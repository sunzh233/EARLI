# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from abc import abstractmethod

import torch
from stable_baselines3.common.utils import set_random_seed
from .head_attention_model import HeadAttentionModel
from .sampler import Sampler


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


def set_device(key):
    if 'cuda' in key and torch.cuda.is_available():
        return key
    else:
        return 'cpu'


class AbstractNetwork(torch.nn.Module):
    def __init__(self, config=None, sampler=None):
        torch.nn.Module.__init__(self)
        self.sampler = Sampler(config) if sampler is None else sampler
        self.eight_rounding = config['model']['eight_rounding']
        set_random_seed(config['train']['seed'])
        self.config = config
        self.dtype = torch.float64 if config['buffer']['buffer_precision'] == 'float64' else torch.float32
        self.macro_state = False
        self.forward_mode = 'tree'
        self.head_model = HeadAttentionModel(config)
        self.head_embedding = torch.nn.Linear(config['model']['embedding_dim'] + (8 if self.eight_rounding else 2),
                                              config['model']['embedding_dim'])

    @abstractmethod
    def initial_inference(self, observation):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action):
        pass

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights, id=None):
        missing_keys, unexpected_keys = self.load_state_dict(weights, strict=False)
        self.update_counter += 1
        if id is not None:
            print(id)
        self.eval()
        if missing_keys or unexpected_keys:
            print(f'missing_keys: {missing_keys}')
            print(f'unexpected_keys: {unexpected_keys}')

    def set_training_mode(self, val):
        self.train(val)

    def forward_inference(self, state, deterministic=False,  unmasked_heads=Ellipsis, copy_to_cpu=True, lazy=False, fabric=None):
        batch_shape = state.batch_size
        state = state.view(-1)
        value, policy_logits, state_representation = self._forward(state=state,
                                                                   unmasked_heads=unmasked_heads, lazy=lazy)
        state_score = torch.ones(batch_shape, device=policy_logits.device)
        policy_logits = policy_logits.view(*batch_shape, *policy_logits.shape[1:])
        value = value.view(*batch_shape)
        if copy_to_cpu:
            state_score = state_score.cpu()
            policy_logits = policy_logits.cpu()
            value = value.cpu()
        return policy_logits, value, state_score

    def forward(self, state, *args, inference_mode=True, lazy=False, batch_shape=None, **kwargs):
        if batch_shape is not None:
            state = TensorDict(state, batch_size=batch_shape)
        if not self.config['speedups']['use_fabric']:
            state = state.to(self.device)
        if inference_mode:
            return self.forward_inference(state, lazy=lazy, *args, **kwargs)
        else:
            return self.evaluate_actions(state, *args, **kwargs)

    def evaluate_actions(self, state, actions, unmasked_heads=None, samples=None, action_grounded=None,
                         action_rank=None):
        values, logits, _ = self._forward(state=state)
        _, log_prob, entropy = self.sampler.sample(logits, unmasked_nodes=state['feasible_nodes'].to(bool), action=actions,
                                                   deterministic=False)
        head_scores = torch.ones([1], device='cuda')
        return values, log_prob, entropy, head_scores

    @torch.inference_mode()
    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        values, logits, _ = self._forward(state=obs)
        feasible_nodes = obs['feasible_nodes']
        if not isinstance(feasible_nodes, torch.Tensor):
            feasible_nodes = torch.as_tensor(feasible_nodes, device=logits.device)
        else:
            feasible_nodes = feasible_nodes.to(logits.device)
        feasible_nodes = feasible_nodes.to(torch.bool)

        action, log_prob, entropy = self.sampler.sample(
            logits,
            unmasked_nodes=feasible_nodes,
            deterministic=deterministic,
        )
        model_info = {'value'    : values,
                      'log_probs': log_prob}
        return action, model_info

    def reset_parameters(self, k=None):
        self.actor.reset_parameters(k)
        self.v.reset_parameters(k)

    def reset_last_layers(self):
        self.actor.reset_last_layers()
        self.v.reset_last_layers()



