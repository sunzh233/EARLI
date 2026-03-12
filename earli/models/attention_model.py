import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_schedule_fn
from tensordict import TensorDict

from .base_model import AbstractNetwork, set_device
from .sampler import Sampler
from .attention_actor import ActorAttentionModel
from .attention_critic import CriticNetwork


class PosAttentionModel(AbstractNetwork, ActorCriticPolicy):
    """PPO Model based on Proximal Policy Optimization (PPO).

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        critic: Critic to use for the algorithm
        policy_kwargs: Keyword arguments for policy
        critic_kwargs: Keyword arguments for critic
    """

    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule=None, use_sde=False, config=None, sampler=None,
            policy_kwargs: dict = {},
            critic_kwargs: dict = {},
            **kwargs,
            ):
        self._config_device = set_device(config['system']['model_device']) if config else 'cpu'
        # if lr_schedule is None:
        #     lr_schedule = get_schedule_fn(float(config['train']['learning_rate']))
        # ActorCriticPolicy.__init__(self, observation_space, action_space, lr_schedule, share_features_extractor=False)
        AbstractNetwork.__init__(self, config=config, sampler=sampler)
        self.actor = ActorAttentionModel(config, **policy_kwargs).to(self.device)
        self.v = CriticNetwork(config, **critic_kwargs).to(self.device)
        self.optimizer = self.get_optimizer()
        self.sampler = Sampler(config) if sampler is None else sampler
        if not self.config['speedups']['use_fabric']:
            self.actor = self.actor.to(self.device)
            self.v = self.v.to(self.device)

    def get_optimizer(self):
        lr = float(self.config['train']['learning_rate'])
        if self.config['train']['optimizer'].lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr,
                                         fused=self.config['speedups']['fused_optimizer'],
                                         foreach=self.config['speedups']['foreach_optimizer'])
        elif self.config['train']['optimizer'].lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr,
                                          fused=self.config['speedups']['fused_optimizer'],
                                          foreach=self.config['speedups']['foreach_optimizer'])
        return optimizer

    @property
    def device(self):
        """Get the device to use (handles the conflict between parent classes)"""
        return self._config_device

    def disable_non_head_layers(self):
        trainable_params = self.actor.disable_non_head_layers()
        return trainable_params

    def set_training_mode(self, val):
        self.train(val)

    def forward(self, state, *args, inference_mode=True, lazy=False, batch_shape=None, copy_to_cpu=True,
                deterministic: bool = False, **kwargs):
        values, policy_logits, batch_shape = self._forward(
            state=state,
            batch_shape=batch_shape,
            lazy=lazy,
            **kwargs,
        )
        if self.config['system']['compatibility_mode'] == 'stable_baselines':
            action, log_prob, entropy = self.sampler.sample(policy_logits, unmasked_nodes=state['feasible_nodes'].to(bool),
                                                            deterministic=deterministic)
            log_prob = log_prob.squeeze(1)
            return action, values, log_prob
        else:
            state_score = torch.ones(batch_shape, device=policy_logits.device)
            policy_logits = policy_logits.view(*batch_shape, *policy_logits.shape[1:])
            values = values.view(*batch_shape)
            if copy_to_cpu:
                state_score = state_score.cpu()
                policy_logits = policy_logits.cpu()
                values = values.cpu()
            return policy_logits, values, state_score

    def _forward(self, state, batch_shape=None, lazy=False, **kwargs):
        if self.config['system']['compatibility_mode'] == 'stable_baselines':
            batch_shape = (state['loc'].shape[0],)
        if batch_shape is not None:
            state = TensorDict(state, batch_size=batch_shape)
        else:
            batch_shape = state.batch_size
        if self.config['system']['compatibility_mode'] == 'stable_baselines':
            state = state.unsqueeze(1)
        state = state.view(-1).to(self.device)
        policy_logits, state_representation = self.actor(state, train_actor=self.actor)
        if lazy:
            values = torch.zeros(state.shape[0], 1, device=self.device)
        else:
            values = self.v(state)
        return values, policy_logits, batch_shape

    def evaluate_actions(self, state, actions, unmasked_heads=None, samples=None, action_grounded=None,
                         action_rank=None):
        """
                Evaluate actions according to the current policy,
                given the observations.

                :param obs: Observation
                :param actions: Actions
                :return: estimated value, log likelihood of taking those actions
                    and entropy of the action distribution.
        """
        values, logits, _ = self._forward(state=state, batch_shape=None, lazy=False)
        # Keep masks/actions on the same device as logits to avoid CPU/CUDA gather errors.
        device = logits.device
        actions = actions.to(device=device)
        unmasked_nodes = state['feasible_nodes'].to(device=device, dtype=torch.bool)
        _, log_prob, entropy = self.sampler.sample(logits, unmasked_nodes=unmasked_nodes, action=actions,
                                                   deterministic=False)
        return values, log_prob, entropy

    def predict_values(self, state):
        values = self.v(state).view(-1)
        return values


class BaselineModel(PosAttentionModel):
    def __init__(self, env, **kwargs):
        PosAttentionModel.__init__(self, **kwargs)
        self.env = env
        self.mode = None

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, state, deterministic=False, unmasked_heads=Ellipsis, copy_to_cpu=True):
        raise NotImplementedError('baselines not implemented for attention')
