# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import random

import numpy as np
import torch
from tensordict import TensorDict

from .models.attention_model import PosAttentionModel
from .models.sampler import Sampler
from .models.sampler import SimulationInstruction
from .search_tree import SearchTree
from .utils.nv import seed_all


def optimize_torch():
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

def set_device(key):
    if 'cuda' in key and torch.cuda.is_available():
        return key
    else:
        return 'cpu'

class SelfPlay:
    """
    Class which runs inference to play games and generate solutions.
    """

    def __init__(self, Game, config, seed, env_type='test', n_beams=0, model=None, datafile=None, sampler=None, worker_id=0,
                 n_problems=None, n_workers=1, **env_args):
        # Setup torch for better performance
        optimize_torch()
        self.k_beams = n_beams if n_beams != 0 else config['train']['n_beams']
        self.config = config
        
        # Fix random generator seed
        seed_all(seed + worker_id, deterministic=self.config['train']['deterministic_algo'])
        self.game = Game(config, env_type=env_type, datafile=datafile, worker_id=worker_id, num_envs=n_problems,
                         n_workers=n_workers, **env_args)
        self.sampler = Sampler(config) if sampler is None else sampler
        self.amp = config['speedups']['amp']

        # Initialize the network
        if model is None:
            model_class = PosAttentionModel
            self.model = model_class(self.game.observation_space, self.game.action_space, config=config,
                                     sampler=self.sampler)
        else:
            self.model = model
        self.model.to(set_device(config['system']['model_device']))
        self.model.eval()

    def set_weights(self, weights, id=None):
        self.model.set_weights(weights, id=id)

    def play_game(self, deterministic=True, detailed_log=False):
        """
        Play one game to generate solutions.
        """
        light_buffer = True
        lazy = True
        inference_mode = True
        n_problems = self.game.n_parallel_problems
        dones = torch.ones(self.game.batch_dim, dtype=bool)
        dones[:, 0] = False
        self.model.eval()
        observations = self.game.reset()
        infos = {'env_id': self.game.ids}
        
        # Initial forward pass
        with torch.inference_mode():
            with torch.autocast(device_type=self.model.device, dtype=torch.float16, enabled=self.amp):
                scores, values, state_score = self.model.forward(observations, lazy=lazy, unmasked_heads=~dones)
        root_state = self.game.get_state()
        policy_logits = scores.clone()
        root_data = {
            'rewards': torch.zeros(observations.shape),
            'dones': dones,
            'actions': -torch.ones(observations.shape, dtype=int),
        }
        roots_data = TensorDict(root_data, batch_size=observations.shape)
        trees = SearchTree(self.config, env=self.game, problem_uid=self.game.ids, roots=roots_data)
        
        # Main game loop
        iter_count = 0
        done = False
        while not done:
            if iter_count == 0:
                n_samples_from_head = torch.zeros(n_problems, self.k_beams, dtype=int)
                n_samples_from_head[:, 0] = self.k_beams
            else:
                n_samples_from_head = torch.ones(n_problems, self.k_beams, dtype=int)
                n_samples_from_head[dones] = 0

            if self.config['eval']['naive_greedy']:
                actions, simulation_instruction, log_prob, info = greedy_step(
                        self.game.distance_matrix, observations, iter_count == 0)
            else:
                actions, log_prob, simulation_instruction, info = (
                    self.sampler.sample_heads_and_actions(state_score, dones,
                                                          policy_logits=policy_logits.clone(),
                                                          legal_actions=observations['feasible_nodes'].clone(),
                                                          deterministic=deterministic,
                                                          n_actions_from_head=n_samples_from_head))

            total_actions_this_step = (actions != -1).sum().item()
            new_observations, rewards, dones, _ = self.game.step(actions, automatic_reset=False)
            
            # Update tree
            trees.expand_and_update_tree(
                dones=dones,
                log_prob=log_prob,
                observations=observations,
                policy_logits=policy_logits,
                rewards=rewards,
                simulation_instruction=simulation_instruction,
                values=values,
                info=info,
                light_buffer=True
            )
            trees.reset_frontier()
            
            # Next state
            observations = new_observations
            with torch.inference_mode(mode=inference_mode):
                if self.config['eval']['naive_greedy']:
                    scores = torch.ones(observations['feasible_nodes'].shape, dtype=float)
                    values = torch.ones(actions.shape, dtype=float)
                    state_score = torch.ones(actions.shape, dtype=float)
                else:
                    with torch.autocast(device_type=self.model.device, dtype=torch.float16, enabled=self.amp):
                        scores, values, state_score = self.model.forward(observations, unmasked_heads=~dones, lazy=True)

            policy_logits = scores.clone()
            
            # Check termination
            iter_count += 1
            done_trees = trees.dones()
            done = iter_count > self.config['muzero']['max_moves'] or dones.all() or all(done_trees)
        
        # Get final results
        game_histories, game_infos = trees.backpropagate_and_fill_buffer(detailed_log=detailed_log,
                                                                         light_buffer=True)
        infos.update(game_infos)
        
        return game_histories, infos


def greedy_step(distances, observations, first_iter):
    actions = get_greedy_actions(distances, observations['feasible_nodes'].clone(),
                                 observations['head'].squeeze(-1).clone())
    sim_actions = - torch.ones(actions.shape[0], actions.shape[1], actions.shape[1] + 1, dtype=int)
    if first_iter:
        sim_actions[:, 0, :-1] = actions
        n_actions_from_head = torch.zeros_like(actions)
        n_actions_from_head[:, 0] = n_actions_from_head.shape[1]
    else:
        sim_actions[:, :, 0] = actions
        n_actions_from_head = torch.ones_like(actions)

    valid_actions = sim_actions > -1
    n_valid_actions = valid_actions.sum(dim=[-1, -2])
    if (n_valid_actions != actions.shape[1]).any():
        valid_action_mask = torch.zeros(actions.shape[0], actions.shape[1], actions.shape[1] + 1, dtype=bool)
        valid_action_mask[:, :, 0] = True
    else:
        valid_action_mask = valid_actions

    simulation_instruction = SimulationInstruction(actions=sim_actions,
                                                   n_actions_from_head=n_actions_from_head,
                                                   valid_actions_mask=valid_action_mask)
    log_prob = torch.ones((actions.shape[0], actions.shape[1], actions.shape[1] + 1))
    info = {}
    return actions, simulation_instruction, log_prob, info


def get_greedy_actions(distances, permitted, head):
    N, M = distances.shape[:2]

    # Initialize the next_visit tensor
    next_visit = torch.empty(N, M, dtype=torch.long)

    # Calculate distances from current locations
    distances = distances[torch.arange(N).unsqueeze(1), torch.arange(M), head]

    # Make depots least preferable
    max_distance = distances.max()
    distances[:, :, 0] = max_distance + 1

    # Mask non-permitted locations
    masked_distances = distances.clone()
    masked_distances[~permitted] = float('inf')

    # For m == 0, find the closest permitted location
    closest_permitted = torch.argmin(masked_distances[:, 0], dim=1)
    no_permitted = ~permitted[:, 0].any(dim=1)
    next_visit[:, 0] = torch.where(no_permitted, torch.tensor(-1), closest_permitted)

    # For m > 0, choose a random permitted location
    if M > 1:
        if False:
            # vectoric code - buggy

            permitted = permitted[:, 1:].clone()
            # Create a tensor to store the count of permitted destinations for each problem
            permitted_counts = torch.sum(permitted, dim=-1)
            # permitted[:, :, 0][torch.where(permitted_counts > 1)[0]] = False
            reset_ids = torch.where(permitted_counts > 1)
            permitted[reset_ids[0], reset_ids[1], 0] = False
            permitted_counts = torch.sum(permitted, dim=-1)

            # Get the indices of permitted destinations for all problems
            permitted_indices = torch.where(permitted)

            # Generate random indices for each problem
            random_indices = torch.rand(N, M - 1)

            # Scale the random indices to the range of permitted destinations
            scaled_indices = (random_indices * permitted_counts.float()).long()

            # Create a tensor to store the cumulative sum of permitted destinations
            cumulative_counts = torch.cumsum(permitted_counts, dim=0)

            # Adjust the scaled indices based on the cumulative sum
            adjusted_indices = scaled_indices + torch.cat(
                    [torch.zeros(1, M - 1, dtype=torch.long), cumulative_counts[:-1]], dim=0)

            # Flatten the adjusted indices
            flat_indices = adjusted_indices.view(-1)

            # Select the permitted destinations using the flat indices
            selected_destinations = permitted_indices[2][flat_indices]

            # Reshape the selected destinations to match the original shape
            random_visit = selected_destinations.view(N, M - 1)

            # Set -1 for problems with no permitted destinations
            random_visit[permitted_counts == 0] = -1

            next_visit[:, 1:] = random_visit
        else:
            # slow code
            for n in range(N):
                for m in range(1, M):
                    permitted_indices = torch.where(permitted[n, m])[0]
                    if len(permitted_indices) > 0:
                        random_idx = torch.randint(0, len(permitted_indices), (1,))
                        next_visit[n, m] = permitted_indices[random_idx]
                    else:
                        next_visit[n, m] = -1  # Handle case where no locations are permitted

    return next_visit
