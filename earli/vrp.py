# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from copy import deepcopy

import numpy
import numpy as np
import torch
from gymnasium import Env as GymEnv
from gymnasium.spaces import Discrete, Box, Dict
from tensordict import TensorDict
from torch.nn.functional import pad

from .vehicle_routing import RoutingBase, GRAPH
from .utils.general_usage_utils import cyclic_indexing

DEPOT_LOCATION = 0
STATE_PROPERTIES = {'head', 'demand', 'capacity'}


def build_attention_matrix(visible_nodes):
    n_nodes = visible_nodes.shape[-1]
    attention_matrix = visible_nodes.unsqueeze(-1).expand(-1, -1, n_nodes)
    attention_matrix = attention_matrix & attention_matrix.transpose(-2, -1)
    # assign one to the diagonal, so that attention will not return NaN
    eye = torch.eye(n_nodes, dtype=torch.bool).view(1, n_nodes, n_nodes).expand_as(attention_matrix)
    attention_matrix[eye] = True
    attention_matrix = attention_matrix.unsqueeze(1)  # [batch_size, 1, graph_size, graph_size]
    return attention_matrix


class VRP(RoutingBase):
    def __init__(self, config, **kwargs):
        GymEnv.__init__(self)
        state_properties = list(STATE_PROPERTIES)
        problem_properties = ['demand', 'capacity']
        prop_types = dict(demand=torch.float32, capacity=torch.float32)
        RoutingBase.__init__(self, config, problem_properties, state_properties, prop_types=prop_types,
                             title='vrp', **kwargs)

        self.last_go_back_to_depot = self.config['problem_setup']['last_return_to_depot']
        self.extra_car_penalty = 0
        self.unused_capacity_penalty = 0
        if self.config['problem_setup']['minimize_vehicles']:
            self.extra_car_penalty = self.config['problem_setup']['vehicle_penalty'] * self.radius
            self.extra_car_penalty *= self.reward_normalization
            if self.env_type == 'train':
                self.unused_capacity_penalty = self.config['problem_setup']['unused_capacity_penalty']
                self.unused_capacity_penalty *= self.radius * self.reward_normalization
        # print(f'Vehicle penalty: {self.extra_car_penalty}; Capacity penalty: {self.unused_capacity_penalty}')

    # noinspection PyAttributeOutsideInit
    def set_spaces(self):
        # initial demand (static) per customer, demand of depot =0
        self.initial_demand = torch.zeros([self.n_parallel_problems, self.problem_size], device=self.device)
        # Vehicle capacity (static)
        # todo: removed extra dimension in max_capacity, capacity, head
        self.max_capacity = torch.zeros(size=[self.n_parallel_problems], device=self.device)

        # dynamic properties
        # remaining demand (dynamic)
        self.demand = torch.zeros(self.maybe_extend([self.n_parallel_problems, self.problem_size]), device=self.device)
        # Current location (dynamic)
        self.head = torch.zeros(self.maybe_extend([self.n_parallel_problems, 1]), dtype=torch.long, device=self.device)
        # Unvisited places (dynamic)
        self.feasible_nodes = torch.ones(self.maybe_extend([self.n_parallel_problems, self.problem_size]),
                                         dtype=torch.bool, device=self.device)
        # Vehicle space left (dynamic)
        self.capacity = torch.zeros(size=self.maybe_extend([self.n_parallel_problems, 1]), device=self.device)
        # nodes with non zero demand (dynamic)
        self.non_zero_demand = torch.ones_like(self.feasible_nodes)
        # padding_mask: True for real nodes, False for padding nodes injected
        # when training with problems of mixed sizes.  Initialised to all-True
        # (no padding) and updated in reset_static_properties.
        self.padding_mask = torch.ones(
            self.maybe_extend([self.n_parallel_problems, self.problem_size]),
            dtype=torch.bool, device=self.device,
        )
        self.action_space = Discrete(self.problem_size)
        # self.filter_action_for_masked_sites = False
        self.normalize = self.config['representation']['normalize']
        self.baseline_return = np.zeros(self.n_parallel_problems)
        self.baseline_vehicles = np.zeros(self.n_parallel_problems)
        self.acc_returns = torch.zeros(size=self.maybe_extend([self.n_parallel_problems, 1]), device=self.device)
        self.n_edge_features = 2  # distance, head_to_depot
        self.depot_one_hot = torch.zeros(1, self.problem_size, dtype=torch.bool, device=self.device)
        self.depot_one_hot[:, DEPOT_LOCATION] = 1
        self.inf_diagonal_matrix = torch.diag(torch.full(fill_value=torch.inf, size=[self.problem_size],
                                                         device=self.device))
        self.k_connectivity = min(self.config['representation']['k_connectivity'], self.problem_size)
        k = self.k_connectivity
        self.top_k_source = torch.arange(self.problem_size, device=self.device)
        if self.extended_output_format:
            self.top_k_source = self.top_k_source.view(1, 1, -1, 1).expand(self.n_parallel_problems, self.n_beams, -1, k)
        else:
            self.top_k_source = self.top_k_source.view(1, -1, 1).expand(self.num_envs, -1, k)
        self.input_repr = GRAPH
        self.self_loops = self.config['representation']['self_loops']
        n_node_features = 4  # feasible, is_depot, demand, is_head
        n_node_features += 1 if self.add_distance_to_head else 0
        self.n_edge_features += 1 if self.self_loops else 0
        self.max_edges = int(self.config['representation'].get('max_edges', k) * self.problem_size)
        self.max_prior_size = 2 * self.max_edges
        # Edge features: distance, prior, type (KNN, link to depot, self loop, prior edge)
        # Node features: depot/non depot, demand, head position, masked / unmasked
        if self.stable_baselines_compatibility:
            self.action_space = Discrete(self.problem_size)
            self.observation_space = Dict(
                    {'loc'             : Box(low=-np.inf, high=np.inf, shape=(self.problem_size, 2)),
                     'demand'          : Box(low=0, high=np.inf, shape=(self.problem_size, 1)),
                     'head'            : Box(low=0, high=self.problem_size + 1, shape=(1,), dtype=int),
                     # Discrete(*self.batch_dim, self.problem_size),
                     'capacity'        : Box(low=0, high=np.inf, shape=(self.problem_size, 1)),
                     'attention_matrix': Box(low=0.0, high=1.0, shape=(1, self.problem_size, self.problem_size),
                                             dtype=bool),
                     'visible_nodes'   : Box(low=0.0, high=1.0, shape=(self.problem_size,), dtype=bool),
                     'feasible_nodes'  : Box(low=0.0, high=1.0, shape=(self.problem_size,), dtype=bool),
                     'remaining_demand': Box(low=0, high=np.inf, shape=(self.problem_size, 1)),
                     'remaining_nodes' : Box(low=0, high=np.inf, shape=(self.problem_size, 1)),
                     'head_feature'    : Box(low=0, high=1, shape=(self.problem_size, 1), dtype=bool),
                     'acc_returns'     : Box(low=-np.inf, high=np.inf, shape=(1,), dtype=float)
                     })
        else:
            self.action_space = Box(low=0, high=self.problem_size - 1, shape=self.batch_dim,
                                    dtype=int)  # Discrete(self.problem_size))
            self.observation_space = Dict(
                    {'loc'             : Box(low=-np.inf, high=np.inf, shape=(*self.batch_dim, self.problem_size, 2)),
                     'demand'          : Box(low=0, high=np.inf, shape=(*self.batch_dim, self.problem_size, 1)),
                     'head'            : Box(low=0, high=self.problem_size + 1, shape=(*self.batch_dim, 1), dtype=int),
                     # Discrete(*self.batch_dim, self.problem_size),
                     'capacity'        : Box(low=0, high=np.inf, shape=(*self.batch_dim, self.problem_size, 1)),
                     'attention_matrix': Box(low=0.0, high=1.0, shape=(*self.batch_dim, 1, self.problem_size, self.problem_size),
                                             dtype=bool),
                     'visible_nodes'   : Box(low=0.0, high=1.0, shape=(*self.batch_dim, self.problem_size,), dtype=bool),
                     'feasible_nodes'  : Box(low=0.0, high=1.0, shape=(*self.batch_dim, self.problem_size,), dtype=bool),
                     'remaining_demand': Box(low=0, high=np.inf, shape=(*self.batch_dim, self.problem_size, 1)),
                     'remaining_nodes' : Box(low=0, high=np.inf, shape=(*self.batch_dim, self.problem_size, 1)),
                     'head_feature'    : Box(low=0, high=1, shape=(*self.batch_dim, self.problem_size, 1), dtype=bool),
                     'acc_returns'     : Box(low=-np.inf, high=np.inf, shape=(*self.batch_dim, 1), dtype=float)
                     })

        self.terminal_state = {'demand'     : torch.zeros([1, self.problem_size], device=self.device),
                               'capacity'   : torch.zeros([1, 1], device=self.device),
                               'head'       : torch.zeros([1, 1], device=self.device),
                               'acc_returns': torch.zeros([1], device=self.device)}
        if self.config['system']['use_tensordict']:
            self.terminal_state = TensorDict(self.terminal_state, batch_size=1)
        self.root_is_source = torch.zeros(self.maybe_extend([self.n_parallel_problems, self.problem_size, 1]), dtype=torch.bool,
                                          device=self.device)
        self.root_is_source[:, DEPOT_LOCATION] = True

    def reset_dynamic_properties(self, indices, n_envs_to_reset):
        # reset dynamic quantities
        # Set head to index of depot
        self.head[indices] = DEPOT_LOCATION
        # set all sites to visitable
        self.feasible_nodes[indices] = True
        # Vehicle is already at depot
        self.feasible_nodes[indices, ..., DEPOT_LOCATION] = False
        # reset and normalize demand
        self.non_zero_demand[indices] = self.feasible_nodes[indices].clone()  # same value
        # Apply padding mask: padding nodes (from mixed-size training) are never
        # feasible and carry no demand, so they should never be selected.
        self.feasible_nodes[indices] &= self.padding_mask[indices]
        self.non_zero_demand[indices] &= self.padding_mask[indices]
        # first index is the accumalted return in this environment, second index is the minimal accumlated return in all environments
        self.acc_returns[indices] = 0

        # the mapping between indices (environment indices being reset), and their initial values to the fixed values that they
        # are being reset to is in reset_static_properties
        if self.normalize:
            demand = self.initial_demand[indices] / self.max_capacity[indices].view(self.initial_demand[indices].shape[0], 1)
            if self.extended_output_format:
                demand = demand.unsqueeze(1)
            self.demand[indices] = demand
            self.capacity[indices] = 1.0
        else:
            demand = self.initial_demand[indices].clone()
            if self.extended_output_format:
                demand = demand.unsqueeze(1).expand_as(self.demand[indices])
            self.demand[indices] = demand
            capacity = self.max_capacity[indices].clone()
            if self.extended_output_format:
                capacity = capacity.unsqueeze(-1).unsqueeze(-1).expand_as(self.capacity[indices])
            self.capacity[indices] = capacity

    def reset_static_properties(self, n_problems_to_reset, copy_all, indices):
        dataset_size = self.reference['positions'].shape[0]
        if not self.fix_envs and self.episode_counter + n_problems_to_reset >= dataset_size - 1:
            self.data = self.problem_generator.create_dataset(dataset_type=self.config['problem_setup']['env'])
            self.set_problem_data(data=self.data)
        # partial indices is probably not going to be supported anymore
        fixed_dataset_indices = cyclic_indexing(self.episode_counter % dataset_size,
                                                n_problems_to_reset,
                                                dataset_size)
        if self.extended_output_format:
            pos = (self.reference['positions'][fixed_dataset_indices]
                   .unsqueeze(1).clone().expand_as(self.pos[indices]))
            dist = (self.reference['distance_matrix'][fixed_dataset_indices]
                    .unsqueeze(1).clone().expand_as(self.distance_matrix[indices]))
            self.pos[indices] = pos if pos.dtype == torch.float else pos.float()
            self.distance_matrix[indices] = dist if dist.dtype == torch.float else dist.float()
        else:
            self.pos[indices] = self.reference['positions'][fixed_dataset_indices].clone()
            self.distance_matrix[indices] = self.reference['distance_matrix'][fixed_dataset_indices].clone()
        self.max_capacity[indices] = (self.reference['capacity'][fixed_dataset_indices]
                                      .to(self.max_capacity.dtype).flatten().clone())
        self.initial_demand[indices] = self.reference['demand'][fixed_dataset_indices].clone()
        if 'baseline_cost' in self.data:
            self.baseline_return[indices] = deepcopy(self.data['baseline_cost'][fixed_dataset_indices]) \
                if (self.data and self.data['baseline_cost'].shape[0] > 0) else [None] * len(fixed_dataset_indices)
            self.baseline_vehicles[indices] = deepcopy(self.data['baseline_vehicles'][fixed_dataset_indices]) \
                if ('baseline_vehicles' in self.data and self.data['baseline_vehicles'].shape[0] > 0) \
                else [None] * len(fixed_dataset_indices)
        if self.fix_envs:
            self.ids[indices] = self.reference['id'][fixed_dataset_indices]
        else:
            self.ids[indices] = None
        # after reset, episode counter points to the index of the next episode in the data
        self.episode_counter += n_problems_to_reset
        if self.normalize:
            if self.config['representation']['normalize_pos_obs_like_reward']:
                self.pos[indices] = self.pos[indices] * self.reward_normalization
            self.distance_matrix[indices] = self.distance_matrix[indices] * self.reward_normalization
        # Update padding_mask for the active environments.
        # For mixed-size training, valid_mask_ref marks real nodes (True) vs
        # padding nodes (False).  Padding nodes have zero demand and will be
        # masked out during reset_dynamic_properties so they are never selected.
        vm = self.valid_mask_ref[fixed_dataset_indices].to(self.device)
        if self.extended_output_format:
            vm = vm.unsqueeze(1).expand_as(self.padding_mask[indices])
        self.padding_mask[indices] = vm
        return n_problems_to_reset

    def _step(self, actions, calc_obs=True):
        '''
        runs one step of the environment and updates demands, loads and masks
        '''
        # fix potential type issues
        if isinstance(actions, numpy.ndarray) or isinstance(actions, list):
            actions = torch.tensor(actions, dtype=torch.long)
        else:
            actions = actions.to(int)
        actions = actions.clone().view(-1)
        n_actions = len(actions)
        head = self.head.view(-1)
        assert n_actions == len(head), 'The number of actions must be equal to the number of environments'

        # Reshaping view of internal properties for easier handling
        distance_matrix = self.distance_matrix.view(-1, self.problem_size, self.problem_size)
        demand = self.demand.view(-1, self.problem_size)
        capacity = self.capacity.view(-1)
        acc_returns = self.acc_returns.view(-1)

        active_env = actions >= 0  # maybe self-consistent with dones?
        actions[~active_env] = 0  # To avoid crashes
        dummy_ind = torch.arange(n_actions, dtype=torch.long)
        from_depot = active_env & (head[:n_actions] == DEPOT_LOCATION)
        depot_visit = active_env & (actions == DEPOT_LOCATION)
        visit_site = actions.clone()

        # reward calculation
        reward = -distance_matrix[dummy_ind, head, visit_site]
        reward[from_depot] -= self.extra_car_penalty
        if self.unused_capacity_penalty > 0:
            reward[depot_visit] -= self.unused_capacity_penalty * capacity[depot_visit]
        acc_returns[active_env] += reward[active_env]
        info = {'active_env': active_env}
        if self.macro_state:
            info['single_env_reward'] = reward
        reward[~active_env] = 0
        reward = reward.view(*self.batch_dim)

        head[active_env] = visit_site[active_env].to(self.device)

        # update demands
        demand_removed = demand[dummy_ind, visit_site]
        demand[dummy_ind, visit_site] = 0  # zero demand, assuming a vehicle cannot visit the same site twice!

        # update capacity (assuming max_capcity is constant for all problems!)
        capacity -= demand_removed
        capacity[depot_visit] = 1.0 if self.normalize else self.max_capacity[depot_visit].clone()  # reset capacity at depot

        self.non_zero_demand = self.demand > 0

        # mask sites with demand > capacity
        self.feasible_nodes = (0 < self.demand) & (self.demand <= self.capacity)

        # unmask depot
        self.feasible_nodes[..., DEPOT_LOCATION] = True

        # set current position as masked
        self.feasible_nodes.view(-1, self.problem_size)[dummy_ind, head] = False

        # the episode is done when all the sites have zeroes their demand,
        dones = self.non_zero_demand.sum(dim=-1) == 0
        if self.last_go_back_to_depot:
            dones = dones & (head == DEPOT_LOCATION).view_as(dones)
        dones = dones | ~active_env.view_as(dones)

        obs = None
        if calc_obs:
            obs = self.get_pos_representation()
        if self.config['train']['method'] == 'ppo':
            reward, dones = reward.numpy(), dones.numpy()
        return obs, reward, dones, info

    def get_pos_representation(self, indices=None, extra_global_features=None):
        """
        returns a feature representation of the current state
        Args:
            indices: indices of the environments to be considered
            extra_global_features: additional global features to be appended to the representation

        Returns:
            a dictionary containing the feature representation

        """
        if indices is None:
            indices = torch.ones(self.n_parallel_problems, dtype=torch.bool)
        n_problems = indices.sum()
        if self.extended_output_format:
            all_ind = torch.arange(n_problems * self.n_beams, dtype=torch.long)
            batch_dim = (n_problems, self.n_beams)
        else:
            all_ind = torch.arange(n_problems, dtype=torch.long)
            batch_dim = (n_problems,)

        head = self.head[indices].view(-1)

        visible_nodes = self.non_zero_demand[indices].clone().view(-1, self.problem_size)
        visible_nodes[all_ind, head] = True
        visible_nodes[:, DEPOT_LOCATION] = True
        attention_matrix = build_attention_matrix(visible_nodes)

        demand = self.demand[indices].view(-1, self.problem_size)
        non_zero_demand = self.non_zero_demand[indices].clone().view(-1, self.problem_size)

        # global features: remaining capacity (max capacity = 1)
        remaining_demand = demand.sum(dim=-1)
        remaining_nodes_fraction = non_zero_demand.sum(dim=-1) / self.net_problem_size

        head_feature = torch.nn.functional.one_hot(head, num_classes=self.problem_size).view(*batch_dim, -1, 1).to(bool)
        remaining_demand_feature = remaining_demand.view(*batch_dim, 1, 1).expand_as(head_feature).clone()
        remaining_nodes_feature = remaining_nodes_fraction.view(*batch_dim, 1, 1).expand_as(head_feature).clone()

        obs = {'loc'             : self.pos[indices],  # todo: is clone neccessary?
               'demand'          : self.demand[indices].unsqueeze(-1).clone(),
               'head'            : self.head[indices].clone(),
               'capacity'        : self.capacity[indices].unsqueeze(-1).expand_as(head_feature).clone(),
               'attention_matrix': attention_matrix.view(*batch_dim, *attention_matrix.shape[1:]),
               'visible_nodes'   : visible_nodes.view(*batch_dim, *visible_nodes.shape[1:]),
               'feasible_nodes'  : self.feasible_nodes[indices].clone(),
               'remaining_demand': remaining_demand_feature,
               'remaining_nodes' : remaining_nodes_feature,
               'head_feature'    : head_feature.type(torch.bool),  # bool for consistency with the buffer
               'acc_returns'     : self.acc_returns[indices].clone()
               }

        if self.config['system']['use_tensordict']:
            obs = TensorDict(obs, batch_size=batch_dim)
            if not self.config['system']['save_obs_on_gpu']:
                obs = obs.to('cpu')
        return obs

    def state_to_hashable(self, state):
        return state['head'].item(), tuple(state['demand'].tolist()), np.round(state['capacity'].item(), 6)

    def baseline_values(self, obs, valid_actions, mode=None):
        assert self.add_distance_to_head, 'baseline requires "add_distance_to_head" to be set to true.'
        distance_to_head = obs['X'][:, :, -1]
        if mode == 'greedy':
            value = - distance_to_head
            value[:, 0] = -self.max_distance
        elif mode == 'random':
            value = torch.rand(distance_to_head.shape)
        else:
            raise ValueError(mode)
        return value
