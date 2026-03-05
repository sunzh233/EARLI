# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""PDPTW (Pickup and Delivery Problem with Time Windows) environment.

Extends the VRPTW environment to enforce pickup-delivery precedence constraints
using the Li&Lim benchmark data format.

Constraints:
- A delivery node may only be visited after its paired pickup node has been
  visited (on the same route).
- The same vehicle must serve both the pickup and the delivery.
- Time-window constraints are inherited from VRPTW.
"""

from copy import deepcopy

import numpy as np
import torch
from gymnasium import Env as GymEnv
from gymnasium.spaces import Box, Dict
from tensordict import TensorDict

from .vrptw import VRPTW, STATE_PROPERTIES_VRPTW
from .vrp import DEPOT_LOCATION
from .vehicle_routing import RoutingBase
from .utils.general_usage_utils import cyclic_indexing

STATE_PROPERTIES_PDPTW = STATE_PROPERTIES_VRPTW | {'pickup_done'}


class PDPTW(VRPTW):
    """PDPTW environment: VRPTW + pickup-delivery pair precedence.

    Additional problem data:
    - ``pairs`` : (n_pairs, 2) int tensor – [pickup_node_idx, delivery_node_idx]

    Additional dynamic state:
    - ``pickup_done``: (batch, n_nodes) bool – True once a pickup node is visited
    - ``is_pickup``  : (batch, n_nodes) bool – True for pickup nodes (static)
    - ``is_delivery``: (batch, n_nodes) bool – True for delivery nodes (static)
    - ``pair_index`` : (n_nodes,) int   – maps each delivery to its pickup (−1 otherwise)
    """

    def __init__(self, config, **kwargs):
        GymEnv.__init__(self)
        state_properties = list(STATE_PROPERTIES_PDPTW)
        problem_properties = ['demand', 'capacity', 'time_windows', 'service_times', 'pairs']
        prop_types = dict(demand=torch.float32, capacity=torch.float32,
                          time_windows=torch.float32, service_times=torch.float32,
                          pairs=torch.long)
        RoutingBase.__init__(self, config, problem_properties, state_properties,
                             prop_types=prop_types, title='pdptw', **kwargs)

        self.last_go_back_to_depot = self.config['problem_setup']['last_return_to_depot']
        self.extra_car_penalty = 0
        self.unused_capacity_penalty = 0
        if self.config['problem_setup']['minimize_vehicles']:
            self.extra_car_penalty = self.config['problem_setup']['vehicle_penalty'] * self.radius
            self.extra_car_penalty *= self.reward_normalization
            if self.env_type == 'train':
                self.unused_capacity_penalty = self.config['problem_setup']['unused_capacity_penalty']
                self.unused_capacity_penalty *= self.radius * self.reward_normalization

    # ------------------------------------------------------------------
    # Space setup
    # ------------------------------------------------------------------

    def set_spaces(self):
        # Call VRPTW's set_spaces (which calls VRP's)
        super().set_spaces()

        # Static pair info (re-built on each reset_static_properties)
        # is_pickup[i] = True if node i is a pickup node
        self.ref_is_pickup = torch.zeros([self.n_parallel_problems, self.problem_size],
                                         dtype=torch.bool, device=self.device)
        # is_delivery[i] = True if node i is a delivery node
        self.ref_is_delivery = torch.zeros([self.n_parallel_problems, self.problem_size],
                                           dtype=torch.bool, device=self.device)
        # pair_of[i] = pickup index for delivery i (−1 if not a delivery)
        self.ref_pair_of = torch.full([self.n_parallel_problems, self.problem_size], -1,
                                      dtype=torch.long, device=self.device)

        # Dynamic: has the pickup of each pair been served?
        self.pickup_done = torch.zeros(
            self.maybe_extend([self.n_parallel_problems, self.problem_size]),
            dtype=torch.bool, device=self.device,
        )

        # Extend observation space
        if not self.stable_baselines_compatibility:
            self.observation_space = Dict({
                **self.observation_space.spaces,
                'is_pickup'  : Box(low=0, high=1, shape=(*self.batch_dim, self.problem_size, 1), dtype=bool),
                'is_delivery': Box(low=0, high=1, shape=(*self.batch_dim, self.problem_size, 1), dtype=bool),
                'pickup_done': Box(low=0, high=1, shape=(*self.batch_dim, self.problem_size, 1), dtype=bool),
            })
        else:
            self.observation_space = Dict({
                **self.observation_space.spaces,
                'is_pickup'  : Box(low=0, high=1, shape=(self.problem_size, 1), dtype=bool),
                'is_delivery': Box(low=0, high=1, shape=(self.problem_size, 1), dtype=bool),
                'pickup_done': Box(low=0, high=1, shape=(self.problem_size, 1), dtype=bool),
            })

        # Extend terminal state
        base_ts = self.terminal_state
        if self.config['system']['use_tensordict']:
            base_ts = base_ts.to_dict()
        base_ts['pickup_done'] = torch.zeros([1, self.problem_size], dtype=torch.bool,
                                             device=self.device)
        if self.config['system']['use_tensordict']:
            self.terminal_state = TensorDict(base_ts, batch_size=1)
        else:
            self.terminal_state = base_ts

    # ------------------------------------------------------------------
    # Static properties reset
    # ------------------------------------------------------------------

    def reset_static_properties(self, n_problems_to_reset, copy_all, indices):
        n = super().reset_static_properties(n_problems_to_reset, copy_all, indices)

        dataset_size = self.reference['positions'].shape[0]
        fixed_dataset_indices = cyclic_indexing(
            (self.episode_counter - n_problems_to_reset) % dataset_size,
            n_problems_to_reset,
            dataset_size,
        )

        pairs = self.reference['pairs'][fixed_dataset_indices]  # (n, max_pairs, 2)
        # Build is_pickup, is_delivery, pair_of masks from pair array
        is_pickup = torch.zeros(n_problems_to_reset, self.problem_size, dtype=torch.bool,
                                device=self.device)
        is_delivery = torch.zeros_like(is_pickup)
        pair_of = torch.full((n_problems_to_reset, self.problem_size), -1,
                             dtype=torch.long, device=self.device)

        for b in range(n_problems_to_reset):
            p = pairs[b]  # (max_pairs, 2)
            for row in range(p.shape[0]):
                pick_idx = p[row, 0].item()
                delv_idx = p[row, 1].item()
                if pick_idx == 0 and delv_idx == 0:
                    continue  # padding row
                if pick_idx < self.problem_size:
                    is_pickup[b, pick_idx] = True
                if delv_idx < self.problem_size:
                    is_delivery[b, delv_idx] = True
                    pair_of[b, delv_idx] = pick_idx

        self.ref_is_pickup[indices] = is_pickup
        self.ref_is_delivery[indices] = is_delivery
        self.ref_pair_of[indices] = pair_of
        return n

    # ------------------------------------------------------------------
    # Dynamic properties reset
    # ------------------------------------------------------------------

    def reset_dynamic_properties(self, indices, n_envs_to_reset):
        super().reset_dynamic_properties(indices, n_envs_to_reset)
        self.pickup_done[indices] = False

        # Delivery nodes are infeasible until their paired pickup is visited.
        if self.extended_output_format:
            is_delv = self.ref_is_delivery[indices].unsqueeze(1)  # (n_reset, 1, n_nodes)
        else:
            is_delv = self.ref_is_delivery[indices]               # (n_reset, n_nodes)
        self.feasible_nodes[indices] &= ~is_delv

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def _step(self, actions, calc_obs=True):
        """Extend VRPTW step with pickup-done tracking."""
        import numpy

        if isinstance(actions, numpy.ndarray) or isinstance(actions, list):
            actions = torch.tensor(actions, dtype=torch.long)
        else:
            actions = actions.to(int)
        actions = actions.clone().view(-1)
        n_actions = len(actions)

        head = self.head.view(-1)
        distance_matrix = self.distance_matrix.view(-1, self.problem_size, self.problem_size)
        demand = self.demand.view(-1, self.problem_size)
        capacity = self.capacity.view(-1)
        acc_returns = self.acc_returns.view(-1)
        current_time = self.current_time.view(-1)
        pickup_done = self.pickup_done.view(-1, self.problem_size)

        # Replicate static data over beams when in extended format
        if self.extended_output_format:
            ref_tw = (self.ref_time_windows
                      .unsqueeze(1).expand(-1, self.n_beams, -1, -1)
                      .reshape(-1, self.problem_size, 2))
            ref_svc = (self.ref_service_times
                       .unsqueeze(1).expand(-1, self.n_beams, -1)
                       .reshape(-1, self.problem_size))
            ref_is_pickup = (self.ref_is_pickup
                             .unsqueeze(1).expand(-1, self.n_beams, -1)
                             .reshape(-1, self.problem_size))
            ref_is_delivery = (self.ref_is_delivery
                               .unsqueeze(1).expand(-1, self.n_beams, -1)
                               .reshape(-1, self.problem_size))
            ref_pair_of_flat = (self.ref_pair_of
                                .unsqueeze(1).expand(-1, self.n_beams, -1)
                                .reshape(-1, self.problem_size))
        else:
            ref_tw = self.ref_time_windows
            ref_svc = self.ref_service_times
            ref_is_pickup = self.ref_is_pickup
            ref_is_delivery = self.ref_is_delivery
            ref_pair_of_flat = self.ref_pair_of

        active_env = actions >= 0
        actions[~active_env] = 0
        dummy_ind = torch.arange(n_actions, dtype=torch.long)
        from_depot = active_env & (head[:n_actions] == DEPOT_LOCATION)
        depot_visit = active_env & (actions == DEPOT_LOCATION)
        visit_site = actions.clone()

        # Reward: travel distance
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

        # Time update
        svc_at_head = ref_svc[dummy_ind, head]
        travel = distance_matrix[dummy_ind, head, visit_site]
        arrive = current_time + svc_at_head + travel
        ready = ref_tw[dummy_ind, visit_site, 0]
        current_time[active_env] = torch.max(arrive, ready)[active_env]
        current_time[depot_visit] = 0.0

        # Mark pickup done when visiting a pickup node
        is_pickup_visit = ref_is_pickup[dummy_ind, visit_site]
        pickup_done[dummy_ind, visit_site] = pickup_done[dummy_ind, visit_site] | (
            is_pickup_visit & active_env
        )

        # Update head, demand, capacity
        head[active_env] = visit_site[active_env].to(self.device)
        demand_removed = demand[dummy_ind, visit_site]
        demand[dummy_ind, visit_site] = 0
        capacity -= demand_removed
        # Expand max_capacity to (n_problems * n_beams,) so the mask aligns.
        if self.extended_output_format:
            max_cap_expanded = (self.max_capacity
                                .unsqueeze(1)
                                .expand(self.n_parallel_problems, self.n_beams)
                                .reshape(-1))
        else:
            max_cap_expanded = self.max_capacity
        capacity[depot_visit] = 1.0 if self.normalize else max_cap_expanded[depot_visit].clone()

        self.non_zero_demand = self.demand > 0

        # Feasibility
        due = ref_tw[:, :, 1]
        svc_h = ref_svc[dummy_ind, head].unsqueeze(-1)
        travel_all = distance_matrix[dummy_ind, head, :]
        arrive_all = current_time.unsqueeze(-1) + svc_h + travel_all
        tw_feasible = arrive_all <= due

        # For deliveries: feasible only if paired pickup is done.
        # Vectorised: for each delivery node j in env b, look up whether
        # pickup_done[b, pair_of[b, j]] is True.
        pickup_done_flat = pickup_done                         # (n_flat, nodes)
        pair_pickup_done = torch.zeros_like(pickup_done_flat)
        # Only need to fill delivery nodes; others stay False (default)
        delivery_mask = ref_is_delivery.bool()                 # (n_flat, nodes)
        if delivery_mask.any():
            # Clamp pair indices to [0, problem_size-1] to avoid invalid indexing
            # (value of -1 for non-delivery nodes is replaced with 0; we only use
            #  results for actual delivery positions anyway)
            safe_pair = ref_pair_of_flat.clamp(min=0)          # (n_flat, nodes)
            env_idx = torch.arange(n_actions, device=pickup_done_flat.device).unsqueeze(1).expand_as(safe_pair)
            looked_up = pickup_done_flat[env_idx, safe_pair]    # (n_flat, nodes)
            pair_pickup_done[delivery_mask] = looked_up[delivery_mask]

        # Delivery feasible only if pickup already done
        delivery_feasible = (~ref_is_delivery) | pair_pickup_done
        # Pickup feasible without additional constraint (capacity handled below)
        self.feasible_nodes = (
            (0 < self.demand) & (self.demand <= self.capacity)
            & tw_feasible.view_as(self.demand)
            & delivery_feasible.view_as(self.demand)
        )
        self.feasible_nodes[..., DEPOT_LOCATION] = True
        self.feasible_nodes.view(-1, self.problem_size)[dummy_ind, head] = False

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

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def get_pos_representation(self, indices=None, extra_global_features=None):
        obs = super().get_pos_representation(indices=indices,
                                             extra_global_features=extra_global_features)
        if indices is None:
            indices = torch.ones(self.n_parallel_problems, dtype=torch.bool)

        n_problems = indices.sum()
        if self.extended_output_format:
            batch_dim = (n_problems, self.n_beams)
        else:
            batch_dim = (n_problems,)

        # ref_is_pickup / ref_is_delivery: (n_parallel, n_nodes) – no beam dim
        is_pick = self.ref_is_pickup[indices].unsqueeze(-1)   # (n, nodes, 1)
        is_delv = self.ref_is_delivery[indices].unsqueeze(-1)  # (n, nodes, 1)
        if self.extended_output_format:
            is_pick = is_pick.unsqueeze(1).expand(*batch_dim, self.problem_size, 1).contiguous()
            is_delv = is_delv.unsqueeze(1).expand(*batch_dim, self.problem_size, 1).contiguous()

        # pickup_done: (n_parallel[, n_beams], n_nodes) – has beam dim in extended mode
        pdone = self.pickup_done[indices]  # (n [, n_beams], n_nodes)
        pdone = pdone.unsqueeze(-1)        # (n [, n_beams], n_nodes, 1)

        extra = {'is_pickup': is_pick, 'is_delivery': is_delv, 'pickup_done': pdone}
        if self.config['system']['use_tensordict']:
            # VRPTW.get_pos_representation() already moved obs to CPU when
            # save_obs_on_gpu=False.  The PDPTW-specific tensors are still on
            # self.device (GPU), so we must bring them to the same device as obs
            # before assigning, to avoid a TensorDict mixed-device error.
            if not self.config['system']['save_obs_on_gpu']:
                extra = {k: v.cpu() for k, v in extra.items()}
            for k, v in extra.items():
                obs[k] = v
        else:
            obs.update(extra)
        return obs

    # ------------------------------------------------------------------
    # State hashing
    # ------------------------------------------------------------------

    def state_to_hashable(self, state):
        base = super().state_to_hashable(state)
        return base + (tuple(state['pickup_done'].cpu().numpy().flatten().tolist()),)
