# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""VRPTW (Vehicle Routing Problem with Time Windows) environment.

Extends the VRP environment to enforce time-window constraints using the
Homberger benchmark data format.  Each customer has a [ready_time, due_date]
time window: the vehicle may arrive early (and wait) but must arrive before
the due_date.  Service at node j finishes at

    finish_time(j) = max(ready_time(j), arrive_time(j)) + service_time(j)

and the vehicle must arrive at j before due_date(j).
"""

from copy import deepcopy

import numpy as np
import torch
from gymnasium import Env as GymEnv
from gymnasium.spaces import Box, Dict
from tensordict import TensorDict

from .vrp import VRP, DEPOT_LOCATION, build_attention_matrix
from .vehicle_routing import RoutingBase
from .utils.general_usage_utils import cyclic_indexing

STATE_PROPERTIES_VRPTW = {'head', 'demand', 'capacity', 'current_time'}


class VRPTW(VRP):
    """VRPTW environment: VRP + per-node time windows.

    Inherits all VRP logic and extends it with:
    - ``time_windows``    : (n_nodes, 2) ready_time / due_date per node
    - ``service_times``   : (n_nodes,) service duration per node
    - ``current_time``    : (batch,) or (batch, n_beams) current clock per vehicle
    """

    def __init__(self, config, **kwargs):
        GymEnv.__init__(self)
        state_properties = list(STATE_PROPERTIES_VRPTW)
        problem_properties = ['demand', 'capacity', 'time_windows', 'service_times']
        prop_types = dict(demand=torch.float32, capacity=torch.float32,
                          time_windows=torch.float32, service_times=torch.float32)
        RoutingBase.__init__(self, config, problem_properties, state_properties,
                             prop_types=prop_types, title='vrptw', **kwargs)

        self.last_go_back_to_depot = self.config['problem_setup']['last_return_to_depot']
        self.extra_car_penalty = 0
        self.unused_capacity_penalty = 0
        if self.config['problem_setup']['minimize_vehicles']:
            self.extra_car_penalty = self.config['problem_setup']['vehicle_penalty'] * self.radius
            self.extra_car_penalty *= self.reward_normalization
            if self.env_type == 'train':
                self.unused_capacity_penalty = self.config['problem_setup']['unused_capacity_penalty']
                self.unused_capacity_penalty *= self.radius * self.reward_normalization

        self._init_constraint_penalty_state()

    def _init_constraint_penalty_state(self):
        ps = self.config.get('problem_setup', {})

        # Keep action selection inside the feasible mask for both train and inference.
        # If an invalid action still arrives from upstream, we project it back to a
        # feasible action; if no feasible action exists, we deactivate that env step.
        self.strict_action_mask_train = bool(ps.get('strict_action_mask_train', True))
        self.strict_action_mask_infer = bool(ps.get('strict_action_mask_infer', True))

        self.use_lagrangian_constraints = bool(ps.get('use_lagrangian_constraints', False))
        self.lagrangian_lr = float(ps.get('lagrangian_lr', 1e-4))
        self.lagrangian_ema = float(ps.get('lagrangian_ema', 0.05))
        self.lagrangian_max = float(ps.get('lagrangian_max', 10.0))

        # PIP (Proactive Infeasibility Prevention) masking:
        # When True, proactively mask actions that would make future nodes permanently
        # infeasible (e.g., for PDPTW: mask pickup p if delivery d_p becomes unreachable).
        self.use_pip_masking = bool(ps.get('use_pip_masking', False))

        self.fixed_constraint_penalties = {
            'late_sum': float(ps.get('penalty_late_sum', 0.0)),
            'late_count': float(ps.get('penalty_late_count', 0.0)),
            'masked_ratio': float(ps.get('penalty_masked_ratio', 0.0)),
            'pair_blocked': float(ps.get('penalty_pair_blocked', 0.0)),
            'depot_with_customer': float(ps.get('penalty_depot_with_customer', 0.0)),
            'vehicle_over_lb': float(ps.get('penalty_vehicle_over_lb', 0.0)),
        }

        self.constraint_targets = {
            'late_sum': float(ps.get('target_late_sum', 0.0)),
            'late_count': float(ps.get('target_late_count', 0.0)),
            'masked_ratio': float(ps.get('target_masked_ratio', 0.0)),
            'pair_blocked': float(ps.get('target_pair_blocked', 0.0)),
            'depot_with_customer': float(ps.get('target_depot_with_customer', 0.0)),
            'vehicle_over_lb': float(ps.get('target_vehicle_over_lb', 0.0)),
        }

        self.lagrangian_state = {
            k: {'lambda': 0.0, 'ema': 0.0}
            for k in self.fixed_constraint_penalties
        }

    def _constraint_weight(self, name):
        lam = self.lagrangian_state[name]['lambda']
        return self.fixed_constraint_penalties[name] + lam

    def _maybe_update_lagrangian(self, name, signal):
        if not self.use_lagrangian_constraints or self.env_type != 'train':
            return
        if signal.numel() == 0:
            return

        signal_mean = float(signal.detach().float().mean().item())
        state = self.lagrangian_state[name]
        state['ema'] = (1.0 - self.lagrangian_ema) * state['ema'] + self.lagrangian_ema * signal_mean
        dual_grad = state['ema'] - self.constraint_targets[name]
        state['lambda'] = float(np.clip(state['lambda'] + self.lagrangian_lr * dual_grad, 0.0, self.lagrangian_max))

    # ------------------------------------------------------------------
    # Space setup
    # ------------------------------------------------------------------

    def set_spaces(self):
        # Delegate to VRP to set all base attributes
        super().set_spaces()

        # Additional VRPTW-specific tensors
        # time windows per node: [ready_time, due_date]
        self.ref_time_windows = torch.zeros([self.n_parallel_problems, self.problem_size, 2],
                                            device=self.device)
        # service times per node
        self.ref_service_times = torch.zeros([self.n_parallel_problems, self.problem_size],
                                             device=self.device)
        # current time for each vehicle (dynamic state)
        self.current_time = torch.zeros(
            self.maybe_extend([self.n_parallel_problems, 1]), device=self.device)
        # Number of vehicle departures from depot in current episode.
        self.vehicle_starts = torch.zeros(
            self.maybe_extend([self.n_parallel_problems, 1]), device=self.device)
        # Demand lower-bound on vehicle count, computed at reset.
        self.vehicle_lb = torch.ones([self.n_parallel_problems], device=self.device)

        # Extend observation space with time-window features
        if not self.stable_baselines_compatibility:
            self.observation_space = Dict({
                **self.observation_space.spaces,
                'tmin'        : Box(low=0, high=np.inf, shape=(*self.batch_dim, self.problem_size, 1)),
                'tmax'        : Box(low=0, high=np.inf, shape=(*self.batch_dim, self.problem_size, 1)),
                'dt'          : Box(low=0, high=np.inf, shape=(*self.batch_dim, self.problem_size, 1)),
                # Dynamic time features (updated every step):
                # time_slack: remaining time until each node's deadline from current time,
                #             normalised by the planning horizon. Conveys urgency.
                'time_slack'  : Box(low=0, high=np.inf, shape=(*self.batch_dim, self.problem_size, 1)),
                # current_time: normalised elapsed time for the current vehicle route.
                #               Broadcast to all nodes so the decoder can use it as a
                #               global context scalar.
                'current_time': Box(low=0, high=np.inf, shape=(*self.batch_dim, self.problem_size, 1)),
            })
        else:
            self.observation_space = Dict({
                **self.observation_space.spaces,
                'tmin'        : Box(low=0, high=np.inf, shape=(self.problem_size, 1)),
                'tmax'        : Box(low=0, high=np.inf, shape=(self.problem_size, 1)),
                'dt'          : Box(low=0, high=np.inf, shape=(self.problem_size, 1)),
                'time_slack'  : Box(low=0, high=np.inf, shape=(self.problem_size, 1)),
                'current_time': Box(low=0, high=np.inf, shape=(self.problem_size, 1)),
            })

        # Extend terminal state
        self.terminal_state = {
            **self.terminal_state,
            'current_time': torch.zeros([1, 1], device=self.device),
            'vehicle_starts': torch.zeros([1, 1], device=self.device),
        }
        if self.config['system']['use_tensordict']:
            self.terminal_state = TensorDict(self.terminal_state, batch_size=1)

    # ------------------------------------------------------------------
    # Static properties reset
    # ------------------------------------------------------------------

    def reset_static_properties(self, n_problems_to_reset, copy_all, indices):
        n = super().reset_static_properties(n_problems_to_reset, copy_all, indices)
        fixed_dataset_indices = getattr(self, 'last_dataset_indices', None)
        if fixed_dataset_indices is None:
            dataset_size = self.reference['positions'].shape[0]
            fixed_dataset_indices = cyclic_indexing(
                (self.episode_counter - n_problems_to_reset) % dataset_size,
                n_problems_to_reset,
                dataset_size,
            )
        tw = self.reference['time_windows'][fixed_dataset_indices].clone()   # (n, n_nodes, 2)
        svc = self.reference['service_times'][fixed_dataset_indices].clone()  # (n, n_nodes)

        # Keep travel time and TW/service on the same scale when distance is normalized.
        if self.normalize:
            tw = tw * self.reward_normalization
            svc = svc * self.reward_normalization

        self.ref_time_windows[indices] = tw.float()
        self.ref_service_times[indices] = svc.float()
        return n

    # ------------------------------------------------------------------
    # Dynamic properties reset
    # ------------------------------------------------------------------

    def reset_dynamic_properties(self, indices, n_envs_to_reset):
        super().reset_dynamic_properties(indices, n_envs_to_reset)
        self.current_time[indices] = 0.0
        self.vehicle_starts[indices] = 0.0

        total_demand = self.initial_demand[indices].sum(dim=-1)
        cap = self.max_capacity[indices].clamp(min=1e-8)
        self.vehicle_lb[indices] = torch.ceil(total_demand / cap).clamp(min=1.0)

        # Apply initial time-window feasibility: block nodes that cannot be
        # reached from the depot before their due_date.
        ref_tw_idx  = self.ref_time_windows[indices]    # (n_reset, n_nodes, 2)
        ref_svc_idx = self.ref_service_times[indices]   # (n_reset, n_nodes)
        due         = ref_tw_idx[:, :, 1]               # (n_reset, n_nodes)
        svc_depot   = ref_svc_idx[:, DEPOT_LOCATION]    # (n_reset,)

        if self.extended_output_format:
            dist_depot = self.distance_matrix[indices][:, :, DEPOT_LOCATION, :]  # (n_reset, n_beams, n_nodes)
            arrive = svc_depot.unsqueeze(-1).unsqueeze(-1) + dist_depot
            tw_ok  = arrive <= due.unsqueeze(1)          # broadcast to (n_reset, n_beams, n_nodes)
        else:
            dist_depot = self.distance_matrix[indices][:, DEPOT_LOCATION, :]     # (n_reset, n_nodes)
            arrive = svc_depot.unsqueeze(-1) + dist_depot
            tw_ok  = arrive <= due                       # (n_reset, n_nodes)

        self.feasible_nodes[indices] &= tw_ok

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def _step(self, actions, calc_obs=True):
        """Extend VRP step with time-window updates."""
        import numpy

        target_device = self.head.device

        if isinstance(actions, numpy.ndarray) or isinstance(actions, list):
            actions = torch.tensor(actions, dtype=torch.long, device=target_device)
        else:
            actions = actions.to(device=target_device, dtype=torch.long)
        actions = actions.clone().view(-1)
        n_actions = len(actions)

        head = self.head.view(-1)
        distance_matrix = self.distance_matrix.view(-1, self.problem_size, self.problem_size)
        demand = self.demand.view(-1, self.problem_size)
        capacity = self.capacity.view(-1)
        acc_returns = self.acc_returns.view(-1)
        current_time = self.current_time.view(-1)
        vehicle_starts = self.vehicle_starts.view(-1)

        # Flat time-window / service-time views (replicated over beams if needed)
        if self.extended_output_format:
            ref_tw = (self.ref_time_windows
                      .unsqueeze(1)
                      .expand(-1, self.n_beams, -1, -1)
                      .reshape(-1, self.problem_size, 2))
            ref_svc = (self.ref_service_times
                       .unsqueeze(1)
                       .expand(-1, self.n_beams, -1)
                       .reshape(-1, self.problem_size))
        else:
            ref_tw = self.ref_time_windows    # (n_parallel, n_nodes, 2)
            ref_svc = self.ref_service_times  # (n_parallel, n_nodes)

        active_env = actions >= 0
        actions[~active_env] = 0
        dummy_ind = torch.arange(n_actions, dtype=torch.long, device=target_device)

        feasible_flat = self.feasible_nodes.view(-1, self.problem_size)
        action_is_feasible = feasible_flat[dummy_ind, actions]
        invalid_active = active_env & (~action_is_feasible)
        if invalid_active.any():
            strict_mask = self.strict_action_mask_train if self.env_type == 'train' else self.strict_action_mask_infer
            feasible_rows = feasible_flat[invalid_active]
            has_feasible = feasible_rows.any(dim=-1)

            if has_feasible.any():
                fallback = feasible_rows[has_feasible].float().argmax(dim=-1).to(actions.dtype)
                invalid_idx = invalid_active.nonzero(as_tuple=False).squeeze(-1)
                actions[invalid_idx[has_feasible]] = fallback

            if (~has_feasible).any():
                invalid_idx = invalid_active.nonzero(as_tuple=False).squeeze(-1)
                no_feasible_idx = invalid_idx[~has_feasible]
                if strict_mask:
                    # No legal continuation: deactivate these envs for this step.
                    actions[no_feasible_idx] = -1
                else:
                    actions[no_feasible_idx] = DEPOT_LOCATION

            # Recompute activity after correction/deactivation.
            active_env = actions >= 0
            actions[~active_env] = 0

        from_depot = active_env & (head[:n_actions] == DEPOT_LOCATION)
        depot_visit = active_env & (actions == DEPOT_LOCATION)
        visit_site = actions.clone()
        new_vehicle_start = from_depot & (visit_site != DEPOT_LOCATION)
        vehicle_starts[new_vehicle_start] += 1.0

        # Travel distance reward
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

        # --- Time update ---
        # Service time at current node (head) finishes, then travel to next node
        svc_at_head = ref_svc[dummy_ind, head]            # service time at old position
        travel = distance_matrix[dummy_ind, head, visit_site]  # travel time = travel dist (unit speed)
        arrive = current_time + svc_at_head + travel
        # Wait if arrived before ready_time
        ready = ref_tw[dummy_ind, visit_site, 0]
        current_time[active_env] = torch.max(arrive, ready)[active_env]
        # Reset clock when going back to depot
        current_time[depot_visit] = 0.0

        # --- Standard VRP updates ---
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

        # Feasibility: demand constraint AND time-window constraint
        due = ref_tw[:, :, 1]                              # (n, n_nodes) due dates
        svc_h = ref_svc[dummy_ind, head].unsqueeze(-1)     # (n, 1)  service at current head
        travel_all = distance_matrix[dummy_ind, head, :]   # (n, n_nodes) all travel times
        arrive_all = current_time.unsqueeze(-1) + svc_h + travel_all
        tw_feasible = arrive_all <= due                    # (n, n_nodes)

        # Soft constraint signals (used for reward shaping and lagrangian updates)
        selected_due = ref_tw[dummy_ind, visit_site, 1]
        lateness = torch.clamp(current_time - selected_due, min=0.0)
        lateness = torch.where(active_env, lateness, torch.zeros_like(lateness))
        late_count = (lateness > 0).float()

        candidate_nodes = (0 < demand) & (demand <= capacity.unsqueeze(-1))
        candidate_nodes[:, DEPOT_LOCATION] = False
        masked_by_tw = candidate_nodes & (~tw_feasible)
        candidate_count = candidate_nodes.float().sum(dim=-1).clamp(min=1.0)
        masked_ratio = masked_by_tw.float().sum(dim=-1) / candidate_count
        masked_ratio = torch.where(active_env, masked_ratio, torch.zeros_like(masked_ratio))

        feasible_non_depot = candidate_nodes & tw_feasible
        has_feasible_non_depot = feasible_non_depot.any(dim=-1)
        depot_with_customer = (depot_visit & has_feasible_non_depot).float()
        depot_with_customer = torch.where(
            active_env,
            depot_with_customer,
            torch.zeros_like(depot_with_customer),
        )

        constraint_penalty = (
            self._constraint_weight('late_sum') * lateness
            + self._constraint_weight('late_count') * late_count
            + self._constraint_weight('masked_ratio') * masked_ratio
            + self._constraint_weight('depot_with_customer') * depot_with_customer
        )

        reward_flat = reward.view(-1)
        reward_flat[active_env] -= constraint_penalty[active_env]
        acc_returns[active_env] -= constraint_penalty[active_env]

        active_mask = active_env.nonzero(as_tuple=False).squeeze(-1)
        if active_mask.numel() > 0:
            self._maybe_update_lagrangian('late_sum', lateness[active_mask])
            self._maybe_update_lagrangian('late_count', late_count[active_mask])
            self._maybe_update_lagrangian('masked_ratio', masked_ratio[active_mask])
            self._maybe_update_lagrangian('depot_with_customer', depot_with_customer[active_mask])

        info.update({
            'late_sum': lateness,
            'late_count': late_count,
            'masked_ratio': masked_ratio,
            'depot_with_customer': depot_with_customer,
            'hard_constraint_override_ratio': invalid_active.float().mean(),
            'hard_constraint_no_feasible_ratio': (
                (invalid_active & (~feasible_flat[dummy_ind].any(dim=-1))).float().mean()
            ),
            'constraint_penalty': constraint_penalty,
            'lagrangian_lambda_late_sum': self.lagrangian_state['late_sum']['lambda'],
            'lagrangian_lambda_late_count': self.lagrangian_state['late_count']['lambda'],
            'lagrangian_lambda_masked_ratio': self.lagrangian_state['masked_ratio']['lambda'],
            'lagrangian_lambda_depot_with_customer': self.lagrangian_state['depot_with_customer']['lambda'],
        })

        self.feasible_nodes = (
            (0 < self.demand) & (self.demand <= self.capacity) & tw_feasible.view_as(self.demand)
        )
        self.feasible_nodes[..., DEPOT_LOCATION] = True
        self.feasible_nodes.view(-1, self.problem_size)[dummy_ind, head] = False

        # ------------------------------------------------------------------
        # PIP (Proactive Infeasibility Prevention) for VRPTW:
        # Optionally mask candidate node j if visiting j would cause at least
        # one currently-feasible must-visit node k to miss its time window.
        # The check: after visiting j (time_at_j), can we reach k from j?
        # Only enabled when use_pip_masking is True (default: False).
        # ------------------------------------------------------------------
        if self.use_pip_masking:
            # Candidate nodes: demand > 0, within capacity, TW feasible, not depot
            cand = self.feasible_nodes.view(-1, self.problem_size).clone()
            cand[:, DEPOT_LOCATION] = False
            if cand.any():
                ready = ref_tw[:, :, 0]                   # (n, nodes)
                due   = ref_tw[:, :, 1]                   # (n, nodes)
                # Arrival time at each candidate j from current head
                arrive_j = current_time.unsqueeze(-1) + svc_h + travel_all  # (n, nodes)
                time_at_j_done = torch.max(arrive_j, ready) + ref_svc       # (n, nodes) time after svc at j

                # From j, can we still reach each remaining node k?
                # (n, nodes_j, nodes_k): travel from j to k
                travel_j_to_k = distance_matrix  # (n, nodes, nodes)
                # time_at_j_done[:, j] + travel_j_to_k[:, j, k]  vs due[:, k]
                # arrive_from_j: (n, nodes_j, nodes_k)
                arrive_from_j = time_at_j_done.unsqueeze(-1) + travel_j_to_k  # (n, nodes_j, nodes_k)
                reachable_from_j = arrive_from_j <= due.unsqueeze(-2)          # (n, nodes_j, nodes_k)

                # must_visit: currently feasible non-depot customer nodes
                must_visit = cand  # (n_flat, nodes_k)
                # pip_blocked[j]: some must_visit k is reachable from head but NOT from j
                reachable_from_head = tw_feasible  # (n_flat, nodes_k)
                # For each j: any k that was reachable but becomes unreachable after j?
                becomes_infeasible = (reachable_from_head.unsqueeze(-2)
                                      & ~reachable_from_j
                                      & must_visit.unsqueeze(-2))  # (n, nodes_j, nodes_k)
                pip_blocks_others = becomes_infeasible.any(dim=-1)  # (n_flat, nodes_j)

                # Only mask customer (non-depot) nodes
                pip_blocks_others[:, DEPOT_LOCATION] = False
                # Apply to feasible_nodes
                fn_flat = self.feasible_nodes.view(-1, self.problem_size)
                fn_flat[pip_blocks_others] = False
                self.feasible_nodes = fn_flat.view_as(self.feasible_nodes)
                info['pip_blocked_vrptw_ratio'] = pip_blocks_others.float().mean()

        dones = self.non_zero_demand.sum(dim=-1) == 0
        if self.last_go_back_to_depot:
            dones = dones & (head == DEPOT_LOCATION).view_as(dones)
        dones = dones | ~active_env.view_as(dones)

        done_active = dones.view(-1) & active_env
        vehicle_over_lb = torch.zeros_like(reward_flat)
        vehicle_over_lb_penalty = torch.zeros_like(reward_flat)
        if done_active.any():
            if self.extended_output_format:
                vehicle_lb = self.vehicle_lb.unsqueeze(1).expand(
                    self.n_parallel_problems, self.n_beams
                ).reshape(-1)
            else:
                vehicle_lb = self.vehicle_lb
            vehicle_over_lb[done_active] = torch.clamp(
                vehicle_starts[done_active] - vehicle_lb[done_active],
                min=0.0,
            )
            vehicle_over_lb_penalty = self._constraint_weight('vehicle_over_lb') * vehicle_over_lb
            reward_flat[done_active] -= vehicle_over_lb_penalty[done_active]
            acc_returns[done_active] -= vehicle_over_lb_penalty[done_active]
            self._maybe_update_lagrangian('vehicle_over_lb', vehicle_over_lb[done_active])

        reward = reward_flat.view(*self.batch_dim)

        info.update({
            'vehicle_over_lb': vehicle_over_lb,
            'vehicle_over_lb_penalty': vehicle_over_lb_penalty,
            'lagrangian_lambda_vehicle_over_lb': self.lagrangian_state['vehicle_over_lb']['lambda'],
        })

        obs = None
        if calc_obs:
            obs = self.get_pos_representation()
        if self.config['train']['method'] == 'ppo':
            reward = reward.detach().cpu().numpy()
            dones = dones.detach().cpu().numpy()
        return obs, reward, dones, info

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def get_pos_representation(self, indices=None, extra_global_features=None):
        """Extend base VRP observation with time-window node features."""
        obs = super().get_pos_representation(indices=indices,
                                             extra_global_features=extra_global_features)

        if indices is None:
            indices = torch.ones(self.n_parallel_problems, dtype=torch.bool)

        n_problems = indices.sum()
        if self.extended_output_format:
            batch_dim = (n_problems, self.n_beams)
        else:
            batch_dim = (n_problems,)

        # Time-window features per node, normalized by the time horizon (max due_date)
        tw = self.ref_time_windows[indices]         # (n, n_nodes, 2)
        svc = self.ref_service_times[indices]       # (n, n_nodes)
        horizon = tw[..., 1].max(dim=-1, keepdim=True).values.clamp(min=1.0)  # (n, 1)

        # Fetch current time before extending tw/horizon so shapes are consistent.
        # current_time: (n, n_beams, 1) in extended format; (n, 1) otherwise.
        ct = self.current_time[indices]

        if self.extended_output_format:
            tw = tw.unsqueeze(1).expand(*batch_dim, self.problem_size, 2).contiguous()
            svc = svc.unsqueeze(1)
            horizon = horizon.unsqueeze(1)   # (n, 1, 1)

        # Static TW features (normalised by horizon)
        tmin = (tw[..., 0:1] / horizon.unsqueeze(-1))   # (*batch_dim, n_nodes, 1)
        tmax = (tw[..., 1:2] / horizon.unsqueeze(-1))
        dt   = tmax - tmin

        # Dynamic feature: remaining time until each node's deadline from the current
        # vehicle position, normalised by the planning horizon.
        # due shape: (*batch_dim, n_nodes); ct shape: (n,[n_beams,] 1);
        # horizon shape: (n,[1,] 1) – all broadcast correctly via PyTorch rules.
        due = tw[..., 1]                                              # (*batch_dim, n_nodes)
        time_slack = ((due - ct) / horizon).clamp(min=0).unsqueeze(-1)  # (*batch_dim, n_nodes, 1)

        # Global context: current (elapsed) time normalised by horizon, broadcast to
        # all nodes so the decoder can use it as a per-env scalar.
        ct_norm = ct / horizon                                         # (n, [n_beams,] 1)
        current_time_feat = (ct_norm
                             .unsqueeze(-2)
                             .expand(*batch_dim, self.problem_size, 1)
                             .contiguous())                            # (*batch_dim, n_nodes, 1)

        tw_obs = {
            'tmin'        : tmin,
            'tmax'        : tmax,
            'dt'          : dt,
            'time_slack'  : time_slack,
            'current_time': current_time_feat,
        }

        if self.config['system']['use_tensordict']:
            for k, v in tw_obs.items():
                obs[k] = v
            if not self.config['system']['save_obs_on_gpu']:
                obs = obs.to('cpu')
        else:
            obs.update(tw_obs)

        return obs

    # ------------------------------------------------------------------
    # State hashing (for tree search)
    # ------------------------------------------------------------------

    def state_to_hashable(self, state):
        base = super().state_to_hashable(state)
        return base + (np.round(state['current_time'].item(), 4),)
