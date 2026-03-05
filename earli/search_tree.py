# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import torch
from tensordict import TensorDict


def fill_to_k_actions(in_mask, k, n_actions_from_head=None):
    # fill up the last elements per problem to ensure each problem has k True masked entries
    final_mask = in_mask.clone()
    delta = k - final_mask.sum(dim=[-1, -2])
    ind = delta > 0
    delta = delta[ind]
    if n_actions_from_head is not None:
        n_actions_from_head[ind, -1] += delta
    max_fill = delta.max()
    mask = (torch.arange(max_fill, device=max_fill.device).flip(0).unsqueeze(0) < delta.unsqueeze(1))
    final_mask[ind, -1, -max_fill:] = final_mask[ind, -1, -max_fill:] | mask
    return final_mask, n_actions_from_head


class SearchTree(object):
    # this store is used to store the state of the search tree and relevant information, such as the frontier
    # the search tree is a tree of states, each state is a node in the tree

    def __init__(self, config, env, problem_uid, roots, gae_lambda=None):
        self.config = config
        self.k_beams = env.n_beams
        self.env = env
        self.problem_size = self.env.problem_size
        self.n_problems = roots.shape[0]
        # The first set of observations contains effectively  the root at 0 only.
        self.initial_mask = torch.zeros((1, self.k_beams), dtype=torch.bool)
        self.initial_mask[:, 0] = True
        self.gae_lambda = 1 if gae_lambda is None else gae_lambda
        max_steps = 2 * (self.problem_size + 2)
        self.buffer = TensorDict({}, batch_size=(max_steps, *roots.shape))
        self.buffer[0] = roots
        self.frontier_index = 0
        self.parent = - torch.ones((max_steps, *roots.shape), dtype=torch.int)
        self.action_head_source = - torch.ones((max_steps, *roots.shape), dtype=torch.int)
        self.done_problems = torch.zeros(roots.shape[0], dtype=torch.bool)
        self.cuopt_route, self.ref_node, self.top_action_index = None, None, None
        self.step_count = 0
        self.action_count = 0
        self.duplication_count = 0

    def reset_frontier(self):
        self.frontier_index += 1

    def dones(self):
        self.done_problems = self.done_problems & self.buffer[self.frontier_index]['dones'].all(dim=-1)
        return self.done_problems

    def build_buffer_from_best_traj(self, problem_output_buffer, t, beam_index, action_index, dummy_ind, observations):
        rewards = problem_output_buffer['rewards'][dummy_ind, action_index]
        dones = problem_output_buffer['dones'][dummy_ind, action_index]
        move_action = problem_output_buffer['actions'][dummy_ind, action_index]
        values = problem_output_buffer['values'][dummy_ind, beam_index]  # the value of the best beam
        unmasked_heads = torch.cat([self.initial_mask, ~problem_output_buffer['dones'][:-1]])
        buffer_data = TensorDict(source={'rewards'     : rewards,
                                         'values'      : values,
                                         'dones'       : dones,
                                         'observations': observations}, batch_size=(t + 1,))
        if observations['feasible_nodes'].dim() > 2:
            feasible_actions_from_head = unmasked_heads * observations['feasible_nodes'].sum(dim=-1)
            buffer_data['feasible_actions_from_head'] = feasible_actions_from_head
        return buffer_data, move_action, rewards

    @property
    def terminal_actions(self):
        terminal_actions = self.buffer['dones'] & (self.buffer['actions'] >= 0)
        return terminal_actions

    def expand_and_update_tree(self, dones, log_prob, observations, policy_logits,
                               rewards, simulation_instruction, values, info=None, light_buffer=False):
        # NOTE: log_prob, actions, rewards stored at step i are properties of the transition from i-1 to i (unlike SB3,
        # in which they are properties of the transition from i to i+1)!
        # NOTE: log_prob, actions, rewards stored at step i are properties of the transition from state/observation i to i+1.
        # a done is recevied if the resulting state at i+1 is terminal (not if state i is terminal).
        # Dones are not registered explicitly, and therefore the last entry in the buffer is always a pre-terminal state.

        n_actions_from_head = simulation_instruction.n_actions_from_head
        assert (self.k_beams == n_actions_from_head.sum(dim=-1)).all(), 'something is wrong with the number of actions'
        dummy_ind = torch.arange(self.k_beams, dtype=torch.int)
        dummy_ind = dummy_ind.unsqueeze(0).expand(self.n_problems, -1)
        active_actions = simulation_instruction.valid_actions_mask
        actions = simulation_instruction.actions[active_actions].view(*dones.shape)
        data = dict(rewards=rewards, dones=dones, actions=actions)
        self.buffer[self.frontier_index] = data
        parents = torch.repeat_interleave(dummy_ind.flatten(), n_actions_from_head.flatten())
        action_from_head = simulation_instruction.valid_actions_mask.nonzero()[:, 1].view(*dones.shape)
        parents = parents.view(self.n_problems, self.k_beams)
        self.action_head_source[self.frontier_index] = action_from_head
        assert (action_from_head == parents).all()
        parents[(actions < 0)] = -1
        self.parent[self.frontier_index + 1] = parents


    def build_training_data_from_history(self, obs_history, actions_history,
                                          policy_logits_history, rewards_history,
                                          sampler):
        """Extract PPO training data from the best trajectory found by tree search.

        Parameters
        ----------
        obs_history : list[TensorDict]
            Observations (on CPU) at each step, shape ``(n_problems, k_beams, ...)``.
        actions_history : list[Tensor]
            Actions (on CPU) taken by each *destination* beam at each step,
            shape ``(n_problems, k_beams)``.
        policy_logits_history : list[Tensor]
            Policy logits (on CPU) for each beam at each step,
            shape ``(n_problems, k_beams, n_nodes)``.
        rewards_history : list[Tensor]
            Rewards (on CPU) received by each beam at each step (same indexing as
            ``actions_history``), shape ``(n_problems, k_beams)``.
        sampler : Sampler
            Used to compute per-node log-probabilities from stored logits.

        Returns
        -------
        TensorDict
            Concatenated training data for all problems with keys:
            ``observations``, ``actions``, ``log_prob``, ``rewards``,
            ``returns``.  ``observations`` is itself a TensorDict.
        """
        terminal_actions = self.terminal_actions          # (max_steps, n, k)
        terminal_actions_positions = terminal_actions.nonzero()  # (M, 3): step,prob,beam

        if terminal_actions_positions.shape[0] == 0:
            # No terminal actions found (degenerate game); return empty buffer.
            return TensorDict({}, batch_size=(0,))

        max_t, n_problems, k_beams = self.buffer['actions'].shape

        # ---- find best beam per problem (mirrors light_buffer=True logic) ----
        terminal_states_returns = self.buffer['rewards'].sum(dim=0)  # (n, k)
        _, best_return_index = terminal_states_returns.max(dim=1)    # (n,)

        # Compute first terminal step per (problem, beam).
        # We replicate the +1 shift from backpropagate_and_fill_buffer so that
        # first_true_indices[i, k] == actual_terminal_step + 1 (= trajectory length).
        # Non-terminal beams keep the default value (max_t - 1); they are never
        # selected as the best beam under normal circumstances.
        terminal_indices = terminal_actions_positions.transpose(1, 0).unbind()
        first_true_indices = torch.full(
            (n_problems, k_beams), max_t - 1, dtype=torch.long,
            device=terminal_states_returns.device,
        )
        shifted_step = terminal_actions_positions[:, 0] + 1   # step + 1 = traj length
        first_true_indices[terminal_indices[1], terminal_indices[2]] = shifted_step

        if self.config['problem_setup']['minimize_vehicles']:
            composite_key = (terminal_actions_positions[:, 1] * k_beams
                             + terminal_actions_positions[:, 2])
            sort_indices = torch.argsort(composite_key)
            sorted_tp = terminal_actions_positions[sort_indices]
            vehicles_per_beam = sorted_tp[:, 0].reshape(n_problems, k_beams)
            best_solution_per_problem, _ = self._select_best_beam(
                terminal_states_returns, vehicles_per_beam)
        else:
            best_solution_per_problem = best_return_index  # (n,)

        # Extract observation keys once – all problems share the same observation space.
        obs_keys = list(obs_history[0].keys()) if obs_history else []

        # ---- extract per-problem training trajectory ----
        output_buffers = []
        for i in range(n_problems):
            best_beam = int(best_solution_per_problem[i].item())
            # T = trajectory length (actual_terminal_step + 1)
            T = int(first_true_indices[i, best_beam].item())

            # Guard: trajectory must have at least one step.
            if T <= 0:
                continue

            # Actual terminal step index (0-based) = T - 1
            t_actual = T - 1

            # The parent of best_beam at the terminal step is action_head_source.
            if t_actual > 0:
                terminal_beam = int(
                    self.action_head_source[t_actual, i, best_beam].item()
                )
            else:
                # Single-step episode: the only beam is its own parent.
                terminal_beam = best_beam

            terminal_action_t = torch.tensor(best_beam, dtype=torch.int)
            beam_index, action_index, _ = self.get_trajectory_upto_head(
                t_actual, i,
                terminal_beam=terminal_beam,
                terminal_action=terminal_action_t,
                calculate_buffer=False,
            )
            # beam_index  (T,): source beam at each step 0..t_actual
            # action_index (T,): destination-beam / action index at each step

            T_eff = beam_index.shape[0]   # == T
            if T_eff == 0:
                continue

            # Guard: history must be long enough.
            if T_eff > len(obs_history):
                continue

            # ---- observations ----
            # Each obs_history[t] is a TensorDict with batch_size (n_problems, k_beams).
            # Index to get a scalar-batch TensorDict for the source beam.
            obs_seq_list = []
            for t in range(T_eff):
                b = int(beam_index[t].item())
                obs_seq_list.append(obs_history[t][i, b])   # batch_size ()

            # Stack into a TensorDict with batch_size (T_eff,).
            # We do it manually to avoid relying on torch.stack for TensorDicts.
            # obs_keys is shared across all problems (same observation space).
            obs_stacked = TensorDict(
                {k: torch.stack([obs_seq_list[t][k] for t in range(T_eff)])
                 for k in obs_keys},
                batch_size=(T_eff,),
            )

            # ---- actions (destination nodes) ----
            actions_seq = torch.stack([
                actions_history[t][i, int(action_index[t].item())]
                for t in range(T_eff)
            ])  # (T_eff,)

            # ---- rewards ----
            rewards_seq = torch.stack([
                rewards_history[t][i, int(action_index[t].item())]
                for t in range(T_eff)
            ]).float()   # (T_eff,)

            # ---- old log-probs: P(destination node | source-beam policy) ----
            log_probs_list = []
            for t in range(T_eff):
                b = int(beam_index[t].item())
                logits_t   = policy_logits_history[t][i, b]              # (n_nodes,)
                feasible_t = obs_history[t]['feasible_nodes'][i, b]      # (n_nodes,)
                action_t   = actions_seq[t].unsqueeze(0)                 # (1,)
                with torch.no_grad():
                    _, lp, _ = sampler.sample(
                        logits_t.unsqueeze(0),
                        feasible_t.unsqueeze(0),
                        action=action_t,
                    )
                log_probs_list.append(lp.squeeze(0))
            log_probs_seq = torch.stack(log_probs_list).float()  # (T_eff,)

            returns_seq = rewards_seq.flip(0).cumsum(0).flip(0)  # (T_eff,)

            output_buffers.append(TensorDict({
                'observations': obs_stacked,
                'actions'     : actions_seq,
                'log_prob'    : log_probs_seq,
                'rewards'     : rewards_seq,
                'returns'     : returns_seq,
            }, batch_size=(T_eff,)))

        if not output_buffers:
            return TensorDict({}, batch_size=(0,))
        return torch.cat(output_buffers)

    def backpropagate_and_fill_buffer(self, detailed_log=0, light_buffer=False):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        output_buffers = []
        optimal_route = []
        all_routes = []
        terminal_actions = self.terminal_actions
        terminal_actions_positions = terminal_actions.nonzero()
        terminal_actions_beam_source = torch.cat([terminal_actions_positions[:, :2],
                                                  self.action_head_source[terminal_actions].unsqueeze(-1)], dim=1)
        if light_buffer:
            terminal_states_returns = self.buffer['rewards'].sum(dim=0)
            terminal_actions_positions[:, 0] += 1
            best_return, best_return_index = terminal_states_returns.max(dim=1)
            max_t, n_problems, k_beams, = self.buffer['actions'].shape
            actions = self.buffer['actions'].permute(1, 2, 0)
            actions = torch.cat([torch.zeros([actions.shape[0], actions.shape[1], 1], dtype=actions.dtype, device=actions.device),
                                 actions], dim=2)

            # # Create a tensor to hold the indices of the first True value for each (i, j) pair
            terminal_indices = terminal_actions_positions.transpose(1,0).unbind()
            first_true_indices = torch.full((n_problems, k_beams), max_t - 1, dtype=torch.long, device=actions.device)
            first_true_indices[terminal_indices[1], terminal_indices[2]] = terminal_indices[0]

            if self.config['problem_setup']['minimize_vehicles']:
                # Sort terminal_actions_positions lexicographically by problem_id & beam_id
                composite_key = terminal_actions_positions[:, 1] * k_beams + terminal_actions_positions[:, 2]
                sort_indices = torch.argsort(composite_key)
                sorted_terminal_positions = terminal_actions_positions[sort_indices]
                vehicles_per_beam = sorted_terminal_positions[:,0].reshape(n_problems, k_beams)
                best_solution_per_problem, best_vehicles = self._select_best_beam(terminal_states_returns, vehicles_per_beam)
            else:
                best_solution_per_problem, best_vehicles = best_return_index, None

            all_routes = []
            for i in range(n_problems):
                problem_list = []
                for j in range(k_beams):
                    # Find the index of the first `True` entry in `terminal_actions[i, j]`
                    first_true_index = first_true_indices[i,j]
                    # Slice the `actions[i, j]` tensor up to this index and convert to list
                    action_list = actions[i, j, :first_true_index + 1]
                    problem_list.append(action_list)
                all_routes.append(problem_list)
            optimal_route = [routes[ind] for routes, ind in zip(all_routes, best_solution_per_problem)]
        else:
            # we also sum up the rewards of the terminal states, but it is more complex
            acc_returns = self.buffer['observations']['acc_returns'].squeeze(-1)[terminal_actions_beam_source.unbind(dim=-1)]
            terminal_states_returns = (acc_returns + self.buffer['rewards'][terminal_actions])
            max_positions = grouped_argmax_index(terminal_states_returns, terminal_actions_positions[:, 1],
                                                 dim_size=self.n_problems)
        max_lengths = grouped_max(terminal_actions_positions[:, 0], terminal_actions_positions[:, 1],
                                  dim_size=self.n_problems)
        if light_buffer:
            info = {'best_return'  : best_return,
                    'num_vehicles' : best_vehicles,
                    'data_samples' : max_lengths,
                    'optimal_route': optimal_route,
                    'all_routes'   : all_routes,
                    'forward_iters': max_lengths,  # not really, but it isn't important
                    'env_steps'    : max_lengths,  # not really, but it isn't important
                    'forward_passes': max_lengths,  # calls to model.forward (=env_steps)
                    }
            return torch.empty([0]), info
        # the best trajectory termination point ( trajectory length, problem index, beam index):
        return_pos = terminal_actions_positions[max_positions]
        for i in range(self.n_problems):
            t = return_pos[i, 0]
            t_last = max_lengths[i]
            terminal_action_index = return_pos[i, 2]
            terminal_beam = self.action_head_source[t, i, terminal_action_index]
            problem_output_buffer = self.buffer[:t + 1, i]
            if self.buffer_type in ('kppo', 'macro_state'):
                beam_index, action_index, _ = self.get_trajectory_upto_head(t, i, terminal_beam=terminal_beam,
                                                                            terminal_action=terminal_action_index,
                                                                            calculate_buffer=False)
                dummy_ind = torch.arange(t + 1, dtype=int)
                observations = problem_output_buffer['observations'][dummy_ind, beam_index]
                buffer_data, move_action, rewards = self.build_buffer_from_best_traj(
                        problem_output_buffer, t, beam_index,
                        action_index, dummy_ind, observations)
                log_prob = problem_output_buffer['log_prob'][dummy_ind, action_index]
                actions = move_action
                optimal_route.append(torch.cat([torch.tensor([0]), actions]))
                if detailed_log >= 1:
                    extended_problem_buffer = self.buffer[:t_last + 1, i]['actions']
                    # problem_terminals: n_beams x (t, i, a_i)
                    problem_terminals = terminal_actions_positions[terminal_actions_positions[:, 1] == i]
                    beams = []
                    for k in range(problem_terminals.shape[0]):
                        curr_t = problem_terminals[k, 0]
                        term_idx = problem_terminals[k, 2]
                        beam = self.action_head_source[curr_t, i, term_idx]
                        a_indices = self.get_trajectory_upto_head(
                                curr_t, i, terminal_beam=beam, terminal_action=term_idx,
                                calculate_buffer=False)[1]
                        beams.append([0] + extended_problem_buffer[
                            torch.arange(curr_t + 1, dtype=int), a_indices].tolist())
                    all_routes.append(beams)

                buffer_data.update({'actions' : actions,
                                    'log_prob': log_prob})
                rewards = rewards.unsqueeze(-1)
            returns = rewards.flip(0).cumsum(dim=0).flip(0)
            buffer_data['returns'] = returns
            output_buffers.append(buffer_data)
        info = {'best_return'  : torch.tensor([problem_data['returns'][0] for problem_data in output_buffers]),
                'data_samples' : torch.tensor([problem_data['returns'].shape[0] for problem_data in output_buffers]),
                'optimal_route': optimal_route,
                'all_routes'   : all_routes,
                # how many loops we had in this play (=tree.step_count)
                'forward_iters': torch.tensor([problem_data.batch_size[0] for problem_data in output_buffers]),
                # how many steps we simulated in the env (~iters*n*K) (~env_steps_count/n_parallel)
                # (this is sometimes less than T*K, since some beams may be pruned near the end; yet we use T*K for simplicity)
                'env_steps'    : torch.tensor([problem_data['observations'].numel() for problem_data in output_buffers]),
                }
        # calls to model.forward (=env_steps)
        info['forward_passes'] = info['env_steps']
        output_buffers = torch.cat(output_buffers)
        output_buffers = update_gaes(output_buffers, gamma=1, lmbda=self.gae_lambda)
        return output_buffers, info

    def get_trajectory_upto_head(self, t, problem_ind, terminal_beam, terminal_action, calculate_buffer=True):
        beam_index = torch.zeros(t + 1, dtype=int)
        beam_index[-1] = terminal_beam
        for j in range(t - 1, -1, -1):
            beam_index[j] = self.parent[j + 1, problem_ind, beam_index[j + 1]]
        # a quick has based on the fact that the last action is always deterministic
        action_index = torch.cat([beam_index[1:], terminal_action.unsqueeze(0)], dim=0)
        if calculate_buffer:
            dummy_ind = torch.arange(t + 1, dtype=int)
            reference_buffer = self.buffer[:t + 1, problem_ind]
            output_buffer = reference_buffer[dummy_ind, beam_index]
            output_buffer['rewards'] = reference_buffer['rewards'][dummy_ind, action_index]
            output_buffer['log_prob'] = reference_buffer['log_prob'][dummy_ind, action_index]
            output_buffer['dones'] = reference_buffer['dones'][dummy_ind, action_index]
            output_buffer['actions'] = reference_buffer['actions'][dummy_ind, action_index]
            output_buffer['returns'] = output_buffer['rewards'].flip(0).cumsum(dim=0).flip(0)
        else:
            output_buffer = None
        return beam_index, action_index, output_buffer

    def _select_best_beam(self, terminal_states_returns, num_vehicles_per_beam):
        """Selects the best beam index for each problem based on config."""
        # Find the minimum vehicle count for each problem
        min_vehicles, _ = num_vehicles_per_beam.min(dim=1, keepdim=True)

        # Identify beams that achieve the minimum vehicle count
        is_min_vehicle = (num_vehicles_per_beam == min_vehicles)

        # Mask returns: prioritize min-vehicle solutions, set others to -inf
        masked_returns = torch.where(is_min_vehicle, terminal_states_returns,
                                     torch.tensor(-torch.inf, device=terminal_states_returns.device))

        # Find the index of the beam with the best return among the min-vehicle solutions
        best_solution_per_problem = masked_returns.argmax(dim=1)

        return best_solution_per_problem, min_vehicles.view(-1)


def grouped_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Return max value per group id in `index` (size = dim_size).
       Empty groups get -inf for floats, or dtype min for ints."""
    if index.dtype != torch.long:
        index = index.long()
    if src.is_floating_point():
        init = torch.full((dim_size,), float("-inf"), device=src.device, dtype=src.dtype)
    else:
        init = torch.full((dim_size,), torch.iinfo(src.dtype).min, device=src.device, dtype=src.dtype)

    init.scatter_reduce_(0, index, src, reduce="amax", include_self=True)
    return init

def grouped_argmax_index(src: torch.Tensor,
                         index: torch.Tensor,
                         dim_size: int) -> torch.Tensor:
    """
    src:   shape [M]         (values to reduce: terminal_states_returns)
    index: shape [M] (long)  (group ids: terminal_actions_positions[:, 1])
    dim_size: number of groups (n_problems)

    returns: shape [dim_size] (arg index in [0..M-1] of the max per group,
                               -1 where a group has no elements)
    """
    if index.dtype != torch.long:
        index = index.long()

    device = src.device
    M = src.numel()

    # 1) Max value per group
    max_vals = torch.full((dim_size,), float("-inf"), device=device, dtype=src.dtype)
    # include_self=True keeps initial -inf when a group has no entries
    max_vals.scatter_reduce_(0, index, src, reduce="amax", include_self=True)

    # 2) Arg index per group (pick the highest position among ties)
    pos = torch.arange(M, device=device)
    mask = src == max_vals.index_select(0, index)
    masked_pos = torch.where(mask, pos, torch.full_like(pos, -1))

    arg_idx = torch.full((dim_size,), -1, device=device, dtype=pos.dtype)
    arg_idx.scatter_reduce_(0, index, masked_pos, reduce="amax", include_self=True)

    return arg_idx
