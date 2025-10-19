# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import logging
from collections import namedtuple

import torch
import torch as th
import torch.nn.functional as F

from ..search_tree import fill_to_k_actions


def update_lob_prob_of_beam(log_prob_of_beam, head_log_prob, masked_heads, selected_heads, sampled_problems, shift=0):
    batch_size, seq_len = selected_heads.shape
    range_tensor = torch.arange(seq_len, device=selected_heads.device).unsqueeze(0).expand(batch_size, -1)
    mask = range_tensor.unsqueeze(-1) < range_tensor.unsqueeze(-2)
    expanded_heads = selected_heads.unsqueeze(-1).expand(-1, -1, seq_len)
    action_index_from_beam = ((expanded_heads == selected_heads.unsqueeze(-2)) & mask).sum(dim=-2)
    # Create a mask for valid selected heads
    valid_mask = ~masked_heads
    # Use the mask to get the valid selected heads, their corresponding action indices, and log probabilities
    valid_selected_heads = selected_heads[valid_mask]
    valid_action_indices = action_index_from_beam[valid_mask]
    valid_log_probs = head_log_prob[valid_mask]
    # Create indices for the first dimension of log_prob_of_beam
    problem_indices, _ = valid_mask.nonzero(as_tuple=True)
    # Use advanced indexing to assign the log probabilities to the corresponding entries in log_prob_of_beam
    problem_indices = sampled_problems.nonzero()[problem_indices].flatten()
    log_prob_of_beam[problem_indices, valid_selected_heads, valid_action_indices + shift] = valid_log_probs


class Sampler(object):
    def __init__(self, config):
        self.config = config
        self.score_to_prob_logits = config['sampler']['score_to_prob'] == 'logits'
        self.logging_level = self.config['logger']['logging_level']
        self.temperature = config['sampler']['temperature']
        self.score_to_prob = self.config['sampler']['score_to_prob']

    def sample(self, logits, unmasked_nodes, action=None, deterministic=False, n_samples=1, max_actions=None, score_to_prob=None):
        dtype = logits.dtype
        if score_to_prob is None:
            score_to_prob = self.score_to_prob
        n_actions = logits.shape[-1]
        # self.logits = logits.view(-1, n_actions)
        unmasked_nodes = unmasked_nodes.view(-1, n_actions)
        if deterministic:
            dist = self.get_dist(logits, unmasked_nodes, score_to_prob=score_to_prob)
            assert n_samples == 1, ('Multinomial sampling not implemented for deterministic sampling. Best action is '
                                    'selected in external function.')
            scores = dist.probs
            if action is None:
                action = scores.argmax(dim=-1).to(int)
            log_prob = dist.log_prob(action)
            entropy = 0. # this part should never be used in backward pass!
        else:
            compute_action = action is None or (action.dim() > 1 and action.shape[-1] > 1)
            if compute_action:
                action, log_prob = self.draw_sample(logits, unmasked_nodes, n_samples_to_draw=n_samples,
                                                    n_feasible_actions_from_bin=max_actions,
                                                    score_to_prob=score_to_prob, action=action)
                entropy = 0.  # we don't need entropy at the collect data stage
            else:
                n_actions_from_head = unmasked_nodes.clone()
                mask = n_actions_from_head > 0
                dist = self.get_dist(logits, unmasked_nodes=mask, score_to_prob=score_to_prob)
                entropy = dist.entropy()
                log_prob = dist.log_prob(action.squeeze(-1)).view_as(action)
        return action, log_prob.to(dtype), entropy

    def sample_heads_and_actions(self, beam_scores, dones, policy_logits, legal_actions, deterministic=False,
                                 n_actions_from_head=None, node_sampler_args={}, action_sampler_args={}):
        beam_scores = beam_scores.to(torch.float32)
        policy_logits = policy_logits.to(torch.float32)
        n_problems, k_beams = dones.shape
        not_dones = ~dones
        frontier_length = not_dones.sum(dim=[-1])
        provided_action = None
        if deterministic:
            n_action = k_beams + 1
            non_det_ind = 1
        else:
            n_action = k_beams
            non_det_ind = 0
        actions = -torch.ones(n_problems, k_beams, n_action, dtype=int)
        if n_actions_from_head is None:
            n_actions_from_head = torch.zeros(n_problems, k_beams, dtype=int, device=actions.device)
        log_prob = torch.ones(n_problems, k_beams, n_action, device=beam_scores.device)

        top_head = torch.zeros(n_problems, k_beams, dtype=bool, device=actions.device)
        remaining_actions = torch.zeros(n_problems, k_beams, dtype=int, device=actions.device)
        # log_prob_of_beam is the log prob of the selected heads
        log_prob_of_beam = torch.zeros(n_problems, k_beams, n_action, dtype=beam_scores.dtype, device=actions.device)
        selected_heads_info = -torch.ones(n_problems, k_beams, dtype=int, device=actions.device)
        # Relaxation: sampling with replacement instead of without replacement

        # step 1 - choose the beams / heads
        # max actions - the maximal number of actions that can be taken from each head
        max_actions = legal_actions.sum(dim=-1)
        max_actions = torch.min(max_actions, n_actions_from_head)
        max_actions[dones] = 0

        # define the different types of problems
        done_problems = frontier_length == 0  # no heads, no actions
        single_head_problems = frontier_length == 1  # single head, multiple actions
        sampled_problems = frontier_length > 1  # multiple heads, multiple actions

        # null problems - basically do nothing, actions are -1, log prob is 0,
        active_problems = ~done_problems

        if deterministic:
            # a trick to find the first True entry at each row
            top_head_ind = not_dones[active_problems].to(int).argmax(dim=-1)
            top_head_log_prob = 0.
            top_head[active_problems, top_head_ind] = True
            log_prob_of_beam[active_problems.nonzero().squeeze(), top_head_ind, 0] = top_head_log_prob
            if single_head_problems.any():
                # deterministic problems - Only a single head exists
                n_actions_from_head[single_head_problems] = max_actions[single_head_problems] * top_head[single_head_problems]

            max_actions[top_head] -= 1
            top_action, top_log_prob, _ = self.sample(logits=policy_logits[top_head], unmasked_nodes=legal_actions[top_head],
                                                      deterministic=True, action=provided_action,
                                                      **action_sampler_args)
            actions[..., 0][top_head] = top_action
            log_prob[..., 0][top_head] = top_log_prob.to(log_prob.dtype)
            selected_heads_info[active_problems, 0] = top_head_ind
            legal_actions[top_head, top_action] = False
            idx = 1
            remaining_samples = torch.min(torch.tensor(k_beams - 1), max_actions.sum(dim=-1))
        else:
            if single_head_problems.any():
                # deterministic problems - Only a single head exists
                n_actions_from_head[single_head_problems] = max_actions[single_head_problems] * not_dones[single_head_problems]
            idx = 0
            remaining_samples = torch.min(torch.tensor(k_beams), max_actions.sum(dim=-1))

        if single_head_problems.any():
            log_prob_of_beam[single_head_problems] = 0
            remaining_actions[single_head_problems.unsqueeze(-1) & not_dones] = remaining_samples[single_head_problems]

        if sampled_problems.any():
            # sampled problems - identify the top head, and sample other heads
            # single_head_problem were addressed in the previous step
            beam_scores = beam_scores[sampled_problems]
            max_actions = max_actions[sampled_problems]
            remaining_samples = remaining_samples[sampled_problems]

            remaining_actions[sampled_problems] = max_actions

        # step 2 - sample actions from the selected beams
        active_heads = remaining_actions > 0
        n_samples = remaining_actions[active_heads]
        if active_heads.any() > 0:
            sampled_action, sampled_log_prob, _ = self.sample(logits=policy_logits[active_heads],
                                                              unmasked_nodes=legal_actions[active_heads],
                                                              n_samples=n_samples, max_actions=1, action=provided_action,
                                                              **node_sampler_args)
            n_sampled_actions = sampled_log_prob.shape[-1]
            actions[..., idx:n_sampled_actions + idx][active_heads] = sampled_action
            log_prob[..., idx:n_sampled_actions + idx][active_heads] = sampled_log_prob.to(log_prob.dtype)

        log_prob += log_prob_of_beam
        valid_actions = actions > -1
        n_valid_actions = valid_actions.sum(dim=[-1, -2]) # skipping the correction for step 0
        if (n_valid_actions != k_beams).any():
            # a quick fix to make sure that for every beam there is a "valid" action so there won't be any state copies
            expanded_dones = dones.unsqueeze(-1).repeat(1, 1, valid_actions.size(-1))
            expanded_dones[:,:,1:] = False
            valid_actions[expanded_dones] = True
            n_actions_from_head[:] = 1
        n_valid_actions = valid_actions.sum(dim=[-1, -2])
        if (n_valid_actions != k_beams).any():
            valid_action_mask, n_actions_from_head = (
                fill_to_k_actions(valid_actions, k=k_beams, n_actions_from_head=n_actions_from_head))
        else:
            valid_action_mask = valid_actions
        simulation_instruction = SimulationInstruction(actions=actions,
                                                        n_actions_from_head=n_actions_from_head,
                                                        valid_actions_mask=valid_action_mask)
        action_list = actions[valid_action_mask].view(n_problems, k_beams)
        info = {'selected_heads': selected_heads_info}
        return action_list, log_prob, simulation_instruction, info

    def draw_sample(self, logits, unmasked_nodes, n_samples_to_draw=1, n_feasible_actions_from_bin=None, score_to_prob=None,
                    action=None):
        if score_to_prob is None:
            score_to_prob = self.config['sampler']['score_to_prob']

        max_samples = n_samples_to_draw if isinstance(n_samples_to_draw, int) else n_samples_to_draw.max()
        n_distributions = logits.shape[0]
        provided_action = None
        predefined_actions = action is not None
        if not predefined_actions:
            action = -torch.ones(n_distributions, max_samples, dtype=int)
        log_prob = -torch.zeros(n_distributions, max_samples, dtype=logits.dtype, device=logits.device)
        flag = True
        use_multinomial = isinstance(n_samples_to_draw, torch.Tensor) and (n_samples_to_draw > 1).any()
        if use_multinomial:
            # make sure that we can sample enough heads with replacement
            if isinstance(n_feasible_actions_from_bin,int) and n_feasible_actions_from_bin == 1:  # default case for sampling actions, disable replacement
                with_replacement = torch.zeros(n_samples_to_draw.shape, dtype=bool)
                without_replacement = torch.ones(n_samples_to_draw.shape, dtype=bool)
            else:
                # cases where we can sample with replacement
                delta = n_feasible_actions_from_bin - n_samples_to_draw.unsqueeze(-1)
                # if we can sample more than the maximal number of actions, we can sample with replacement. This condition
                # should be checked only for unmasked nodes
                with_replacement = torch.logical_or(delta >= 0, ~unmasked_nodes).all(dim=1)
                # cases where we must sample without replacement
                without_replacement = (n_feasible_actions_from_bin < 2).all(dim=1)  # if we need to sample a single action or head, we can't sample with replacement
                with_replacement = with_replacement & ~without_replacement  # make sure that we don't sample with and without replacement
            repeated_sampling = ~torch.logical_or(with_replacement, without_replacement)

            if with_replacement.any():
                if predefined_actions:
                    provided_action = action[with_replacement]
                k = n_samples_to_draw[with_replacement].max()
                res_action, log_prob[with_replacement, :k] = (
                    self.draw_multinomial(logits[with_replacement], unmasked_nodes=unmasked_nodes[with_replacement],
                                          n_samples=n_samples_to_draw[with_replacement], with_replacement=True, k=k,
                                          score_to_prob=score_to_prob,action=provided_action))
                if not predefined_actions:
                    action[with_replacement, :k] = res_action
            if without_replacement.any():
                if predefined_actions:
                    provided_action = action[without_replacement]
                k = n_samples_to_draw[without_replacement].max()
                if isinstance(n_feasible_actions_from_bin, int) and n_feasible_actions_from_bin == 1:
                    maximal_number_of_samples = n_feasible_actions_from_bin
                else:
                    maximal_number_of_samples = n_feasible_actions_from_bin[without_replacement]
                res_action, log_prob[without_replacement, :k] = (
                    self.draw_multinomial(logits[without_replacement], unmasked_nodes=unmasked_nodes[without_replacement],
                                          n_samples=n_samples_to_draw[without_replacement],
                                          with_replacement=False, k=k,
                                          max_actions=maximal_number_of_samples,
                                          score_to_prob=score_to_prob,
                                          action=provided_action))
                if not predefined_actions:
                    action[without_replacement, :k] = res_action
            if repeated_sampling.any():
                k = n_samples_to_draw[repeated_sampling].max()
                log_prob[repeated_sampling, :k], repeated_actions = (
                    self.sequential_sampling(unmasked_nodes=unmasked_nodes[repeated_sampling],
                                             action=action[repeated_sampling], logits=logits[repeated_sampling],
                                             n_feasible_actions_from_bin=n_feasible_actions_from_bin[repeated_sampling],
                                             n_samples_to_draw=n_samples_to_draw[repeated_sampling],
                                             predefined_actions=predefined_actions, score_to_prob=score_to_prob))
                if not predefined_actions:
                    action[repeated_sampling] = repeated_actions
        else:
            dist = self.get_dist(logits, unmasked_nodes, score_to_prob=score_to_prob)
            while flag:
                action = dist.sample([1]).transpose(1, 0)
                if action.dim() == 2 and unmasked_nodes.dim() == 1:
                    action = action.squeeze(0)
                if action.dim() == 2:
                    # basically, they do the same
                    dummy_ind = torch.arange(unmasked_nodes.shape[0])
                    sample_valid = unmasked_nodes[dummy_ind, action.squeeze(-1)]
                else:
                    sample_valid = unmasked_nodes[action]
                flag = not sample_valid.all()
                if flag:
                    logging.warning('Warning: sampling a 0 probability action')
            log_prob = dist.log_prob(action.transpose(1, 0)).transpose(1, 0)
            action = action.to(int)
        return action, log_prob

    def sequential_sampling(self, unmasked_nodes, action, logits, n_feasible_actions_from_bin, n_samples_to_draw,
                            predefined_actions, prior, score_to_prob):
        ind = torch.ones(action.shape[0], dtype=bool)
        log_prob = torch.zeros(action.shape[0], n_samples_to_draw.max(), dtype=logits.dtype, device=logits.device)
        for action_counter in range(n_samples_to_draw.max()):
            dist = self.get_dist(logits[ind], unmasked_nodes[ind], prior=prior, score_to_prob=score_to_prob)
            if not predefined_actions:
                action[ind, action_counter] = dist.sample([1]).view(-1)
            selected_action = action[ind, action_counter].view(-1)
            dummy_ind = ind.nonzero().flatten()
            log_prob[ind, action_counter] = dist.logits[torch.arange(selected_action.shape[0]), selected_action]
            n_feasible_actions_from_bin[dummy_ind, selected_action] -= 1
            unmasked_nodes[n_feasible_actions_from_bin == 0] = False
            ind = ind.clone()  # this is weird, no idea why cloning is necessary, but it is a must
            n_samples_to_draw -= 1
            ind[n_samples_to_draw == 0] = False
        return log_prob, action

    def draw_multinomial(self, logits, unmasked_nodes, n_samples, with_replacement, k, max_actions=None, prior=None,
                         score_to_prob=None, penalty_factor=None, action=None):
        dtype = logits.dtype
        initial_dist = self.get_dist(logits, unmasked_nodes, prior=prior, score_to_prob=score_to_prob,
                                     penalty_factor=penalty_factor)
        # log_prob_of_dist = initial_dist.probs.log()
        log_prob_of_dist = initial_dist.probs
        n_distributions = logits.shape[0]
        compute_action = action is None
        if compute_action:
            action = -torch.ones(n_distributions, k, dtype=int)
        log_prob = -torch.zeros(n_distributions, k, dtype=log_prob_of_dist.dtype, device=logits.device)
        for i in range(1, k + 1):
            indices = n_samples == i
            n_indices = indices.sum()
            if n_indices > 0:
                if compute_action:
                    action[indices, :i] = torch.multinomial(initial_dist.probs[indices], num_samples=i,
                                                            replacement=with_replacement).to(int)
                if with_replacement or i == 1:  # independent trails
                    dummy_ind = indices.nonzero().expand(-1, i).flatten()
                    log_prob[indices, :i] = torch.log(log_prob_of_dist[dummy_ind, action[indices, :i].flatten()].view(
                            n_indices, i))
                else:  # dependent trails
                    dummy_ind = indices.nonzero().squeeze(-1)
                    log_prob[dummy_ind, 0] = torch.log(log_prob_of_dist[dummy_ind, action[indices, 0]])
                    unmasked_nodes = unmasked_nodes.clone()
                    for j in range(1, i):
                        unmasked_nodes[dummy_ind, action[indices, j - 1]] = False
                        dist = self.get_dist(logits[indices], unmasked_nodes[indices], prior=prior, score_to_prob=score_to_prob,
                                             penalty_factor=penalty_factor)
                        log_prob[indices, j] = dist.log_prob(action[indices, j])
        log_prob = log_prob.to(dtype)
        return action, log_prob

    def get_dist(self, logits, unmasked_nodes, prior=None, score_to_prob=None, penalty_factor=None):
        if score_to_prob is None:
            score_to_prob = self.score_to_prob
        if logits.device.type == 'cpu':
            logits = logits.to(torch.float32)
        unmasked_nodes = unmasked_nodes.view_as(logits)
        if score_to_prob in ('probs', 'softplus'):
            if score_to_prob == 'probs':
                logits = logits.abs()
            else:
                # softplus: log(1+exp(logits))
                # log-sum-exp trick for stabilization: shift all samples towards 0, before taking exp().
                inf_mask = torch.zeros_like(unmasked_nodes, dtype=torch.float)
                inf_mask[~unmasked_nodes] = torch.inf
                lmax = (logits - inf_mask).max(dim=1).values.unsqueeze(1)
                logits = lmax + ((-lmax).exp() + (logits - lmax).exp()).log()
            logits[~unmasked_nodes] = 0
            if self.temperature != 1:
                logits = logits ** (1 / self.temperature)
            probs = logits if prior is None else logits * prior
            zero_rows = probs.sum(dim=-1) == 0  # todo: does it take a lot of time
            if zero_rows.any():
                print('zero rows found in probs, setting to 1 at unmasked actions')
                probs[zero_rows] = unmasked_nodes[zero_rows].to(probs.dtype)
            dist = th.distributions.categorical.Categorical(probs=probs, validate_args=True)
        elif score_to_prob == 'logits':
            assert logits is None or prior is None, 'need to verify that logits and prior works together'
            min_score = logits.abs().min()
            min_score = float(-50 * max(1, min_score))
            logits[torch.logical_not(unmasked_nodes)] = min_score
            if self.temperature != 1:
                logits = logits / self.temperature
            logits_in = logits if prior is None else logits + torch.log(prior)
            dist = th.distributions.categorical.Categorical(logits=logits_in)
        else:
            raise NotImplementedError(f"Unknown score_to_prob method: "
                                      f"{score_to_prob}")
        return dist



SimulationInstruction = namedtuple('SimulationInstruction',
                                   ['actions', 'n_actions_from_head', 'valid_actions_mask'])
