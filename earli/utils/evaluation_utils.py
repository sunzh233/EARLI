# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import warnings

import numpy as np
import pandas as pd
import torch


def maybe_update_best_result(best_result, candidate_result):
    if candidate_result is not None:
        if isinstance(best_result, dict):
            return maybe_update_best_result_per_eq(best_result, candidate_result)
        elif isinstance(best_result, list):
            assert len(best_result) == len(candidate_result)
            best_result = [maybe_update_best_result_per_eq(best_result[i], candidate_result[i])
                           for i in range(len(candidate_result))]
        elif best_result is None:
            best_result = candidate_result
    return best_result


def maybe_update_best_result_per_eq(best_result, candidate_result):
    if candidate_result['total_reward'] > best_result['total_reward'] and not candidate_result['baseline_policy']:
        best_result = candidate_result
    return best_result


def get_best_result(env, same_eq_subsets=None):
    # candidate_result_array = env.get_best_result()
    candidate_result_array = env.get_attr('best_result')
    if same_eq_subsets:
        best_result_list = []
        for same_eq_envs in same_eq_subsets:
            best_result = None
            for ind in same_eq_envs:
                best_result = maybe_update_best_result(best_result, candidate_result_array[ind])
            best_result_list.append(best_result)
        return best_result_list
    else:
        best_result = None
        for ind in candidate_result_array:
            best_result = maybe_update_best_result(best_result, ind)
        return best_result


def symmetric_diff(path1, path2, directed=True, normalize=True):
    edges1 = {(a, b) for a, b in zip(path1[:-1], path1[1:])}
    edges2 = {(a, b) for a, b in zip(path2[:-1], path2[1:])}
    tot_edges = len(edges1) + len(edges2)
    if directed:
        intersect = [e for e in edges1 if e in edges2]
    else:
        intersect = [e for e in edges1 if e in edges2 or e[::-1] in edges2]
    diff = tot_edges - 2*len(intersect)  # union=tot-intersect; diff=union-intersect
    if normalize:
        diff = diff / tot_edges

    return diff


def is_worse_equal(vehicles, cand_ret, ref_vehicles, ref_ret, cost_only=False):
    if cost_only:
        return cand_ret <= ref_ret
    else:
        return vehicles > ref_vehicles or (vehicles == ref_vehicles and cand_ret <= ref_ret)

def is_worse(vehicles, cand_ret, ref_vehicles, ref_ret, cost_only=False, eps=0):
    if not cost_only:
        if vehicles is None:
            warnings.warn('Unknown vehicle count. Using cost only.')
        elif ref_vehicles is None:
            warnings.warn('Unknown ref vehicle count. Using cost only.')
        else:
            return vehicles > ref_vehicles or (vehicles == ref_vehicles and cand_ret + eps < ref_ret)

    # if cost only, or if we failed to read vehicles / ref_vehicles:
    return cand_ret + eps < ref_ret

def is_accepted_to_population(cand_path, cand_vehicles, cand_ret, population, vehicles, rets, radius,
                              are_valids=None, cost_only=False, require_best=False, **kwargs):
    if are_valids is None:
        are_valids = len(rets) * [True]
    if vehicles is None:
        vehicles = len(rets) * [True]

    if not np.any(are_valids):
        # no valid solutions
        return True

    is_worst = True
    for ref_path, ref_vehicles, ref_ret, is_valid in zip(population, vehicles, rets, are_valids):
        if is_valid:
            candidate_is_better_equal = is_worse_equal(ref_vehicles, ref_ret, cand_vehicles, cand_ret, cost_only)
            candidate_is_better = is_worse(ref_vehicles, ref_ret, cand_vehicles, cand_ret, cost_only)
            if require_best and not candidate_is_better_equal:
                # candidate is (strictly-)inferior to another valid solution
                return False
            if (not candidate_is_better) and (symmetric_diff(cand_path, ref_path, **kwargs) <= 1 - radius):
                # candidate is inferior(/equal) to a valid solution within clearing radius
                return False
            if candidate_is_better:
                is_worst = False

    return not is_worst

def test_acceptance(cand_path, cand_vehicles, cand_ret, populations, cost_only=False,
                    reward_normalization=1, **kwargs):
    if (not cost_only) and (cand_vehicles is None or not hasattr(populations, 'snapshot_vehicles')):
        if cand_vehicles is None:
            warnings.warn('Cannot find vehicles count in solution. Using cost only.')
        else:
            warnings.warn('Cannot find vehicles count within snapshots. Using cost only.')
        cost_only = True

    sols = populations.population_snapshots
    vehicles = len(sols)*[None] if cost_only else populations.snapshot_vehicles
    costs = populations.snapshot_costs
    radii = populations.clearing_radii
    times = populations.times
    valids = populations.valid_snapshots
    if not radii:
        warnings.warn(f'CuOpt clearing radius was not found in data. Assuming default=0.8.')
        radii = len(sols) * [0.8]

    accepted = []
    strongly_accepted = []
    for population, sol_vehicles, sol_costs, radius, sol_valids in zip(sols, vehicles, costs, radii, valids):
        rets = - reward_normalization * sol_costs
        paths = [[node for route in sol for node in [0] + route]
                 for sol in population]
        accepted.append(
            is_accepted_to_population(cand_path, cand_vehicles, cand_ret, paths, sol_vehicles, rets, radius,
                                      sol_valids, cost_only, require_best=False, **kwargs))
        strongly_accepted.append(
            is_accepted_to_population(cand_path, cand_vehicles, cand_ret, paths, sol_vehicles, rets, radius,
                                      sol_valids, cost_only, require_best=True, **kwargs))
    return times, accepted, strongly_accepted

def evaluate_populations_gaps(data):
    problem_ids = []
    times = []
    gaps = []
    criterion = []
    pool_size = []
    final_costs = []

    for i_prob in range(data['n_problems']):
        final_cost = data['baseline_cost_unnormalized'][i_prob] if 'baseline_cost_unnormalized' in data else (
            data)['baseline_cost'][i_prob]
        populations = data['baseline_population'][i_prob]
        for t, population_costs, population_valids in zip(
                populations.times, populations.snapshot_costs, populations.valid_snapshots):
            population_costs = [c for c, v in zip(population_costs, population_valids) if v]

            # log worst solution in population
            problem_ids.append(i_prob)
            # ensure we operate on numpy array (items may be torch tensors)
            worst_cost = np.max(np.asarray(population_costs))
            times.append(t)
            gaps.append(worst_cost / final_cost)
            criterion.append('worst')
            pool_size.append(len(population_costs))
            final_costs.append(final_cost)

            # log best solution in population
            problem_ids.append(i_prob)
            # ensure we operate on numpy array (items may be torch tensors)
            best_cost = np.min(np.asarray(population_costs))
            times.append(t)
            gaps.append(best_cost / final_cost)
            criterion.append('best')
            pool_size.append(len(population_costs))
            final_costs.append(final_cost)

    rr = pd.DataFrame(dict(i_env=problem_ids, time=times, criterion=criterion, pool_size=pool_size,
                           final_cost=final_costs, optimality_gap=gaps))
    return rr


def get_trajectory_cost(distance_matrix, trajectory):
    # Ensure the sequence starts with 0
    if trajectory[0] != 0:
        trajectory.insert(0, 0)

    # Convert list to tensor if it's not
    if not isinstance(trajectory, torch.Tensor):
        trajectory = torch.tensor(trajectory)

    # Sum the values d(l[i], l[i+1])
    total = 0
    for i in range(len(trajectory) - 1):
        total += distance_matrix[trajectory[i], trajectory[i + 1]]

    return total

def solution_cost(sol, positions, Lp=2):
    if positions.shape[1] > 2:
        # interpret positions as distances
        distances = positions
        return np.sum([distances[i1,i2] for i1, i2 in zip(sol[:-1], sol[1:])])
    return np.sum([(np.linalg.norm(positions[i2] - positions[i1], ord=Lp))
                   for i1, i2 in zip(sol[:-1], sol[1:])])

def solutions_costs(sols, positions, **kwargs):
    return [solution_cost(sol, pos, **kwargs) for sol, pos in zip(sols, positions)]


def verify_solution(solution, demands, capacity=None, max_vehicles=None, verbose=False,
                    verbose_fun=warnings.warn):
    solution = np.array(solution)
    n = len(demands)
    if set(solution) != set(range(n)):
        if verbose:
            verbose_fun(f'Solution has missing nodes ({len(np.unique(solution))} instead of {n})')
        return False
    depot_indices = np.where(solution == 0)[0]
    if capacity is not None:
        route_demands = np.split(demands[solution[1:-1]], depot_indices[1:-1] - 1)
        route_loads = np.array([np.sum(route) for route in route_demands])
        if np.any(route_loads > capacity):
            if verbose:
                verbose_fun(f'Solution has overloaded route ({max(route_loads)} > {capacity})')
            return False
    if max_vehicles is not None:
        if len(depot_indices) - 1 > max_vehicles:
            if verbose:
                verbose_fun(f'Solution has {len(depot_indices) - 1} > {max_vehicles} vehicles')
            return False
    return True


def test_solution(solution, dist_matrix):
    total = 0
    for i, j in zip(solution[:-1], solution[1:]):
        total += dist_matrix[i, j]
    print(f'Total distance: {total}, Number of vehicles: {(solution == 0).sum()}')


def update_stats(env_steps, info, stats, restart_iteration_log, n_workers=1, worker_id=0):
    n = len(info['env_id'])

    if restart_iteration_log:
        stats['total_games'] = 0
        stats['data_samples'] = 0
        # stats['stage_time'] = 0
        stats['game_clocktime'] = 0

    stats['total_games'] += n
    stats['data_samples'] += info['data_samples'].sum().item()
    # stats['stage_time'] += info['game_clocktime']
    stats['game_clocktime'] = ((stats['total_games'] - n) * stats['game_clocktime'] +
                               n * info['game_clocktime']) / stats['total_games']
    # stats['total_env_steps'] = env_steps

    if info['env_id'][0] is not None:
        env_ids = info['env_id']
    elif n_workers > 1:
        env_ids = torch.arange(worker_id, n_workers * n, n_workers)
    else:
        env_ids = list(range(n))
    for i, id in enumerate(env_ids):
        if 'baseline_return' in info:
            stats['baseline_returns'][id] = info['baseline_return'][i].item()
        if 'baseline_vehicles' in info:
            stats['baseline_vehicles'][id] = info['baseline_vehicles'][i].item()
        stats['recent_returns'][id] = info['best_return'][i].item()
        past_problem_return = stats['best_returns'].get(id, -np.inf)
        stats['best_returns'][id] = max(info['best_return'][i].item(), past_problem_return)
        stats['game_iters'][id] = info['forward_iters'][i]
        if 'vehicle_penalty' in info:
            stats['net_returns'][id] = info['best_return'][i].item() + info['vehicle_penalty'][i].item()
            stats['number_of_vehicles'][id] = info['num_vehicles'][i].item()
        if 'optimal_route' in info:
            stats['optimal_route'][id] = info['optimal_route'][i]
            if 'all_routes' in info and len(info['all_routes']) > 0:
                stats['all_routes'][id] = info['all_routes'][i]
    return stats


def update_batch_log(stats, buffer_usage):
    stats['buffer_usage'] = buffer_usage[2]
