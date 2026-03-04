# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""cuOpt injection experiment for VRPTW and PDPTW.

This script runs the RL + cuOpt injection pipeline for VRPTW/PDPTW benchmark
instances.  It:
  1. Loads a PKL dataset produced by ``benchmark_parser.py`` (VRPTW or PDPTW).
  2. Optionally loads RL-generated initial solutions from a test-log PKL file.
  3. Passes those solutions as initial population into cuOpt.
  4. Reports solution quality (cost and vehicle count).

Usage::

    # VRPTW with cuOpt refinement:
    python -m earli.cuopt_injection_vrptw \\
        --config config_vrptw_cuopt.yaml \\
        --solutions outputs/test_logs.pkl

    # PDPTW with cuOpt refinement:
    python -m earli.cuopt_injection_vrptw \\
        --config config_pdptw_cuopt.yaml \\
        --solutions outputs/test_logs_pdptw.pkl

Both ``ppo`` and ``tree_based`` inference output formats are supported
(the solutions log format is the same regardless of the RL pipeline used).
"""

import argparse
import os
import pickle as pkl
import time
import warnings

import numpy as np
import yaml

from .unified_logger import UnifiedLogger
from .utils import evaluation_utils as eval_utils
from .test_injection import (
    load_problems,
    load_external_solutions,
    preprocess_external_solutions,
    get_initial_solutions,
    get_initialization_runtimes,
    run_initialization_only,
    summarize_results,
    update_accepted_stats,
    get_num_carriers_candidates,
    get_pools_bests,
    convert_cuopt_solution,
)


# ---------------------------------------------------------------------------
# Time-window aware cuOpt runner
# ---------------------------------------------------------------------------

def run_cuopt_tw(external_solutions, problems, demands_for_solver, n_carriers, ind,
                 ga_time, ENV, LAST_RETURN_TO_DEPOT, FIXED_VEHICLES, PULL_FREQ,
                 CUOPT_NORMALIZATION):
    """Run cuOpt with time-window (and optionally pickup-delivery) constraints.

    Extends the base ``run_cuopt`` from ``test_injection.py`` by extracting
    ``time_windows`` and ``service_times`` from the problems dict and forwarding
    them to ``CuOptSolver.create_data_model``.
    """
    try:
        import cudf
        import cuopt
        from .cuopt_solver.cuopt_solver import CuOptSolver
    except ImportError as e:
        raise ImportError(
            "cuOpt is required for this experiment. "
            "Please install cuopt and its dependencies."
        ) from e

    cuopt_config = dict(
        problem_setup=dict(
            env=ENV,
            last_return_to_depot=LAST_RETURN_TO_DEPOT,
        ),
        cuopt=dict(
            time_limit=ga_time,
            climbers=2048,
            pull_frequency=PULL_FREQ,
        ),
    )
    solver = CuOptSolver(cuopt_config)

    distance_matrix = problems['distance_matrix'][ind]
    capacity = problems['capacity'][ind]

    # Build time-window arguments for VRPTW / PDPTW
    t_min_arg = t_max_arg = dt_arg = None
    if 'time_windows' in problems:
        tw = problems['time_windows'][ind].numpy()       # (n_nodes, 2)
        svc = problems['service_times'][ind].numpy()     # (n_nodes,)
        norm = float(CUOPT_NORMALIZATION)
        t_min_arg = (tw[:, 0] / norm).tolist()
        t_max_arg = (tw[:, 1] / norm).tolist()
        dt_arg    = (svc / norm).tolist()

    problem = solver.create_data_model(
        demand=demands_for_solver,
        distance_matrix=distance_matrix / CUOPT_NORMALIZATION,
        vehicle_capacity=np.full(shape=(n_carriers,),
                                 fill_value=int(capacity),
                                 dtype=int),
        n_carriers=n_carriers,
        t_min=t_min_arg,
        t_max=t_max_arg,
        dt=dt_arg,
        set_fixed_carriers=FIXED_VEHICLES,
        last_drop_return_trip=not LAST_RETURN_TO_DEPOT,
        injected_solutions=external_solutions,
    )

    t0 = time.time()
    try:
        routing_solution, populations = solver.solve(problem, verbose=0)
        actual_runtime = time.time() - t0
        success = routing_solution.get_status() == 0
        vehicle_count = routing_solution.get_vehicle_count()
        cost = CUOPT_NORMALIZATION * routing_solution.get_total_objective()
        solution = convert_cuopt_solution(routing_solution)
        accepted = routing_solution.get_accepted_solutions()
    except Exception as exc:
        warnings.warn(f"cuOpt failed: {exc}")
        actual_runtime = time.time() - t0
        success, vehicle_count, cost, solution = False, np.nan, np.nan, []
        import pandas as pd
        accepted = pd.Series(dtype='float')
        populations = None

    if populations is not None:
        dm_np = distance_matrix.numpy() if hasattr(distance_matrix, 'numpy') else distance_matrix
        populations.update_costs(
            distance_matrix=dm_np,
            capacity=int(capacity),
            demands_for_solver=demands_for_solver,
            n_customers=dm_np.shape[0] - 1,
            return_to_depot=LAST_RETURN_TO_DEPOT,
        )

    return success, vehicle_count, cost, solution, accepted, actual_runtime, np.nan, populations


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(config_path=None, solution_path=None):
    # ---- argument parsing ----
    if config_path is None:
        parser = argparse.ArgumentParser(
            description='cuOpt injection experiment for VRPTW / PDPTW'
        )
        parser.add_argument('--config', default='config_vrptw_cuopt.yaml',
                            help='Path to cuOpt injection config YAML')
        parser.add_argument('--solutions', default=None,
                            help='Path to RL inference test_logs.pkl (overrides config paths.solutions)')
        args = parser.parse_args()
        config_path = args.config
        solution_path = args.solutions

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if solution_path is not None:
        config['paths']['solutions'] = solution_path

    logger = UnifiedLogger(config)

    RUN_TITLE = config['names']['run_title']
    PROBLEMS_PATH = config['paths']['problems']
    SOLUTION_PATHS = config['paths']['solutions']
    if isinstance(SOLUTION_PATHS, str):
        SOLUTION_PATHS = [SOLUTION_PATHS]
    SOLUTION_SOURCES = config['paths'].get('solutions_source')
    if not isinstance(SOLUTION_SOURCES, (tuple, list)):
        SOLUTION_SOURCES = [SOLUTION_SOURCES]
    NAIVE_SOURCE = config['paths'].get('naive_source')
    PROBLEMS_RANGE = config['problem'].get('problems_range')
    if PROBLEMS_RANGE is not None:
        PROBLEMS_RANGE = [int(x) for x in PROBLEMS_RANGE]

    OUT_PATH = os.path.join('./outputs',
                            f'{config["paths"]["out_summary"]}_{RUN_TITLE}.pkl')
    VERBOSE = config['names']['verbose']
    METHODS = config['names']['methods']
    BASELINE = config['names']['baseline']
    MAIN_METHOD = config['names']['main_method']
    SOLVER_RUNTIMES = [float(x) for x in config['cuopt']['runtimes']]
    if BASELINE not in METHODS:
        BASELINE = None
    if MAIN_METHOD not in METHODS:
        MAIN_METHOD = None

    MAX_EXTERNAL_SOLUTIONS = config['injection']['max_injections']
    if MAX_EXTERNAL_SOLUTIONS is not None:
        MAX_EXTERNAL_SOLUTIONS = (
            [int(n) for n in MAX_EXTERNAL_SOLUTIONS]
            if isinstance(MAX_EXTERNAL_SOLUTIONS, (list, tuple))
            else int(MAX_EXTERNAL_SOLUTIONS)
        )
    CHOOSE_EXTERNAL_SOLUTIONS = config['injection']['can_choose_injections']
    BOTH_RL_AND_NAIVE = config['injection']['both_rl_and_naive']
    SOLUTIONS_FORMAT_HAS_NO_LAST_DEPOT = config['problem']['remove_last_depot_from_solution']
    AVOID_SUBOPTIMAL_CARS = config['injection']['avoid_suboptimal']
    SORT_SOLUTIONS = config['injection']['sort']
    REPETITIONS = config['cuopt']['repetitions']
    ENV = config['problem']['env']
    LAST_RETURN_TO_DEPOT = config['problem']['last_return_to_depot']
    FIXED_VEHICLES = config['cuopt']['fixed_vehicles']
    SPARE_VEHICLES = config['cuopt']['spare_vehicles']
    if SPARE_VEHICLES > 0:
        FIXED_VEHICLES = False
    USE_REF_VEHICLES = config['cuopt'].get('use_reference_for_num_vehicles', False)
    PULL_FREQ = config['cuopt'].get('pull_freq', 0)

    print(f'\n{RUN_TITLE}')
    print('Problems:', PROBLEMS_PATH)
    delete_solutions = False
    if SOLUTION_SOURCES[0] is None:
        print('Solutions:', SOLUTION_PATHS)
    else:
        from .test_injection import download_wandb_file
        SOLUTION_PATHS = [
            download_wandb_file(run_id,
                                config['system']['wandb_user'],
                                config['system']['wandb_project'])
            for run_id in SOLUTION_SOURCES
        ]
        delete_solutions = True

    # Load problem instances
    problems, problems_range, problem_size, n_problems, CUOPT_NORMALIZATION, LP = load_problems(
        PROBLEMS_PATH, PROBLEMS_RANGE, config)
    log10_problems = int(np.ceil(np.log10(max(n_problems, 2))))

    INIT_RUNTIMES, NAIVE_RUNTIME = get_initialization_runtimes(config, problem_size, METHODS)

    rlopt_solutions, rlopt_vehicles, rlopt_costs, naive_solutions = get_initial_solutions(
        config, problems, problems_range, METHODS, SOLUTION_PATHS, NAIVE_SOURCE,
        MAX_EXTERNAL_SOLUTIONS, CHOOSE_EXTERNAL_SOLUTIONS, delete_solutions, LP)

    # Collect results
    (methods_log, solver_times_log, init_times_log, problem_ids_log, vehicles_log,
     costs_log, n_external_log, n_accepted_log, n_rejected_log, n_unconsidered_log,
     accepted_best_log, accepted_mean_log, accepted_cover_log, accepted_mean_sharing_log,
     extra_naive_log, solver_iter_log, successes_log, successes2_log, runtimes_log) = (
        [] for _ in range(19))

    solutions_store: dict = {}
    (pool_methods, pool_full_times, pool_times, pool_problems,
     pool_sizes, pool_vehicles, pool_costs_p) = ([] for _ in range(7))

    for i_time, solver_time in enumerate(SOLVER_RUNTIMES):
        print(f'\nSOLVER_RUNTIME = {solver_time}:')
        for ind in range(*problems_range):
            for i_method, method in enumerate(METHODS):
                init_runtime = INIT_RUNTIMES.get(method, 0)
                ga_time = solver_time - init_runtime

                demands_for_solver = problems['demand'][ind]
                if hasattr(demands_for_solver, 'numpy'):
                    demands_for_solver = demands_for_solver.numpy()
                tot_demand = float(np.sum(demands_for_solver))
                max_capacity = problems['capacity'][ind]

                if USE_REF_VEHICLES and 'baseline_vehicles' in problems:
                    n_carriers = int(problems['baseline_vehicles'][ind]) + SPARE_VEHICLES
                else:
                    n_carriers = get_num_carriers_candidates(
                        tot_demand, float(max_capacity),
                        margin=0 if FIXED_VEHICLES else SPARE_VEHICLES)

                external_solutions, best_cars, best_cost, extra_time = preprocess_external_solutions(
                    method, ind, problems, n_carriers, rlopt_solutions, naive_solutions,
                    problems_range,
                    avoid_suboptimal_cars=AVOID_SUBOPTIMAL_CARS,
                    force_feasible=(ga_time == 0),
                    LP=LP,
                    BOTH_RL_AND_NAIVE=BOTH_RL_AND_NAIVE,
                    NAIVE_RUNTIME=NAIVE_RUNTIME,
                    USE_REF_VEHICLES=USE_REF_VEHICLES,
                    ENV=ENV,
                    SORT_SOLUTIONS=SORT_SOLUTIONS,
                    SOLUTIONS_FORMAT_HAS_NO_LAST_DEPOT=SOLUTIONS_FORMAT_HAS_NO_LAST_DEPOT,
                )
                if extra_time > 0:
                    init_runtime += extra_time
                    ga_time = max(ga_time - extra_time, 0)

                for repetition in range(REPETITIONS):
                    populations = None

                    if ('naive' in method or 'RL' in method) and ga_time == 0:
                        (success, vehicle_count, cost, solution,
                         accepted, actual_runtime, num_iterations) = run_initialization_only(
                            external_solutions, best_cars, best_cost, method)
                    else:
                        (success, vehicle_count, cost, solution, accepted,
                         actual_runtime, num_iterations, populations) = run_cuopt_tw(
                            external_solutions, problems, demands_for_solver, n_carriers,
                            ind, ga_time, ENV, LAST_RETURN_TO_DEPOT, FIXED_VEHICLES,
                            PULL_FREQ, CUOPT_NORMALIZATION)

                    # Re-verify
                    cap_val = float(max_capacity.item()) if hasattr(max_capacity, 'item') else float(max_capacity)
                    if success:
                        success2 = eval_utils.verify_solution(
                            solution, demands_for_solver, cap_val, n_carriers, True, print)
                        if not success2:
                            print(f'  Validation failed ({solver_time}, {ind}, {method}).')
                    else:
                        try:
                            success2 = eval_utils.verify_solution(
                                solution, demands_for_solver, cap_val, n_carriers, False)
                        except Exception:
                            success2 = False

                    problem_id = ind if REPETITIONS == 1 else str(ind) + chr(97 + repetition)
                    methods_log.append(method)
                    solver_times_log.append(solver_time)
                    init_times_log.append(init_runtime)
                    runtimes_log.append(init_runtime + actual_runtime)
                    problem_ids_log.append(problem_id)
                    successes_log.append(success)
                    successes2_log.append(success2)
                    vehicles_log.append(vehicle_count)
                    costs_log.append(cost)
                    n_external_log.append(len(accepted))
                    n_accepted_log.append((accepted == 1).astype(int).sum()
                                          if hasattr(accepted, '__len__') and len(accepted) else 0)
                    n_rejected_log.append((accepted == 0).astype(int).sum()
                                          if hasattr(accepted, '__len__') and len(accepted) else 0)
                    n_unconsidered_log.append((accepted == -1).astype(int).sum()
                                              if hasattr(accepted, '__len__') and len(accepted) else 0)
                    extra_naive_log.append(extra_time > 0)
                    solver_iter_log.append(num_iterations)
                    dist_ind = problems['distance_matrix'][ind]
                    update_accepted_stats(external_solutions, dist_ind, accepted,
                                         accepted_best_log, accepted_mean_log,
                                         accepted_cover_log, accepted_mean_sharing_log, LP)
                    solutions_store[f'time_{solver_time:.0f}_p{ind:d}_{method}_r{repetition}'] = solution

                    if PULL_FREQ > 0 and method.startswith('cuOpt') and populations is not None:
                        tmp = get_pools_bests(populations)
                        n_snap = len(tmp[0])
                        pool_times.extend(tmp[0])
                        pool_sizes.extend(tmp[1])
                        pool_costs_p.extend(tmp[2])
                        pool_methods.extend(n_snap * [method])
                        pool_full_times.extend(n_snap * [solver_time])
                        pool_problems.extend(n_snap * [problem_id])
                        pool_vehicles.extend(n_snap * [None])

                print(f'  {i_time+1}/{len(SOLVER_RUNTIMES)} '
                      f'problem {ind - problems_range[0] + 1}/{n_problems} '
                      f'method {i_method+1}/{len(METHODS)} done.')

        res = dict(
            method=methods_log,
            total_runtime=solver_times_log,
            init_runtime=init_times_log,
            problem_id=problem_ids_log,
            vehicles=vehicles_log,
            cost=costs_log,
            attempted_injections=n_external_log,
            accepted=n_accepted_log,
            rejected=n_rejected_log,
            unconsidered=n_unconsidered_log,
            accepted_best=accepted_best_log,
            accepted_mean=accepted_mean_log,
            accepted_cover=accepted_cover_log,
            accepted_mean_sharing=accepted_mean_sharing_log,
            extra_naive_injection=extra_naive_log,
            solver_iterations=solver_iter_log,
            success=successes_log,
            success_verification=successes2_log,
        )
        pools = dict(
            method=pool_methods,
            final_time=pool_full_times,
            problem_id=pool_problems,
            time=pool_times,
            pool_size=pool_sizes,
            vehicles=pool_vehicles,
            best_cost=pool_costs_p,
        )
        info = dict(
            config=config,
            external_costs=dict(vehicles=rlopt_vehicles, costs=rlopt_costs),
            solutions=solutions_store,
        )
        summary_verbosity = VERBOSE if i_time == len(SOLVER_RUNTIMES) - 1 else 0
        os.makedirs('./outputs', exist_ok=True)
        summarize_results(res, pools, info, OUT_PATH,
                          config['system']['allow_wandb'],
                          BASELINE, MAIN_METHOD, METHODS, summary_verbosity)


if __name__ == '__main__':
    main()
