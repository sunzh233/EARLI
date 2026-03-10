# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import warnings

import pandas as pd
import cudf
import numpy
import numpy as np
import torch
import cuopt
from cuopt import routing

from .cuopt_parser import SolutionPopulation


def solver_initialization(time_limit, climbers, pull_frequency=0):
    solver_settings = routing.SolverSettings()
    # Set seconds update and climbers
    solver_settings.set_time_limit(time_limit)
    # solver_settings.set_number_of_climbers(climbers)
    if pull_frequency > 0:
        solver_settings.pull("population.txt", pull_frequency)
    return solver_settings


def convert_solutions_to_series(sols):
    """
    sols = [
        [timestamp_1, [route_1 ... route_2 ... route_3]],
        [timestamp_2, [route_1 ... route_2 ... route_3 .. route_4]],
        [timestamp_3, [route_1 ... route_2]]
    ]
    """
    routes = []
    route_ids = []
    types = []
    sol_offsets = [0]
    for i, sol in enumerate(sols):
        sol_routes = sol[1]
        routes.extend(sol_routes)
        route_ids.extend(np.cumsum(np.array(sol[1]) == 0) - 1)
        types.extend(["Depot" if node==0 else "Delivery" for node in sol_routes])
        # types.extend(["Depot"] + ["Delivery"] * (len(sol_routes) - 1))
        sol_offsets.append(sol_offsets[i] + len(sol_routes))
    return  cudf.Series(route_ids, dtype=int), cudf.Series(routes, dtype=int), cudf.Series(types), cudf.Series(sol_offsets, dtype=int)


class CuOptSolver(object):
    def __init__(self, config):
        self.config = config
        self.output_population = config['cuopt']['pull_frequency'] > 0
        self.solver_settings = solver_initialization(time_limit=config['cuopt']['time_limit'],
                                                     climbers=config['cuopt']['climbers'],
                                                     pull_frequency=config['cuopt']['pull_frequency'])
        self.env_type = config['problem_setup']['env']
        self.cuopt_version = cuopt.__version__

    def solve(self, data_model, verbose=1):
        routing_solution = routing.Solve(data_model, self.solver_settings)
        final_cost = routing_solution.get_total_objective()
        vehicle_count = routing_solution.get_vehicle_count()
        cu_status = routing_solution.get_status()
        if cu_status != 0:
            print("""
            --------------------------------------------------------------------------------------------------
              !!!!!!!!!!!!        Failed: Solution within constraints could not be found     !!!!!!!!!!!!!!!!
            -------------------------------------------------------------------------------------------------- """)
        elif verbose >= 1:
            print("Final Cost         : ", final_cost)
            print("Number of Vehicles : ", vehicle_count)
        if self.output_population:
            population = SolutionPopulation("population.txt")
        else:
            population = None
        return routing_solution, population

    def create_data_model(self, demand, distance_matrix, vehicle_capacity, n_carriers, t_min=None, t_max=None, dt=None,
                          set_fixed_carriers=False, drop_return_trips=False, last_drop_return_trip=None, injected_solutions=None):
        if type(distance_matrix) == torch.Tensor:
            distance_matrix = distance_matrix.cpu().numpy()
        n_nodes = distance_matrix.shape[0]
        data_model = routing.DataModel(n_nodes, n_carriers)
        # Normalize distance matrix to cudf.DataFrame
        distance_matrix = (
            distance_matrix.tolist()
            if not isinstance(distance_matrix, cudf.DataFrame)
            else distance_matrix
        )
        distance_df = cudf.DataFrame(distance_matrix).astype(np.float32)
        data_model.add_cost_matrix(distance_df)

        # Normalize capacity to cudf.Series
        if isinstance(vehicle_capacity, cudf.Series):
            capacity = vehicle_capacity.astype('int32')
        else:
            try:
                # pandas Series or numpy array or list
                capacity = cudf.Series(np.asarray(vehicle_capacity, dtype=np.int32))
            except Exception:
                capacity = cudf.Series(list(vehicle_capacity)).astype('int32')
        if self.env_type.startswith('pdp'):
            pickup_indices = np.arange(1, n_nodes // 2 +1 ).tolist()
            delivery_indices = np.arange(n_nodes // 2 + 1, n_nodes).tolist()
            data_model.set_order_locations=np.arange(1, 1 + n_nodes // 2).tolist()
            data_model.set_pickup_delivery_pairs(cudf.Series(pickup_indices, dtype='int32'), cudf.Series(delivery_indices, dtype='int32'))

        if self.env_type.endswith('tw'):
            # t_min, t_max, dt may be lists, numpy arrays, pandas Series or cudf.Series
            def _to_cudf_series(x, dtype=None):
                if x is None:
                    return None
                if isinstance(x, cudf.Series):
                    return x if dtype is None else x.astype(dtype)
                if isinstance(x, pd.Series):
                    return cudf.Series(x.to_numpy()).astype(dtype) if dtype is not None else cudf.Series(x.to_numpy())
                return cudf.Series(np.asarray(x)).astype(dtype) if dtype is not None else cudf.Series(np.asarray(x))

            # vehicle windows: scalar per vehicle provided as first element or list-like
            if t_min is not None and t_max is not None:
                try:
                    v_earliest = [t_min[0]] * n_carriers if not isinstance(t_min, (pd.Series, cudf.Series, list, tuple, np.ndarray)) else [t_min[0]] * n_carriers
                except Exception:
                    v_earliest = [t_min] * n_carriers
                try:
                    v_latest = [t_max[0]] * n_carriers if not isinstance(t_max, (pd.Series, cudf.Series, list, tuple, np.ndarray)) else [t_max[0]] * n_carriers
                except Exception:
                    v_latest = [t_max] * n_carriers
                data_model.set_vehicle_time_windows(_to_cudf_series(v_earliest, dtype='int32'), _to_cudf_series(v_latest, dtype='int32'))

            data_model.set_order_time_windows(_to_cudf_series(t_min, dtype='int32'), _to_cudf_series(t_max, dtype='int32'))
            if dt is not None:
                data_model.set_order_service_times(_to_cudf_series(dt, dtype='int32'))

        if set_fixed_carriers:
            if self.env_type != 'vrp':
                warnings.warn('Setting fixed number of vehicles in non-VRP env.')
            data_model.set_min_vehicles(n_carriers)

        # Ensure demand is a cudf.Series of int32
        if isinstance(demand, cudf.Series):
            demand_series = demand.astype('int32')
        else:
            try:
                demand_series = cudf.Series(np.asarray(demand, dtype=np.int32))
            except Exception:
                demand_series = cudf.Series([int(x) for x in list(demand)])

        data_model.add_capacity_dimension("demand", demand_series, capacity)
        drop_return_trips_ = [drop_return_trips] * n_carriers
        if last_drop_return_trip is not None:
            drop_return_trips_[-1] = last_drop_return_trip
        data_model.set_drop_return_trips(cudf.Series(drop_return_trips_))

        if injected_solutions is not None and len(injected_solutions) > 0:
            data_model.add_initial_solutions(*convert_solutions_to_series(injected_solutions))

        return data_model
