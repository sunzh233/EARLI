# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
This module provides an API for HGS' local search:
https://github.com/vidalt/HGS-CVRP
via the PyHygese Python wrapping for HGS:
https://github.com/chkwon/PyHygese
"""

import hygese as ohgs
import numpy as np

def local_search(solutions, dist_matrix, demands, vehicle_capacity,
                 penalty_capacity=1000, duration_limit=10000, is_duration_constraint=True, solver=None):
    if solver is None:
        solver = ohgs.Solver()
    improved_solutions, n_routes, costs = solver.apply_local_search_multiple(
        solutions, dist_matrix, demands, vehicle_capacity, penalty_capacity,
        duration_limit, is_duration_constraint)
    return improved_solutions, n_routes, costs
