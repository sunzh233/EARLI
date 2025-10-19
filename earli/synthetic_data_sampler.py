# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import torch

def sample_problems(n_problems, n_nodes, capacity_range=None, demand_range=(1,9), p_norm=2):
    # capacity
    if capacity_range is None:
        # use paper's default
        capacity_range = [50,50] if n_nodes == 100 else [40,80]
    capacities = torch.randint(low=capacity_range[0], high=capacity_range[1]+1, size=(n_problems,)).float()

    # demand
    demands = torch.randint(demand_range[0], demand_range[1]+1, [n_problems, n_nodes])
    demands[:, 0] = 0  # no demand at the depot

    # positions & distances
    positions = torch.rand(size=(n_problems, n_nodes, 2))
    distance_matrix = torch.cdist(positions, positions, p=p_norm)
    # numeric issue: cdist sometimes returns non-zero diagonal. so we force zeros:
    diag_ids = torch.arange(n_nodes)
    distance_matrix[:, diag_ids, diag_ids] = 0

    return positions, distance_matrix, demands, capacities
