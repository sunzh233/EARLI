# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# class Route:
#     def __init__(self):
#         self.number = None
#         self.ids = []
import logging

import numpy as np


class Solution:
    def __init__(self):
        self.number = None
        self.routes = []


class SolutionPopulation:
    def __init__(self, file_path):
        self.times = []
        self.population_snapshots = []
        self.clearing_radii = []
        self.snapshot_costs = []
        self.snapshot_vehicles = []
        self.valid_snapshots = []

        solutions = []
        current_solution = None

        with open(file_path, 'r') as file:
            for line in file:
                stripped_line = line.strip()
                if stripped_line.startswith('Reserve update at:'):
                    if solutions:
                        self.population_snapshots.append(solutions)
                    current_population_time = int(stripped_line.split(': ')[1][:-1])
                    self.times.append(current_population_time)
                    solutions = []
                elif stripped_line.startswith('Clearing radius:'):
                    current_clearing_radius = float(stripped_line.split(': ')[1])
                    self.clearing_radii.append(current_clearing_radius)
                elif stripped_line.startswith('Solution'):
                    if current_solution:
                        solutions.append(current_solution)
                    current_solution = []
                    # current_solution_number = int(stripped_line.split(' ')[1])
                elif stripped_line.startswith('Route'):
                    split_line = stripped_line.split(':')
                    route = [int(x) for x in split_line[1].split()]
                    current_solution.append(route)
            # Append any remaining objects
            if current_solution:
                solutions.append(current_solution)
            if solutions:
                self.population_snapshots.append(solutions)

    def update_costs(self, distance_matrix, capacity, demands_for_solver, n_customers,
                     return_to_depot=False):
        # Note: n_customers does not include depot
        for i_snap, snapshot in enumerate(self.population_snapshots):
            vehicles = []
            costs = []
            valid_solutions = []
            for solution in snapshot:
                cost = 0
                valid = np.sum([len(route) for route in solution]) >= n_customers
                for ind, route in enumerate(solution):
                    load = 0
                    sites = [0] + route
                    if ind != len(solution) - 1 or return_to_depot:
                        sites += [0]
                    for source, target in zip(sites[:-1], sites[1:]):
                        cost += distance_matrix[int(source), int(target)]
                        load += demands_for_solver[target]
                    if load > capacity:
                        # logging.warning(f'Snapshot {i_snap+1}: Route {ind} is overloaded. Invalid solution!')
                        valid = False
                vehicles.append(len(solution))
                costs.append(cost)
                valid_solutions.append(valid)
            self.snapshot_vehicles.append(np.array(vehicles))
            self.snapshot_costs.append(np.array(costs))
            self.valid_snapshots.append(valid_solutions)


if __name__ == 'main':
    # replace with your file path
    file_path = "/home/emeirom/code/RLOptimizer/population.txt"
    parsed_data = SolutionPopulation(file_path)
    print(parsed_data.times)
