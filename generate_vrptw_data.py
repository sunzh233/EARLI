#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Script to generate random VRPTW (Vehicle Routing Problem with Time Windows) training data.
This generates synthetic datasets with positions, demands, capacities, and time windows.
"""

import argparse
import os
import pickle
import numpy as np


def generate_vrptw_data(
    n_problems=128,
    problem_size=100,
    capacity_range=None,
    demand_range=(1, 9),
    p_norm=2,
    time_horizon=480.0,  # Total time available (e.g., 8 hours = 480 minutes)
    service_time_range=(5, 15),  # Service time range in minutes
    seed=None
):
    """
    Generate random VRPTW problem instances.
    
    Args:
        n_problems: Number of problem instances to generate
        problem_size: Number of nodes (including depot) in each problem
        capacity_range: Range for vehicle capacity [min, max]. If None, uses defaults
        demand_range: Range for customer demands (min, max)
        p_norm: Norm to use for distance calculation (2 for Euclidean)
        time_horizon: Total time available for routing
        service_time_range: Range for service time at each customer (min, max)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing problem data
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Set capacity range based on problem size if not provided
    if capacity_range is None:
        if problem_size <= 100:
            capacity_range = [50, 50]
        else:
            capacity_range = [40, 80]
    
    # Generate capacities
    capacities = np.random.randint(
        low=capacity_range[0], 
        high=capacity_range[1] + 1, 
        size=(n_problems,)
    ).astype(np.float32)
    
    # Generate demands
    demands = np.random.randint(
        demand_range[0], 
        demand_range[1] + 1, 
        size=(n_problems, problem_size)
    ).astype(np.float32)
    demands[:, 0] = 0  # Depot has no demand
    
    # Generate positions (uniformly distributed in [0, 1] x [0, 1])
    positions = np.random.rand(n_problems, problem_size, 2).astype(np.float32)
    
    # Calculate distance matrix
    distance_matrix = np.zeros((n_problems, problem_size, problem_size), dtype=np.float32)
    for i in range(n_problems):
        for j in range(problem_size):
            for k in range(problem_size):
                if j != k:
                    dist = np.linalg.norm(
                        positions[i, j] - positions[i, k], 
                        ord=p_norm
                    )
                    distance_matrix[i, j, k] = dist
    
    # Generate time windows
    # Strategy: Generate arrival time windows based on distance from depot
    t_min = np.zeros((n_problems, problem_size), dtype=np.float32)
    t_max = np.zeros((n_problems, problem_size), dtype=np.float32)
    dt = np.zeros((n_problems, problem_size), dtype=np.float32)  # service time
    
    for i in range(n_problems):
        # Depot time window: always available
        t_min[i, 0] = 0.0
        t_max[i, 0] = time_horizon
        dt[i, 0] = 0.0  # No service time at depot
        
        # For other nodes, generate time windows
        for j in range(1, problem_size):
            # Service time
            dt[i, j] = np.random.uniform(service_time_range[0], service_time_range[1])
            
            # Distance from depot to node
            dist_from_depot = distance_matrix[i, 0, j]
            
            # Earliest start time: account for travel from depot
            # Add some randomness to create variety
            base_earliest = dist_from_depot * time_horizon * 0.5
            earliest = base_earliest + np.random.uniform(-time_horizon * 0.1, time_horizon * 0.1)
            earliest = max(0.0, earliest)
            t_min[i, j] = earliest
            
            # Latest start time: ensure there's a feasible window
            # Window width varies between 10% to 50% of time horizon
            window_width = np.random.uniform(time_horizon * 0.1, time_horizon * 0.5)
            latest = earliest + window_width
            latest = min(latest, time_horizon - dt[i, j])  # Must finish before time horizon
            t_max[i, j] = max(latest, earliest + 1.0)  # Ensure window is at least 1 time unit
    
    # Prepare data dictionary
    data = {
        'positions': positions,
        'distance_matrix': distance_matrix,
        'demand': demands,
        'capacity': capacities,
        't_min': t_min,
        't_max': t_max,
        'dt': dt,  # service time
        'env_type': 'vrptw',
        'n_problems': n_problems,
        'problem_size': problem_size,
    }
    
    return data


def save_dataset(data, filename):
    """Save dataset to pickle file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved dataset to {filename}")
    print(f"  - Number of problems: {data['n_problems']}")
    print(f"  - Problem size: {data['problem_size']}")
    print(f"  - Capacity range: [{data['capacity'].min():.1f}, {data['capacity'].max():.1f}]")
    print(f"  - Demand range: [{data['demand'][data['demand']>0].min():.1f}, {data['demand'].max():.1f}]")
    print(f"  - Time window range: [{data['t_min'].min():.2f}, {data['t_max'].max():.2f}]")


def main():
    parser = argparse.ArgumentParser(
        description='Generate random VRPTW training data'
    )
    parser.add_argument(
        '--n_problems', 
        type=int, 
        default=128,
        help='Number of problem instances to generate (default: 128)'
    )
    parser.add_argument(
        '--problem_size', 
        type=int, 
        default=100,
        help='Number of nodes including depot (default: 100)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='datasets/vrptw_train_100.pkl',
        help='Output file path (default: datasets/vrptw_train_100.pkl)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=None,
        help='Random seed for reproducibility (default: None)'
    )
    parser.add_argument(
        '--capacity_min',
        type=int,
        default=None,
        help='Minimum vehicle capacity (default: 50 for size<=100, else 40)'
    )
    parser.add_argument(
        '--capacity_max',
        type=int,
        default=None,
        help='Maximum vehicle capacity (default: 50 for size<=100, else 80)'
    )
    parser.add_argument(
        '--time_horizon',
        type=float,
        default=480.0,
        help='Total time available for routing in minutes (default: 480)'
    )
    
    args = parser.parse_args()
    
    # Set capacity range
    capacity_range = None
    if args.capacity_min is not None and args.capacity_max is not None:
        capacity_range = [args.capacity_min, args.capacity_max]
    
    print(f"Generating VRPTW data...")
    print(f"  - Number of problems: {args.n_problems}")
    print(f"  - Problem size: {args.problem_size}")
    print(f"  - Time horizon: {args.time_horizon}")
    print(f"  - Seed: {args.seed}")
    
    # Generate data
    data = generate_vrptw_data(
        n_problems=args.n_problems,
        problem_size=args.problem_size,
        capacity_range=capacity_range,
        time_horizon=args.time_horizon,
        seed=args.seed
    )
    
    # Save data
    save_dataset(data, args.output)
    
    # Generate additional datasets of different sizes for comprehensive training
    if args.output == 'datasets/vrptw_train_100.pkl':  # Only auto-generate if using default
        print("\nGenerating additional datasets...")
        
        # Small dataset (50 nodes)
        print("\nGenerating dataset with 50 nodes...")
        data_50 = generate_vrptw_data(
            n_problems=128,
            problem_size=50,
            time_horizon=args.time_horizon,
            seed=args.seed + 1 if args.seed is not None else None
        )
        save_dataset(data_50, 'datasets/vrptw_train_50.pkl')
        
        # Medium dataset (200 nodes)
        print("\nGenerating dataset with 200 nodes...")
        data_200 = generate_vrptw_data(
            n_problems=64,  # Fewer problems for larger size
            problem_size=200,
            capacity_range=[40, 80],
            time_horizon=args.time_horizon * 1.5,  # More time for larger problems
            seed=args.seed + 2 if args.seed is not None else None
        )
        save_dataset(data_200, 'datasets/vrptw_train_200.pkl')
        
        print("\n✓ All datasets generated successfully!")


if __name__ == '__main__':
    main()
