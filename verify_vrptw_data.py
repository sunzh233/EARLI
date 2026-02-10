#!/usr/bin/env python3
"""
Simple verification script to show that generated VRPTW data is correct.
This script has minimal dependencies (only numpy and pickle).
"""

import pickle
import numpy as np


def main():
    print("\n" + "=" * 70)
    print(" VRPTW Generated Data Verification")
    print("=" * 70 + "\n")
    
    # Check all generated datasets
    datasets = [
        'datasets/vrptw_train_50.pkl',
        'datasets/vrptw_train_100.pkl',
        'datasets/vrptw_train_200.pkl',
    ]
    
    for dataset_path in datasets:
        print(f"\nDataset: {dataset_path}")
        print("-" * 70)
        
        try:
            with open(dataset_path, 'rb') as f:
                data = pickle.load(f)
            
            # Basic info
            print(f"✓ Number of problems: {data['n_problems']}")
            print(f"✓ Problem size: {data['problem_size']} nodes (including depot)")
            print(f"✓ Environment type: {data['env_type']}")
            
            # Data shapes
            print(f"\nData Shapes:")
            print(f"  positions:        {data['positions'].shape}")
            print(f"  distance_matrix:  {data['distance_matrix'].shape}")
            print(f"  demand:           {data['demand'].shape}")
            print(f"  capacity:         {data['capacity'].shape}")
            print(f"  t_min:            {data['t_min'].shape}")
            print(f"  t_max:            {data['t_max'].shape}")
            print(f"  dt (service):     {data['dt'].shape}")
            
            # Statistics
            print(f"\nStatistics:")
            print(f"  Capacity range:      [{data['capacity'].min():.1f}, {data['capacity'].max():.1f}]")
            customer_demands = data['demand'][:, 1:]  # Exclude depot
            print(f"  Customer demand:     [{customer_demands.min():.1f}, {customer_demands.max():.1f}]")
            customer_service = data['dt'][:, 1:]  # Exclude depot
            print(f"  Service time:        [{customer_service.min():.2f}, {customer_service.max():.2f}]")
            print(f"  Time horizon:        [0.00, {data['t_max'].max():.2f}]")
            
            # Example problem
            print(f"\nExample Problem (problem 0):")
            print(f"  Vehicle capacity:    {data['capacity'][0]:.1f}")
            total_demand = data['demand'][0].sum()
            min_vehicles = int(np.ceil(total_demand / data['capacity'][0]))
            print(f"  Total demand:        {total_demand:.1f}")
            print(f"  Min vehicles needed: {min_vehicles}")
            print(f"  Depot time window:   [0.00, {data['t_max'][0, 0]:.2f}]")
            print(f"  Customer 1 at:       ({data['positions'][0, 1, 0]:.3f}, {data['positions'][0, 1, 1]:.3f})")
            print(f"  Customer 1 demand:   {data['demand'][0, 1]:.1f}")
            print(f"  Customer 1 window:   [{data['t_min'][0, 1]:.2f}, {data['t_max'][0, 1]:.2f}]")
            print(f"  Customer 1 service:  {data['dt'][0, 1]:.2f}")
            
            # Validation checks
            print(f"\nValidation Checks:")
            checks = [
                (np.all(data['demand'][:, 0] == 0), "Depot has zero demand"),
                (np.all(data['t_min'][:, 0] == 0), "Depot time window starts at 0"),
                (np.all(data['dt'][:, 0] == 0), "Depot has zero service time"),
                (np.all(customer_demands > 0), "All customers have positive demand"),
                (np.all(data['t_max'] >= data['t_min']), "Valid time windows (t_max >= t_min)"),
                (data['positions'].shape == (data['n_problems'], data['problem_size'], 2), "Correct position shape"),
                (data['distance_matrix'].shape == (data['n_problems'], data['problem_size'], data['problem_size']), "Correct distance matrix shape"),
            ]
            
            all_valid = True
            for check, desc in checks:
                status = "✓" if check else "✗"
                print(f"  {status} {desc}")
                if not check:
                    all_valid = False
            
            if all_valid:
                print(f"\n✅ Dataset {dataset_path} is VALID and ready for training!")
            else:
                print(f"\n❌ Dataset {dataset_path} has validation errors!")
                
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(" Summary")
    print("=" * 70)
    print("\nGenerated VRPTW datasets are ready for training!")
    print("\nTo use in training, update config_train.yaml:")
    print("  eval:")
    print("    data_file: datasets/vrptw_train_100.pkl")
    print("\nTo generate new datasets:")
    print("  python3 generate_vrptw_data.py --help")
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()
