#!/usr/bin/env python3
"""
Test script to verify that generated VRPTW data works with the VRPTW environment.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import pickle
import torch
import numpy as np

def test_data_loading():
    """Test that data can be loaded from pickle file."""
    print("=" * 60)
    print("Test 1: Loading generated VRPTW data...")
    print("=" * 60)
    
    data_file = 'datasets/vrptw_train_100.pkl'
    
    if not os.path.exists(data_file):
        print(f"❌ FAILED: Data file {data_file} not found!")
        return False
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    # Check required fields
    required_fields = ['positions', 'distance_matrix', 'demand', 'capacity', 
                       't_min', 't_max', 'dt', 'env_type']
    
    for field in required_fields:
        if field not in data:
            print(f"❌ FAILED: Missing required field '{field}'")
            return False
    
    print(f"✓ All required fields present: {required_fields}")
    print(f"✓ Environment type: {data['env_type']}")
    print(f"✓ Number of problems: {data['n_problems']}")
    print(f"✓ Problem size: {data['problem_size']}")
    print()
    return True


def test_problem_loader():
    """Test that ProblemLoader can load the data."""
    print("=" * 60)
    print("Test 2: Testing ProblemLoader...")
    print("=" * 60)
    
    try:
        from earli.generate_data import ProblemLoader
        
        config = {
            'problem_setup': {
                'env': 'vrptw',
                'problem_range': None
            }
        }
        
        data = ProblemLoader.load_problem_data(
            config, 
            'datasets/vrptw_train_100.pkl',
            problem_range=None
        )
        
        print(f"✓ Data loaded successfully via ProblemLoader")
        print(f"✓ Number of problems: {data['n_problems']}")
        print(f"✓ Environment type: {data['env_type']}")
        
        # Verify time window fields
        assert 't_min' in data, "Missing t_min"
        assert 't_max' in data, "Missing t_max"
        assert 'dt' in data, "Missing dt"
        print(f"✓ Time window fields present: t_min, t_max, dt")
        
        # Verify data types
        assert isinstance(data['t_min'], torch.Tensor), "t_min should be torch.Tensor"
        assert isinstance(data['t_max'], torch.Tensor), "t_max should be torch.Tensor"
        assert isinstance(data['dt'], torch.Tensor), "dt should be torch.Tensor"
        print(f"✓ All fields are torch.Tensor as expected")
        print()
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vrptw_environment():
    """Test that VRPTW environment can be initialized with the data."""
    print("=" * 60)
    print("Test 3: Testing VRPTW environment initialization...")
    print("=" * 60)
    
    try:
        from earli.vrptw import VRPTW
        
        # Load config
        with open('config_train.yaml') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        
        # Update config to use CPU
        config['system']['env_device'] = 'cpu'
        config['system']['model_device'] = 'cpu'
        
        # Create environment
        print(f"Creating VRPTW environment with data file: {config['eval']['data_file']}")
        env = VRPTW(config, datafile=config['eval']['data_file'], env_type='eval')
        
        print(f"✓ VRPTW environment created successfully")
        print(f"✓ Number of parallel problems: {env.n_parallel_problems}")
        print(f"✓ Problem size: {env.problem_size}")
        
        # Test reset
        print("\nTesting environment reset...")
        obs = env.reset()
        print(f"✓ Environment reset successful")
        print(f"✓ Observation type: {type(obs)}")
        
        # Check that time window data is loaded
        assert hasattr(env, 't_min'), "Environment should have t_min"
        assert hasattr(env, 't_max'), "Environment should have t_max"
        assert hasattr(env, 'service_time'), "Environment should have service_time"
        print(f"✓ Time window attributes present in environment")
        
        # Test a few steps
        print("\nTesting environment steps...")
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"  Step {i+1}: action={action}, reward={reward:.4f}, done={done}")
        
        print(f"✓ Environment steps executed successfully")
        print()
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_validity():
    """Test that generated data is valid (time windows, demands, etc.)."""
    print("=" * 60)
    print("Test 4: Validating data properties...")
    print("=" * 60)
    
    try:
        with open('datasets/vrptw_train_100.pkl', 'rb') as f:
            data = pickle.load(f)
        
        # Check depot properties
        assert np.all(data['demand'][:, 0] == 0), "Depot demand should be 0"
        print(f"✓ Depot has zero demand")
        
        assert np.all(data['t_min'][:, 0] == 0), "Depot t_min should be 0"
        print(f"✓ Depot time window starts at 0")
        
        assert np.all(data['dt'][:, 0] == 0), "Depot service time should be 0"
        print(f"✓ Depot has zero service time")
        
        # Check that all customers have positive demands
        customer_demands = data['demand'][:, 1:]
        assert np.all(customer_demands > 0), "All customers should have positive demand"
        print(f"✓ All customers have positive demand")
        
        # Check time window validity
        assert np.all(data['t_max'] >= data['t_min']), "t_max should be >= t_min"
        print(f"✓ Time windows are valid (t_max >= t_min)")
        
        # Check capacity vs demand
        for i in range(data['n_problems']):
            total_demand = data['demand'][i].sum()
            capacity = data['capacity'][i]
            # At least one vehicle should be needed
            assert total_demand > 0, "Total demand should be positive"
            print(f"  Problem {i}: total_demand={total_demand:.1f}, capacity={capacity:.1f}, " 
                  f"min_vehicles={int(np.ceil(total_demand / capacity))}")
            if i >= 2:  # Just show first 3
                break
        
        print(f"✓ Demand and capacity are reasonable")
        print()
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("VRPTW Data Generation Test Suite")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Data Loading", test_data_loading()))
    results.append(("ProblemLoader", test_problem_loader()))
    results.append(("VRPTW Environment", test_vrptw_environment()))
    results.append(("Data Validity", test_data_validity()))
    
    # Print summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests PASSED! The generated VRPTW data is ready for training.\n")
        return 0
    else:
        print("\n⚠️  Some tests FAILED. Please review the errors above.\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
