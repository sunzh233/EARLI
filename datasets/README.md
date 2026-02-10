# VRPTW Training Datasets

This directory contains generated training datasets for the Vehicle Routing Problem with Time Windows (VRPTW).

## Generated Datasets

The following datasets have been generated using `generate_vrptw_data.py`:

- **vrptw_train_50.pkl**: 128 problems with 50 nodes each
- **vrptw_train_100.pkl**: 256 problems with 100 nodes each  
- **vrptw_train_200.pkl**: 64 problems with 200 nodes each

## Dataset Format

Each dataset is a pickle file containing a dictionary with the following fields:

- `positions`: Node positions (n_problems, problem_size, 2) - coordinates in [0, 1] x [0, 1]
- `distance_matrix`: Pairwise distances (n_problems, problem_size, problem_size)
- `demand`: Customer demands (n_problems, problem_size) - depot has 0 demand
- `capacity`: Vehicle capacities (n_problems,)
- `t_min`: Earliest arrival times (n_problems, problem_size) - time windows start
- `t_max`: Latest arrival times (n_problems, problem_size) - time windows end
- `dt`: Service times at each node (n_problems, problem_size)
- `env_type`: 'vrptw' - environment type identifier
- `n_problems`: Number of problem instances
- `problem_size`: Number of nodes (including depot)

## Generating New Datasets

To generate new VRPTW training datasets, use the `generate_vrptw_data.py` script:

```bash
# Basic usage - generates default datasets
python generate_vrptw_data.py

# Custom dataset
python generate_vrptw_data.py \
  --n_problems 512 \
  --problem_size 150 \
  --output datasets/vrptw_train_150.pkl \
  --seed 42 \
  --time_horizon 600

# View all options
python generate_vrptw_data.py --help
```

### Available Options

- `--n_problems`: Number of problem instances (default: 128)
- `--problem_size`: Number of nodes including depot (default: 100)
- `--output`: Output file path (default: datasets/vrptw_train_100.pkl)
- `--seed`: Random seed for reproducibility
- `--capacity_min`: Minimum vehicle capacity
- `--capacity_max`: Maximum vehicle capacity
- `--time_horizon`: Total time available in minutes (default: 480)

## Time Window Generation Strategy

Time windows are generated to create realistic and challenging routing problems:

1. **Depot**: Always has time window [0, time_horizon] with no service time
2. **Customers**: 
   - Service time: Randomly selected from 5-15 minutes
   - Earliest time: Based on distance from depot with added randomness
   - Latest time: Creates windows of varying widths (10-50% of time horizon)
   - All time windows are guaranteed to be feasible

## Using Generated Data for Training

Update `config_train.yaml` to point to your generated dataset:

```yaml
eval:
  data_file: datasets/vrptw_train_100.pkl
```

Then run training:

```bash
python train_vrptw.py
```
