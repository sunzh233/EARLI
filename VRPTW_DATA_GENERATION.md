# VRPTW Data Generation

This document explains how to generate training data for the Vehicle Routing Problem with Time Windows (VRPTW).

## Quick Start

Generate VRPTW training datasets:

```bash
python3 generate_vrptw_data.py
```

This creates three datasets with different problem sizes:
- `datasets/vrptw_train_50.pkl` - 128 problems with 50 nodes
- `datasets/vrptw_train_100.pkl` - 256 problems with 100 nodes
- `datasets/vrptw_train_200.pkl` - 64 problems with 200 nodes

Verify the generated data:

```bash
python3 verify_vrptw_data.py
```

## Generated Datasets

The script has already generated the following datasets:

| Dataset | Problems | Nodes | Capacity | Time Horizon |
|---------|----------|-------|----------|--------------|
| vrptw_train_50.pkl | 128 | 50 | 50 | 480 min |
| vrptw_train_100.pkl | 256 | 100 | 50 | 480 min |
| vrptw_train_200.pkl | 64 | 200 | 40-80 | 720 min |

All datasets are ready for training and have been validated.

## Data Format

Each dataset contains:
- **positions**: Node coordinates in [0, 1] x [0, 1] space
- **distance_matrix**: Euclidean distances between all node pairs
- **demand**: Customer demands (depot = 0, customers > 0)
- **capacity**: Vehicle capacity for each problem
- **t_min**: Earliest arrival time at each node (time window start)
- **t_max**: Latest arrival time at each node (time window end)
- **dt**: Service time required at each node
- **env_type**: 'vrptw' identifier

## Using for Training

To train with the generated data, update `config_train.yaml`:

```yaml
eval:
  data_file: datasets/vrptw_train_100.pkl
```

Then run training:

```bash
python train_vrptw.py
```

## Generating Custom Datasets

Generate a custom dataset with specific parameters:

```bash
python3 generate_vrptw_data.py \
  --n_problems 512 \
  --problem_size 150 \
  --output datasets/vrptw_train_150.pkl \
  --seed 42 \
  --time_horizon 600 \
  --capacity_min 40 \
  --capacity_max 80
```

### Command Line Options

- `--n_problems`: Number of problem instances (default: 128)
- `--problem_size`: Number of nodes including depot (default: 100)
- `--output`: Output file path
- `--seed`: Random seed for reproducibility
- `--capacity_min`: Minimum vehicle capacity
- `--capacity_max`: Maximum vehicle capacity
- `--time_horizon`: Total time available in minutes (default: 480)

## Time Window Generation

The script generates realistic time windows using the following strategy:

1. **Depot**: Time window [0, time_horizon] with no service time
2. **Customers**:
   - Service time: Random 5-15 minutes
   - Earliest time: Based on distance from depot ± randomness
   - Latest time: Creates feasible windows of varying widths (10-50% of horizon)
   - All constraints are guaranteed to be satisfiable

## Validation

Run validation on any dataset:

```bash
python3 verify_vrptw_data.py
```

The validation checks:
- ✓ Depot has zero demand
- ✓ Depot time window starts at 0
- ✓ Depot has zero service time
- ✓ All customers have positive demand
- ✓ Valid time windows (t_max >= t_min)
- ✓ Correct data shapes and types

## Data Statistics

Example statistics from vrptw_train_100.pkl:
- 256 problems with 100 nodes each
- Capacity: 50 units
- Customer demand: 1-9 units
- Service time: 5-15 minutes
- Time horizon: 0-480 minutes
- Minimum vehicles needed: ~10-15 per problem

## Problem Statement (中文)

目前 earli 中用于训练 VRPTW 模型的数据不足，本脚本可以随机生成一批数据用于训练。

已生成的数据集：
- 小规模 (50节点): 128个问题实例
- 中规模 (100节点): 256个问题实例
- 大规模 (200节点): 64个问题实例

所有数据集包含完整的时间窗口、需求和容量约束，可直接用于训练。
