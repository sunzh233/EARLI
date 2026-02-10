# Quick Start Guide - VRPTW Data Generation

## 问题 (Problem)
目前 earli 中用于训练 VRPTW 模型的数据不足。

Currently, there is insufficient data for training the VRPTW model in earli.

## 解决方案 (Solution)
已实现随机数据生成工具，可生成用于训练的 VRPTW 数据集。

Implemented a random data generation tool to create VRPTW training datasets.

## 快速开始 (Quick Start)

### 1. 生成数据 (Generate Data)
```bash
cd /home/runner/work/EARLI/EARLI
python3 generate_vrptw_data.py
```

这会生成三个不同规模的数据集：
This generates three datasets of different sizes:
- 50 节点 (nodes): 128 个问题 (problems)
- 100 节点 (nodes): 256 个问题 (problems)
- 200 节点 (nodes): 64 个问题 (problems)

### 2. 验证数据 (Verify Data)
```bash
python3 verify_vrptw_data.py
```

### 3. 训练模型 (Train Model)
```bash
python train_vrptw.py
```

配置文件 `config_train.yaml` 已更新为使用新生成的数据。
The config file `config_train.yaml` is already updated to use the generated data.

## 自定义生成 (Custom Generation)

生成特定规模的数据集：
Generate a dataset with specific size:

```bash
python3 generate_vrptw_data.py \
  --n_problems 512 \
  --problem_size 150 \
  --output datasets/vrptw_train_150.pkl \
  --seed 42
```

查看所有选项：
View all options:
```bash
python3 generate_vrptw_data.py --help
```

## 数据格式 (Data Format)

每个数据集包含：
Each dataset contains:
- positions: 节点坐标 (node coordinates)
- distance_matrix: 距离矩阵 (distance matrix)
- demand: 客户需求 (customer demands)
- capacity: 车辆容量 (vehicle capacity)
- t_min: 时间窗口开始 (time window start)
- t_max: 时间窗口结束 (time window end)
- dt: 服务时间 (service time)

## 已生成的数据集 (Generated Datasets)

✅ datasets/vrptw_train_50.pkl
✅ datasets/vrptw_train_100.pkl
✅ datasets/vrptw_train_200.pkl

所有数据集已验证并可以直接使用。
All datasets are validated and ready to use.

## 文档 (Documentation)

详细文档：
Detailed documentation:
- 📖 VRPTW_DATA_GENERATION.md - 完整使用指南 (Complete usage guide)
- 📖 datasets/README.md - 数据集说明 (Dataset documentation)
- 📖 IMPLEMENTATION_SUMMARY.md - 技术实现细节 (Technical details)

## 问题和帮助 (Issues & Help)

如有问题，请查看文档或运行验证脚本：
For issues, check documentation or run verification:
```bash
python3 verify_vrptw_data.py
python3 test_vrptw_data.py
```
