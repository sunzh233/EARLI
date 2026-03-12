# Hybrid 训练与监控使用指南

本文档说明如何使用混合流程脚本完成 VRPTW 训练，并通过 TensorBoard 监测 PPO 与 tree_based 两段训练效果。

适用脚本与配置：
- run_full_vrptw_synthetic_curriculum_hybrid.sh
- config_train_vrptw_synthetic_curriculum_hybrid.yaml
- config_infer_vrptw_homberger_hybrid.yaml
- config_initialization_vrptw_homberger_hybrid.yaml

## 1. 流程概览

混合流程按阶段执行：
1. 每个规模先跑 PPO 训练。
2. 然后立刻用 tree_based 做一段精炼训练。
3. tree_based 输出模型作为下一阶段 warm start。
4. 可选在指定规模做 Homberger 推理与注入评测。

默认阶段：
- 50,100,150,200,300,400,500,600,800,1000

默认输出根目录：
- outputs/hybrid

## 2. 目录与产物说明

训练完成后主要文件在：
- outputs/hybrid/models
- outputs/hybrid/tensorboard
- outputs/hybrid/test_logs_homberger_*.pkl

模型命名规则：
- PPO 阶段模型：vrptw_model_synth_hybrid_ppo_N.m
- Tree 精炼模型：vrptw_model_synth_hybrid_N.m

其中 N 为阶段规模，例如 200、400、1000。

## 3. 最常用启动命令

3.1 全流程（生成数据 + 训练 + 测试）

./run_full_vrptw_synthetic_curriculum_hybrid.sh --profile conservative --run-test

3.2 已有数据时跳过数据生成

./run_full_vrptw_synthetic_curriculum_hybrid.sh --skip-data --profile conservative --run-test

3.3 快速烟测（仅到 200）

./run_full_vrptw_synthetic_curriculum_hybrid.sh --skip-data --stages 50,100,150,200 --eval-sizes 200 --run-test

3.4 只训练不测

./run_full_vrptw_synthetic_curriculum_hybrid.sh --skip-data

## 4. 关键参数说明（看这个就知道如何调）

脚本参数可分三类：

4.1 全局参数
- --profile conservative|aggressive
  - 选择默认阶段超参模板。
- --out-dir PATH
  - 指定输出根目录，模型和 TensorBoard 都会写到这个目录下。
- --stages CSV
  - 指定训练阶段规模列表。
- --initial-model PATH
  - 第一阶段的初始模型。
- --run-test
  - 训练阶段结束后执行注入评测。

4.2 PPO 阶段参数
- --stage-steps CSV
  - 每阶段总训练步数。
- --stage-n-steps CSV
  - PPO rollout horizon（train.n_steps）。
- --stage-lrs CSV
  - PPO 学习率。
- --stage-epochs CSV
  - PPO 每次更新的 epoch 数。
- --stage-batch-sizes CSV
  - PPO batch size。
- --stage-parallel CSV
  - PPO 并行问题数（train.n_parallel_problems）。

说明：
- 当前 hybrid 脚本内置了按 profile 的 PPO 正则调度（ent_coef、vf_coef、clip_range、target_kl）。
- 这些调度会随阶段自动变化，不需要额外传参。
- 目标是让 conservative 配置下 critic 更稳地拟合，同时抑制 aggressive 配置下 explained_variance 过快冲高带来的过拟合风险。

4.3 tree_based 精炼参数
- --tree-epochs CSV
  - 每阶段 tree 精炼 epoch 数。
- --tree-beams CSV
  - train.n_beams。
- --tree-data-steps CSV
  - muzero.data_steps_per_epoch。
- --tree-batch-sizes CSV
  - tree 训练 batch size。
- --tree-parallel CSV
  - tree 并行问题数。
- --tree-lrs CSV
  - tree 学习率。

重要约束：
- 所有 CSV 参数长度必须与 --stages 长度一致，否则脚本会报错并退出。

## 5. TensorBoard 使用方法

## 5.1 日志目录位置

混合脚本会把两段训练都写到：
- outputs/hybrid/tensorboard

并按阶段写入独立 run 名称：
- PPO：hybrid_ppo_stage_N
- Tree：hybrid_tree_stage_N

其中 N 为阶段规模。

## 5.2 启动 TensorBoard

在 EARLI 根目录执行：

tensorboard --logdir outputs/hybrid/tensorboard

## 5.3 如何筛选曲线

建议在 TensorBoard Scalars 页按以下前缀过滤：
- PPO 常用：train/, rollout/, eval/, custom/
- Tree 常用：tree_based/iter/, tree_based/epoch/

## 6. Tree 重采样监测（重点）

当前实现中，tree_based 每次数据重采样都会记录一组 iter 指标。这里的“每次重采样”对应一次 collector.play_game 调用。

可直接观察的重采样级指标：
- tree_based/iter/mean_best_return
- tree_based/iter/mean_forward_iters
- tree_based/iter/mean_data_samples
- tree_based/iter/mean_game_clocktime
- tree_based/iter/mean_num_vehicles
- tree_based/iter/training_samples_collected
- tree_based/iter/epoch
- tree_based/iter/iteration_in_epoch

每个 epoch 汇总指标：
- tree_based/epoch/training_samples_total
- tree_based/epoch/mean_best_return
- tree_based/epoch/mean_forward_passes
- tree_based/epoch/mean_env_steps
- tree_based/epoch/learning_rate

## 7. 你应该重点盯哪些曲线

1. tree_based/iter/mean_best_return
- 持续上升通常代表采样质量提升。

2. tree_based/iter/mean_num_vehicles
- 车辆数应稳定下降或持平，不应长期上升。

3. tree_based/iter/mean_game_clocktime
- 持续陡增通常说明搜索开销过大，需要降 beams 或并行。

4. tree_based/epoch/training_samples_total
- 过低可能意味着采样失败或有效轨迹不足。

5. PPO 的 custom/constraint_* 与 train/explained_variance
- explained_variance 高不等于解质量一定高，需要结合车辆数和注入结果一起看。

6. train/explained_variance 的实战判读
- conservative 若长期低于 0.7：优先看是否在缓慢上升，并结合 validation reward、车辆数趋势一起判断。
- aggressive 若过快冲高到 0.9+：重点观察 eval/mean_reward、注入质量是否同步提升；若不同步，通常是 critic 过拟合迹象。
- 推荐同时看 tree_based/iter/mean_best_return 与最终注入 summary，而不是单看 explained_variance。

## 7.1 本次 profile 稳定化策略

conservative：
- 降低熵系数曲线（ent_coef）
- 提高价值损失权重（vf_coef）
- 稍宽到中等 clip 与 target_kl

aggressive：
- 提高熵系数曲线（ent_coef）
- 降低价值损失权重（vf_coef）
- 更严格 clip 与 target_kl

作用：
- conservative 提升 critic 可学习性，缓解 explained_variance 长期偏低。
- aggressive 放慢“看起来很好”的 critic 拟合速度，降低策略/价值偏离验证分布的风险。

## 8. 常见问题与处理

1. 长度不匹配报错
- 检查 CSV 参数个数是否与 --stages 一致。

2. 显存压力过大
- 优先降低 --tree-parallel、--tree-batch-sizes、--tree-beams。
- 同时可用 conservative profile。

3. TensorBoard 看不到新曲线
- 确认训练实际在跑。
- 确认 logdir 指向 outputs/hybrid/tensorboard。
- 确认 run 名称是否筛选到了对应 stage。

4. run-test 阶段找不到模型
- 检查 outputs/hybrid/models 下是否存在 vrptw_model_synth_hybrid_N.m。

## 9. 推荐起步配置（24G 显存）

建议先这样跑：

./run_full_vrptw_synthetic_curriculum_hybrid.sh --skip-data --profile conservative --run-test

若你要验证“EV 过快增长是否被抑制”，推荐先跑 aggressive 短链：

./run_full_vrptw_synthetic_curriculum_hybrid.sh --skip-data --profile aggressive --stages 50,100,150,200 --eval-sizes 200 --run-test

若后段仍有压力，可先缩小阶段范围：

./run_full_vrptw_synthetic_curriculum_hybrid.sh --skip-data --profile conservative --stages 50,100,150,200,300,400 --eval-sizes 200,400 --run-test

