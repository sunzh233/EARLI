# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Training entry point for VRPTW and PDPTW.

Usage
-----
# VRPTW (prepare dataset first):
    python -m earli.benchmark_parser vrptw homberger/homberger_200_customer_instances \
        datasets/vrptw_200.pkl
    python -m earli.train --config config_vrptw_train.yaml

# PDPTW (prepare dataset first):
    python -m earli.benchmark_parser pdptw "li&lim benchmark/pdp_100" \
        datasets/pdptw_100.pkl
    python -m earli.train --config config_pdptw_train.yaml

Two training methods are supported, controlled by ``train.method`` in the config:

* ``ppo`` (Stable-Baselines3 PPO, requires ``compatibility_mode: stable_baselines``)
  Simple on-policy rollout training – the same path used for CVRP in the
  ExampleTrain notebook.

* ``tree_based`` (custom tree-search PPO, requires ``compatibility_mode: null``)
  Data is collected via multi-beam tree search (``SelfPlay``), which
  explores several trajectory branches at each step and selects the
  highest-return one.  The resulting trajectories are of higher quality
  than a plain single-beam rollout, which speeds up learning.
  ``train.n_beams`` controls the breadth of the tree (set >= 2 for real
  tree search; 1 degenerates to a single-beam rollout with PPO updates).
"""

import argparse
import os
import shutil
from copy import deepcopy
from datetime import datetime

import torch
import math
import logging
import numpy as np
import torch.nn.functional as F
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env import VecMonitor
from yaml import SafeLoader

from .models.attention_model import PosAttentionModel
from .models.sampler import Sampler
from .self_play import SelfPlay
from .utils.general_usage_utils import ignore_legacy_wandb_warnings
from .utils.nv import verify_consistent_config
from .vrp import VRP
from .vrptw import VRPTW
from .pdptw import PDPTW
from .pomo_tw_utils import augment_coords_8fold, augment_vrptw_dataset

# SummaryWriter is imported lazily inside functions that use it to avoid
# requiring tensorboard as a hard dependency for all modules.
_SummaryWriter = None


def _get_summary_writer(log_dir):
    """Lazily import and return a SummaryWriter instance."""
    global _SummaryWriter
    if _SummaryWriter is None:
        from torch.utils.tensorboard import SummaryWriter
        _SummaryWriter = SummaryWriter
    return _SummaryWriter(log_dir=log_dir)

ignore_legacy_wandb_warnings()

ENV_CLASSES = {
    'vrp'  : VRP,
    'vrptw': VRPTW,
    'pdptw': PDPTW,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_datafile(config, env_name):
    """Return the datafile path (or list of paths) from config, raising if missing."""
    data_files = config['eval'].get('data_files')
    data_file = config['eval'].get('data_file')

    if data_files:
        for f in data_files:
            if not os.path.exists(f):
                raise FileNotFoundError(
                    f"Dataset file not found: {f}\n"
                    f"Generate it first with:\n"
                    f"  python -m earli.benchmark_parser {env_name} <benchmark_dir> {f}"
                )
        return data_files
    elif data_file:
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"Dataset file not found: {data_file}\n"
                f"Generate it first with:\n"
                f"  python -m earli.benchmark_parser {env_name} <benchmark_dir> {data_file}"
            )
        return data_file
    else:
        raise ValueError("Config must specify either 'eval.data_file' or 'eval.data_files'.")
                


def make_lr_schedule(initial_lr: float,
                     schedule_type: str = "constant",
                     min_lr: float = 0.0,
                     exp_decay: float = 5.0,
                     step_ratio: float = 0.5,
                     step_fraction: float = 0.5):
    schedule_type = (schedule_type or "constant").lower()
    min_lr = float(min_lr)
    initial_lr = float(initial_lr)

    if schedule_type == "constant":
        return initial_lr

    if schedule_type == "linear":
        def linear_schedule(progress_remaining: float):
            return min_lr + (initial_lr - min_lr) * float(progress_remaining)
        return linear_schedule

    if schedule_type == "cosine":
        def cosine_schedule(progress_remaining: float):
            completed = 1.0 - float(progress_remaining)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * completed))
            return min_lr + (initial_lr - min_lr) * cosine_factor
        return cosine_schedule

    if schedule_type == "exp":
        def exp_schedule(progress_remaining: float):
            completed = 1.0 - float(progress_remaining)
            lr = initial_lr * math.exp(-float(exp_decay) * completed)
            return max(min_lr, lr)
        return exp_schedule

    if schedule_type == "step":
        step_ratio_clamped = min(max(float(step_ratio), 0.0), 1.0)
        step_fraction_clamped = min(max(float(step_fraction), 0.0), 1.0)

        def step_schedule(progress_remaining: float):
            completed = 1.0 - float(progress_remaining)
            lr = initial_lr if completed < step_fraction_clamped else initial_lr * step_ratio_clamped
            return max(min_lr, lr)
        return step_schedule

    raise ValueError(f"Unknown lr schedule type: {schedule_type}")


def _ppo_update(model, training_data, config,
                clip_range: float = 0.2,
                value_coef: float = 0.5,
                n_epochs: int = 4):
    """Apply PPO gradient updates to *model* using the collected *training_data*.

    Parameters
    ----------
    model : PosAttentionModel
        The policy/value network.
    training_data : TensorDict
        Must contain keys ``observations``, ``actions``, ``log_prob``,
        ``returns``.  Produced by
        ``SearchTree.build_training_data_from_history()``.
    config : dict
        Full training config.
    clip_range : float
        PPO clipping coefficient ε.
    value_coef : float
        Weight of the value-function loss term.
    n_epochs : int
        Number of mini-batch passes over the collected data.
    """
    n_samples  = len(training_data)
    if n_samples == 0:
        return
    batch_size = min(config['train']['batch_size'], n_samples)
    device = next(model.parameters()).device
    model.train()

    for _ in range(n_epochs):
        indices = torch.randperm(n_samples)
        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start: start + batch_size]
            batch     = training_data[batch_idx]

            obs          = batch['observations'].to(device)
            actions      = batch['actions'].to(device)
            old_log_prob = batch['log_prob'].to(device)
            returns      = batch['returns'].to(device).float()

            # Forward pass with gradient tracking
            values, new_log_prob, entropy = model.evaluate_actions(obs, actions)
            values = values.squeeze(-1).float()

            # Advantage (no GAE; use simple Monte-Carlo returns)
            advantages = returns - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO clipped surrogate objective
            log_ratio   = new_log_prob.float() - old_log_prob.float()
            ratio       = log_ratio.exp()
            surr1       = ratio * advantages
            surr2       = ratio.clamp(1.0 - clip_range, 1.0 + clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss  = F.mse_loss(values, returns)

            loss = policy_loss + value_coef * value_loss

            model.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            model.optimizer.step()

    model.eval()


# ---------------------------------------------------------------------------
# POMO-TW utilities (re-exported from pomo_tw_utils for internal convenience)
# ---------------------------------------------------------------------------

_augment_coords_8fold = augment_coords_8fold
_augment_vrptw_dataset = augment_vrptw_dataset


# ---------------------------------------------------------------------------
# Tree-based training
# ---------------------------------------------------------------------------

def _train_tree_based(config, env_class, datafile, total_epochs):
    """Training loop for the ``tree_based`` method.

    Data is collected via multi-beam tree search (``SelfPlay.play_game``
    with ``training=True``).  The best trajectory per problem instance is
    extracted from the tree and used for PPO-style gradient updates.

    Parameters
    ----------
    config : dict
        Verified training config (``compatibility_mode`` must be *None*).
    env_class : type
        Environment class (VRP, VRPTW, or PDPTW).
    datafile : str or list[str]
        Path(s) to the dataset PKL file(s).
    total_epochs : int
        Number of training epochs.
    """
    print(f"[tree_based] Starting training for {total_epochs} epochs …")
    env_name   = config['problem_setup']['env'].upper()
    n_parallel = config['train']['n_parallel_problems']
    sampler    = Sampler(config)

    # Build model (no stable_baselines wrapper needed)
    dummy_env = env_class(config, datafile=datafile, env_type='train')
    obs_space, act_space = dummy_env.spaces
    model = PosAttentionModel(
        obs_space, act_space, config=config, sampler=sampler
    )

    # Load pretrained weights if provided
    pretrained = config['train'].get('pretrained_fname')
    if pretrained and os.path.exists(pretrained):
        checkpoint = torch.load(pretrained, weights_only=False)
        params = checkpoint.get('model_state_dict', checkpoint)
        params = {k.replace('._orig_mod', ''): v for k, v in params.items()}
        missing, unexpected = model.load_state_dict(params, strict=False)
        if missing:
            print(f"[tree_based] Missing keys from pretrained model: {missing}")
        if unexpected:
            print(f"[tree_based] Unexpected keys in pretrained model: {unexpected}")
        print(f"[tree_based] Loaded pretrained weights from {pretrained}")

    data_steps = config['muzero']['data_steps_per_epoch']

    print(f"[tree_based] Training {env_name} for {total_epochs} epochs "
          f"({data_steps} problems/epoch, {n_parallel} parallel) …")

    tb_dir = config.get('system', {}).get('tensorboard_logdir', 'outputs/tensorboard')
    default_tb_name = f"earli_tree_{os.path.splitext(os.path.basename(config['train']['save_model_path']))[0]}"
    tb_name = config.get('system', {}).get('run_name') or default_tb_name
    tb_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tb_run_dir = os.path.join(tb_dir, f"{tb_name}_{tb_stamp}")
    os.makedirs(tb_run_dir, exist_ok=True)
    tb_writer = _get_summary_writer(tb_run_dir)
    print(f"[tree_based] TensorBoard run dir: {tb_run_dir}")

    logger = logging.getLogger(__name__)
    logger.info(
        f"[tree_based] Training {env_name} for {total_epochs} epochs "
        f"({data_steps} problems/epoch, {n_parallel} parallel) …"
    )

    global_resample_step = 0

    for epoch in range(total_epochs):
        # ---- data collection via tree search ----
        collector = SelfPlay(
            Game=env_class,
            config=config,
            seed=config['train']['seed'] + epoch,
            env_type='train',
            n_beams=config['train']['n_beams'],
            model=model,
            datafile=datafile,
            n_problems=n_parallel,
        )

        n_iterations = max(1, data_steps // n_parallel)
        all_training_data = []

        # Accumulate per-iteration diagnostic stats for logging
        iter_best_returns = []
        iter_forward_passes = []
        iter_env_steps = []
        for iter_idx in range(n_iterations):
            print(f"[tree_based] Epoch {epoch + 1}/{total_epochs}, iteration {iter_idx + 1}/{n_iterations} …")
            _, infos = collector.play_game(deterministic=False, training=True)
            td = infos.get('training_data')
            if td is not None and len(td) > 0:
                all_training_data.append(td)

            iter_samples = None
            if td is not None:
                try:
                    iter_samples = int(len(td))
                except Exception:
                    iter_samples = None

            # collect diagnostics if available
            try:
                if 'best_return' in infos and infos['best_return'] is not None:
                    br = infos['best_return']
                    # convert to scalar mean if tensor/array/list
                    try:
                        br_mean = float(np.mean(br))
                    except Exception:
                        try:
                            br_mean = float(torch.tensor(br).mean())
                        except Exception:
                            br_mean = None
                    if br_mean is not None:
                        iter_best_returns.append(br_mean)
            except Exception:
                pass
            try:
                if 'forward_passes' in infos and infos['forward_passes'] is not None:
                    fp = infos['forward_passes']
                    try:
                        fp_mean = float(np.mean(fp))
                    except Exception:
                        try:
                            fp_mean = float(torch.tensor(fp).mean())
                        except Exception:
                            fp_mean = None
                    if fp_mean is not None:
                        iter_forward_passes.append(fp_mean)
            except Exception:
                pass
            try:
                if 'env_steps' in infos and infos['env_steps'] is not None:
                    es = infos['env_steps']
                    try:
                        es_mean = float(np.mean(es))
                    except Exception:
                        try:
                            es_mean = float(torch.tensor(es).mean())
                        except Exception:
                            es_mean = None
                    if es_mean is not None:
                        iter_env_steps.append(es_mean)
            except Exception:
                pass

            # Per-resample TensorBoard logging (one point per collector.play_game call)
            try:
                def _safe_mean(value):
                    if value is None:
                        return None
                    if isinstance(value, torch.Tensor):
                        return float(value.detach().float().mean().cpu().item())
                    if isinstance(value, (list, tuple)):
                        if len(value) == 0:
                            return None
                        return float(np.mean(value))
                    if isinstance(value, np.ndarray):
                        if value.size == 0:
                            return None
                        return float(value.mean())
                    return float(value)

                iter_best_return = _safe_mean(infos.get('best_return'))
                iter_forward_iters = _safe_mean(infos.get('forward_iters'))
                iter_data_samples = _safe_mean(infos.get('data_samples'))
                iter_game_clocktime = _safe_mean(infos.get('game_clocktime'))
                iter_num_vehicles = _safe_mean(infos.get('num_vehicles'))

                if iter_best_return is not None:
                    tb_writer.add_scalar('tree_based/iter/mean_best_return', iter_best_return, global_resample_step)
                if iter_forward_iters is not None:
                    tb_writer.add_scalar('tree_based/iter/mean_forward_iters', iter_forward_iters, global_resample_step)
                if iter_data_samples is not None:
                    tb_writer.add_scalar('tree_based/iter/mean_data_samples', iter_data_samples, global_resample_step)
                if iter_game_clocktime is not None:
                    tb_writer.add_scalar('tree_based/iter/mean_game_clocktime', iter_game_clocktime, global_resample_step)
                if iter_num_vehicles is not None:
                    tb_writer.add_scalar('tree_based/iter/mean_num_vehicles', iter_num_vehicles, global_resample_step)
                if iter_samples is not None:
                    tb_writer.add_scalar('tree_based/iter/training_samples_collected', float(iter_samples), global_resample_step)
                tb_writer.add_scalar('tree_based/iter/epoch', float(epoch + 1), global_resample_step)
                tb_writer.add_scalar('tree_based/iter/iteration_in_epoch', float(iter_idx + 1), global_resample_step)
            except Exception:
                pass

            global_resample_step += 1

        if not all_training_data:
            logger.warning(f"[tree_based] Epoch {epoch}: no training data collected, skipping update.")
            continue

        training_data = torch.cat(all_training_data)

        # ---- gradient update ----
        _ppo_update(model, training_data, config)

        # ---- logging ----
        mean_best_return = np.mean(iter_best_returns) if len(iter_best_returns) > 0 else None
        mean_forward_passes = np.mean(iter_forward_passes) if len(iter_forward_passes) > 0 else None
        mean_env_steps = np.mean(iter_env_steps) if len(iter_env_steps) > 0 else None

        msg = (
            f"[tree_based] Epoch {epoch + 1}/{total_epochs}: {len(training_data)} training samples"
        )
        if mean_best_return is not None:
            msg += f", mean_best_return={mean_best_return:.4f}"
        if mean_forward_passes is not None:
            msg += f", mean_forward_passes={mean_forward_passes:.1f}"
        if mean_env_steps is not None:
            msg += f", mean_env_steps={mean_env_steps:.1f}"

        logger.info(msg)

        # Epoch summary TensorBoard logging
        try:
            tb_writer.add_scalar('tree_based/epoch/training_samples_total', float(len(training_data)), epoch + 1)
            if mean_best_return is not None:
                tb_writer.add_scalar('tree_based/epoch/mean_best_return', float(mean_best_return), epoch + 1)
            if mean_forward_passes is not None:
                tb_writer.add_scalar('tree_based/epoch/mean_forward_passes', float(mean_forward_passes), epoch + 1)
            if mean_env_steps is not None:
                tb_writer.add_scalar('tree_based/epoch/mean_env_steps', float(mean_env_steps), epoch + 1)
            if hasattr(model, 'optimizer') and model.optimizer is not None:
                lr = float(model.optimizer.param_groups[0]['lr'])
                tb_writer.add_scalar('tree_based/epoch/learning_rate', lr, epoch + 1)
            tb_writer.flush()
        except Exception:
            pass

    # ---- save ----
    save_path = config['train']['save_model_path']
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, save_path)
    print(f"[tree_based] Model saved to {save_path}")
    try:
        tb_writer.close()
    except Exception:
        pass
    return model


# ---------------------------------------------------------------------------
# POMO-TW training
# ---------------------------------------------------------------------------

def _train_pomo_tw(config, env_class, datafile, total_epochs):
    """POMO-TW training loop.

    Implements Policy Optimization with Multiple Optima for Time-Window
    problems (POMO-TW).  Key ideas from the POMO paper (Kwon et al., 2020):

    1. **Coordinate augmentation** – each problem instance is transformed into
       *n_augments* equivalent instances by applying rotations and reflections
       of the 2D node coordinates (all 8 symmetries of the square, or a subset).
       Because augmentations preserve L2 distances, they define genuinely
       different search landscapes while sharing the same optimal tour length,
       which increases training diversity at zero labelling cost.

    2. **Shared REINFORCE baseline** – within each group of augmented instances
       (derived from the same original problem), the mean return is used as the
       REINFORCE baseline.  This reduces variance without requiring a separate
       value network.

    3. **PIP integration** – when ``problem_setup.use_pip_masking: true`` the
       environment enforces proactive infeasibility prevention (blocking pickup
       nodes whose paired delivery would become unreachable), making the search
       more feasibility-aware.

    Parameters
    ----------
    config : dict
        Full training config.  ``train.method`` must be ``'pomo_tw'``.
    env_class : type
        Environment class (VRPTW or PDPTW).
    datafile : str or list[str]
        Path(s) to the dataset PKL file(s).
    total_epochs : int
        Number of training epochs.
    """
    import tempfile
    import pickle
    from .generate_data import ProblemLoader, MultiSizeDataLoader

    pomo_cfg   = config.get('pomo_tw', {})
    n_augments = int(pomo_cfg.get('n_augments', 8))
    assert n_augments in (1, 2, 4, 8), "pomo_tw.n_augments must be 1, 2, 4 or 8"

    env_name   = config['problem_setup']['env'].upper()
    n_parallel = config['train']['n_parallel_problems']
    sampler    = Sampler(config)

    # Load base dataset
    if isinstance(datafile, (list, tuple)):
        base_data = MultiSizeDataLoader.load_mixed_data(config, datafile)
    else:
        base_data = ProblemLoader.load_problem_data(config, datafile)

    n_base = base_data['n_problems']

    # Build augmented config (n_parallel problems × n_augments per problem,
    # processed together in one batch so the shared baseline can be computed).
    aug_config = deepcopy(config)
    aug_config['train']['n_parallel_problems'] = n_parallel * n_augments

    # Build model on a dummy un-augmented environment so the architecture is
    # determined by the original problem size / feature set.
    dummy_env = env_class(config, datafile=datafile, env_type='train')
    obs_space, act_space = dummy_env.spaces
    model = PosAttentionModel(obs_space, act_space, config=config, sampler=sampler)

    # Load pretrained weights if provided
    pretrained = config['train'].get('pretrained_fname')
    if pretrained and os.path.exists(pretrained):
        checkpoint = torch.load(pretrained, weights_only=False)
        params = checkpoint.get('model_state_dict', checkpoint)
        params = {k.replace('._orig_mod', ''): v for k, v in params.items()}
        missing, unexpected = model.load_state_dict(params, strict=False)
        if missing:
            print(f"[pomo_tw] Missing keys from pretrained model: {missing}")
        if unexpected:
            print(f"[pomo_tw] Unexpected keys in pretrained model: {unexpected}")
        print(f"[pomo_tw] Loaded pretrained weights from {pretrained}")

    data_steps = config['muzero']['data_steps_per_epoch']

    tb_dir = config.get('system', {}).get('tensorboard_logdir', 'outputs/tensorboard')
    default_tb_name = (f"earli_pomo_tw_"
                       f"{os.path.splitext(os.path.basename(config['train']['save_model_path']))[0]}")
    tb_name = config.get('system', {}).get('run_name') or default_tb_name
    tb_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tb_run_dir = os.path.join(tb_dir, f"{tb_name}_{tb_stamp}")
    os.makedirs(tb_run_dir, exist_ok=True)
    tb_writer = _get_summary_writer(tb_run_dir)
    print(f"[pomo_tw] TensorBoard run dir: {tb_run_dir}")

    logger = logging.getLogger(__name__)
    logger.info(
        f"[pomo_tw] Training {env_name} for {total_epochs} epochs "
        f"({data_steps} base problems/epoch, {n_augments} augments each, "
        f"{n_parallel} parallel) …"
    )

    device = next(model.parameters()).device

    for epoch in range(total_epochs):
        # ---- data collection ----
        # Sample a batch of base problems and augment them
        dataset_size = n_base
        start_idx = (epoch * data_steps) % dataset_size
        prob_indices = torch.arange(start_idx, start_idx + data_steps) % dataset_size

        # Build augmented dataset for this epoch
        subset = {k: (v[prob_indices] if isinstance(v, torch.Tensor)
                       and v.dim() > 0 and v.shape[0] == dataset_size
                       else (v[prob_indices] if isinstance(v, np.ndarray)
                              and v.ndim > 0 and v.shape[0] == dataset_size
                              else v))
                  for k, v in base_data.items()}
        subset['n_problems'] = len(prob_indices)
        aug_data = _augment_vrptw_dataset(subset, n_augments=n_augments)

        # Write augmented data to a temp file (env uses file-based loading)
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_f:
            pickle.dump(aug_data, tmp_f)
            tmp_path = tmp_f.name

        try:
            aug_config['train']['n_parallel_problems'] = min(
                n_parallel * n_augments, aug_data['n_problems']
            )
            collector = SelfPlay(
                Game=env_class,
                config=aug_config,
                seed=config['train']['seed'] + epoch,
                env_type='train',
                n_beams=config['train']['n_beams'],
                model=model,
                datafile=tmp_path,
                n_problems=aug_config['train']['n_parallel_problems'],
            )

            n_iterations = max(1, (data_steps * n_augments)
                               // aug_config['train']['n_parallel_problems'])
            all_obs: list = []
            all_actions: list = []
            all_returns: list = []
            all_groups: list = []
            aug_best_returns: list = []

            for iter_idx in range(n_iterations):
                _, infos = collector.play_game(deterministic=False, training=True)
                td = infos.get('training_data')
                if td is None or len(td) == 0:
                    continue

                # Per-augmentation group assignment: group by original problem index
                # (The augmented env runs aug_config['n_parallel_problems'] problems,
                #  cycling through the n_augments versions of each original problem.)
                n_batch_probs = aug_config['train']['n_parallel_problems']
                n_orig = n_batch_probs // n_augments
                group_ids = torch.arange(n_orig).repeat(n_augments)[:n_batch_probs]
                # Expand group IDs to match the number of training samples
                samples_per_prob = max(1, len(td) // n_batch_probs)
                group_expanded = group_ids.repeat_interleave(samples_per_prob)
                if len(group_expanded) < len(td):
                    pad = torch.zeros(len(td) - len(group_expanded), dtype=torch.long)
                    group_expanded = torch.cat([group_expanded, pad])
                group_expanded = group_expanded[:len(td)]

                all_obs.append(td['observations'])
                all_actions.append(td['actions'])
                all_returns.append(td['returns'])
                all_groups.append(group_expanded)

                # Best return for logging
                try:
                    br = infos.get('best_return')
                    if br is not None:
                        aug_best_returns.append(float(np.mean(br)))
                except Exception:
                    pass

        finally:
            os.unlink(tmp_path)

        if not all_obs:
            logger.warning(f"[pomo_tw] Epoch {epoch}: no training data, skipping update.")
            continue

        obs_cat     = torch.cat(all_obs)
        actions_cat = torch.cat(all_actions)
        returns_cat = torch.cat(all_returns).float()
        groups_cat  = torch.cat(all_groups)

        # ---- POMO shared-baseline advantage ----
        # For each group (original problem), compute mean return as baseline.
        n_groups = int(groups_cat.max().item()) + 1
        group_mean = torch.zeros(n_groups, dtype=torch.float32)
        group_count = torch.zeros(n_groups, dtype=torch.float32)
        for g in range(n_groups):
            mask_g = groups_cat == g
            if mask_g.any():
                group_mean[g] = returns_cat[mask_g].mean()
                group_count[g] = mask_g.float().sum()

        baseline  = group_mean[groups_cat]
        advantage = returns_cat - baseline
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # ---- REINFORCE gradient update ----
        # POMO uses a pure policy-gradient update — no critic/value network.
        # The shared baseline (mean return per augmentation group) already
        # provides variance reduction; no PPO clipping is needed.
        n_samples  = len(obs_cat)
        batch_size = min(config['train']['batch_size'], n_samples)
        model.train()
        indices = torch.randperm(n_samples)
        for start in range(0, n_samples, batch_size):
            bidx  = indices[start: start + batch_size]
            obs_b = obs_cat[bidx].to(device)
            act_b = actions_cat[bidx].to(device)
            adv_b = advantage[bidx].to(device)

            # evaluate_actions returns (values, log_prob, entropy); values are
            # not needed because POMO uses a group-mean baseline, not a critic.
            _, new_log_prob, _ = model.evaluate_actions(obs_b, act_b)
            loss = -(new_log_prob.float() * adv_b).mean()

            model.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            model.optimizer.step()

        model.eval()

        # ---- logging ----
        mean_best = float(np.mean(aug_best_returns)) if aug_best_returns else None
        msg = f"[pomo_tw] Epoch {epoch + 1}/{total_epochs}: {n_samples} training samples"
        if mean_best is not None:
            msg += f", mean_best_return={mean_best:.4f}"
        logger.info(msg)

        try:
            tb_writer.add_scalar('pomo_tw/epoch/training_samples', float(n_samples), epoch + 1)
            if mean_best is not None:
                tb_writer.add_scalar('pomo_tw/epoch/mean_best_return', float(mean_best), epoch + 1)
            if hasattr(model, 'optimizer') and model.optimizer is not None:
                lr = float(model.optimizer.param_groups[0]['lr'])
                tb_writer.add_scalar('pomo_tw/epoch/learning_rate', lr, epoch + 1)
            tb_writer.flush()
        except Exception:
            pass

    # ---- save ----
    save_path = config['train']['save_model_path']
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, save_path)
    print(f"[pomo_tw] Model saved to {save_path}")
    try:
        tb_writer.close()
    except Exception:
        pass
    return model


# ---------------------------------------------------------------------------
# PPO training (SB3)
# ---------------------------------------------------------------------------


def _train_ppo(config, env_class, datafile, total_steps):
    """Training loop for the ``ppo`` method using Stable-Baselines3.

    This replicates the CVRP training path from *ExampleTrain.ipynb* for
    VRPTW and PDPTW environments.

    Parameters
    ----------
    config : dict
        Verified training config (``compatibility_mode`` must be
        ``'stable_baselines'``).
    env_class : type
        Environment class (VRP, VRPTW, or PDPTW).
    datafile : str or list[str]
        Path(s) to the dataset PKL file(s).
    total_steps : int
        Total environment steps for SB3 PPO.
    """
    env_name = config['problem_setup']['env'].upper()
    env      = env_class(config, datafile=datafile, env_type='train')
    # Wrap environment so SB3 can track episode rewards/lengths. If the
    # environment is already a VecEnv (RoutingBase), use VecMonitor; otherwise
    # use Monitor for single-env wrappers.
    if isinstance(env, VecEnv):
        env = VecMonitor(env)
    else:
        env = Monitor(env)
    # Keep rollout horizon independent from total_steps so changing total_steps
    # increases the number of PPO rollouts/iterations instead of only enlarging
    # a single rollout. Priority:
    #   1) train.n_steps (if explicitly provided in config)
    #   2) muzero.data_steps_per_epoch / n_parallel_problems
    #   3) fallback to 1
    n_parallel = max(1, int(config['train']['n_parallel_problems']))
    cfg_n_steps = config['train'].get('n_steps')
    if cfg_n_steps is not None:
        n_steps = max(1, int(cfg_n_steps))
    else:
        per_epoch_steps = int(config['muzero']['data_steps_per_epoch'])
        n_steps = max(1, per_epoch_steps // n_parallel)

    rollout_size = n_steps * n_parallel

    # Optional periodic validation on eval.val_data_file.
    val_datafile = config['eval'].get('val_data_file')
    eval_callback = None
    best_policy_path = None
    if val_datafile:
        if not os.path.exists(val_datafile):
            raise FileNotFoundError(
                f"Validation dataset file not found: {val_datafile}"
            )
        # Validation uses a dedicated config so we can random-sample validation
        # problems without affecting inference/test configs.
        eval_config = deepcopy(config)
        eval_config.setdefault('eval', {})['sampling_mode'] = (
            config.get('eval', {}).get('sampling_mode', 'random_with_replacement')
        )
        n_eval_episodes = int(config['eval'].get('n_eval_episodes', 50))
        n_eval_episodes = max(1, n_eval_episodes)
        eval_env = env_class(eval_config, datafile=val_datafile, env_type='eval')
        if isinstance(eval_env, VecEnv):
            eval_env = VecMonitor(eval_env)
        else:
            eval_env = Monitor(eval_env)

        class SaveBestPolicyStateDictCallback(BaseCallback):
            def __init__(self, save_path: str, verbose=0):
                super().__init__(verbose)
                self.save_path = save_path

            def _on_step(self) -> bool:
                self._save_best()
                return True

            def _on_training_start(self) -> None:
                os.makedirs(os.path.dirname(os.path.abspath(self.save_path)), exist_ok=True)

            def _on_rollout_end(self) -> None:
                return None

            def _save_best(self) -> None:
                torch.save({'model_state_dict': self.model.policy.state_dict()}, self.save_path)
                if self.verbose:
                    print(f"[ppo] Saved new best validation checkpoint to {self.save_path}")

        save_path = config['train']['save_model_path']
        best_policy_path = f"{save_path}.best"
        best_callback = SaveBestPolicyStateDictCallback(best_policy_path, verbose=1)
        eval_callback = EvalCallback(
            eval_env,
            eval_freq=max(1, n_steps),
            n_eval_episodes=n_eval_episodes,
            deterministic=bool(config['eval'].get('deterministic_test_beam', True)),
            best_model_save_path=None,
            callback_on_new_best=best_callback,
            verbose=1,
        )

    # Prepare TensorBoard logging directory/name
    tb_dir = config.get('system', {}).get('tensorboard_logdir', 'outputs/tensorboard')
    tb_name = config.get('system', {}).get('run_name') or 'earli_ppo'
    os.makedirs(tb_dir, exist_ok=True)

    sb3_model = PPO(
        policy=PosAttentionModel,
        env=env,
        policy_kwargs={'config': config},
        n_steps=n_steps,
        batch_size=config['train']['batch_size'],
        n_epochs=int(config['train'].get('epochs', 10)),
        gamma=float(config['train'].get('gamma', 0.99)),
        gae_lambda=float(config['train'].get('gae_lambda', 0.95)),
        clip_range=float(config['train'].get('clip_range', 0.2)),
        vf_coef=float(config['train'].get('vf_coef', 0.5)),
        max_grad_norm=float(config['train'].get('max_grad_norm', 0.5)),
        target_kl=(
            float(config['train']['target_kl'])
            if config['train'].get('target_kl', None) is not None
            else None
        ),
        learning_rate=(
            # build a schedule if configured, otherwise accept float or callable
            make_lr_schedule(
                initial_lr=config['train'].get('learning_rate', 1e-4),
                schedule_type=config['train'].get('lr_schedule', None) or 'constant',
                min_lr=config['train'].get('min_learning_rate', 0.0),
                exp_decay=config['train'].get('lr_exp_decay', 5.0),
                step_ratio=config['train'].get('lr_step_ratio', 0.5),
                step_fraction=config['train'].get('lr_step_fraction', 0.5),
            ) if not callable(config['train'].get('learning_rate')) and config['train'].get('lr_schedule', None) is not None
            else (
                config['train']['learning_rate']
                if callable(config['train']['learning_rate'])
                else float(config['train']['learning_rate'])
            )
        ),
        ent_coef=float(config['train'].get('ent_coef', 0.0)),
        verbose=1,
        tensorboard_log=tb_dir,
    )

    # Load pretrained weights into the SB3 policy if specified and available
    pretrained_fname = config['train'].get('pretrained_fname')
    if pretrained_fname:
        if os.path.exists(pretrained_fname):
            checkpoint = torch.load(pretrained_fname, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model_state = sb3_model.policy.state_dict()
            compatible_state = {}
            skipped = []
            for key, value in state_dict.items():
                if key not in model_state:
                    skipped.append(key)
                    continue
                try:
                    if model_state[key].shape != value.shape:
                        skipped.append(key)
                        continue
                except Exception:
                    skipped.append(key)
                    continue
                compatible_state[key] = value
            missing, unexpected = sb3_model.policy.load_state_dict(compatible_state, strict=False)
            if skipped or missing or unexpected:
                print("[ppo] Loaded pretrained weights with skipped/missing keys.")
                if skipped:
                    print(f"[ppo] Skipped keys (shape or missing): {skipped}")
                if missing:
                    print(f"[ppo] Missing keys: {missing}")
                if unexpected:
                    print(f"[ppo] Unexpected keys: {unexpected}")
            else:
                print(f"[ppo] Successfully loaded pretrained weights from {pretrained_fname}")
        else:
            print(f"[ppo] Pretrained file not found: {pretrained_fname}")

    print(
        f"[ppo] Training {env_name} for {total_steps} steps "
        f"({config['train']['epochs']} epochs × "
        f"{config['muzero']['data_steps_per_epoch']} data steps/epoch, "
        f"n_steps={n_steps}, n_parallel={n_parallel}, rollout_size={rollout_size}) …"
    )
    if eval_callback is not None:
        print(
            f"[ppo] Validation enabled: val_data_file={val_datafile}, "
            f"eval_freq={max(1, n_steps)} (matches rollout collection frequency), "
            f"n_eval_episodes={n_eval_episodes}, "
            f"sampling_mode={config.get('eval', {}).get('sampling_mode', 'random_with_replacement')}"
        )
    # Create a callback that records per-rollout scalars to TensorBoard.
    class PerRolloutTensorboardCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self._constraint_sums = {}
            self._constraint_counts = {}

        def _on_step(self) -> bool:
            infos = self.locals.get('infos', None)
            if infos:
                tracked = [
                    'hard_constraint_override_ratio',
                    'late_sum',
                    'late_count',
                    'masked_ratio',
                    'depot_with_customer',
                    'vehicle_over_lb',
                    'constraint_penalty',
                ]
                for info in infos:
                    if not isinstance(info, dict):
                        continue
                    for key in tracked:
                        if key not in info:
                            continue
                        val = info[key]
                        try:
                            if hasattr(val, 'detach'):
                                import torch as _torch
                                v = float(_torch.as_tensor(val).float().mean().item())
                            else:
                                v = float(val)
                        except Exception:
                            continue
                        self._constraint_sums[key] = self._constraint_sums.get(key, 0.0) + v
                        self._constraint_counts[key] = self._constraint_counts.get(key, 0) + 1
            return True

        def _on_rollout_end(self) -> None:
            # Try to copy a selection of train/rollout/eval scalars into a
            # custom namespace so they appear reliably in TensorBoard.
            try:
                name_to_value = getattr(self.logger, 'name_to_value', {})
            except Exception:
                name_to_value = {}

            tags = [
                'train/entropy_loss',
                'train/explained_variance',
                'train/loss',
                'train/learning_rate',
                'rollout/ep_rew_mean',
                'rollout/ep_len_mean',
                'eval/mean_reward',
                'time/fps',
            ]
            for tag in tags:
                val = name_to_value.get(tag)
                if val is not None:
                    # map slashes to underscores under custom/ namespace
                    safe_tag = tag.replace('/', '_')
                    self.logger.record(f'custom/{safe_tag}', float(val))

            for key, total in self._constraint_sums.items():
                cnt = self._constraint_counts.get(key, 0)
                if cnt > 0:
                    self.logger.record(f'custom/constraint_{key}', float(total / cnt))
            self._constraint_sums.clear()
            self._constraint_counts.clear()

            # Additionally, if the model has an episode info buffer, compute
            # and log mean episode reward/length from it (more reliable).
            try:
                ep_buf = getattr(self.model, 'ep_info_buffer', None)
                if ep_buf is not None and len(ep_buf) > 0:
                    import numpy as _np
                    rewards = [e.get('r') for e in ep_buf if 'r' in e]
                    lengths = [e.get('l') for e in ep_buf if 'l' in e]
                    if rewards:
                        self.logger.record('custom/ep_info_mean_reward', float(_np.mean(rewards)))
                    if lengths:
                        self.logger.record('custom/ep_info_mean_length', float(_np.mean(lengths)))
            except Exception:
                pass
            # Flush to ensure values are written out
            try:
                self.logger.dump(self.num_timesteps)
            except Exception:
                pass

    step_cb = PerRolloutTensorboardCallback()
    cb = CallbackList([c for c in (eval_callback, step_cb) if c is not None])

    # Pass tb_log_name so TensorBoard groups logs under a readable run name.
    sb3_model.learn(total_steps, log_interval=1, callback=cb, tb_log_name=tb_name)

    save_path = config['train']['save_model_path']
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    last_policy_path = f"{save_path}.last"
    torch.save({'model_state_dict': sb3_model.policy.state_dict()}, last_policy_path)
    print(f"[ppo] Saved last-step checkpoint to {last_policy_path}")

    if best_policy_path is not None and os.path.exists(best_policy_path):
        shutil.copy2(best_policy_path, save_path)
        print(f"[ppo] Validation-best checkpoint promoted to {save_path}")
    else:
        shutil.copy2(last_policy_path, save_path)
        print(f"[ppo] No validation-best checkpoint found. Using last-step checkpoint at {save_path}")
    return sb3_model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train(config_path: str, total_steps: int | None = None) -> None:
    """Dispatch to either the PPO or tree_based training loop.

    The training method is read from ``config['train']['method']`` (either
    ``'ppo'`` or ``'tree_based'``).  Unlike the previous implementation,
    the method setting is **no longer overridden** – it is respected as
    configured.

    Args:
        config_path: Path to a YAML config file (e.g.
            ``config_vrptw_train.yaml``).
        total_steps: Total environment / training steps.  When *None* the
            value from the config
            (``train.epochs * muzero.data_steps_per_epoch``) is used.
    """
    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)

    # NOTE: do NOT force config['train']['method'] = 'ppo' here.
    # verify_consistent_config will apply method-specific adjustments
    # (e.g. setting compatibility_mode=None for tree_based).
    config = verify_consistent_config(config)

    env_name = config['problem_setup']['env'].lower()
    if env_name not in ENV_CLASSES:
        raise ValueError(
            f"Unknown env '{env_name}'. Choose from: {list(ENV_CLASSES.keys())}"
        )
    env_class = ENV_CLASSES[env_name]
    datafile  = _resolve_datafile(config, env_name)
    method    = config['train']['method'].lower()

    if method == 'ppo':
        if total_steps is None:
            total_steps = (config['train']['epochs']
                           * config['muzero']['data_steps_per_epoch'])
            total_steps = max(total_steps, config['train']['batch_size'])
        _train_ppo(config, env_class, datafile, total_steps)

    elif method == 'tree_based':
        total_epochs = config['train']['epochs']
        if total_steps is not None:
            # Convert an explicit step count to an approximate epoch count
            data_steps = config['muzero']['data_steps_per_epoch']
            n_parallel = config['train']['n_parallel_problems']
            # steps_per_epoch ≈ data_steps (each epoch collects data_steps problems)
            total_epochs = max(1, total_steps // max(data_steps, n_parallel))
        _train_tree_based(config, env_class, datafile, total_epochs)

    elif method == 'pomo_tw':
        total_epochs = config['train']['epochs']
        if total_steps is not None:
            data_steps = config['muzero']['data_steps_per_epoch']
            n_parallel = config['train']['n_parallel_problems']
            total_epochs = max(1, total_steps // max(data_steps, n_parallel))
        _train_pomo_tw(config, env_class, datafile, total_epochs)

    else:
        raise ValueError(
            f"Unknown train.method '{method}'. Choose 'ppo', 'tree_based', or 'pomo_tw'."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Train a VRP/VRPTW/PDPTW RL agent.\n\n'
            'Set train.method in the YAML config to select the training algorithm:\n'
            '  ppo         – Stable-Baselines3 PPO (fast, requires compatibility_mode: stable_baselines)\n'
            '  tree_based  – Tree-search guided PPO (higher data quality, requires compatibility_mode: null)\n'
            '  pomo_tw     – POMO-TW: augmented multi-start REINFORCE for VRPTW/PDPTW (requires compatibility_mode: null)'
        )
    )
    parser.add_argument(
        '--config', default='config_vrptw_train.yaml',
        help='Path to the YAML config file (default: config_vrptw_train.yaml)',
    )
    parser.add_argument(
        '--total-steps', type=int, default=None,
        help=(
            'Total PPO environment steps / training steps '
            '(overrides epochs × data_steps_per_epoch)'
        ),
    )
    args = parser.parse_args()
    train(config_path=args.config, total_steps=args.total_steps)


if __name__ == '__main__':
    main()
