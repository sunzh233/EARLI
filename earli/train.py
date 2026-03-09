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

    logger = logging.getLogger(__name__)
    logger.info(
        f"[tree_based] Training {env_name} for {total_epochs} epochs "
        f"({data_steps} problems/epoch, {n_parallel} parallel) …"
    )

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
        for _ in range(n_iterations):
            print(f"[tree_based] Epoch {epoch + 1}/{total_epochs}, iteration {_ + 1}/{n_iterations} …")
            _, infos = collector.play_game(deterministic=False, training=True)
            td = infos.get('training_data')
            if td is not None and len(td) > 0:
                all_training_data.append(td)

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

    # ---- save ----
    save_path = config['train']['save_model_path']
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, save_path)
    print(f"[tree_based] Model saved to {save_path}")
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
    if val_datafile:
        if not os.path.exists(val_datafile):
            raise FileNotFoundError(
                f"Validation dataset file not found: {val_datafile}"
            )
        # Use eval mode so validation remains stable and does not inherit
        # training-time random sampling behavior.
        n_eval_episodes = int(config['eval'].get('n_eval_episodes', 50))
        n_eval_episodes = max(1, n_eval_episodes)
        eval_env = env_class(config, datafile=val_datafile, env_type='eval')
        if isinstance(eval_env, VecEnv):
            eval_env = VecMonitor(eval_env)
        else:
            eval_env = Monitor(eval_env)
        eval_callback = EvalCallback(
            eval_env,
            eval_freq=max(1, n_steps),
            n_eval_episodes=n_eval_episodes,
            deterministic=bool(config['eval'].get('deterministic_test_beam', True)),
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
            f"n_eval_episodes={n_eval_episodes}"
        )
    # Create a callback that records per-rollout scalars to TensorBoard.
    class PerRolloutTensorboardCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)

        def _on_step(self) -> bool:
            # Required abstract implementation; do nothing per step.
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
    torch.save({'model_state_dict': sb3_model.policy.state_dict()}, save_path)
    print(f"[ppo] Model saved to {save_path}")
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

    else:
        raise ValueError(
            f"Unknown train.method '{method}'. Choose 'ppo' or 'tree_based'."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Train a VRP/VRPTW/PDPTW RL agent.\n\n'
            'Set train.method in the YAML config to select the training algorithm:\n'
            '  ppo         – Stable-Baselines3 PPO (fast, requires compatibility_mode: stable_baselines)\n'
            '  tree_based  – Tree-search guided PPO (higher data quality, requires compatibility_mode: null)'
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
