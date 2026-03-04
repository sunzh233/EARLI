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

The script uses Stable-Baselines3 PPO with the custom attention policy.
Set ``train.method: ppo`` and ``system.compatibility_mode: stable_baselines``
in the config to use the SB3 training path.
"""

import argparse
import os

import torch
import yaml
from stable_baselines3 import PPO
from yaml import SafeLoader

from .models.attention_model import PosAttentionModel
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


def train(config_path: str, total_steps: int | None = None) -> None:
    """Run PPO training for a VRP/VRPTW/PDPTW environment.

    Args:
        config_path: Path to a YAML config file (e.g. ``config_vrptw_train.yaml``).
        total_steps: Total environment steps for training.  When *None* the value
            from the config (``train.epochs * muzero.data_steps_per_epoch``) is
            used as a rough proxy; set explicitly to override.
    """
    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)

    # Force PPO-compatible settings
    config['train']['method'] = 'ppo'
    config = verify_consistent_config(config)

    env_name = config['problem_setup']['env'].lower()
    if env_name not in ENV_CLASSES:
        raise ValueError(
            f"Unknown env '{env_name}'. Choose from: {list(ENV_CLASSES.keys())}"
        )
    env_class = ENV_CLASSES[env_name]

    data_file = config['eval']['data_file']
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"Dataset file not found: {data_file}\n"
            f"Generate it first with:\n"
            f"  python -m earli.benchmark_parser {env_name} <benchmark_dir> {data_file}"
        )

    env = env_class(config, datafile=data_file, env_type='train')

    if total_steps is None:
        total_steps = config['train']['epochs'] * config['muzero']['data_steps_per_epoch']
        # Ensure a sensible minimum
        total_steps = max(total_steps, config['train']['batch_size'])

    n_steps = max(1, total_steps // config['train']['n_parallel_problems'])

    model = PPO(
        policy=PosAttentionModel,
        env=env,
        policy_kwargs={'config': config},
        n_steps=n_steps,
        batch_size=config['train']['batch_size'],
        learning_rate=config['train']['learning_rate'],
        ent_coef=0,
        verbose=1,
    )

    print(
        f"Training {env_name.upper()} for {total_steps} steps "
        f"({config['train']['epochs']} epochs × "
        f"{config['muzero']['data_steps_per_epoch']} data steps/epoch) …"
    )
    model.learn(total_steps, log_interval=1)

    save_path = config['train']['save_model_path']
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save({'model_state_dict': model.policy.state_dict()}, save_path)
    print(f"Model saved to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train a VRPTW or PDPTW RL agent via PPO'
    )
    parser.add_argument(
        '--config', default='config_vrptw_train.yaml',
        help='Path to the YAML config file (default: config_vrptw_train.yaml)',
    )
    parser.add_argument(
        '--total-steps', type=int, default=None,
        help='Total PPO environment steps (overrides epochs × data_steps_per_epoch)',
    )
    args = parser.parse_args()
    train(config_path=args.config, total_steps=args.total_steps)


if __name__ == '__main__':
    main()
