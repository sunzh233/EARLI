# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import argparse
import yaml
from matplotlib import pyplot as plt
from yaml import SafeLoader

from .utils.general_usage_utils import ignore_legacy_wandb_warnings
ignore_legacy_wandb_warnings()
import wandb
from .evaluator import Evaluator
from .unified_logger import UnifiedLogger
from .utils.nv import verify_consistent_config
from .vrp import VRP
from .vrptw import VRPTW
from .pdptw import PDPTW


ENV_CLASSES = {
    'vrp'  : VRP,
    'vrptw': VRPTW,
    'pdptw': PDPTW,
}


def main(config_path=None):
    if config_path is None:
        parser = argparse.ArgumentParser(
            description='Run RL inference for VRP/VRPTW/PDPTW.'
        )
        parser.add_argument(
            '--config', default='config.yaml',
            help='Path to the YAML config file (default: config.yaml)',
        )
        args = parser.parse_args()
        config_path = args.config
    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)
    config = verify_consistent_config(config)
    logger = UnifiedLogger(config)
    wandb_mode = 'disabled'
    wandb.init(mode=wandb_mode)
    env_name = config['problem_setup']['env'].lower()
    if env_name not in ENV_CLASSES:
        raise ValueError(f"Unknown env type '{env_name}'. Choose from: {list(ENV_CLASSES.keys())}")
    env_class = ENV_CLASSES[env_name]
    evaluator = Evaluator(env_class, config, logger)
    print(f'Applying RL solver to {evaluator.n_problems["test"]} problem instances...')
    evaluator.inference()

if __name__ == '__main__':
    main()
