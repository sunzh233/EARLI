# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

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

def main(config_path='config.yaml'):
    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)
    
    print(f"Loaded config from {config_path}:")
    config = verify_consistent_config(config)
    logger = UnifiedLogger(config)
    # wandb_mode = 'online' if config['system']['allow_wandb'] else 'disabled'
    wandb_mode = 'disabled'
    wandb.init(mode=wandb_mode)
    env_class = VRPTW if config['problem_setup']['env'] == 'vrptw' else VRP
    evaluator = Evaluator(env_class, config, logger)
    print(f'Applying RL solver to {evaluator.n_problems["test"]} problem instances...')
    evaluator.inference()

if __name__ == '__main__':
    main()
