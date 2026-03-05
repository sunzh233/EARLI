# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import logging
import os
import pickle
import random
import time

import numpy as np
import torch
from colorama import Fore


def seed_all(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def verify_consistent_config(config, warn=True):
    def set_field(k, v):
        if warn and v != config[k[0]][k[1]]:
            logging.info(Fore.GREEN + f'Replacing {k[1]} from {config[k[0]][k[1]]} to {v}.')
        config[k[0]][k[1]] = v

    def assert_consistent_property(k1, v1, k2, v2):
        if not isinstance(v1, (tuple, list)): v1 = [v1]
        if not isinstance(v2, (tuple, list)): v2 = [v2]
        if config[k1[0]][k1[1]] in v1:
            if not config[k2[0]][k2[1]] in v2:
                set_field(k2, v2[0])

    # PPO
    assert_consistent_property(
            ('muzero', 'expansion_method'), 'POLICY_LEAD_SEARCH',
            ('train', 'n_beams'), 1)
    assert_consistent_property(
            ('train', 'n_beams'), (0, 1),
            ('sampler', 'diversity_penalty'), 0)
    assert_consistent_property(
            ('muzero', 'expansion_method'), 'POLICY_LEAD_SEARCH',
            ('muzero', 'deterministic_branch_in_k_beams'), False)
    assert_consistent_property(
            ('muzero', 'expansion_method'), 'POLICY_LEAD_SEARCH',
            ('sampler', 'complement_k_beams_calc'), False)

    # torch.compile
    assert_consistent_property(
            ('speedups', 'compile_mode'), ('default', 'reduce-overhead', 'max-autotune'),
            ('speedups', 'amp'), False)

    # SB3 / tree_based assertions should be last!
    if config['train']['method'].lower() == 'ppo':
        config['train']['method'] = 'ppo'
    assert_consistent_property(
            ('train', 'method'), 'ppo',
            ('buffer', 'buffer_precision'), 'float32')
    assert_consistent_property(
            ('train', 'method'), 'ppo',
            ('train', 'n_beams'), 0)
    assert_consistent_property(
            ('train', 'method'), 'ppo',
            ('system', 'use_tensordict'), 0)
    assert_consistent_property(
            ('train', 'method'), 'ppo',
            ('muzero', 'expansion_method'), 'POLICY_LEAD_SEARCH')  # to avoid a k-beams env

    # tree_based training requires tensor-mode (not stable_baselines compatibility)
    if config['train']['method'].lower() == 'tree_based':
        if config['system'].get('compatibility_mode') == 'stable_baselines':
            set_field(('system', 'compatibility_mode'), None)
        assert_consistent_property(
                ('train', 'method'), 'tree_based',
                ('system', 'use_tensordict'), True)

    return config


def printable_time(t0=None, t1=None, dt=None, n_repeats=1, in_seconds=False):
    if dt is None:
        if t1 is None:
            t1 = time.time()
        dt = (t1 - t0) / n_repeats
    if in_seconds:
        return f'{dt:.2f} [s]'
    else:
        hh = f'{int(dt // 3600):01d}'
        mm = f'{int(dt // 60) % 60:02d}'
        ss = f'{int(dt % 60):02d}'
        return f'{hh}:{mm}:{ss}'


def find_largest_common_prefixes(strings):
    def longest_common_prefix(str1, str2):
        min_length = min(len(str1), len(str2))
        for i in range(min_length):
            if str1[i] != str2[i]:
                return str1[:i]
        return str1[:min_length]

    if len(strings) <= 1:
        return strings

    prefixes = {}
    for i in range(len(strings) - 1):
        prefix = longest_common_prefix(strings[i], strings[i + 1])
        if prefix:
            prefixes[prefix] = len(prefix)

    return list(prefixes.keys())
