# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
The MIT License
Copyright (c) 2025 NVIDIA Inc.
Copyright (c) 2019 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import logging

import torch


class RolloutBuffer(object):
    """
    Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the RolloutBuffer to use dictionary observations

    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to Monte-Carlo advantage estimate when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(self, config, data=None, gae_lambda=None, gamma: float = 1):
        self.start_idx = None
        self.batch_indices = None
        self.config = config
        self.allow_overflow = self.config['buffer']['allow_overflow']
        if config['buffer']['buffer_precision'] == 'float32':
            self.float_type = torch.float32
        elif config['buffer']['buffer_precision'] == 'float64':
            self.float_type = torch.float64
        self.gae_lambda = 1 if gae_lambda is None else gae_lambda
        self.n_samples_since_reset = 0
        self.gamma = gamma
        self.data = data
        self.max_buffer_size = self.config['buffer']['max_buffer_size']
        self.macro_state_buffer = False
        self.update_log_probs = False
        self.warn = True
        self.updated_ind = []


    def reset(self) -> None:
        self.n_samples_since_reset = 0
        self.data = None

    def append(self, external_buffer, progress=None):
        if self.data is None:
            if isinstance(external_buffer, list):
                self.data = torch.cat(external_buffer.data)
            else:
                self.data = external_buffer.data
            self.n_samples_since_reset = len(self.data)
        else:
            if isinstance(external_buffer, list):
                self.data = torch.cat([self.data] + external_buffer.data)
                self.n_samples_since_reset += sum([len(x) for x in external_buffer.data])
            else:
                self.data = torch.cat([self.data, external_buffer.data])
                self.n_samples_since_reset += len(self.data)
        if len(self.data) >= self.max_buffer_size:
            if self.allow_overflow:
                if self.warn:
                    progress_str = '' if progress is None else f' after {100 * progress:.1f}% of the epoch'
                    logging.info(f'Replay buffer is full{progress_str}. Starting overwriting cyclically.')
                    self.warn = False
                crop_ind = len(self.data) - self.max_buffer_size
                self.data = self.data[crop_ind:]
            else:
                raise OverflowError("Buffer overflows")

    def __len__(self):
        return len(self.data)

    def get_usage(self):
        return self.n_samples_since_reset, self.max_buffer_size, self.n_samples_since_reset / self.max_buffer_size


