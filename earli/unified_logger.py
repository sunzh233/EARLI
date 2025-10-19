# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import logging
import os
import uuid
from copy import copy

import torch
from colorama import Fore, Style
from stable_baselines3.common.logger import make_output_format, Logger

import wandb


class ColorfulFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.WARNING:
            prefix = Fore.YELLOW
        elif record.levelno == logging.ERROR:
            prefix = Fore.RED
        # elif record.levelno == logging.INFO:
        #     prefix = Fore.GREEN
        elif record.levelno == logging.CRITICAL:
            prefix = Fore.RED + Style.BRIGHT
        else:
            prefix = Style.RESET_ALL
        msg = super().format(record)
        return prefix + msg + Style.RESET_ALL


class UnifiedLogger(object):
    def __init__(self, config, format_strings=["stdout"], log_to_file=False):
        """
            Configure the current logger.

            :param folder: the save location
                (if None, $SB3_LOGDIR, if still None, tempdir/SB3-[date & time])
            :param format_strings: the output logging format
                (if None, $SB3_LOG_FORMAT, if still None, ['stdout', 'log', 'csv'])
            :return: The logger object.
            """
        self.histograms = {}
        self.config = config
        folder = 'logs'
        if log_to_file:
            os.makedirs(folder, exist_ok=True)
            id = uuid.uuid4()
            self.log_file = os.path.join(folder, f'run_{id}.log')
        else:
            self.log_file = None
        self.setup_stdout_format(log_file=self.log_file)

        log_suffix = ""
        format_strings = list(filter(None, format_strings))
        output_formats = [make_output_format(f, folder, log_suffix) for f in format_strings]
        self.stdout_logger = Logger(folder=folder, output_formats=output_formats)

    def setup_wandb_logger(self):
        self.wandb_logger = wandb.run

    def setup_stdout_format(self, log_file=None):
        config = self.config
        # logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12s] %(message)s", datefmt='%d-%m')
        logFormatter = ColorfulFormatter("%(asctime)s [%(threadName)-12s] %(message)s", datefmt='%d-%m')
        rootLogger = logging.getLogger()

        if log_file is not None:
            fileHandler = logging.FileHandler(log_file)
            fileHandler.setFormatter(logFormatter)
            rootLogger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)
        rootLogger.setLevel(config['logger']['logging_level'])
        return rootLogger

    def log(self, *args, **kwargs):
        self.wandb_logger.log(*args, **kwargs)

    def record(self, stage, key, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.stdout_logger.record(f"{stage}/{key}", value)

    def log_histogram(self, key, data):
        self.histograms[key] = wandb.Histogram(data)

    def dump(self, step, fabric=None, force=False):
        data = copy(self.stdout_logger.name_to_value)
        data.update(self.histograms)
        self.histograms = {}
        if hasattr(self, 'wandb_logger'):
            self.wandb_logger.log(data, step)
        self.stdout_logger.dump(step)

    def close(self):
        self.stdout_logger.removeHandler()
        self.stdout_logger.file_handler.close()
