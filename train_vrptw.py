import pickle as pkl
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import yaml
from stable_baselines3 import PPO

from earli.models.attention_model import PosAttentionModel
from earli.utils.nv import verify_consistent_config
from earli.vrp import VRP
from earli.vrptw import VRPTW
from earli.utils import analysis_utils as utils
from earli import download_data, main, test_injection
import torch

total_train_steps = 1000000

with open('config_train.yaml') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
config = verify_consistent_config(config) # optional

print('Creating environment...')

env = VRPTW(config, datafile=config['eval']['data_file'], env_type='eval')

print('Creating model and starting training...')

model = PPO(policy=PosAttentionModel, env=env, policy_kwargs={'config': config},
            n_steps=200,
            batch_size=config['train']['batch_size'],
            ent_coef=0)

print('Training model...')

model.learn(total_train_steps, log_interval=1)

model_path = config['train']['save_model_path']
print('Saving model to:', model_path)
torch.save({'model_state_dict': model.policy.state_dict()}, model_path)

main.main(config_path='config_test.yaml')