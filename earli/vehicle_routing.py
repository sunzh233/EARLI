# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import logging
from collections import namedtuple
from copy import deepcopy
from typing import List, Any

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium import Env as GymEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices
from stable_baselines3.common.vec_env.util import obs_space_info
from tensordict import TensorDict

from .generate_data import ProblemLoader
from .utils.nv import seed_all

SIMPLE_MLP = 0
GRAPH = 1
EDGE_FREQUENCY = namedtuple('EDGE_FREQUENCY', ['edge_index', 'edge_counter'])


def render(static, tour_indices, save_path):
    """Plots the found solution."""

    plt.close('all')

    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        idx = idx.expand(static.size(1), -1)
        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        start = static[i, :, 0].cpu().data.numpy()
        x = np.hstack((start[0], data[0], start[0]))
        y = np.hstack((start[1], data[1], start[1]))

        # Assign each subtour a different colour & label in order traveled
        idx = np.hstack((0, tour_indices[i].cpu().numpy().flatten(), 0))
        where = np.where(idx == 0)[0]

        for j in range(len(where) - 1):

            low = where[j]
            high = where[j + 1]

            if low + 1 == high:
                continue

            ax.plot(x[low: high + 1], y[low: high + 1], zorder=1, label=j)

        ax.legend(loc="upper right", fontsize=3, framealpha=0.5)
        ax.scatter(x, y, s=4, c='r', zorder=2)
        ax.scatter(x[0], y[0], s=20, c='k', marker='*', zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)


def update_obs(env_idx, new_obs, target_obs):
    for key in new_obs.keys():
        if key is None:
            target_obs[key][env_idx] = new_obs
        else:
            target_obs[key][env_idx] = new_obs[key].view_as(target_obs[key][env_idx])
    return target_obs


def slice_data(data, start_ind, skip, length):
    res = {}
    for k, v in data.items():
        if (isinstance(v, torch.Tensor) and v.dim() > 0) or isinstance(v, np.ndarray) or isinstance(v, list):
            res[k] = v[start_ind: length: skip]
        elif isinstance(v, dict):
            res[k] = slice_data(v, start_ind, skip, length)
        else:
            res[k] = v
    return res


def get_problem_dataset(config, datafile, n_workers=1, worker_id=0):
    data = ProblemLoader.load_problem_data(config=config, problem_range=config['problem_setup']['problem_range'], fname=datafile)
    n_problems = data['positions'].shape[0] # n_problems is now part of loaded data
    starting_problem_ind = worker_id
    if n_workers > 1:
        data = slice_data(data, starting_problem_ind, n_workers, n_problems)
    return data


class RoutingBase(GymEnv, VecEnv):
    def __init__(self, config, problem_properties, state_properties=None, prop_types=None,
                 seed=None, datafile=None, data=None, env_type='train', title='', num_envs=None, depot=True, worker_id=0,
                 n_workers=1, radius=None):
        GymEnv.__init__(self)
        self.gym_format = False
        self.stable_baselines_compatibility = False
        if config['system']['compatibility_mode'] == 'gym':
            self.gym_format = True
        elif config['system']['compatibility_mode'] == 'stable_baselines':
            self.stable_baselines_compatibility = True
        self.worker_id = worker_id
        self.title = title
        self.config = config
        self.graph_representation = config['model']['model_type'].lower() == 'gnn'
        base_properties = ['id', 'positions', 'distance_matrix']
        self.problem_properties = base_properties + problem_properties
        self.prop_types = prop_types
        self.dataset_size = None
        self.is_depot = depot
        self.episode_counter = 0
        self.env_type = env_type
        self.reference = {k: None for k in self.problem_properties}

        # Number of customers (static)
        self.device = config['system']['env_device']
        self.fix_envs = datafile is not None
        if not self.fix_envs:
            assert num_envs is not None, "If no datafile is provided, the number of problems is required"
        self.macro_state = config['muzero']['expansion_method'] == 'K_BEAMS_POLICY_SEARCH'
        self.add_distance_to_head = config['representation']['add_distance_to_head']
        if self.fix_envs:
            data = get_problem_dataset(config, datafile, n_workers=n_workers, worker_id=worker_id)
            self.data = data
            self.net_problem_size = self.data['demand'].shape[1] - int(self.is_depot)
            self.radius = self.data['radius'] if radius is None else radius
            self.max_distance = 1.5 * self.radius  # sqrt(2)*radius + spare
            self.reward_normalization = 1 / self.radius
            if self.config['representation']['normalize_reward_by_problem_size']:
                self.reward_normalization /= math.sqrt(3.14 * self.net_problem_size)
            self.set_problem_data(data=self.data)
        else:
            raise NotImplementedError("On-the-fly problem generation is not implemented")

        # Check consistency between config and env/baseline
        if 'env_type' in self.data:  # TODO temp for backward compatibility; should keep w/o the "if"
            if self.title and self.data['env_type'] != self.title:
                raise ValueError(f"Expected dataset of type {self.title}, but got {self.data['env_type']}.\n"
                                 f"File: {datafile}")
        if 'config' in self.data and 'last_return_to_depot' in self.data['config']['problem_setup']:
            baseline_returns_to_depot = self.data['config']['problem_setup']['last_return_to_depot']
            config_returns_to_depot = self.config['problem_setup']['last_return_to_depot']
            if config_returns_to_depot != baseline_returns_to_depot:
                raise ValueError(f"Last vehicle returning to depot: config ({config_returns_to_depot}) "
                                 f"must match baseline ({baseline_returns_to_depot})")

        self.problem_size = self.net_problem_size + int(self.is_depot)
        self.n_parallel_problems = config['train']['n_parallel_problems'] if num_envs is None else num_envs
        self.num_envs = self.n_parallel_problems  # for backward compatability with SB3 PPO
        self.n_beams = config['train']['n_beams']
        self.extended_output_format = self.n_beams > 0
        if self.extended_output_format:
            self.batch_dim = (self.n_parallel_problems, self.n_beams)
        else:
            self.batch_dim = [self.n_parallel_problems]

        # 2D locations of all customers and the depot in last index (static)
        self.pos = torch.zeros(self.maybe_extend([self.n_parallel_problems, self.problem_size, 2]), device=self.device)
        # Distance between every two points calculated directly from position
        self.distance_matrix = torch.zeros(self.maybe_extend([self.n_parallel_problems, self.problem_size, self.problem_size]),
                                           device=self.device)
        self.ids = np.array([None] * self.n_parallel_problems)
        self.set_spaces()

        self.keys, shapes, dtypes = obs_space_info(self.observation_space)
        self.actions = None
        self.config = config
        self.single_site_visit = config['problem_setup']['single_site_visit']
        self.initialize_static = True
        self.extra_car_penalty = 0

        # for compatibility with stable-baselines
        self.disable_infos = 'ppo' in config['train']['method'].lower()
        self.render_mode = np.full(self.num_envs, None)
        self.state_properties = state_properties
        if self.macro_state:
            self.state_properties += ['acc_returns']
        VecEnv.__init__(self, num_envs=self.num_envs, observation_space=self.observation_space, action_space=self.action_space)
        self.spaces = self.observation_space, self.action_space
        self.seed(seed)
        # placeholder for baselines results
        self.baselines = []

    def maybe_extend(self, x_in):
        if self.extended_output_format:
            x = [x_in[0]] + [self.n_beams] + x_in[1:]
            if isinstance(x_in, tuple):
                x = tuple(x)
            return x
        else:
            return x_in

    def set_spaces(self):
        '''Define observation and action spaces, as well as other class attributes.'''
        raise NotImplementedError

    def reset(self, indices=None, copy_all=False, reset_static=True, seed=None):
        '''
        Resets the environment. This environment might be used with different decoders.
        In case of using with beam-search decoder, we need to have to increase
        the rows of the mask by a factor of beam_width.
        indices is a list of size num_envs of True\False
        '''
        if seed is not None:
            seed_all(seed)
        if indices is None:
            indices = torch.ones(self.n_parallel_problems, dtype=torch.bool)
        if isinstance(indices, list):
            if np.all([isinstance(x, bool) for x in indices]):  # a list of booleans
                indices = torch.tensor(indices)  # a list of booleans
            else:
                new_indices = torch.zeros(self.n_parallel_problems, dtype=torch.bool)
                new_indices[[indices]] = True
                indices = new_indices
        reset_static = (reset_static and not self.env_type == 'eval') or self.initialize_static
        indices = np.pad(indices, (0, self.n_parallel_problems - len(indices)))
        n_problems_to_reset = indices.sum().item()
        if reset_static:
            # self.initialize_static = False
            self.reset_static_properties(n_problems_to_reset, copy_all, indices)
        self.reset_dynamic_properties(indices, n_problems_to_reset)
        state = self.get_pos_representation(indices=indices)
        if self.stable_baselines_compatibility:
            # get_pos_representation may return a TensorDict (when use_tensordict=True)
            # or a plain dict (when use_tensordict=False). Handle both.
            try:
                from tensordict import TensorDict as _TD
            except Exception:
                _TD = None

            if _TD is not None and isinstance(state, _TD):
                state = state.squeeze(1).to_dict()
            # If state is a plain dict, leave tensor shapes intact. The original
            # implementation assumed a TensorDict and squeezed the batch dimension
            # at the TensorDict level; performing per-tensor squeeze here can
            # remove internal singleton dims (e.g. attention channels) and break
            # downstream attention masks. So do nothing for plain dicts.
        if self.gym_format: #convert from TensorDict or dict to numpy dict
            if hasattr(state, 'to_dict'):
                state_dict = state.to_dict()
            else:
                state_dict = state
            state = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
            return state, {}
        else:
            return state

    def set_problem_data(self, data):
        self.episode_counter = 0
        for k in self.problem_properties:
            self.reference[k] = data[k]
            if self.prop_types is not None and k in self.prop_types:
                self.reference[k] = self.reference[k].to(self.prop_types[k])
        self.dataset_size = data['n_problems']
        if self.config['problem_setup']['spare_numeric_capacity'] > 0:
            self.reference['capacity'] += self.config['problem_setup']['spare_numeric_capacity']
        if 'baseline_cost' in data:
            data['baseline_cost_unnormalized'] = np.array(data['baseline_cost'])
            data['baseline_cost'] = np.array(data['baseline_cost']) * self.reward_normalization

    def reset_dynamic_properties(self, indices, n_envs_to_reset):
        raise NotImplementedError

    def reset_static_properties(self, n_envs_to_reset, copy_all, indices):
        raise NotImplementedError

    def set_state(self, states_array):
        states_data, n_actions_from_head = states_array.states, states_array.n_actions_from_head
        if self.state_properties is None:
            raise NotImplementedError
        for problem_index, (states, n_actions) in enumerate(zip(states_data, n_actions_from_head)):
            counter = 0
            for head_index, n in enumerate(n_actions):
                if n > 0:
                    for k in self.state_properties:
                        arr = getattr(self, k)
                        if n > 1:
                            arr[problem_index, counter: counter + n] = states[k][head_index].clone()
                        else:
                            arr[problem_index, counter] = states[k][head_index].clone()
                    counter += n
            if counter < self.n_beams:
                for k in self.state_properties:
                    arr = getattr(self, k)
                    arr[problem_index, counter:] = self.terminal_state[k].clone()

    def get_state(self, ind=Ellipsis):
        if self.extended_output_format and ind is not Ellipsis:
            # assumes ind is an integer!
            res = {k: self.__dict__[k][:, ind:ind + 1].clone() for k in self.state_properties}  # sampling only the first
        else:
            res = {k: self.__dict__[k].clone() for k in self.state_properties}
        batch_dim = res['demand'].shape[:-1]
        res = TensorDict(res, batch_size=batch_dim)
        return res

    def step(self, action, automatic_reset=True, **kwargs):
        obs, buf_rews, buf_dones, buf_infos = self._step(action, **kwargs)
        # Robustly handle different tensor/array types for stable-baselines compatibility
        if self.stable_baselines_compatibility:
            # Normalize buf_dones to a 1D numpy boolean array of length num_envs
            if isinstance(buf_dones, torch.Tensor):
                buf_dones = buf_dones.squeeze(1).cpu().numpy() if buf_dones.dim() > 1 else buf_dones.cpu().numpy()
            elif isinstance(buf_dones, np.ndarray):
                buf_dones = np.squeeze(buf_dones, axis=1) if buf_dones.ndim > 1 else buf_dones
            else:
                try:
                    buf_dones = np.array(buf_dones)
                except Exception:
                    pass
        # save final observation where user can get it, then reset
        if buf_dones.any() and automatic_reset:
            new_obs = self.reset(buf_dones)
            obs = update_obs(buf_dones, new_obs, obs)
        if self.disable_infos:
            infos = [{} for _ in range(self.num_envs)]
        else:
            infos = deepcopy(buf_infos)

        if self.stable_baselines_compatibility:
            # Normalize buffer rewards to numpy 1D
            if isinstance(buf_rews, torch.Tensor):
                buf_rews = buf_rews.squeeze(1).cpu().numpy() if buf_rews.dim() > 1 else buf_rews.cpu().numpy()
            elif isinstance(buf_rews, np.ndarray):
                buf_rews = np.squeeze(buf_rews, axis=1) if buf_rews.ndim > 1 else buf_rews
            else:
                try:
                    buf_rews = np.array(buf_rews)
                except Exception:
                    pass

            # Convert obs to a plain dict of numpy arrays (expected by SB3)
            if hasattr(obs, 'squeeze') and hasattr(obs, 'to_dict'):
                # TensorDict path
                obs_td = obs.squeeze(1).to_dict()
            elif isinstance(obs, dict):
                obs_td = obs
            else:
                obs_td = obs

            # Convert tensor values to numpy arrays where necessary
            obs_dict = {}
            if isinstance(obs_td, dict):
                for k, v in obs_td.items():
                    if isinstance(v, torch.Tensor):
                        obs_dict[k] = v.cpu().numpy()
                    else:
                        obs_dict[k] = v
            else:
                obs_dict = obs_td

            obs = obs_dict

            sb_info = [{} for _ in range(self.num_envs)]
            # `infos` may be a dict of per-key arrays (original code) or a
            # list of per-env dicts (e.g., when we filled with placeholders).
            if isinstance(infos, dict):
                for i in range(self.num_envs):
                    sb_info[i].update({k: (v[i] if (isinstance(v, (list, tuple, np.ndarray)) or isinstance(v, torch.Tensor)) else v)
                                        for k, v in infos.items()})
                    sb_info[i].update({'episode': None, 'is_success': None})
            elif isinstance(infos, list):
                for i in range(self.num_envs):
                    if isinstance(infos[i], dict):
                        sb_info[i].update(infos[i])
                    sb_info[i].update({'episode': None, 'is_success': None})
            else:
                for i in range(self.num_envs):
                    sb_info[i].update({'episode': None, 'is_success': None})
            infos = sb_info
        if self.gym_format:
            # obs may be a TensorDict or a plain dict
            if hasattr(obs, 'to_dict'):
                state_dict = obs.to_dict()
            else:
                state_dict = obs
            obs = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}

            if isinstance(buf_rews, torch.Tensor):
                buf_rews = buf_rews.cpu().numpy()
            elif not isinstance(buf_rews, np.ndarray):
                try:
                    buf_rews = np.array(buf_rews)
                except Exception:
                    pass

            if isinstance(buf_dones, torch.Tensor):
                buf_dones = buf_dones.cpu().numpy()
            elif not isinstance(buf_dones, np.ndarray):
                try:
                    buf_dones = np.array(buf_dones)
                except Exception:
                    pass

            # new gym standard: obs, rewards, terminated, truncated, infos
            return obs, buf_rews, buf_dones, np.zeros_like(buf_dones), infos
        else:
            # previous gym standard: obs, rewards, dones, infos
            return obs, buf_rews, buf_dones, infos


    def _step(self, actions, calc_obs=True):
        raise NotImplementedError

    def write_state(self, states, multiplicity):
        """
        write a state to the env
        Args:
            states: list of k states
            multiplicity:  the number of copies for each state
            env_indices: k tuples of indices
        Distance matrix and positions are assumed fixed.
        """
        if self.state_properties is None:
            raise NotImplementedError
        counter = 0
        for s, n in zip(states, multiplicity):
            for k in self.state_properties:
                arr = getattr(self, k)
                arr[counter:counter + n] = s[k]
            self.feasible_nodes[counter:counter + n] = s['unmasked']
            counter += n

    def get_feature_representation(self, indices=None, extra_global_features=None):
        raise NotImplementedError

    def seed(self, seed=None):
        if seed is None:
            seed = self.config['train']['seed']
        seed_all(seed + self.worker_id, deterministic=self.config['train']['deterministic_algo'])

    def env_is_wrapped(self, wrapper_class, indices=None):
        return False

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        indices = self._get_indices(indices)
        return getattr(self, attr_name)[indices]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        raise NotImplementedError
        # return self.venv.set_attr(attr_name, value, indices)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        raise NotImplementedError
        # return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def step_wait(self):
        raise NotImplementedError

    def step_async(self, actions: np.ndarray):
        raise NotImplementedError

    def is_delivery_action(self, action):
        return action != 0

    def is_reset_action(self, action):
        return action == 0

    def state_to_hashable(self, state):
        raise NotImplementedError

    def baseline_values(self, obs, valid_actions, mode=None):
        raise NotImplementedError
