# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import logging
import os
import pickle
import gc
import torch
import numpy as np

# Removed: shutil, traceback, uuid, warnings, defaultdict, cudf, wandb, cuOpt imports
# Removed: DEFAULT_FIX_CAPACITY, DEFAULT_CAPACITIES, DEPOT_LOCATION, RESET_LOCATION, UNMASKED_INDEX
# Removed: write_dataset (function)

class ProblemLoader(object):
    """
    Loads pre-generated problem data from a pickle file.
    """
    def __init__(self, config):
        # This __init__ is not strictly necessary if all methods are static,
        # but kept for consistency or future non-static helpers.
        # self.config = config
        pass

    @staticmethod
    def load_problem_data(config, fname: str, problem_range: list = None):
        """
        Loads problem data from a .pkl file.

        Args:
            config: The main configuration dictionary (currently not used by this static method
                    but kept for API consistency or future use).
            fname: Path to the .pkl file containing the problem data.

        Returns:
            A dictionary containing the loaded and processed problem data.
        """
        if not os.path.exists(fname):
            logging.error(f'Data file {fname} not found.')
            raise FileNotFoundError(f'Data file {fname} not found.')

        logging.info(f'Loading data from {fname}')
        with open(fname, 'rb') as f:
            loaded_data_pkl = pickle.load(f)

        data = {}

        # 1. env_type (String)
        data['env_type'] = loaded_data_pkl.get('env_type', 'vrp')

        # 2. positions (torch.Tensor)
        if 'positions' in loaded_data_pkl:
            pos_val = loaded_data_pkl['positions']
            if problem_range is not None:
                pos_val = pos_val[problem_range[0]:problem_range[1]]
            # prefer float32 to reduce memory footprint
            if isinstance(pos_val, np.ndarray):
                if pos_val.dtype == np.float64:
                    pos_val = pos_val.astype(np.float32)
                data['positions'] = torch.from_numpy(pos_val)
            elif isinstance(pos_val, torch.Tensor):
                data['positions'] = pos_val
            else:
                raise TypeError(f"Field 'positions' has unsupported type: {type(pos_val)}")
        else:
            raise KeyError("Field 'positions' is missing from the data file.")

        # 3. demand (torch.Tensor) - handles renaming from 'demands'
        demand_val = None
        if 'demand' in loaded_data_pkl:
            demand_val = loaded_data_pkl['demand']
        elif 'demands' in loaded_data_pkl:
            demand_val = loaded_data_pkl['demands']
        
        if demand_val is not None:
            if problem_range is not None:
                demand_val = demand_val[problem_range[0]:problem_range[1]]
            if isinstance(demand_val, np.ndarray):
                data['demand'] = torch.from_numpy(demand_val)
            elif isinstance(demand_val, torch.Tensor):
                data['demand'] = demand_val
            else:
                raise TypeError(f"Field 'demand'/'demands' has unsupported type: {type(demand_val)}")
        else:
            raise KeyError("Field 'demand' or 'demands' is missing from the data file.")

        # 4. distance_matrix (torch.Tensor)
        if 'distance_matrix' in loaded_data_pkl:
            dm_val = loaded_data_pkl['distance_matrix']
            if problem_range is not None:
                dm_val = dm_val[problem_range[0]:problem_range[1]]
            # distance matrices are often the largest tensors; cast to float32
            if isinstance(dm_val, np.ndarray):
                if dm_val.dtype == np.float64:
                    dm_val = dm_val.astype(np.float32)
                data['distance_matrix'] = torch.from_numpy(dm_val)
            elif isinstance(dm_val, torch.Tensor):
                data['distance_matrix'] = dm_val
            else:
                raise TypeError(f"Field 'distance_matrix' has unsupported type: {type(dm_val)}")
        else:
            # If distance_matrix is optional and can be computed from positions,
            # this could be a warning and a computation step.
            # For now, assuming it's required based on typical VRP datasets.
            raise KeyError("Field 'distance_matrix' is missing from the data file.")

        # 5. capacity (torch.Tensor, float) - handles renaming from 'capacities'
        capacity_val = None
        if 'capacity' in loaded_data_pkl:
            capacity_val = loaded_data_pkl['capacity']
        elif 'capacities' in loaded_data_pkl:
            capacity_val = loaded_data_pkl['capacities']

        if capacity_val is not None:
            if problem_range is not None:
                capacity_val = capacity_val[problem_range[0]:problem_range[1]]
            if isinstance(capacity_val, np.ndarray):
                data['capacity'] = torch.from_numpy(capacity_val).float()
            elif isinstance(capacity_val, torch.Tensor):
                data['capacity'] = capacity_val.float()
            elif isinstance(capacity_val, (int, float)): # Handle scalar capacity
                data['capacity'] = torch.tensor([capacity_val] * data['positions'].shape[0], dtype=torch.float32) \
                                   if data['positions'].shape[0] > 1 and np.isscalar(capacity_val) \
                                   else torch.tensor(capacity_val, dtype=torch.float32)
            else:
                raise TypeError(f"Field 'capacity'/'capacities' has unsupported type: {type(capacity_val)}")
        else:
            raise KeyError("Field 'capacity' or 'capacities' is missing from the data file.")

        # Derived and other important fields for vehicle_routing.py
        data['n_problems'] = data['positions'].shape[0]

        if 'radius' in loaded_data_pkl:
            radius_val = loaded_data_pkl['radius']
        else:
            radius_val = data['positions'].abs().max().item()
        data['radius'] = radius_val # Keep as float/int as per original direct usage

        if 'id' in loaded_data_pkl:
            ids_val = loaded_data_pkl['id']
            if isinstance(ids_val, (np.ndarray, list)):
                data['id'] = np.array(ids_val) # vehicle_routing.py expects numpy array for ids
            else:
                logging.warning(f"Field 'id' has an unsupported type: {type(ids_val)}. Skipping.")
        else:
            # Generate default IDs if missing, as RoutingBase expects self.ids
            data['id'] = np.arange(data['n_problems'])
            logging.info(f"Field 'id' missing. Generated default IDs: {data['id']}")


        # Load original problem config if present (for consistency checks in vehicle_routing.py)
        if 'config' in loaded_data_pkl:
            data['config'] = loaded_data_pkl['config']

        # Load dataset_type if present (used for consistency check in vehicle_routing.py as data['env_type'])
        # This might be redundant if loaded_data_pkl['env_type'] is already used for data['env_type']
        if 'dataset_type' in loaded_data_pkl and 'env_type' not in loaded_data_pkl:
            # If original pkl had 'dataset_type' but not 'env_type', use it for 'env_type'.
            data['env_type'] = loaded_data_pkl['dataset_type']


        # 6. time_windows (optional, for VRPTW / PDPTW)
        if 'time_windows' in loaded_data_pkl:
            tw_val = loaded_data_pkl['time_windows']
            if problem_range is not None:
                tw_val = tw_val[problem_range[0]:problem_range[1]]
            if isinstance(tw_val, np.ndarray):
                data['time_windows'] = torch.from_numpy(tw_val).float()
            elif isinstance(tw_val, torch.Tensor):
                data['time_windows'] = tw_val.float()
            else:
                raise TypeError(f"Field 'time_windows' has unsupported type: {type(tw_val)}")

        # 7. service_times (optional, for VRPTW / PDPTW)
        if 'service_times' in loaded_data_pkl:
            svc_val = loaded_data_pkl['service_times']
            if problem_range is not None:
                svc_val = svc_val[problem_range[0]:problem_range[1]]
            if isinstance(svc_val, np.ndarray):
                data['service_times'] = torch.from_numpy(svc_val).float()
            elif isinstance(svc_val, torch.Tensor):
                data['service_times'] = svc_val.float()
            else:
                raise TypeError(f"Field 'service_times' has unsupported type: {type(svc_val)}")

        # 8. pairs (optional, for PDPTW) - pickup-delivery pairs (n_pairs, 2)
        if 'pairs' in loaded_data_pkl:
            pairs_val = loaded_data_pkl['pairs']
            if problem_range is not None:
                pairs_val = pairs_val[problem_range[0]:problem_range[1]]
            if isinstance(pairs_val, np.ndarray):
                data['pairs'] = torch.from_numpy(pairs_val).long()
            elif isinstance(pairs_val, torch.Tensor):
                data['pairs'] = pairs_val.long()
            else:
                raise TypeError(f"Field 'pairs' has unsupported type: {type(pairs_val)}")

        # Try to free the original loaded pickle to reduce peak memory usage
        try:
            del loaded_data_pkl
            gc.collect()
        except Exception:
            pass

        # Ensure essential data used by RoutingBase.set_problem_data is present
        for key_check in ['positions', 'demand', 'distance_matrix', 'capacity', 'id', 'n_problems', 'radius', 'env_type']:
            if key_check not in data:
                # Some like 'radius' or 'id' might have logged warnings but are critical
                if key_check == 'radius' and key_check not in data: # If radius was skipped due to type
                     raise KeyError(f"Field '{key_check}' is missing or has an unsupported type and is required.")
                # id is now defaulted if missing
                # Others would have raised KeyError already if missing mandatory field from pkl
                pass
        return data

# Removed: load_real_data, generate_data, sample_real_distances, generate_distances, generate_demands,
#          generate_time_windows, create_dataset, is_better
# Removed: get_num_carriers_candidates, convert_cuopt_solution, test_cuopt_solution
# Removed global constants related to data generation


def _pad_dataset(data: dict, max_size: int) -> dict:
    """Pad all node-indexed tensor fields in *data* so the node dimension equals *max_size*.

    A ``valid_mask`` boolean tensor of shape ``(n_problems, max_size)`` is added
    where ``True`` marks real nodes and ``False`` marks padding slots.

    Args:
        data: Dataset dict as returned by :meth:`ProblemLoader.load_problem_data`.
        max_size: Target number of nodes (must be >= current size).

    Returns:
        A new dict with all node-level tensors padded to *max_size*.
    """
    n, cur_size = data['positions'].shape[:2]
    pad = max_size - cur_size

    result = dict(data)

    # valid_mask: True for real nodes, False for padding
    valid_mask = torch.zeros(n, max_size, dtype=torch.bool)
    valid_mask[:, :cur_size] = True
    result['valid_mask'] = valid_mask

    if pad == 0:
        return result

    def _pad2d(t):
        """(n, cur_size) -> (n, max_size)"""
        return torch.cat([t, torch.zeros(n, pad, dtype=t.dtype)], dim=1)

    def _pad3d(t):
        """(n, cur_size, k) -> (n, max_size, k)"""
        k = t.shape[2]
        return torch.cat([t, torch.zeros(n, pad, k, dtype=t.dtype)], dim=1)

    def _pad_dm(t):
        """(n, cur_size, cur_size) -> (n, max_size, max_size)"""
        t = torch.cat([t, torch.zeros(n, cur_size, pad, dtype=t.dtype)], dim=2)
        return torch.cat([t, torch.zeros(n, pad, max_size, dtype=t.dtype)], dim=1)

    result['positions'] = _pad3d(data['positions'])
    result['demand'] = _pad2d(data['demand'])
    result['distance_matrix'] = _pad_dm(data['distance_matrix'])

    if 'time_windows' in data:
        result['time_windows'] = _pad3d(data['time_windows'])
    if 'service_times' in data:
        result['service_times'] = _pad2d(data['service_times'])
    # 'pairs' indices reference original node IDs -- no padding needed
    return result


def _merge_datasets(datasets: list) -> dict:
    """Concatenate a list of (same max-size) datasets along the problem axis.

    Tensor fields are concatenated; scalar / non-tensor fields are taken from
    the first dataset.  ``n_problems`` is updated to the total count.

    Args:
        datasets: List of data dicts with identical node-dimension sizes.

    Returns:
        A single merged data dict.
    """
    if len(datasets) == 1:
        return datasets[0]

    result = {}
    keys = set(datasets[0].keys())
    for k in keys:
        v0 = datasets[0].get(k)
        if v0 is None:
            continue
        if isinstance(v0, torch.Tensor) and v0.dim() > 0:
            try:
                result[k] = torch.cat([d[k] for d in datasets if k in d], dim=0)
            except Exception:
                result[k] = v0
        elif isinstance(v0, np.ndarray):
            result[k] = np.concatenate([d[k] for d in datasets if k in d], axis=0)
        else:
            result[k] = v0  # scalar: keep first dataset's value

    result['n_problems'] = sum(d['n_problems'] for d in datasets)
    # Re-generate contiguous IDs for the merged dataset
    result['id'] = np.arange(result['n_problems'])
    return result


class MultiSizeDataLoader:
    """Loads and merges problem datasets of potentially different problem sizes.

    When datasets have different numbers of nodes, all problems are padded to
    the maximum size.  A boolean ``valid_mask`` tensor of shape
    ``(n_total_problems, max_size)`` is added to the merged dataset, where
    ``True`` marks real nodes and ``False`` marks padding slots.

    The padding approach allows a single environment instance (with a fixed
    ``problem_size = max_size``) to train on problems of mixed sizes within the
    same training run.  Padding nodes carry zero demand and are kept
    infeasible throughout the episode, so they are never selected as actions.

    Example usage (config)::

        eval:
          data_files:
            - datasets/vrp_50.pkl
            - datasets/vrp_100.pkl
            - datasets/vrp_200.pkl

    The environment reads ``data_files`` (list) when present; otherwise it
    falls back to the single ``data_file`` string.
    """

    @staticmethod
    def load_mixed_data(config, fnames: list, problem_range: list = None) -> dict:
        """Load and merge multiple pkl dataset files, padding to the largest size.

        Args:
            config: Main configuration dictionary (passed through to
                :meth:`ProblemLoader.load_problem_data`).
            fnames: Ordered list of paths to ``.pkl`` dataset files.
            problem_range: Optional ``[start, end]`` slice applied to each
                dataset independently.

        Returns:
            A merged data dict where every problem has the same node count
            (``max_size`` across all loaded files).  An extra ``valid_mask``
            field ``(n_total, max_size)`` bool tensor is included.
        """
        if not fnames:
            raise ValueError("fnames must contain at least one file path.")

        datasets = [
            ProblemLoader.load_problem_data(config=config, fname=fname,
                                            problem_range=problem_range)
            for fname in fnames
        ]

        if len(datasets) == 1:
            d = datasets[0]
            n, s = d['positions'].shape[:2]
            if 'valid_mask' not in d:
                d['valid_mask'] = torch.ones(n, s, dtype=torch.bool)
            return d

        max_size = max(d['positions'].shape[1] for d in datasets)
        padded = [_pad_dataset(d, max_size) for d in datasets]
        return _merge_datasets(padded)
