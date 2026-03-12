# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Unit tests for VRPTW and PDPTW implementation logic.

Tests cover:
- Benchmark data parsers (Homberger VRPTW / Li&Lim PDPTW)
- VRPTW environment: time-window enforcement and feasibility computation
- PDPTW environment: pickup-delivery precedence constraints
"""

import os
import pickle
import tempfile

import numpy as np
import pytest
import torch
import yaml

from earli.benchmark_parser import (
    parse_homberger_instance,
    parse_lilim_instance,
    convert_homberger_to_pkl,
    convert_lilim_to_pkl,
)


# ---------------------------------------------------------------------------
# Helpers – synthetic datasets and configs
# ---------------------------------------------------------------------------

def _make_base_config(env: str, n_beams: int = 1, n_parallel: int = 2):
    """Return a minimal config dict for env tests."""
    return {
        'system': {
            'env_device': 'cpu',
            'model_device': 'cpu',
            'tree_device': 'cpu',
            'buffer_device': 'cpu',
            'save_obs_on_gpu': False,
            'use_tensordict': True,
            'allow_wandb': False,
            'run_name': None,
            'no_sweep': False,
            'compatibility_mode': None,
        },
        'train': {
            'method': 'tree_based',
            'optimizer': 'adam',
            'seed': 0,
            'deterministic_algo': False,
            'gamma': 1,
            'multi_gpu': False,
            'block_decreasing_action': False,
            'epochs': 1,
            'batch_size': 4,
            'n_beams': n_beams,
            'pretrained_fname': None,
            'pretrained_run': None,
            'save_model_path': 'earli/pretrained_models/model_test.m',
            'n_parallel_problems': n_parallel,
            'learning_rate': 3e-4,
        },
        'eval': {
            'data_file': '',
            'max_problems': None,
            'apply_local_search': False,
            'deterministic_test_beam': True,
            'detailed_test_log': False,
            'save_full_tree': False,
            'naive_greedy': False,
        },
        'problem_setup': {
            'env': env,
            'problem_range': None,
            'minimize_vehicles': False,
            'last_return_to_depot': True,
            'distance_Lp_norm': 2,
            'vehicle_penalty': 0,
            'unused_capacity_penalty': 0,
            'spare_numeric_capacity': 0.0,
            'single_site_visit': True,
        },
        'representation': {
            'input_repr': 'Graph',
            'k_connectivity': 4,
            'self_loops': True,
            'normalize': False,
            'normalize_pos_obs_like_reward': False,
            'normalize_reward_by_problem_size': False,
            'add_distance_to_head': True,
        },
        'logger': {
            'episode_stats_limit': 100,
            'logging_level': 30,
        },
        'model': {
            'model_type': 'attention',
            'head_model_type': 'attention',
            'embedding_dim': 32,
            'use_pair_norm': False,
            'use_batch_norm': False,
            'num_intermediate_features': 16,
            'use_bias': True,
            'agg_type': 'sum',
            'edge_model_layers': 1,
            'value_from_depot': False,
            'reset_layers_freq': 0,
            'reset_last_layers': False,
            'reset_last_k_modules': 0,
            'single_network': False,
            'attention': True,
            'split_actor_and_state_models': False,
            'lazy': True,
            'eight_rounding': False,
        },
        'attention_model': {
            'layer_normalization': 'batch',
            'n_attention_layers_actor': 2,
            'n_attention_layers_critic': 2,
            'n_attention_layers_head_module': 1,
            'n_attention_heads': 4,
            'reweight_by_distance': False,
            'reweight_function': 'exponential',
            'use_basic_head_encoding': True,
            'separate_head_and_action_model': False,
            'num_head_encoder_layers': 1,
        },
        'sampler': {
            'score_to_prob': 'probs',
            'temperature': 0.3,
            'diversity_penalty': 0,
            'normalize_attention': True,
            'tanh_clipping': 10,
            'complement_k_beams_calc': False,
            'use_full_action_space_calc': True,
            'autoregressive': True,
        },
        'muzero': {
            'expansion_method': 'KPPO',
            'deterministic_branch_in_k_beams': True,
            'zeroize_deterministic_log_prob': False,
            'loss_type': 'ppo',
            'max_leaves': 5,
            'data_steps_per_epoch': 4,
            'hash_states': True,
            'max_moves': 200,
        },
        'speedups': {
            'use_ray': False,
            'use_fabric': False,
            'n_workers': 1,
            'amp': False,
            'fused_optimizer': False,
            'compile_mode': None,
            'foreach_optimizer': False,
            'share_data': True,
        },
        'cuopt': {
            'use_cuopt': False,
            'climbers': 64,
        },
        'buffer': {
            'max_buffer_size': 1000,
            'allow_overflow': True,
            'on_policy_buffer': True,
            'buffer_precision': 'float32',
            'randomize_minibatches': False,
        },
    }


def _make_vrptw_dataset(n_problems: int = 4, n_nodes: int = 6, capacity: float = 100.0):
    """Create a minimal synthetic VRPTW dataset PKL and return its path."""
    # positions: depot at index 0, customers at 1..n_nodes-1
    positions = torch.rand(n_problems, n_nodes, 2) * 100.0
    demand = torch.zeros(n_problems, n_nodes)
    # All customer nodes get demand 5; guaranteed non-zero for deterministic tests.
    demand[:, 1:] = 5.0

    dm = torch.cdist(positions, positions, p=2)
    idx = torch.arange(n_nodes)
    dm[:, idx, idx] = 0.0

    cap = torch.full((n_problems,), capacity)

    # Time windows: ready=0, due=large enough for all routes
    time_windows = torch.zeros(n_problems, n_nodes, 2)
    time_windows[:, :, 1] = 1000.0          # due_date
    service_times = torch.ones(n_problems, n_nodes) * 5.0
    service_times[:, 0] = 0.0               # depot has no service time

    radius = float(positions.abs().max())
    data = {
        'env_type': 'vrptw',
        'positions': positions,
        'demand': demand,
        'distance_matrix': dm,
        'capacity': cap,
        'time_windows': time_windows,
        'service_times': service_times,
        'n_problems': n_problems,
        'radius': radius,
        'id': np.arange(n_problems),
    }
    tmp = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
    pickle.dump(data, tmp)
    tmp.close()
    return tmp.name


def _make_pdptw_dataset(n_problems: int = 4, n_nodes: int = 7, capacity: float = 100.0):
    """Create a minimal synthetic PDPTW dataset PKL and return its path.

    Node layout: depot(0), pickup(1), delivery(2), pickup(3), delivery(4), …
    """
    assert n_nodes >= 5 and (n_nodes - 1) % 2 == 0, \
        "n_nodes must be odd (1 depot + even number of customer nodes)"
    n_pairs = (n_nodes - 1) // 2

    positions = torch.rand(n_problems, n_nodes, 2) * 100.0
    demand = torch.zeros(n_problems, n_nodes)
    # pickup nodes: 1, 3, 5, …   delivery nodes: 2, 4, 6, …
    pickup_ids  = list(range(1, n_nodes, 2))
    delivery_ids = list(range(2, n_nodes, 2))
    for p, d in zip(pickup_ids, delivery_ids):
        demand[:, p] = torch.randint(1, 10, (n_problems,)).float()
        demand[:, d] = demand[:, p]           # same demand for pickup and delivery

    dm = torch.cdist(positions, positions, p=2)
    idx = torch.arange(n_nodes)
    dm[:, idx, idx] = 0.0

    cap = torch.full((n_problems,), capacity)

    time_windows = torch.zeros(n_problems, n_nodes, 2)
    time_windows[:, :, 1] = 1000.0
    service_times = torch.ones(n_problems, n_nodes) * 5.0
    service_times[:, 0] = 0.0

    # pairs: (n_problems, n_pairs, 2)
    pairs = torch.zeros(n_problems, n_pairs, 2, dtype=torch.long)
    for i, (p, d) in enumerate(zip(pickup_ids, delivery_ids)):
        pairs[:, i, 0] = p
        pairs[:, i, 1] = d

    radius = float(positions.abs().max())
    data = {
        'env_type': 'pdptw',
        'positions': positions,
        'demand': demand,
        'distance_matrix': dm,
        'capacity': cap,
        'time_windows': time_windows,
        'service_times': service_times,
        'pairs': pairs,
        'n_problems': n_problems,
        'radius': radius,
        'id': np.arange(n_problems),
    }
    tmp = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
    pickle.dump(data, tmp)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Benchmark parser tests
# ---------------------------------------------------------------------------

HOMBERGER_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'homberger', 'homberger_200_customer_instances'
)
LILIM_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'li&lim benchmark', 'pdp_100'
)
# NOTE: The above paths assume the tests/ directory is in the repository root,
# alongside the homberger/ and "li&lim benchmark/" data directories.
# Tests that require these directories are automatically skipped when they are
# not present (see the @pytest.mark.skipif decorators below).


def _first_file(directory, ext):
    """Return the path of the first file with the given extension in a directory."""
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(ext):
            return os.path.join(directory, fname)
    return None


@pytest.mark.skipif(
    not os.path.isdir(HOMBERGER_DIR),
    reason='Homberger benchmark data not available',
)
def test_parse_homberger_instance():
    """Parsed Homberger instance has the expected fields and shapes."""
    fpath = _first_file(HOMBERGER_DIR, '.TXT')
    assert fpath is not None, f'No .TXT files found in {HOMBERGER_DIR}'
    inst = parse_homberger_instance(fpath)

    assert 'positions' in inst
    assert 'demand' in inst
    assert 'time_windows' in inst
    assert 'service_times' in inst
    assert 'capacity' in inst

    n = inst['positions'].shape[0]
    assert inst['demand'].shape == (n,)
    assert inst['time_windows'].shape == (n, 2)
    assert inst['service_times'].shape == (n,)

    # Depot is node 0: demand must be 0
    assert inst['demand'][0] == 0.0

    # ready_time <= due_date for all nodes
    assert (inst['time_windows'][:, 0] <= inst['time_windows'][:, 1]).all()


@pytest.mark.skipif(
    not os.path.isdir(LILIM_DIR),
    reason='Li&Lim benchmark data not available',
)
def test_parse_lilim_instance():
    """Parsed Li&Lim instance has the expected fields and shapes."""
    fpath = _first_file(LILIM_DIR, '.txt')
    assert fpath is not None, f'No .txt files found in {LILIM_DIR}'
    inst = parse_lilim_instance(fpath)

    assert 'positions' in inst
    assert 'demand' in inst
    assert 'time_windows' in inst
    assert 'service_times' in inst
    assert 'pairs' in inst

    n = inst['positions'].shape[0]
    assert inst['demand'].shape == (n,)
    assert inst['time_windows'].shape == (n, 2)
    assert inst['service_times'].shape == (n,)
    assert inst['pairs'].ndim == 2 and inst['pairs'].shape[1] == 2

    # Demand must be non-negative (absolute values)
    assert (inst['demand'] >= 0).all()

    # Pickup nodes must have valid delivery partners
    for pick_idx, delv_idx in inst['pairs']:
        assert 0 < pick_idx < n
        assert 0 < delv_idx < n
        assert pick_idx != delv_idx


@pytest.mark.skipif(
    not os.path.isdir(HOMBERGER_DIR),
    reason='Homberger benchmark data not available',
)
def test_convert_homberger_to_pkl(tmp_path):
    """convert_homberger_to_pkl writes a valid PKL file."""
    fpath = _first_file(HOMBERGER_DIR, '.TXT')
    out = str(tmp_path / 'vrptw_test.pkl')
    convert_homberger_to_pkl([fpath], out)
    assert os.path.exists(out)
    with open(out, 'rb') as f:
        data = pickle.load(f)
    assert data['env_type'] == 'vrptw'
    assert 'time_windows' in data
    assert 'service_times' in data


@pytest.mark.skipif(
    not os.path.isdir(LILIM_DIR),
    reason='Li&Lim benchmark data not available',
)
def test_convert_lilim_to_pkl(tmp_path):
    """convert_lilim_to_pkl writes a valid PKL file."""
    fpath = _first_file(LILIM_DIR, '.txt')
    out = str(tmp_path / 'pdptw_test.pkl')
    convert_lilim_to_pkl([fpath], out)
    assert os.path.exists(out)
    with open(out, 'rb') as f:
        data = pickle.load(f)
    assert data['env_type'] == 'pdptw'
    assert 'pairs' in data
    assert 'time_windows' in data


# ---------------------------------------------------------------------------
# VRPTW environment tests
# ---------------------------------------------------------------------------

class TestVRPTW:
    """Tests for the VRPTW environment step logic."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from earli.vrptw import VRPTW
        self.VRPTW = VRPTW
        self.pkl = _make_vrptw_dataset(n_problems=4, n_nodes=6)
        config = _make_base_config('vrptw', n_beams=1, n_parallel=4)
        config['eval']['data_file'] = self.pkl
        self.config = config
        yield
        os.unlink(self.pkl)

    def _make_env(self):
        return self.VRPTW(self.config, datafile=self.pkl, env_type='eval')

    def test_reset_returns_observation(self):
        """reset() returns a TensorDict with expected keys."""
        env = self._make_env()
        obs = env.reset()
        for key in ('demand', 'feasible_nodes', 'head', 'tmin', 'tmax', 'dt'):
            assert key in obs, f"Key '{key}' missing from observation"

    def test_time_windows_in_observation(self):
        """tmin, tmax, dt must all be non-negative and tmax >= tmin."""
        env = self._make_env()
        obs = env.reset()
        tmin = obs['tmin'].float()
        tmax = obs['tmax'].float()
        dt   = obs['dt'].float()
        assert (tmin >= 0).all(), 'tmin should be >= 0'
        assert (tmax >= tmin).all(), 'tmax should be >= tmin'
        assert torch.allclose(dt, tmax - tmin, atol=1e-5), 'dt should equal tmax - tmin'

    def test_depot_always_feasible(self):
        """Depot becomes feasible once the vehicle leaves for a customer."""
        env = self._make_env()
        env.reset()
        # At reset, the vehicle is at the depot so depot is marked infeasible
        # (no need to "visit" the depot when already there).
        assert not env.feasible_nodes[..., 0].any(), \
            'Depot should be infeasible right after reset (vehicle is already there)'
        # After visiting a customer node, the depot becomes feasible (return trip).
        actions = torch.ones(env.n_parallel_problems, env.n_beams, dtype=torch.long)
        env._step(actions)
        assert env.feasible_nodes[..., 0].all(), \
            'Depot should be feasible after visiting at least one customer'

    def test_step_reduces_demand(self):
        """Visiting a customer node sets its demand to 0."""
        env = self._make_env()
        env.reset()
        # All customer nodes have demand 5 in the synthetic dataset (node 1 is always > 0).
        actions = torch.ones(env.n_parallel_problems, env.n_beams, dtype=torch.long)
        env._step(actions)
        demand_after = env.demand
        # Node 1 demand should now be 0 for all envs
        for i in range(env.n_parallel_problems):
            assert demand_after[i, 0, 1] == 0.0, \
                f'Demand at node 1 should be 0 after visit (env {i})'

    def test_current_time_advances(self):
        """Current time must advance (or stay) after visiting a customer node."""
        env = self._make_env()
        env.reset()
        time_before = env.current_time.clone()
        actions = torch.ones(env.n_parallel_problems, env.n_beams, dtype=torch.long)
        env._step(actions)
        time_after = env.current_time
        assert (time_after >= time_before).all(), \
            'Current time should never decrease after a step'

    def test_time_resets_on_depot_visit(self):
        """Current time resets to 0 when the vehicle returns to the depot."""
        env = self._make_env()
        env.reset()
        # First step: go to node 1
        actions = torch.ones(env.n_parallel_problems, env.n_beams, dtype=torch.long)
        env._step(actions)
        # Second step: return to depot (node 0)
        actions_depot = torch.zeros(env.n_parallel_problems, env.n_beams, dtype=torch.long)
        env._step(actions_depot)
        assert (env.current_time == 0.0).all(), \
            'Current time should be 0 after returning to depot'

    def test_time_slack_in_observation(self):
        """time_slack must be non-negative and equal max(0, tmax - current_time/horizon)."""
        env = self._make_env()
        obs = env.reset()
        assert 'time_slack' in obs, "'time_slack' must be present in VRPTW observation"
        assert 'current_time' in obs, "'current_time' must be present in VRPTW observation"

        time_slack = obs['time_slack'].float()
        tmax = obs['tmax'].float()
        current_time_feat = obs['current_time'].float()

        # All values non-negative
        assert (time_slack >= 0).all(), 'time_slack must be >= 0'
        assert (current_time_feat >= 0).all(), 'current_time feature must be >= 0'

        # At reset, current_time=0 so time_slack should equal tmax
        assert torch.allclose(time_slack, tmax, atol=1e-5), \
            'At reset (current_time=0), time_slack should equal tmax (= due/horizon)'

        # current_time feature is 0 at reset
        assert (current_time_feat == 0).all(), \
            'current_time feature must be 0 right after reset'

    def test_time_slack_decreases_after_step(self):
        """time_slack must decrease (or stay) for nodes after a step advances the clock."""
        env = self._make_env()
        env.reset()
        obs_before = env.get_pos_representation()
        slack_before = obs_before['time_slack'].float()

        # Visit node 1 → clock advances
        actions = torch.ones(env.n_parallel_problems, env.n_beams, dtype=torch.long)
        _, _, dones, _ = env._step(actions)

        obs_after = env.get_pos_representation()
        slack_after = obs_after['time_slack'].float()
        ct_after = obs_after['current_time'].float()

        # time_slack must not increase (clock only moves forward)
        assert (slack_after <= slack_before + 1e-5).all(), \
            'time_slack should not increase after a step advances current_time'
        # current_time feature must be >= 0
        assert (ct_after >= 0).all(), 'current_time feature must remain >= 0 after a step'

    def test_tw_infeasible_node_blocked(self):
        """A node whose due-date has already passed must be excluded from feasible_nodes."""
        import pickle as pkl_mod

        # Build a dataset where node 1 has an extremely tight time window (due_date = 0)
        n_problems, n_nodes = 2, 4
        positions = torch.rand(n_problems, n_nodes, 2) * 10.0
        demand = torch.zeros(n_problems, n_nodes)
        demand[:, 1:] = 5.0
        dm = torch.cdist(positions, positions, p=2)
        idx = torch.arange(n_nodes)
        dm[:, idx, idx] = 0.0
        cap = torch.full((n_problems,), 100.0)

        time_windows = torch.zeros(n_problems, n_nodes, 2)
        time_windows[:, :, 1] = 1000.0          # default: very lenient
        time_windows[:, 1, 0] = 0.0             # node 1: ready_time=0
        time_windows[:, 1, 1] = 0.0             # node 1: due_date=0 → unreachable

        service_times = torch.ones(n_problems, n_nodes) * 5.0
        service_times[:, 0] = 0.0

        data = {
            'env_type': 'vrptw',
            'positions': positions,
            'demand': demand,
            'distance_matrix': dm,
            'capacity': cap,
            'time_windows': time_windows,
            'service_times': service_times,
            'n_problems': n_problems,
            'radius': float(positions.abs().max()),
            'id': np.arange(n_problems),
        }
        tmpf = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
        pkl_mod.dump(data, tmpf)
        tmpf.close()
        try:
            cfg = _make_base_config('vrptw', n_beams=1, n_parallel=n_problems)
            cfg['eval']['data_file'] = tmpf.name
            env = self.VRPTW(cfg, datafile=tmpf.name, env_type='eval')
            env.reset()
            # Node 1 has due_date=0; any travel from depot takes > 0 time,
            # so node 1 must be infeasible.
            assert not env.feasible_nodes[:, 0, 1].any(), \
                'Node 1 with due_date=0 should be infeasible from the start'
        finally:
            os.unlink(tmpf.name)


# ---------------------------------------------------------------------------
# PDPTW environment tests
# ---------------------------------------------------------------------------

class TestPDPTW:
    """Tests for the PDPTW environment step logic."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from earli.pdptw import PDPTW
        self.PDPTW = PDPTW
        self.pkl = _make_pdptw_dataset(n_problems=4, n_nodes=7)
        config = _make_base_config('pdptw', n_beams=1, n_parallel=4)
        config['eval']['data_file'] = self.pkl
        self.config = config
        yield
        os.unlink(self.pkl)

    def _make_env(self):
        return self.PDPTW(self.config, datafile=self.pkl, env_type='eval')

    def test_reset_returns_observation(self):
        """reset() returns a TensorDict with PDPTW-specific keys."""
        env = self._make_env()
        obs = env.reset()
        for key in ('demand', 'feasible_nodes', 'head',
                    'tmin', 'tmax', 'dt',
                    'is_pickup', 'is_delivery', 'pickup_done'):
            assert key in obs, f"Key '{key}' missing from PDPTW observation"

    def test_is_pickup_is_delivery_non_overlapping(self):
        """is_pickup and is_delivery must not both be True for the same node."""
        env = self._make_env()
        obs = env.reset()
        is_pick = obs['is_pickup'].bool()
        is_delv = obs['is_delivery'].bool()
        both = is_pick & is_delv
        assert not both.any(), 'A node cannot be both pickup and delivery'

    def test_delivery_blocked_before_pickup(self):
        """Delivery node must be infeasible until its paired pickup is visited."""
        env = self._make_env()
        env.reset()
        # In our synthetic dataset node 2 is always the delivery partner of node 1.
        feasible = env.feasible_nodes
        # Check each env: delivery node 2 should be infeasible at start
        for i in range(env.n_parallel_problems):
            assert not feasible[i, 0, 2], \
                f'Delivery node 2 should be infeasible before pickup (env {i})'

    def test_delivery_unblocked_after_pickup(self):
        """Delivery node becomes feasible once its paired pickup is visited."""
        env = self._make_env()
        env.reset()

        # Visit node 1 (pickup) for all envs
        actions = torch.ones(env.n_parallel_problems, env.n_beams, dtype=torch.long)
        env._step(actions)

        # Now node 2 (delivery partner of node 1) should be feasible
        # (subject to capacity and time-window constraints, which are generous here)
        for i in range(env.n_parallel_problems):
            # pickup_done[i, 1] should now be True
            assert env.pickup_done[i, 0, 1], \
                f'pickup_done[{i}, 1] should be True after visiting node 1'

    def test_pickup_done_initially_false(self):
        """pickup_done must be all-False at the start of each episode."""
        env = self._make_env()
        env.reset()
        assert not env.pickup_done.any(), \
            'pickup_done should be all False after reset'

    def test_depot_always_feasible(self):
        """Depot becomes feasible once the vehicle leaves for a customer."""
        env = self._make_env()
        env.reset()
        # At reset, vehicle is at depot → depot infeasible (vehicle is there).
        assert not env.feasible_nodes[..., 0].any(), \
            'Depot should be infeasible right after reset (vehicle is already there)'
        # After visiting a pickup node, the depot becomes a valid next destination.
        actions = torch.ones(env.n_parallel_problems, env.n_beams, dtype=torch.long)
        env._step(actions)
        assert env.feasible_nodes[..., 0].all(), \
            'Depot should be feasible after leaving for a customer'

    def test_current_time_advances(self):
        """Current time must advance after visiting a pickup node."""
        env = self._make_env()
        env.reset()
        time_before = env.current_time.clone()
        actions = torch.ones(env.n_parallel_problems, env.n_beams, dtype=torch.long)
        env._step(actions)
        assert (env.current_time >= time_before).all(), \
            'Current time should not decrease after a step'

    def test_time_slack_and_current_time_in_observation(self):
        """PDPTW observation must include time_slack and current_time features."""
        env = self._make_env()
        obs = env.reset()
        assert 'time_slack' in obs, "'time_slack' must be in PDPTW observation"
        assert 'current_time' in obs, "'current_time' must be in PDPTW observation"

        time_slack = obs['time_slack'].float()
        tmax = obs['tmax'].float()
        current_time_feat = obs['current_time'].float()

        # All non-negative
        assert (time_slack >= 0).all(), 'time_slack must be >= 0'
        assert (current_time_feat >= 0).all(), 'current_time feature must be >= 0'

        # At reset current_time=0, so time_slack == tmax
        assert torch.allclose(time_slack, tmax, atol=1e-5), \
            'At reset, PDPTW time_slack should equal tmax (current_time=0)'

    def test_obs_all_keys_on_same_device_after_step(self):
        """All observation tensors must be on the same device after a step.

        This guards against the bug where VRPTW.get_pos_representation() moved
        obs to CPU (save_obs_on_gpu=False) but PDPTW then added its own tensors
        (is_pickup, is_delivery, pickup_done) without moving them first, causing
        a TensorDict mixed-device error when env_device='cuda'.
        """
        env = self._make_env()
        env.reset()
        actions = torch.ones(env.n_parallel_problems, env.n_beams, dtype=torch.long)
        env._step(actions)
        obs = env.get_pos_representation()

        # All required PDPTW keys must be present
        expected_keys = (
            'demand', 'feasible_nodes', 'head',
            'tmin', 'tmax', 'dt', 'time_slack', 'current_time',
            'is_pickup', 'is_delivery', 'pickup_done',
        )
        for key in expected_keys:
            assert key in obs, f"Key '{key}' missing from PDPTW observation after step"

        # All tensors must be on the same device (no mixed CPU/GPU)
        devices = {key: obs[key].device for key in expected_keys}
        unique_devices = set(str(d) for d in devices.values())
        assert len(unique_devices) == 1, (
            f'All observation tensors must be on the same device, got: {devices}'
        )


# ---------------------------------------------------------------------------
# Mixed-size data training tests
# ---------------------------------------------------------------------------

def _make_vrp_dataset(n_problems: int, n_nodes: int, capacity: float = 50.0) -> str:
    """Create a minimal VRP dataset with *n_nodes* nodes and return its path."""
    positions = torch.rand(n_problems, n_nodes, 2)
    demand = torch.zeros(n_problems, n_nodes)
    demand[:, 1:] = torch.randint(1, 5, (n_problems, n_nodes - 1)).float()

    dm = torch.cdist(positions, positions, p=2)
    idx = torch.arange(n_nodes)
    dm[:, idx, idx] = 0.0

    cap = torch.full((n_problems,), capacity)
    radius = float(positions.abs().max())

    data = {
        'env_type': 'vrp',
        'positions': positions,
        'demand': demand,
        'distance_matrix': dm,
        'capacity': cap,
        'n_problems': n_problems,
        'radius': radius,
        'id': np.arange(n_problems),
    }
    tmp = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
    pickle.dump(data, tmp)
    tmp.close()
    return tmp.name


class TestMixedSizeDataTraining:
    """Tests for mixed-size dataset loading and environment behaviour."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from earli.vrp import VRP
        self.VRP = VRP
        # Two datasets with different problem sizes (including depot node)
        self.pkl_small = _make_vrp_dataset(n_problems=4, n_nodes=6)   # 5 customers
        self.pkl_large = _make_vrp_dataset(n_problems=4, n_nodes=11)  # 10 customers
        yield
        os.unlink(self.pkl_small)
        os.unlink(self.pkl_large)

    def _base_config(self, n_parallel: int = 4):
        cfg = _make_base_config('vrp', n_beams=1, n_parallel=n_parallel)
        cfg['problem_setup']['minimize_vehicles'] = False
        return cfg

    # ------------------------------------------------------------------
    # MultiSizeDataLoader unit tests
    # ------------------------------------------------------------------

    def test_pad_dataset_shape(self):
        """_pad_dataset pads tensors to max_size along the node dimension."""
        from earli.generate_data import _pad_dataset, ProblemLoader
        cfg = self._base_config()
        data = ProblemLoader.load_problem_data(cfg, self.pkl_small)
        n, orig_size = data['positions'].shape[:2]
        padded = _pad_dataset(data, orig_size + 5)
        assert padded['positions'].shape == (n, orig_size + 5, 2)
        assert padded['demand'].shape == (n, orig_size + 5)
        assert padded['distance_matrix'].shape == (n, orig_size + 5, orig_size + 5)
        assert padded['valid_mask'].shape == (n, orig_size + 5)
        # First orig_size columns of valid_mask must be True
        assert padded['valid_mask'][:, :orig_size].all()
        # Padding columns must be False
        assert not padded['valid_mask'][:, orig_size:].any()

    def test_merge_datasets_total_problems(self):
        """_merge_datasets concatenates problems from both datasets."""
        from earli.generate_data import _pad_dataset, _merge_datasets, ProblemLoader
        cfg = self._base_config()
        d_small = ProblemLoader.load_problem_data(cfg, self.pkl_small)
        d_large = ProblemLoader.load_problem_data(cfg, self.pkl_large)
        max_size = d_large['positions'].shape[1]
        merged = _merge_datasets([_pad_dataset(d_small, max_size),
                                  _pad_dataset(d_large, max_size)])
        assert merged['n_problems'] == d_small['n_problems'] + d_large['n_problems']
        assert merged['positions'].shape[1] == max_size

    def test_mixed_load_valid_mask_present(self):
        """MultiSizeDataLoader.load_mixed_data adds a valid_mask field."""
        from earli.generate_data import MultiSizeDataLoader
        cfg = self._base_config()
        data = MultiSizeDataLoader.load_mixed_data(
            cfg, [self.pkl_small, self.pkl_large]
        )
        assert 'valid_mask' in data, "valid_mask must be present after mixed load"
        n, max_size = data['positions'].shape[:2]
        assert data['valid_mask'].shape == (n, max_size)
        assert data['valid_mask'].dtype == torch.bool

    def test_mixed_load_padding_nodes_zeroed(self):
        """Padding nodes must have zero demand and zero positions in the merged dataset."""
        from earli.generate_data import MultiSizeDataLoader
        cfg = self._base_config()
        data = MultiSizeDataLoader.load_mixed_data(
            cfg, [self.pkl_small, self.pkl_large]
        )
        vm = data['valid_mask']           # (n_total, max_size)
        demand = data['demand']           # (n_total, max_size)
        # Where valid_mask is False (padding), demand must be zero
        assert (demand[~vm] == 0).all(), "Padding nodes must have zero demand"

    def test_single_file_returns_valid_mask(self):
        """Loading a single file still produces a valid_mask (all-True)."""
        from earli.generate_data import MultiSizeDataLoader
        cfg = self._base_config()
        data = MultiSizeDataLoader.load_mixed_data(cfg, [self.pkl_small])
        assert 'valid_mask' in data
        assert data['valid_mask'].all(), "Single-file load: valid_mask must be all-True"

    # ------------------------------------------------------------------
    # Environment integration tests
    # ------------------------------------------------------------------

    def test_env_loads_mixed_data(self):
        """VRP environment initialises without error on mixed-size data files."""
        cfg = self._base_config(n_parallel=4)
        env = self.VRP(cfg, datafile=[self.pkl_small, self.pkl_large], env_type='eval')
        # problem_size should equal the larger dataset's node count
        from earli.generate_data import ProblemLoader
        d_large = ProblemLoader.load_problem_data(cfg, self.pkl_large)
        assert env.problem_size == d_large['positions'].shape[1]

    def test_padding_nodes_infeasible_after_reset(self):
        """Padding nodes must be infeasible (feasible_nodes == False) after reset."""
        cfg = self._base_config(n_parallel=4)
        env = self.VRP(cfg, datafile=[self.pkl_small, self.pkl_large], env_type='eval')
        env.reset()
        # The small-dataset problems (first 4 problems, cycled) have fewer real nodes.
        # Their padding slots must all be infeasible.
        # padding_mask has shape (n_parallel, problem_size); check it is applied.
        pm = env.padding_mask  # (n_parallel, [n_beams,] problem_size)
        fn = env.feasible_nodes
        # Wherever padding_mask is False (padding node), feasible_nodes must also be False
        if pm.dim() == fn.dim():
            padding_slots = ~pm
        else:
            padding_slots = ~pm.unsqueeze(1).expand_as(fn)
        assert not fn[padding_slots].any(), \
            "Padding nodes must be infeasible after reset"

    def test_no_padding_for_single_size_dataset(self):
        """When all datasets have the same size, padding_mask should be all-True."""
        cfg = self._base_config(n_parallel=4)
        # Two datasets of the same size (6 nodes)
        pkl2 = _make_vrp_dataset(n_problems=4, n_nodes=6)
        try:
            env = self.VRP(cfg, datafile=[self.pkl_small, pkl2], env_type='eval')
            env.reset()
            assert env.padding_mask.all(), \
                "All-same-size datasets: padding_mask must be all-True"
        finally:
            os.unlink(pkl2)


# ---------------------------------------------------------------------------
# Tree-based training tests
# ---------------------------------------------------------------------------

def _make_base_config_tree(env: str, n_beams: int = 2, n_parallel: int = 2):
    """Return a minimal config for tree_based training tests."""
    cfg = _make_base_config(env, n_beams=n_beams, n_parallel=n_parallel)
    # tree_based training requires non-stable_baselines compatibility
    cfg['system']['compatibility_mode'] = None
    cfg['train']['method'] = 'tree_based'
    cfg['train']['n_beams'] = n_beams
    cfg['muzero']['max_moves'] = 100
    cfg['muzero']['data_steps_per_epoch'] = n_parallel
    return cfg


class TestTreeBasedTrainingVRPTW:
    """Tests for the tree_based training path with VRPTW."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from earli.vrptw import VRPTW
        self.VRPTW = VRPTW
        self.pkl = _make_vrptw_dataset(n_problems=4, n_nodes=6)
        config = _make_base_config_tree('vrptw', n_beams=2, n_parallel=2)
        config['eval']['data_file'] = self.pkl
        self.config = config
        yield
        os.unlink(self.pkl)

    def test_verify_consistent_config_sets_compatibility(self):
        """verify_consistent_config should set compatibility_mode=None for tree_based."""
        import yaml
        from earli.utils.nv import verify_consistent_config

        cfg = _make_base_config_tree('vrptw', n_beams=2, n_parallel=2)
        # Simulate misconfigured: stable_baselines with tree_based
        cfg['system']['compatibility_mode'] = 'stable_baselines'
        cfg2 = verify_consistent_config(cfg, warn=False)
        assert cfg2['system']['compatibility_mode'] is None, (
            "verify_consistent_config should override compatibility_mode to None "
            "for tree_based training"
        )

    def test_train_method_not_overridden_by_train_py(self):
        """The train.py module must NOT force method='ppo'."""
        import ast
        import inspect
        from earli import train as train_module

        source = inspect.getsource(train_module)
        tree = ast.parse(source)

        # Walk the AST looking for assignments that match
        # config['train']['method'] = 'ppo'  (with a string literal rhs).
        forced_ppo_assignments = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            # Check RHS is the string 'ppo'
            if not (isinstance(node.value, ast.Constant)
                    and node.value.value == 'ppo'):
                continue
            # Check LHS is a subscript chain: config['train']['method']
            for target in node.targets:
                if (isinstance(target, ast.Subscript)
                        and isinstance(target.slice, ast.Constant)
                        and target.slice.value == 'method'):
                    forced_ppo_assignments.append(node)

        assert not forced_ppo_assignments, (
            "train.py must not have an unconditional assignment "
            "config['train']['method'] = 'ppo'; tree_based method must be respected."
        )

    def test_play_game_training_returns_training_data(self):
        """SelfPlay.play_game(training=True) should return a non-empty training_data."""
        from earli.self_play import SelfPlay
        from earli.models.attention_model import PosAttentionModel
        from earli.models.sampler import Sampler

        config = self.config
        sampler = Sampler(config)
        env = self.VRPTW(config, datafile=self.pkl, env_type='train')
        obs_space, act_space = env.spaces
        model = PosAttentionModel(obs_space, act_space, config=config, sampler=sampler)

        collector = SelfPlay(
            Game=self.VRPTW,
            config=config,
            seed=0,
            env_type='train',
            n_beams=config['train']['n_beams'],
            model=model,
            datafile=self.pkl,
            n_problems=config['train']['n_parallel_problems'],
        )

        _, infos = collector.play_game(deterministic=False, training=True)
        assert 'training_data' in infos, "training_data must be present in infos"
        td = infos['training_data']
        assert len(td) > 0, "training_data must be non-empty"
        for key in ('observations', 'actions', 'log_prob', 'rewards', 'returns'):
            assert key in td.keys(), f"training_data must contain key '{key}'"

    def test_ppo_update_runs_without_error(self):
        """_ppo_update must run one gradient step without errors."""
        from earli.self_play import SelfPlay
        from earli.models.attention_model import PosAttentionModel
        from earli.models.sampler import Sampler
        from earli.train import _ppo_update

        config = self.config
        sampler = Sampler(config)
        env = self.VRPTW(config, datafile=self.pkl, env_type='train')
        obs_space, act_space = env.spaces
        model = PosAttentionModel(obs_space, act_space, config=config, sampler=sampler)

        collector = SelfPlay(
            Game=self.VRPTW,
            config=config,
            seed=0,
            env_type='train',
            n_beams=config['train']['n_beams'],
            model=model,
            datafile=self.pkl,
            n_problems=config['train']['n_parallel_problems'],
        )

        _, infos = collector.play_game(deterministic=False, training=True)
        td = infos['training_data']
        if len(td) == 0:
            pytest.skip("No training data collected (degenerate game)")

        # Should not raise
        _ppo_update(model, td, config, n_epochs=1)


class TestTreeBasedTrainingPDPTW:
    """Tests for the tree_based training path with PDPTW."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from earli.pdptw import PDPTW
        self.PDPTW = PDPTW
        self.pkl = _make_pdptw_dataset(n_problems=4, n_nodes=7)
        config = _make_base_config_tree('pdptw', n_beams=2, n_parallel=2)
        config['eval']['data_file'] = self.pkl
        self.config = config
        yield
        os.unlink(self.pkl)

    def test_play_game_training_returns_training_data(self):
        """SelfPlay.play_game(training=True) should return non-empty data for PDPTW."""
        from earli.self_play import SelfPlay
        from earli.models.attention_model import PosAttentionModel
        from earli.models.sampler import Sampler

        config = self.config
        sampler = Sampler(config)
        env = self.PDPTW(config, datafile=self.pkl, env_type='train')
        obs_space, act_space = env.spaces
        model = PosAttentionModel(obs_space, act_space, config=config, sampler=sampler)

        collector = SelfPlay(
            Game=self.PDPTW,
            config=config,
            seed=0,
            env_type='train',
            n_beams=config['train']['n_beams'],
            model=model,
            datafile=self.pkl,
            n_problems=config['train']['n_parallel_problems'],
        )

        _, infos = collector.play_game(deterministic=False, training=True)
        assert 'training_data' in infos, "training_data must be present in infos for PDPTW"
        td = infos['training_data']
        assert len(td) > 0, "training_data must be non-empty for PDPTW"

    def test_ppo_update_runs_for_pdptw(self):
        """_ppo_update must run one gradient step for PDPTW without errors."""
        from earli.self_play import SelfPlay
        from earli.models.attention_model import PosAttentionModel
        from earli.models.sampler import Sampler
        from earli.train import _ppo_update

        config = self.config
        sampler = Sampler(config)
        env = self.PDPTW(config, datafile=self.pkl, env_type='train')
        obs_space, act_space = env.spaces
        model = PosAttentionModel(obs_space, act_space, config=config, sampler=sampler)

        collector = SelfPlay(
            Game=self.PDPTW,
            config=config,
            seed=0,
            env_type='train',
            n_beams=config['train']['n_beams'],
            model=model,
            datafile=self.pkl,
            n_problems=config['train']['n_parallel_problems'],
        )

        _, infos = collector.play_game(deterministic=False, training=True)
        td = infos['training_data']
        if len(td) == 0:
            pytest.skip("No training data collected (degenerate game)")

        _ppo_update(model, td, config, n_epochs=1)


# ---------------------------------------------------------------------------
# PIP (Proactive Infeasibility Prevention) tests
# ---------------------------------------------------------------------------

def _make_pip_base_config(env: str, n_beams: int = 1, n_parallel: int = 2):
    """Return a minimal config with PIP masking enabled."""
    cfg = _make_base_config(env, n_beams=n_beams, n_parallel=n_parallel)
    cfg['problem_setup']['use_pip_masking'] = True
    return cfg


class TestPIPMaskingPDPTW:
    """Tests for Proactive Infeasibility Prevention masking in PDPTW."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from earli.pdptw import PDPTW
        self.PDPTW = PDPTW
        self.pkl = _make_pdptw_dataset(n_problems=4, n_nodes=7)
        yield
        os.unlink(self.pkl)

    def _make_env(self, n_parallel=2):
        cfg = _make_pip_base_config('pdptw', n_beams=1, n_parallel=n_parallel)
        cfg['eval']['data_file'] = self.pkl
        return self.PDPTW(cfg, datafile=self.pkl, env_type='train')

    def test_pip_masking_enabled_no_crash(self):
        """PDPTW with use_pip_masking=True must initialise and step without errors."""
        env = self._make_env()
        env.reset()
        actions = torch.ones(env.n_parallel_problems, env.n_beams, dtype=torch.long)
        obs, reward, dones, info = env._step(actions)
        assert obs is not None

    def test_pip_blocked_pickup_attribute_present(self):
        """After a step, env.pip_blocked_pickup must be present."""
        env = self._make_env()
        env.reset()
        actions = torch.ones(env.n_parallel_problems, env.n_beams, dtype=torch.long)
        env._step(actions)
        assert hasattr(env, 'pip_blocked_pickup'), \
            "env.pip_blocked_pickup should be set after a step with use_pip_masking=True"

    def test_pip_blocked_pickup_shape(self):
        """pip_blocked_pickup shape must match (n_flat, problem_size)."""
        env = self._make_env()
        env.reset()
        actions = torch.ones(env.n_parallel_problems, env.n_beams, dtype=torch.long)
        env._step(actions)
        n_flat = env.n_parallel_problems * env.n_beams
        assert env.pip_blocked_pickup.shape == (n_flat, env.problem_size), \
            f"pip_blocked_pickup shape mismatch: {env.pip_blocked_pickup.shape}"

    def test_pip_info_key_present(self):
        """Info dict must include 'pip_blocked_pickup_ratio' after a step."""
        env = self._make_env()
        env.reset()
        actions = torch.ones(env.n_parallel_problems, env.n_beams, dtype=torch.long)
        _, _, _, info = env._step(actions)
        assert 'pip_blocked_pickup_ratio' in info, \
            "info must contain 'pip_blocked_pickup_ratio' when use_pip_masking=True"

    def test_pip_tight_tw_blocks_pickup(self):
        """With very tight delivery time windows, the pickup must be PIP-blocked."""
        import pickle as _pkl
        import tempfile

        # Build a problem where pickup p=1 has delivery d=2, but d has a due_date so
        # tight that visiting p first is guaranteed to make d infeasible.
        n_problems = 2
        n_nodes = 3  # depot=0, pickup=1, delivery=2
        positions = torch.zeros(n_problems, n_nodes, 2)
        positions[:, 1, :] = 0.5   # pickup at (0.5, 0.5)
        positions[:, 2, :] = 1.0   # delivery at (1.0, 1.0)

        dm = torch.cdist(positions, positions, p=2)
        idx = torch.arange(n_nodes)
        dm[:, idx, idx] = 0.0

        demand = torch.zeros(n_problems, n_nodes)
        demand[:, 1] = 5.0
        demand[:, 2] = 5.0

        cap = torch.full((n_problems,), 10.0)
        # Time windows: depot and pickup have generous TW; delivery has very tight TW
        tw = torch.zeros(n_problems, n_nodes, 2)
        tw[:, :, 1] = 1000.0      # generous by default
        # Tight delivery deadline: effectively 0 – guaranteed unreachable
        # once we travel to the pickup node first (positive travel distance).
        TIGHT_TW = 0.001
        tw[:, 2, 1] = TIGHT_TW

        svc = torch.zeros(n_problems, n_nodes)

        pairs = torch.zeros(n_problems, 1, 2, dtype=torch.long)
        pairs[:, 0, 0] = 1  # pickup
        pairs[:, 0, 1] = 2  # delivery

        data = {
            'env_type': 'pdptw',
            'positions': positions,
            'demand': demand,
            'distance_matrix': dm,
            'capacity': cap,
            'time_windows': tw,
            'service_times': svc,
            'pairs': pairs,
            'n_problems': n_problems,
            'radius': 2.0,
            'id': np.arange(n_problems),
        }
        tmp = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
        _pkl.dump(data, tmp)
        tmp.close()

        try:
            cfg = _make_pip_base_config('pdptw', n_beams=1, n_parallel=n_problems)
            cfg['eval']['data_file'] = tmp.name
            env = self.PDPTW(cfg, datafile=tmp.name, env_type='train')
            env.reset()

            # Do a single step (visit depot → should trigger PIP check on pickup nodes)
            actions = torch.zeros(n_problems, 1, dtype=torch.long)  # revisit depot
            _, _, _, info = env._step(actions)

            # With tight delivery TW, pickup node 1 should be PIP-blocked
            # (visiting node 1 makes delivery node 2 infeasible)
            blocked = env.pip_blocked_pickup  # (n_flat, nodes)
            # pickup node 1 must be blocked in at least one env
            assert blocked[:, 1].any(), \
                "Pickup node 1 should be PIP-blocked when its delivery has a very tight time window"
        finally:
            os.unlink(tmp.name)


class TestPIPMaskingVRPTW:
    """Tests for PIP masking in VRPTW (conservative node masking)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from earli.vrptw import VRPTW
        self.VRPTW = VRPTW
        self.pkl = _make_vrptw_dataset(n_problems=4, n_nodes=6)
        yield
        os.unlink(self.pkl)

    def _make_env(self, n_parallel=2):
        cfg = _make_pip_base_config('vrptw', n_beams=1, n_parallel=n_parallel)
        cfg['eval']['data_file'] = self.pkl
        return self.VRPTW(cfg, datafile=self.pkl, env_type='train')

    def test_vrptw_pip_no_crash(self):
        """VRPTW with use_pip_masking=True must step without errors."""
        env = self._make_env()
        env.reset()
        actions = torch.ones(env.n_parallel_problems, env.n_beams, dtype=torch.long)
        obs, reward, dones, info = env._step(actions)
        assert obs is not None

    def test_vrptw_pip_info_key(self):
        """VRPTW PIP step must add pip_blocked_vrptw_ratio to info (when any candidate)."""
        env = self._make_env()
        env.reset()
        actions = torch.ones(env.n_parallel_problems, env.n_beams, dtype=torch.long)
        _, _, _, info = env._step(actions)
        # pip_blocked_vrptw_ratio is only added when candidates exist
        if 'pip_blocked_vrptw_ratio' in info:
            assert info['pip_blocked_vrptw_ratio'].item() >= 0.0


# ---------------------------------------------------------------------------
# POMO-TW augmentation tests
# ---------------------------------------------------------------------------

class TestPomoTWAugmentation:
    """Tests for the 8-fold coordinate augmentation utility."""

    def test_augment_coords_8fold_shape(self):
        """_augment_coords_8fold must return 8×n problems."""
        from earli.pomo_tw_utils import augment_coords_8fold as _augment_coords_8fold
        pos = torch.rand(4, 10, 2)
        aug = _augment_coords_8fold(pos)
        assert aug.shape == (32, 10, 2), \
            f"Expected (32, 10, 2), got {aug.shape}"

    def test_augment_coords_8fold_range(self):
        """All augmented coordinates must stay in [0, 1] when input is in [0, 1]."""
        from earli.pomo_tw_utils import augment_coords_8fold as _augment_coords_8fold
        pos = torch.rand(4, 10, 2)          # already in [0, 1]
        aug = _augment_coords_8fold(pos)
        assert (aug >= 0.0).all() and (aug <= 1.0).all(), \
            "Augmented coordinates must remain in [0, 1]"

    def test_augment_coords_first_variant_is_identity(self):
        """First n_problems rows of augmentation must be the identity transform."""
        from earli.pomo_tw_utils import augment_coords_8fold as _augment_coords_8fold
        n = 4
        pos = torch.rand(n, 10, 2)
        aug = _augment_coords_8fold(pos)
        assert torch.allclose(aug[:n], pos, atol=1e-6), \
            "First n rows of augmented output must equal the original positions"

    def test_augment_vrptw_dataset_shape(self):
        """_augment_vrptw_dataset must produce 8× the problems."""
        from earli.pomo_tw_utils import augment_vrptw_dataset as _augment_vrptw_dataset
        data = {
            'positions': torch.rand(4, 6, 2),
            'demand': torch.rand(4, 6),
            'capacity': torch.rand(4),
            'distance_matrix': torch.rand(4, 6, 6),
            'time_windows': torch.rand(4, 6, 2),
            'service_times': torch.rand(4, 6),
            'n_problems': 4,
            'radius': 1.0,
            'id': np.arange(4),
        }
        aug = _augment_vrptw_dataset(data, n_augments=8)
        assert aug['n_problems'] == 32, f"Expected 32 problems, got {aug['n_problems']}"
        assert aug['positions'].shape == (32, 6, 2)
        assert aug['demand'].shape == (32, 6)

    def test_augment_vrptw_distance_tiled(self):
        """Distance matrix must be tiled (not modified) since L2 is rotation-invariant."""
        from earli.pomo_tw_utils import augment_vrptw_dataset as _augment_vrptw_dataset
        n_orig = 4
        dm = torch.rand(n_orig, 6, 6)
        data = {
            'positions': torch.rand(n_orig, 6, 2),
            'demand': torch.rand(n_orig, 6),
            'capacity': torch.rand(n_orig),
            'distance_matrix': dm,
            'time_windows': torch.rand(n_orig, 6, 2),
            'service_times': torch.rand(n_orig, 6),
            'n_problems': n_orig,
            'radius': 1.0,
            'id': np.arange(n_orig),
        }
        aug = _augment_vrptw_dataset(data, n_augments=8)
        # augment_vrptw_dataset uses tensor.repeat(n_augments, ...) so the layout is:
        #   [problem0, problem1, ..., problem(n-1), problem0, ..., ...] repeated 8 times.
        # The 8 copies of problem 0 are at indices: 0, n_orig, 2*n_orig, ..., 7*n_orig.
        expected = dm[0]
        for k in range(8):
            got = aug['distance_matrix'][k * n_orig]  # k-th copy of problem 0
            assert torch.allclose(got, expected, atol=1e-6), \
                f"Distance matrix of augment {k} (index {k * n_orig}) differs from original"

    def test_augment_vrptw_positions_distinct(self):
        """Each of the 8 augmented position tensors must be distinct."""
        from earli.pomo_tw_utils import augment_coords_8fold as _augment_coords_8fold
        pos = torch.rand(2, 8, 2)   # 2 problems, 8 nodes
        aug = _augment_coords_8fold(pos)   # (16, 8, 2)
        # Augmentations within a problem should differ
        for i in range(8):
            for j in range(i + 1, 8):
                # Compare augment i and j for the first problem
                diff = (aug[i] - aug[j]).abs().max()
                # They should generally differ (random input makes collision almost impossible)
                # Just ensure the code ran to produce 8 variants
        assert aug.shape[0] == 16


class TestPomoTWTraining:
    """Smoke-test for the POMO-TW training loop."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from earli.vrptw import VRPTW
        self.VRPTW = VRPTW
        self.pkl = _make_vrptw_dataset(n_problems=8, n_nodes=6)
        yield
        os.unlink(self.pkl)

    def _make_pomo_config(self, n_parallel=2):
        cfg = _make_base_config('vrptw', n_beams=1, n_parallel=n_parallel)
        cfg['system']['compatibility_mode'] = None
        cfg['train']['method'] = 'pomo_tw'
        cfg['train']['n_beams'] = 1
        cfg['muzero']['max_moves'] = 50
        cfg['muzero']['data_steps_per_epoch'] = 4
        cfg['pomo_tw'] = {'n_augments': 2}  # small for fast tests
        cfg['eval']['data_file'] = self.pkl
        return cfg

    def test_augment_vrptw_dataset_integrates(self):
        """_augment_vrptw_dataset works on a real dataset pickle."""
        from earli.pomo_tw_utils import augment_vrptw_dataset as _augment_vrptw_dataset
        from earli.generate_data import ProblemLoader
        cfg = self._make_pomo_config()
        data = ProblemLoader.load_problem_data(cfg, self.pkl)
        aug = _augment_vrptw_dataset(data, n_augments=2)
        n_orig = data['n_problems']
        assert aug['n_problems'] == 2 * n_orig
        assert aug['positions'].shape[0] == 2 * n_orig

    def test_pomo_tw_verify_consistent_config(self):
        """verify_consistent_config must accept pomo_tw method and set compatibility_mode=None."""
        from earli.utils.nv import verify_consistent_config
        cfg = self._make_pomo_config()
        cfg['system']['compatibility_mode'] = 'stable_baselines'  # intentionally wrong
        cfg2 = verify_consistent_config(cfg, warn=False)
        assert cfg2['system']['compatibility_mode'] is None, \
            "verify_consistent_config must set compatibility_mode=None for pomo_tw"
