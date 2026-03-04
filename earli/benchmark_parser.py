# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Parsers for VRPTW and PDPTW benchmark data formats.

Supported formats:
- Homberger benchmark (VRPTW): .TXT files with VEHICLE/CUSTOMER sections
- Li&Lim benchmark (PDPTW): space-separated files with pickup-delivery pairs
"""

import os
import pickle
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Homberger VRPTW parser
# ---------------------------------------------------------------------------

def parse_homberger_instance(file_path: str) -> dict:
    """Parse a single Homberger VRPTW benchmark instance.

    File format::

        <instance_name>

        VEHICLE
        NUMBER     CAPACITY
          <n_vehicles>  <capacity>

        CUSTOMER
        CUST NO.  XCOORD.  YCOORD.  DEMAND  READY TIME  DUE DATE  SERVICE TIME
           0      <x0>     <y0>     0        0          <horizon>   0
           1      <x1>     <y1>    <d1>    <r1>        <d1>        <s1>
           ...

    Args:
        file_path: Path to a Homberger .TXT file.

    Returns:
        dict with keys:
            - positions: np.ndarray (n_nodes, 2)  [depot first]
            - demand: np.ndarray (n_nodes,)
            - capacity: float
            - time_windows: np.ndarray (n_nodes, 2)  [ready_time, due_date]
            - service_times: np.ndarray (n_nodes,)
            - n_vehicles: int
            - instance_name: str
    """
    with open(file_path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]

    instance_name = ''
    n_vehicles = 0
    capacity = 0.0
    customers = []

    section = None
    for line in lines:
        if not line:
            continue
        upper = line.upper()
        if upper.startswith('VEHICLE'):
            section = 'vehicle_header'
            continue
        if upper.startswith('NUMBER') and section == 'vehicle_header':
            section = 'vehicle_data'
            continue
        if upper.startswith('CUSTOMER'):
            section = 'customer_header'
            continue
        if upper.startswith('CUST') and section == 'customer_header':
            section = 'customer_data'
            continue

        if section is None and instance_name == '':
            instance_name = line
            continue

        if section == 'vehicle_data':
            parts = line.split()
            if len(parts) >= 2:
                n_vehicles = int(parts[0])
                capacity = float(parts[1])
            section = 'done_vehicle'
            continue

        if section == 'customer_data':
            parts = line.split()
            if len(parts) >= 7:
                cust_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                demand = float(parts[3])
                ready_time = float(parts[4])
                due_date = float(parts[5])
                service_time = float(parts[6])
                customers.append((cust_id, x, y, demand, ready_time, due_date, service_time))

    customers.sort(key=lambda c: c[0])
    n_nodes = len(customers)

    positions = np.array([[c[1], c[2]] for c in customers], dtype=np.float32)
    demand = np.array([c[3] for c in customers], dtype=np.float32)
    time_windows = np.array([[c[4], c[5]] for c in customers], dtype=np.float32)
    service_times = np.array([c[6] for c in customers], dtype=np.float32)

    return {
        'instance_name': instance_name,
        'n_vehicles': n_vehicles,
        'capacity': capacity,
        'positions': positions,
        'demand': demand,
        'time_windows': time_windows,
        'service_times': service_times,
    }


# ---------------------------------------------------------------------------
# Li&Lim PDPTW parser
# ---------------------------------------------------------------------------

def parse_lilim_instance(file_path: str) -> dict:
    """Parse a single Li&Lim PDPTW benchmark instance.

    File format (first line then one row per node)::

        <n_vehicles>  <max_capacity>  <max_horizon>
        <id> <x> <y> <demand> <ready_time> <due_date> <service_time> <pickup_node> <delivery_node>
        ...

    Depot is node 0 (demand=0, pickup_node=0, delivery_node=0).
    Pickup nodes have demand > 0 and delivery_node > 0.
    Delivery nodes have demand < 0 and pickup_node > 0.

    Args:
        file_path: Path to a Li&Lim .txt file.

    Returns:
        dict with keys:
            - positions: np.ndarray (n_nodes, 2)  [depot first]
            - demand: np.ndarray (n_nodes,)  [unsigned; 0 for depot]
            - capacity: float
            - time_windows: np.ndarray (n_nodes, 2)
            - service_times: np.ndarray (n_nodes,)
            - pairs: np.ndarray (n_pairs, 2) [pickup_idx, delivery_idx]
            - n_vehicles: int
            - instance_name: str
    """
    with open(file_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # First line: n_vehicles capacity max_horizon
    header = lines[0].split()
    n_vehicles = int(header[0])
    capacity = float(header[1])

    nodes = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 9:
            continue
        node_id = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        demand = float(parts[3])
        ready_time = float(parts[4])
        due_date = float(parts[5])
        service_time = float(parts[6])
        pickup_node = int(parts[7])
        delivery_node = int(parts[8])
        nodes.append((node_id, x, y, demand, ready_time, due_date, service_time,
                      pickup_node, delivery_node))

    nodes.sort(key=lambda n: n[0])
    n_nodes = len(nodes)

    positions = np.array([[n[1], n[2]] for n in nodes], dtype=np.float32)
    # Use unsigned demand (pickup has positive, delivery has negative in original)
    demand = np.abs(np.array([n[3] for n in nodes], dtype=np.float32))
    time_windows = np.array([[n[4], n[5]] for n in nodes], dtype=np.float32)
    service_times = np.array([n[6] for n in nodes], dtype=np.float32)

    # Build pickup-delivery pairs
    # For each pickup node (demand > 0 and delivery_node > 0), record (pickup_idx, delivery_idx)
    pairs = []
    for n in nodes:
        node_id, orig_demand, delivery_node_ref = n[0], n[3], n[8]
        if orig_demand > 0 and delivery_node_ref > 0:
            # This is a pickup node; delivery_node_ref is the delivery partner
            pairs.append([node_id, delivery_node_ref])
    pairs = np.array(pairs, dtype=np.int64) if pairs else np.zeros((0, 2), dtype=np.int64)

    instance_name = os.path.splitext(os.path.basename(file_path))[0]

    return {
        'instance_name': instance_name,
        'n_vehicles': n_vehicles,
        'capacity': capacity,
        'positions': positions,
        'demand': demand,
        'time_windows': time_windows,
        'service_times': service_times,
        'pairs': pairs,
    }


# ---------------------------------------------------------------------------
# Dataset conversion utilities
# ---------------------------------------------------------------------------

def _compute_distance_matrix(positions: np.ndarray, p_norm: int = 2) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix from (n_nodes, 2) positions."""
    pos_t = torch.from_numpy(positions).unsqueeze(0)  # (1, n_nodes, 2)
    dm = torch.cdist(pos_t, pos_t, p=p_norm).squeeze(0).numpy()
    n = positions.shape[0]
    dm[np.arange(n), np.arange(n)] = 0.0
    return dm.astype(np.float32)


def _check_uniform_size(all_positions, instance_files):
    """Raise ValueError if instances have different node counts."""
    sizes = [p.shape[0] for p in all_positions]
    if len(set(sizes)) > 1:
        size_map = {}
        for f, s in zip(instance_files, sizes):
            size_map.setdefault(s, []).append(os.path.basename(f))
        detail = '; '.join(f'{s} nodes: {fs[:3]}{"..." if len(fs)>3 else ""}' for s, fs in sorted(size_map.items()))
        raise ValueError(
            f"All instances must have the same number of nodes, but got different counts: {detail}. "
            "Use convert_*_dir_by_size() to auto-group by node count."
        )


def convert_homberger_to_pkl(
        instance_files,
        output_path: str,
        p_norm: int = 2,
) -> str:
    """Convert a list of Homberger VRPTW instance files to a single PKL dataset.

    The PKL file is compatible with ``ProblemLoader.load_problem_data()``.

    Args:
        instance_files: List of paths to Homberger .TXT files.
        output_path: Path for the output .pkl file.
        p_norm: Distance norm (default 2 = Euclidean).

    Returns:
        Path to the written PKL file.
    """
    all_positions = []
    all_demand = []
    all_distance_matrix = []
    all_capacity = []
    all_time_windows = []
    all_service_times = []
    all_ids = []

    for idx, fpath in enumerate(instance_files):
        inst = parse_homberger_instance(fpath)
        dm = _compute_distance_matrix(inst['positions'], p_norm=p_norm)

        all_positions.append(inst['positions'])
        all_demand.append(inst['demand'])
        all_distance_matrix.append(dm)
        all_capacity.append(inst['capacity'])
        all_time_windows.append(inst['time_windows'])
        all_service_times.append(inst['service_times'])
        all_ids.append(inst['instance_name'])

    _check_uniform_size(all_positions, instance_files)

    positions_arr = np.stack(all_positions, axis=0)
    demand_arr = np.stack(all_demand, axis=0)
    dm_arr = np.stack(all_distance_matrix, axis=0)
    capacity_arr = np.array(all_capacity, dtype=np.float32)
    tw_arr = np.stack(all_time_windows, axis=0)
    svc_arr = np.stack(all_service_times, axis=0)

    radius = float(np.abs(positions_arr).max())

    data = {
        'env_type': 'vrptw',
        'positions': torch.from_numpy(positions_arr),
        'demand': torch.from_numpy(demand_arr),
        'distance_matrix': torch.from_numpy(dm_arr),
        'capacity': torch.from_numpy(capacity_arr),
        'time_windows': torch.from_numpy(tw_arr),
        'service_times': torch.from_numpy(svc_arr),
        'n_problems': positions_arr.shape[0],
        'radius': radius,
        'id': np.array(all_ids),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Saved {positions_arr.shape[0]} VRPTW instances to {output_path}")
    return output_path


def convert_lilim_to_pkl(
        instance_files,
        output_path: str,
        p_norm: int = 2,
) -> str:
    """Convert a list of Li&Lim PDPTW instance files to a single PKL dataset.

    The PKL file is compatible with ``ProblemLoader.load_problem_data()``.

    Args:
        instance_files: List of paths to Li&Lim .txt files.
        output_path: Path for the output .pkl file.
        p_norm: Distance norm (default 2 = Euclidean).

    Returns:
        Path to the written PKL file.
    """
    all_positions = []
    all_demand = []
    all_distance_matrix = []
    all_capacity = []
    all_time_windows = []
    all_service_times = []
    all_pairs = []
    all_ids = []

    for idx, fpath in enumerate(instance_files):
        inst = parse_lilim_instance(fpath)
        dm = _compute_distance_matrix(inst['positions'], p_norm=p_norm)

        all_positions.append(inst['positions'])
        all_demand.append(inst['demand'])
        all_distance_matrix.append(dm)
        all_capacity.append(inst['capacity'])
        all_time_windows.append(inst['time_windows'])
        all_service_times.append(inst['service_times'])
        all_pairs.append(inst['pairs'])
        all_ids.append(inst['instance_name'])

    _check_uniform_size(all_positions, instance_files)

    positions_arr = np.stack(all_positions, axis=0)
    demand_arr = np.stack(all_demand, axis=0)
    dm_arr = np.stack(all_distance_matrix, axis=0)
    capacity_arr = np.array(all_capacity, dtype=np.float32)
    tw_arr = np.stack(all_time_windows, axis=0)
    svc_arr = np.stack(all_service_times, axis=0)

    # Pairs may differ in count across instances; pad to max count
    max_pairs = max(p.shape[0] for p in all_pairs) if all_pairs else 0
    pairs_padded = []
    for p in all_pairs:
        pad_rows = max_pairs - p.shape[0]
        if pad_rows > 0:
            p = np.concatenate([p, np.zeros((pad_rows, 2), dtype=np.int64)], axis=0)
        pairs_padded.append(p)
    pairs_arr = np.stack(pairs_padded, axis=0) if pairs_padded else np.zeros((len(all_pairs), 0, 2), dtype=np.int64)

    radius = float(np.abs(positions_arr).max())

    data = {
        'env_type': 'pdptw',
        'positions': torch.from_numpy(positions_arr),
        'demand': torch.from_numpy(demand_arr),
        'distance_matrix': torch.from_numpy(dm_arr),
        'capacity': torch.from_numpy(capacity_arr),
        'time_windows': torch.from_numpy(tw_arr),
        'service_times': torch.from_numpy(svc_arr),
        'pairs': torch.from_numpy(pairs_arr),
        'n_problems': positions_arr.shape[0],
        'radius': radius,
        'id': np.array(all_ids),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Saved {positions_arr.shape[0]} PDPTW instances to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI convenience: convert benchmark directories
# ---------------------------------------------------------------------------

def convert_homberger_dir(benchmark_dir: str, output_path: str, glob_pattern: str = '*.TXT') -> str:
    """Convert all Homberger files in a directory to a PKL dataset.

    All files must contain instances with the same number of nodes.
    Use ``convert_homberger_dir_by_size()`` to automatically group by node count.
    """
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(benchmark_dir, glob_pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {glob_pattern} in {benchmark_dir}")
    return convert_homberger_to_pkl(files, output_path)


def convert_lilim_dir(benchmark_dir: str, output_path: str, glob_pattern: str = '*.txt') -> str:
    """Convert all Li&Lim files in a directory to a PKL dataset.

    All files must contain instances with the same number of nodes.
    Use ``convert_lilim_dir_by_size()`` to automatically group by node count.
    """
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(benchmark_dir, glob_pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {glob_pattern} in {benchmark_dir}")
    return convert_lilim_to_pkl(files, output_path)


def convert_homberger_dir_by_size(
        benchmark_dir: str, output_dir: str, glob_pattern: str = '*.TXT', p_norm: int = 2
) -> list:
    """Convert Homberger files grouped by node count, writing one PKL per size.

    Returns:
        List of output PKL file paths.
    """
    import glob as _glob
    from collections import defaultdict

    files = sorted(_glob.glob(os.path.join(benchmark_dir, glob_pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {glob_pattern} in {benchmark_dir}")

    groups = defaultdict(list)
    for fpath in files:
        inst = parse_homberger_instance(fpath)
        n_nodes = inst['positions'].shape[0]
        groups[n_nodes].append(fpath)

    os.makedirs(output_dir, exist_ok=True)
    outputs = []
    for n_nodes, group_files in sorted(groups.items()):
        out_path = os.path.join(output_dir, f'vrptw_{n_nodes - 1}_nodes.pkl')
        convert_homberger_to_pkl(group_files, out_path, p_norm=p_norm)
        outputs.append(out_path)
    return outputs


def convert_lilim_dir_by_size(
        benchmark_dir: str, output_dir: str, glob_pattern: str = '*.txt', p_norm: int = 2
) -> list:
    """Convert Li&Lim files grouped by node count, writing one PKL per size.

    Returns:
        List of output PKL file paths.
    """
    import glob as _glob
    from collections import defaultdict

    files = sorted(_glob.glob(os.path.join(benchmark_dir, glob_pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {glob_pattern} in {benchmark_dir}")

    groups = defaultdict(list)
    for fpath in files:
        inst = parse_lilim_instance(fpath)
        n_nodes = inst['positions'].shape[0]
        groups[n_nodes].append(fpath)

    os.makedirs(output_dir, exist_ok=True)
    outputs = []
    for n_nodes, group_files in sorted(groups.items()):
        n_customers = n_nodes - 1  # subtract depot
        out_path = os.path.join(output_dir, f'pdptw_{n_customers}_nodes.pkl')
        convert_lilim_to_pkl(group_files, out_path, p_norm=p_norm)
        outputs.append(out_path)
    return outputs


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert VRP benchmark instances to PKL format')
    parser.add_argument('problem_type', choices=['vrptw', 'pdptw'],
                        help='Problem type: vrptw (Homberger) or pdptw (Li&Lim)')
    parser.add_argument('input_dir', help='Directory containing benchmark instance files')
    parser.add_argument('output_path', help='Output .pkl file path (or output directory when --by-size is set)')
    parser.add_argument('--glob', default=None,
                        help='Glob pattern for instance files (default: *.TXT for vrptw, *.txt for pdptw)')
    parser.add_argument('--by-size', action='store_true',
                        help='Auto-group instances by node count and write one PKL per size '
                             '(output_path is treated as an output directory)')
    args = parser.parse_args()

    if args.problem_type == 'vrptw':
        pattern = args.glob or '*.TXT'
        if args.by_size:
            convert_homberger_dir_by_size(args.input_dir, args.output_path, glob_pattern=pattern)
        else:
            convert_homberger_dir(args.input_dir, args.output_path, glob_pattern=pattern)
    else:
        pattern = args.glob or '*.txt'
        if args.by_size:
            convert_lilim_dir_by_size(args.input_dir, args.output_path, glob_pattern=pattern)
        else:
            convert_lilim_dir(args.input_dir, args.output_path, glob_pattern=pattern)
