#!/usr/bin/env python3
"""Consistency checks for VRPTW time-window and service-time handling.

Checks performed:
1) Homberger parser alignment between benchmark-style parsing and EARLI parser.
2) Order-level mapping consistency (benchmark uses customers excluding depot).
3) Runtime enforcement in EARLI VRPTW env (reset mask + invalid action correction).
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import tempfile
from typing import Dict, List

import numpy as np
import torch
import yaml

from earli.benchmark_parser import parse_homberger_instance
from earli.vrptw import VRPTW


def parse_homberger_benchmark_style(filepath: str) -> Dict:
    """Parse Homberger file similarly to benchmark script in docs.

    Returns customers list including depot at index 0.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    instance_name = lines[0].strip()
    vehicle_line = lines[4].strip().split()
    num_vehicles = int(vehicle_line[0])
    vehicle_capacity = int(vehicle_line[1])

    customers = []
    for line in lines[9:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 7:
            customers.append(
                {
                    "customer_id": int(parts[0]),
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "demand": float(parts[3]),
                    "ready_time": float(parts[4]),
                    "due_date": float(parts[5]),
                    "service_time": float(parts[6]),
                }
            )

    return {
        "instance_name": instance_name,
        "num_vehicles": num_vehicles,
        "vehicle_capacity": float(vehicle_capacity),
        "customers": customers,
    }


def build_distance_matrix(positions: np.ndarray) -> np.ndarray:
    pos_t = torch.from_numpy(positions).unsqueeze(0)
    dm = torch.cdist(pos_t, pos_t, p=2).squeeze(0).numpy().astype(np.float32)
    n = dm.shape[0]
    dm[np.arange(n), np.arange(n)] = 0.0
    return dm


def check_parser_alignment(file_paths: List[str]) -> List[str]:
    messages = []
    for path in file_paths:
        bench = parse_homberger_benchmark_style(path)
        earli = parse_homberger_instance(path)

        customers = bench["customers"]
        n = len(customers)

        ok = True
        if earli["positions"].shape[0] != n:
            ok = False
            messages.append(f"[FAIL] {os.path.basename(path)}: node count mismatch")
            continue

        # full-node alignment (including depot)
        for i, c in enumerate(customers):
            x, y = float(earli["positions"][i, 0]), float(earli["positions"][i, 1])
            d = float(earli["demand"][i])
            r = float(earli["time_windows"][i, 0])
            du = float(earli["time_windows"][i, 1])
            s = float(earli["service_times"][i])
            if not (
                np.isclose(x, c["x"]) and np.isclose(y, c["y"]) and np.isclose(d, c["demand"])
                and np.isclose(r, c["ready_time"]) and np.isclose(du, c["due_date"]) and np.isclose(s, c["service_time"])
            ):
                ok = False
                messages.append(f"[FAIL] {os.path.basename(path)}: field mismatch at node {i}")
                break

        # benchmark mapping check: orders exclude depot -> should match earli arrays[1:]
        if ok:
            order_ready = np.array([c["ready_time"] for c in customers[1:]], dtype=np.float32)
            order_due = np.array([c["due_date"] for c in customers[1:]], dtype=np.float32)
            order_svc = np.array([c["service_time"] for c in customers[1:]], dtype=np.float32)
            if not (
                np.allclose(order_ready, earli["time_windows"][1:, 0])
                and np.allclose(order_due, earli["time_windows"][1:, 1])
                and np.allclose(order_svc, earli["service_times"][1:])
            ):
                ok = False
                messages.append(f"[FAIL] {os.path.basename(path)}: order(excl depot) mapping mismatch")

        if ok:
            messages.append(f"[PASS] {os.path.basename(path)}: parser + order mapping consistent")

    return messages


def check_runtime_enforcement(sample_path: str, config_path: str) -> str:
    inst = parse_homberger_instance(sample_path)

    positions = inst["positions"].astype(np.float32)
    demand = inst["demand"].astype(np.float32)
    tw = inst["time_windows"].astype(np.float32).copy()
    svc = inst["service_times"].astype(np.float32).copy()
    dm = build_distance_matrix(positions)

    # Force one customer to be unreachable from depot at reset:
    # due_date = 0 while distance from depot is normally > 0.
    blocked_node = 1
    tw[blocked_node, 0] = 0.0
    tw[blocked_node, 1] = 0.0

    cap = np.array([inst["capacity"]], dtype=np.float32)
    radius = float(np.abs(positions).max()) if positions.size else 1.0

    data = {
        "env_type": "vrptw",
        "positions": torch.from_numpy(positions[None, ...]),
        "demand": torch.from_numpy(demand[None, ...]),
        "distance_matrix": torch.from_numpy(dm[None, ...]),
        "capacity": torch.from_numpy(cap),
        "time_windows": torch.from_numpy(tw[None, ...]),
        "service_times": torch.from_numpy(svc[None, ...]),
        "n_problems": 1,
        "radius": radius,
        "id": np.array(["consistency_check"]),
    }

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf:
        tmp_pkl = tf.name

    with open(tmp_pkl, "wb") as f:
        pickle.dump(data, f)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Minimal overrides for a lightweight env check.
    config["problem_setup"]["env"] = "vrptw"
    config["eval"]["data_file"] = tmp_pkl
    config["train"]["n_parallel_problems"] = 1
    config["train"]["n_beams"] = 1
    config["train"]["method"] = "tree_based"
    config["system"]["compatibility_mode"] = None

    env = VRPTW(config, datafile=tmp_pkl, env_type="train")
    env.reset()

    blocked_after_reset = bool(env.feasible_nodes[0, 0, blocked_node].item())

    # Try forcing an invalid action to blocked node; env should correct/override.
    action = torch.tensor([[blocked_node]], dtype=torch.long)
    env.step(action, automatic_reset=False)
    selected_head = int(env.head[0, 0].item())

    os.unlink(tmp_pkl)

    if blocked_after_reset:
        return "[FAIL] Runtime enforcement: blocked node is still feasible after reset"
    if selected_head == blocked_node:
        return "[FAIL] Runtime enforcement: invalid blocked-node action was not corrected"
    return "[PASS] Runtime enforcement: TW mask active at reset and invalid action corrected at step"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check VRPTW TW/service consistency")
    parser.add_argument(
        "--glob",
        type=str,
        default="homberger/homberger_200_customer_instances/*.TXT",
        help="Glob pattern for Homberger instances",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of instances to sample",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_train_vrptw_homberger.yaml",
        help="EARLI config for env runtime check",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"[ERROR] No files matched: {args.glob}")
        return 2

    sample_files = files[: max(1, args.samples)]
    print(f"[INFO] Sampled {len(sample_files)} files")

    parser_msgs = check_parser_alignment(sample_files)
    for m in parser_msgs:
        print(m)

    runtime_msg = check_runtime_enforcement(sample_files[0], args.config)
    print(runtime_msg)

    has_fail = any(msg.startswith("[FAIL]") for msg in parser_msgs) or runtime_msg.startswith("[FAIL]")
    if has_fail:
        print("[SUMMARY] CONSISTENCY CHECK FAILED")
        return 1

    print("[SUMMARY] CONSISTENCY CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
