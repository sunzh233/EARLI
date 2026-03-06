#!/usr/bin/env python3
"""Prepare Homberger VRPTW datasets for curriculum train/infer/compare pipeline.

This script converts raw Homberger TXT files to PKL via ``earli.benchmark_parser``
and creates per-size splits for multi-stage training:
- stage train/val from each size in ``train-sizes``
- final test from ``test-size``
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from earli.benchmark_parser import convert_homberger_dir


def _slice_first_dim(value, indices: np.ndarray, total: int):
    """Slice values that have problem dimension on axis 0."""
    if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == total:
        return value[indices]
    if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == total:
        return value[indices]
    if isinstance(value, list) and len(value) == total:
        return [value[i] for i in indices]
    return value


def _subset_dataset(dataset: dict, indices: np.ndarray) -> dict:
    n_total = int(dataset["positions"].shape[0])
    out = {}
    for key, value in dataset.items():
        out[key] = _slice_first_dim(value, indices, n_total)
    out["n_problems"] = len(indices)
    return out


def _load_pkl(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def _save_pkl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Homberger VRPTW curriculum train/val/test PKLs")
    parser.add_argument("--root", default=".", help="EARLI repository root")
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=[200, 400, 600, 800],
        help="Curriculum train sizes (customers), e.g. 200 400 600 800",
    )
    parser.add_argument("--test-size", type=int, default=1000, help="Homberger test size (customers)")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio inside train-size dataset")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for split")
    parser.add_argument(
        "--output-dir",
        default="datasets/homberger_vrptw_curriculum",
        help="Output directory for generated PKLs",
    )
    parser.add_argument("--force-parse", action="store_true", help="Re-run TXT->PKL conversion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root).resolve()
    out_dir = (root / args.output_dir).resolve()
    raw_dir = out_dir / "raw"

    train_sizes = [int(x) for x in args.train_sizes]
    if len(train_sizes) == 0:
        raise ValueError("--train-sizes must contain at least one size")

    for s in train_sizes:
        train_raw_txt = root / f"homberger/homberger_{s}_customer_instances"
        if not train_raw_txt.exists():
            raise FileNotFoundError(f"Train Homberger dir not found: {train_raw_txt}")

    test_raw_txt = root / f"homberger/homberger_{args.test_size}_customer_instances"
    if not test_raw_txt.exists():
        raise FileNotFoundError(f"Test Homberger dir not found: {test_raw_txt}")

    raw_test_pkl = raw_dir / f"vrptw_{args.test_size}.pkl"

    raw_train_pkls = []
    for s in train_sizes:
        train_raw_txt = root / f"homberger/homberger_{s}_customer_instances"
        raw_train_pkl = raw_dir / f"vrptw_{s}.pkl"
        raw_train_pkls.append((s, raw_train_pkl))
        if args.force_parse or not raw_train_pkl.exists():
            convert_homberger_dir(str(train_raw_txt), str(raw_train_pkl), glob_pattern="*.TXT")

    if args.force_parse or not raw_test_pkl.exists():
        convert_homberger_dir(str(test_raw_txt), str(raw_test_pkl), glob_pattern="*.TXT")
    test_data = _load_pkl(raw_test_pkl)

    ratio = float(args.train_ratio)
    if not 0.0 < ratio < 1.0:
        raise ValueError("--train-ratio must be in (0, 1)")

    print("Prepared Homberger VRPTW curriculum datasets:")
    for i_stage, (s, raw_train_pkl) in enumerate(raw_train_pkls):
        train_data = _load_pkl(raw_train_pkl)
        n_train_total = int(train_data["positions"].shape[0])
        if n_train_total < 3:
            raise ValueError(f"Need at least 3 instances for size {s}, got {n_train_total}.")

        rng = np.random.default_rng(args.seed + i_stage)
        indices = np.arange(n_train_total)
        rng.shuffle(indices)

        n_train = int(round(n_train_total * ratio))
        n_train = max(1, min(n_train, n_train_total - 1))

        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_split = _subset_dataset(train_data, train_indices)
        val_split = _subset_dataset(train_data, val_indices)

        out_train = out_dir / f"vrptw_train_{s}.pkl"
        out_val = out_dir / f"vrptw_val_{s}.pkl"
        _save_pkl(out_train, train_split)
        _save_pkl(out_val, val_split)

        print(f"  raw size {s}: {raw_train_pkl}")
        print(f"  train split ({len(train_indices)}): {out_train}")
        print(f"  val split   ({len(val_indices)}): {out_val}")

    test_split = test_data
    out_test = out_dir / f"vrptw_test_{args.test_size}.pkl"
    _save_pkl(out_test, test_split)
    print(f"  test-size raw: {raw_test_pkl}")
    print(f"  test split ({int(test_split['positions'].shape[0])}): {out_test}")


if __name__ == "__main__":
    main()
