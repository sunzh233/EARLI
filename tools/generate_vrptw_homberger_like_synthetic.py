#!/usr/bin/env python3
"""Generate synthetic VRPTW curriculum datasets with Homberger-like marginals.

The generator uses aggregate statistics from Homberger benchmark files
(marginal pools + scale trends), then samples entirely new instances.
No original benchmark instance is copied into the training set.
"""

from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import numpy as np
import torch

from earli.benchmark_parser import convert_homberger_dir, parse_homberger_instance


DEFAULT_SIZES = [50, 100, 150, 200, 300, 400, 500, 600, 800, 1000]
DEFAULT_COUNTS = {
    50: 3000,
    100: 2500,
    150: 2000,
    200: 1500,
    300: 800,
    400: 500,
    500: 360,
    600: 260,
    800: 160,
    1000: 100,
}


def parse_counts(items: list[str]) -> dict[int, int]:
    out: dict[int, int] = {}
    for item in items:
        k, v = item.split(":", 1)
        out[int(k)] = int(v)
    return out


def _save_pkl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _collect_homberger_reference(root: Path, fit_sizes: list[int], max_files_per_size: int) -> dict:
    demand_pool = []
    service_pool = []
    capacity_pool = []
    xy_pool = []
    ready_ratio_pool = []
    width_ratio_pool = []
    depot_due_by_size = {}

    # Per-category pools (e.g., R, C, RC, other)
    cat_pools: dict[str, dict[str, list]] = {}
    # Per-group pools for detailed Homberger prefixes (R1,R2,C1,C2,RC1,RC2)
    group_keys = ['R1', 'R2', 'C1', 'C2', 'RC1', 'RC2']
    group_pools: dict[str, dict[str, list]] = {}

    for size in fit_sizes:
        in_dir = root / f"homberger/homberger_{size}_customer_instances"
        files = sorted(glob.glob(str(in_dir / "*.TXT")))
        if max_files_per_size > 0:
            files = files[:max_files_per_size]
        if not files:
            continue

        depot_dues = []
        for fp in files:
            inst = parse_homberger_instance(fp)
            pos = inst["positions"].astype(np.float32)
            dem = inst["demand"].astype(np.float32)
            tw = inst["time_windows"].astype(np.float32)
            svc = inst["service_times"].astype(np.float32)
            cap = float(inst["capacity"])
            # Infer category and detailed group from instance name
            inst_name = str(inst.get('instance_name', '')).upper()
            # determine coarse category (old behavior)
            if inst_name.startswith('RC'):
                cat = 'RC'
            elif inst_name.startswith('C'):
                cat = 'C'
            elif inst_name.startswith('R'):
                cat = 'R'
            else:
                cat = 'OTHER'

            # determine detailed group (R1,R2,C1,C2,RC1,RC2)
            grp = None
            for g in ['RC1', 'RC2', 'R1', 'R2', 'C1', 'C2']:
                if inst_name.startswith(g):
                    grp = g
                    break
            if grp is None:
                # fallback: try single-letter+digit patterns
                if len(inst_name) >= 2 and inst_name[0] in ('R', 'C') and inst_name[1] in ('1', '2'):
                    grp = inst_name[:2]
                else:
                    grp = 'OTHER'

            if cat not in cat_pools:
                cat_pools[cat] = {
                    'demand_pool': [],
                    'service_pool': [],
                    'capacity_pool': [],
                    'xy_pool': [],
                    'ready_ratio_pool': [],
                    'width_ratio_pool': [],
                    'depot_dues': [],
                }

            if grp not in group_pools:
                group_pools[grp] = {
                    'demand_pool': [],
                    'service_pool': [],
                    'capacity_pool': [],
                    'xy_pool': [],
                    'ready_ratio_pool': [],
                    'width_ratio_pool': [],
                    'depot_dues': [],
                }

            max_xy = float(np.max(pos))
            if max_xy <= 0:
                max_xy = 1.0
            pos_norm = np.clip(pos / max_xy, 0.0, 1.0)
            horizon = float(tw[0, 1])
            if horizon <= 0:
                horizon = max(float(np.max(tw[:, 1])), 1.0)

            ready_ratio = np.clip(tw[1:, 0] / horizon, 0.0, 1.0)
            width_ratio = np.clip((tw[1:, 1] - tw[1:, 0]) / horizon, 0.0, 1.0)

            demand_pool.append(dem[1:])
            service_pool.append(svc[1:])
            capacity_pool.append(cap)
            xy_pool.append(pos_norm)
            ready_ratio_pool.append(ready_ratio)
            width_ratio_pool.append(width_ratio)
            depot_dues.append(float(tw[0, 1]))

            # also accumulate into category-specific pools
            cat_pools[cat]['demand_pool'].append(dem[1:])
            cat_pools[cat]['service_pool'].append(svc[1:])
            cat_pools[cat]['capacity_pool'].append(cap)
            cat_pools[cat]['xy_pool'].append(pos_norm)
            cat_pools[cat]['ready_ratio_pool'].append(ready_ratio)
            cat_pools[cat]['width_ratio_pool'].append(width_ratio)
            cat_pools[cat]['depot_dues'].append(float(tw[0, 1]))

            # accumulate into group-specific pools
            group_pools[grp]['demand_pool'].append(dem[1:])
            group_pools[grp]['service_pool'].append(svc[1:])
            group_pools[grp]['capacity_pool'].append(cap)
            group_pools[grp]['xy_pool'].append(pos_norm)
            group_pools[grp]['ready_ratio_pool'].append(ready_ratio)
            group_pools[grp]['width_ratio_pool'].append(width_ratio)
            group_pools[grp]['depot_dues'].append(float(tw[0, 1]))

        if depot_dues:
            depot_due_by_size[size] = float(np.median(depot_dues))

    if not demand_pool:
        raise RuntimeError("No Homberger reference files found for distribution fitting.")

    fit_x = np.array(sorted(depot_due_by_size.keys()), dtype=np.float32)
    fit_y = np.array([depot_due_by_size[s] for s in fit_x], dtype=np.float32)
    deg = 2 if len(fit_x) >= 3 else 1
    horizon_poly = np.polyfit(fit_x, fit_y, deg=deg)

    # Build per-category refs (compute horizon poly per category if possible)
    by_category: dict[str, dict] = {}
    for cat, pools in cat_pools.items():
        if not pools['demand_pool']:
            continue
        try:
            cat_fit_x = fit_x
            # compute median depot due for sizes present in pools if possible
            # fallback to global horizon poly when insufficient
            cat_depot_dues = np.array(pools['depot_dues'], dtype=np.float32)
            if cat_depot_dues.size >= 3:
                # fit a simple constant poly (deg=0) across available dues as a fallback
                cat_poly = np.polyfit(np.arange(cat_depot_dues.size), cat_depot_dues, deg=1).astype(np.float32)
            else:
                cat_poly = horizon_poly.astype(np.float32)
        except Exception:
            cat_poly = horizon_poly.astype(np.float32)

        by_category[cat] = {
            'demand_pool': np.concatenate(pools['demand_pool']).astype(np.float32),
            'service_pool': np.concatenate(pools['service_pool']).astype(np.float32),
            'capacity_pool': np.array(pools['capacity_pool'], dtype=np.float32),
            'xy_pool': np.concatenate(pools['xy_pool'], axis=0).astype(np.float32),
            'ready_ratio_pool': np.concatenate(pools['ready_ratio_pool']).astype(np.float32),
            'width_ratio_pool': np.concatenate(pools['width_ratio_pool']).astype(np.float32),
            'horizon_poly': cat_poly,
        }

    # Build per-group refs (R1,R2,C1,C2,RC1,RC2 and OTHER)
    by_group: dict[str, dict] = {}
    for grp, pools in group_pools.items():
        if not pools['demand_pool']:
            continue
        try:
            grp_depot_dues = np.array(pools['depot_dues'], dtype=np.float32)
            if grp_depot_dues.size >= 3:
                grp_poly = np.polyfit(np.arange(grp_depot_dues.size), grp_depot_dues, deg=1).astype(np.float32)
            else:
                grp_poly = horizon_poly.astype(np.float32)
        except Exception:
            grp_poly = horizon_poly.astype(np.float32)

        by_group[grp] = {
            'demand_pool': np.concatenate(pools['demand_pool']).astype(np.float32),
            'service_pool': np.concatenate(pools['service_pool']).astype(np.float32),
            'capacity_pool': np.array(pools['capacity_pool'], dtype=np.float32),
            'xy_pool': np.concatenate(pools['xy_pool'], axis=0).astype(np.float32),
            'ready_ratio_pool': np.concatenate(pools['ready_ratio_pool']).astype(np.float32),
            'width_ratio_pool': np.concatenate(pools['width_ratio_pool']).astype(np.float32),
            'horizon_poly': grp_poly,
        }

    return {
        "demand_pool": np.concatenate(demand_pool).astype(np.float32),
        "service_pool": np.concatenate(service_pool).astype(np.float32),
        "capacity_pool": np.array(capacity_pool, dtype=np.float32),
        "xy_pool": np.concatenate(xy_pool, axis=0).astype(np.float32),
        "ready_ratio_pool": np.concatenate(ready_ratio_pool).astype(np.float32),
        "width_ratio_pool": np.concatenate(width_ratio_pool).astype(np.float32),
        "horizon_poly": horizon_poly.astype(np.float32),
        "by_category": by_category,
        "by_group": by_group,
    }


def _sample_horizon(n_customers: int, poly: np.ndarray, rng: np.random.Generator) -> float:
    base = float(np.polyval(poly, float(n_customers)))
    jitter = rng.uniform(0.88, 1.12)
    return max(600.0, base * jitter)


def _sample_capacity(capacity_pool: np.ndarray, rng: np.random.Generator) -> float:
    return float(capacity_pool[rng.integers(0, len(capacity_pool))])


def _sample_instance(n_customers: int, ref: dict, rng: np.random.Generator) -> dict:
    n_nodes = n_customers + 1
    max_coord = float(max(30.0, n_customers / 2.0))
    horizon = _sample_horizon(n_customers, ref["horizon_poly"], rng)

    xy_idx = rng.integers(0, len(ref["xy_pool"]), size=n_nodes)
    positions = ref["xy_pool"][xy_idx].copy() * max_coord

    demand = np.zeros(n_nodes, dtype=np.float32)
    dem_idx = rng.integers(0, len(ref["demand_pool"]), size=n_customers)
    demand[1:] = np.clip(np.round(ref["demand_pool"][dem_idx]), 1.0, 50.0)

    service = np.zeros(n_nodes, dtype=np.float32)
    svc_idx = rng.integers(0, len(ref["service_pool"]), size=n_customers)
    service[1:] = np.clip(np.round(ref["service_pool"][svc_idx]), 1.0, 120.0)

    tw = np.zeros((n_nodes, 2), dtype=np.float32)
    tw[0, 0] = 0.0
    tw[0, 1] = horizon

    rr_idx = rng.integers(0, len(ref["ready_ratio_pool"]), size=n_customers)
    wr_idx = rng.integers(0, len(ref["width_ratio_pool"]), size=n_customers)
    ready_ratio = np.clip(ref["ready_ratio_pool"][rr_idx], 0.0, 0.98)
    width_ratio = np.clip(ref["width_ratio_pool"][wr_idx], 0.01, 1.0)

    ready = np.floor(ready_ratio * horizon)
    width = np.floor(width_ratio * horizon)
    width = np.clip(width, 5.0, horizon)
    due = np.minimum(horizon, ready + width)
    due = np.maximum(due, ready + 5.0)

    tw[1:, 0] = ready.astype(np.float32)
    tw[1:, 1] = due.astype(np.float32)

    return {
        "positions": positions.astype(np.float32),
        "demand": demand,
        "service_times": service,
        "time_windows": tw,
        "capacity": _sample_capacity(ref["capacity_pool"], rng),
    }


def _build_dataset(n_customers: int, n_instances: int, ref: dict, seed: int, chunk_size: int) -> dict:
    rng = np.random.default_rng(seed)
    n_nodes = n_customers + 1

    pos_chunks = []
    dm_chunks = []
    dem_chunks = []
    cap_chunks = []
    tw_chunks = []
    svc_chunks = []
    ids = []

    for start in range(0, n_instances, chunk_size):
        cur = min(chunk_size, n_instances - start)
        positions = np.zeros((cur, n_nodes, 2), dtype=np.float32)
        demand = np.zeros((cur, n_nodes), dtype=np.float32)
        time_windows = np.zeros((cur, n_nodes, 2), dtype=np.float32)
        service_times = np.zeros((cur, n_nodes), dtype=np.float32)
        capacities = np.zeros((cur,), dtype=np.float32)

        for i in range(cur):
            inst = _sample_instance(n_customers, ref, rng)
            positions[i] = inst["positions"]
            demand[i] = inst["demand"]
            time_windows[i] = inst["time_windows"]
            service_times[i] = inst["service_times"]
            capacities[i] = inst["capacity"]
            ids.append(f"synthetic_{n_customers}_{start + i:06d}")

        pos_t = torch.from_numpy(positions)
        dm = torch.cdist(pos_t, pos_t, p=2)
        diag = torch.arange(n_nodes)
        dm[:, diag, diag] = 0.0

        pos_chunks.append(positions)
        dm_chunks.append(dm.numpy().astype(np.float32))
        dem_chunks.append(demand)
        cap_chunks.append(capacities)
        tw_chunks.append(time_windows)
        svc_chunks.append(service_times)

    positions_arr = np.concatenate(pos_chunks, axis=0)
    dm_arr = np.concatenate(dm_chunks, axis=0)
    demand_arr = np.concatenate(dem_chunks, axis=0)
    capacity_arr = np.concatenate(cap_chunks, axis=0)
    tw_arr = np.concatenate(tw_chunks, axis=0)
    svc_arr = np.concatenate(svc_chunks, axis=0)

    return {
        "env_type": "vrptw",
        "positions": torch.from_numpy(positions_arr),
        "demand": torch.from_numpy(demand_arr),
        "distance_matrix": torch.from_numpy(dm_arr),
        "capacity": torch.from_numpy(capacity_arr),
        "time_windows": torch.from_numpy(tw_arr),
        "service_times": torch.from_numpy(svc_arr),
        "n_problems": n_instances,
        "radius": float(np.abs(positions_arr).max()),
        "id": np.array(ids),
    }


def _split_dataset(dataset: dict, train_ratio: float, seed: int) -> tuple[dict, dict]:
    n = int(dataset["positions"].shape[0])
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(round(n * train_ratio))
    n_train = max(1, min(n_train, n - 1))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    def _pick(v, indices):
        if isinstance(v, torch.Tensor) and v.ndim > 0 and int(v.shape[0]) == n:
            return v[indices]
        if isinstance(v, np.ndarray) and v.ndim > 0 and int(v.shape[0]) == n:
            return v[indices]
        if isinstance(v, list) and len(v) == n:
            return [v[i] for i in indices]
        return v

    train = {k: _pick(v, train_idx) for k, v in dataset.items()}
    val = {k: _pick(v, val_idx) for k, v in dataset.items()}
    train["n_problems"] = len(train_idx)
    val["n_problems"] = len(val_idx)
    train["id"] = np.array([f"{x}" for x in train["id"]])
    val["id"] = np.array([f"{x}" for x in val["id"]])
    return train, val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic Homberger-like VRPTW curriculum datasets")
    parser.add_argument("--root", default=".", help="EARLI repository root")
    parser.add_argument("--output-dir", default="datasets/vrptw_homberger_like_curriculum", help="Output dataset directory")
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES, help="Customer sizes to generate")
    parser.add_argument(
        "--counts",
        nargs="*",
        default=[],
        help="Per-size counts as size:count (overrides defaults), e.g. 200:2000 400:800",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio for synthetic datasets")
    parser.add_argument("--seed", type=int, default=1234, help="Base random seed")
    parser.add_argument("--chunk-size", type=int, default=64, help="Batch size used while computing distance matrices")
    parser.add_argument(
        "--fit-sizes",
        nargs="+",
        type=int,
        default=[200, 400, 600, 800, 1000],
        help="Homberger sizes used to fit aggregate distribution statistics",
    )
    parser.add_argument("--max-fit-files-per-size", type=int, default=60, help="Max reference files per Homberger size (0 = all)")
    parser.add_argument("--export-by-category", action="store_true", help="Also export per-category synthetic datasets (R/C/RC/OTHER)")
    parser.add_argument("--group-reduction", type=float, default=0.5, help="Fraction of overall count to generate for each detailed group (0-1)")
    parser.add_argument("--export-homberger-test", action="store_true", help="Also export Homberger test PKLs for 200/400/600/800/1000")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = (root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    custom_counts = parse_counts(args.counts) if args.counts else {}
    counts = dict(DEFAULT_COUNTS)
    counts.update(custom_counts)

    sizes = [int(s) for s in args.sizes]
    missing = [s for s in sizes if s not in counts]
    if missing:
        raise ValueError(f"Missing counts for sizes: {missing}. Provide them via --counts.")

    if not (0.0 < float(args.train_ratio) < 1.0):
        raise ValueError("--train-ratio must be in (0, 1)")

    print("Collecting Homberger aggregate statistics...")
    ref = _collect_homberger_reference(
        root=root,
        fit_sizes=[int(x) for x in args.fit_sizes],
        max_files_per_size=int(args.max_fit_files_per_size),
    )

    manifest = {
        "sizes": sizes,
        "counts": {int(k): int(v) for k, v in counts.items() if int(k) in sizes},
        "train_ratio": float(args.train_ratio),
        "seed": int(args.seed),
    }

    for i, n_customers in enumerate(sizes):
        n_instances = int(counts[n_customers])
        print(f"Generating synthetic n={n_customers}, instances={n_instances} ...")
        dataset = _build_dataset(
            n_customers=n_customers,
            n_instances=n_instances,
            ref=ref,
            seed=int(args.seed) + i,
            chunk_size=int(args.chunk_size),
        )
        train_ds, val_ds = _split_dataset(dataset, float(args.train_ratio), seed=int(args.seed) + 100 + i)

        train_path = out_dir / f"vrptw_train_{n_customers}.pkl"
        val_path = out_dir / f"vrptw_val_{n_customers}.pkl"
        _save_pkl(train_path, train_ds)
        _save_pkl(val_path, val_ds)

        print(f"  train: {train_path} ({train_ds['n_problems']})")
        print(f"  val  : {val_path} ({val_ds['n_problems']})")

    # Optionally export per-group (detailed) synthetic datasets and build mixed files
    if getattr(args, 'export_by_category', False):
        by_group = ref.get('by_group', {}) if isinstance(ref, dict) else {}
        if not by_group:
            print("No per-group pools found; skipping per-group export")
        else:
            print("Exporting per-group synthetic datasets (R1,R2,C1,C2,RC1,RC2,OTHER)...")
            # For each detailed group, generate a reduced number of instances
            for grp, grp_ref in by_group.items():
                grp_dir = out_dir / f"by_group/{grp}"
                grp_dir.mkdir(parents=True, exist_ok=True)
                for i, n_customers in enumerate(sizes):
                    base_count = int(counts[n_customers])
                    n_instances = max(1, int(round(base_count * float(args.group_reduction))))
                    print(f"  [group={grp}] Generating n={n_customers}, instances={n_instances} ...")
                    dataset = _build_dataset(
                        n_customers=n_customers,
                        n_instances=n_instances,
                        ref=grp_ref,
                        seed=int(args.seed) + 1000 + i,
                        chunk_size=int(args.chunk_size),
                    )
                    train_ds, val_ds = _split_dataset(dataset, float(args.train_ratio), seed=int(args.seed) + 2000 + i)
                    train_path = grp_dir / f"vrptw_train_{n_customers}_{grp}.pkl"
                    val_path = grp_dir / f"vrptw_val_{n_customers}_{grp}.pkl"
                    _save_pkl(train_path, train_ds)
                    _save_pkl(val_path, val_ds)
                    print(f"    train: {train_path} ({train_ds['n_problems']})")
                    print(f"    val  : {val_path} ({val_ds['n_problems']})")

            # After exporting per-group datasets, build a single mixed file per size that
            # concatenates: the original global file (if present) + all per-group files.
            print("Building mixed per-size datasets by concatenating per-group files + global file...")
            import pickle
            import torch
            for i, n_customers in enumerate(sizes):
                train_parts = []
                val_parts = []

                # include the original global parts if they exist
                global_train = out_dir / f"vrptw_train_{n_customers}.pkl"
                global_val = out_dir / f"vrptw_val_{n_customers}.pkl"
                if global_train.exists():
                    with global_train.open('rb') as f:
                        train_parts.append(pickle.load(f))
                if global_val.exists():
                    with global_val.open('rb') as f:
                        val_parts.append(pickle.load(f))

                # add each group's train/val parts
                for grp in by_group.keys():
                    grp_train = out_dir / f"by_group/{grp}/vrptw_train_{n_customers}_{grp}.pkl"
                    grp_val = out_dir / f"by_group/{grp}/vrptw_val_{n_customers}_{grp}.pkl"
                    if grp_train.exists():
                        with grp_train.open('rb') as f:
                            train_parts.append(pickle.load(f))
                    if grp_val.exists():
                        with grp_val.open('rb') as f:
                            val_parts.append(pickle.load(f))

                def _concat_parts(parts, seed_base=0):
                    if not parts:
                        return None
                    first = parts[0]
                    out = {}
                    for k, v in first.items():
                        if isinstance(v, torch.Tensor):
                            elems = [p[k] for p in parts if k in p]
                            out[k] = torch.cat(elems, dim=0)
                        elif isinstance(v, (list, tuple)):
                            elems = []
                            for p in parts:
                                if k in p and isinstance(p[k], (list, tuple)):
                                    elems.extend(list(p[k]))
                            out[k] = np.array(elems)
                        elif isinstance(v, np.ndarray):
                            elems = [p[k] for p in parts if k in p]
                            out[k] = np.concatenate(elems, axis=0)
                        else:
                            out[k] = v
                    # set n_problems
                    if 'positions' in out:
                        n_total = int(out['positions'].shape[0])
                    elif 'n_problems' in out:
                        n_total = int(sum(p.get('n_problems', 0) for p in parts))
                    else:
                        n_total = 0
                    out['n_problems'] = n_total
                    # shuffle
                    rng = np.random.default_rng(int(args.seed) + seed_base + int(n_customers))
                    if 'positions' in out:
                        idx = rng.permutation(int(out['positions'].shape[0]))
                        for k in list(out.keys()):
                            v = out[k]
                            try:
                                if isinstance(v, torch.Tensor) and v.ndim > 0 and int(v.shape[0]) == len(idx):
                                    out[k] = v[idx]
                                elif isinstance(v, np.ndarray) and v.ndim > 0 and int(v.shape[0]) == len(idx):
                                    out[k] = v[idx]
                                elif isinstance(v, (list, tuple)) and len(v) == len(idx):
                                    out[k] = [v[j] for j in idx]
                            except Exception:
                                pass
                    return out

                mixed_train = _concat_parts(train_parts, seed_base=5000)
                mixed_val = _concat_parts(val_parts, seed_base=6000)

                if mixed_train is not None:
                    train_path = out_dir / f"vrptw_train_{n_customers}.pkl"
                    _save_pkl(train_path, mixed_train)
                    print(f"  Mixed train saved: {train_path} ({mixed_train['n_problems']})")
                if mixed_val is not None:
                    val_path = out_dir / f"vrptw_val_{n_customers}.pkl"
                    _save_pkl(val_path, mixed_val)
                    print(f"  Mixed val   saved: {val_path} ({mixed_val['n_problems']})")

    if args.export_homberger_test:
        print("Exporting Homberger test datasets for evaluation sizes...")
        for s in [200, 400, 600, 800, 1000]:
            in_dir = root / f"homberger/homberger_{s}_customer_instances"
            if not in_dir.exists():
                print(f"  skip n={s}: {in_dir} not found")
                continue
            out_path = out_dir / f"vrptw_test_homberger_{s}.pkl"
            convert_homberger_dir(str(in_dir), str(out_path), glob_pattern="*.TXT")
            print(f"  test : {out_path}")

    manifest_path = out_dir / "manifest.pkl"
    _save_pkl(manifest_path, manifest)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
