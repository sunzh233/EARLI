#!/usr/bin/env python3
import argparse
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from earli import test_injection


def _subset_problems_dict(problems: dict, indices: np.ndarray) -> dict:
    subset = {}
    n_total = None
    if 'distance_matrix' in problems:
        n_total = int(np.asarray(problems['distance_matrix']).shape[0])
    elif 'demand' in problems:
        n_total = int(np.asarray(problems['demand']).shape[0])
    elif 'demands' in problems:
        n_total = int(np.asarray(problems['demands']).shape[0])

    for key, value in problems.items():
        if n_total is None:
            subset[key] = value
            continue

        try:
            arr = np.asarray(value)
            if arr.ndim >= 1 and arr.shape[0] == n_total:
                subset[key] = arr[indices]
            else:
                subset[key] = value
        except Exception:
            subset[key] = value

    if 'n_problems' in subset:
        subset['n_problems'] = len(indices)
    return subset


def _build_cfg(
    base_cfg: dict,
    problems_path: str,
    run_title: str,
    runtime_sec: float,
    n_samples: int,
    fixed_vehicles: bool,
    spare_vehicles: int,
) -> dict:
    cfg = dict(base_cfg)
    cfg['names'] = dict(base_cfg.get('names', {}))
    cfg['paths'] = dict(base_cfg.get('paths', {}))
    cfg['problem'] = dict(base_cfg.get('problem', {}))
    cfg['cuopt'] = dict(base_cfg.get('cuopt', {}))
    cfg['system'] = dict(base_cfg.get('system', {}))

    cfg['names']['run_title'] = run_title
    cfg['names']['methods'] = ['cuOpt']
    cfg['names']['baseline'] = 'cuOpt'
    cfg['names']['main_method'] = 'cuOpt'

    cfg['paths']['problems'] = problems_path
    cfg['paths']['solutions'] = None
    cfg['paths']['solutions_source'] = None
    cfg['paths']['naive_source'] = None
    cfg['paths']['out_summary'] = 'test_summary'

    cfg['problem']['problems_range'] = [0, n_samples]

    cfg['cuopt']['runtimes'] = [float(runtime_sec)]
    cfg['cuopt']['repetitions'] = 1
    cfg['cuopt']['fixed_vehicles'] = bool(fixed_vehicles)
    cfg['cuopt']['spare_vehicles'] = int(spare_vehicles)
    cfg['cuopt']['use_reference_for_num_vehicles'] = False
    cfg['system']['allow_wandb'] = False

    return cfg


def _parse_size(path: Path) -> int:
    stem = path.stem
    return int(stem.split('_')[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description='Run cuOpt 15s on sampled validation sets and summarize metrics.')
    parser.add_argument('--data-dir', default='datasets/vrptw_homberger_like_curriculum')
    parser.add_argument('--pattern', default='vrptw_val_*.pkl')
    parser.add_argument('--base-config', default='config_cuopt.yaml')
    parser.add_argument('--runtime', type=float, default=15.0)
    parser.add_argument('--samples', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sizes', default='all', help='Comma-separated sizes, e.g. 200,400,600. Default: all found.')
    parser.add_argument('--output-csv', default='outputs/cuopt_15s_val_summary.csv')
    parser.add_argument(
        '--fixed-vehicles',
        action='store_true',
        help='If set, force fixed vehicle count. Default is NOT fixed, allowing cuOpt to use fewer vehicles.',
    )
    parser.add_argument('--spare-vehicles', type=int, default=0)
    args = parser.parse_args()

    root = Path.cwd()
    data_dir = (root / args.data_dir).resolve()
    out_csv = (root / args.output_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(root / args.base_config) as f:
        base_cfg = yaml.safe_load(f)

    all_files = sorted(data_dir.glob(args.pattern), key=_parse_size)
    if not all_files:
        raise FileNotFoundError(f'No files matched {args.pattern} under {data_dir}')

    if args.sizes.strip().lower() != 'all':
        selected_sizes = {int(x.strip()) for x in args.sizes.split(',') if x.strip()}
        files = [p for p in all_files if _parse_size(p) in selected_sizes]
    else:
        files = all_files

    rng = np.random.default_rng(args.seed)
    rows = []

    print(f'[INFO] data_dir={data_dir}')
    print(f'[INFO] sizes={[ _parse_size(p) for p in files ]}')
    print(f'[INFO] runtime={args.runtime}s, samples_per_size={args.samples}, seed={args.seed}')
    print(f'[INFO] fixed_vehicles={args.fixed_vehicles}, spare_vehicles={args.spare_vehicles}')

    for fpath in files:
        size = _parse_size(fpath)
        print(f'\n[RUN] size={size}, file={fpath.name}')

        with open(fpath, 'rb') as f:
            problems = pickle.load(f)

        if 'distance_matrix' not in problems:
            raise KeyError(f'distance_matrix missing in {fpath}')
        n_total = int(np.asarray(problems['distance_matrix']).shape[0])
        if n_total < args.samples:
            raise ValueError(f'size={size} has only {n_total} samples, fewer than requested {args.samples}')

        sampled = np.sort(rng.choice(n_total, size=args.samples, replace=False))
        subset = _subset_problems_dict(problems, sampled)

        with tempfile.NamedTemporaryFile('wb', suffix=f'_val_{size}_subset.pkl', delete=False) as pf:
            pickle.dump(subset, pf)
            subset_path = Path(pf.name)

        run_title = f'cuopt_{int(args.runtime)}s_val_{size}_n{args.samples}'
        cfg = _build_cfg(
            base_cfg=base_cfg,
            problems_path=str(subset_path),
            run_title=run_title,
            runtime_sec=args.runtime,
            n_samples=args.samples,
            fixed_vehicles=args.fixed_vehicles,
            spare_vehicles=args.spare_vehicles,
        )

        with tempfile.NamedTemporaryFile('w', suffix=f'_cfg_{size}.yaml', delete=False) as cf:
            yaml.safe_dump(cfg, cf, sort_keys=False)
            cfg_path = Path(cf.name)

        try:
            test_injection.main(str(cfg_path))
        finally:
            if cfg_path.exists():
                cfg_path.unlink()
            if subset_path.exists():
                subset_path.unlink()

        summary_path = root / 'outputs' / f'test_summary_{run_title}.pkl'
        if not summary_path.exists():
            raise FileNotFoundError(f'Missing summary output: {summary_path}')

        with open(summary_path, 'rb') as f:
            out = pickle.load(f)
        df = out['summary']
        feasible = df[df['success_verification'] == True].copy()

        mean_vehicles = float(feasible['vehicles'].mean())
        mean_distance = float(feasible['cost'].mean())
        feasible_count = int(len(feasible))

        row = {
            'size': int(size),
            'sampled': int(args.samples),
            'runtime_sec': float(args.runtime),
            'feasible_count': feasible_count,
            'avg_best_vehicles': mean_vehicles,
            'avg_best_distance': mean_distance,
            'summary_file': str(summary_path),
        }
        rows.append(row)

        pd.DataFrame(rows).sort_values('size').to_csv(out_csv, index=False)
        print(
            f"[DONE] size={size} feasible={feasible_count}/{args.samples} "
            f"avg_best_vehicles={mean_vehicles:.4f} avg_best_distance={mean_distance:.4f}"
        )

    res = pd.DataFrame(rows).sort_values('size').reset_index(drop=True)
    res.to_csv(out_csv, index=False)
    print(f'\n[FINISH] wrote summary to {out_csv}')
    print(res.to_string(index=False))


if __name__ == '__main__':
    main()
