#!/usr/bin/env python3
import argparse
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Prefer local workspace package over site-packages for consistent behavior.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from earli import main as earli_main
from earli import test_injection


def subset_problems(problems: dict, indices: np.ndarray) -> dict:
    out = {}
    n_total = None
    if 'distance_matrix' in problems:
        n_total = int(np.asarray(problems['distance_matrix']).shape[0])
    elif 'demand' in problems:
        n_total = int(np.asarray(problems['demand']).shape[0])

    for k, v in problems.items():
        if n_total is None:
            out[k] = v
            continue
        try:
            arr = np.asarray(v)
            if arr.ndim >= 1 and arr.shape[0] == n_total:
                out[k] = arr[indices]
            else:
                out[k] = v
        except Exception:
            out[k] = v

    if 'n_problems' in out:
        out['n_problems'] = len(indices)

    # Some datasets carry non-unique string IDs. Evaluator/test logs index per-problem
    # stats by `id`, so duplicated IDs collapse entries (e.g., 50 samples -> 45 loaded).
    # Reindex sampled subset IDs to guarantee one-to-one mapping with sampled problems.
    out['id'] = np.arange(len(indices), dtype=np.int64)
    return out


def _pick_parallel_problems(n_samples: int, preferred: int = 12) -> int:
    candidates = [preferred, 10, 8, 6, 5, 4, 3, 2, 1]
    for c in candidates:
        if c > 0 and n_samples % c == 0:
            return c
    return 1


def build_infer_cfg(base_cfg: dict, model_path: Path, data_path: Path, n_samples: int) -> dict:
    cfg = dict(base_cfg)
    cfg['train'] = dict(base_cfg.get('train', {}))
    cfg['eval'] = dict(base_cfg.get('eval', {}))
    cfg['system'] = dict(base_cfg.get('system', {}))

    cfg['train']['pretrained_fname'] = str(model_path)
    cfg['eval']['data_file'] = str(data_path)
    cfg['eval']['max_problems'] = None
    cfg['eval']['detailed_test_log'] = True
    cfg['train']['n_parallel_problems'] = _pick_parallel_problems(n_samples)
    cfg['system']['allow_wandb'] = False
    return cfg


def build_injection_cfg(
    base_cfg: dict,
    problems_path: Path,
    solutions_path: Path,
    run_title: str,
    runtime_sec: float,
    n_samples: int,
    fixed_vehicles: bool,
    spare_vehicles: int,
    use_reference_for_num_vehicles: bool,
) -> dict:
    cfg = dict(base_cfg)
    cfg['names'] = dict(base_cfg.get('names', {}))
    cfg['paths'] = dict(base_cfg.get('paths', {}))
    cfg['problem'] = dict(base_cfg.get('problem', {}))
    cfg['cuopt'] = dict(base_cfg.get('cuopt', {}))
    cfg['system'] = dict(base_cfg.get('system', {}))

    cfg['names']['run_title'] = run_title
    cfg['names']['methods'] = ['cuOpt', 'cuOpt_RL']
    cfg['names']['baseline'] = 'cuOpt'
    cfg['names']['main_method'] = 'cuOpt_RL'

    cfg['paths']['problems'] = str(problems_path)
    cfg['paths']['solutions'] = str(solutions_path)
    cfg['paths']['solutions_source'] = None
    cfg['paths']['naive_source'] = None
    cfg['paths']['out_summary'] = 'test_summary_model_vs_cuopt_val'

    cfg['problem']['problems_range'] = [0, n_samples]

    cfg['cuopt']['runtimes'] = [float(runtime_sec)]
    cfg['cuopt']['repetitions'] = 1
    cfg['cuopt']['fixed_vehicles'] = bool(fixed_vehicles)
    cfg['cuopt']['spare_vehicles'] = int(spare_vehicles)
    cfg['cuopt']['use_reference_for_num_vehicles'] = bool(use_reference_for_num_vehicles)
    cfg['injection'] = dict(base_cfg.get('injection', {}))
    cfg['injection']['rl_runtime'] = 0
    cfg['injection']['naive_runtime'] = 0
    cfg['system']['allow_wandb'] = False
    return cfg


def summarize_pair(summary_df: pd.DataFrame) -> dict:
    feas = summary_df[summary_df['success_verification'] == True].copy()

    out = {}
    for method in ['cuOpt', 'cuOpt_RL']:
        mm = feas[feas['method'] == method]
        out[method] = {
            'feasible_count': int(len(mm)),
            'avg_vehicles': float(mm['vehicles'].mean()),
            'avg_distance': float(mm['cost'].mean()),
        }

    if out['cuOpt']['feasible_count'] > 0 and out['cuOpt_RL']['feasible_count'] > 0:
        out['delta_rl_minus_cuopt'] = {
            'vehicles': out['cuOpt_RL']['avg_vehicles'] - out['cuOpt']['avg_vehicles'],
            'distance': out['cuOpt_RL']['avg_distance'] - out['cuOpt']['avg_distance'],
        }
    else:
        out['delta_rl_minus_cuopt'] = {'vehicles': np.nan, 'distance': np.nan}

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare cuOpt vs cuOpt_RL on validate data for n=50,100 with 15s runtime.')
    parser.add_argument('--sizes', default='50,100')
    parser.add_argument('--samples', type=int, default=50)
    parser.add_argument('--runtime', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-dir', default='datasets/vrptw_homberger_like_curriculum')
    parser.add_argument('--model-pattern', default='outputs/hybrid/models/vrptw_model_synth_hybrid_{size}.m')
    parser.add_argument('--infer-config', default='config_infer_vrptw_homberger.yaml')
    parser.add_argument('--inject-config', default='config_initialization_vrptw_homberger.yaml')
    parser.add_argument('--out-csv', default='outputs/model_vs_cuopt_val_15s_summary.csv')
    parser.add_argument(
        '--fixed-vehicles',
        action='store_true',
        help='If set, force fixed vehicle count in cuOpt.',
    )
    parser.add_argument(
        '--spare-vehicles',
        type=int,
        default=0,
        help='Extra vehicles above baseline/lower-bound when not fixed (use 1-2 for harder VRPTW sets).',
    )
    parser.add_argument(
        '--use-reference-vehicles',
        action='store_true',
        help='Use reference baseline_vehicles from dataset when available.',
    )
    args = parser.parse_args()

    root = Path.cwd()
    sizes = [int(x.strip()) for x in args.sizes.split(',') if x.strip()]
    rng = np.random.default_rng(args.seed)

    with open(root / args.infer_config) as f:
        infer_base = yaml.safe_load(f)
    with open(root / args.inject_config) as f:
        inject_base = yaml.safe_load(f)

    out_rows = []

    print(f'[INFO] sizes={sizes}, runtime={args.runtime}s, samples={args.samples}, seed={args.seed}')
    print(
        f"[INFO] fixed_vehicles={args.fixed_vehicles}, spare_vehicles={args.spare_vehicles}, "
        f"use_reference_vehicles={args.use_reference_vehicles}"
    )

    for size in sizes:
        model_path = root / args.model_pattern.format(size=size)
        val_path = root / args.data_dir / f'vrptw_val_{size}.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f'model not found: {model_path}')
        if not val_path.exists():
            raise FileNotFoundError(f'val data not found: {val_path}')

        print(f'\n[RUN] size={size}')
        print(f'[RUN] model={model_path}')
        print(f'[RUN] val={val_path}')

        with open(val_path, 'rb') as f:
            problems = pickle.load(f)

        n_total = int(np.asarray(problems['distance_matrix']).shape[0])
        if n_total < args.samples:
            raise ValueError(f'size={size} has only {n_total} problems, fewer than samples={args.samples}')

        indices = np.sort(rng.choice(n_total, size=args.samples, replace=False))
        subset = subset_problems(problems, indices)

        with tempfile.NamedTemporaryFile('wb', suffix=f'_val_{size}_subset.pkl', delete=False) as pf:
            pickle.dump(subset, pf)
            subset_path = Path(pf.name)

        infer_cfg = build_infer_cfg(
            infer_base,
            model_path=model_path,
            data_path=subset_path,
            n_samples=args.samples,
        )
        with tempfile.NamedTemporaryFile('w', suffix=f'_infer_{size}.yaml', delete=False) as cf:
            yaml.safe_dump(infer_cfg, cf, sort_keys=False)
            infer_cfg_path = Path(cf.name)

        run_title = f'model_vs_cuopt_val{size}_n{args.samples}_{int(args.runtime)}s'
        logs_path = root / 'outputs' / f'test_logs_val_model_{size}.pkl'

        inject_cfg = build_injection_cfg(
            inject_base,
            problems_path=subset_path,
            solutions_path=logs_path,
            run_title=run_title,
            runtime_sec=args.runtime,
            n_samples=args.samples,
            fixed_vehicles=args.fixed_vehicles,
            spare_vehicles=args.spare_vehicles,
            use_reference_for_num_vehicles=args.use_reference_vehicles,
        )
        with tempfile.NamedTemporaryFile('w', suffix=f'_inject_{size}.yaml', delete=False) as jf:
            yaml.safe_dump(inject_cfg, jf, sort_keys=False)
            inject_cfg_path = Path(jf.name)

        try:
            # 1) generate RL solutions from trained model
            earli_main.main(str(infer_cfg_path))
            src_logs = root / 'outputs' / 'test_logs.pkl'
            if not src_logs.exists():
                raise FileNotFoundError(f'missing inference logs: {src_logs}')
            shutil.copy2(src_logs, logs_path)

            # 2) compare cuOpt vs cuOpt_RL with 15s
            test_injection.main(str(inject_cfg_path))
        finally:
            for p in [infer_cfg_path, inject_cfg_path, subset_path]:
                if p.exists():
                    p.unlink()

        summary_path = root / 'outputs' / f'test_summary_model_vs_cuopt_val_{run_title}.pkl'
        if not summary_path.exists():
            # fallback: locate newest matching summary from out_summary + run_title naming.
            candidates = sorted(
                (root / 'outputs').glob(f'test_summary_model_vs_cuopt_val*{run_title}*.pkl'),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                summary_path = candidates[0]
        if not summary_path.exists():
            raise FileNotFoundError(f'missing summary file for size={size}, run_title={run_title}')

        with open(summary_path, 'rb') as f:
            result = pickle.load(f)
        pair = summarize_pair(result['summary'])

        for method in ['cuOpt', 'cuOpt_RL']:
            out_rows.append(
                {
                    'size': size,
                    'method': method,
                    'runtime_sec': args.runtime,
                    'samples': args.samples,
                    'feasible_count': pair[method]['feasible_count'],
                    'avg_vehicles': pair[method]['avg_vehicles'],
                    'avg_distance': pair[method]['avg_distance'],
                    'summary_file': str(summary_path),
                    'rl_logs_file': str(logs_path),
                }
            )

        out_rows.append(
            {
                'size': size,
                'method': 'delta_rl_minus_cuopt',
                'runtime_sec': args.runtime,
                'samples': args.samples,
                'feasible_count': np.nan,
                'avg_vehicles': pair['delta_rl_minus_cuopt']['vehicles'],
                'avg_distance': pair['delta_rl_minus_cuopt']['distance'],
                'summary_file': str(summary_path),
                'rl_logs_file': str(logs_path),
            }
        )

        pd.DataFrame(out_rows).to_csv(root / args.out_csv, index=False)
        print(f"[DONE] size={size} cuOpt_avg_dist={pair['cuOpt']['avg_distance']:.4f} cuOpt_RL_avg_dist={pair['cuOpt_RL']['avg_distance']:.4f}")

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(root / args.out_csv, index=False)
    print(f'\n[FINISH] wrote {root / args.out_csv}')
    print(out_df.to_string(index=False))


if __name__ == '__main__':
    main()
