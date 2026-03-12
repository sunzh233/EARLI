#!/usr/bin/env bash

CUOPT_LOCAL_LIB="/home/sunzh/cuopt/cpp/build"
CUOPT_DEV_LIB="/home/cslabuser/miniconda3/envs/cuopt_dev_szh/lib"
CUOPT_PREFIX=""
if [[ -d "${CUOPT_LOCAL_LIB}" ]]; then
    CUOPT_PREFIX="${CUOPT_LOCAL_LIB}"
fi
if [[ -d "${CUOPT_DEV_LIB}" ]]; then
    if [[ -n "${CUOPT_PREFIX}" ]]; then
        CUOPT_PREFIX="${CUOPT_PREFIX}:${CUOPT_DEV_LIB}"
    else
        CUOPT_PREFIX="${CUOPT_DEV_LIB}"
    fi
fi
if [[ -n "${CUOPT_PREFIX}" ]]; then
    export LD_LIBRARY_PATH="${CUOPT_PREFIX}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

PY=/home/cslabuser/miniconda3/envs/earli_env/bin/python
$PY - <<'PY'
import os
import pickle
import subprocess
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

root = Path('/home/sunzh/EARLI')
py = '/home/cslabuser/miniconda3/envs/cuopt_dev_szh/bin/python'

dataset = 'datasets/vrptw_homberger_like_curriculum/vrptw_test_homberger_200.pkl'
models = [
    (200, 'outputs/vrptw_model_synth_200.m'),
    (150, 'outputs/vrptw_model_synth_150.m'),
    (50, 'outputs/vrptw_model_synth_50.m'),
    (100, 'outputs/vrptw_model_synth_100.m'),
]

all_rows = []

for size, model in models:
    print(f'===== MODEL n={size} : {model} =====', flush=True)

    infer_cfg_path = Path(f'/tmp/config_infer_h200_m{size}.yaml')
    with open(root / 'config_infer_vrptw_homberger.yaml') as f:
        infer_cfg = yaml.safe_load(f)
    infer_cfg.setdefault('train', {})['pretrained_fname'] = model
    infer_cfg.setdefault('eval', {})['data_file'] = dataset
    infer_cfg['eval']['detailed_test_log'] = True
    with open(infer_cfg_path, 'w') as f:
        yaml.safe_dump(infer_cfg, f, sort_keys=False)

    subprocess.run([py, '-m', 'earli.main', '--config', str(infer_cfg_path)], cwd=str(root), check=True)

    rl_log_path = root / f'outputs/test_logs_h200_model{size}.pkl'
    subprocess.run(['cp', 'outputs/test_logs.pkl', str(rl_log_path)], cwd=str(root), check=True)

    with open(rl_log_path, 'rb') as f:
        rl_logs = pickle.load(f)
    returns = np.asarray(rl_logs.get('returns', []), dtype=float)
    total_games = int(rl_logs.get('total_games', len(returns)))
    # Prefer mean_game_clocktime if available; otherwise fall back to game_clocktime.
    # Note: these fields are already a per-game (per-problem) average, do NOT divide by total_games.
    clock_mean = float(rl_logs.get('mean_game_clocktime', rl_logs.get('game_clocktime', 0.0)) or 0.0)
    rl_runtime = clock_mean
    rl_mean_reward = float(np.mean(returns)) if returns.size else np.nan

    init_cfg_path = Path(f'/tmp/config_init_h200_m{size}_15s.yaml')
    with open(root / 'config_initialization_vrptw_homberger.yaml') as f:
        init_cfg = yaml.safe_load(f)

    init_cfg.setdefault('names', {})['run_title'] = f'm{size}'
    init_cfg['names']['methods'] = ['cuOpt', 'cuOpt_RL']
    init_cfg['names']['baseline'] = 'cuOpt'
    init_cfg['names']['main_method'] = 'cuOpt_RL'

    init_cfg.setdefault('paths', {})['problems'] = dataset
    init_cfg['paths']['solutions'] = str(rl_log_path)
    init_cfg['paths']['out_summary'] = 'test_summary_h200_15s'

    init_cfg.setdefault('cuopt', {})['runtimes'] = [15, 30, 45]
    init_cfg['cuopt']['repetitions'] = 1

    init_cfg.setdefault('injection', {})['rl_runtime'] = float(rl_runtime)
    init_cfg['injection']['log_per_problem'] = False

    with open(init_cfg_path, 'w') as f:
        yaml.safe_dump(init_cfg, f, sort_keys=False)

    subprocess.run([py, '-c', f"from earli import test_injection; test_injection.main(config_path='{init_cfg_path}')"], cwd=str(root), check=True)

    summary_path = root / f'outputs/test_summary_h200_15s_m{size}.pkl'
    with open(summary_path, 'rb') as f:
        res = pickle.load(f)
    df = res['summary']
    success_col = 'success_verification' if 'success_verification' in df.columns else (
        'success2' if 'success2' in df.columns else None
    )

    for method in ['cuOpt', 'cuOpt_RL']:
        sub = df[(df['method'] == method) & (df['total_runtime'] == 15.0)]
        row = {
            'model_size': size,
            'method': method,
            'total_runtime': 15.0,
            'success_rate': float(sub[success_col].mean()) if (len(sub) and success_col is not None) else np.nan,
            'mean_cost': float(sub['cost'].mean()) if len(sub) else np.nan,
            'mean_vehicles': float(sub['vehicles'].mean()) if len(sub) else np.nan,
            'rl_mean_reward': rl_mean_reward,
            'rl_runtime_per_problem_sec': float(rl_runtime),
        }
        all_rows.append(row)

    print(f"[MODEL {size}] RL mean_reward={rl_mean_reward:.4f}, rl_runtime={rl_runtime:.4f}s/problem", flush=True)

out_csv = root / 'outputs/h200_15s_model_compare.csv'
pd.DataFrame(all_rows).to_csv(out_csv, index=False)
print(f'WROTE {out_csv}', flush=True)
PY