#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config_train_vrptw_synthetic_curriculum.yaml}"
DATA_DIR="${DATA_DIR:-datasets/vrptw_homberger_like_curriculum}"
SKIP_DATA=0
SKIP_TRAIN=0
RUN_TEST=0
PROFILE="conservative"
EVAL_DURING_TRAIN=0
INFER_CONFIG="config_infer_vrptw_homberger.yaml"
INIT_CONFIG="config_initialization_vrptw_homberger.yaml"
INITIAL_MODEL=""
USER_SET_STAGE_STEPS=0
USER_SET_STAGE_N_STEPS=0
USER_SET_STAGE_LRS=0
USER_SET_STAGE_EPOCHS=0
USER_SET_STAGE_BATCH_SIZES=0
USER_SET_STAGE_PARALLEL=0
USER_STAGE_STEPS_CSV=""
USER_STAGE_N_STEPS_CSV=""
USER_STAGE_LRS_CSV=""
USER_STAGE_EPOCHS_CSV=""
USER_STAGE_BATCH_SIZES_CSV=""
USER_STAGE_PARALLEL_CSV=""
EVAL_SIZES=(200 400 600 800 1000)
EVAL_RUNTIMES=(10 15 25)

CURRICULUM_STAGES=(50 100 150 200 300 400 500 600 800 1000)
STAGE_STEPS=()
STAGE_N_STEPS=()
STAGE_LRS=()
STAGE_EPOCHS=()
STAGE_BATCH_SIZES=()
STAGE_PARALLEL_PROBLEMS=()
STAGE_LAG_LRS=()
STAGE_PEN_LATE_SUM=()
STAGE_PEN_LATE_COUNT=()
STAGE_PEN_MASKED_RATIO=()
STAGE_TARGET_MASKED_RATIO=()
STAGE_TARGET_KLS=()
STAGE_CLIP_RANGES=()
STAGE_MAX_GRAD_NORMS=()
STAGE_VEHICLE_PENALTIES=()
STAGE_UNUSED_CAP_PENALTIES=()
STAGE_PEN_DEPOT_WITH_CUSTOMER=()
STAGE_TARGET_DEPOT_WITH_CUSTOMER=()
STAGE_PEN_VEHICLE_OVER_LB=()
STAGE_TARGET_VEHICLE_OVER_LB=()
RUN_TEST_DONE_SIZES=()

# Empirical wall-time model (seconds per env step) for 1x RTX 4090,
# used only for ETA display before training starts.
ETA_SEC_PER_STEP=(0.0040 0.0046 0.0054 0.0064 0.0096 0.0120 0.0144 0.0176 0.0240 0.0310)

set_profile_defaults() {
  local p="$1"
  case "$p" in
    conservative)
      STAGE_STEPS=(400000 340000 300000 260000 220000 180000 140000 110000 90000 90000)
      STAGE_N_STEPS=(224 512 640 704 768 768 704 640 576 576)
      STAGE_LRS=(1.5e-5 6.0e-6 4.8e-6 4.2e-6 3.8e-6 3.4e-6 3.0e-6 2.6e-6 2.2e-6 1.9e-6)
      STAGE_EPOCHS=(8 6 6 5 5 4 4 3 3 3)
      # 24G GPU safe defaults: reduce parallel envs and update batch for large n.
      STAGE_BATCH_SIZES=(128 192 192 192 160 128 128 96 64 64)
      STAGE_PARALLEL_PROBLEMS=(12 12 10 8 6 5 4 3 2 1)
      # Lagrangian and constraint-shaping schedule per stage size.
      STAGE_LAG_LRS=(8e-5 7e-5 5.5e-5 4.8e-5 4.2e-5 3.6e-5 3.0e-5 2.6e-5 2.2e-5 2.0e-5)
      STAGE_PEN_LATE_SUM=(0.015 0.018 0.020 0.022 0.024 0.026 0.028 0.030 0.032 0.035)
      STAGE_PEN_LATE_COUNT=(0.060 0.070 0.080 0.090 0.100 0.110 0.120 0.130 0.140 0.150)
      STAGE_PEN_MASKED_RATIO=(0.010 0.012 0.014 0.016 0.018 0.020 0.022 0.024 0.026 0.030)
      STAGE_TARGET_MASKED_RATIO=(0.18 0.17 0.16 0.15 0.14 0.13 0.12 0.11 0.10 0.10)
      STAGE_TARGET_KLS=(none 0.06 0.05 0.05 0.045 0.04 0.04 0.035 0.03 0.03)
      STAGE_CLIP_RANGES=(0.15 0.12 0.12 0.11 0.11 0.10 0.10 0.10 0.10 0.10)
      STAGE_VEHICLE_PENALTIES=(10 11 12 14 16 18 19 20 22 24)
      STAGE_UNUSED_CAP_PENALTIES=(0.20 0.22 0.24 0.27 0.30 0.33 0.36 0.40 0.45 0.50)
      STAGE_PEN_DEPOT_WITH_CUSTOMER=(0.015 0.020 0.025 0.035 0.045 0.055 0.065 0.075 0.085 0.095)
      STAGE_TARGET_DEPOT_WITH_CUSTOMER=(0.12 0.10 0.08 0.06 0.05 0.04 0.03 0.03 0.02 0.02)
      STAGE_PEN_VEHICLE_OVER_LB=(0.08 0.10 0.12 0.16 0.20 0.24 0.28 0.32 0.36 0.40)
      STAGE_TARGET_VEHICLE_OVER_LB=(0.20 0.16 0.14 0.12 0.10 0.08 0.06 0.05 0.04 0.03)
      STAGE_MAX_GRAD_NORMS=(0.40 0.30 0.30 0.28 0.28 0.25 0.25 0.25 0.22 0.22)
      ;;
    aggressive)
      STAGE_STEPS=(600000 520000 430000 340000 260000 210000 170000 140000 110000 90000)
      STAGE_N_STEPS=(256 512 640 768 832 832 768 704 640 576)
      STAGE_LRS=(2.0e-5 9.0e-6 6.8e-6 5.8e-6 5.0e-6 4.3e-6 3.7e-6 3.1e-6 2.5e-6 2.1e-6)
      STAGE_EPOCHS=(10 8 7 6 5 5 4 4 3 3)
      STAGE_BATCH_SIZES=(128 192 192 192 160 128 128 96 64 64)
      STAGE_PARALLEL_PROBLEMS=(12 12 10 8 6 5 4 3 2 1)
      STAGE_LAG_LRS=(1e-4 8e-5 7e-5 6.2e-5 5.4e-5 4.6e-5 3.8e-5 3.3e-5 2.8e-5 2.4e-5)
      STAGE_PEN_LATE_SUM=(0.020 0.022 0.024 0.026 0.028 0.030 0.032 0.035 0.038 0.040)
      STAGE_PEN_LATE_COUNT=(0.080 0.090 0.100 0.110 0.120 0.130 0.140 0.150 0.160 0.170)
      STAGE_PEN_MASKED_RATIO=(0.015 0.017 0.019 0.021 0.023 0.025 0.028 0.030 0.033 0.035)
      STAGE_TARGET_MASKED_RATIO=(0.12 0.10 0.08 0.06 0.05 0.04 0.04 0.03 0.02 0.02)
      STAGE_TARGET_KLS=(none 0.07 0.06 0.055 0.05 0.045 0.04 0.04 0.035 0.035)
      STAGE_CLIP_RANGES=(0.15 0.13 0.12 0.12 0.11 0.11 0.10 0.10 0.10 0.10)
      STAGE_VEHICLE_PENALTIES=(10 12 13 15 17 19 21 23 25 27)
      STAGE_UNUSED_CAP_PENALTIES=(0.20 0.23 0.26 0.30 0.34 0.38 0.42 0.46 0.50 0.55)
      STAGE_PEN_DEPOT_WITH_CUSTOMER=(0.020 0.025 0.030 0.040 0.050 0.060 0.070 0.080 0.090 0.100)
      STAGE_TARGET_DEPOT_WITH_CUSTOMER=(0.10 0.08 0.07 0.05 0.04 0.03 0.02 0.02 0.01 0.01)
      STAGE_PEN_VEHICLE_OVER_LB=(0.10 0.12 0.14 0.18 0.22 0.26 0.30 0.34 0.38 0.42)
      STAGE_TARGET_VEHICLE_OVER_LB=(0.18 0.14 0.12 0.10 0.08 0.06 0.05 0.04 0.03 0.02)
      STAGE_MAX_GRAD_NORMS=(0.40 0.32 0.30 0.30 0.28 0.28 0.26 0.25 0.24 0.24)
      ;;
    *)
      echo "[ERROR] Unknown profile: $p (use conservative or aggressive)"
      exit 1
      ;;
  esac
}

print_eta_table() {
  "$PYTHON_BIN" - <<PY
stages = [int(x) for x in "${CURRICULUM_STAGES[*]}".split()]
steps = [int(x) for x in "${STAGE_STEPS[*]}".split()]
n_steps = [int(x) for x in "${STAGE_N_STEPS[*]}".split()]
lrs = [float(x) for x in "${STAGE_LRS[*]}".split()]
epochs = [int(x) for x in "${STAGE_EPOCHS[*]}".split()]
sec_per_step = [float(x) for x in "${ETA_SEC_PER_STEP[*]}".split()]

print("[Schedule] profile=${PROFILE}, gpu=1x4090")
print("[Schedule] stage | total_steps | n_steps | lr | lag_lr | target_kl | clip | grad | veh_pen | p_depot | p_overlb | tgt_overlb | p_late | p_masked | tgt_masked | epochs | est_minutes")
total_sec = 0.0
lag_lrs = [float(x) for x in "${STAGE_LAG_LRS[*]}".split()]
p_late = [float(x) for x in "${STAGE_PEN_LATE_SUM[*]}".split()]
p_masked = [float(x) for x in "${STAGE_PEN_MASKED_RATIO[*]}".split()]
tgt_masked = [float(x) for x in "${STAGE_TARGET_MASKED_RATIO[*]}".split()]
target_kls = "${STAGE_TARGET_KLS[*]}".split()
clips = [float(x) for x in "${STAGE_CLIP_RANGES[*]}".split()]
grads = [float(x) for x in "${STAGE_MAX_GRAD_NORMS[*]}".split()]
veh_pen = [float(x) for x in "${STAGE_VEHICLE_PENALTIES[*]}".split()]
p_depot = [float(x) for x in "${STAGE_PEN_DEPOT_WITH_CUSTOMER[*]}".split()]
tgt_depot = [float(x) for x in "${STAGE_TARGET_DEPOT_WITH_CUSTOMER[*]}".split()]
p_overlb = [float(x) for x in "${STAGE_PEN_VEHICLE_OVER_LB[*]}".split()]
tgt_overlb = [float(x) for x in "${STAGE_TARGET_VEHICLE_OVER_LB[*]}".split()]
for s, t, ns, lr, llr, tkl, clip, grad, vp, pd, pov, tov, pl, pm, tm, ep, c in zip(stages, steps, n_steps, lrs, lag_lrs, target_kls, clips, grads, veh_pen, p_depot, p_overlb, tgt_overlb, p_late, p_masked, tgt_masked, epochs, sec_per_step):
    est = t * c
    total_sec += est
    print(f"[Schedule] {s:>4} | {t:>10} | {ns:>7} | {lr:>8.2e} | {llr:>8.2e} | {tkl:>9} | {clip:>4.2f} | {grad:>4.2f} | {vp:>7.2f} | {pd:>7.3f} | {pov:>8.3f} | {tov:>10.3f} | {pl:>6.3f} | {pm:>8.3f} | {tm:>10.3f} | {ep:>6} | {est/60:>10.1f}")
print(f"[Schedule] Estimated total: {total_sec/3600:.2f} hours")
print("[Schedule] Note: ETA assumes no contention and includes only RL training, not data generation/test.")
PY
}

contains_value() {
  local needle="$1"
  shift
  for x in "$@"; do
    if [[ "$x" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

ensure_stage_mix_pkls() {
  for size in "${CURRICULUM_STAGES[@]}"; do
    local train_pkl="$DATA_DIR/vrptw_train_${size}.pkl"
    local val_pkl="$DATA_DIR/vrptw_val_${size}.pkl"
    if [[ ! -f "$train_pkl" ]]; then
      echo "[ERROR] Missing mixed train pkl for n=${size}: $train_pkl"
      exit 1
    fi
    if [[ ! -f "$val_pkl" ]]; then
      echo "[ERROR] Missing mixed val pkl for n=${size}: $val_pkl"
      exit 1
    fi
  done
}

summarize_test_log() {
  local log_path="$1"
  local label="${2:-test}"

  if [[ ! -f "$log_path" ]]; then
    echo "[SUMMARY][${label}] Missing log file: $log_path"
    return
  fi

  "$PYTHON_BIN" - <<PY
import pickle
import numpy as np

path = r'''$log_path'''
label = r'''$label'''

with open(path, 'rb') as f:
    logs = pickle.load(f)

returns = np.asarray(logs.get('returns', []), dtype=float)
iters = np.asarray(logs.get('game_iters', []), dtype=float)
total_games = int(logs.get('total_games', len(returns)))
clock = float(logs.get('game_clocktime', 0.0) or 0.0)
problems_path = logs.get('problems_path', 'unknown')

print(f"[SUMMARY][{label}] problems={problems_path}")
print(f"[SUMMARY][{label}] total_games={total_games}, clock_sec={clock:.3f}")

if returns.size > 0:
    print(
        f"[SUMMARY][{label}] returns(mean/std/min/max)="
        f"{returns.mean():.4f}/{returns.std():.4f}/{returns.min():.4f}/{returns.max():.4f}"
    )
else:
    print(f"[SUMMARY][{label}] returns=empty")

if iters.size > 0:
    print(
        f"[SUMMARY][{label}] game_iters(mean/std/min/max)="
        f"{iters.mean():.2f}/{iters.std():.2f}/{iters.min():.0f}/{iters.max():.0f}"
    )
else:
    print(f"[SUMMARY][{label}] game_iters=empty")
PY
}

run_stage_eval() {
  local size="$1"
  local model_path="$2"
  local test_path="$DATA_DIR/vrptw_test_homberger_${size}.pkl"

  if [[ ! -f "$model_path" ]]; then
    echo "[EVAL][WARN] Missing model for n=${size}: $model_path"
    return
  fi
  if [[ ! -f "$test_path" ]]; then
    echo "[EVAL][WARN] Missing test data for n=${size}: $test_path"
    return
  fi
  if [[ ! -f "$INFER_CONFIG" ]]; then
    echo "[EVAL][WARN] Missing infer config: $INFER_CONFIG"
    return
  fi
  if [[ ! -f "$INIT_CONFIG" ]]; then
    echo "[EVAL][WARN] Missing init config: $INIT_CONFIG"
    return
  fi

  local infer_cfg
  infer_cfg="$(mktemp /tmp/config_infer_vrptw_stage_${size}.XXXXXX.yaml)"

  echo "[EVAL][n=${size}] Run RL inference to generate inject solutions"
  "$PYTHON_BIN" - <<PY
import yaml

with open('$INFER_CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg.setdefault('train', {})['pretrained_fname'] = '$model_path'
if $size >= 1000:
    cfg['train']['n_parallel_problems'] = 1
elif $size >= 800:
    cfg['train']['n_parallel_problems'] = 2
elif $size >= 600:
    cfg['train']['n_parallel_problems'] = 3
elif $size >= 400:
    cfg['train']['n_parallel_problems'] = 4
else:
    cfg['train']['n_parallel_problems'] = min(int(cfg['train'].get('n_parallel_problems', 8)), 8)
cfg.setdefault('eval', {})['data_file'] = '$test_path'
cfg['eval']['detailed_test_log'] = True
with open('$infer_cfg', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print('infer_cfg=', '$infer_cfg')
PY

  "$PYTHON_BIN" -m earli.main --config "$infer_cfg"

  local rl_logs
  rl_logs="outputs/test_logs_stage_${size}.pkl"
  if [[ -f "outputs/test_logs.pkl" ]]; then
    cp "outputs/test_logs.pkl" "$rl_logs"
    echo "[EVAL][n=${size}] Saved RL logs: $rl_logs"
    summarize_test_log "$rl_logs" "stage_${size}"
  else
    echo "[EVAL][WARN] Missing outputs/test_logs.pkl after inference"
  fi
}

run_homberger_test_size() {
  local SIZE="$1"
  local MODEL_PATH="outputs/vrptw_model_synth_${SIZE}.m"
  local TEST_PATH="$DATA_DIR/vrptw_test_homberger_${SIZE}.pkl"
  local OUT_LOG="outputs/test_logs_homberger_${SIZE}.pkl"
  local RUN_TEST_LOG_DIR="outputs/run_test_replay"
  local SIZE_LOG="$RUN_TEST_LOG_DIR/size_${SIZE}.log"
  local INFER_CFG_SAVE="$RUN_TEST_LOG_DIR/config_infer_homberger_${SIZE}.yaml"
  local INIT_CFG_SAVE="$RUN_TEST_LOG_DIR/config_init_homberger_${SIZE}.yaml"

  mkdir -p "$RUN_TEST_LOG_DIR"

  {
    echo "===== [RUN-TEST][n=${SIZE}] START $(date '+%F %T') ====="

    if [[ ! -f "$MODEL_PATH" ]]; then
      echo "[WARN] Missing model, skip n=${SIZE}: $MODEL_PATH"
      echo "===== [RUN-TEST][n=${SIZE}] END (SKIPPED) ====="
      exit 0
    fi
    if [[ ! -f "$TEST_PATH" ]]; then
      echo "[WARN] Missing Homberger test pkl, skip n=${SIZE}: $TEST_PATH"
      echo "===== [RUN-TEST][n=${SIZE}] END (SKIPPED) ====="
      exit 0
    fi

    INFER_CFG="$(mktemp /tmp/config_infer_vrptw_synth_${SIZE}.XXXXXX.yaml)"
    "$PYTHON_BIN" - <<PY
import yaml

with open('$INFER_CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg.setdefault('train', {})['pretrained_fname'] = '$MODEL_PATH'
if $SIZE >= 1000:
    cfg['train']['n_parallel_problems'] = 1
elif $SIZE >= 800:
    cfg['train']['n_parallel_problems'] = 2
elif $SIZE >= 600:
    cfg['train']['n_parallel_problems'] = 3
elif $SIZE >= 400:
    cfg['train']['n_parallel_problems'] = 4
else:
    cfg['train']['n_parallel_problems'] = min(int(cfg['train'].get('n_parallel_problems', 8)), 8)
cfg.setdefault('eval', {})['data_file'] = '$TEST_PATH'
cfg['eval']['detailed_test_log'] = True
with open('$INFER_CFG', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print('infer_cfg=', '$INFER_CFG')
PY

    cp "$INFER_CFG" "$INFER_CFG_SAVE"

    echo "[RUN-TEST][n=${SIZE}] Inference begin"
    "$PYTHON_BIN" -m earli.main --config "$INFER_CFG"

    RL_RUNTIME="0"
    if [[ -f "outputs/test_logs.pkl" ]]; then
      cp "outputs/test_logs.pkl" "$OUT_LOG"
      echo "Saved test log: $OUT_LOG"
      summarize_test_log "$OUT_LOG" "homberger_${SIZE}"
      RL_RUNTIME="$($PYTHON_BIN - <<PY
import pickle
import numpy as np
with open('$OUT_LOG', 'rb') as f:
    logs = pickle.load(f)
clock_mean = float(logs.get('mean_game_clocktime', logs.get('game_clocktime', 0.0)) or 0.0)
print(clock_mean)
PY
)"
    else
      echo "[WARN] Missing outputs/test_logs.pkl after inference for n=${SIZE}"
    fi

    INIT_CFG="$(mktemp /tmp/config_init_vrptw_synth_${SIZE}.XXXXXX.yaml)"
    "$PYTHON_BIN" - <<PY
import yaml

with open('$INIT_CONFIG') as f:
    cfg = yaml.safe_load(f)

cfg.setdefault('names', {})['run_title'] = 'homberger_${SIZE}'
cfg['names']['methods'] = ['cuOpt', 'cuOpt_RL']
cfg['names']['baseline'] = 'cuOpt'
cfg['names']['main_method'] = 'cuOpt_RL'

cfg.setdefault('paths', {})['problems'] = '$TEST_PATH'
cfg['paths']['solutions'] = '$OUT_LOG'
cfg['paths']['out_summary'] = 'test_summary_homberger_injection'

cfg.setdefault('cuopt', {})['runtimes'] = [15, 30, 45]
cfg['cuopt']['repetitions'] = 1

cfg.setdefault('injection', {})['rl_runtime'] = float('$RL_RUNTIME')
cfg['injection']['log_per_problem'] = False

with open('$INIT_CFG', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print('init_cfg=', '$INIT_CFG')
PY

    cp "$INIT_CFG" "$INIT_CFG_SAVE"

    echo "[RUN-TEST][n=${SIZE}] Injection comparison begin"
    "$PYTHON_BIN" -c "from earli import test_injection; test_injection.main(config_path='$INIT_CFG')"

    SUMMARY_PKL="outputs/test_summary_homberger_injection_homberger_${SIZE}.pkl"
    if [[ -f "$SUMMARY_PKL" ]]; then
      echo "Saved injection summary: $SUMMARY_PKL"
    else
      echo "[WARN] Expected summary not found: $SUMMARY_PKL"
    fi

    echo "===== [RUN-TEST][n=${SIZE}] END $(date '+%F %T') ====="
  } 2>&1 | tee "$SIZE_LOG"

  echo "[RUN-TEST][n=${SIZE}] Full log saved: $SIZE_LOG"
}

usage() {
  cat <<'EOF'
Usage:
  ./run_full_vrptw_synthetic_curriculum.sh [options]

Options:
  --python BIN             Python executable (default: python)
  --profile NAME           1x4090 preset: conservative|aggressive (default: conservative)
  --train-config PATH      Base train config
  --infer-config PATH      Inference config used during eval (default: config_infer_vrptw_homberger.yaml)
  --init-config PATH       cuOpt injection config used during eval (default: config_initialization_vrptw_homberger.yaml)
  --initial-model PATH     Initial pretrained model for first stage (skip retraining n=50)
  --data-dir PATH          Synthetic data directory
  --skip-data              Skip synthetic data generation
  --skip-train             Skip curriculum training
  --eval-during-train      After stages 200/400/600/800/1000, run eval pipeline automatically
  --eval-sizes L           CSV sizes to evaluate during training (default: 200,400,600,800,1000)
  --eval-runtimes L        CSV cuOpt time budgets in seconds (default: 10,15,25)
  --run-test               Run Homberger tests for saved checkpoints (200/400/600/800/1000)
  --stages L               Comma list, e.g. 50,100,150,200,300,400,500,600,800,1000
  --stage-steps L          Comma list of total steps per stage
  --stage-n-steps L        Comma list of train.n_steps per stage
  --stage-lrs L            Comma list of train.learning_rate per stage (e.g. 1e-5,8e-6,...)
  --stage-epochs L         Comma list of train.epochs per stage
  --stage-batch-sizes L    Comma list of train.batch_size per stage
  --stage-parallel L       Comma list of train.n_parallel_problems per stage
  -h, --help               Show help
EOF
}

parse_csv_to_array() {
  local csv="$1"
  local -n out_arr="$2"
  IFS=',' read -r -a out_arr <<< "$csv"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"; shift 2 ;;
    --profile)
      PROFILE="$2"; shift 2 ;;
    --train-config)
      TRAIN_CONFIG="$2"; shift 2 ;;
    --infer-config)
      INFER_CONFIG="$2"; shift 2 ;;
    --init-config)
      INIT_CONFIG="$2"; shift 2 ;;
    --initial-model)
      INITIAL_MODEL="$2"; shift 2 ;;
    --data-dir)
      DATA_DIR="$2"; shift 2 ;;
    --skip-data)
      SKIP_DATA=1; shift ;;
    --skip-train)
      SKIP_TRAIN=1; shift ;;
    --run-test)
      RUN_TEST=1; shift ;;
    --eval-during-train)
      EVAL_DURING_TRAIN=1; shift ;;
    --eval-sizes)
      parse_csv_to_array "$2" EVAL_SIZES; shift 2 ;;
    --eval-runtimes)
      parse_csv_to_array "$2" EVAL_RUNTIMES; shift 2 ;;
    --stages)
      parse_csv_to_array "$2" CURRICULUM_STAGES; shift 2 ;;
    --stage-steps)
      USER_SET_STAGE_STEPS=1
      USER_STAGE_STEPS_CSV="$2"
      parse_csv_to_array "$2" STAGE_STEPS; shift 2 ;;
    --stage-n-steps)
      USER_SET_STAGE_N_STEPS=1
      USER_STAGE_N_STEPS_CSV="$2"
      parse_csv_to_array "$2" STAGE_N_STEPS; shift 2 ;;
    --stage-lrs)
      USER_SET_STAGE_LRS=1
      USER_STAGE_LRS_CSV="$2"
      parse_csv_to_array "$2" STAGE_LRS; shift 2 ;;
    --stage-epochs)
      USER_SET_STAGE_EPOCHS=1
      USER_STAGE_EPOCHS_CSV="$2"
      parse_csv_to_array "$2" STAGE_EPOCHS; shift 2 ;;
    --stage-batch-sizes)
      USER_SET_STAGE_BATCH_SIZES=1
      USER_STAGE_BATCH_SIZES_CSV="$2"
      parse_csv_to_array "$2" STAGE_BATCH_SIZES; shift 2 ;;
    --stage-parallel)
      USER_SET_STAGE_PARALLEL=1
      USER_STAGE_PARALLEL_CSV="$2"
      parse_csv_to_array "$2" STAGE_PARALLEL_PROBLEMS; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1 ;;
  esac
done

set_profile_defaults "$PROFILE"

if [[ "$USER_SET_STAGE_STEPS" -eq 1 ]]; then
  parse_csv_to_array "$USER_STAGE_STEPS_CSV" STAGE_STEPS
fi
if [[ "$USER_SET_STAGE_N_STEPS" -eq 1 ]]; then
  parse_csv_to_array "$USER_STAGE_N_STEPS_CSV" STAGE_N_STEPS
fi
if [[ "$USER_SET_STAGE_LRS" -eq 1 ]]; then
  parse_csv_to_array "$USER_STAGE_LRS_CSV" STAGE_LRS
fi
if [[ "$USER_SET_STAGE_EPOCHS" -eq 1 ]]; then
  parse_csv_to_array "$USER_STAGE_EPOCHS_CSV" STAGE_EPOCHS
fi
if [[ "$USER_SET_STAGE_BATCH_SIZES" -eq 1 ]]; then
  parse_csv_to_array "$USER_STAGE_BATCH_SIZES_CSV" STAGE_BATCH_SIZES
fi
if [[ "$USER_SET_STAGE_PARALLEL" -eq 1 ]]; then
  parse_csv_to_array "$USER_STAGE_PARALLEL_CSV" STAGE_PARALLEL_PROBLEMS
fi

if [[ "${#CURRICULUM_STAGES[@]}" -ne "${#STAGE_STEPS[@]}" ]]; then
  echo "[ERROR] --stages and --stage-steps length mismatch"
  exit 1
fi
if [[ "${#CURRICULUM_STAGES[@]}" -ne "${#STAGE_N_STEPS[@]}" ]]; then
  echo "[ERROR] --stages and --stage-n-steps length mismatch"
  exit 1
fi
if [[ "${#CURRICULUM_STAGES[@]}" -ne "${#STAGE_LRS[@]}" ]]; then
  echo "[ERROR] --stages and --stage-lrs length mismatch"
  exit 1
fi
if [[ "${#CURRICULUM_STAGES[@]}" -ne "${#STAGE_EPOCHS[@]}" ]]; then
  echo "[ERROR] --stages and --stage-epochs length mismatch"
  exit 1
fi
if [[ "${#CURRICULUM_STAGES[@]}" -ne "${#STAGE_BATCH_SIZES[@]}" ]]; then
  echo "[ERROR] --stages and --stage-batch-sizes length mismatch"
  exit 1
fi
if [[ "${#CURRICULUM_STAGES[@]}" -ne "${#STAGE_PARALLEL_PROBLEMS[@]}" ]]; then
  echo "[ERROR] --stages and --stage-parallel length mismatch"
  exit 1
fi
if [[ "${#CURRICULUM_STAGES[@]}" -ne "${#STAGE_LAG_LRS[@]}" ]]; then
  echo "[ERROR] --stages and STAGE_LAG_LRS length mismatch"
  exit 1
fi
if [[ "${#CURRICULUM_STAGES[@]}" -ne "${#STAGE_PEN_LATE_SUM[@]}" ]]; then
  echo "[ERROR] --stages and STAGE_PEN_LATE_SUM length mismatch"
  exit 1
fi
if [[ "${#CURRICULUM_STAGES[@]}" -ne "${#STAGE_PEN_LATE_COUNT[@]}" ]]; then
  echo "[ERROR] --stages and STAGE_PEN_LATE_COUNT length mismatch"
  exit 1
fi
if [[ "${#CURRICULUM_STAGES[@]}" -ne "${#STAGE_PEN_MASKED_RATIO[@]}" ]]; then
  echo "[ERROR] --stages and STAGE_PEN_MASKED_RATIO length mismatch"
  exit 1
fi
if [[ "${#CURRICULUM_STAGES[@]}" -ne "${#STAGE_TARGET_MASKED_RATIO[@]}" ]]; then
  echo "[ERROR] --stages and STAGE_TARGET_MASKED_RATIO length mismatch"
  exit 1
fi
if [[ "${#CURRICULUM_STAGES[@]}" -ne "${#STAGE_TARGET_KLS[@]}" ]]; then
  echo "[ERROR] --stages and STAGE_TARGET_KLS length mismatch"
  exit 1
fi
if [[ "${#CURRICULUM_STAGES[@]}" -ne "${#STAGE_CLIP_RANGES[@]}" ]]; then
  echo "[ERROR] --stages and STAGE_CLIP_RANGES length mismatch"
  exit 1
fi
if [[ "${#CURRICULUM_STAGES[@]}" -ne "${#STAGE_MAX_GRAD_NORMS[@]}" ]]; then
  echo "[ERROR] --stages and STAGE_MAX_GRAD_NORMS length mismatch"
  exit 1
fi
if [[ "${#CURRICULUM_STAGES[@]}" -ne "${#STAGE_VEHICLE_PENALTIES[@]}" ]]; then
  echo "[ERROR] --stages and STAGE_VEHICLE_PENALTIES length mismatch"
  exit 1
fi
if [[ "${#CURRICULUM_STAGES[@]}" -ne "${#STAGE_UNUSED_CAP_PENALTIES[@]}" ]]; then
  echo "[ERROR] --stages and STAGE_UNUSED_CAP_PENALTIES length mismatch"
  exit 1
fi
if [[ "${#CURRICULUM_STAGES[@]}" -ne "${#STAGE_PEN_DEPOT_WITH_CUSTOMER[@]}" ]]; then
  echo "[ERROR] --stages and STAGE_PEN_DEPOT_WITH_CUSTOMER length mismatch"
  exit 1
fi
if [[ "${#CURRICULUM_STAGES[@]}" -ne "${#STAGE_TARGET_DEPOT_WITH_CUSTOMER[@]}" ]]; then
  echo "[ERROR] --stages and STAGE_TARGET_DEPOT_WITH_CUSTOMER length mismatch"
  exit 1
fi
if [[ "${#CURRICULUM_STAGES[@]}" -ne "${#STAGE_PEN_VEHICLE_OVER_LB[@]}" ]]; then
  echo "[ERROR] --stages and STAGE_PEN_VEHICLE_OVER_LB length mismatch"
  exit 1
fi
if [[ "${#CURRICULUM_STAGES[@]}" -ne "${#STAGE_TARGET_VEHICLE_OVER_LB[@]}" ]]; then
  echo "[ERROR] --stages and STAGE_TARGET_VEHICLE_OVER_LB length mismatch"
  exit 1
fi

if [[ ! -f "$TRAIN_CONFIG" ]]; then
  echo "[ERROR] Missing train config: $TRAIN_CONFIG"
  exit 1
fi

print_eta_table

if [[ "$SKIP_DATA" -eq 0 ]]; then
  echo "[1/3] Generate synthetic Homberger-like curriculum data"
  "$PYTHON_BIN" tools/generate_vrptw_homberger_like_synthetic.py \
    --root "$ROOT_DIR" \
    --output-dir "$DATA_DIR" \
    --sizes "${CURRICULUM_STAGES[@]}" \
    --export-homberger-test \
    --export-by-category
fi

ensure_stage_mix_pkls

echo "[INFO] Using per-size mixed single PKLs for train/val: $DATA_DIR/vrptw_train_<n>.pkl, $DATA_DIR/vrptw_val_<n>.pkl"

if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  echo "[2/3] Curriculum training (single mixed pkl per size)"
  PREV_MODEL="$INITIAL_MODEL"

  if [[ -n "$PREV_MODEL" && ! -f "$PREV_MODEL" ]]; then
    echo "[ERROR] --initial-model file not found: $PREV_MODEL"
    exit 1
  fi

  for i in "${!CURRICULUM_STAGES[@]}"; do
    SIZE="${CURRICULUM_STAGES[$i]}"
    TOTAL_STEPS="${STAGE_STEPS[$i]}"
    N_STEPS="${STAGE_N_STEPS[$i]}"
    STAGE_LR="${STAGE_LRS[$i]}"
    STAGE_EPOCH="${STAGE_EPOCHS[$i]}"
    STAGE_BATCH_SIZE="${STAGE_BATCH_SIZES[$i]}"
    STAGE_PARALLEL="${STAGE_PARALLEL_PROBLEMS[$i]}"
    STAGE_LAG_LR="${STAGE_LAG_LRS[$i]}"
    PEN_LATE_SUM="${STAGE_PEN_LATE_SUM[$i]}"
    PEN_LATE_COUNT="${STAGE_PEN_LATE_COUNT[$i]}"
    PEN_MASKED_RATIO="${STAGE_PEN_MASKED_RATIO[$i]}"
    TARGET_MASKED_RATIO="${STAGE_TARGET_MASKED_RATIO[$i]}"
    STAGE_TARGET_KL="${STAGE_TARGET_KLS[$i]}"
    STAGE_CLIP_RANGE="${STAGE_CLIP_RANGES[$i]}"
    STAGE_MAX_GRAD_NORM="${STAGE_MAX_GRAD_NORMS[$i]}"
    STAGE_VEHICLE_PENALTY="${STAGE_VEHICLE_PENALTIES[$i]}"
    STAGE_UNUSED_CAP_PENALTY="${STAGE_UNUSED_CAP_PENALTIES[$i]}"
    PEN_DEPOT_WITH_CUSTOMER="${STAGE_PEN_DEPOT_WITH_CUSTOMER[$i]}"
    TARGET_DEPOT_WITH_CUSTOMER="${STAGE_TARGET_DEPOT_WITH_CUSTOMER[$i]}"
    PEN_VEHICLE_OVER_LB="${STAGE_PEN_VEHICLE_OVER_LB[$i]}"
    TARGET_VEHICLE_OVER_LB="${STAGE_TARGET_VEHICLE_OVER_LB[$i]}"

    STAGE_CFG="$(mktemp /tmp/config_train_vrptw_synth_stage_${SIZE}.XXXXXX.yaml)"
    MODEL_PATH="outputs/vrptw_model_synth_${SIZE}.m"
    TRAIN_PKL="$DATA_DIR/vrptw_train_${SIZE}.pkl"
    VAL_PKL="$DATA_DIR/vrptw_val_${SIZE}.pkl"

    "$PYTHON_BIN" - <<PY
import yaml

with open('$TRAIN_CONFIG') as f:
    cfg = yaml.safe_load(f)

cfg.setdefault('eval', {})['data_file'] = '$TRAIN_PKL'
cfg['eval']['val_data_file'] = '$VAL_PKL'
cfg['eval']['sampling_mode'] = 'random_with_replacement'
# Force single-file mode so every stage uses one mixed train/val file.
cfg['eval'].pop('data_files', None)

cfg.setdefault('train', {})['save_model_path'] = '$MODEL_PATH'
cfg['train']['pretrained_fname'] = '$PREV_MODEL' if '$PREV_MODEL' else None
cfg['train']['n_parallel_problems'] = int('$STAGE_PARALLEL')
cfg['train']['n_steps'] = int('$N_STEPS')
cfg['train']['learning_rate'] = float('$STAGE_LR')
cfg['train']['epochs'] = int('$STAGE_EPOCH')
cfg['train']['batch_size'] = int('$STAGE_BATCH_SIZE')
cfg['train']['target_kl'] = None if '$STAGE_TARGET_KL' == 'none' else float('$STAGE_TARGET_KL')
cfg['train']['clip_range'] = float('$STAGE_CLIP_RANGE')
cfg['train']['max_grad_norm'] = float('$STAGE_MAX_GRAD_NORM')
cfg['train']['sampling_mode'] = 'random_with_replacement'

ps = cfg.setdefault('problem_setup', {})
ps['use_lagrangian_constraints'] = True
ps['lagrangian_lr'] = float('$STAGE_LAG_LR')
ps['lagrangian_ema'] = float(ps.get('lagrangian_ema', 0.05))
ps['lagrangian_max'] = float(ps.get('lagrangian_max', 10.0))
ps['vehicle_penalty'] = float('$STAGE_VEHICLE_PENALTY')
ps['unused_capacity_penalty'] = float('$STAGE_UNUSED_CAP_PENALTY')
ps['penalty_late_sum'] = float('$PEN_LATE_SUM')
ps['penalty_late_count'] = float('$PEN_LATE_COUNT')
ps['penalty_masked_ratio'] = float('$PEN_MASKED_RATIO')
ps['penalty_pair_blocked'] = float(ps.get('penalty_pair_blocked', 0.0))
ps['penalty_depot_with_customer'] = float('$PEN_DEPOT_WITH_CUSTOMER')
ps['penalty_vehicle_over_lb'] = float('$PEN_VEHICLE_OVER_LB')
ps['target_late_sum'] = float(ps.get('target_late_sum', 0.0))
ps['target_late_count'] = float(ps.get('target_late_count', 0.0))
ps['target_masked_ratio'] = float('$TARGET_MASKED_RATIO')
ps['target_pair_blocked'] = float(ps.get('target_pair_blocked', 0.0))
ps['target_depot_with_customer'] = float('$TARGET_DEPOT_WITH_CUSTOMER')
ps['target_vehicle_over_lb'] = float('$TARGET_VEHICLE_OVER_LB')

with open('$STAGE_CFG', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print('stage_cfg=', '$STAGE_CFG')
print('train_data=', cfg['eval'].get('data_file'))
print('val_data=', cfg['eval'].get('val_data_file'))
print('n_steps=', cfg['train']['n_steps'])
print('n_parallel_problems=', cfg['train']['n_parallel_problems'])
print('learning_rate=', cfg['train']['learning_rate'])
print('epochs=', cfg['train']['epochs'])
print('batch_size=', cfg['train']['batch_size'])
print('target_kl=', cfg['train'].get('target_kl'))
print('clip_range=', cfg['train'].get('clip_range'))
print('ent_coef=', cfg['train'].get('ent_coef'))
print('max_grad_norm=', cfg['train'].get('max_grad_norm'))
print('val_sampling_mode=', cfg['eval'].get('sampling_mode'))
print('lagrangian_lr=', cfg['problem_setup'].get('lagrangian_lr'))
print('vehicle_penalty=', cfg['problem_setup'].get('vehicle_penalty'))
print('unused_capacity_penalty=', cfg['problem_setup'].get('unused_capacity_penalty'))
print('penalty_late_sum=', cfg['problem_setup'].get('penalty_late_sum'))
print('penalty_late_count=', cfg['problem_setup'].get('penalty_late_count'))
print('penalty_masked_ratio=', cfg['problem_setup'].get('penalty_masked_ratio'))
print('penalty_depot_with_customer=', cfg['problem_setup'].get('penalty_depot_with_customer'))
print('penalty_vehicle_over_lb=', cfg['problem_setup'].get('penalty_vehicle_over_lb'))
print('target_masked_ratio=', cfg['problem_setup'].get('target_masked_ratio'))
print('target_depot_with_customer=', cfg['problem_setup'].get('target_depot_with_customer'))
print('target_vehicle_over_lb=', cfg['problem_setup'].get('target_vehicle_over_lb'))
print('save_model=', cfg['train']['save_model_path'])
print('pretrained=', cfg['train']['pretrained_fname'])
PY

  echo "[2/3][n=${SIZE}] total_steps=${TOTAL_STEPS}, n_steps=${N_STEPS}, n_parallel=${STAGE_PARALLEL}, lr=${STAGE_LR}, target_kl=${STAGE_TARGET_KL}, clip=${STAGE_CLIP_RANGE}, grad=${STAGE_MAX_GRAD_NORM}, lag_lr=${STAGE_LAG_LR}, veh_pen=${STAGE_VEHICLE_PENALTY}, p_depot=${PEN_DEPOT_WITH_CUSTOMER}, p_overlb=${PEN_VEHICLE_OVER_LB}, tgt_overlb=${TARGET_VEHICLE_OVER_LB}, p_late=${PEN_LATE_SUM}, p_masked=${PEN_MASKED_RATIO}, target_masked=${TARGET_MASKED_RATIO}, epochs=${STAGE_EPOCH}, batch_size=${STAGE_BATCH_SIZE}, val_sampling=random_with_replacement"

    if [[ -n "$PREV_MODEL" && ! -f "$PREV_MODEL" ]]; then
      echo "[WARN] PREV_MODEL is set but file missing: $PREV_MODEL. Unsetting PREV_MODEL."
      PREV_MODEL=""
    fi

    "$PYTHON_BIN" -m earli.train --config "$STAGE_CFG" --total-steps "$TOTAL_STEPS"

    if [[ -f "$MODEL_PATH" ]]; then
      PREV_MODEL="$MODEL_PATH"
    else
      echo "[WARN] Expected model not found after training n=${SIZE}: $MODEL_PATH. PREV_MODEL remains='${PREV_MODEL}'"
    fi

    if [[ "$EVAL_DURING_TRAIN" -eq 1 ]] && contains_value "$SIZE" "${EVAL_SIZES[@]}"; then
      run_stage_eval "$SIZE" "$MODEL_PATH"
    fi

    if [[ "$RUN_TEST" -eq 1 ]] && contains_value "$SIZE" "${EVAL_SIZES[@]}"; then
      run_homberger_test_size "$SIZE"
      RUN_TEST_DONE_SIZES+=("$SIZE")
    fi
  done
fi

if [[ "$RUN_TEST" -eq 1 ]]; then
  echo "[3/3] Homberger testing for checkpoints: 200/400/600/800/1000"
  for SIZE in 200 400 600 800 1000; do
    if contains_value "$SIZE" "${RUN_TEST_DONE_SIZES[@]}"; then
      echo "[RUN-TEST][n=${SIZE}] already finished during training, skip replay"
      continue
    fi
    run_homberger_test_size "$SIZE"
  done
fi

echo "[DONE] Synthetic curriculum pipeline complete"
