#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config_train_vrptw_synthetic_curriculum_hybrid.yaml}"
INFER_CONFIG="${INFER_CONFIG:-config_infer_vrptw_homberger_hybrid.yaml}"
INIT_CONFIG="${INIT_CONFIG:-config_initialization_vrptw_homberger_hybrid.yaml}"
DATA_DIR="${DATA_DIR:-datasets/vrptw_homberger_like_curriculum}"
OUT_DIR="${OUT_DIR:-outputs/tree_curriculum}"
MODEL_DIR="$OUT_DIR/models"

SKIP_DATA=0
SKIP_TRAIN=0
RUN_TEST=0
PROFILE="conservative"
INITIAL_MODEL=""

EVAL_SIZES=(200 400 600 800 1000)
EVAL_RUNTIMES=(15 30 45)
RUN_TEST_DONE_SIZES=()

CURRICULUM_STAGES=(50 100 150 200 300 400 500 600 800 1000)
TREE_EPOCHS=()
TREE_BEAMS=()
TREE_DATA_STEPS=()
TREE_BATCH_SIZES=()
TREE_PARALLEL_PROBLEMS=()
TREE_LRS=()

# PIP-style constrained loss schedule (via lagrangian penalties in problem_setup)
PIP_USE_LAGRANGIAN=1
PIP_LAGRANGIAN_LRS=()
PIP_LAGRANGIAN_EMAS=()
PIP_LAGRANGIAN_MAXS=()
PIP_PEN_LATE_SUM=()
PIP_PEN_LATE_COUNT=()
PIP_PEN_MASKED_RATIO=()
PIP_PEN_DEPOT_WITH_CUSTOMER=()
PIP_PEN_VEHICLE_OVER_LB=()
PIP_TARGET_LATE_SUM=()
PIP_TARGET_LATE_COUNT=()
PIP_TARGET_MASKED_RATIO=()
PIP_TARGET_DEPOT_WITH_CUSTOMER=()
PIP_TARGET_VEHICLE_OVER_LB=()

VEHICLE_PENALTIES=()

usage() {
cat <<'EOF'
Usage:
  ./run_full_vrptw_synthetic_curriculum_tree.sh [options]

Options:
  --python BIN              Python executable (default: python)
  --profile NAME            Preset: conservative|aggressive (default: conservative)
  --train-config PATH       Base train config
  --infer-config PATH       Inference config used by run-test
  --init-config PATH        Injection config used by run-test
  --data-dir PATH           Synthetic data directory
  --out-dir PATH            Output root (default: outputs/tree_curriculum)
  --initial-model PATH      Warm-start model for stage 1
  --skip-data               Skip synthetic data generation
  --skip-train              Skip staged training
  --run-test                Run Homberger injection tests
  --stages L                CSV sizes, e.g. 50,100,150,...,1000
  --eval-sizes L            CSV sizes for run-test (default: 200,400,600,800,1000)
  --eval-runtimes L         CSV cuOpt runtimes (default: 15,30,45)

  Tree stage overrides:
  --tree-epochs L           CSV tree train.epochs per stage
  --tree-beams L            CSV tree train.n_beams per stage
  --tree-data-steps L       CSV tree muzero.data_steps_per_epoch per stage
  --tree-batch-sizes L      CSV tree train.batch_size per stage
  --tree-parallel L         CSV tree train.n_parallel_problems per stage
  --tree-lrs L              CSV tree train.learning_rate per stage
  --vehicle-penalties L     CSV problem_setup.vehicle_penalty per stage

  PIP (constraint loss) overrides:
  --pip-use-lagrangian 0|1
  --pip-lagrangian-lrs L
  --pip-lagrangian-emas L
  --pip-lagrangian-maxs L
  --pip-pen-late-sum L
  --pip-pen-late-count L
  --pip-pen-masked-ratio L
  --pip-pen-depot-with-customer L
  --pip-pen-vehicle-over-lb L
  --pip-target-late-sum L
  --pip-target-late-count L
  --pip-target-masked-ratio L
  --pip-target-depot-with-customer L
  --pip-target-vehicle-over-lb L

  -h, --help                Show help
EOF
}

parse_csv_to_array() {
  local csv="$1"
  local -n out_arr="$2"
  IFS=',' read -r -a out_arr <<< "$csv"
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

infer_parallel_by_size() {
  local size="$1"
  if [[ "$size" -ge 1000 ]]; then
    echo 1
  elif [[ "$size" -ge 800 ]]; then
    echo 2
  elif [[ "$size" -ge 600 ]]; then
    echo 3
  elif [[ "$size" -ge 400 ]]; then
    echo 4
  else
    echo 6
  fi
}

set_profile_defaults() {
  local p="$1"
  case "$p" in
    conservative)
      TREE_EPOCHS=(4 4 4 4 3 3 3 2 2 2)
      TREE_BEAMS=(20 20 18 18 16 14 12 10 8 8)
      TREE_DATA_STEPS=(384 384 352 352 320 288 256 224 192 192)
      TREE_BATCH_SIZES=(512 512 512 448 384 320 256 192 128 128)
      TREE_PARALLEL_PROBLEMS=(6 6 5 4 3 3 2 2 1 1)
      TREE_LRS=(1.0e-4 8.5e-5 7.2e-5 6.2e-5 5.2e-5 4.6e-5 4.0e-5 3.5e-5 3.0e-5 2.6e-5)

      VEHICLE_PENALTIES=(10 11 12 14 16 18 20 22 24 26)

      PIP_LAGRANGIAN_LRS=(5.0e-5 5.0e-5 5.0e-5 4.5e-5 4.5e-5 4.0e-5 4.0e-5 3.5e-5 3.0e-5 3.0e-5)
      PIP_LAGRANGIAN_EMAS=(0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05)
      PIP_LAGRANGIAN_MAXS=(10 10 10 10 10 10 10 10 10 10)

      PIP_PEN_LATE_SUM=(0.015 0.018 0.020 0.022 0.024 0.026 0.028 0.030 0.032 0.034)
      PIP_PEN_LATE_COUNT=(0.08 0.09 0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17)
      PIP_PEN_MASKED_RATIO=(0.015 0.018 0.020 0.022 0.024 0.026 0.028 0.030 0.032 0.034)
      PIP_PEN_DEPOT_WITH_CUSTOMER=(0.015 0.020 0.025 0.030 0.035 0.040 0.045 0.050 0.055 0.060)
      PIP_PEN_VEHICLE_OVER_LB=(0.08 0.10 0.12 0.14 0.16 0.18 0.20 0.22 0.24 0.26)

      PIP_TARGET_LATE_SUM=(0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)
      PIP_TARGET_LATE_COUNT=(0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)
      PIP_TARGET_MASKED_RATIO=(0.12 0.11 0.10 0.09 0.08 0.07 0.06 0.05 0.04 0.03)
      PIP_TARGET_DEPOT_WITH_CUSTOMER=(0.05 0.05 0.04 0.04 0.03 0.03 0.02 0.02 0.01 0.01)
      PIP_TARGET_VEHICLE_OVER_LB=(0.16 0.14 0.12 0.10 0.09 0.08 0.07 0.06 0.05 0.04)
      ;;
    aggressive)
      TREE_EPOCHS=(5 5 5 4 4 3 3 3 2 2)
      TREE_BEAMS=(24 24 22 20 18 16 14 12 10 8)
      TREE_DATA_STEPS=(448 448 416 384 352 320 288 256 224 192)
      TREE_BATCH_SIZES=(512 512 512 512 448 384 320 256 192 128)
      TREE_PARALLEL_PROBLEMS=(6 6 5 4 3 3 2 2 1 1)
      TREE_LRS=(1.2e-4 1.0e-4 8.8e-5 7.6e-5 6.4e-5 5.4e-5 4.8e-5 4.2e-5 3.4e-5 2.8e-5)

      VEHICLE_PENALTIES=(10 12 14 16 18 20 22 24 26 28)

      PIP_LAGRANGIAN_LRS=(6.0e-5 6.0e-5 5.5e-5 5.0e-5 5.0e-5 4.5e-5 4.0e-5 3.5e-5 3.2e-5 3.0e-5)
      PIP_LAGRANGIAN_EMAS=(0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05)
      PIP_LAGRANGIAN_MAXS=(10 10 10 10 10 10 10 10 10 10)

      PIP_PEN_LATE_SUM=(0.02 0.022 0.024 0.026 0.028 0.030 0.032 0.034 0.036 0.038)
      PIP_PEN_LATE_COUNT=(0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19)
      PIP_PEN_MASKED_RATIO=(0.02 0.022 0.024 0.026 0.028 0.030 0.032 0.034 0.036 0.038)
      PIP_PEN_DEPOT_WITH_CUSTOMER=(0.02 0.025 0.03 0.035 0.04 0.045 0.05 0.055 0.06 0.065)
      PIP_PEN_VEHICLE_OVER_LB=(0.10 0.12 0.14 0.16 0.18 0.20 0.22 0.24 0.26 0.28)

      PIP_TARGET_LATE_SUM=(0 0 0 0 0 0 0 0 0 0)
      PIP_TARGET_LATE_COUNT=(0 0 0 0 0 0 0 0 0 0)
      PIP_TARGET_MASKED_RATIO=(0.12 0.11 0.10 0.09 0.08 0.07 0.06 0.05 0.04 0.03)
      PIP_TARGET_DEPOT_WITH_CUSTOMER=(0.08 0.07 0.06 0.05 0.04 0.03 0.02 0.02 0.01 0.01)
      PIP_TARGET_VEHICLE_OVER_LB=(0.18 0.16 0.14 0.12 0.10 0.08 0.06 0.05 0.04 0.03)
      ;;
    *)
      echo "[ERROR] Unknown profile: $p (use conservative or aggressive)"
      exit 1
      ;;
  esac
}

check_len() {
  local name="$1"
  local arr_len="$2"
  if [[ "${#CURRICULUM_STAGES[@]}" -ne "$arr_len" ]]; then
    echo "[ERROR] --stages and ${name} length mismatch"
    exit 1
  fi
}

validate_int_array_positive() {
  local name="$1"
  shift
  local values=("$@")
  "$PYTHON_BIN" - "$name" "${values[@]}" <<'PY'
import sys
name = sys.argv[1]
vals = sys.argv[2:]
for v in vals:
    try:
        x = int(v)
    except Exception:
        raise SystemExit(f"[ERROR] {name} has non-integer value: {v}")
    if x <= 0:
        raise SystemExit(f"[ERROR] {name} requires > 0, got: {x}")
print(f"[CHECK] {name}: ok ({len(vals)} values)")
PY
}

validate_float_array_range() {
  local name="$1"
  local min_v="$2"
  local max_v="$3"
  shift 3
  local values=("$@")
  "$PYTHON_BIN" - "$name" "$min_v" "$max_v" "${values[@]}" <<'PY'
import sys
name = sys.argv[1]
mn = float(sys.argv[2])
mx = float(sys.argv[3])
vals = sys.argv[4:]
for v in vals:
    try:
        x = float(v)
    except Exception:
        raise SystemExit(f"[ERROR] {name} has non-float value: {v}")
    if x < mn or x > mx:
        raise SystemExit(f"[ERROR] {name} expects [{mn}, {mx}], got: {x}")
print(f"[CHECK] {name}: ok ({len(vals)} values)")
PY
}

ensure_stage_mix_pkls() {
  for size in "${CURRICULUM_STAGES[@]}"; do
    local train_pkl="$DATA_DIR/vrptw_train_${size}.pkl"
    local val_pkl="$DATA_DIR/vrptw_val_${size}.pkl"
    if [[ ! -f "$train_pkl" ]]; then
      echo "[ERROR] Missing train pkl for n=${size}: $train_pkl"
      exit 1
    fi
    if [[ ! -f "$val_pkl" ]]; then
      echo "[ERROR] Missing val pkl for n=${size}: $val_pkl"
      exit 1
    fi
  done
}

append_stage_log() {
  local stage="$1"
  local model_path="$2"
  local status="$3"
  local tree_cfg="$4"
  local stage_log_csv="$OUT_DIR/tree_stage_log.csv"

  "$PYTHON_BIN" - <<PY
import csv
import os
from datetime import datetime

path = r'''$stage_log_csv'''
is_new = not os.path.exists(path)
with open(path, 'a', newline='') as f:
    w = csv.writer(f)
    if is_new:
        w.writerow([
            'timestamp', 'stage', 'status', 'model_path', 'tree_cfg'
        ])
    w.writerow([
        datetime.now().isoformat(timespec='seconds'),
        '$stage', '$status', r'''$model_path''', r'''$tree_cfg'''
    ])
print('[LOG] appended stage row ->', path)
PY
}

summarize_test_log() {
  local log_path="$1"
  local label="$2"
  if [[ ! -f "$log_path" ]]; then
    echo "[SUMMARY][${label}] Missing log: $log_path"
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
clock = float(logs.get('mean_game_clocktime', logs.get('game_clocktime', 0.0)) or 0.0)

print(f"[SUMMARY][{label}] clock_sec={clock:.4f}")
if returns.size:
    print(f"[SUMMARY][{label}] returns={returns.mean():.4f}/{returns.std():.4f}/{returns.min():.4f}/{returns.max():.4f}")
else:
    print(f"[SUMMARY][{label}] returns=empty")
if iters.size:
    print(f"[SUMMARY][{label}] game_iters={iters.mean():.2f}/{iters.std():.2f}/{iters.min():.0f}/{iters.max():.0f}")
else:
    print(f"[SUMMARY][{label}] game_iters=empty")
PY
}

run_homberger_test_size() {
  local size="$1"
  local model_path="$2"
  local test_path="$DATA_DIR/vrptw_test_homberger_${size}.pkl"

  if [[ ! -f "$model_path" ]]; then
    echo "[RUN-TEST][n=${size}] Missing model: $model_path"
    return
  fi
  if [[ ! -f "$test_path" ]]; then
    echo "[RUN-TEST][n=${size}] Missing test pkl: $test_path"
    return
  fi

  mkdir -p "$OUT_DIR" "$OUT_DIR/run_test_logs"

  local infer_cfg
  infer_cfg="$(mktemp /tmp/config_infer_vrptw_tree_${size}.XXXXXX.yaml)"
  local infer_parallel
  infer_parallel="$(infer_parallel_by_size "$size")"

  "$PYTHON_BIN" - <<PY
import yaml

with open('$INFER_CONFIG') as f:
    cfg = yaml.safe_load(f)

cfg.setdefault('train', {})['pretrained_fname'] = '$model_path'
cfg['train']['n_parallel_problems'] = int('$infer_parallel')
cfg.setdefault('eval', {})['data_file'] = '$test_path'
cfg['eval']['detailed_test_log'] = True

with open('$infer_cfg', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print('infer_cfg=', '$infer_cfg')
PY

  echo "[RUN-TEST][n=${size}] RL inference"
  "$PYTHON_BIN" -m earli.main --config "$infer_cfg"

  local out_log="$OUT_DIR/test_logs_homberger_${size}.pkl"
  if [[ -f "outputs/test_logs.pkl" ]]; then
    cp "outputs/test_logs.pkl" "$out_log"
    summarize_test_log "$out_log" "tree_homberger_${size}"
  else
    echo "[RUN-TEST][n=${size}] Missing outputs/test_logs.pkl"
    return
  fi

  local rl_runtime
  rl_runtime="$($PYTHON_BIN - <<PY
import pickle
with open('$out_log', 'rb') as f:
    logs = pickle.load(f)
print(float(logs.get('mean_game_clocktime', logs.get('game_clocktime', 0.0)) or 0.0))
PY
)"

  local init_cfg
  init_cfg="$(mktemp /tmp/config_init_vrptw_tree_${size}.XXXXXX.yaml)"

  "$PYTHON_BIN" - <<PY
import yaml

with open('$INIT_CONFIG') as f:
    cfg = yaml.safe_load(f)

cfg.setdefault('names', {})['run_title'] = 'homberger_tree_${size}'
cfg['names']['methods'] = ['cuOpt', 'cuOpt_RL']
cfg['names']['baseline'] = 'cuOpt'
cfg['names']['main_method'] = 'cuOpt_RL'

cfg.setdefault('paths', {})['problems'] = '$test_path'
cfg['paths']['solutions'] = '$out_log'
cfg['paths']['out_summary'] = 'test_summary_homberger_injection_tree'

cfg.setdefault('cuopt', {})['runtimes'] = [int(x) for x in '${EVAL_RUNTIMES[*]}'.split()]
cfg['cuopt']['repetitions'] = 1

cfg.setdefault('injection', {})['rl_runtime'] = float('$rl_runtime')
cfg['injection']['log_per_problem'] = False

with open('$init_cfg', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print('init_cfg=', '$init_cfg')
PY

  echo "[RUN-TEST][n=${size}] Injection comparison"
  "$PYTHON_BIN" -c "from earli import test_injection; test_injection.main(config_path='$init_cfg')"

  echo "[RUN-TEST][n=${size}] done"
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
    --data-dir)
      DATA_DIR="$2"; shift 2 ;;
    --out-dir)
      OUT_DIR="$2"; MODEL_DIR="$OUT_DIR/models"; shift 2 ;;
    --initial-model)
      INITIAL_MODEL="$2"; shift 2 ;;
    --skip-data)
      SKIP_DATA=1; shift ;;
    --skip-train)
      SKIP_TRAIN=1; shift ;;
    --run-test)
      RUN_TEST=1; shift ;;
    --stages)
      parse_csv_to_array "$2" CURRICULUM_STAGES; shift 2 ;;
    --eval-sizes)
      parse_csv_to_array "$2" EVAL_SIZES; shift 2 ;;
    --eval-runtimes)
      parse_csv_to_array "$2" EVAL_RUNTIMES; shift 2 ;;
    --tree-epochs)
      parse_csv_to_array "$2" TREE_EPOCHS; shift 2 ;;
    --tree-beams)
      parse_csv_to_array "$2" TREE_BEAMS; shift 2 ;;
    --tree-data-steps)
      parse_csv_to_array "$2" TREE_DATA_STEPS; shift 2 ;;
    --tree-batch-sizes)
      parse_csv_to_array "$2" TREE_BATCH_SIZES; shift 2 ;;
    --tree-parallel)
      parse_csv_to_array "$2" TREE_PARALLEL_PROBLEMS; shift 2 ;;
    --tree-lrs)
      parse_csv_to_array "$2" TREE_LRS; shift 2 ;;
    --vehicle-penalties)
      parse_csv_to_array "$2" VEHICLE_PENALTIES; shift 2 ;;
    --pip-use-lagrangian)
      PIP_USE_LAGRANGIAN="$2"; shift 2 ;;
    --pip-lagrangian-lrs)
      parse_csv_to_array "$2" PIP_LAGRANGIAN_LRS; shift 2 ;;
    --pip-lagrangian-emas)
      parse_csv_to_array "$2" PIP_LAGRANGIAN_EMAS; shift 2 ;;
    --pip-lagrangian-maxs)
      parse_csv_to_array "$2" PIP_LAGRANGIAN_MAXS; shift 2 ;;
    --pip-pen-late-sum)
      parse_csv_to_array "$2" PIP_PEN_LATE_SUM; shift 2 ;;
    --pip-pen-late-count)
      parse_csv_to_array "$2" PIP_PEN_LATE_COUNT; shift 2 ;;
    --pip-pen-masked-ratio)
      parse_csv_to_array "$2" PIP_PEN_MASKED_RATIO; shift 2 ;;
    --pip-pen-depot-with-customer)
      parse_csv_to_array "$2" PIP_PEN_DEPOT_WITH_CUSTOMER; shift 2 ;;
    --pip-pen-vehicle-over-lb)
      parse_csv_to_array "$2" PIP_PEN_VEHICLE_OVER_LB; shift 2 ;;
    --pip-target-late-sum)
      parse_csv_to_array "$2" PIP_TARGET_LATE_SUM; shift 2 ;;
    --pip-target-late-count)
      parse_csv_to_array "$2" PIP_TARGET_LATE_COUNT; shift 2 ;;
    --pip-target-masked-ratio)
      parse_csv_to_array "$2" PIP_TARGET_MASKED_RATIO; shift 2 ;;
    --pip-target-depot-with-customer)
      parse_csv_to_array "$2" PIP_TARGET_DEPOT_WITH_CUSTOMER; shift 2 ;;
    --pip-target-vehicle-over-lb)
      parse_csv_to_array "$2" PIP_TARGET_VEHICLE_OVER_LB; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1 ;;
  esac
done

set_profile_defaults "$PROFILE"

check_len TREE_EPOCHS "${#TREE_EPOCHS[@]}"
check_len TREE_BEAMS "${#TREE_BEAMS[@]}"
check_len TREE_DATA_STEPS "${#TREE_DATA_STEPS[@]}"
check_len TREE_BATCH_SIZES "${#TREE_BATCH_SIZES[@]}"
check_len TREE_PARALLEL_PROBLEMS "${#TREE_PARALLEL_PROBLEMS[@]}"
check_len TREE_LRS "${#TREE_LRS[@]}"
check_len VEHICLE_PENALTIES "${#VEHICLE_PENALTIES[@]}"

check_len PIP_LAGRANGIAN_LRS "${#PIP_LAGRANGIAN_LRS[@]}"
check_len PIP_LAGRANGIAN_EMAS "${#PIP_LAGRANGIAN_EMAS[@]}"
check_len PIP_LAGRANGIAN_MAXS "${#PIP_LAGRANGIAN_MAXS[@]}"
check_len PIP_PEN_LATE_SUM "${#PIP_PEN_LATE_SUM[@]}"
check_len PIP_PEN_LATE_COUNT "${#PIP_PEN_LATE_COUNT[@]}"
check_len PIP_PEN_MASKED_RATIO "${#PIP_PEN_MASKED_RATIO[@]}"
check_len PIP_PEN_DEPOT_WITH_CUSTOMER "${#PIP_PEN_DEPOT_WITH_CUSTOMER[@]}"
check_len PIP_PEN_VEHICLE_OVER_LB "${#PIP_PEN_VEHICLE_OVER_LB[@]}"
check_len PIP_TARGET_LATE_SUM "${#PIP_TARGET_LATE_SUM[@]}"
check_len PIP_TARGET_LATE_COUNT "${#PIP_TARGET_LATE_COUNT[@]}"
check_len PIP_TARGET_MASKED_RATIO "${#PIP_TARGET_MASKED_RATIO[@]}"
check_len PIP_TARGET_DEPOT_WITH_CUSTOMER "${#PIP_TARGET_DEPOT_WITH_CUSTOMER[@]}"
check_len PIP_TARGET_VEHICLE_OVER_LB "${#PIP_TARGET_VEHICLE_OVER_LB[@]}"

validate_int_array_positive CURRICULUM_STAGES "${CURRICULUM_STAGES[@]}"
validate_int_array_positive TREE_EPOCHS "${TREE_EPOCHS[@]}"
validate_int_array_positive TREE_BEAMS "${TREE_BEAMS[@]}"
validate_int_array_positive TREE_DATA_STEPS "${TREE_DATA_STEPS[@]}"
validate_int_array_positive TREE_BATCH_SIZES "${TREE_BATCH_SIZES[@]}"
validate_int_array_positive TREE_PARALLEL_PROBLEMS "${TREE_PARALLEL_PROBLEMS[@]}"

validate_float_array_range TREE_LRS 1e-9 1.0 "${TREE_LRS[@]}"
validate_float_array_range VEHICLE_PENALTIES 1.0 200.0 "${VEHICLE_PENALTIES[@]}"

validate_float_array_range PIP_LAGRANGIAN_LRS 1e-9 1.0 "${PIP_LAGRANGIAN_LRS[@]}"
validate_float_array_range PIP_LAGRANGIAN_EMAS 0.0 1.0 "${PIP_LAGRANGIAN_EMAS[@]}"
validate_float_array_range PIP_LAGRANGIAN_MAXS 0.1 50.0 "${PIP_LAGRANGIAN_MAXS[@]}"
validate_float_array_range PIP_PEN_LATE_SUM 0.0 5.0 "${PIP_PEN_LATE_SUM[@]}"
validate_float_array_range PIP_PEN_LATE_COUNT 0.0 5.0 "${PIP_PEN_LATE_COUNT[@]}"
validate_float_array_range PIP_PEN_MASKED_RATIO 0.0 5.0 "${PIP_PEN_MASKED_RATIO[@]}"
validate_float_array_range PIP_PEN_DEPOT_WITH_CUSTOMER 0.0 5.0 "${PIP_PEN_DEPOT_WITH_CUSTOMER[@]}"
validate_float_array_range PIP_PEN_VEHICLE_OVER_LB 0.0 5.0 "${PIP_PEN_VEHICLE_OVER_LB[@]}"
validate_float_array_range PIP_TARGET_LATE_SUM 0.0 1.0 "${PIP_TARGET_LATE_SUM[@]}"
validate_float_array_range PIP_TARGET_LATE_COUNT 0.0 1.0 "${PIP_TARGET_LATE_COUNT[@]}"
validate_float_array_range PIP_TARGET_MASKED_RATIO 0.0 1.0 "${PIP_TARGET_MASKED_RATIO[@]}"
validate_float_array_range PIP_TARGET_DEPOT_WITH_CUSTOMER 0.0 1.0 "${PIP_TARGET_DEPOT_WITH_CUSTOMER[@]}"
validate_float_array_range PIP_TARGET_VEHICLE_OVER_LB 0.0 1.0 "${PIP_TARGET_VEHICLE_OVER_LB[@]}"

validate_int_array_positive EVAL_SIZES "${EVAL_SIZES[@]}"
validate_int_array_positive EVAL_RUNTIMES "${EVAL_RUNTIMES[@]}"

"$PYTHON_BIN" - "$PIP_USE_LAGRANGIAN" <<'PY'
import sys
v = sys.argv[1]
if v not in {'0', '1'}:
    raise SystemExit(f"[ERROR] --pip-use-lagrangian expects 0/1, got: {v}")
print(f"[CHECK] pip_use_lagrangian: {v}")
PY

for cfg in "$TRAIN_CONFIG" "$INFER_CONFIG" "$INIT_CONFIG"; do
  if [[ ! -f "$cfg" ]]; then
    echo "[ERROR] Missing config: $cfg"
    exit 1
  fi
done

mkdir -p "$OUT_DIR" "$MODEL_DIR"

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

echo "[INFO] Tree curriculum outputs in: $OUT_DIR"

if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  echo "[2/3] Stage training: pure tree_based with curriculum + PIP"
  PREV_MODEL="$INITIAL_MODEL"

  if [[ -n "$PREV_MODEL" && ! -f "$PREV_MODEL" ]]; then
    echo "[ERROR] --initial-model file not found: $PREV_MODEL"
    exit 1
  fi

  for i in "${!CURRICULUM_STAGES[@]}"; do
    SIZE="${CURRICULUM_STAGES[$i]}"
    T_EPOCHS="${TREE_EPOCHS[$i]}"
    T_BEAMS="${TREE_BEAMS[$i]}"
    T_DATA_STEPS="${TREE_DATA_STEPS[$i]}"
    T_BATCH="${TREE_BATCH_SIZES[$i]}"
    T_PARALLEL="${TREE_PARALLEL_PROBLEMS[$i]}"
    T_LR="${TREE_LRS[$i]}"
    VEH_PEN="${VEHICLE_PENALTIES[$i]}"

    PIP_LR="${PIP_LAGRANGIAN_LRS[$i]}"
    PIP_EMA="${PIP_LAGRANGIAN_EMAS[$i]}"
    PIP_MAX="${PIP_LAGRANGIAN_MAXS[$i]}"
    PIP_LATE_SUM="${PIP_PEN_LATE_SUM[$i]}"
    PIP_LATE_COUNT="${PIP_PEN_LATE_COUNT[$i]}"
    PIP_MASKED="${PIP_PEN_MASKED_RATIO[$i]}"
    PIP_DEPOT="${PIP_PEN_DEPOT_WITH_CUSTOMER[$i]}"
    PIP_VEH_OVER="${PIP_PEN_VEHICLE_OVER_LB[$i]}"
    PIP_TGT_LATE_SUM="${PIP_TARGET_LATE_SUM[$i]}"
    PIP_TGT_LATE_COUNT="${PIP_TARGET_LATE_COUNT[$i]}"
    PIP_TGT_MASKED="${PIP_TARGET_MASKED_RATIO[$i]}"
    PIP_TGT_DEPOT="${PIP_TARGET_DEPOT_WITH_CUSTOMER[$i]}"
    PIP_TGT_VEH_OVER="${PIP_TARGET_VEHICLE_OVER_LB[$i]}"

    TRAIN_PKL="$DATA_DIR/vrptw_train_${SIZE}.pkl"
    VAL_PKL="$DATA_DIR/vrptw_val_${SIZE}.pkl"
    TREE_MODEL="$MODEL_DIR/vrptw_model_tree_curriculum_${SIZE}.m"

    TREE_CFG="$(mktemp /tmp/config_train_vrptw_tree_curriculum_${SIZE}.XXXXXX.yaml)"
    "$PYTHON_BIN" - <<PY
import yaml

with open('$TRAIN_CONFIG') as f:
    cfg = yaml.safe_load(f)

cfg.setdefault('system', {})['compatibility_mode'] = None
cfg['system']['use_tensordict'] = True
cfg['system']['save_obs_on_gpu'] = True
cfg['system']['tensorboard_logdir'] = '$OUT_DIR/tensorboard'
cfg['system']['run_name'] = 'tree_curriculum_stage_$SIZE'

cfg.setdefault('eval', {})['data_file'] = '$TRAIN_PKL'
cfg['eval']['val_data_file'] = '$VAL_PKL'
cfg['eval']['sampling_mode'] = 'random_with_replacement'
cfg['eval']['deterministic_test_beam'] = True
cfg['eval']['apply_local_search'] = False
cfg['eval'].pop('data_files', None)

cfg.setdefault('train', {})['method'] = 'tree_based'
cfg['train']['pretrained_fname'] = '$PREV_MODEL' if '$PREV_MODEL' else None
cfg['train']['save_model_path'] = '$TREE_MODEL'
cfg['train']['epochs'] = int('$T_EPOCHS')
cfg['train']['n_beams'] = int('$T_BEAMS')
cfg['train']['batch_size'] = int('$T_BATCH')
cfg['train']['n_parallel_problems'] = int('$T_PARALLEL')
cfg['train']['learning_rate'] = float('$T_LR')
cfg['train']['sampling_mode'] = 'random_with_replacement'

cfg.setdefault('muzero', {})['expansion_method'] = 'KPPO'
cfg['muzero']['data_steps_per_epoch'] = int('$T_DATA_STEPS')
cfg['muzero']['deterministic_branch_in_k_beams'] = False
cfg['muzero']['max_leaves'] = int(max(12, int('$T_BEAMS')))
cfg['muzero']['max_moves'] = int(max(8000, int('$SIZE') * 10))

cfg.setdefault('sampler', {})['temperature'] = 0.2
cfg['sampler']['diversity_penalty'] = 0.01

ps = cfg.setdefault('problem_setup', {})
ps['vehicle_penalty'] = float('$VEH_PEN')
ps['use_lagrangian_constraints'] = bool(int('$PIP_USE_LAGRANGIAN'))
ps['lagrangian_lr'] = float('$PIP_LR')
ps['lagrangian_ema'] = float('$PIP_EMA')
ps['lagrangian_max'] = float('$PIP_MAX')
ps['penalty_late_sum'] = float('$PIP_LATE_SUM')
ps['penalty_late_count'] = float('$PIP_LATE_COUNT')
ps['penalty_masked_ratio'] = float('$PIP_MASKED')
ps['penalty_depot_with_customer'] = float('$PIP_DEPOT')
ps['penalty_vehicle_over_lb'] = float('$PIP_VEH_OVER')
ps['target_late_sum'] = float('$PIP_TGT_LATE_SUM')
ps['target_late_count'] = float('$PIP_TGT_LATE_COUNT')
ps['target_masked_ratio'] = float('$PIP_TGT_MASKED')
ps['target_depot_with_customer'] = float('$PIP_TGT_DEPOT')
ps['target_vehicle_over_lb'] = float('$PIP_TGT_VEH_OVER')

with open('$TREE_CFG', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print('tree_cfg=', '$TREE_CFG')
print('stage=', '$SIZE', 'save_model=', '$TREE_MODEL')
PY

    echo "[2/3][n=${SIZE}] Tree train: epochs=${T_EPOCHS}, beams=${T_BEAMS}, data_steps=${T_DATA_STEPS}, parallel=${T_PARALLEL}, lr=${T_LR}, pip_lagrangian=${PIP_USE_LAGRANGIAN}, pip_lr=${PIP_LR}, pip_masked_pen=${PIP_MASKED}, pip_vehicle_over_lb_pen=${PIP_VEH_OVER}"
    "$PYTHON_BIN" -m earli.train --config "$TREE_CFG"

    if [[ -f "$TREE_MODEL" ]]; then
      PREV_MODEL="$TREE_MODEL"
      append_stage_log "$SIZE" "$TREE_MODEL" "ok" "$TREE_CFG"
    else
      echo "[WARN] Missing tree model for n=${SIZE}: $TREE_MODEL"
      append_stage_log "$SIZE" "$TREE_MODEL" "missing_model" "$TREE_CFG"
      continue
    fi

    if [[ "$RUN_TEST" -eq 1 ]] && contains_value "$SIZE" "${EVAL_SIZES[@]}"; then
      run_homberger_test_size "$SIZE" "$PREV_MODEL"
      RUN_TEST_DONE_SIZES+=("$SIZE")
    fi
  done
fi

if [[ "$RUN_TEST" -eq 1 ]]; then
  echo "[3/3] Replay run-test for remaining sizes"
  for size in "${EVAL_SIZES[@]}"; do
    if contains_value "$size" "${RUN_TEST_DONE_SIZES[@]}"; then
      echo "[RUN-TEST][n=${size}] already done during stage loop"
      continue
    fi
    final_model="$MODEL_DIR/vrptw_model_tree_curriculum_${size}.m"
    run_homberger_test_size "$size" "$final_model"
  done
fi

echo "[DONE] Pure tree_based curriculum pipeline complete"
