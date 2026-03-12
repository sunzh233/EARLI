#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config_train_pomo_tw.yaml}"
INFER_CONFIG="${INFER_CONFIG:-config_infer_vrptw_homberger.yaml}"
INIT_CONFIG="${INIT_CONFIG:-config_initialization_vrptw_homberger.yaml}"

DATA_DIR="${DATA_DIR:-datasets/vrptw_homberger_like_curriculum}"
OUT_DIR="${OUT_DIR:-outputs/pomo_curriculum}"
MODEL_DIR="$OUT_DIR/models"
TB_DIR="$OUT_DIR/tensorboard"

PROFILE="conservative"
SKIP_DATA=0
SKIP_TRAIN=0
RUN_TEST=1
DRY_RUN=0
INITIAL_MODEL=""

CURRICULUM_STAGES=(50 100 150 200 300 400 500 600 800 1000)
PROFILE_BASE_STAGES=(50 100 150 200 300 400 500 600 800 1000)
EVAL_SIZES=(200 400 600 800 1000)
EVAL_RUNTIMES=(15 30 45)

STAGE_EPOCHS=()
STAGE_DATA_STEPS=()
STAGE_LRS=()
STAGE_BATCH_SIZES=()
STAGE_PARALLEL_PROBLEMS=()
STAGE_N_AUGMENTS=()
STAGE_ENT_COEFS=()
STAGE_VF_COEFS=()
STAGE_CLIP_RANGES=()
STAGE_USE_PIP=()

usage() {
  cat <<'EOF'
Usage:
  ./run_curriculum_pomo_tw.sh [options]

A complete one-click POMO-TW curriculum pipeline:
1) generate synthetic Homberger-like datasets (optional skip)
2) stage-by-stage curriculum training (small-n to large-n)
3) test on Homberger sizes and run cuOpt injection comparison

Options:
  --python BIN                 Python executable path
  --profile NAME               conservative|aggressive (default: conservative)
  --train-config PATH          Base train config (default: config_train_pomo_tw.yaml)
  --infer-config PATH          Base inference config (default: config_infer_vrptw_homberger.yaml)
  --init-config PATH           Base injection config (default: config_initialization_vrptw_homberger.yaml)
  --data-dir PATH              Dataset root (default: datasets/vrptw_homberger_like_curriculum)
  --out-dir PATH               Output root (default: outputs/pomo_curriculum)
  --initial-model PATH         Warm-start model for stage 1

  --skip-data                  Skip dataset generation
  --skip-train                 Skip stage training and only run test/injection from saved models
  --skip-test                  Skip inference+injection tests
  --dry-run                    Print commands/configs without launching train/test

  --stages L                   CSV stage sizes, e.g. 50,100,150,...,1000
  --eval-sizes L               CSV evaluation sizes (default: 200,400,600,800,1000)
  --eval-runtimes L            CSV cuOpt runtimes (default: 15,30,45)

  Stage overrides (CSV length must equal --stages length):
  --stage-epochs L
  --stage-data-steps L         maps to muzero.data_steps_per_epoch
  --stage-lrs L
  --stage-batch-sizes L
  --stage-parallel L
  --stage-augments L           maps to pomo_tw.n_augments (1/2/4/8)
  --stage-ent-coefs L
  --stage-vf-coefs L
  --stage-clip-ranges L
  --stage-use-pip L            0/1 flags for problem_setup.use_pip_masking

Examples:
  ./run_curriculum_pomo_tw.sh

  ./run_curriculum_pomo_tw.sh \
    --profile aggressive \
    --stage-use-pip 0,0,0,0,0,1,1,1,1,1
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

check_len() {
  local name="$1"
  local arr_len="$2"
  if [[ "$arr_len" -eq 0 ]]; then
    return 0
  fi
  if [[ "${#CURRICULUM_STAGES[@]}" -ne "$arr_len" ]]; then
    echo "[ERROR] --stages length (${#CURRICULUM_STAGES[@]}) and ${name} length ($arr_len) mismatch"
    exit 1
  fi
}

remap_profile_array_to_stages() {
  local arr_name="$1"
  local -n arr_ref="$arr_name"

  if [[ "${#arr_ref[@]}" -eq 0 ]]; then
    return 0
  fi

  # If length already matches stages, nothing to do.
  if [[ "${#arr_ref[@]}" -eq "${#CURRICULUM_STAGES[@]}" ]]; then
    return 0
  fi

  # Only auto-remap arrays that still follow profile-base length.
  if [[ "${#arr_ref[@]}" -ne "${#PROFILE_BASE_STAGES[@]}" ]]; then
    return 0
  fi

  local remapped=()
  local s i matched
  for s in "${CURRICULUM_STAGES[@]}"; do
    matched=0
    for i in "${!PROFILE_BASE_STAGES[@]}"; do
      if [[ "$s" == "${PROFILE_BASE_STAGES[$i]}" ]]; then
        remapped+=("${arr_ref[$i]}")
        matched=1
        break
      fi
    done
    if [[ "$matched" -eq 0 ]]; then
      echo "[ERROR] Stage size ${s} is not in profile defaults (${PROFILE_BASE_STAGES[*]})."
      echo "        Please pass explicit CSV overrides for all --stage-* options."
      exit 1
    fi
  done

  arr_ref=("${remapped[@]}")
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
      # PPO-style conservative schedule adapted to POMO-TW.
      if [[ "${#STAGE_EPOCHS[@]}" -eq 0 ]]; then STAGE_EPOCHS=(14 12 10 8 7 6 5 4 3 3); fi
      if [[ "${#STAGE_DATA_STEPS[@]}" -eq 0 ]]; then STAGE_DATA_STEPS=(128 112 96 80 64 56 48 40 32 24); fi
      if [[ "${#STAGE_LRS[@]}" -eq 0 ]]; then STAGE_LRS=(1.5e-5 6.0e-6 4.8e-6 4.2e-6 3.8e-6 3.4e-6 3.0e-6 2.6e-6 2.2e-6 1.9e-6); fi
      if [[ "${#STAGE_BATCH_SIZES[@]}" -eq 0 ]]; then STAGE_BATCH_SIZES=(256 256 256 256 192 192 160 128 96 64); fi
      if [[ "${#STAGE_PARALLEL_PROBLEMS[@]}" -eq 0 ]]; then STAGE_PARALLEL_PROBLEMS=(16 16 14 12 10 8 6 4 2 1); fi
      if [[ "${#STAGE_N_AUGMENTS[@]}" -eq 0 ]]; then STAGE_N_AUGMENTS=(4 4 4 8 8 8 8 8 8 8); fi
      if [[ "${#STAGE_ENT_COEFS[@]}" -eq 0 ]]; then STAGE_ENT_COEFS=(0.0025 0.0022 0.0020 0.0018 0.0016 0.0014 0.0012 0.0010 0.0008 0.0008); fi
      if [[ "${#STAGE_VF_COEFS[@]}" -eq 0 ]]; then STAGE_VF_COEFS=(0.70 0.70 0.68 0.66 0.64 0.62 0.60 0.58 0.56 0.56); fi
      if [[ "${#STAGE_CLIP_RANGES[@]}" -eq 0 ]]; then STAGE_CLIP_RANGES=(0.15 0.14 0.13 0.12 0.12 0.11 0.11 0.10 0.10 0.10); fi
      if [[ "${#STAGE_USE_PIP[@]}" -eq 0 ]]; then STAGE_USE_PIP=(0 0 0 0 0 0 0 0 0 0); fi
      ;;
    aggressive)
      if [[ "${#STAGE_EPOCHS[@]}" -eq 0 ]]; then STAGE_EPOCHS=(18 16 14 12 10 8 7 6 5 4); fi
      if [[ "${#STAGE_DATA_STEPS[@]}" -eq 0 ]]; then STAGE_DATA_STEPS=(160 144 128 112 96 80 64 52 40 32); fi
      if [[ "${#STAGE_LRS[@]}" -eq 0 ]]; then STAGE_LRS=(2.0e-5 9.0e-6 6.8e-6 5.8e-6 5.0e-6 4.3e-6 3.7e-6 3.1e-6 2.5e-6 2.1e-6); fi
      if [[ "${#STAGE_BATCH_SIZES[@]}" -eq 0 ]]; then STAGE_BATCH_SIZES=(256 256 256 256 224 192 160 128 96 64); fi
      if [[ "${#STAGE_PARALLEL_PROBLEMS[@]}" -eq 0 ]]; then STAGE_PARALLEL_PROBLEMS=(16 16 14 12 10 8 6 4 2 1); fi
      if [[ "${#STAGE_N_AUGMENTS[@]}" -eq 0 ]]; then STAGE_N_AUGMENTS=(4 4 8 8 8 8 8 8 8 8); fi
      if [[ "${#STAGE_ENT_COEFS[@]}" -eq 0 ]]; then STAGE_ENT_COEFS=(0.0060 0.0055 0.0050 0.0045 0.0040 0.0035 0.0030 0.0025 0.0020 0.0020); fi
      if [[ "${#STAGE_VF_COEFS[@]}" -eq 0 ]]; then STAGE_VF_COEFS=(0.45 0.45 0.44 0.43 0.42 0.41 0.40 0.40 0.38 0.38); fi
      if [[ "${#STAGE_CLIP_RANGES[@]}" -eq 0 ]]; then STAGE_CLIP_RANGES=(0.12 0.12 0.11 0.11 0.10 0.10 0.10 0.09 0.09 0.09); fi
      if [[ "${#STAGE_USE_PIP[@]}" -eq 0 ]]; then STAGE_USE_PIP=(0 0 0 0 0 0 0 0 0 0); fi
      ;;
    *)
      echo "[ERROR] Unknown profile: $p (use conservative or aggressive)"
      exit 1
      ;;
  esac
}

resolve_python() {
  local requested="$1"
  local candidates=()

  if [[ -n "$requested" ]]; then
    candidates+=("$requested")
  fi

  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    candidates+=("${CONDA_PREFIX}/bin/python")
  fi

  if command -v python >/dev/null 2>&1; then
    candidates+=("$(command -v python)")
  fi
  if command -v python3 >/dev/null 2>&1; then
    candidates+=("$(command -v python3)")
  fi

  # Common local env fallback used in this workspace
  if [[ -x "/home/cslabuser/miniconda3/envs/earli_env/bin/python" ]]; then
    candidates+=("/home/cslabuser/miniconda3/envs/earli_env/bin/python")
  fi

  for py in "${candidates[@]}"; do
    if "$py" - <<'PY' >/dev/null 2>&1
import yaml
import torch
import stable_baselines3
import tensordict
PY
    then
      echo "$py"
      return 0
    fi
  done

  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --train-config) TRAIN_CONFIG="$2"; shift 2 ;;
    --infer-config) INFER_CONFIG="$2"; shift 2 ;;
    --init-config) INIT_CONFIG="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; MODEL_DIR="$OUT_DIR/models"; TB_DIR="$OUT_DIR/tensorboard"; shift 2 ;;
    --initial-model) INITIAL_MODEL="$2"; shift 2 ;;

    --skip-data) SKIP_DATA=1; shift ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --skip-test) RUN_TEST=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;

    --stages) parse_csv_to_array "$2" CURRICULUM_STAGES; shift 2 ;;
    --eval-sizes) parse_csv_to_array "$2" EVAL_SIZES; shift 2 ;;
    --eval-runtimes) parse_csv_to_array "$2" EVAL_RUNTIMES; shift 2 ;;

    --stage-epochs) parse_csv_to_array "$2" STAGE_EPOCHS; shift 2 ;;
    --stage-data-steps) parse_csv_to_array "$2" STAGE_DATA_STEPS; shift 2 ;;
    --stage-lrs) parse_csv_to_array "$2" STAGE_LRS; shift 2 ;;
    --stage-batch-sizes) parse_csv_to_array "$2" STAGE_BATCH_SIZES; shift 2 ;;
    --stage-parallel) parse_csv_to_array "$2" STAGE_PARALLEL_PROBLEMS; shift 2 ;;
    --stage-augments) parse_csv_to_array "$2" STAGE_N_AUGMENTS; shift 2 ;;
    --stage-ent-coefs) parse_csv_to_array "$2" STAGE_ENT_COEFS; shift 2 ;;
    --stage-vf-coefs) parse_csv_to_array "$2" STAGE_VF_COEFS; shift 2 ;;
    --stage-clip-ranges) parse_csv_to_array "$2" STAGE_CLIP_RANGES; shift 2 ;;
    --stage-use-pip) parse_csv_to_array "$2" STAGE_USE_PIP; shift 2 ;;

    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

set_profile_defaults "$PROFILE"

# If user changes --stages to a subset of profile stages, auto-remap defaults.
remap_profile_array_to_stages STAGE_EPOCHS
remap_profile_array_to_stages STAGE_DATA_STEPS
remap_profile_array_to_stages STAGE_LRS
remap_profile_array_to_stages STAGE_BATCH_SIZES
remap_profile_array_to_stages STAGE_PARALLEL_PROBLEMS
remap_profile_array_to_stages STAGE_N_AUGMENTS
remap_profile_array_to_stages STAGE_ENT_COEFS
remap_profile_array_to_stages STAGE_VF_COEFS
remap_profile_array_to_stages STAGE_CLIP_RANGES
remap_profile_array_to_stages STAGE_USE_PIP

check_len STAGE_EPOCHS "${#STAGE_EPOCHS[@]}"
check_len STAGE_DATA_STEPS "${#STAGE_DATA_STEPS[@]}"
check_len STAGE_LRS "${#STAGE_LRS[@]}"
check_len STAGE_BATCH_SIZES "${#STAGE_BATCH_SIZES[@]}"
check_len STAGE_PARALLEL_PROBLEMS "${#STAGE_PARALLEL_PROBLEMS[@]}"
check_len STAGE_N_AUGMENTS "${#STAGE_N_AUGMENTS[@]}"
check_len STAGE_ENT_COEFS "${#STAGE_ENT_COEFS[@]}"
check_len STAGE_VF_COEFS "${#STAGE_VF_COEFS[@]}"
check_len STAGE_CLIP_RANGES "${#STAGE_CLIP_RANGES[@]}"
check_len STAGE_USE_PIP "${#STAGE_USE_PIP[@]}"

PYTHON_BIN="$(resolve_python "$PYTHON_BIN" || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "[ERROR] Could not find a Python interpreter with required deps: yaml, torch, stable_baselines3, tensordict"
  echo "Hint: activate your EARLI env first, then rerun."
  exit 1
fi

echo "[INFO] Using Python: $PYTHON_BIN"

for cfg in "$TRAIN_CONFIG" "$INFER_CONFIG" "$INIT_CONFIG"; do
  if [[ ! -f "$cfg" ]]; then
    echo "[ERROR] Missing config: $cfg"
    exit 1
  fi
done

mkdir -p "$OUT_DIR" "$MODEL_DIR" "$TB_DIR" "$OUT_DIR/test_logs"

if [[ "$SKIP_DATA" -eq 0 ]]; then
  echo "[1/3] Generate synthetic Homberger-like curriculum datasets"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "$PYTHON_BIN tools/generate_vrptw_homberger_like_synthetic.py --root $ROOT_DIR --output-dir $DATA_DIR --sizes ${CURRICULUM_STAGES[*]} --export-homberger-test --export-by-category"
  else
    "$PYTHON_BIN" tools/generate_vrptw_homberger_like_synthetic.py \
      --root "$ROOT_DIR" \
      --output-dir "$DATA_DIR" \
      --sizes "${CURRICULUM_STAGES[@]}" \
      --export-homberger-test \
      --export-by-category
  fi
fi

for s in "${CURRICULUM_STAGES[@]}"; do
  if [[ ! -f "$DATA_DIR/vrptw_train_${s}.pkl" || ! -f "$DATA_DIR/vrptw_val_${s}.pkl" ]]; then
    echo "[ERROR] Missing train/val dataset for stage n=${s} under $DATA_DIR"
    exit 1
  fi
done

if [[ "$RUN_TEST" -eq 1 ]]; then
  for s in "${EVAL_SIZES[@]}"; do
    if [[ ! -f "$DATA_DIR/vrptw_test_homberger_${s}.pkl" ]]; then
      echo "[ERROR] Missing test dataset: $DATA_DIR/vrptw_test_homberger_${s}.pkl"
      echo "Run without --skip-data to auto-generate Homberger test PKLs."
      exit 1
    fi
  done
fi

run_eval_size() {
  local size="$1"
  local model_path="$2"

  local test_pkl="$DATA_DIR/vrptw_test_homberger_${size}.pkl"
  local infer_cfg
  local init_cfg
  local infer_parallel
  local out_log
  local rl_runtime

  if [[ ! -f "$model_path" ]]; then
    echo "[RUN-TEST][n=${size}] Missing model: $model_path"
    return
  fi

  infer_parallel="$(infer_parallel_by_size "$size")"
  infer_cfg="$(mktemp /tmp/config_infer_pomo_tw_${size}.XXXXXX.yaml)"

  # Build inference config from TRAIN_CONFIG so architecture exactly matches
  # the trained POMO model (avoids model-shape mismatch from generic infer cfg).
  "$PYTHON_BIN" - <<PY
import yaml
with open('$TRAIN_CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg.setdefault('system', {})['compatibility_mode'] = None
cfg['system']['use_tensordict'] = True
cfg.setdefault('train', {})['method'] = 'pomo_tw'
cfg.setdefault('train', {})['pretrained_fname'] = '$model_path'
cfg.setdefault('eval', {})['data_file'] = '$test_pkl'
cfg['eval']['detailed_test_log'] = True
cfg['eval']['deterministic_test_beam'] = True
cfg['train']['n_parallel_problems'] = int('$infer_parallel')

# Evaluator expects eval.max_problems as dict (e.g. {'test': N}) when set.
mp = cfg.get('eval', {}).get('max_problems', None)
if isinstance(mp, int):
    cfg['eval']['max_problems'] = {'test': mp}

with open('$infer_cfg', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "$PYTHON_BIN -m earli.main --config $infer_cfg"
    return
  fi

  echo "[RUN-TEST][n=${size}] RL inference"
  "$PYTHON_BIN" -m earli.main --config "$infer_cfg"

  out_log="$OUT_DIR/test_logs/test_logs_homberger_${size}.pkl"
  if [[ -f "outputs/test_logs.pkl" ]]; then
    cp "outputs/test_logs.pkl" "$out_log"
  else
    echo "[RUN-TEST][n=${size}] Missing outputs/test_logs.pkl after inference"
    return
  fi

  rl_runtime="$($PYTHON_BIN - <<PY
import pickle
with open('$out_log', 'rb') as f:
    logs = pickle.load(f)
print(float(logs.get('mean_game_clocktime', logs.get('game_clocktime', 0.0)) or 0.0))
PY
)"

  # Keep test_injection robust: if logs contain fewer problems than the
  # source dataset (e.g. due to max_problems), create a matching subset PKL.
  inj_problem_pkl="$test_pkl"
  subset_pkl="$OUT_DIR/test_logs/test_subset_homberger_${size}.pkl"
  subset_info="$($PYTHON_BIN - <<PY
import pickle
import numpy as np
import torch

test_pkl = '$test_pkl'
log_pkl = '$out_log'
subset_pkl = '$subset_pkl'

with open(log_pkl, 'rb') as f:
    logs = pickle.load(f)
with open(test_pkl, 'rb') as f:
    data = pickle.load(f)

n_data = int(data.get('n_problems', data['positions'].shape[0]))
returns = logs.get('returns', [])
n_log = int(len(returns)) if hasattr(returns, '__len__') else n_data

if n_log <= 0 or n_log >= n_data:
    print(test_pkl)
else:
    out = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor) and v.ndim > 0 and int(v.shape[0]) == n_data:
            out[k] = v[:n_log]
        elif isinstance(v, np.ndarray) and v.ndim > 0 and int(v.shape[0]) == n_data:
            out[k] = v[:n_log]
        elif isinstance(v, list) and len(v) == n_data:
            out[k] = v[:n_log]
        else:
            out[k] = v
    out['n_problems'] = n_log
    with open(subset_pkl, 'wb') as f:
        pickle.dump(out, f)
    print(subset_pkl)
PY
)"
  inj_problem_pkl="$subset_info"

  init_cfg="$(mktemp /tmp/config_init_pomo_tw_${size}.XXXXXX.yaml)"
  "$PYTHON_BIN" - <<PY
import yaml
with open('$INIT_CONFIG') as f:
    cfg = yaml.safe_load(f)

cfg.setdefault('names', {})['run_title'] = 'pomo_tw_homberger_${size}'
cfg['names']['methods'] = ['cuOpt', 'cuOpt_RL']
cfg['names']['baseline'] = 'cuOpt'
cfg['names']['main_method'] = 'cuOpt_RL'

cfg.setdefault('paths', {})['problems'] = '$inj_problem_pkl'
cfg['paths']['solutions'] = '$out_log'
cfg['paths']['out_summary'] = 'test_summary_homberger_pomo_tw'

cfg.setdefault('cuopt', {})['runtimes'] = [int(x) for x in '${EVAL_RUNTIMES[*]}'.split()]
cfg['cuopt']['repetitions'] = 1

cfg.setdefault('injection', {})['rl_runtime'] = float('$rl_runtime')
cfg['injection']['log_per_problem'] = False

with open('$init_cfg', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

  echo "[RUN-TEST][n=${size}] Injection comparison"
  "$PYTHON_BIN" -c "from earli import test_injection; test_injection.main(config_path='$init_cfg')"
}

echo "[2/3] POMO-TW curriculum training"
PREV_MODEL="$INITIAL_MODEL"
if [[ -n "$PREV_MODEL" && ! -f "$PREV_MODEL" ]]; then
  echo "[ERROR] --initial-model not found: $PREV_MODEL"
  exit 1
fi

if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  for i in "${!CURRICULUM_STAGES[@]}"; do
    SIZE="${CURRICULUM_STAGES[$i]}"
    TRAIN_PKL="$DATA_DIR/vrptw_train_${SIZE}.pkl"
    VAL_PKL="$DATA_DIR/vrptw_val_${SIZE}.pkl"
    MODEL_PATH="$MODEL_DIR/pomo_tw_model_${SIZE}.m"

    EPOCHS="${STAGE_EPOCHS[$i]}"
    DATA_STEPS="${STAGE_DATA_STEPS[$i]}"
    LR="${STAGE_LRS[$i]}"
    BATCH_SIZE="${STAGE_BATCH_SIZES[$i]}"
    PARALLEL="${STAGE_PARALLEL_PROBLEMS[$i]}"
    N_AUG="${STAGE_N_AUGMENTS[$i]}"
    ENT_COEF="${STAGE_ENT_COEFS[$i]}"
    VF_COEF="${STAGE_VF_COEFS[$i]}"
    CLIP_RANGE="${STAGE_CLIP_RANGES[$i]}"
    USE_PIP="${STAGE_USE_PIP[$i]}"

    CFG_TMP="$(mktemp /tmp/config_train_pomo_tw_stage_${SIZE}.XXXXXX.yaml)"
    "$PYTHON_BIN" - <<PY
import yaml
with open('$TRAIN_CONFIG') as f:
    cfg = yaml.safe_load(f)

cfg.setdefault('system', {})['compatibility_mode'] = None
cfg['system']['use_tensordict'] = True
cfg['system']['tensorboard_logdir'] = '$TB_DIR'
cfg['system']['run_name'] = 'pomo_tw_stage_${SIZE}'

cfg.setdefault('eval', {})['data_file'] = '$TRAIN_PKL'
cfg['eval']['val_data_file'] = '$VAL_PKL'
cfg['eval']['sampling_mode'] = 'random_with_replacement'
cfg['eval'].pop('data_files', None)

cfg.setdefault('train', {})['method'] = 'pomo_tw'
cfg['train']['save_model_path'] = '$MODEL_PATH'
cfg['train']['pretrained_fname'] = '$PREV_MODEL' if '$PREV_MODEL' else None
cfg['train']['epochs'] = int('$EPOCHS')
cfg['train']['batch_size'] = int('$BATCH_SIZE')
cfg['train']['n_parallel_problems'] = int('$PARALLEL')
cfg['train']['learning_rate'] = float('$LR')
cfg['train']['ent_coef'] = float('$ENT_COEF')
cfg['train']['vf_coef'] = float('$VF_COEF')
cfg['train']['clip_range'] = float('$CLIP_RANGE')

cfg.setdefault('muzero', {})['data_steps_per_epoch'] = int('$DATA_STEPS')

cfg.setdefault('pomo_tw', {})['n_augments'] = int('$N_AUG')
cfg.setdefault('problem_setup', {})['use_pip_masking'] = bool(int('$USE_PIP'))

with open('$CFG_TMP', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

    echo "[TRAIN][n=${SIZE}] epochs=${EPOCHS}, data_steps=${DATA_STEPS}, lr=${LR}, batch=${BATCH_SIZE}, parallel=${PARALLEL}, aug=${N_AUG}, pip=${USE_PIP}"
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "$PYTHON_BIN -m earli.train --config $CFG_TMP"
    else
      "$PYTHON_BIN" -m earli.train --config "$CFG_TMP"
    fi

    if [[ -f "$MODEL_PATH" ]]; then
      PREV_MODEL="$MODEL_PATH"
    else
      echo "[WARN] Missing model after stage n=${SIZE}: $MODEL_PATH"
    fi

    if [[ "$RUN_TEST" -eq 1 ]] && contains_value "$SIZE" "${EVAL_SIZES[@]}"; then
      run_eval_size "$SIZE" "$MODEL_PATH"
    fi
  done
fi

if [[ "$RUN_TEST" -eq 1 ]]; then
  echo "[3/3] Replay tests for remaining eval sizes"
  for size in "${EVAL_SIZES[@]}"; do
    model_path="$MODEL_DIR/pomo_tw_model_${size}.m"
    run_eval_size "$size" "$model_path"
  done
fi

echo "[DONE] POMO-TW curriculum pipeline complete"
echo "[DONE] Models: $MODEL_DIR"
echo "[DONE] Test logs: $OUT_DIR/test_logs"
