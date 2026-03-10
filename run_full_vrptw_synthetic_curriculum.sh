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
USER_STAGE_STEPS_CSV=""
USER_STAGE_N_STEPS_CSV=""
USER_STAGE_LRS_CSV=""
USER_STAGE_EPOCHS_CSV=""
USER_STAGE_BATCH_SIZES_CSV=""
EVAL_SIZES=(200 400 600 800 1000)
EVAL_RUNTIMES=(10 15 25)

CURRICULUM_STAGES=(50 100 150 200 300 400 500 600 800 1000)
STAGE_STEPS=()
STAGE_N_STEPS=()
STAGE_LRS=()
STAGE_EPOCHS=()
STAGE_BATCH_SIZES=()
STAGE_LAG_LRS=()
STAGE_PEN_LATE_SUM=()
STAGE_PEN_LATE_COUNT=()
STAGE_PEN_MASKED_RATIO=()
STAGE_TARGET_MASKED_RATIO=()

# Empirical wall-time model (seconds per env step) for 1x RTX 4090,
# used only for ETA display before training starts.
ETA_SEC_PER_STEP=(0.0040 0.0046 0.0054 0.0064 0.0096 0.0120 0.0144 0.0176 0.0240 0.0310)

set_profile_defaults() {
  local p="$1"
  case "$p" in
    conservative)
      STAGE_STEPS=(400000 320000 240000 180000 120000 90000 70000 60000 45000 45000)
      # Stabilize fine-tuning on larger scales: bigger rollout horizon + lower LR.
      STAGE_N_STEPS=(224 512 512 512 512 512 448 448 384 384)
      STAGE_LRS=(1.5e-5 6.0e-6 5.5e-6 5.0e-6 4.5e-6 3.8e-6 3.2e-6 2.8e-6 2.3e-6 2.0e-6)
      STAGE_EPOCHS=(8 6 6 5 5 4 4 3 3 3)
      STAGE_BATCH_SIZES=(128 256 256 256 256 256 256 256 256 256)
      # Lagrangian and constraint-shaping schedule per stage size.
      STAGE_LAG_LRS=(8e-5 7e-5 6e-5 5e-5 4e-5 3.5e-5 3e-5 2.5e-5 2e-5 2e-5)
      STAGE_PEN_LATE_SUM=(0.015 0.018 0.020 0.022 0.024 0.026 0.028 0.030 0.032 0.035)
      STAGE_PEN_LATE_COUNT=(0.060 0.070 0.080 0.090 0.100 0.110 0.120 0.130 0.140 0.150)
      STAGE_PEN_MASKED_RATIO=(0.010 0.012 0.014 0.016 0.018 0.020 0.022 0.024 0.026 0.030)
      STAGE_TARGET_MASKED_RATIO=(0.18 0.17 0.16 0.15 0.14 0.13 0.12 0.11 0.10 0.10)
      ;;
    aggressive)
      STAGE_STEPS=(600000 500000 400000 300000 180000 150000 120000 100000 80000 60000)
      STAGE_N_STEPS=(256 512 640 768 768 768 704 640 576 512)
      STAGE_LRS=(2.0e-5 9.0e-6 7.5e-6 6.5e-6 5.5e-6 4.8e-6 4.0e-6 3.2e-6 2.6e-6 2.2e-6)
      STAGE_EPOCHS=(10 8 7 6 5 5 4 4 3 3)
      STAGE_BATCH_SIZES=(128 256 256 256 256 256 256 256 256 256)
      STAGE_LAG_LRS=(1e-4 9e-5 8e-5 7e-5 6e-5 5e-5 4e-5 3.5e-5 3e-5 2.5e-5)
      STAGE_PEN_LATE_SUM=(0.020 0.022 0.024 0.026 0.028 0.030 0.032 0.035 0.038 0.040)
      STAGE_PEN_LATE_COUNT=(0.080 0.090 0.100 0.110 0.120 0.130 0.140 0.150 0.160 0.170)
      STAGE_PEN_MASKED_RATIO=(0.015 0.017 0.019 0.021 0.023 0.025 0.028 0.030 0.033 0.035)
      STAGE_TARGET_MASKED_RATIO=(0.20 0.19 0.18 0.17 0.16 0.15 0.14 0.13 0.12 0.12)
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
print("[Schedule] stage | total_steps | n_steps | lr | lag_lr | p_late | p_masked | tgt_masked | epochs | est_minutes")
total_sec = 0.0
lag_lrs = [float(x) for x in "${STAGE_LAG_LRS[*]}".split()]
p_late = [float(x) for x in "${STAGE_PEN_LATE_SUM[*]}".split()]
p_masked = [float(x) for x in "${STAGE_PEN_MASKED_RATIO[*]}".split()]
tgt_masked = [float(x) for x in "${STAGE_TARGET_MASKED_RATIO[*]}".split()]
for s, t, ns, lr, llr, pl, pm, tm, ep, c in zip(stages, steps, n_steps, lrs, lag_lrs, p_late, p_masked, tgt_masked, epochs, sec_per_step):
    est = t * c
    total_sec += est
    print(f"[Schedule] {s:>4} | {t:>10} | {ns:>7} | {lr:>8.2e} | {llr:>8.2e} | {pl:>6.3f} | {pm:>8.3f} | {tm:>10.3f} | {ep:>6} | {est/60:>10.1f}")
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
    STAGE_LAG_LR="${STAGE_LAG_LRS[$i]}"
    PEN_LATE_SUM="${STAGE_PEN_LATE_SUM[$i]}"
    PEN_LATE_COUNT="${STAGE_PEN_LATE_COUNT[$i]}"
    PEN_MASKED_RATIO="${STAGE_PEN_MASKED_RATIO[$i]}"
    TARGET_MASKED_RATIO="${STAGE_TARGET_MASKED_RATIO[$i]}"

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
# Force single-file mode so every stage uses one mixed train/val file.
cfg['eval'].pop('data_files', None)

cfg.setdefault('train', {})['save_model_path'] = '$MODEL_PATH'
cfg['train']['pretrained_fname'] = '$PREV_MODEL' if '$PREV_MODEL' else None
cfg['train']['n_steps'] = int('$N_STEPS')
cfg['train']['learning_rate'] = float('$STAGE_LR')
cfg['train']['epochs'] = int('$STAGE_EPOCH')
cfg['train']['batch_size'] = int('$STAGE_BATCH_SIZE')
# Allow larger initial policy movement on the first (n=50) stage.
if int('$SIZE') == 50:
  cfg['train']['target_kl'] = None
else:
  cfg['train']['target_kl'] = float(cfg.get('train', {}).get('target_kl', 0.0) or 0.2)
cfg['train']['clip_range'] = float(cfg.get('train', {}).get('clip_range', 0.0) or 0.15)
cfg['train']['max_grad_norm'] = float(cfg.get('train', {}).get('max_grad_norm', 0.0) or 0.4)
cfg['train']['sampling_mode'] = 'random_with_replacement'

ps = cfg.setdefault('problem_setup', {})
ps['use_lagrangian_constraints'] = True
ps['lagrangian_lr'] = float('$STAGE_LAG_LR')
ps['lagrangian_ema'] = float(ps.get('lagrangian_ema', 0.05))
ps['lagrangian_max'] = float(ps.get('lagrangian_max', 10.0))
ps['penalty_late_sum'] = float('$PEN_LATE_SUM')
ps['penalty_late_count'] = float('$PEN_LATE_COUNT')
ps['penalty_masked_ratio'] = float('$PEN_MASKED_RATIO')
ps['penalty_pair_blocked'] = float(ps.get('penalty_pair_blocked', 0.0))
ps['target_late_sum'] = float(ps.get('target_late_sum', 0.0))
ps['target_late_count'] = float(ps.get('target_late_count', 0.0))
ps['target_masked_ratio'] = float('$TARGET_MASKED_RATIO')
ps['target_pair_blocked'] = float(ps.get('target_pair_blocked', 0.0))

with open('$STAGE_CFG', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print('stage_cfg=', '$STAGE_CFG')
print('train_data=', cfg['eval'].get('data_file'))
print('val_data=', cfg['eval'].get('val_data_file'))
print('n_steps=', cfg['train']['n_steps'])
print('learning_rate=', cfg['train']['learning_rate'])
print('epochs=', cfg['train']['epochs'])
print('batch_size=', cfg['train']['batch_size'])
print('target_kl=', cfg['train'].get('target_kl'))
print('clip_range=', cfg['train'].get('clip_range'))
print('lagrangian_lr=', cfg['problem_setup'].get('lagrangian_lr'))
print('penalty_late_sum=', cfg['problem_setup'].get('penalty_late_sum'))
print('penalty_late_count=', cfg['problem_setup'].get('penalty_late_count'))
print('penalty_masked_ratio=', cfg['problem_setup'].get('penalty_masked_ratio'))
print('target_masked_ratio=', cfg['problem_setup'].get('target_masked_ratio'))
print('save_model=', cfg['train']['save_model_path'])
print('pretrained=', cfg['train']['pretrained_fname'])
PY

  echo "[2/3][n=${SIZE}] total_steps=${TOTAL_STEPS}, n_steps=${N_STEPS}, lr=${STAGE_LR}, lag_lr=${STAGE_LAG_LR}, p_late=${PEN_LATE_SUM}, p_masked=${PEN_MASKED_RATIO}, target_masked=${TARGET_MASKED_RATIO}, epochs=${STAGE_EPOCH}, batch_size=${STAGE_BATCH_SIZE}"

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
  done
fi

if [[ "$RUN_TEST" -eq 1 ]]; then
  echo "[3/3] Homberger testing for checkpoints: 200/400/600/800/1000"
  for SIZE in 200 400 600 800 1000; do
    MODEL_PATH="outputs/vrptw_model_synth_${SIZE}.m"
    TEST_PATH="$DATA_DIR/vrptw_test_homberger_${SIZE}.pkl"
    OUT_LOG="outputs/test_logs_homberger_${SIZE}.pkl"

    if [[ ! -f "$MODEL_PATH" ]]; then
      echo "[WARN] Missing model, skip n=${SIZE}: $MODEL_PATH"
      continue
    fi
    if [[ ! -f "$TEST_PATH" ]]; then
      echo "[WARN] Missing Homberger test pkl, skip n=${SIZE}: $TEST_PATH"
      continue
    fi

    INFER_CFG="$(mktemp /tmp/config_infer_vrptw_synth_${SIZE}.XXXXXX.yaml)"
    "$PYTHON_BIN" - <<PY
import yaml

with open('config_infer_vrptw_homberger.yaml') as f:
    cfg = yaml.safe_load(f)
cfg.setdefault('train', {})['pretrained_fname'] = '$MODEL_PATH'
cfg.setdefault('eval', {})['data_file'] = '$TEST_PATH'
cfg['eval']['detailed_test_log'] = True
with open('$INFER_CFG', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print('infer_cfg=', '$INFER_CFG')
PY

    "$PYTHON_BIN" -m earli.main --config "$INFER_CFG"
    if [[ -f "outputs/test_logs.pkl" ]]; then
      cp "outputs/test_logs.pkl" "$OUT_LOG"
      echo "Saved test log: $OUT_LOG"
        summarize_test_log "$OUT_LOG" "homberger_${SIZE}"
    fi
  done
fi

echo "[DONE] Synthetic curriculum pipeline complete"
