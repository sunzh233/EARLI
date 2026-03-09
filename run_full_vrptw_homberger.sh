#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TRAIN_CONFIG="config_train_vrptw_homberger.yaml"
INFER_CONFIG="config_infer_vrptw_homberger.yaml"
INIT_CONFIG="config_initialization_vrptw_homberger.yaml"
TRAIN_STEPS="512000"
STAGE_TOTAL_STEPS=""
STAGE_N_STEPS=""
SKIP_TRAIN=0
CURRICULUM_STAGES=(200 400 600 800)
DATA_DIR="datasets/homberger_vrptw_curriculum"
MAX_INJECTIONS=""

usage() {
  cat <<'EOF'
Usage:
  ./run_full_vrptw_homberger.sh [options]

Options:
  --train-config PATH   Train config (default: config_train_vrptw_homberger.yaml)
  --infer-config PATH   Inference config (default: config_infer_vrptw_homberger.yaml)
  --init-config PATH    Injection config (default: config_initialization_vrptw_homberger.yaml)
  --train-steps N       Total training steps for earli.train (default: 512000)
  --stage-total-steps L Comma list for per-stage total_steps (e.g. 128000,256000,384000,512000)
  --stage-n-steps L     Comma list for per-stage train.n_steps (e.g. 512,1024,2048,4096)
  --stages S1,S2,...    Curriculum stage sizes (default: 200,400,600,800)
  --max-injections N    Max number of RL solutions injected into cuOpt_RL per problem
  --skip-train          Skip training and reuse configured pretrained model
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-config)
      TRAIN_CONFIG="$2"; shift 2 ;;
    --infer-config)
      INFER_CONFIG="$2"; shift 2 ;;
    --init-config)
      INIT_CONFIG="$2"; shift 2 ;;
    --train-steps)
      TRAIN_STEPS="$2"; shift 2 ;;
    --stage-total-steps)
      STAGE_TOTAL_STEPS="$2"; shift 2 ;;
    --stage-n-steps)
      STAGE_N_STEPS="$2"; shift 2 ;;
    --stages)
      IFS=',' read -r -a CURRICULUM_STAGES <<< "$2"; shift 2 ;;
    --max-injections)
      MAX_INJECTIONS="$2"; shift 2 ;;
    --skip-train)
      SKIP_TRAIN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1 ;;
  esac
done

STAGE_TOTAL_STEPS_ARR=()
if [[ -n "$STAGE_TOTAL_STEPS" ]]; then
  IFS=',' read -r -a STAGE_TOTAL_STEPS_ARR <<< "$STAGE_TOTAL_STEPS"
  if [[ "${#STAGE_TOTAL_STEPS_ARR[@]}" -ne "${#CURRICULUM_STAGES[@]}" ]]; then
    echo "[ERROR] --stage-total-steps length (${#STAGE_TOTAL_STEPS_ARR[@]}) must match --stages length (${#CURRICULUM_STAGES[@]})"
    exit 1
  fi
fi

STAGE_N_STEPS_ARR=()
if [[ -n "$STAGE_N_STEPS" ]]; then
  IFS=',' read -r -a STAGE_N_STEPS_ARR <<< "$STAGE_N_STEPS"
  if [[ "${#STAGE_N_STEPS_ARR[@]}" -ne "${#CURRICULUM_STAGES[@]}" ]]; then
    echo "[ERROR] --stage-n-steps length (${#STAGE_N_STEPS_ARR[@]}) must match --stages length (${#CURRICULUM_STAGES[@]})"
    exit 1
  fi
fi

for cfg in "$TRAIN_CONFIG" "$INFER_CONFIG" "$INIT_CONFIG"; do
  if [[ ! -f "$cfg" ]]; then
    echo "[ERROR] Missing config: $cfg"
    exit 1
  fi
done

echo "[1/4] Prepare Homberger VRPTW datasets"
python tools/prepare_vrptw_homberger_pipeline.py --root "$ROOT_DIR" --train-sizes "${CURRICULUM_STAGES[@]}"

echo "[2/4] Train VRPTW model"
if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  PREV_MODEL=""
  FINAL_MODEL=""
  for IDX in "${!CURRICULUM_STAGES[@]}"; do
    SIZE="${CURRICULUM_STAGES[$IDX]}"
    STAGE_TOTAL_STEP="${TRAIN_STEPS}"
    STAGE_N_STEP=""
    if [[ "${#STAGE_TOTAL_STEPS_ARR[@]}" -gt 0 ]]; then
      STAGE_TOTAL_STEP="${STAGE_TOTAL_STEPS_ARR[$IDX]}"
    fi
    if [[ "${#STAGE_N_STEPS_ARR[@]}" -gt 0 ]]; then
      STAGE_N_STEP="${STAGE_N_STEPS_ARR[$IDX]}"
    fi

    STAGE_CFG="$(mktemp /tmp/config_train_vrptw_stage_${SIZE}.XXXXXX.yaml)"
    MODEL_PATH="outputs/vrptw_model_homberger_${SIZE}.m"
    python - <<PY
import yaml

with open('$TRAIN_CONFIG') as f:
    cfg = yaml.safe_load(f)

cfg.setdefault('eval', {})['data_file'] = '$DATA_DIR/vrptw_train_${SIZE}.pkl'
cfg['eval']['val_data_file'] = '$DATA_DIR/vrptw_val_${SIZE}.pkl'
cfg.setdefault('train', {})['save_model_path'] = '$MODEL_PATH'
prev = '$PREV_MODEL'.strip()
cfg['train']['pretrained_fname'] = prev if prev else 'outputs/vrptw_model_homberger_600.m'
stage_n_steps = '$STAGE_N_STEP'.strip()
if stage_n_steps:
  cfg['train']['n_steps'] = int(stage_n_steps)

with open('$STAGE_CFG', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print('Stage config:', '$STAGE_CFG')
print('Train file  :', cfg['eval']['data_file'])
print('Val file    :', cfg['eval']['val_data_file'])
print('Init model  :', cfg['train']['pretrained_fname'])
print('n_steps     :', cfg['train'].get('n_steps'))
print('Save model  :', cfg['train']['save_model_path'])
PY

  echo "[2/4][stage ${SIZE}] train --total-steps ${STAGE_TOTAL_STEP} (n_steps=${STAGE_N_STEP:-from_config})"
  python -m earli.train --config "$STAGE_CFG" --total-steps "$STAGE_TOTAL_STEP"
    PREV_MODEL="$MODEL_PATH"
    FINAL_MODEL="$MODEL_PATH"
  done

  INFER_CFG_RUNTIME="$(mktemp /tmp/config_infer_vrptw_curriculum.XXXXXX.yaml)"
  python - <<PY
import yaml

with open('$INFER_CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg.setdefault('train', {})['pretrained_fname'] = '$FINAL_MODEL'
cfg.setdefault('eval', {})['data_file'] = '$DATA_DIR/vrptw_test_1000.pkl'
with open('$INFER_CFG_RUNTIME', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print('Inference config:', '$INFER_CFG_RUNTIME')
PY
else
  echo "[SKIP] Training skipped"
  INFER_CFG_RUNTIME="$INFER_CONFIG"
fi

echo "[3/4] Run RL inference to generate injected solutions"
python -m earli.main --config "$INFER_CFG_RUNTIME"

RL_RUNTIME=$(python - <<'PY'
import pickle
import numpy as np

with open('outputs/test_logs.pkl', 'rb') as f:
    logs = pickle.load(f)

v = logs.get('mean_game_clocktime', logs.get('game_clocktime', 0.0))
if isinstance(v, (list, tuple)):
    v = float(np.mean(v)) if len(v) else 0.0
print(float(v))
PY
)

echo "[INFO] Measured RL runtime per problem: ${RL_RUNTIME}s"

echo "[4/4] Run cuOpt vs cuOpt_RL comparison with runtime compensation"
TMP_INIT="$(mktemp /tmp/config_initialization_vrptw_homberger_runtime.XXXXXX.yaml)"
python - <<PY
import yaml

with open('$INIT_CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg.setdefault('injection', {})['rl_runtime'] = float('$RL_RUNTIME')
max_injections = '$MAX_INJECTIONS'.strip()
if max_injections:
  cfg['injection']['max_injections'] = int(max_injections)
with open('$TMP_INIT', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print('Runtime-patched config:', '$TMP_INIT')
if max_injections:
  print('Patched max_injections:', cfg['injection']['max_injections'])
PY

python -c "from earli import test_injection; test_injection.main(config_path='$TMP_INIT')"

echo "[DONE] VRPTW full pipeline completed."
echo "[DONE] Summary PKL is under outputs/test_summary_vrptw_homberger_*.pkl"
