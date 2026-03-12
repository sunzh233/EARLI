#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config_train_vrptw_synthetic_curriculum_hybrid.yaml}"
INFER_CONFIG="${INFER_CONFIG:-config_infer_vrptw_homberger_hybrid.yaml}"
INIT_CONFIG="${INIT_CONFIG:-config_initialization_vrptw_homberger_hybrid.yaml}"
DATA_DIR="${DATA_DIR:-datasets/vrptw_homberger_like_curriculum}"
OUT_DIR="${OUT_DIR:-outputs/hybrid}"
MODEL_DIR="$OUT_DIR/models"

SKIP_DATA=0
SKIP_TRAIN=0
RUN_TEST=0
PROFILE="conservative"
INITIAL_MODEL=""
FIRST_STAGE_NO_KL_ITERS=0

EVAL_SIZES=(200 400 600 800 1000)
EVAL_RUNTIMES=(15 30 45)
RUN_TEST_DONE_SIZES=()

CURRICULUM_STAGES=(50 100 150 200 300 400 500 600 800 1000)
STAGE_STEPS=()
STAGE_N_STEPS=()
STAGE_LRS=()
STAGE_EPOCHS=()
STAGE_BATCH_SIZES=()
STAGE_PARALLEL_PROBLEMS=()
STAGE_ENT_COEFS=()
STAGE_VF_COEFS=()
STAGE_CLIP_RANGES=()
STAGE_TARGET_KLS=()
STAGE_VEHICLE_PENALTIES=()
STAGE_PEN_VEHICLE_OVER_LB=()
STAGE_TARGET_VEHICLE_OVER_LB=()
STAGE_PEN_DEPOT_WITH_CUSTOMER=()
STAGE_TARGET_DEPOT_WITH_CUSTOMER=()

TREE_EPOCHS=()
TREE_BEAMS=()
TREE_DATA_STEPS=()
TREE_BATCH_SIZES=()
TREE_PARALLEL_PROBLEMS=()
TREE_LRS=()

usage() {
	cat <<'EOF'
Usage:
	./run_full_vrptw_synthetic_curriculum_hybrid.sh [options]

Options:
	--python BIN             Python executable (default: python)
	--profile NAME           Preset: conservative|aggressive (default: conservative)
	--train-config PATH      Base hybrid train config
	--infer-config PATH      Hybrid inference config
	--init-config PATH       Hybrid injection config
	--data-dir PATH          Synthetic data directory
	--out-dir PATH           Output root (default: outputs/hybrid)
	--initial-model PATH     Warm-start model for stage 1
	--skip-data              Skip synthetic data generation
	--skip-train             Skip stage training
	--run-test               Run Homberger injection tests
	--first-stage-no-kl-iters K
	                         Disable KL constraint for first-stage first K PPO iterations (default: 0)
	--stages L               CSV sizes, e.g. 50,100,150,...,1000
	--eval-sizes L           CSV sizes for run-test (default: 200,400,600,800,1000)
	--eval-runtimes L        CSV cuOpt runtimes (default: 15,30,45)

	PPO stage overrides:
	--stage-steps L          CSV total PPO steps per stage
	--stage-n-steps L        CSV PPO train.n_steps per stage
	--stage-lrs L            CSV PPO train.learning_rate per stage
	--stage-epochs L         CSV PPO train.epochs per stage
	--stage-batch-sizes L    CSV PPO train.batch_size per stage
	--stage-parallel L       CSV PPO train.n_parallel_problems per stage

	Tree refine overrides:
	--tree-epochs L          CSV tree refine epochs per stage
	--tree-beams L           CSV tree refine n_beams per stage
	--tree-data-steps L      CSV tree muzero.data_steps_per_epoch per stage
	--tree-batch-sizes L     CSV tree train.batch_size per stage
	--tree-parallel L        CSV tree train.n_parallel_problems per stage
	--tree-lrs L             CSV tree train.learning_rate per stage

	-h, --help               Show help
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
			STAGE_STEPS=(400000 340000 300000 260000 220000 180000 140000 110000 90000 90000)
			STAGE_N_STEPS=(224 512 640 704 768 768 704 640 576 576)
			STAGE_LRS=(1.5e-5 6.0e-6 4.8e-6 4.2e-6 3.8e-6 3.4e-6 3.0e-6 2.6e-6 2.2e-6 1.9e-6)
			STAGE_EPOCHS=(8 6 6 5 5 4 4 3 3 3)
			STAGE_BATCH_SIZES=(128 192 192 192 160 128 128 96 64 64)
			STAGE_PARALLEL_PROBLEMS=(12 12 10 8 6 5 4 3 2 1)
			STAGE_ENT_COEFS=(0.0025 0.0022 0.0020 0.0018 0.0016 0.0014 0.0012 0.0010 0.0008 0.0008)
			STAGE_VF_COEFS=(0.70 0.70 0.68 0.66 0.64 0.62 0.60 0.58 0.56 0.56)
			STAGE_CLIP_RANGES=(0.15 0.14 0.13 0.12 0.12 0.11 0.11 0.10 0.10 0.10)
			# Relaxed-then-tight schedule: allow larger policy moves early, then stabilize.
			STAGE_TARGET_KLS=(0.24 0.12 0.10 0.085 0.07 0.06 0.05 0.045 0.04 0.035)
			STAGE_VEHICLE_PENALTIES=(10 11 12 14 16 18 19 20 22 24)
			STAGE_PEN_VEHICLE_OVER_LB=(0.08 0.10 0.12 0.16 0.20 0.24 0.28 0.32 0.36 0.40)
			STAGE_TARGET_VEHICLE_OVER_LB=(0.20 0.16 0.14 0.12 0.10 0.08 0.06 0.05 0.04 0.03)
			STAGE_PEN_DEPOT_WITH_CUSTOMER=(0.015 0.020 0.025 0.035 0.045 0.055 0.065 0.075 0.085 0.095)
			STAGE_TARGET_DEPOT_WITH_CUSTOMER=(0.02 0.01 0.01 0.0 0.0 0.0 0.0 0.0 0.0 0.0)

			TREE_EPOCHS=(3 3 3 3 3 2 2 2 2 2)
			TREE_BEAMS=(16 16 16 16 12 12 12 10 8 8)
			TREE_DATA_STEPS=(320 320 320 320 288 288 256 256 224 192)
			TREE_BATCH_SIZES=(512 512 512 512 384 384 320 256 192 128)
			TREE_PARALLEL_PROBLEMS=(6 6 5 4 3 3 2 2 1 1)
			TREE_LRS=(1.0e-4 8.0e-5 7.0e-5 6.0e-5 5.0e-5 4.5e-5 4.0e-5 3.5e-5 3.0e-5 2.5e-5)
			;;
		aggressive)
			STAGE_STEPS=(600000 520000 430000 340000 260000 210000 170000 140000 110000 90000)
			STAGE_N_STEPS=(256 512 640 768 832 832 768 704 640 576)
			STAGE_LRS=(2.0e-5 9.0e-6 6.8e-6 5.8e-6 5.0e-6 4.3e-6 3.7e-6 3.1e-6 2.5e-6 2.1e-6)
			STAGE_EPOCHS=(10 8 7 6 5 5 4 4 3 3)
			STAGE_BATCH_SIZES=(128 192 192 192 160 128 128 96 64 64)
			STAGE_PARALLEL_PROBLEMS=(12 12 10 8 6 5 4 3 2 1)
			STAGE_ENT_COEFS=(0.0060 0.0055 0.0050 0.0045 0.0040 0.0035 0.0030 0.0025 0.0020 0.0020)
			STAGE_VF_COEFS=(0.45 0.45 0.44 0.43 0.42 0.41 0.40 0.40 0.38 0.38)
			STAGE_CLIP_RANGES=(0.12 0.12 0.11 0.11 0.10 0.10 0.10 0.09 0.09 0.09)
			# Aggressive profile keeps higher KL budget in early stages to avoid over-constraining updates.
			STAGE_TARGET_KLS=(0.25 0.08 0.07 0.06 0.05 0.045 0.04 0.035 0.03 0.025)
			STAGE_VEHICLE_PENALTIES=(10 12 13 15 17 19 21 23 25 27)
			STAGE_PEN_VEHICLE_OVER_LB=(0.10 0.12 0.14 0.18 0.22 0.26 0.30 0.34 0.38 0.42)
			STAGE_TARGET_VEHICLE_OVER_LB=(0.18 0.14 0.12 0.10 0.08 0.06 0.05 0.04 0.03 0.02)
			STAGE_PEN_DEPOT_WITH_CUSTOMER=(0.020 0.025 0.030 0.040 0.050 0.060 0.070 0.080 0.090 0.100)
			STAGE_TARGET_DEPOT_WITH_CUSTOMER=(0.10 0.08 0.07 0.05 0.04 0.03 0.02 0.02 0.01 0.01)

			TREE_EPOCHS=(4 4 4 3 3 3 2 2 2 2)
			TREE_BEAMS=(20 20 18 18 16 14 12 10 8 8)
			TREE_DATA_STEPS=(384 384 384 352 320 320 288 256 224 192)
			TREE_BATCH_SIZES=(512 512 512 512 448 384 320 256 192 128)
			TREE_PARALLEL_PROBLEMS=(6 6 5 4 3 3 2 2 1 1)
			TREE_LRS=(1.2e-4 1.0e-4 8.5e-5 7.2e-5 6.2e-5 5.2e-5 4.6e-5 4.0e-5 3.2e-5 2.8e-5)
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
	infer_cfg="$(mktemp /tmp/config_infer_vrptw_hybrid_${size}.XXXXXX.yaml)"
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
		summarize_test_log "$out_log" "hybrid_homberger_${size}"
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
	init_cfg="$(mktemp /tmp/config_init_vrptw_hybrid_${size}.XXXXXX.yaml)"

	"$PYTHON_BIN" - <<PY
import yaml

with open('$INIT_CONFIG') as f:
		cfg = yaml.safe_load(f)

cfg.setdefault('names', {})['run_title'] = 'homberger_hybrid_${size}'
cfg['names']['methods'] = ['cuOpt', 'cuOpt_RL']
cfg['names']['baseline'] = 'cuOpt'
cfg['names']['main_method'] = 'cuOpt_RL'

cfg.setdefault('paths', {})['problems'] = '$test_path'
cfg['paths']['solutions'] = '$out_log'
cfg['paths']['out_summary'] = 'test_summary_homberger_injection_hybrid'

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
		--first-stage-no-kl-iters)
			FIRST_STAGE_NO_KL_ITERS="$2"; shift 2 ;;
		--stages)
			parse_csv_to_array "$2" CURRICULUM_STAGES; shift 2 ;;
		--eval-sizes)
			parse_csv_to_array "$2" EVAL_SIZES; shift 2 ;;
		--eval-runtimes)
			parse_csv_to_array "$2" EVAL_RUNTIMES; shift 2 ;;
		--stage-steps)
			parse_csv_to_array "$2" STAGE_STEPS; shift 2 ;;
		--stage-n-steps)
			parse_csv_to_array "$2" STAGE_N_STEPS; shift 2 ;;
		--stage-lrs)
			parse_csv_to_array "$2" STAGE_LRS; shift 2 ;;
		--stage-epochs)
			parse_csv_to_array "$2" STAGE_EPOCHS; shift 2 ;;
		--stage-batch-sizes)
			parse_csv_to_array "$2" STAGE_BATCH_SIZES; shift 2 ;;
		--stage-parallel)
			parse_csv_to_array "$2" STAGE_PARALLEL_PROBLEMS; shift 2 ;;
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
		-h|--help)
			usage; exit 0 ;;
		*)
			echo "Unknown option: $1"
			usage
			exit 1 ;;
	esac
done

set_profile_defaults "$PROFILE"

check_len STAGE_STEPS "${#STAGE_STEPS[@]}"
check_len STAGE_N_STEPS "${#STAGE_N_STEPS[@]}"
check_len STAGE_LRS "${#STAGE_LRS[@]}"
check_len STAGE_EPOCHS "${#STAGE_EPOCHS[@]}"
check_len STAGE_BATCH_SIZES "${#STAGE_BATCH_SIZES[@]}"
check_len STAGE_PARALLEL_PROBLEMS "${#STAGE_PARALLEL_PROBLEMS[@]}"
check_len STAGE_ENT_COEFS "${#STAGE_ENT_COEFS[@]}"
check_len STAGE_VF_COEFS "${#STAGE_VF_COEFS[@]}"
check_len STAGE_CLIP_RANGES "${#STAGE_CLIP_RANGES[@]}"
check_len STAGE_TARGET_KLS "${#STAGE_TARGET_KLS[@]}"
check_len STAGE_VEHICLE_PENALTIES "${#STAGE_VEHICLE_PENALTIES[@]}"
check_len STAGE_PEN_VEHICLE_OVER_LB "${#STAGE_PEN_VEHICLE_OVER_LB[@]}"
check_len STAGE_TARGET_VEHICLE_OVER_LB "${#STAGE_TARGET_VEHICLE_OVER_LB[@]}"
check_len STAGE_PEN_DEPOT_WITH_CUSTOMER "${#STAGE_PEN_DEPOT_WITH_CUSTOMER[@]}"
check_len STAGE_TARGET_DEPOT_WITH_CUSTOMER "${#STAGE_TARGET_DEPOT_WITH_CUSTOMER[@]}"
check_len TREE_EPOCHS "${#TREE_EPOCHS[@]}"
check_len TREE_BEAMS "${#TREE_BEAMS[@]}"
check_len TREE_DATA_STEPS "${#TREE_DATA_STEPS[@]}"
check_len TREE_BATCH_SIZES "${#TREE_BATCH_SIZES[@]}"
check_len TREE_PARALLEL_PROBLEMS "${#TREE_PARALLEL_PROBLEMS[@]}"
check_len TREE_LRS "${#TREE_LRS[@]}"

# Value-level checks
validate_int_array_positive CURRICULUM_STAGES "${CURRICULUM_STAGES[@]}"
validate_int_array_positive STAGE_STEPS "${STAGE_STEPS[@]}"
validate_int_array_positive STAGE_N_STEPS "${STAGE_N_STEPS[@]}"
validate_int_array_positive STAGE_EPOCHS "${STAGE_EPOCHS[@]}"
validate_int_array_positive STAGE_BATCH_SIZES "${STAGE_BATCH_SIZES[@]}"
validate_int_array_positive STAGE_PARALLEL_PROBLEMS "${STAGE_PARALLEL_PROBLEMS[@]}"

validate_float_array_range STAGE_LRS 1e-9 1.0 "${STAGE_LRS[@]}"
validate_float_array_range STAGE_ENT_COEFS 0.0 0.1 "${STAGE_ENT_COEFS[@]}"
validate_float_array_range STAGE_VF_COEFS 0.1 2.0 "${STAGE_VF_COEFS[@]}"
validate_float_array_range STAGE_CLIP_RANGES 0.01 0.5 "${STAGE_CLIP_RANGES[@]}"
validate_float_array_range STAGE_TARGET_KLS 0.001 1.0 "${STAGE_TARGET_KLS[@]}"
validate_float_array_range STAGE_VEHICLE_PENALTIES 1.0 200.0 "${STAGE_VEHICLE_PENALTIES[@]}"
validate_float_array_range STAGE_PEN_VEHICLE_OVER_LB 0.0 5.0 "${STAGE_PEN_VEHICLE_OVER_LB[@]}"
validate_float_array_range STAGE_TARGET_VEHICLE_OVER_LB 0.0 1.0 "${STAGE_TARGET_VEHICLE_OVER_LB[@]}"
validate_float_array_range STAGE_PEN_DEPOT_WITH_CUSTOMER 0.0 2.0 "${STAGE_PEN_DEPOT_WITH_CUSTOMER[@]}"
validate_float_array_range STAGE_TARGET_DEPOT_WITH_CUSTOMER 0.0 1.0 "${STAGE_TARGET_DEPOT_WITH_CUSTOMER[@]}"

validate_int_array_positive TREE_EPOCHS "${TREE_EPOCHS[@]}"
validate_int_array_positive TREE_BEAMS "${TREE_BEAMS[@]}"
validate_int_array_positive TREE_DATA_STEPS "${TREE_DATA_STEPS[@]}"
validate_int_array_positive TREE_BATCH_SIZES "${TREE_BATCH_SIZES[@]}"
validate_int_array_positive TREE_PARALLEL_PROBLEMS "${TREE_PARALLEL_PROBLEMS[@]}"
validate_float_array_range TREE_LRS 1e-9 1.0 "${TREE_LRS[@]}"

validate_int_array_positive EVAL_SIZES "${EVAL_SIZES[@]}"
validate_int_array_positive EVAL_RUNTIMES "${EVAL_RUNTIMES[@]}"

"$PYTHON_BIN" - "$FIRST_STAGE_NO_KL_ITERS" <<'PY'
import sys
v = sys.argv[1]
try:
	x = int(v)
except Exception:
	raise SystemExit(f"[ERROR] --first-stage-no-kl-iters must be integer, got: {v}")
if x < 0:
 	raise SystemExit(f"[ERROR] --first-stage-no-kl-iters must be >= 0, got: {x}")
print(f"[CHECK] first_stage_no_kl_iters: {x}")
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

echo "[INFO] Hybrid outputs in: $OUT_DIR"

if [[ "$SKIP_TRAIN" -eq 0 ]]; then
	echo "[2/3] Stage training: PPO then tree_based refine"
	PREV_MODEL="$INITIAL_MODEL"

	if [[ -n "$PREV_MODEL" && ! -f "$PREV_MODEL" ]]; then
		echo "[ERROR] --initial-model file not found: $PREV_MODEL"
		exit 1
	fi

	for i in "${!CURRICULUM_STAGES[@]}"; do
		SIZE="${CURRICULUM_STAGES[$i]}"

		TOTAL_STEPS="${STAGE_STEPS[$i]}"
		PPO_N_STEPS="${STAGE_N_STEPS[$i]}"
		PPO_LR="${STAGE_LRS[$i]}"
		PPO_EPOCHS="${STAGE_EPOCHS[$i]}"
		PPO_BATCH="${STAGE_BATCH_SIZES[$i]}"
		PPO_PARALLEL="${STAGE_PARALLEL_PROBLEMS[$i]}"
		PPO_ENT="${STAGE_ENT_COEFS[$i]}"
		PPO_VF="${STAGE_VF_COEFS[$i]}"
		PPO_CLIP="${STAGE_CLIP_RANGES[$i]}"
		PPO_TKL="${STAGE_TARGET_KLS[$i]}"
		PPO_VEH_PEN="${STAGE_VEHICLE_PENALTIES[$i]}"
		PPO_PEN_OVER_LB="${STAGE_PEN_VEHICLE_OVER_LB[$i]}"
		PPO_TGT_OVER_LB="${STAGE_TARGET_VEHICLE_OVER_LB[$i]}"
		PPO_PEN_DEPOT="${STAGE_PEN_DEPOT_WITH_CUSTOMER[$i]}"
		PPO_TGT_DEPOT="${STAGE_TARGET_DEPOT_WITH_CUSTOMER[$i]}"

		T_EPOCHS="${TREE_EPOCHS[$i]}"
		T_BEAMS="${TREE_BEAMS[$i]}"
		T_DATA_STEPS="${TREE_DATA_STEPS[$i]}"
		T_BATCH="${TREE_BATCH_SIZES[$i]}"
		T_PARALLEL="${TREE_PARALLEL_PROBLEMS[$i]}"
		T_LR="${TREE_LRS[$i]}"

		TRAIN_PKL="$DATA_DIR/vrptw_train_${SIZE}.pkl"
		VAL_PKL="$DATA_DIR/vrptw_val_${SIZE}.pkl"

		PPO_MODEL="$MODEL_DIR/vrptw_model_synth_hybrid_ppo_${SIZE}.m"
		TREE_MODEL="$MODEL_DIR/vrptw_model_synth_hybrid_${SIZE}.m"

		PPO_PHASE_PRETRAIN="$PREV_MODEL"
		PPO_PHASE_STEPS="$TOTAL_STEPS"
		PPO_PHASE_EPOCHS="$PPO_EPOCHS"
		PPO_PHASE_TKL="$PPO_TKL"

		# Optional first-stage warmup: first K PPO iterations without KL, then resume with configured KL.
		if [[ "$i" -eq 0 && "$FIRST_STAGE_NO_KL_ITERS" -gt 0 ]]; then
			read -r WARMUP_STEPS MAIN_STEPS ROLLOUT_UNIT <<< "$($PYTHON_BIN - <<PY
total_steps = int('$TOTAL_STEPS')
rollout_unit = int('$PPO_N_STEPS') * int('$PPO_PARALLEL')
warmup_iters = int('$FIRST_STAGE_NO_KL_ITERS')
if rollout_unit <= 0 or warmup_iters <= 0:
    print('0', total_steps, max(1, rollout_unit))
    raise SystemExit
warmup_steps = warmup_iters * rollout_unit
warmup_steps = max(rollout_unit, warmup_steps)
max_warmup = total_steps - rollout_unit
if max_warmup < rollout_unit:
    warmup_steps = 0
else:
    warmup_steps = min(warmup_steps, max_warmup)
main_steps = total_steps - warmup_steps
if warmup_steps <= 0 or main_steps <= 0:
    print('0', total_steps, rollout_unit)
else:
    print(warmup_steps, main_steps, rollout_unit)
PY
)"

			if [[ "$WARMUP_STEPS" -gt 0 && "$MAIN_STEPS" -gt 0 ]]; then
				WARMUP_MODEL="$MODEL_DIR/vrptw_model_synth_hybrid_ppo_${SIZE}_warmup.m"
				PPO_WARMUP_CFG="$(mktemp /tmp/config_train_vrptw_hybrid_ppo_${SIZE}_warmup.XXXXXX.yaml)"
				"$PYTHON_BIN" - <<PY
import yaml

with open('$TRAIN_CONFIG') as f:
	cfg = yaml.safe_load(f)

cfg.setdefault('system', {})['compatibility_mode'] = 'stable_baselines'
cfg['system']['use_tensordict'] = False
cfg['system']['tensorboard_logdir'] = '$OUT_DIR/tensorboard'
cfg['system']['run_name'] = 'hybrid_ppo_stage_${SIZE}_warmup_no_kl'

cfg.setdefault('eval', {})['data_file'] = '$TRAIN_PKL'
cfg['eval']['val_data_file'] = '$VAL_PKL'
cfg['eval']['sampling_mode'] = 'random_with_replacement'
cfg['eval'].pop('data_files', None)

cfg.setdefault('train', {})['method'] = 'ppo'
cfg['train']['save_model_path'] = '$WARMUP_MODEL'
cfg['train']['pretrained_fname'] = '$PPO_PHASE_PRETRAIN' if '$PPO_PHASE_PRETRAIN' else None
cfg['train']['n_parallel_problems'] = int('$PPO_PARALLEL')
cfg['train']['n_steps'] = int('$PPO_N_STEPS')
cfg['train']['learning_rate'] = float('$PPO_LR')
cfg['train']['epochs'] = int('$PPO_EPOCHS')
cfg['train']['batch_size'] = int('$PPO_BATCH')
cfg['train']['ent_coef'] = float('$PPO_ENT')
cfg['train']['vf_coef'] = float('$PPO_VF')
cfg['train']['clip_range'] = float('$PPO_CLIP')
cfg['train']['target_kl'] = None
cfg['train']['sampling_mode'] = 'random_with_replacement'

ps = cfg.setdefault('problem_setup', {})
ps['vehicle_penalty'] = float('$PPO_VEH_PEN')
ps['penalty_vehicle_over_lb'] = float('$PPO_PEN_OVER_LB')
ps['target_vehicle_over_lb'] = float('$PPO_TGT_OVER_LB')
ps['penalty_depot_with_customer'] = float('$PPO_PEN_DEPOT')
ps['target_depot_with_customer'] = float('$PPO_TGT_DEPOT')

with open('$PPO_WARMUP_CFG', 'w') as f:
	yaml.safe_dump(cfg, f, sort_keys=False)

print('ppo_warmup_cfg=', '$PPO_WARMUP_CFG')
print('stage=', '$SIZE', 'save_model=', '$WARMUP_MODEL')
PY

				echo "[2/3][n=${SIZE}] PPO warmup(no KL): steps=${WARMUP_STEPS}, iters=${FIRST_STAGE_NO_KL_ITERS}, n_steps=${PPO_N_STEPS}, parallel=${PPO_PARALLEL}, rollout_unit=${ROLLOUT_UNIT}"
				"$PYTHON_BIN" -m earli.train --config "$PPO_WARMUP_CFG" --total-steps "$WARMUP_STEPS"

				if [[ -f "$WARMUP_MODEL" ]]; then
					PPO_PHASE_PRETRAIN="$WARMUP_MODEL"
					PPO_PHASE_STEPS="$MAIN_STEPS"
					echo "[2/3][n=${SIZE}] PPO main(with KL): steps=${MAIN_STEPS}, epochs=${PPO_EPOCHS}, target_kl=${PPO_TKL}"
				else
					echo "[WARN] Missing warmup model for n=${SIZE}: $WARMUP_MODEL; fallback to single constrained PPO run"
				fi
			else
				echo "[WARN] Cannot split first stage into warmup/main with current step budget; fallback to single constrained PPO run"
			fi
		fi

		PPO_CFG="$(mktemp /tmp/config_train_vrptw_hybrid_ppo_${SIZE}.XXXXXX.yaml)"
		"$PYTHON_BIN" - <<PY
import yaml

with open('$TRAIN_CONFIG') as f:
		cfg = yaml.safe_load(f)

cfg.setdefault('system', {})['compatibility_mode'] = 'stable_baselines'
cfg['system']['use_tensordict'] = False
cfg['system']['tensorboard_logdir'] = '$OUT_DIR/tensorboard'
cfg['system']['run_name'] = 'hybrid_ppo_stage_$SIZE'

cfg.setdefault('eval', {})['data_file'] = '$TRAIN_PKL'
cfg['eval']['val_data_file'] = '$VAL_PKL'
cfg['eval']['sampling_mode'] = 'random_with_replacement'
cfg['eval'].pop('data_files', None)

cfg.setdefault('train', {})['method'] = 'ppo'
cfg['train']['save_model_path'] = '$PPO_MODEL'
cfg['train']['pretrained_fname'] = '$PPO_PHASE_PRETRAIN' if '$PPO_PHASE_PRETRAIN' else None
cfg['train']['n_parallel_problems'] = int('$PPO_PARALLEL')
cfg['train']['n_steps'] = int('$PPO_N_STEPS')
cfg['train']['learning_rate'] = float('$PPO_LR')
cfg['train']['epochs'] = int('$PPO_PHASE_EPOCHS')
cfg['train']['batch_size'] = int('$PPO_BATCH')
cfg['train']['ent_coef'] = float('$PPO_ENT')
cfg['train']['vf_coef'] = float('$PPO_VF')
cfg['train']['clip_range'] = float('$PPO_CLIP')
cfg['train']['target_kl'] = float('$PPO_PHASE_TKL')
cfg['train']['sampling_mode'] = 'random_with_replacement'

ps = cfg.setdefault('problem_setup', {})
ps['vehicle_penalty'] = float('$PPO_VEH_PEN')
ps['penalty_vehicle_over_lb'] = float('$PPO_PEN_OVER_LB')
ps['target_vehicle_over_lb'] = float('$PPO_TGT_OVER_LB')
ps['penalty_depot_with_customer'] = float('$PPO_PEN_DEPOT')
ps['target_depot_with_customer'] = float('$PPO_TGT_DEPOT')

with open('$PPO_CFG', 'w') as f:
		yaml.safe_dump(cfg, f, sort_keys=False)

print('ppo_cfg=', '$PPO_CFG')
print('stage=', '$SIZE', 'save_model=', '$PPO_MODEL')
PY

		echo "[2/3][n=${SIZE}] PPO train: steps=${PPO_PHASE_STEPS}, epochs=${PPO_PHASE_EPOCHS}, n_steps=${PPO_N_STEPS}, parallel=${PPO_PARALLEL}, lr=${PPO_LR}, ent=${PPO_ENT}, vf=${PPO_VF}, clip=${PPO_CLIP}, target_kl=${PPO_PHASE_TKL}, vehicle_penalty=${PPO_VEH_PEN}, pen_over_lb=${PPO_PEN_OVER_LB}, target_over_lb=${PPO_TGT_OVER_LB}, pen_depot=${PPO_PEN_DEPOT}, target_depot=${PPO_TGT_DEPOT}"
		"$PYTHON_BIN" -m earli.train --config "$PPO_CFG" --total-steps "$PPO_PHASE_STEPS"

		if [[ ! -f "$PPO_MODEL" ]]; then
			echo "[WARN] Missing PPO model for n=${SIZE}: $PPO_MODEL"
			continue
		fi

		TREE_CFG="$(mktemp /tmp/config_train_vrptw_hybrid_tree_${SIZE}.XXXXXX.yaml)"
		"$PYTHON_BIN" - <<PY
import yaml

with open('$TRAIN_CONFIG') as f:
		cfg = yaml.safe_load(f)

cfg.setdefault('system', {})['compatibility_mode'] = None
cfg['system']['use_tensordict'] = True
cfg['system']['save_obs_on_gpu'] = True
cfg['system']['tensorboard_logdir'] = '$OUT_DIR/tensorboard'
cfg['system']['run_name'] = 'hybrid_tree_stage_$SIZE'

cfg.setdefault('eval', {})['data_file'] = '$TRAIN_PKL'
cfg['eval']['val_data_file'] = '$VAL_PKL'
cfg['eval']['sampling_mode'] = 'random_with_replacement'
cfg['eval']['deterministic_test_beam'] = False
cfg['eval']['apply_local_search'] = False
cfg['eval'].pop('data_files', None)

cfg.setdefault('train', {})['method'] = 'tree_based'
cfg['train']['pretrained_fname'] = '$PPO_MODEL'
cfg['train']['save_model_path'] = '$TREE_MODEL'
cfg['train']['epochs'] = int('$T_EPOCHS')
cfg['train']['n_beams'] = int('$T_BEAMS')
cfg['train']['batch_size'] = int('$T_BATCH')
cfg['train']['n_parallel_problems'] = int('$T_PARALLEL')
cfg['train']['learning_rate'] = float('$T_LR')

cfg.setdefault('muzero', {})['expansion_method'] = 'KPPO'
cfg['muzero']['data_steps_per_epoch'] = int('$T_DATA_STEPS')
cfg['muzero']['deterministic_branch_in_k_beams'] = False
cfg['muzero']['max_leaves'] = int(max(12, int('$T_BEAMS')))
cfg['muzero']['max_moves'] = int(max(8000, int('$SIZE') * 10))

cfg.setdefault('sampler', {})['temperature'] = 0.2
cfg['sampler']['diversity_penalty'] = 0.01

ps = cfg.setdefault('problem_setup', {})
ps['vehicle_penalty'] = float('$PPO_VEH_PEN')
ps['penalty_vehicle_over_lb'] = float('$PPO_PEN_OVER_LB')
ps['target_vehicle_over_lb'] = float('$PPO_TGT_OVER_LB')
ps['penalty_depot_with_customer'] = float('$PPO_PEN_DEPOT')
ps['target_depot_with_customer'] = float('$PPO_TGT_DEPOT')

with open('$TREE_CFG', 'w') as f:
		yaml.safe_dump(cfg, f, sort_keys=False)

print('tree_cfg=', '$TREE_CFG')
print('stage=', '$SIZE', 'save_model=', '$TREE_MODEL')
PY

		echo "[2/3][n=${SIZE}] Tree refine: epochs=${T_EPOCHS}, beams=${T_BEAMS}, data_steps=${T_DATA_STEPS}, parallel=${T_PARALLEL}, lr=${T_LR}"
		"$PYTHON_BIN" -m earli.train --config "$TREE_CFG"

		if [[ -f "$TREE_MODEL" ]]; then
			PREV_MODEL="$TREE_MODEL"
		else
			echo "[WARN] Missing tree-refined model for n=${SIZE}: $TREE_MODEL"
			PREV_MODEL="$PPO_MODEL"
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
		final_model="$MODEL_DIR/vrptw_model_synth_hybrid_${size}.m"
		run_homberger_test_size "$size" "$final_model"
	done
fi

echo "[DONE] Hybrid PPO+Tree curriculum pipeline complete"
