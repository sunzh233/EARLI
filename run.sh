#!/usr/bin/env bash
# =============================================================================
# run.sh -- One-click training + cuOpt vs cuOpt+RL comparison experiment
#
# Usage:
#   ./run.sh ppo          # train with Stable-Baselines3 PPO
#   ./run.sh tree_based   # train with tree-search guided PPO (recommended)
#
# What this script does:
#   1. Parses all benchmark datasets into PKL files.
#        Training data : scale <=500  (Homberger 200+400 / Li&Lim 100+200+400)
#        Test data     : scale ~1000  (Homberger 1000 / Li&Lim 1000)
#   2. Trains RL models on the training data using the chosen pipeline.
#   3. Runs RL inference on the 1000-scale test instances and measures
#      per-problem inference time.
#   4. Runs cuOpt vs cuOpt+RL comparison experiments at two time budgets
#      (20 s and 60 s).  For cuOpt+RL the measured RL inference time is
#      subtracted from the cuOpt solver time so that both methods consume
#      the same total wall time.
#
# Prerequisites:
#   - NVIDIA cuOpt installed (step 4 only; steps 1-3 work without it)
#   - Python environment with earli package installed
#   - Run from the repository root directory
# =============================================================================

set -euo pipefail

PIPELINE="${1:-}"
if [[ "$PIPELINE" != "ppo" && "$PIPELINE" != "tree_based" ]]; then
    echo "Usage: $0 ppo|tree_based"
    echo ""
    echo "  ppo         – Stable-Baselines3 PPO training"
    echo "  tree_based  – Tree-search guided PPO training (higher quality, recommended)"
    exit 1
fi

echo "============================================================"
echo "  EARLI pipeline: $PIPELINE"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# STEP 1 – Prepare datasets
# ---------------------------------------------------------------------------
echo "[1/4] Preparing datasets ..."
mkdir -p datasets outputs

# --- VRPTW training data (Homberger, n <= 500) ---
for SIZE in 200 400; do
    PKL="datasets/vrptw_${SIZE}.pkl"
    DIR="homberger/homberger_${SIZE}_customer_instances"
    if [[ ! -f "$PKL" ]]; then
        echo "  Parsing VRPTW n=${SIZE} → $PKL"
        python -m earli.benchmark_parser vrptw "$DIR" "$PKL"
    else
        echo "  $PKL already exists, skipping."
    fi
done

# --- PDPTW training data (Li&Lim, n <= 500) ---
for SIZE in 100 200 400; do
    PKL="datasets/pdptw_${SIZE}.pkl"
    DIR="li&lim benchmark/pdp_${SIZE}"
    if [[ ! -f "$PKL" ]]; then
        echo "  Parsing PDPTW n=${SIZE} → $PKL"
        python -m earli.benchmark_parser pdptw "$DIR" "$PKL"
    else
        echo "  $PKL already exists, skipping."
    fi
done

# --- VRPTW test data (Homberger n=1000) ---
if [[ ! -f "datasets/vrptw_1000.pkl" ]]; then
    echo "  Parsing VRPTW n=1000 → datasets/vrptw_1000.pkl"
    python -m earli.benchmark_parser vrptw \
        homberger/homberger_1000_customer_instances \
        datasets/vrptw_1000.pkl
else
    echo "  datasets/vrptw_1000.pkl already exists, skipping."
fi

# --- PDPTW test data (Li&Lim n=1000) ---
if [[ ! -f "datasets/pdptw_1000.pkl" ]]; then
    echo "  Parsing PDPTW n=1000 → datasets/pdptw_1000.pkl"
    python -m earli.benchmark_parser pdptw \
        "li&lim benchmark/pdptw1000" \
        datasets/pdptw_1000.pkl
else
    echo "  datasets/pdptw_1000.pkl already exists, skipping."
fi

echo ""

# ---------------------------------------------------------------------------
# STEP 2 – Train RL models
# ---------------------------------------------------------------------------
echo "[2/4] Training RL models (pipeline=$PIPELINE) ..."
echo ""

echo "  Training VRPTW model ..."
python -m earli.train --config "config_vrptw_${PIPELINE}_train_mixed.yaml"

echo ""
echo "  Training PDPTW model ..."
python -m earli.train --config "config_pdptw_${PIPELINE}_train_mixed.yaml"

echo ""

# ---------------------------------------------------------------------------
# STEP 3 – RL inference on 1000-scale test instances
# ---------------------------------------------------------------------------
echo "[3/4] Running RL inference on 1000-scale test instances ..."
echo ""

# -- VRPTW --
echo "  VRPTW n=1000 inference ..."
python -m earli.main --config "config_vrptw_infer_1000_${PIPELINE}.yaml"
cp outputs/test_logs.pkl outputs/vrptw_1000_rl_solutions.pkl

VRPTW_RL_TIME=$(python - <<'PYEOF'
import pickle, sys
try:
    with open('outputs/vrptw_1000_rl_solutions.pkl', 'rb') as f:
        stats = pickle.load(f)
    t = float(stats.get('game_clocktime', 0))
    print(f'{t:.4f}')
except Exception as e:
    print('0.0', file=sys.stderr)
    print('0.0')
PYEOF
)
echo "  VRPTW RL inference time per problem: ${VRPTW_RL_TIME} s"

# -- PDPTW --
echo "  PDPTW n=1000 inference ..."
python -m earli.main --config "config_pdptw_infer_1000_${PIPELINE}.yaml"
cp outputs/test_logs.pkl outputs/pdptw_1000_rl_solutions.pkl

PDPTW_RL_TIME=$(python - <<'PYEOF'
import pickle, sys
try:
    with open('outputs/pdptw_1000_rl_solutions.pkl', 'rb') as f:
        stats = pickle.load(f)
    t = float(stats.get('game_clocktime', 0))
    print(f'{t:.4f}')
except Exception as e:
    print('0.0', file=sys.stderr)
    print('0.0')
PYEOF
)
echo "  PDPTW RL inference time per problem: ${PDPTW_RL_TIME} s"
echo ""

# ---------------------------------------------------------------------------
# STEP 4 – cuOpt vs cuOpt+RL comparison experiments (requires cuOpt)
# ---------------------------------------------------------------------------
echo "[4/4] Running cuOpt vs cuOpt+RL comparison experiments ..."
echo "  Time budgets: 20 s and 60 s"
echo "  cuOpt_RL solver time = budget - RL inference time"
echo ""

# Check whether cuOpt is available
if ! python -c "import cuopt" 2>/dev/null; then
    echo "  WARNING: cuOpt is not installed – skipping comparison experiments."
    echo "  RL solutions are saved to outputs/vrptw_1000_rl_solutions.pkl"
    echo "                         and outputs/pdptw_1000_rl_solutions.pkl"
    echo "  Install cuOpt and re-run step 4 manually (patch rl_runtime first):"
    echo "    python -c \\"
    echo "      \"import yaml; cfg=yaml.safe_load(open('config_vrptw_compare_1000.yaml'));"
    echo "       cfg['injection']['rl_runtime']=<measured_time>;"
    echo "       yaml.dump(cfg, open('/tmp/config_vrptw_compare_1000_runtime.yaml','w'))\""
    echo "    python -m earli.cuopt_injection_vrptw \\"
    echo "        --config /tmp/config_vrptw_compare_1000_runtime.yaml \\"
    echo "        --solutions outputs/vrptw_1000_rl_solutions.pkl"
    echo "    # earli.cuopt_injection_vrptw also handles PDPTW via config problem.env:"
    echo "    python -m earli.cuopt_injection_vrptw \\"
    echo "        --config /tmp/config_pdptw_compare_1000_runtime.yaml \\"
    echo "        --solutions outputs/pdptw_1000_rl_solutions.pkl"
    exit 0
fi

# Helper: patch rl_runtime into a comparison config and write a runtime copy
patch_and_run() {
    local ENV="$1"          # vrptw | pdptw
    local RL_TIME="$2"      # per-problem RL inference time in seconds
    local SOLUTIONS="$3"    # path to RL solutions PKL

    local TEMPLATE="config_${ENV}_compare_1000.yaml"
    local RUNTIME_CFG="/tmp/config_${ENV}_compare_1000_runtime.yaml"

    echo "  Patching rl_runtime=${RL_TIME} s into ${TEMPLATE} ..."
    python - <<PYEOF
import yaml
with open('${TEMPLATE}') as f:
    cfg = yaml.safe_load(f)
cfg['injection']['rl_runtime'] = float('${RL_TIME}')
with open('${RUNTIME_CFG}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
print('  Written:', '${RUNTIME_CFG}')
PYEOF

    echo "  Running ${ENV^^} comparison experiment ..."
    python -m earli.cuopt_injection_vrptw \
        --config "${RUNTIME_CFG}" \
        --solutions "${SOLUTIONS}"
}

patch_and_run vrptw "${VRPTW_RL_TIME}" outputs/vrptw_1000_rl_solutions.pkl
echo ""
patch_and_run pdptw "${PDPTW_RL_TIME}" outputs/pdptw_1000_rl_solutions.pkl

echo ""
echo "============================================================"
echo "  All done!"
echo "  Experiment results saved to ./outputs/"
echo "    VRPTW: outputs/compare_vrptw_1000_compare_vrptw_1000.pkl"
echo "    PDPTW: outputs/compare_pdptw_1000_compare_pdptw_1000.pkl"
echo "============================================================"
