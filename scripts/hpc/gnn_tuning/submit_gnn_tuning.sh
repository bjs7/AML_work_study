#!/bin/bash
# =============================================================================
# GNN Parallel Hyperparameter Tuning — Pipeline Coordinator
# =============================================================================
# Submits the full two-stage tuning pipeline as a chain of dependent SLURM
# jobs. All jobs are submitted upfront; SLURM dependencies ensure correct
# ordering even if individual jobs spend time in the queue.
#
# Pipeline:
#   1. Generate stage 1 HP configs locally (fast, runs here on login node)
#   2. Stage 1 array   — N parallel HP evaluations
#   3. Stage 1 aggregator (afterok stage 1) — picks top 5, narrows intervals,
#                                             writes stage2_configs.json
#   4. Stage 2 array   (afterok aggregator) — N parallel HP evaluations
#   5. Final selection (afterok stage 2)    — picks best, saves to tuned HP file
#
# Usage:
#   bash scripts/hpc/gnn_tuning/submit_gnn_tuning.sh [size] [ir] [eval_mode]
#
# Examples:
#   bash scripts/hpc/gnn_tuning/submit_gnn_tuning.sh small HI system
#   bash scripts/hpc/gnn_tuning/submit_gnn_tuning.sh medium LO system
#
# Logs are written to $VSC_DATA/AML_work_study/batch_jobs/logs/.
# After this pipeline completes, run train_gnn_full_info.sh to train 4 seeds.
# =============================================================================

set -e  # Exit on first error

SIZE=${1:-small}
IR=${2:-HI}
EVAL_MODE=${3:-system}

# All gnn_tuning scripts (both .py and .sh) live together in batch_jobs/gnn_tuning/
BATCH_JOBS_DIR="$VSC_DATA/AML_work_study/batch_jobs"
SCRIPT_DIR="$BATCH_JOBS_DIR/gnn_tuning"

# Project directory — needed only for cd before running Python
PROJECT_DIR="$VSC_DATA/AML_work_study/AML_work_study"

# Single logs folder for all jobs; created here before any sbatch call so
# SLURM can open output files immediately (it does not create missing dirs)
LOGS_DIR="$BATCH_JOBS_DIR/gnn_tuning/logs"
mkdir -p "$LOGS_DIR"

# Args forwarded to all Python scripts and SLURM jobs
DATA_FLAGS="--fl_algo full_info --model GINe --size $SIZE --ir $IR --eval_mode $EVAL_MODE --batching --emlps"

export PATH=$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc

echo "======================================================================"
echo "GNN Parallel Tuning Pipeline"
echo "  Dataset   : $SIZE / $IR"
echo "  Eval mode : $EVAL_MODE"
echo "  Project   : $PROJECT_DIR"
echo "  Logs      : $LOGS_DIR"
echo "======================================================================"

# Step 0 — Generate stage 1 HP configs (runs locally, no GPU needed)
echo ""
echo "[0/4] Generating stage 1 HP configs..."
cd "$PROJECT_DIR"
GEN_OUTPUT=$(python "$SCRIPT_DIR/generate_hp_configs.py" $DATA_FLAGS 2>/dev/null)
echo "$GEN_OUTPUT"

N_CONFIGS=$(echo "$GEN_OUTPUT" | grep "^N_CONFIGS=" | cut -d= -f2)
if [ -z "$N_CONFIGS" ] || [ "$N_CONFIGS" -lt 1 ]; then
    echo "ERROR: Could not determine number of HP configs from generate_hp_configs.py output."
    exit 1
fi

ARRAY_SPEC="0-$((N_CONFIGS - 1))"
echo "Array spec: --array=$ARRAY_SPEC  ($N_CONFIGS configs)"

# Step 1 — Submit stage 1 evaluation array
echo ""
echo "[1/4] Submitting stage 1 array ($N_CONFIGS tasks)..."
JID1=$(sbatch \
    --array="$ARRAY_SPEC" \
    --output="$LOGS_DIR/gnn_hp_eval_%A_%a.log" \
    --error="$LOGS_DIR/gnn_hp_eval_%A_%a.err" \
    "$SCRIPT_DIR/run_hp_eval.sh" \
    --stage 1 $DATA_FLAGS \
    | awk '{print $4}')
echo "  Stage 1 array job ID : $JID1"

# Step 2 — Submit stage 1 aggregator (waits for ALL stage 1 tasks)
echo ""
echo "[2/4] Submitting stage 1 aggregator (depends on $JID1)..."
JID_AGG1=$(sbatch \
    --dependency=afterok:$JID1 \
    --output="$LOGS_DIR/gnn_aggregate1_%j.log" \
    --error="$LOGS_DIR/gnn_aggregate1_%j.err" \
    "$SCRIPT_DIR/run_aggregate.sh" \
    --stage 1 $DATA_FLAGS \
    | awk '{print $4}')
echo "  Stage 1 aggregator job ID : $JID_AGG1"

# Step 3 — Submit stage 2 evaluation array (waits for aggregator to write configs)
echo ""
echo "[3/4] Submitting stage 2 array ($N_CONFIGS tasks, depends on $JID_AGG1)..."
JID2=$(sbatch \
    --dependency=afterok:$JID_AGG1 \
    --array="$ARRAY_SPEC" \
    --output="$LOGS_DIR/gnn_hp_eval_%A_%a.log" \
    --error="$LOGS_DIR/gnn_hp_eval_%A_%a.err" \
    "$SCRIPT_DIR/run_hp_eval.sh" \
    --stage 2 $DATA_FLAGS \
    | awk '{print $4}')
echo "  Stage 2 array job ID : $JID2"

# Step 4 — Submit final selection (waits for ALL stage 2 tasks)
echo ""
echo "[4/4] Submitting final selection (depends on $JID2)..."
JID_FINAL=$(sbatch \
    --dependency=afterok:$JID2 \
    --output="$LOGS_DIR/gnn_final_%j.log" \
    --error="$LOGS_DIR/gnn_final_%j.err" \
    "$SCRIPT_DIR/run_aggregate.sh" \
    --final $DATA_FLAGS \
    | awk '{print $4}')
echo "  Final selection job ID : $JID_FINAL"

echo ""
echo "======================================================================"
echo "Full pipeline submitted:"
echo "  [1] Stage 1 eval  ($N_CONFIGS parallel) : $JID1"
echo "  [2] Stage 1 aggregate                    : $JID_AGG1"
echo "  [3] Stage 2 eval  ($N_CONFIGS parallel) : $JID2"
echo "  [4] Final selection                      : $JID_FINAL"
echo ""
echo "Monitor : squeue -u \$USER"
echo "Logs    : $LOGS_DIR/"
echo ""
echo "When $JID_FINAL completes, run:"
echo "  sbatch $BATCH_JOBS_DIR/train_gnn_full_info.sh $SIZE $IR $EVAL_MODE"
echo "======================================================================"
