#!/bin/bash
# =============================================================================
# Booster Timing Test — estimates SecureBoost walltime before a full run.
#
#   Runs SecureBoost on full data with --sb_num_rounds 5 and 1 seed.
#   Extrapolate: full_time ≈ measured_time × (num_rounds / 5) × 4 seeds
#
# Usage:
#   bash scripts/run_booster_timing_test.sh [eval_mode]
#
# Examples:
#   bash scripts/run_booster_timing_test.sh comparable   # default
#   bash scripts/run_booster_timing_test.sh system
# =============================================================================

SIZE="small"
IR="HI"
MODEL="xgboost"
EVAL_MODE=${1:-comparable}

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc"

PYTHON="python \$VSC_DATA/AML_work_study/AML_work_study/main.py"

mkdir -p logs

SCENARIO="SecureBoost"
JOB="aml_booster_timing_${SCENARIO}_${SIZE}_${IR}_${EVAL_MODE}"
FLAGS="--model $MODEL --size $SIZE --ir $IR --eval_mode $EVAL_MODE --testing_seeds 1 --sb_num_rounds 5"

echo "Submitting SecureBoost timing test (${SIZE} / ${IR} / eval_mode=${EVAL_MODE})"

sbatch \
    -M genius \
    --account=lp_aml_work_study \
    --partition=batch \
    --job-name="$JOB" \
    --output="logs/${JOB}_%j.log" \
    --error="logs/${JOB}_%j.err" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=32 \
    --mem=160G \
    --time=02:00:00 \
    --mail-type=END,FAIL \
    --mail-user=bjoern.strandgaard@kuleuven.be \
    --wrap "$CONDA_SETUP
echo ======================================================================
echo SecureBoost timing test: $SIZE $IR eval_mode=$EVAL_MODE
echo Full data, 5 rounds, 1 seed.
echo Started: \$(date)
echo ======================================================================
$PYTHON --fl_algo $SCENARIO $FLAGS
EXIT_CODE=\$?
echo ======================================================================
echo Finished: \$(date)
echo Exit: \$EXIT_CODE
echo ======================================================================"

echo "  Submitted: $SCENARIO → $JOB"
echo ""
echo "Check log then multiply per-round time by (num_rounds / 5) x 4 seeds to estimate full run."