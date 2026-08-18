#!/bin/bash
# =============================================================================
# Booster Comparable — TEST RUNS (timing estimates)
# =============================================================================
# Submits a SecureBoost timing job to estimate walltime before the full run.
#
#   SecureBoost : full data, --testing_seeds 1, --sb_num_rounds 5
#                 (no --testing so all 630 banks load; 5 rounds gives realistic per-round time)
#                 Extrapolate: full_time ≈ measured_time × (960 / 5)
#
# Usage:
#   bash scripts/run_booster_comparable_test.sh
# =============================================================================

SIZE="small"
IR="HI"
MODEL="xgboost"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc"

PYTHON="python \$VSC_DATA/AML_work_study/AML_work_study/main.py"

mkdir -p logs

echo "Submitting booster comparable test runs (${SIZE} / ${IR})"
echo ""

# --- SecureBoost: full data, capped rounds, 1 seed ---
# No --testing so all 630 banks load. --sb_num_rounds 5 gives a realistic per-round estimate.
# Full run = measured_time x (960 / 5)
SCENARIO="SecureBoost"
JOB="aml_booster_cmp_test_${SCENARIO}_${SIZE}_${IR}"
FLAGS="--model $MODEL --size $SIZE --ir $IR --eval_mode comparable --testing_seeds 1 --sb_num_rounds 5"

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
echo Booster comparable TEST RUN: $SCENARIO $SIZE $IR
echo Full data, 5 rounds, 1 seed. Extrapolate: full = this x \(960/5\) x 4 seeds
echo Started: \$(date)
echo ======================================================================
$PYTHON --fl_algo $SCENARIO $FLAGS
EXIT_CODE=\$?
echo ======================================================================
echo Finished: \$(date)
echo Exit: \$EXIT_CODE
echo ======================================================================"

echo "  Submitted: $SCENARIO (full data, 5 rounds, 1 seed) → $JOB"

echo ""
echo "SecureBoost test job submitted."
echo "SecureBoost timing: check log then multiply by (960/5) x 4 to estimate full run."
