#!/bin/bash
# =============================================================================
# SecureBoost Training — comparable eval, 2 parallel seed jobs.
#
# Each job runs 1 seed with a shared --run_id so both write to the same
# experiment directory. Run again with --first_seed 3 later to add seeds 3+4.
#
# Runtime estimate: ~45-70 min/round × 60 rounds ≈ 45-70 h per seed
#
# Usage:
#   bash scripts/run_secureboost_comparable.sh           # seeds 1 and 2
#   bash scripts/run_secureboost_comparable.sh 3 <run_id>  # seeds 3 and 4 (same dir)
# =============================================================================

FIRST_SEED=${1:-1}
RUN_ID=${2:-$(date +%Y%m%d_%H%M%S)}

HP="configs/tuned_hyperparams/booster/xgboost/small_HI_comparable_r150.json"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc"

PYTHON="python \$VSC_DATA/AML_work_study/AML_work_study/main.py"

BASE_FLAGS="--fl_algo SecureBoost --model xgboost --size small --ir HI \
--eval_mode comparable --testing_seeds 1 \
--hp_path $HP --run_id $RUN_ID --sb_num_rounds 60"

mkdir -p logs

SECOND_SEED=$((FIRST_SEED + 1))

echo "======================================================================"
echo "Submitting SecureBoost comparable: seeds $FIRST_SEED and $SECOND_SEED"
echo "run_id: $RUN_ID"
echo "======================================================================"

# --- Seed A ---
JOB_A="aml_sb_comparable_s${FIRST_SEED}"
sbatch \
    -M wice \
    --account=lp_aml_work_study \
    --partition=batch_icelake \
    --job-name="$JOB_A" \
    --output="logs/${JOB_A}_%j.log" \
    --error="logs/${JOB_A}_%j.err" \
    --nodes=1 --ntasks=1 \
    --cpus-per-task=32 \
    --mem=160G \
    --time=72:00:00 \
    --mail-type=END,FAIL \
    --mail-user=bjoern.strandgaard@kuleuven.be \
    --wrap "$CONDA_SETUP
echo 'SecureBoost comparable — seed $FIRST_SEED (run_id=$RUN_ID)'
echo 'Started: \$(date)'
$PYTHON $BASE_FLAGS --first_seed $FIRST_SEED
echo 'Finished: \$(date)'"

sleep 1

# --- Seed B ---
JOB_B="aml_sb_comparable_s${SECOND_SEED}"
sbatch \
    -M wice \
    --account=lp_aml_work_study \
    --partition=batch_icelake \
    --job-name="$JOB_B" \
    --output="logs/${JOB_B}_%j.log" \
    --error="logs/${JOB_B}_%j.err" \
    --nodes=1 --ntasks=1 \
    --cpus-per-task=32 \
    --mem=160G \
    --time=72:00:00 \
    --mail-type=END,FAIL \
    --mail-user=bjoern.strandgaard@kuleuven.be \
    --wrap "$CONDA_SETUP
echo 'SecureBoost comparable — seed $SECOND_SEED (run_id=$RUN_ID)'
echo 'Started: \$(date)'
$PYTHON $BASE_FLAGS --first_seed $SECOND_SEED
echo 'Finished: \$(date)'"

echo ""
echo "Submitted seeds $FIRST_SEED and $SECOND_SEED → run_id=$RUN_ID"
echo "To add seeds 3+4 later: bash scripts/run_secureboost_comparable.sh 3 $RUN_ID"