#!/bin/bash
#SBATCH --job-name=aml_tune_booster
#SBATCH --account=lp_aml_work_study
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bjoern.strandgaard@kuleuven.be
#SBATCH --output=output_tune_booster_%j.txt
#SBATCH --error=error_tune_booster_%j.txt

# =============================================================================
# Booster Hyperparameter Tuning — Full Info
# =============================================================================
# Tunes XGBoost hyperparameters on the full dataset (no federation) using
# successive halving, then saves the result to:
#   configs/tuned_hyperparams/booster/xgboost/{size}_{ir}.json
#
# These saved HPs are reused by individual_booster, FedAvg_booster, and
# SecureBoost_booster without re-tuning.
#
# Usage:
#   sbatch scripts/tune_booster.sh [size] [ir] [model]
#
# Examples:
#   sbatch scripts/tune_booster.sh small HI
#   sbatch scripts/tune_booster.sh medium LO
#   sbatch scripts/tune_booster.sh large HI xgboost
# =============================================================================

SIZE=${1:-small}
IR=${2:-HI}
MODEL=${3:-xgboost}
shift 3
EXTRA_FLAGS="$@"

export PATH=$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc

echo "======================================================================"
echo "Booster Hyperparameter Tuning (full_info / tune-only)"
echo "Job started : $(date)"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_NODELIST"
echo "CPUs        : $SLURM_CPUS_PER_TASK"
echo "======================================================================"
echo "  Dataset     : $SIZE / $IR"
echo "  Model       : $MODEL"
echo "  Extra flags : ${EXTRA_FLAGS:-none}"
echo "======================================================================"

python $VSC_DATA/AML_work_study/AML_work_study/main.py \
    --fl_algo full_info \
    --model "$MODEL" \
    --size "$SIZE" \
    --ir "$IR" \
    --tune \
    $EXTRA_FLAGS

EXIT_CODE=$?

echo "======================================================================"
echo "Finished at : $(date)"
echo "Exit code   : $EXIT_CODE"
echo "======================================================================"

exit $EXIT_CODE
