#!/bin/bash
#SBATCH --job-name=aml_train_booster
#SBATCH --account=lp_aml_work_study
#SBATCH --clusters=wice
#SBATCH --partition=batch_icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --mem=160G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bjoern.strandgaard@kuleuven.be
#SBATCH --output=logs/aml_booster_%j.log
#SBATCH --error=logs/aml_booster_%j.err

# =============================================================================
# Booster Training — CPU (no GPU)
# =============================================================================
# Loads pre-tuned hyperparameters and runs full training across all seeds.
# Pass --hp_path to use a specific HP file for all scenarios.
#
# Usage:
#   sbatch scripts/train_booster_cpu.sh [fl_algo] [size] [ir] [model] [eval_mode] [ibm_fe] [hp_path]
#
# Examples:
#   sbatch scripts/train_booster_cpu.sh full_info small HI xgboost system
#   sbatch scripts/train_booster_cpu.sh full_info small HI xgboost system ibm_fe configs/tuned_hyperparams/booster/xgboost/small_HI_comparable_r150.json
#   sbatch scripts/train_booster_cpu.sh individual small HI xgboost comparable "" configs/tuned_hyperparams/booster/xgboost/small_HI_comparable_r150.json
# =============================================================================

FL_ALGO=${1:-full_info}
SIZE=${2:-small}
IR=${3:-HI}
MODEL=${4:-xgboost}
EVAL_MODE=${5:-system}
IBM_FE=${6:-}
HP_PATH=${7:-}

IBM_FE_FLAG=""
if [ "$IBM_FE" = "ibm_fe" ]; then
    IBM_FE_FLAG="--ibm_fe"
fi

HP_PATH_FLAG=""
if [ -n "$HP_PATH" ]; then
    HP_PATH_FLAG="--hp_path $HP_PATH"
fi

export PATH=$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc

echo "======================================================================"
echo "Booster Training — CPU"
echo "Job started : $(date)"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_NODELIST"
echo "CPUs        : $SLURM_CPUS_PER_TASK"
echo "======================================================================"
echo "  FL algo   : $FL_ALGO"
echo "  Dataset   : $SIZE / $IR"
echo "  Model     : $MODEL"
echo "  Eval mode : $EVAL_MODE"
echo "  IBM FE    : ${IBM_FE:-no}"
echo "  HP path   : ${HP_PATH:-auto}"
echo "======================================================================"

python $VSC_DATA/AML_work_study/AML_work_study/main.py \
    --fl_algo "$FL_ALGO" \
    --model "$MODEL" \
    --size "$SIZE" \
    --ir "$IR" \
    --eval_mode "$EVAL_MODE" \
    $IBM_FE_FLAG \
    $HP_PATH_FLAG

EXIT_CODE=$?

echo "======================================================================"
echo "Finished at : $(date)"
echo "Exit code   : $EXIT_CODE"
echo "======================================================================"

exit $EXIT_CODE
