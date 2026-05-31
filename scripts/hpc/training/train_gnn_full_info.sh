#!/bin/bash
#SBATCH --job-name=aml_train_gnn
#SBATCH --account=lp_aml_work_study
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=36:00:00
#SBATCH --mem=42G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bjoern.strandgaard@kuleuven.be
#SBATCH --output=logs/train_gnn_full_info_%j.log
#SBATCH --error=logs/train_gnn_full_info_%j.err

# =============================================================================
# GNN Training — Full Info (loads saved HPs, runs 4 seeds)
# =============================================================================
# Loads the tuned GINe hyperparameters saved by tune_gnn_full_info.sh and
# trains the full_info model across 4 seeds.
#
# Prerequisite: tune_gnn_full_info.sh must have been run first to produce:
#   configs/tuned_hyperparams/gnn/GINe/{size}_{ir}_{eval_mode}.json
#
# Usage:
#   sbatch scripts/hpc/train_gnn_full_info.sh [size] [ir] [eval_mode]
#
# Examples:
#   sbatch scripts/hpc/train_gnn_full_info.sh small HI system
#   sbatch scripts/hpc/train_gnn_full_info.sh medium LO system
# =============================================================================

SIZE=${1:-small}
IR=${2:-HI}
EVAL_MODE=${3:-system}
shift 3
EXTRA_FLAGS="$@"

export PATH=$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc

echo "======================================================================"
echo "GNN Training (full_info / 4 seeds)"
echo "Job started : $(date)"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_NODELIST"
echo "CPUs        : $SLURM_CPUS_PER_TASK"
echo "======================================================================"
echo "  Dataset   : $SIZE / $IR"
echo "  Eval mode : $EVAL_MODE"
echo "  Seeds     : 4"
echo "  Loads HPs : configs/tuned_hyperparams/gnn/GINe/${SIZE}_${IR}_${EVAL_MODE}.json"
echo "======================================================================"

python $VSC_DATA/AML_work_study/AML_work_study/main.py \
    --fl_algo full_info \
    --model GINe \
    --size "$SIZE" \
    --ir "$IR" \
    --eval_mode "$EVAL_MODE" \
    --batching \
    --emlps \
    --testing_seeds 4 \
    $EXTRA_FLAGS

EXIT_CODE=$?

echo "======================================================================"
echo "Finished at : $(date)"
echo "Exit code   : $EXIT_CODE"
echo "======================================================================"

exit $EXIT_CODE
