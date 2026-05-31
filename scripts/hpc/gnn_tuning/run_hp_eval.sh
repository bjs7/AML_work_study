#!/bin/bash
# =============================================================================
# GNN HP Evaluation — SLURM Array Job
# =============================================================================
# Each array task evaluates one HP config and saves its F1 score.
# Called by submit_gnn_tuning.sh; do not submit this directly.
#
# $SLURM_ARRAY_TASK_ID is used as --hp_idx.
# All remaining positional args are forwarded to eval_single_hp.py.
# =============================================================================
#SBATCH --job-name=aml_gnn_hp_eval
#SBATCH --account=lp_aml_work_study
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --mem=42G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=bjoern.strandgaard@kuleuven.be
#SBATCH --output=logs/gnn_hp_eval_%A_%a.log
#SBATCH --error=logs/gnn_hp_eval_%A_%a.err

SCRIPT_DIR="$VSC_DATA/AML_work_study/batch_jobs/gnn_tuning"

export PATH=$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc

echo "======================================================================"
echo "GNN HP Eval  stage=$(echo "$@" | grep -oP '(?<=--stage )\d') | idx=$SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID  Task: $SLURM_ARRAY_TASK_ID  Node: $SLURM_NODELIST"
echo "======================================================================"

python "$SCRIPT_DIR/eval_single_hp.py" \
    --hp_idx "$SLURM_ARRAY_TASK_ID" \
    "$@"

EXIT_CODE=$?
echo "======================================================================"
echo "Finished at: $(date)  Exit code: $EXIT_CODE"
echo "======================================================================"
exit $EXIT_CODE
