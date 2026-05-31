#!/bin/bash
# =============================================================================
# GNN HP Aggregation — SLURM CPU Job
# =============================================================================
# Runs between stages: picks top 5 from stage N, narrows intervals, generates
# stage N+1 configs. Also used for final selection (--final flag).
# Called by submit_gnn_tuning.sh with a dependency on the preceding array job.
#
# All positional args are forwarded to aggregate_stage.py.
# =============================================================================
#SBATCH --job-name=aml_gnn_aggregate
#SBATCH --account=lp_aml_work_study
#SBATCH --clusters=wice
#SBATCH --partition=batch_icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=bjoern.strandgaard@kuleuven.be
#SBATCH --output=logs/gnn_aggregate_%j.log
#SBATCH --error=logs/gnn_aggregate_%j.err

SCRIPT_DIR="$VSC_DATA/AML_work_study/batch_jobs/gnn_tuning"

export PATH=$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc

echo "======================================================================"
echo "GNN HP Aggregation: $@"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "======================================================================"

python "$SCRIPT_DIR/aggregate_stage.py" "$@"

EXIT_CODE=$?
echo "======================================================================"
echo "Finished at: $(date)  Exit code: $EXIT_CODE"
echo "======================================================================"
exit $EXIT_CODE
