#!/bin/bash
# =============================================================================
# Correctness + timing test for the batched GNN forward pass in SplitFed.
#
# Runs batched_forward_splitfed.py on a GPU node and reports:
#   1. Whether _collect_embeddings_batched == _collect_embeddings_sequential
#   2. Wall-clock speedup across different party counts (5, 10, 25, 50, 100)
#
# Cluster: genius (V100)
# Walltime: 30 min (well within for synthetic data benchmarks)
# =============================================================================
# Usage:
#   bash scripts/hpc/tests/run_batched_forward_test.sh
# =============================================================================

CLUSTER="genius"
ACCOUNT="lp_aml_work_study"
PARTITION="gpu_v100"
TIME="00:30:00"
CPUS="4"
MEM="16G"
GPUS="1"

PYTHON_SCRIPT="$VSC_DATA/AML_work_study/AML_work_study/scripts/hpc/tests/batched_forward_splitfed.py"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc"

# Representative batch: 50 parties, 100 nodes, 60 edges each
FLAGS="--device cuda --n_parties 50 --n_nodes 100 --n_edges 60 --n_warmup 5 --n_runs 30"

JOB_NAME="aml_batched_fwd_test"

mkdir -p logs

echo "Submitting $JOB_NAME (V100, 50 parties × 100 nodes × 60 edges) ..."

sbatch \
    -M "$CLUSTER" \
    --account="$ACCOUNT" \
    --job-name="$JOB_NAME" \
    --output="logs/${JOB_NAME}_%j.log" \
    --error="logs/${JOB_NAME}_%j.err" \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --time="$TIME" \
    --mem="$MEM" \
    --cpus-per-task="$CPUS" \
    --gpus-per-node="$GPUS" \
    --mail-type=END,FAIL \
    --mail-user=bjoern.strandgaard@kuleuven.be \
    --wrap "$CONDA_SETUP
echo '======================================================================'
echo 'Batched forward test started at: \$(date)'
echo 'Job ID: \$SLURM_JOB_ID  Node: \$SLURM_NODELIST'
echo '======================================================================'
python $PYTHON_SCRIPT $FLAGS
echo '======================================================================'
echo 'Batched forward test finished at: \$(date)'
echo '======================================================================'
"

echo ""
echo "Log: logs/${JOB_NAME}_<jobid>.log"
