#!/bin/bash
# =============================================================================
# Gradient equivalence test — real AML data, full dataset.
#
# Verifies that the explicit multi-party backward pass (detach at embedding
# boundary, receive ∂L/∂emb from manager, backpropagate locally) produces
# identical GNN parameter gradients to the joint PyTorch backward pass.
#
# Runs on CPU (GPU scatter_add is non-deterministic; test requires exact
# floating-point reproducibility). Requests minimal resources.
#
# Output: gradient_equivalence_results.csv in scripts/hpc/tests/
#
# Cluster: genius
# Walltime: 2 h (full small/HI dataset, 10 batches of 8192 transactions)
# =============================================================================
# Usage:
#   bash scripts/hpc/tests/run_gradient_equivalence.sh
# =============================================================================

CLUSTER="wice"
ACCOUNT="lp_aml_work_study"
PARTITION="batch_icelake"  # CPU-only partition
TIME="02:00:00"
CPUS="4"
MEM="32G"

PYTHON_SCRIPT="$VSC_DATA/AML_work_study/AML_work_study/scripts/hpc/tests/gradient_equivalence_real.py"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc"

FLAGS="--size small --ir HI --n_rows 0 --n_batch 8192 --n_batches 10"

JOB_NAME="aml_grad_equiv_test"

mkdir -p logs

echo "Submitting $JOB_NAME (CPU, full small/HI dataset, 10 × 8192 batches) ..."

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
    --mail-type=END,FAIL \
    --mail-user=bjoern.strandgaard@kuleuven.be \
    --wrap "$CONDA_SETUP
echo '======================================================================'
echo 'Gradient equivalence test started at: \$(date)'
echo 'Job ID: \$SLURM_JOB_ID  Node: \$SLURM_NODELIST'
echo '======================================================================'
python $PYTHON_SCRIPT $FLAGS
echo '======================================================================'
echo 'Gradient equivalence test finished at: \$(date)'
echo '======================================================================'
"

echo ""
echo "Results saved to scripts/hpc/tests/gradient_equivalence_results.csv"
echo "Log: logs/${JOB_NAME}_<jobid>.log"
