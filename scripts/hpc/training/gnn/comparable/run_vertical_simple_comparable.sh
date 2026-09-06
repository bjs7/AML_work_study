#!/bin/bash
# =============================================================================
# Submit FedGraphSimple (lazy_link_neighbor) — single seed timing verification.
#
# Runs seed 1 to check that the batched GNN forward pass (Batch.from_data_list)
# produces the same results as the old sequential approach and to measure the
# actual wall-clock improvement. Previous runs took ~93h/seed on A100 with the
# sequential forward pass; the batched pass showed ~8-10x speedup in benchmarks
# so ~10-12h is expected, with 24h walltime as headroom.
#
# After confirming results match existing seeds, remaining seeds can be queued
# at the shorter walltime.
# =============================================================================
# Usage:
#   bash scripts/hpc/training/gnn/comparable/run_vertical_simple_comparable.sh
# =============================================================================

CLUSTER="wice"
ACCOUNT="lp_aml_work_study"
PARTITION="gpu_a100"
CPUS="16"
MEM="88G"
GPUS="1"
TIME="24:00:00"

PYTHON_CMD="python $VSC_DATA/AML_work_study/AML_work_study/main.py"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc"

BASE_FLAGS="--fl_algo SplitFed --model GINe --size small --ir HI \
--batching --ibm_hp --emlps --eval_mode comparable \
--max_workers $CPUS --testing_seeds 1 --batching_mode lazy_link_neighbor --num_rounds 50"

RUN_ID=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

JOB_NAME="aml_fedgraphsimple_lazy_s1"
echo "Submitting: $JOB_NAME (seed 1, run_id=$RUN_ID)"
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
echo 'Job started at: \$(date)'
echo 'Job ID: \$SLURM_JOB_ID  Node: \$SLURM_NODELIST'
echo 'FedGraphSimple lazy — seed 1, batched forward pass (run_id=$RUN_ID)'
echo '======================================================================'
$PYTHON_CMD $BASE_FLAGS --first_seed 1 --run_id $RUN_ID
echo '======================================================================'
echo 'Job finished at: \$(date)'
echo '======================================================================'
"

echo ""
echo "Submitted $JOB_NAME (run_id=$RUN_ID)."
echo "Check logs/${JOB_NAME}_<jobid>.log for round timings and final F1."
echo "Compare result against existing seed-1 run to verify correctness."
