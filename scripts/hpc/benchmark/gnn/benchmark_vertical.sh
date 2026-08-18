#!/bin/bash
# =============================================================================
# Submit benchmark jobs for vertical FL batching modes
# Queues two separate single-node jobs (one per batching mode)
# =============================================================================
# Usage:
#   bash scripts/benchmark_vertical.sh
# =============================================================================

CLUSTER="wice"
ACCOUNT="lp_aml_work_study"
PARTITION="gpu_a100"
TIME="03:00:00"
GPUS="1"
CPUS="16"
MEM="88G"

PYTHON_SCRIPT="$VSC_DATA/AML_work_study/AML_work_study/scripts/benchmark/benchmark_vertical.py"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc"

BASE_FLAGS="--fl_algo FedGraph --model GINe --size small --ir HI --batching --ibm_hp --eval_mode comparable --max_workers $CPUS --testing --testing_frac 0.20"

MODES=("lazy_link_neighbor" "simple")

mkdir -p logs

for mode in "${MODES[@]}"; do
    JOB_NAME="aml_bench_${mode}"

    echo "Submitting: $JOB_NAME"

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
        --wrap "$CONDA_SETUP
echo '======================================================================'
echo 'Benchmark started at: \$(date)'
echo 'Job ID: \$SLURM_JOB_ID  Node: \$SLURM_NODELIST'
echo 'Mode: $mode'
echo '======================================================================'
python $PYTHON_SCRIPT $BASE_FLAGS --batching_mode $mode
echo '======================================================================'
echo 'Benchmark finished at: \$(date)'
echo '======================================================================'
"
    sleep 1
done

echo ""
echo "Submitted ${#MODES[@]} benchmark jobs."
