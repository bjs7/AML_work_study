#!/bin/bash
# =============================================================================
# Submit benchmark jobs for FedGraphSimple (lazy_link_neighbor):
#   Job 1: Training timing under comparable eval mode (2 epochs)
#   Job 2: Full training timing under system eval mode (2 epochs)
#   Job 3: Inference-only timing under system eval mode (pre-trained weights)
#
# Usage:
#   bash scripts/benchmark/benchmark_vertical_simple.sh
#   bash scripts/benchmark/benchmark_vertical_simple.sh --inference_only /path/to/model.pth
#
# With --inference_only <path>, only Job 3 is submitted.
# =============================================================================

CLUSTER="wice"
ACCOUNT="lp_aml_work_study"
PARTITION="gpu_h100"
TIME="06:00:00"
GPUS="1"
CPUS="16"
MEM="192G"

PYTHON_SCRIPT="/data/leuven/362/vsc36278/AML_work_study/batch_jobs/benchmark/benchmark_vertical_simple.py"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc"

BASE_FLAGS="--fl_algo FedGraphSimple --model GINe --size small --ir HI \
--batching --batching_mode lazy_link_neighbor --ibm_hp --emlps \
--max_workers $CPUS --testing_seeds 1"

INFERENCE_ONLY=false
WEIGHTS_PATH=""

if [[ "$1" == "--inference_only" ]]; then
    INFERENCE_ONLY=true
    WEIGHTS_PATH="${2:?--inference_only requires a model.pth path as second argument}"
fi

LOGS_DIR="/data/leuven/362/vsc36278/AML_work_study/batch_jobs/benchmark/logs"
mkdir -p "$LOGS_DIR"

submit_job() {
    local job_name=$1
    local extra_flags=$2

    echo "Submitting: $job_name"
    sbatch \
        -M "$CLUSTER" \
        --account="$ACCOUNT" \
        --job-name="$job_name" \
        --output="${LOGS_DIR}/${job_name}_%j.log" \
        --error="${LOGS_DIR}/${job_name}_%j.err" \
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
echo 'Job: $job_name'
echo '======================================================================'
python $PYTHON_SCRIPT $BASE_FLAGS $extra_flags
echo '======================================================================'
echo 'Benchmark finished at: \$(date)'
echo '======================================================================'
"
    sleep 1
}

if [[ "$INFERENCE_ONLY" == true ]]; then
    # Inference-only: load pre-trained weights, time test evaluation (system eval)
    submit_job "aml_bench_simple_infer_system" \
        "--eval_mode system --inference_only --load_weights $WEIGHTS_PATH"
else
    # Training timing, system eval
    submit_job "aml_bench_simple_train_system" \
        "--eval_mode system --n_benchmark_epochs 2"
fi

echo ""
if [[ "$INFERENCE_ONLY" == true ]]; then
    echo "Submitted inference-only benchmark (system eval, weights: $WEIGHTS_PATH)"
else
    echo "Submitted training benchmark job (system eval, 2 epochs)."
    echo "To benchmark inference only, run:"
    echo "  bash scripts/benchmark/benchmark_vertical_simple.sh --inference_only /path/to/seed_1/model.pth"
fi
