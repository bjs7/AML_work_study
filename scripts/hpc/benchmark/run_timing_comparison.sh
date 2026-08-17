#!/bin/bash
# =============================================================================
# Timing benchmark — FedGraph vs FedAvgSplit (single job, --fl_algo all)
#
# Runs benchmark_timing_comparison.py which:
#   1. Loads data once and runs both FedGraph and FedAvgSplit back-to-back.
#   2. Reports per-step wall-clock times: batch_setup, party_gnn, exchange,
#      manager_head, forward_total, backward, validation, epoch_total.
#   3. Runs an explicit-backward mini-benchmark for FedAvgSplit that separates
#      manager-only backward time from per-party GNN backward time — used to
#      estimate whether explicit multi-GPU would be faster.
#   4. Prints a parallelism speedup table for N=2/4/8 GPUs.
#
# Cluster: genius (Cascadelake GPU, gpu_v100)
# Node specs: 8× V100 32GB, 36 cores, 768 GB RAM
# Per-job policy: 1 GPU, 4 CPUs, 82 GiB
# Walltime: 4 h (5 benchmark epochs × 2 algos, small/HI dataset)
# =============================================================================
# Usage:
#   bash scripts/hpc/benchmark/run_timing_comparison.sh
# =============================================================================

CLUSTER="genius"
ACCOUNT="lp_aml_work_study"
PARTITION="gpu_v100"
TIME="04:00:00"
GPUS="1"
CPUS="4"
MEM="82G"

PYTHON_SCRIPT="$VSC_DATA/AML_work_study/AML_work_study/scripts/hpc/benchmark/benchmark_timing_comparison.py"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc"

FLAGS="--fl_algo all \
--model GINe --size small --ir HI \
--ibm_hp --emlps --batching --batching_mode lazy_link_neighbor \
--eval_mode system --max_workers $CPUS \
--n_benchmark_epochs 5 --n_explicit_batches 20"

JOB_NAME="aml_timing_bench"

mkdir -p logs

echo "Submitting $JOB_NAME (FedGraph + SplitFed, V100, genius) ..."

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
echo 'Timing benchmark started at: \$(date)'
echo 'Job ID: \$SLURM_JOB_ID  Node: \$SLURM_NODELIST'
echo '======================================================================'
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $FLAGS
echo '======================================================================'
echo 'Timing benchmark finished at: \$(date)'
echo '======================================================================'
"

echo ""
echo "Results will appear in the job log: logs/${JOB_NAME}_<jobid>.log"
echo "grep 'ms' logs/${JOB_NAME}_<jobid>.log  # to pull timing rows quickly"
