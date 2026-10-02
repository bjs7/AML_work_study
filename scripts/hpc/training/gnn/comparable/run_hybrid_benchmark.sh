#!/bin/bash
# =============================================================================
# FedAvgSplit Phase 2 CPU timing benchmark.
#
# Runs a minimal FedAvgSplit job to estimate Phase 2 wall-time on CPU before
# committing to a full run:
#   Phase 1 — 3 FedAvg rounds (enough to produce a model, not for quality)
#   Phase 2 — 2 MLP rounds on CPU
#
# Check the log for lines like:
#   "FedAvgSplit Phase 2: training mlp_vert ..."
#   "Epoch 1/2 - Train Loss: ..."
#   "Epoch 2/2 - Train Loss: ..."
# and compare the timestamps to get per-epoch Phase 2 wall-time.
# Multiply by 25 to estimate the full --num_phase2_rounds 25 cost.
#
# Node specs (Cascadelake GPU, gpu_v100): 36 cores, 768 GB RAM, 8× V100 32GB
# Per-GPU policy limit: 4 cores, 84000 MiB (~82 GiB).
# =============================================================================
# Usage:
#   bash scripts/hpc/training/gnn/comparable/run_hybrid_benchmark.sh
# =============================================================================

CLUSTER="genius"
ACCOUNT="lp_aml_work_study"
PARTITION="gpu_v100"
CPUS="4"
MEM="82G"
GPUS="1"
TIME="6:00:00"

PYTHON_CMD="python $VSC_DATA/AML_work_study/AML_work_study/main.py"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc"

BASE_FLAGS="--fl_algo FedAvgSplit --model GINe --size small --ir HI \
--batching --batching_mode lazy_link_neighbor \
--ibm_hp --emlps \
--eval_mode comparable \
--mu 0.1 --num_local_epochs 5 --client_fraction 0.1 \
--num_rounds 3 --num_phase2_rounds 2 \
--max_workers $CPUS --testing_seeds 1"

RUN_ID=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

JOB_NAME="aml_hybrid_bench_s1"
echo "Submitting: $JOB_NAME (benchmark, run_id=$RUN_ID)"
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
echo 'FedAvgSplit benchmark — Phase 1: 3 rounds  Phase 2: 2 rounds (run_id=$RUN_ID)'
echo '======================================================================'
$PYTHON_CMD $BASE_FLAGS --first_seed 1 --run_id $RUN_ID
echo '======================================================================'
echo 'Job finished at: \$(date)'
echo '======================================================================'
"

echo ""
echo "Submitted $JOB_NAME (run_id=$RUN_ID)."
echo "Once done, check logs/${JOB_NAME}_<jobid>.log:"
echo "  grep 'Phase 2\|Epoch [12]/2' logs/${JOB_NAME}_<jobid>.log"
echo "Compare Phase 2 start timestamp to Epoch 2/2 timestamp to get per-epoch time."
echo "Multiply by 25 to estimate the full run_hybrid_comparable.sh Phase 2 cost."
