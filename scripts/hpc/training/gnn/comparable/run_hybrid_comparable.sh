#!/bin/bash
# =============================================================================
# Submit FedAvgSplit (comparable, single seed) for an initial timing estimate.
#
# Two-phase algorithm:
#   Phase 1 — FedProx (mu=0.1): separate per-party models, --num_rounds FedAvg rounds.
#   Phase 2 — Vertical MLP: frozen GNN embeddings, --num_rounds vertical rounds.
# Both phases use the same --num_rounds flag.
#
# Phase 1 uses --num_rounds (100); Phase 2 uses --num_phase2_rounds (50).
# Expected runtime (V100): ~8h Phase 1 (100 FedAvg rounds, same as solo FedAvg) +
# Phase 2 (~FedGraphSimple comparable cost × 50 rounds). Total expected within 24h.
# Check 'Phase 1'/'Phase 2' log timestamps after the run to tune these if needed.
#
# Node specs (Cascadelake GPU, gpu_v100): 36 cores, 768 GB RAM, 8× V100 32GB
# Per-GPU policy limit: 4 cores, 84000 MiB (~82 GiB).
# =============================================================================
# Usage:
#   bash scripts/hpc/training/gnn/comparable/run_hybrid_comparable.sh
# =============================================================================

CLUSTER="genius"
ACCOUNT="lp_aml_work_study"
PARTITION="gpu_v100"
CPUS="4"
MEM="82G"
GPUS="1"
TIME="48:00:00"

PYTHON_CMD="python $VSC_DATA/AML_work_study/AML_work_study/main.py"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc"

BASE_FLAGS="--fl_algo FedAvgSplit --model GINe --size small --ir HI \
--batching --batching_mode lazy_link_neighbor \
--ibm_hp --emlps \
--eval_mode comparable \
--mu 0.1 --num_local_epochs 5 --client_fraction 0.1 \
--num_rounds 100 --num_phase2_rounds 50 \
--max_workers $CPUS --testing_seeds 1"

RUN_ID=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

JOB_NAME="aml_hybrid_comparable_s1"
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
echo 'FedAvgSplit comparable — seed 1 (run_id=$RUN_ID)'
echo 'Phase 1: FedProx mu=0.1, 100 rounds'
echo 'Phase 2: frozen GNN, vertical mlp_vert, 50 rounds'
echo '======================================================================'
$PYTHON_CMD $BASE_FLAGS --first_seed 1 --run_id $RUN_ID
echo '======================================================================'
echo 'Job finished at: \$(date)'
echo '======================================================================'
"

echo ""
echo "Submitted FedAvgSplit comparable timing run (run_id=$RUN_ID)."
echo "Check logs/${JOB_NAME}_<jobid>.log for 'Phase 1'/'Phase 2' timestamps"
echo "to see how long each phase takes before scaling up to more seeds."
