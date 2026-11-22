#!/bin/bash
# =============================================================================
# Submit RestrictedMerge (system, lazy_link_neighbor) — one job per seed, all
# seeds writing into the same run directory (--run_id). The job that finishes
# last writes the aggregated result across all seeds.
#
# RestrictedMerge experiment: same per-party restricted subgraph as SplitFed,
# but merges (From Bank, To Bank) pair subgraphs into one graph per pair and
# runs a single GNN with a standard mlp head (not mlp_vert concatenation).
#
# Goal: isolate whether the SplitFed vs full_info performance gap is due to
# restricted information or the two-view concatenation architecture.
#
# Comparable with SplitFed run (run_vertical_simple_system.sh):
#   lazy_link_neighbor, 20 rounds, wice gpu_h100, 182G
# =============================================================================
# Usage:
#   bash scripts/hpc/training/gnn/system/run_restricted_merge_system.sh
#       → new run, seeds 1-2
#   bash scripts/hpc/training/gnn/system/run_restricted_merge_system.sh <RUN_ID> <FIRST_SEED> <LAST_SEED>
#       → add seeds to an existing run, e.g. <RUN_ID> 3 4
# =============================================================================

if [[ $# -eq 3 ]]; then
    RUN_ID="$1"; FIRST_SEED="$2"; LAST_SEED="$3"
elif [[ $# -eq 0 ]]; then
    RUN_ID=$(date +%Y%m%d_%H%M%S); FIRST_SEED=1; LAST_SEED=2
else
    echo "Usage: $0 [<RUN_ID> <FIRST_SEED> <LAST_SEED>]"
    exit 1
fi

CLUSTER="wice"
ACCOUNT="lp_aml_work_study"
PARTITION="gpu_h100"
CPUS="16"
MEM="182G"
GPUS="1"
# Estimated ~51 min per epoch (chunk size 256, lookup subgraph build) + setup/test
# → ~17-22 h for 20 rounds. Check the first epochs' "Vali F1" timestamps.
TIME="30:00:00"

PYTHON_CMD="python $VSC_DATA/AML_work_study/AML_work_study/main.py"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

BASE_FLAGS="--fl_algo RestrictedMerge --model GINe --size small --ir HI \
--batching --ibm_hp --emlps --eval_mode system \
--max_workers $CPUS --testing_seeds 1 --batching_mode lazy_link_neighbor --num_rounds 20"

mkdir -p logs

for SEED in $(seq "$FIRST_SEED" "$LAST_SEED"); do
    JOB_NAME="aml_restrmerge_sys_s${SEED}"
    echo "Submitting: $JOB_NAME (seed $SEED, run_id=$RUN_ID)"
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
echo 'RestrictedMerge system — seed $SEED (run_id=$RUN_ID)'
echo '======================================================================'
$PYTHON_CMD $BASE_FLAGS --first_seed $SEED --run_id $RUN_ID
echo '======================================================================'
echo 'Finished'
echo '======================================================================'
"
    sleep 1
done

echo ""
echo "Submitted seeds $FIRST_SEED-$LAST_SEED (run_id=$RUN_ID)."
echo "Results: $VSC_DATA/AML_work_study/experiments/small_HI/split_0.6_0.2/system/RestrictedMerge/GINe__emlps/batching__ibm_hp/$RUN_ID/"
echo "To add more seeds later: bash $0 $RUN_ID 3 4"
