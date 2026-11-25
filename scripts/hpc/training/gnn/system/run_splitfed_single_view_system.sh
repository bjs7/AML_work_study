#!/bin/bash
# =============================================================================
# Single-view SplitFed (system, lazy_link_neighbor): the manager only ever receives
# one bank's embedding, in training and evaluation (--view_mode). Answers "what
# if a transaction only ever has one view?", as opposed to the test-time view
# ablation of the two-view model (robustness to a view going missing).
#
#   sender_only    manager input [s | 0]
#   receiver_only  manager input [0 | r]   (intra-bank: the bank's embedding)
#
# Same flags as V1 (run_vertical_simple_system.sh) apart from --view_mode.
# One job per seed into a shared run directory; results under
#   system/SplitFed/view_<VIEW>/GINe__emlps/batching__ibm_hp/<RUN_ID>/
# =============================================================================
# Usage:
#   bash scripts/hpc/training/gnn/system/run_splitfed_single_view_system.sh <sender_only|receiver_only>
#       → new run, seeds 1-4
#   bash scripts/hpc/training/gnn/system/run_splitfed_single_view_system.sh <VIEW> <RUN_ID> <FIRST_SEED> <LAST_SEED>
#       → add seeds to an existing run
# =============================================================================

VIEW="$1"
if [[ "$VIEW" != "sender_only" && "$VIEW" != "receiver_only" ]]; then
    echo "Usage: $0 <sender_only|receiver_only> [<RUN_ID> <FIRST_SEED> <LAST_SEED>]"
    exit 1
fi
if [[ $# -eq 4 ]]; then
    RUN_ID="$2"; FIRST_SEED="$3"; LAST_SEED="$4"
elif [[ $# -eq 1 ]]; then
    RUN_ID=$(date +%Y%m%d_%H%M%S); FIRST_SEED=1; LAST_SEED=4
else
    echo "Usage: $0 <sender_only|receiver_only> [<RUN_ID> <FIRST_SEED> <LAST_SEED>]"
    exit 1
fi

CLUSTER="wice"
ACCOUNT="lp_aml_work_study"
PARTITION="gpu_h100"
CPUS="16"
MEM="182G"
GPUS="1"
TIME="12:00:00"   # two-view SplitFed seeds take ~7 h

PYTHON_CMD="python $VSC_DATA/AML_work_study/AML_work_study/main.py"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

BASE_FLAGS="--fl_algo SplitFed --model GINe --size small --ir HI \
--batching --ibm_hp --emlps --eval_mode system \
--max_workers $CPUS --testing_seeds 1 --batching_mode lazy_link_neighbor --num_rounds 20 \
--view_mode $VIEW"

mkdir -p logs

for SEED in $(seq "$FIRST_SEED" "$LAST_SEED"); do
    JOB_NAME="aml_sf_${VIEW}_sys_s${SEED}"
    echo "Submitting: $JOB_NAME (seed $SEED, run_id=$RUN_ID)"
    sbatch \
        -M "$CLUSTER" \
        --account="$ACCOUNT" \
        --job-name="$JOB_NAME" \
        --output="logs/${JOB_NAME}_%j.log" \
        --error="logs/${JOB_NAME}_%j.err" \
        --partition="$PARTITION" \
        --nodes=1 --ntasks=1 \
        --time="$TIME" --mem="$MEM" \
        --cpus-per-task="$CPUS" --gpus-per-node="$GPUS" \
        --mail-type=END,FAIL \
        --mail-user=bjoern.strandgaard@kuleuven.be \
        --wrap "$CONDA_SETUP
echo '======================================================================'
echo 'SplitFed $VIEW system — seed $SEED (run_id=$RUN_ID)'
echo '======================================================================'
$PYTHON_CMD $BASE_FLAGS --first_seed $SEED --run_id $RUN_ID
echo '======================================================================'
echo 'Finished'
echo '======================================================================'
"
    sleep 1
done

echo ""
echo "Submitted $VIEW seeds $FIRST_SEED-$LAST_SEED (run_id=$RUN_ID)."
echo "Results: $VSC_DATA/AML_work_study/experiments/small_HI/split_0.6_0.2/system/SplitFed/view_$VIEW/GINe__emlps/batching__ibm_hp/$RUN_ID/"
echo "To add seeds later: bash $0 $VIEW $RUN_ID <FIRST_SEED> <LAST_SEED>"
