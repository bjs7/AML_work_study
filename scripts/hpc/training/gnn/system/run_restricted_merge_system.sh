#!/bin/bash
# =============================================================================
# Submit RestrictedMerge (system, lazy_link_neighbor) — single seed.
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
# =============================================================================

CLUSTER="wice"
ACCOUNT="lp_aml_work_study"
PARTITION="gpu_h100"
CPUS="16"
MEM="182G"
GPUS="1"
# Measured ~2h26 per epoch (2h05 train + 17 min vali) + ~35 min setup/test → ~50 h for 20 rounds.
TIME="60:00:00"

PYTHON_CMD="python $VSC_DATA/AML_work_study/AML_work_study/main.py"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

BASE_FLAGS="--fl_algo RestrictedMerge --model GINe --size small --ir HI \
--batching --ibm_hp --emlps --eval_mode system \
--max_workers $CPUS --testing_seeds 1 --batching_mode lazy_link_neighbor --num_rounds 20"

RUN_ID=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

JOB_NAME="aml_restrmerge_sys_lazy_s1"
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
echo 'RestrictedMerge system — seed 1 (run_id=$RUN_ID)'
echo '======================================================================'
$PYTHON_CMD $BASE_FLAGS --first_seed 1 --run_id $RUN_ID
echo '======================================================================'
echo 'Job finished at: \$(date)'
echo '======================================================================'
"

echo ""
echo "Submitted $JOB_NAME (run_id=$RUN_ID)."
echo "This run should be compared with SplitFed system results."
