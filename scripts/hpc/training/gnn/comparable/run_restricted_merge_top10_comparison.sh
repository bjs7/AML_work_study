#!/bin/bash
# =============================================================================
# Three-way comparison: SplitFed vs RestrictedMerge vs full_info
# — top-10 banks, local_neighbor batching.
#
# This is a quick overnight feasibility test for RestrictedMerge on a small
# subset (top 10 banks by training edge count) so all three run within a few
# hours on one node.
#
# All three algorithms use:
#   --batching_mode local_neighbor   (pre-computed fixed subgraphs per party)
#   --bank_filter only_top10         (top 10 banks by training edge count)
#   --eval_mode comparable
#   --num_rounds 50
#
# Comparison logic:
#   SplitFed:          2 GNNs + mlp_vert (restricted info, two-view arch)
#   RestrictedMerge:   1 GNN + mlp      (restricted info, single-view arch)
#   full_info:         1 GNN + mlp      (full info, single-view arch)
#
# If RestrictedMerge ≈ SplitFed: performance gap vs full_info is from
#   restricted information, not the architecture.
# If RestrictedMerge ≈ full_info: performance gap is from the architecture
#   (mlp_vert concatenation), not restricted information.
#
# Submits three jobs, each requesting 1 GPU (modest — top 10 banks is small).
# =============================================================================
# Usage:
#   bash scripts/hpc/training/gnn/comparable/run_restricted_merge_top10_comparison.sh
# =============================================================================

CLUSTER="wice"
ACCOUNT="lp_aml_work_study"
PARTITION="gpu_a100"
CPUS="8"
MEM="64G"
GPUS="1"
TIME="12:00:00"

PYTHON_CMD="python $VSC_DATA/AML_work_study/AML_work_study/main.py"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc"

COMMON_FLAGS="--model GINe --size small --ir HI \
--batching --ibm_hp --emlps --eval_mode comparable \
--max_workers $CPUS --testing_seeds 1 \
--batching_mode local_neighbor --num_rounds 50"

# SplitFed and RestrictedMerge use only the top-10 banks for a tractable overnight test.
# full_info trains on all banks (bank_filter is not applicable to its single-party design).
TOP10_FLAGS="--bank_filter only_top10"

RUN_ID=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

# ----- Job 1: SplitFed -----
JOB1="aml_sf_top10_s1"
echo "Submitting: $JOB1 (SplitFed, top-10, local_neighbor)"
sbatch \
    -M "$CLUSTER" \
    --account="$ACCOUNT" \
    --job-name="$JOB1" \
    --output="logs/${JOB1}_%j.log" \
    --error="logs/${JOB1}_%j.err" \
    --partition="$PARTITION" \
    --nodes=1 --ntasks=1 \
    --time="$TIME" --mem="$MEM" \
    --cpus-per-task="$CPUS" --gpus-per-node="$GPUS" \
    --mail-type=END,FAIL \
    --mail-user=bjoern.strandgaard@kuleuven.be \
    --wrap "$CONDA_SETUP
echo '======================================================================'
echo 'Job: SplitFed top-10 banks local_neighbor (run_id=$RUN_ID)'
echo 'Node: \$SLURM_NODELIST  Started: \$(date)'
echo '======================================================================'
$PYTHON_CMD $COMMON_FLAGS $TOP10_FLAGS --fl_algo SplitFed --first_seed 1 --run_id ${RUN_ID}_sf
echo '======================================================================'
echo 'Finished: \$(date)'
echo '======================================================================'
"

# ----- Job 2: RestrictedMerge -----
JOB2="aml_rm_top10_s1"
echo "Submitting: $JOB2 (RestrictedMerge, top-10, local_neighbor)"
sbatch \
    -M "$CLUSTER" \
    --account="$ACCOUNT" \
    --job-name="$JOB2" \
    --output="logs/${JOB2}_%j.log" \
    --error="logs/${JOB2}_%j.err" \
    --partition="$PARTITION" \
    --nodes=1 --ntasks=1 \
    --time="$TIME" --mem="$MEM" \
    --cpus-per-task="$CPUS" --gpus-per-node="$GPUS" \
    --mail-type=END,FAIL \
    --mail-user=bjoern.strandgaard@kuleuven.be \
    --wrap "$CONDA_SETUP
echo '======================================================================'
echo 'Job: RestrictedMerge top-10 banks local_neighbor (run_id=$RUN_ID)'
echo 'Node: \$SLURM_NODELIST  Started: \$(date)'
echo '======================================================================'
$PYTHON_CMD $COMMON_FLAGS $TOP10_FLAGS --fl_algo RestrictedMerge --first_seed 1 --run_id ${RUN_ID}_rm
echo '======================================================================'
echo 'Finished: \$(date)'
echo '======================================================================'
"

# ----- Job 3: full_info -----
JOB3="aml_fi_top10_s1"
echo "Submitting: $JOB3 (full_info, top-10, local_neighbor)"
sbatch \
    -M "$CLUSTER" \
    --account="$ACCOUNT" \
    --job-name="$JOB3" \
    --output="logs/${JOB3}_%j.log" \
    --error="logs/${JOB3}_%j.err" \
    --partition="$PARTITION" \
    --nodes=1 --ntasks=1 \
    --time="$TIME" --mem="$MEM" \
    --cpus-per-task="$CPUS" --gpus-per-node="$GPUS" \
    --mail-type=END,FAIL \
    --mail-user=bjoern.strandgaard@kuleuven.be \
    --wrap "$CONDA_SETUP
echo '======================================================================'
echo 'Job: full_info top-10 banks local_neighbor (run_id=$RUN_ID)'
echo 'Node: \$SLURM_NODELIST  Started: \$(date)'
echo '======================================================================'
$PYTHON_CMD $COMMON_FLAGS --fl_algo full_info --first_seed 1 --run_id ${RUN_ID}_fi
echo '======================================================================'
echo 'Finished: \$(date)'
echo '======================================================================'
"

echo ""
echo "Submitted 3 jobs (run_id=$RUN_ID):"
echo "  $JOB1  — SplitFed  (2 GNNs + mlp_vert)"
echo "  $JOB2  — RestrictedMerge (1 GNN + mlp, merged pair subgraphs)"
echo "  $JOB3  — full_info (1 GNN + mlp, global graph)"
echo ""
echo "Compare Test F1 across all three to test info-restriction vs architecture hypothesis."
