#!/bin/bash
# =============================================================================
# Sanity-check smoke test: full_info vs FedGraphSimple (lazy)
# Both run 2 epochs / 2 seeds on full data so F1 scores are comparable.
#
# Expect full_info to converge faster (sees the complete global graph).
# If FedGraphSimple ≈ full_info after only 2 epochs → red flag suggesting
# a data leak or bug.
#
# Expected walltime:
#   full_info : ~30-60 min for 2 epochs
#   lazy      : ~4-5h for 2 epochs (confirmed from prior smoke test)
# =============================================================================
# Usage:
#   bash scripts/run_gnn_comparison_smoketest.sh
# =============================================================================

CLUSTER="wice"
ACCOUNT="lp_aml_work_study"
PARTITION="gpu_a100"
CPUS="16"
MEM="88G"
GPUS="1"

PYTHON_CMD="python $VSC_DATA/AML_work_study/AML_work_study/main.py"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc"

COMMON_FLAGS="--model GINe --size small --ir HI --ibm_hp --emlps \
--eval_mode comparable --batching --num_rounds 3 --testing_seeds 2"

mkdir -p logs

# --- full_info ---
JOB_NAME="aml_full_info_smoketest"
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
    --time="02:00:00" \
    --mem="$MEM" \
    --cpus-per-task="$CPUS" \
    --gpus-per-node="$GPUS" \
    --mail-type=END,FAIL \
    --mail-user=bjoern.strandgaard@kuleuven.be \
    --wrap "$CONDA_SETUP
echo '======================================================================'
echo 'Job started at: \$(date)'
echo 'Job ID: \$SLURM_JOB_ID  Node: \$SLURM_NODELIST'
echo 'Scenario: full_info (smoke test: 2 epochs, full data)'
echo '======================================================================'
$PYTHON_CMD --fl_algo full_info $COMMON_FLAGS
echo '======================================================================'
echo 'Job finished at: \$(date)'
echo '======================================================================'
"
sleep 1

# --- FedGraphSimple lazy_link_neighbor ---
JOB_NAME="aml_fedgraphsimple_lazy_smoketest2"
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
    --time="08:00:00" \
    --mem="$MEM" \
    --cpus-per-task="$CPUS" \
    --gpus-per-node="$GPUS" \
    --mail-type=END,FAIL \
    --mail-user=bjoern.strandgaard@kuleuven.be \
    --wrap "$CONDA_SETUP
echo '======================================================================'
echo 'Job started at: \$(date)'
echo 'Job ID: \$SLURM_JOB_ID  Node: \$SLURM_NODELIST'
echo 'Scenario: FedGraphSimple lazy_link_neighbor (smoke test: 2 epochs, full data)'
echo '======================================================================'
$PYTHON_CMD --fl_algo FedGraphSimple --batching_mode lazy_link_neighbor \
    --max_workers $CPUS $COMMON_FLAGS
echo '======================================================================'
echo 'Job finished at: \$(date)'
echo '======================================================================'
"

echo ""
echo "Submitted 2 comparison smoke-test jobs (2 epochs, full data)."
echo "Compare Test F1 scores when both complete:"
echo "  If full_info >> FedGraphSimple → results look legitimate."
echo "  If FedGraphSimple ≈ full_info  → red flag, investigate (data leak / bug)."
