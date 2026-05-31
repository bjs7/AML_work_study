#!/bin/bash
# =============================================================================
# Submit FedGraphSimple (lazy_link_neighbor) as two parallel seed jobs.
#
# Each job runs 1 seed with a shared --run_id so both write to the same
# experiment directory. Whichever job finishes last picks up the other's
# saved seed and writes the full aggregated result.
#
# Walltime: ~93h/seed on A100; 60h per job fits within cluster limits.
# Seed 1 and seed 2 run in parallel so total wall time is ~93h, not ~186h.
# =============================================================================
# Usage:
#   bash scripts/run_vertical_simple.sh
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

BASE_FLAGS="--fl_algo FedGraphSimple --model GINe --size small --ir HI \
--batching --ibm_hp --emlps --eval_mode comparable \
--max_workers $CPUS --testing_seeds 1 --batching_mode lazy_link_neighbor --num_rounds 50"

# Shared run ID so both jobs write to the same experiment directory
RUN_ID=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

# --- Seed 1 ---
JOB_NAME="aml_fedgraphsimple_lazy_s1"
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
    --time="72:00:00" \
    --mem="$MEM" \
    --cpus-per-task="$CPUS" \
    --gpus-per-node="$GPUS" \
    --mail-type=END,FAIL \
    --mail-user=bjoern.strandgaard@kuleuven.be \
    --wrap "$CONDA_SETUP
echo '======================================================================'
echo 'Job started at: \$(date)'
echo 'Job ID: \$SLURM_JOB_ID  Node: \$SLURM_NODELIST'
echo 'FedGraphSimple lazy — seed 1 of 2 (run_id=$RUN_ID)'
echo '======================================================================'
$PYTHON_CMD $BASE_FLAGS --first_seed 1 --run_id $RUN_ID
echo '======================================================================'
echo 'Job finished at: \$(date)'
echo '======================================================================'
"
sleep 1

# --- Seed 2 ---
JOB_NAME="aml_fedgraphsimple_lazy_s2"
echo "Submitting: $JOB_NAME (seed 2, run_id=$RUN_ID)"
sbatch \
    -M "$CLUSTER" \
    --account="$ACCOUNT" \
    --job-name="$JOB_NAME" \
    --output="logs/${JOB_NAME}_%j.log" \
    --error="logs/${JOB_NAME}_%j.err" \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --time="72:00:00" \
    --mem="$MEM" \
    --cpus-per-task="$CPUS" \
    --gpus-per-node="$GPUS" \
    --mail-type=END,FAIL \
    --mail-user=bjoern.strandgaard@kuleuven.be \
    --wrap "$CONDA_SETUP
echo '======================================================================'
echo 'Job started at: \$(date)'
echo 'Job ID: \$SLURM_JOB_ID  Node: \$SLURM_NODELIST'
echo 'FedGraphSimple lazy — seed 2 of 2 (run_id=$RUN_ID)'
echo '======================================================================'
$PYTHON_CMD $BASE_FLAGS --first_seed 2 --run_id $RUN_ID
echo '======================================================================'
echo 'Job finished at: \$(date)'
echo '======================================================================'
"

echo ""
echo "Submitted 2 parallel FedGraphSimple lazy jobs (run_id=$RUN_ID)."
echo "Both seeds write to the same experiment directory."
echo "The job that finishes last will write the full 2-seed aggregated result."
