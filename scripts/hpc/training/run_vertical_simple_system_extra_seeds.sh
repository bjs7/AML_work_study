#!/bin/bash
# =============================================================================
# Submit additional FedGraphSimple system eval seed jobs into an existing run directory.
#
# Usage:
#   bash scripts/hpc/training/run_vertical_simple_system_extra_seeds.sh <RUN_ID> <FIRST_SEED> <LAST_SEED>
#
# Example — add seeds 3 and 4 to an existing run:
#   bash scripts/hpc/training/run_vertical_simple_system_extra_seeds.sh 20250501_143022 3 4
# =============================================================================

RUN_ID="${1:?Usage: $0 <RUN_ID> <FIRST_SEED> <LAST_SEED>}"
FIRST_SEED="${2:?Usage: $0 <RUN_ID> <FIRST_SEED> <LAST_SEED>}"
LAST_SEED="${3:?Usage: $0 <RUN_ID> <FIRST_SEED> <LAST_SEED>}"

CLUSTER="wice"
ACCOUNT="lp_aml_work_study"
PARTITION="gpu_h100"
CPUS="16"
MEM="192G"
GPUS="1"

PYTHON_CMD="python $VSC_DATA/AML_work_study/AML_work_study/main.py"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc"

BASE_FLAGS="--fl_algo FedGraphSimple --model GINe --size small --ir HI \
--batching --ibm_hp --emlps --eval_mode system \
--max_workers $CPUS --testing_seeds 1 --batching_mode lazy_link_neighbor --num_rounds 50"

mkdir -p logs

for SEED in $(seq $FIRST_SEED $LAST_SEED); do
    JOB_NAME="aml_fedgraphsimple_sys_s${SEED}"
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
echo 'FedGraphSimple lazy system — seed $SEED (run_id=$RUN_ID)'
echo '======================================================================'
$PYTHON_CMD $BASE_FLAGS --first_seed $SEED --run_id $RUN_ID
echo '======================================================================'
echo 'Job finished at: \$(date)'
echo '======================================================================'
"
    sleep 1
done

echo ""
echo "Submitted seeds $FIRST_SEED-$LAST_SEED into existing system eval run directory (run_id=$RUN_ID)."
echo "The job that finishes last will write the full aggregated result across all seeds."
