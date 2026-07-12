#!/bin/bash
# =============================================================================
# Submit FedGraphSimple DP runs (comparable eval) across noise levels.
#
# For each dp_noise_scale, submits 2 parallel seed jobs sharing a run_id.
# Results for each noise level are saved to a separate experiment directory.
#
# Noise levels: 0.1, 0.5, 1.0
# Seeds: 1 and 2 (parallel per noise level)
# Walltime: ~25h/seed on A100 (50 rounds, comparable eval)
# =============================================================================
# Usage:
#   bash scripts/hpc/training/run_vertical_simple_dp.sh
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

DP_NOISE_LEVELS="0.1 0.5 1.0"

mkdir -p logs

for DP_NOISE in $DP_NOISE_LEVELS; do
    RUN_ID=$(date +%Y%m%d_%H%M%S)
    sleep 1  # ensure unique RUN_IDs across noise levels

    echo "Submitting noise_scale=$DP_NOISE (run_id=$RUN_ID)"

    for SEED in 1 2; do
        JOB_NAME="aml_fgs_dp${DP_NOISE}_s${SEED}"
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
echo 'FedGraphSimple DP — noise_scale=$DP_NOISE seed $SEED of 2 (run_id=$RUN_ID)'
echo '======================================================================'
$PYTHON_CMD $BASE_FLAGS --dp_noise_scale $DP_NOISE --first_seed $SEED --run_id $RUN_ID
echo '======================================================================'
echo 'Job finished at: \$(date)'
echo '======================================================================'
"
        sleep 1
    done
done

echo ""
echo "Submitted FedGraphSimple DP jobs for noise levels: $DP_NOISE_LEVELS"
echo "Each noise level has 2 parallel seeds in its own experiment directory."
