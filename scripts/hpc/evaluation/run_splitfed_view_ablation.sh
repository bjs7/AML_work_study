#!/bin/bash
# =============================================================================
# Test-time view ablation for the trained SplitFed system models (evaluation only).
# Submits one job per seed (1-4) of run 20260712_195549. Each job loads
# seed_k/model.pth, runs one test pass and scores three conditions
# (both / sender_only / receiver_only), writing
#   <RUN_DIR>/evaluation/seed_k/predictions_evalseed{k}.pkl
#   <RUN_DIR>/evaluation/seed_k/eval_config_evalseed{k}.json
# =============================================================================
# Usage:
#   bash scripts/hpc/evaluation/run_splitfed_view_ablation.sh
#   bash scripts/hpc/evaluation/run_splitfed_view_ablation.sh --eval_seed 101   # sampling-variance check
# =============================================================================

RUN_DIR="/data/leuven/362/vsc36278/AML_work_study/experiments/small_HI/split_0.6_0.2/system/FedGraphSimple/GINe__emlps/batching__ibm_hp/20260712_195549"
SEEDS=(1 2 3 4)
EXTRA_FLAGS="$*"

CLUSTER="wice"
ACCOUNT="lp_aml_work_study"
PARTITION="gpu_h100"
CPUS="16"
MEM="182G"
GPUS="1"
TIME="04:00:00"

PYTHON_CMD="python $VSC_DATA/AML_work_study/AML_work_study/scripts/hpc/evaluation/splitfed_view_ablation.py"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc"

# Must match the flags the models were trained with (run_vertical_simple_system.sh).
FLAGS="--fl_algo SplitFed --model GINe --size small --ir HI \
--batching --ibm_hp --emlps --eval_mode system \
--max_workers $CPUS --batching_mode lazy_link_neighbor \
--run_dir $RUN_DIR"

mkdir -p logs

for SEED in "${SEEDS[@]}"; do
    JOB_NAME="aml_sf_viewabl_sys_s${SEED}"
    echo "Submitting: $JOB_NAME"
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
echo 'SplitFed view ablation (system) seed $SEED  Node: \$SLURM_NODELIST  Started: \$(date)'
echo '======================================================================'
$PYTHON_CMD $FLAGS --seeds $SEED $EXTRA_FLAGS
echo '======================================================================'
echo 'Finished: \$(date)'
echo '======================================================================'
"
done

echo ""
echo "Submitted ${#SEEDS[@]} jobs. Outputs: $RUN_DIR/evaluation/seed_*/"
