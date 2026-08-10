#!/bin/bash
# =============================================================================
# SGD diagnostic: full_info with SGD optimizer — system eval.
# Runs HP tuning (--tune) so hyper_sampler uses the SGD-aware LR range
# [0.01, 0.5] rather than the Adam-tuned IBM default (~0.006).
# If full_info learns with SGD, the issue in FedAvg is federation-specific.
#
# Cluster: genius (Cascadelake GPU, gpu_v100)
# Resources: 1 GPU, 4 CPUs, 82 GiB — same per-GPU policy as packed jobs.
# Walltime: 32 h (tuning + training on full graph).
# =============================================================================

CLUSTER="genius"
ACCOUNT="lp_aml_work_study"
PARTITION="gpu_v100"
TIME="36:00:00"
CPUS="4"
MEM="82G"
GPUS="1"

PYTHON_CMD="python $VSC_DATA/AML_work_study/AML_work_study/main.py"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc"

JOB_NAME="aml_f1_sgd_system_full_info"

mkdir -p logs

echo "Submitting F1-SGD system job..."

sbatch \
    -M "$CLUSTER" \
    --account="$ACCOUNT" \
    --job-name="$JOB_NAME" \
    --output="logs/${JOB_NAME}_%j.log" \
    --error="logs/${JOB_NAME}_%j.err" \
    --partition="$PARTITION" \
    --nodes=1 \
    --time="$TIME" \
    --mem="$MEM" \
    --cpus-per-task="$CPUS" \
    --gpus-per-node="$GPUS" \
    --mail-type=END,FAIL \
    --mail-user=bjoern.strandgaard@kuleuven.be \
    --wrap "$CONDA_SETUP
CUDA_VISIBLE_DEVICES=0 $PYTHON_CMD \
    --fl_algo full_info --model GINe --size small --ir HI \
    --ibm_hp --emlps --batching --eval_mode system \
    --max_workers 1 --optimizer sgd --testing_seeds 1"

echo "Done."
