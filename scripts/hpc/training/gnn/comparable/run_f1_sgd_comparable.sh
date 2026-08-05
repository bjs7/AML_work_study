#!/bin/bash
# =============================================================================
# F1-SGD: FedAvg comparable eval with SGD optimizer.
# Mirrors the F1 experiment from multiple_jobs_comparable.sh (B=8192, C=0.1,
# E=5, proportional weighting) but uses --optimizer sgd instead of Adam.
#
# Cluster: genius (Cascadelake GPU, gpu_v100)
# Resources: 1 GPU, 4 CPUs, 82 GiB — same per-GPU policy as packed jobs.
# Walltime: 12 h (same as other FedAvg comparable jobs).
# =============================================================================

CLUSTER="genius"
ACCOUNT="lp_aml_work_study"
PARTITION="gpu_v100"
TIME="12:00:00"
CPUS="4"
MEM="82G"
GPUS="1"

PYTHON_CMD="python $VSC_DATA/AML_work_study/AML_work_study/main.py"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc"

JOB_NAME="aml_f1_sgd_comparable"

mkdir -p logs

echo "Submitting F1-SGD comparable job..."

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
    --fl_algo FedAvg --model GINe --size small --ir HI \
    --ibm_hp --emlps --batching --eval_mode comparable \
    --num_local_epochs 5 --client_fraction 0.1 --max_workers 1 \
    --optimizer sgd"

echo "Done."
