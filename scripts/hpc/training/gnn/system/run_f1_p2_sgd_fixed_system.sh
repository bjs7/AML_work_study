#!/bin/bash
# =============================================================================
# F1 + P2 with SGD optimizer, fixed LR=0.5, system evaluation.
#
# Uses --lr_override 0.5: skips the LR grid search entirely and trains
# directly with LR=0.5 (the best value found during full_info SGD tuning).
# All other HPs stay at IBM defaults (--ibm_hp).
#
#   F1-SGD: FedAvg,  B=8192, C=0.1, E=5, proportional weighting
#   P2-SGD: FedProx, B=8192, C=0.1, E=5, mu=0.1
#
# Two independent sbatch jobs, submitted separately.
#
# Cluster: genius (Cascadelake GPU, gpu_v100)
# Node specs: 8× V100 32GB, 36 cores, 768 GB RAM
# Per-job policy: 1 GPU, 4 CPUs, 82 GiB
# Walltime: 36 h (2 seeds × ~15 h system-mode FedAvg on V100)
# =============================================================================
# Usage:
#   bash scripts/hpc/training/gnn/system/run_f1_p2_sgd_fixed_system.sh
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

mkdir -p logs

COMMON="--model GINe --size small --ir HI \
--ibm_hp --emlps --batching --eval_mode system \
--num_local_epochs 5 --client_fraction 0.1 --max_workers $CPUS \
--optimizer sgd --lr_override 0.5 --testing_seeds 2"

# --- Job 1: FedAvg SGD ---
JOB_F1="aml_f1_sgd_fixed_system"
echo "Submitting $JOB_F1 ..."
sbatch \
    -M "$CLUSTER" \
    --account="$ACCOUNT" \
    --job-name="$JOB_F1" \
    --output="logs/${JOB_F1}_%j.log" \
    --error="logs/${JOB_F1}_%j.err" \
    --partition="$PARTITION" \
    --nodes=1 \
    --time="$TIME" \
    --mem="$MEM" \
    --cpus-per-task="$CPUS" \
    --gpus-per-node="$GPUS" \
    --mail-type=END,FAIL \
    --mail-user=bjoern.strandgaard@kuleuven.be \
    --wrap "$CONDA_SETUP
CUDA_VISIBLE_DEVICES=0 $PYTHON_CMD --fl_algo FedAvg $COMMON"

# --- Job 2: FedProx SGD (mu=0.1) ---
JOB_P2="aml_p2_sgd_fixed_system"
echo "Submitting $JOB_P2 ..."
sbatch \
    -M "$CLUSTER" \
    --account="$ACCOUNT" \
    --job-name="$JOB_P2" \
    --output="logs/${JOB_P2}_%j.log" \
    --error="logs/${JOB_P2}_%j.err" \
    --partition="$PARTITION" \
    --nodes=1 \
    --time="$TIME" \
    --mem="$MEM" \
    --cpus-per-task="$CPUS" \
    --gpus-per-node="$GPUS" \
    --mail-type=END,FAIL \
    --mail-user=bjoern.strandgaard@kuleuven.be \
    --wrap "$CONDA_SETUP
CUDA_VISIBLE_DEVICES=0 $PYTHON_CMD --fl_algo FedProx $COMMON --mu 0.1"

echo ""
echo "Submitted 2 independent jobs (F1 and P2 SGD fixed LR=0.5, system eval, V100)."
