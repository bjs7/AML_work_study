#!/bin/bash
# =============================================================================
# F1 + P2 SGD LR tuning — system evaluation, expanded search range.
#
# Two independent jobs submitted separately (one for FedAvg, one for FedProx).
# Each job:
#   1. Runs the LR grid search (15 log-spaced candidates, 0.1 → 5.0).
#      Grid (approx): 0.100, 0.130, 0.169, 0.219, 0.285, 0.370, 0.481,
#                     0.707, 0.919, 1.194, 1.552, 2.017, 2.620, 3.405, 5.000
#   2. Each candidate trains until no validation F1 improvement for 50
#      consecutive rounds (--patience 50), then moves to the next candidate.
#   3. Best LR is used for a single-seed final training run (--testing_seeds 1).
#
# Why expanded range? The previous grid (0.3–3.0) found LR near the top;
# expanding to 0.1–5.0 with 15 points gives finer coverage across a wider range.
#
# Cluster: genius (Cascadelake GPU, gpu_v100)
# Node specs: 8× V100 32GB, 36 cores, 768 GB RAM
# Per-job policy: 1 GPU, 4 CPUs, 82 GiB
# Walltime: 36 h (15 candidates × patience-capped + 1-seed training)
# =============================================================================
# Usage:
#   bash scripts/hpc/tuning/gnn/system/run_f1_p2_sgd_tuning_system.sh
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

RUN_ID="$(date +%Y%m%d_%H%M%S)"

COMMON="--model GINe --size small --ir HI \
--ibm_hp --emlps --batching --eval_mode system \
--num_local_epochs 5 --client_fraction 0.1 --max_workers $CPUS \
--optimizer sgd --lr_lower 0.1 --lr_upper 5.0 \
--patience 50 --testing_seeds 1 --tune_run"

# --- Job 1: FedAvg SGD ---
JOB_F1="aml_f1_sgd_tune_system"
echo "Submitting $JOB_F1 (run_id=${RUN_ID}_f1, saved under experiments/tuning/) ..."
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
CUDA_VISIBLE_DEVICES=0 $PYTHON_CMD --fl_algo FedAvg $COMMON --run_id ${RUN_ID}_f1"

# --- Job 2: FedProx SGD (mu=0.1) ---
JOB_P2="aml_p2_sgd_tune_system"
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
CUDA_VISIBLE_DEVICES=0 $PYTHON_CMD --fl_algo FedProx $COMMON --mu 0.1 --run_id ${RUN_ID}_p2"

echo ""
echo "Submitted 2 independent jobs (F1 and P2 SGD tuning, system eval, V100)."
echo "Best LR per job visible in the log: grep 'SGD LR grid' <logfile>"
echo "Winning HP saved to hyper_parameters.json in the results folder."
