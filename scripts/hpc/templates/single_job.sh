#!/bin/bash
# =============================================================================
# Easy Single Job Submission Script
# =============================================================================
# Usage:
#   ./scripts/submit_single.sh --fl_algo full_info --model GINe
#   ./scripts/submit_single.sh --fl_algo fedavg --size medium --time 12:00:00
#   ./scripts/submit_single.sh --fl_algo individual --testing --tqdm
#   ./scripts/submit_single.sh --fl_algo FedGraph --ibm_fe --ibm_hp --batching
# =============================================================================

CLUSTER="wice"
ACCOUNT="lp_aml_work_study"
PARTITION="gpu_h100"
NODES="1"
NTASKS="1"

# Default values for main arguments
FL_ALGO="Individual"
MODEL="GINe"
SIZE="small"
IR="HI"

# Default SBATCH resources
TIME="36:00:00"
MEM="64G"
CPUS="8"
GPU="1"

# Extra flags to pass to main.py
EXTRA_FLAGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        # Main experiment parameters
        --fl_algo)
            FL_ALGO="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --size)
            SIZE="$2"
            shift 2
            ;;
        --ir)
            IR="$2"
            shift 2
            ;;
        --seed)
            EXTRA_FLAGS="$EXTRA_FLAGS --seed $2"
            shift 2
            ;;
        --aggregation)
            EXTRA_FLAGS="$EXTRA_FLAGS --aggregation $2"
            shift 2
            ;;
        --banks)
            EXTRA_FLAGS="$EXTRA_FLAGS --banks $2"
            shift 2
            ;;
        --testing_seeds)
            EXTRA_FLAGS="$EXTRA_FLAGS --testing_seeds $2"
            shift 2
            ;;

        # SBATCH resource parameters
        --time)
            TIME="$2"
            shift 2
            ;;
        --mem)
            MEM="$2"
            shift 2
            ;;
        --cpus)
            CPUS="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;

        # Boolean flags (pass through to main.py)
        --testing|--tqdm|--batching|--overwrite|--train_for_final|\
        --ibm_fe|--ibm_hp|--emlps|--ports|--tds|--reverse_mp|--use_global_stats)
            EXTRA_FLAGS="$EXTRA_FLAGS $1"
            shift
            ;;

        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Create job name
JOB_NAME="aml_${FL_ALGO}_${MODEL}_${SIZE}"

# Submit job
echo "Submitting job: $JOB_NAME"
echo "Configuration: FL_ALGO=$FL_ALGO MODEL=$MODEL SIZE=$SIZE IR=$IR"
echo "Resources: TIME=$TIME MEM=$MEM CPUS=$CPUS GPU=$GPU"
echo "Extra flags: $EXTRA_FLAGS"
echo ""

sbatch \
    --clusters="$CLUSTER" \
    --account="$ACCOUNT" \
    --job-name="$JOB_NAME" \
    --output="logs/${JOB_NAME}_%j.log" \
    --error="logs/${JOB_NAME}_%j.err" \
    --time="$TIME" \
    --mem="$MEM" \
    --nodes="$NODES" \
    --ntasks="$NTASKS" \
    --cpus-per-task="$CPUS" \
    --gpus-per-node="$GPU" \
    --partition="$PARTITION" \
    $VSC_DATA/AML_work_study/batch_jobs/job_template.sh "$FL_ALGO" "$MODEL" "$SIZE" "$IR" $EXTRA_FLAGS

echo "Job submitted successfully!"
