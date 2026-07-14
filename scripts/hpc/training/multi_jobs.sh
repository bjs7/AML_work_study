#!/bin/bash
# =============================================================================
# Simple Batch Job Submission
# =============================================================================

CLUSTER="wice"
ACCOUNT="lp_aml_work_study"
PARTITION="gpu_a100"

TIME="16:00:00"
MEM="42G"
CPUS="2"
GPU="1"

# =============================================================================
# CONFIGURE YOUR EXPERIMENTS HERE
# Format: "fl_algo model size ir [extra_flags]"
# Extra flags are optional - only add them when you need them
# =============================================================================

EXPERIMENTS=(
    "FedAvg GINe small HI --ibm_hp --batching --client_fraction 0.1 --num_local_epochs 5 --emlps"
)

# =============================================================================
# Submit the jobs
# =============================================================================

echo "Submitting ${#EXPERIMENTS[@]} jobs..."
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    # Read first 4 arguments, everything else goes into extra_flags
    read -r fl_algo model size ir extra_flags <<< "$exp"

    # Create job name
    JOB_NAME="aml_${fl_algo}_${model}_${size}"

    echo "Submitting: $JOB_NAME"
    if [[ -n "$extra_flags" ]]; then
        echo "  Extra flags: $extra_flags"
    fi

    sbatch \
        -M "$CLUSTER" \
        --account="$ACCOUNT" \
        --job-name="$JOB_NAME" \
        --output="logs/${JOB_NAME}_%j.log" \
        --error="logs/${JOB_NAME}_%j.err" \
        --partition="$PARTITION" \
        --time="$TIME" \
        --mem="$MEM" \
        --cpus-per-task="$CPUS" \
        --gpus-per-node="$GPU" \
        $VSC_DATA/AML_work_study/batch_jobs/job_template.sh "$fl_algo" "$model" "$size" "$ir" $extra_flags

    sleep 1
    echo ""
done

echo "Done! Submitted ${#EXPERIMENTS[@]} jobs"
