#!/bin/bash
# =============================================================================
# Batch Job Submission - supports packing up to 4 jobs per node
# =============================================================================

CLUSTER="wice"
ACCOUNT="lp_aml_work_study"
PARTITION="gpu_h100"
TIME="14:00:00"
#MEM="64G"
#CPUS="16"
# Single jobs
SINGLE_GPUS="4"
SINGLE_CPUS="16"
SINGLE_MEM="88G"
# Packed jobs (shared across all GPUs on the node)
PACKED_CPUS="4"
PACKED_MEM="83G"
JOB_TEMPLATE="$VSC_DATA/AML_work_study/batch_jobs/job_template.sh"
PYTHON_CMD="python $VSC_DATA/AML_work_study/AML_work_study/main.py"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc"

# =============================================================================
# SINGLE jobs (one experiment per node, 1 GPU)
# Use for: experiments with very different runtimes, or standalone runs
# =============================================================================
SINGLE_EXPERIMENTS=(
"FedAvg GINe small HI --testing --ibm_hp --emlps --batching --num_local_epochs 5 --client_fraction 0.1 --max_workers 4 --testing_seeds 1"
"FedAvg GINe small HI --testing --ibm_hp --emlps --batching --num_local_epochs 5 --client_fraction 0.1 --max_workers 4 --testing_seeds 1 --eval_mode comparable"
)

# =============================================================================
# PACKED jobs (up to 4 experiments per node, 4 GPUs)
# Group experiments with similar expected runtimes together
# Separate groups with "---"
# =============================================================================
PACKED_EXPERIMENTS=(
)

# =============================================================================
# Submit single jobs (1 GPU each)
# =============================================================================
echo "Submitting ${#SINGLE_EXPERIMENTS[@]} single jobs..."
echo ""

for exp in "${SINGLE_EXPERIMENTS[@]}"; do
    read -r fl_algo model size ir extra_flags <<< "$exp"
    JOB_NAME="aml_${fl_algo}_${model}_${size}"

    echo "Submitting single: $JOB_NAME"

    sbatch \
        -M "$CLUSTER" \
        --account="$ACCOUNT" \
        --job-name="$JOB_NAME" \
        --output="logs/${JOB_NAME}_%j.log" \
        --error="logs/${JOB_NAME}_%j.err" \
        --partition="$PARTITION" \
        --time="$TIME" \
        --mem="$SINGLE_MEM" \
        --cpus-per-task="$SINGLE_CPUS" \
        --gpus-per-node="$SINGLE_GPUS" \
        "$JOB_TEMPLATE" "$fl_algo" "$model" "$size" "$ir" $extra_flags

    sleep 1
    echo ""
done

# =============================================================================
# Submit packed jobs (up to 4 GPUs each)
# =============================================================================
echo ""
echo "Submitting packed jobs..."
echo ""

group=()
group_num=1

submit_group() {
    local num_gpus=${#group[@]}
    if [[ $num_gpus -eq 0 ]]; then
        return
    fi

    local JOB_NAME="aml_packed_group${group_num}"
    echo "Submitting packed group $group_num ($num_gpus experiments on $num_gpus GPUs)"

    # Build the GPU commands
    local gpu_cmds=""
    for i in "${!group[@]}"; do
        read -r fl_algo model size ir extra_flags <<< "${group[$i]}"
        echo "  GPU $i: $fl_algo $model $size $ir $extra_flags"
        gpu_cmds="${gpu_cmds}CUDA_VISIBLE_DEVICES=$i $PYTHON_CMD --fl_algo $fl_algo --model $model --size $size --ir $ir $extra_flags &"$'\n'
    done

    sbatch \
        -M "$CLUSTER" \
        --account="$ACCOUNT" \
        --job-name="$JOB_NAME" \
        --output="logs/${JOB_NAME}_%j.log" \
        --error="logs/${JOB_NAME}_%j.err" \
        --partition="$PARTITION" \
        --nodes=1 \
        --time="$TIME" \
        --mem="$PACKED_MEM" \
        --cpus-per-task="$PACKED_CPUS" \
        --gpus-per-node="$num_gpus" \
        --wrap "$CONDA_SETUP
echo 'Starting $num_gpus experiments on $num_gpus GPUs'
${gpu_cmds}wait
echo 'All experiments complete'"

    group_num=$((group_num + 1))
    sleep 1
    echo ""
}

for exp in "${PACKED_EXPERIMENTS[@]}"; do
    if [[ "$exp" == "---" ]]; then
        submit_group
        group=()
        continue
    fi
    group+=("$exp")
done
# Submit last group if not empty
submit_group

echo "======================================================================"
echo "Done!"
