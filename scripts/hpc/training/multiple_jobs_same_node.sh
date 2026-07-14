#!/bin/bash
# =============================================================================
# Batch Job Submission - supports packing up to 4 jobs per node
# =============================================================================

CLUSTER="wice"
ACCOUNT="lp_aml_work_study"
PARTITION="gpu_a100"
TIME="24:00:00"
#MEM="64G"
#CPUS="16"
# Single jobs
SINGLE_CPUS="8"
SINGLE_MEM="32G"
# Packed jobs (shared across all GPUs on the node)
PACKED_CPUS="32"
PACKED_MEM="128G"
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
    "full_info GINe small HI --replicate_ibm --emlps"
    "full_info GINe small HI --ibm_hp --batching"
    "full_info GINe small HI --ibm_hp --batching --emlps"
    "individual GINe small HI --ibm_hp --batching --emlps"
    "individual GINe small HI --ibm_hp --emlps"
)

# =============================================================================
# PACKED jobs (up to 4 experiments per node, 4 GPUs)
# Group experiments with similar expected runtimes together
# Separate groups with "---"
# =============================================================================
PACKED_EXPERIMENTS=(
    # --- Group 1: FedAvg batch/epoch sensitivity ---
    "FedAvg GINe small HI --ibm_hp --emlps --num_local_epochs 1 --client_fraction 1.0 --weighting uniform"
    "FedAvg GINe small HI --ibm_hp --emlps --batching --num_local_epochs 5 --client_fraction 0.1"
    "FedAvg GINe small HI --ibm_hp --emlps --batching --batch_size 4096 --num_local_epochs 5 --client_fraction 0.1"
    "FedAvg GINe small HI --ibm_hp --emlps --num_local_epochs 5 --client_fraction 0.1"
    "---"
    # --- Group 2: Local epochs ---
    "FedAvg GINe small HI --ibm_hp --emlps --batching --num_local_epochs 1 --client_fraction 0.1"
    "FedAvg GINe small HI --ibm_hp --emlps --batching --num_local_epochs 10 --client_fraction 0.1"
    "FedAvg GINe small HI --ibm_hp --emlps --batching --num_local_epochs 25 --client_fraction 0.1"
    "FedAvg GINe small HI --ibm_hp --emlps --batching --num_local_epochs 5 --client_fraction 0.1 --num_rounds 50"
    "---"
    # --- Group 3: Client fraction + weighting ---
    "FedAvg GINe small HI --ibm_hp --emlps --batching --num_local_epochs 5 --client_fraction 0.25"
    "FedAvg GINe small HI --ibm_hp --emlps --batching --num_local_epochs 5 --client_fraction 0.50"
    "FedAvg GINe small HI --ibm_hp --emlps --batching --num_local_epochs 5 --client_fraction 0.1 --weighting uniform"
    "---"
    # --- Group 4: FedProx ---
    "FedAvg GINe small HI --ibm_hp --emlps --batching --num_local_epochs 5 --client_fraction 0.1 --mu 0.01"
    "FedAvg GINe small HI --ibm_hp --emlps --batching --num_local_epochs 5 --client_fraction 0.1 --mu 0.1"
    "FedAvg GINe small HI --ibm_hp --emlps --batching --num_local_epochs 5 --client_fraction 0.1 --mu 1.0"
    "---"
    # --- Group 5: Bank filters ---
    "FedAvg GINe small HI --ibm_hp --emlps --batching --num_local_epochs 5 --client_fraction 0.1 --bank_filter no_top10"
    "FedAvg GINe small HI --ibm_hp --emlps --batching --num_local_epochs 5 --client_fraction 0.1 --bank_filter no_top1"
    "FedAvg GINe small HI --ibm_hp --emlps --batching --num_local_epochs 5 --client_fraction 0.1 --bank_filter no_bottom10"
    "FedAvg GINe small HI --ibm_hp --emlps --batching --num_local_epochs 5 --client_fraction 0.1 --bank_filter no_bottom5pct"
    "---"
    # --- Group 6: Loss ratio + currency ---
    "FedAvg GINe small HI --ibm_hp --emlps --batching --num_local_epochs 5 --client_fraction 0.1 --loss_ratio 1"
    "FedAvg GINe small HI --ibm_hp --emlps --batching --num_local_epochs 5 --client_fraction 0.1 --loss_ratio 980"
    "FedAvg GINe small HI --ibm_hp --emlps --batching --num_local_epochs 5 --client_fraction 0.1 --loss_ratio 50"
    "FedAvg GINe small HI --ibm_hp --emlps --batching --num_local_epochs 5 --client_fraction 0.1 --normalize_currency"
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
        --gpus-per-node=1 \
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
