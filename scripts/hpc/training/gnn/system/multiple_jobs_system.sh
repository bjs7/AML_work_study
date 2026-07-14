#!/bin/bash
# =============================================================================
# System eval mode jobs — packed on A100 nodes (wice, 4 GPUs/node)
# Each experiment gets 1 GPU via CUDA_VISIBLE_DEVICES.
# Group experiments with similar expected runtimes; separate groups with "---".
# ~100 min per FL experiment on A100 system mode, 1 worker.
#
# Node specs (Icelake GPU, gpu_a100): 72 cores, 512 GB RAM, 4x A100 SXM2 80GB
# With 4 jobs packed per node: 18 CPUs and 125 GB RAM per process.
#
# NOTE: Verify A100 partition name — check with: sinfo -M wice
# =============================================================================

CLUSTER="wice"
ACCOUNT="lp_aml_work_study"
PARTITION="gpu_a100"       # Verify: may also be gpu_a100_t4 or similar
TIME="20:00:00"
FULL_INFO_TIME="24:00:00" # Full_info is slow: ~12h on H100 comparable; estimate longer on A100 system mode
# Packed jobs (A100 node: 72 cores, 503 GiB RAM, 4x A100-SXM4-80GB; MaxMemPerCPU=7000 MiB)
NODE_GPUS="4"
NODE_CPUS="72"
NODE_MEM="488"        # Max safe RAM: MaxMemPerCPU=7000 MiB × 72 CPUs = 492 GiB; use 488 for headroom
PYTHON_CMD="python $VSC_DATA/AML_work_study/AML_work_study/main.py"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc"

# =============================================================================
# PACKED experiments (up to 4 per node, 1 GPU each)
# Common flags: --eval_mode system --max_workers 1
#
# Flag notes:
#   S1  R0 replication: --replicate_ibm --emlps (batching auto-enabled by parser)
#   F0  FedSGD-like: full batch (no --batching), C=1.0, E=1, uniform — may be SLOW
#   F3  Full batch variant: no --batching, C=0.1, E=5
#   S4  Individual no batching: no --batching
#   H5  loss [1,1]  → --loss_ratio 1
#   H6  loss [1,980]→ --loss_ratio 980
#   H7  loss [1,50] → --loss_ratio 50
# =============================================================================
PACKED_EXPERIMENTS=(
# --- Group 1 --- (full_info: slow, gets FULL_INFO_TIME)
# S1: Full-info, R0, BatchNorm (IBM replication)
"full_info GINe small HI --replicate_ibm --emlps --max_workers 1 --eval_mode system"
# S2: Full-info, R1, LayerNorm
"full_info GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode system --batching"
---

# --- Group 2 ---
# S3: Individual, R1, batching
"individual GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode system --batching"
# S4: Individual, R1, no batching (full batch)
"individual GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode system"
# F0: FedSGD-like (full batch, C=1.0, E=1, uniform) — WARNING: potentially very slow
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode system --num_local_epochs 1 --client_fraction 1.0 --weighting uniform"
# F1/S5/H0: FedAvg baseline (B=8192, C=0.1, E=5, proportional)
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode system --batching --num_local_epochs 5 --client_fraction 0.1"
---

# --- Group 3 ---
# F2: FedAvg batch size 4096
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode system --batching --num_local_epochs 5 --client_fraction 0.1 --batch_size 4096"
# F3: FedAvg full batch (B=full, C=0.1, E=5)
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode system --num_local_epochs 5 --client_fraction 0.1"
# F4: FedAvg 1 local epoch (B=8192, C=0.1, E=1)
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode system --batching --num_local_epochs 1 --client_fraction 0.1"
# F5: FedAvg 10 local epochs
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode system --batching --num_local_epochs 10 --client_fraction 0.1"
---

# --- Group 4 ---
# F6: FedAvg 25 local epochs (stress)
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode system --batching --num_local_epochs 25 --client_fraction 0.1"
# F7: FedAvg C=0.25
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode system --batching --num_local_epochs 5 --client_fraction 0.25"
# F8: FedAvg C=0.50
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode system --batching --num_local_epochs 5 --client_fraction 0.50"
# F9: FedAvg uniform weighting
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode system --batching --num_local_epochs 5 --client_fraction 0.1 --weighting uniform"
---

# --- Group 5 ---
# F10: FedAvg 50 rounds (B=8192, C=0.1, E=5, T=50)
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode system --batching --num_local_epochs 5 --client_fraction 0.1 --num_rounds 50"
# P1: FedProx mu=0.01
"FedProx GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode system --batching --num_local_epochs 5 --client_fraction 0.1 --mu 0.01"
# P2/S6: FedProx mu=0.1
"FedProx GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode system --batching --num_local_epochs 5 --client_fraction 0.1 --mu 0.1"
# P3: FedProx mu=1.0
"FedProx GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode system --batching --num_local_epochs 5 --client_fraction 0.1 --mu 1.0"
---

# --- Group 6 ---
# H1: Remove largest 10 training parties
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode system --batching --num_local_epochs 5 --client_fraction 0.1 --bank_filter no_top10"
# H2: Remove largest 1 training party
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode system --batching --num_local_epochs 5 --client_fraction 0.1 --bank_filter no_top1"
# H3: Remove smallest 10 training parties
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode system --batching --num_local_epochs 5 --client_fraction 0.1 --bank_filter no_bottom10"
# H4: Remove bottom 5% training parties
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode system --batching --num_local_epochs 5 --client_fraction 0.1 --bank_filter no_bottom5pct"
---

# --- Group 7 ---
# H5: Loss weights [1,1] (no class weighting)
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode system --batching --num_local_epochs 5 --client_fraction 0.1 --loss_ratio 1"
# H6: Loss weights [1,980] (inverse-frequency)
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode system --batching --num_local_epochs 5 --client_fraction 0.1 --loss_ratio 980"
# H7: Loss weights [1,50] (clipped inverse-frequency)
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode system --batching --num_local_epochs 5 --client_fraction 0.1 --loss_ratio 50"
# H8: Normalize currency
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode system --batching --num_local_epochs 5 --client_fraction 0.1 --normalize_currency"
)

# =============================================================================
# Submit packed jobs (up to 4 GPUs each)
# =============================================================================
echo "Submitting system mode jobs (packed on A100)..."
echo ""

group=()
group_num=1

submit_group() {
    local num_gpus=${#group[@]}
    if [[ $num_gpus -eq 0 ]]; then
        return
    fi

    # Scale CPUs and memory proportionally to number of GPUs used
    local cpus=$(( NODE_CPUS * num_gpus / NODE_GPUS ))
    local mem=$(( NODE_MEM * num_gpus / NODE_GPUS ))G
    # For very small groups, enforce a sensible minimum
    if [[ $cpus -lt 4 ]]; then cpus=4; fi
    if [[ ${mem%G} -lt 20 ]]; then mem=20G; fi

    # Group 1 contains full_info experiments which are much slower
    local group_time="$TIME"
    if [[ $group_num -eq 1 ]]; then
        group_time="$FULL_INFO_TIME"
    fi

    local JOB_NAME="aml_system_group${group_num}"
    echo "Submitting group $group_num ($num_gpus experiments on $num_gpus GPUs, ${cpus} CPUs, ${mem} RAM, walltime=${group_time})"

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
        --time="$group_time" \
        --mem="$mem" \
        --cpus-per-task="$cpus" \
        --gpus-per-node="$num_gpus" \
        --mail-type=END,FAIL \
        --mail-user=bjoern.strandgaard@kuleuven.be \
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
echo "Done! Submitted 7 packed groups (system eval mode)"
