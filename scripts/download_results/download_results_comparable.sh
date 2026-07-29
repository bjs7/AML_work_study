#!/bin/bash
# =============================================================================
# Download comparable eval mode results from HPC (genius, V100)
# Mirrors experiment list from multiple_jobs_comparable.sh
# =============================================================================
# Usage:
#   ./download_results_comparable.sh             # Download all comparable experiments
#   ./download_results_comparable.sh --dry-run   # Preview constructed paths
#   ./download_results_comparable.sh --all       # Download everything under HPC_BASE
# =============================================================================

HPC_USER="vsc36278"
HPC_HOST="login.hpc.kuleuven.be"
HPC_BASE="/data/leuven/362/vsc36278/AML_work_study/experiments"
LOCAL_BASE="$HOME/projects/AML_work_study/experiments"

DRY_RUN=false
DOWNLOAD_ALL=false

if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "DRY RUN MODE - No files will be downloaded"
    echo "======================================================================"
fi

if [[ "$1" == "--all" ]]; then
    DOWNLOAD_ALL=true
fi

# =============================================================================
# All comparable-mode experiments (mirrors multiple_jobs_comparable.sh)
# =============================================================================
EXPERIMENTS=(
# S1: Full-info, R0, BatchNorm (IBM replication)
"full_info GINe small HI --replicate_ibm --emlps --max_workers 1 --eval_mode comparable"
# S2: Full-info, R1, LayerNorm
"full_info GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode comparable --batching"
# S3: Individual, R1, batching
"individual GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode comparable --batching"
# S4: Individual, R1, no batching (full batch)
"individual GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode comparable"
# F0: FedAvg FedSGD-like (full batch, C=1.0, E=1, uniform)
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode comparable --num_local_epochs 1 --client_fraction 1.0 --weighting uniform"
# F1/S5/H0: FedAvg baseline (B=8192, C=0.1, E=5, proportional)
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode comparable --batching --num_local_epochs 5 --client_fraction 0.1"
# F2: FedAvg batch size 4096
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode comparable --batching --num_local_epochs 5 --client_fraction 0.1 --batch_size 4096"
# F3: FedAvg full batch (B=full, C=0.1, E=5)
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode comparable --num_local_epochs 5 --client_fraction 0.1"
# F4: FedAvg 1 local epoch (B=8192, C=0.1, E=1)
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode comparable --batching --num_local_epochs 1 --client_fraction 0.1"
# F5: FedAvg 10 local epochs
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode comparable --batching --num_local_epochs 10 --client_fraction 0.1"
# F6: FedAvg 25 local epochs (stress)
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode comparable --batching --num_local_epochs 25 --client_fraction 0.1"
# F7: FedAvg C=0.25
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode comparable --batching --num_local_epochs 5 --client_fraction 0.25"
# F8: FedAvg C=0.50
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode comparable --batching --num_local_epochs 5 --client_fraction 0.5"
# F9: FedAvg uniform weighting
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode comparable --batching --num_local_epochs 5 --client_fraction 0.1 --weighting uniform"
# F10: FedAvg 50 rounds
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode comparable --batching --num_local_epochs 5 --client_fraction 0.1 --num_rounds 50"
# P1: FedProx mu=0.01
"FedProx GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode comparable --batching --num_local_epochs 5 --client_fraction 0.1 --mu 0.01"
# P2/S6: FedProx mu=0.1
"FedProx GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode comparable --batching --num_local_epochs 5 --client_fraction 0.1 --mu 0.1"
# P3: FedProx mu=1.0
"FedProx GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode comparable --batching --num_local_epochs 5 --client_fraction 0.1 --mu 1.0"
# H1: Remove largest 10 training parties
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode comparable --batching --num_local_epochs 5 --client_fraction 0.1 --bank_filter no_top10"
# H2: Remove largest 1 training party
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode comparable --batching --num_local_epochs 5 --client_fraction 0.1 --bank_filter no_top1"
# H3: Remove smallest 10 training parties
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode comparable --batching --num_local_epochs 5 --client_fraction 0.1 --bank_filter no_bottom10"
# H4: Remove bottom 5% training parties
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode comparable --batching --num_local_epochs 5 --client_fraction 0.1 --bank_filter no_bottom5pct"
# H5: Loss weights [1,1]
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode comparable --batching --num_local_epochs 5 --client_fraction 0.1 --loss_ratio 1"
# H6: Loss weights [1,980]
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode comparable --batching --num_local_epochs 5 --client_fraction 0.1 --loss_ratio 980"
# H7: Loss weights [1,50]
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode comparable --batching --num_local_epochs 5 --client_fraction 0.1 --loss_ratio 50"
# H8: Normalize currency
"FedAvg GINe small HI --ibm_hp --emlps --max_workers 1 --eval_mode comparable --batching --num_local_epochs 5 --client_fraction 0.1 --normalize_currency"
)

# =============================================================================
# construct_path: mirrors save_results.py directory structure
#   {size}_{ir}/split_0.6_0.2/{eval_mode}/{fl_algo}/{algo_subfolder}/{model_folder}/{data_folder}
#
# Notes:
#   --replicate_ibm auto-enables batching=True and batchnorm=True in the parser,
#   so those appear in data_folder even without explicit --batching/--batchnorm flags.
# =============================================================================
construct_path() {
    local fl_algo=$1
    local model=$2
    local size=$3
    local ir=$4
    shift 4
    local extra_flags="$@"

    # eval_mode: extract from --eval_mode flag, default to system
    local eval_mode="system"
    if [[ "$extra_flags" == *"--eval_mode"* ]]; then
        eval_mode=$(echo "$extra_flags" | grep -oP '(?<=--eval_mode )\S+')
    fi

    local base_path="${size}_${ir}/split_0.6_0.2/${eval_mode}/${fl_algo}"

    # Algo subfolder for FedAvg/FedProx
    local algo_subfolder=""
    if [[ "$fl_algo" == "FedAvg" || "$fl_algo" == "FedProx" ]]; then
        local weighting="proportional"
        [[ "$extra_flags" == *"--weighting"* ]] && weighting=$(echo "$extra_flags" | grep -oP '(?<=--weighting )\S+')

        local C="1.0"
        [[ "$extra_flags" == *"--client_fraction"* ]] && C=$(echo "$extra_flags" | grep -oP '(?<=--client_fraction )\S+')

        local E="1"
        [[ "$extra_flags" == *"--num_local_epochs"* ]] && E=$(echo "$extra_flags" | grep -oP '(?<=--num_local_epochs )\S+')

        local R="100"
        [[ "$extra_flags" == *"--num_rounds"* ]] && R=$(echo "$extra_flags" | grep -oP '(?<=--num_rounds )\S+')

        local mu="0.0"
        [[ "$extra_flags" == *"--mu"* ]] && mu=$(echo "$extra_flags" | grep -oP '(?<=--mu )\S+')

        local ve="1"
        [[ "$extra_flags" == *"--validate_every"* ]] && ve=$(echo "$extra_flags" | grep -oP '(?<=--validate_every )\S+')

        algo_subfolder="${weighting}_C${C}_E${E}"
        [[ "$R" != "100" ]] && algo_subfolder="${algo_subfolder}_R${R}"
        [[ "$mu" != "0.0" && "$mu" != "0" ]] && algo_subfolder="${algo_subfolder}_mu${mu}"
        [[ "$ve" != "1" ]] && algo_subfolder="${algo_subfolder}_ve${ve}"
    fi

    # Model folder with GNN flags
    local model_folder="$model"
    [[ "$extra_flags" == *"--emlps"* ]]      && model_folder="${model_folder}__emlps"
    [[ "$extra_flags" == *"--ports"* ]]      && model_folder="${model_folder}__ports"
    [[ "$extra_flags" == *"--tds"* ]]        && model_folder="${model_folder}__tds"
    [[ "$extra_flags" == *"--reverse_mp"* ]] && model_folder="${model_folder}__reverse_mp"

    # Data flags (order matches save_results.py: batching, batchnorm, ibm_fe, ibm_hp, use_global_stats, ...)
    # --replicate_ibm auto-sets batching=True and batchnorm=True
    local data_flags=""
    local has_batching=false
    local has_batchnorm=false
    [[ "$extra_flags" == *"--batching"* || "$extra_flags" == *"--replicate_ibm"* ]] && has_batching=true
    [[ "$extra_flags" == *"--batchnorm"* || "$extra_flags" == *"--replicate_ibm"* ]] && has_batchnorm=true

    $has_batching  && data_flags="${data_flags}batching__"
    $has_batchnorm && data_flags="${data_flags}batchnorm__"
    [[ "$extra_flags" == *"--ibm_fe"* ]]           && data_flags="${data_flags}ibm_fe__"
    [[ "$extra_flags" == *"--ibm_hp"* ]]           && data_flags="${data_flags}ibm_hp__"
    [[ "$extra_flags" == *"--use_global_stats"* ]] && data_flags="${data_flags}use_global_stats__"

    if [[ "$extra_flags" == *"--batching_mode"* ]]; then
        local bm=$(echo "$extra_flags" | grep -oP '(?<=--batching_mode )\S+')
        [[ "$bm" != "lazy_link_neighbor" ]] && data_flags="${data_flags}bm_${bm}__"
    fi

    [[ "$extra_flags" == *"--normalize_currency"* ]] && data_flags="${data_flags}normalize_currency__"

    if [[ "$extra_flags" == *"--bank_filter"* ]]; then
        local bf=$(echo "$extra_flags" | grep -oP '(?<=--bank_filter )\S+')
        data_flags="${data_flags}bank_filter_${bf}__"
    fi

    if [[ "$extra_flags" == *"--loss_ratio"* ]]; then
        local lr=$(echo "$extra_flags" | grep -oP '(?<=--loss_ratio )\S+')
        data_flags="${data_flags}loss_ratio_${lr}__"
    fi

    if [[ "$extra_flags" == *"--batch_size"* ]]; then
        local bs=$(echo "$extra_flags" | grep -oP '(?<=--batch_size )\S+')
        [[ "$bs" != "8192" ]] && data_flags="${data_flags}batch_size_${bs}__"
    fi

    data_flags=${data_flags%__}
    [[ -z "$data_flags" ]] && data_flags="default"

    if [[ -n "$algo_subfolder" ]]; then
        echo "${base_path}/${algo_subfolder}/${model_folder}/${data_flags}"
    else
        echo "${base_path}/${model_folder}/${data_flags}"
    fi
}

# =============================================================================
# Download function
# =============================================================================
download_experiment() {
    local exp_path=$1
    local exp_name=$2

    local remote_path="${HPC_USER}@${HPC_HOST}:${HPC_BASE}/${exp_path}/"

    echo "Checking for experiments in: $exp_path"

    if [[ "$DRY_RUN" == false ]]; then
        local latest_folder
        latest_folder=$(ssh ${HPC_USER}@${HPC_HOST} "ls -1 ${HPC_BASE}/${exp_path} 2>/dev/null | sort -r | head -1")

        if [[ -z "$latest_folder" ]]; then
            echo "  ⚠ No results found for: $exp_name"
            return 1
        fi

        echo "  ✓ Found latest run: $latest_folder"

        local local_path="${LOCAL_BASE}/${exp_path}/${latest_folder}"
        mkdir -p "$local_path"

        echo "  → Downloading to: $local_path"
        rsync -avz --progress \
            --include='*/' \
            --include='*.json' \
            --include='*.pkl' \
            --exclude='*.pth' \
            --exclude='*' \
            "${HPC_USER}@${HPC_HOST}:${HPC_BASE}/${exp_path}/${latest_folder}/" \
            "$local_path/"

        echo "  ✓ Download complete"
    else
        echo "  [DRY RUN] Would download from: $remote_path"
        echo "  [DRY RUN] Constructed path: $exp_path"
    fi

    echo ""
}

# =============================================================================
# Main download logic
# =============================================================================

if [[ "$DOWNLOAD_ALL" == true ]]; then
    echo "Downloading ALL experiments..."
    echo ""

    mkdir -p "$LOCAL_BASE"

    rsync -avz --progress \
        --include='*/' \
        --include='aggregated_results.json' \
        --include='experiment_config.json' \
        --include='hyper_parameters.json' \
        --include='model_tuning_configs.json' \
        --include='*.pkl' \
        --exclude='*.pth' \
        --exclude='*' \
        "${HPC_USER}@${HPC_HOST}:${HPC_BASE}/" \
        "$LOCAL_BASE/"

    echo "Download complete!"
else
    echo "Downloading ${#EXPERIMENTS[@]} comparable-mode experiments..."
    echo "======================================================================"
    echo ""

    for exp in "${EXPERIMENTS[@]}"; do
        read -r fl_algo model size ir extra_flags <<< "$exp"

        exp_name="${fl_algo}_${model}_${size}_${ir}"
        echo "[$exp_name]"

        exp_path=$(construct_path "$fl_algo" "$model" "$size" "$ir" $extra_flags)
        download_experiment "$exp_path" "$exp_name"
    done

    echo "======================================================================"
    echo "Download complete! Results in: $LOCAL_BASE"
fi
