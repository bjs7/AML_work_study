#!/bin/bash
# =============================================================================
# Download booster (XGBoost / SecureBoost) results from HPC.
# Covers the 6 standard booster jobs (submit_booster_training.sh) and
# the SecureBoost comparable job (run_secureboost_comparable.sh).
#
# All paths follow:
#   experiments/small_HI/split_0.6_0.2/{eval_mode}/{fl_algo}/xgboost/{data_folder}/
#
# Usage:
#   bash scripts/data/download_booster_results.sh             # download all
#   bash scripts/data/download_booster_results.sh --dry-run   # preview paths only
# =============================================================================

HPC_USER="vsc36278"
HPC_HOST="login.hpc.kuleuven.be"
HPC_BASE="/data/leuven/362/vsc36278/AML_work_study/experiments"
LOCAL_BASE="$HOME/projects/AML_work_study/experiments"

DRY_RUN=false
[[ "$1" == "--dry-run" ]] && DRY_RUN=true && echo "DRY RUN — no files will be downloaded"

# =============================================================================
# Experiment paths (relative to HPC_BASE)
# Format: "description|relative/path"
# =============================================================================
EXPERIMENTS=(
    "full_info system standard FE|small_HI/split_0.6_0.2/system/full_info/xgboost/default"
    "full_info system IBM FE      |small_HI/split_0.6_0.2/system/full_info/xgboost/ibm_fe"
    "individual system standard FE|small_HI/split_0.6_0.2/system/individual/xgboost/default"
    "full_info comparable standard FE|small_HI/split_0.6_0.2/comparable/full_info/xgboost/default"
    "full_info comparable IBM FE      |small_HI/split_0.6_0.2/comparable/full_info/xgboost/ibm_fe"
    "individual comparable standard FE|small_HI/split_0.6_0.2/comparable/individual/xgboost/default"
    "SecureBoost comparable|small_HI/split_0.6_0.2/comparable/SecureBoost/xgboost/default"
)

# =============================================================================
# Download one experiment (latest run folder only)
# =============================================================================
download_experiment() {
    local desc=$1
    local rel_path=$2

    echo "[$desc]"
    echo "  Path: $rel_path"

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] Would download from: ${HPC_USER}@${HPC_HOST}:${HPC_BASE}/${rel_path}/"
        echo ""
        return
    fi

    local latest
    latest=$(ssh ${HPC_USER}@${HPC_HOST} "ls -1 ${HPC_BASE}/${rel_path} 2>/dev/null | sort -r | head -1")

    if [[ -z "$latest" ]]; then
        echo "  ⚠  No results found — skipping"
        echo ""
        return
    fi

    echo "  Latest run: $latest"
    local local_path="${LOCAL_BASE}/${rel_path}/${latest}"
    mkdir -p "$local_path"

    rsync -avz --progress \
        --include='*/' \
        --include='*.json' \
        --include='*.pkl' \
        --exclude='*.pth' \
        --exclude='*.ubj' \
        --exclude='*' \
        "${HPC_USER}@${HPC_HOST}:${HPC_BASE}/${rel_path}/${latest}/" \
        "$local_path/"

    echo "  ✓ Saved to: $local_path"
    echo ""
}

# =============================================================================
# Main
# =============================================================================
echo "======================================================================"
echo "Downloading booster results from HPC"
echo "======================================================================"
echo ""

mkdir -p "$LOCAL_BASE"

for entry in "${EXPERIMENTS[@]}"; do
    desc="${entry%%|*}"
    rel_path="${entry##*|}"
    download_experiment "$desc" "$rel_path"
done

echo "======================================================================"
echo "Done. Results in: $LOCAL_BASE"
