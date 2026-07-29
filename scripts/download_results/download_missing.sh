#!/bin/bash
# Download only the missing comparable experiments

HPC_USER="vsc36278"
HPC_HOST="login.hpc.kuleuven.be"
HPC_BASE="/data/leuven/362/vsc36278/AML_work_study/experiments"
LOCAL_BASE="$HOME/projects/AML_work_study/experiments"

MISSING_PATHS=(
    "small_HI/split_0.6_0.2/system/full_info/GINe__emlps/batching__batchnorm__ibm_fe__ibm_hp"                          # S1
)

echo "Downloading ${#MISSING_PATHS[@]} missing experiments..."
echo "======================================================================"

for exp_path in "${MISSING_PATHS[@]}"; do
    echo "[$exp_path]"

    latest_folder=$(ssh ${HPC_USER}@${HPC_HOST} "ls -1 ${HPC_BASE}/${exp_path} 2>/dev/null | sort -r | head -1")

    if [[ -z "$latest_folder" ]]; then
        echo "  ⚠ No results found on server"
        echo ""
        continue
    fi

    echo "  ✓ Found: $latest_folder"

    local_path="${LOCAL_BASE}/${exp_path}/${latest_folder}"
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

    echo "  ✓ Done"
    echo ""
done

echo "======================================================================"
echo "Complete! Results in: $LOCAL_BASE"