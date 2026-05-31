#!/bin/bash
# =============================================================================
# Download FedGraphSimple (lazy_link_neighbor) results from HPC.
#
# Usage:
#   bash scripts/data/download_vertical_simple.sh <RUN_ID>
#
# Example:
#   bash scripts/data/download_vertical_simple.sh 20260420_214839
# =============================================================================

RUN_ID="${1:?Usage: $0 <RUN_ID>}"

HPC_USER="vsc36278"
HPC_HOST="login.hpc.kuleuven.be"
HPC_BASE="/data/leuven/362/vsc36278/AML_work_study/experiments"
LOCAL_BASE="$HOME/projects/AML_work_study/experiments"

EXP_PATH="small_HI/split_0.6_0.2/comparable/FedGraphSimple/GINe__emlps/batching__ibm_hp/${RUN_ID}"

REMOTE="${HPC_USER}@${HPC_HOST}:${HPC_BASE}/${EXP_PATH}/"
LOCAL="${LOCAL_BASE}/${EXP_PATH}/"

echo "Downloading FedGraphSimple lazy run: $RUN_ID"
echo "Remote: $REMOTE"
echo "Local:  $LOCAL"
echo ""

mkdir -p "$LOCAL"

rsync -avz --progress \
    --include='*/' \
    --include='*.json' \
    --include='*.pkl' \
    --exclude='*.pth' \
    --exclude='*' \
    "$REMOTE" \
    "$LOCAL"

echo ""
echo "Download complete: $LOCAL"
