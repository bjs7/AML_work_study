#!/bin/bash
# =============================================================================
# Push local source code changes to HPC (wice, KU Leuven VSC)
# Mirrors the pattern of download_results_system.sh / download_missing.sh
#
# Usage:
#   ./scripts/push_to_hpc.sh             # Push all source code
#   ./scripts/push_to_hpc.sh --dry-run   # Preview what would be synced
#   ./scripts/push_to_hpc.sh --check     # Show files that differ (no transfer)
# =============================================================================

HPC_USER="vsc36278"
HPC_HOST="login.hpc.kuleuven.be"
HPC_BASE="/data/leuven/362/vsc36278/AML_work_study/AML_work_study"
LOCAL_BASE="$HOME/projects/AML_work_study/AML_work_study"

DRY_RUN=false
CHECK_ONLY=false

case "$1" in
    --dry-run) DRY_RUN=true  ; echo "DRY RUN MODE — no files will be transferred" ; echo "" ;;
    --check)   CHECK_ONLY=true ;;
esac

# =============================================================================
# Check: show which source files differ from HPC without transferring
# =============================================================================
if [[ "$CHECK_ONLY" == true ]]; then
    echo "Checking which files differ from HPC..."
    echo "======================================================================"
    rsync --dry-run -avz --checksum \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='.git/' \
        --exclude='results/' \
        --exclude='experiments/' \
        --exclude='logs/' \
        --exclude='profiling/' \
        --exclude='.vscode/' \
        --exclude='.idea/' \
        --exclude='*.egg-info/' \
        "${LOCAL_BASE}/" \
        "${HPC_USER}@${HPC_HOST}:${HPC_BASE}/"
    echo ""
    echo "Run without --check to push these changes."
    exit 0
fi

# =============================================================================
# Push source code to HPC
# =============================================================================
echo "Pushing source code to HPC..."
echo "  From: $LOCAL_BASE"
echo "  To:   ${HPC_USER}@${HPC_HOST}:${HPC_BASE}"
echo "======================================================================"

RSYNC_FLAGS="-avz --progress"
[[ "$DRY_RUN" == true ]] && RSYNC_FLAGS="$RSYNC_FLAGS --dry-run"

rsync $RSYNC_FLAGS \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    --exclude='results/' \
    --exclude='experiments/' \
    --exclude='logs/' \
    --exclude='profiling/' \
    --exclude='.vscode/' \
    --exclude='.idea/' \
    --exclude='*.egg-info/' \
    "${LOCAL_BASE}/" \
    "${HPC_USER}@${HPC_HOST}:${HPC_BASE}/"

echo ""
echo "======================================================================"
if [[ "$DRY_RUN" == true ]]; then
    echo "Dry run complete. Run without --dry-run to push."
else
    echo "Push complete."
fi
