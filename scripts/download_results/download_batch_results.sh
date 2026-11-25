#!/bin/bash
# =============================================================================
# Download batch analysis outputs from the HPC.
#
# all   (default): syncs hpc_output/<EVAL_MODE>/ from the cluster, then copies the
#                  figures (*.pdf) to figs/party_visibility/<EVAL_MODE>/ and the tables
#                  (*.tex, *.csv) to tables/party_visibility/<EVAL_MODE>/ in the writing repo.
# cones          : syncs only hpc_output/<EVAL_MODE>/cone_candidates/ and copies nothing
#                  into the writing repo.
# Run this from your local machine after the job completes.
#
# Usage:
#   bash scripts/download_results/download_batch_results.sh [system|comparable] [all|cones] [user@host]
#
# Defaults: system, all, and the host vsc36278@login.hpc.kuleuven.be (same as the other
# download scripts; uses the login.hpc.kuleuven.be entry in ~/.ssh/config).
# =============================================================================

set -euo pipefail

EVAL_MODE="${1:-system}"
WHAT="${2:-all}"
HOST="${3:-vsc36278@login.hpc.kuleuven.be}"
if [[ "$EVAL_MODE" != "system" && "$EVAL_MODE" != "comparable" ]] || \
   [[ "$WHAT" != "all" && "$WHAT" != "cones" ]]; then
    echo "Usage: bash download_batch_results.sh [system|comparable] [all|cones] [user@host]  (got: '$EVAL_MODE' '$WHAT')" >&2
    exit 1
fi

# Full path ($VSC_DATA): rsync >= 3.2.4 no longer lets the remote shell expand $VSC_DATA.
REMOTE_BASE="/data/leuven/362/vsc36278/AML_work_study/AML_work_study/scripts/hpc/analysis_hpc"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_OUT="$SCRIPT_DIR/hpc_output"

WRITING="$HOME/projects/AML_work_study/writing/Experimental-Protocol"
FIGS_DEST="$WRITING/figs/party_visibility/$EVAL_MODE"
TABLES_DEST="$WRITING/tables/party_visibility/$EVAL_MODE"

# --- cone candidates only ---
if [[ "$WHAT" == "cones" ]]; then
    mkdir -p "$LOCAL_OUT/$EVAL_MODE"
    echo "Downloading from $HOST:$REMOTE_BASE/hpc_output/$EVAL_MODE/cone_candidates/"
    echo "          → $LOCAL_OUT/$EVAL_MODE/cone_candidates/"
    echo ""
    rsync -avz --progress \
        "${HOST}:${REMOTE_BASE}/hpc_output/$EVAL_MODE/cone_candidates" \
        "$LOCAL_OUT/$EVAL_MODE/"
    echo ""
    echo "Download complete. Files:"
    ls -lh "$LOCAL_OUT/$EVAL_MODE/cone_candidates/"
    exit 0
fi

mkdir -p "$LOCAL_OUT"

echo "Downloading from $HOST:$REMOTE_BASE/hpc_output/$EVAL_MODE/"
echo "          → $LOCAL_OUT/$EVAL_MODE/"
echo ""

# No trailing slash on the source, so the files land in $LOCAL_OUT/$EVAL_MODE/.
rsync -avz --progress \
    "${HOST}:${REMOTE_BASE}/hpc_output/$EVAL_MODE" \
    "$LOCAL_OUT/"

echo ""
echo "Download complete. Files:"
ls -lh "$LOCAL_OUT/$EVAL_MODE/"

# --- copy into the writing repo ---
mkdir -p "$FIGS_DEST" "$TABLES_DEST"
shopt -s nullglob
figs=("$LOCAL_OUT/$EVAL_MODE"/*.pdf)
tables=("$LOCAL_OUT/$EVAL_MODE"/*.tex "$LOCAL_OUT/$EVAL_MODE"/*.csv)

echo ""
if (( ${#figs[@]} )); then
    cp "${figs[@]}" "$FIGS_DEST/"
    echo "Copied ${#figs[@]} figure(s)  → $FIGS_DEST/"
else
    echo "No figures found (viz job not run or not finished?)"
fi
if (( ${#tables[@]} )); then
    cp "${tables[@]}" "$TABLES_DEST/"
    echo "Copied ${#tables[@]} table file(s) → $TABLES_DEST/"
else
    echo "No tables found (tables job not run or not finished?)"
fi
