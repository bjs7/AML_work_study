#!/bin/bash
# =============================================================================
# Download batch analysis outputs from the HPC.
#
# Syncs the hpc_output/ folder from the cluster to the local machine.
# Run this from your local machine after the job completes.
#
# Usage:
#   bash scripts/hpc/analysis_hpc/download_results.sh [vsc_login_alias]
#
# The optional argument overrides the default SSH host alias.
# Default: vsc  (set up in ~/.ssh/config, e.g. Host vsc → login-genius.hpc.kuleuven.be)
#
# Example ~/.ssh/config entry:
#   Host vsc
#       HostName login-genius.hpc.kuleuven.be
#       User vsc36278
#       IdentityFile ~/.ssh/id_rsa_vsc
# =============================================================================

set -euo pipefail

HOST="${1:-vsc}"
VSC_USER="vsc36278"
REMOTE_BASE="\$VSC_DATA/AML_work_study/AML_work_study/scripts/hpc/analysis_hpc"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_OUT="$SCRIPT_DIR/hpc_output"

mkdir -p "$LOCAL_OUT"

echo "Downloading from $HOST:$REMOTE_BASE/hpc_output/"
echo "          → $LOCAL_OUT/"
echo ""

rsync -avz --progress \
    "${HOST}:${REMOTE_BASE}/hpc_output/" \
    "$LOCAL_OUT/"

echo ""
echo "Download complete. Files:"
ls -lh "$LOCAL_OUT/"
