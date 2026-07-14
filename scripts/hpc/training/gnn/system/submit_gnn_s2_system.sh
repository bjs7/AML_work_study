#!/bin/bash
# =============================================================================
# Submit S2 GNN training — system evaluation, R1 tuned HPs.
# Full-info GINe with batching + edge updates, using HPs tuned by the parallel
# GNN tuning pipeline (not IBM HPs). Runs 4 seeds.
#
# Prerequisite: GNN tuning pipeline must have completed for system eval:
#   bash scripts/hpc/gnn_tuning/submit_gnn_tuning.sh small HI system
#   → produces configs/tuned_hyperparams/gnn/GINe/small_HI_system.json
#
# Usage:
#   bash submit_gnn_s2_system.sh
# =============================================================================

SCRIPT="train_gnn_full_info.sh"
LOGS="logs"

mkdir -p "$LOGS"

echo "Submitting S2 GNN training — system evaluation (R1 tuned HPs)"
echo ""

sbatch --job-name=aml_s2_gnn_system \
       --output=$LOGS/s2_gnn_system_%j.log \
       --error=$LOGS/s2_gnn_system_%j.err \
       $SCRIPT small HI system

echo ""
echo "Done."
