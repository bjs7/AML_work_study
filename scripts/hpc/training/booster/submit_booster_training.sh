#!/bin/bash
# =============================================================================
# Submit all 6 booster training jobs (full_info + individual, system + comparable,
# standard FE + IBM FE) using the agreed shared HP config.
#
# Usage:
#   bash scripts/submit_booster_training.sh
# =============================================================================

HP="configs/tuned_hyperparams/booster/xgboost/small_HI_comparable_r150.json"
SCRIPT="train_booster_cpu.sh"
LOGS="logs"

mkdir -p "$LOGS"

echo "Submitting 6 booster training jobs with HP: $HP"
echo ""

# System eval
sbatch --job-name=aml_fi_sys     --output=$LOGS/aml_fi_sys_%j.log     --error=$LOGS/aml_fi_sys_%j.err     $SCRIPT full_info  small HI xgboost system ""      $HP
sbatch --job-name=aml_fi_sys_ibm --output=$LOGS/aml_fi_sys_ibm_%j.log --error=$LOGS/aml_fi_sys_ibm_%j.err $SCRIPT full_info  small HI xgboost system ibm_fe  $HP
sbatch --job-name=aml_ind_sys    --output=$LOGS/aml_ind_sys_%j.log    --error=$LOGS/aml_ind_sys_%j.err    $SCRIPT individual small HI xgboost system ""      $HP

# Comparable eval
sbatch --job-name=aml_fi_cmp     --output=$LOGS/aml_fi_cmp_%j.log     --error=$LOGS/aml_fi_cmp_%j.err     $SCRIPT full_info  small HI xgboost comparable ""      $HP
sbatch --job-name=aml_fi_cmp_ibm --output=$LOGS/aml_fi_cmp_ibm_%j.log --error=$LOGS/aml_fi_cmp_ibm_%j.err $SCRIPT full_info  small HI xgboost comparable ibm_fe  $HP
sbatch --job-name=aml_ind_cmp    --output=$LOGS/aml_ind_cmp_%j.log    --error=$LOGS/aml_ind_cmp_%j.err    $SCRIPT individual small HI xgboost comparable ""      $HP

echo ""
echo "All 6 jobs submitted."