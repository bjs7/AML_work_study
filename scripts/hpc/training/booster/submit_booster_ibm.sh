#!/bin/bash
# =============================================================================
# Submit full_info XGBoost training with IBM feature engineering.
# Uses HPs tuned with r1000 (max_rounds=1000) and IBM FE — different from the
# shared r150 HPs used by submit_booster_training.sh.
#
# Usage:
#   bash submit_booster_ibm.sh
# =============================================================================

# HPs tuned specifically for IBM FE with 1000-round budget (not the shared r150 HPs)
HP="configs/tuned_hyperparams/booster/xgboost/small_HI_system_ibm_r1000.json"
SCRIPT="train_booster_cpu.sh"
LOGS="logs"

mkdir -p "$LOGS"

echo "Submitting full_info IBM FE booster training with HP: $HP"
echo ""

sbatch --job-name=aml_fi_sys_ibm_r1000 \
       --output=$LOGS/aml_fi_sys_ibm_r1000_%j.log \
       --error=$LOGS/aml_fi_sys_ibm_r1000_%j.err \
       $SCRIPT full_info small HI xgboost system ibm_fe $HP

echo ""
echo "Done."
