#!/bin/bash
# =============================================================================
# Tabular baseline (XGBoost) for the paper: pooled and bank-level, with and
# without the snapML graph features.
#
#   ID  Scenario                      Features                    HP file
#   B5  pooled (full_info)            raw transaction features    $HP_RAW  (tuned here)
#   B2  pooled (full_info)            raw + graph features        $HP_GFP  (shared, already on HPC)
#   B6  bank-level (individual)       raw transaction features    $HP_RAW
#   B3  bank-level (individual)       raw + graph features        $HP_GFP
#
# Stages:
#   tune   tunes the raw-features model the same way the shared graph-features HPs
#          were tuned (full_info, comparable, max 150 rounds) → $HP_RAW
#   train  submits the four training jobs above (4 seeds each)
#   all    tune, then train once tuning has finished OK (slurm dependency)
#
# Usage (from anywhere on the HPC):
#   bash scripts/hpc/training/booster/submit_tabular_baseline.sh [all|tune|train] [system|comparable]
# Defaults: all, system.
#
# Results land in experiments/small_HI/split_0.6_0.2/<eval_mode>/{full_info,individual}/xgboost/
#   no_gfp/   (raw features)   and   default/   (raw + graph features)
# =============================================================================

STAGE="${1:-all}"
EVAL_MODE="${2:-system}"
if [[ ! "$STAGE" =~ ^(all|tune|train)$ ]] || \
   [[ "$EVAL_MODE" != "system" && "$EVAL_MODE" != "comparable" ]]; then
    echo "Usage: bash submit_tabular_baseline.sh [all|tune|train] [system|comparable]  (got: '$STAGE' '$EVAL_MODE')" >&2
    exit 1
fi

CLUSTER="wice"
ACCOUNT="lp_aml_work_study"
REPO="$VSC_DATA/AML_work_study/AML_work_study"
LOGS="$REPO/scripts/hpc/training/booster/logs"
mkdir -p "$LOGS"

# HP paths are relative to $REPO (see --hp_path in utils.py).
HP_GFP="configs/tuned_hyperparams/booster/xgboost/small_HI_comparable_r150.json"
HP_RAW="configs/tuned_hyperparams/booster/xgboost/small_HI_comparable_nogfp_r150.json"

if [[ ! -f "$REPO/$HP_GFP" ]]; then
    echo "Warning: shared graph-features HP file not found: $REPO/$HP_GFP" >&2
fi

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc
cd $REPO"

# submit <job-name> <partition> <cpus> <time> <dependency or ''> <python args...>
# Prints the job id.
submit() {
    local name="$1" partition="$2" cpus="$3" time="$4" dep="$5"; shift 5
    local dep_flag=()
    [[ -n "$dep" ]] && dep_flag=(--dependency="afterok:$dep")
    sbatch --parsable \
        -M "$CLUSTER" \
        --account="$ACCOUNT" \
        --job-name="$name" \
        --output="$LOGS/${name}_%j.log" \
        --error="$LOGS/${name}_%j.err" \
        --partition="$partition" \
        --nodes=1 --ntasks=1 \
        --cpus-per-task="$cpus" \
        --mem=160G \
        --time="$time" \
        "${dep_flag[@]}" \
        --mail-type=END,FAIL \
        --mail-user=bjoern.strandgaard@kuleuven.be \
        --wrap "$CONDA_SETUP
echo 'Job started: '\$(date)'  ID: '\$SLURM_JOB_ID'  Node: '\$SLURM_NODELIST
echo 'python main.py $*'
python main.py $*
EXIT_CODE=\$?
echo 'Job finished: '\$(date)'  exit code: '\$EXIT_CODE
exit \$EXIT_CODE
" | cut -d';' -f1
}

TUNE_ID=""
if [[ "$STAGE" == "all" || "$STAGE" == "tune" ]]; then
    if [[ -f "$REPO/$HP_RAW" ]]; then
        echo "Note: $HP_RAW already exists and will be overwritten by the tuning job."
    fi
    TUNE_ID=$(submit aml_tab_tune_nogfp batch_sapphirerapids 96 08:00:00 "" \
        --fl_algo full_info --model xgboost --size small --ir HI \
        --eval_mode comparable --tune_max_rounds 150 --tune --no_gfp)
    echo "Tuning (raw features)          job $TUNE_ID  → $HP_RAW"
fi

if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
    if [[ "$STAGE" == "train" && ! -f "$REPO/$HP_RAW" ]]; then
        echo "Error: $REPO/$HP_RAW not found. Run the tune stage first." >&2
        exit 1
    fi
    COMMON="--model xgboost --size small --ir HI --eval_mode $EVAL_MODE"
    m="${EVAL_MODE:0:3}"
    id=$(submit "aml_tab_B5_pooled_raw_$m" batch_icelake 32 08:00:00 "$TUNE_ID" \
        --fl_algo full_info  $COMMON --no_gfp --hp_path "$HP_RAW")
    echo "B5 pooled, raw features        job $id"
    id=$(submit "aml_tab_B2_pooled_gfp_$m" batch_icelake 32 08:00:00 "" \
        --fl_algo full_info  $COMMON          --hp_path "$HP_GFP")
    echo "B2 pooled, + graph features    job $id"
    id=$(submit "aml_tab_B6_bank_raw_$m"   batch_icelake 32 08:00:00 "$TUNE_ID" \
        --fl_algo individual $COMMON --no_gfp --hp_path "$HP_RAW")
    echo "B6 bank-level, raw features    job $id"
    id=$(submit "aml_tab_B3_bank_gfp_$m"   batch_icelake 32 08:00:00 "" \
        --fl_algo individual $COMMON          --hp_path "$HP_GFP")
    echo "B3 bank-level, + graph features job $id"
fi

echo ""
echo "Submitted ($STAGE, $EVAL_MODE). Monitor with:"
echo "  squeue -u \$USER -M $CLUSTER"
echo "  tail -f $LOGS/<job-name>_<JOBID>.log"
