#!/bin/bash
#SBATCH --job-name=aml_tune_booster_cpu
#SBATCH --account=lp_aml_work_study
#SBATCH --clusters=wice
#SBATCH --partition=batch_sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=08:00:00
#SBATCH --mem=160G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bjoern.strandgaard@kuleuven.be
#SBATCH --output=output_tune_booster_cpu_%j.txt
#SBATCH --error=error_tune_booster_cpu_%j.txt

# =============================================================================
# Booster Hyperparameter Tuning — CPU parallel (no GPU)
# =============================================================================
# Uses all allocated CPU cores to run HP configs in parallel via
# ThreadPoolExecutor. No GPU needed: XGBoost hist on CPU with many
# parallel single-threaded fits is faster than sequential GPU fits
# for this many-independent-configs workload.
#
# Partition options:
#   genius batch      : 32 cores, 160 GB  (Cascadelake, Xeon Gold 6240)
#   wice <cpu-part>   : 96 cores, 256 GB  (Sapphire Rapids, Xeon 8468)
#     -> change --clusters=wice --partition=<name> --cpus-per-task=96
#
# Usage:
#   sbatch scripts/tune_booster_cpu.sh [size] [ir] [model] [eval_mode] [ibm_fe]
#
# Examples:
#   sbatch scripts/tune_booster_cpu.sh small HI xgboost system           # system + standard FE
#   sbatch scripts/tune_booster_cpu.sh small HI xgboost system ibm_fe    # system + IBM FE
#   sbatch scripts/tune_booster_cpu.sh small HI xgboost comparable       # comparable + standard FE
# =============================================================================

SIZE=${1:-small}
IR=${2:-HI}
MODEL=${3:-xgboost}
EVAL_MODE=${4:-system}   # system or comparable
IBM_FE=${5:-}            # pass "ibm_fe" to enable IBM feature engineering
MAX_ROUNDS=${6:-150}     # upper bound for num_rounds in HP sampler

IBM_FE_FLAG=""
IBM_FE_SUFFIX=""
if [ "$IBM_FE" = "ibm_fe" ]; then
    IBM_FE_FLAG="--ibm_fe"
    IBM_FE_SUFFIX="_ibm"
fi

export PATH=$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc

echo "======================================================================"
echo "Booster Hyperparameter Tuning — CPU parallel"
echo "Job started : $(date)"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_NODELIST"
echo "CPUs        : $SLURM_CPUS_PER_TASK"
echo "======================================================================"
echo "  Dataset   : $SIZE / $IR"
echo "  Model     : $MODEL"
echo "  Eval mode : $EVAL_MODE"
echo "  IBM FE    : ${IBM_FE:-no}"
echo "  Max rounds: $MAX_ROUNDS"
echo "  Saves to  : configs/tuned_hyperparams/booster/$MODEL/${SIZE}_${IR}_${EVAL_MODE}${IBM_FE_SUFFIX}_r${MAX_ROUNDS}.json (best HP)"
echo "             configs/tuned_hyperparams/booster/$MODEL/${SIZE}_${IR}_${EVAL_MODE}${IBM_FE_SUFFIX}_r${MAX_ROUNDS}_top31.json (top-31 ranked)"
echo "======================================================================"

python $VSC_DATA/AML_work_study/AML_work_study/main.py \
    --fl_algo full_info \
    --model "$MODEL" \
    --size "$SIZE" \
    --ir "$IR" \
    --eval_mode "$EVAL_MODE" \
    --tune_max_rounds "$MAX_ROUNDS" \
    --tune \
    $IBM_FE_FLAG

EXIT_CODE=$?

echo "======================================================================"
echo "Finished at : $(date)"
echo "Exit code   : $EXIT_CODE"
echo "======================================================================"

exit $EXIT_CODE
