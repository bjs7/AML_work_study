#!/bin/bash
# =============================================================================
# Add seeds to the system-evaluation GNN baselines in tables/baselines/
# baselines_grouped_system (S2, S3, S4, V1). One job per (scenario, seed), each
# writing into the scenario's existing run folder (--run_id), with the same flags
# as the original runs. The job that finishes last per scenario rewrites
# aggregated_results.json over all seeds in that folder; the run's original
# config files are kept.
#
#   S2  full_info GNN (R1, LayerNorm)   full_info/GINe__emlps/batching__ibm_hp/20260306_171841
#   S3  individual GNN, batching         individual/GINe__emlps/batching__ibm_hp/20260306_204247
#   S4  individual GNN, full batch       individual/GINe__emlps/ibm_hp/20260306_185326
#   V1  SplitFed                         FedGraphSimple/GINe__emlps/batching__ibm_hp/20260712_195549
#       (run started under the old name FedGraphSimple; save_results adds new
#        SplitFed seeds to that folder)
#
# Original flags: S2-S4 from multiple_jobs_system.sh, V1 from run_vertical_simple_system.sh.
# Time limits: S2-S4 use the original jobs' limits, which covered 4 seeds each.
# =============================================================================
# Usage:
#   bash scripts/hpc/training/gnn/system/run_extra_seeds_system.sh                  # seeds 5-8, all four
#   bash scripts/hpc/training/gnn/system/run_extra_seeds_system.sh 5 8 S2 V1        # chosen seeds/scenarios
# =============================================================================

FIRST_SEED="${1:-5}"
LAST_SEED="${2:-8}"
shift 2 2>/dev/null
if [[ $# -gt 0 ]]; then SCENARIOS=("$@"); else SCENARIOS=(S2 S3 S4 V1); fi

CLUSTER="wice"
ACCOUNT="lp_aml_work_study"
COMMON="--model GINe --size small --ir HI --testing_seeds 1"

declare -A ALGO RUN_ID FLAGS PARTITION CPUS MEM TIME
ALGO[S2]=full_info;   RUN_ID[S2]=20260306_171841
FLAGS[S2]="--ibm_hp --emlps --max_workers 1 --eval_mode system --batching"
PARTITION[S2]=gpu_a100; CPUS[S2]=18; MEM[S2]=122G; TIME[S2]=24:00:00

ALGO[S3]=individual;  RUN_ID[S3]=20260306_204247
FLAGS[S3]="--ibm_hp --emlps --max_workers 1 --eval_mode system --batching"
PARTITION[S3]=gpu_a100; CPUS[S3]=18; MEM[S3]=122G; TIME[S3]=20:00:00

ALGO[S4]=individual;  RUN_ID[S4]=20260306_185326
FLAGS[S4]="--ibm_hp --emlps --max_workers 1 --eval_mode system"
PARTITION[S4]=gpu_a100; CPUS[S4]=18; MEM[S4]=122G; TIME[S4]=20:00:00

ALGO[V1]=SplitFed;    RUN_ID[V1]=20260712_195549
FLAGS[V1]="--batching --ibm_hp --emlps --eval_mode system --max_workers 16 --batching_mode lazy_link_neighbor --num_rounds 20"
PARTITION[V1]=gpu_h100; CPUS[V1]=16; MEM[V1]=182G; TIME[V1]=12:00:00   # ~5.3 h/seed after the speed-ups

PYTHON_CMD="python $VSC_DATA/AML_work_study/AML_work_study/main.py"
CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

mkdir -p logs

for S in "${SCENARIOS[@]}"; do
    if [[ -z "${ALGO[$S]}" ]]; then
        echo "Unknown scenario '$S' (choose from S2 S3 S4 V1)"; exit 1
    fi
    for SEED in $(seq "$FIRST_SEED" "$LAST_SEED"); do
        JOB_NAME="aml_${S}_sys_s${SEED}"
        echo "Submitting: $JOB_NAME (${ALGO[$S]}, seed $SEED, run_id=${RUN_ID[$S]})"
        sbatch \
            -M "$CLUSTER" \
            --account="$ACCOUNT" \
            --job-name="$JOB_NAME" \
            --output="logs/${JOB_NAME}_%j.log" \
            --error="logs/${JOB_NAME}_%j.err" \
            --partition="${PARTITION[$S]}" \
            --nodes=1 --ntasks=1 \
            --time="${TIME[$S]}" --mem="${MEM[$S]}" \
            --cpus-per-task="${CPUS[$S]}" --gpus-per-node=1 \
            --mail-type=END,FAIL \
            --mail-user=bjoern.strandgaard@kuleuven.be \
            --wrap "$CONDA_SETUP
echo '======================================================================'
echo '$S (${ALGO[$S]}) system — seed $SEED (run_id=${RUN_ID[$S]})'
echo '======================================================================'
$PYTHON_CMD --fl_algo ${ALGO[$S]} $COMMON ${FLAGS[$S]} --first_seed $SEED --run_id ${RUN_ID[$S]}
echo '======================================================================'
echo 'Finished'
echo '======================================================================'
"
        sleep 1
    done
done

echo ""
echo "Submitted seeds $FIRST_SEED-$LAST_SEED for: ${SCENARIOS[*]}"
