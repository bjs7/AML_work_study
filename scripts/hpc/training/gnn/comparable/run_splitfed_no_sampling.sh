#!/bin/bash
# =============================================================================
# SplitFed with no sampling — seed 1 benchmark.
#
# Each party applies the GNN to its full local subgraph (all transactions for
# that mode), rather than a sampled neighbourhood. Uses _collect_embeddings_
# sequential (use_batched=False) to process one party at a time and avoid
# assembling all 630 full graphs into a single GPU batch.
#
# Expected to be slower per epoch than lazy_link_neighbor (~28 min/epoch)
# but may produce better embeddings. 72h walltime as conservative headroom.
# =============================================================================
# Usage:
#   bash scripts/hpc/training/gnn/comparable/run_splitfed_no_sampling.sh
# =============================================================================

CLUSTER="wice"
ACCOUNT="lp_aml_work_study"
PARTITION="gpu_a100"
CPUS="16"
MEM="88G"
GPUS="1"
TIME="72:00:00"

PYTHON_CMD="python $VSC_DATA/AML_work_study/AML_work_study/main.py"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc"

BASE_FLAGS="--fl_algo SplitFed --model GINe --size small --ir HI \
--ibm_hp --emlps --eval_mode comparable \
--max_workers $CPUS --testing_seeds 1 --num_rounds 50"

RUN_ID=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

JOB_NAME="aml_splitfed_no_samp_s1"
echo "Submitting: $JOB_NAME (seed 1, run_id=$RUN_ID)"
sbatch \
    -M "$CLUSTER" \
    --account="$ACCOUNT" \
    --job-name="$JOB_NAME" \
    --output="logs/${JOB_NAME}_%j.log" \
    --error="logs/${JOB_NAME}_%j.err" \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --time="$TIME" \
    --mem="$MEM" \
    --cpus-per-task="$CPUS" \
    --gpus-per-node="$GPUS" \
    --mail-type=END,FAIL \
    --mail-user=bjoern.strandgaard@kuleuven.be \
    --wrap "$CONDA_SETUP
echo '======================================================================'
echo 'Job started at: \$(date)'
echo 'Job ID: \$SLURM_JOB_ID  Node: \$SLURM_NODELIST'
echo 'SplitFed no-sampling (full local subgraph) — seed 1 (run_id=$RUN_ID)'
echo '======================================================================'
$PYTHON_CMD $BASE_FLAGS --first_seed 1 --run_id $RUN_ID
echo '======================================================================'
echo 'Job finished at: \$(date)'
echo '======================================================================'
"

echo ""
echo "Submitted $JOB_NAME (run_id=$RUN_ID)."
echo "Check logs/${JOB_NAME}_<jobid>.log for epoch timings and final F1."
echo "Compare F1 and epoch time against lazy_link_neighbor baseline."
