#!/bin/bash
# =============================================================================
# Timing breakdown — SplitFed (system, lazy_link_neighbor) + batched-vs-sequential
# equivalence check on the trained system model.
#
# Times N train + N vali batches (benchmark_timing_splitfed.py): loader sampling,
# party subgraph build, GNN, head, other, backward, peak GPU memory, extrapolated
# epoch / full-run time. Then checks that the batched forward equals the
# sequential (per-party) forward on N vali batches and times both paths.
#
# Same cluster and flags as the training runs (wice gpu_h100, 182G, 16 CPUs).
# =============================================================================
# Usage:
#   bash scripts/hpc/benchmark/gnn/run_timing_splitfed.sh
# Results:
#   grep -A30 'Per train batch'      logs/aml_timing_SplitFed_<jobid>.log
#   grep -A10 'Batched vs sequential' logs/aml_timing_SplitFed_<jobid>.log
# =============================================================================

CLUSTER="wice"
ACCOUNT="lp_aml_work_study"
PARTITION="gpu_h100"
CPUS="16"
MEM="182G"
GPUS="1"
TIME="02:00:00"
N_BATCHES=30

PYTHON_SCRIPT="$VSC_DATA/AML_work_study/AML_work_study/scripts/hpc/benchmark/gnn/benchmark_timing_splitfed.py"
SF_MODEL="/data/leuven/362/vsc36278/AML_work_study/experiments/small_HI/split_0.6_0.2/system/FedGraphSimple/GINe__emlps/batching__ibm_hp/20260712_195549/seed_1/model.pth"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

FLAGS="--fl_algo SplitFed \
--model GINe --size small --ir HI \
--ibm_hp --emlps --batching --batching_mode lazy_link_neighbor \
--eval_mode system --max_workers $CPUS --num_rounds 20 \
--n_benchmark_epochs 1 --n_benchmark_batches $N_BATCHES --n_explicit_batches 0 \
--check_batched_equivalence $N_BATCHES --load_weights $SF_MODEL"

mkdir -p logs

JOB_NAME="aml_timing_SplitFed"
echo "Submitting: $JOB_NAME ($N_BATCHES train + $N_BATCHES vali batches, equivalence check)"
sbatch \
    -M "$CLUSTER" \
    --account="$ACCOUNT" \
    --job-name="$JOB_NAME" \
    --output="logs/${JOB_NAME}_%j.log" \
    --error="logs/${JOB_NAME}_%j.err" \
    --partition="$PARTITION" \
    --nodes=1 --ntasks=1 \
    --time="$TIME" --mem="$MEM" \
    --cpus-per-task="$CPUS" --gpus-per-node="$GPUS" \
    --mail-type=END,FAIL \
    --mail-user=bjoern.strandgaard@kuleuven.be \
    --wrap "$CONDA_SETUP
echo '======================================================================'
echo 'Timing breakdown SplitFed (system)'
echo '======================================================================'
python $PYTHON_SCRIPT $FLAGS
echo '======================================================================'
echo 'Finished'
echo '======================================================================'
"

echo ""
echo "Timing:      grep -A30 'Per train batch' logs/${JOB_NAME}_<jobid>.log"
echo "Equivalence: grep -A10 'Batched vs sequential' logs/${JOB_NAME}_<jobid>.log"
