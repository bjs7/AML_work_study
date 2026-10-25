#!/bin/bash
# =============================================================================
# Timing breakdown — RestrictedMerge (system, lazy_link_neighbor) + merge-graph
# equivalence check (vectorised merge vs the original loop-based reference).
#
# Times N train + N vali batches (benchmark_timing_restricted_merge.py): loader
# sampling, party subgraph build, GNN, merge-graph build (CPU), head, other,
# backward (+ checkpoint recompute), peak GPU memory, extrapolated epoch /
# full-run time. Then rebuilds every pair's merged graph in N_EQUIV_BATCHES vali
# batches both ways, requires exact equality, and times both.
#
# Same cluster and flags as the training runs (wice gpu_h100, 182G, 16 CPUs).
# =============================================================================
# Usage:
#   bash scripts/hpc/benchmark/gnn/run_timing_restricted_merge.sh
# Results:
#   grep -A30 'Per train batch'         logs/aml_timing_RestrictedMerge_<jobid>.log
#   grep -A8  'Merge-graph equivalence' logs/aml_timing_RestrictedMerge_<jobid>.log
# =============================================================================

CLUSTER="wice"
ACCOUNT="lp_aml_work_study"
PARTITION="gpu_h100"
CPUS="16"
MEM="182G"
GPUS="1"
TIME="02:00:00"
N_BATCHES=30
N_EQUIV_BATCHES=10   # vectorised vs reference merge, exact equality

PYTHON_SCRIPT="$VSC_DATA/AML_work_study/AML_work_study/scripts/hpc/benchmark/gnn/benchmark_timing_restricted_merge.py"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

FLAGS="--fl_algo RestrictedMerge \
--model GINe --size small --ir HI \
--ibm_hp --emlps --batching --batching_mode lazy_link_neighbor \
--eval_mode system --max_workers $CPUS --num_rounds 20 \
--n_benchmark_epochs 1 --n_benchmark_batches $N_BATCHES --n_explicit_batches 0 \
--check_merge_equivalence $N_EQUIV_BATCHES"

mkdir -p logs

JOB_NAME="aml_timing_RestrictedMerge"
echo "Submitting: $JOB_NAME ($N_BATCHES train + $N_BATCHES vali batches)"
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
echo 'Timing breakdown RestrictedMerge (system)'
echo '======================================================================'
python $PYTHON_SCRIPT $FLAGS
echo '======================================================================'
echo 'Finished'
echo '======================================================================'
"

echo ""
echo "Timing:      grep -A30 'Per train batch' logs/${JOB_NAME}_<jobid>.log"
echo "Equivalence: grep -A8 'Merge-graph equivalence' logs/${JOB_NAME}_<jobid>.log"
