#!/bin/bash
# =============================================================================
# Timing breakdown — SplitFed vs RestrictedMerge (system, lazy_link_neighbor).
#
# One job per algorithm (each needs its own full system data in RAM). Each job
# times N train + N vali batches with benchmark_timing_comparison.py and prints a
# per-batch breakdown: loader sampling, party subgraph build, GNN, merge-graph
# build (RestrictedMerge), head, other, backward (+ checkpoint recompute), peak
# GPU memory, and the extrapolated epoch / full-run time.
#
# Same cluster and flags as the training runs (wice gpu_h100, 182G, 16 CPUs).
# =============================================================================
# Usage:
#   bash scripts/hpc/benchmark/gnn/run_timing_sf_vs_rm.sh
# Results: grep -A30 "Per train batch" logs/aml_timing_<algo>_<jobid>.log
# =============================================================================

CLUSTER="wice"
ACCOUNT="lp_aml_work_study"
PARTITION="gpu_h100"
CPUS="16"
MEM="182G"
GPUS="1"
TIME="02:00:00"
N_BATCHES=30

PYTHON_SCRIPT="$VSC_DATA/AML_work_study/AML_work_study/scripts/hpc/benchmark/gnn/benchmark_timing_comparison.py"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

# SplitFed job also runs the LayerNorm coupling check (batched vs per-party
# normalisation) on the trained system model, over N_BATCHES vali batches.
SF_MODEL="/data/leuven/362/vsc36278/AML_work_study/experiments/small_HI/split_0.6_0.2/system/FedGraphSimple/GINe__emlps/batching__ibm_hp/20260712_195549/seed_1/model.pth"

mkdir -p logs

for ALGO in SplitFed RestrictedMerge; do
    FLAGS="--fl_algo $ALGO \
--model GINe --size small --ir HI \
--ibm_hp --emlps --batching --batching_mode lazy_link_neighbor \
--eval_mode system --max_workers $CPUS --num_rounds 20 \
--n_benchmark_epochs 1 --n_benchmark_batches $N_BATCHES --n_explicit_batches 0"
    if [[ "$ALGO" == "SplitFed" ]]; then
        FLAGS="$FLAGS --check_norm_coupling $N_BATCHES --load_weights $SF_MODEL"
    fi

    JOB_NAME="aml_timing_${ALGO}"
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
echo 'Timing breakdown $ALGO (system)  Node: \$SLURM_NODELIST  Started: \$(date)'
echo '======================================================================'
python $PYTHON_SCRIPT $FLAGS
echo '======================================================================'
echo 'Finished: \$(date)'
echo '======================================================================'
"
done

echo ""
echo "Results: grep -A30 'Per train batch' logs/aml_timing_<algo>_<jobid>.log"
echo "Norm check: grep -A10 'LayerNorm coupling' logs/aml_timing_SplitFed_<jobid>.log"
