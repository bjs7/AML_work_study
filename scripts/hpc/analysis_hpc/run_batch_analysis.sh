#!/bin/bash
# =============================================================================
# Submit batch utilization analysis jobs.
#
# Two independent jobs:
#   1. batch_tables_hpc.py  — iterates ALL batches, samples ALL seed edges,
#      writes CSV + LaTeX tables.  CPU-heavy; typically 3–8 h.
#   2. batch_viz_hpc.py     — loads a small set of BATCH_INDICES batches and
#      produces full-batch NetworkX visualizations.  GPU-heavy for setup;
#      CPU-heavy for layout.  Typically 1–3 h.
#
# Arguments: [system|comparable] [all|tables|viz|cones]  (defaults: system, all)
#   bash scripts/hpc/analysis_hpc/run_batch_analysis.sh system           # both jobs
#   bash scripts/hpc/analysis_hpc/run_batch_analysis.sh system viz       # figures only
#   bash scripts/hpc/analysis_hpc/run_batch_analysis.sh system cones     # cone candidates only
# 'cones' draws only the Cell 3 cone figure for several pairs of batches
# (CONE_CANDIDATE_PAIRS in batch_viz_hpc.py), to choose batches for _VIZ_PICKS.
# System mode needs its own BATCH_INDICES / SEED_OVERRIDE in batch_viz_hpc.py
# (_VIZ_PICKS), since batches differ between modes.
#
# Outputs (all in scripts/hpc/analysis_hpc/hpc_output/<EVAL_MODE>/):
#   Tables job:
#     batch_stats_batch.csv / batch_stats_party.csv   — raw per-batch stats
#     cone_coverage_combined_{a,b}.{csv,tex}          — Table 1a / 1b
#     party_batch_coverage_summary.{csv,tex}          — Table 2
#   Viz job:
#     batch_viz_cell{0a,0b,1,1b,2,3,3b,4}_hpc.pdf    — visualizations
#   Cones job:
#     cone_candidates/batch_viz_cell3_mp_cone_hpc_b<i>_b<j>.pdf — one per batch pair
#
# Usage:
#   cd $VSC_DATA/AML_work_study/AML_work_study
#   bash scripts/hpc/analysis_hpc/run_batch_analysis.sh [system|comparable] [all|tables|viz|cones]
# =============================================================================

EVAL_MODE="${1:-system}"
JOBS="${2:-all}"
if [[ "$EVAL_MODE" != "system" && "$EVAL_MODE" != "comparable" ]] || \
   [[ ! "$JOBS" =~ ^(all|tables|viz|cones)$ ]]; then
    echo "Usage: bash run_batch_analysis.sh [system|comparable] [all|tables|viz|cones]  (got: '$EVAL_MODE' '$JOBS')" >&2
    exit 1
fi
CLUSTER="wice"
ACCOUNT="lp_aml_work_study"
if [[ "$EVAL_MODE" == "system" ]]; then
    # System sets up all ~30.5k banks; at 64G it ran out after ~20k, which scales
    # to ~100G for the full setup. 160G leaves room for what runs after setup.
    # Memory on batch_icelake is tied to cores (3400 MiB each): 50 x 3400 MiB ~ 166 GiB.
    # Max on this partition is a full node (72 cores, ~230G); beyond that use bigmem.
    PARTITION="batch_icelake"
    CPUS_TABLES="50"
    CPUS_VIZ="50"
    MEM="160G"
else
    PARTITION="batch_icelake"
    CPUS_TABLES="32"   # tables iterates all batches; more threads helps loader
    CPUS_VIZ="16"     # viz loads a handful of batches; 16 is enough
    MEM="64G"
fi
TIME_TABLES="24:00:00"
TIME_VIZ="06:00:00"

REPO="$VSC_DATA/AML_work_study/AML_work_study"
SCRIPT_DIR="$REPO/scripts/hpc/analysis_hpc"
LOGS="$SCRIPT_DIR/logs"

mkdir -p "$LOGS"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc
export EVAL_MODE=$EVAL_MODE"


# =============================================================================
# JOB 1: Tables (all batches, all cone samples) — JOBS=all|tables
# =============================================================================

if [[ "$JOBS" == "all" || "$JOBS" == "tables" ]]; then
JOB_TABLES="aml_batch_tables_${EVAL_MODE}"
SCRIPT_TABLES="$SCRIPT_DIR/batch_tables_hpc.py"

echo "Eval mode:  $EVAL_MODE"
echo "Submitting tables job: $JOB_TABLES"
echo "Script:     $SCRIPT_TABLES"
echo "Resources:  $CPUS_TABLES CPUs · ${MEM} RAM · walltime $TIME_TABLES · partition $PARTITION"
echo ""

sbatch \
    -M "$CLUSTER" \
    --account="$ACCOUNT" \
    --job-name="$JOB_TABLES" \
    --output="$LOGS/${JOB_TABLES}_%j.log" \
    --error="$LOGS/${JOB_TABLES}_%j.err" \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --time="$TIME_TABLES" \
    --mem="$MEM" \
    --cpus-per-task="$CPUS_TABLES" \
    --mail-type=END,FAIL \
    --mail-user=bjoern.strandgaard@kuleuven.be \
    --wrap "$CONDA_SETUP
echo '======================================================================'
echo 'Job started at: \$(date)'
echo 'Job ID: \$SLURM_JOB_ID  Node: \$SLURM_NODELIST'
echo 'Tables — all batches, all cone samples'
echo '======================================================================'
cd $REPO
python $SCRIPT_TABLES
EXIT_CODE=\$?
echo '======================================================================'
echo 'Job finished at: \$(date)  exit code: '\$EXIT_CODE
echo '======================================================================'
exit \$EXIT_CODE
"
fi


# =============================================================================
# JOB 2: Visualizations — JOBS=all|viz (all cells, _VIZ_PICKS batches)
#                         or JOBS=cones (Cell 3 only, CONE_CANDIDATE_PAIRS)
# Edit _VIZ_PICKS / SEED_PICK / CONE_CANDIDATE_PAIRS in batch_viz_hpc.py before submitting.
# =============================================================================

if [[ "$JOBS" != "tables" ]]; then
if [[ "$JOBS" == "cones" ]]; then
    VIZ_TARGET="cone_candidates"
    JOB_VIZ="aml_batch_cones_${EVAL_MODE}"
else
    VIZ_TARGET="all"
    JOB_VIZ="aml_batch_viz_${EVAL_MODE}"
fi
SCRIPT_VIZ="$SCRIPT_DIR/batch_viz_hpc.py"

echo "Eval mode:  $EVAL_MODE"
echo "Submitting viz job: $JOB_VIZ"
echo "Script:     $SCRIPT_VIZ"
echo "Resources:  $CPUS_VIZ CPUs · ${MEM} RAM · walltime $TIME_VIZ · partition $PARTITION"
echo ""

sbatch \
    -M "$CLUSTER" \
    --account="$ACCOUNT" \
    --job-name="$JOB_VIZ" \
    --output="$LOGS/${JOB_VIZ}_%j.log" \
    --error="$LOGS/${JOB_VIZ}_%j.err" \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --time="$TIME_VIZ" \
    --mem="$MEM" \
    --cpus-per-task="$CPUS_VIZ" \
    --mail-type=END,FAIL \
    --mail-user=bjoern.strandgaard@kuleuven.be \
    --wrap "$CONDA_SETUP
export VIZ_TARGET=$VIZ_TARGET
echo '======================================================================'
echo 'Job started at: \$(date)'
echo 'Job ID: \$SLURM_JOB_ID  Node: \$SLURM_NODELIST'
echo 'Viz — VIZ_TARGET=$VIZ_TARGET'
echo '======================================================================'
cd $REPO
python $SCRIPT_VIZ
EXIT_CODE=\$?
echo '======================================================================'
echo 'Job finished at: \$(date)  exit code: '\$EXIT_CODE
echo '======================================================================'
exit \$EXIT_CODE
"
fi


echo ""
echo "Submitted ($JOBS, $EVAL_MODE, partition $PARTITION). Monitor with:"
echo "  squeue -u \$USER -M $CLUSTER"
echo "  tail -f $LOGS/<job-name>_<JOBID>.log"
