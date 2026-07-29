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
# To run only one job, comment out the other sbatch block below.
#
# Outputs (all in scripts/hpc/analysis_hpc/hpc_output/):
#   Tables job:
#     batch_stats_batch.csv / batch_stats_party.csv   — raw per-batch stats
#     cone_coverage_combined_{a,b}.{csv,tex}          — Table 1a / 1b
#     party_batch_coverage_summary.{csv,tex}          — Table 2
#   Viz job:
#     batch_viz_cell{0a,0b,1,1b,2,3,3b,4}_hpc.pdf    — visualizations
#
# Usage:
#   cd $VSC_DATA/AML_work_study/AML_work_study
#   bash scripts/hpc/analysis_hpc/run_batch_analysis.sh
# =============================================================================

CLUSTER="wice"
ACCOUNT="lp_aml_work_study"
PARTITION="batch_icelake"
CPUS_TABLES="32"   # tables iterates all batches; more threads helps loader
CPUS_VIZ="16"     # viz loads a handful of batches; 16 is enough
MEM="64G"
TIME_TABLES="24:00:00"
TIME_VIZ="06:00:00"

REPO="$VSC_DATA/AML_work_study/AML_work_study"
SCRIPT_DIR="$REPO/scripts/hpc/analysis_hpc"
LOGS="$SCRIPT_DIR/logs"

mkdir -p "$LOGS"

CONDA_SETUP="export PATH=\$PATH:/data/leuven/362/vsc36278/miniconda3/bin
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh
conda activate multignn_hpc"


# =============================================================================
# JOB 1: Tables (all batches, all cone samples)
# Comment out this block to skip the tables job.
# =============================================================================

JOB_TABLES="aml_batch_tables"
SCRIPT_TABLES="$SCRIPT_DIR/batch_tables_hpc.py"

echo "Submitting tables job: $JOB_TABLES"
echo "Script:     $SCRIPT_TABLES"
echo "Resources:  $CPUS_TABLES CPUs · ${MEM} RAM · walltime $TIME_TABLES (CPU only)"
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


# =============================================================================
# JOB 2: Visualizations (selected BATCH_INDICES only)
# Comment out this block to skip the viz job.
# Edit BATCH_INDICES / SEED_PICK / SEED_OVERRIDE in batch_viz_hpc.py before
# submitting, or set them from the batch scanner in batch_viz_explore_nb.py.
# =============================================================================

JOB_VIZ="aml_batch_viz"
SCRIPT_VIZ="$SCRIPT_DIR/batch_viz_hpc.py"

echo "Submitting viz job: $JOB_VIZ"
echo "Script:     $SCRIPT_VIZ"
echo "Resources:  $CPUS_VIZ CPUs · ${MEM} RAM · walltime $TIME_VIZ (CPU only)"
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
echo '======================================================================'
echo 'Job started at: \$(date)'
echo 'Job ID: \$SLURM_JOB_ID  Node: \$SLURM_NODELIST'
echo 'Viz — BATCH_INDICES from batch_viz_hpc.py'
echo '======================================================================'
cd $REPO
python $SCRIPT_VIZ
EXIT_CODE=\$?
echo '======================================================================'
echo 'Job finished at: \$(date)  exit code: '\$EXIT_CODE
echo '======================================================================'
exit \$EXIT_CODE
"


echo ""
echo "Both jobs submitted (CPU-only, batch_icelake). Monitor with:"
echo "  squeue -u \$USER -M $CLUSTER"
echo "  tail -f $LOGS/${JOB_TABLES}_<JOBID>.log"
echo "  tail -f $LOGS/${JOB_VIZ}_<JOBID>.log"
