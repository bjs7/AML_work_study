#!/bin/bash
#SBATCH --job-name=aml_experiment
#SBATCH --account=lp_aml_work_study
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=36:00:00
#SBATCH --mem=42G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bjoern.strandgaard@kuleuven.be
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt

# =============================================================================
# HPC Job Template for AML Experiments
# =============================================================================
# Usage:
#   sbatch job_template.sh full_info GINe small HI --batching --ibm_fe --ibm_hp
#   sbatch job_template.sh FedAvg GINe small HI --batching --client_fraction 0.25 --num_local_epochs 10
#   sbatch job_template.sh FedProx GINe small HI --batching --mu 0.01
#
# Arguments (all optional):
#   $1: fl_algo    (default: FedAvg)
#   $2: model      (default: GINe)
#   $3: size       (default: small)
#   $4: ir         (default: HI)
#   $5+: extra flags (e.g., --testing, --batching, --ibm_fe, --ibm_hp,
#                     --client_fraction, --num_local_epochs, --mu, --weighting)
# =============================================================================

# Parse arguments with defaults
FL_ALGO=${1:-Individual}
MODEL=${2:-GINe}
SIZE=${3:-small}
IR=${4:-HI}
shift 4 2>/dev/null || true  # Remove first 4 args, keep rest as extra flags
EXTRA_FLAGS="$@"

export PATH=$PATH:/data/leuven/362/vsc36278/miniconda3/bin

# Activate virtual environment (CHANGE THIS PATH)
source /data/leuven/362/vsc36278/miniconda3/etc/profile.d/conda.sh

conda activate multignn_hpc

# Print job info
echo "======================================================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "======================================================================"
echo "Experiment Configuration:"
echo "  FL Algorithm: $FL_ALGO"
echo "  Model:        $MODEL"
echo "  Size:         $SIZE"
echo "  IR:           $IR"
echo "  Extra Flags:  $EXTRA_FLAGS"
echo "======================================================================"

# Run the experiment
python $VSC_DATA/AML_work_study/AML_work_study/main.py \
    --fl_algo "$FL_ALGO" \
    --model "$MODEL" \
    --size "$SIZE" \
    --ir "$IR" \
    $EXTRA_FLAGS

EXIT_CODE=$?

# Print completion info
echo "======================================================================"
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================================================"

exit $EXIT_CODE

