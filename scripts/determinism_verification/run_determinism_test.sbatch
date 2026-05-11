#!/bin/bash
#SBATCH --job-name=nsosim_determinism
#SBATCH --partition=BMR-AI
#SBATCH --output=scripts/determinism_verification/logs/determinism-%j.out
#SBATCH --error=scripts/determinism_verification/logs/determinism-%j.err
#SBATCH --chdir=/dataNAS/people/aagatti/programming/nsosim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G

# Run the GPU-only determinism tests for nsosim. These verify that two
# runs of decode_latent_to_osim / decode_joint_from_descriptors with the
# same seed produce bit-identical mesh outputs.
#
# Submit:
#   sbatch scripts/determinism_verification/run_determinism_test.sbatch
#
# Tail logs:
#   tail -f logs/determinism-<jobid>.out

set -euo pipefail

REPO_DIR="/dataNAS/people/aagatti/programming/nsosim"
cd "$REPO_DIR"
mkdir -p scripts/determinism_verification/logs

echo "=================================================="
echo "nsosim determinism verification"
echo "Job ID:  ${SLURM_JOB_ID:-<interactive>}"
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "Repo:    $REPO_DIR"
echo "=================================================="

nvidia-smi || echo "nvidia-smi failed — no GPU?"
echo ""

# Activate the comak conda env (has pytest, torch+CUDA, opensim, etc.)
# `conda run` keeps the surrounding shell clean.
echo "Using conda env: comak"
conda run -n comak python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo ""
echo "Running determinism tests..."
echo "----------------------------"

conda run -n comak python -m pytest tests/test_determinism.py -v --tb=short

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=================================================="

exit $EXIT_CODE
