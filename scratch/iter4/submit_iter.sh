#!/bin/bash
set -e
WORKTREE=/dataNAS/people/aagatti/programming/nsosim/.claude/worktrees/wrap-fitter-robustness
SCRATCH=$WORKTREE/scratch/iter4
SLURM_DIR=$SCRATCH/slurm_outputs
mkdir -p "$SLURM_DIR"
TS=$(date +%Y%m%d_%H%M%S)

BATCH=$(mktemp "${SLURM_DIR}/refit_${TS}_XXXXXX.sh")
cat > "$BATCH" <<SLURM_EOF
#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00
#SBATCH --output=${SLURM_DIR}/refit_${TS}-%j.out
#SBATCH --job-name=wrap_fit_quality

set -e
echo "Started: \$(date)"
echo "Worktree HEAD: \$(cd ${WORKTREE} && git rev-parse HEAD)"

source /dataNAS/people/aagatti/miniconda/etc/profile.d/conda.sh
conda activate comak
export PYTHONPATH=${WORKTREE}:\${PYTHONPATH}

PY=/dataNAS/people/aagatti/miniconda/envs/comak/bin/python
\$PY ${SCRATCH}/refit_template_and_compare.py
echo "Finished: \$(date)"
SLURM_EOF

chmod +x "$BATCH"
OUT=$(sbatch "$BATCH")
JOB_ID=$(echo "$OUT" | awk '{print $NF}')
echo "$JOB_ID" > $SCRATCH/last_job_id
echo "Submitted job $JOB_ID"
echo "Log: ${SLURM_DIR}/refit_${TS}-${JOB_ID}.out"
