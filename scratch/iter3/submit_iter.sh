#!/bin/bash
# Submit the isolation test against the worktree's nsosim via PYTHONPATH override.
set -e

WORKTREE=/dataNAS/people/aagatti/programming/nsosim/.claude/worktrees/wrap-fitter-robustness
SCRATCH=$WORKTREE/scratch/iter3
SLURM_DIR=$SCRATCH/slurm_outputs
mkdir -p "$SLURM_DIR"

RESULTS_BASE=/dataNAS/people/aagatti/projects/comak_gait_simulation_results
A_RUN=$RESULTS_BASE/e2e_determinism_20260510_221720_A
B_RUN=$RESULTS_BASE/e2e_determinism_20260510_221720_B
TS=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT=$SCRATCH/build_isolation_${TS}
mkdir -p "$OUTPUT_ROOT"

SCRIPT=/dataNAS/people/aagatti/projects/comak_gait_simulation/tests/swap_experiments/isolate_build_joint_model.py

BATCH=$(mktemp "${SLURM_DIR}/isolation_${TS}_XXXXXX.sh")
cat > "$BATCH" <<SLURM_EOF
#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=24gb
#SBATCH --gres=gpu:1
#SBATCH --time=0-01:00:00
#SBATCH --output=${SLURM_DIR}/isolation_${TS}-%j.out
#SBATCH --job-name=wrap_fitter_iter3

set -e
echo "Started: \$(date)"
echo "Worktree HEAD: \$(cd ${WORKTREE} && git rev-parse HEAD)"
echo "Worktree branch: \$(cd ${WORKTREE} && git branch --show-current)"

source /dataNAS/people/aagatti/miniconda/etc/profile.d/conda.sh
conda activate comak
export PYTHONPATH=${WORKTREE}:\${PYTHONPATH}

PY=/dataNAS/people/aagatti/miniconda/envs/comak/bin/python

# Verify which nsosim is being used
\$PY -c "import nsosim; print('nsosim from:', nsosim.__file__)"

\$PY ${SCRIPT} \\
    --a-run "${A_RUN}" \\
    --b-run "${B_RUN}" \\
    --output-root "${OUTPUT_ROOT}"

echo "Finished: \$(date)"
echo "Output: ${OUTPUT_ROOT}"
SLURM_EOF

chmod +x "$BATCH"
OUT=$(sbatch "$BATCH")
JOB_ID=$(echo "$OUT" | awk '{print $NF}')
echo "$JOB_ID" > $SCRATCH/last_job_id
echo "Submitted job $JOB_ID"
echo "Output:   $OUTPUT_ROOT"
echo "Log:      ${SLURM_DIR}/isolation_${TS}-${JOB_ID}.out"
echo "Monitor:  tail -f ${SLURM_DIR}/isolation_${TS}-${JOB_ID}.out"
