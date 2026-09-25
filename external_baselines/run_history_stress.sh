#!/usr/bin/env bash
# Uses the already installed DFoT environment and cached native Mini dataset.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
unset PYTHONHOME PYTHONPATH
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Run inside the existing Newton GPU allocation, not the login node.' >&2
    exit 2
fi
export DFOT_ENV_PREFIX="${DFOT_ENV_PREFIX:-$HOME/.conda/envs/dfot}"
export PATH="$DFOT_ENV_PREFIX/bin:$PATH"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=disabled
OUT="${DFOT_STRESS_OUTPUT:-$HOME/memcam_results/dfot_history_stress_mini_n10}"
mkdir -p "$OUT"
LOG="$OUT/launch_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S).log"
run_stress() {
    "$DFOT_ENV_PREFIX/bin/python" -u -m external_baselines.environment_smoke --gpu inherit
    "$DFOT_ENV_PREFIX/bin/python" -m unittest discover -s tests -p 'test_history*.py' -v
    "$DFOT_ENV_PREFIX/bin/python" -u -m external_baselines.history_stress \
        --output "$OUT" --max-seconds "${DFOT_STRESS_SECONDS:-7200}" "$@"
}
run_stress "$@" 2>&1 | tee "$LOG"
