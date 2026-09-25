#!/usr/bin/env bash
# Run in the existing GPU allocation, not on a login node.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
unset PYTHONHOME PYTHONPATH
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Run this launcher inside your allocated Newton GPU shell.' >&2
    exit 2
fi
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi unavailable in this allocation.' >&2; exit 1; }
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
if [[ ! -f "../MemCam/diffsynth/pipelines/memory_policies.py" ]]; then
    echo 'The sibling ~/MemCam checkout is required for the production KEEPSAKE scorer.' >&2
    exit 1
fi
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=disabled
export DFOT_ENV_PREFIX="${DFOT_ENV_PREFIX:-$HOME/.conda/envs/dfot}"
TOTAL_SECONDS="${DFOT_TOTAL_SECONDS:-12600}"
if [[ ! "$TOTAL_SECONDS" =~ ^[0-9]+$ ]] || (( TOTAL_SECONDS < 1 )); then
    echo 'DFOT_TOTAL_SECONDS must be a positive integer.' >&2
    exit 2
fi
OUT="${DFOT_OUTPUT:-$HOME/memcam_results/dfot_history_transfer_mini_n5}"
mkdir -p "$OUT"
LOG="$OUT/launch_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S).log"
STARTED=$SECONDS
run_pilot() {
    printf 'Job=%s GPU mask=%s\n' "$SLURM_JOB_ID" "${CUDA_VISIBLE_DEVICES:-unset}"
    bash external_baselines/setup_dfot.sh inherit
    "$DFOT_ENV_PREFIX/bin/python" -m unittest discover -s tests -p 'test_history_transfer.py' -v
    "$DFOT_ENV_PREFIX/bin/python" -m unittest discover -s tests -p 'test_dfot*.py' -v
    local remaining=$(( TOTAL_SECONDS - (SECONDS - STARTED) ))
    if (( remaining < 300 )); then
        echo 'Setup consumed the launcher time budget. Resume in another allocation.' >&2
        return 2
    fi
    "$DFOT_ENV_PREFIX/bin/python" -u -m external_baselines.native_history_transfer \
        --output "$OUT" --scenes 5 --max-seconds "$remaining"
}
run_pilot 2>&1 | tee "$LOG"
printf '\nResults: %s/scores.csv\nLog: %s\n' "$OUT" "$LOG"
