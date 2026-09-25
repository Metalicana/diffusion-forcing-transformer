#!/bin/bash
#SBATCH --job-name=dfot_keep_pilot
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=dfot_keep_%j.out
#SBATCH --error=dfot_keep_%j.err

set -euo pipefail
cd "${DFOT_ROOT:-$HOME/diffusion-forcing-transformer}"
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=disabled
PYTHON="${DFOT_ENV_PREFIX:-$HOME/.conda/envs/dfot}/bin/python"
printf 'Job=%s Node=%s CUDA_VISIBLE_DEVICES=%s\n' "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "${CUDA_VISIBLE_DEVICES:-unset}"
"$PYTHON" -u -m external_baselines.environment_smoke --gpu inherit
"$PYTHON" -u -m external_baselines.native_history_transfer \
  --output "${DFOT_TRANSFER_OUTPUT:-$HOME/memcam_results/dfot_history_transfer_mini_n5}" \
  --max-seconds 12600 "$@"
