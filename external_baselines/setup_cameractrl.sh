#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
CAMERA_ENV="${CAMERA_ENV:-$HOME/.conda/envs/cameractrl}"
CAMERA_REPO="${CAMERA_REPO:-$HOME/CameraCtrl}"
CAMERA_COMMIT=1f9d8baeff66f6c60bb9848dfafcc667b3a22a28
command -v conda >/dev/null
if [[ ! -x "$CAMERA_ENV/bin/python" ]]; then
  conda create -y --prefix "$CAMERA_ENV" --override-channels -c conda-forge python=3.10 pip ffmpeg
fi
if [[ ! -d "$CAMERA_REPO" ]]; then
  git clone --branch svd https://github.com/hehao13/CameraCtrl.git "$CAMERA_REPO"
  git -C "$CAMERA_REPO" checkout "$CAMERA_COMMIT"
fi
if [[ "$(git -C "$CAMERA_REPO" rev-parse HEAD)" != "$CAMERA_COMMIT" ]]; then
  echo 'CameraCtrl checkout differs from inspected revision; use a separate CAMERA_REPO path.' >&2
  exit 1
fi
"$CAMERA_ENV/bin/python" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"$CAMERA_ENV/bin/python" -m pip install 'numpy==1.26.4' 'setuptools==80.9.0' \
  'diffusers==0.24.0' 'transformers==4.39.3' 'huggingface-hub==0.25.2' \
  'imageio[ffmpeg]' 'opencv-python<4.12' einops decord omegaconf safetensors wandb termcolor
"$CAMERA_ENV/bin/python" -m pip check
export PYTHONPATH="$CAMERA_REPO${PYTHONPATH:+:$PYTHONPATH}"
"$CAMERA_ENV/bin/python" -u -c 'print("Importing CameraCtrl SVD pipeline", flush=True); from inference import get_pipeline; import torch; print("Torch:",torch.__version__,"architectures:",torch.cuda.get_arch_list())'
printf '\nCameraCtrl environment: %s\nNext: run external_baselines.cameractrl_smoke on allocated GPU 1.\n' "$CAMERA_ENV/bin/python"
