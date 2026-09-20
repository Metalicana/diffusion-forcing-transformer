#!/usr/bin/env bash
# Run on the cluster after pulling. Does not connect to any other machine.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ $# -ne 1 ]]; then
    echo 'Usage: bash external_baselines/setup_dfot.sh GPU_INDEX_OR_UUID' >&2
    exit 2
fi
command -v conda >/dev/null || { echo 'Load conda first (e.g. source your Miniconda conda.sh).' >&2; exit 1; }
DFOT_ENV_PREFIX="${DFOT_ENV_PREFIX:-$HOME/.conda/envs/dfot}"
case "$DFOT_ENV_PREFIX" in
    /*) ;;
    *) echo 'DFOT_ENV_PREFIX must be absolute.' >&2; exit 2 ;;
esac
if [[ ! -x "$DFOT_ENV_PREFIX/bin/python" ]]; then
    conda create --yes --prefix "$DFOT_ENV_PREFIX" --override-channels -c conda-forge python=3.10 pip ffmpeg
fi
DFOT_PYTHON="$DFOT_ENV_PREFIX/bin/python"
"$DFOT_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use a dedicated Python 3.10 DFoT environment"'
"$DFOT_PYTHON" -m pip install --upgrade pip
"$DFOT_PYTHON" -m pip install -c external_baselines/constraints-dfot.txt setuptools
"$DFOT_PYTHON" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"$DFOT_PYTHON" -m pip install -r requirements.txt -c external_baselines/constraints-dfot.txt 'imageio[ffmpeg]' setuptools
"$DFOT_PYTHON" -m pip check
mkdir -p outputs/dfot_environment
"$DFOT_PYTHON" -m pip freeze > outputs/dfot_environment/pip-freeze.txt
export PATH="$DFOT_ENV_PREFIX/bin:$PATH"
"$DFOT_PYTHON" -u -m external_baselines.environment_smoke --gpu "$1" --output outputs/dfot_environment/smoke.json
"$DFOT_PYTHON" -m unittest discover -s tests -p test_external_baselines.py -v
printf '\nEnvironment ready: %s\n' "$DFOT_PYTHON"
printf 'Set models.dfot.python in study.json to this path.\n'
