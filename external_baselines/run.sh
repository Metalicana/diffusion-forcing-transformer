#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec "${STUDY_PYTHON:-python3}" -u -m external_baselines.run "$@"
