#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SUGARCREPE_CONFIG=configs/sugarcrepe_cuhk_pedes.yaml
uv run python sugarcrepe-pedes.py
