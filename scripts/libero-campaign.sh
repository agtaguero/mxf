#!/usr/bin/env bash
# One serial policy stream per suite. Each suite starts a fresh seeded worker.
set -euo pipefail
cd "$(dirname "$0")/.."
scenes=${1:?Pass the prepared four-suite export root}
output=${2:?Pass the output root}
gpu=${3:?Pass the policy GPU index}
shift 3
export WGPU_BACKEND=vulkan OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$output"
for suite in "$@"; do
    .venv/bin/python -m mxf.eval --engine motrixsim --suite "$suite" --scenes "$scenes/$suite" \
        --tasks {0..9} --episodes 50 --gpu "$gpu" --compact --no-video --output "$output/$suite" \
        > "$output/$suite.log" 2>&1
done
