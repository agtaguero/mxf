#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

reference=${1:?Pass the full MuJoCo run directory}
root=${2:?Pass a new directory for the MotrixSim retry}
platform=${MXF_PLATFORM:-"$PWD/../sim-platform"}
python="$PWD/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export WGPU_BACKEND="${WGPU_BACKEND:-vulkan}"

mkdir -p "$(dirname "$root")"
mkdir "$root"
printf '%s\n' "$$" >"$root/pid"
child=
finish() {
    status=$?
    if [[ -n "$child" ]]; then
        kill "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
    printf '%s\n' "$status" >"$root/exit-code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

"$python" -m mxf.eval --platform "$platform" --engine motrixsim \
    --scenes "$reference" --wait-for-scenes 240 --tasks {0..9} \
    --episodes 50 --gpu "${MXF_MOTRIX_GPU:-0}" --output "$root/motrixsim" \
    >"$root/motrixsim.log" 2>&1 &
child=$!
wait "$child"
child=
python3 -m mxf.compare "$reference" "$root/motrixsim" --output "$root/comparison.json"
"$python" -m mxf.audit "$reference" "$root/motrixsim" --output "$root/evidence"
