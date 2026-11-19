#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

root=${1:?Pass a new path for the campaign}
platform=${MXF_PLATFORM:-"$PWD/../sim-platform"}
reference_python="$platform/.runtimes/envs/libero/bin/python"
motrix_python="$PWD/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$(dirname "$root")"
mkdir "$root"
printf '%s\n' "$$" >"$root/pid"
children=()
finish() {
    status=$?
    if (("${#children[@]}")); then
        kill "${children[@]}" 2>/dev/null || true
        wait "${children[@]}" 2>/dev/null || true
    fi
    printf '%s\n' "$status" >"$root/exit-code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

run_pair() {
    local phase=$1 episodes=$2 first_task left_status=0 right_status=0
    shift 2
    local output="$root/$phase"
    mkdir "$output"
    printf '%s\n' "$phase" >"$root/phase"
    MUJOCO_EGL_DEVICE_ID="${MXF_MUJOCO_EGL_DEVICE:-1}" \
    "$reference_python" -m mxf.eval --platform "$platform" --tasks "$@" \
        --episodes "$episodes" --gpu 1 --output "$output/mujoco" >"$output/mujoco.log" 2>&1 &
    children+=("$!")
    printf -v first_task 'task-%02d' "$1"
    until [[ -s "$output/mujoco/$first_task/state-00/result.json" ]]; do
        kill -0 "${children[0]}"
        sleep 2
    done
    WGPU_BACKEND="${WGPU_BACKEND:-vulkan}" \
    "$motrix_python" -m mxf.eval --platform "$platform" --engine motrixsim \
        --scenes "$output/mujoco" --wait-for-scenes 240 --tasks "$@" \
        --episodes "$episodes" --gpu 0 --output "$output/motrixsim" >"$output/motrixsim.log" 2>&1 &
    children+=("$!")
    # Keep an intact reference run available if MotrixSim has to be relaunched.
    wait "${children[0]}" || left_status=$?
    wait "${children[1]}" || right_status=$?
    children=()
    if (( left_status || right_status )); then
        printf 'MuJoCo exited %s; MotrixSim exited %s\n' "$left_status" "$right_status" >&2
        return 1
    fi
    python3 -m mxf.compare "$output/mujoco" "$output/motrixsim" --output "$output/comparison.json"
    "$motrix_python" -m mxf.audit "$output/mujoco" "$output/motrixsim" --output "$output/evidence"
    python3 - "$output/comparison.json" "$episodes" "$#" <<'PY'
import json
from pathlib import Path
import sys

data = json.loads(Path(sys.argv[1]).read_text())
if not data["complete"] or data["pairs"] != int(sys.argv[2]) * int(sys.argv[3]):
    raise SystemExit("Incomplete paired comparison")
if int(sys.argv[2]) == 20 and any(
    t["both_success"] + t["motrixsim_only"] == 0 for t in data["tasks"]
):
    raise SystemExit("A pilot task has no successful MotrixSim rollout")
PY
}

run_pair pilot 20 1 2 8
run_pair spatial 50 {0..9}
printf '%s\n' complete >"$root/phase"
