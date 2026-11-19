"""Validate raw state identities against the installed LIBERO task catalog."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from .compare import read
from .eval import write_json


def verify(run, source, partial):
    sys.path.insert(0, str(source))
    from libero.libero import benchmark

    specification = read(run / "run.json")
    if specification["engine"] != "mujoco" or specification["suite"] != "libero_spatial":
        raise ValueError("Expected a LIBERO Spatial reference run")
    catalog = benchmark.get_benchmark_dict()["libero_spatial"]()
    policy_path = run / "policy.json"
    instructions = read(policy_path)["instructions"] if policy_path.is_file() else {}
    count = 0
    entries = []
    for task_id in specification["tasks"]:
        task = catalog.get_task(task_id)
        if instructions and instructions[str(task_id)] != task.language:
            raise ValueError(f"Reference prompt differs from LIBERO for task {task_id}")
        states = catalog.get_task_init_states(task_id)
        if len(states) < specification["episodes"]:
            raise ValueError(f"LIBERO has fewer requested initial states for task {task_id}")
        verified = 0
        for index in range(specification["episodes"]):
            record = run / f"task-{task_id:02}" / f"state-{index:02}" / "initial-state.json"
            if partial and not record.is_file():
                continue
            state = np.asarray(states[index], dtype="<f8")
            digest = hashlib.sha256(state.tobytes()).hexdigest()
            initial = read(record)
            if digest != initial["initial_state_sha256"]:
                raise ValueError(f"{record}: hash differs from LIBERO state {index}")
            verified += 1
        count += verified
        entries.append({"native_task": task_id, "prompt": task.language, "verified_states": verified})
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    return {"libero_commit": commit, "checked_states": count, "tasks": entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    root = source / "libero/libero"
    with tempfile.TemporaryDirectory(prefix="mxf-reference-audit-") as temporary:
        os.environ["LIBERO_CONFIG_PATH"] = temporary
        write_json(Path(temporary) / "config.yaml", {"init_states": str(root / "init_files")})
        result = verify(args.run, source, args.partial)
    write_json(args.output, result)
    print(f"Verified {result['checked_states']} source LIBERO initial states")


if __name__ == "__main__":
    main()
