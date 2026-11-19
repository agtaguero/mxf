"""Verify same-state burner probes and the 50 paired π0.5 reruns."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
from pathlib import Path
from statistics import median
import xml.etree.ElementTree as ET

import numpy as np

from .actuator_replay import BOWL
from .assets import STOVE_BURNER, extend_stove_burner
from .audit import audit_episode
from .compare import read
from .eval import write_json


@lru_cache(maxsize=None)
def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def scene_signature(tree):
    for asset in tree.findall("asset/*"):
        if file := asset.get("file"):
            asset.set("file", sha256(file))
    return ET.tostring(tree.getroot())


def burner_top(probe):
    geom = probe["geometry"][STOVE_BURNER]
    return geom["center_m"][2] + geom["half_extents_m"][2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Original full MuJoCo suite run")
    parser.add_argument("before", type=Path, help="Task-07 closed-loop run with only the UV fix")
    parser.add_argument("work", type=Path, help="Run output with normals, solver-backends and closed-loop")
    args = parser.parse_args()
    reference, before, work = (value.resolve() for value in (args.reference, args.before, args.work))
    after = work / "closed-loop"
    old_run, new_run = (read(root / "run.json") for root in (before, after))
    if old_run.pop("extend_stove_burner", False) or not new_run.pop("extend_stove_burner", True):
        raise ValueError("Expected original geometry followed by the explicit burner experiment")
    for metadata in (old_run, new_run):
        metadata.setdefault("motrix_solver_override", None)
        metadata.setdefault("osc_reset", "legacy")
        metadata.setdefault("inertia_groups", False)
    if old_run != new_run or new_run["tasks"] != [7] or new_run["episodes"] != 50:
        raise ValueError("Paired retests require identical single-task protocols")
    if read(before / "policy.json") != read(after / "policy.json"):
        raise ValueError("Policy inputs differ in weights, prompt or per-episode random offsets")

    pairs = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        replays = [pool.submit(audit_episode, reference, 7, state, after,
                               new_run["max_steps"], new_run["camera_asset_warmup_min_seconds"])
                   for state in range(50)]
        for index in range(50):
            name = f"state-{index:02}"
            original, old, new = (root / "task-07" / name for root in (reference, before, after))
            native, motrix, solvers = (read(work / key / f"{name}.json") for key in
                                      ("normals-mujoco", "normals", "solver-backends"))
            old_result, new_result = (read(root / "result.json") for root in (old, new))
            for field in ("initial_state_sha256", "fixtures_sha256"):
                if len({result[field] for result in (read(original / "initial-state.json"),
                        native, motrix, solvers, old_result, new_result)}) != 1:
                    raise ValueError(f"{name}: initial-state or fixture identity differs")
            desired = ET.parse(old / "case.xml")
            if not extend_stove_burner(desired) or scene_signature(desired) != scene_signature(ET.parse(new / "case.xml")):
                raise ValueError(f"{name}: compiled input changed beyond the burner extension")

            base, fixed = (motrix["variants"][key] for key in ("original", "extend_1mm"))
            native_probe = native["variants"]["original"]
            for geom, native_geom in native_probe["geometry"].items():
                quaternion = np.asarray(native_geom["quat_xyzw"])
                imported_quaternion = np.asarray(base["geometry"][geom]["quat_xyzw"])
                if any(not np.allclose(native_geom[key], base["geometry"][geom][key], atol=1e-7, rtol=0.)
                       for key in ("center_m", "half_extents_m")) or min(
                           np.linalg.norm(quaternion - imported_quaternion),
                           np.linalg.norm(quaternion + imported_quaternion)) > 1e-6:
                    raise ValueError(f"{name}: original imported geometry does not match the native scene")
            if (abs(burner_top(base) - burner_top(fixed)) > 1e-7
                    or base["normals_before_step"] != native_probe["normals"]
                    or base["normals_before_step"].get("down", 0) == 0 or not base["bowl_force_z_n"] < 0
                    or fixed["normals_before_step"].get("down", 0) != 0 or not fixed["bowl_force_z_n"] > 0):
                raise ValueError(f"{name}: directional fix or top-plane check failed")
            if any(item["normals_before_step"] != base["normals_before_step"] for item in solvers["variants"].values()):
                raise ValueError(f"{name}: solver backend trial changed the generated normals")

            trajectories = [read(root / "trace.json") for root in (old, new)]
            native_trace = read(original / "trace.json")
            native_warmup_z = native_trace[9]["object_positions"][BOWL][2]
            pairs.append({"state": index, "before": old_result["success"], "after": new_result["success"],
                          "native_force_z_n": native_probe["bowl_force_z_n"],
                          "before_force_z_n": base["bowl_force_z_n"], "after_force_z_n": fixed["bowl_force_z_n"],
                          "backend_force_z_n": {key: val["bowl_force_z_n"] for key, val in solvers["variants"].items()},
                          "before_normals": base["normals_before_step"], "after_normals": fixed["normals_before_step"],
                          "warmup_bowl_height_gap_mm": [1000 * (trace[9]["object_positions"][BOWL][2] - native_warmup_z)
                                                       for trace in trajectories],
                          "bowl_lift_m": [max(row["object_positions"][BOWL][2] for row in trace[10:])
                                          - trace[9]["object_positions"][BOWL][2] for trace in trajectories],
                          "replay_frames": replays[index].result()["frames"]})
    summary = {"pairs": len(pairs), "before_successes": sum(row["before"] for row in pairs),
               "after_successes": sum(row["after"] for row in pairs),
               "improved": sum(not row["before"] and row["after"] for row in pairs),
               "regressed": sum(row["before"] and not row["after"] for row in pairs),
               "lifted": [sum(row["bowl_lift_m"][i] >= .03 for row in pairs) for i in range(2)],
               "warmup_bowl_height_gap_mm": [median(row["warmup_bowl_height_gap_mm"][i] for row in pairs)
                                             for i in range(2)],
               "force_z_medians_n": {field: median(row[field] for row in pairs) for field in
                                     ("native_force_z_n", "before_force_z_n", "after_force_z_n")},
               "backends": {key: {"median_force_z_n": median(row["backend_force_z_n"][key] for row in pairs),
                                  "upward_force_cases": sum(row["backend_force_z_n"][key] > 0 for row in pairs)}
                            for key in pairs[0]["backend_force_z_n"]},
               "replay_frames": sum(row["replay_frames"] for row in pairs)}
    write_json(work / "summary.json", {**summary, "episodes": pairs})
    print(summary)


if __name__ == "__main__":
    main()
