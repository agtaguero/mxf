"""Verify paired state, contact, and replay evidence for the Spatial free-space comparison."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .compare import read
from .eval import write_json


def stats(values):
    data = np.asarray(values)
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite comparison")
    return {"count": len(values), "median": float(np.median(data)), "p95": float(np.percentile(data, 95)),
            "max": float(data.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("work", type=Path)
    args = parser.parse_args()
    work = args.work
    metrics, stages, components = defaultdict(list), defaultdict(list), defaultdict(list)
    validation, split = [], {"diagnosis": [], "held_out": []}
    per_task = {}
    for task in range(10):
        folder = f"task-{task:02}"
        ablations = {engine: read(work / "ablations" / engine / (folder + ".json"))
                     for engine in ("native", "motrix")}
        for engine, table in ablations.items():
            air = read(work / "air-details" / engine / (folder + ".json"))
            for name, extra in air.items():
                table[name].update(extra)
        task_validation = []
        for index in range(50):
            name = f"state-{index:02}"
            path = Path(folder) / name
            state = read(args.reference / path / "initial-state.json")
            check = read(work / "validation" / path / "validation.json")
            assert all(state[k] == check[k] for k in ("fixtures_sha256", "initial_state_sha256"))
            row = [check[k]["eef_error_mm"][n] for n in (9, 11) for k in ("legacy", "restored")]
            validation.append(row)
            task_validation.append(row)
            diagnosis = index % 10 == 0
            split["diagnosis" if diagnosis else "held_out"].append(row)
            if not diagnosis:
                continue
            ref = read(work / "native-matched" / folder / (name + ".json"))
            assert max(ref["historical_eef_error_m"]) < 1e-12
            assert all(state[k] == ref[k] for k in ("fixtures_sha256", "initial_state_sha256"))
            motrix = read(work / "motrix" / path / "result.json")
            metrics["legacy_reset_orientation_error_deg"].append(np.degrees(motrix["reset_goal_delta_rad"]))
            metrics["legacy_reset_rest_rms_rad"].append(motrix["rest_delta_rad"])
            metrics["legacy_first_goal_position_error_mm"].append(1000*np.linalg.norm(
                np.subtract(ref["first_goal_pos"], ref["frames"]["0"]["pose"])))
            for tick, frame in ref["frames"].items():
                other = motrix["frames"][tick]
                assert not frame["contacts"] and not other["contacts"]
                for key, tests in frame["step"].items():
                    probe = other["step"][key]
                    assert not tests["contacts"] and not any(probe["contacts"])
                    stages[f"step_{tick}_same_torque_arm_vel_rms_rad_s"].append(probe["arm_vel_error_rms"])
                for probe in other["targets"].values():
                    stages[f"step_{tick}_same_target_torque_rms_nm"].append(probe["error_rms"])
                for key in ("jac_max_error", "mass_max_error", "bias_error_rms"):
                    stages[key].append(other[key])
            for variant in ablations["native"][name]:
                a, b = (np.asarray(ablations[e][name][variant]) for e in ("native", "motrix"))
                if variant == "baseline":
                    np.testing.assert_allclose(a[:7], ref["frames"]["12"]["step"]["osc"]["vel"], rtol=0, atol=1e-12)
                    np.testing.assert_allclose(b[:7], motrix["frames"]["12"]["step"]["osc"]["vel"], rtol=0, atol=1e-7)
                metrics["moving_osc_arm_vel_rms_rad_s_" + variant].append(np.sqrt(np.mean((a[:7]-b[:7])**2)))
            cmp = read(work / "components" / path / "validation.json")
            for mode in ("legacy", "goals_only", "rest_only", "bias_only", "restored"):
                components[mode].append(cmp[mode]["eef_error_mm"][9])
            for mode in ("legacy","restored"):
                np.testing.assert_allclose(cmp[mode]["eef_error_mm"],check[mode]["eef_error_mm"],rtol=0,atol=1e-7)
        per_task[folder] = summarize(task_validation)
    write_json(work / "summary.json", {
        "protocol": "MotrixSim 0.10.1 vs MuJoCo 2.3.7, original geometry, no rendering; 5 states/task for matched free-space probes; 50 states/task for 12 fixed actions (10 warmup, 2 recorded actions).",
        "matched_probes": {k: stats(v) for k, v in stages.items()},
        "reset_and_ablations": {k: stats(v) for k, v in metrics.items()},
        "warmup_components_mm": {k: stats(v) for k,v in components.items()},
        "all": summarize(validation), "by_split": {k:summarize(v) for k,v in split.items()},
        "by_task": per_task})


def summarize(rows):
    a, b, c, d = np.asarray(rows).T
    return {"legacy_warmup_mm": stats(a), "restored_warmup_mm": stats(b),
            "warmup_improved": int(np.sum(b < a)),
            "legacy_after_two_actions_mm": stats(c), "restored_after_two_actions_mm": stats(d),
            "after_two_improved": int(np.sum(d < c))}


if __name__ == "__main__":
    main()
