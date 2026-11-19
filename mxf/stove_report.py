"""Verify and summarize the task-07 texture correction and physical controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image, ImageDraw

from .audit import audit_episode
from .compare import read
from .eval import write_json
from .stove_physics import BOWL

TASK = "task-07"
STATES = 50
STATE_FIELDS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
CAMERA_FIELDS = ("agentview_image", "robot0_eye_in_hand_image")


def verify(reference, before, work):
    fixed_before, fixed_after, after = (work / name for name in
                                        ("fixed-before", "fixed-uv", "closed-loop-uv"))
    source_run, old_run, new_run = (read(root / "run.json") for root in (reference, before, after))
    for field in ("suite", "weights_sha256", "model_manifest_sha256", "max_steps", "policy_seed"):
        if source_run[field] != old_run[field] or old_run[field] != new_run[field]:
            raise ValueError(f"Policy or run setting differs: {field}")
    if new_run["tasks"] != [7] or new_run["episodes"] != STATES or new_run["policy_rng_source"] != str(before):
        raise ValueError("Expected a task-07-only rerun with original draw offsets")
    policies = [read(root / "policy.json") for root in (reference, before, after)]
    if len({policy["instructions"]["7"] for policy in policies}) != 1:
        raise ValueError("Task prompt differs")

    pairs = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        replays = [pool.submit(audit_episode, reference, 7, state, after,
                               new_run["max_steps"], new_run["camera_asset_warmup_min_seconds"])
                   for state in range(STATES)]
        for state in range(STATES):
            name = f"state-{state:02}"
            native_case, old_case, new_case, old_fixed_case, new_fixed_case = (
                root / TASK / name for root in (reference, before, after, fixed_before, fixed_after))
            results = [read(folder / "result.json") for folder in
                       (native_case, old_case, new_case, old_fixed_case, new_fixed_case)]
            if any(len({row[field] for row in results}) != 1
                   for field in ("initial_state_sha256", "fixtures_sha256")):
                raise ValueError(f"{name}: initial state or fixture differs")
            if (old_fixed_case / "trace.json").read_bytes() != (new_fixed_case / "trace.json").read_bytes():
                raise ValueError(f"{name}: fixed-action trace changed after the texture correction")
            records = [read(folder / "trace.json") for folder in
                       (native_case, old_case, new_case, new_fixed_case)]
            if any(left["action"] != right["action"] for left, right in zip(records[0], records[3], strict=True)):
                raise ValueError(f"{name}: fixed actions differ from original MuJoCo commands")
            with np.load(old_case / "policy-inputs.npz") as old, np.load(new_case / "policy-inputs.npz") as new:
                if not all(np.array_equal(old[field][0], new[field][0]) for field in STATE_FIELDS):
                    raise ValueError(f"{name}: proprioception changed before the first policy query")
            contact_data = read(work / "fixed-contacts" / name / "contacts.json")
            if any(contact_data[field] != results[0][field]
                   for field in ("initial_state_sha256", "fixtures_sha256")):
                raise ValueError(f"{name}: contact probe has a different initial state or fixture")
            touches = contact_data["steps"]
            core = [row for row in touches if row["reference_window"]]
            if [row["success"] for row in core] != [row["success"] for row in records[3]]:
                raise ValueError(f"{name}: contact probe changed the fixed-action replay")
            first_touch = next((i for i, row in enumerate(core) if row["finger_stove_contacts"] or any(
                geom.startswith("gripper0_") for geom in row["bowl_contacts"])), None)
            label = "success"
            if not results[4]["success"]:
                label = "within_xy_and_height_without_contact" if any(
                    row["step"] > 0 and row["bowl_plate_horizontal_distance_m"] < .03
                    and row["bowl_plate_height_m"] >= 0 for row in core) else "outside_xy_or_height"
            pairs.append({
                "state": state, "before": results[1]["success"], "after": results[2]["success"],
                "fixed_result": label,
                "fixed_recovered_with_extra_open": label != "success" and any(row["success"] for row in touches),
                "lift_m": {key: max(row["object_positions"][BOWL][2] for row in trace[10:])
                           - trace[9]["object_positions"][BOWL][2]
                           for key, trace in zip(("mujoco", "before", "after", "fixed"), records, strict=True)},
                "bowl_warmup_height_gap_m": (core[9]["bowl_z_m"] - records[0][9]["object_positions"][BOWL][2]),
                "eef_before_first_touch_error_m": core[first_touch - 1]["eef_error_m"] if first_touch else None,
                "replay_frames": replays[state].result()["frames"],
            })
    return pairs


def camera_montage(sources, output):
    image = Image.new("RGB", (768, 552))
    draw = ImageDraw.Draw(image)
    for column, (root, label) in enumerate(sources):
        with np.load(root / TASK / "state-00/policy-inputs.npz") as inputs:
            for row, field in enumerate(CAMERA_FIELDS):
                left, top = column * 256, row * 276
                draw.text((left + 4, top + 4), f"{label} / {field.removesuffix('_image')}", fill="white")
                image.paste(Image.fromarray(inputs[field][0, ::-1, ::-1]), (left, top + 20))
    image.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "before", "work"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    reference, before, work = (path.resolve() for path in (args.reference, args.before, args.work))
    pairs = verify(reference, before, work)
    summary = {
        "pairs": len(pairs),
        "before_successes": sum(row["before"] for row in pairs),
        "after_successes": sum(row["after"] for row in pairs),
        "improved": sum(not row["before"] and row["after"] for row in pairs),
        "regressed": sum(row["before"] and not row["after"] for row in pairs),
        "lift_at_least_3_cm": {name: sum(row["lift_m"][name] >= .03 for row in pairs)
                               for name in pairs[0]["lift_m"]},
        "fixed_replay_outcomes": {label: sum(row["fixed_result"] == label for row in pairs) for label in
                                  ("success", "within_xy_and_height_without_contact", "outside_xy_or_height")},
        "fixed_recovered_with_extra_open": sum(row["fixed_recovered_with_extra_open"] for row in pairs),
        "median_bowl_warmup_height_gap_mm": 1000 * median(row["bowl_warmup_height_gap_m"] for row in pairs),
        "median_pre_touch_eef_error_mm": 1000 * median(row["eef_before_first_touch_error_m"] for row in pairs
                                                       if row["eef_before_first_touch_error_m"] is not None),
        "replay_frames": sum(row["replay_frames"] for row in pairs),
    }
    write_json(work / "summary.json", {**summary, "episodes": pairs})
    camera_montage(((reference, "MuJoCo"), (before, "MotrixSim before UV fix"),
                    (work / "closed-loop-uv", "MotrixSim after UV fix")), work / "first-query-cameras.png")
    print(summary)


if __name__ == "__main__":
    main()
