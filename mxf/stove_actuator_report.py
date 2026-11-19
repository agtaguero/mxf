"""Audit the actuator replay and test the burner contact at the same pose."""
import argparse
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import motrixsim as mx
import numpy as np

from .actuator_replay import BOWL
from .compare import read
from .eval import write_json
from .stove_actuator_motrix import dof_map


def burner_step(xml, layout, q, v, ctrl, bowl_body):
    model = mx.load_model(xml)
    model.reports.contact_forces = True
    data = mx.SceneData(model)
    pos_map, vel_map = dof_map(model, layout, data)
    pos, vel = data.dof_pos.copy(), data.dof_vel.copy()
    pos[pos_map], vel[vel_map] = q, v
    data.reset(model, pos, vel, forward_kinematic=True)
    bowl = model.get_link(bowl_body)
    before = float(bowl.get_position(data)[2])
    data.actuator_ctrls = ctrl.astype(np.float32)
    model.step(data)
    force = float(model.get_link_net_contact_forces(data, [bowl.index])[0, 2])
    model.forward_kinematic(data)
    return {"bowl_contact_force_z_n": force,
            "bowl_step_z_mm": 1000 * (float(bowl.get_position(data)[2]) - before)}


def baseline_gripper_error(actions, reference_ctrl, config, ranges):
    """Check the exact finger target stream produced by the existing MotrixSim OSC adapter."""
    substeps = round(1 / (20 * config["physics_dt"]))
    command = 0.
    close_signs = np.repeat(np.sign([row["action"][-1] for row in actions]), substeps)
    gripper = np.empty_like(reference_ctrl)
    for i, sign in enumerate(close_signs):
        command = min(1., max(-1., command - config["gripper_speed"] * sign))
        gripper[i] = ((1. + np.array([command, -command])) / 2.) * np.diff(ranges, axis=1)[:, 0] + ranges[:, 0]
    return float(np.max(np.abs(gripper - reference_ctrl)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Original MuJoCo task-07 directory")
    parser.add_argument("baseline", type=Path, help="Fixed-action MotrixSim task-07 directory using its own OSC")
    parser.add_argument("work", type=Path, help="Directory with native, matched-native and motrix captures")
    args = parser.parse_args()
    cfg = read(args.reference / "scene.json")
    substeps = round(1 / (20 * cfg["physics_dt"]))
    warmup = 10 * substeps
    xml = ET.parse(args.reference / "scene.xml")
    ranges = np.asarray([[float(x) for x in xml.find(f".//position[@name='{name}']").get("ctrlrange").split()]
                         for name in cfg["gripper_actuators"]])
    arm_vel = [read(args.work / "native/state-00/result.json")["joints"][joint][1][0]
               for joint in cfg["arm_joints"]]
    finger_vel = [read(args.work / "native/state-00/result.json")["joints"][joint][1][0]
                  for joint in cfg["gripper_joints"]]
    rows, phases = [], defaultdict(list)
    with TemporaryDirectory(prefix="mxf-burner-contact-") as scratch:
        disabled = Path(scratch) / "disabled.xml"
        for index in range(50):
            tag = f"state-{index:02}"
            n, m = (args.work / engine / tag for engine in ("native", "motrix"))
            native, motrix = read(n / "result.json"), read(m / "result.json")
            cold = read(args.work / "matched-native" / (tag + ".json"))
            initial = read(args.reference / tag / "initial-state.json")
            baseline = read(args.baseline / tag / "result.json")
            for field in ("initial_state_sha256", "fixtures_sha256"):
                if any(result[field] != initial[field] for result in (native, motrix, cold, baseline)):
                    raise ValueError(f"{tag}: state/fixture hash differs")
            if (not native["historical_success_flags_match"] or not native["success"]
                    or native["historical_eef_max_error_m"] != 0 or native["historical_object_max_error_m"] != 0):
                raise ValueError(f"{tag}: MuJoCo did not reproduce the original fixed-action rollout")
            with np.load(n / "physics.npz") as native_trace, np.load(m / "physics.npz") as motrix_trace:
                a, b = dict(native_trace), dict(motrix_trace)
            if (not np.array_equal(a["ctrl"].astype(np.float32), b["ctrl"])
                    or native["physics_steps"] != motrix["physics_steps"]):
                raise ValueError(f"{tag}: replay controls or physics steps differ")
            old = read(args.baseline / tag / "trace.json")
            finger_command_error = baseline_gripper_error(old, a["ctrl"][:, -2:], cfg, ranges)
            if finger_command_error > 1e-9:
                raise ValueError(f"{tag}: original OSC and actuator replay Gripper targets differ")
            eef = np.linalg.norm(a["eef"] - b["eef"], axis=1)
            finger_contacts = a["contacts"][:, :2] | b["contacts"][:, :2]
            touch = int(np.flatnonzero(finger_contacts.any(axis=1))[0])
            for left, right in zip(cold["probes"], motrix["probes"], strict=True):
                tick = left["tick"]
                pre_eef = np.linalg.norm(a["eef"][tick] - right["eef_before"])
                pre_obj = np.linalg.norm(a["objects"][tick] - right["objects_before"], axis=1).max()
                if (left["phase"] != right["phase"] or tick != right["tick"]
                        or pre_eef > 2e-6 or pre_obj > 2e-6):
                    raise ValueError(f"{tag}: one-step probe geometry or tick differs")
                qvel = np.asarray(right["qvel_after"]) - left["qvel_after"]
                bowl_dz = float(right["objects_after"][0][2] - left["objects_after"][0][2])
                phases[left["phase"]].append({"state": index, "tick": tick,
                    "arm_rms_rad_s": float(np.sqrt(np.mean(qvel[arm_vel]**2))),
                    "finger_rms_mm_s": float(1000 * np.sqrt(np.mean(qvel[finger_vel]**2))),
                    "native_bowl_external_force_z_n": left["bowl_external_force_z_n"],
                    "bowl_z_gap_mm": 1000 * bowl_dz,
                    "matching_contact_flags": left["contacts"] == right["contacts"]})
            template = m / "case.xml"
            tree = ET.parse(template)
            burner = tree.find(".//geom[@name='flat_stove_1_collision_burner']")
            burner.set("contype", "0")
            burner.set("conaffinity", "0")
            tree.write(disabled)
            state = [a[key][warmup] for key in ("qpos", "qvel", "ctrl")]
            burner_on, burner_off = (burner_step(path, native["joints"], *state, cfg["objects"][BOWL]["body"])
                                     for path in (template, disabled))
            rows.append({"state": index, "physics_steps": native["physics_steps"],
                         "osc_success": baseline["success"], "torque_success": motrix["success"],
                         "torque_lift_m": float(b["objects"][warmup:, 0, 2].max() - b["objects"][warmup, 0, 2]),
                         "torque_plate_contacts": bool(b["contacts"][:, 3].any()),
                         "osc_warmup_eef_error_mm": 1000 * old[9]["eef_error_m"],
                         "torque_warmup_eef_error_mm": 1000 * float(eef[warmup]),
                         "torque_first_contact_pre_eef_error_mm": 1000 * float(eef[max(0, touch - 1)]),
                         "torque_final_eef_error_mm": 1000 * float(eef[-1]),
                         "max_arm_torque_conversion_nm": float(np.abs(a["ctrl"][:, :7] - b["ctrl"][:, :7]).max()),
                         "baseline_gripper_max_diff_m": finger_command_error,
                         "burner_on": burner_on, "burner_off": burner_off})
    phase_summary = {phase: {**{field: float(np.median([row[field] for row in values])) for field in
                              ("arm_rms_rad_s", "finger_rms_mm_s", "bowl_z_gap_mm",
                               "native_bowl_external_force_z_n")},
                             "identical_contact_flags": sum(row["matching_contact_flags"] for row in values),
                             "probe_count": len(values)} for phase, values in phases.items()}
    aggregates = {field: float(np.median([row[field] for row in rows])) for field in
                  ("osc_warmup_eef_error_mm", "torque_warmup_eef_error_mm", "torque_first_contact_pre_eef_error_mm",
                   "torque_final_eef_error_mm")}
    totals = {"cases": len(rows), "physics_steps": sum(row["physics_steps"] for row in rows),
              "osc_successes": sum(row["osc_success"] for row in rows),
              "torque_successes": sum(row["torque_success"] for row in rows),
              "torque_bowl_lifts": sum(row["torque_lift_m"] >= .03 for row in rows),
              "torque_bowl_plate_contacts": sum(row["torque_plate_contacts"] for row in rows),
              "max_arm_torque_conversion_nm": max(row["max_arm_torque_conversion_nm"] for row in rows),
              "baseline_gripper_max_diff_m": max(row["baseline_gripper_max_diff_m"] for row in rows),
              "medians": aggregates,
              **{variant: {field: float(np.median([row[variant][field] for row in rows])) for field in
                            ("bowl_contact_force_z_n", "bowl_step_z_mm")} for variant in ("burner_on", "burner_off")},
              "matched_state_phases": phase_summary}
    write_json(args.work / "summary.json", {**totals, "episodes": rows, "matched_state_probes": phases})
    print(totals)


if __name__ == "__main__":
    main()
