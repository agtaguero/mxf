"""Ablate passive forces in matched one-step probes on temporary model copies."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import numpy as np

from .compare import read
from .eval import write_json

VARIANTS = ("baseline", "air", "arm_friction", "arm_damping", "finger_friction", "finger_limits", "combined")


def variant_xml(path, config, fixtures, variant):
    tree = ET.parse(path)
    for a in tree.findall("asset/*"):
        if file := a.get("file"):
            a.set("file", str((path.parent / file).resolve()))
    for b in tree.iter("body"):
        if b.get("name") in fixtures:
            fixture = fixtures[b.get("name")]
            b.set("pos", " ".join(map(str, fixture["position"])))
            b.set("quat", " ".join(map(str, fixture["quaternion_wxyz"])))
    for j in tree.iter("joint"):
        if j.get("name") in config["arm_joints"]:
            if variant in ("arm_friction", "air_arm_friction", "combined"):
                j.set("frictionloss", "0")
            if variant in ("arm_damping", "combined"):
                j.set("damping", "0")
        if j.get("name") in config["gripper_joints"]:
            if variant in ("finger_friction", "combined"):
                j.set("frictionloss", "0")
            if variant in ("finger_limits", "combined"):
                j.set("limited", "false")
    if variant in ("air", "air_arm_friction", "density", "combined"):
        tree.find("option").set("density", "0")
    if variant in ("air", "air_arm_friction", "viscosity", "combined"):
        tree.find("option").set("viscosity", "0")
    return ET.tostring(tree.getroot(), encoding="unicode")


def native_step(xml, config, sample):
    import mujoco
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:], data.ctrl[:] = sample["qpos"], sample["qvel"], sample["step"]["osc"]["ctrl"]
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data)
    names = config["arm_joints"] + config["gripper_joints"]
    return data.qvel[[model.joint(n).dofadr.item() for n in names]].tolist()


def motrix_step(xml, config, sample, layout, temp):
    import motrixsim as mx
    from .stove_actuator_motrix import dof_map
    temp.write_text(xml)
    model = mx.load_model(temp)
    data = mx.SceneData(model)
    pm, vm = dof_map(model, layout, data)
    pos, vel = data.dof_pos.copy(), data.dof_vel.copy()
    pos[pm], vel[vm] = sample["qpos"], sample["qvel"]
    data.reset(model, pos, vel, forward_kinematic=True)
    data.actuator_ctrls = np.asarray(sample["step"]["osc"]["ctrl"], dtype=np.float32)
    model.step(data)
    names = config["arm_joints"] + config["gripper_joints"]
    return data.dof_vel[[model.get_joint(n).dof_vel_index for n in names]].tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("native", "motrix"))
    parser.add_argument("reference", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--tasks", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--air-details", action="store_true", help="Separate density, viscosity and residual joint friction")
    args = parser.parse_args()
    variants = ("density", "viscosity", "air_arm_friction") if args.air_details else VARIANTS
    with TemporaryDirectory() as scratch:
        for task in args.tasks:
            name = f"task-{task:02}"
            folder = args.reference / name
            config = read(folder / "scene.json")
            results = {}
            for recording in sorted((args.work / "native-matched" / name).glob("state-*.json")):
                ref = read(recording)
                fixtures = read(folder / recording.stem / "initial-state.json")["fixtures"]
                path = (folder / "scene.xml" if args.engine == "native" else
                        args.work / "motrix" / name / recording.stem / "case.xml")
                frame = ref["frames"]["12"]
                cases = {}
                for variant in variants:
                    xml = variant_xml(path, config, fixtures, variant)
                    cases[variant] = (native_step(xml, config, frame) if args.engine == "native" else
                                      motrix_step(xml, config, frame, ref["layout"], Path(scratch) / "case.xml"))
                results[recording.stem] = cases
            dest = args.work / ("air-details" if args.air_details else "ablations") / args.engine
            dest.mkdir(parents=True, exist_ok=True)
            write_json(dest / (name + ".json"), results)
            print(name, flush=True)


if __name__ == "__main__":
    main()
