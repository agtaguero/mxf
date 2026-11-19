"""One-step restoring the same native state, separating finger servos and collision response."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import numpy as np

from .actuator_replay import BOWL
from .assets import apply_inertia_group_range
from .compare import read
from .contact_phases import confirmed_release, groups
from .eval import write_json
from .free_space_dynamics import variant_xml


def anchors(n, m, native, mx):
    tip = min(v for v in (n["tip"], m["tip"], len(native["bits"]) - 1) if v is not None)
    divergence = np.linalg.norm(native["bowl"] - mx["bowl"], axis=1) >= 0.001
    crossed = np.flatnonzero(divergence[: tip + 1])
    anchors = {"initial": 0, "settled": 249, "before_tip": max(tip - 1, 0), "earliest_tip": tip, **n}
    mismatch = np.flatnonzero((native["bits"] & 3) != (mx["bits"] & 3))
    if mismatch.size:
        anchors["first_finger_mismatch"] = int(mismatch[0])
        anchors["before_finger_mismatch"] = max(0, int(mismatch[0]) - 1)
    if len(crossed):
        anchors["first_bowl_mm"] = int(crossed[0])
        anchors["before_bowl_mm"] = max(0, int(crossed[0]) - 1)
    if n["grip"] is not None:
        anchors["holding"] = min(n["grip"] + 25, len(divergence) - 1)
    return {k: v for k, v in anchors.items() if v is not None}


def model_xml(path, config, fixtures, variant):
    tree = ET.fromstring(variant_xml(path, config, fixtures, "baseline"))
    if variant == "inertia":
        apply_inertia_group_range(tree)
    if variant == "motors":
        for a in tree.findall("actuator/position"):
            if a.get("name") in config["gripper_actuators"]:
                a.tag = "motor"
                a.attrib.pop("kp")
                a.set("ctrlrange", a.attrib["forcerange"])
    if variant in ("no_fingers", "no_bowl"):
        for g in tree.iter("geom"):
            n = g.get("name", "")
            finger = n.startswith("gripper0_finger") and n.endswith("collision")
            if (variant == "no_fingers" and finger) or (
                variant == "no_bowl" and n in config["objects"][BOWL]["contact_geoms"]
            ):
                g.set("contype", "0")
                g.set("conaffinity", "0")
    return ET.tostring(tree, encoding="unicode")


def native_batch(xml, config, info, trace, ticks, variant):
    import mujoco

    m = mujoco.MjModel.from_xml_string(xml)
    finger = [m.joint(n).dofadr.item() for n in config["gripper_joints"]]
    body = m.body(config["objects"][BOWL]["body"]).id
    bowl = info["layout"][BOWL + "_joint0"][1]
    gn = [m.geom(i).name for i in range(m.ngeom)]
    ids = [{m.geom(n).id for n in g} for g in groups(gn, config)]
    out = {}
    for phase, t in ticks.items():
        d = mujoco.MjData(m)
        d.qpos[:], d.qvel[:], d.ctrl[:] = trace["qpos"][t], trace["qvel"][t], trace["ctrl"][t]
        if variant == "motors":
            d.ctrl[-2:] = trace["force"][t]
        mujoco.mj_forward(m, d)
        mujoco.mj_step(m, d)
        mujoco.mj_rnePostConstraint(m, d)
        counts = [0, 0, 0]
        for c in d.contact:
            if c.efc_address < 0:
                continue
            peer = c.geom2 if c.geom1 in ids[2] else c.geom1 if c.geom2 in ids[2] else None
            for j, k in enumerate((0, 1, 3)):
                counts[j] += int(peer in ids[k])
        out[phase] = {
            "counts": counts,
            "finger_v": d.qvel[finger].tolist(),
            "bowl_v": d.qvel[bowl[:3]].tolist(),
            "bowl_force": d.cfrc_ext[body, 3:].tolist(),
            "motor_force": d.actuator_force[-2:].tolist(),
        }
    return out


def motrix_batch(path, config, info, trace, ticks, variant):
    import motrixsim as mx
    from .stove_actuator_motrix import dof_map

    m = mx.load_model(path)
    m.reports.contact_forces = True
    pm, vm = dof_map(m, info["layout"], mx.SceneData(m))
    fingers = [m.get_joint(n).dof_vel_index for n in config["gripper_joints"]]
    bowl = next(b for b in m.floating_bases if b.name == BOWL + "_joint0")
    link = m.get_link(config["objects"][BOWL]["body"])
    ids = [
        {c for n in g for c in (m.low.get_collider_indices(n) or [])} for g in groups(m.geom_names, config)
    ]
    out = {}
    for phase, t in ticks.items():
        d = mx.SceneData(m)
        pos, vel = d.dof_pos.copy(), d.dof_vel.copy()
        pos[pm], vel[vm] = trace["qpos"][t], trace["qvel"][t]
        d.reset(m, pos, vel, forward_kinematic=True)
        for b in m.floating_bases:
            vi = info["layout"][b.name][1]
            b.set_local_angular_velocity(d, trace["qvel"][t, vi[3:]])
            np.testing.assert_allclose(
                b.get_local_angular_velocity(d), trace["qvel"][t, vi[3:]], atol=5e-6, rtol=1e-5
            )
        m.forward_kinematic(d)
        np.testing.assert_allclose(link.get_position(d), trace["bowl"][t], atol=1e-6, rtol=0)
        np.testing.assert_allclose(
            m.get_site(config["eef_site"]).get_position(d), trace["eef"][t], atol=4e-6, rtol=0
        )
        ctrl = trace["ctrl"][t].copy()
        if variant == "motors":
            ctrl[-2:] = trace["force"][t]
        d.actuator_ctrls = ctrl.astype(np.float32)
        m.step(d)
        counts = [0, 0, 0]
        for c in d.low.get_contacts():
            a, b = c.collider_index_a, c.collider_index_b
            peer = a if b in ids[2] else b if a in ids[2] else None
            for j, k in enumerate((0, 1, 3)):
                counts[j] += int(peer in ids[k])
        out[phase] = {
            "counts": counts,
            "finger_v": d.dof_vel[fingers].tolist(),
            "bowl_v": bowl.get_global_linear_velocity(d).tolist(),
            "bowl_force": m.get_link_net_contact_forces(d, [link.index])[0, :3].tolist(),
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("engine", choices=("native", "motrix"))
    p.add_argument("reference", type=Path)
    p.add_argument("work", type=Path)
    p.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--states", type=int, nargs="+", help="Restrict to specific reference cases")
    p.add_argument("--held-out", action="store_true", help="Check five reserved states per task")
    args = p.parse_args()
    inputs = ("native-held-samples", "baseline-held") if args.held_out else ("native", "baseline")
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "scene.xml"
        for ti in args.tasks:
            task = f"task-{ti:02}"
            scene = args.reference / task
            config = read(scene / "scene.json")
            folder = args.work / ("held-probes" if args.held_out else "probes") / args.engine / task
            folder.mkdir(parents=True, exist_ok=True)
            for i in args.states or range(int(args.held_out), 50, 10):
                name = f"state-{i:02}"
                native, mx = (args.work / k / task / name for k in inputs)
                a, b = read(native / "result.json"), read(mx / "result.json")
                assert all(
                    a[k] == b[k]
                    for k in ("initial_state_sha256", "fixtures_sha256", "action_sha256", "ticks")
                )
                with np.load(native / "trace.npz") as f:
                    trace = dict(f)
                commands = np.repeat([r["action"][-1] for r in read(scene / name / "trace.json")], 25)
                a["events"]["release"] = confirmed_release(a["events"]["open"], trace["bits"], commands)
                with np.load(mx / "trace.npz") as f:
                    ticks = anchors(a["events"], b["events"], trace, f)
                fixtures = read(scene / name / "initial-state.json")["fixtures"]
                forces = [
                    next(
                        a
                        for a in ET.parse(scene / "scene.xml").findall("actuator/position")
                        if a.get("name") == n
                    )
                    for n in config["gripper_actuators"]
                ]
                kp = np.array([float(a.get("kp")) for a in forces])
                bounds = np.array([[float(v) for v in a.get("forcerange").split()] for a in forces])
                trace["force"] = np.clip(kp * (trace["ctrl"][:, -2:] - trace["finger"]), *bounds.T)
                results = {}
                variants = ["original", "motors", "no_fingers", "no_bowl"]
                if args.held_out:
                    variants = ["original", "motors", "no_fingers"]
                if args.engine == "motrix":
                    variants.append("inertia")
                for variant in variants:
                    src = scene / "scene.xml" if args.engine == "native" else mx / "case.xml"
                    xml = model_xml(src, config, fixtures, variant)
                    if args.engine == "native":
                        rows = native_batch(xml, config, a, trace, ticks, variant)
                        if variant == "original":
                            for k, row in rows.items():
                                np.testing.assert_allclose(
                                    row["motor_force"], trace["force"][ticks[k]], atol=1e-10
                                )
                    else:
                        path.write_text(xml)
                        rows = motrix_batch(path, config, a, trace, ticks, variant)
                    results[variant] = rows
                write_json(folder / (name + ".json"), {"ticks": ticks, "variants": results})
            print(task, flush=True)


if __name__ == "__main__":
    main()
