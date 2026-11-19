"""Measure burner contact normals and one-step forces at the saved MuJoCo pose."""
import argparse
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import numpy as np

from .actuator_replay import BOWL
from .assets import STOVE_BURNER, extend_stove_burner
from .compare import read
from .eval import write_json

def normal_counts(contacts):
    return dict(Counter("up" if row["normal_z"] > .5 else "down" if row["normal_z"] < -.5 else "side"
                        for row in contacts))


def native_model(reference):
    import mujoco

    return mujoco.MjModel.from_xml_path(str(reference / "scene.xml"))


def probe_native(model, fixture_poses, state, bowl_names, bowl_body):
    import mujoco

    for name, pose in fixture_poses.items():
        model.body(name).pos[:] = pose["position"]
        model.body(name).quat[:] = pose["quaternion_wxyz"]
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:], data.ctrl[:] = state
    mujoco.mj_forward(model, data)
    mujoco.mj_rnePostConstraint(model, data)
    ids = {model.geom(name).id: name for name in bowl_names}
    burner = model.geom(STOVE_BURNER).id
    contacts = []
    for c in data.contact:
        if c.geom1 == burner and c.geom2 in ids:
            contacts.append({"bowl_geom": ids[c.geom2], "normal_z": float(c.frame[2]), "distance_mm": float(c.dist * 1000)})
        elif c.geom2 == burner and c.geom1 in ids:
            contacts.append({"bowl_geom": ids[c.geom1], "normal_z": float(-c.frame[2]), "distance_mm": float(c.dist * 1000)})
    geometry = {}
    for name in (STOVE_BURNER, *bowl_names):
        geom = model.geom(name)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, data.geom_xmat[geom.id])
        geometry[name] = {"center_m": data.geom_xpos[geom.id].tolist(),
                          "quat_xyzw": quat[[1, 2, 3, 0]].tolist(), "half_extents_m": geom.size.tolist()}
    return {"geometry": geometry,
            "normals": normal_counts(contacts), "contacts": contacts,
            "bowl_force_z_n": float(data.cfrc_ext[model.body(bowl_body).id, 5])}


def probe_motrix(model, layout, state, bowl_names, bowl_body, *, bowl_offset_mm=0.):
    import motrixsim as mx
    from .stove_actuator_motrix import dof_map

    data = mx.SceneData(model)
    pos_map, vel_map = dof_map(model, layout, data)
    pos, vel = data.dof_pos.copy(), data.dof_vel.copy()
    pos[pos_map], vel[vel_map] = state[:2]
    pos[pos_map[layout[BOWL + "_joint0"][0][2]]] += bowl_offset_mm / 1000
    data.reset(model, pos, vel, forward_kinematic=True)
    geometry = {}
    for name in (STOVE_BURNER, *bowl_names):
        geom = model.get_geom(name)
        pose = geom.get_pose(data)
        geometry[name] = {"center_m": pose[:3].tolist(), "quat_xyzw": pose[3:].tolist(),
                          "half_extents_m": geom.half_extents.tolist()}
    bowl = model.get_link(bowl_body)
    before = float(bowl.get_position(data)[2])
    burner = set(model.low.get_collider_indices(STOVE_BURNER))
    ids = {index: name for name in bowl_names for index in model.low.get_collider_indices(name)}

    def read_contacts():
        contacts = []
        for c in data.low.get_contacts():
            if c.collider_index_a in burner and c.collider_index_b in ids:
                contacts.append({"bowl_geom": ids[c.collider_index_b], "normal_z": c.normal[2], "distance_mm": 1000 * c.depth})
            elif c.collider_index_b in burner and c.collider_index_a in ids:
                contacts.append({"bowl_geom": ids[c.collider_index_a], "normal_z": -c.normal[2], "distance_mm": 1000 * c.depth})
        return contacts

    contacts_before = read_contacts()
    data.actuator_ctrls = state[2].astype(np.float32)
    model.step(data)
    contacts = read_contacts()
    force = float(model.get_link_net_contact_forces(data, [bowl.index])[0, 2])
    model.forward_kinematic(data)
    return {"geometry": geometry, "normals_before_step": normal_counts(contacts_before),
            "contacts_before_step": contacts_before, "normals": normal_counts(contacts), "contacts": contacts,
            "solver_iterations": model.options.max_iterations, "bowl_force_z_n": force,
            "bowl_step_z_mm": 1000 * (float(bowl.get_position(data)[2]) - before)}


def motrix_variants(case, modified, state, layout, config, solver_extras):
    import motrixsim as mx

    parts = config["contact_geoms"], config["body"]
    model = mx.load_model(case)
    model.reports.contact_forces = True
    variants = {"original": probe_motrix(model, layout, state, *parts)}
    if solver_extras:
        variants["raise_bowl_0.5mm"] = probe_motrix(model, layout, state, *parts, bowl_offset_mm=.5)
        for iterations in (4, 256):
            model.options.max_iterations = iterations
            variants[f"iterations_{iterations}"] = probe_motrix(model, layout, state, *parts)
    for mm in (.5, 1., 2.):
        tree = ET.parse(case)
        if not extend_stove_burner(tree, mm / 1000):
            raise ValueError(f"{case}: expected the original 1 mm burner collider")
        tree.write(modified)
        model = mx.load_model(modified)
        model.reports.contact_forces = True
        variants[f"extend_{mm:g}mm"] = probe_motrix(model, layout, state, *parts)
    return variants


def motrix_solvers(case, state, layout, config):
    from motrixsim import msd

    world = msd.from_str(case.read_text(), format="mjcf", file_path=str(case.resolve()))
    options = world.simulate_option
    variants = {}
    for solver in ("PGS", "PGSIMPR", "Newton"):
        options.solver_type = getattr(msd.SolverType, solver)
        world.simulate_option = options
        model = msd.build(world)
        model.reports.contact_forces = True
        variants[solver] = probe_motrix(model, layout, state, config["contact_geoms"], config["body"])
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("mujoco", "motrixsim"))
    parser.add_argument("reference", type=Path, help="Original MuJoCo task-07 directory")
    parser.add_argument("recordings", type=Path, help="Per-step actuator experiment output")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-solvers", action="store_true", help="Sweep MotrixSim backends on the original box")
    args = parser.parse_args()
    if args.compare_solvers and args.engine != "motrixsim":
        parser.error("--compare-solvers requires motrixsim")
    config = read(args.reference / "scene.json")
    warmup = 10 * round(1 / (20 * config["physics_dt"]))
    bowl = config["objects"][BOWL]
    model = native_model(args.reference) if args.engine == "mujoco" else None
    args.output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="mxf-burner-normal-") as temporary:
        modified = Path(temporary) / "burner.xml"
        for index in range(50):
            name = f"state-{index:02}"
            recording = args.recordings / "native" / name
            meta = read(recording / "result.json")
            initial = read(args.reference / name / "initial-state.json")
            if any(meta[key] != initial[key] for key in ("initial_state_sha256", "fixtures_sha256")):
                raise ValueError(f"{name}: source and recording identities differ")
            with np.load(recording / "physics.npz") as values:
                state = tuple(values[key][warmup] for key in ("qpos", "qvel", "ctrl"))
            if model is not None:
                variants = {"original": probe_native(model, initial["fixtures"], state, bowl["contact_geoms"], bowl["body"])}
            else:
                case = args.recordings / "motrix" / name / "case.xml"
                if args.compare_solvers:
                    variants = motrix_solvers(case, state, meta["joints"], bowl)
                else:
                    variants = motrix_variants(case, modified, state, meta["joints"], bowl, solver_extras=index == 0)
            write_json(args.output / f"{name}.json", {"initial_state_sha256": meta["initial_state_sha256"],
                      "fixtures_sha256": meta["fixtures_sha256"], "tick": warmup, "variants": variants})


if __name__ == "__main__":
    main()
