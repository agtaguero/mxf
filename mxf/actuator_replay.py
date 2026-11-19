"""Record and replay LIBERO task-07 actuator inputs at the physics timestep."""
import argparse
import hashlib
import itertools
import os
from pathlib import Path
import sys

import numpy as np

from .compare import read
from .eval import ENV_SEED, fixture_sha256, write_json
from .policy import DEFAULT_PLATFORM

BOWL = "akita_black_bowl_1"
PLATE = "plate_1"
CONTACTS = ("finger_bowl", "finger_stove", "bowl_stove", "bowl_plate")


def contact_pairs(names, config):
    indices = {name: index for index, name in enumerate(names) if name}
    fingers = [index for name, index in indices.items()
               if name.startswith("gripper0_finger") and name.endswith("_collision")]
    stove = [index for name, index in indices.items() if name.startswith("flat_stove_1_")]
    bowl, plate = ([indices[name] for name in config["objects"][obj]["contact_geoms"]]
                   for obj in (BOWL, PLATE))
    return [np.asarray(list(itertools.product(a, b)), dtype=np.uint32)
            for a, b in ((fingers, bowl), (fingers, stove), (bowl, stove), (bowl, plate))]


def indices(address):
    return list(range(*address)) if isinstance(address, tuple) else [int(address)]


def series(size, nq, nv, na):
    return {"qpos": np.empty((size + 1, nq)), "qvel": np.empty((size + 1, nv)),
            "eef": np.empty((size + 1, 3)), "objects": np.empty((size + 1, 2, 3)),
            "contacts": np.empty((size, len(CONTACTS)), dtype=bool),
            "ctrl": np.empty((size, na))}


def anchor_ticks(contact, warmup):
    first_robot = np.flatnonzero(contact[:, 0] | contact[:, 1])
    phases = {"initial": 0, "after_warmup": warmup}
    if first_robot.size:
        phases["before_finger_contact"] = max(0, int(first_robot[0]) - 1)
    for name, column in (("first_finger_bowl", 0), ("first_bowl_plate", 3)):
        active = np.flatnonzero(contact[:, column])
        if active.size:
            phases[name] = int(active[0])
    return phases


def native_contact_group(model, config):
    pairs = contact_pairs(model.geom_names, config)
    group = np.zeros((model.ngeom, model.ngeom, len(pairs)), dtype=bool)
    for i, pair in enumerate(pairs):
        group[pair[:, 0], pair[:, 1], i] = group[pair[:, 1], pair[:, 0], i] = True
    return group


def native_contacts(data, group):
    contact = data.contact
    active = (contact.efc_address >= 0) & (contact.geom1 >= 0) & (contact.geom2 >= 0)
    return group[contact.geom1[active], contact.geom2[active]].any(axis=0)


def restore_fixtures(model, fixtures):
    for name, pose in fixtures.items():
        index = model.body_name2id(name)
        model.body_pos[index] = pose["position"]
        model.body_quat[index] = pose["quaternion_wxyz"]


def record_case(env, state, initial, actions, config):
    from .native import fixtures, snapshot

    sim = env.sim
    model, data = sim.model, sim.data._data
    restore_fixtures(model, initial["fixtures"])
    env.set_init_state(state.copy())
    if snapshot(env) != initial["joints"] or fixtures(env) != initial["fixtures"]:
        raise ValueError("MuJoCo reset differs from the exported state or fixture poses")
    site = model.site_name2id(config["eef_site"])
    objects = [model.body_name2id(config["objects"][name]["body"]) for name in (BOWL, PLATE)]
    actuators = config["arm_actuators"] + config["gripper_actuators"]
    act = [model.actuator_name2id(name) for name in actuators]
    step_count = round(1 / (20 * config["physics_dt"])) * len(actions)
    values = series(step_count, model.nq, model.nv, len(act))
    values["actuator_force"] = np.empty_like(values["ctrl"])
    group = native_contact_group(model, config)
    offset = 0
    step = sim.step

    def frame(tick):
        values["qpos"][tick] = data.qpos
        values["qvel"][tick] = data.qvel
        values["eef"][tick] = data.site_xpos[site]
        values["objects"][tick] = data.xpos[objects]

    def capture(*args, **kwargs):
        nonlocal offset
        frame(offset)
        values["ctrl"][offset] = data.ctrl[act]
        values["contacts"][offset] = native_contacts(data, group)
        step(*args, **kwargs)
        values["actuator_force"][offset] = data.actuator_force[act]
        offset += 1

    sim.step = capture
    try:
        seen_eef, seen_objects, scored = [], [], []
        for action in actions:
            obs, *_ = env.step(np.asarray(action["action"]))
            seen_eef.append(obs["robot0_eef_pos"])
            seen_objects.append([obs[name + "_pos"] for name in (BOWL, PLATE)])
            scored.append(bool(env.check_success()) if action["step"] > 0 else False)
    finally:
        sim.step = step
    if offset != step_count:
        raise ValueError(f"Expected {step_count} MuJoCo physics steps, recorded {offset}")
    sim.forward()
    frame(offset)
    arm = len(config["arm_actuators"])
    if not np.array_equal(values["ctrl"][:, :arm], values["actuator_force"][:, :arm]):
        raise ValueError("MuJoCo arm motor force differs from its unit-gear torque command")
    pose_error = np.max(np.linalg.norm(np.asarray(seen_eef) - [row["eef"] for row in actions], axis=1))
    obj_error = np.max(np.linalg.norm(np.asarray(seen_objects) - [
        [row["object_positions"][name] for name in (BOWL, PLATE)] for row in actions
    ], axis=-1))
    layout = {name: [indices(model.get_joint_qpos_addr(name)), indices(model.get_joint_qvel_addr(name))]
              for name in model.joint_names}
    return values, {"actuators": actuators, "joints": layout, "success": any(scored),
                    "physics_steps": offset, "historical_eef_max_error_m": float(pose_error),
                    "historical_object_max_error_m": float(obj_error),
                    "historical_success_flags_match": scored == [row["success"] for row in actions]}


def record(args):
    sys.path.insert(0, str(args.platform.resolve() / ".runtimes/sources/libero-source"))
    from local_platform.libero_execution.suite import EnvironmentRNG
    from .native import open_task, task_catalog

    suite, bddl_root = task_catalog()
    task = suite.get_task(7)
    states = suite.get_task_init_states(7)
    config = read(args.reference / "scene.json")
    rng = EnvironmentRNG(ENV_SEED)
    env = rng.call(open_task, bddl_root / task.problem_folder / task.bddl_file, render=False)
    try:
        rng.call(env.seed, ENV_SEED)
        for index in range(args.episodes):
            name = f"state-{index:02}"
            source = args.reference / name
            initial = read(source / "initial-state.json")
            actions = read(source / "trace.json")
            state = np.asarray(states[index], dtype="<f8")
            if (hashlib.sha256(state.tobytes()).hexdigest() != initial["initial_state_sha256"]
                    or fixture_sha256(initial["fixtures"]) != initial["fixtures_sha256"]):
                raise ValueError(f"Initial state or fixture identity differs for {source}")
            rng.call(env.reset)
            values, info = rng.call(record_case, env, state, initial, actions, config)
            folder = args.output / name
            folder.mkdir()
            np.savez_compressed(folder / "physics.npz", **values)
            write_json(folder / "result.json", {**info,
                       **{key: initial[key] for key in ("initial_state_sha256", "fixtures_sha256")}})
            print(name, info["physics_steps"], info["success"],
                  info["historical_eef_max_error_m"], info["historical_object_max_error_m"], flush=True)
    finally:
        rng.call(env.close)


def matched_native(args):
    import mujoco
    from robosuite.utils.binding_utils import MjSim

    sim = MjSim(mujoco.MjModel.from_xml_path(str(args.reference / "scene.xml")))
    model = sim.model
    config = read(args.reference / "scene.json")
    group = native_contact_group(model, config)
    site = model.site_name2id(config["eef_site"])
    objects = [model.body_name2id(config["objects"][name]["body"]) for name in (BOWL, PLATE)]
    warmup = 10 * round(1 / (20 * config["physics_dt"]))
    for index in range(args.episodes):
        name = f"state-{index:02}"
        initial = read(args.reference / name / "initial-state.json")
        recorded = args.controls / name
        info = read(recorded / "result.json")
        act = [model.actuator_name2id(key) for key in info["actuators"]]
        restore_fixtures(model, initial["fixtures"])
        probes = []
        with np.load(recorded / "physics.npz") as source:
            for phase, tick in anchor_ticks(source["contacts"], warmup).items():
                data = mujoco.MjData(model._model)
                data.qpos[:] = source["qpos"][tick]
                data.qvel[:] = source["qvel"][tick]
                data.ctrl[act] = source["ctrl"][tick]
                mujoco.mj_forward(model._model, data)
                if (not np.allclose(data.site_xpos[site], source["eef"][tick], atol=1e-9, rtol=0.)
                        or not np.allclose(data.xpos[objects], source["objects"][tick], atol=1e-9, rtol=0.)):
                    raise ValueError(f"{name} {phase}: XML-based MuJoCo geometry differs")
                flags = native_contacts(data, group)
                mujoco.mj_rnePostConstraint(model._model, data)
                bowl_force_z = float(data.cfrc_ext[objects[0], 5])
                mujoco.mj_step(model._model, data)
                mujoco.mj_forward(model._model, data)
                probes.append({"phase": phase, "tick": tick, "contacts": flags.tolist(),
                               "bowl_external_force_z_n": bowl_force_z,
                               "qvel_after": data.qvel.tolist(), "eef_after": data.site_xpos[site].tolist(),
                               "objects_after": data.xpos[objects].tolist()})
        write_json(args.output / f"{name}.json", {"initial_state_sha256": info["initial_state_sha256"],
                   "fixtures_sha256": info["fixtures_sha256"], "probes": probes})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Original MuJoCo task-07 directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--platform", type=Path, default=DEFAULT_PLATFORM)
    parser.add_argument("--controls", type=Path,
                        help="Run fresh MuJoCo one-step probes from these recorded actuator traces")
    args = parser.parse_args()
    if not 1 <= args.episodes <= 50:
        parser.error("Expected 1–50 episodes")
    workspace = Path(__file__).resolve().parents[1]
    for key, name in (("LIBERO_CONFIG_PATH", "libero"), ("NUMBA_CACHE_DIR", "numba"),
                      ("XDG_CACHE_HOME", ""), ("MPLCONFIGDIR", "matplotlib")):
        os.environ.setdefault(key, str(workspace / ".cache" / name))
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(args.platform.resolve()))
    args.reference = args.reference.resolve()
    args.output.mkdir(parents=True, exist_ok=args.controls is not None)
    (record if args.controls is None else matched_native)(args)


if __name__ == "__main__":
    main()
