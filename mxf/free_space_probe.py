"""Compare LIBERO free-space arm dynamics and OSC outputs without rendering."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

from .actuator_replay import indices, restore_fixtures
from .compare import read
from .eval import ENV_SEED, fixture_sha256, write_json
from .policy import DEFAULT_PLATFORM


def native_probe(env, config):
    import mujoco
    from robosuite.utils import transform_utils as T

    sim = env.sim
    controller = env.robots[0].controller
    m = sim.model._model
    d = sim.data._data
    controller.update(force=True)
    site = m.site(config["eef_site"]).id
    arm = [m.joint(n).dofadr.item() for n in config["arm_joints"]]
    robot_geoms = {i for i in range(m.ngeom)
                   if m.body(m.geom_bodyid[i]).name.startswith(("robot0_", "gripper0_"))}

    def contacts(data):
        return sum(c.geom1 in robot_geoms or c.geom2 in robot_geoms for c in data.contact)

    pose = d.site_xpos[site].copy()
    ori = d.site_xmat[site].copy().reshape(3, 3)
    goal_pos, goal_ori = controller.goal_pos.copy(), controller.goal_ori.copy()
    targets = {}
    for name, shift, angle in (("hold", [0, 0, 0], [0, 0, 0]),
                               ("translate", [.01, -.012, .008], [0, 0, 0]),
                               ("rotate", [0, 0, 0], [.06, -.04, .05]),
                               ("history", goal_pos - pose, None)):
        position = pose + shift
        rotation = goal_ori if angle is None else T.quat2mat(T.axisangle2quat(np.array(angle))) @ ori
        controller.set_goal(np.zeros(6), set_pos=position, set_ori=rotation)
        torque = controller.run_controller()
        targets[name] = {"position": position.tolist(), "quat": T.mat2quat(rotation).tolist(),
                         "torque": torque.tolist()}
    controller.goal_pos, controller.goal_ori = goal_pos, goal_ori
    controller.new_update = True

    result = {"qpos": d.qpos.tolist(), "qvel": d.qvel.tolist(), "pose": pose.tolist(),
              "targets": targets, "contacts": int(contacts(d)), "bias": controller.torque_compensation.tolist(),
              "mass": controller.mass_matrix.tolist(), "rest": controller.initial_joint.tolist(),
              "jac": controller.J_full.tolist(), "step": {}}
    for name, torque in (("zero", np.zeros(7)), ("osc", targets["translate"]["torque"]),
                         *((f"joint-{i+1}", t) for i, t in enumerate(2 * np.eye(7)))):
        fresh = mujoco.MjData(m)
        fresh.qpos[:], fresh.qvel[:] = d.qpos, d.qvel
        fresh.ctrl[:] = d.ctrl
        fresh.ctrl[:7] = np.clip(torque, *m.actuator_ctrlrange[:7].T)
        mujoco.mj_forward(m, fresh)
        mujoco.mj_step(m, fresh)
        mujoco.mj_forward(m, fresh)
        result["step"][name] = {"ctrl": fresh.ctrl.tolist(), "vel": fresh.qvel[arm].tolist(),
                                "finger_vel": fresh.qvel[[m.joint(n).dofadr.item() for n in config["gripper_joints"]]].tolist(),
                                "contacts": int(contacts(fresh))}
    return result


def native(args):
    from local_platform.libero_execution.suite import EnvironmentRNG
    from robosuite.utils.transform_utils import mat2quat
    from .native import fixtures, open_task, snapshot, task_catalog

    suite, bddl = task_catalog()
    for task_id in args.tasks:
        task = suite.get_task(task_id)
        states = suite.get_task_init_states(task_id)
        folder = args.reference / f"task-{task_id:02}"
        config = read(folder / "scene.json")
        rng = EnvironmentRNG(ENV_SEED)
        env = rng.call(open_task, bddl / task.problem_folder / task.bddl_file, render=False)
        try:
            rng.call(env.seed, ENV_SEED)
            for index in args.states:
                case = f"state-{index:02}"
                initial = read(folder / case / "initial-state.json")
                reference = read(folder / case / "trace.json")
                state = np.asarray(states[index], dtype="<f8")
                assert hashlib.sha256(state.tobytes()).hexdigest() == initial["initial_state_sha256"]
                assert fixture_sha256(initial["fixtures"]) == initial["fixtures_sha256"]
                rng.call(env.reset)
                restore_fixtures(env.sim.model, initial["fixtures"])
                rng.call(env.set_init_state, state.copy())
                assert snapshot(env) == initial["joints"] and fixtures(env) == initial["fixtures"]
                controller = env.robots[0].controller
                model = env.sim.model
                assert list(model.actuator_names) == config["arm_actuators"] + config["gripper_actuators"]
                layout = {n: [indices(model.get_joint_qpos_addr(n)), indices(model.get_joint_qvel_addr(n))]
                          for n in model.joint_names}
                result = {"initial_state_sha256": initial["initial_state_sha256"],
                          "fixtures_sha256": initial["fixtures_sha256"], "layout": layout,
                          "reset_goal_quat": mat2quat(controller.goal_ori).tolist(),
                          "frames": {}, "historical_eef_error_m": []}
                captures = {}
                data = env.sim.data._data
                for tick in range(13):
                    if tick in (0, 12):
                        captures[tick] = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(),
                                          controller.goal_pos.copy(), controller.goal_ori.copy())
                    if tick < 12:
                        obs, *_ = rng.call(env.step, np.array(reference[tick]["action"]))
                        if tick == 0:
                            result["first_goal_pos"] = controller.goal_pos.tolist()
                        error = np.linalg.norm(obs["robot0_eef_pos"] - reference[tick]["eef"])
                        result["historical_eef_error_m"].append(float(error))
                for tick, values in captures.items():
                    data.qpos[:], data.qvel[:], data.ctrl[:], controller.goal_pos, controller.goal_ori = values
                    result["frames"][str(tick)] = native_probe(env, config)
                dest = args.output / folder.name
                dest.mkdir(exist_ok=True)
                write_json(dest / (case + ".json"), result)
            print(folder.name, flush=True)
        finally:
            env.close()


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def measure_frame(env, maps, native, robot_colliders):
    import motrixsim as mx
    from .motrix import rotvec

    model = env.model
    pos_map, vel_map = maps
    pos = np.empty(pos_map.size, dtype=np.float32)
    vel = np.empty(vel_map.size, dtype=np.float32)
    pos[pos_map], vel[vel_map] = native["qpos"], native["qvel"]
    arm = [model.get_joint(n).dof_vel_index for n in env.config["arm_joints"]]
    fingers = [joint.dof_vel_index for joint in env.finger_joints]

    def contacts(data):
        return sum(c.collider_index_a in robot_colliders or c.collider_index_b in robot_colliders
                   for c in data.low.get_contacts())

    result = {"step": {}, "targets": {}, "contacts": 0}
    for name, ref in native["step"].items():
        data = mx.SceneData(model)
        data.reset(model, pos, vel, forward_kinematic=True)
        before = contacts(data)
        np.testing.assert_allclose(env.site.get_position(data), native["pose"], atol=3e-6, rtol=0)
        cmd = np.asarray(ref["ctrl"], dtype=np.float32)
        data.actuator_ctrls = cmd
        model.step(data)
        assert np.array_equal(cmd, data.actuator_ctrls)
        after = contacts(data)
        model.forward_kinematic(data)
        result["step"][name] = {"vel": data.dof_vel[arm].tolist(), "contacts": [before, after, contacts(data)],
                                "arm_vel_error_rms": rms(data.dof_vel[arm] - ref["vel"]),
                                "finger_vel_error_rms": rms(data.dof_vel[fingers] - ref["finger_vel"])}
    # Inertia and bias are produced at the input of model.step. Restore q and v
    # for the OSC query without advancing the simulation a second time.
    data.set_dof_pos(pos, model)
    data.set_dof_vel(vel)
    model.forward_kinematic(data)
    assert np.array_equal(data.dof_vel, vel) and np.allclose(data.dof_pos, pos, atol=2e-7, rtol=0)
    result["contacts"] = contacts(data)
    J = env.chain.get_end_effector_jac(data)[[3, 4, 5, 0, 1, 2]]
    result["jac_max_error"] = float(np.max(np.abs(J - native["jac"])))
    result["mass_max_error"] = float(np.max(np.abs(env.chain.get_inertia_matrix(data) - native["mass"])))
    result["bias_error_rms"] = rms(env.chain.get_bias_force(data) - native["bias"])
    for name, target in native["targets"].items():
        torque = env.solver.solve(env.chain, target["position"], rotvec(np.array(target["quat"])),
                                  native["rest"], data)
        result["targets"][name] = {"torque": torque.tolist(), "error_rms": rms(torque - target["torque"])}
    return result


def motrix(args):
    from .assets import prepare_scene
    from .motrix import MotrixEnv, rotvec, multiply
    from .stove_actuator_motrix import dof_map

    for task_id in args.tasks:
        folder = args.reference / f"task-{task_id:02}"
        dest = args.output / folder.name
        dest.mkdir(exist_ok=True)
        template = prepare_scene(folder, dest, inertia_groups=False)
        for index in args.states:
            name = f"state-{index:02}"
            case = dest / name
            case.mkdir(exist_ok=True)
            initial = read(folder / name / "initial-state.json")
            reference = read(args.native / folder.name / (name + ".json"))
            for key in ("fixtures_sha256", "initial_state_sha256"):
                assert reference[key] == initial[key]
            env = MotrixEnv(folder, template, case, initial["fixtures"], camera_enabled=False)
            try:
                env.set_init_state(initial["joints"], legacy=True)
                assert env.renderer is None
                names = [geom.attrib["name"] for body in ET.parse(template).iter("body")
                         if body.get("name", "").startswith(("robot0_", "gripper0_"))
                         for geom in body.findall("geom")]
                colliders = {c for n in names for c in (env.model.low.get_collider_indices(n) or [])}
                assert colliders
                maps = dof_map(env.model, reference["layout"], env.data)
                qinv = np.array(reference["reset_goal_quat"]) * [-1, -1, -1, 1]
                result = {"reset_goal_delta_rad": float(np.linalg.norm(rotvec(multiply(qinv, env.goal_quat)))),
                          "rest_delta_rad": rms(env.rest - reference["frames"]["0"]["rest"]), "frames": {}}
                for tick, frame in reference["frames"].items():
                    result["frames"][tick] = measure_frame(env, maps, frame, colliders)
                write_json(case / "result.json", result)
            finally:
                env.close()
        print(folder.name, flush=True)


def validate(args):
    from .assets import prepare_scene
    from .motrix import MotrixEnv

    for task_id in args.tasks:
        folder = args.reference / f"task-{task_id:02}"
        dest = args.output / folder.name
        dest.mkdir(exist_ok=True)
        template = prepare_scene(folder, dest, inertia_groups=False)
        assert [g.attrib for g in ET.parse(template).iter("geom")] == [
            g.attrib for g in ET.parse(folder / "scene.xml").iter("geom")]
        for index in args.states:
            name = f"state-{index:02}"
            initial = read(folder / name / "initial-state.json")
            trace = read(folder / name / "trace.json")[:12]
            case = dest / name
            case.mkdir(exist_ok=True)
            env = MotrixEnv(folder, template, case, initial["fixtures"], camera_enabled=False)
            try:
                if not initial.get("osc_reset") and not env.archived_panda:
                    raise ValueError("Legacy robot reset profile unavailable")
                result = {k: initial[k] for k in ("initial_state_sha256", "fixtures_sha256")}
                modes = ("legacy", "restored")
                if args.components:
                    modes += ("goals_only", "rest_only", "bias_only")
                for mode in modes:
                    env.set_init_state(initial["joints"], initial.get("osc_reset"), legacy=mode == "legacy")
                    if mode.endswith("_only"):
                        goal = env.goal_quat.copy()
                        cached = env.start_pose.copy()
                        rest = env.rest.copy()
                        bias = env.start_bias.copy()
                        env.set_init_state(initial["joints"], legacy=True)
                        if mode == "goals_only":
                            env.goal_quat, env.start_pose = goal, cached
                        elif mode == "rest_only":
                            env.rest = rest
                        else:
                            for motor, value in zip(env.arm, bias, strict=True):
                                motor.set_ctrl(env.data, float(np.clip(value, *motor.ctrl_range)))
                    errors = []
                    for row in trace:
                        obs, *_ = env.step(row["action"])
                        if mode == "goals_only":
                            env.start_pose = None
                        errors.append(float(1000 * np.linalg.norm(obs["robot0_eef_pos"] - row["eef"])))
                    result[mode] = {"eef_error_mm": errors}
                write_json(case / "validation.json", result)
            finally:
                env.close()
        print(folder.name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("native", "motrix", "validate"))
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--components", action="store_true", help="Ablate individual controller startup changes during validation")
    parser.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--states", type=int, nargs="+", default=[0, 10, 20, 30, 40])
    args = parser.parse_args()
    if args.engine == "motrix" and args.native is None:
        parser.error("motrix requires --native recordings")
    args.reference = args.reference.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.engine == "native":
        cache = Path(__file__).resolve().parents[1] / ".cache"
        for key, name in (("LIBERO_CONFIG_PATH", "libero"), ("NUMBA_CACHE_DIR", "numba"),
                          ("XDG_CACHE_HOME", ""), ("MPLCONFIGDIR", "matplotlib")):
            os.environ.setdefault(key, str(cache / name))
        sys.path[:0] = [str(DEFAULT_PLATFORM), str(DEFAULT_PLATFORM / ".runtimes/sources/libero-source")]
        native(args)
    elif args.engine == "motrix":
        motrix(args)
    else:
        validate(args)


if __name__ == "__main__":
    main()
