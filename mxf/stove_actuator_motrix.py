"""Play the recorded actuator stream into MotrixSim without its OSC controller."""
import argparse
from pathlib import Path

import motrixsim as mx
import numpy as np

from .actuator_replay import BOWL, PLATE, anchor_ticks, contact_pairs, series
from .assets import add_burner_options, prepare_scene
from .compare import read
from .eval import fixture_sha256, write_json
from .motrix import MotrixEnv


def dof_map(model, layout, data):
    """Map MuJoCo coordinate order to MotrixSim's named DOFs and xyzw quaternions."""
    pos_map = np.full(data.dof_pos.size, -1, dtype=int)
    vel_map = np.full(data.dof_vel.size, -1, dtype=int)
    floating = {base.name: base for base in model.floating_bases}
    if set(layout) != set(floating) | set(model.joint_names):
        raise ValueError("Native and MotrixSim joint layouts differ")
    for name, (qpos, qvel) in layout.items():
        if name in floating:
            base = floating[name]
            pos_map[qpos] = np.asarray(base.dof_pos_indices)[[0, 1, 2, 6, 3, 4, 5]]
            vel_map[qvel] = base.dof_vel_indices
        else:
            joint = model.get_joint(name)
            pos_map[qpos] = np.arange(joint.dof_pos_index, joint.dof_pos_index + joint.num_dof_pos)
            vel_map[qvel] = np.arange(joint.dof_vel_index, joint.dof_vel_index + joint.num_dof_vel)
    if (not np.array_equal(np.sort(pos_map), np.arange(pos_map.size))
            or not np.array_equal(np.sort(vel_map), np.arange(vel_map.size))):
        raise ValueError("Native and MotrixSim DOF maps must be bijective")
    return pos_map, vel_map


def replay_case(scene, template, output, initial, native, info):
    env = MotrixEnv(scene, template, output, initial["fixtures"], camera_enabled=False)
    try:
        env.set_init_state(initial["joints"], legacy=True)  # All actuator commands are externally supplied.
        model, data = env.model, env.data
        if model.actuator_names != info["actuators"]:
            raise ValueError("Native and MotrixSim actuator ordering differs")
        pos_map, vel_map = dof_map(model, info["joints"], data)
        links = [env.goal_links[name] for name in (BOWL, PLATE)]
        pairs = contact_pairs(model.geom_names, env.config)
        starts = np.cumsum([0] + [len(pair) for pair in pairs[:-1]])
        all_pairs = np.concatenate(pairs)
        controls = native["ctrl"].astype(np.float32)
        values = series(len(controls), pos_map.size, vel_map.size, controls.shape[1])

        def contact_flags(state):
            return np.logical_or.reduceat(model.get_contact_query(state).is_colliding(all_pairs), starts)

        def frame(tick):
            values["qpos"][tick] = data.dof_pos[pos_map]
            values["qvel"][tick] = data.dof_vel[vel_map]
            values["eef"][tick] = env.site.get_position(data)
            values["objects"][tick] = [link.get_position(data) for link in links]

        frame(0)
        if (not np.allclose(values["qpos"][0], native["qpos"][0], atol=1e-6, rtol=0.)
                or not np.allclose(values["qvel"][0], native["qvel"][0], atol=1e-6, rtol=0.)):
            raise ValueError("MotrixSim started from a different physical state")
        warmup = 10 * env.substeps
        success = False
        for tick, act in enumerate(controls):
            data.actuator_ctrls = act
            model.step(data)
            values["ctrl"][tick] = data.actuator_ctrls
            values["contacts"][tick] = contact_flags(data)
            model.forward_kinematic(data)
            frame(tick + 1)
            if tick + 1 > warmup and (tick + 1) % env.substeps == 0:
                success |= env.check_success()
        if (not np.array_equal(values["ctrl"], controls)
                or not all(np.isfinite(value).all() for value in values.values())):
            raise ValueError("MotrixSim controls or physical state differ from expected stream")

        probes = []
        for phase, tick in anchor_ticks(native["contacts"], warmup).items():
            fresh = mx.SceneData(model)
            pos, vel = np.empty(pos_map.size, dtype=np.float32), np.empty(vel_map.size, dtype=np.float32)
            pos[pos_map], vel[vel_map] = native["qpos"][tick], native["qvel"][tick]
            fresh.reset(model, pos, vel, forward_kinematic=True)
            pre_eef = env.site.get_position(fresh)
            pre_objects = np.asarray([link.get_position(fresh) for link in links])
            fresh.actuator_ctrls = controls[tick]
            model.step(fresh)
            flags = contact_flags(fresh)
            model.forward_kinematic(fresh)
            probes.append({"phase": phase, "tick": tick, "contacts": flags.tolist(),
                           "eef_before": pre_eef.tolist(), "objects_before": pre_objects.tolist(),
                           "qvel_after": fresh.dof_vel[vel_map].tolist(),
                           "eef_after": env.site.get_position(fresh).tolist(),
                           "objects_after": [link.get_position(fresh).tolist() for link in links]})
        return values, {"success": bool(success), "physics_steps": len(controls), "probes": probes}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Original MuJoCo task-07 directory")
    parser.add_argument("controls", type=Path, help="Native per-physics-step actuator recording")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    add_burner_options(parser)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 50:
        parser.error("Expected 1–50 episodes")
    scene = args.reference.resolve()
    args.output.mkdir(parents=True)
    template = prepare_scene(scene, args.output, original_stove_burner=not args.extend_stove_burner,
                             inertia_groups=False)
    for index in range(args.episodes):
        name = f"state-{index:02}"
        initial = read(scene / name / "initial-state.json")
        info = read(args.controls / name / "result.json")
        ids = {key: initial[key] for key in ("initial_state_sha256", "fixtures_sha256")}
        if (any(info[key] != value for key, value in ids.items())
                or fixture_sha256(initial["fixtures"]) != ids["fixtures_sha256"]):
            raise ValueError(f"{name}: input traces and scene state differ")
        folder = args.output / name
        folder.mkdir()
        with np.load(args.controls / name / "physics.npz") as source:
            values, result = replay_case(scene, template, folder, initial, dict(source), info)
        np.savez_compressed(folder / "physics.npz", **values)
        write_json(folder / "result.json", {**result, **ids})
        print(name, result["physics_steps"], result["success"], flush=True)


if __name__ == "__main__":
    main()
