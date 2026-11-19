"""Pair fixed-action Spatial contact events at the 2 ms physics clock."""

import argparse
import hashlib
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .actuator_replay import BOWL, PLATE, indices, restore_fixtures
from .compare import read
from .eval import ENV_SEED, WARMUP, fixture_sha256, write_json
from .policy import DEFAULT_PLATFORM


def groups(names, config):
    return [
        [n for n in names if (n or "").startswith(f"gripper0_finger{i}_") and n.endswith("collision")]
        for i in (1, 2)
    ] + [config["objects"][n]["contact_geoms"] for n in (BOWL, PLATE)]


def masks(ids, count):
    mask = np.zeros((count, count), dtype=np.uint8)
    for group, bit in ((0, 1), (1, 2), (3, 4)):
        mask[np.ix_(ids[group], ids[2])] = bit
        mask[np.ix_(ids[2], ids[group])] = bit
    return mask


def first(condition, start=0, consecutive=1):
    if start is None or len(condition) - start < consecutive:
        return None
    match = np.convolve(condition[start:].astype(int), np.ones(consecutive, dtype=int), "valid")
    hits = np.flatnonzero(match == consecutive)
    return int(hits[0] + start) if hits.size else None


def confirmed_release(opened, bits, commands):
    if opened is None:
        return None
    touch = (bits & 3) != 0
    for t in np.flatnonzero(touch[:-1] & ~touch[1:]) + 1:
        if (
            t >= opened
            and t + 5 <= len(bits)
            and not touch[t : t + 5].any()
            and (commands[t : t + 5] < 0).all()
        ):
            return int(t)
    return None


def terminal_support(bits):
    """Bowl–plate and finger occupancy on the common 2 ms sampling clock."""
    contact = (bits[-50:] & 4) != 0
    boundaries = np.flatnonzero(np.r_[True, contact, True])
    return {
        "strict": bool(contact[-5:].all()),
        "any10": bool(contact[-5:].any()),
        "hits10": int(contact[-5:].sum()),
        "duty100": float(contact.mean()),
        "gap100_ms": int((np.diff(boundaries).max() - 1) * 2),
        "finger_free100": bool(((bits[-50:] & 3) == 0).all()),
    }


def events(bits, bowl_z, action):
    fingers = bits & 3
    tip = first(fingers != 0)
    grip = first(fingers == 3, tip, 5)
    lift = first(bowl_z >= bowl_z[250] + 0.03, 250)
    opened = first(action < 0, grip)
    return {
        "tip": tip,
        "grip": grip,
        "lift": lift,
        "open": opened,
        "loss": first(fingers == 0, grip, 5),
        "release": confirmed_release(opened, bits, action),
        "plate": first((bits & 4) != 0, lift),
    }


def native_case(env, state, initial, trace, config):
    from .native import fixtures, snapshot

    sim = env.sim
    model, data = sim.model, sim.data._data
    restore_fixtures(model, initial["fixtures"])
    env.set_init_state(state.copy())
    assert snapshot(env) == initial["joints"] and fixtures(env) == initial["fixtures"]
    ids = [[model.geom_name2id(n) for n in g] for g in groups(model.geom_names, config)]
    mask = masks(ids, model.ngeom)
    layout = {
        n: [indices(model.get_joint_qpos_addr(n)), indices(model.get_joint_qvel_addr(n))]
        for n in model.joint_names
    }
    assert list(model.actuator_names) == config["arm_actuators"] + config["gripper_actuators"]
    site = model.site_name2id(config["eef_site"])
    size = len(trace) * 25
    q = np.empty((size, model.nq))
    v = np.empty((size, model.nv))
    ctrl = np.empty((size, model.nu))
    eef = np.empty((size, 3))
    bits = np.empty(size, dtype=np.uint8)
    tick = 0
    step = sim.step

    def observe():
        nonlocal tick
        q[tick], v[tick], ctrl[tick], eef[tick] = data.qpos, data.qvel, data.ctrl, data.site_xpos[site]
        step()
        c = data.contact
        active = c.efc_address >= 0
        bits[tick] = np.bitwise_or.reduce(mask[c.geom1[active], c.geom2[active]], initial=0)
        tick += 1

    hist_errors = []
    scored = []
    sim.step = observe
    try:
        for row in trace:
            obs, *_ = env.step(np.array(row["action"]))
            if "eef" not in row:
                continue
            errors = [np.linalg.norm(obs["robot0_eef_pos"] - row["eef"])]
            for obj in (BOWL, PLATE):
                errors.append(np.linalg.norm(obs[obj + "_pos"] - row["object_positions"][obj]))
            hist_errors.append(errors)
            scored.append(bool(row["step"] > 0 and env.check_success()))
    finally:
        sim.step = step
    assert tick == size
    finger = [layout[n][0][0] for n in config["gripper_joints"]]
    bowl = layout[BOWL + "_joint0"][0][:3]
    return (
        {
            "qpos": q,
            "qvel": v,
            "ctrl": ctrl,
            "bits": bits,
            "bowl": q[:, bowl],
            "eef": eef,
            "finger": q[:, finger],
            "historical_error_m": np.array(hist_errors),
            "historical_flag_match": np.array(scored) == [r["success"] for r in trace if "success" in r],
        },
        layout,
        any(scored),
    )


class StepRecorder:
    """Observe the normal SDK step without updating transforms on the live state."""

    def __init__(self, env, size, layout=None, physics_substeps=1):
        self.model = env.model
        self.physics_substeps = physics_substeps
        self.env = env
        gs = groups(env.model.geom_names, env.config)
        ids = [[c for n in g for c in (env.model.low.get_collider_indices(n) or [])] for g in gs]
        assert all(ids)
        self.bits_by_pair = {
            (a, b): bit
            for group, bit in ((0, 1), (1, 2), (3, 4))
            for x in ids[group]
            for y in ids[2]
            for a, b in ((x, y), (y, x))
        }
        self.fingers = [j.dof_pos_index for j in env.finger_joints]
        self.bowl = next(b for b in env.model.floating_bases if b.name == BOWL + "_joint0").dof_pos_indices[
            :3
        ]
        self.trace = {
            k: np.empty((size, n)) for k, n in (("ctrl", 9), ("eef", 3), ("bowl", 3), ("finger", 2))
        }
        self.trace["bits"] = np.zeros(size, dtype=np.uint8)
        self.layout = layout
        if layout is not None:
            from .stove_actuator_motrix import dof_map

            self.pm, self.vm = dof_map(env.model, layout, env.data)
            self.trace["qpos"] = np.empty((size, len(self.pm)))
            self.trace["qvel"] = np.empty((size, len(self.vm)))
            self.free_bases = [(base, layout[base.name][1]) for base in env.model.floating_bases]
        self.tick = 0

    def __getattr__(self, name):
        return getattr(self.model, name)

    def step(self, data):
        t = self.tick
        q = data.dof_pos
        if self.layout is not None:
            self.trace["qpos"][t] = q[self.pm]
            self.trace["qvel"][t] = data.dof_vel[self.vm]
            for base, v in self.free_bases:
                self.trace["qvel"][t, v[:3]] = base.get_global_linear_velocity(data)
                self.trace["qvel"][t, v[3:]] = base.get_local_angular_velocity(data)
        self.trace["finger"][t], self.trace["bowl"][t] = q[self.fingers], q[self.bowl]
        self.trace["eef"][t] = self.env.site.get_position(data)
        self.trace["ctrl"][t] = data.actuator_ctrls
        for _ in range(self.physics_substeps):
            self.model.step(data)
        bits = 0
        for c in data.low.get_contacts():
            bits |= self.bits_by_pair.get((c.collider_index_a, c.collider_index_b), 0)
        self.trace["bits"][t] = bits
        self.tick += 1


def motrix_case(scene, template, case, initial, trace, solver, layout=None, physics_substeps=1):
    from .motrix import MotrixEnv

    env = MotrixEnv(scene, template, case, initial["fixtures"], camera_enabled=False, physics_solver=solver)
    try:
        env.set_init_state(initial["joints"], initial.get("osc_reset"))
        assert env.osc_reset_source != "legacy" and env.renderer is None and env.substeps == 25
        recorder = StepRecorder(env, len(trace) * 25, layout, physics_substeps)
        env.model = recorder
        success = False
        for row in trace:
            env.step(row["action"])
            if "eef" in row and row["step"] > 0:
                success |= env.check_success()
        assert recorder.tick == len(trace) * 25
        return recorder.trace, layout, bool(success)
    finally:
        env.close()


def run(args):
    if args.engine == "native":
        sys.path[:0] = [str(DEFAULT_PLATFORM), str(DEFAULT_PLATFORM / ".runtimes/sources/libero-source")]
        from local_platform.libero_execution.suite import EnvironmentRNG

        from .native import open_task, task_catalog

        suite, bddl = task_catalog()
    for task in args.tasks:
        scene = args.reference / f"task-{task:02}"
        dest = args.output / scene.name
        dest.mkdir(exist_ok=True, parents=True)
        config = read(scene / "scene.json")
        assert config["physics_dt"] == 0.002
        env = None
        if args.engine == "native":
            spec = suite.get_task(task)
            states = suite.get_task_init_states(task)
            rng = EnvironmentRNG(ENV_SEED)
            env = rng.call(open_task, bddl / spec.problem_folder / spec.bddl_file, render=False)
            if args.release_tail:
                env.env.horizon += 20
            rng.call(env.seed, ENV_SEED)
        else:
            from .assets import apply_inertia_group_range, prepare_scene

            template = prepare_scene(scene, dest, inertia_groups=args.inertia_groups)
            expected = ET.parse(scene / "scene.xml")
            if args.inertia_groups:
                apply_inertia_group_range(expected)
            assert [g.attrib for g in ET.parse(template).iter("geom")] == [
                g.attrib for g in expected.iter("geom")
            ]
            if args.physics_substeps > 1:
                tree = ET.parse(template)
                tree.find("option").set("timestep", str(config["physics_dt"] / args.physics_substeps))
                template = dest / "substeps.xml"
                tree.write(template, encoding="utf-8")
        try:
            for index in args.states:
                name = f"state-{index:02}"
                case = dest / name
                case.mkdir(exist_ok=True)
                initial = read(scene / name / "initial-state.json")
                trace = read(scene / name / "trace.json")
                cutoff = len(trace) * 25
                if args.release_tail:
                    trace.extend({"action": WARMUP, "step": 0} for _ in range(20))
                actions = np.asarray([r["action"] for r in trace], dtype="<f8")
                ids = {k: initial[k] for k in ("initial_state_sha256", "fixtures_sha256")}
                assert fixture_sha256(initial["fixtures"]) == ids["fixtures_sha256"]
                action_hash = hashlib.sha256(actions.tobytes()).hexdigest()
                if args.resume and (case / "result.json").is_file():
                    old = read(case / "result.json")
                    if (
                        any(old.get(k) != v for k, v in ids.items())
                        or old["action_sha256"] != action_hash
                        or old["solver"] != args.solver
                        or old.get("inertia_groups", False) != args.inertia_groups
                        or old.get("physics_substeps", 1) != args.physics_substeps
                        or old.get("engine", "native" if old["layout"] is not None else "motrix")
                        != args.engine
                    ):
                        raise ValueError(f"Existing run uses different inputs at {case}")
                    with np.load(case / "trace.npz") as previous:
                        if not args.full_states or "qpos" in previous:
                            continue
                if env is not None:
                    state = np.asarray(states[index], dtype="<f8")
                    assert hashlib.sha256(state.tobytes()).hexdigest() == ids["initial_state_sha256"]
                    rng.call(env.reset)
                    values, layout, success = rng.call(native_case, env, state, initial, trace, config)
                else:
                    layout = None
                    if args.native_layouts:
                        manifest = read(args.native_layouts / scene.name / name / "result.json")
                        if any(manifest[k] != v for k, v in ids.items()):
                            raise ValueError("Reference layout source has a different state or fixture")
                        layout = manifest["layout"]
                    values, layout, success = motrix_case(
                        scene, template, case, initial, trace, args.solver, layout, args.physics_substeps
                    )
                event = events(values["bits"], values["bowl"][:, 2], np.repeat(actions[:, -1], 25))
                if index % 10 and not args.full_states:
                    for key in ("qpos", "qvel"):
                        values.pop(key, None)
                np.savez_compressed(case / "trace.npz", **values)
                write_json(
                    case / "result.json",
                    {
                        **ids,
                        "engine": args.engine,
                        "action_sha256": action_hash,
                        "ticks": len(values["bits"]),
                        "official_ticks": cutoff,
                        "events": event,
                        **(
                            {"terminal_support": terminal_support(values["bits"])}
                            if args.release_tail
                            else {}
                        ),
                        "success": success,
                        "layout": layout,
                        "solver": args.solver,
                        "physics_substeps": args.physics_substeps,
                        "inertia_groups": args.inertia_groups,
                    },
                )
                print(scene.name, name, success, event, flush=True)
        finally:
            if env is not None:
                env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("engine", choices=("native", "motrix"))
    p.add_argument("reference", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--states", type=int, nargs="+", default=list(range(0, 50, 10)))
    p.add_argument("--solver", choices=("Newton", "PGS"))
    p.add_argument(
        "--physics-substeps",
        type=int,
        choices=(1, 2),
        default=1,
        help="Motrix diagnostic only. Hold each 2 ms control for this many physics steps.",
    )
    p.add_argument("--inertia-groups", action="store_true", help="Use native MJCF inertia-group filtering")
    p.add_argument(
        "--release-tail", action="store_true", help="Append one second of fixed opening, outside the score"
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--native-layouts", type=Path, help="Capture Motrix states in the reference joint coordinate order"
    )
    p.add_argument("--full-states", action="store_true")
    args = p.parse_args()
    if args.engine == "native" and args.physics_substeps != 1:
        p.error("Reference runs use the native timestep")
    if args.engine == "motrix" and args.full_states and args.native_layouts is None:
        p.error("Full Motrix state capture requires --native-layouts")
    cache = Path(__file__).resolve().parents[1] / ".cache"
    for k, n in (
        ("LIBERO_CONFIG_PATH", "libero"),
        ("NUMBA_CACHE_DIR", "numba"),
        ("XDG_CACHE_HOME", ""),
        ("MPLCONFIGDIR", "matplotlib"),
    ):
        os.environ.setdefault(k, str(cache / n))
    run(args)


if __name__ == "__main__":
    main()
