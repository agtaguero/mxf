"""Run paired LIBERO episodes and retain the inputs needed for replay."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import time

from .policy import DEFAULT_MANIFEST, DEFAULT_PLATFORM, policy_worker

WARMUP = [0., 0., 0., 0., 0., 0., -1.]
POLICY_SEED = 7
ENV_SEED = 0


SUITE_LIMITS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def fixture_sha256(layout):
    return hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()


def object_positions(obs):
    return {name: obs[name + "_pos"].tolist()
            for name in ("akita_black_bowl_1", "plate_1") if name + "_pos" in obs}


def wait_for_scene(path, timeout):
    deadline = time.monotonic() + timeout
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise FileNotFoundError(path)
        time.sleep(0.25)
    return path


def reference_jobs(args):
    from .native import task_catalog

    suite, bddl_root = task_catalog(args.suite)
    for index in args.tasks:
        task = suite.get_task(index)
        yield index, task.language, (bddl_root / task.problem_folder / task.bddl_file,
                                     suite.get_task_init_states(index))


def rollout(env, client, obs, max_steps, output, video, *, compact=False):
    import imageio.v2 as imageio
    import numpy as np
    from local_platform.vla.libero_suite import observation_input

    timing = {"environment_seconds": 0., "policy_rpc_seconds": 0.}
    trace, queries = [], []
    query_digest, action_digest = hashlib.sha256(), hashlib.sha256()
    forwards = 0
    video_writer = imageio.get_writer(output / "replay.mp4", fps=20) if video else nullcontext()
    with video_writer as writer:
        for i in range(10 + max_steps):
            if i < 10:
                action = WARMUP
            else:
                if writer is not None:
                    writer.append_data(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                prepared = observation_input(obs)
                if (i - 10) % 5 == 0:
                    forwards += 1
                    if compact:
                        for key in sorted(prepared):
                            query_digest.update(prepared[key].tobytes())
                    else:
                        queries.append({"step": i - 9, **prepared})
                started = time.perf_counter()
                action = client.call("get_action", prepared)
                timing["policy_rpc_seconds"] += time.perf_counter() - started
            bounded = np.asarray(action)
            if bounded.shape != (7,) or not np.isfinite(bounded).all() or np.any(np.abs(bounded) > 1.000001):
                raise ValueError("Invalid LIBERO action")
            action_digest.update(np.asarray(action, dtype="<f8").tobytes())
            started = time.perf_counter()
            obs, *_ = env.step(action)
            success = i >= 10 and bool(env.check_success())
            timing["environment_seconds"] += time.perf_counter() - started
            for field, duration in getattr(env, "step_timing", {}).items():
                timing[field] = timing.get(field, 0.) + duration
            if not compact:
                trace.append({"step": i - 9, "action": list(map(float, action)),
                          "eef": obs["robot0_eef_pos"].tolist(), "success": success,
                          "object_positions": object_positions(obs)})
            if success and writer is not None:
                writer.append_data(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            if success:
                break
    if not compact:
        write_json(output / "trace.json", trace)
    if queries:
        np.savez_compressed(output / "policy-inputs.npz", **{
            key: np.stack([query[key] for query in queries]) for key in queries[0]
        })
    return {"success": success, "steps": i - 9, "policy_forwards": forwards,
            "action_sha256": action_digest.hexdigest(),
            **({"query_sha256": query_digest.hexdigest()} if compact else {}),
            "termination": "success" if success else "horizon", **timing}


def run_reference(args):
    import numpy as np
    from local_platform.libero_execution.suite import EnvironmentRNG
    from .native import export_scene, fixtures, open_task, osc_reset, snapshot

    jobs = list(reference_jobs(args))
    instructions = {str(index): instruction for index, instruction, _ in jobs}
    rng = EnvironmentRNG(ENV_SEED)
    context = nullcontext((None, {})) if args.export_only else policy_worker(
        args.platform, args.manifest, instructions, POLICY_SEED, args.output, args.gpu
    )
    with context as (client, model):
        write_json(args.output / "run.json", {
            "engine": "mujoco", "suite": args.suite, "tasks": args.tasks,
            "mode": "export" if args.export_only else "closed_loop",
            "engine_version": importlib.metadata.version("mujoco"),
            "episodes": args.episodes, "max_steps": args.max_steps,
            "environment_seed": ENV_SEED, "policy_seed": POLICY_SEED, **model,
        })
        for task_position, (index, instruction, (bddl, initial_states)) in enumerate(jobs):
            task_output = args.output / f"task-{index:02}"
            task_output.mkdir()
            env = rng.call(open_task, bddl, render=not args.export_only, max_steps=args.max_steps)
            try:
                if task_position == 0:
                    rng.call(env.seed, ENV_SEED)
                if client:
                    client.call("select_task", str(index))
                for case in range(args.episodes):
                    case_output = task_output / f"state-{case:02}"
                    case_output.mkdir()
                    state = np.asarray(initial_states[case], dtype="<f8")
                    rng.call(env.reset)
                    obs = rng.call(env.set_init_state, state.copy())
                    if case == 0:
                        export_scene(env, task_output, instruction)
                    identity = hashlib.sha256(state.tobytes()).hexdigest()
                    layout = fixtures(env)
                    identities = {"initial_state_sha256": identity, "fixtures_sha256": fixture_sha256(layout)}
                    write_json(case_output / "initial-state.json", {
                        **identities, "joints": snapshot(env), "fixtures": layout, "osc_reset": osc_reset(env),
                    })
                    if client:
                        client.call("reset")
                        try:
                            with rng.active():
                                result = rollout(env, client, obs, args.max_steps,
                                                 case_output, not args.no_video, compact=args.compact)
                        except Exception as error:
                            write_json(case_output / "result.json", {
                                "termination": "error", "error": str(error),
                                **identities,
                            })
                            raise
                        write_json(case_output / "result.json",
                                   {**result, **identities})
                        print(f"Task {index} state {case}: {result['termination']} "
                              f"at step {result['steps']}", flush=True)
            finally:
                rng.call(env.close)


def run_motrix(args):
    from .assets import prepare_scene
    from .motrix import ASSET_WARMUP_SECONDS

    source_path = args.scenes / "run.json"
    source_suite = json.loads(source_path.read_text())["suite"] if source_path.exists() else "libero_spatial"
    if source_suite != args.suite:
        raise ValueError("Scene suite differs from requested suite")
    jobs = {str(index): args.scenes / f"task-{index:02}" for index in args.tasks}
    reference_policy = args.scenes / "policy.json"
    if reference_policy.exists():
        source_instructions = json.loads(reference_policy.read_text())["instructions"]
        instructions = {index: source_instructions[index] for index in jobs}
    else:
        instructions = {index: json.loads(wait_for_scene(
            scene / "scene.json", args.wait_for_scenes).read_text())["instruction"]
                        for index, scene in jobs.items()}
    context = nullcontext((None, {})) if args.replay else policy_worker(
        args.platform, args.manifest, instructions, POLICY_SEED, args.output, args.gpu,
        rng_source=args.policy_rng_source, episodes=args.episodes,
    )
    with context as (client, model):
        write_json(args.output / "run.json", {
            "engine": "motrixsim", "suite": args.suite, "tasks": args.tasks,
            "engine_version": importlib.metadata.version("motrixsim"),
            "camera_asset_warmup_min_seconds": ASSET_WARMUP_SECONDS,
            "inertia_groups": not args.legacy_inertia,
            "osc_reset": "legacy" if args.legacy_osc_reset else "libero",
            "motrix_solver_override": args.motrix_solver, "extend_stove_burner": args.extend_stove_burner,
            "mode": "reference_action_replay" if args.replay else "closed_loop",
            "physics_dt": 0.002, "control_hz": 20, "warmup_steps": 10, "compact": args.compact,
            "episodes": args.episodes, "max_steps": args.max_steps,
            "episodes_per_process": args.episodes_per_process,
            "environment_seed": ENV_SEED, "policy_seed": POLICY_SEED, **model,
        })
        if args.compact:
            write_json(args.output / "scene-digests.json", {
                index: {name: hashlib.sha256((scene / name).read_bytes()).hexdigest()
                        for name in ("scene.xml", "scene.json")}
                for index, scene in jobs.items()
            })
        for index, scene in jobs.items():
            metadata = json.loads(wait_for_scene(
                scene / "scene.json", args.wait_for_scenes).read_text())
            if "bddl_instruction" in metadata and metadata["instruction"] != instructions[index]:
                raise ValueError(f"Instruction changed in exported task {index}")
            task_output = args.output / scene.name
            task_output.mkdir()
            template = prepare_scene(scene, task_output, original_stove_burner=not args.extend_stove_burner,
                                     inertia_groups=not args.legacy_inertia)
            if client:
                client.call("select_task", index)
            for first in range(0, args.episodes, args.episodes_per_process):
                batch = range(first, min(first + args.episodes_per_process, args.episodes))
                run_episode_process(args, index, scene, task_output, template, batch, client)

        if client and args.policy_rng_source is None:
            write_json(args.output / "policy-summary.json", client.call("summary"))


def run_episode_process(args, index, scene, output, template, cases, client):
    """Release all native renderer allocations at each batch boundary."""
    import multiprocessing
    import random
    import numpy as np

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=episode_process,
                              args=(args, index, scene, output, template, cases, client,
                                    client.connection.gettimeout() if client is not None else None,
                                    random.getstate(), np.random.get_state(), sender))
    try:
        process.start()
        # Only the child uses the shared, already-connected policy socket until
        # it exits. There is one policy process and one sequence for the suite.
        sender.close()
        try:
            sequence, python_state, numpy_state = receiver.recv()
        except EOFError:
            process.join()
            raise RuntimeError(f"Simulation process failed for task {index}, cases {cases}, "
                               f"exit code {process.exitcode}") from None
        process.join()
        if process.exitcode:
            raise RuntimeError(f"Simulation process exited with {process.exitcode}")
        if client is not None:
            client.sequence = sequence
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    finally:
        receiver.close()
        sender.close()
        if process.pid is not None and process.is_alive():
            process.terminate()
            process.join()
        process.close()


def episode_process(args, index, scene, task_output, template, cases, client, timeout, python_state, numpy_state, sender):
    import random
    import numpy as np
    from .motrix import CameraStartupError, MotrixEnv

    if client is not None:
        # Spawn duplicates the descriptor, but not Python's socket timeout metadata.
        client.connection.settimeout(timeout)
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    for case in cases:
        case_output = task_output / f"state-{case:02}"
        case_output.mkdir()
        source = scene / case_output.name
        state = json.loads(wait_for_scene(
            source / "initial-state.json", args.wait_for_scenes).read_text())
        layout = state["fixtures"]
        identities = {"initial_state_sha256": state["initial_state_sha256"],
                      "fixtures_sha256": fixture_sha256(layout)}
        try:
            if identities["fixtures_sha256"] != state["fixtures_sha256"]:
                raise ValueError("Fixture layout differs from the original export")
            for attempt in range(3):
                env = None
                try:
                    env = MotrixEnv(scene, template, case_output, layout,
                                    camera_enabled=not (args.replay and args.no_video),
                                    physics_solver=args.motrix_solver)
                    obs = env.set_init_state(state["joints"], state.get("osc_reset"), legacy=args.legacy_osc_reset)
                except BaseException as error:
                    if env is not None:
                        env.close()
                    if not isinstance(error, CameraStartupError) or attempt == 2:
                        raise
                    print(f"Task {index} state {case}: retrying renderer before inference ({error})",
                          flush=True)
                    time.sleep(2 ** attempt)
                else:
                    break
            try:
                if client:
                    if args.policy_rng_source is None:
                        client.call("reset")
                    else:
                        client.call("reset", int(index), case)
                    result = rollout(env, client, obs, args.max_steps,
                                     case_output, not args.no_video, compact=args.compact)
                else:
                    result = replay_actions(env, source, case_output, not args.no_video)
            finally:
                env.close()
        except BaseException as error:
            write_json(case_output / "result.json",
                       {"termination": "error", "error": str(error), **identities})
            raise
        result["osc_reset_source"] = env.osc_reset_source
        if hasattr(env, "camera_asset_warmup_seconds"):
            result["camera_asset_warmup_seconds"] = env.camera_asset_warmup_seconds
            result["camera_startup_retries"] = attempt
        write_json(case_output / "result.json", {**result, **identities})
        print(f"Task {index} state {case}: {result['termination']} "
              f"at step {result['steps']}", flush=True)

    sender.send((client.sequence if client is not None else None,
                 random.getstate(), np.random.get_state()))
    sender.close()


def replay_actions(env, source, output, video):
    import imageio.v2 as imageio
    import numpy as np

    actions = json.loads((source / "trace.json").read_text())
    rows = []
    context = imageio.get_writer(output / "replay.mp4", fps=20) if video else nullcontext()
    with context as writer:
        for native in actions:
            obs, *_ = env.step(native["action"])
            success = native["step"] > 0 and env.check_success()
            eef = obs["robot0_eef_pos"]
            rows.append({**native, "eef": eef.tolist(), "success": success,
                         "object_positions": object_positions(obs),
                         "reference_eef": native["eef"],
                         "reference_success": native["success"],
                         "eef_error_m": float(np.linalg.norm(eef - native["eef"]))})
            if writer is not None and native["step"] >= 0:
                writer.append_data(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
    write_json(output / "trace.json", rows)
    return {"success": any(row["success"] for row in rows), "steps": len(rows) - 10,
            "termination": "replay_complete",
            "max_eef_error_m": max(row["eef_error_m"] for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", type=Path, default=DEFAULT_PLATFORM)
    parser.add_argument("--engine", choices=["mujoco", "motrixsim"], default="mujoco")
    parser.add_argument("--scenes", type=Path, help="Path to a native export or evaluation run")
    parser.add_argument("--wait-for-scenes", type=int, default=0,
                        help="Seconds to wait for each scene file during concurrent reference evaluation")
    parser.add_argument("--replay", action="store_true", help="Replay recorded reference actions")
    from .assets import add_burner_options
    add_burner_options(parser)
    parser.add_argument("--legacy-inertia", action="store_true", help="Reproduce the former all-group inertia inference")
    parser.add_argument("--legacy-osc-reset", action="store_true", help="Reproduce the original Motrix OSC startup")
    parser.add_argument("--motrix-solver", choices=("PGS", "Newton", "PGSIMPR"),
                        help="Override the MotrixSim solver for all selected tasks")
    parser.add_argument("--policy-rng-source", type=Path,
                        help="Restore each episode's π0.5 draw index from a prior MotrixSim run")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--tasks", type=int, nargs="+", default=[2])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episodes-per-process", type=int, default=3,
                        help="Recycle the MotrixSim process after this many episodes; the suite policy persists")
    parser.add_argument("--suite", choices=SUITE_LIMITS, default="libero_spatial")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--compact", action="store_true", help="Retain closed-loop outcome, input and action hashes without raw tensors")
    args = parser.parse_args()
    if args.compact and (args.replay or args.export_only):
        parser.error("--compact requires closed-loop evaluation")
    if args.engine == "motrixsim" and (args.scenes is None or args.export_only):
        parser.error("MotrixSim requires --scenes and supports closed loop or --replay")
    if args.engine != "motrixsim" and args.replay:
        parser.error("--replay requires --engine motrixsim")
    if args.engine != "motrixsim" and (args.original_stove_burner or args.extend_stove_burner or args.motrix_solver or args.legacy_osc_reset or args.legacy_inertia):
        parser.error("MotrixSim solver and geometry options require --engine motrixsim")
    if args.policy_rng_source is not None:
        if args.engine != "motrixsim" or args.replay:
            parser.error("--policy-rng-source requires a closed-loop MotrixSim evaluation")
        args.policy_rng_source = args.policy_rng_source.resolve()
    if args.max_steps is None:
        args.max_steps = SUITE_LIMITS[args.suite]
    if not 1 <= args.episodes <= 50 or not 1 <= args.max_steps <= SUITE_LIMITS[args.suite]:
        parser.error("Episode count or horizon outside suite protocol")
    if args.episodes_per_process < 1:
        parser.error("--episodes-per-process must be positive")
    if args.wait_for_scenes < 0:
        parser.error("--wait-for-scenes must be nonnegative")
    if not set(args.tasks) <= set(range(10)) or len(args.tasks) != len(set(args.tasks)):
        parser.error("Select distinct native task indices in 0–9")
    args.tasks.sort()
    args.platform = args.platform.resolve()
    args.manifest = (args.manifest or args.platform / DEFAULT_MANIFEST).resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    cache = Path(__file__).resolve().parents[1] / ".cache"
    for name, value in {
        "NUMBA_CACHE_DIR": str(cache / "numba"), "LIBERO_CONFIG_PATH": str(cache / "libero"),
        "XDG_CACHE_HOME": str(cache), "MPLCONFIGDIR": str(cache / "matplotlib"),
    }.items():
        os.environ.setdefault(name, value)
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(args.platform))
    import random
    import numpy as np
    random.seed(POLICY_SEED)
    np.random.seed(POLICY_SEED)
    if args.engine == "motrixsim":
        args.scenes = args.scenes.resolve()
        run_motrix(args)
        return
    source = args.platform / ".runtimes/sources/libero-source"
    sys.path.insert(0, str(source))
    config = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
    if not config.exists():
        root = source / "libero/libero"
        # JSON is valid YAML and avoids the upstream interactive first-run prompt.
        write_json(config, {
            "benchmark_root": str(root), "bddl_files": str(root / "bddl_files"),
            "init_states": str(root / "init_files"), "assets": str(root / "assets"),
        })
    run_reference(args)


if __name__ == "__main__":
    def interrupt(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    main()
