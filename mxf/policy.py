"""Reuse sim platform's locked π0.5 runtime, adapter and local transport."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

DEFAULT_PLATFORM = Path(__file__).resolve().parents[2] / "sim-platform"
DEFAULT_MANIFEST = (
    ".local-data/validation/libero-vla-default-20261118/pi05-full/"
    "data/models/1cbaa6017e6d4c07/manifest.json"
)


def original_forward_offsets(source, tasks, episodes, model):
    """Recover the original OpenPI draw index at the start of each episode."""
    run = json.loads((source / "run.json").read_text())
    if (run["engine"] != "motrixsim" or run["suite"] != "libero_spatial"
            or run["mode"] != "closed_loop" or "policy_rng_source" in run
            or run["weights_sha256"] != model["weights_sha256"]
            or run["model_manifest_sha256"] != model["model_manifest_sha256"]
            or run["policy_seed"] != model["seed"]
            or not set(tasks) <= set(run["tasks"]) or episodes > run["episodes"]):
        raise ValueError("RNG source must be an original closed-loop run of the same policy")
    offsets, forwards = {}, 0
    for task in run["tasks"]:
        if task > max(tasks):
            break
        for case in range(run["episodes"]):
            folder = source / f"task-{task:02}" / f"state-{case:02}"
            result = json.loads((folder / "result.json").read_text())
            steps = result["steps"]
            if (result["termination"] not in ("success", "horizon")
                    or type(steps) is not int or not 1 <= steps <= run["max_steps"]):
                raise ValueError(f"Invalid original policy episode {folder}")
            if task in tasks and case < episodes:
                offsets[f"{task}:{case}"] = forwards
            forwards += (steps + 4) // 5
    return offsets


@contextmanager
def policy_worker(platform, manifest, instructions, seed, output, gpu, *, rng_source=None,
                  episodes=None):
    sys.path.insert(0, str(platform))
    from local_platform.native.policy_rpc import Client
    from local_platform.vla.artifacts import digest

    model = json.loads(manifest.read_text())
    config = {
        "schema": "libero-suite-policy.v1",
        "boundary": "libero-raw256-suite.v1",
        "vla_key": "pi05-libero",
        "manifest_path": str(manifest),
        "model_manifest_sha256": digest(model),
        "instructions": instructions,
        "seed": seed,
    }
    module = "local_platform.vla.libero_metric_worker"
    python_path = str(platform)
    identity_metadata = {"model_manifest_sha256": config["model_manifest_sha256"],
                         "weights_sha256": model["weights_sha256"]}
    if rng_source is not None:
        config["forward_offsets"] = original_forward_offsets(
            rng_source, [int(task) for task in instructions], episodes,
            {**identity_metadata, "seed": seed},
        )
        module = "mxf.policy_replay_worker"
        python_path = os.pathsep.join((str(Path(__file__).resolve().parents[1]), python_path))
        identity_metadata["policy_rng_source"] = str(rng_source)
    encoded = json.dumps(config).encode()
    identity = hashlib.sha256(encoded).hexdigest()
    (output / "policy.json").write_bytes(encoded)
    with tempfile.TemporaryDirectory(prefix="mxf-policy-") as temporary:
        socket = Path(temporary) / "policy.sock"
        ready = Path(temporary) / "ready.json"
        env = {
            **os.environ,
            "PYTHONPATH": python_path,
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": str(gpu),
        }
        argv = [
            str(platform / ".runtimes/envs/policy-pi05/bin/python"),
            "-m", module,
            "--config", str(output / "policy.json"), "--identity", identity,
            "--socket", str(socket), "--ready", str(ready),
        ]
        with (output / "policy.log").open("w") as log, subprocess.Popen(
            argv, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=platform
        ) as process:
            client = None
            try:
                deadline = time.monotonic() + float(os.environ.get("MXF_POLICY_STARTUP_SECONDS", "1200"))
                while not ready.exists():
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError(f"π0.5 worker did not start. See {log.name}")
                    time.sleep(0.1)
                client = Client(socket, identity)
                yield client, identity_metadata
            finally:
                if client is not None:
                    client.close()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
