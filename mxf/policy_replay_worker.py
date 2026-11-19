"""Serve π0.5 with the original OpenPI draw index restored per episode."""
import argparse
import hashlib
import json
from pathlib import Path

from local_platform.native.policy_rpc import serve
from local_platform.vla.libero_metric_worker import create_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "socket", "ready"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--identity", required=True)
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if len(raw) > 4 * 1024**2 or hashlib.sha256(raw).hexdigest() != args.identity:
        raise ValueError("π0.5 replay config identity differs")
    config = json.loads(raw)
    offsets = config.pop("forward_offsets")
    if config["vla_key"] != "pi05-libero" or not offsets or any(
            type(value) is not int or not 0 <= value <= 22000 for value in offsets.values()):
        raise ValueError("Expected π0.5 source forward offsets")
    policy = create_policy(config)
    adapter = policy.policy.native
    jax = adapter._jax

    @jax.jit
    def advance(key, draws):
        return jax.lax.fori_loop(0, draws, lambda _, prior: jax.random.split(prior)[0], key)

    with jax.default_device(adapter._device):
        key = jax.random.key(config["seed"])
        previous = 0
        keys = {}
        for name, offset in sorted(offsets.items(), key=lambda item: item[1]):
            key = advance(key, offset - previous)
            keys[name] = key
            previous = offset

    def reset(task, case):
        if policy.instruction is None or policy.instruction != config["instructions"][str(task)]:
            raise ValueError("RNG restore requested for a different instruction")
        policy.reset()
        adapter._policy._rng = keys[f"{task}:{case}"]

    def ready():
        temporary = args.ready.with_suffix(".tmp")
        temporary.write_text(json.dumps({"identity": args.identity}))
        temporary.replace(args.ready)

    serve(args.socket, args.identity, {"select_task": policy.select_task, "reset": reset,
                                      "get_action": policy.get_action, "summary": policy.summary},
          ready=ready)


if __name__ == "__main__":
    main()
