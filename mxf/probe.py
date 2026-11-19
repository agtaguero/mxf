"""Verify exported LIBERO poses, OSC warmup, and optionally both cameras."""
import argparse
import json
from pathlib import Path

import numpy as np

from .eval import WARMUP, write_json
from .assets import prepare_scene
from .motrix import CAMERAS, MotrixEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-cameras", action="store_true", help="Verify and save both warmup frames")
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    results = []
    for scene in sorted(args.scenes.glob("task-*")):
        output = args.output / scene.name
        output.mkdir()
        state = json.loads((scene / "state-00/initial-state.json").read_text())
        env = MotrixEnv(scene, prepare_scene(scene, output), output, state["fixtures"],
                        camera_enabled=args.check_cameras)
        try:
            expected = state["joints"]
            first = env.set_init_state(expected, state.get("osc_reset"))
            for base in env.model.floating_bases:
                position = np.asarray(expected[base.name]["qpos"])
                np.testing.assert_allclose(base.get_dof_pos(env.data),
                                           position[[0, 1, 2, 4, 5, 6, 3]], atol=1e-6)
            for joint in env.model.joints:
                np.testing.assert_allclose(joint.get_dof_pos(env.data),
                                           expected[joint.name]["qpos"], atol=1e-6)
            for _ in range(10):
                last, *_ = env.step(WARMUP)
            last_position = last["robot0_eef_pos"]
            if not np.isfinite(last_position).all() or env.check_success():
                raise ValueError(f"Invalid warmup state for {scene.name}")
            drift = float(np.linalg.norm(last_position - first["robot0_eef_pos"]))
            if drift > 0.02:
                raise ValueError(f"More than 2 cm of end-effector drift during warmup: {scene.name}")
            row = {"task": scene.name, "initial_state_sha256": state["initial_state_sha256"],
                   "warmup_eef_drift_m": drift,
                   "contacts": int(env.model.get_contact_query(env.data).num_contacts)}
            if args.check_cameras:
                from PIL import Image

                frames = []
                for camera in CAMERAS:
                    rgb = last[camera + "_image"]
                    if rgb.dtype != np.uint8 or rgb.shape != (256, 256, 3) or np.ptp(rgb) < 32:
                        raise ValueError(f"{scene.name}: invalid RGB frame from {camera}")
                    path = output / (camera + ".png")
                    Image.fromarray(np.ascontiguousarray(rgb[::-1, ::-1])).save(path)
                    frames.append(str(path.relative_to(args.output)))
                row.update(camera_frames=frames,
                           camera_asset_warmup_seconds=env.camera_asset_warmup_seconds)
            results.append(row)
            print(scene.name, "OK", flush=True)
        finally:
            env.close()
    if not results:
        raise ValueError("The source did not contain exported task bundles")
    write_json(args.output / "probe.json", results)


if __name__ == "__main__":
    main()
