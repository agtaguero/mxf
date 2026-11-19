"""Trace contact and grasp events during headless task-07 fixed-action replay."""
import argparse
from itertools import product
from pathlib import Path

import numpy as np

from .assets import add_burner_options, prepare_scene
from .compare import read
from .eval import WARMUP, fixture_sha256, write_json
from .motrix import MotrixEnv

BOWL = "akita_black_bowl_1"
PLATE = "plate_1"
SETTLE_STEPS = 20


def trace_case(scene, template, source, output, *, legacy=False):
    initial = read(source / "initial-state.json")
    reference = read(source / "trace.json")
    if not reference[-1]["success"] or fixture_sha256(initial["fixtures"]) != initial["fixtures_sha256"]:
        raise ValueError(f"Missing successful reference actions or matched fixtures at {source}")
    env = MotrixEnv(scene, template, output, initial["fixtures"], camera_enabled=False)
    try:
        env.set_init_state(initial["joints"], initial.get("osc_reset"), legacy=legacy)
        model = env.model
        bowl_geoms = [model.get_geom_index(name) for name in env.config["objects"][BOWL]["contact_geoms"]]
        names = model.geom_names
        finger_geoms = [index for index, name in enumerate(names)
                        if name and name.startswith("gripper0_finger") and name.endswith("_collision")]
        stove_geoms = [index for index, name in enumerate(names) if (name or "").startswith("flat_stove_1_")]
        plate_geoms = [model.get_geom_index(name) for name in env.config["objects"][PLATE]["contact_geoms"]]
        # Test the explicit pairs, as the goal predicate does. MotrixSim 0.10.1's
        # contacting_geoms() returned empty lists even for known active pairs.
        bowl_pairs = np.asarray(list(product(bowl_geoms, finger_geoms + stove_geoms + plate_geoms)),
                                dtype=np.uint32)
        stove_pairs = np.asarray(list(product(finger_geoms, stove_geoms)), dtype=np.uint32)
        rows = []
        steps = [(row["step"], row["action"], row["eef"]) for row in reference]
        steps.extend((reference[-1]["step"] + extra, WARMUP, None)
                     for extra in range(1, SETTLE_STEPS + 1))
        for step, action, reference_eef in steps:
            obs, *_ = env.step(action)
            query = model.get_contact_query(env.data)
            bowl_contacts = {names[index] for index in bowl_pairs[query.is_colliding(bowl_pairs), 1]}
            finger_to_stove = {names[index] for index in stove_pairs[query.is_colliding(stove_pairs), 1]}
            bowl, plate = obs[BOWL + "_pos"], obs[PLATE + "_pos"]
            rows.append({
                "step": step, "reference_window": reference_eef is not None,
                "eef_error_m": (float(np.linalg.norm(obs["robot0_eef_pos"] - reference_eef))
                                if reference_eef is not None else None),
                "bowl_z_m": float(bowl[2]),
                "bowl_plate_horizontal_distance_m": float(np.linalg.norm(bowl[:2] - plate[:2])),
                "bowl_plate_height_m": float(bowl[2] - plate[2]),
                "gripper_qpos": obs["robot0_gripper_qpos"].tolist(),
                "bowl_contacts": sorted(bowl_contacts), "finger_stove_contacts": sorted(finger_to_stove),
                "success": step > 0 and bool(env.check_success()),
            })
        return {"initial_state_sha256": initial["initial_state_sha256"],
                "fixtures_sha256": initial["fixtures_sha256"], "steps": rows}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Native LIBERO task-07 directory")
    parser.add_argument("--output", type=Path, required=True)
    add_burner_options(parser)
    parser.add_argument("--legacy-osc-reset", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    scene = args.reference.resolve()
    template = prepare_scene(scene, args.output, original_stove_burner=not args.extend_stove_burner,
                             inertia_groups=False)
    for source in sorted(scene.glob("state-*")):
        output = args.output / source.name
        output.mkdir()
        result = trace_case(scene, template, source, output, legacy=args.legacy_osc_reset)
        write_json(output / "contacts.json", result)
        print(source.name, flush=True)


if __name__ == "__main__":
    main()
