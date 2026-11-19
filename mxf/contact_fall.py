"""Separate medium drag and free-joint damping before the first finger contact."""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from .actuator_replay import BOWL
from .compare import read
from .eval import write_json
from .free_space_dynamics import variant_xml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--held-out", action="store_true")
    args = parser.parse_args()
    group = "held-probes" if args.held_out else "probes"
    native = "native-held-samples" if args.held_out else "native"
    result = {}
    for task in range(10):
        folder = f"task-{task:02}"
        scene = args.reference / folder
        cfg = read(scene / "scene.json")
        for index in range(int(args.held_out), 50, 10):
            name = f"state-{index:02}"
            case = Path(folder) / name
            initial = read(scene / name / "initial-state.json")
            probes = read(args.work / group / "motrix" / folder / (name + ".json"))
            if "first_bowl_mm" not in probes["ticks"]:
                continue
            tick = probes["ticks"]["first_bowl_mm"]
            m = mujoco.MjModel.from_xml_string(
                variant_xml(scene / "scene.xml", cfg, initial["fixtures"], "baseline")
            )
            velocity = m.joint(BOWL + "_joint0").dofadr.item()
            ix = slice(velocity, velocity + 3)
            density, viscosity = m.opt.density, m.opt.viscosity
            damping = m.dof_damping.copy()
            with np.load(args.work / native / case / "trace.npz") as ref:
                q, v, ctrl = (ref[key][tick] for key in ("qpos", "qvel", "ctrl"))
            out = {}
            mx = probes["variants"]["original"]["first_bowl_mm"]
            for var in ("original", "no_air", "no_free_damping", "neither"):
                m.opt.density = density if var in ("original", "no_free_damping") else 0
                m.opt.viscosity = viscosity if var in ("original", "no_free_damping") else 0
                m.dof_damping[:] = damping
                if var in ("no_free_damping", "neither"):
                    m.dof_damping[velocity : velocity + 6] = 0
                d = mujoco.MjData(m)
                d.qpos[:], d.qvel[:], d.ctrl[:] = q, v, ctrl
                mujoco.mj_forward(m, d)
                mujoco.mj_step(m, d)
                out[var] = {
                    "bowl_v_error_mm_s": float(1000 * np.linalg.norm(d.qvel[ix] - mx["bowl_v"])),
                    "passive_force_n": d.qfrc_passive[ix].tolist(),
                }
            result[str(case)] = {"tick": tick, "variants": out}
    write_json(args.work / ("fall-held.json" if args.held_out else "fall.json"), result)


if __name__ == "__main__":
    main()
