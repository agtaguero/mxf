"""Replay identical pre-impact states and controls in both engines without rendering."""

import argparse
from pathlib import Path

import numpy as np

from .actuator_replay import BOWL, PLATE
from .compare import read
from .contact_phases import groups, masks
from .eval import write_json
from .free_space_dynamics import variant_xml


def rotate(q, v):
    """Rotate v by unit quaternions in wxyz order; leading dimensions broadcast."""
    xyz = q[..., 1:]
    u = 2 * np.cross(xyz, v)
    return v + q[..., :1] * u + np.cross(xyz, u)


def product(q, r):
    return np.concatenate(
        (
            q[..., :1] * r[..., :1] - (q[..., 1:] * r[..., 1:]).sum(-1, keepdims=True),
            q[..., :1] * r[..., 1:] + r[..., :1] * q[..., 1:] + np.cross(q[..., 1:], r[..., 1:]),
        ),
        axis=-1,
    )


def relative(q, v, layout):
    bp, bv = layout[BOWL + "_joint0"]
    pp, pv = layout[PLATE + "_joint0"]
    bowl, plate = q[..., bp], q[..., pp]
    inv = plate[..., 3:] * [1, -1, -1, -1]
    offset = bowl[..., :3] - plate[..., :3]
    wp = rotate(plate[..., 3:], v[..., pv[3:]])
    wb = rotate(bowl[..., 3:], v[..., bv[3:]])
    return (
        rotate(inv, offset),
        product(inv, bowl[..., 3:]),
        rotate(inv, v[..., bv[:3]] - v[..., pv[:3]] - np.cross(wp, offset)),
        rotate(inv, wb - wp),
    )


def anchors(meta, other, bits):
    # Earliest post-lift plate contact, retaining never-arrivals at the paired comparison clock.
    hit = meta["events"]["plate"]
    tick = hit if hit is not None else other["events"]["plate"]
    if tick is None:
        tick = meta["official_ticks"]
    return {"entry": max(0, tick - 1), "support": len(bits) - 100}


def describes(xyz, rot, vel, angular, bits):
    contact = (bits & 4) != 0
    last = xyz[-50:]
    return {
        "end_contact_5": bool(contact[-5:].all()),
        "end_contact_any_10ms": bool(contact[-5:].any()),
        "end_duty_100ms": float(contact[-50:].mean()),
        "ever": bool(contact.any()),
        "end_finger": bool((bits[-5:] & 3).any()),
        "end_offset_mm": (1000 * xyz[-1]).tolist(),
        "end_range_mm": (1000 * np.ptp(last, axis=0)).tolist(),
        "end_tangent_speed_mm_s": float(1000 * np.median(np.linalg.norm(vel[-50:, :2], axis=1))),
        "end_normal_speed_mm_s": float(1000 * np.median(np.abs(vel[-50:, 2]))),
        "entry_offset_mm": (xyz[0] * 1000).tolist(),
        "entry_rel_quat_wxyz": rot[0].tolist(),
        "entry_rel_v_m_s": vel[0].tolist(),
        "entry_rel_omega_rad_s": angular[0].tolist(),
        "entry_tilt_deg": float(
            np.degrees(np.arccos(np.clip(rotate(rot[0], np.array([0.0, 0.0, 1.0]))[2], -1, 1)))
        ),
        "entry_tangent_speed_mm_s": float(1000 * np.linalg.norm(vel[0, :2])),
    }


class Native:
    def __init__(self, scene, initial, cfg):
        import mujoco

        self.api = mujoco
        xml = variant_xml(scene / "scene.xml", cfg, initial["fixtures"], "baseline")
        self.model = m = mujoco.MjModel.from_xml_string(xml)
        assert [m.actuator(i).name for i in range(m.nu)] == cfg["arm_actuators"] + cfg["gripper_actuators"]
        names = [m.geom(i).name for i in range(m.ngeom)]
        ids = [[m.geom(n).id for n in g] for g in groups(names, cfg)]
        self.mask = masks(ids, m.ngeom)

    def reset(self, q, v):
        self.d = self.api.MjData(self.model)
        self.d.qpos[:], self.d.qvel[:] = q, v
        self.api.mj_forward(self.model, self.d)

    def state(self):
        return self.d.qpos, self.d.qvel

    def step(self, ctrl):
        d = self.d
        d.ctrl[:] = ctrl
        self.api.mj_step(self.model, d)
        c = d.contact
        ix = c.efc_address >= 0
        return np.bitwise_or.reduce(self.mask[c.geom1[ix], c.geom2[ix]], initial=0)


class Motrix:
    def __init__(self, path, cfg, layout, mode):
        import motrixsim as mx

        from .stove_actuator_motrix import dof_map

        self.mx = mx
        self.substeps = 2 if mode == "halfstep" else 1
        if self.substeps == 2:
            import tempfile
            import xml.etree.ElementTree as ET

            tree = ET.parse(path)
            tree.find("option").set("timestep", str(cfg["physics_dt"] / self.substeps))
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".xml") as xml:
                tree.write(xml, encoding="utf-8")
                xml.flush()
                self.model = m = mx.load_model(xml.name)
        else:
            self.model = m = mx.load_model(path)
        assert list(m.actuator_names) == cfg["arm_actuators"] + cfg["gripper_actuators"]
        self.pm, self.vm = dof_map(m, layout, mx.SceneData(m))
        self.bases = [(b, layout[b.name][1]) for b in m.floating_bases]
        ids = [
            [c for n in g for c in (m.low.get_collider_indices(n) or [])] for g in groups(m.geom_names, cfg)
        ]
        self.bowl_colliders, self.plate_colliders = set(ids[2]), set(ids[3])
        self.bowl_link = m.get_link(cfg["objects"][BOWL]["body"])
        self.mask = {
            (a, b): bit
            for i, bit in ((0, 1), (1, 2), (3, 4))
            for x in ids[i]
            for y in ids[2]
            for a, b in ((x, y), (y, x))
        }

    def reset(self, q, v):
        self.d = d = self.mx.SceneData(self.model)
        qp, vt = d.dof_pos.copy(), d.dof_vel.copy()
        qp[self.pm], vt[self.vm] = q, v
        d.reset(self.model, qp, vt, forward_kinematic=True)
        for b, ind in self.bases:
            b.set_global_linear_velocity(d, v[ind[:3]])
            b.set_local_angular_velocity(d, v[ind[3:]])
        self.model.forward_kinematic(d)

    def state(self):
        q = self.d.dof_pos[self.pm]
        v = self.d.dof_vel[self.vm]
        for b, i in self.bases:
            v[i[:3]] = b.get_global_linear_velocity(self.d)
            v[i[3:]] = b.get_local_angular_velocity(self.d)
        return q, v

    def step(self, ctrl):
        self.d.actuator_ctrls = ctrl.astype(np.float32)
        for _ in range(self.substeps):
            self.model.step(self.d)
        # Sample the final physical substep, identical sampling clock for every mode.
        bit = 0
        for c in self.d.low.get_contacts():
            bit |= self.mask.get((c.collider_index_a, c.collider_index_b), 0)
        return bit

    def support_forces(self, trace, t):
        """Measure world vertical force at each solver step and flag other supports."""
        self.model.reports.contact_forces = True
        self.reset(trace["qpos"][t], trace["qvel"][t])
        samples = []
        for control in trace["ctrl"][t:]:
            self.d.actuator_ctrls = control.astype(np.float32)
            for _ in range(self.substeps):
                self.model.step(self.d)
                peers = set()
                for c in self.d.low.get_contacts():
                    a, b = c.collider_index_a, c.collider_index_b
                    if a in self.bowl_colliders:
                        peers.add(b)
                    if b in self.bowl_colliders:
                        peers.add(a)
                fz = self.model.get_link_net_contact_forces(self.d, [self.bowl_link.index])[0, 2]
                samples.append((bool(peers & self.plate_colliders), bool(peers - self.plate_colliders), fz))
        return np.asarray(samples)


def replay(engine, trace, t, layout):
    q, v, ctrl = trace["qpos"], trace["qvel"], trace["ctrl"]
    engine.reset(q[t], v[t])
    qp, vt = engine.state()
    np.testing.assert_allclose(qp, q[t], atol=2e-6, rtol=0)
    np.testing.assert_allclose(vt, v[t], atol=3e-5, rtol=2e-6)
    g = np.empty_like(q[t:])
    u = np.empty_like(v[t:])
    bits = np.empty(len(g), dtype=np.uint8)
    for i, act in enumerate(ctrl[t:]):
        g[i], u[i] = engine.state()
        bits[i] = engine.step(act)
    return (*relative(g, u, layout), bits)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("engine", choices=("native", "motrix", "halfstep"))
    p.add_argument("--reference", type=Path, default=Path("runs/spatial-campaign-v2/spatial/mujoco"))
    p.add_argument("--work", type=Path, default=Path("runs/plate-support"))
    p.add_argument("--tasks", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--states", nargs="+", type=int, default=list(range(0, 50, 10)))
    p.add_argument(
        "--common-control",
        action="store_true",
        help="Start both origins at the same clock and apply the reference actuators to both",
    )
    p.add_argument(
        "--support-forces",
        action="store_true",
        help="Measure Motrix bowl contact forces at every solver step from support checkpoints",
    )
    args = p.parse_args()
    if args.support_forces and args.engine == "native":
        p.error("Force-report sampling uses the Motrix SDK")
    if args.common_control and args.support_forces:
        p.error("Select common controls or force sampling for each invocation")
    for task in args.tasks:
        folder = f"task-{task:02}"
        scene = args.reference / folder
        cfg = read(scene / "scene.json")
        experiment = "forces" if args.support_forces else "common-control" if args.common_control else "cross"
        dest = args.work / experiment / args.engine / folder
        dest.mkdir(parents=True, exist_ok=True)
        for index in args.states:
            name = f"state-{index:02}"
            case = Path(folder) / name
            ini = read(scene / name / "initial-state.json")
            metas = {e: read(args.work / e / case / "result.json") for e in ("native", "motrix")}
            layout = metas["native"]["layout"]
            assert layout == metas["motrix"]["layout"]
            for k in ("fixtures_sha256", "initial_state_sha256", "action_sha256", "official_ticks", "ticks"):
                assert metas["native"][k] == metas["motrix"][k]
            model = (
                Native(scene, ini, cfg)
                if args.engine == "native"
                else Motrix(args.work / "motrix" / case / "case.xml", cfg, layout, args.engine)
            )
            result = {}
            if args.common_control:
                with np.load(args.work / "native" / case / "trace.npz") as reference:
                    common_ctrl = reference["ctrl"]
                hits = [m["events"]["plate"] for m in metas.values() if m["events"]["plate"] is not None]
                first_entry = min(hits, default=metas["native"]["official_ticks"])
            for src, other in (("native", "motrix"), ("motrix", "native")):
                with np.load(args.work / src / case / "trace.npz") as f:
                    trace = {k: f[k] for k in ("qpos", "qvel", "bits", "ctrl")}
                ticks = anchors(metas[src], metas[other], trace["bits"])
                if args.common_control:
                    trace["ctrl"] = common_ctrl
                    ticks["entry"] = max(0, first_entry - 1)
                if args.support_forces:
                    samples = model.support_forces(trace, ticks["support"])
                    np.savez_compressed(
                        dest / f"{name}-{src}.npz",
                        plate=samples[:, 0].astype(bool),
                        other=samples[:, 1].astype(bool),
                        world_fz_N=samples[:, 2],
                    )
                    continue
                result[src] = {}
                for phase, t in ticks.items():
                    arr = replay(model, trace, t, layout)
                    row = describes(*arr)
                    row["tick"] = t
                    row["has_entry"] = metas[src]["events"]["plate"] is not None
                    result[src][phase] = row
                    np.savez_compressed(
                        dest / f"{name}-{src}-{phase}.npz",
                        pos=arr[0],
                        quat=arr[1],
                        vel=arr[2],
                        omega=arr[3],
                        bits=arr[4],
                    )
            if not args.support_forces:
                write_json(dest / (name + ".json"), result)
        print(folder, flush=True)


if __name__ == "__main__":
    main()
