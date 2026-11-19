"""MJCF inertia-box air drag evaluated on Motrix link states."""

import numpy as np


def compiled_fluid(model):
    """Snapshot native principal frames, including inferred and fused mass properties."""
    if not (model.opt.density or model.opt.viscosity):
        return None
    if np.any(model.geom_fluid[:, 0]):
        raise ValueError("Ellipsoid fluid interaction requires a separate force law")
    import mujoco

    links = {}
    for i in range(1, model.nbody):
        if model.body_weldid[i] == 0 or model.body_mass[i] <= 1e-15:
            continue
        axes = np.empty(9)
        mujoco.mju_quat2Mat(axes, model.body_iquat[i])
        inertia = model.body_inertia[i]
        sides = np.sqrt(6 * np.maximum(1e-15, inertia.sum() - 2 * inertia) / model.body_mass[i])
        links[model.body(i).name] = [model.body_ipos[i].tolist(), axes.reshape(3, 3).tolist(), sides.tolist()]
    return {
        "links": links,
        "density": model.opt.density,
        "viscosity": model.opt.viscosity,
        "wind": model.opt.wind.tolist(),
    }


class FluidForces:
    def __init__(self, model, spec):
        self.model = model
        self.links = [model.get_link(name) for name in spec["links"]]
        self.indices = [link.index for link in self.links]
        self.centers, self.axes, sides = (np.array(v) for v in zip(*spec["links"].values()))
        diameter = sides.mean(axis=1, keepdims=True)
        a, b = np.roll(sides, 1, axis=1), np.roll(sides, -1, axis=1)
        self.force_drag = -0.5 * spec["density"] * a * b
        self.torque_drag = -spec["density"] * sides * (a**4 + b**4) / 64
        self.force_visc = -3 * np.pi * spec["viscosity"] * diameter
        self.torque_visc = -np.pi * spec["viscosity"] * diameter**3
        self.wind = np.asarray(spec["wind"])

    def apply(self, data):
        rotation = self.model.get_link_rotation_mats(data, self.indices)
        vel = self.model.get_link_velocities(data, self.indices)
        inertia_frame = rotation @ self.axes
        com_offset = (rotation @ self.centers[..., None])[..., 0]
        linear = vel[:, :3] + np.cross(vel[:, 3:], com_offset) - self.wind
        to_local = inertia_frame.transpose(0, 2, 1)
        local_v = (to_local @ linear[..., None])[..., 0]
        local_w = (to_local @ vel[:, 3:, None])[..., 0]
        force = self.force_visc * local_v + self.force_drag * np.abs(local_v) * local_v
        torque = self.torque_visc * local_w + self.torque_drag * np.abs(local_w) * local_w
        local_f = (self.axes @ force[..., None])[..., 0].astype(np.float32)
        local_t = (self.axes @ torque[..., None])[..., 0].astype(np.float32)
        for link, point, force, torque in zip(self.links, self.centers, local_f, local_t):
            link.add_external_force(data, force, point, local=True)
            link.add_external_torque(data, torque, local=True)
