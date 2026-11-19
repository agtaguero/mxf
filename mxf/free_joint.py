"""Restore MJCF free-joint viscous damping omitted by the community importer."""

import numpy as np


def compiled_free_damping(model):
    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mass = np.empty((model.nv, model.nv))
    mujoco.mj_fullM(model, mass, data.qM)
    specs = {}
    for i in range(model.njnt):
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        adr = model.jnt_dofadr[i]
        damping = model.dof_damping[adr : adr + 6]
        if not damping.any():
            continue
        body = model.jnt_bodyid[i]
        if any(
            model.body_rootid[other] == body and model.body_jntnum[other]
            for other in range(body + 1, model.nbody)
        ):
            raise ValueError(
                "Damped free base with moving child joints needs a configuration-dependent mass matrix"
            )
        basis = np.eye(6)
        basis[:3, :3] = data.xmat[body].reshape(3, 3)
        inertia = basis.T @ mass[adr : adr + 6, adr : adr + 6] @ basis
        if not np.all(damping[:3] == damping[0]):
            raise ValueError("Anisotropic free-joint world damping requires a rotating damping matrix")
        specs[model.joint(i).name] = {
            "body": model.body(body).name,
            "inertia": inertia.tolist(),
            "damping": damping.tolist(),
        }
    return specs


class FreeJointDamping:
    def __init__(self, model, spec, timestep):
        self.model = model
        bases = {base.name: base for base in model.floating_bases}
        self.bases = [bases[name] for name in spec]
        self.links = [model.get_link(r["body"]) for r in spec.values()]
        self.ids = [link.index for link in self.links]
        self.gains = []
        for row in spec.values():
            inertia = np.asarray(row["inertia"])
            damping = np.diag(row["damping"])
            # A rigid body's isolated backward-Euler damping impulse as a wrench.
            self.gains.append(-damping @ np.linalg.solve(inertia + timestep * damping, inertia))
        self.gains = np.asarray(self.gains)

    def apply(self, data):
        rotation = self.model.get_link_rotation_mats(data, self.ids)
        velocity = np.asarray([base.get_global_linear_velocity(data) for base in self.bases])
        spin = np.asarray([base.get_local_angular_velocity(data) for base in self.bases])
        linear = (rotation.transpose(0, 2, 1) @ velocity[..., None])[..., 0]
        twist = np.concatenate((linear, spin), axis=1)
        wrench = (self.gains @ twist[..., None])[..., 0].astype(np.float32)
        for link, ft in zip(self.links, wrench):
            link.add_external_force(data, ft[:3], (0.0, 0.0, 0.0), local=True)
            link.add_external_torque(data, ft[3:], local=True)
