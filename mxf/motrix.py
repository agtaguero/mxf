"""MotrixSim implementation of the LIBERO Spatial environment boundary."""
import hashlib
import json
import operator
import time
import xml.etree.ElementTree as ET

import motrixsim as mx
from motrixsim import ik, msd, osc, render
import numpy as np

from .assets import place_fixtures
from .goals import TaskGoals
from .fluid import FluidForces
from .free_joint import FreeJointDamping

CAMERAS = ("agentview", "robot0_eye_in_hand")
ASSET_WARMUP_SECONDS = 2.
ASSET_STABILITY_SECONDS = 0.4
ASSET_TIMEOUT_SECONDS = 15.


class CameraStartupError(RuntimeError):
    """An offscreen renderer initialization failure before policy inference."""


def stable_camera(a, b):
    # Compare 16-pixel tiles to filter per-pixel antialiasing and shading
    # fluctuations while detecting the spatial appearance of a late mesh.
    def tiles(frame):
        return frame.reshape(16, 16, 16, 16, 3).mean(axis=(1, 3))

    change = np.abs(tiles(a) - tiles(b)).max(axis=-1)
    return change.mean() <= 2. and np.count_nonzero(change > 8.) <= 1


def multiply(a, b):
    av, bv = a[:3], b[:3]
    return np.r_[a[3] * bv + b[3] * av + np.cross(av, bv), a[3] * b[3] - av @ bv]


def from_rotvec(v):
    angle = np.linalg.norm(v)
    return np.r_[v * (0.5 * np.sinc(angle / (2 * np.pi))), np.cos(angle / 2)]


def rotvec(quat):
    quat = quat / np.linalg.norm(quat)
    if quat[-1] < 0:
        quat = -quat
    sin_half = np.linalg.norm(quat[:3])
    return quat[:3] * (2 * np.arctan2(sin_half, quat[3]) / sin_half if sin_half else 2.)


class MotrixEnv:
    def __init__(self, scene, template, output, fixtures, *, camera_enabled=True, physics_solver=None):
        self.config = json.loads((scene / "scene.json").read_text())
        # Earlier exports omit controller memory. Identify the archived LIBERO
        # MountedPanda model before using its published reset joint posture.
        robot = ET.parse(scene / "scene.xml").find(".//body[@name='robot0_base']")
        self.archived_panda = (robot is not None and hashlib.sha256(ET.tostring(robot)).hexdigest()
                               == "c99017c81beb5770a6bbb795b656b8f0258ab3c97bbfa54a5c6c864e03fa22dc")
        case = place_fixtures(template, output / "case.xml", fixtures)
        visual_groups = {}
        if camera_enabled and self.config.get("indicators"):
            tree = ET.parse(case)
            occupied = {int(g.get("group", 0)) for g in tree.iter("geom")}
            available = iter(sorted(set(range(1, 6)) - occupied))
            for parent in tree.iter("body"):
                for site in parent.findall("site"):
                    name = site.get("name")
                    if name in self.config["indicators"]:
                        group = next(available)
                        attrs = {k: v for k, v in site.attrib.items() if k in
                                 {"type", "size", "pos", "quat", "rgba", "axisangle", "euler", "fromto"}}
                        ET.SubElement(parent, "geom", **attrs, name="__mxf_indicator_" + name,
                                      group=str(group), mass="0", contype="0", conaffinity="0")
                        visual_groups[name] = group
            if visual_groups.keys() != self.config["indicators"].keys():
                raise ValueError("An exported visualization site is missing")
            tree.write(case, encoding="utf-8")
        if physics_solver is None:
            model = mx.load_model(case)
        else:
            world = msd.from_str(case.read_text(), format="mjcf", file_path=str(case.resolve()))
            options = world.simulate_option
            options.solver_type = getattr(msd.SolverType, physics_solver)
            world.simulate_option = options
            model = msd.build(world)
        self.model = model
        self.fluid = FluidForces(model, self.config["fluid"]) if self.config.get("fluid") else None
        damping = self.config.get("free_damping")
        self.free_damping = FreeJointDamping(model, damping, self.config["physics_dt"]) if damping else None
        self.fixtures = fixtures
        self.site = model.get_site(self.config["eef_site"])
        self.hand = model.get_link(self.config["eef_body"])
        self.arm = [model.get_actuator(name) for name in self.config["arm_actuators"]]
        self.fingers = [model.get_actuator(name) for name in self.config["gripper_actuators"]]
        self.finger_joints = [model.get_joint(name) for name in self.config["gripper_joints"]]
        self.chain = ik.IkChain(model, self.site.parent_link.name,
                                end_effector_offset=np.r_[self.site.local_pos, self.site.local_quat])
        if self.chain.num_dof_vel != len(self.arm):
            raise ValueError("OSC arm chain does not match actuator count")
        gains = self.config["osc"]
        self.solver = osc.OscSolver(kp=gains["kp"])
        self.range_min, self.range_max = np.array(gains["output_min"]), np.array(gains["output_max"])
        self.substeps = round(1 / (20 * self.config["physics_dt"]))
        self.goals = TaskGoals(model, self.config)
        self.goal_links = self.goals.links

        self.indicators = []
        for name, group in visual_groups.items():
            rule = self.config["indicators"][name]
            self.indicators.append((group, [model.get_joint(n) for n in rule["joints"]],
                                    getattr(operator, rule["op"]), rule["limit"]))
        self.indicator_state = {}
        self.renderer = None
        if camera_enabled:
            cameras = {camera.name: camera for camera in model.cameras.tolist()}
            for name in CAMERAS:
                cameras[name].set_render_target("image", 256, 256)
            self.renderer = render.RenderApp(headless=True)
            try:
                self.renderer.launch(model, render_settings=render.RenderSettings(
                    simplify_render_mesh=False, enable_shadow=False, enable_ssao=False,
                    enable_oit=False, share_lights_between_envs=False,
                ))
                self.renderer.opt.set_group_vis(0, False)
                self.cameras = [self.renderer.get_camera(cameras[name].index) for name in CAMERAS]
            except BaseException as error:
                self.renderer.close()
                if "OutOfMemory" in str(error):
                    raise CameraStartupError("MotrixSim could not allocate its render device") from error
                raise

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    def set_init_state(self, joints, controller_reset=None, *, legacy=False):
        model = self.model
        # Allocate fresh data to discard cached contacts and solver warm starts.
        data = self.data = mx.SceneData(model)
        pos, vel = data.dof_pos.copy(), data.dof_vel.copy()
        bases = {base.name: base for base in model.floating_bases}
        if set(joints) != set(bases) | set(model.joint_names):
            raise ValueError("Native and MotrixSim joint names differ")
        for name, state in joints.items():
            qpos, qvel = np.asarray(state["qpos"]), np.asarray(state["qvel"])
            if name in bases:
                base = bases[name]
                # MuJoCo free joint qpos uses wxyz. MotrixSim floating poses use xyzw.
                pos[base.dof_pos_indices] = qpos[[0, 1, 2, 4, 5, 6, 3]]
                vel[base.dof_vel_indices] = qvel
            else:
                joint = model.get_joint(name)
                start, velocity_start = joint.dof_pos_index, joint.dof_vel_index
                pos[start:start + joint.num_dof_pos] = qpos
                vel[velocity_start:velocity_start + joint.num_dof_vel] = qvel
        data.reset(model, pos, vel, forward_kinematic=True)
        for name, base in bases.items():
            velocity = joints[name]["qvel"]
            if np.any(velocity):
                base.set_global_linear_velocity(data, velocity[:3])
                base.set_local_angular_velocity(data, velocity[3:])
        model.forward_kinematic(data)
        for name, expected in self.fixtures.items():
            pose = model.get_link(name).get_pose(data)
            quat = np.asarray(expected["quaternion_wxyz"])[[1, 2, 3, 0]]
            if not (np.allclose(pose[:3], expected["position"], atol=1e-6, rtol=0.)
                    and abs(float(quat @ pose[3:])) > 1. - 1e-6):
                raise ValueError(f"MotrixSim fixture pose differs for {name}")
        self.rest = self.chain.get_dof_pos(data).copy()
        self.goal_quat = self.site.get_pose(data)[3:].copy()
        self.start_pose = None
        self.start_data = None
        self.osc_reset_source = "legacy"
        if not legacy and (controller_reset is not None or self.archived_panda):
            # Motrix computes inertia and bias on step. Prime a scratch state
            # without consuming a physical timestep in the rollout.
            def dynamics_at(qpos, qvel):
                scratch = mx.SceneData(model)
                scratch.reset(model, qpos, qvel, forward_kinematic=True)
                model.step(scratch)
                scratch.set_dof_pos(qpos, model)
                scratch.set_dof_vel(qvel)
                model.forward_kinematic(scratch)
                return scratch

            self.osc_reset_source = "exported" if controller_reset is not None else "archived-mounted-panda"
            self.start_bias = self.chain.get_bias_force(dynamics_at(pos, vel))
            if controller_reset is None:
                rest = np.array([0., -.161037389, 0., -2.44459747, 0., 2.22675220, np.pi / 4])
                cached_q, cached_v = rest, np.zeros_like(rest)
            else:
                rest = np.asarray(controller_reset["rest"])
                cached_q = controller_reset["joint_pos"]
                cached_v = controller_reset["joint_vel"]
            self.rest = rest
            pos, vel = pos.copy(), vel.copy()
            for name, q, v in zip(self.config["arm_joints"], cached_q, cached_v, strict=True):
                joint = model.get_joint(name)
                pos[joint.dof_pos_index], vel[joint.dof_vel_index] = q, v
            self.start_data = dynamics_at(pos, vel)
            self.start_pose = self.site.get_pose(self.start_data)
            self.goal_quat = self.start_pose[3:].copy()
            if controller_reset is not None:
                self.start_pose = np.r_[controller_reset["position"], controller_reset["orientation_xyzw"]]
                self.goal_quat = np.asarray(controller_reset["goal_xyzw"])
        self.grip = np.zeros(2)
        self.physics_started = False
        self.step_timing = {}
        return self.wait_for_camera_assets() if self.renderer is not None else self.observe()

    def wait_for_camera_assets(self):
        # CaptureTask completion guarantees a readback, but asset drawing can
        # still be pending. Pump both cameras at a fixed pose before warmup.
        start = unchanged_since = time.perf_counter()
        previous = self.observe()
        while time.perf_counter() - start < ASSET_TIMEOUT_SECONDS:
            time.sleep(0.075)
            observation = self.observe()
            now = time.perf_counter()
            if not all(stable_camera(observation[camera + "_image"], previous[camera + "_image"])
                       for camera in CAMERAS):
                unchanged_since = now
            if now - start >= ASSET_WARMUP_SECONDS and now - unchanged_since >= ASSET_STABILITY_SECONDS:
                self.camera_asset_warmup_seconds = now - start
                return observation
            previous = observation
        raise CameraStartupError("MotrixSim camera frames did not settle at the initial pose")

    def observe(self):
        quat = self.hand.get_pose(self.data)[3:]
        # Match the sign convention of the reference Panda pose near its
        # nominal downward orientation before π0.5's axis-angle transform.
        if quat[0] < 0:
            quat = -quat
        obs = {"robot0_eef_pos": self.site.get_position(self.data),
               "robot0_eef_quat": quat,
               "robot0_gripper_qpos": np.array([
                   joint.get_dof_pos(self.data).item() for joint in self.finger_joints
               ], dtype=np.float32)}
        for name, link in self.goal_links.items():
            obs[name + "_pos"] = link.get_position(self.data)
        if self.renderer is not None:
            for group, joints, comparison, limit in self.indicators:
                visible = any(comparison(joint.get_dof_pos(self.data).item(), limit) for joint in joints)
                if self.indicator_state.get(group) != visible:
                    self.renderer.opt.set_group_vis(group, visible)
                    self.indicator_state[group] = visible
            self.renderer.sync(self.data, wait=True)
            captures = [camera.capture() for camera in self.cameras]
            self.renderer.sync(self.data, wait=True)
            for name, capture in zip(CAMERAS, captures):
                image = capture.take_image()
                if image is None:
                    raise RuntimeError(f"MotrixSim did not return {name} RGB")
                # Motrix renders top-left origin. The LIBERO adapter expects the
                # raw MuJoCo buffer, whose scanlines start at the bottom.
                obs[name + "_image"] = np.ascontiguousarray(image.pixels[::-1, :, :3])
        return obs

    def step(self, action):
        started = time.perf_counter()
        action = np.asarray(action)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Expected a finite LIBERO OSC 7D action")
        delta = self.range_min + (np.clip(action[:6], -1., 1.) + 1.) * (
            self.range_max - self.range_min) / 2.
        pose = self.start_pose if self.start_pose is not None else self.site.get_pose(self.data)
        position = pose[:3] + delta[:3]
        if np.any(delta[3:]):
            self.goal_quat = multiply(from_rotvec(delta[3:]), pose[3:])
        target = rotvec(self.goal_quat)
        gripper_delta = np.array([-1., 1.]) * self.config["gripper_speed"] * np.sign(action[6])
        for _ in range(self.substeps):
            # MotrixSim produces the inertia and bias cache on its first step.
            if self.physics_started or self.start_data is not None:
                state = self.data if self.physics_started else self.start_data
                torques = self.solver.solve(self.chain, position, target, self.rest, state)
                if not self.physics_started:
                    # robosuite uses the cached reset kinematics in the first
                    # controller update and the restored state's live bias.
                    torques += self.start_bias - self.chain.get_bias_force(state)
                    self.start_data = self.start_pose = None
                for actuator, torque in zip(self.arm, torques):
                    actuator.set_ctrl(self.data, float(np.clip(torque, *actuator.ctrl_range)))
            self.grip = np.clip(self.grip + gripper_delta, -1., 1.)
            for actuator, command in zip(self.fingers, self.grip):
                low, high = actuator.ctrl_range
                actuator.set_ctrl(self.data, float(low + (1. + command) * (high - low) / 2.))
            if self.free_damping is not None:
                self.free_damping.apply(self.data)
            if self.fluid is not None:
                self.fluid.apply(self.data)
            self.model.step(self.data)
            self.physics_started = True
        physics = time.perf_counter()
        obs = self.observe()
        self.step_timing = {"physics_and_control_seconds": physics - started,
                            "observation_and_render_seconds": time.perf_counter() - physics}
        return obs, None, None, None

    def check_success(self):
        query = self.model.get_contact_query(self.data)
        return self.goals.satisfied(self.data, query)
