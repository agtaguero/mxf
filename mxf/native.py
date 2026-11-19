"""Original LIBERO environment and export of named, portable scene state."""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np


def task_catalog(suite_name="libero_spatial"):
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()[suite_name]()
    return suite, Path(get_libero_path("bddl_files"))


def open_task(bddl, render=True, max_steps=220):
    from libero.libero.envs.env_wrapper import ControlEnv

    return ControlEnv(
        bddl_file_name=str(bddl), control_freq=20, horizon=max_steps + 11,
        camera_heights=256, camera_widths=256, use_camera_obs=render,
        has_offscreen_renderer=render, hard_reset=True,
    )


def articulation_rule(obj, predicate, joints):
    from copy import copy, deepcopy

    probe = copy(obj)
    probe.object_properties = deepcopy(obj.object_properties)
    method = getattr(probe, {"open": "is_open", "close": "is_close", "turnon": "turn_on"}[predicate])
    limits = probe.object_properties["articulation"]["default_" + predicate + "_ranges"]
    cuts = []
    for threshold in set(limits):
        below, at, above = [bool(method(q)) for q in
            (np.nextafter(threshold, -np.inf), threshold, np.nextafter(threshold, np.inf))]
        if below != above:
            cuts.append((threshold, ("le" if at else "lt") if below else ("ge" if at else "gt")))
    if len(cuts) != 1 or not joints:
        raise ValueError(f"Unsupported joint predicate {predicate} for {obj.name}")
    return {"joints": list(joints), "limit": cuts[0][0], "op": cuts[0][1]}


def visual_indicators(env):
    indicators = {}
    for name in set(env.env.visualization_sites_list):
        obj = env.env.get_object(name)
        if not hasattr(obj, "turn_on"):
            raise ValueError(f"Unknown visual indicator on {name}")
        rule = articulation_rule(obj, "turnon", obj.joints)
        for site, _ in obj.object_properties["vis_site_names"].values():
            indicators[site] = rule
    return indicators


def export_scene(env, output, instruction):
    model = env.sim.model
    xml = env.env.model.get_xml()
    root = ET.fromstring(xml)
    assets = output / "assets"
    assets.mkdir()
    for element in root.findall("asset/*"):
        file = element.get("file")
        if file:
            path = Path(file)
            if not path.is_absolute():
                raise ValueError(f"Unexpected relative asset path: {file}")
            name = hashlib.sha256(path.read_bytes()).hexdigest()[:20] + path.suffix
            if not (assets / name).exists():
                shutil.copyfile(path, assets / name)
            element.set("file", f"assets/{name}")
    for compiler in root.findall("compiler"):
        for field in ("assetdir", "meshdir", "texturedir"):
            compiler.attrib.pop(field, None)
    ET.ElementTree(root).write(output / "scene.xml", encoding="utf-8")
    from .contact_reference import compiled_contact_reference
    from .free_joint import compiled_free_damping
    from .mesh_uv import export_planar_meshes
    from .fluid import compiled_fluid
    robot = env.robots[0]
    goal = env.env.parsed_problem["goal_state"]
    objects, sites, joint_rules = {}, {}, {}
    for predicate, *terms in goal:
        for name in terms:
            state = env.env.object_states_dict[name]
            parent = name
            if state.object_state_type == "site":
                site = env.env.object_sites_dict[name]
                parent = state.parent_name
                sites[name] = {"parent": parent, "size": np.asarray(site.size).tolist()}
            obj = env.env.get_object(parent)
            if obj is not None:
                objects[parent] = {"body": obj.root_body, "contact_geoms": list(obj.contact_geoms)}
            if predicate in ("open", "close", "turnon"):
                joints = obj.joints if state.object_state_type == "object" else site.joints
                joint_rules[predicate + "/" + name] = articulation_rule(obj, predicate, joints)
    config = {
        "schema": "mxf-libero-scene.v1",
        "instruction": instruction,
        "bddl_instruction": env.language_instruction,
        "goals": goal, "sites": sites, "joint_rules": joint_rules, "indicators": visual_indicators(env),
        "objects": objects,
        "eef_site": robot.gripper.important_sites["grip_site"],
        "eef_body": robot.robot_model.eef_name,
        "arm_joints": list(robot.robot_model.joints),
        "arm_actuators": list(robot.robot_model.actuators),
        "gripper_joints": list(robot.gripper.joints),
        "gripper_actuators": list(robot.gripper.actuators),
        "gripper_speed": robot.gripper.speed,
        "osc": {
            key: np.asarray(getattr(robot.controller, key)).tolist()
            for key in ("input_min", "input_max", "output_min", "output_max", "kp", "kd")
        },
        "physics_dt": float(model.opt.timestep),
        "fluid": compiled_fluid(model._model),
        "contact_reference": compiled_contact_reference(model._model),
        "free_damping": compiled_free_damping(model._model),
        "planar_meshes": export_planar_meshes(model._model, output),
    }
    target = output / "scene.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n")
    temporary.replace(target)


def snapshot(env):
    sim = env.sim
    result = {}
    for name in sim.model.joint_names:
        result[name] = {
            "qpos": np.asarray(sim.data.get_joint_qpos(name)).reshape(-1).tolist(),
            "qvel": np.asarray(sim.data.get_joint_qvel(name)).reshape(-1).tolist(),
        }
    return result


def fixtures(env):
    """Named XML body transforms written by LIBERO's reset sampler."""
    model = env.sim.model
    return {
        fixture.root_body: {
            "position": model.body_pos[model.body_name2id(fixture.root_body)].tolist(),
            "quaternion_wxyz": model.body_quat[model.body_name2id(fixture.root_body)].tolist(),
        }
        for fixture in env.env.fixtures_dict.values()
    }

def osc_reset(env):
    """Controller memory survives LIBERO's set_init_state physics restore."""
    from robosuite.utils.transform_utils import mat2quat

    ctrl = env.robots[0].controller
    return {"rest": ctrl.initial_joint.tolist(), "joint_pos": ctrl.joint_pos.tolist(),
            "joint_vel": ctrl.joint_vel.tolist(), "position": ctrl.ee_pos.tolist(),
            "orientation_xyzw": mat2quat(ctrl.ee_ori_mat).tolist(),
            "goal_xyzw": mat2quat(ctrl.goal_ori).tolist()}
