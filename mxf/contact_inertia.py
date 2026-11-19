"""Audit compiled mass, center of mass and local inertia of the exported floating bodies."""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .eval import write_json


def properties_native(scene, names):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene))
    result = {}
    for name in names:
        body = model.body(name)
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, body.iquat)
        rot = rot.reshape(3, 3)
        inertia = rot @ np.diag(body.inertia) @ rot.T
        result[name] = [float(body.mass[0]), body.ipos.tolist(), inertia.tolist()]
    return result


def properties_motrix(scene, names):
    import motrixsim as mx

    model = mx.load_model(scene)
    data = mx.SceneData(model)
    result = {}
    for name in names:
        body = model.get_link(name)
        xx, yy, zz, xy, xz, yz = map(float, body.get_inertia_override(data))
        result[name] = [
            float(body.mass),
            body.center_of_mass.tolist(),
            [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]],
        ]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("engine", choices=("native", "motrix"))
    p.add_argument("reference", type=Path)
    p.add_argument("work", type=Path)
    args = p.parse_args()
    result = {}
    for t in range(10):
        task = f"task-{t:02}"
        src = args.reference / task / "scene.xml"
        names = [
            body.attrib["name"]
            for body in ET.parse(src).iter("body")
            if body.find("joint[@type='free']") is not None or body.find("freejoint") is not None
        ]
        if args.engine == "native":
            result[task] = properties_native(src, names)
        else:
            result[task] = {
                k: properties_motrix(args.work / k / task / "state-00/case.xml", names)
                for k in ("baseline", "inertia")
            }
    write_json(args.work / ("body-inertias-" + args.engine + ".json"), result)


if __name__ == "__main__":
    main()
