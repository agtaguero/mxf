"""Capture same-state, no-step external and wrist views and a native surface mask."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from .assets import prepare_scene, place_fixtures

CAMERAS = ("agentview", "robot0_eye_in_hand")


def capture(engine, scene, state, dest, chart=False, legacy=False):
    scene = scene.resolve()
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    ini = json.loads((scene / f"state-{state:02}" / "initial-state.json").read_text())
    source = ET.parse(scene / "scene.xml")
    materials = {
        t.get("name")
        for t in source.findall("asset/texture")
        if t.get("type", "cube") == "cube" and t.get("file")
    }
    mats = {m.get("name") for m in source.findall("asset/material") if m.get("texture") in materials}
    import tempfile

    # Keep the input chart inside the scene bundle to exercise the normal asset importer.
    with tempfile.TemporaryDirectory(dir=scene) as scratch:
        if chart:
            y, x = np.indices((256, 256))
            colors = np.stack(
                (32 + 3 * x // 4, 32 + 3 * y // 4, 60 + 100 * ((x // 32 + y // 32) % 2)), axis=-1
            ).astype(np.uint8)
            chart_file = Path(scratch) / "uv.png"
            Image.fromarray(colors).save(chart_file)
            for texture in source.findall("asset/texture"):
                if texture.get("name") in materials:
                    texture.set("file", str(chart_file.relative_to(scene)))
        if engine == "native":
            for a in source.findall("asset/*"):
                if file := a.get("file"):
                    a.set("file", str((scene / file).resolve()))
            # The reference renderer loads the assets after the temporary directory is closed.
            if chart:
                saved = dest / "uv.png"
                Image.fromarray(colors).save(saved)
                for tex in source.findall("asset/texture"):
                    if tex.get("name") in materials:
                        tex.set("file", str(saved.resolve()))
            path = dest / "native.xml"
            source.write(path)
        else:
            xml = Path(scratch) / "test.xml"
            source.write(xml)
            path = prepare_scene(scene, dest, scene_file=xml)
            if legacy:
                tree = ET.parse(path)
                for tex in tree.findall("asset/texture"):
                    if tex.get("name") in materials:
                        original = source.find(f"asset/texture[@name='{tex.get('name')}']")
                        asset = (scene / original.get("file")).resolve()
                        if chart:
                            asset = dest.resolve() / "legacy-uv.png"
                            Image.fromarray(colors).save(asset)
                        tex.set("file", str(asset))
                tree.write(path)
    if engine == "native":
        import mujoco

        m = mujoco.MjModel.from_xml_path(str(place_fixtures(path, dest / "placed.xml", ini["fixtures"])))
        d = mujoco.MjData(m)
        for name, s in ini["joints"].items():
            d.joint(name).qpos[:] = s["qpos"]
            d.joint(name).qvel[:] = s["qvel"]
        mujoco.mj_forward(m, d)
        opt = mujoco.MjvOption()
        opt.geomgroup[0] = 0
        opt.sitegroup[:] = 0
        renderer = mujoco.Renderer(m, 256, 256)
        geom_ids = [
            g for g in range(m.ngeom) if m.geom_matid[g] >= 0 and m.material(m.geom_matid[g]).name in mats
        ]
        try:
            for camera in CAMERAS:
                renderer.update_scene(d, camera, scene_option=opt)
                Image.fromarray(renderer.render()).save(dest / (camera + ".png"))
                renderer.enable_segmentation_rendering()
                seg = renderer.render()
                mask = (seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[:, :, 0], geom_ids)
                Image.fromarray(mask).save(dest / (camera + "-mask.png"))
                bindings = json.loads((scene / "scene.json").read_text()).get("planar_meshes", {})
                subset = [m.geom(name).id for name in bindings]
                planar = (seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[:, :, 0], subset)
                Image.fromarray(planar).save(dest / (camera + "-planar.png"))
                renderer.disable_segmentation_rendering()
        finally:
            if hasattr(renderer, "close"):
                renderer.close()
    else:
        from .motrix import MotrixEnv

        env = MotrixEnv(scene, path, dest, ini["fixtures"])
        try:
            obs = env.set_init_state(ini["joints"], ini.get("osc_reset"))
            for camera in CAMERAS:
                Image.fromarray(obs[camera + "_image"][::-1]).save(dest / (camera + ".png"))
        finally:
            env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("engine", choices=("native", "motrix"))
    p.add_argument("scene", type=Path)
    p.add_argument("dest", type=Path)
    p.add_argument("--state", type=int, default=0)
    p.add_argument("--chart", action="store_true")
    p.add_argument("--legacy", action="store_true")
    args = p.parse_args()
    capture(args.engine, args.scene, args.state, args.dest, args.chart, args.legacy)
