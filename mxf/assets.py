"""Translate the MuJoCo binary mesh container without changing geometry."""
from pathlib import Path
import struct
import json
import xml.etree.ElementTree as ET

import numpy as np

STOVE_BURNER = "flat_stove_1_collision_burner"
STOVE_BURNER_EXTENSION_M = 0.001


def msh_to_obj(source, destination):
    raw = source.read_bytes()
    vertices, normals, texcoords, faces = struct.unpack_from("<4i", raw)
    if (min(vertices, faces) <= 0 or normals not in (0, vertices)
            or texcoords not in (0, vertices)
            or len(raw) != 16 + 12 * (vertices + normals + faces) + 8 * texcoords):
        raise ValueError(f"Invalid MuJoCo mesh: {source}")
    points = np.frombuffer(raw, "<f4", vertices * 3, 16).reshape(-1, 3)
    offset = 16 + 12 * vertices
    directions = np.frombuffer(raw, "<f4", normals * 3, offset).reshape(-1, 3)
    offset += 12 * normals
    uv = np.frombuffer(raw, "<f4", texcoords * 2, offset).reshape(-1, 2)
    triangles = np.frombuffer(raw, "<i4", faces * 3, offset + 8 * texcoords).reshape(-1, 3)
    if (triangles.min() < 0 or triangles.max() >= vertices
            or not all(np.isfinite(array).all() for array in (points, directions, uv))):
        raise ValueError(f"Invalid MuJoCo mesh data: {source}")
    if texcoords:
        # MuJoCo reads MSH texcoords as stored but flips OBJ's V in LoadOBJ.
        # Write the OBJ convention so imported textures match the native MSH.
        uv = uv.copy()
        uv[:, 1] = 1. - uv[:, 1]
    with destination.open("w") as stream:
        for prefix, array in (("v", points), ("vt", uv), ("vn", directions)):
            for row in array:
                stream.write(prefix + " " + " ".join(f"{value:.9g}" for value in row) + "\n")
        for triangle in triangles + 1:
            refs = []
            for index in triangle:
                ref = str(index)
                if normals or texcoords:
                    ref += "/" + (str(index) if texcoords else "")
                if normals:
                    ref += "/" + str(index)
                refs.append(ref)
            stream.write("f " + " ".join(refs) + "\n")


def extend_stove_burner(tree, extension=STOVE_BURNER_EXTENSION_M):
    """Extend the hidden, two-sided LIBERO burner box downward at a fixed top."""
    burner = tree.find(f".//geom[@name='{STOVE_BURNER}']")
    if burner is None or burner.get("type") != "box" or burner.get("group") != "0":
        return False
    size = [float(value) for value in burner.get("size").split()]
    if size[2] != 0.0005:
        return False
    # At the saved MuJoCo resting pose, the base of the bowl straddles the
    # 1 mm box. Moving its center down avoids underside contacts on the bowl.
    position = [float(value) for value in burner.get("pos").split()]
    size[2] += extension / 2
    position[2] -= extension / 2
    burner.set("size", " ".join(f"{value:.9g}" for value in size))
    burner.set("pos", " ".join(f"{value:.9g}" for value in position))
    return True


def apply_inertia_group_range(tree):
    """Honor MJCF's inclusive geom-group range when Motrix infers body inertia."""
    compiler = tree.find("compiler")
    if compiler is None or compiler.get("inertiagrouprange") is None:
        return
    low, high = map(int, compiler.attrib["inertiagrouprange"].split())
    inference = compiler.get("inertiafromgeom", "auto")
    if inference == "false":
        return
    group_defaults = {"main": 0}

    def read_defaults(node, inherited):
        geom = node.find("geom")
        group = inherited if geom is None else int(geom.get("group", inherited))
        group_defaults[node.get("class", "main")] = group
        for child in node.findall("default"):
            read_defaults(child, group)

    for node in tree.findall("default"):
        read_defaults(node, group_defaults["main"])

    def visit(body, parent_class):
        default_class = body.get("childclass", parent_class)
        if body.find("inertial") is None or inference == "true":
            for geom in body.findall("geom"):
                group = int(geom.get("group", group_defaults[geom.get("class", default_class)]))
                if not low <= group <= high:
                    geom.set("mass", "0")
        for child in body.findall("body"):
            visit(child, default_class)

    for body in tree.findall("worldbody/body"):
        visit(body, "main")


def prepare_scene(directory, destination, *, original_stove_burner=True, inertia_groups=True, scene_file=None):
    """Keep the original export intact and retain an inspectable converted copy."""
    tree = ET.parse(scene_file or directory / "scene.xml")
    meshes = destination / "meshes"
    meshes.mkdir(exist_ok=True)
    for asset in tree.findall("asset/*"):
        if (asset.tag == "texture" and asset.get("type", "cube") == "2d"
                and asset.get("builtin") == "flat" and asset.get("mark", "none") == "none"
                and not asset.get("file")):
            from PIL import Image
            import hashlib
            rgb = tuple(int(255 * float(x)) for x in asset.get("rgb1", "0.8 0.8 0.8").split())
            dims = tuple(int(asset.get(attr, 1)) for attr in ("width", "height"))
            texture_dir = destination / "textures"
            texture_dir.mkdir(exist_ok=True)
            token = hashlib.sha256(repr((rgb, dims)).encode()).hexdigest()[:16]
            file = texture_dir / ("flat-" + token + ".png")
            if not file.exists():
                Image.new("RGB", dims, rgb).save(file)
            for attr in ("builtin", "mark", "rgb1", "rgb2", "width", "height"):
                asset.attrib.pop(attr, None)
            asset.set("file", str(file.resolve()))
            continue
        single_cube_tile = (asset.tag == "texture" and asset.get("type", "cube") == "cube"
                            and asset.get("file") and asset.get("gridsize", "1 1").split() == ["1", "1"]
                            and not asset.get("gridlayout"))
        if file := asset.get("file"):
            source = (directory / file).resolve()
            if not source.is_relative_to(directory.resolve()):
                raise ValueError(f"Asset outside scene bundle: {file}")
            if single_cube_tile:
                # A repeated cube face and Motrix's 2-D primitive UVs have opposite V.
                # Convert this binding only; the same image may also be used on a mesh.
                from PIL import Image
                texture_dir = destination / "textures"
                texture_dir.mkdir(exist_ok=True)
                import hashlib
                key = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
                tile = texture_dir / (key + "-cube-face.png")
                if not tile.exists():
                    with Image.open(source) as image:
                        image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(tile)
                asset.set("type", "2d")
                asset.set("file", str(tile.resolve()))
            elif source.suffix.lower() == ".msh":
                path = meshes / (source.stem + "-obj-uv.obj")
                if not path.exists():
                    msh_to_obj(source, path)
                asset.set("file", str(path.resolve()))
            else:
                asset.set("file", str(source))
    # Replace only collision-disabled visual meshes. The vertices are exported
    # in the native compiler's principal frame with its matching geom pose.
    config_path = directory / "scene.json"
    from .contact_reference import apply_contact_reference
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    apply_contact_reference(tree, config.get("contact_reference"))
    bindings = config.get("planar_meshes", {})
    material_meshes = {}
    for geom in tree.iter("geom"):
        if binding := bindings.get(geom.get("name")):
            filename = binding["file"]
            if filename not in material_meshes:
                name = "__mxf_" + Path(filename).stem
                source = (directory / filename).resolve()
                if not source.is_relative_to(directory.resolve()):
                    raise ValueError("Visual mesh outside scene bundle")
                ET.SubElement(tree.find("asset"), "mesh", name=name, file=str(source))
                material_meshes[filename] = name
            geom.set("mesh", material_meshes[filename])
            geom.set("pos", " ".join(map(str, binding["pos"])))
            for attr in ("axisangle", "euler", "xyaxes", "zaxis"):
                geom.attrib.pop(attr, None)
            geom.set("quat", " ".join(map(str, binding["quat"])))
    if inertia_groups:
        apply_inertia_group_range(tree)
    if not original_stove_burner:
        extend_stove_burner(tree)
    path = destination / "motrix.xml"
    tree.write(path, encoding="utf-8")
    return path


def place_fixtures(template, destination, fixtures):
    tree = ET.parse(template)
    pending = set(fixtures)
    for body in tree.iter("body"):
        name = body.get("name")
        if name in pending:
            pose = fixtures[name]
            body.set("pos", " ".join(map(str, pose["position"])))
            body.set("quat", " ".join(map(str, pose["quaternion_wxyz"])))
            pending.remove(name)
    if pending:
        raise ValueError(f"Fixture bodies missing from task XML: {sorted(pending)}")
    tree.write(destination, encoding="utf-8")
    return destination


def add_burner_options(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--original-stove-burner", action="store_true",
                       help="Preserve the exported collision geometry (default)")
    group.add_argument("--extend-stove-burner", action="store_true",
                       help="Reproduce the task-07 burner thickness experiment")
