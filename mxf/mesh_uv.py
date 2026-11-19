"""Export implicit MuJoCo planar UV coordinates for unwrapped visual meshes."""

from pathlib import Path

import numpy as np


def export_planar_meshes(model, folder):
    import mujoco

    bindings = {}
    meshes = {}
    for g in range(model.ngeom):
        mat, mid = model.geom_matid[g], model.geom_dataid[g]
        if (
            model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
            or mat < 0
            or model.mat_texid[mat] < 0
            or model.tex_type[model.mat_texid[mat]] != mujoco.mjtTexture.mjTEXTURE_2D
            or model.mesh_texcoordadr[mid] >= 0
            or model.geom_group[g] == 0
            or model.geom_contype[g]
            or model.geom_conaffinity[g]
        ):
            continue
        key = (int(mid), int(mat))
        if key not in meshes:
            start = model.mesh_vertadr[mid]
            verts = model.mesh_vert[start : start + model.mesh_vertnum[mid]]
            start = model.mesh_faceadr[mid]
            faces = model.mesh_face[start : start + model.mesh_facenum[mid]]
            normal_faces = model.mesh_facenormal[start : start + len(faces)]
            start = model.mesh_normaladr[mid]
            normals = model.mesh_normal[start : start + model.mesh_normalnum[mid]]
            scale = model.mat_texrepeat[mat].copy()
            if not model.mat_texuniform[mat]:
                scale /= np.maximum(1e-15, model.geom_size[g, :2])
            uv = verts[:, :2] * scale * [0.5, -0.5] - 0.5
            # Use a common integer offset per axis to preserve wrap across triangles.
            uv -= np.floor(uv.min(axis=0))
            uv[:, 1] = 1 - uv[:, 1]
            path = Path("assets") / f"planar-{mid}-{mat}.obj"
            with (folder / path).open("w") as f:
                for label, rows in (("v", verts), ("vn", normals), ("vt", uv)):
                    for row in rows:
                        f.write(label + " " + " ".join(f"{v:.9g}" for v in row) + "\n")
                for vs, ns in zip(faces + 1, normal_faces + 1):
                    f.write("f " + " ".join(f"{v}/{v}/{n}" for v, n in zip(vs, ns)) + "\n")
            meshes[key] = str(path)
        bindings[model.geom(g).name] = {
            "file": meshes[key],
            "pos": model.geom_pos[g].tolist(),
            "quat": model.geom_quat[g].tolist(),
        }
    return bindings
