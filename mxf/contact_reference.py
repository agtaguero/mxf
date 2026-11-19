"""Export MuJoCo's post-mixing lower bound on contact time constants."""

from itertools import combinations
import xml.etree.ElementTree as ET

import numpy as np


def compiled_contact_reference(model):
    import mujoco as mj

    flags = model.opt.disableflags
    threshold = 0.0 if flags & mj.mjtDisableBit.mjDSBL_REFSAFE else 2 * model.opt.timestep
    names = list(model.geom(i).name for i in range(model.ngeom))
    anonymous = {}
    pairs = {}
    material = {}

    def name(i):
        if not names[i]:
            key = f"__mxf_contact_geom_{i}"
            if key in names:
                raise ValueError(f"Generated name already exists: {key}")
            body = model.geom_bodyid[i]
            anonymous[key] = [model.body(body).name, int(i - model.body_geomadr[body])]
            names[i] = key
        return names[i]

    def add(i, j, ref, imp, friction, dimension, margin, gap):
        time_constant = max(threshold, ref[0]) if ref[0] > 0 else ref[0]
        attrs = dict(
            solref=[time_constant, ref[1]],
            solimp=imp,
            friction=friction,
            condim=[dimension],
            margin=[margin],
            gap=[gap],
        )
        values = tuple(" ".join(format(x, ".15g") for x in value) for value in attrs.values())
        if values not in material:
            material[values] = len(material)
        pairs.setdefault(material[values], []).append([name(i), name(j)])

    explicit = {tuple(sorted((int(i), int(j)))) for i, j in zip(model.pair_geom1, model.pair_geom2)}
    excludes = set(model.exclude_signature.tolist())
    filter_parent = not flags & mj.mjtDisableBit.mjDSBL_FILTERPARENT
    geoms = np.flatnonzero(model.geom_contype | model.geom_conaffinity)
    # Equal materials can share the clamped per-geom spring. Mixed materials
    # must combine the original reference parameters before computing stiffness.
    per_geom = {
        name(i): f"{threshold:.15g} {model.geom_solref[i, 1]:.15g}"
        for i in geoms
        if 0 < model.geom_solref[i, 0] < threshold
    }
    for i, j in combinations(geoms, 2):
        if (i, j) in explicit or model.opt.collision == int(mj.mjtCollision.mjCOL_PAIR):
            continue
        if not (
            (model.geom_contype[i] & model.geom_conaffinity[j])
            or (model.geom_contype[j] & model.geom_conaffinity[i])
        ):
            continue
        b, c = sorted((model.geom_bodyid[i], model.geom_bodyid[j]))
        w, v = model.body_weldid[[b, c]]
        if ((int(b) + 1) << 16) + int(c) + 1 in excludes or w == v:
            continue
        if (
            filter_parent
            and w
            and v
            and (
                model.body_weldid[model.body_parentid[w]] == v
                or model.body_weldid[model.body_parentid[v]] == w
            )
        ):
            continue
        if max(model.geom_type[i], model.geom_type[j]) <= int(mj.mjtGeom.mjGEOM_HFIELD):
            continue
        if model.geom_priority[i] != model.geom_priority[j]:
            g = i if model.geom_priority[i] > model.geom_priority[j] else j
            ref, imp, fr, dim = (
                model.geom_solref[g],
                model.geom_solimp[g],
                model.geom_friction[g],
                model.geom_condim[g],
            )
        else:
            r, s = model.geom_solref[[i, j]]
            if np.array_equal(r, s) and np.array_equal(model.geom_solimp[i], model.geom_solimp[j]):
                continue
            a, bmix = model.geom_solmix[[i, j]]
            a, bmix = (x if x >= mj.mjMINVAL else 0.0 for x in (a, bmix))
            weight = a / (a + bmix) if a + bmix else 0.5
            ref = weight * r + (1 - weight) * s if min(r[0], s[0]) > 0 else np.minimum(r, s)
            imp = weight * model.geom_solimp[i] + (1 - weight) * model.geom_solimp[j]
            fr = np.maximum(model.geom_friction[i], model.geom_friction[j])
            dim = max(model.geom_condim[i], model.geom_condim[j])
        add(
            i,
            j,
            ref,
            imp,
            fr[[0, 0, 1, 2, 2]],
            dim,
            max(model.geom_margin[[i, j]]),
            max(model.geom_gap[[i, j]]),
        )
    authored = {}
    for i in range(model.npair):
        updates = {}
        for field in ("solref", "solreffriction"):
            vals = getattr(model, f"pair_{field}")[i]
            if 0 < vals[0] < threshold:
                updates[field] = f"{threshold:.15g} {vals[1]:.15g}"
        if updates:
            first, second = sorted((name(model.pair_geom1[i]), name(model.pair_geom2[i])))
            authored.setdefault(first, {})[second] = updates
    return {
        "per_geom": per_geom,
        "anonymous": anonymous,
        "authored": authored,
        "materials": [
            {
                "attrs": dict(zip(("solref", "solimp", "friction", "condim", "margin", "gap"), key)),
                "geoms": pairs[index],
            }
            for key, index in material.items()
        ],
    }


def apply_contact_reference(tree, spec):
    if not spec or not (spec.get("per_geom") or spec["authored"] or spec["materials"]):
        return
    bodies = {node.get("name"): node for node in tree.findall(".//worldbody//body")}
    bodies["world"] = tree.find("worldbody")
    for name, (body, slot) in spec["anonymous"].items():
        geom = bodies[body].findall("geom")[slot]
        if geom.get("name"):
            raise ValueError(f"Native anonymous contact geometry mismatch: {body}/{slot}")
        geom.set("name", name)
    by_name = {geom.get("name"): geom for geom in tree.findall(".//worldbody//geom")}
    for name, ref in spec.get("per_geom", {}).items():
        by_name[name].set("solref", ref)
    for node in tree.findall("contact/pair"):
        first, second = sorted((node.attrib["geom1"], node.attrib["geom2"]))
        node.attrib.update(spec["authored"].get(first, {}).get(second, {}))
    contact = tree.find("contact")
    if contact is None:
        contact = ET.SubElement(tree.getroot(), "contact")
    for group in spec["materials"]:
        for first, second in group["geoms"]:
            ET.SubElement(contact, "pair", geom1=first, geom2=second, **group["attrs"])
