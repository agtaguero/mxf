"""Summarize full-duration replays and paired checkpoints without changing the LIBERO score."""

import argparse
import hashlib
import xml.etree.ElementTree as ET
from functools import cache
from pathlib import Path

import numpy as np

from .compare import read
from .contact_phases import terminal_support
from .eval import write_json
from .plate_support import anchors, describes, relative


def norm(x):
    return float(np.linalg.norm(x))


def snapshot(p, quat, v, omega, t):
    return {
        "xyz_mm": (1000 * p[t]).tolist(),
        "quat_wxyz": quat[t].tolist(),
        "vel_m_s": v[t].tolist(),
        "omega_rad_s": omega[t].tolist(),
    }


def full_row(file):
    m = read(file)
    with np.load(file.with_name("trace.npz")) as data:
        bits = data["bits"]
    return {
        "success": m["success"],
        "lift": m["events"]["lift"] is not None,
        "arrival": m["events"]["plate"] is not None,
        **terminal_support(bits),
    }


def summarize(rows):
    return {k: sum(int(r[k]) for r in rows) for k in ("success", "lift", "arrival", "strict", "any10")}


@cache
def asset_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_scene(path, expected_dt):
    tree = ET.parse(path)
    option = tree.find("option")
    assert float(option.attrib.pop("timestep")) == expected_dt
    for node in tree.findall("asset/*"):
        if name := node.get("file"):
            node.set("file", asset_hash((path.parent / name).resolve()))
    return ET.tostring(tree.getroot())


def analyze(work, archive):
    cases = {}
    for f in sorted((work / "native").glob("task-*/state-*/result.json")):
        case = f.parent.relative_to(work / "native")
        target = {}
        for engine in ("native", "motrix"):
            other = "motrix" if engine == "native" else "native"
            meta = read(work / engine / case / "result.json")
            peer = read(work / other / case / "result.json")
            for k in ("initial_state_sha256", "fixtures_sha256", "action_sha256", "ticks", "official_ticks"):
                assert meta[k] == peer[k]
            with np.load(work / engine / case / "trace.npz") as data:
                bits = data["bits"]
                original = archive / ("tail-native" if engine == "native" else "tail-inertia") / case
                with np.load(original / "trace.npz") as prev:
                    matches = int(np.count_nonzero(prev["bits"] != bits))
                p, quat, v, omega = relative(data["qpos"], data["qvel"], meta["layout"])
            t = anchors(meta, peer, bits)["entry"]
            paired = anchors(peer, meta, bits)["entry"]
            row = full_row(work / engine / case / "result.json")
            row.update(describes(p[t:], quat[t:], v[t:], omega[t:], bits[t:]))
            row["entry_tick"] = t
            row["original_contact_bit_differences"] = matches
            row["at_peer_clock"] = snapshot(p, quat, v, omega, paired)
            row["events"] = meta["events"]
            row["released_end_100ms"] = bool(((bits[-50:] & 3) == 0).all())
            row["end_xy_diameter_upper_mm"] = norm(np.ptp(p[-50:, :2], axis=0)) * 1000
            target[engine] = row
        key = str(case)
        target["cross"] = {}
        target["common_control"] = {
            e: read((work / "common-control" / e / case).with_suffix(".json")) for e in ("native", "motrix")
        }
        target["forces"] = {}
        for engine in ("native", "motrix", "halfstep"):
            target["cross"][engine] = read((work / "cross" / engine / case).with_suffix(".json"))
            if engine == "native":
                continue
            for src in ("native", "motrix"):
                fzfile = work / "forces" / engine / case.parent / f"{case.name}-{src}.npz"
                with np.load(fzfile) as d:
                    n = len(d["plate"]) // 2  # final 100 ms of the 200 ms restart
                    hit, other, fz = (d[k][-n:] for k in ("plate", "other", "world_fz_N"))
                target["forces"][engine + "-" + src] = {
                    "other_steps": int(other.sum()),
                    "plate_steps": int(hit.sum()),
                    "mean_fz_N": float(fz.mean()),
                    "peak_fz_N": float(fz.max()),
                    "max_abs_fz_off_N": float(np.abs(fz[~hit]).max()) if (~hit).any() else None,
                }
        cases[key] = target
    write_json(work / "cases.json", cases)
    totals = {}
    for name in ("native", "motrix", "full-halfstep", "fresh-native", "fresh-motrix", "fresh-halfstep"):
        rows = {
            str(f.parent.relative_to(work / name)): full_row(f)
            for f in sorted((work / name).glob("task-*/state-*/result.json"))
        }
        assert len(rows) == (50 if name.startswith("fresh-") else 100)
        for k in rows:
            reference = "fresh-native" if name.startswith("fresh-") else "native"
            a = read(work / name / k / "result.json")
            b = read(work / reference / k / "result.json")
            assert all(
                a[ident] == b[ident]
                for ident in ("initial_state_sha256", "fixtures_sha256", "action_sha256", "ticks")
            )
        by_split = {}
        for suffix in sorted({k[-1] for k in rows}):
            selected = {k: v for k, v in rows.items() if k.endswith(suffix)}
            by_split[suffix] = {
                "total": summarize(selected.values()),
                "tasks": {
                    task: summarize([v for k, v in selected.items() if k.startswith(task + "/")])
                    for task in sorted({k.split("/")[0] for k in selected})
                },
            }
        totals[name] = by_split
    write_json(work / "totals.json", totals)
    scene_checks = {}
    for baseline, candidate in (("motrix", "full-halfstep"), ("fresh-motrix", "fresh-halfstep")):
        count = 0
        for file in (work / baseline).glob("task-*/state-*/case.xml"):
            ref = file.relative_to(work / baseline)
            assert normalized_scene(file, 0.002) == normalized_scene(work / candidate / ref, 0.001), ref
            count += 1
        scene_checks[candidate] = count
    write_json(work / "scene-checks.json", scene_checks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("runs/plate-support"))
    parser.add_argument("--archive", type=Path, default=Path("runs/contact-alignment"))
    args = parser.parse_args()
    analyze(args.work, args.archive)
