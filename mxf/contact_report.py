"""Audit and summarize paired fixed actions, contact transitions and matched probes."""

import argparse
from collections import defaultdict, Counter
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

import numpy as np

from .compare import read
from .contact_phases import confirmed_release
from .eval import write_json
from .stove_burner_audit import scene_signature


def summary(values):
    x = np.asarray(values, dtype=float)
    if not x.size:
        return {"n": 0}
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite comparison")
    return {
        "n": len(x),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "max": float(x.max()),
    }


def first_run(condition, start):
    spans = np.flatnonzero(np.convolve(condition[start:].astype(int), np.ones(5, int), "valid") == 5)
    return int(spans[0] + start) if spans.size else None


def trajectories(reference, work, held):
    suffix = "-held" if held else ""
    labels = ("native", "baseline", "inertia") if held else ("native", "baseline", "inertia", "newton")
    raw = []
    hist = []
    changed_flags = []
    for task in range(10):
        for i in range(50):
            if bool(i % 10) != held:
                continue
            key = f"task-{task:02}/state-{i:02}"
            trace = read(reference / key / "trace.json")
            commands = np.asarray([r["action"] for r in trace], dtype="<f8")
            action_hash = hashlib.sha256(commands.tobytes()).hexdigest()
            initial = read(reference / key / "initial-state.json")
            row = {"case": key, "native_success_in_archive": any(r["success"] for r in trace)}
            sources = {}
            for label in labels:
                parent = work / (label + suffix) / key
                recheck = work / "reference-recheck" / key
                if held and label == "native" and (recheck / "result.json").exists():
                    parent = recheck
                meta = read(parent / "result.json")
                assert all(meta[k] == initial[k] for k in ("fixtures_sha256", "initial_state_sha256"))
                assert meta["action_sha256"] == action_hash and len(trace) * 25 == meta["ticks"]
                with np.load(parent / "trace.npz") as arr:
                    sources[label] = {k: arr[k] for k in ("bits", "ctrl", "bowl", "finger")}
                    if label == "native" and "historical_error_m" in arr:
                        hist.append(float(arr["historical_error_m"].max()))
                        if not arr["historical_flag_match"].all():
                            changed_flags.append(key)
                row[label] = dict(meta["events"], success=meta["success"])
                row[label]["release"] = confirmed_release(
                    row[label]["open"], sources[label]["bits"], np.repeat(commands[:, -1], 25)
                )
            n = sources["native"]
            for label in labels[1:]:
                m = sources[label]
                t = min(t for t in (row[label]["tip"], row["native"]["tip"], len(n["bits"])) if t is not None)
                delta = np.linalg.norm(n["bowl"][:t] - m["bowl"][:t], axis=1) * 1000
                gaps = np.flatnonzero(delta >= 1.0)
                row[label]["first_pre_tip_bowl_mm_tick"] = int(gaps[0]) if gaps.size else None
                row[label]["before_tip_bowl_mm"] = float(delta[-1]) if t else 0.0
                row[label]["max_gripper_command_delta_m"] = float(
                    np.max(abs(n["ctrl"][:, 7:] - m["ctrl"][:, 7:]))
                )
                mismatch = np.flatnonzero((n["bits"] & 3) != (m["bits"] & 3))
                row[label]["first_finger_mismatch_tick"] = int(mismatch[0]) if mismatch.size else None
            raw.append(row)
    write_json(work / ("held-cases.json" if held else "diagnosis-cases.json"), raw)
    result = {
        "historical_error_m_on_retries": summary(hist),
        "historical_flag_mismatch_on_retries": changed_flags,
    }
    for label in labels:
        r = {"success": sum(v[label]["success"] for v in raw)}
        for stage in ("tip", "grip", "lift", "open", "release", "plate"):
            r[stage] = {
                "observed": sum(v[label][stage] is not None for v in raw),
                "abs_delay_ms": summary(
                    [
                        2 * abs(v[label][stage] - v["native"][stage])
                        for v in raw
                        if None not in (v[label][stage], v["native"][stage])
                    ]
                ),
            }
        if label != "native":
            r["first_tip_tick_differs"] = sum(v[label]["tip"] != v["native"]["tip"] for v in raw)
            r["pre_tip_bowl_over_1mm"] = sum(v[label]["first_pre_tip_bowl_mm_tick"] is not None for v in raw)
            r["before_tip_bowl_mm"] = summary([v[label]["before_tip_bowl_mm"] for v in raw])
            r["max_gripper_command_delta_m"] = max(v[label]["max_gripper_command_delta_m"] for v in raw)
        result[label] = r
    result["by_task_success"] = {
        f"task-{t:02}": {
            k: sum(v[k]["success"] for v in raw if v["case"].startswith(f"task-{t:02}")) for k in labels
        }
        for t in range(10)
    }
    result["success_transitions"] = dict(
        Counter(f"{int(v['baseline']['success'])}->{int(v['inertia']['success'])}" for v in raw)
    )
    return result


def matched(work, held):
    all_metrics = defaultdict(list)
    for t in range(10):
        for i in range(int(held), 50, 10):
            rel = Path(f"task-{t:02}") / f"state-{i:02}.json"
            root = work / ("held-probes" if held else "probes")
            a, b = (read(root / e / rel) for e in ("native", "motrix"))
            assert a["ticks"] == b["ticks"]
            for phase in a["ticks"]:
                for variant in b["variants"]:
                    n = a["variants"]["original" if variant == "inertia" else variant][phase]
                    m = b["variants"][variant][phase]
                    for metric in ("finger_v", "bowl_v", "bowl_force"):
                        key = (phase, variant, metric)
                        all_metrics[key].append(float(np.linalg.norm(np.subtract(n[metric], m[metric]))))
                    if variant == "no_fingers":
                        assert not any(n["counts"][:2] + m["counts"][:2])
                    if variant == "no_bowl":
                        assert not any(n["counts"] + m["counts"])
                    pairs = [p > 0 for p in n["counts"][:2]], [p > 0 for p in m["counts"][:2]]
                    all_metrics[(phase, variant, "finger_presence_mismatch")].append(
                        int(pairs[0] != pairs[1])
                    )
    out = {}
    for (phase, variant, metric), v in all_metrics.items():
        out.setdefault(phase, {}).setdefault(variant, {})[metric] = (
            {"mismatches": sum(v), "n": len(v)} if metric == "finger_presence_mismatch" else summary(v)
        )
    return out


def tails(work):
    result = {}
    for label, official in (("native", "native"), ("motrix", "baseline"), ("inertia", "inertia")):
        observations = []
        released = []
        on_plate = 0
        for t in range(10):
            for i in range(50):
                if i % 10 not in (0, 1):
                    continue
                key = f"task-{t:02}/state-{i:02}"
                source = work / ("tail-" + label) / key
                end = read(source / "result.json")["official_ticks"]
                original = work / (official + ("-held" if i % 10 else "")) / key
                with np.load(source / "trace.npz") as f, np.load(original / "trace.npz") as ref:
                    bits = f["bits"]
                    assert len(bits) == end + 500
                    np.testing.assert_array_equal(bits[:end], ref["bits"])
                    np.testing.assert_allclose(f["ctrl"][:end], ref["ctrl"], rtol=0, atol=1e-9)
                    if bits[end - 1] & 3:
                        observations.append(key)
                        off = first_run((bits & 3) == 0, end)
                        if off is not None:
                            released.append(2 * (off - end))
                    on_plate += int(((bits[-5:] & 4) != 0).all())
        result[label] = {
            "n": 100,
            "touching_at_cutoff": len(observations),
            "release_ms": summary(released),
            "on_plate_last_five_ticks": on_plate,
        }
    return result


def inertias(work):
    n, m = (read(work / f"body-inertias-{e}.json") for e in ("native", "motrix"))
    errors = defaultdict(list)
    for task, objects in n.items():
        for name, (mass, com, tensor) in objects.items():
            for k in ("baseline", "inertia"):
                mass2, com2, tensor2 = m[task][k][name]
                for field, value in (
                    ("mass_relative", abs(mass2 / mass - 1)),
                    ("com_m", np.linalg.norm(np.subtract(com, com2))),
                    (
                        "inertia_relative",
                        np.linalg.norm(np.subtract(tensor, tensor2)) / np.linalg.norm(tensor),
                    ),
                ):
                    errors[(k, field)].append(value)
    return {
        k: {field: summary(errors[(k, field)]) for field in ("mass_relative", "com_m", "inertia_relative")}
        for k in ("baseline", "inertia")
    }


def audit_geometry(work):
    changed = Counter()
    for task in range(10):
        for index in range(50):
            key = Path(f"task-{task:02}") / f"state-{index:02}" / "case.xml"
            suffix = "-held" if index % 10 else ""
            source, actual = [ET.parse(work / (mode + suffix) / key) for mode in ("baseline", "inertia")]
            for before, after in zip(source.iter("geom"), actual.iter("geom"), strict=True):
                if before.attrib == after.attrib:
                    continue
                assert before.get("group") == "1" and after.get("mass") == "0"
                before.set("mass", "0")
                assert before.attrib == after.attrib
                changed[key.parts[0]] += 1
            assert scene_signature(source) == scene_signature(actual)
    return dict(changed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("work", type=Path)
    args = p.parse_args()
    result = {
        key: trajectories(args.reference, args.work, held)
        for key, held in (("diagnosis", False), ("held", True))
    }
    result["inertial_errors"] = inertias(args.work)
    result["verified_geom_mass_changes"] = audit_geometry(args.work)
    result["release_tail"] = tails(args.work)
    result["matched"] = matched(args.work, False)
    result["held_matched"] = matched(args.work, True)
    write_json(args.work / "summary.json", result)


if __name__ == "__main__":
    main()
