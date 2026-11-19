"""Summarize closed-loop results only across matching official initial states."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def cases(run, task, count):
    for index in range(count):
        path = run / f"task-{task:02}" / f"state-{index:02}" / "result.json"
        if path.is_file():
            yield index, read(path)


def compare(reference, motrix):
    a, b = (read(path / "run.json") for path in (reference, motrix))
    if (a["engine"] != "mujoco" or b["engine"] != "motrixsim"
            or b["mode"] != "closed_loop"):
        raise ValueError("Select a MuJoCo reference run and a MotrixSim closed-loop run")
    for key in ("suite", "tasks", "episodes", "max_steps", "environment_seed", "policy_seed",
                "weights_sha256", "model_manifest_sha256"):
        if a[key] != b[key]:
            raise ValueError(f"Mismatched run settings: {key}")
    reference_policy, motrix_policy = (read(path / "policy.json") for path in (reference, motrix))
    if reference_policy["instructions"] != motrix_policy["instructions"]:
        raise ValueError("The model received different task prompts")

    tasks = []
    for task in a["tasks"]:
        reference_cases = dict(cases(reference, task, a["episodes"]))
        motrix_cases = dict(cases(motrix, task, a["episodes"]))
        outcomes = {"both_success": 0, "mujoco_only": 0, "motrixsim_only": 0, "neither": 0}
        examples = {}
        errors = []
        for index in sorted(reference_cases.keys() & motrix_cases.keys()):
            left, right = reference_cases[index], motrix_cases[index]
            if left["initial_state_sha256"] != right["initial_state_sha256"]:
                raise ValueError(f"Initial state differs for task {task} case {index}")
            if not left.get("fixtures_sha256") or left["fixtures_sha256"] != right.get("fixtures_sha256"):
                raise ValueError(f"Fixture layout differs for task {task} case {index}")
            if any(item["termination"] not in ("horizon", "success") for item in (left, right)):
                errors.append(index)
                continue
            names = {(True, True): "both_success", (True, False): "mujoco_only",
                     (False, True): "motrixsim_only", (False, False): "neither"}
            name = names[left["success"], right["success"]]
            outcomes[name] += 1
            examples.setdefault(name, index)
        valid_pairs = sum(outcomes.values())
        tasks.append({
            "task": task, "expected_pairs": a["episodes"], "valid_pairs": valid_pairs,
            "prompt": reference_policy["instructions"][str(task)],
            "reference_completed": len(reference_cases), "motrixsim_completed": len(motrix_cases),
            "mujoco_success_rate": (outcomes["both_success"] + outcomes["mujoco_only"]) / valid_pairs
            if valid_pairs == a["episodes"] else None,
            "motrixsim_success_rate": (outcomes["both_success"] + outcomes["motrixsim_only"]) / valid_pairs
            if valid_pairs == a["episodes"] else None,
            "replay_examples": examples, "error_cases": errors, **outcomes,
        })

    report = {
        "suite": a["suite"], "mujoco_version": a.get("engine_version"),
        "motrixsim_version": b["engine_version"], "weights_sha256": b["weights_sha256"],
        "reference": str(reference), "motrixsim": str(motrix), "tasks": tasks,
        "complete": all(row["valid_pairs"] == row["expected_pairs"] for row in tasks),
    }
    if report["complete"]:
        count = sum(row["valid_pairs"] for row in tasks)
        left = sum(row["both_success"] + row["mujoco_only"] for row in tasks)
        right = sum(row["both_success"] + row["motrixsim_only"] for row in tasks)
        report.update({"pairs": count, "mujoco_successes": left, "motrixsim_successes": right,
                       "mujoco_success_rate": left / count, "motrixsim_success_rate": right / count,
                       "difference_percentage_points": 100 * (right - left) / count})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("motrix", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.reference.resolve(), args.motrix.resolve())
    result = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    print(result, end="")


if __name__ == "__main__":
    main()
