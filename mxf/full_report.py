"""Validate four complete policy streams and publish paired per-episode results."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from .eval import SUITE_LIMITS, fixture_sha256, write_json


def read(path):
    return json.loads(path.read_text())


def require(valid, message):
    if not valid:
        raise ValueError(message)


def collect(scored, scenes, reference):
    acceptance = read(reference / "acceptance.json")
    require(
        acceptance["passed"] and acceptance["full_default_budget"] and acceptance["requested_cases"] == 2000,
        "The native reference is not a completed 40×50 acceptance",
    )
    manifest = read(reference / "data" / "models" / acceptance["model_id"] / "manifest.json")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    require(
        hashlib.sha256(canonical).hexdigest() == acceptance["model_manifest_sha256"],
        "Native manifest content changed",
    )
    reference_runs = {}
    for item in acceptance["completed_suites"]:
        path = reference / "data" / "runs" / item["run_id"] / "native"
        native = read(path / "native-result.json")
        suite = native["suites"][0]["suite"]
        require(
            suite not in reference_runs
            and native["completed_cases"] == 500
            and not native["execution_errors"],
            f"Incomplete or duplicate reference {suite}",
        )
        reference_runs[suite] = (
            native,
            read(path / "policy-config.json"),
            hashlib.sha256((path / "native-result.json").read_bytes()).hexdigest(),
        )
    episodes, suites = [], {}
    for suite, horizon in SUITE_LIMITS.items():
        native, reference_policy, native_digest = reference_runs[suite]
        folder = scored / suite
        run = read(folder / "run.json")
        policy = read(folder / "policy.json")
        require(
            run["suite"] == suite
            and run["engine"] == "motrixsim"
            and run["mode"] == "closed_loop"
            and run["episodes"] == 50
            and run["tasks"] == list(range(10))
            and run["max_steps"] == horizon
            and run["osc_reset"] == "libero"
            and run["inertia_groups"]
            and not run["extend_stove_burner"]
            and run["physics_dt"] == 0.002
            and run["control_hz"] == 20
            and run["warmup_steps"] == 10
            and run["environment_seed"] == 0
            and run["policy_seed"] == 7
            and run["compact"]
            and run["motrix_solver_override"] is None
            and "policy_rng_source" not in run,
            f"Protocol drift: {suite}",
        )
        require(
            policy["model_manifest_sha256"]
            == reference_policy["model_manifest_sha256"]
            == acceptance["model_manifest_sha256"]
            == run["model_manifest_sha256"]
            and policy["vla_key"] == reference_policy["vla_key"] == "pi05-libero",
            "Checkpoint differs",
        )
        summary = read(folder / "policy-summary.json")
        require(
            run["weights_sha256"] == manifest["weights_sha256"]
            and policy["seed"] == 7
            and policy["boundary"] == reference_policy["boundary"] == "libero-raw256-suite.v1",
            f"Policy identity differs: {suite}",
        )
        signatures = read(folder / "scene-digests.json")
        require(set(signatures) == {str(i) for i in range(10)}, f"Missing scene identities: {suite}")
        require(len(list(folder.glob("task-*/state-*/result.json"))) == 500, f"Result count differs: {suite}")
        counts = Counter()
        paired = Counter()
        tasks = []
        for i, task in enumerate(native["tasks"]):
            task_name = f"task-{i:02}"
            scene = scenes / suite / task_name
            meta = read(scene / "scene.json")
            require(
                signatures[str(i)]
                == {
                    n: hashlib.sha256((scene / n).read_bytes()).hexdigest()
                    for n in ("scene.xml", "scene.json")
                },
                f"Scene changed since scoring: {suite}/{task_name}",
            )
            require(
                all(
                    k in meta
                    for k in ("contact_reference", "free_damping", "fluid", "planar_meshes", "indicators")
                ),
                f"Missing scene adapter inputs: {suite}/{task_name}",
            )
            require(
                task["native_task_id"] == i
                and task["suite"] == suite
                and len(task["cases"]) == 50
                and policy["instructions"][str(i)]
                == reference_policy["instructions"][task["task_id"]]
                == meta["instruction"],
                f"Task identity differs: {suite}/{task_name}",
            )
            start = len(episodes)
            for j, expected in enumerate(task["cases"]):
                relative = Path(task_name) / f"state-{j:02}"
                state = read(scenes / suite / relative / "initial-state.json")
                row = read(folder / relative / "result.json")
                require(
                    expected["episode_index"] == expected["init_state_index"] == j
                    and state["initial_state_sha256"]
                    == row["initial_state_sha256"]
                    == expected["initial_state_sha256"]
                    and fixture_sha256(state["fixtures"])
                    == state["fixtures_sha256"]
                    == row["fixtures_sha256"],
                    f"Episode identity differs: {suite}/{relative}",
                )
                steps = row["steps"]
                require(
                    type(row["success"]) is bool
                    and row["termination"] == ("success" if row["success"] else "horizon")
                    and 1 <= steps <= horizon
                    and (row["success"] or steps == horizon)
                    and row["policy_forwards"] == (steps + 4) // 5
                    and row["osc_reset_source"] == "exported"
                    and row["camera_asset_warmup_seconds"] >= 2
                    and all(len(bytes.fromhex(row[key])) == 32 for key in ("action_sha256", "query_sha256")),
                    f"Invalid scored episode: {suite}/{relative}",
                )
                episodes.append(
                    {
                        "suite": suite,
                        "task_index": i,
                        "task": task["task_id"],
                        "initial_state_index": j,
                        **{
                            key: row[key]
                            for key in (
                                "initial_state_sha256",
                                "fixtures_sha256",
                                "success",
                                "steps",
                                "termination",
                                "policy_forwards",
                                "query_sha256",
                                "action_sha256",
                            )
                        },
                        "reference_success": expected["success"],
                        "reference_steps": expected["steps"],
                        "reference_termination": expected["termination"],
                    }
                )
                paired[(row["success"], expected["success"])] += 1
                counts.update(steps=steps, forwards=row["policy_forwards"])
            cases = episodes[start:]
            tasks.append(
                {
                    "task_index": i,
                    "task": task["task_id"],
                    "instruction": meta["instruction"],
                    "episodes": 50,
                    "scene_sha256": signatures[str(i)],
                    "successes": sum(x["success"] for x in cases),
                    "reference_successes": sum(x["reference_success"] for x in cases),
                }
            )
        require(
            summary["algorithm_resets"] == 500
            and summary["action_batches"] == counts["steps"]
            and summary["model_forwards"] == counts["forwards"]
            and summary["policy_seeded_once"]
            and summary["inference_order"] == "serial_task_episode_action",
            f"Stream counters differ: {suite}",
        )
        suites[suite] = {
            "tasks": tasks,
            "episodes": 500,
            "successes": sum(t["successes"] for t in tasks),
            "reference_successes": sum(t["reference_successes"] for t in tasks),
            "paired_outcomes": {
                "both_success": paired[True, True],
                "motrix_only": paired[True, False],
                "reference_only": paired[False, True],
                "both_timeout": paired[False, False],
            },
            "policy_summary": summary,
            "max_steps": horizon,
            "native_result_sha256": native_digest,
            "motrixsim_version": run["engine_version"],
            "episodes_per_simulation_process": run.get("episodes_per_process"),
            "weights_sha256": run["weights_sha256"],
        }
    require(len(episodes) == 2000, "Incomplete campaign")
    return episodes, {
        "protocol": acceptance["protocol_id"],
        "model_manifest_sha256": acceptance["model_manifest_sha256"],
        "reference_execution": acceptance["execution_id"],
        "weights_sha256": manifest["weights_sha256"],
        "checkpoint_source": manifest["source"],
        "policy_runtime_sha256": manifest["runtime"]["code_sha256"],
        "protocol_settings": {
            "policy_seed": 7,
            "environment_seed": 0,
            "physics_dt": 0.002,
            "control_hz": 20,
            "warmup_steps": 10,
            "action_queue": 5,
            "camera_resolution": [256, 256],
            "episodes_per_task": 50,
            "inference_order": "serial_task_episode_action",
            "solver_override": None,
            "engine": "motrixsim",
            "reference_engine": "mujoco",
            "original_collision_geometry": True,
        },
        "episodes": 2000,
        "successes": sum(r["success"] for r in episodes),
        "reference_successes": sum(r["reference_success"] for r in episodes),
        "suites": suites,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("scored", "scenes", "reference", "output"):
        p.add_argument(key, type=Path)
    args = p.parse_args()
    episodes, summary = collect(args.scored, args.scenes, args.reference)
    args.output.mkdir(parents=True, exist_ok=True)
    tmp = args.output / "episodes.tmp"
    with tmp.open("w") as f:
        for row in episodes:
            print(json.dumps(row, ensure_ascii=False, separators=(",", ":")), file=f)
    tmp.replace(args.output / "episodes.jsonl")
    write_json(args.output / "results.json", summary)
    lines = [
        "# MotrixSim 社区版 π0.5 LIBERO 40×50 评测",
        "",
        f"MotrixSim 完成 2,000 个闭环回合，成功 {summary['successes']}/2,000（{summary['successes'] / 20:.2f}%）。"
        f"平台归档的同权重 MuJoCo 参考为 {summary['reference_successes']}/2,000（{summary['reference_successes'] / 20:.2f}%）。",
        "",
        "| 套件 | 每回合动作上限 | MotrixSim | MuJoCo 参考 | 成功率差 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, s in summary["suites"].items():
        a, b = s["successes"], s["reference_successes"]
        lines.append(
            f"| {name} | {s['max_steps']} | {a}/500（{a / 5:.1f}%） | {b}/500（{b / 5:.1f}%） | {(a - b) / 5:+.1f} 个百分点 |"
        )
    lines += [
        "",
        "四套件独立初始化，套件内依次执行任务 0–9 和初态 0–49。策略种子 7，环境种子 0。"
        "每回合先开爪预热 10 步，随后每 5 步至多调用一次 π0.5。两路 256×256 RGB 输入由 MotrixSim 渲染，动作控制为 20 Hz，物理步长为 2 ms。每个控制步检查原生任务谓词，成功立即终止。",
        "",
        "参考来自 sim-platform 已封存的 2,000 回合。逐回合配对核对初态，逐套件核对模型清单、指令和调用计数。"
        "两引擎各自闭环运行，其动作与后续轨迹会分化。成绩适用于本次 MotrixSim 移植、模型和平台约定的评测协议。",
        "",
        "## 相同初态的结果分布",
        "",
        "| 套件 | 两端均成功 | 仅 MotrixSim 成功 | 仅 MuJoCo 成功 | 两端均超时 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, suite in summary["suites"].items():
        paired = suite["paired_outcomes"]
        lines.append(
            f"| {name} | {paired['both_success']} | {paired['motrix_only']} | "
            f"{paired['reference_only']} | {paired['both_timeout']} |"
        )
    lines += [
        "",
        "## 逐任务成绩",
        "",
        "| 套件 | 任务 | MotrixSim | MuJoCo 参考 |",
        "|---|---|---:|---:|",
    ]
    for suite, s in summary["suites"].items():
        for t in s["tasks"]:
            lines.append(
                f"| {suite} | {t['task_index']:02} · {t['instruction']} | {t['successes']}/50 | {t['reference_successes']}/50 |"
            )
    lines += [
        "",
        "[逐回合结果](episodes.jsonl) · [结构化汇总](results.json) · [物理和渲染验证记录](validation.md)",
        "",
    ]
    records = Path(__file__).resolve().parents[1] / "results" / "libero-pi05"
    for name in ("validation.md", "validation-history.md"):
        source = records / name
        dest = args.output / name
        if source.resolve() != dest.resolve():
            shutil.copyfile(source, dest)
    (args.output / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
