"""Audit paired rollout evidence and write a human-readable Spatial report."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from urllib.parse import quote
import xml.etree.ElementTree as ET

import imageio_ffmpeg
import numpy as np

from .compare import cases, compare, read
from .eval import WARMUP, fixture_sha256, write_json
from .policy import DEFAULT_PLATFORM


def audit_video(path, expected_frames):
    output = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-xerror", "-nostdin",
         "-threads", "1", "-i", str(path), "-map", "0:v:0", "-an",
         "-threads", "1", "-f", "framehash", "-"],
        capture_output=True, text=True, timeout=30,
    )
    if output.returncode:
        raise ValueError(f"{path}: video decode failed: {output.stderr.strip()}")
    rows = output.stdout.splitlines()
    frames = sum(row.startswith("0,") for row in rows)
    if "#dimensions 0: 256x256" not in rows or "#tb 0: 1/20" not in rows or frames != expected_frames:
        raise ValueError(f"{path}: unexpected video dimensions, rate or frame count ({frames})")
    return frames


def audit_episode(reference, task_number, index, root, maximum, camera_warmup_min):
    task = f"task-{task_number:02}"
    case = f"state-{index:02}"
    source = reference / task / case
    folder = root / task / case
    initial = read(source / "initial-state.json")
    result = read(folder / "result.json")
    sha = result["initial_state_sha256"]
    if (len(sha) != 64 or sha != initial["initial_state_sha256"]
            or result["fixtures_sha256"] != fixture_sha256(initial["fixtures"])):
        raise ValueError(f"{folder}: initial state or fixture hash differs")
    if root != reference:
        bodies = {body.get("name"): body for body in ET.parse(folder / "case.xml").iter("body")}
        for name, pose in initial["fixtures"].items():
            body = bodies[name]
            if ([float(value) for value in body.attrib["pos"].split()] != pose["position"]
                    or [float(value) for value in body.attrib["quat"].split()] != pose["quaternion_wxyz"]):
                raise ValueError(f"{folder}: compiled fixture placement differs for {name}")
    if camera_warmup_min is not None:
        warmup_seconds = result.get("camera_asset_warmup_seconds")
        if warmup_seconds is None or not math.isfinite(warmup_seconds) or warmup_seconds < camera_warmup_min:
            raise ValueError(f"{folder}: no settled camera readback before physics warmup")
    score_steps = result["steps"]
    success = result["success"]
    if (type(score_steps) is not int or not 1 <= score_steps <= maximum
            or type(success) is not bool
            or result["termination"] != ("success" if success else "horizon")
            or (not success and score_steps != maximum)):
        raise ValueError(f"{folder}: invalid episode termination")

    trace = read(folder / "trace.json")
    steps = [row["step"] for row in trace]
    if steps != list(range(-9, score_steps + 1)):
        raise ValueError(f"{folder}: trace steps are missing or out of order")
    actions = np.asarray([row["action"] for row in trace])
    eef = np.asarray([row["eef"] for row in trace])
    if (actions.shape != (len(trace), 7) or not np.isfinite(actions).all()
            or np.any(np.abs(actions) > 1.000001) or not np.allclose(actions[:10], WARMUP)
            or eef.shape != (len(trace), 3) or not np.isfinite(eef).all()
            or [row["success"] for row in trace] != [False] * (len(trace) - 1) + [success]):
        raise ValueError(f"{folder}: invalid actions, end-effector positions or success flags")

    queries = list(range(1, score_steps + 1, 5))
    with np.load(folder / "policy-inputs.npz", allow_pickle=False) as inputs:
        if inputs["step"].tolist() != queries:
            raise ValueError(f"{folder}: incorrect five-step policy query schedule")
        for field, width in (("robot0_eef_pos", 3), ("robot0_eef_quat", 4),
                             ("robot0_gripper_qpos", 2)):
            value = inputs[field]
            if value.dtype != np.float32 or value.shape != (len(queries), width) or not np.isfinite(value).all():
                raise ValueError(f"{folder}: invalid model input {field}")
        for field in ("agentview_image", "robot0_eye_in_hand_image"):
            value = inputs[field]
            if value.dtype != np.uint8 or value.shape != (len(queries), 256, 256, 3):
                raise ValueError(f"{folder}: invalid model input {field}")
    frames = audit_video(folder / "replay.mp4", score_steps + int(success))
    return {"frames": frames, "steps": score_steps, **{
        name: result[name] for name in ("environment_seconds", "policy_rpc_seconds",
                                       "physics_and_control_seconds", "observation_and_render_seconds")
        if name in result
    }}


def audit(reference, motrix, *, partial=False, workers=4):
    comparison = compare(reference, motrix)
    if not partial and not comparison["complete"]:
        raise ValueError("Full audit requires a complete paired comparison")
    run = read(reference / "run.json")
    details = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for engine, root in (("mujoco", reference), ("motrixsim", motrix)):
            camera_warmup_min = read(root / "run.json").get("camera_asset_warmup_min_seconds")
            jobs = {(task, index): pool.submit(audit_episode, reference, task, index, root,
                                              run["max_steps"], camera_warmup_min)
                    for task in run["tasks"] for index, result in cases(root, task, run["episodes"])
                    if result["termination"] in ("success", "horizon")}
            if not jobs:
                raise ValueError(f"No completed {engine} rollout evidence")
            results = {key: job.result() for key, job in jobs.items()}
            details[engine] = {
                "verified_replays": len(results),
                "decoded_frames": sum(row["frames"] for row in results.values()),
                **{field + "_median_ms_per_control_step": 1000 * statistics.median(
                    row[field] / (row["steps"] if field == "policy_rpc_seconds" else row["steps"] + 10)
                    for row in results.values() if field in row)
                   for field in ("environment_seconds", "policy_rpc_seconds",
                                 "physics_and_control_seconds", "observation_and_render_seconds")
                   if all(field in row for row in results.values())},
            }
    return {"comparison": comparison, "rollout_evidence": details,
            "max_scored_steps": run["max_steps"],
            "camera_asset_warmup_min_seconds": read(motrix / "run.json").get("camera_asset_warmup_min_seconds"),
            "policy_config_sha256": {
                engine: hashlib.sha256((root / "policy.json").read_bytes()).hexdigest()
                for engine, root in (("mujoco", reference), ("motrixsim", motrix))
            }}


def report_text(result, output):
    report = result["comparison"]
    lines = ["# LIBERO Spatial on MotrixSim", ""]
    if report["complete"]:
        lines.extend([
            f"完成 **{report['pairs']} 对**评测。MuJoCo 成功 {report['mujoco_successes']} 次"
            f"（{report['mujoco_success_rate']:.1%}），MotrixSim 成功 {report['motrixsim_successes']} 次"
            f"（{report['motrixsim_success_rate']:.1%}），相差 {report['difference_percentage_points']:+.1f} 个百分点。",
            "",
        ])
    else:
        lines.extend(["当前为进行中的证据审计。汇总成功率在全部配对完成后生成。", ""])
    lines.extend([
        f"引擎版本为 MuJoCo {report['mujoco_version']} 和 MotrixSim {report['motrixsim_version']}。"
        "两侧复用 sim platform 的 π0.5 LIBERO 模型，使用相同提示词、官方保存初态、柜子和灶台位姿。",
        f"已从 LIBERO 原始任务目录复核 {result['reference_evidence']['checked_states']} 个初态"
        f"（源码 `{result['reference_evidence']['libero_commit']}`）。",
        "评测以 20 Hz 运行，前 10 步预热，每 5 个评分动作重新推理，"
        f"最多执行 {result['max_scored_steps']} 个评分动作。",
        "",
        "| Spatial 任务 | 有效配对 | MuJoCo 成功率 | MotrixSim 成功率 | 两侧成功 | 仅 MuJoCo 成功 | 仅 MotrixSim 成功 | 两侧超时 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for task in report["tasks"]:
        rates = [f"{task[key]:.1%}" if task[key] is not None else "—"
                 for key in ("mujoco_success_rate", "motrixsim_success_rate")]
        lines.append(f"| {task['task']:02} | {task['valid_pairs']}/{task['expected_pairs']} | "
                     f"{rates[0]} | {rates[1]} | {task['both_success']} | "
                     f"{task['mujoco_only']} | {task['motrixsim_only']} | {task['neither']} |")
    warmup = result["camera_asset_warmup_min_seconds"]
    if warmup is not None:
        lines.extend(["", f"每个 MotrixSim 回合先在静止初态下等待至少 {warmup:g} 秒，"
                      "确认双相机画面稳定后再开始物理预热。"])
    lines.extend(["", "| 引擎 | 已解码回放 | 帧数 | 环境/步中位数 | 策略 RPC/评分步中位数 |",
                  "|---|---:|---:|---:|---:|"])
    for engine, evidence in result["rollout_evidence"].items():
        lines.append(f"| {engine} | {evidence['verified_replays']} | {evidence['decoded_frames']} | "
                     f"{evidence['environment_seconds_median_ms_per_control_step']:.1f} ms | "
                     f"{evidence['policy_rpc_seconds_median_ms_per_control_step']:.1f} ms |")
    motrix_timing = result["rollout_evidence"]["motrixsim"]
    control = motrix_timing.get("physics_and_control_seconds_median_ms_per_control_step")
    observation = motrix_timing.get("observation_and_render_seconds_median_ms_per_control_step")
    if control is not None and observation is not None:
        lines.extend(["", f"MotrixSim 每步物理与控制中位数为 {control:.1f} ms，"
                      f"观测与渲染为 {observation:.1f} ms。"])
    lines.extend(["", "环境时间包括控制、物理和渲染，策略 RPC 时间含传输与模型推理。"
                  "结果包含引擎之间的动力学与图像差异。"])
    lines.extend(["", "| Spatial 任务提示词 | 配对结果 | 初态 | MuJoCo | MotrixSim |",
                  "|---|---|---:|---|---|"])
    labels = {"both_success": "两侧成功", "mujoco_only": "仅 MuJoCo 成功",
              "motrixsim_only": "仅 MotrixSim 成功", "neither": "两侧超时"}
    for task in report["tasks"]:
        for outcome, state in task["replay_examples"].items():
            case = Path(f"task-{task['task']:02}") / f"state-{state:02}" / "replay.mp4"
            links = [f"[回放]({quote(os.path.relpath(Path(report[engine]) / case, output))})"
                     for engine in ("reference", "motrixsim")]
            prompt = task["prompt"].replace("|", r"\|")
            lines.append(f"| {task['task']:02} {prompt} | {labels[outcome]} | {state:02} | "
                         f"{links[0]} | {links[1]} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("motrix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", type=Path, default=Path(os.environ.get("MXF_PLATFORM", DEFAULT_PLATFORM)))
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    result = audit(args.reference.resolve(), args.motrix.resolve(), partial=args.partial)
    args.output.mkdir(parents=True, exist_ok=True)
    platform = args.platform.resolve()
    source_evidence = args.output / "reference-evidence.json"
    command = [str(platform / ".runtimes/envs/libero/bin/python"), "-m", "mxf.reference_evidence",
               str(args.reference.resolve()), "--source", str(platform / ".runtimes/sources/libero-source"),
               "--output", str(source_evidence.resolve())]
    if args.partial:
        command.append("--partial")
    subprocess.run(command, check=True, timeout=60)
    result["reference_evidence"] = read(source_evidence)
    write_json(args.output / "audit.json", result)
    (args.output / "report.md").write_text(report_text(result, args.output))
    for engine, evidence in result["rollout_evidence"].items():
        print(engine, evidence["verified_replays"], "videos", evidence["decoded_frames"], "frames")


if __name__ == "__main__":
    main()
