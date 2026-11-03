#!/usr/bin/env python3
"""Sample HiPhi and export T-pose candidates and two review examples.

This is an analysis tool: it never changes source motions. Candidate
times are threshold crossings, not manually annotated ground truth. Joint and
body indices follow data/hiphi/README.md (mviz order, not MuJoCo order).
With --export-review, export conservative T-pose holds only after verifying
stable, matching standing poses both before and after the full arm cycle.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.spatial.transform import Rotation


EXAMPLES = (
    "Body_movement/shiver/Body_movement-shiver_0039/motion_actor.npz",
    "Self_motion/hop/Self_motion-hop_0029/motion_actor.npz",
)
ROLL_INDICES = [15, 16]


def runs(mask):
    """Return zero-based half-open intervals [start, end)."""
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def load_features(path):
    with np.load(path, allow_pickle=False) as data:
        fps = float(data["fps"].item())
        q = np.rad2deg(data["joint_pos"])
        pos = data["body_pos_w"]
        quat = data["body_quat_w"][:, 9, :]
        velocity = data["body_lin_vel_w"]
        root_speed = np.linalg.norm(velocity[:, 0, :2], axis=1)
        foot_speed = np.linalg.norm(velocity[:, [18, 19]], axis=2)
    torso = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
    up = torso.apply(np.broadcast_to([0., 0., 1.], (len(q), 3)))
    tilt = np.rad2deg(np.arccos(np.clip(up[:, 2], -1, 1)))
    # Upper-arm direction in the torso frame, independent of world heading.
    upper_arm = np.stack([
        torso.inv().apply(pos[:, 22] - pos[:, 12]),
        torso.inv().apply(pos[:, 23] - pos[:, 13]),
    ], axis=1)
    upper_arm /= np.linalg.norm(upper_arm, axis=2, keepdims=True).clip(1e-8)
    forearm = np.stack([
        torso.inv().apply(pos[:, 28] - pos[:, 22]),
        torso.inv().apply(pos[:, 29] - pos[:, 23]),
    ], axis=1)
    forearm /= np.linalg.norm(forearm, axis=2, keepdims=True).clip(1e-8)
    # G1 elbow joint zero is not an anatomically straight elbow. Use geometry
    # to measure arm alignment; zero deflection means shoulder/elbow/wrist align.
    elbow_deflection = np.rad2deg(np.arccos(np.clip(
        np.sum(upper_arm * forearm, axis=2), -1, 1)))
    return dict(fps=fps, q=q, pos=pos, torso=torso, root_speed=root_speed,
                tilt=tilt, upper_arm=upper_arm, elbow_deflection=elbow_deflection,
                foot_speed=foot_speed)


def analyze(path, high_deg, low_deg, min_seconds):
    f = load_features(path)
    q, fps = f["q"], f["fps"]
    roll = q[:, ROLL_INDICES] * np.array([1., -1.])
    window = max(3, int(round(.5 * fps)) | 1)
    variation = maximum_filter1d(roll, window, axis=0, mode="nearest") - minimum_filter1d(
        roll, window, axis=0, mode="nearest")
    # Supporting evidence only: not yet a validated automatic deletion rule.
    support = ((variation < 10).all(axis=1)
               & (np.abs(roll[:, 0] - roll[:, 1]) < 25)
               & (f["root_speed"] < .10)
               & (f["tilt"] < 30)
               & (f["pos"][:, 0, 2] > .60)
               & (f["upper_arm"][:, 0, 1] > .85)
               & (f["upper_arm"][:, 1, 1] < -.85))
    candidates = []
    for start, end in runs((roll > high_deg).all(axis=1)):
        if (end - start) / fps < min_seconds:
            continue
        lo, hi = start, end
        while lo > 0 and np.max(roll[lo - 1]) > low_deg:
            lo -= 1
        while hi < len(q) and np.max(roll[hi]) > low_deg:
            hi += 1
        support_runs = runs(support[start:end])
        longest = max((b - a for a, b in support_runs), default=0) / fps
        candidates.append(dict(
            core_frames=[int(start), int(end)],
            core_seconds=[start / fps, end / fps],
            transition_frames=[int(lo), int(hi)],
            transition_seconds=[lo / fps, hi / fps],
            low_pose_before=bool(lo > 0), low_pose_after=bool(hi < len(q)),
            median_roll_deg=np.median(q[start:end, ROLL_INDICES], axis=0).tolist(),
            median_elbow_deg=np.median(q[start:end, [21, 22]], axis=0).tolist(),
            median_geometric_elbow_deflection_deg=np.median(
                f["elbow_deflection"][start:end], axis=0).tolist(),
            median_root_xy_speed_m_s=float(np.median(f["root_speed"][start:end])),
            median_torso_tilt_deg=float(np.median(f["tilt"][start:end])),
            median_upper_arm_lateral_cosine=np.median(
                f["upper_arm"][start:end, :, 1], axis=0).tolist(),
            longest_support_seconds=longest,
        ))
    return dict(fps=fps, frames=len(q), duration_seconds=len(q) / fps,
                candidates=candidates)


PRECISION_RULES = {
    "core_roll_min_deg": 60., "standing_roll_max_deg": 40.,
    "standing_min_seconds": .4, "standing_search_seconds": 1.5,
    "variation_window_seconds": .4, "standing_roll_range_deg": 10.,
    "standing_tilt_max_deg": 20., "standing_root_xy_speed_max_m_s": .12,
    "foot_speed_max_m_s": .12, "foot_height_max_m": .10,
    "foot_height_difference_max_m": .04,
    "pelvis_height_range_m": [.68, .90], "knee_abs_max_deg": 20.,
    "hip_pitch_abs_max_deg": 30., "standing_upper_arm_down_cosine": .60,
    "matching_root_distance_max_m": .10, "matching_leg_angle_max_deg": 10.,
    "matching_torso_rotation_max_deg": 12.,
    "hold_roll_range_deg": [70., 110.], "hold_roll_asymmetry_max_deg": 15.,
    "hold_upper_arm_lateral_cosine": .95, "hold_elbow_deflection_max_deg": 20.,
    "hold_tilt_max_deg": 25., "hold_root_xy_speed_max_m_s": .08,
    "hold_roll_window_range_max_deg": 8., "hold_min_seconds": .8,
    "transition_max_seconds": 1.5,
}


def precision_candidates(path, *, features=None):
    """Reject ambiguous cycles; both standing contexts are mandatory.

    Return the longest uninterrupted strict hold within each accepted cycle.
    Exporting the hold keeps every output frame in the T-pose mask; the full
    cycle and standing witnesses are recorded separately for review.
    """
    f = load_features(path) if features is None else features
    q, p, fps = f["q"], f["pos"], f["fps"]
    cfg = PRECISION_RULES
    roll = q[:, ROLL_INDICES] * [1., -1.]
    window = max(3, round(cfg["variation_window_seconds"] * fps) | 1)
    variation = maximum_filter1d(roll, window, axis=0, mode="nearest") - minimum_filter1d(
        roll, window, axis=0, mode="nearest")
    height_min, height_max = cfg["pelvis_height_range_m"]
    upright_legs = ((p[:, 0, 2] > height_min) & (p[:, 0, 2] < height_max)
                    & (np.abs(q[:, [9, 10]]) < cfg["knee_abs_max_deg"]).all(axis=1)
                    & (np.abs(q[:, [0, 1]]) < cfg["hip_pitch_abs_max_deg"]).all(axis=1))
    planted = ((f["foot_speed"] < cfg["foot_speed_max_m_s"]).all(axis=1)
               & (np.abs(p[:, [18, 19], 2]) < cfg["foot_height_max_m"]).all(axis=1)
               & (np.abs(p[:, 18, 2] - p[:, 19, 2])
                  < cfg["foot_height_difference_max_m"]))
    standing = (upright_legs & planted & (roll >= 0).all(axis=1)
                & (roll <= cfg["standing_roll_max_deg"]).all(axis=1)
                & (f["upper_arm"][:, :, 2] < -cfg["standing_upper_arm_down_cosine"]).all(axis=1)
                & (f["tilt"] < cfg["standing_tilt_max_deg"])
                & (f["root_speed"] < cfg["standing_root_xy_speed_max_m_s"])
                & (variation < cfg["standing_roll_range_deg"]).all(axis=1))
    minimum_stand = int(np.ceil(cfg["standing_min_seconds"] * fps))
    standing_runs = [(a, b) for a, b in runs(standing) if b - a >= minimum_stand]
    hold_min, hold_max = cfg["hold_roll_range_deg"]
    holding = (upright_legs & planted & (roll > hold_min).all(axis=1)
               & (roll < hold_max).all(axis=1)
               & (np.abs(roll[:, 0] - roll[:, 1]) < cfg["hold_roll_asymmetry_max_deg"])
               & (f["upper_arm"][:, 0, 1] > cfg["hold_upper_arm_lateral_cosine"])
               & (f["upper_arm"][:, 1, 1] < -cfg["hold_upper_arm_lateral_cosine"])
               & (f["elbow_deflection"] < cfg["hold_elbow_deflection_max_deg"]).all(axis=1)
               & (f["tilt"] < cfg["hold_tilt_max_deg"])
               & (f["root_speed"] < cfg["hold_root_xy_speed_max_m_s"])
               & (variation < cfg["hold_roll_window_range_max_deg"]).all(axis=1))
    accepted = []
    search = round(cfg["standing_search_seconds"] * fps)
    max_transition = round(cfg["transition_max_seconds"] * fps)
    leg_indices = [0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18]
    for core_start, core_end in runs((roll > cfg["core_roll_min_deg"]).all(axis=1)):
        if (core_end - core_start) / fps < cfg["hold_min_seconds"]:
            continue
        lo, hi = core_start, core_end
        while lo > 0 and np.max(roll[lo - 1]) > cfg["standing_roll_max_deg"]:
            lo -= 1
        while hi < len(q) and np.max(roll[hi]) > cfg["standing_roll_max_deg"]:
            hi += 1
        if core_start - lo > max_transition or hi - core_end > max_transition:
            continue
        before = [(a, b) for a, b in standing_runs if b <= lo and lo - b <= search]
        after = [(a, b) for a, b in standing_runs if a >= hi and a - hi <= search]
        if not before or not after:
            continue
        # Compare nearby fixed-length witnesses, never distant unrelated stands.
        _, before_end = max(before, key=lambda interval: interval[1])
        after_start, _ = min(after)
        bs, be = before_end - minimum_stand, before_end
        ats, ate = after_start, after_start + minimum_stand
        root_distance = float(np.linalg.norm(np.median(p[bs:be, 0], axis=0)
                                             - np.median(p[ats:ate, 0], axis=0)))
        leg_difference = float(np.max(np.abs(np.median(q[bs:be, leg_indices], axis=0)
                                             - np.median(q[ats:ate, leg_indices], axis=0))))
        pre_rotation = f["torso"][bs:be].mean()
        post_rotation = f["torso"][ats:ate].mean()
        rotation_difference = float(np.rad2deg((pre_rotation.inv() * post_rotation).magnitude()))
        if (root_distance >= cfg["matching_root_distance_max_m"]
                or leg_difference >= cfg["matching_leg_angle_max_deg"]
                or rotation_difference >= cfg["matching_torso_rotation_max_deg"]):
            continue
        # Arms may move during the cycle, but legs must stay planted and upright.
        if not (upright_legs[bs:ate] & planted[bs:ate]
                & (f["tilt"][bs:ate] < cfg["hold_tilt_max_deg"])).all():
            continue
        intervals = [(core_start + a, core_start + b)
                     for a, b in runs(holding[core_start:core_end])
                     if (b - a) / fps >= cfg["hold_min_seconds"]]
        if not intervals:
            continue
        start, end = max(intervals, key=lambda interval: interval[1] - interval[0])
        accepted.append(dict(
            fps=fps, source_frames=len(q), export_frames=[start, end],
            export_seconds=[start / fps, end / fps], duration_seconds=(end - start) / fps,
            cycle_frames=[lo, hi], cycle_seconds=[lo / fps, hi / fps],
            before_standing_frames=[bs, be], before_standing_seconds=[bs / fps, be / fps],
            after_standing_frames=[ats, ate], after_standing_seconds=[ats / fps, ate / fps],
            standing_root_distance_m=root_distance, standing_leg_difference_deg=leg_difference,
            standing_torso_rotation_difference_deg=rotation_difference,
            median_roll_deg=np.median(q[start:end, ROLL_INDICES], axis=0).tolist(),
            max_geometric_elbow_deflection_deg=float(f["elbow_deflection"][start:end].max()),
            max_root_xy_speed_m_s=float(f["root_speed"][start:end].max()),
        ))
    return accepted


def draw_review_pose(ax, f, frame):
    forward = f["torso"][frame].apply([1., 0., 0.])
    yaw = np.arctan2(forward[1], forward[0])
    pos = Rotation.from_rotvec([0., 0., -yaw]).apply(f["pos"][frame] - f["pos"][frame, 0])
    chains = [[0, 3, 6, 9], [0, 1, 4, 7, 10, 14, 18],
              [0, 2, 5, 8, 11, 15, 19], [9, 12, 16, 20, 22, 24, 26, 28],
              [9, 13, 17, 21, 23, 25, 27, 29]]
    for chain, color in zip(chains, ["black", "tab:blue", "tab:orange", "tab:blue", "tab:orange"]):
        ax.plot(pos[chain, 1], pos[chain, 2], "o-", color=color, markersize=2, lw=1.5)
    ax.set_aspect("equal")
    ax.set_xlim(-.65, .65)
    ax.set_ylim(-.85, .65)
    ax.grid(alpha=.2)


def plot_precision_review(path, record, output):
    f = load_features(path)
    fps = f["fps"]
    bs, be = record["before_standing_frames"]
    ats, ate = record["after_standing_frames"]
    start, end = record["export_frames"]
    frames = [(bs + be) // 2, start, (start + end) // 2, end - 1, (ats + ate) // 2]
    titles = ["Standing before", "Export first", "T-pose hold", "Export last", "Standing after"]
    fig = plt.figure(figsize=(14, 6))
    grid = fig.add_gridspec(2, 5, height_ratios=[2, 1])
    for i, (frame, title) in enumerate(zip(frames, titles)):
        ax = fig.add_subplot(grid[0, i])
        draw_review_pose(ax, f, frame)
        ax.set_title(f"{title}\n{frame / fps:.2f}s", fontsize=10)
    ax = fig.add_subplot(grid[1, :])
    t = np.arange(bs, ate) / fps
    ax.plot(t, f["q"][bs:ate, 15], label="Left outward roll")
    ax.plot(t, -f["q"][bs:ate, 16], label="Right outward roll")
    ax.axvspan(bs / fps, be / fps, color="green", alpha=.18)
    ax.axvspan(ats / fps, ate / fps, color="green", alpha=.18)
    ax.axvspan(start / fps, end / fps, color="red", alpha=.18)
    ax.set(xlabel="Original motion time (s)", ylabel="Outward roll (deg)")
    ax.legend(loc="lower center", ncol=2, fontsize=8)
    ax.grid(alpha=.2)
    fig.suptitle(f"{record['id']:02d}: {path.parent.name} | green: standing witnesses; red: exported frames")
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def write_precision_readme(output, payload):
    clips = payload["clips"]
    lines = [
        "# 高精度 T-pose 人工复核样本", "",
        f"扫描 {payload['scanned_files']} 条不含镜像的原始动作，"
        f"有 {payload['accepted_source_files']} 个源文件通过严格规则；"
        f"从中选取 {len(clips)} 个不同源文件，覆盖 "
        f"{len({c['source'].split('/')[0] for c in clips})} 个大类。", "",
        "## 文件内容", "",
        "每个 NPZ **只包含连续的 T-pose 保持段**，不含前后站立和抬放臂过渡。"
        "所有字段、FPS、dtype、坐标和速度均保留原始数值，仅按时间维切片；"
        "没有插值、重定向、补帧或姿态合成。已逐字段核对与原始切片完全一致。", "",
        "前后站立是接受该片段的必要条件，证据保存在每条对应的 `_context.png`："
        "五个快照依次为前站立、导出首帧、保持中点、导出末帧、后站立；"
        "下方曲线中绿色标记前后站立窗口，红色标记 NPZ 实际导出区间。", "",
        "所有时间和帧索引均相对于原文件，采用从 0 开始的左闭右开区间。"
        "NPZ 播放时从截取片段的第 0 帧开始。", "",
        "## 必须同时满足的规则", "",
        "1. 前后各有连续至少 0.4 秒的稳定站立：双肩外展 0°～40°、"
        "上臂朝下、躯干倾斜 <20°、膝角绝对值 <20°、髋 pitch 绝对值 <30°、"
        "pelvis 高度 0.68～0.90 m、root 平面速度 <0.12 m/s。",
        "2. 双脚接近地面、速度均 <0.12 m/s；站立窗口内肩角的 0.4 秒滑动极差 <10°。"
        "前后站立窗口距完整抬放臂周期的边界均不超过 1.5 秒。",
        "3. 前后站立姿态匹配：root 中位位置距离 <0.10 m、"
        "各腿部关节中位角之差 <10°、躯干平均旋转差 <12°。"
        "从前站立窗口到后站立窗口的全过程要求双脚着地、腿部直立且躯干倾斜 <25°。",
        "4. 双肩外展均超过 60°的核心段必须位于上述前后站立之间；"
        "向外扩展至双肩均回到 40°以内，抬臂/放臂边缘各不超过 1.5 秒。",
        "5. 实际导出区间至少连续 0.8 秒，每一帧均满足：双肩外展 70°～110°、"
        "左右差 <15°、上臂与各自侧向的方向余弦 >0.95、几何肘偏折 <20°、"
        "root 平面速度 <0.08 m/s、双肩各自的 0.4 秒滑动极差 <8°，"
        "并满足直立、双脚着地条件。每个合格周期只取最长的连续保持段。", "",
        "刻意采用保守条件：不满足前后完整站立过程的片段直接跳过，"
        "不为凑数量放宽规则。几何规则仍不能证明动作的采集意图，最终由本次人工复核确认。", "",
        "## 快速查看", "", "![20 条保持段中点](overview.png)", "",
        "| 编号 | NPZ | 原文件起止秒 | 时长/秒 | 前后站立证据 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in clips:
        start, end = c["export_seconds"]
        lines.append(f"| {c['id']:02d} | [{Path(c['source']).parent.name}]({c['npz']}) | "
                     f"{start:.2f}–{end:.2f} | {c['duration_seconds']:.2f} | "
                     f"[对照图]({c['context_plot']}) |")
    lines += ["", "详细来源和各条件的量化结果见 [manifest.json](manifest.json)，"
              "便于表格检查的清单见 [manifest.csv](manifest.csv)。", "",
              "## 复现", "", "在项目根目录运行；为避免覆盖人工复核文件，复现时指定新的输出目录。", "",
              "```bash", "MPLCONFIGDIR=/tmp/textop-tpose-mpl OPENBLAS_NUM_THREADS=1 \\",
              "  /home/hanyi/miniforge3/envs/textop/bin/python \\",
              "  dataset/data_analyze/tpose/analyze_hiphi_tpose.py \\",
              "  --export-review 20 --scan-per-category 120 --seed 42 --workers 8 \\",
              "  --output /tmp/hiphi_tpose_precision_reproduce", "```", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def export_precision_review(args, groups):
    rng = random.Random(args.seed)
    selected = []
    for _, files in sorted(groups.items()):
        selected.extend(rng.sample(sorted(files), min(args.scan_per_category, len(files))))

    def inspect(relative):
        return [dict(source=relative, **c) for c in precision_candidates(
            args.dataset_dir / "motions" / relative)]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        per_file = list(pool.map(inspect, selected))
    candidates = [c for records in per_file for c in records]
    print(f"Strict scan: {len(selected)} files, {len(candidates)} cycles, "
          f"{len({c['source'] for c in candidates})} distinct source files", flush=True)
    # One hold per source, then round-robin categories to avoid duplicates and
    # selecting every example from one action class. No threshold relaxation.
    by_category = defaultdict(list)
    seen_sources = set()
    for c in sorted(candidates, key=lambda item: (-item["duration_seconds"], item["source"])):
        if c["source"] not in seen_sources:
            seen_sources.add(c["source"])
            by_category[c["source"].split("/")[0]].append(c)
    chosen = []
    while len(chosen) < args.export_review and any(by_category.values()):
        for category in sorted(by_category):
            if by_category[category] and len(chosen) < args.export_review:
                chosen.append(by_category[category].pop(0))
    if len(chosen) < args.export_review:
        raise RuntimeError(f"Only {len(chosen)} distinct sources pass strict rules; "
                           "increase --scan-per-category. Rules were not relaxed.")
    output = args.output / f"review_{args.export_review}"
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("*.npz")):
        raise FileExistsError(f"Review output already contains NPZs: {output}")
    for index, record in enumerate(chosen, 1):
        source = args.dataset_dir / "motions" / record["source"]
        start, end = record["export_frames"]
        name = f"{index:02d}_{source.parent.name}_f{start:05d}-{end:05d}"
        record.update(id=index, npz=f"{name}.npz", context_plot=f"{name}_context.png")
        with np.load(source, allow_pickle=False) as data:
            arrays = {key: (data[key][start:end].copy()
                            if data[key].ndim and data[key].shape[0] == record["source_frames"]
                            else data[key].copy()) for key in data.files}
        np.savez_compressed(output / record["npz"], **arrays)
        # Verify lossless original slicing, all fields, shapes, dtypes and values.
        with np.load(source, allow_pickle=False) as original, np.load(output / record["npz"]) as saved:
            assert set(original.files) == set(saved.files)
            for key, expected in arrays.items():
                assert expected.dtype == saved[key].dtype
                np.testing.assert_array_equal(saved[key], expected)
        plot_precision_review(source, record, output / record["context_plot"])
    payload = dict(rules=PRECISION_RULES, seed=args.seed, scanned_files=len(selected),
                   accepted_cycles=len(candidates),
                   accepted_source_files=len(seen_sources),
                   exported_files=len(chosen),
                   export_policy="Pure uninterrupted strict T-pose holds; standing context only in plots and manifest.",
                   interval_convention="zero-based [start, end), seconds = frame / fps",
                   clips=chosen)
    (output / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (output / "manifest.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "npz", "source", "fps", "start_frame", "end_frame_exclusive",
                         "start_seconds", "end_seconds", "duration_seconds", "context_plot"])
        for c in chosen:
            writer.writerow([c["id"], c["npz"], c["source"], c["fps"], *c["export_frames"],
                             *c["export_seconds"], c["duration_seconds"], c["context_plot"]])
    fig, axes = plt.subplots(int(np.ceil(len(chosen) / 5)), 5, figsize=(15, 3.5 * np.ceil(len(chosen) / 5)))
    for ax, c in zip(np.asarray(axes).reshape(-1), chosen):
        f = load_features(args.dataset_dir / "motions" / c["source"])
        start, end = c["export_frames"]
        draw_review_pose(ax, f, (start + end) // 2)
        ax.set_title(f"{c['id']:02d}: {Path(c['source']).parent.name}\n"
                     f"{start / c['fps']:.2f}-{end / c['fps']:.2f}s", fontsize=8)
    for ax in np.asarray(axes).reshape(-1)[len(chosen):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "overview.png", dpi=140)
    plt.close(fig)
    write_precision_readme(output, payload)
    print(f"Exported and verified {len(chosen)} NPZ clips in {output}", flush=True)


def plot_example(path, record, output, high_deg, low_deg):
    f = load_features(path)
    q, fps = f["q"], f["fps"]
    t = np.arange(len(q)) / fps
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(t, q[:, 15], label="Left shoulder roll", lw=1)
    axes[0].plot(t, -q[:, 16], label="Right shoulder roll (sign reversed)", lw=1)
    axes[0].axhline(high_deg, color="black", ls="--", lw=.8, label="Core threshold")
    axes[0].axhline(low_deg, color="gray", ls=":", lw=.8, label="Transition boundary")
    axes[0].set_ylabel("Outward roll (deg)")
    axes[0].legend(loc="upper center", ncol=2, fontsize=8)
    axes[1].plot(t, f["root_speed"], color="tab:green", lw=.8)
    axes[1].axhline(.10, color="gray", ls=":", lw=.8)
    axes[1].set_ylabel("Root XY speed (m/s)")
    axes[2].plot(t, f["tilt"], color="tab:purple", lw=.8)
    axes[2].axhline(30, color="gray", ls=":", lw=.8)
    axes[2].set_ylabel("Torso tilt (deg)")
    axes[2].set_xlabel("Time since first frame (s)")
    for c in record["candidates"]:
        lo, hi = c["transition_seconds"]
        start, end = c["core_seconds"]
        for ax in axes:
            ax.axvspan(lo, hi, color="orange", alpha=.18)
            ax.axvspan(start, end, color="red", alpha=.2)
        axes[0].text((start + end) / 2, 112,
                     f"{start:.2f}-{end:.2f}s", ha="center", fontsize=9)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.set_xlim(0, len(q) / fps)
    axes[0].set_ylim(0, max(125, q[:, 15].max() + 10))
    fig.suptitle(f"{path.parent.name}: red = high-roll core; orange = transitions")
    fig.tight_layout()
    fig.savefig(output / f"{path.parent.name}_timeline.png", dpi=160)
    plt.close(fig)

    first = record["candidates"][0]
    lo, hi = first["transition_seconds"]
    start, end = first["core_seconds"]
    times = [max(0, lo - .4), (lo + start) / 2, (start + end) / 2,
             (end + hi) / 2, min(len(q) / fps - 1 / fps, hi + .4)]
    labels = ["Before", "Raising", "Holding", "Lowering", "After"]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex="row", sharey=True)
    chains = [[0, 3, 6, 9], [0, 1, 4, 7, 10, 14, 18],
              [0, 2, 5, 8, 11, 15, 19], [9, 12, 16, 20, 22, 24, 26, 28],
              [9, 13, 17, 21, 23, 25, 27, 29]]
    colors = ["black", "tab:blue", "tab:orange", "tab:blue", "tab:orange"]
    for col, (time, label) in enumerate(zip(times, labels)):
        frame = min(len(q) - 1, round(time * fps))
        time = frame / fps
        # Use torso heading only, preserving tilt and world vertical.
        forward = f["torso"][frame].apply([1., 0., 0.])
        yaw = np.arctan2(forward[1], forward[0])
        p = Rotation.from_rotvec([0., 0., -yaw]).apply(
            f["pos"][frame] - f["pos"][frame, 0])
        for row, horizontal in enumerate([1, 0]):
            ax = axes[row, col]
            for chain, color in zip(chains, colors):
                ax.plot(p[chain, horizontal], p[chain, 2], "o-", color=color,
                        markersize=3, lw=2)
            ax.set_aspect("equal")
            ax.set_xlim(-.6, .6)
            ax.set_ylim(-.85, .65)
            ax.grid(alpha=.2)
            if row == 0:
                ax.set_title(f"{label}\n{time:.2f}s")
            if col == 0:
                ax.set_ylabel("Front / Z (m)" if row == 0 else "Side / Z (m)")
    fig.suptitle(f"{path.parent.name}: observed link positions (blue left, orange right)")
    fig.tight_layout()
    fig.savefig(output / f"{path.parent.name}_poses.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/hiphi"))
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "analysis_output/hiphi_tpose")
    parser.add_argument("--per-category", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--high-deg", type=float, default=60)
    parser.add_argument("--low-deg", type=float, default=40)
    parser.add_argument("--min-seconds", type=float, default=.5)
    parser.add_argument("--export-review", type=int, default=0,
                        help="Export this many strict holds, each from a different source")
    parser.add_argument("--scan-per-category", type=int, default=120,
                        help="Files sampled per category for strict review export")
    args = parser.parse_args()
    if not (0 < args.low_deg < args.high_deg) or args.min_seconds <= 0:
        parser.error("Require 0 < low-deg < high-deg and min-seconds > 0")
    if args.per_category < 1 or args.workers < 1:
        parser.error("per-category and workers must be positive")
    if args.export_review < 0 or args.scan_per_category < 1:
        parser.error("export-review must be nonnegative and scan-per-category positive")
    groups = defaultdict(list)
    for relative in (args.dataset_dir / "filenames.txt").read_text().splitlines():
        relative = relative.strip()
        if relative.endswith(".npz") and "__mirror" not in relative:
            groups[relative.split("/")[0]].append(relative)
    if args.export_review:
        export_precision_review(args, groups)
        return
    rng = random.Random(args.seed)
    selected = []
    for _, files in sorted(groups.items()):
        selected.extend(rng.sample(sorted(files), min(args.per_category, len(files))))

    def inspect(relative):
        record = analyze(args.dataset_dir / "motions" / relative, args.high_deg,
                         args.low_deg, args.min_seconds)
        return dict(path=relative, **record)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(inspect, selected))
    examples = [next((r for r in records if r["path"] == path), None)
                or inspect(path) for path in EXAMPLES]
    candidates = [c for r in records for c in r["candidates"]]
    summary = dict(
        canonical_files=sum(map(len, groups.values())), categories=len(groups),
        sampled_files=len(records), seed=args.seed,
        sampled_seconds=sum(r["duration_seconds"] for r in records),
        fps_values=sorted({r["fps"] for r in records}),
        files_with_high_roll_candidates=sum(bool(r["candidates"]) for r in records),
        candidate_count=len(candidates),
        candidate_seconds=sum(c["core_seconds"][1] - c["core_seconds"][0]
                              for c in candidates),
        candidates_with_at_least_half_second_support=sum(
            c["longest_support_seconds"] >= .5 for c in candidates),
        high_deg=args.high_deg, low_deg=args.low_deg, min_seconds=args.min_seconds,
        interval_convention="zero-based [start, end), seconds = frame / fps",
        warning="Candidates require visual review; sample is balanced by category, not representative of all frames.",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sample_analysis.json").write_text(json.dumps(
        dict(summary=summary, examples=examples, samples=records), indent=2) + "\n")
    for example in examples:
        plot_example(args.dataset_dir / "motions" / example["path"], example,
                     args.output, args.high_deg, args.low_deg)
    print(json.dumps(dict(summary=summary, examples=examples), indent=2))
    print(f"Saved analysis to {args.output}")


if __name__ == "__main__":
    main()
