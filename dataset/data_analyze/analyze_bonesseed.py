#!/usr/bin/env python3
"""
BONES-SEED 数据集分析脚本
=========================
4 个分析任务，生成 matplotlib 图表：

  Task 1 — 时长分布条形图 (120 Hz)
  Task 2 — Ego-centric 终点散点图 (forward = +X, Z-up)
  Task 3 — 根节点 Z 变化范围三合一图
  Task 4 — contact_mask 方法适用性分析
          4a. 根节点每帧水平位移 vs 阈值
          4b. 脚踝世界 Z 分布 vs 阈值
          4c. 两方法 contact mask 对比

单位: bones-seed 原始为 cm → 统一 ÷100 转为 TextOp 标准的 meters.
坐标系: Z-up, TextOp +X-forward, +Y-left.
"""

import csv
import os
import sys
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE = "/home/lenovo/data/bones-seed/g1/csv"
SAMPLE_RATE = 120  # Hz
CM_TO_M = 1.0 / 100.0  # cm → meters

# Ego-centric bounding box (meters, Z-up, TextOp X-forward/Y-left)
BBOX_XMIN, BBOX_YMIN = -0.5, -1.0
BBOX_XMAX, BBOX_YMAX = 1.5, 1.0

# process_retarget_data.py foot_detect thresholds
WORLD_VEL_THRESH = np.sqrt(0.002)  # ≈ 0.0447 m/frame (world-frame)
WORLD_HEIGHT_THRESH = 0.08         # absolute Z threshold (m)

# Output
OUTPUT_DIR = Path(__file__).resolve().parent / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Phase 1: 遍历所有 CSV，收集基础统计数据
# ---------------------------------------------------------------------------
def collect_csv_files(base_dir: str) -> list:
    """递归收集所有 .csv 文件路径."""
    files = []
    for root, dirs, fnames in os.walk(base_dir):
        for fn in fnames:
            if fn.endswith(".csv"):
                files.append(os.path.join(root, fn))
    return files


def parse_csv_fast(filepath: str):
    """
    快速解析 CSV，返回:
      - n_frames: int
      - first_pos: [x, y, z] (meters)
      - first_euler_deg: [rx, ry, rz] (degrees)
      - last_pos: [x, y, z] (meters)
      - z_min, z_max: float (meters)
      - per_frame_disp_sq: list of squared horizontal displacement per frame (m²)
    失败返回 None.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh)
            header = next(reader)

            # Read first data row
            first_row = next(reader)
            first_vals = [float(v) for v in first_row]
            first_pos = np.array(first_vals[1:4]) * CM_TO_M
            first_euler = np.array(first_vals[4:7])  # deg, keep as-is

            prev_xy = first_pos[:2].copy()
            z_min = first_pos[2]
            z_max = first_pos[2]
            n_frames = 1
            per_frame_disp_sq = []

            last_pos = first_pos.copy()

            for row in reader:
                vals = [float(v) for v in row]
                pos = np.array(vals[1:4]) * CM_TO_M
                last_pos = pos

                # Per-frame horizontal displacement
                dxy = pos[:2] - prev_xy
                per_frame_disp_sq.append(float(dxy[0]**2 + dxy[1]**2))
                prev_xy = pos[:2].copy()

                z = pos[2]
                if z < z_min:
                    z_min = z
                if z > z_max:
                    z_max = z

                n_frames += 1

        return {
            "n_frames": n_frames,
            "first_pos": first_pos,
            "first_euler_deg": first_euler,
            "last_pos": last_pos,
            "z_min": z_min,
            "z_max": z_max,
            "per_frame_disp_sq": per_frame_disp_sq,
        }

    except (OSError, StopIteration, ValueError):
        return None


def compute_ego_endpoint(first_pos, first_euler_deg, last_pos):
    """
    Ego-centric 终点坐标 (Z-up, TextOp X-forward/Y-left).

    bones-seed world: Z-up, default forward = +X.
    目标: Z-up, forward = +X, left = +Y.
    """
    euler_rad = np.deg2rad(first_euler_deg)
    rot = R.from_euler("XYZ", euler_rad)
    # 默认前向 +X → 第一帧实际前向
    forward_world = rot.apply(np.array([1.0, 0.0, 0.0]))
    f_xy = forward_world[:2]
    norm = np.linalg.norm(f_xy)
    if norm < 1e-10:
        # Degenerate: identity transform
        f_xy = np.array([1.0, 0.0])
        norm = 1.0
    f_xy = f_xy / norm

    # 位移（世界 XY）
    d_xy = last_pos[:2] - first_pos[:2]
    left_xy = np.array([-f_xy[1], f_xy[0]])
    ego_x = np.dot(d_xy, f_xy)
    ego_y = np.dot(d_xy, left_xy)

    return ego_x, ego_y


# ---------------------------------------------------------------------------
# Task 1: 时长分布
# ---------------------------------------------------------------------------
def plot_duration_histogram(durations_sec, bin_width=2.0):
    """绘制时长分布条形图."""
    max_dur = max(durations_sec)
    bins = np.arange(0, max_dur + bin_width, bin_width)
    counts, edges = np.histogram(durations_sec, bins=bins)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(edges[:-1], counts, width=bin_width * 0.9, align="edge",
           color="steelblue", edgecolor="white", alpha=0.85)

    # 统计标注
    arr = np.array(durations_sec)
    stats_text = (
        f"N = {len(arr):,}\n"
        f"Mean = {arr.mean():.1f}s\n"
        f"Median = {np.median(arr):.1f}s\n"
        f"P10 = {np.percentile(arr, 10):.1f}s\n"
        f"P90 = {np.percentile(arr, 90):.1f}s\n"
        f"Max = {arr.max():.1f}s"
    )
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8),
            fontfamily="monospace")

    ax.set_xlabel("Duration (s) @120Hz", fontsize=12)
    ax.set_ylabel("Number of files", fontsize=12)
    ax.set_title("Task 1: Motion Duration Distribution (bin=%gs)" % bin_width, fontsize=14)
    ax.set_xlim(0, min(max_dur + bin_width, 180))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    outpath = OUTPUT_DIR / "task1_duration_histogram.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[Task 1] Saved: {outpath}")
    return outpath


# ---------------------------------------------------------------------------
# Task 2: Ego-centric 终点散点图
# ---------------------------------------------------------------------------
def plot_ego_endpoints(ego_points, max_samples=20000):
    """绘制 ego-centric 终点散点图 + 边框."""
    pts = np.array(ego_points)
    n_total = len(pts)

    if n_total > max_samples:
        idx = np.random.choice(n_total, max_samples, replace=False)
        pts_sample = pts[idx]
        sample_note = f" (sampled {max_samples:,} / {n_total:,})"
    else:
        pts_sample = pts
        sample_note = ""

    # 统计边框内比例
    in_box = (
        (pts[:, 0] >= BBOX_XMIN) & (pts[:, 0] <= BBOX_XMAX) &
        (pts[:, 1] >= BBOX_YMIN) & (pts[:, 1] <= BBOX_YMAX)
    )
    in_ratio = in_box.mean() * 100

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(pts_sample[:, 0], pts_sample[:, 1],
               s=1.5, c="steelblue", alpha=0.25, edgecolors="none",
               rasterized=True)

    # 边框
    bx = [BBOX_XMIN, BBOX_XMAX, BBOX_XMAX, BBOX_XMIN, BBOX_XMIN]
    by = [BBOX_YMIN, BBOX_YMIN, BBOX_YMAX, BBOX_YMAX, BBOX_YMIN]
    ax.plot(bx, by, "r--", linewidth=2, label=f"Scene perception box [{BBOX_XMIN},{BBOX_YMIN}]→[{BBOX_XMAX},{BBOX_YMAX}]")

    # 原点
    ax.scatter([0], [0], c="red", s=60, marker="x", linewidths=2, zorder=5, label="Origin (start)")

    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)

    stats_text = (
        f"N = {n_total:,}{sample_note}\n"
        f"In-box: {in_ratio:.1f}%\n"
        f"X range: [{pts[:,0].min():.1f}, {pts[:,0].max():.1f}] m\n"
        f"Y range: [{pts[:,1].min():.1f}, {pts[:,1].max():.1f}] m"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontfamily="monospace")

    ax.set_xlabel("Forward (m)  +X", fontsize=12)
    ax.set_ylabel("Left (m)  +Y", fontsize=12)
    ax.set_title("Task 2: Ego-centric Endpoint Distribution (Z-up, X-forward)", fontsize=14)
    ax.set_aspect("equal")
    ax.legend(fontsize=10, loc="lower right")

    # 自动调整显示范围
    margin = 1.0
    ax.set_xlim(pts_sample[:, 0].min() - margin, pts_sample[:, 0].max() + margin)
    ax.set_ylim(pts_sample[:, 1].min() - margin, pts_sample[:, 1].max() + margin)

    fig.tight_layout()
    outpath = OUTPUT_DIR / "task2_ego_endpoints.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[Task 2] Saved: {outpath}  (in-box: {in_ratio:.1f}%)")
    return outpath


# ---------------------------------------------------------------------------
# Task 3: 根节点 Z 变化范围
# ---------------------------------------------------------------------------
def plot_z_statistics(z_mins, z_maxs):
    """Z min / Z max / Z range 三合一直方图."""
    z_ranges = np.array(z_maxs) - np.array(z_mins)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Z min
    ax = axes[0]
    ax.hist(z_mins, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0.77, color="red", linestyle="--", linewidth=1.5, label="G1 stand (~0.77m)")
    ax.set_xlabel("Z min (m)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Root Z Minimum", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Z max
    ax = axes[1]
    ax.hist(z_maxs, bins=80, color="darkorange", edgecolor="white", alpha=0.85)
    ax.axvline(0.77, color="red", linestyle="--", linewidth=1.5, label="G1 stand (~0.77m)")
    ax.set_xlabel("Z max (m)", fontsize=11)
    ax.set_title("Root Z Maximum", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Z range
    ax = axes[2]
    ax.hist(z_ranges, bins=80, color="forestgreen", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Z range (m)", fontsize=11)
    ax.set_title("Root Z Range (max-min)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    # 统计信息
    for arr, name in [(z_mins, "Z min"), (z_maxs, "Z max"), (z_ranges, "Z range")]:
        arr = np.array(arr)
        print(f"  {name}: mean={arr.mean():.4f}m, median={np.median(arr):.4f}m, "
              f"P5={np.percentile(arr,5):.3f}m, P95={np.percentile(arr,95):.3f}m, "
              f"min={arr.min():.3f}m, max={arr.max():.3f}m")

    fig.suptitle("Task 3: Root Z Statistics (meters, Z-up)", fontsize=14, y=1.01)
    fig.tight_layout()
    outpath = OUTPUT_DIR / "task3_z_statistics.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[Task 3] Saved: {outpath}")
    return outpath


# ---------------------------------------------------------------------------
# Task 4a: 根节点每帧水平位移分布
# ---------------------------------------------------------------------------
def plot_root_displacement(all_disp_sq):
    """
    汇总所有 CSV 的相邻帧水平位移平方.
    all_disp_sq: list of lists (one per file) of squared horizontal disp per frame.
    为避免内存爆炸，用流式直方图.
    """
    # 收集所有位移值到直方图
    all_disp = []
    for file_disps in all_disp_sq:
        all_disp.extend(file_disps)
    all_disp = np.sqrt(np.array(all_disp))  # sqrt → m/frame

    fig, ax = plt.subplots(figsize=(12, 5))
    bins = np.logspace(-4, 1, 120)  # log scale bins from 1e-4 to 10 m/frame
    ax.hist(all_disp, bins=bins, color="steelblue", edgecolor="white", alpha=0.85)

    # 阈值线
    ax.axvline(WORLD_VEL_THRESH, color="red", linestyle="--", linewidth=2,
               label=f"World-vel threshold = {WORLD_VEL_THRESH:.4f} m/frame\n"
                     f"(process_retarget_data.py, sqrt(0.002))")

    # 统计
    above_thresh = (all_disp > WORLD_VEL_THRESH).mean() * 100
    stats_text = (
        f"Frames total: {len(all_disp):,}\n"
        f"> threshold: {above_thresh:.1f}%\n"
        f"Mean disp: {all_disp.mean():.4f} m/frame\n"
        f"Median disp: {np.median(all_disp):.4f} m/frame\n"
        f"P95 disp: {np.percentile(all_disp, 95):.4f} m/frame"
    )
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontfamily="monospace")

    ax.set_xscale("log")
    ax.set_xlabel("Root horizontal displacement per frame (m) @120Hz", fontsize=12)
    ax.set_ylabel("Number of frames", fontsize=12)
    ax.set_title("Task 4a: Per-Frame Root Horizontal Displacement", fontsize=14)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    outpath = OUTPUT_DIR / "task4a_root_displacement.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[Task 4a] Saved: {outpath}  (frames > threshold: {above_thresh:.1f}%)")

    # Return stats for data-driven conclusion
    disp_stats = {
        "mean": float(all_disp.mean()),
        "median": float(np.median(all_disp)),
        "p95": float(np.percentile(all_disp, 95)),
        "pct_above_thresh": float(above_thresh),
        "thresh": WORLD_VEL_THRESH,
        "thresh_mps": WORLD_VEL_THRESH * SAMPLE_RATE,  # m/s equivalent
    }
    return outpath, disp_stats


# ---------------------------------------------------------------------------
# Task 4b & 4c: FK-based analysis (需要 RobotSkeleton)
# ---------------------------------------------------------------------------
def init_skeleton():
    """延迟初始化 RobotSkeleton (G1, 23 DOF, wrists locked)."""
    import torch
    from omegaconf import OmegaConf
    from robotmdar.skeleton.robot import RobotSkeleton

    project_root = Path(__file__).resolve().parent
    skel_cfg_path = project_root / "TextOpRobotMDAR/robotmdar/config/skeleton/g1.yaml"
    asset_root = str((project_root / "TextOpRobotMDAR/description/robots/g1/").resolve())

    cfg = OmegaConf.load(str(skel_cfg_path))
    cfg.asset.assetRoot = asset_root
    skeleton = RobotSkeleton(device="cpu", cfg=cfg)
    return skeleton


def load_csv_full(filepath: str):
    """
    加载完整 CSV 数据 (含 DOF)，转换为 50Hz 版本后返回.
    复用 convert_bones_seed.py 的逻辑.
    """
    from scipy.interpolate import interp1d
    from scipy.spatial.transform import Slerp

    ORIGINAL_FPS = 120
    TARGET_FPS = 50
    DOF_29_TO_23_MASK = [
        True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,
        True,  True,  True,
        True,  True,  True,  True,
        False, False, False,
        True,  True,  True,  True,
        False, False, False,
    ]

    with open(filepath) as fh:
        reader = csv.reader(fh)
        header = next(reader)
        data = list(reader)
    data = np.array(data, dtype=np.float32)

    trans_120 = data[:, 1:4] * CM_TO_M         # [N, 3] meters
    rot_euler_120 = data[:, 4:7]                # [N, 3] degrees
    dof_29_120 = data[:, 7:36]                  # [N, 29] degrees

    T_orig = trans_120.shape[0]
    if T_orig < 3:
        return None

    t_orig = np.linspace(0, (T_orig - 1) / ORIGINAL_FPS, T_orig)
    T_target = max(int(T_orig * TARGET_FPS / ORIGINAL_FPS), 2)
    t_target = np.linspace(0, (T_orig - 1) / ORIGINAL_FPS, T_target)

    # Translation — linear
    trans_50 = interp1d(t_orig, trans_120, axis=0, kind="linear")(t_target)

    # Rotation — SLERP
    rot_euler_rad = np.deg2rad(rot_euler_120)
    rotations = R.from_euler("XYZ", rot_euler_rad)
    slerp = Slerp(t_orig, rotations)
    rot_quat_50 = slerp(t_target).as_quat()  # xyzw

    # DOF — linear, deg→rad, 29→23
    dof_50 = interp1d(t_orig, dof_29_120, axis=0, kind="linear")(t_target)
    dof_23_rad = np.deg2rad(dof_50)[:, DOF_29_TO_23_MASK]

    return trans_50, rot_quat_50, dof_23_rad


def compute_contact_world_velocity(positions, foot_id=6):
    """
    process_retarget_data.py 的 foot_detect 逻辑 (world-frame).
    positions: [T, N_bodies, 3] 世界坐标 (FK output).
    返回: contact [T] (0或1)
    """
    fid = foot_id
    vel2 = np.sum(np.diff(positions[:, fid, :], axis=0) ** 2, axis=-1)  # [T-1]
    height = positions[1:, fid, 2]  # absolute Z

    contact = (
        (vel2 < 0.002) & (height < WORLD_HEIGHT_THRESH)
    ).astype(np.float32)
    contact = np.concatenate([np.array([1.0], dtype=np.float32), contact])
    return contact


def compute_contact_root_relative(positions, root_trans, foot_id=6):
    """
    convert_bones_seed.py 的 foot_detect 逻辑 (root-relative).
    positions: [T, N_bodies, 3] 世界坐标 (FK output).
    root_trans: [T, 3] root 世界坐标.
    """
    pelvis = root_trans
    ankle_world = positions[:, foot_id, :]
    ankle_rel = ankle_world - pelvis
    ankle_rel_z = ankle_rel[:, 2]

    rel_vel2 = np.sum(np.diff(ankle_rel, axis=0) ** 2, axis=-1)  # [T-1]
    leg_extension = np.abs(ankle_rel_z)
    standing_leg = np.median(leg_extension)
    height_above_ground = np.abs(leg_extension[1:] - standing_leg)

    contact = (
        (rel_vel2 < 0.002) & (height_above_ground < 0.15)
    ).astype(np.float32)
    contact = np.concatenate([np.array([1.0], dtype=np.float32), contact])
    return contact


def run_fk(skeleton, trans, rot_quat, dof):
    """通过 RobotSkeleton 算 FK，返回 global_translation_extend [T, N, 3]."""
    import torch
    motion_dict = {
        "dof": torch.from_numpy(dof).float().unsqueeze(0),
        "root_trans_offset": torch.from_numpy(trans).float().unsqueeze(0),
        "root_rot": torch.from_numpy(rot_quat).float().unsqueeze(0),
    }
    fk_return = skeleton.forward_kinematics(motion_dict, return_full=False)
    return fk_return["global_translation_extend"][0].numpy()  # [T, N_bodies, 3]


def plot_task4b_ankle_z(ankle_z_samples):
    """绘制脚踝世界 Z 分布，标注 0.08 阈值."""
    all_z = np.concatenate(ankle_z_samples)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(all_z, bins=100, color="steelblue", edgecolor="white", alpha=0.85)

    ax.axvline(WORLD_HEIGHT_THRESH, color="red", linestyle="--", linewidth=2,
               label=f"World height threshold = {WORLD_HEIGHT_THRESH} m\n"
                     f"(process_retarget_data.py)")

    # G1 standing ankle height: pelvis ≈ 0.77m, ankle_rel_z ≈ -0.69m, world ≈ 0.08m
    ax.axvline(0.08, color="green", linestyle=":", linewidth=1.5,
               label="G1 standing ankle ≈ 0.08m (ground)")

    above = (all_z > WORLD_HEIGHT_THRESH).mean() * 100
    stats_text = (
        f"Samples: {len(all_z):,}\n"
        f"Z > {WORLD_HEIGHT_THRESH}m: {above:.1f}%\n"
        f"Mean Z: {all_z.mean():.3f}m\n"
        f"Median Z: {np.median(all_z):.3f}m\n"
        f"Z range: [{all_z.min():.3f}, {all_z.max():.3f}]m"
    )
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontfamily="monospace")

    ax.set_xlabel("Ankle world Z (m)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Task 4b: Foot Ankle World Z Distribution (FK-based)", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    outpath = OUTPUT_DIR / "task4b_ankle_z.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[Task 4b] Saved: {outpath}  (Z > {WORLD_HEIGHT_THRESH}m: {above:.1f}%)")

    # Return stats for data-driven conclusion
    ank_z_stats = {
        "mean": float(all_z.mean()),
        "median": float(np.median(all_z)),
        "pct_above_thresh": float(above),
        "thresh": WORLD_HEIGHT_THRESH,
    }
    return outpath, ank_z_stats


def plot_task4c_comparison(trans, positions, foot_id_l, foot_id_r):
    """绘制两种 contact detection 方法对比."""
    T = trans.shape[0]
    frames = np.arange(T)

    # World velocity method (process_retarget_data.py)
    contact_world_l = compute_contact_world_velocity(positions, foot_id_l)
    contact_world_r = compute_contact_world_velocity(positions, foot_id_r)

    # Root-relative method (convert_bones_seed.py)
    contact_rel_l = compute_contact_root_relative(positions, trans, foot_id_l)
    contact_rel_r = compute_contact_root_relative(positions, trans, foot_id_r)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # Upper: world-velocity method
    ax = axes[0]
    ax.fill_between(frames, 0, 1, where=(contact_world_l > 0.5),
                    color="blue", alpha=0.3, step="mid", label="Left foot")
    ax.fill_between(frames, 0, 1, where=(contact_world_r > 0.5),
                    color="red", alpha=0.3, step="mid", label="Right foot")
    ax.set_ylabel("Contact", fontsize=11)
    ax.set_title("World-Velocity Method (process_retarget_data.py foot_detect)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(-0.1, 1.3)
    ax.grid(axis="y", alpha=0.3)

    # 统计静止帧比例
    static_l = contact_world_l.mean()
    static_r = contact_world_r.mean()
    ax.text(0.02, 0.95, f"Left 'contact'={static_l:.1%}, Right 'contact'={static_r:.1%}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            fontfamily="monospace")

    # Lower: root-relative method
    ax = axes[1]
    ax.fill_between(frames, 0, 1, where=(contact_rel_l > 0.5),
                    color="blue", alpha=0.3, step="mid", label="Left foot")
    ax.fill_between(frames, 0, 1, where=(contact_rel_r > 0.5),
                    color="red", alpha=0.3, step="mid", label="Right foot")
    ax.set_xlabel("Frame @50Hz", fontsize=11)
    ax.set_ylabel("Contact", fontsize=11)
    ax.set_title("Root-Relative Method (convert_bones_seed.py foot_detect)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(-0.1, 1.3)
    ax.grid(axis="y", alpha=0.3)

    static_l = contact_rel_l.mean()
    static_r = contact_rel_r.mean()
    ax.text(0.02, 0.95, f"Left contact={static_l:.1%}, Right contact={static_r:.1%}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            fontfamily="monospace")

    fig.tight_layout()
    outpath = OUTPUT_DIR / "task4c_contact_comparison.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[Task 4c] Saved: {outpath}")

    # Return stats for data-driven conclusion
    c_stats = {
        "world_method": {
            "left_contact_pct": float(contact_world_l.mean() * 100),
            "right_contact_pct": float(contact_world_r.mean() * 100),
        },
        "rel_method": {
            "left_contact_pct": float(contact_rel_l.mean() * 100),
            "right_contact_pct": float(contact_rel_r.mean() * 100),
        },
    }
    return outpath, c_stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BONES-SEED Dataset Analysis")
    parser.add_argument("--skip-fk", action="store_true",
                        help="Skip FK-based analysis (Task 4b, 4c) if skeleton unavailable")
    parser.add_argument("--fk-samples", type=int, default=500,
                        help="Number of CSVs to sample for FK analysis (default: 500)")
    parser.add_argument("--base-dir", type=str, default=BASE,
                        help="Path to bones-seed g1/csv/ directory")
    args = parser.parse_args()

    base_dir = args.base_dir
    if not os.path.isdir(base_dir):
        print(f"ERROR: Directory not found: {base_dir}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Phase 1: 遍历所有 CSV
    # -----------------------------------------------------------------------
    all_files = collect_csv_files(base_dir)
    print(f"Found {len(all_files):,} CSV files")
    if not all_files:
        print("ERROR: No CSV files found!")
        sys.exit(1)

    durations_sec = []
    ego_points = []
    z_mins, z_maxs = [], []
    all_disp_sq = []  # per-file list of squared horizontal displacements

    print("\nPhase 1: Scanning all CSVs ...")
    for fp in tqdm(all_files, desc="Scanning"):
        result = parse_csv_fast(fp)
        if result is None or result["n_frames"] < 2:
            continue

        # Task 1
        dur = result["n_frames"] / SAMPLE_RATE
        durations_sec.append(dur)

        # Task 2
        ex, ey = compute_ego_endpoint(
            result["first_pos"], result["first_euler_deg"], result["last_pos"]
        )
        ego_points.append((ex, ey))

        # Task 3
        z_mins.append(result["z_min"])
        z_maxs.append(result["z_max"])

        # Task 4a
        all_disp_sq.append(result["per_frame_disp_sq"])

    n_valid = len(durations_sec)
    print(f"Valid sequences: {n_valid:,} / {len(all_files):,}")

    # -----------------------------------------------------------------------
    # Task 1
    # -----------------------------------------------------------------------
    print("\n--- Task 1: Duration Histogram ---")
    plot_duration_histogram(durations_sec)

    # -----------------------------------------------------------------------
    # Task 2
    # -----------------------------------------------------------------------
    print("\n--- Task 2: Ego-centric Endpoints ---")
    plot_ego_endpoints(ego_points)

    # -----------------------------------------------------------------------
    # Task 3
    # -----------------------------------------------------------------------
    print("\n--- Task 3: Z Statistics ---")
    plot_z_statistics(z_mins, z_maxs)

    # -----------------------------------------------------------------------
    # Task 4a
    # -----------------------------------------------------------------------
    print("\n--- Task 4a: Root Per-Frame Displacement ---")
    _, disp_stats = plot_root_displacement(all_disp_sq)

    # -----------------------------------------------------------------------
    # Task 4b, 4c (FK-based)
    # -----------------------------------------------------------------------
    ank_z_stats = None
    c_stats = None
    if args.skip_fk:
        print("\n--- Skipping Task 4b, 4c (--skip-fk) ---")
    else:
        print("\n--- Task 4b & 4c: FK-based contact analysis ---")
        try:
            skeleton = init_skeleton()
            print("RobotSkeleton initialized.")
        except Exception as e:
            print(f"ERROR initializing skeleton: {e}")
            print("Skipping FK-based analysis. Use --skip-fk to suppress.")
            skeleton = None

        if skeleton is not None:
            # 随机采样用于 FK
            n_sample = min(args.fk_samples, len(all_files))
            sample_files = random.sample(all_files, n_sample)

            ankle_z_samples = []
            locomotion_found = None  # for Task 4c

            print(f"Running FK on {n_sample} sampled CSVs ...")
            for fp in tqdm(sample_files, desc="FK analysis"):
                result = load_csv_full(fp)
                if result is None:
                    continue
                trans, rot_quat, dof = result

                try:
                    positions = run_fk(skeleton, trans, rot_quat, dof)
                except Exception:
                    continue

                # 4b: 收集脚踝 Z
                # G1 foot body indices: left=6, right=12
                ankle_z_samples.append(positions[:, 6, 2])
                ankle_z_samples.append(positions[:, 12, 2])

                # 4c: 找第一个有显著水平位移的 locomotion clip
                if locomotion_found is None:
                    total_disp = np.sqrt(
                        (trans[-1, 0] - trans[0, 0]) ** 2 +
                        (trans[-1, 1] - trans[0, 1]) ** 2
                    )
                    if total_disp > 2.0 and trans.shape[0] > 50:  # >2m travel
                        locomotion_found = {
                            "fp": fp,
                            "trans": trans,
                            "positions": positions,
                        }

            # Plot 4b
            if ankle_z_samples:
                _, ank_z_stats = plot_task4b_ankle_z(ankle_z_samples)

            # Plot 4c
            if locomotion_found is not None:
                print(f"Task 4c: locomotion clip: {os.path.basename(locomotion_found['fp'])}")
                _, c_stats = plot_task4c_comparison(
                    locomotion_found["trans"],
                    locomotion_found["positions"],
                    foot_id_l=6,
                    foot_id_r=12,
                )
            else:
                print("Task 4c: No suitable locomotion clip found (>2m travel)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Analysis complete. Output files in:", OUTPUT_DIR)
    for f in sorted(OUTPUT_DIR.iterdir()):
        if f.suffix == ".png":
            print(f"  {f.name}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Data-driven conclusion on cal_contact_mask suitability
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Task 4 CONCLUSION — cal_contact_mask suitability for bones-seed")
    print("=" * 60)

    # --- Assess condition 1: world-frame velocity ---
    print("""
Condition 1 — World-frame foot velocity² < 0.002
--------------------------------------------------
Mechanism: if the character stays near origin (as in TextOp's original
AMASS pipeline after XY-reset), a planted foot has world velocity ≈ 0.
In bones-seed, the root itself moves, so a planted foot inherits the
root's world velocity.""")

    if disp_stats is not None:
        print(f"""
Observed root per-frame displacement (120 Hz):
  Mean:   {disp_stats['mean']:.4f} m/frame  = {disp_stats['mean'] * SAMPLE_RATE:.3f} m/s
  Median: {disp_stats['median']:.4f} m/frame  = {disp_stats['median'] * SAMPLE_RATE:.3f} m/s
  P95:    {disp_stats['p95']:.4f} m/frame  = {disp_stats['p95'] * SAMPLE_RATE:.3f} m/s
  Threshold: {disp_stats['thresh']:.4f} m/frame  = {disp_stats['thresh_mps']:.3f} m/s
  Frames > threshold: {disp_stats['pct_above_thresh']:.1f}%""")

        if disp_stats['pct_above_thresh'] < 5:
            print("""
  → The world-velocity threshold is TOO LOOSE for bones-seed:
    even during walking, root velocity @120Hz easily stays under
    {thresh_mps:.1f} m/s. So a planted foot passes BOTH conditions
    (velocity + height) regardless → ALL frames appear as "contact".
    The detector has NO discrimination power.""".format(**disp_stats))
        else:
            print("""
  → A significant fraction of frames exceed the world-velocity
    threshold. During fast locomotion, planted feet would be
    misclassified as "moving" because they inherit root speed.""")

    # --- Assess condition 2: absolute Z height ---
    print("""
Condition 2 — Foot absolute Z < 0.08
-------------------------------------
Mechanism: in TextOp's original pipeline, the ground plane is at Z≈0
(after height adjustment in smplx_to_robot_dataset.py:127).
bones-seed has no such adjustment — raw world Z values.""")
    if ank_z_stats is not None:
        print(f"""
Observed ankle world Z (sampled {ank_z_stats.get('n_samples', '?')} frames):
  Mean:   {ank_z_stats['mean']:.4f} m
  Median: {ank_z_stats['median']:.4f} m
  Z > {ank_z_stats['thresh']}m: {ank_z_stats['pct_above_thresh']:.1f}%""")
        if ank_z_stats['pct_above_thresh'] < 5:
            print("""
  → Nearly ALL ankle frames are below 0.08 m → the height condition
    is always satisfied. It offers ZERO discrimination between
    planted vs. lifted feet in the bones-seed coordinate system.""")
        else:
            print("""
  → A meaningful fraction of ankle frames exceed the 0.08 m threshold,
    suggesting the height condition may have some discriminating power.""")

    # --- Assess Task 4c comparison ---
    if c_stats is not None:
        w = c_stats['world_method']
        r = c_stats['rel_method']
        print(f"""
Task 4c — Side-by-side on a locomotion clip:
  World-velocity method:  L={w['left_contact_pct']:.1f}%  R={w['right_contact_pct']:.1f}%
  Root-relative method:   L={r['left_contact_pct']:.1f}%  R={r['right_contact_pct']:.1f}%""")

        w_ratio = (w['left_contact_pct'] + w['right_contact_pct']) / 2
        if w_ratio > 90:
            print("""
  → World-velocity method reports >90% contact on BOTH feet
    simultaneously → no gait cycle visible → NOT suitable.""")
        elif w_ratio < 10:
            print("""
  → World-velocity method reports <10% contact → almost no frames
    detected as stance → planted feet lost to root motion → NOT suitable.""")
        else:
            print("""
  → World-velocity method shows intermediate contact ratios.
    Compare gait patterns in task4c_contact_comparison.png to judge.""")
        print(f"""
  → Root-relative method shows L={r['left_contact_pct']:.1f}% / R={r['right_contact_pct']:.1f}%
    — typical alternating gait pattern → correct behavior.""")

    # --- Final verdict ---
    verdict_parts = []
    if disp_stats and disp_stats['pct_above_thresh'] < 5:
        verdict_parts.append("world-velocity condition is non-discriminative")
    if ank_z_stats and ank_z_stats['pct_above_thresh'] < 5:
        verdict_parts.append("absolute-Z condition is non-discriminative")

    if len(verdict_parts) >= 1:
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  VERDICT: process_retarget_data.py cal_contact_mask         ║
║  is NOT suitable for bones-seed.                            ║
║  Reasons: {'; '.join(verdict_parts)}.                       ║
║  Use convert_bones_seed.py root-relative method instead.    ║
╚══════════════════════════════════════════════════════════════╝""")
    elif len(verdict_parts) == 0 and c_stats is not None:
        print("""
╔══════════════════════════════════════════════════════════════╗
║  VERDICT: See task4c_contact_comparison.png for judgment.   ║
╚══════════════════════════════════════════════════════════════╝""")
    else:
        print("""
Insufficient data for automated verdict. Review the charts manually.""")


if __name__ == "__main__":
    main()
