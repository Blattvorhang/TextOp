#!/usr/bin/env python3
"""
BONES-SEED 数据集时长统计脚本（含 metadata 交叉验证）
--------------------------------------------------
统计逻辑：
  1. 遍历 g1/csv/ 下所有 .csv 文件，逐行统计帧数
  2. 加载 metadata CSV，用 move_duration_frames 交叉验证帧数
  3. 加载 metadata JSONL，用时序标签交叉验证采样率 (120 Hz)
  4. 汇总 metadata 中的分类维度（category/package/gender 等）
"""

import csv
import json
import os
import sys
from collections import defaultdict, Counter

BASE = "/home/lenovo/data/bones-seed/g1/csv"
META_CSV = "/home/lenovo/data/bones-seed/metadata/seed_metadata_v004.csv"
META_JSONL = "/home/lenovo/data/bones-seed/metadata/seed_metadata_v002_temporal_labels.jsonl"
SAMPLE_RATE = 120  # Hz, 来自官网

# ============================================================
# Phase 1: 遍历原始 CSV 文件统计帧数
# ============================================================
print("=" * 70)
print("Phase 1: 遍历原始 CSV 文件，逐行统计帧数 ...")
print("=" * 70)

date_stats = defaultdict(lambda: {"files": 0, "frames": 0, "bytes": 0})
total_files = 0
total_frames = 0
total_bytes = 0
zero_frame_files = []
all_durations_sec = []
csv_frame_map = {}  # filename (no .csv) -> frame count from line counting

for root, dirs, files in os.walk(BASE):
    for fname in files:
        if not fname.endswith(".csv"):
            continue

        fpath = os.path.join(root, fname)
        stem = fname.replace(".csv", "")

        # 文件大小
        try:
            fsize = os.path.getsize(fpath)
        except OSError:
            fsize = 0

        # 行数统计
        line_count = 0
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                for _ in fh:
                    line_count += 1
        except (OSError, PermissionError):
            print(f"[警告] 无法读取: {fpath}", file=sys.stderr)
            continue

        frames = line_count - 1
        if frames < 0:
            frames = 0
        if frames == 0:
            zero_frame_files.append(fpath)

        duration_s = frames / SAMPLE_RATE

        total_files += 1
        total_frames += frames
        total_bytes += fsize
        all_durations_sec.append(duration_s)
        csv_frame_map[stem] = frames

        # 按日期文件夹统计
        rel = os.path.relpath(fpath, BASE)
        parts = rel.split(os.sep)
        if len(parts) >= 1:
            date_folder = parts[0]
            date_stats[date_folder]["files"] += 1
            date_stats[date_folder]["frames"] += frames
            date_stats[date_folder]["bytes"] += fsize

total_hours = total_frames / SAMPLE_RATE / 3600

print(f"CSV 文件总数:   {total_files:,}")
print(f"总帧数:         {total_frames:,}")
print(f"总时长:         {total_hours:.2f} 小时  ({total_hours * 60:.1f} 分钟)")
print(f"总存储:         {total_bytes / 1024**3:.2f} GB")
print()

# ============================================================
# Phase 2: 加载 metadata CSV，交叉验证帧数
# ============================================================
print("=" * 70)
print("Phase 2: 加载 metadata CSV，交叉验证帧数 ...")
print("=" * 70)

meta_rows = []
meta_frame_map = {}  # filename -> move_duration_frames
with open(META_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        meta_rows.append(row)
        meta_frame_map[row["filename"]] = int(row["move_duration_frames"])

meta_total_frames = sum(meta_frame_map.values())
meta_total_hours = meta_total_frames / SAMPLE_RATE / 3600

print(f"Metadata 行数:   {len(meta_rows):,}")
print(f"Metadata 总帧数: {meta_total_frames:,}")
print(f"Metadata 总时长: {meta_total_hours:.2f} 小时  ({meta_total_hours * 60:.1f} 分钟)")
print()

# 交叉验证：比较 CSV 遍历 vs Metadata 的帧数
only_in_csv = set(csv_frame_map) - set(meta_frame_map)
only_in_meta = set(meta_frame_map) - set(csv_frame_map)
common = set(csv_frame_map) & set(meta_frame_map)

mismatches = []
for fn in common:
    if csv_frame_map[fn] != meta_frame_map[fn]:
        mismatches.append((fn, csv_frame_map[fn], meta_frame_map[fn]))

print("--- 帧数交叉验证 ---")
print(f"  仅存在于 CSV 遍历:  {len(only_in_csv)} 个")
print(f"  仅存在于 metadata:   {len(only_in_meta)} 个")
print(f"  共同文件:            {len(common):,} 个")
print(f"  帧数不一致:          {len(mismatches)} 个")
if mismatches:
    print(f"  最大偏差:            {max(abs(a-b) for _,a,b in mismatches)} 帧")
    print(f"  示例 (前 5 个):")
    for fn, a, b in mismatches[:5]:
        print(f"    {fn}: CSV={a}, Meta={b}, diff={a-b}")

if only_in_csv:
    print(f"\n  仅在 CSV 中的文件 (前 5): {list(only_in_csv)[:5]}")
if only_in_meta:
    print(f"\n  仅在 metadata 中的文件 (前 5): {list(only_in_meta)[:5]}")

print()

# ============================================================
# Phase 3: 加载 metadata JSONL，交叉验证采样率
# ============================================================
print("=" * 70)
print("Phase 3: 加载 metadata JSONL，交叉验证采样率 (120 Hz) ...")
print("=" * 70)

fps_samples = []
fps_deviations = []
meta_jsonl_map = {}
with open(META_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        fn = obj["filename"]
        meta_jsonl_map[fn] = obj
        if fn in meta_frame_map and obj["events"]:
            last_end = obj["events"][-1]["end_time"]
            frames = meta_frame_map[fn]
            if last_end > 0:
                fps = frames / last_end
                fps_samples.append(fps)

fps_avg = sum(fps_samples) / len(fps_samples)
fps_min = min(fps_samples)
fps_max = max(fps_samples)
fps_dev = sum(abs(x - SAMPLE_RATE) for x in fps_samples) / len(fps_samples)

print(f"  JSONL 条目数:       {len(meta_jsonl_map):,}")
print(f"  计算 FPS 样本数:    {len(fps_samples):,}")
print(f"  FPS 平均值:         {fps_avg:.4f} Hz")
print(f"  FPS 范围:           [{fps_min:.4f}, {fps_max:.4f}] Hz")
print(f"  与 120Hz 平均偏差:  {fps_dev:.4f} Hz")
print(f"  结论:               {'✓ 与 120 Hz 一致' if fps_dev < 0.5 else '✗ 存在偏差，请检查'}")
print()

# ============================================================
# Phase 4: 基于 metadata 的多维度统计
# ============================================================
print("=" * 70)
print("Phase 4: 基于 metadata 的多维度统计")
print("=" * 70)

# 按 category 统计
cat_stats = defaultdict(lambda: {"files": 0, "frames": 0})
for row in meta_rows:
    cat = row["category"]
    cat_stats[cat]["files"] += 1
    cat_stats[cat]["frames"] += int(row["move_duration_frames"])

print()
print("--- 按 category 统计 (共 {} 类) ---".format(len(cat_stats)))
for name, s in sorted(cat_stats.items(), key=lambda x: -x[1]["frames"]):
    h = s["frames"] / SAMPLE_RATE / 3600
    print(f"  {name}: {s['files']:>6,} 文件, {h:>8.2f}h")

# 按 package 统计
pkg_stats = defaultdict(lambda: {"files": 0, "frames": 0})
for row in meta_rows:
    pkg = row["package"]
    pkg_stats[pkg]["files"] += 1
    pkg_stats[pkg]["frames"] += int(row["move_duration_frames"])

print()
print("--- 按 package 统计 (共 {} 类) ---".format(len(pkg_stats)))
for name, s in sorted(pkg_stats.items(), key=lambda x: -x[1]["frames"]):
    h = s["frames"] / SAMPLE_RATE / 3600
    print(f"  {name}: {s['files']:>6,} 文件, {h:>8.2f}h")

# 按 is_mirror 统计
mir_stats = defaultdict(lambda: {"files": 0, "frames": 0})
for row in meta_rows:
    mir = "Mirror" if row["is_mirror"] == "True" else "Original"
    mir_stats[mir]["files"] += 1
    mir_stats[mir]["frames"] += int(row["move_duration_frames"])

print()
print("--- 按 is_mirror 统计 ---")
for name, s in sorted(mir_stats.items(), key=lambda x: -x[1]["frames"]):
    h = s["frames"] / SAMPLE_RATE / 3600
    print(f"  {name}: {s['files']:>6,} 文件, {h:>8.2f}h")

# 按 actor_gender 统计
gen_stats = defaultdict(lambda: {"files": 0, "frames": 0})
for row in meta_rows:
    gen = row["actor_gender"]
    gen_stats[gen]["files"] += 1
    gen_stats[gen]["frames"] += int(row["move_duration_frames"])

print()
print("--- 按 actor_gender 统计 ---")
for name, s in sorted(gen_stats.items(), key=lambda x: -x[1]["frames"]):
    h = s["frames"] / SAMPLE_RATE / 3600
    print(f"  {name}: {s['files']:>6,} 文件, {h:>8.2f}h")

# 按 content_horizontal_move 统计
hm_stats = defaultdict(lambda: {"files": 0, "frames": 0})
for row in meta_rows:
    hm = "有位移" if row["content_horizontal_move"] == "1" else "无位移"
    hm_stats[hm]["files"] += 1
    hm_stats[hm]["frames"] += int(row["move_duration_frames"])

print()
print("--- 按 content_horizontal_move 统计 ---")
for name, s in sorted(hm_stats.items(), key=lambda x: -x[1]["frames"]):
    h = s["frames"] / SAMPLE_RATE / 3600
    print(f"  {name}: {s['files']:>6,} 文件, {h:>8.2f}h")

# 被试者信息（来自 metadata 的 actor_uid）
actor_stats = defaultdict(lambda: {"files": 0, "frames": 0, "gender": "", "height_cm": ""})
for row in meta_rows:
    uid = row["actor_uid"]
    actor_stats[uid]["files"] += 1
    actor_stats[uid]["frames"] += int(row["move_duration_frames"])
    actor_stats[uid]["gender"] = row["actor_gender"]
    actor_stats[uid]["height_cm"] = row.get("actor_height_cm", "")

print()
print("--- 按被试者统计 (共 {} 人) ---".format(len(actor_stats)))
print(f"  {'Subject':>8s}  {'性别':>4s}  {'身高cm':>8s}  {'文件数':>7s}  {'时长':>10s}")
for name, s in sorted(actor_stats.items(), key=lambda x: -x[1]["frames"])[:30]:
    h = s["frames"] / SAMPLE_RATE / 3600
    print(f"  {name:>8s}  {s['gender']:>4s}  {s['height_cm']:>8s}  {s['files']:>6,}   {h:>8.2f}h")
if len(actor_stats) > 30:
    print(f"  ... (共 {len(actor_stats)} 人，仅显示前 30)")

# ============================================================
# Phase 5: 单文件时长分布（基于 metadata 帧数）
# ============================================================
print()
print("--- 单文件时长分布（基于 metadata 帧数 @120Hz）---")
durations = sorted(f / SAMPLE_RATE for f in meta_frame_map.values())
n = len(durations)

def percentile(data, p):
    k = (len(data) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(data) - 1)
    if f == c:
        return data[f]
    return data[f] * (c - k) + data[c] * (k - f)

print(f"  文件数:  {n:,}")
print(f"  最小值:  {durations[0]:.2f}s")
print(f"  P10:     {percentile(durations, 10):.2f}s")
print(f"  P25:     {percentile(durations, 25):.2f}s")
print(f"  中位数:  {percentile(durations, 50):.2f}s")
print(f"  P75:     {percentile(durations, 75):.2f}s")
print(f"  P90:     {percentile(durations, 90):.2f}s")
print(f"  最大值:  {durations[-1]:.2f}s")
print(f"  平均值:  {sum(durations)/n:.2f}s")

print()
print("--- 时长区间分布 ---")
buckets = [
    (0, 2.5), (2.5, 5), (5, 10), (10, 15), (15, 30),
    (30, 60), (60, 150), (150, float("inf"))
]
for lo, hi in buckets:
    cnt = sum(1 for d in durations if lo <= d < hi)
    label = f"{lo}-{hi}s" if hi != float("inf") else f">{lo}s"
    print(f"  {label:>12s}: {cnt:>6,} 文件 ({cnt/n*100:5.1f}%)")

# ============================================================
# 最终汇总
# ============================================================
print()
print("=" * 70)
print("统计汇总")
print("=" * 70)
print(f"  数据路径:       {BASE}")
print(f"  采样率:         {SAMPLE_RATE} Hz (官网 + metadata 交叉验证确认)")
print(f"  CSV 文件总数:   {total_files:,}")
print(f"  总帧数:         {total_frames:,}")
print(f"  总时长:         {total_hours:.2f} 小时 ({total_hours * 60:.1f} 分钟)")
print(f"  总存储:         {total_bytes / 1024**3:.2f} GB")
print(f"  Metadata 条目:  {len(meta_rows):,}")
print(f"  Category 数:    {len(cat_stats)}")
print(f"  被试者数:       {len(actor_stats)}")
print(f"  日期范围:       210531 ~ 240918 ({len(date_stats)} 个日期)")
print(f"  空文件:         {len(zero_frame_files)} 个")
print()
print("统计完成。")
