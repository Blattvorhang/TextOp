#!/usr/bin/env python3
"""Precompute dataset durations and BONES-SEED filter decisions for plotting."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from math import fsum
from pathlib import Path
import sys
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from dataset.data_process.filter_and_copy_bones_data import (  # noqa: E402
    DEFAULT_FILTER_KEYWORDS,
    should_filter_out,
)

OUTPUT = ROOT / "dataset/data_analyze/analysis_output/dataset_statistics/duration_comparison"
BONES_CARD = "https://huggingface.co/datasets/bones-studio/seed/blob/main/README.md"
G1_GUIDE = "https://github.com/NVlabs/ProtoMotions/blob/main/docs/source/getting_started/seed_g1_csv_preparation.rst"
PLACEHOLDERS = Path(__file__).with_name("duration_placeholders.json")


def sample_evenly(items, count=5):
    return [items[i] for i in np.linspace(0, len(items) - 1, min(count, len(items)), dtype=int)]


def count_csv_frames(path):
    with path.open("rb") as stream:
        return sum(bool(line.strip()) for line in stream) - 1


def read_bones(csv_dir, metadata, keywords, include_keywords):
    files = sorted(csv_dir.rglob("*.csv"))
    if not files:
        raise ValueError(f"No CSV motions in {csv_dir}")
    with metadata.open(newline="", encoding="utf-8") as stream:
        frames_by_path = {}
        for row in csv.DictReader(stream):
            relative = row["move_g1_path"].removeprefix("g1/csv/")
            if relative in frames_by_path:
                raise ValueError(f"Duplicate BONES-SEED metadata path: {relative}")
            frames_by_path[relative] = int(row["move_duration_frames"])

    selected, excluded = [], []
    raw_frames = processed_frames = 0
    for path in files:
        relative = path.relative_to(csv_dir).as_posix()
        if relative not in frames_by_path:
            raise ValueError(f"Missing BONES-SEED metadata: {relative}")
        frames = frames_by_path[relative]
        raw_frames += frames
        # The conversion stage writes date/stem.pkl; match the PKL filter's
        # parent/basename test without creating intermediate PKLs.
        filter_name = f"{path.parent.name}/{path.stem}.pkl"
        if should_filter_out(filter_name, keywords, include_keywords):
            excluded.append(relative)
        else:
            selected.append(relative)
            processed_frames += frames

    samples = sample_evenly(files)
    with ThreadPoolExecutor(max_workers=5) as pool:
        actual_counts = list(pool.map(count_csv_frames, samples))
    evidence = []
    for path, actual in zip(samples, actual_counts):
        expected = frames_by_path[path.relative_to(csv_dir).as_posix()]
        if actual != expected:
            raise ValueError(f"CSV/metadata frame mismatch: {path}: {actual} != {expected}")
        evidence.append(dict(path=str(path), csv_frames=actual, metadata_frames=expected))

    manifest = dict(schema_version=1, source=str(csv_dir.resolve()),
                    filter_script="dataset/data_process/filter_and_copy_bones_data.py",
                    filter_name="parent/stem.pkl", filter_keywords=keywords,
                    include_keywords=include_keywords, fps=120,
                    selected_files=len(selected), excluded_files=len(excluded),
                    selected=selected, excluded=excluded)
    data = dict(name="BONES-SEED", source=str(csv_dir.resolve()), files=len(files),
                fps_counts={"120": len(files)}, raw_frames=raw_frames,
                processed_frames=processed_frames, raw_seconds=raw_frames / 120,
                processed_seconds=processed_frames / 120,
                cleaning_breakdown=dict(filtered_files=len(excluded),
                                        retained_files=len(selected),
                                        filtered_frames=raw_frames - processed_frames,
                                        filtered_seconds=(raw_frames - processed_frames) / 120,
                                        filtered_share_of_raw=(raw_frames - processed_frames) / raw_frames),
                processing="Default filter_and_copy_bones_data keyword filter; CSV names mapped to PKL names",
                evidence=dict(metadata=str(metadata.resolve()), fps=120,
                              fps_sources=[BONES_CARD, G1_GUIDE],
                              filter_manifest="bones_filter_manifest.json", samples=evidence))
    return data, manifest


def read_hiphi(cleaning_json, collision_json, motions_dir):
    cleaning_bytes = cleaning_json.read_bytes()
    records = json.loads(cleaning_bytes)["motions"]
    if any(record["status"] != "ok" for record in records):
        raise ValueError("T-pose JSON contains failed motions")
    by_source = {record["source"]: record for record in records}
    if len(by_source) != len(records):
        raise ValueError("T-pose JSON contains duplicate sources")
    fps_counts = Counter(record["fps"] for record in records)
    samples = []
    for record in sample_evenly(records):
        path = motions_dir / record["source"]
        with zipfile.ZipFile(path) as archive, archive.open("fps.npy") as stream:
            fps = float(np.load(stream, allow_pickle=False).item())
        if fps != record["fps"]:
            raise ValueError(f"NPZ/T-pose JSON FPS mismatch: {path}")
        samples.append(dict(path=str(path), npz_fps=fps, cleaning_json_fps=record["fps"]))

    collision = json.loads(collision_json.read_text(encoding="utf-8"))
    selection = collision["selection"]
    if selection["scan_mirror_policy"] != "independent":
        raise ValueError("Duration comparison requires independently measured mirror collisions")
    if selection.get("tpose_json_sha256") != hashlib.sha256(cleaning_bytes).hexdigest():
        raise ValueError("Collision estimate was not computed from this T-pose JSON")
    if len({item["source"] for item in collision["candidates"]}) != len(collision["candidates"]):
        raise ValueError("Collision estimate contains duplicate sources")
    collision_frames = 0
    collision_seconds = []
    for candidate in collision["candidates"]:
        record = by_source[candidate["source"]]
        if ((candidate["fps"], candidate["frames"], candidate["kept_frames"])
                != (record["fps"], record["frames"], record["remaining_frames"])):
            raise ValueError(f"Collision/T-pose timeline mismatch: {candidate['source']}")
        collision_frames += record["remaining_frames"]
        collision_seconds.append(record["remaining_frames"] / record["fps"])
    raw_frames = sum(record["frames"] for record in records)
    post_tpose_frames = sum(record["remaining_frames"] for record in records)
    raw_seconds = fsum(record["frames"] / record["fps"] for record in records)
    post_tpose_seconds = fsum(record["remaining_frames"] / record["fps"] for record in records)
    collision_seconds = fsum(collision_seconds)
    if (collision["summary"]["total_motions"] != len(records)
            or collision["summary"]["post_tpose_frames"] != post_tpose_frames
            or collision["summary"]["candidate_full_motion_frames"] != collision_frames):
        raise ValueError("Collision estimate summary does not match T-pose records")
    return dict(name="HiPhi", source=str(motions_dir.resolve()), files=len(records),
                fps_counts={str(key): value for key, value in sorted(fps_counts.items())},
                raw_frames=raw_frames, processed_frames=post_tpose_frames - collision_frames,
                raw_seconds=raw_seconds, processed_seconds=post_tpose_seconds - collision_seconds,
                cleaning_breakdown=dict(
                    tpose_removed_frames=raw_frames - post_tpose_frames,
                    tpose_removed_seconds=raw_seconds - post_tpose_seconds,
                    tpose_removed_share_of_raw=(raw_seconds - post_tpose_seconds) / raw_seconds,
                    post_tpose_frames=post_tpose_frames,
                    post_tpose_seconds=post_tpose_seconds,
                    collision_candidate_motions=len(collision["candidates"]),
                    collision_removed_frames=collision_frames,
                    collision_removed_seconds=collision_seconds,
                    collision_removed_share_of_post_tpose=collision_seconds / post_tpose_seconds,
                    collision_removed_share_of_raw=collision_seconds / raw_seconds,
                    final_retained_share_of_raw=(post_tpose_seconds - collision_seconds) / raw_seconds),
                processing="T-pose removal then full-motion self-collision removal estimate; includes mirrors",
                evidence=dict(cleaning_json=str(cleaning_json.resolve()),
                              collision_json=str(collision_json.resolve()), samples=samples))


def read_placeholders(path):
    return [dict(name=item["name"], source=item["source"], placeholder=True,
                 files=None, fps_counts={}, raw_frames=None, processed_frames=None,
                 raw_seconds=item["hours"] * 3600, processed_seconds=item["hours"] * 3600,
                 processing="Published duration placeholder; no additional preprocessing or mirroring",
                 evidence=dict(item["evidence"], duration_source=item["source"]))
            for item in json.loads(path.read_text(encoding="utf-8"))]


def add_shares(datasets):
    totals = {}
    for stage in ("raw", "processed"):
        known = [data for data in datasets if data[f"{stage}_frames"] is not None]
        frames = sum(data[f"{stage}_frames"] for data in known)
        seconds = fsum(data[f"{stage}_seconds"] for data in datasets)
        totals[stage] = dict(frames=frames if len(known) == len(datasets) else None,
                             known_frames=frames, frame_share_scope="datasets_with_known_frames_only",
                             seconds=seconds, hours=seconds / 3600)
        for data in datasets:
            data[f"{stage}_hours"] = data[f"{stage}_seconds"] / 3600
            data[f"{stage}_duration_share"] = data[f"{stage}_seconds"] / seconds
            data[f"{stage}_frame_share"] = (data[f"{stage}_frames"] / frames
                                             if data[f"{stage}_frames"] is not None else None)
    return totals


def write_report(datasets, totals, output):
    bones = next(data for data in datasets if data["name"] == "BONES-SEED")
    hiphi = next(data for data in datasets if data["name"] == "HiPhi")
    bones_filter = bones["cleaning_breakdown"]
    hiphi_filter = hiphi["cleaning_breakdown"]
    rows = ["# 数据集时长占比", "",
            "外圈是原始数据，内圈是按筛选规则计算的拟保留数据；两圈分别归一化。原始 CSV/NPZ 均未改写。", "",
            "| 数据集 | 文件数 | 原始帧数 | 拟保留帧数 | 原始小时 | 拟保留小时 | 原始时长占比 | 拟保留时长占比 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for data in datasets:
        value = lambda key: f"{data[key]:,}" if data[key] is not None else "—"
        rows.append(f"| {data['name']} | {value('files')} | {value('raw_frames')} | {value('processed_frames')} | "
                    f"{data['raw_hours']:.4f} | {data['processed_hours']:.4f} | "
                    f"{data['raw_duration_share']:.2%} | {data['processed_duration_share']:.2%} |")
    rows += ["", f"总时长：原始 {totals['raw']['hours']:.4f} 小时；拟保留 {totals['processed']['hours']:.4f} 小时。", "",
             "## BONES-SEED 筛选", "",
             f"按 `filter_and_copy_bones_data.py` 的默认关键词排除 {bones_filter['filtered_files']:,} / {bones['files']:,} 条，"
             f"{bones_filter['filtered_frames']:,} 帧（{bones_filter['filtered_seconds'] / 3600:.4f} 小时，"
             f"占原始 BONES-SEED 时长 {bones_filter['filtered_share_of_raw']:.2%}）。",
             "逐文件决定及关键词见 `bones_filter_manifest.json`。按原始 CSV 的 120 fps 元数据计算；实际流水线在转换为 50 fps PKL 后过滤。", "",
             "## HiPhi 分阶段剔除", "",
             "| 阶段 | 小时 | 占 HiPhi 原始时长 | 占 T-pose 后时长 |",
             "| --- | ---: | ---: | ---: |",
             f"| T-pose 删除 | {hiphi_filter['tpose_removed_seconds'] / 3600:.4f} | {hiphi_filter['tpose_removed_share_of_raw']:.2%} | — |",
             f"| 自碰撞拟剔除（{hiphi_filter['collision_candidate_motions']:,} 条） | {hiphi_filter['collision_removed_seconds'] / 3600:.4f} | {hiphi_filter['collision_removed_share_of_raw']:.2%} | {hiphi_filter['collision_removed_share_of_post_tpose']:.2%} |",
             f"| 最终拟保留 | {hiphi['processed_hours']:.4f} | {hiphi_filter['final_retained_share_of_raw']:.2%} | — |",
             "", "自碰撞阈值：连续至少 0.5 秒、每帧穿透至少 3 cm、区间峰值至少 5 cm；命中后拟剔除该动作的全部 T-pose 保留区间。", "",
             "AMASS-G1 与 LAFAN1-G1 的时长为公开版本占位值，尚未下载实测，不额外镜像。", "",
             "## 证据", ""]
    for data in datasets:
        rows += [f"### {data['name']}", "", "```json",
                 json.dumps(data["evidence"], ensure_ascii=False, indent=2), "```", ""]
    (output / "REPORT.md").write_text("\n".join(rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bones-csv", type=Path, default=ROOT / "data/bones-seed/g1/csv")
    parser.add_argument("--bones-metadata", type=Path, default=ROOT / "data/bones-seed/metadata/seed_metadata_v004.csv")
    parser.add_argument("--filter-keywords", nargs="+", default=DEFAULT_FILTER_KEYWORDS)
    parser.add_argument("--include-file", type=Path)
    parser.add_argument("--hiphi-motions", type=Path, default=ROOT / "data/hiphi/motions")
    parser.add_argument("--hiphi-json", type=Path, default=ROOT / "dataset/data_analyze/analysis_output/hiphi_tpose/full_scan_endpoints_adaptive/removal_intervals.json")
    parser.add_argument("--collision-json", type=Path, default=ROOT / "dataset/data_analyze/analysis_output/hiphi_self_collision_after_tpose/removal_estimate.json")
    parser.add_argument("--placeholders", type=Path, default=PLACEHOLDERS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    include_keywords = args.include_file.read_text(encoding="utf-8").splitlines() if args.include_file else None
    bones, manifest = read_bones(args.bones_csv, args.bones_metadata, args.filter_keywords, include_keywords)
    datasets = [bones, read_hiphi(args.hiphi_json, args.collision_json, args.hiphi_motions),
                *read_placeholders(args.placeholders)]
    totals = add_shares(datasets)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "bones_filter_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "statistics.json").write_text(
        json.dumps(dict(datasets=datasets, totals=totals), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    write_report(datasets, totals, args.output)
    print(f"BONES-SEED: {bones['files']:,} -> {bones['cleaning_breakdown']['retained_files']:,} motions; "
          f"{bones['raw_hours']:.2f} -> {bones['processed_hours']:.2f} h")
    print(f"HiPhi: {datasets[1]['raw_hours']:.2f} -> {datasets[1]['processed_hours']:.2f} h")
    print(f"Saved statistics and filter manifest to {args.output}")


if __name__ == "__main__":
    main()
