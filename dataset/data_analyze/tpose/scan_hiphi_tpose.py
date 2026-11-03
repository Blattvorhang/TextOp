#!/usr/bin/env python3
"""Index T-pose removal/keep intervals and compute remaining HiPhi duration.

Source NPZs are read-only. Default: trim confirmed endpoint calibration only.
The previous two-standing-witness cycle detector is available with --detector cycle.
The JSON is reusable: --from-json recomputes all statistics without NPZ reads.
"""

import argparse
from collections import defaultdict
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import multiprocessing
from pathlib import Path
import random
import time
import zipfile

if __package__:
    from .analyze_hiphi_tpose import PRECISION_RULES, load_features, precision_candidates
    from .detect_endpoint_tpose import ENDPOINT_RULES, REMOVAL_POLICY as ENDPOINT_REMOVAL_POLICY, detect_endpoints
else:
    from analyze_hiphi_tpose import PRECISION_RULES, load_features, precision_candidates
    from detect_endpoint_tpose import ENDPOINT_RULES, REMOVAL_POLICY as ENDPOINT_REMOVAL_POLICY, detect_endpoints


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "analysis_output/hiphi_tpose/full_scan"
DEFAULT_ENDPOINT_OUTPUT = DEFAULT_OUTPUT.with_name("full_scan_endpoints")
SCHEMA_VERSION = 1
REMOVAL_POLICY = "[before_standing_end, after_standing_start); preserve both standing witnesses"


def merge_intervals(intervals, frames):
    merged = []
    for start, end in sorted(intervals):
        if not 0 <= start < end <= frames:
            raise ValueError(f"Invalid interval [{start}, {end}) for {frames} frames")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def complement(intervals, frames):
    result, cursor = [], 0
    for start, end in intervals:
        if cursor < start:
            result.append([cursor, start])
        cursor = end
    if cursor < frames:
        result.append([cursor, frames])
    return result


def interval_fields(frames, fps, intervals):
    """Normalize intervals and build the timeline fields shared by all records."""
    intervals = merge_intervals(intervals, frames)
    removed = sum(b - a for a, b in intervals)
    return dict(frames=frames, fps=fps, duration_seconds=frames / fps,
                remove_intervals_frames=intervals,
                remove_intervals_seconds=[[a / fps, b / fps] for a, b in intervals],
                keep_intervals_frames=complement(intervals, frames),
                removed_frames=removed, removed_seconds=removed / fps,
                remaining_frames=frames - removed, remaining_seconds=(frames - removed) / fps)


def fingerprint(path):
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def canonical_source(relative):
    path = Path(relative)
    if not path.parent.name.endswith("__mirror"):
        return None
    return str(path.parent.with_name(path.parent.name.removesuffix("__mirror")) / path.name)


def read_timeline(path):
    """Read the joint array header and scalar FPS, without loading trajectories."""
    import numpy as np
    with zipfile.ZipFile(path) as archive:
        with archive.open("joint_pos.npy") as stream:
            version = np.lib.format.read_magic(stream)
            reader = {(1, 0): np.lib.format.read_array_header_1_0,
                      (2, 0): np.lib.format.read_array_header_2_0}.get(version)
            if reader is None:
                raise ValueError(f"Unsupported NPY header {version}")
            shape, _, _ = reader(stream)
        with archive.open("fps.npy") as stream:
            fps = float(np.load(stream, allow_pickle=False).item())
    if len(shape) != 2 or shape[1] != 29 or shape[0] <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid timeline: {shape}, {fps}")
    return int(shape[0]), fps


def can_inherit_mirror(original, mirror):
    return (original is not None and original["status"] == mirror["status"] == "ok"
            and original["fps"] == mirror["fps"] and abs(original["frames"] - mirror["frames"]) <= 1)


def inherit_mirror(original, mirror):
    if not can_inherit_mirror(original, mirror):
        raise ValueError("Mirror inheritance requires matching FPS and at most one frame difference")
    result = deepcopy(original)
    for key in ("source", "source_stat", "is_mirror", "blacklist_matches"):
        result[key] = deepcopy(mirror[key])
    frames, fps = mirror["frames"], mirror["fps"]
    # Keep interior cut points; a tail deletion must reach the mirror's actual end.
    intervals = [[min(a, frames), frames if b == original["frames"] else min(b, frames)]
                 for a, b in original["remove_intervals_frames"]]
    result.update(interval_fields(frames, fps, [[a, b] for a, b in intervals if a < b]))
    result.update(interval_method="canonical_inherited", interval_source=original["source"],
                  evidence_source=original["source"], canonical_source_stat=deepcopy(original["source_stat"]))
    return result


def scan_mirror(task):
    dataset, relative, patterns, detector, original = task
    reason = "missing_or_invalid_canonical"
    if original is not None and original["status"] == "ok":
        try:
            source = Path(dataset) / "motions" / relative
            frames, fps = read_timeline(source)
            source_stat = fingerprint(source)
        except (OSError, ValueError, EOFError, zipfile.BadZipFile):
            reason = "timeline_metadata_unavailable"
        else:
            semantic = "/".join(Path(relative).parts[:2])
            metadata = dict(status="ok", source=relative, is_mirror=True, frames=frames, fps=fps,
                            source_stat=source_stat,
                            blacklist_matches=[p for p in patterns if fnmatch.fnmatch(semantic, p)
                                               or fnmatch.fnmatch(relative, p)])
            if can_inherit_mirror(original, metadata):
                return inherit_mirror(original, metadata)
            reason = "timeline_mismatch"
    result = scan_one((dataset, relative, patterns, detector))
    result["mirror_reuse_status"] = reason
    return result


def scan_one(task):
    dataset, relative, patterns, *selection = task
    detector = selection[0] if selection else "cycle"  # Preserve existing sampling callers.
    source = Path(dataset) / "motions" / relative
    semantic = "/".join(Path(relative).parts[:2])
    record = dict(source=relative, detector=detector, is_mirror="__mirror" in relative,
                  interval_method="detected", interval_source=relative, evidence_source=relative,
                  blacklist_matches=[p for p in patterns if fnmatch.fnmatch(semantic, p)
                                     or fnmatch.fnmatch(relative, p)])
    try:
        record["source_stat"] = fingerprint(source)
        features = load_features(source)
        fps, frames = float(features["fps"]), len(features["q"])
        import numpy as np
        if not np.isfinite(fps) or fps <= 0 or frames <= 0:
            raise ValueError(f"Invalid fps/frames: {fps}/{frames}")
        for key in ("q", "pos", "root_speed", "foot_speed", "tilt", "upper_arm", "elbow_deflection"):
            if not np.isfinite(features[key]).all():
                raise ValueError(f"Non-finite {key}")
        if detector == "endpoint":
            detected = detect_endpoints(source, features=features)
            cycles = detected["candidates"]
            intervals = [cycle["remove_frames"] for cycle in cycles]
            record["endpoint_review_flags"] = detected["review_flags"]
        elif detector == "cycle":
            cycles = precision_candidates(source, features=features)
            intervals = [
                [cycle["before_standing_frames"][1], cycle["after_standing_frames"][0]]
                for cycle in cycles]
        else:
            raise ValueError(f"Unknown detector: {detector}")
        record.update(interval_fields(frames, fps, intervals), status="ok", detected_cycles=cycles)
    except (OSError, ValueError, EOFError) as error:
        record.update(status="error", error=f"{type(error).__name__}: {error}")
    return record


def aggregate(records):
    result = dict(files=0, valid_files=0, errors=0, files_with_removals=0,
                  detected_cycles=0, removal_intervals=0, remaining_segments=0,
                  original_frames=0, removed_frames=0, remaining_frames=0,
                  original_seconds=0., removed_seconds=0., remaining_seconds=0.,
                  remaining_segments_under_64_frames=0, remaining_frames_in_segments_under_64=0)
    result.update(endpoint_head_detections=0, endpoint_tail_detections=0,
                  endpoint_files_both_ends=0, endpoint_unconfirmed_ends=0,
                  files_head_trimmed=0, files_tail_trimmed=0, files_both_ends_trimmed=0,
                  mirror_inherited_files=0, mirror_independent_files=0)
    for record in records:
        result["files"] += 1
        if record["status"] != "ok":
            result["errors"] += 1
            continue
        frames, fps = record["frames"], record["fps"]
        intervals = record["remove_intervals_frames"]
        kept = record["keep_intervals_frames"]
        removed = record["removed_frames"]
        result["valid_files"] += 1
        inherited = record.get("interval_method") == "canonical_inherited"
        result["mirror_inherited_files"] += record["is_mirror"] and inherited
        result["mirror_independent_files"] += record["is_mirror"] and not inherited
        result["files_with_removals"] += bool(intervals)
        head_trimmed = bool(intervals and intervals[0][0] == 0)
        tail_trimmed = bool(intervals and intervals[-1][1] == frames)
        result["files_head_trimmed"] += head_trimmed
        result["files_tail_trimmed"] += tail_trimmed
        result["files_both_ends_trimmed"] += head_trimmed and tail_trimmed
        result["detected_cycles"] += len(record["detected_cycles"])
        locations = {c.get("location") for c in record["detected_cycles"]}
        result["endpoint_head_detections"] += "head" in locations
        result["endpoint_tail_detections"] += "tail" in locations
        result["endpoint_files_both_ends"] += {"head", "tail"}.issubset(locations)
        result["endpoint_unconfirmed_ends"] += len(record.get("endpoint_review_flags", []))
        result["removal_intervals"] += len(intervals)
        result["remaining_segments"] += len(kept)
        result["original_frames"] += frames
        result["removed_frames"] += removed
        result["remaining_frames"] += frames - removed
        result["original_seconds"] += frames / fps
        result["removed_seconds"] += removed / fps
        result["remaining_seconds"] += (frames - removed) / fps
        short = [b - a for a, b in kept if b - a < 64]
        result["remaining_segments_under_64_frames"] += len(short)
        result["remaining_frames_in_segments_under_64"] += sum(short)
    for field in ("original", "removed", "remaining"):
        result[f"{field}_hours"] = result[f"{field}_seconds"] / 3600
    result["removed_frame_fraction"] = result["removed_frames"] / max(1, result["original_frames"])
    result["removed_duration_fraction"] = result["removed_seconds"] / max(1e-8, result["original_seconds"])
    result["remaining_duration_fraction"] = result["remaining_seconds"] / max(1e-8, result["original_seconds"])
    return result


def build_summary(records):
    categories = defaultdict(list)
    for record in records:
        # Validate once before aggregating the same records in several views.
        if record["status"] == "ok":
            expected = interval_fields(record["frames"], record["fps"], record["remove_intervals_frames"])
            if any(record[key] != expected[key] for key in (
                    "remove_intervals_frames", "keep_intervals_frames", "removed_frames", "remaining_frames")):
                raise ValueError(f"Inconsistent intervals/counts in {record['source']}")
        categories[record["source"].split("/")[0]].append(record)
    return dict(
        all_indexed=aggregate(records),
        canonical_only=aggregate(r for r in records if not r["is_mirror"]),
        mirror_only=aggregate(r for r in records if r["is_mirror"]),
        after_existing_blacklist=aggregate(r for r in records if not r["blacklist_matches"]),
        blacklist_excluded=aggregate(r for r in records if r["blacklist_matches"]),
        per_category={category: aggregate(rows) for category, rows in sorted(categories.items())},
        errors=[dict(source=r["source"], error=r["error"]) for r in records if r["status"] != "ok"],
    )


def write_json(path, value, *, indent=None):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=indent, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def write_outputs(output, payload):
    output.mkdir(parents=True, exist_ok=True)
    summary = build_summary(payload["motions"])
    payload["summary"] = summary
    write_json(output / "removal_intervals.json", payload)
    write_json(output / "statistics.json", summary, indent=2)
    endpoint_mode = payload["config"].get("detector") == "endpoint"
    adaptive_mode = payload["config"].get("detector") == "endpoint_adaptive"
    policy_rows = ([
        "- 仅检测靠近开头/结尾的校准姿态，中间抬放臂动作全部保留。",
        "- 搜索区为每端最多 10 秒且不超过文件时长的 15%；每端最多删除 12 秒且不超过时长的 20%。",
        "- 每端只取最靠边的一次：必须存在连续至少 0.6 秒的稳定双臂侧向伸展证据，并验证直立、肘部、双脚与速度。",
        "- 肩外展 65°–120°、左右差 ≤20°、上臂侧向余弦 ≥0.90、肘偏折 ≤25°；0.4 秒肩角滑动极差 ≤10°。",
        "- 无需外侧站立证据；内侧须在 2 秒内存在至少 0.25 秒双臂放下、躯干直立的姿态，允许腿部仍在行走过渡。",
        "- 开头删除 `[0, 内侧低臂证据窗口末尾)`；结尾删除 `[内侧低臂证据窗口起点, 总帧数)`，覆盖外侧准备与收尾小动作。",
        "- 没有足够姿态证据或超过删除上限时保留，并记录 endpoint_review_flags；不无条件固定裁掉每条首尾。",
    ] if endpoint_mode else [
        "- 必须先通过严格 T-pose 检测，以及前后稳定站立、站姿匹配条件。",
        "- 剔除区间为 `[前站立证据窗口末帧之后, 后站立证据窗口首帧)`；包含抬臂、保持、放臂及恢复站立前的小动作。",
        "- 前后站立证据本身保留；这是历史全时域周期规则，可能误检动作中的重复抬放臂。",
    ])
    if adaptive_mode:
        rules = payload["config"]["rules"]
        policy_rows = [
            "- 从每端向内寻找最靠边的双肩 ≥60°、持续 ≥0.2 秒的片段；起点限于 min(10 秒, 15% 时长)。",
            "- 随后 3 秒内寻找双肩 ≤45°且持续 ≥0.2 秒的窗口，以窗口内侧末端为测量边界。结尾反向测量。",
            "- 边界超过 min(12 秒, 20% 时长) 视为无法可靠定界，避免将持续抬臂的真实动作整段删除。",
            f"- 可测量时裁到转换边界后 {rules['adaptive_margin_seconds']:g} 秒；无法定界时该端固定裁掉 {rules['fallback_seconds']:g} 秒。每条动作两端均裁剪。",
            "- 保留旧规则更长的删除区间；测量预算不限制额外余量、兜底或旧规则区间。两端重叠则整条剔除。",
            "- 镜像直接继承原片切点，末尾对齐自身实际帧数；中间动作不额外检测。",
            "- endpoint_trim_decisions 保存各端 adaptive/fallback 决策及测量状态；detected_cycles / endpoint_review_flags 仅为旧规则证据。",
        ]
    rows = [
        "# HiPhi T-pose 剔除后的数据量", "",
        f"按 {'首尾自适应裁剪规则' if adaptive_mode else '首尾校准姿态规则' if endpoint_mode else '历史严格周期规则'} 统计原始动作，包括镜像。",
        ("镜像默认继承原片区间，允许相差 1 帧，末尾删除边界对齐镜像实际末帧；帧率不同、帧数相差超过 1 帧或原片结果无效时独立检测。"
         if payload["config"].get("mirror_policy") == "inherit" else "本次镜像独立检测。"),
        "JSON 只记录建议剔除和保留区间，不改写原始 NPZ。", "",
        "## 统计口径", "",
        *policy_rows,
        "- 重叠/相邻剔除区间先合并，剩余每个连续区间视为独立片段，不跨越被删间隔拼接。",
        "- 时长按 `帧数 / 各文件 FPS`，帧索引从 0 开始，区间左闭右开。",
        "- 保留全部剩余帧，不默认丢弃短片段；JSON 另统计不足 64 帧的段数和帧数。",
        "- 已有 blacklist 仅提供一个独立统计视图；全量统计不把类别黑名单混入 T-pose 删除量。", "",
        "| 范围 | 文件数 | 原始帧数 | 剔除帧数 | 剩余帧数 | 原始小时 | 剔除小时 | 剩余小时 | 时长剔除比例 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, key in [("全部（含镜像）", "all_indexed"), ("原片（不含镜像）", "canonical_only"),
                       ("仅镜像", "mirror_only"), ("应用已有 blacklist 后", "after_existing_blacklist")]:
        s = summary[key]
        rows.append(f"| {label} | {s['valid_files']:,} | {s['original_frames']:,} | {s['removed_frames']:,} | "
                    f"{s['remaining_frames']:,} | {s['original_hours']:.4f} | {s['removed_hours']:.4f} | "
                    f"{s['remaining_hours']:.4f} | {s['removed_duration_fraction']:.3%} |")
    all_stats = summary["all_indexed"]
    if adaptive_mode:
        rows += ["", f"实际裁剪开头 {all_stats['files_head_trimmed']:,} 条，结尾 {all_stats['files_tail_trimmed']:,} 条，"
                 f"首尾均裁剪 {all_stats['files_both_ends_trimmed']:,} 条。",
                 f"独立原片端点：自适应 {payload['run']['adaptive_original_edges']:,} 个，兜底 {payload['run']['fallback_original_edges']:,} 个。"]
    rows += ["", f"受影响文件 {all_stats['files_with_removals']:,} 条；{'历史姿态证据段' if adaptive_mode else '检测段'} {all_stats['detected_cycles']:,} 个；"
             f"合并后剔除区间 {all_stats['removal_intervals']:,} 段；剩余连续片段 {all_stats['remaining_segments']:,} 段。",
             f"不足 64 帧的剩余片段 {all_stats['remaining_segments_under_64_frames']:,} 段，"
             f"共 {all_stats['remaining_frames_in_segments_under_64']:,} 帧；上述总量仍包含这些片段。", "",
             f"读取错误 {all_stats['errors']} 条；错误条目不计入已知时长，详见 statistics.json 的 errors。", "",
             f"镜像继承原片 {all_stats['mirror_inherited_files']:,} 条，独立判断 {all_stats['mirror_independent_files']:,} 条。", "",
             *( [f"开头确认 {all_stats['endpoint_head_detections']:,} 条，结尾确认 {all_stats['endpoint_tail_detections']:,} 条，"
                 f"两端均确认 {all_stats['endpoint_files_both_ends']:,} 条；未确认端点 {all_stats['endpoint_unconfirmed_ends']:,} 个。",
                 "这些是规则命中数量，不是人工标注 precision/recall；五条复核样例不能证明全数据集均有首尾校准。", ""] if endpoint_mode else []),
             "## 复用区间 JSON", "",
             "`removal_intervals.json` 的 `motions` 数组包含每条索引文件；"
             "`source` 相对于 `data/hiphi/motions/`。", "",
             "- `remove_intervals_frames`：需要剔除的区间；空列表表示完整保留。",
             "- `keep_intervals_frames`：后续清洗可直接切片的连续保留段。",
             "- `detected_cycles`：检测证据列表；首尾规则含 location、保持段、内侧低臂窗口及 remove_frames；历史规则含前后站立证据。",
             "- `endpoint_review_flags`：旧姿态规则未确认的端点及原因；自适应模式中仅作历史证据，实际决策见 endpoint_trim_decisions。",
             "- `fps`、`frames`、`source_stat`：帧率、总帧数和源文件变更检查信息。",
             "- `is_mirror`、`blacklist_matches`：独立统计分组信息。", "",
             "- `interval_method`、`interval_source`、`evidence_source`：区间是否继承、来源原片及姿态证据来源；继承结果中的 detected_cycles 保留原片证据，实际切片使用 remove/keep_intervals_frames。",
             "- `canonical_source_stat`：继承依赖的原片指纹；原片变化时镜像缓存也失效。", "",
             "```python", "import json", "import numpy as np", "from pathlib import Path", "",
             "index = json.loads(Path('removal_intervals.json').read_text())",
             "for record in index['motions']:", "    if record['status'] != 'ok':", "        continue",
             "    # Each [start, end) is a separate output motion; never concatenate across gaps.",
             "    for start, end in record['keep_intervals_frames']:",
             "        pass  # Read the source NPZ and slice all frame-aligned arrays with [start:end].", "```", "",
             "## 无需再次读取 NPZ 的统计", "", "```bash",
             "/home/hanyi/miniforge3/envs/textop/bin/python \\",
             "  dataset/data_analyze/tpose/scan_hiphi_tpose.py \\",
             f"  --from-json {output / 'removal_intervals.json'}", "```", "",
             "## 扫描记录", "", f"运行信息：`{json.dumps(payload['run'], ensure_ascii=False)}`", ""]
    (output / "REPORT.md").write_text("\n".join(rows), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("all_indexed", "canonical_only", "after_existing_blacklist")},
                     indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/hiphi")
    parser.add_argument("--output", type=Path, help="Defaults to full_scan_endpoints_shared_mirror")
    parser.add_argument("--detector", choices=["endpoint", "cycle"], default="endpoint")
    parser.add_argument("--mirror-policy", choices=["inherit", "independent"], default="inherit")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--executor", choices=["process", "thread"], default="process")
    parser.add_argument("--limit", type=int, default=0,
                        help="Random subset; inherit mode adds required original partners; 0 means all")
    parser.add_argument("--seed", type=int, default=42)
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument("--from-json", type=Path, help="Only recompute statistics; do not change interval policy")
    existing.add_argument("--reuse-json", type=Path, help="Convert an existing scan to mirror inheritance without trajectory reads")
    parser.add_argument("--benchmark-only", action="store_true", help="Do not create final statistics, just cache/timing")
    args = parser.parse_args()
    if args.from_json:
        args.output = args.output or args.from_json.parent
        payload = json.loads(args.from_json.read_text())
        if payload.get("schema_version") != SCHEMA_VERSION:
            parser.error("Unsupported JSON schema version")
        write_outputs(args.output, payload)
        return
    default_output = DEFAULT_ENDPOINT_OUTPUT if args.detector == "endpoint" else DEFAULT_OUTPUT
    if args.mirror_policy == "inherit":
        default_output = default_output.with_name(default_output.name + "_shared_mirror")
    args.output = args.output or default_output
    if args.workers < 1 or args.limit < 0:
        parser.error("workers must be positive and limit must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset_dir.resolve()
    index_path = dataset / "filenames.txt"
    raw_index = index_path.read_bytes()
    relatives = [line.strip() for line in raw_index.decode().splitlines() if line.strip().endswith(".npz")]
    if len(set(relatives)) != len(relatives):
        raise ValueError("Duplicate paths in filenames.txt")
    for relative in relatives:
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"Unsafe relative source path: {relative}")
    indexed = set(relatives)
    if args.limit:
        relatives = random.Random(args.seed).sample(relatives, min(args.limit, len(relatives)))
        if args.mirror_policy == "inherit":
            dependencies = {canonical_source(r) for r in relatives} & indexed
            relatives = list(dict.fromkeys([*relatives, *sorted(dependencies)]))
    blacklist = dataset / "blacklist.txt"
    patterns = [line.strip() for line in blacklist.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")] if blacklist.is_file() else []
    detector_function = detect_endpoints if args.detector == "endpoint" else precision_candidates
    detector_source = Path(detector_function.__code__.co_filename).read_bytes()
    config = dict(schema_version=SCHEMA_VERSION, dataset_dir=str(dataset), detector=args.detector,
                  mirror_policy=args.mirror_policy,
                  rules=ENDPOINT_RULES if args.detector == "endpoint" else PRECISION_RULES,
                  removal_policy=ENDPOINT_REMOVAL_POLICY if args.detector == "endpoint" else REMOVAL_POLICY,
                  blacklist_patterns=patterns,
                  detector_sha256=hashlib.sha256(detector_source).hexdigest(),
                  features_sha256=hashlib.sha256(Path(load_features.__code__.co_filename).read_bytes()).hexdigest(),
                  scanner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  index_sha256=hashlib.sha256(raw_index).hexdigest())
    config_path = args.output / "scan_config.json"
    if config_path.is_file() and json.loads(config_path.read_text()) != config:
        raise ValueError("Checkpoint configuration changed; use a new --output directory")
    write_json(config_path, config, indent=2)
    checkpoint = args.output / "scan_checkpoint.jsonl"
    if args.reuse_json:
        if args.mirror_policy != "inherit" or args.limit:
            parser.error("--reuse-json requires --mirror-policy inherit and a full index")
        start = time.monotonic()
        previous = json.loads(args.reuse_json.read_text())
        if previous.get("schema_version") != SCHEMA_VERSION:
            parser.error("Unsupported JSON schema version")
        for field in ("dataset_dir", "rules", "detector_sha256", "features_sha256",
                      "index_sha256", "blacklist_patterns"):
            if previous["config"].get(field) != config[field]:
                raise ValueError(f"Existing JSON has different {field}; run a fresh scan")
        old_records = {r["source"]: r for r in previous["motions"]}
        if set(old_records) != indexed:
            raise ValueError("Existing JSON must cover every indexed source")
        for relative, record in old_records.items():
            if record["status"] != "ok" or record["source_stat"] != fingerprint(dataset / "motions" / relative):
                raise ValueError(f"Invalid/stale JSON source: {relative}; run a fresh scan")
        converted = []
        for relative in relatives:
            record = deepcopy(old_records[relative])
            partner = canonical_source(relative)
            original = old_records.get(partner)
            if can_inherit_mirror(original, record):
                record = inherit_mirror(original, record)
            else:
                record.update(interval_method="detected", interval_source=relative, evidence_source=relative)
                if partner:
                    record["mirror_reuse_status"] = "timeline_mismatch_cached" if original else "missing_canonical_cached"
            converted.append(record)
        # Cache the converted full result so future scans can resume normally.
        temporary = checkpoint.with_suffix(".jsonl.tmp")
        with temporary.open("w") as stream:
            for record in converted:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(checkpoint)
        run = dict(created_utc=datetime.now(timezone.utc).isoformat(), scanned_files=0,
                   reused_files=len(converted), elapsed_seconds=time.monotonic() - start,
                   mirror_inherited_files=sum(r.get("interval_method") == "canonical_inherited" for r in converted),
                   mirror_fallback_cached_files=sum(bool(r.get("mirror_reuse_status")) for r in converted),
                   source_json=str(args.reuse_json.resolve()), complete_index=True)
        write_json(args.output / "timing.json", run, indent=2)
        write_outputs(args.output, dict(schema_version=SCHEMA_VERSION, config=config, run=run, motions=converted))
        return
    cached = {}
    if checkpoint.is_file():
        # An interrupted final write is ignored; complete records are reused.
        with checkpoint.open() as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record["status"] == "ok":
                    cached[record["source"]] = record
    requested = set(relatives)
    records = {}
    for relative, record in cached.items():
        if relative not in requested:
            continue
        try:
            dependency_valid = (record.get("interval_method") != "canonical_inherited"
                                or record["canonical_source_stat"] == fingerprint(
                                    dataset / "motions" / record["interval_source"]))
            if dependency_valid and record["source_stat"] == fingerprint(dataset / "motions" / relative):
                records[relative] = record
        except OSError:
            pass
    pending = [r for r in relatives if r not in records]
    print(f"Indexed: {len(relatives):,}; cached: {len(records):,}; pending: {len(pending):,}; "
          f"{args.executor} workers: {args.workers}", flush=True)
    start = time.monotonic()
    completed = 0
    geometry_scans = 0
    executor_type = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor
    kwargs = {"mp_context": multiprocessing.get_context("spawn")} if args.executor == "process" else {}
    with checkpoint.open("a") as stream, executor_type(max_workers=args.workers, **kwargs) as executor:
        first = [r for r in pending if args.mirror_policy == "independent" or not canonical_source(r)]
        first_set = set(first)
        second = [r for r in pending if r not in first_set]
        for phase, sources in enumerate((first, second)):
            if phase == 0:
                tasks = ((str(dataset), r, patterns, args.detector) for r in sources)
                worker = scan_one
            else:
                tasks = ((str(dataset), r, patterns, args.detector, records.get(canonical_source(r))) for r in sources)
                worker = scan_mirror
            for record in executor.map(worker, tasks, chunksize=8):
                records[record["source"]] = record
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                completed += 1
                geometry_scans += record.get("interval_method") != "canonical_inherited"
                if completed % 128 == 0 or completed == len(pending):
                    stream.flush()
                    elapsed = time.monotonic() - start
                    rate = completed / max(elapsed, 1e-8)
                    print(f"Processed {len(records):,}/{len(relatives):,} | {rate:.1f} files/s | "
                          f"elapsed {elapsed:.1f}s | ETA {(len(pending) - completed) / rate:.1f}s", flush=True)
    elapsed = time.monotonic() - start
    run = dict(created_utc=datetime.now(timezone.utc).isoformat(), scanned_files=geometry_scans,
               processed_files=completed,
               mirror_inherited_files=sum(r.get("interval_method") == "canonical_inherited" for r in records.values()),
               reused_files=len(relatives) - completed, elapsed_seconds=elapsed,
               files_per_second=completed / max(elapsed, 1e-8), workers=args.workers,
               executor=args.executor, complete_index=not bool(args.limit))
    write_json(args.output / "timing.json", run, indent=2)
    print(f"Timing: {json.dumps(run)}", flush=True)
    if not args.benchmark_only:
        payload = dict(schema_version=SCHEMA_VERSION, config=config, run=run,
                       motions=[records[relative] for relative in relatives])
        write_outputs(args.output, payload)


if __name__ == "__main__":
    main()
