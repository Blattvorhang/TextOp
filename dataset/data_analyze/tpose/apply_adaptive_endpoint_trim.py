#!/usr/bin/env python3
"""Build HiPhi cleaning intervals from cached endpoint measurements; never modify NPZs."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

if __package__:
    from .scan_hiphi_tpose import canonical_source, inherit_mirror, interval_fields, write_json, write_outputs
else:
    from scan_hiphi_tpose import canonical_source, inherit_mirror, interval_fields, write_json, write_outputs

BASE = Path(__file__).resolve().parents[1] / "analysis_output/hiphi_tpose"


def trim_original(record, measurement, margin, fallback):
    if (record["frames"], record["fps"]) != (measurement["frames"], measurement["fps"]):
        raise ValueError(f"Measurement timeline differs: {record['source']}")
    result = deepcopy(record)
    frames, fps = record["frames"], record["fps"]
    decisions = {}
    for edge in ("head", "tail"):
        evidence = measurement["measurements"]["45"][edge]
        measured = evidence["status"] == "measured"
        seconds = evidence["boundary_seconds"] + margin if measured else fallback
        trim = min(frames, max(1, round(seconds * fps)))
        decisions[edge] = dict(method="adaptive" if measured else "fallback",
                               measurement=deepcopy(evidence), proposed_trim_frames=trim,
                               proposed_trim_seconds=trim / fps)
    head = decisions["head"]["proposed_trim_frames"]
    tail = decisions["tail"]["proposed_trim_frames"]
    intervals = [[0, head], [frames - tail, frames], *record["remove_intervals_frames"]]
    result.update(interval_fields(frames, fps, intervals))
    result.update(detector="endpoint_adaptive", interval_method="endpoint_adaptive",
                  interval_source=record["source"], evidence_source=record["source"],
                  baseline_remove_intervals_frames=deepcopy(record["remove_intervals_frames"]),
                  endpoint_trim_decisions=decisions)
    return result


def write_review(output, records):
    manifest = BASE / "review_unconfirmed_10/manifest.json"
    review = output / "review_10"
    review.mkdir(exist_ok=True)
    motions = []
    rows = ["# 十条完整原片：正式自适应清洗区间", "",
            "复用原完整 NPZ；标记来自当前正式 JSON，NPZ 未实际裁剪。结尾秒数为原片绝对时间。", "",
            "| 编号 | 动作 | 删除区间 / 秒 | 首 / 尾方法 |", "| --- | --- | --- | --- |"]
    for sample in json.loads(manifest.read_text())["samples"]:
        record = records[sample["record"]["source"]]
        intervals = record["remove_intervals_frames"]
        label = f"{sample['sample_id']:02d} {Path(record['source']).parent.name}"
        methods = " / ".join(record["endpoint_trim_decisions"][e]["method"] for e in ("head", "tail"))
        motions.append(dict(path=os.path.relpath(manifest.parent / sample["npz"], review), label=label,
                            source=record["source"], notes=f"正式自适应清洗：{methods}；完整原片未裁剪。",
                            annotations=[dict(label="建议删除", frames=[a, b]) for a, b in intervals]))
        times = "; ".join(f"{a / record['fps']:.2f}–{b / record['fps']:.2f}" for a, b in intervals)
        rows.append(f"| {sample['sample_id']:02d} | {label} | {times} | {methods} |")
    write_json(review / "motion_files.json", dict(version=1, motions=motions), indent=2)
    (review / "README.md").write_text("\n".join(rows) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASE / "full_scan_endpoints_shared_mirror/removal_intervals.json")
    parser.add_argument("--measurements", type=Path, default=BASE / "endpoint_duration_analysis/measurements.json")
    parser.add_argument("--output", type=Path, default=BASE / "full_scan_endpoints_adaptive")
    parser.add_argument("--margin", type=float, default=.3)
    parser.add_argument("--fallback", type=float, default=7.)
    args = parser.parse_args()
    start = time.monotonic()
    baseline_bytes, measurement_bytes = args.baseline.read_bytes(), args.measurements.read_bytes()
    previous = json.loads(baseline_bytes)
    measurements = {r["source"]: r for r in json.loads(measurement_bytes)}
    originals = {r["source"]: trim_original(r, measurements[r["source"]], args.margin, args.fallback)
                 for r in previous["motions"] if not r["is_mirror"]}
    records = []
    for record in previous["motions"]:
        if record["is_mirror"]:
            result = inherit_mirror(originals[canonical_source(record["source"])], record)
            result["baseline_remove_intervals_frames"] = record["remove_intervals_frames"]
        else:
            result = originals[record["source"]]
        records.append(result)
    decisions = [r["endpoint_trim_decisions"][e] for r in originals.values() for e in ("head", "tail")]
    config = dict(previous["config"], detector="endpoint_adaptive",
                  rules=dict(shoulder_high_degrees=60, shoulder_low_degrees=45,
                             high_hold_seconds=.2, low_hold_seconds=.2, transition_search_seconds=3.,
                             core_start_search_seconds=10., core_start_search_fraction=.15,
                             measurement_budget_seconds=12., measurement_budget_fraction=.2,
                             adaptive_margin_seconds=args.margin, fallback_seconds=args.fallback,
                             preserve_longer_baseline_trims=True, short_motion_policy="drop_if_no_frames_remaining"),
                  removal_policy="measured endpoint transition + margin; fallback on every unmeasurable edge; preserve longer baseline deletions",
                  baseline_json=str(args.baseline.resolve()), measurements_json=str(args.measurements.resolve()),
                  baseline_sha256=hashlib.sha256(baseline_bytes).hexdigest(),
                  measurements_sha256=hashlib.sha256(measurement_bytes).hexdigest(),
                  trimmer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    run = dict(created_utc=datetime.now(timezone.utc).isoformat(), scanned_files=0,
               reused_files=len(records), complete_index=previous["run"].get("complete_index", True),
               adaptive_original_edges=sum(d["method"] == "adaptive" for d in decisions),
               fallback_original_edges=sum(d["method"] == "fallback" for d in decisions),
               elapsed_seconds=time.monotonic() - start)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "scan_config.json", config, indent=2)
    write_json(args.output / "timing.json", run, indent=2)
    write_outputs(args.output, dict(schema_version=previous["schema_version"], config=config, run=run, motions=records))
    write_review(args.output, {r["source"]: r for r in records})


if __name__ == "__main__":
    main()
