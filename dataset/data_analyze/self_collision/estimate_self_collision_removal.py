#!/usr/bin/env python3
"""Estimate full-motion removals from an existing self-collision scan."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "analysis_output/hiphi_self_collision/severe_intervals.json"


def estimate(scan: dict, peak_depth_m: float, min_duration_seconds: float) -> dict:
    config = scan["config"]
    if config["stride"] != 1:
        raise ValueError("Removal estimates require a stride-1, every-frame scan")
    if peak_depth_m < config["severe_depth_m"]:
        raise ValueError("Peak threshold cannot be below the scan's per-frame depth threshold")

    candidates = []
    for motion in scan["motions"]:
        fps = motion["fps"]
        intervals = [
            interval for interval in motion["severe_intervals"]
            if interval["max_penetration_m"] >= peak_depth_m
            and (interval["end_frame"] - interval["start_frame"]) / fps >= min_duration_seconds
        ]
        if not intervals:
            continue
        intervals.sort(key=lambda item: (-item["max_penetration_m"], item["start_frame"]))
        candidates.append({
            "source": motion["source"],
            "source_stat": motion["source_stat"],
            "collision_method": motion.get("collision_method", "independent"),
            "evidence_source": motion.get("evidence_source", motion["source"]),
            "is_mirror": "__mirror" in Path(motion["source"]).parent.name,
            "fps": fps,
            "frames": motion["frames"],
            "kept_frames": motion.get("kept_frames", motion["frames"]),
            "keep_intervals_frames": motion.get("keep_intervals_frames", [[0, motion["frames"]]]),
            "duration_seconds": motion.get("kept_frames", motion["frames"]) / fps,
            "qualifying_interval_frames": sum(
                item["end_frame"] - item["start_frame"] for item in intervals
            ),
            "peak_penetration_m": intervals[0]["max_penetration_m"],
            "qualifying_intervals": intervals,
        })
    candidates.sort(key=lambda item: (
        -item["peak_penetration_m"], -item["qualifying_interval_frames"], item["source"]
    ))

    total = scan["summary"]
    removed_frames = sum(item["kept_frames"] for item in candidates)
    removed_seconds = sum(item["duration_seconds"] for item in candidates)
    return {
        "schema_version": 1,
        "selection": {
            "per_frame_penetration_m": config["severe_depth_m"],
            "scan_mirror_policy": config.get("mirror_policy", "independent"),
            "tpose_json_sha256": config.get("tpose_json_sha256"),
            "minimum_interval_peak_penetration_m": peak_depth_m,
            "minimum_interval_duration_seconds": min_duration_seconds,
            "removal_unit": "entire motion if any interval qualifies",
            "actual_source_data_removed": False,
        },
        "summary": {
            "total_motions": total["valid_files"],
            "candidate_motions": len(candidates),
            "candidate_motion_fraction": len(candidates) / total["valid_files"],
            "remaining_motions": total["valid_files"] - len(candidates),
            "candidate_original_motions": sum(not item["is_mirror"] for item in candidates),
            "candidate_mirror_motions": sum(item["is_mirror"] for item in candidates),
            "candidate_motions_with_inherited_collision": sum(
                item["collision_method"] == "mirror_inherited" for item in candidates
            ),
            "qualifying_intervals": sum(len(item["qualifying_intervals"]) for item in candidates),
            "qualifying_interval_frames": sum(item["qualifying_interval_frames"] for item in candidates),
            "total_frames": total["frames"],
            "tpose_removed_frames": total.get("removed_tpose_frames", 0),
            "post_tpose_frames": total.get("kept_frames", total["frames"]),
            "candidate_full_motion_frames": removed_frames,
            "candidate_frame_fraction": removed_frames / total.get("kept_frames", total["frames"]),
            "remaining_frames": total.get("kept_frames", total["frames"]) - removed_frames,
            "total_duration_hours": total["duration_seconds"] / 3600,
            "post_tpose_duration_hours": total.get("kept_seconds", total["duration_seconds"]) / 3600,
            "candidate_full_motion_hours": removed_seconds / 3600,
            "candidate_duration_fraction": removed_seconds / total.get("kept_seconds", total["duration_seconds"]),
            "remaining_duration_hours": (total.get("kept_seconds", total["duration_seconds"]) - removed_seconds) / 3600,
            "candidate_source_bytes": sum(item["source_stat"]["size_bytes"] for item in candidates),
        },
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--peak-depth-m", type=float, default=0.05)
    parser.add_argument("--min-duration-s", type=float, default=0.5)
    args = parser.parse_args()
    if not (math.isfinite(args.peak_depth_m) and args.peak_depth_m > 0
            and math.isfinite(args.min_duration_s) and args.min_duration_s > 0):
        parser.error("Thresholds must be finite and positive")
    with args.input.open(encoding="utf-8") as stream:
        scan = json.load(stream)
    result = estimate(scan, args.peak_depth_m, args.min_duration_s)
    result["source_scan"] = str(args.input.resolve())
    output = args.output or args.input.with_name("removal_estimate.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(output)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Candidate list: {output}")


if __name__ == "__main__":
    main()
