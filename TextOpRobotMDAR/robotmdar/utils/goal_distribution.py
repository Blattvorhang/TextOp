"""Diagnostics for sampled, unclamped training goals (not physical limits)."""

import math

import torch
from robotmdar.utils.goal_time import clamp_goal_time


REFERENCE_HEIGHT_RANGE = (0.70, 0.775)
_QUANTILES = [0.5, 0.75, 0.8, 0.9, 0.95, 0.99, 0.995]


def reference_goal_distribution(height, distance, arrival_time, fps,
                                height_range=REFERENCE_HEIGHT_RANGE,
                                time_min=0.0, time_max=None):
    """Pool raw goals in an active-height interval before taking quantiles.

    This is a height-based proxy, not a locomotion classifier. Static goals,
    recovery and other modes may remain. Selection and counts are persisted
    so that the calibration source is reviewable alongside the runtime curve.
    """
    low, high = (float(v) for v in height_range)
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ValueError("reference height range requires finite low < high")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    h, d, t = [torch.as_tensor(v).detach().cpu().reshape(-1).double()
               for v in (height, distance, arrival_time)]
    if not (h.numel() == d.numel() == t.numel()):
        raise ValueError("height, distance and arrival_time must have equal lengths")
    selected = (torch.isfinite(h) & torch.isfinite(d) & torch.isfinite(t)
                & (d >= 0) & (t > 0) & (h >= low) & (h < high))
    d, t = d[selected], t[selected]
    levels = torch.tensor(_QUANTILES, dtype=torch.float64)
    result = {
        "schema_version": 2,
        "selection": "pooled_active_height_goals",
        "height_range": [low, high],
        "range_convention": "left_closed_right_open",
        "motion_filter": "none; active-height proxy, not locomotion-only",
        "n_samples": d.numel(),
        "n_total": h.numel(),
        "quantile_levels": (levels * 100).tolist(),
        "time_min": time_min,
        "time_max": time_max,
        "speed_definition": "distance / clamp(arrival_time, max(time_min, 1/fps), time_max)",
    }
    for key, values in (("dist_quantiles", d),
                        ("speed_quantiles", d / clamp_goal_time(t, fps, time_min, time_max))):
        result[key] = torch.quantile(values, levels).tolist() if d.numel() else []
    return result


def height_goal_distribution(height, distance, arrival_time, fps, bin_width=0.025,
                             time_min=0.0, time_max=None):
    """Fixed-width bins preserve height occupancy, including empty intervals.

    Each observation is one goal window produced by the dataset sampler. This
    measures its sampled condition distribution, not unique frames or seconds
    of motion. Mixed arrival horizons are retained and reported explicitly.
    Empty bins have null centers/quantiles; no global-quantile fallback is used.
    """
    if (not math.isfinite(bin_width) or bin_width <= 0
            or not math.isfinite(fps) or fps <= 0):
        raise ValueError("bin_width and fps must be positive and finite")
    height, distance, arrival_time = [
        torch.as_tensor(v).detach().cpu().reshape(-1).double()
        for v in (height, distance, arrival_time)
    ]
    if not (height.numel() == distance.numel() == arrival_time.numel()):
        raise ValueError("height, distance and arrival_time must have equal lengths")
    valid = (torch.isfinite(height) & torch.isfinite(distance)
             & torch.isfinite(arrival_time) & (distance >= 0) & (arrival_time > 0))
    total = height.numel()
    height, distance, arrival_time = [v[valid] for v in (height, distance, arrival_time)]
    if not height.numel():
        raise ValueError("No finite goals with nonnegative distance and positive arrival time")
    first = math.floor(float(height.min()) / bin_width)
    last = math.floor(float(height.max()) / bin_width) + 1
    edges = torch.arange(first, last + 1, dtype=torch.float64) * bin_width
    bucket = torch.bucketize(height, edges[1:-1], right=True)
    levels = torch.tensor(_QUANTILES, dtype=torch.float64)
    result = {
        "schema_version": 2,
        "condition": "reference_root_height",
        "sampling": "dataset_sampled_goal_windows; no locomotion/posture filtering",
        "binning": "fixed_width_left_closed_right_open_last_closed",
        "bin_width": bin_width,
        "bin_edges": edges.tolist(),
        "quantile_levels": (levels * 100).tolist(),
        "n_total": total,
        "n_valid": height.numel(),
        "n_excluded": total - height.numel(),
        "n_samples": [],
        "sample_fraction": [],
        "centers": [],
        "dist_quantiles": [],
        "speed_quantiles": [],
        "arrival_time_quantiles": [],
        "time_budget_quantiles": [],
        "time_min": time_min,
        "time_max": time_max,
        "speed_definition": "distance / clamp(arrival_time, max(time_min, 1/fps), time_max)",
    }
    time_budget = clamp_goal_time(arrival_time, fps, time_min, time_max)
    speed = distance / time_budget
    for index in range(len(edges) - 1):
        mask = bucket == index
        count = int(mask.sum())
        result["n_samples"].append(count)
        result["sample_fraction"].append(count / height.numel())
        result["centers"].append(float(height[mask].mean()) if count else None)
        for key, values in (("dist_quantiles", distance), ("speed_quantiles", speed),
                            ("arrival_time_quantiles", arrival_time),
                            ("time_budget_quantiles", time_budget)):
            result[key].append(torch.quantile(values[mask], levels).tolist()
                               if count else [None] * len(levels))
    return result
