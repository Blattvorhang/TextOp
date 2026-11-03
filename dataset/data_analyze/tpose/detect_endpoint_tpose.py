"""Detect HiPhi calibration poses only at the two ends of a recording.

Endpoint priors come from manual review of full recordings. Do not infer that
every file contains calibration, and do not delete matching interior actions.
The original two-sided standing detector remains available for comparison.
"""

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d

if __package__:
    from .analyze_hiphi_tpose import load_features, runs
else:
    from analyze_hiphi_tpose import load_features, runs


ENDPOINT_RULES = {
    "edge_search_seconds": 10., "edge_search_fraction": .15,
    "max_trim_seconds_per_edge": 12., "max_trim_fraction_per_edge": .20,
    "core_roll_min_deg": 60.,
    "hold_roll_range_deg": [65., 120.], "hold_roll_asymmetry_max_deg": 20.,
    "hold_upper_arm_lateral_cosine": .90, "hold_elbow_deflection_max_deg": 25.,
    "hold_tilt_max_deg": 25., "hold_root_xy_speed_max_m_s": .12,
    "hold_foot_speed_max_m_s": .15, "hold_foot_height_max_m": .12,
    "hold_foot_height_difference_max_m": .06,
    "hold_pelvis_height_range_m": [.60, .95],
    "hold_knee_abs_max_deg": 25., "hold_hip_pitch_abs_max_deg": 35.,
    "variation_window_seconds": .4, "hold_roll_window_range_max_deg": 10.,
    "hold_min_seconds": .6,
    "inner_boundary_search_seconds": 2., "inner_low_arm_min_seconds": .25,
    "inner_low_arm_roll_max_deg": 45., "inner_upper_arm_down_cosine": .55,
    "inner_tilt_max_deg": 35.,
}

REMOVAL_POLICY = "endpoint-only; head [0, low_arm_witness_end), tail [low_arm_witness_start, frames); interior preserved"


def detect_endpoints(path, *, features=None):
    f = load_features(path) if features is None else features
    cfg = ENDPOINT_RULES
    q, p, fps = f["q"], f["pos"], f["fps"]
    n = len(q)
    roll = q[:, [15, 16]] * [1., -1.]
    window = max(3, round(cfg["variation_window_seconds"] * fps) | 1)
    variation = maximum_filter1d(roll, window, axis=0, mode="nearest") - minimum_filter1d(
        roll, window, axis=0, mode="nearest")
    height_min, height_max = cfg["hold_pelvis_height_range_m"]
    upright_legs = ((p[:, 0, 2] >= height_min) & (p[:, 0, 2] <= height_max)
                    & (np.abs(q[:, [9, 10]]) <= cfg["hold_knee_abs_max_deg"]).all(axis=1)
                    & (np.abs(q[:, [0, 1]]) <= cfg["hold_hip_pitch_abs_max_deg"]).all(axis=1))
    planted = ((f["foot_speed"] <= cfg["hold_foot_speed_max_m_s"]).all(axis=1)
               & (np.abs(p[:, [18, 19], 2]) <= cfg["hold_foot_height_max_m"]).all(axis=1)
               & (np.abs(p[:, 18, 2] - p[:, 19, 2]) <= cfg["hold_foot_height_difference_max_m"]))
    lo_roll, hi_roll = cfg["hold_roll_range_deg"]
    holding = (upright_legs & planted
               & ((roll >= lo_roll) & (roll <= hi_roll)).all(axis=1)
               & (np.abs(roll[:, 0] - roll[:, 1]) <= cfg["hold_roll_asymmetry_max_deg"])
               & (f["upper_arm"][:, 0, 1] >= cfg["hold_upper_arm_lateral_cosine"])
               & (f["upper_arm"][:, 1, 1] <= -cfg["hold_upper_arm_lateral_cosine"])
               & (f["elbow_deflection"] <= cfg["hold_elbow_deflection_max_deg"]).all(axis=1)
               & (f["tilt"] <= cfg["hold_tilt_max_deg"])
               & (f["root_speed"] <= cfg["hold_root_xy_speed_max_m_s"])
               & (variation <= cfg["hold_roll_window_range_max_deg"]).all(axis=1))
    # The witness is on the retained-action side only. A recording may begin
    # or end within calibration, so the outer standing witness is unnecessary.
    low_arms = ((roll <= cfg["inner_low_arm_roll_max_deg"]).all(axis=1)
                & (f["upper_arm"][:, :, 2] <= -cfg["inner_upper_arm_down_cosine"]).all(axis=1)
                & (p[:, 0, 2] >= height_min) & (p[:, 0, 2] <= height_max)
                & (f["tilt"] <= cfg["inner_tilt_max_deg"]))
    minimum_low = int(np.ceil(cfg["inner_low_arm_min_seconds"] * fps))
    low_runs = [(a, b) for a, b in runs(low_arms) if b - a >= minimum_low]
    core_runs = runs((roll >= cfg["core_roll_min_deg"]).all(axis=1))
    search = min(round(cfg["edge_search_seconds"] * fps), int(n * cfg["edge_search_fraction"]))
    budget = min(round(cfg["max_trim_seconds_per_edge"] * fps), int(n * cfg["max_trim_fraction_per_edge"]))
    inner_search = round(cfg["inner_boundary_search_seconds"] * fps)
    minimum_hold = int(np.ceil(cfg["hold_min_seconds"] * fps))
    detections, flags = [], []
    for location in ("head", "tail"):
        eligible = [(a, b) for a, b in core_runs if (a <= search if location == "head" else b >= n - search)]
        if location == "tail":
            eligible.reverse()
        failures = []
        for core_start, core_end in eligible:
            intervals = [(core_start + a, core_start + b)
                         for a, b in runs(holding[core_start:core_end]) if b - a >= minimum_hold]
            if not intervals:
                failures.append("no_continuous_geometry_hold")
                continue
            start, end = max(intervals, key=lambda pair: pair[1] - pair[0])
            if location == "head":
                witnesses = [(a, b) for a, b in low_runs if core_end <= a <= core_end + inner_search]
                if not witnesses:
                    failures.append("no_inner_low_arm_witness")
                    continue
                witness_start, _ = min(witnesses)
                witness_end = witness_start + minimum_low
                remove = [0, witness_end]
            else:
                witnesses = [(a, b) for a, b in low_runs if core_start - inner_search <= b <= core_start]
                if not witnesses:
                    failures.append("no_inner_low_arm_witness")
                    continue
                _, witness_end = max(witnesses, key=lambda pair: pair[1])
                witness_start = witness_end - minimum_low
                remove = [witness_start, n]
            if remove[1] - remove[0] > budget:
                failures.append("exceeds_per_edge_trim_budget")
                continue
            detections.append(dict(
                location=location, fps=fps, source_frames=n,
                export_frames=[start, end], export_seconds=[start / fps, end / fps],
                duration_seconds=(end - start) / fps,
                core_frames=[core_start, core_end], core_seconds=[core_start / fps, core_end / fps],
                inner_low_arm_frames=[witness_start, witness_end],
                remove_frames=remove, remove_seconds=[v / fps for v in remove],
                median_outward_roll_deg=np.median(roll[start:end], axis=0).tolist(),
                minimum_upper_arm_lateral_cosine=float(np.min(
                    f["upper_arm"][start:end, :, 1] * [1., -1.])),
                max_geometric_elbow_deflection_deg=float(f["elbow_deflection"][start:end].max()),
                max_root_xy_speed_m_s=float(f["root_speed"][start:end].max()),
            ))
            break  # Earliest head / latest tail only, not further arm cycles.
        else:
            flags.append(dict(location=location, reason="no_endpoint_core" if not eligible else "not_confirmed",
                              candidate_count=len(eligible), rejection_reasons=sorted(set(failures))))
    return dict(candidates=detections, review_flags=flags)
