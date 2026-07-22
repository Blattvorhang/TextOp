#!/usr/bin/env python3
"""
Calculate per-category sampling weights for data.weighted_sample.

Reads train.pkl (TextOp SkeletonPrimitiveDataset format), accumulates
total duration per coarse action category from frame_ann, and computes
weights that produce a controlled training distribution.

Design — target_mass (NOT uniform):
    Unlike BABEL/TextOp text-conditioned training where every prompt must
    work equally, our goal+scene-conditioned planner uses conditions to
    select behavior. Walk is the default — when the goal is far away the
    model should walk. Crouch/step_over are triggered by specific occupancy
    patterns. Gesture/dance are irrelevant noise.

    Therefore we do NOT weight toward uniformity. Instead, each category is
    assigned a target_mass that directly encodes its desired training share.
    Core locomotion dominates; irrelevant categories are suppressed.

Formula:
    weight[cat] = target_mass[cat] / total_duration[cat]

The data loader (data.py:_cal_sample_weight) then computes:
    seq_weight = Σ(seg_duration × weight[seg_category])
    P(sequence) ∝ seq_weight

With single-category-per-sequence frame_ann, each category contributes
~target_mass[cat] units of unnormalized probability mass:
    P(cat) = target_mass[cat] / Σ(target_mass)     ← directly controlled

Training distribution (default TARGET_MASS):

    Stage 1 — Core locomotion (dominates, ~43%):
        walk 3.0, jog 3.0, turn 2.0, idle 2.0

    Stage 2 — Body-level obstacle avoidance (~15%):
        crouch 2.0, kneel 1.5

    Stage 3 — Limb-level obstacle negotiation (~9%):
        step_over 2.0

    Stage 4 — Scene interaction (~7%):
        sit 1.5

    De-emphasized (~11%):
        jump 0.5, climb 0.5, push 0.5, carry 0.5, reach 0.5, fall 0.3

    Suppressed (~15%):
        gesture 1.0, dance 0.5, injured 0.5, sport 0.3, crutch 0.2, other 0.5

Usage:
    cd dataset/data_process

    python cal_weighted_statistics.py \
        --data_folder ../g1_textop \
        --trg_filename ../RobotMDAR-statistics/action_statistics.json

    # With custom target_mass config (YAML/JSON):
    python cal_weighted_statistics.py \
        --data_folder ../g1_textop \
        --trg_filename ../action_statistics.json \
        --mass_config ./target_mass.yaml
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Default TARGET_MASS
# ---------------------------------------------------------------------------
# Each value is the unnormalized probability mass for that category.
# P(cat) = mass[cat] / Σ(mass).
#
# Design rationale (different from BABEL text-conditioned weighted_sample):
#   - Walk/jog are the DEFAULT behavior — when goal is far, walk there.
#     They must dominate training because they dominate inference.
#   - Crouch/kneel/step_over are triggered by occupancy patterns — boost
#     them above their raw data share so the VAE learns them well.
#   - Gesture/dance are irrelevant noise — suppress them so they don't
#     waste VAE capacity.
#   - Jump/climb/push are hard to track or occupancy-invisible — low mass.

TARGET_MASS = {
    # ---- Stage 1: Core locomotion (dominant — ~43% combined) ----
    "walk":      3.0,
    "jog":       3.0,
    "turn":      2.0,
    "idle":      2.0,

    # ---- Stage 2: Body-level obstacle avoidance (~15% combined) ----
    "crouch":    2.0,
    "kneel":     1.5,

    # ---- Stage 3: Limb-level obstacle negotiation (~9%) ----
    "step_over": 2.0,

    # ---- Stage 4: Scene interaction (~7%) ----
    "sit":       1.5,

    # ---- De-emphasized: hard to track or occupancy-invisible (~11%) ----
    "jump":      0.5,
    "climb":     0.5,
    "push":      0.5,
    "carry":     0.5,
    "reach":     0.5,
    "fall":      0.3,

    # ---- Suppressed: irrelevant to navigation (~15%) ----
    "gesture":   1.0,
    "dance":     0.5,
    "injured":   0.5,
    "sport":     0.3,
    "crutch":    0.2,
    "other":     0.5,
}


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def compute_action_statistics(
    data_folder: str,
    json_filename: str,
    target_mass: dict | None = None,
    max_weight_ratio: float = 10.0,
):
    """Compute per-category weights from frame_ann in train.pkl.

    Parameters
    ----------
    data_folder : str
        Directory containing train.pkl (TextOp format).
    json_filename : str
        Output path for action_statistics.json.
    target_mass : dict | None
        Category → unnormalized probability mass.
        P(cat) ≈ mass[cat] / Σ(mass).
        If None, uses built-in TARGET_MASS.
    max_weight_ratio : float
        Cap: weight ≤ max_weight_ratio × median_weight across all categories.
        Set to 0 or negative to disable capping.
    """
    if target_mass is None:
        target_mass = TARGET_MASS

    # ── 1. Load data & accumulate duration per category ──
    dataset_path = Path(data_folder) / "train.pkl"
    if not dataset_path.exists():
        print(f"ERROR: {dataset_path} not found")
        sys.exit(1)

    data = joblib.load(dataset_path)
    print(f"Loaded {len(data):,} sequences from {dataset_path}")

    cat_duration: dict[str, float] = defaultdict(float)
    cat_count: dict[str, int] = defaultdict(int)
    missing_frame_ann = 0

    for item in data:
        frame_ann = item.get("frame_ann", [])
        if not frame_ann:
            missing_frame_ann += 1
            continue

        for (_start_t, _end_t, _text, act_cat_list) in frame_ann:
            duration = _end_t - _start_t
            for cat in act_cat_list:
                cat_duration[cat] += duration
                cat_count[cat] += 1

    if missing_frame_ann:
        print(f"WARNING: {missing_frame_ann} sequences have empty frame_ann — skipped")

    if not cat_duration:
        print("ERROR: no valid frame_ann found. Did you wire classify_coarse() "
              "into pack_motion_lib_to_textop.py?")
        sys.exit(1)

    total_dur = sum(cat_duration.values())
    print(f"Total motion duration: {total_dur / 3600:.2f} hours "
          f"across {len(cat_duration)} categories\n")

    # ── 2. Assign default mass for categories not in TARGET_MASS ──
    # New categories (not explicitly listed) get mass = 0.5 (treated as "other")
    for cat in cat_duration:
        if cat not in target_mass:
            target_mass[cat] = 0.5

    total_mass = sum(target_mass[c] for c in cat_duration)

    # ── 3. Compute weight = target_mass / total_duration ──
    action_stats: dict[str, dict] = {}
    for cat, dur in cat_duration.items():
        mass = target_mass.get(cat, 0.5)
        w = mass / dur if dur > 0 else 0.0
        action_stats[cat] = {
            "total_len": round(dur, 3),
            "target_mass": mass,
            "target_pct": round(mass / total_mass * 100, 1),
            "weight": round(w, 10),
        }

    # ── 4. Soft-cap extreme weights ──
    if max_weight_ratio > 0:
        weights = np.array([s["weight"] for s in action_stats.values()])
        median_w = float(np.median(weights))
        cap = median_w * max_weight_ratio
        capped_cats = []
        for cat in action_stats:
            if action_stats[cat]["weight"] > cap:
                action_stats[cat]["weight"] = round(cap, 10)
                action_stats[cat]["capped"] = True
                capped_cats.append(cat)
        if capped_cats:
            print(f"Weight cap: {max_weight_ratio}× median = {cap:.6f}")
            for cat in capped_cats:
                orig_w = action_stats[cat]["target_mass"] / action_stats[cat]["total_len"]
                print(f"  Capped: {cat}  "
                      f"({orig_w:.6f} → {cap:.6f})")
            print()

    # ── 5. Report ──
    _print_report(action_stats, cat_duration, target_mass, total_mass)

    # ── 6. Save (data loader needs 'weight' and 'total_len') ──
    export = {}
    for cat, stats in action_stats.items():
        export[cat] = {
            "total_len": stats["total_len"],
            "weight": stats["weight"],
        }

    export_path = Path(json_filename)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, "w") as f:
        json.dump(export, f, indent=4)

    print(f"\nSaved: {export_path}")
    print(f"Ready for training:")
    print(f"  data.weighted_sample=true")
    print(f"  data.action_statistics_path={export_path.resolve()}")
    return export


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _print_report(
    action_stats: dict[str, dict],
    cat_duration: dict[str, float],
    target_mass: dict[str, float],
    total_mass: float,
):
    """Print a human-readable table: data share vs training share."""
    total_hours = sum(cat_duration.values()) / 3600

    # Sort: by target_mass descending
    sorted_cats = sorted(action_stats.items(),
                         key=lambda x: -x[1]["target_mass"])

    header = (f"{'Category':<14s} {'Hours':>7s}  {'%Data':>6s}  "
              f"{'Mass':>5s}  {'%Train':>7s}  {'Weight':>10s}")
    sep = "-" * len(header)

    print("Training distribution report:")
    print(sep)
    print(header)
    print(sep)

    for cat, stats in sorted_cats:
        hours = cat_duration[cat] / 3600
        pct_data = cat_duration[cat] / (total_hours * 3600) * 100
        mass = stats["target_mass"]
        pct_train = stats["target_pct"]
        capped = " *" if stats.get("capped") else ""

        # Ratio: how much we boost/suppress
        if pct_data > 0.001:
            ratio = pct_train / pct_data
            note = ""
            if ratio > 5.0:
                note = f"  ↑{ratio:.0f}×"
            elif ratio < 0.3:
                note = f"  ↓{1/ratio:.0f}×"
            else:
                note = ""
        else:
            note = ""

        print(f"  {cat:<12s} {hours:>7.2f}  {pct_data:>5.2f}%  "
              f"{mass:>4.1f}  {pct_train:>6.2f}%  "
              f"{stats['weight']:>10.6f}{capped}{note}")

    print(sep)
    print(f"  Total data: {total_hours:.2f} hours across {len(cat_duration)} categories")
    print(f"  Total mass: {total_mass:.1f}")
    print(f"  Cap applied: "
          f"{sum(1 for _, s in sorted_cats if s.get('capped'))} categories")
    print()
    print("  %Data  = raw share of motion duration in train.pkl")
    print("  Mass   = target_mass (directly controls training share)")
    print("  %Train = target share in training batches (≈ mass / Σmass)")
    print("  Weight = mass / total_duration  (consumed by data loader)")
    print("  ↑/↓    = how much we boost or suppress vs. raw data share")
    print("  *      = weight was soft-capped")
    print()
    print("  Unlike BABEL text-conditioned training, we do NOT aim for uniform.")
    print("  Walk/jog dominate because they are the default inference behavior.")
    print("  Crouch/kneel/step_over are boosted above raw share for obstacle learning.")
    print("  Gesture/dance are suppressed — they waste VAE capacity for navigation.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute per-category sampling weights for weighted_sample"
    )
    parser.add_argument(
        "--data_folder", type=str, required=True,
        help="Directory containing train.pkl (output of pack_motion_lib_to_textop.py)",
    )
    parser.add_argument(
        "--trg_filename", type=str, required=True,
        help="Output path for action_statistics.json",
    )
    parser.add_argument(
        "--mass_config", type=str, default=None,
        help="Optional YAML/JSON file with category → target_mass map. "
             "Overrides the built-in TARGET_MASS.",
    )
    parser.add_argument(
        "--max_weight_ratio", type=float, default=10.0,
        help="Cap weight at max_weight_ratio × median_weight (default: 10.0). "
             "Set to 0 to disable. Prevents ultra-rare categories from dominating.",
    )

    args = parser.parse_args()

    target_mass = TARGET_MASS
    if args.mass_config:
        config_path = Path(args.mass_config)
        if config_path.suffix in (".yaml", ".yml"):
            import yaml
            with open(config_path) as f:
                target_mass = yaml.safe_load(f)
        else:
            with open(config_path) as f:
                target_mass = json.load(f)
        print(f"Loaded target_mass config from {config_path}")

    compute_action_statistics(
        data_folder=args.data_folder,
        json_filename=args.trg_filename,
        target_mass=target_mass,
        max_weight_ratio=args.max_weight_ratio,
    )


if __name__ == "__main__":
    main()
