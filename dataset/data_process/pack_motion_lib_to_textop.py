#!/usr/bin/env python3
"""Pack motion_lib PKL(s) into TextOp RobotMDAR SkeletonPrimitiveDataset format.

This is the second stage of the BONES-SEED → TextOp pipeline:

  Stage 1: convert_soma_csv_to_motion_lib.py
      BONES-SEED CSV (120Hz, 29-DOF, cm+deg)
      → motion_lib PKL (50Hz, 29-DOF, m+rad, per-name dict)

  Stage 2: pack_motion_lib_to_textop.py  ← this script
      motion_lib PKL
      → TextOp train.pkl / val.pkl (23-DOF, list-of-dicts)

Usage:
    # Single motion_lib PKL
    python pack_motion_lib_to_textop.py \
        --input bones_seed_all.pkl \
        --output ./g1_textop

    # Multiple motion_lib PKLs (merged)
    python pack_motion_lib_to_textop.py \
        --input pkl1.pkl pkl2.pkl pkl3.pkl \
        --output ./g1_textop

    # With custom train/val ratio
    python pack_motion_lib_to_textop.py \
        --input bones_seed_all.pkl \
        --output ./g1_textop --val_ratio 0.05
"""

import argparse
import random
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 29-DOF → 23-DOF crop mask (same as convert_soma_csv_to_motion_lib.py)
# ---------------------------------------------------------------------------
DOF_29_TO_23 = np.array([
    True,  True,  True,  True,  True,  True,    # left leg  0-5
    True,  True,  True,  True,  True,  True,    # right leg 6-11
    True,  True,  True,                          # waist     12-14
    True,  True,  True,  True,                   # left arm  15-18
    False, False, False,                         # left wrist 19-21
    True,  True,  True,  True,                   # right arm 22-25
    False, False, False,                         # right wrist 26-28
])
TARGET_DOF = 23
assert DOF_29_TO_23.sum() == TARGET_DOF


# ---------------------------------------------------------------------------
# Coarse action classification (mirrors analyze_action_distribution.py)
# ---------------------------------------------------------------------------
def _extract_action_name(filename: str) -> str:
    """Extract fine-grained action name from a BONES-SEED filename stem.

    Handles both raw CSV names and prefixed motion_lib dict keys:
      walk_ff_stop_180_R_003__A047           →  walk_ff_stop_180_R
      210531__jump_and_land_heavy_001__A001  →  jump_and_land_heavy
      idle_one_foot_left__003__A023          →  idle_one_foot_left
      brush_of_dust__A029                    →  brush_of_dust
    """
    stem = filename.replace(".csv", "")

    # 1. Strip trailing camera ID (last "__A" + digits)
    stem = re.sub(r'__A\d+$', '', stem)

    # 2. Strip leading date prefix (YYMMDD__) added by load_motion_lib_dicts
    stem = re.sub(r'^\d{6}__', '', stem)

    # 3. Strip trailing sequence number: action_001, action_001o, action_007_ns
    m = re.match(r'^(.+)_(\d+(?:[a-z_]\w*)?)$', stem)
    if m:
        return m.group(1)

    # 4. Double-underscore pattern: action__003
    m = re.match(r'^(.+?)__(\d+)$', stem)
    if m:
        return m.group(1)

    # 5. Sequence number directly appended (no underscore): nameR001
    m = re.match(r'^(.+[a-zA-Z])(\d+[a-z_]*\w*)$', stem)
    if m:
        return m.group(1)

    return stem


# Keyword rules in priority order — first match wins.
# See dataset/data_analyze/analyze_action_distribution.py for the full taxonomy.
_COARSE_RULES = [
    ("injured",       ["injured"]),
    ("crutch",        ["crutch", "crutches"]),
    ("jump",          ["jump", "hop", "leap", "flip", "vault_over",
                       "jump_and_land", "jump_over", "jump_twice",
                       "high_jump", "jump_ff", "jump_sideway",
                       "jump_and_down", "turn_jump", "fire_in_the_hole"]),
    ("jog",           ["jog", "jogging", "run_"]),
    ("walk",          ["walk", "moonwalk", "step_forward", "step_backward"]),
    ("dance",         ["dance", "dancing", "choreography", "macarena",
                       "dancecard", "expressionism", "krakowiak"]),
    ("climb",         ["climb", "ladder", "come_up_", "come_down_",
                       "crouch_cupboard"]),
    ("fall",          ["fall", "faint", "toxic_gas", "postmortem",
                       "death", "lying", "lie_", "flying_",
                       "stand_up_lying", "on_ground"]),
    ("crouch",        ["crouch", "crawl", "on_all_fours",
                       "crouch_idle", "crouch_walk", "stoop"]),
    ("kneel",         ["kneel", "sit_on_heels"]),
    ("sit",           ["sitting", "sit_cross", "sit_",
                       "read_newspaper_sitting", "eat_hotdog_sitting",
                       "play_guitar_sitting", "having_a_sit"]),
    ("carry",         ["carry", "lift", "crate", "heavy_", "light_",
                       "pick_up", "put_down", "hold_",
                       "moving_object", "pass_",
                       "item_give", "item_take", "item_pick", "item_put",
                       "item_switch", "item_hold",
                       "lasso_catch", "lasso_dance", "lasso_pull",
                       "watering_plants", "walk_the_dog",
                       "medium_big", "small_heavy", "small_light",
                       "big_heavy", "big_light",
                       "medium_heavy", "medium_light"]),
    ("reach",         ["reach", "reaching"]),
    ("push",          ["push", "pull", "crank", "valve", "handle", "lever",
                       "_knob_", "door_", "shut", "slam",
                       "open_walk", "close_",
                       "horizontal_lever", "vertical_lever",
                       "neutral_button", "operating"]),
    ("step_over",     ["step_over", "step_in", "avoid_obstacle",
                       "bump_into", "jump_over_obstacle", "neutral_avoid"]),
    ("turn",          ["turn_handstand", "mohak", "step_rotate",
                       "idle_turn", "spin_"]),
    ("idle",          ["idle", "stand", "standing", "legs_relax",
                       "looking_around", "looking_in_the_mirror",
                       "look_around", "looking_R", "looking_",
                       "neutral_sit", "neutral_stand",
                       "neutral_idle", "neutral_laugh", "neutral_fear",
                       "neutral_cry", "neutral_looking",
                       "neutral_dancecard_idle", "neutral_dancecard_looking",
                       "idle_hands", "idle_one_foot",
                       "idle_to_", "one_leg_idle"]),
    ("gesture",       ["wave", "salute", "clap", "cheer", "triumph",
                       "thumbs", "point", "welcome", "greet", "bye",
                       "raise_your_hand", "show_", "shhh", "rock_out",
                       "mic_drop", "count_it", "i_got_this", "eureka",
                       "fist_pump", "bicep", "body_check",
                       "pray", "cross_your", "no_see", "no_hear",
                       "lament", "scream", "confusion", "think",
                       "don_t_know", "omg_", "yawn", "listen",
                       "checking_time", "looking_at",
                       "itching", "scratch", "brush_of_dust",
                       "dusting", "wipe", "rub", "fixing",
                       "body_stretch", "body_search", "pocket_search",
                       "freezing_cold", "shiver", "cough", "sneeze",
                       "puke", "boss_dust", "dust_brushing",
                       "chefs_kiss", "shoulder_clap", "step_in_shit",
                       "meditate", "horse_riding", "shuffle_cards",
                       "drinking_bottle", "eat_burger",
                       "zippo", "smoke", "drink_",
                       "clear_ear", "rubbing_",
                       "neutral_cry", "neutral_fear",
                       "rage_", "proud_", "neutral_laugh",
                       "looking_in_the_mirror",
                       "wiping_shoes", "maybe", "just_realised",
                       "tasty", "no_speak", "no_say",
                       "on_the_edge", "eating", "painting",
                       "stinky", "binoculars", "brush_off",
                       "tarzan", "cry_", "laugh",
                       "welcom", "clear", "alone",
                       "playing", "grating", "peeling", "looting",
                       "chainsaw", "cutting",
                       ]),
    ("sport",         ["swim", "throw_", "catch_", "kick_", "punch_",
                       "dodge_", "play_tennis", "play_guitar",
                       "petting_dog", "dribble", "shoot_",
                       "ib_combat", "ib_dodge", "exercise",
                       "cartwheel"]),
]


def classify_coarse(fine_name: str) -> str:
    """Map a fine-grained action name to a coarse category (~20 classes)."""
    name_lower = fine_name.lower()
    for category, keywords in _COARSE_RULES:
        for kw in keywords:
            if kw in name_lower:
                return category
    return "other"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_motion_lib_dicts(input_paths: list[str]) -> dict[str, dict]:
    """Load and merge motion_lib entries from PKL files and/or directories.

    Supported inputs:
      - Single combined PKL:  {name: entry, ...}
      - Directory of individual PKLs (e.g. --individual output or filtered dir):
        walks the tree, loads every .pkl, merges all {name: entry} dicts.
    """
    merged: dict[str, dict] = {}
    for raw_path in input_paths:
        p = Path(raw_path)
        if not p.exists():
            print(f"ERROR: {raw_path} not found")
            sys.exit(1)

        pkl_paths: list[Path] = []
        if p.is_dir():
            pkl_paths = sorted(p.rglob("*.pkl"))
        elif p.suffix == ".pkl":
            pkl_paths = [p]
        else:
            print(f"WARNING: {raw_path} is neither .pkl nor directory, skipping")
            continue

        for pkl_f in pkl_paths:
            try:
                data = joblib.load(pkl_f)
            except Exception:
                print(f"  WARNING: failed to load {pkl_f}, skipping")
                continue
            if not isinstance(data, dict):
                continue
            # Use relative path prefix to avoid name collisions across sessions
            prefix = ""
            if p.is_dir():
                prefix = str(pkl_f.relative_to(p).with_suffix("")).replace("/", "__") + "__"
            for name, entry in data.items():
                unique_name = prefix + name if prefix else name
                if unique_name in merged:
                    unique_name = f"{pkl_f.stem}__{name}"
                merged[unique_name] = entry

    return merged


def motion_lib_entry_to_textop(name: str, entry: dict) -> dict | None:
    """Convert one motion_lib entry to TextOp SkeletonPrimitiveDataset format.

    motion_lib entry (from convert_soma_csv_to_motion_lib.py):
        {
            "root_trans_offset": ndarray [T, 3],    # meters
            "root_rot":          ndarray [T, 4],    # xyzw quaternion
            "dof":               ndarray [T, 29],   # radians, 29-DOF MJCF order
            "contact_mask":      ndarray [T, 2],    # left/right ∈ {0,1}
            "scene": {                              # optional, from --mob
                "occu_global":    ndarray [X,Y,Z],
                "unit":           float,
                "llb":            ndarray [3],
            },
            "fps":               int,
            "pose_aa":           ...,               # ignored
            "smpl_joints":       ...,               # ignored
        }

    TextOp format:
        {
            "length": int,
            "motion": {                                        # ← from motion_lib entry
                "root_trans_offset": ndarray [T, 3],
                "root_rot":          ndarray [T, 4],
                "dof":               ndarray [T, 23],         29-DOF → 23-DOF (wrists locked)
                "contact_mask":      ndarray [T, 2],
                "fps":               int,
                "motion_len":        int,
            },
            "scene": {                                        # inferred pseudo-obstacles
                "occu_global":       ndarray [X, Y, Z],  bool, 1=occupied (vacant space → obstacle)
                "unit":              float,               voxel size (m)
                "llb":               ndarray [3],         float32, world origin
            },
        }
    """
    dof_29 = entry["dof"]
    if dof_29.shape[1] != 29:
        return None  # unexpected DOF count, skip

    dof_23 = dof_29[:, DOF_29_TO_23]                     # (T, 23)
    T = dof_23.shape[0]
    fps_val = entry.get("fps", 50)

    # ── Coarse action label from filename ──
    duration = T / fps_val
    fine = _extract_action_name(name)
    coarse = classify_coarse(fine)

    return {
        "length": T,
        "motion": {
            "root_trans_offset": entry["root_trans_offset"].astype(np.float32),
            "root_rot": entry["root_rot"].astype(np.float32),
            "dof": dof_23.astype(np.float32),
            "contact_mask": entry["contact_mask"].astype(np.float32),
            "fps": fps_val,
            "motion_len": T,
        },
        # inferred pseudo-obstacles: vacant space treated as occupied
        # (computed by convert_soma_csv_to_motion_lib.py --mob)
        "scene": entry.get("scene", {}),
        # Coarse category → weighted_sample via cal_weighted_statistics.py
        "frame_ann": [(0.0, duration, coarse, [coarse])],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Pack motion_lib PKL(s) → TextOp train.pkl / val.pkl"
    )
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="One or more motion_lib PKL files (output of convert_soma_csv_to_motion_lib.py)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for train.pkl, val.pkl, statistics.yaml",
    )
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min_frames", type=int, default=35,
        help="Minimum sequence length in frames (segment_len with default config)",
    )
    args = parser.parse_args()

    # ── Load all motion_lib entries (supports files + directories) ──
    all_entries = load_motion_lib_dicts(args.input)
    print(f"Loaded {len(all_entries)} total sequences from {len(args.input)} input(s)")

    # ── Convert ──
    converted: list[dict] = []
    skipped = 0
    for name, entry in tqdm(all_entries.items(), desc="Converting"):
        item = motion_lib_entry_to_textop(name, entry)
        if item is None:
            skipped += 1
            continue
        if item["length"] < args.min_frames:
            skipped += 1
            continue
        item["_source"] = name
        converted.append(item)

    print(f"Converted {len(converted)} (skipped {skipped} — too short or wrong DOF count)")

    if not converted:
        print("ERROR: No valid sequences!")
        sys.exit(1)

    # ── Shuffle & split ──
    random.seed(args.seed)
    random.shuffle(converted)
    n_val = max(1, int(len(converted) * args.val_ratio))
    train_data = converted[n_val:]
    val_data = converted[:n_val]

    # ── Save ──
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    train_path = out / "train.pkl"
    val_path = out / "val.pkl"
    print(f"\nSaving train: {len(train_data)} sequences → {train_path}")
    joblib.dump(train_data, train_path)
    print(f"Saving val:   {len(val_data)} sequences → {val_path}")
    joblib.dump(val_data, val_path)

    # ── Statistics ──
    fps_val = int(train_data[0]["motion"]["fps"])
    train_hours = sum(d["motion"]["motion_len"] for d in train_data) / fps_val / 3600
    val_hours = sum(d["motion"]["motion_len"] for d in val_data) / fps_val / 3600

    stats = {
        "dataset name": "BONES-SEED → TextOp (G1 23-DOF, 50fps)",
        "fps": fps_val,
        "nfeats": 57,
        "train count": len(train_data),
        "val count": len(val_data),
        "train hours": round(train_hours, 1),
        "val hours": round(val_hours, 1),
    }
    stats_path = out / "statistics.yaml"
    with open(stats_path, "w") as f:
        yaml.dump(stats, f)
    print(f"Statistics: {stats_path}")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # ── Done ──
    print(f"\nDone. Ready for TextOp VAE training:")
    print(f"  data.datadir={out.resolve()}")
    print(f"  data.weighted_sample=false")
    print(f"  skeleton.asset.assetRoot=<path/to/description/robots/g1/>")


if __name__ == "__main__":
    main()
