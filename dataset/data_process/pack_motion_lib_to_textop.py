#!/usr/bin/env python3
"""Pack motion_lib PKL(s) into TextOp RobotMDAR SkeletonPrimitiveDataset format.

This is the second stage of the BONES-SEED → TextOp pipeline:

  Stage 1: convert_soma_csv_to_motion_lib.py
      BONES-SEED CSV (120Hz, 29-DOF, cm+deg)
      → motion_lib PKL (50Hz, 29-DOF, m+rad, per-name dict)

  Stage 2: pack_motion_lib_to_textop.py  ← this script
      motion_lib PKL
      → TextOp train.pkl / val.pkl manifests + samples/ (29-DOF)

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
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os
import random
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import yaml
from tqdm import tqdm

TARGET_DOF = 29
FEATURE_DIM_V3 = 11 + 2 * TARGET_DOF
TARGET_DOF_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
assert len(TARGET_DOF_NAMES) == TARGET_DOF


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


# ── Flat-lying fall recovery detection ──
# Side-lying and crutch variants are excluded by filter_and_copy_bones_data.py.
# The remaining kept recovery actions are:
#   stand_up_lying, stand_up_lying_stomach,
#   faint_stand_up_lying, faint_stand_up_lying_stomach,
#   faint_stand_up_lying_puke_walk_ff*
_RECOVERY_FINE_PATTERNS = (
    "stand_up_lying",        # stand_up_lying, stand_up_lying_stomach
    "faint_stand_up_lying",  # faint_stand_up_lying, faint_stand_up_lying_stomach,
                             # faint_stand_up_lying_puke_walk_ff*
)


def _is_flat_recovery(fine_name: str) -> bool:
    """Check whether a fine action name is a kept flat-lying recovery motion."""
    lower = fine_name.lower()
    # Belt-and-suspenders: side-lying should already be filtered, but
    # explicitly exclude it in case the filter script wasn't used.
    if "lying_side" in lower:
        return False
    return any(pat in lower for pat in _RECOVERY_FINE_PATTERNS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def discover_motion_lib_pkls(input_paths: list[str]) -> list[tuple[Path, Path | None]]:
    """Return input PKLs and their optional directory roots.

    Supported inputs:
      - Single combined PKL:  {name: entry, ...}
      - Directory of individual PKLs (e.g. --individual output or filtered dir):
        walks the tree without loading their contents.
    """
    discovered: list[tuple[Path, Path | None]] = []
    for raw_path in input_paths:
        p = Path(raw_path)
        if not p.exists():
            print(f"ERROR: {raw_path} not found")
            sys.exit(1)

        if p.is_dir():
            discovered.extend((pkl_path, p) for pkl_path in sorted(p.rglob("*.pkl")))
        elif p.suffix == ".pkl":
            discovered.append((p, None))
        else:
            print(f"WARNING: {raw_path} is neither .pkl nor directory, skipping")
    return discovered


def iter_motion_lib_dicts(
    input_paths: list[str],
):
    """Yield entries while retaining at most one source PKL in memory."""
    seen_names: set[str] = set()
    pkl_paths = discover_motion_lib_pkls(input_paths)
    for pkl_f, root in tqdm(pkl_paths, desc="Reading source PKLs"):
        try:
            data = joblib.load(pkl_f)
        except Exception as exc:
            print(f"  WARNING: failed to load {pkl_f}: {exc}")
            continue
        if not isinstance(data, dict):
            print(f"  WARNING: {pkl_f} does not contain a dict, skipping")
            continue

        prefix = ""
        if root is not None:
            prefix = str(pkl_f.relative_to(root).with_suffix("")).replace("/", "__") + "__"
        for raw_name, entry in data.items():
            name = str(raw_name)
            unique_name = prefix + name if prefix else name
            if unique_name in seen_names:
                base_name = f"{pkl_f.stem}__{name}"
                unique_name = base_name
                suffix = 2
                while unique_name in seen_names:
                    unique_name = f"{base_name}__{suffix}"
                    suffix += 1
            seen_names.add(unique_name)
            yield unique_name, entry


def load_motion_lib_dicts(input_paths: list[str]) -> dict[str, dict]:
    """Compatibility helper; prefer iter_motion_lib_dicts for large datasets."""
    return dict(iter_motion_lib_dicts(input_paths))


def _manifest_hours(records: list[dict], fallback_fps: int) -> float:
    return sum(
        record["length"] / float(record.get("_fps", fallback_fps))
        for record in records
    ) / 3600


def _recovery_manifest_stats(
    records: list[dict],
    total_hours: float,
    fallback_fps: int,
) -> dict[str, float]:
    recovery = [record for record in records if record.get("_recovery_boost")]
    hours = _manifest_hours(recovery, fallback_fps)
    return {
        "count": len(recovery),
        "hours": round(hours, 3),
        "pct_data": round(hours / total_hours * 100, 3)
        if total_hours > 0 else 0.0,
    }


def _pack_source_file(task: tuple) -> tuple[list[dict], int, set[int], str | None]:
    """Load, convert, and save one source PKL.

    This function is thread-safe: every source index owns a disjoint output
    filename prefix. Returning metadata only keeps executor memory bounded.
    """
    source_idx, pkl_f, root, out, min_frames, sample_compress = task
    try:
        data = joblib.load(pkl_f)
    except Exception as exc:
        return [], 0, set(), f"failed to load {pkl_f}: {exc}"
    if not isinstance(data, dict):
        return [], 0, set(), f"{pkl_f} does not contain a dict"

    prefix = ""
    if root is not None:
        prefix = str(pkl_f.relative_to(root).with_suffix("")).replace("/", "__") + "__"

    records: list[dict] = []
    skipped = 0
    fps_values: set[int] = set()
    warning = None
    for entry_idx, (raw_name, entry) in enumerate(data.items()):
        name = prefix + str(raw_name) if prefix else str(raw_name)
        try:
            item = motion_lib_entry_to_textop(name, entry)
        except (TypeError, ValueError) as exc:
            warning = f"invalid motion {name}: {exc}"
            item = None
        if item is None or (min_frames and item["length"] < min_frames):
            skipped += 1
            continue

        item["_source"] = name
        sample_relpath = (
            Path("samples") / f"{source_idx:08d}_{entry_idx:04d}.pkl"
        )
        joblib.dump(item, out / sample_relpath, compress=sample_compress)
        fps = int(item["motion"]["fps"])
        fps_values.add(fps)
        records.append({
            "length": item["length"],
            "frame_ann": item["frame_ann"],
            "_source": name,
            "_data_path": sample_relpath.as_posix(),
            "_fps": fps,
            "_recovery_boost": item.get("_recovery_boost", False),
        })
    return records, skipped, fps_values, warning


def pack_source_files(
    source_pkls: list[tuple[Path, Path | None]],
    out: Path,
    min_frames: int,
    sample_compress: int,
    workers: int,
) -> tuple[list[dict], int, set[int]]:
    """Pack source files with a bounded number of in-flight tasks."""
    tasks = (
        (idx, pkl_f, root, out, min_frames, sample_compress)
        for idx, (pkl_f, root) in enumerate(source_pkls)
    )
    manifest: list[dict] = []
    skipped = 0
    fps_values: set[int] = set()

    if workers == 1:
        results = map(_pack_source_file, tasks)
        for records, task_skipped, task_fps, warning in tqdm(
            results, total=len(source_pkls), desc="Packing source PKLs"
        ):
            manifest.extend(records)
            skipped += task_skipped
            fps_values.update(task_fps)
            if warning:
                tqdm.write(f"  WARNING: {warning}")
        return manifest, skipped, fps_values

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = deque()
        task_iter = iter(tasks)
        for _ in range(min(len(source_pkls), workers * 2)):
            pending.append(executor.submit(_pack_source_file, next(task_iter)))

        with tqdm(total=len(source_pkls), desc="Packing source PKLs") as progress:
            while pending:
                records, task_skipped, task_fps, warning = pending.popleft().result()
                manifest.extend(records)
                skipped += task_skipped
                fps_values.update(task_fps)
                if warning:
                    tqdm.write(f"  WARNING: {warning}")
                progress.update()
                try:
                    pending.append(executor.submit(_pack_source_file, next(task_iter)))
                except StopIteration:
                    pass
    return manifest, skipped, fps_values


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
                "dof":               ndarray [T, 29],         native G1 order, wrists retained
                "contact_mask":      ndarray [T, 2],
                "sliding_mask":      ndarray [T, 2],         diagnostic/training side channel
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
    required_keys = ("dof", "root_trans_offset", "root_rot", "contact_mask")
    if not isinstance(entry, dict) or any(key not in entry for key in required_keys):
        return None

    dof = np.asarray(entry["dof"])
    if dof.ndim != 2 or dof.shape[1] != TARGET_DOF:
        return None  # unexpected DOF count, skip
    dof_order = entry.get("dof_order")
    if dof_order is not None and str(dof_order).lower() not in ("mj", "mujoco"):
        raise ValueError(f"Expected MuJoCo DOF order, got {dof_order!r}")
    dof_names = entry.get("dof_names")
    if dof_names is not None and tuple(dof_names) != TARGET_DOF_NAMES:
        raise ValueError(
            "Motion joint order differs from the TextOp 29-DOF contract: "
            f"got {tuple(dof_names)}"
        )

    T = dof.shape[0]
    root_trans = np.asarray(entry["root_trans_offset"])
    root_rot = np.asarray(entry["root_rot"])
    contact_mask = np.asarray(entry["contact_mask"])
    sliding_mask = np.asarray(entry.get("sliding_mask", np.zeros_like(contact_mask)))
    if (
        root_trans.shape != (T, 3)
        or root_rot.shape != (T, 4)
        or contact_mask.shape != (T, 2)
        or sliding_mask.shape != (T, 2)
    ):
        return None
    if not all(np.isfinite(array).all() for array in (
        dof, root_trans, root_rot, contact_mask, sliding_mask
    )):
        return None

    fps_val = int(entry.get("fps", 50))
    if fps_val <= 0:
        return None

    # ── Coarse action label from filename ──
    duration = T / fps_val
    fine = _extract_action_name(str(name))
    coarse = classify_coarse(fine)
    recovery_boost = _is_flat_recovery(fine)

    return {
        "length": T,
        "_recovery_boost": recovery_boost,  # marks flat-lying recovery
        "motion": {
            "root_trans_offset": root_trans.astype(np.float32, copy=False),
            "root_rot": root_rot.astype(np.float32, copy=False),
            "dof": dof.astype(np.float32, copy=False),
            "dof_order": "mujoco",
            "dof_names": list(TARGET_DOF_NAMES),
            "contact_mask": contact_mask.astype(np.float32, copy=False),
            "sliding_mask": sliding_mask.astype(np.float32, copy=False),
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
        "--sample_compress", type=int, default=3, choices=range(0, 10),
        metavar="0-9",
        help="Joblib compression for individual sample files (default: 3).",
    )
    parser.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 1),
        help="Parallel source-file workers (default: min(8, CPU count)).",
    )
    parser.add_argument(
        "--min_frames", type=int, default=0,
        help="Optional model-independent data-quality filter. Disabled by "
        "default; training-window validity is evaluated from the active config.",
    )
    args = parser.parse_args()
    if args.min_frames < 0:
        parser.error("--min_frames must be non-negative")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if not 0.0 <= args.val_ratio < 1.0:
        parser.error("--val_ratio must be in [0, 1)")

    source_pkls = discover_motion_lib_pkls(args.input)
    if not source_pkls:
        print("ERROR: No input PKL files found")
        sys.exit(1)
    print(f"Found {len(source_pkls):,} PKL files from {len(args.input)} input(s)")
    effective_workers = min(args.workers, len(source_pkls))
    print(
        f"Packing with {effective_workers} worker(s), "
        f"joblib compression={args.sample_compress}"
    )

    out = Path(args.output)
    samples_dir = out / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Each worker persists samples directly and returns metadata only. The
    # bounded task queue limits peak memory to roughly 2 * workers source files.
    manifest, skipped, fps_values = pack_source_files(
        source_pkls=source_pkls,
        out=out,
        min_frames=args.min_frames,
        sample_compress=args.sample_compress,
        workers=effective_workers,
    )

    print(f"Converted {len(manifest)} (skipped {skipped} - too short or invalid)")

    if not manifest:
        print("ERROR: No valid sequences!")
        sys.exit(1)
    if len(fps_values) != 1:
        print(f"ERROR: Mixed source frame rates are unsupported: {sorted(fps_values)}")
        sys.exit(1)

    # ── Shuffle & split ──
    random.Random(args.seed).shuffle(manifest)
    n_val = int(len(manifest) * args.val_ratio)
    if args.val_ratio > 0 and len(manifest) > 1:
        n_val = max(1, min(n_val, len(manifest) - 1))
    train_data = manifest[n_val:]
    val_data = manifest[:n_val]

    # ── Save ──
    train_path = out / "train.pkl"
    val_path = out / "val.pkl"
    print(f"\nSaving train: {len(train_data)} sequences → {train_path}")
    joblib.dump(train_data, train_path)
    print(f"Saving val:   {len(val_data)} sequences → {val_path}")
    joblib.dump(val_data, val_path)

    # ── Statistics ──
    fps_val = next(iter(fps_values))
    train_hours = _manifest_hours(train_data, fps_val)
    val_hours = _manifest_hours(val_data, fps_val)
    total_hours = train_hours + val_hours
    train_recovery = _recovery_manifest_stats(train_data, train_hours, fps_val)
    val_recovery = _recovery_manifest_stats(val_data, val_hours, fps_val)
    all_recovery = _recovery_manifest_stats(manifest, total_hours, fps_val)

    stats = {
        "dataset name": "BONES-SEED → TextOp (G1 29-DOF, 50fps)",
        "fps": fps_val,
        "dof_dim": TARGET_DOF,
        "dof_order": "mujoco",
        "dof_names": list(TARGET_DOF_NAMES),
        "nfeats": FEATURE_DIM_V3,
        "storage": "lazy-sample-manifest-v1",
        "train count": len(train_data),
        "val count": len(val_data),
        "train hours": round(train_hours, 1),
        "val hours": round(val_hours, 1),
        "recovery_boost": {
            "train": train_recovery,
            "val": val_recovery,
            "all": all_recovery,
        },
    }
    stats_path = out / "statistics.yaml"
    with open(stats_path, "w") as f:
        yaml.dump(stats, f)
    print(f"Statistics: {stats_path}")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # ── Done ──
    print("\nDone. Ready for TextOp VAE training:")
    print(f"  data.datadir={out.resolve()}")
    print("  data.weighted_sample=false")
    print("  skeleton.asset.assetRoot=<path/to/description/robots/g1/>")


if __name__ == "__main__":
    main()
