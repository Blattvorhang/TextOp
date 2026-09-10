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
import csv
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import yaml
from tqdm import tqdm

try:
    from dataset.data_analyze.analyze_action_distribution import (  # noqa: E402
        classify_coarse as _shared_classify_coarse,
    )
except Exception:  # pragma: no cover - optional fallback
    _shared_classify_coarse = None

TARGET_DOF = 29
FEATURE_DIM_V3 = 11 + 2 * TARGET_DOF
CLIP_ALIGN_FPS = 50
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

_SOURCE_EXT_RE = re.compile(r"\.(?:csv|pkl)$", flags=re.IGNORECASE)
_DATE_PREFIX_RE = re.compile(r"^\d{6}__")
_AUG_SUFFIX_RE = re.compile(r"_aug_\d+$", flags=re.IGNORECASE)
_AUG_FALL_RECOVERY_PREFIX_RE = re.compile(
    r"^(?:.*__)?aug_fall_recovery__",
    flags=re.IGNORECASE,
)
SUBJECT_PREFIX_RE = re.compile(
    r"^(?:a|an|the)\s+"
    r"(?:(?:standing|seated|upright|injured|wounded|crouched|kneeling|sitting|lying|bent)\s+)*"
    r"(?:person(?:'s)?|character(?:'s)?|individual(?:'s)?|figure(?:'s)?|man|woman|dancer|player|actor|someone|somebody)\b[\s,]*"
)
COPULA_PREFIX_RE = re.compile(r"^(?:is|are|was|were|be|been|being)\s+")


# ---------------------------------------------------------------------------
# Metadata-driven text annotations
# ---------------------------------------------------------------------------
def compact_text(text: object) -> str:
    return " ".join(str(text).strip().split())


def normalize_text(text: object) -> str:
    return compact_text(text).lower()


def normalize_motion_text(text: object) -> str:
    text = normalize_text(text)
    text = SUBJECT_PREFIX_RE.sub("", text)
    text = COPULA_PREFIX_RE.sub("", text)
    return text.strip(" ,;:.!?\"'")


def _canonical_motion_name(name: str) -> str:
    stem = compact_text(name)
    stem = stem.replace("\\", "/").split("/")[-1]
    stem = _SOURCE_EXT_RE.sub("", stem)
    stem = _AUG_FALL_RECOVERY_PREFIX_RE.sub("", stem)
    stem = _DATE_PREFIX_RE.sub("", stem)
    stem = _AUG_SUFFIX_RE.sub("", stem)
    return stem


def _lookup_by_motion_name(lookup: dict[str, dict], name: str) -> dict | None:
    raw = compact_text(name)
    if not raw:
        return None
    for candidate in (raw, _canonical_motion_name(raw)):
        if candidate and candidate in lookup:
            return lookup[candidate]
    return None


def load_metadata_lookup(metadata_csv: str) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    with open(metadata_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            short_text = compact_text(row.get("content_short_description", ""))
            if not short_text:
                continue
            row["content_short_description"] = short_text
            key = compact_text(row.get("filename") or row.get("move_name") or "")
            if not key:
                continue
            lookup[key] = row
            lookup[_canonical_motion_name(key)] = row
    return lookup


def load_temporal_lookup(jsonl_path: str) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            key = compact_text(obj.get("filename", ""))
            if not key:
                continue
            lookup[key] = obj
            lookup[_canonical_motion_name(key)] = obj
    return lookup


def build_text_candidates(*texts: object) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for text in texts:
        candidate = compact_text(text)
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    return candidates


def _fallback_classify_coarse(label: str) -> str:
    label_space = re.sub(r"[^a-z0-9]+", " ", str(label).lower()).strip()
    label_underscore = label_space.replace(" ", "_")
    coarse_rules = (
        ("injured", ["injured"]),
        ("crutch", ["crutch", "crutches"]),
        ("jump", ["jump", "hop", "leap", "flip", "vault_over", "jump_and_land"]),
        ("jog", ["jog", "jogging", "run"]),
        ("walk", ["walk", "moonwalk", "step_forward", "step_backward", "stroll"]),
        ("dance", ["dance", "dancing", "choreography", "macarena"]),
        ("climb", ["climb", "ladder", "box"]),
        ("fall", ["fall", "faint", "lying", "lie"]),
        ("crouch", ["crouch", "crawl", "on_all_fours", "stoop"]),
        ("kneel", ["kneel"]),
        ("sit", ["sit", "sitting"]),
        ("carry", ["carry", "lift", "hold", "pick_up", "put_down", "crate"]),
        ("reach", ["reach", "reaching"]),
        ("push", ["push", "pull", "crank", "valve", "handle", "lever", "knob", "door"]),
        ("step_over", ["step_over", "step_in", "avoid_obstacle"]),
        ("turn", ["turn", "spin"]),
        ("idle", ["idle", "stand", "standing", "neutral"]),
        ("gesture", ["wave", "salute", "clap", "cheer", "thumbs", "point", "scratch", "wipe", "rub"]),
        ("sport", ["swim", "throw", "catch", "kick", "punch", "dribble", "shoot", "exercise", "cartwheel"]),
    )
    for category, keywords in coarse_rules:
        for kw in keywords:
            kw_space = re.sub(r"[^a-z0-9]+", " ", kw.lower()).strip()
            kw_underscore = kw_space.replace(" ", "_")
            if kw_space and (kw_space in label_space or kw_underscore in label_underscore):
                return category
    return "other"


def classify_coarse_text(label: object) -> str:
    text = compact_text(label)
    if not text:
        return "other"
    if _shared_classify_coarse is not None:
        try:
            coarse = _shared_classify_coarse(text)
            if coarse != "other":
                return coarse
        except Exception:
            pass
    return _fallback_classify_coarse(text)


def _metadata_act_cat(row: dict[str, str]) -> list[str]:
    candidates = [
        row.get("content_type_of_movement", ""),
        row.get("content_short_description", ""),
        row.get("content_short_description_2", ""),
    ]
    for candidate in candidates:
        coarse = classify_coarse_text(candidate)
        if coarse != "other":
            return [coarse]

    return ["other"]


def _snap_event_times_to_fps(
    start_time: float | int | str | None,
    end_time: float | int | str | None,
    fps: int,
    max_frames: int,
) -> tuple[float, float] | None:
    if start_time is None or end_time is None:
        return None
    start_f = int(math.floor(float(start_time) * fps + 1e-9))
    end_f = int(math.ceil(float(end_time) * fps - 1e-9))
    start_f = max(0, min(start_f, max_frames))
    end_f = max(start_f + 1, min(end_f, max_frames))
    return start_f / float(fps), end_f / float(fps)


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


def _directory_source_prefix(pkl_f: Path, root: Path | None) -> str:
    """Return the source-folder prefix for a directory-packed motion PKL."""
    if root is None:
        return ""
    rel_parent = pkl_f.relative_to(root).parent
    if str(rel_parent) == ".":
        return ""
    return str(rel_parent).replace("/", "__") + "__"


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

        prefix = _directory_source_prefix(pkl_f, root)
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


_SOURCE_EXT_RE = re.compile(r"\.(?:csv|pkl)$", flags=re.IGNORECASE)
_AUG_SUFFIX_RE = re.compile(r"_aug_\d+$", flags=re.IGNORECASE)
_MIRROR_SUFFIX_RE = re.compile(r"_M$", flags=re.IGNORECASE)


def _motion_split_key(source: str) -> str:
    """Return a canonical key for split-only grouping.

    Only the BONES-SEED suffix conventions are normalized:
      - date folder prefixes: ``221010__motion`` -> ``motion``
      - fall-recovery augmentation folder prefix:
        ``aug_fall_recovery__motion_aug_003`` -> ``motion``
      - mirrored files: ``motion_M`` -> ``motion``
      - augmented files: ``motion_aug_003`` -> ``motion``
      - mirrored augmented files: ``motion_M_aug_003`` -> ``motion``
    """
    stem = _canonical_motion_name(source)
    return _MIRROR_SUFFIX_RE.sub("", stem)


def _is_mirrored_motion(source: str) -> bool:
    """Check whether a source name carries the BONES-SEED mirror marker."""
    stem = _SOURCE_EXT_RE.sub("", str(source))
    stem = _AUG_SUFFIX_RE.sub("", stem)
    return _MIRROR_SUFFIX_RE.search(stem) is not None


def _is_augmented_motion(source: str) -> bool:
    """Check whether a source name carries the generated augmentation suffix."""
    stem = _SOURCE_EXT_RE.sub("", str(source))
    return _AUG_SUFFIX_RE.search(stem) is not None


def split_manifest_train_val(
    manifest: list[dict],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    """Split records by canonical motion group to prevent mirror leakage."""
    groups: dict[str, list[dict]] = {}
    for idx, record in enumerate(manifest):
        source = record.get("_source") or record.get("_data_path") or f"record_{idx}"
        key = _motion_split_key(str(source))
        groups.setdefault(key, []).append(record)

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)

    for _, records in group_items:
        rng.shuffle(records)

    total_records = len(manifest)
    can_make_val = val_ratio > 0 and total_records > 1 and len(group_items) > 1
    target_val_count = int(total_records * val_ratio) if can_make_val else 0
    if can_make_val:
        target_val_count = max(1, min(target_val_count, total_records - 1))

    train_data: list[dict] = []
    val_data: list[dict] = []
    train_keys: set[str] = set()
    val_keys: set[str] = set()

    for group_idx, (key, records) in enumerate(group_items):
        keep_train_group = group_idx == len(group_items) - 1
        if len(val_data) < target_val_count and not keep_train_group:
            val_data.extend(records)
            val_keys.add(key)
        else:
            train_data.extend(records)
            train_keys.add(key)

    leakage_keys = train_keys & val_keys
    if leakage_keys:
        examples = ", ".join(sorted(leakage_keys)[:5])
        raise RuntimeError(
            "Mirror-safe split failed: canonical motion group(s) appear in "
            f"both train and val: {examples}"
        )

    mirror_records = 0
    mirror_groups = 0
    augmented_records = 0
    augmented_groups = 0
    paired_original_mirror_groups = 0
    for _key, records in group_items:
        mirror_flags = [
            _is_mirrored_motion(str(record.get("_source", "")))
            for record in records
        ]
        augmented_flags = [
            _is_augmented_motion(str(record.get("_source", "")))
            for record in records
        ]
        mirror_count = sum(mirror_flags)
        augmented_count = sum(augmented_flags)
        mirror_records += mirror_count
        augmented_records += augmented_count
        if mirror_count:
            mirror_groups += 1
        if augmented_count:
            augmented_groups += 1
        if 0 < mirror_count < len(records):
            paired_original_mirror_groups += 1

    stats = {
        "unit": "canonical_motion_group",
        "groups": len(group_items),
        "train_groups": len(train_keys),
        "val_groups": len(val_keys),
        "target_val_count": target_val_count,
        "mirror_records": mirror_records,
        "mirror_groups": mirror_groups,
        "augmented_records": augmented_records,
        "augmented_groups": augmented_groups,
        "paired_original_mirror_groups": paired_original_mirror_groups,
        "leakage_groups": 0,
    }
    return train_data, val_data, stats


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
    source_idx, pkl_f, root, out, min_frames, sample_compress, metadata_lookup, temporal_lookup = task
    try:
        data = joblib.load(pkl_f)
    except Exception as exc:
        return [], 0, set(), f"failed to load {pkl_f}: {exc}"
    if not isinstance(data, dict):
        return [], 0, set(), f"{pkl_f} does not contain a dict"

    prefix = _directory_source_prefix(pkl_f, root)

    records: list[dict] = []
    skipped = 0
    fps_values: set[int] = set()
    warning = None
    for entry_idx, (raw_name, entry) in enumerate(data.items()):
        label_name = str(raw_name)
        source_name = prefix + label_name if prefix else label_name
        metadata_row = (
            _lookup_by_motion_name(metadata_lookup, source_name)
            if metadata_lookup is not None else None
        )
        temporal_obj = (
            _lookup_by_motion_name(temporal_lookup, source_name)
            if temporal_lookup is not None else None
        )
        try:
            item = motion_lib_entry_to_textop(
                source_name,
                entry,
                metadata_row=metadata_row,
                temporal_obj=temporal_obj,
            )
        except (TypeError, ValueError) as exc:
            warning = f"invalid motion {source_name}: {exc}"
            item = None
        if item is None:
            skipped += 1
            continue
        if not item.get("frame_ann"):
            skipped += 1
            warning = f"missing metadata labels for {source_name}"
            continue
        if min_frames and item["length"] < min_frames:
            skipped += 1
            continue

        item["_source"] = source_name
        sample_relpath = (
            Path("samples") / f"{source_idx:08d}_{entry_idx:04d}.pkl"
        )
        joblib.dump(item, out / sample_relpath, compress=sample_compress)
        fps = int(item["motion"]["fps"])
        fps_values.add(fps)
        records.append({
            "length": item["length"],
            "frame_ann": item["frame_ann"],
            "_source": source_name,
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
    metadata_lookup: dict[str, dict[str, str]] | None = None,
    temporal_lookup: dict[str, dict] | None = None,
) -> tuple[list[dict], int, set[int]]:
    """Pack source files with a bounded number of in-flight tasks."""
    tasks = (
        (idx, pkl_f, root, out, min_frames, sample_compress, metadata_lookup, temporal_lookup)
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


def motion_lib_entry_to_textop(
    name: str,
    entry: dict,
    *,
    metadata_row: dict[str, str] | None = None,
    temporal_obj: dict | None = None,
) -> dict | None:
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

    duration = T / fps_val
    recovery_boost = _is_flat_recovery(str(name))

    frame_ann: list[tuple[float, float, list[str], list[str]]] = []
    if metadata_row is not None:
        short_text = compact_text(metadata_row.get("content_short_description", ""))
        if short_text:
            sequence_texts = build_text_candidates(
                short_text,
                normalize_motion_text(short_text),
            )
            frame_ann.append(
                (0.0, duration, sequence_texts, _metadata_act_cat(metadata_row))
            )

            if temporal_obj is not None:
                events = temporal_obj.get("events") or []
                for event in events:
                    temporal_raw = compact_text(event.get("description", ""))
                    if not temporal_raw:
                        continue
                    snapped = _snap_event_times_to_fps(
                        event.get("start_time"),
                        event.get("end_time"),
                        fps_val,
                        T,
                    )
                    if snapped is None:
                        continue
                    event_core = normalize_motion_text(temporal_raw) or normalize_text(temporal_raw)
                    event_texts = build_text_candidates(
                        temporal_raw,
                        event_core,
                        short_text,
                    )
                    frame_ann.append(
                        (snapped[0], snapped[1], event_texts, _metadata_act_cat(metadata_row))
                    )

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
        "frame_ann": frame_ann,
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
    parser.add_argument(
        "--metadata-csv",
        default="/home/lenovo/data/bones-seed/metadata/seed_metadata_v004.csv",
        help="BONES-SEED metadata CSV with content_short_description",
    )
    parser.add_argument(
        "--temporal-jsonl",
        default="/home/lenovo/data/bones-seed/metadata/seed_metadata_v002_temporal_labels.jsonl",
        help="BONES-SEED temporal label JSONL with per-event descriptions",
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
    print(f"Metadata CSV: {args.metadata_csv}")
    print(f"Temporal JSONL: {args.temporal_jsonl}")
    effective_workers = min(args.workers, len(source_pkls))
    print(
        f"Packing with {effective_workers} worker(s), "
        f"joblib compression={args.sample_compress}"
    )

    metadata_lookup = load_metadata_lookup(args.metadata_csv)
    temporal_lookup = load_temporal_lookup(args.temporal_jsonl)
    print(
        f"Loaded {len(metadata_lookup):,} metadata keys and "
        f"{len(temporal_lookup):,} temporal keys"
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
        metadata_lookup=metadata_lookup,
        temporal_lookup=temporal_lookup,
    )

    print(f"Converted {len(manifest)} (skipped {skipped} - too short or invalid)")

    if not manifest:
        print("ERROR: No valid sequences!")
        sys.exit(1)
    if len(fps_values) != 1:
        print(f"ERROR: Mixed source frame rates are unsupported: {sorted(fps_values)}")
        sys.exit(1)

    # ── Grouped shuffle & split ──
    train_data, val_data, split_stats = split_manifest_train_val(
        manifest, args.val_ratio, args.seed
    )
    print(
        "Split by canonical motion groups: "
        f"train={len(train_data)} seqs/{split_stats['train_groups']} groups, "
        f"val={len(val_data)} seqs/{split_stats['val_groups']} groups "
        f"(target val seqs={split_stats['target_val_count']})"
    )
    if args.val_ratio > 0 and split_stats["groups"] <= 1:
        print(
            "WARNING: Validation split is empty because only one canonical "
            "motion group is available; keeping the group intact in train."
        )

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
        "split": split_stats,
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
