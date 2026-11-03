#!/usr/bin/env python3
"""Scan G1 training motions for self-collision with MuJoCo's fast detector.

The scanner supports the two source layouts used by this repository::

    data/bones-seed/g1/csv       # flat Bones-SEED CSV files
    data/hiphi/motions           # HiPhi motion_actor.npz files

For every sampled frame the script writes the root pose and the 29 joint
positions into a G1 MuJoCo model, runs ``mj_kinematics`` followed by
``mj_collision``, and keeps only contacts between two robot bodies.  Floor
contacts are therefore excluded.  ``mj_collision`` is deliberately used
instead of ``mj_forward``: collision detection still needs FK, but it does
not need the dynamics and sensor stages.

The output is JSON and JSONL.  Each motion record contains source metadata,
collision intervals, and contact-pair statistics, so a later cleaning step
can recover the exact frame ranges without reopening the original data.

Run from the repository root, for example::

    python dataset/data_analyze/self_collision/analyze_self_collision.py --workers 8

Use ``--limit`` before a full run to check the model and throughput.  The
default process executor is useful for the large HiPhi collection; threads
are also available when the installed MuJoCo build releases the GIL well.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Iterable
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BONES_DIR = ROOT / "data/bones-seed/g1/csv"
DEFAULT_HIPHI_DIR = ROOT / "data/hiphi/motions"
DEFAULT_MODEL = ROOT / "TextOpRobotMDAR/description/robots/g1/g1_29dof_with_collision.xml"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "analysis_output/self_collision"
DEFAULT_TPOSE_JSON = Path(__file__).resolve().parents[1] / "analysis_output/hiphi_tpose/full_scan_endpoints_adaptive/removal_intervals.json"
SCHEMA_VERSION = 1
SEVERE_DEPTH_M = 0.03
SEVERE_MIN_SECONDS = 0.10

# MuJoCo order: values[..., MUJOCO_TO_ISAACLAB] changes IsaacLab input to
# the actuator/joint order in the G1 MJCF.  This is the same permutation as
# dataset/data_process/convert_soma_csv_to_motion_lib.py.
MUJOCO_TO_ISAACLAB = np.asarray(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19,
     21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int64,
)

BONES_JOINT_COLUMNS = (
    "left_hip_pitch_joint_dof", "left_hip_roll_joint_dof", "left_hip_yaw_joint_dof",
    "left_knee_joint_dof", "left_ankle_pitch_joint_dof", "left_ankle_roll_joint_dof",
    "right_hip_pitch_joint_dof", "right_hip_roll_joint_dof", "right_hip_yaw_joint_dof",
    "right_knee_joint_dof", "right_ankle_pitch_joint_dof", "right_ankle_roll_joint_dof",
    "waist_yaw_joint_dof", "waist_roll_joint_dof", "waist_pitch_joint_dof",
    "left_shoulder_pitch_joint_dof", "left_shoulder_roll_joint_dof",
    "left_shoulder_yaw_joint_dof", "left_elbow_joint_dof", "left_wrist_roll_joint_dof",
    "left_wrist_pitch_joint_dof", "left_wrist_yaw_joint_dof",
    "right_shoulder_pitch_joint_dof", "right_shoulder_roll_joint_dof",
    "right_shoulder_yaw_joint_dof", "right_elbow_joint_dof", "right_wrist_roll_joint_dof",
    "right_wrist_pitch_joint_dof", "right_wrist_yaw_joint_dof",
)

# One model and one data object per worker.  MjModel is read-only while a scan
# is running; MjData stays private to the task, which makes both process and
# thread executors safe.
_MODEL = None
_MODEL_SOURCE = None
_MODEL_LOCK = threading.Lock()


def _import_mujoco():
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "MuJoCo is required. Use the project's textop environment or install mujoco."
        ) from exc
    return mujoco


def _load_collision_model(model_path: str):
    """Load the collision MJCF, fixing the repository's stale meshdir in memory.

    ``g1_29dof_with_collision.xml`` was authored next to a ``meshes/g1``
    directory, while this checkout stores the meshes directly in ``g1/meshes``.
    A temporary XML next to the source makes the relative mesh path valid and
    avoids changing a tracked asset.
    """
    mujoco = _import_mujoco()
    source = Path(model_path).resolve()
    xml = source.read_text(encoding="utf-8")
    temporary = None
    if 'meshdir="../meshes/g1"' in xml:
        xml = xml.replace('meshdir="../meshes/g1"', 'meshdir="meshes"')
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".xml", prefix=".textop_collision_",
            dir=source.parent, delete=False,
        )
        handle.write(xml)
        handle.close()
        temporary = Path(handle.name)
        load_path = temporary
    else:
        load_path = source
    try:
        model = mujoco.MjModel.from_xml_path(str(load_path))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    if model.nq < 36:
        raise ValueError(f"Expected a free joint plus 29 DOFs, got nq={model.nq}")
    if not np.any(np.asarray(model.geom_contype) != 0):
        raise ValueError(f"Collision model has no enabled geoms: {source}")
    return model


def _get_model(model_path: str):
    global _MODEL, _MODEL_SOURCE
    with _MODEL_LOCK:
        if _MODEL is None or _MODEL_SOURCE != model_path:
            _MODEL = _load_collision_model(model_path)
            _MODEL_SOURCE = model_path
    return _MODEL


def _as_scalar(value: np.ndarray | Any) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar, got shape {array.shape}")
    return float(array.reshape(-1)[0])


def _validate_motion(root_pos: np.ndarray, root_quat: np.ndarray, dof: np.ndarray, fps: float):
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"Expected root position [T, 3], got {root_pos.shape}")
    if root_quat.shape != (len(root_pos), 4):
        raise ValueError(f"Expected root quaternion [T, 4], got {root_quat.shape}")
    if dof.shape != (len(root_pos), 29):
        raise ValueError(f"Expected 29 DOFs [T, 29], got {dof.shape}")
    if len(root_pos) == 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid motion length/FPS: {len(root_pos)}/{fps}")
    if not (np.isfinite(root_pos).all() and np.isfinite(root_quat).all() and np.isfinite(dof).all()):
        raise ValueError("Motion contains non-finite pose values")
    lengths = np.linalg.norm(root_quat, axis=1)
    if np.any(lengths < 1e-8):
        raise ValueError("Motion contains zero-length root quaternions")
    root_quat /= lengths[:, None]


def _load_bones_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any]]:
    import pandas as pd
    from scipy.spatial.transform import Rotation

    columns = [
        "root_translateX", "root_translateY", "root_translateZ",
        "root_rotateX", "root_rotateY", "root_rotateZ", *BONES_JOINT_COLUMNS,
    ]
    frame = pd.read_csv(path, usecols=columns)
    values = frame.to_numpy(dtype=np.float64, copy=True)
    root_pos = values[:, 0:3].astype(np.float32) / 100.0  # cm -> m
    root_quat_xyzw = Rotation.from_euler("xyz", values[:, 3:6], degrees=True).as_quat()
    root_quat = root_quat_xyzw[:, [3, 0, 1, 2]].astype(np.float32)  # xyzw -> wxyz
    dof = np.deg2rad(values[:, 6:]).astype(np.float32)
    _validate_motion(root_pos, root_quat, dof, 120.0)
    return root_pos, root_quat, dof, 120.0, {"format": "bones-seed-csv", "angle_unit": "degree", "position_unit": "cm"}


def _load_hiphi_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        required = {"fps", "joint_pos", "body_pos_w", "body_quat_w"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"missing NPZ fields: {missing}")
        fps = _as_scalar(data["fps"])
        dof_il = np.asarray(data["joint_pos"], dtype=np.float32)
        body_pos = np.asarray(data["body_pos_w"], dtype=np.float32)
        body_quat = np.asarray(data["body_quat_w"], dtype=np.float32)
    if dof_il.ndim != 2 or dof_il.shape[1] != 29:
        raise ValueError(f"Expected joint_pos [T, 29], got {dof_il.shape}")
    if body_pos.ndim != 3 or body_pos.shape[1:] != (30, 3):
        raise ValueError(f"Expected body_pos_w [T, 30, 3], got {body_pos.shape}")
    if body_quat.ndim != 3 or body_quat.shape[1:] != (30, 4):
        raise ValueError(f"Expected body_quat_w [T, 30, 4], got {body_quat.shape}")
    if len(dof_il) != len(body_pos) or len(dof_il) != len(body_quat):
        raise ValueError("NPZ pose arrays have different frame counts")
    dof = dof_il[:, MUJOCO_TO_ISAACLAB]
    root_pos = body_pos[:, 0, :]
    root_quat = body_quat[:, 0, :].copy()  # HiPhi stores wxyz
    _validate_motion(root_pos, root_quat, dof, fps)
    return root_pos, root_quat, dof, fps, {"format": "hiphi-motion-actor-npz", "joint_order_input": "isaaclab", "joint_order_model": "mujoco"}


def _load_sidecar_json(path: Path) -> dict[str, Any] | None:
    """Keep a small source-side JSON annotation when one is present."""
    candidates = (path.with_suffix(".json"), path.parent / "metadata.json")
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return {"path": str(candidate), "value": value}
    return None


def _runs(frames: Iterable[int]) -> list[list[int]]:
    values = list(frames)
    if not values:
        return []
    intervals: list[list[int]] = []
    start = previous = values[0]
    for frame in values[1:]:
        if frame != previous + 1:
            intervals.append([start, previous + 1])
            start = frame
        previous = frame
    intervals.append([start, previous + 1])
    return intervals


def _sampled_runs(frames: Iterable[int], stride: int) -> list[list[int]]:
    """Return frame intervals for sampled collision evidence."""
    values = list(frames)
    if not values:
        return []
    intervals: list[list[int]] = []
    start = previous = values[0]
    for frame in values[1:]:
        if frame != previous + stride:
            intervals.append([start, previous + stride])
            start = frame
        previous = frame
    intervals.append([start, previous + stride])
    return intervals


def _validate_keep_intervals(intervals: list[list[int]], frames: int) -> list[list[int]]:
    previous = 0
    for start, end in intervals:
        if not 0 <= previous <= start < end <= frames:
            raise ValueError(f"Invalid or overlapping keep interval [{start}, {end}) for {frames} frames")
        previous = end
    return intervals


def _analyze_task(task: tuple[str, str, str, str, int, dict[str, Any] | None]) -> dict[str, Any]:
    dataset, source, relative, model_path, stride, trim = task
    path = Path(source)
    record: dict[str, Any] = {
        "dataset": dataset,
        "source": relative,
        "source_path": str(path),
        "status": "error",
    }
    try:
        if dataset == "bones-seed":
            root_pos, root_quat, dof, fps, input_info = _load_bones_csv(path)
        elif dataset == "hiphi":
            root_pos, root_quat, dof, fps, input_info = _load_hiphi_npz(path)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        keep_intervals = (_validate_keep_intervals(trim["keep_intervals_frames"], len(dof))
                          if trim is not None else [[0, len(dof)]])
        if trim is not None and (trim["frames"], trim["fps"]) != (len(dof), fps):
            raise ValueError("T-pose JSON timeline differs from source NPZ")
        model = _get_model(str(Path(model_path).resolve()))
        mujoco = _import_mujoco()
        data = mujoco.MjData(model)
        geom_names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
            for geom_id in range(model.ngeom)
        )
        collision_frames: list[int] = []
        severe_frames: list[int] = []
        severe_frame_pairs: dict[int, set[tuple[str, str]]] = {}
        severe_frame_depths: dict[int, float] = {}
        frame_contact_counts: dict[int, int] = {}
        pair_stats: dict[tuple[str, str], dict[str, Any]] = {}
        max_contacts = 0
        checked_frames = sum(len(range(start, end, stride)) for start, end in keep_intervals)
        for frame in (frame for start, end in keep_intervals for frame in range(start, end, stride)):
            data.qpos[:3] = root_pos[frame]
            data.qpos[3:7] = root_quat[frame]
            data.qpos[7:36] = dof[frame]
            # FK is required to update geom_xpos/geom_xmat.  mj_collision then
            # runs broadphase+narrowphase only; no dynamics are evaluated.
            mujoco.mj_kinematics(model, data)
            mujoco.mj_collision(model, data)
            contacts = []
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                geom1, geom2 = int(contact.geom[0]), int(contact.geom[1])
                body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
                # World body 0 is the floor; same-body geometry contacts are
                # not robot self-collisions either.
                if body1 == 0 or body2 == 0 or body1 == body2:
                    continue
                pair = tuple(sorted((geom_names[geom1], geom_names[geom2])))
                distance = float(contact.dist)
                if distance < 0:
                    contacts.append((pair, distance))
            if not contacts:
                continue
            collision_frames.append(frame)
            severe_contacts = [(pair, -distance) for pair, distance in contacts
                               if distance <= -SEVERE_DEPTH_M]
            if severe_contacts:
                severe_frames.append(frame)
                severe_frame_pairs[frame] = {pair for pair, _ in severe_contacts}
                severe_frame_depths[frame] = max(depth for _, depth in severe_contacts)
            frame_contact_counts[frame] = len(contacts)
            max_contacts = max(max_contacts, len(contacts))
            seen_pairs: set[tuple[str, str]] = set()
            for pair, distance in contacts:
                stats = pair_stats.setdefault(pair, {"geom1": pair[0], "geom2": pair[1], "frames": 0, "contacts": 0, "first_frame": frame, "last_frame": frame, "min_distance_m": distance})
                if pair not in seen_pairs:
                    stats["frames"] += 1
                    seen_pairs.add(pair)
                stats["contacts"] += 1
                stats["first_frame"] = min(stats["first_frame"], frame)
                stats["last_frame"] = max(stats["last_frame"], frame)
                stats["min_distance_m"] = min(stats["min_distance_m"], distance)
        def runs_within_keep(frames: list[int]) -> list[list[int]]:
            return [[a, min(end, b)] for start, end in keep_intervals
                    for a, b in _sampled_runs((f for f in frames if start <= f < end), stride)]

        intervals = runs_within_keep(collision_frames)
        severe_intervals = []
        min_severe_samples = max(1, math.ceil(SEVERE_MIN_SECONDS * fps / stride))
        for start, end in runs_within_keep(severe_frames):
            frames_in_run = range(start, end, stride)
            if len(frames_in_run) < min_severe_samples:
                continue
            pairs = sorted(set().union(*(severe_frame_pairs[f] for f in frames_in_run)))
            severe_intervals.append({
                "start_frame": start, "end_frame": end,
                "start_seconds": start / fps, "end_seconds": end / fps,
                "sampled_frames": len(frames_in_run),
                "max_penetration_m": max(severe_frame_depths[f] for f in frames_in_run),
                "contact_pairs": [[a, b] for a, b in pairs],
            })
        # A stride greater than one represents sampled evidence.  Keep the
        # half-open interval convention and make its upper edge cover the
        # final sampled frame without claiming unseen frames were checked.
        segments = []
        for start, end in intervals:
            segment_end = end
            frames_in_segment = [f for f in collision_frames if start <= f < end]
            segments.append({
                "dataset": dataset,
                "source": relative,
                "start_frame": start,
                "end_frame": segment_end,
                "start_seconds": start / fps,
                "end_seconds": segment_end / fps,
                "collision_frames_sampled": len(frames_in_segment),
                "max_contacts": max((frame_contact_counts[f] for f in frames_in_segment), default=0),
                "contact_pairs": [
                    [pair[0], pair[1]]
                    for pair, stats in sorted(pair_stats.items())
                    if stats["first_frame"] < segment_end and stats["last_frame"] >= start
                ],
            })
        stat = path.stat()
        record.update(
            status="ok",
            input=input_info,
            source_stat={"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
            sidecar_json=_load_sidecar_json(path),
            fps=float(fps),
            frames=int(len(dof)),
            duration_seconds=float(len(dof) / fps),
            keep_intervals_frames=keep_intervals,
            kept_frames=sum(end - start for start, end in keep_intervals),
            removed_tpose_frames=len(dof) - sum(end - start for start, end in keep_intervals),
            sampling_stride=int(stride),
            checked_frames=checked_frames,
            collision_frames=int(len(collision_frames)),
            severe_frames=int(len(severe_frames)),
            severe_intervals=severe_intervals,
            max_penetration_m=max((-item["min_distance_m"] for item in pair_stats.values()), default=0.0),
            collision_frame_fraction=float(len(collision_frames) / max(1, checked_frames)),
            collision_intervals_frames=intervals,
            collision_intervals_seconds=[[a / fps, min(len(dof), b) / fps] for a, b in intervals],
            segments=segments,
            max_contacts=int(max_contacts),
            contact_pairs=sorted(pair_stats.values(), key=lambda item: (item["geom1"], item["geom2"])),
        )
    except Exception as exc:  # keep one malformed source from stopping a full scan
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def _fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _canonical_source(relative: str) -> str | None:
    path = Path(relative)
    if not path.parent.name.endswith("__mirror"):
        return None
    return (path.parent.with_name(path.parent.name.removesuffix("__mirror")) / path.name).as_posix()


def _read_hiphi_timeline(path: Path) -> tuple[int, float]:
    """Read only the array header and FPS from a mirror NPZ."""
    with zipfile.ZipFile(path) as archive:
        with archive.open("joint_pos.npy") as stream:
            version = np.lib.format.read_magic(stream)
            reader = {
                (1, 0): np.lib.format.read_array_header_1_0,
                (2, 0): np.lib.format.read_array_header_2_0,
                (3, 0): np.lib.format.read_array_header_2_0,
            }.get(version)
            if reader is None:
                raise ValueError(f"Unsupported NPY header: {version}")
            shape, _, _ = reader(stream)
        with archive.open("fps.npy") as stream:
            fps = _as_scalar(np.load(stream, allow_pickle=False))
    if len(shape) != 2 or shape[1] != 29 or shape[0] <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid mirror timeline: {shape}, {fps}")
    return int(shape[0]), fps


def _mirror_geom(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return name


def _mirror_pairs(pairs: list[list[str]]) -> list[list[str]]:
    return [list(pair) for pair in sorted({tuple(sorted(map(_mirror_geom, pair))) for pair in pairs})]


def _inherit_mirror(original: dict[str, Any], task: tuple[str, str, str],
                    source_stat: dict[str, int]) -> dict[str, Any]:
    dataset, source, relative = task
    record = deepcopy(original)
    record.update(dataset=dataset, source=relative, source_path=source,
                  source_stat=source_stat, sidecar_json=_load_sidecar_json(Path(source)),
                  collision_method="mirror_inherited", evidence_source=original["source"],
                  canonical_source_stat=original["source_stat"])
    for interval in record["severe_intervals"]:
        interval["contact_pairs"] = _mirror_pairs(interval["contact_pairs"])
    for segment in record["segments"]:
        segment["source"] = relative
        segment["contact_pairs"] = _mirror_pairs(segment["contact_pairs"])
    for pair in record["contact_pairs"]:
        pair["geom1"], pair["geom2"] = sorted((_mirror_geom(pair["geom1"]),
                                                  _mirror_geom(pair["geom2"])))
    record["contact_pairs"].sort(key=lambda item: (item["geom1"], item["geom2"]))
    return record


def _collect_files(dataset: str, bones_dir: Path, hiphi_dir: Path) -> list[tuple[str, str, str]]:
    if dataset == "bones-seed":
        root = bones_dir.resolve()
        return [(dataset, str(path), path.relative_to(root).as_posix())
                for path in sorted(root.rglob("*.csv")) if path.is_file()]
    if dataset == "hiphi":
        root = hiphi_dir.resolve()
        index = root.parent / "filenames.txt"
        if index.is_file():
            result = []
            for line in index.read_text(encoding="utf-8").splitlines():
                relative = line.strip()
                if not relative or not relative.endswith(".npz"):
                    continue
                path = root / relative
                if Path(relative).is_absolute() or ".." in Path(relative).parts:
                    raise ValueError(f"Unsafe HiPhi index path: {relative}")
                result.append((dataset, str(path), relative))
            return result
        return [(dataset, str(path), path.relative_to(root).as_posix())
                for path in sorted(root.rglob("*.npz")) if path.is_file()]
    raise ValueError(f"Unknown dataset: {dataset}")


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "files": len(records), "valid_files": 0, "errors": 0,
        "mirror_inherited_files": 0,
        "files_with_collision": 0, "collision_intervals": 0,
        "files_with_severe_collision": 0, "severe_intervals": 0, "severe_frames": 0,
        "frames": 0, "checked_frames": 0, "collision_frames": 0,
        "kept_frames": 0, "removed_tpose_frames": 0,
        "duration_seconds": 0.0, "kept_seconds": 0.0, "collision_sampled_seconds": 0.0,
    }
    for record in records:
        if record.get("status") != "ok":
            result["errors"] += 1
            continue
        result["valid_files"] += 1
        result["mirror_inherited_files"] += record.get("collision_method") == "mirror_inherited"
        result["files_with_collision"] += bool(record["collision_frames"])
        result["collision_intervals"] += len(record["collision_intervals_frames"])
        result["files_with_severe_collision"] += bool(record["severe_intervals"])
        result["severe_intervals"] += len(record["severe_intervals"])
        result["severe_frames"] += record["severe_frames"]
        result["frames"] += record["frames"]
        result["kept_frames"] += record.get("kept_frames", record["frames"])
        result["removed_tpose_frames"] += record.get("removed_tpose_frames", 0)
        result["checked_frames"] += record["checked_frames"]
        result["collision_frames"] += record["collision_frames"]
        result["duration_seconds"] += record["duration_seconds"]
        result["kept_seconds"] += record.get("kept_frames", record["frames"]) / record["fps"]
        result["collision_sampled_seconds"] += record["collision_frames"] / record["fps"]
    result["collision_file_fraction"] = result["files_with_collision"] / max(1, result["valid_files"])
    result["collision_frame_fraction"] = result["collision_frames"] / max(1, result["checked_frames"])
    result["severe_frame_fraction"] = result["severe_frames"] / max(1, result["checked_frames"])
    result["duration_hours"] = result["duration_seconds"] / 3600.0
    result["collision_sampled_hours"] = result["collision_sampled_seconds"] / 3600.0
    return result


def _write_json(path: Path, value: Any, indent: int | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=indent, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "bones-seed", "hiphi"), default="all")
    parser.add_argument("--bones-dir", type=Path, default=DEFAULT_BONES_DIR)
    parser.add_argument("--hiphi-dir", type=Path, default=DEFAULT_HIPHI_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=max(1, min(8, multiprocessing.cpu_count())))
    parser.add_argument("--executor", choices=("process", "thread"), default="process")
    parser.add_argument("--mirror-policy", choices=("independent", "inherit"), default="independent",
                        help="Inherit HiPhi mirror collisions from originals (approximate; default: independent)")
    parser.add_argument("--tpose-json", type=Path, default=DEFAULT_TPOSE_JSON,
                        help="Scan only HiPhi keep_intervals_frames from this T-pose JSON")
    parser.add_argument("--no-tpose-trim", action="store_true", help="Scan all HiPhi frames")
    parser.add_argument("--stride", type=int, default=1, help="Check every Nth frame (default: 1)")
    parser.add_argument("--limit", type=int, default=0, help="Only scan the first N files after sorting")
    parser.add_argument("--resume", action="store_true", help="Reuse matching records in the checkpoint JSONL")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.workers < 1 or args.stride < 1 or args.limit < 0:
        raise SystemExit("--workers and --stride must be positive; --limit must be nonnegative")
    model = args.model.resolve()
    if not model.is_file():
        raise SystemExit(f"MuJoCo model does not exist: {model}")
    datasets = ("bones-seed", "hiphi") if args.dataset == "all" else (args.dataset,)
    tasks = []
    for name in datasets:
        tasks.extend(_collect_files(name, args.bones_dir, args.hiphi_dir))
    tasks.sort(key=lambda item: (item[0], item[2]))
    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks:
        raise SystemExit("No input motions found")
    trim_by_source = {}
    trim_bytes = None
    if "hiphi" in datasets and not args.no_tpose_trim:
        trim_bytes = args.tpose_json.read_bytes()
        trim_payload = json.loads(trim_bytes)
        trim_by_source = {item["source"]: item for item in trim_payload["motions"]}
        hiphi_sources = {relative for dataset, _, relative in tasks if dataset == "hiphi"}
        if len(trim_by_source) != len(trim_payload["motions"]) or not hiphi_sources <= trim_by_source.keys():
            raise SystemExit("T-pose JSON has duplicate or missing HiPhi sources")
        if any(trim_by_source[source]["status"] != "ok" for source in hiphi_sources):
            raise SystemExit("T-pose JSON contains failed HiPhi records")
    existing_config_path = args.output / "scan_config.json"
    if existing_config_path.is_file():
        with existing_config_path.open(encoding="utf-8") as stream:
            existing_config = json.load(stream)
        if (existing_config.get("mirror_policy", "independent") != args.mirror_policy
                or existing_config.get("tpose_json_sha256") !=
                (hashlib.sha256(trim_bytes).hexdigest() if trim_bytes is not None else None)):
            raise SystemExit("Output contains a different mirror/T-pose policy; choose a separate --output directory")
    args.output.mkdir(parents=True, exist_ok=True)

    config = {
        "schema_version": SCHEMA_VERSION,
        "datasets": list(datasets),
        "bones_dir": str(args.bones_dir.resolve()),
        "hiphi_dir": str(args.hiphi_dir.resolve()),
        "model": str(model),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "stride": args.stride,
        "mirror_policy": args.mirror_policy,
        "tpose_json": str(args.tpose_json.resolve()) if trim_bytes is not None else None,
        "tpose_json_sha256": hashlib.sha256(trim_bytes).hexdigest() if trim_bytes is not None else None,
        "severe_depth_m": SEVERE_DEPTH_M,
        "severe_min_seconds": SEVERE_MIN_SECONDS,
        "scanner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "collision_logic": "mj_kinematics then mj_collision; robot body pairs only; world/same-body contacts excluded",
        "mirror_reuse_is_approximate": args.mirror_policy == "inherit",
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    _write_json(args.output / "scan_config.json", config, indent=2)

    checkpoint_path = args.output / "self_collision_checkpoint.jsonl"
    cached: dict[tuple[str, str], dict[str, Any]] = {}
    if args.resume and checkpoint_path.is_file():
        with checkpoint_path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                    key = (item["dataset"], item["source"])
                    if item.get("config_sha256") == config_hash:
                        cached[key] = item["record"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
    pending = []
    for task in tasks:
        key = (task[0], task[2])
        cached_record = cached.get(key)
        if cached_record is None:
            pending.append(task)
            continue
        try:
            if cached_record.get("source_stat") != _fingerprint(Path(task[1])):
                pending.append(task)
            elif cached_record.get("collision_method") == "mirror_inherited":
                canonical_path = args.hiphi_dir.resolve() / cached_record["evidence_source"]
                if cached_record["canonical_source_stat"] != _fingerprint(canonical_path):
                    pending.append(task)
        except (OSError, KeyError):
            pending.append(task)
    started = time.perf_counter()
    records = dict(cached)
    executor_type = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor
    geometry_scans = 0
    inherited = 0

    def save_record(record: dict[str, Any]) -> None:
        records[(record["dataset"], record["source"])] = record
        with checkpoint_path.open("a", encoding="utf-8") as stream:
            json.dump({"config_sha256": config_hash, "dataset": record["dataset"],
                       "source": record["source"], "record": record},
                      stream, ensure_ascii=False, allow_nan=False)
            stream.write("\n")

    with executor_type(max_workers=args.workers) as executor:
        def scan_tasks(batch: list[tuple[str, str, str]]) -> None:
            nonlocal geometry_scans
            worker_tasks = [(dataset, source, relative, str(model), args.stride,
                             trim_by_source.get(relative) if dataset == "hiphi" else None)
                            for dataset, source, relative in batch]
            for index, record in enumerate(executor.map(_analyze_task, worker_tasks, chunksize=1), 1):
                save_record(record)
                geometry_scans += 1
                if index == 1 or index % 100 == 0 or index == len(worker_tasks):
                    print(f"scanned {index}/{len(worker_tasks)} pending motions", flush=True)

        if args.mirror_policy == "inherit":
            scan_tasks([task for task in pending
                        if task[0] != "hiphi" or _canonical_source(task[2]) is None])
            fallback = []
            for task in pending:
                canonical = _canonical_source(task[2]) if task[0] == "hiphi" else None
                if canonical is None:
                    continue
                original = records.get(("hiphi", canonical))
                try:
                    frames, fps = _read_hiphi_timeline(Path(task[1]))
                    source_stat = _fingerprint(Path(task[1]))
                except (OSError, ValueError, EOFError, zipfile.BadZipFile, KeyError):
                    fallback.append(task)
                    continue
                if original is None or original.get("status") != "ok" or (frames, fps) != (original["frames"], original["fps"]):
                    fallback.append(task)
                    continue
                if trim_bytes is not None and trim_by_source[task[2]]["keep_intervals_frames"] != original["keep_intervals_frames"]:
                    fallback.append(task)
                    continue
                try:
                    if original["source_stat"] != _fingerprint(args.hiphi_dir.resolve() / canonical):
                        fallback.append(task)
                        continue
                except OSError:
                    fallback.append(task)
                    continue
                save_record(_inherit_mirror(original, task, source_stat))
                inherited += 1
            if inherited:
                print(f"inherited {inherited} mirror motions", flush=True)
            scan_tasks(fallback)
        else:
            scan_tasks(pending)

    ordered = [records[(dataset, relative)] for dataset, _, relative in tasks]
    summary = _aggregate(ordered)
    summary["by_dataset"] = {name: _aggregate([record for record in ordered if record["dataset"] == name]) for name in datasets}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "run": {"workers": args.workers, "executor": args.executor, "stride": args.stride,
                "limit": args.limit, "geometry_scans": geometry_scans,
                "mirrors_inherited": inherited, "elapsed_seconds": time.perf_counter() - started},
        "summary": summary,
        "motions": ordered,
    }
    _write_json(args.output / "self_collision.json", payload)
    _write_json(args.output / "severe_intervals.json", {
        "schema_version": SCHEMA_VERSION, "config": config,
        "summary": summary,
        "motions": [
            {"source": record["source"], "fps": record["fps"], "frames": record["frames"],
             "source_stat": record["source_stat"], "severe_intervals": record["severe_intervals"],
             "keep_intervals_frames": record["keep_intervals_frames"],
             "kept_frames": record["kept_frames"],
             "collision_method": record.get("collision_method", "independent"),
             "evidence_source": record.get("evidence_source", record["source"])}
            for record in ordered if record["status"] == "ok" and record["severe_intervals"]
        ],
        "errors": [{"source": record["source"], "error": record["error"]}
                   for record in ordered if record["status"] != "ok"],
    })
    _write_json(args.output / "statistics.json", summary, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
