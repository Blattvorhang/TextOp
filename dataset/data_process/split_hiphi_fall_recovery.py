#!/usr/bin/env python3
"""Split HiPHI fall/recovery recordings into individual recovery clips.

Each reference-height recovery candidate is tracked independently with the
same SONIC/MuJoCo validator used by ``augment_fall_recovery.py``.  Resetting
SONIC for every candidate prevents one fall from contaminating the next.
Final boundaries use the aligned reference and simulated SONIC pelvis
heights: recovery completes only after both robots remain upright.

Every output keeps the HiPHI NPZ schema.  Arrays whose first dimension is the
motion length are sliced; metadata such as ``fps`` and ``body_names`` is
copied unchanged.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "data/hiphi/get_up"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/hiphi/augmented"
MOTION_FILE = "motion_actor.npz"


@dataclass(frozen=True)
class Segment:
    """Half-open frame interval and the first jointly upright frame."""

    start: int
    end: int
    upright: int


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open intervals for all true runs in a 1-D mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError(f"Expected a 1-D mask, got {mask.shape}")
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [(int(start), int(end)) for start, end in edges.reshape(-1, 2)]


def _first_run(
    mask: np.ndarray,
    minimum_length: int,
    start: int,
) -> tuple[int, int] | None:
    for run_start, run_end in _true_runs(mask[start:]):
        run_start += start
        run_end += start
        if run_end - run_start >= minimum_length:
            return run_start, run_end
    return None


def _rise_onset(
    height: np.ndarray,
    start: int,
    end: int,
    *,
    window: int,
    min_rise: float,
) -> int:
    """Return the first sustained upward trend, or ``end`` if unavailable."""
    if end <= start or window < 1:
        return end
    height = np.asarray(height, dtype=np.float64)
    delta = np.diff(height, prepend=height[0])
    for index in range(start, max(start, end - window + 1)):
        values = delta[index + 1:index + window + 1]
        if len(values) < window:
            break
        if (height[index + window] - height[index] >= min_rise
                and np.count_nonzero(values > 0.0) >= 0.6 * window):
            return index
    return end


def find_reference_segments(
    height: np.ndarray,
    *,
    low_height: float,
    upright_height: float,
    low_frames: int,
    trend_frames: int,
    post_rise_frames: int,
    min_frames: int,
) -> list[Segment]:
    """Split reference height into low-to-high action clips.

    A clip starts at the minimum of each sustained low-height interval (or at
    frame zero when the sequence starts low). After the height reaches the
    upright threshold, it ends immediately before the next sustained decline,
    or at the end of the sequence.
    """
    height = np.asarray(height, dtype=np.float64)
    if height.ndim != 1 or not np.isfinite(height).all():
        raise ValueError("height must be a finite 1-D array")
    low_runs = _true_runs(height < low_height)
    segments: list[Segment] = []
    for run_index, (run_start, run_end) in enumerate(low_runs):
        if run_end - run_start < low_frames:
            continue
        start = 0 if run_start == 0 else run_start + int(
            np.argmin(height[run_start:run_end])
        )
        search_end = (
            low_runs[run_index + 1][0]
            if run_index + 1 < len(low_runs) else len(height)
        )
        high = np.flatnonzero(height[run_end:search_end] >= upright_height)
        if high.size == 0:
            continue
        high_start = run_end + int(high[0])
        smooth = np.convolve(
            height, np.ones(trend_frames) / trend_frames, mode="same"
        )
        end = len(height)
        decline_start = high_start + post_rise_frames
        for index in range(decline_start + trend_frames,
                           search_end - trend_frames + 1):
            if (smooth[index + trend_frames - 1] < smooth[index - 1]
                    and np.all(np.diff(smooth[index - trend_frames:index]) < 0)):
                end = index
                break
        if end - start >= min_frames:
            segments.append(Segment(start, end, high_start))
    return segments


def find_recovery_segments(
    reference_height: np.ndarray,
    sonic_height: np.ndarray,
    *,
    low_height: float,
    upright_height: float,
    low_frames: int,
    upright_frames: int,
    stable_frames: int,
    drop_frames: int,
    min_frames: int,
) -> list[Segment]:
    """Find low-root -> jointly-upright intervals in aligned trajectories.

    A segment begins at the first sustained low frame in either trajectory.
    Recovery completes only after both roots remain above ``upright_height``.
    The upright tail continues until either root falls below that threshold
    for ``drop_frames`` frames, or until the recording ends. ``stable_frames``
    is used only to reject a drop immediately after the initial recovery.
    """
    reference_height = np.asarray(reference_height, dtype=np.float64)
    sonic_height = np.asarray(sonic_height, dtype=np.float64)
    if reference_height.ndim != 1 or sonic_height.ndim != 1:
        raise ValueError("Root-height arrays must be one-dimensional")
    if reference_height.shape != sonic_height.shape:
        raise ValueError(
            "Reference and SONIC root-height arrays must have the same shape: "
            f"{reference_height.shape} != {sonic_height.shape}"
        )
    if not (np.isfinite(reference_height).all()
            and np.isfinite(sonic_height).all()):
        raise ValueError("Root-height arrays contain NaN or Inf")

    frames = len(reference_height)
    low = (reference_height < low_height) | (sonic_height < low_height)
    jointly_upright = ((reference_height >= upright_height)
                       & (sonic_height >= upright_height))
    dropped = ~jointly_upright
    segments: list[Segment] = []
    cursor = 0

    while cursor < frames:
        low_run = _first_run(low, low_frames, cursor)
        if low_run is None:
            break
        segment_start = low_run[0]

        upright_run = _first_run(
            jointly_upright, upright_frames, low_run[1]
        )
        if upright_run is None:
            break  # An unfinished recovery is not useful training data.
        upright_start = upright_run[0]

        # A drop before the initial upright hold means this is not a recovery.
        drop_run = _first_run(dropped, drop_frames, upright_run[1])
        if drop_run is not None and drop_run[0] < upright_run[1] + stable_frames:
            cursor = drop_run[1]
            continue
        segment_end = drop_run[0] if drop_run is not None else frames
        if segment_end - segment_start >= min_frames:
            segments.append(Segment(segment_start, segment_end, upright_start))

        # Reconsider the detected drop as the possible start of the next fall.
        cursor = segment_end if segment_end > segment_start else segment_start + 1

    return segments


def _load_motion(path: Path) -> tuple[dict[str, np.ndarray], int, float]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    required = {"fps", "joint_pos", "body_pos_w"}
    missing = sorted(required - arrays.keys())
    if missing:
        raise ValueError(f"{path}: missing required fields: {missing}")
    fps_array = np.asarray(arrays["fps"])
    if fps_array.size != 1:
        raise ValueError(f"{path}: fps must contain one value, got {fps_array.shape}")
    fps = float(fps_array.item())
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{path}: invalid fps {fps}")
    frames = int(np.asarray(arrays["joint_pos"]).shape[0])
    body_pos = np.asarray(arrays["body_pos_w"])
    if body_pos.ndim != 3 or body_pos.shape[0] != frames or body_pos.shape[2] != 3:
        raise ValueError(
            f"{path}: body_pos_w must have shape (T, N, 3), got {body_pos.shape}"
        )
    return arrays, frames, fps


def _slice_motion(
    arrays: dict[str, np.ndarray],
    frames: int,
    start: int,
    end: int,
) -> dict[str, np.ndarray]:
    result = {}
    for key, value in arrays.items():
        if value.ndim > 0 and value.shape[0] == frames and key != "body_names":
            result[key] = np.ascontiguousarray(value[start:end])
        else:
            result[key] = value
    return result


def _slice_entry(entry: dict, start: int, end: int) -> dict:
    """Slice a packed motion entry for an independent SONIC rollout."""
    frames = int(np.asarray(entry["dof"]).shape[0])
    result = {}
    for key, value in entry.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 \
                and value.shape[0] == frames:
            result[key] = np.ascontiguousarray(value[start:end])
        else:
            result[key] = value
    if "motion_len" in result:
        result["motion_len"] = end - start
    return result


def _save_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _source_files(input_dir: Path, output_dir: Path) -> list[Path]:
    sources = []
    for path in sorted(input_dir.rglob(MOTION_FILE)):
        try:
            path.resolve().relative_to(output_dir.resolve())
        except ValueError:
            sources.append(path)
    return sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split HiPHI recordings into low-root-to-upright clips."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--low-height", type=float, default=0.35,
                        help="Pelvis height that starts a fallen interval (m)")
    parser.add_argument("--upright-height", type=float, default=0.68,
                        help="Pelvis height considered upright (m)")
    parser.add_argument("--min-low-seconds", type=float, default=0.20)
    parser.add_argument("--trend-seconds", type=float, default=0.20,
                        help="Sustained downward trend needed to end a clip")
    parser.add_argument("--post-rise-seconds", type=float, default=3.0,
                        help="Minimum time retained after reaching upright height")
    parser.add_argument("--min-clip-seconds", type=float, default=0.75)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.low_height >= args.upright_height:
        parser.error("--low-height must be lower than --upright-height")
    for name in ("min_low_seconds", "trend_seconds", "post_rise_seconds",
                 "min_clip_seconds"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    sources = _source_files(input_dir, output_dir)
    if not sources:
        raise FileNotFoundError(f"No {MOTION_FILE} files found under {input_dir}")

    total_segments = 0
    written = 0
    skipped = 0
    for source in sources:
        arrays, frames, fps = _load_motion(source)
        reference_height = np.asarray(arrays["body_pos_w"])[:, 0, 2]
        to_frames = lambda seconds: max(1, int(round(seconds * fps)))
        segments = find_reference_segments(
            reference_height,
            low_height=args.low_height,
            upright_height=args.upright_height,
            low_frames=to_frames(args.min_low_seconds),
            trend_frames=to_frames(args.trend_seconds),
            post_rise_frames=to_frames(args.post_rise_seconds),
            min_frames=to_frames(args.min_clip_seconds),
        )
        total_segments += len(segments)
        print(
            f"{source.parent.name}: {frames} frames -> {len(segments)} clips"
        )
        for index, segment in enumerate(segments, start=1):
            clip_name = f"{source.parent.name}_{index:04d}"
            destination = output_dir / clip_name / MOTION_FILE
            duration = (segment.end - segment.start) / fps
            print(
                f"  {clip_name}: [{segment.start}:{segment.end}) "
                f"upright={segment.upright}, {duration:.2f}s"
            )
            if args.dry_run:
                continue
            if destination.exists() and not args.overwrite:
                print(f"    skip existing: {destination}")
                skipped += 1
                continue
            clip = _slice_motion(
                arrays, frames, segment.start, segment.end
            )
            _save_npz_atomic(destination, clip)
            written += 1

    mode = "Dry run" if args.dry_run else "Done"
    print(
        f"{mode}: {len(sources)} sources, {total_segments} clips, "
        f"{written} written, {skipped} skipped"
    )


if __name__ == "__main__":
    main()
