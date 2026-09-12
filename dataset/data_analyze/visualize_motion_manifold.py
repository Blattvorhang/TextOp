#!/usr/bin/env python3
"""Embed G1 motion poses in a 3-D UMAP manifold and plot one trajectory.

The embedded state is exactly ``[dof(29), root_z, root_pitch, root_roll]``.
Root X/Y/Yaw are deliberately omitted.  Input can be a Bones-Seed CSV or a
TextOp/motion-lib pickle containing ``dof``, ``root_trans_offset`` and
``root_rot`` arrays.

Examples
--------
python dataset/data_analyze/visualize_motion_manifold.py \
    --input /home/lenovo/data/bones-seed/g1/csv --motion walk \
    --output dataset/data_analyze/analysis_output/walk_manifold.png \
    --html dataset/data_analyze/analysis_output/walk_manifold.html

python dataset/data_analyze/visualize_motion_manifold.py \
    --input data/motion_lib/221010/walk_ff_loop_180_R_003__A045_M.pkl \
    --show

# Several motions in ONE shared UMAP embedding:
python dataset/data_analyze/visualize_motion_manifold.py \
    --input /home/lenovo/data/bones-seed/g1/csv --motion walk --num-motions 10 \
    --output dataset/data_analyze/analysis_output/walk_manifold_multi.png \
    --html dataset/data_analyze/analysis_output/walk_manifold_multi.html
"""

from __future__ import annotations

import argparse
import csv
import pickle
from itertools import islice
from pathlib import Path
from typing import Any, Iterator
import xml.etree.ElementTree as ET

import numpy as np


DOF_COLUMNS = [
    "left_hip_pitch_joint_dof", "left_hip_roll_joint_dof", "left_hip_yaw_joint_dof",
    "left_knee_joint_dof", "left_ankle_pitch_joint_dof", "left_ankle_roll_joint_dof",
    "right_hip_pitch_joint_dof", "right_hip_roll_joint_dof", "right_hip_yaw_joint_dof",
    "right_knee_joint_dof", "right_ankle_pitch_joint_dof", "right_ankle_roll_joint_dof",
    "waist_yaw_joint_dof", "waist_roll_joint_dof", "waist_pitch_joint_dof",
    "left_shoulder_pitch_joint_dof", "left_shoulder_roll_joint_dof", "left_shoulder_yaw_joint_dof",
    "left_elbow_joint_dof", "left_wrist_roll_joint_dof", "left_wrist_pitch_joint_dof",
    "left_wrist_yaw_joint_dof", "right_shoulder_pitch_joint_dof", "right_shoulder_roll_joint_dof",
    "right_shoulder_yaw_joint_dof", "right_elbow_joint_dof", "right_wrist_roll_joint_dof",
    "right_wrist_pitch_joint_dof", "right_wrist_yaw_joint_dof",
]
DEFAULT_LIMITS_XML = Path(__file__).resolve().parents[2] / "TextOpRobotMDAR/description/robots/g1/g1_29dof.xml"
# Categorical colors for distinct motions, assigned in this fixed order (light surface).
CURVE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def load_joint_limits(xml_path: Path = DEFAULT_LIMITS_XML) -> tuple[np.ndarray, np.ndarray]:
    """Read the 29 MuJoCo joint ranges in the same order as ``DOF_COLUMNS``."""
    root = ET.parse(xml_path).getroot()
    ranges = {joint.attrib["name"] + "_dof": joint.attrib.get("range")
              for joint in root.iter("joint") if "range" in joint.attrib}
    missing = [name for name in DOF_COLUMNS if name not in ranges]
    if missing:
        raise ValueError(f"Joint-limit XML is missing {len(missing)} joints: {missing}")
    bounds = np.array([[float(x) for x in ranges[name].split()] for name in DOF_COLUMNS])
    return bounds[:, 0], bounds[:, 1]


def normalize_dof(dof: Any, xml_path: Path = DEFAULT_LIMITS_XML) -> np.ndarray:
    """Map each joint independently to [0, 1] using its XML range."""
    lower, upper = load_joint_limits(xml_path)
    dof = np.asarray(dof, dtype=float)
    return np.clip((dof - lower) / (upper - lower), 0.0, 1.0)


def _euler_xyz_from_xyzw(q: np.ndarray) -> np.ndarray:
    """Return XYZ roll, pitch, yaw from xyzw quaternions."""
    q = np.asarray(q, dtype=float)
    x, y, z, w = np.moveaxis(q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12), -1, 0)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack((roll, pitch, yaw), axis=-1)


def features_from_arrays(dof: Any, root_z: Any, root_euler_xyz: Any, *, degrees: bool = False,
                         limits_xml: Path = DEFAULT_LIMITS_XML) -> np.ndarray:
    """Build the 32D manifold state, validating frame alignment."""
    dof = np.asarray(dof, dtype=float)
    root_z = np.asarray(root_z, dtype=float).reshape(-1)
    euler = np.asarray(root_euler_xyz, dtype=float)
    if dof.ndim != 2 or dof.shape[1] != 29:
        raise ValueError(f"Expected dof with shape [T, 29], got {dof.shape}")
    if euler.shape != (len(dof), 3) or root_z.shape != (len(dof),):
        raise ValueError("root arrays must have the same number of frames as dof")
    if degrees:
        euler = np.deg2rad(euler)
    # Columns are normalized dof, z, pitch, roll; yaw never enters the embedding.
    return np.concatenate((normalize_dof(dof, limits_xml), root_z[:, None], euler[:, 1:2], euler[:, 0:1]), axis=1)


def load_csv(path: Path, limits_xml: Path = DEFAULT_LIMITS_XML) -> np.ndarray:
    with path.open(newline="", encoding="utf-8", errors="replace") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV has no frames: {path}")
    try:
        dof = np.array([[float(row[c]) for c in DOF_COLUMNS] for row in rows])
        z = np.array([float(row["root_translateZ"]) for row in rows]) / 100.0
        euler = np.array([[float(row[f"root_rotate{axis}"]) for axis in "XYZ"] for row in rows])
    except KeyError as exc:
        raise ValueError(f"Missing expected Bones-Seed CSV column: {exc}") from exc
    return features_from_arrays(dof, z, euler, degrees=True, limits_xml=limits_xml)


def _iter_motions(value: Any, path: str = "") -> Iterator[tuple[dict, str]]:
    """Yield every motion dict (dof/root_trans_offset/root_rot) with its key path."""
    if isinstance(value, dict) and {"dof", "root_trans_offset", "root_rot"} <= set(value):
        yield value, path or "<root>"
        return
    if isinstance(value, dict):
        motion = value.get("motion")
        if isinstance(motion, dict):
            yield from _iter_motions(motion, f"{path}.motion" if path else "motion")
        for name, item in value.items():
            if item is motion:
                continue
            yield from _iter_motions(item, f"{path}.{name}" if path else str(name))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_motions(item, f"{path}[{index}]" if path else f"[{index}]")


def _pickle_motions(path: Path, motion_name: str, max_motions: int,
                    limits_xml: Path = DEFAULT_LIMITS_XML) -> tuple[list[str], list[np.ndarray]]:
    """Load up to ``max_motions`` motions whose pickle key path contains ``motion_name``."""
    try:
        with path.open("rb") as stream:
            data = pickle.load(stream)
    except Exception as exc:
        try:
            import joblib
            data = joblib.load(path)
        except ImportError:
            raise RuntimeError("Install joblib to read this pickle") from exc
    found = [(item, found_path) for item, found_path in _iter_motions(data)
             if motion_name.lower() in found_path.lower()]
    total = len(found)
    if not found:
        found = list(islice(_iter_motions(data), 1))
        if found:
            print(f"Warning: no motion key containing '{motion_name}'; using the first motion found")
    if not found:
        raise ValueError(f"Could not find a motion containing '{motion_name}' in {path}")
    found = found[:max_motions]
    if len(found) == 1:
        item, found_path = found[0]
        print(f"Using motion '{motion_name}' found at pickle key path '{found_path}' "
              f"({len(item['dof'])} frames)")
    else:
        print(f"Using {len(found)} of {total} motions matching '{motion_name}' in {path}:")
        for item, found_path in found:
            print(f"  {found_path} ({len(item['dof'])} frames)")
    names, features = [], []
    for item, found_path in found:
        root = np.asarray(item["root_trans_offset"])
        rot = np.asarray(item["root_rot"])
        names.append(found_path)
        features.append(features_from_arrays(item["dof"], root[:, 2],
                                             _euler_xyz_from_xyzw(rot), limits_xml=limits_xml))
    return names, features


def load_inputs(path: Path, motion_name: str, num_motions: int,
                limits_xml: Path = DEFAULT_LIMITS_XML) -> tuple[list[str], list[np.ndarray]]:
    """Load up to ``num_motions`` motions; return their display names and feature arrays."""
    if path.is_dir():
        matches = sorted(p for p in path.rglob("*.csv") if motion_name.lower() in p.stem.lower())
        if not matches:
            raise FileNotFoundError(f"No CSV matching '{motion_name}' under {path}")
        total = len(matches)
        matches = matches[:num_motions]
        if total > len(matches):
            print(f"Found {total} CSVs matching '{motion_name}' under {path}; "
                  f"using the first {len(matches)}:")
        else:
            print(f"Using {len(matches)} CSV(s) matching '{motion_name}':")
        for match in matches:
            print(f"  {match}")
        return [match.stem for match in matches], [load_csv(match, limits_xml) for match in matches]
    if path.suffix.lower() == ".csv":
        return [path.stem], [load_csv(path, limits_xml)]
    return _pickle_motions(path, motion_name, num_motions, limits_xml)


def _time_features(lengths: list[int], temporal_weight: float, cyclic_time: bool) -> np.ndarray:
    """Per-motion temporal features for UMAP's neighbor graph: one (linear) or
    two (cyclic) columns per motion, arranged block-diagonally.

    Each motion's local frame index restarts at 0 in its own column(s), so
    different motions are never aligned by time: frame t of motion A and frame t
    of motion B live in orthogonal time blocks and only come close if their POSE
    is close. Only frames of the same motion share a time axis — that is what
    chains them into a continuous trajectory.
    """
    width = 2 if cyclic_time else 1
    block = np.zeros((sum(lengths), width * len(lengths)), dtype=float)
    start = 0
    for index, length in enumerate(lengths):
        rows = slice(start, start + length)
        if cyclic_time:
            angle = 2 * np.pi * np.arange(length, dtype=float) / length
            radius = temporal_weight * length / (2 * np.pi)
            block[rows, 2 * index] = radius * np.sin(angle)
            block[rows, 2 * index + 1] = radius * np.cos(angle)
        else:
            block[rows, index] = np.arange(length, dtype=float) * temporal_weight
        start += length
    return block


def embed(features: np.ndarray, *, neighbors: int, seed: int, temporal_weight: float = 0.0,
          lengths: list[int] | None = None, cyclic_time: bool = False) -> np.ndarray:
    """Compute the 3-D UMAP embedding from the 32D pose features.

    By default the embedding is pose-only (``temporal_weight=0``): adjacent
    frames of a motion are already each other's closest poses, so pose
    similarity alone chains a trajectory, and similar poses stay close no
    matter how far apart in time — a stop motion's standing start and end land
    on the same point, symmetric gait phases overlap, and motions with similar
    poses come near each other.

    ``temporal_weight > 0`` adds an optional continuity prior to UMAP's
    neighbor-graph input only (never to the plotted state): each motion gets
    its own time column(s), block-diagonal and restarting at 0 per motion,
    which guarantees consecutive frames are neighbors even where pose proximity
    is weak (fast segments, sparse data). The cost is that pose-similar but
    time-distant frames are pushed apart. With ``cyclic_time`` (weight > 0),
    local time is a circle (sin/cos pair) so periodic motions close start->end.

    ``lengths`` gives the frame count of each motion when ``features`` is the
    concatenation of several motions.
    """
    try:
        from umap import UMAP
    except ImportError as exc:
        raise RuntimeError("UMAP requires umap-learn; install with `pip install umap-learn matplotlib`") from exc
    if len(features) < 3:
        raise ValueError("Need at least 3 frames for a 3-D embedding")
    pose = features.copy()
    pose[:, 29:] = (pose[:, 29:] - pose[:, 29:].mean(axis=0)) / np.where(
        pose[:, 29:].std(axis=0) > 1e-8, pose[:, 29:].std(axis=0), 1.0)
    if lengths is None:
        lengths = [len(pose)]
    if temporal_weight > 0:
        frame_axis = _time_features(lengths, temporal_weight, cyclic_time)
        normalized = np.concatenate((pose, frame_axis), axis=1)
    else:
        normalized = pose
    reducer = UMAP(
        n_components=3,
        n_neighbors=min(neighbors, len(features) - 1),
        metric="euclidean",
        random_state=seed,
    )
    return reducer.fit_transform(normalized)


def print_closure_stats(names: list[str], curves: list[np.ndarray],
                        features_list: list[np.ndarray]) -> None:
    """Report how close each trajectory's start and end are.

    Cyclic motions (walk, jog) and stop-marked data should close: the last pose
    matches the first. ``embed-closure`` is measured in the shared embedding,
    ``dof-closure`` on the normalized 29 joint values; both are small for a loop.
    """
    print("Trajectory closure (start <-> end):")
    for name, curve, features in zip(names, curves, features_list):
        steps = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        median_step = float(np.median(steps)) if len(steps) else np.nan
        embed_closure = float(np.linalg.norm(curve[-1] - curve[0]))
        ratio = embed_closure / median_step if median_step > 0 else float("inf")
        dof_closure = float(np.linalg.norm(features[-1, :29] - features[0, :29]))
        flag = "  [stop]" if "stop" in name.lower() and "no_stop" not in name.lower() else ""
        print(f"  {name[:32]:32s} frames={len(curve):5d}  embed-closure={embed_closure:8.4f}  "
              f"xmedian-step={ratio:7.1f}  dof-closure={dof_closure:7.3f}{flag}")


def plot_static(embedding: np.ndarray, output: Path, *, title: str, show: bool = False,
                curves: list[np.ndarray] | None = None, names: list[str] | None = None) -> None:
    """Save the trajectory as a static PNG (+ .npy); optionally open a live window.

    In the live window, drag with the left mouse button to rotate the view.
    Pass ``curves``/``names`` to overlay several motions from one shared embedding.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    multi = curves is not None
    fig = plt.figure(figsize=(12, 9) if multi else (10, 8))
    ax = fig.add_subplot(111, projection="3d")
    if not multi:
        time = np.arange(len(embedding))
        trajectory_name = names[0][:28] if names else "trajectory"
        ax.plot(*embedding.T, color=CURVE_COLORS[0], linewidth=1.4, alpha=0.9, label=trajectory_name)
        points = ax.scatter(*embedding.T, c=time, cmap="viridis", s=12, linewidths=0)
        ax.scatter(*embedding[0], c="lime", s=70, marker="o", label="start", edgecolors="black")
        ax.scatter(*embedding[-1], c="red", s=70, marker="X", label="end", edgecolors="black")
        ax.plot(*embedding[[0, -1]].T, color="0.35", linestyle="--", linewidth=1.0, alpha=0.8)
        colorbar_label = "frame"
    else:
        ax.scatter(*embedding.T, s=2, c="0.55", alpha=0.12, depthshade=False, label="all poses")
        points = None
        max_frames = max(len(curve) for curve in curves)
        for index, curve in enumerate(curves):
            color = CURVE_COLORS[index % len(CURVE_COLORS)]
            local_time = np.arange(len(curve))
            scatter = ax.scatter(*curve.T, c=local_time, cmap="viridis", s=12, linewidths=0,
                                 vmin=0, vmax=max_frames - 1)
            if points is None:
                points = scatter
            ax.plot(*curve.T, color=color, linewidth=1.7, label=names[index][:28])
            ax.plot(*curve[[0, -1]].T, color=color, linestyle="--", linewidth=0.9, alpha=0.7)
            ax.scatter(*curve[0], c=color, s=60, marker="o", edgecolors="black")
            ax.scatter(*curve[-1], c=color, s=60, marker="X", edgecolors="black")
        if len(curves) > len(CURVE_COLORS):
            print(f"Warning: {len(curves)} curves share {len(CURVE_COLORS)} colors; "
                  f"reduce --num-motions to keep identities unambiguous")
        colorbar_label = "local frame"
    ax.set(xlabel="UMAP-1", ylabel="UMAP-2", zlabel="UMAP-3", title=title)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="0.35", linestyle="--", label="start↔end closure"))
    ax.legend(handles=handles, fontsize=7 if multi else None, loc="best")
    fig.colorbar(points, ax=ax, pad=0.1, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    if show:
        try:
            plt.show()
        except Exception as exc:  # headless backend
            print(f"Interactive window unavailable ({exc}); open {output} instead")
    plt.close(fig)
    np.save(output.with_suffix(".npy"), embedding)
    print(f"Saved {len(embedding)} frames to {output} (embedding: {output.with_suffix('.npy')})")


def plot_html(embedding: np.ndarray, output: Path, *, title: str,
              curves: list[np.ndarray] | None = None, names: list[str] | None = None) -> None:
    """Save an interactive plotly HTML: drag to rotate, scroll to zoom, hover for the frame index.

    Pass ``curves``/``names`` to overlay several motions from one shared embedding;
    click legend entries to toggle motions on and off.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("Interactive HTML requires plotly; install with `pip install plotly`") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    multi = curves is not None
    fig = go.Figure()
    if not multi:
        time = np.arange(len(embedding))
        trajectory_name = names[0][:28] if names else "trajectory"
        fig.add_trace(go.Scatter3d(
            x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2],
            mode="lines",
            line=dict(color=CURVE_COLORS[0], width=3),
            name=trajectory_name,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter3d(
            x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2],
            mode="markers",
            marker=dict(size=3, color=time, colorscale="Viridis", showscale=True,
                        colorbar=dict(title="frame")),
            customdata=time,
            hovertemplate="frame %{customdata}<extra></extra>",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=embedding[:1, 0], y=embedding[:1, 1], z=embedding[:1, 2],
            mode="markers",
            marker=dict(size=7, color="lime", symbol="circle",
                        line=dict(color="black", width=1)),
            name="start", hovertemplate="frame 0 (start)<extra></extra>",
        ))
        fig.add_trace(go.Scatter3d(
            x=embedding[-1:, 0], y=embedding[-1:, 1], z=embedding[-1:, 2],
            mode="markers",
            marker=dict(size=7, color="red", symbol="x",
                        line=dict(color="black", width=1)),
            name="end", hovertemplate=f"frame {len(embedding) - 1} (end)<extra></extra>",
        ))
        closure = float(np.linalg.norm(embedding[-1] - embedding[0]))
        fig.add_trace(go.Scatter3d(
            x=embedding[[0, -1], 0], y=embedding[[0, -1], 1], z=embedding[[0, -1], 2],
            mode="lines",
            line=dict(color="rgba(60,60,60,0.7)", width=2, dash="dash"),
            name="start↔end closure",
            hovertemplate=f"closure distance: {closure:.3f}<extra></extra>",
        ))
    else:
        fig.add_trace(go.Scatter3d(
            x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2],
            mode="markers",
            marker=dict(size=1.5, color="rgba(130,130,130,0.35)"),
            hoverinfo="skip", showlegend=False,
        ))
        max_frames = max(len(curve) for curve in curves)
        for index, curve in enumerate(curves):
            color = CURVE_COLORS[index % len(CURVE_COLORS)]
            name = names[index][:28]
            local_time = np.arange(len(curve))
            # Line and markers as separate traces: an array-valued marker color
            # must not be able to bleed into the line color (known plotly quirk
            # where lines render white when combined with a colorscale array).
            fig.add_trace(go.Scatter3d(
                x=curve[:, 0], y=curve[:, 1], z=curve[:, 2],
                mode="lines",
                line=dict(color=color, width=3),
                name=name, legendgroup=name,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter3d(
                x=curve[:, 0], y=curve[:, 1], z=curve[:, 2],
                mode="markers",
                marker=dict(size=3, color=local_time, colorscale="Viridis",
                            cmin=0, cmax=max_frames - 1,
                            colorbar=dict(title="local frame"), showscale=(index == 0)),
                name=name, legendgroup=name, showlegend=False, customdata=local_time,
                hovertemplate="frame %{customdata}<extra>%{fullData.name}</extra>",
            ))
            fig.add_trace(go.Scatter3d(
                x=curve[:1, 0], y=curve[:1, 1], z=curve[:1, 2],
                mode="markers",
                marker=dict(size=6, color=color, symbol="circle",
                            line=dict(color="black", width=1)),
                showlegend=False, hovertemplate=f"{name} — start<extra></extra>",
            ))
            fig.add_trace(go.Scatter3d(
                x=curve[-1:, 0], y=curve[-1:, 1], z=curve[-1:, 2],
                mode="markers",
                marker=dict(size=6, color=color, symbol="x",
                            line=dict(color="black", width=1)),
                showlegend=False, hovertemplate=f"{name} — end<extra></extra>",
            ))
            closure = float(np.linalg.norm(curve[-1] - curve[0]))
            fig.add_trace(go.Scatter3d(
                x=curve[[0, -1], 0], y=curve[[0, -1], 1], z=curve[[0, -1], 2],
                mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                showlegend=(index == 0),
                name="start↔end closure" if index == 0 else None,
                hovertemplate=f"closure distance: {closure:.3f}<extra>{name}</extra>",
            ))
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="UMAP-1", yaxis_title="UMAP-2", zaxis_title="UMAP-3",
                   aspectmode="data"),
        margin=dict(l=0, r=0, t=45, b=0),
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.6)"),
    )
    fig.write_html(output, include_plotlyjs=True)
    print(f"Saved interactive plot with {len(embedding)} frames to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="CSV, motion-lib pickle, or directory of CSVs")
    parser.add_argument("--motion", default="walk", help="Substring used to select a motion (default: walk)")
    parser.add_argument("--output", type=Path, default=Path("walk_manifold.png"))
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--temporal-weight", type=float, default=0.0,
                        help="Continuity prior strength for the neighbor graph "
                             "(0 = pure pose embedding, default; > 0 adds per-motion time columns)")
    parser.add_argument("--limits-xml", type=Path, default=DEFAULT_LIMITS_XML)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-motions", type=int, default=1,
                        help="Number of motions to load and show together in one shared embedding (default: 1)")
    parser.add_argument("--cyclic-time", action="store_true",
                        help="With --temporal-weight > 0, encode local time on a circle so periodic "
                             "motions (walk/jog/stop) close start->end into a loop in the embedding")
    parser.add_argument("--html", type=Path, default=None,
                        help="Also save an interactive plotly HTML (drag to rotate, zoom, hover for frame)")
    parser.add_argument("--show", action="store_true",
                        help="Open a live matplotlib window (drag with the mouse to rotate)")
    args = parser.parse_args()
    if args.cyclic_time and args.temporal_weight <= 0:
        print("Note: --cyclic-time needs --temporal-weight > 0; embedding is pose-only")
    names, features_list = load_inputs(args.input, args.motion, args.num_motions, args.limits_xml)
    lengths = [len(features) for features in features_list]
    all_features = np.concatenate(features_list, axis=0)
    title = f"G1 motion manifold: {args.motion}"
    if len(features_list) > 1:
        title += f" ({len(features_list)} motions)"
    embedding = embed(all_features, neighbors=args.neighbors, seed=args.seed,
                      temporal_weight=args.temporal_weight, lengths=lengths,
                      cyclic_time=args.cyclic_time)
    slices, start = [], 0
    for length in lengths:
        slices.append(embedding[start:start + length])
        start += length
    print_closure_stats(names, slices, features_list)
    curves = slices if len(slices) > 1 else None
    plot_static(embedding, args.output, title=title, show=args.show, curves=curves, names=names)
    if args.html is not None:
        plot_html(embedding, args.html, title=title, curves=curves, names=names)


if __name__ == "__main__":
    main()
