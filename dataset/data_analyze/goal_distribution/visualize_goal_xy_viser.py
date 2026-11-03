#!/usr/bin/env python3
"""Interactively visualize cached ego-XY trajectories with the 29-DoF G1."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import viser
sys.path.insert(0, str(Path(__file__).resolve().parent))

if __package__:
    from .visualize_end_effector_xyz_viser import DEFAULT_MJCF, add_g1_meshes
else:
    from visualize_end_effector_xyz_viser import DEFAULT_MJCF, add_g1_meshes


FUTURE_LEN = 64
ACTION_COLORS = {
    "walking": (0, 119, 255),  # #0077FF
    "jogging": (254, 114, 53),  # #FE7235
    "other": (144, 194, 115),  # #90C273
}
TRAJECTORY_ALPHA = {"walking": 0.6, "jogging": 0.6, "other": 0.3}
DEFAULT_TRAJECTORY_ALPHA = 0.6  # categories not listed in TRAJECTORY_ALPHA
GUIDE_COLOR = (128, 128, 128)
GUIDE_Z = -0.002
FALLBACK_COLORS = (
    (66, 133, 244),
    (52, 168, 83),
    (251, 188, 4),
    (234, 67, 53),
)
DEFAULT_TRAJECTORIES = Path(__file__).resolve().parents[1] / "analysis_output/hiphi_goal_xy.npz"


def load_trajectories(path):
    """Load and validate cached [N, 64, 2] ego-frame trajectories."""
    with np.load(path, allow_pickle=False) as data:
        required = {"trajectories", "categories", "coordinate_frame", "future_len"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        trajectories = np.asarray(data["trajectories"], dtype=np.float32)
        categories = np.asarray(data["categories"]).astype(str)
        coordinate_frame = str(data["coordinate_frame"])
        future_len = int(data["future_len"])
    if len(trajectories) == 0:
        raise ValueError("Trajectory cache is empty")
    if trajectories.ndim != 3 or trajectories.shape[1:] != (FUTURE_LEN, 2):
        raise ValueError(
            f"Expected trajectories shaped [N, {FUTURE_LEN}, 2], "
            f"got {trajectories.shape}"
        )
    if categories.shape != (len(trajectories),):
        raise ValueError(
            f"Expected one category per trajectory, got {categories.shape}"
        )
    if coordinate_frame != "ego_xy_forward_left" or future_len != FUTURE_LEN:
        raise ValueError(
            f"Unsupported trajectory metadata: frame={coordinate_frame!r}, "
            f"future_len={future_len}"
        )
    if not np.isfinite(trajectories).all():
        raise ValueError("Trajectories contain non-finite coordinates")
    return trajectories, categories


def blend_on_white(color, alpha):
    """Match an RGBA line's appearance on the viewer's white background."""
    rgb = np.asarray(color, dtype=np.float32)
    return tuple(np.rint(alpha * rgb + (1.0 - alpha) * 255.0).astype(np.uint8))


def dashed_circle_segments(radius, dash_count=64, duty_cycle=0.55):
    """Return separated line segments forming a dashed circle at z=0."""
    starts = np.linspace(0.0, 2.0 * np.pi, dash_count, endpoint=False)
    ends = starts + (2.0 * np.pi / dash_count) * duty_cycle
    points = np.stack(
        (
            np.stack((radius * np.cos(starts), radius * np.sin(starts)), axis=-1),
            np.stack((radius * np.cos(ends), radius * np.sin(ends)), axis=-1),
        ),
        axis=1,
    )
    return np.pad(points, ((0, 0), (0, 0), (0, 1)), constant_values=GUIDE_Z)


def add_plot_guides(server, limit):
    """Add thin ego X/Y axes and dashed 1 m / 1.5 m / 2 m radius circles."""
    axes = {
        "x_axis": ((-limit, 0.0, GUIDE_Z), (limit, 0.0, GUIDE_Z)),
        "y_axis": ((0.0, -limit, GUIDE_Z), (0.0, limit, GUIDE_Z)),
    }
    for name, endpoints in axes.items():
        server.scene.add_line_segments(
            f"/guides/{name}",
            points=np.asarray([endpoints], dtype=np.float32),
            colors=GUIDE_COLOR,
            thickness=0.5,
            thickness_units="screen",
        )
    circle_color = blend_on_white(GUIDE_COLOR, 0.65)
    for radius in (1.0, 1.5, 2.0):
        server.scene.add_line_segments(
            f"/guides/radius_{radius:g}m",
            points=dashed_circle_segments(radius),
            colors=circle_color,
            thickness=2.0,
            thickness_units="screen",
        )


def trajectories_to_segments(trajectories):
    """Convert [N, 64, 2] paths to independent z=0 Viser line segments."""
    origins = np.zeros((len(trajectories), 1, 2), dtype=np.float32)
    xy = np.concatenate((origins, trajectories), axis=1)
    xyz = np.pad(xy, ((0, 0), (0, 0), (0, 1)))
    return np.stack((xyz[:, :-1], xyz[:, 1:]), axis=2).reshape(-1, 2, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJECTORIES)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--density", type=int, default=25,
                        help="Initial trajectory density percentage for categories "
                             "without a dedicated flag (default: 25)")
    parser.add_argument("--line-width", type=float, default=1.5,
                        help="Trajectory width in screen pixels (default: 1.5)")
    parser.add_argument(
        "--walking-density", type=int, default=None, metavar="PERCENT",
        help="Initial walking trajectory density (default: 25)",
    )
    parser.add_argument(
        "--jogging-density", type=int, default=None, metavar="PERCENT",
        help="Initial jogging trajectory density (default: 25)",
    )
    parser.add_argument(
        "--other-density", type=int, default=None, metavar="PERCENT",
        help="Initial other trajectory density (default: 25)",
    )
    parser.add_argument(
        "--walking-opacity", type=int, default=None, metavar="PERCENT",
        help="Initial walking trajectory opacity (default: 60)",
    )
    parser.add_argument(
        "--jogging-opacity", type=int, default=None, metavar="PERCENT",
        help="Initial jogging trajectory opacity (default: 60)",
    )
    parser.add_argument(
        "--other-opacity", type=int, default=None, metavar="PERCENT",
        help="Initial other trajectory opacity (default: 30)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.density <= 100:
        raise ValueError(f"--density must be in [1, 100], got {args.density}")
    if args.line_width <= 0:
        raise ValueError(f"--line-width must be positive, got {args.line_width}")
    opacity_args = {
        "walking": args.walking_opacity,
        "jogging": args.jogging_opacity,
        "other": args.other_opacity,
    }
    for category, value in opacity_args.items():
        if value is not None and not 0 <= value <= 100:
            raise ValueError(f"--{category}-opacity must be in [0, 100], got {value}")
    density_args = {
        "walking": args.walking_density,
        "jogging": args.jogging_density,
        "other": args.other_density,
    }
    for category, value in density_args.items():
        if value is not None and not 1 <= value <= 100:
            raise ValueError(f"--{category}-density must be in [1, 100], got {value}")

    trajectories, categories = load_trajectories(args.trajectories)
    unique_categories = sorted(set(categories.tolist()))
    rng = np.random.default_rng(args.seed)
    category_indices = {
        category: indices[rng.permutation(len(indices))]
        for category in unique_categories
        for indices in [np.flatnonzero(categories == category)]
    }
    base_colors = {
        category: ACTION_COLORS.get(
            category, FALLBACK_COLORS[index % len(FALLBACK_COLORS)]
        )
        for index, category in enumerate(unique_categories)
    }
    initial_opacities = {
        category: (
            opacity_args[category]
            if category in opacity_args and opacity_args[category] is not None
            else round(100.0 * TRAJECTORY_ALPHA.get(category, DEFAULT_TRAJECTORY_ALPHA))
        )
        for category in unique_categories
    }
    initial_densities = {
        category: (
            density_args[category]
            if category in density_args and density_args[category] is not None
            else args.density
        )
        for category in unique_categories
    }

    server = viser.ViserServer(host=args.host, port=args.port, label="G1 goal trajectories")
    server.scene.set_background_image(np.full((2, 2, 3), 255, dtype=np.uint8))
    server.scene.world_axes.visible = False
    server.scene.set_up_direction("+x")

    extent = max(float(np.max(np.abs(trajectories))), 1.0)
    view_limit = max(extent * 1.05, 2.2)
    add_plot_guides(server, view_limit)
    server.initial_camera.position = (0.0, 0.0, view_limit * 2.5)
    server.initial_camera.look_at = (0.0, 0.0, 0.0)
    server.initial_camera.up_direction = (1.0, 0.0, 0.0)

    mesh_count = add_g1_meshes(server, args.mjcf)
    density_sliders = {
        category: server.gui.add_slider(
            f"{category} density (%)", min=1, max=100, step=1,
            initial_value=initial_densities[category],
        )
        for category in unique_categories
    }
    opacity_sliders = {
        category: server.gui.add_slider(
            f"{category} opacity (%)", min=0, max=100, step=1,
            initial_value=initial_opacities[category],
        )
        for category in unique_categories
    }
    trajectory_count = server.gui.add_text("Trajectories shown", "", disabled=True)

    def update_trajectories(_) -> None:
        total = 0
        for category in unique_categories:
            indices = category_indices[category]
            count = max(
                1, round(len(indices) * density_sliders[category].value / 100)
            )
            sampled = trajectories[indices[:count]]
            server.scene.add_line_segments(
                f"/trajectories/{category}",
                points=trajectories_to_segments(sampled),
                colors=blend_on_white(
                    base_colors[category],
                    opacity_sliders[category].value / 100.0,
                ),
                thickness=args.line_width,
                thickness_units="screen",
            )
            total += count
        trajectory_count.value = f"{total:,}"

    for slider in density_sliders.values():
        slider.on_update(update_trajectories)
    for slider in opacity_sliders.values():
        slider.on_update(update_trajectories)
    update_trajectories(None)
    print(f"Loaded {len(trajectories):,} trajectories and {mesh_count} G1 meshes")
    print(f"Open http://localhost:{args.port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
