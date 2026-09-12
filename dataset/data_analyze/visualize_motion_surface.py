#!/usr/bin/env python3
"""Fit a shared UMAP manifold to ten similar motions and loft their curves.

The surface is a ruled, triangulated approximation: each trajectory is
resampled by arc length, trajectories are ordered by their centroid along the
first principal direction, and adjacent curves are joined into quads.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from visualize_motion_manifold import DEFAULT_LIMITS_XML, embed, load_csv


def resample_curve(curve: np.ndarray, samples: int) -> np.ndarray:
    if len(curve) == 1:
        return np.repeat(curve, samples, axis=0)
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    if distance[-1] <= 1e-12:
        return np.repeat(curve[:1], samples, axis=0)
    target = np.linspace(0.0, distance[-1], samples)
    return np.column_stack([np.interp(target, distance, curve[:, axis]) for axis in range(3)])


def plot_surface(embedding: np.ndarray, curves: list[np.ndarray], names: list[str], output: Path,
                 surface_samples: int) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    resampled = np.stack([resample_curve(c, surface_samples) for c in curves])
    centers = resampled.mean(axis=1)
    _, _, vh = np.linalg.svd(centers - centers.mean(axis=0), full_matrices=False)
    ordered = np.argsort((centers - centers.mean(axis=0)) @ vh[0])
    resampled = resampled[ordered]
    names = [names[i] for i in ordered]

    vertices = resampled.reshape(-1, 3)
    faces = []
    for row in range(len(resampled) - 1):
        for col in range(surface_samples - 1):
            a = row * surface_samples + col
            b, c, d = a + 1, a + surface_samples, a + surface_samples + 1
            faces.extend(((a, c, b), (b, c, d)))
    triangles = np.asarray(faces, dtype=int)
    triangulation = Triangulation(vertices[:, 0], vertices[:, 1], triangles)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*embedding.T, s=2, c="0.55", alpha=0.10, depthshade=False, label="all poses")
    ax.plot_trisurf(triangulation, vertices[:, 2], color="steelblue", alpha=0.28,
                    linewidth=0.25, edgecolor="0.25")
    colors = plt.cm.tab10(np.linspace(0, 1, len(resampled)))
    for curve, name, color in zip(resampled, names, colors):
        ax.plot(*curve.T, color=color, linewidth=1.7, label=name[:28])
    ax.set(xlabel="UMAP-1", ylabel="UMAP-2", zlabel="UMAP-3",
           title="Walk motion manifold: pose cloud, trajectories, and lofted surface")
    ax.legend(fontsize=7, loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    np.save(output.with_suffix(".npy"), vertices)
    plt.close(fig)
    print(f"Saved {len(curves)} curves and {len(vertices)} surface vertices to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory containing Bones-Seed CSV files")
    parser.add_argument("--motion", default="walk", help="Filename substring used for selection")
    parser.add_argument("--num-motions", type=int, default=10)
    parser.add_argument("--surface-samples", type=int, default=80)
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--temporal-weight", type=float, default=0.0,
                        help="Continuity prior strength (0 = pure pose embedding, default)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limits-xml", type=Path, default=DEFAULT_LIMITS_XML)
    parser.add_argument("--output", type=Path, default=Path("walk_motion_surface.png"))
    args = parser.parse_args()
    paths = sorted(p for p in args.input.rglob("*.csv") if args.motion.lower() in p.stem.lower())
    if len(paths) < args.num_motions:
        raise FileNotFoundError(f"Found {len(paths)} matching motions, need {args.num_motions}")
    paths = paths[:args.num_motions]
    features_by_motion = [load_csv(path, args.limits_xml) for path in paths]
    lengths = [len(item) for item in features_by_motion]
    all_features = np.concatenate(features_by_motion, axis=0)
    all_embedding = embed(all_features, lengths=lengths, neighbors=args.neighbors, seed=args.seed,
                          temporal_weight=args.temporal_weight)
    curves = []
    start = 0
    for length in lengths:
        curves.append(all_embedding[start:start + length])
        start += length
    plot_surface(all_embedding, curves, [path.stem for path in paths], args.output, args.surface_samples)


if __name__ == "__main__":
    main()
