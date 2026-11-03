#!/usr/bin/env python3
"""Plot manual horizontal average-speed limits and height distribution diagnostics.

    python TextOpRobotMDAR/scripts/plot_goal_clamp.py /path/to/goal_stats.pkl
    python TextOpRobotMDAR/scripts/plot_goal_clamp.py --speed-low 0.6 --speed-full 3

Clamp curves use only YAML/manual settings: R(h,T) = v(h)*T_eff.
Optional goal_stats.pkl provides observational height counts and quantiles;
it never determines the runtime envelope or the model's time horizon.
When supplied, observed P95 speed is overlaid on goal_clamp.png.
Observed displacement mixes arrival horizons; it is not a fixed-time radius.
Counts refer to sampled goal windows, not unique frames or motion duration.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent / "deps" / "isaac_utils"))

from robotmdar.utils.goal import GoalClamp
from robotmdar.utils.goal_time import clamp_goal_time

CLAMP_OPTIONS = ("height_low", "height_full", "speed_low", "speed_full", "time_min")


def plot_supported_quantile(ax, heights, values, counts, min_samples, label, **kwargs):
    """Break lines at sparse bins; show their observations as hollow markers."""
    import numpy as np

    heights, values = np.asarray(heights, dtype=float), np.asarray(values, dtype=float)
    counts = np.asarray(counts)
    supported = counts >= min_samples
    line, = ax.plot(heights, np.where(supported, values, np.nan), ".-",
                    label=label, **kwargs)
    sparse = (counts > 0) & ~supported & np.isfinite(values)
    ax.scatter(heights[sparse], values[sparse], facecolors="none",
               edgecolors=line.get_color(), s=30)


def overlay_observed_p95(stats, speed_ax, min_samples):
    """Show stored P95 observations without fitting, resampling or clamping data."""
    import numpy as np

    tables = stats.get("goal_clamp", {})
    fixed = tables.get("height_distribution")
    table = fixed or tables.get("height_conditioned")
    if table is None:
        print("No per-height statistics; P95 overlay omitted.")
        return "No per-height P95 stored."
    levels = np.asarray(table["quantile_levels"], dtype=float)
    columns = np.flatnonzero(np.isclose(levels, 95))
    if not len(columns):
        print("P95 is absent; overlay omitted. Stored percentiles are not interpolated.")
        return "P95 not stored."
    column = int(columns[0])
    # Preserve gaps in empty height bins; do not substitute global quantiles.
    heights = np.asarray(table["centers"], dtype=float)
    source = "height bins" if fixed else "legacy rank buckets"
    if "speed_quantiles" in table:
        values = np.asarray(table["speed_quantiles"], dtype=float)[:, column]
        plot_supported_quantile(
            speed_ax, heights, values, table["n_samples"], min_samples,
            f"Observed P95 implied speed; {source}", color="tab:red")
    if table.get("time_max") is not None:
        time_note = f"stored speed budget=[{table.get('time_min', 0):g}, {table['time_max']:g}] s"
    else:
        time_note = "stored speed time rule is unspecified"
    print(f"P95 overlay: {source}; {time_note}; distance mixes arrival horizons.")
    return f"P95: {time_note}; observed distance mixes arrival times."


def plot_height_distribution(stats, output, percentiles, count_log=False, min_samples=100):
    """Plot/export fixed-height diagnostics, or explicitly labeled legacy buckets."""
    import numpy as np
    import matplotlib.pyplot as plt

    tables = stats["goal_clamp"]
    fixed = tables.get("height_distribution")
    table = fixed or tables.get("height_conditioned")
    if table is None:
        print("No per-height statistics. Refresh goal_stats.pkl to plot height distributions.")
        return
    counts = np.asarray(table["n_samples"], dtype=int)
    levels = np.asarray(table["quantile_levels"], dtype=float)
    selected = [(p, int(np.flatnonzero(np.isclose(levels, p))[0]))
                for p in percentiles if np.any(np.isclose(levels, p))]
    missing = [p for p in percentiles if not np.any(np.isclose(levels, p))]
    if missing:
        print(f"Percentiles {missing} are absent and will be omitted; refresh stats to add them.")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    count_ax, dist_ax, speed_ax, time_ax = axes.flat
    if fixed:
        edges = np.asarray(table["bin_edges"])
        centers = (edges[:-1] + edges[1:]) / 2
        count_ax.bar(centers, counts, width=np.diff(edges) * 0.9)
        count_ax.set_xlabel("Reference root height (m)")
        source = "Fixed height intervals; sampled training goal windows"
        print(f"Height diagnostics: {table['n_valid']}/{table['n_total']} valid goals; "
              f"{table['n_excluded']} excluded; bin width={table['bin_width']:g} m")
    else:
        centers = np.asarray(table["centers"])
        indices = np.arange(len(counts))
        count_ax.bar(indices, counts)
        count_ax.set_xticks(indices, [f"B{i + 1}\nmean={h:.3f} m"
                                     for i, h in enumerate(centers)])
        count_ax.set_xlabel("Legacy rank bucket (equal sample counts, unequal height widths)")
        source = "Legacy rank buckets: height occupancy cannot be recovered"
        print("Legacy equal-count buckets: counts are NOT a height histogram. "
              "Refresh goal_stats.pkl for fixed-height occupancy and arrival-time quantiles.")
    count_ax.set(title=f"Goal window counts (N={counts.sum():,})", ylabel="Sampled goal windows")
    if count_log:
        count_ax.set_yscale("symlog", linthresh=1)
    # A secondary axis expresses each bar as a percentage of all valid windows.
    if counts.sum():
        percent_ax = count_ax.secondary_yaxis(
            "right", functions=(lambda y: y * 100 / counts.sum(),
                                lambda y: y * counts.sum() / 100))
        percent_ax.set_ylabel("Share of valid goal windows (%)")
    for ax, key, title, unit in (
        (dist_ax, "dist_quantiles", "Horizontal goal displacement", "m"),
        (speed_ax, "speed_quantiles", "Implied goal speed: distance / arrival budget", "m/s"),
        (time_ax, "arrival_time_quantiles", "Raw arrival time (before time clamp)", "s"),
    ):
        if key not in table:
            ax.text(0.5, 0.5, "Not stored in this file; refresh goal statistics",
                    ha="center", transform=ax.transAxes)
        else:
            values = np.asarray(table[key], dtype=float)
            for percentile, column in selected:
                # Null quantiles in empty bins become NaN and break the curve.
                plot_supported_quantile(ax, centers, values[:, column], counts,
                                        min_samples, f"P{percentile:g}")
            if selected:
                ax.legend()
        ax.set(title=title, xlabel="Reference root height (m)", ylabel=unit)
        ax.set_ylim(bottom=0)
    for ax in axes.flat:
        ax.grid(True, alpha=0.2)
    time_note = (f"speed uses time budget [{table['time_min']:g}, {table['time_max']:g}] s"
                 if table.get("time_max") is not None else "speed uses stored arrival budget")
    fig.suptitle(source + f"\nRaw goal windows; {time_note}; no action filtering"
                 + f"\nHollow markers: N < {min_samples}; window counts are not independent samples")
    image_path = output.with_name(output.stem + "_height_distribution" + output.suffix)
    fig.savefig(image_path, dpi=180, bbox_inches="tight")
    csv_path = output.with_name(output.stem + "_height_distribution.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["height_lower_m", "height_upper_m", "height_center_m", "n_samples",
                         "sample_fraction", "percentile", "distance_m", "implied_speed_m_s",
                         "arrival_time_s", "effective_time_budget_s", "sparse_bin"])
        for i, count in enumerate(counts):
            for j, percentile in enumerate(levels):
                writer.writerow([
                    table["bin_edges"][i] if fixed else table.get("height_min", [None] * len(counts))[i],
                    table["bin_edges"][i + 1] if fixed else table.get("height_max", [None] * len(counts))[i],
                    table["centers"][i], count, count / counts.sum() if counts.sum() else 0,
                    percentile, table["dist_quantiles"][i][j], table["speed_quantiles"][i][j],
                    table["arrival_time_quantiles"][i][j] if "arrival_time_quantiles" in table else None,
                    table["time_budget_quantiles"][i][j] if "time_budget_quantiles" in table else None,
                    0 < count < min_samples,
                ])
    print(f"Saved: {image_path.resolve()}")
    print(f"Saved: {csv_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("goal_stats", type=Path, nargs="?",
                        help="Optional goal_stats.pkl for observational distributions only")
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "robotmdar/config/planner_dar.yaml")
    parser.add_argument("--output", type=Path,
                        help="Default: <stats directory or cwd>/goal_clamp.png")
    parser.add_argument("--height-min", type=float, default=0.0, help="Root height in m")
    parser.add_argument("--height-max", type=float, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--distribution-quantiles", type=float, nargs="+", default=[50, 75, 90, 95, 99])
    parser.add_argument("--count-log", action="store_true")
    parser.add_argument("--min-bin-samples", type=int, default=100,
                        help="Display threshold for sparse bins; not a confidence bound")
    parser.add_argument("--arrival-times", type=float, nargs="+",
                        help="Positive arrival times (s); default: minimum, half and full horizon")
    for key in CLAMP_OPTIONS:
        parser.add_argument("--" + key.replace("_", "-"), type=float,
                            help=f"Override goal_clamp.{key} from the YAML")
    args = parser.parse_args()
    if args.min_bin_samples < 1:
        parser.error("--min-bin-samples must be >= 1")
    config = OmegaConf.load(args.config)
    options = dict(config.get("goal_clamp", {}))
    for key in CLAMP_OPTIONS:
        value = getattr(args, key)
        if value is not None:
            options[key] = value
    try:
        fps = float(config.get("motion_fps", 50))
        clamp = GoalClamp(**options, fps=fps, time_max=int(config.data.future_len) / fps)
        stats = (torch.load(args.goal_stats, map_location="cpu", weights_only=False)
                 if args.goal_stats is not None else None)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    height_max = args.height_max if args.height_max is not None else max(1.0, clamp.height_full + 0.1)
    if not args.height_min < height_max:
        parser.error("--height-min must be less than --height-max")
    arrival_times = args.arrival_times or [clamp.time_min, clamp.time_max / 2, clamp.time_max]
    if any(not 0 < t < float("inf") for t in arrival_times):
        parser.error("--arrival-times must be finite and positive")
    heights = torch.linspace(args.height_min, height_max, 1000, dtype=torch.float64)
    knots = [h for h in (clamp.height_low, clamp.height_full)
             if args.height_min <= h <= height_max]
    heights = torch.cat((heights, heights.new_tensor(knots))).sort().values
    speeds = clamp.speed_for_height(heights)
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (speed_ax, radius_ax) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    speed_ax.plot(heights.numpy(), speeds.numpy(), linewidth=2.5, label="Manual limit v(h)")
    radius_curves = []
    for arrival in arrival_times:
        radius = clamp.radius_for_height(heights, arrival)
        radius_curves.append(radius.numpy())
        effective = float(clamp_goal_time(arrival, clamp.fps, clamp.time_min, clamp.time_max))
        label = f"T={arrival:g} s"
        if abs(effective - arrival) > 1e-6:
            label += f"; used {effective:g} s"
        radius_ax.plot(heights.numpy(), radius.numpy(), label=label + " radius")
    observation_note = overlay_observed_p95(stats, speed_ax, args.min_bin_samples) if stats is not None else ""
    for ax, label in ((speed_ax, "Horizontal average-speed limit (m/s)"),
                      (radius_ax, "Allowed horizontal radius (m)")):
        for height in (clamp.height_low, clamp.height_full):
            ax.axvline(height, color="gray", alpha=0.3, linestyle="--")
        ax.set(xlabel="Measured root height h (m)", ylabel=label,
               xlim=(args.height_min, height_max), ylim=(0, None))
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    enabled = bool(config.get("goal_clamp_enabled", False))
    fig.suptitle(
        f"Manual goal clamp: speeds=[{clamp.speed_low:g}, {clamp.speed_full:g}] m/s\n"
        f"height transition=[{clamp.height_low:g}, {clamp.height_full:g}] m; "
        f"time=[{clamp.time_min:g}, {clamp.time_max:g}] s")
    fig.text(0.5, -0.035,
             f"YAML goal_clamp_enabled={enabled}; envelope shown as if enabled. "
             "Statistics are observational only; limits come from manual settings.\n"
             + observation_note,
             ha="center", fontsize=8)
    output = args.output or (args.goal_stats.with_name("goal_clamp.png")
                             if args.goal_stats is not None else Path("goal_clamp.png"))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    csv_path = output.with_name(output.stem + "_envelope.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["height_m", "speed_limit_m_s"]
                        + [f"radius_T{t:g}s_m" for t in arrival_times])
        writer.writerows(zip(heights.tolist(), speeds.tolist(), *radius_curves))
    provenance = {
        "planner_config": str(args.config.resolve()),
        "goal_clamp": options,
        "fps": fps, "time_max": clamp.time_max,
        "stats_path": str(args.goal_stats.resolve()) if args.goal_stats else None,
        "stats_meta": stats.get("meta", {}) if stats else {},
        "min_bin_samples": args.min_bin_samples,
        "arrival_times": arrival_times,
    }
    OmegaConf.save(OmegaConf.create(provenance),
                   output.with_name(output.stem + "_calibration.yaml"))
    print(f"Config: {args.config.resolve()}")
    print(f"goal_clamp_enabled={enabled}; plotting the enabled envelope.")
    print(f"Manual limits: speeds=[{clamp.speed_low:g}, {clamp.speed_full:g}] m/s; "
          f"heights=[{clamp.height_low:g}, {clamp.height_full:g}] m; "
          f"time=[{clamp.time_min:g}, {clamp.time_max:g}] s")
    print(f"Saved: {output.resolve()}")
    print(f"Saved: {csv_path.resolve()}")
    if stats is not None:
        plot_height_distribution(stats, output, args.distribution_quantiles,
                                 args.count_log, args.min_bin_samples)
        reference = stats.get("goal_clamp", {}).get("reference_distribution")
        if reference:
            print(f"Reference height proxy {reference['height_range']} m: "
                  f"N={reference['n_samples']}/{reference['n_total']}; "
                  f"{reference['motion_filter']}")
            print(f"Reference percentiles {reference['quantile_levels']}: "
                  f"implied speed={reference['speed_quantiles']} m/s")
    if args.show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
