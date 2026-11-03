#!/usr/bin/env python3
"""Draw duration-share charts from precomputed dataset statistics."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "dataset/data_analyze/analysis_output/dataset_statistics/duration_comparison"
DATASET_COLORS = {
    "BONES-SEED": "#0077FF",
    "HiPhi": "#FE7235",
    "AMASS-G1": "#90C273",
    "LAFAN1-G1": "#00C3FF",
}
BREAKDOWN_COLORS = ("#FE7235", "#FEA735", "#90C273")


def duration_autopct(hours):
    """Format each pie wedge with its share and duration."""
    index = 0

    def formatter(percent):
        nonlocal index
        value = hours[index]
        index += 1
        return f"{percent:.1f}%\n{value:,.1f} h" if percent >= 3 else ""

    return formatter


def plot_duration(datasets, totals, output):
    colors = [DATASET_COLORS[data["name"]] for data in datasets]
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
    small_index = next(i for i, data in enumerate(datasets) if data["name"] == "LAFAN1-G1")
    explode = [0.12 if i == small_index else 0 for i in range(len(datasets))]
    for stage, radius in (("raw", 1.0), ("processed", .67)):
        hours = [data[f"{stage}_hours"] for data in datasets]
        wedges, _, _ = ax.pie(
            hours, radius=radius, startangle=90, counterclock=False,
            colors=colors, explode=explode,
            wedgeprops=dict(width=.29, edgecolor="white", linewidth=1.5),
            autopct=duration_autopct(hours),
            pctdistance=(radius - .145) / radius,
            textprops=dict(fontsize=11, color="white", weight="bold"),
        )
        wedges[small_index].set_edgecolor("#0088AD")
        wedges[small_index].set_linewidth(.7)
        if stage == "raw":
            small_wedge = wedges[small_index]
            angle = np.deg2rad((small_wedge.theta1 + small_wedge.theta2) / 2)
            tip = ((1 + explode[small_index]) * np.cos(angle),
                   (1 + explode[small_index]) * np.sin(angle))
            ax.scatter(*tip, s=75, color=colors[small_index], edgecolor="white",
                       linewidth=1.4, zorder=5, clip_on=False)
            small = datasets[small_index]
            ax.annotate(
                f"LAFAN1-G1\nBefore {small['raw_duration_share']:.2%}, "
                f"{small['raw_hours']:.1f} h\n"
                f"After {small['processed_duration_share']:.2%}, "
                f"{small['processed_hours']:.1f} h",
                xy=tip, xytext=(-1.48, 1.04), ha="left", va="center", fontsize=9,
                color="#126A87", annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color="#00A2D1", lw=1.5,
                                connectionstyle="angle,angleA=0,angleB=90,rad=0"),
            )
    ax.text(0, 0, f"Before  {totals['raw']['hours']:,.1f} h\n"
            f"After    {totals['processed']['hours']:,.1f} h",
            ha="center", va="center", fontsize=11)
    fig.suptitle("Dataset duration share", fontsize=20, y=.965)
    fig.text(.5, .925, "Outer ring: before   |   Inner ring: after dataset filtering",
             ha="center", fontsize=11, color="#555555")
    legend = [Patch(color=color, label=data["name"])
              for color, data in zip(colors, datasets)]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .275),
               ncol=4, frameon=False, fontsize=10)
    hiphi = next(d for d in datasets if d["name"] == "HiPhi")
    breakdown = hiphi["cleaning_breakdown"]
    segments = [
        (breakdown["tpose_removed_seconds"], BREAKDOWN_COLORS[0]),
        (breakdown["collision_removed_seconds"], BREAKDOWN_COLORS[1]),
        (hiphi["processed_seconds"], BREAKDOWN_COLORS[2]),
    ]
    bar = fig.add_axes((.16, .178, .68, .026))
    cursor = 0.0
    for seconds, color in segments:
        bar.barh(0, seconds, left=cursor, color=color, height=1)
        cursor += seconds
    bar.set_xlim(0, hiphi["raw_seconds"])
    bar.axis("off")
    fig.text(.5, .226, "HiPhi cleaning composition",
             ha="center", fontsize=11, color="#333333")
    for x, label, seconds, color in zip(
            (.18, .5, .82), ("T-pose removed", "Collision candidate", "Retained"),
            (breakdown["tpose_removed_seconds"], breakdown["collision_removed_seconds"],
             hiphi["processed_seconds"]), ("#C24E18", "#9B6412", "#41752F")):
        fig.text(x, .149, f"{label}\n{seconds / hiphi['raw_seconds']:.1%}",
                 ha="center", va="top", fontsize=9, color=color)
    fig.text(.5, .045, "AMASS-G1 and LAFAN1-G1 are published duration placeholders.",
             ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(top=.87, bottom=.30)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"duration_share.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, default=OUTPUT / "statistics.json")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    statistics = json.loads(args.statistics.read_text(encoding="utf-8"))
    datasets, totals = statistics["datasets"], statistics["totals"]
    if not datasets or any(stage not in totals for stage in ("raw", "processed")):
        raise ValueError("Invalid precomputed duration statistics")
    args.output.mkdir(parents=True, exist_ok=True)
    plot_duration(datasets, totals, args.output)
    print(f"Saved duration_share.png and duration_share.svg to {args.output}")


if __name__ == "__main__":
    main()
