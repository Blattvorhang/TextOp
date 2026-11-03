#!/usr/bin/env python3
"""Measure endpoint shoulder transitions to inform trimming, not produce cleaning labels."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1] / "analysis_output/hiphi_tpose"
ROOT = Path(__file__).resolve().parents[3]


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def measure(roll, fps, low=45.):
    """Walk inward from one edge; estimate when the first raised-arm phase ends."""
    n = len(roll)
    search = min(round(10 * fps), int(.15 * n))
    high = [(a, b) for a, b in runs((roll >= 60.).all(axis=1))
            if a <= search and b - a >= round(.2 * fps)]
    if not high:
        return dict(status="no_bilateral_raised_arm_phase")
    a, b = high[0]
    lows = [(c, d) for c, d in runs((roll <= low).all(axis=1))
            if b <= c <= b + round(3 * fps) and d - c >= round(.2 * fps)]
    result = dict(core_seconds=[a / fps, b / fps])
    if not lows:
        return dict(result, status="no_low_arm_transition")
    c, _ = lows[0]
    return dict(result, status="measured", boundary_seconds=(c + round(.2 * fps)) / fps)


def analyze(task):
    motions, record = task
    with np.load(Path(motions) / record["source"], allow_pickle=False) as data:
        roll = np.rad2deg(data["joint_pos"][:, [15, 16]]) * [1., -1.]
    fps = record["fps"]
    measurements = {str(low): {"head": measure(roll, fps, low), "tail": measure(roll[::-1], fps, low)}
                    for low in (40, 45, 50)}
    return dict(source=record["source"], frames=len(roll), fps=fps,
                baseline_remove_intervals_frames=record["remove_intervals_frames"], measurements=measurements)


def summarize(records):
    result = {}
    for low in (40, 45, 50):
        result[str(low)] = {}
        for edge in ("head", "tail"):
            entries = [r["measurements"][str(low)][edge] for r in records]
            values = np.array([r["boundary_seconds"] for r in entries if r["status"] == "measured"])
            result[str(low)][edge] = dict(total=len(entries), measured=len(values),
                missing_core=sum(r["status"] == "no_bilateral_raised_arm_phase" for r in entries),
                missing_transition=sum(r["status"] == "no_low_arm_transition" for r in entries),
                outside_endpoint_budget=sum(r["status"] == "outside_endpoint_budget" for r in entries),
                quantiles_seconds={str(p): float(np.percentile(values, p)) for p in (50, 75, 90, 95, 99, 100)},
                fixed_trim_coverage={str(s): float(np.mean(values <= s)) for s in (2, 3, 4, 5, 6, 7, 8, 10)})
    return result


def plot_reviews(output, by_source):
    review = json.loads((BASE / "review_unconfirmed_10/manifest.json").read_text())
    fig, axes = plt.subplots(10, 2, figsize=(16, 26))
    rows = []
    for sample, pair in zip(review["samples"], axes):
        record = sample["record"]
        with np.load(BASE / "review_unconfirmed_10" / sample["npz"], allow_pickle=False) as data:
            roll = np.rad2deg(data["joint_pos"][:, [15, 16]]) * [1., -1.]
        measurement = by_source[record["source"]]["measurements"]["45"]
        row = dict(sample_id=sample["sample_id"], source=record["source"], measurements=measurement)
        rows.append(row)
        for edge, ax in zip(("head", "tail"), pair):
            values = roll if edge == "head" else roll[::-1]
            count = min(len(values), round(12 * record["fps"]))
            x = np.arange(count) / record["fps"]
            ax.plot(x, values[:count, 0], label="Left shoulder", lw=.8)
            ax.plot(x, values[:count, 1], label="Right shoulder", lw=.8)
            ax.axhline(60, color="gray", ls="--", lw=.6)
            ax.axhline(45, color="gray", ls=":", lw=.6)
            m = measurement[edge]
            if m["status"] == "measured":
                ax.axvline(m["boundary_seconds"], color="green", label=f"Low-arm boundary {m['boundary_seconds']:.2f}s")
            ax.set(title=f"{sample['sample_id']:02d} {Path(record['source']).parent.name} / {edge}",
                   xlim=(0, 12), ylim=(-30, 140), xlabel="Seconds inward from file edge", ylabel="Shoulder roll (deg)")
            ax.grid(alpha=.2)
            ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(output / "review_10_shoulders.png", dpi=120)
    plt.close(fig)
    (output / "review_10_measurements.json").write_text(json.dumps(rows, indent=2) + "\n")


def write_report(output, records, payload, summary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for edge, ax in zip(("head", "tail"), axes):
        for low in (40, 45, 50):
            values = sorted(r["measurements"][str(low)][edge]["boundary_seconds"] for r in records
                            if r["measurements"][str(low)][edge]["status"] == "measured")
            ax.plot(values, np.arange(1, len(values) + 1) / len(values) * 100, label=f"Low-arm threshold {low} deg")
        ax.set(title=edge, xlabel="Seconds inward from file edge", ylabel="Cumulative percent of measured endpoints", xlim=(0, 12), ylim=(0, 100))
        ax.axvline(7, color="gray", ls="--", lw=.8)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "endpoint_duration_cdf.png", dpi=160)
    plt.close(fig)

    motions = [r for r in payload["motions"] if r["status"] == "ok"]
    original_by_source = {r["source"]: r for r in records}
    total_frames = sum(r["frames"] for r in motions)
    raw_hours = sum(r["frames"] / r["fps"] / 3600 for r in motions)
    alternatives = []
    for seconds in (2, 4, 6, 7, 8, 10):
        kept_frames = sum(max(0, r["frames"] - 2 * round(seconds * r["fps"])) for r in motions)
        kept_hours = sum(max(0, r["frames"] - 2 * round(seconds * r["fps"])) / r["fps"] / 3600 for r in motions)
        alternatives.append(dict(seconds_per_edge=seconds, remaining_frames=kept_frames, remaining_hours=kept_hours,
                                 removed_hours=raw_hours - kept_hours,
                                 head_coverage=summary["45"]["head"]["fixed_trim_coverage"][str(seconds)],
                                 tail_coverage=summary["45"]["tail"]["fixed_trim_coverage"][str(seconds)],
                                 empty_motions=sum(r["frames"] <= 2 * round(seconds * r["fps"]) for r in motions)))
    # Compare a concrete adaptive proposal, without changing any cleaning JSON.
    adaptive_frames, adaptive_hours = 0, 0.
    for r in motions:
        source = r["source"].replace("__mirror/", "/")
        measured = original_by_source[source]["measurements"]["45"]
        edges = [round((measured[e]["boundary_seconds"] + .3 if measured[e]["status"] == "measured" else 7.) * r["fps"])
                 for e in ("head", "tail")]
        for a, b in r["remove_intervals_frames"]:
            if a == 0:
                edges[0] = max(edges[0], b)
            if b == r["frames"]:
                edges[1] = max(edges[1], r["frames"] - a)
        remaining = max(0, r["frames"] - sum(edges))
        adaptive_frames += remaining
        adaptive_hours += remaining / r["fps"] / 3600
    results = dict(raw_frames=total_frames, raw_hours=raw_hours, fixed_trim_scenarios=alternatives,
                   adaptive_proposal=dict(margin_seconds=.3, fallback_seconds=7., preserve_longer_baseline_trims=True,
                                          remaining_frames=adaptive_frames, remaining_hours=adaptive_hours,
                                          removed_hours=raw_hours - adaptive_hours))
    (output / "trim_scenarios.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["# HiPhi 首尾裁剪时长：全量实测", "",
             f"独立原片 {len(records):,} 条；镜像不重复计入分布。时长试算包含 {len(motions):,} 条原片与镜像，共 {raw_hours:.4f} 小时。",
             "", "## 测量口径", "",
             "从文件两端向内观察，寻找最靠边的双肩外展 ≥60°、持续 ≥0.2 秒的片段；起点限于前/后 min(10 秒, 15% 时长)。随后 3 秒内寻找双肩 ≤45°且持续 ≥0.2 秒的低臂窗口，边界包括该窗口。结尾用反向时间计算。",
             "超过 min(12 秒, 20% 时长) 的边界标记为不确定，不作为普通首尾 T-pose 时长；持续抬臂的真实动作可能导致这种情况。40°/50°分别用于敏感性对照。",
             "这是宽松肩角转换的自动测量，不是人工真值；覆盖率仅表示覆盖已测出的端点边界，不能当作整个数据集的 recall。末尾计时均指从末帧向内的长度。", "",
             "## 首尾边界分布（45°）", "",
             "| 端点 | 可测数量 | 无抬臂段 | 无低臂过渡 | 超出首尾预算 | P50 / 秒 | P90 / 秒 | P95 / 秒 | P99 / 秒 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for edge in ("head", "tail"):
        s = summary["45"][edge]
        q = s["quantiles_seconds"]
        lines.append(f"| {edge} | {s['measured']} | {s['missing_core']} | {s['missing_transition']} | {s['outside_endpoint_budget']} | {q['50']:.2f} | {q['90']:.2f} | {q['95']:.2f} | {q['99']:.2f} |")
    lines += ["", "## 固定两端等长裁剪试算", "",
              "下表独立计算固定裁剪，不叠加旧检测区间；含镜像、不额外应用 blacklist。", "",
              "| 每端秒数 | 开头边界覆盖 | 结尾边界覆盖 | 剩余小时 | 删除小时 | 删空条数 |",
              "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in alternatives:
        lines.append(f"| {s['seconds_per_edge']} | {s['head_coverage']:.2%} | {s['tail_coverage']:.2%} | {s['remaining_hours']:.2f} | {s['removed_hours']:.2f} | {s['empty_motions']} |")
    lines += ["", "## 建议", "",
              "不建议把 2 秒作为固定裁剪值。若优先代码最简，两端各 7 秒约对应 P95 加少量余量；各 8 秒可覆盖约 98% 的可测边界，但会损失更多有效动作。",
              f"更建议自适应：能测量时裁到转换边界后 0.3 秒；不能测量或超出预算时该端用 7 秒兜底；保留旧规则更长的删除区间。如此每端均有裁剪，典型只删除 3–5 秒，试算保留 {adaptive_hours:.2f} 小时（{adaptive_frames:,} 帧）。正式方案已采用该规则，结果见 `../full_scan_endpoints_adaptive/removal_intervals.json`；此处为时长试算，镜像继承切点时可能有少量帧差。",
              "", "## 十条人工复核样本", "",
              "肩角曲线见 `review_10_shoulders.png`，数值见 `review_10_measurements.json`。其中 #06 开头没有找到抬臂段，与人工认为不明显一致；#05 开头没有足够连续低臂窗口，属于不能可靠自动定界的情况。",
              "", "![首尾边界累计分布](endpoint_duration_cdf.png)", ""]
    (output / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motions", type=Path, default=ROOT / "data/hiphi/motions")
    parser.add_argument("--index", type=Path, default=BASE / "full_scan_endpoints_shared_mirror/removal_intervals.json")
    parser.add_argument("--output", type=Path, default=BASE / "endpoint_duration_analysis")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.index.read_text())
    originals = [r for r in payload["motions"] if r["status"] == "ok" and not r["is_mirror"]]
    cache = args.output / "measurements.json"
    if cache.is_file():
        records = json.loads(cache.read_text())
    else:
        records = []
        start = time.monotonic()
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for record in executor.map(analyze, ((str(args.motions), r) for r in originals), chunksize=16):
                records.append(record)
                if len(records) % 1024 == 0:
                    print(f"Measured {len(records)}/{len(originals)} originals; {time.monotonic() - start:.1f}s", flush=True)
        cache.write_text(json.dumps(records) + "\n")
    for r in records:
        budget = min(12., .2 * r["frames"] / r["fps"])
        for measurements in r["measurements"].values():
            for m in measurements.values():
                if m["status"] == "measured" and m["boundary_seconds"] > budget:
                    m["status"] = "outside_endpoint_budget"
    cache.write_text(json.dumps(records) + "\n")
    summary = summarize(records)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_reviews(args.output, {r["source"]: r for r in records})
    write_report(args.output, records, payload, summary)
    print(json.dumps(summary["45"], indent=2), flush=True)


if __name__ == "__main__":
    main()
