#!/usr/bin/env python3
"""Lightweight BONES-SEED text similarity analysis.

This version keeps the final text grain simple:
- short_description is the sequence-level fallback
- temporal descriptions are lightly normalized into event_core
- all temporal events are used, instead of sampling a single event per file
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from clip.simple_tokenizer import SimpleTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robotmdar.model.clip import encode_text, load_and_freeze_clip  # noqa: E402


METADATA_CSV = "/home/lenovo/data/bones-seed/metadata/seed_metadata_v004.csv"
TEMPORAL_JSONL = "/home/lenovo/data/bones-seed/metadata/seed_metadata_v002_temporal_labels.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "analysis_output" / "text_description_similarity_simple"
DEFAULT_CLIP_VERSION = "ViT-B/32"
DEFAULT_SAMPLE_SIZE = 400  # roughly ~1k expanded event samples on BONES-SEED
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 128
CLIP_CONTEXT_LENGTH = 77
TOKENIZER = SimpleTokenizer()

SUBJECT_PREFIX_RE = re.compile(
    r"^(?:a|an|the)\s+"
    r"(?:(?:standing|seated|upright|injured|wounded|crouched|kneeling|sitting|lying|bent)\s+)*"
    r"(?:person(?:'s)?|character(?:'s)?|individual(?:'s)?|figure(?:'s)?|man|woman|dancer|player|actor|someone|somebody)\b[\s,]*"
)
COPULA_PREFIX_RE = re.compile(r"^(?:is|are|was|were|be|been|being)\s+")


@dataclass
class EventSample:
    sample_id: str
    filename: str
    event_index: int
    is_mirror: bool
    source: str
    start_time: float | None
    end_time: float | None
    short_description: str
    temporal_description: str
    event_core: str
    short_token_count: int = 0
    temporal_token_count: int = 0
    event_core_token_count: int = 0
    short_overflow: bool = False
    temporal_overflow: bool = False
    event_core_overflow: bool = False
    short_temporal_similarity: float = 0.0
    temporal_event_similarity: float = 0.0


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0", "yes"}


def load_metadata_rows(metadata_csv: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(metadata_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            short_text = str(row.get("content_short_description", "")).strip()
            if not short_text:
                continue
            row["content_short_description"] = short_text
            rows.append(row)
    return rows


def load_temporal_map(jsonl_path: str) -> dict[str, dict]:
    temporal_map: dict[str, dict] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            temporal_map[obj["filename"]] = obj
    return temporal_map


def stable_sample_rows(
    rows: list[dict[str, str]],
    sample_size: int,
    seed: int,
    include_mirrored: bool,
) -> list[dict[str, str]]:
    filtered = [
        row for row in rows
        if include_mirrored or not truthy(row.get("is_mirror", ""))
    ]
    if sample_size >= len(filtered):
        return list(filtered)
    ranked = sorted(
        filtered,
        key=lambda row: hashlib.sha1(
            f"{seed}|{row['filename']}|{row.get('is_mirror', '')}".encode("utf-8")
        ).hexdigest(),
    )
    return ranked[:sample_size]


def normalize_motion_text(text: str) -> str:
    text = normalize_text(text)
    text = SUBJECT_PREFIX_RE.sub("", text)
    text = COPULA_PREFIX_RE.sub("", text)
    return text.strip(" ,;:.!?\"'")


def expand_event_samples(
    sampled_rows: list[dict[str, str]],
    temporal_map: dict[str, dict],
) -> tuple[list[EventSample], int]:
    samples: list[EventSample] = []
    fallback_files = 0

    for row in sampled_rows:
        filename = row["filename"]
        is_mirror = truthy(row.get("is_mirror", ""))
        short_text = str(row["content_short_description"]).strip()
        temporal_obj = temporal_map.get(filename, {})
        events = temporal_obj.get("events") or []

        if not events:
            fallback_files += 1
            events = [
                {
                    "start_time": None,
                    "end_time": None,
                    "description": short_text,
                }
            ]
            source = "fallback_short"
        else:
            source = "temporal"

        for event_index, event in enumerate(events):
            temporal_raw = str(event.get("description", "")).strip()
            if not temporal_raw:
                continue
            event_core = normalize_motion_text(temporal_raw) or normalize_text(temporal_raw)
            samples.append(
                EventSample(
                    sample_id=f"{filename}#{event_index}",
                    filename=filename,
                    event_index=event_index,
                    is_mirror=is_mirror,
                    source=source,
                    start_time=(
                        float(event["start_time"])
                        if event.get("start_time") is not None
                        else None
                    ),
                    end_time=(
                        float(event["end_time"])
                        if event.get("end_time") is not None
                        else None
                    ),
                    short_description=short_text,
                    temporal_description=temporal_raw,
                    event_core=event_core,
                )
            )

    return samples, fallback_files


def clip_token_count(text: str) -> int:
    return len(TOKENIZER.encode(str(text)))


def clip_overflow(text: str) -> bool:
    return clip_token_count(text) + 2 > CLIP_CONTEXT_LENGTH


def encode_texts(texts: list[str], clip_model, batch_size: int) -> torch.Tensor:
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        batch_emb = encode_text(clip_model, batch, force_empty_zero=True)
        embeddings.append(F.normalize(batch_emb.float(), dim=-1))
    return torch.cat(embeddings, dim=0)


def summarize_scores(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_similarity_means(stats: dict[str, dict[str, float]], outpath: Path) -> None:
    labels = ["short vs temporal", "temporal vs event_core"]
    keys = ["short_temporal", "temporal_event"]
    means = np.array([stats[k]["mean"] for k in keys], dtype=float)
    p10s = np.array([stats[k]["p10"] for k in keys], dtype=float)
    p90s = np.array([stats[k]["p90"] for k in keys], dtype=float)
    x = np.arange(len(labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(x, means, width=width, color=["#4c78a8", "#54a24b"])
    ax.errorbar(
        x,
        means,
        yerr=[means - p10s, p90s - means],
        fmt="none",
        ecolor="black",
        elinewidth=1,
        capsize=4,
    )
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + 0.01,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("cosine similarity")
    ax.set_ylim(max(0.0, float(np.min(p10s)) - 0.05), min(1.0, float(np.max(p90s)) + 0.05))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Lightweight BONES-SEED text analysis using raw short descriptions, "
            "raw temporal labels, and lightly cleaned event_core text."
        )
    )
    parser.add_argument("--metadata-csv", default=METADATA_CSV)
    parser.add_argument("--temporal-jsonl", default=TEMPORAL_JSONL)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="number of metadata rows/files to sample before expanding all temporal events",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--clip-version", default=DEFAULT_CLIP_VERSION)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--include-mirrored", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Loading metadata: {args.metadata_csv}")
    print(f"Loading temporal labels: {args.temporal_jsonl}")
    print(f"Sampling files: {args.sample_size}")
    print(f"Include mirrored rows: {args.include_mirrored}")
    print(f"CLIP version: {args.clip_version}")
    print(f"CLIP device: {device}")

    metadata_rows = load_metadata_rows(args.metadata_csv)
    temporal_map = load_temporal_map(args.temporal_jsonl)
    sampled_rows = stable_sample_rows(
        metadata_rows,
        args.sample_size,
        args.seed,
        args.include_mirrored,
    )
    samples, fallback_files = expand_event_samples(sampled_rows, temporal_map)
    if not samples:
        raise RuntimeError("No event samples were collected")

    print(f"Sampled files: {len(sampled_rows)}")
    print(f"Expanded event samples: {len(samples)}")
    print(f"Fallback-only files: {fallback_files}")

    clip_model = load_and_freeze_clip(args.clip_version, device=device)

    short_texts = [sample.short_description for sample in samples]
    temporal_texts = [sample.temporal_description for sample in samples]
    event_texts = [sample.event_core for sample in samples]

    short_emb = encode_texts(short_texts, clip_model, args.batch_size)
    temporal_emb = encode_texts(temporal_texts, clip_model, args.batch_size)
    event_emb = encode_texts(event_texts, clip_model, args.batch_size)

    short_temporal_sim = (short_emb * temporal_emb).sum(dim=-1).cpu().numpy()
    temporal_event_sim = (temporal_emb * event_emb).sum(dim=-1).cpu().numpy()

    short_counts = np.array([clip_token_count(text) for text in short_texts], dtype=float)
    temporal_counts = np.array([clip_token_count(text) for text in temporal_texts], dtype=float)
    event_counts = np.array([clip_token_count(text) for text in event_texts], dtype=float)

    short_empty = np.array([not normalize_text(text) for text in short_texts], dtype=bool)
    temporal_empty = np.array([not normalize_text(text) for text in temporal_texts], dtype=bool)
    event_empty = np.array([not normalize_text(text) for text in event_texts], dtype=bool)

    short_overflow = np.array([clip_overflow(text) for text in short_texts], dtype=bool)
    temporal_overflow = np.array([clip_overflow(text) for text in temporal_texts], dtype=bool)
    event_overflow = np.array([clip_overflow(text) for text in event_texts], dtype=bool)

    token_change = 1.0 - event_counts / np.maximum(temporal_counts, 1.0)

    short_stats = {
        "empty_rate": float(np.mean(short_empty)),
        "overflow_rate": float(np.mean(short_overflow)),
        "token_mean": float(np.mean(short_counts)),
        "token_median": float(np.median(short_counts)),
    }
    temporal_stats = {
        "empty_rate": float(np.mean(temporal_empty)),
        "overflow_rate": float(np.mean(temporal_overflow)),
        "token_mean": float(np.mean(temporal_counts)),
        "token_median": float(np.median(temporal_counts)),
    }
    event_stats = {
        "empty_rate": float(np.mean(event_empty)),
        "overflow_rate": float(np.mean(event_overflow)),
        "token_mean": float(np.mean(event_counts)),
        "token_median": float(np.median(event_counts)),
    }

    pair_stats = {
        "short_temporal": summarize_scores(short_temporal_sim),
        "temporal_event": summarize_scores(temporal_event_sim),
    }

    rows = []
    for sample, short_count, temporal_count, event_count, short_of, temporal_of, event_of, s2t, t2e, t_change in zip(
        samples,
        short_counts,
        temporal_counts,
        event_counts,
        short_overflow,
        temporal_overflow,
        event_overflow,
        short_temporal_sim,
        temporal_event_sim,
        token_change,
    ):
        rows.append(
            {
                "sample_id": sample.sample_id,
                "filename": sample.filename,
                "event_index": sample.event_index,
                "is_mirror": sample.is_mirror,
                "source": sample.source,
                "start_time": sample.start_time,
                "end_time": sample.end_time,
                "short_description": sample.short_description,
                "temporal_description": sample.temporal_description,
                "event_core": sample.event_core,
                "short_token_count": int(short_count),
                "temporal_token_count": int(temporal_count),
                "event_core_token_count": int(event_count),
                "temporal_to_event_token_change": float(t_change),
                "short_overflow": bool(short_of),
                "temporal_overflow": bool(temporal_of),
                "event_core_overflow": bool(event_of),
                "short_temporal_similarity": float(s2t),
                "temporal_event_similarity": float(t2e),
            }
        )

    write_csv(output_dir / "sampled_alignments.csv", rows)

    worst_rows = sorted(rows, key=lambda row: row["temporal_event_similarity"])[:30]
    write_csv(output_dir / "worst_temporal_event_matches.csv", worst_rows)

    summary_path = output_dir / "similarity_summary.txt"
    lines = [
        "=" * 72,
        "BONES-SEED CLIP Text Similarity",
        "=" * 72,
        f"Metadata CSV          : {args.metadata_csv}",
        f"Temporal JSONL        : {args.temporal_jsonl}",
        f"Sampled files         : {len(sampled_rows)}",
        f"Expanded event samples: {len(samples)}",
        f"Fallback-only files   : {fallback_files}",
        f"Seed                  : {args.seed}",
        f"Include mirrored rows : {args.include_mirrored}",
        f"CLIP version          : {args.clip_version}",
        f"CLIP device           : {device}",
        "",
        "Text stats:",
        f"  short      empty {short_stats['empty_rate']*100:.1f}% | overflow {short_stats['overflow_rate']*100:.1f}% "
        f"| tokens mean/median {short_stats['token_mean']:.1f} / {short_stats['token_median']:.1f}",
        f"  temporal   empty {temporal_stats['empty_rate']*100:.1f}% | overflow {temporal_stats['overflow_rate']*100:.1f}% "
        f"| tokens mean/median {temporal_stats['token_mean']:.1f} / {temporal_stats['token_median']:.1f}",
        f"  event_core empty {event_stats['empty_rate']*100:.1f}% | overflow {event_stats['overflow_rate']*100:.1f}% "
        f"| tokens mean/median {event_stats['token_mean']:.1f} / {event_stats['token_median']:.1f}",
        f"  temporal->event_core token change mean/median: {np.mean(token_change):.3f} / {np.median(token_change):.3f}",
        "",
        "Cosine similarity:",
        f"{'pair':<22} {'mean':>8} {'median':>8} {'p10':>8} {'p90':>8}",
    ]
    for key, label in (("short_temporal", "short-temporal"), ("temporal_event", "temporal-event")):
        stats = pair_stats[key]
        lines.append(
            f"{label:<22} {stats['mean']:>8.3f} {stats['median']:>8.3f} "
            f"{stats['p10']:>8.3f} {stats['p90']:>8.3f}"
        )
    lines.extend(["", "Worst temporal-event samples:"])
    for row in worst_rows[:12]:
        lines.append(
            f"  {row['temporal_event_similarity']:+.3f} | {row['temporal_description']} -> {row['event_core']}"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot_similarity_means(pair_stats, output_dir / "similarity_means.png")

    print(f"Summary saved: {summary_path}")
    print(f"Samples saved: {output_dir / 'sampled_alignments.csv'}")
    print(f"Worst samples saved: {output_dir / 'worst_temporal_event_matches.csv'}")
    print(f"Plot saved: {output_dir / 'similarity_means.png'}")


if __name__ == "__main__":
    main()
