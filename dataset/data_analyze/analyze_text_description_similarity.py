#!/usr/bin/env python3
"""Compare CLIP text similarity across three BONES-SEED description tiers.

Tiers:
  1. short content_short_description
  2. one random temporal event description from seed_metadata_v002_temporal_labels.jsonl
  3. core phrase extracted from content_short_description

Outputs:
  - similarity_summary.txt
  - sampled_triplets.csv
  - core_rule_counts.csv
  - worst_core_matches.csv
  - similarity_means.png
  - similarity_margin_scatter.png
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robotmdar.model.clip import encode_text, load_and_freeze_clip  # noqa: E402


METADATA_CSV = "/home/lenovo/data/bones-seed/metadata/seed_metadata_v004.csv"
TEMPORAL_JSONL = "/home/lenovo/data/bones-seed/metadata/seed_metadata_v002_temporal_labels.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "analysis_output" / "text_description_similarity"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CLIP_VERSION = "ViT-B/32"
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_SEED = 42

DIRECT_CORE_RULES = (
    ("direct_high_five", "high five", "high five"),
    ("direct_low_five", "low five", "low five"),
    ("direct_handshake", "handshake", "handshake"),
    ("direct_handshake", "shake hands", "handshake"),
    ("direct_fist_chest", "fist to the chest", "fist to chest"),
    ("direct_present", "presenting something", "present"),
    ("direct_present", "present something", "present"),
    ("direct_cellphone", "talking on cellphone", "talk on cellphone"),
    ("direct_wander", "wandering around", "wander"),
    ("direct_slide_forward", "sliding forward", "slide forward"),
    ("direct_oath", "making an oath", "oath"),
    ("direct_quip", "making a quip", "quip"),
    ("direct_five", "giving self a high five", "high five"),
    ("direct_scratch", "itching", "scratch"),
    ("direct_greet", "character greets someone with a hat", "greet"),
    ("direct_greet", "welcoming", "welcome"),
    ("direct_cheer", "cheering", "cheer"),
    ("direct_triumph", "triumphing two handed", "triumph"),
    ("direct_triumph", "triumphing", "triumph"),
    ("direct_swat_bug", "drive off a flying bug", "swat bug"),
    ("direct_swat_bug", "brush away a bug", "swat bug"),
    ("direct_swat_bug", "drive off bug", "swat bug"),
    ("direct_shrug", "expressing don't know without using hands", "shrug"),
    ("direct_shrug", "being confused", "shrug"),
    ("direct_check", "checking whole body", "check body"),
    ("direct_eat", "eating an apple", "eat"),
    ("direct_eat", "eating", "eat"),
    ("direct_eat", "eats", "eat"),
    ("direct_straighten", "changing position from relaxed to straight", "straighten"),
    ("direct_facepalm", "face palming", "facepalm"),
    ("direct_kiss", "chef's kiss", "kiss"),
    ("direct_realize", "just realized something", "realize"),
    ("direct_realize", "character has an eureka moment", "realize"),
    ("direct_yawn", "yawning", "yawn"),
    ("direct_pray", "praying", "pray"),
    ("direct_lonely", "displaying lonely behaviour", "lonely"),
    ("direct_fix", "character fixes something", "fix"),
    ("direct_clear_ear", "clearing the ear", "clear ear"),
    ("direct_search_pocket", "pocket searching", "search pocket"),
    ("direct_tarzan", "acting like tarzan", "swing"),
    ("direct_brush_off", "brushing dust off", "brush off"),
    ("direct_hurry_back", "in a hurry position backwards", "hurry back"),
    ("direct_massage", "belly massage", "massage"),
    ("direct_rub", "rubbing hands", "rub hands"),
    ("direct_horse", "ride a horse", "ride horse"),
    ("direct_shock", "shocked gesture with both hands", "shock"),
    ("direct_wipe", "wiping shoes", "wipe shoes"),
    ("direct_pound", "pounding meat", "pound meat"),
    ("direct_lament", "lamenting posture", "lament"),
    ("direct_hurry", "in a hurry position", "hurry"),
    ("direct_chain_saw", "cutting tree with chainsaw vertically", "cut with chainsaw"),
    ("direct_not_speak", "not speaking", "silent"),
    ("direct_pull_leash", "being pulled diagonally to the right and back by a dog on a leash", "pull"),
    ("direct_drink", "drinking from a cup at a table", "drink"),
    ("direct_not_hearing", "not hearing", "listen"),
    ("direct_think", "thinking", "think"),
    ("direct_body_search", "body search", "body search"),
    ("direct_body_stretch", "arms to the sides", "stretch"),
    ("direct_body_stretch", "bend your right hand to your left ankle", "stretch"),
)

SPORT_CORE_RULES = (
    ("swim", ("swim", "swimming")),
    ("throw", ("throw", "throwing")),
    ("catch", ("catch", "catching")),
    ("kick", ("kick", "kicking")),
    ("punch", ("punch", "punching")),
    ("dribble", ("dribble", "dribbling")),
    ("shoot", ("shoot", "shooting")),
    ("cartwheel", ("cartwheel",)),
    ("exercise", ("exercise",)),
    ("boxing", ("boxing", "box")),
)

CATEGORY_REGEXES = (
    (
        "injured",
        (
            r"\binjured\b",
        ),
    ),
    (
        "crutch",
        (
            r"\bcrutch(?:es)?\b",
        ),
    ),
    (
        "jump",
        (
            r"\bjump(?:ing|s)?\b",
            r"\bhop(?:ping|s)?\b",
            r"\bleap(?:ing|s)?\b",
            r"\bflip(?:ping|s)?\b",
            r"\bvault(?:ing|s)?\b",
            r"\bjump over\b",
            r"\bhigh jump\b",
            r"\bjumping high\b",
        ),
    ),
    (
        "jog",
        (
            r"\bjog(?:ging|s)?\b",
            r"\brun(?:ning|s)?\b",
        ),
    ),
    (
        "walk",
        (
            r"\bwalk(?:ing|s)?\b",
            r"\bmoonwalk\b",
            r"\bstroll(?:ing|s)?\b",
            r"\bstride(?:s|ing)?\b",
            r"\bstep forward\b",
            r"\bstep backward\b",
            r"\bslid(?:e|ing) forward\b",
            r"\bskip(?:ping|s)?\b",
            r"\bwalked\b",
        ),
    ),
    (
        "dance",
        (
            r"\bdanc(?:e|ing|es)\b",
            r"\bchoreograph(?:y|ic)\b",
            r"\bmacarena\b",
            r"\bvogue\b",
            r"\bsalsa\b",
            r"\bhip hop\b",
            r"\bballet\b",
        ),
    ),
    (
        "climb",
        (
            r"\bclimb(?:ing|s)?\b",
            r"\bladder\b",
            r"\bcome up\b",
            r"\bcome down\b",
            r"\bget up onto\b",
            r"\bget down from\b",
        ),
    ),
    (
        "fall",
        (
            r"\bfall(?:ing|s)?\b",
            r"\bfaint(?:ing|s)?\b",
            r"\blying\b",
            r"\blie(?:s|d)?\b",
            r"\broll(?:ing|s)?\b",
        ),
    ),
    (
        "crouch",
        (
            r"\bcrouch(?:ing|es)?\b",
            r"\bcrawl(?:ing|s)?\b",
            r"\bon all fours\b",
            r"\bstoop(?:ing|s)?\b",
            r"\bsquat(?:ting|s)?\b",
            r"\bbend forward\b",
            r"\bplank\b",
        ),
    ),
    (
        "kneel",
        (
            r"\bkneel(?:ing|s)?\b",
            r"\bsit on heels\b",
        ),
    ),
    (
        "sit",
        (
            r"\bsit(?:ting|s|down)?\b",
            r"\bsitting\b",
        ),
    ),
    (
        "carry",
        (
            r"\bcarry(?:ing|s)?\b",
            r"\blift(?:ing|s)?\b",
            r"\bpick(?:ing|s)? up\b",
            r"\bput(?:ting|s)? down\b",
            r"\bhold(?:ing|s)?\b",
            r"\bpass(?:ing|es)?\b",
            r"\bpicks? up\b",
            r"\bputs? down\b",
        ),
    ),
    (
        "reach",
        (
            r"\breach(?:ing|es)?\b",
        ),
    ),
    (
        "push",
        (
            r"\bpush(?:ing|es)?\b",
            r"\bpull(?:ing|s)?\b",
            r"\bcrank\b",
            r"\bvalve\b",
            r"\bhandle\b",
            r"\blever\b",
            r"\bknob\b",
            r"\bdoor\b",
            r"\bpress(?:ing|es)? button\b",
            r"\bopen door\b",
            r"\bclose door\b",
        ),
    ),
    (
        "step_over",
        (
            r"\bstep over\b",
            r"\bstep in\b",
            r"\bstepp?ing in\b",
            r"\bavoid obstacle\b",
            r"\bbump into\b",
            r"\bjump over obstacle\b",
            r"\bneutral avoid\b",
        ),
    ),
    (
        "turn",
        (
            r"\bturn(?:ing|s)?\b",
            r"\bspin(?:ning|s)?\b",
            r"\bturn around\b",
            r"\bturn back\b",
            r"\bturn forward\b",
        ),
    ),
    (
        "idle",
        (
            r"\bidle\b",
            r"\bstand(?:ing|s)?\b",
            r"\blooking around\b",
            r"\blook around\b",
            r"\blooking up\b",
            r"\blooking down\b",
            r"\bneutral\b",
            r"\brelax(?:ing|s)?\b",
            r"\bwatch(?:ing)?\b",
        ),
    ),
    (
        "gesture",
        (
            r"\bhigh five\b",
            r"\blow five\b",
            r"\bhandshake\b",
            r"\bwave\b",
            r"\bsalute\b",
            r"\bclap(?:ping|s)?\b",
            r"\bcheer\b",
            r"\bpoint\b",
            r"\bthumbs\b",
            r"\bgreet\b",
            r"\bbye\b",
            r"\bshhh?\b",
            r"\bfist\b",
            r"\bpray\b",
            r"\bbow\b",
            r"\blaugh\b",
            r"\bscratch\b",
            r"\bstretch\b",
            r"\bcellphone\b",
            r"\bphone\b",
            r"\boath\b",
            r"\bpresent(?:ing)?\b",
            r"\bbody search\b",
            r"\bquip\b",
        ),
    ),
    (
        "sport",
        (
            r"\bswim(?:ming)?\b",
            r"\bthrow(?:ing)?\b",
            r"\bcatch(?:ing)?\b",
            r"\bkick(?:ing)?\b",
            r"\bpunch(?:ing)?\b",
            r"\bdribble(?:ing|s)?\b",
            r"\bshoot(?:ing)?\b",
            r"\bcartwheel\b",
            r"\bexercise\b",
            r"\bboxing\b",
        ),
    ),
)


@dataclass
class SampleRecord:
    filename: str
    is_mirror: bool
    short_description: str
    temporal_description: str
    core_description: str
    core_rule: str
    coarse_category: str


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0", "yes"}


def stable_choice(items: list[str], seed: int, key: str) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    digest = hashlib.sha1(f"{seed}|{key}".encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(items)
    return items[idx]


def contains_any(text: str, phrases: Iterable[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def pick_first(text: str, candidates: Iterable[str], default: str) -> str:
    for candidate in candidates:
        if candidate in text:
            return candidate
    return default


def phrase_matches(text: str, phrase: str) -> bool:
    if " " in phrase or "'" in phrase or "-" in phrase:
        return phrase in text
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def load_metadata_rows(metadata_csv: str) -> list[dict[str, str]]:
    rows = []
    with open(metadata_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = str(row.get("content_short_description", "")).strip()
            if not raw:
                continue
            row["content_short_description"] = raw
            rows.append(row)
    return rows


def load_temporal_map(jsonl_path: str) -> dict[str, dict]:
    temporal_map = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            temporal_map[obj["filename"]] = obj
    return temporal_map


def _directional_core(text: str, base: str, rule_prefix: str) -> tuple[str, str]:
    if contains_any(text, ("sideways", "lateral", "side step", "side-step")):
        return f"{base} sideways", f"{rule_prefix}_sideways"
    if contains_any(text, ("backward", "backwards")):
        return f"{base} backward", f"{rule_prefix}_backward"
    if contains_any(text, ("back diagonal", "back right", "back left", " back ")):
        return f"{base} backward", f"{rule_prefix}_backward"
    if contains_any(text, ("forward", "front", "straight")):
        return f"{base} forward", f"{rule_prefix}_forward"
    if contains_any(text, ("around", "circle", "circular")):
        return f"{base} around", f"{rule_prefix}_around"
    return base, rule_prefix


def _walk_core(text: str) -> tuple[str, str]:
    if "turn" in text:
        return "turn walk", "walk_turn"
    base = pick_first(
        text,
        ("moonwalk", "stroll", "stride", "slide", "skip", "step", "walk"),
        "walk",
    )
    return _directional_core(text, base, f"walk_{base.replace(' ', '_')}")


def _jog_core(text: str) -> tuple[str, str]:
    if "turn" in text:
        return "turn jog", "jog_turn"
    base = pick_first(text, ("run", "jog"), "jog")
    return _directional_core(text, base, f"jog_{base.replace(' ', '_')}")


def _jump_core(text: str) -> tuple[str, str]:
    if "turn" in text:
        return "turn jump", "jump_turn"
    if contains_any(text, ("high jump", "jump high")):
        return "high jump", "jump_high"
    if contains_any(text, ("jump over", "jump_over", "over obstacle", "jump over obstacle")):
        return "jump over", "jump_over"
    base = pick_first(text, ("jump", "hop", "leap", "flip", "vault"), "jump")
    if base in {"hop", "leap", "flip", "vault"}:
        return base, f"jump_{base}"
    return _directional_core(text, base, f"jump_{base.replace(' ', '_')}")


def _dance_core(text: str) -> tuple[str, str]:
    base = pick_first(
        text,
        (
            "hip hop dance",
            "hiphop dance",
            "hip hop",
            "salsa dance",
            "salsa",
            "vogue dance",
            "vogue",
            "country dance",
            "country",
            "ballet",
            "macarena",
            "dance",
        ),
        "dance",
    )
    if base == "hiphop dance":
        base = "hip hop dance"
    return base, f"dance_{base.replace(' ', '_')}"


def _climb_core(text: str) -> tuple[str, str]:
    if contains_any(text, ("down from", "come down", "get down", "climb down", "off")):
        return "climb down", "climb_down"
    if contains_any(text, ("up onto", "come up", "get up", "climb up", "onto")):
        return "climb up", "climb_up"
    if "ladder" in text:
        return "climb ladder", "climb_ladder"
    return "climb", "climb_base"


def _fall_core(text: str) -> tuple[str, str]:
    if contains_any(text, ("roll", "rolling")):
        return "roll", "fall_roll"
    if contains_any(text, ("stand up", "get up", "rise")):
        return "stand up", "fall_recover"
    if contains_any(text, ("lie", "lying", "on ground", "floor")):
        return "lie down", "fall_lie"
    if "faint" in text:
        return "faint", "fall_faint"
    return "fall", "fall_base"


def _crouch_core(text: str) -> tuple[str, str]:
    if contains_any(text, ("crawl", "on all fours", "plank")):
        return "crawl", "crouch_crawl"
    if "squat" in text:
        return "squat", "crouch_squat"
    if contains_any(text, ("stoop", "bend forward")):
        return "bend forward", "crouch_bend"
    return "crouch", "crouch_base"


def _sit_core(text: str) -> tuple[str, str]:
    if "cross" in text:
        return "sit cross", "sit_cross"
    if contains_any(text, ("sit down", "sitting")):
        return "sit down", "sit_down"
    return "sit", "sit_base"


def _carry_core(text: str) -> tuple[str, str]:
    if contains_any(
        text,
        ("pick up", "picking something up", "pickup", "take up", "picks up"),
    ):
        return "pick up", "carry_pick_up"
    if contains_any(text, ("put down", "drops", "drop", "dropping", "puts down")):
        return "put down", "carry_put_down"
    if "hold" in text:
        return "hold", "carry_hold"
    if "pass" in text:
        return "pass", "carry_pass"
    if "lift" in text:
        return "lift", "carry_lift"
    return "carry", "carry_base"


def _push_core(text: str) -> tuple[str, str]:
    if "open" in text:
        return "open", "push_open"
    if "close" in text:
        return "close", "push_close"
    if "pull" in text:
        return "pull", "push_pull"
    if "press" in text or "button" in text:
        return "press button", "push_press"
    if "push" in text:
        return "push", "push_base"
    return "push", "push_base"


def _turn_core(text: str) -> tuple[str, str]:
    if "around" in text:
        return "turn around", "turn_around"
    if "back" in text:
        return "turn back", "turn_back"
    if "forward" in text:
        return "turn forward", "turn_forward"
    return "turn", "turn_base"


def _idle_core(text: str) -> tuple[str, str]:
    if contains_any(text, ("look around", "looking around")):
        return "look around", "idle_look_around"
    if contains_any(text, ("look up", "looking up")):
        return "look up", "idle_look_up"
    if contains_any(text, ("look down", "looking down")):
        return "look down", "idle_look_down"
    if "stretch" in text:
        return "stretch", "idle_stretch"
    if "stand" in text:
        return "stand", "idle_stand"
    if "relax" in text:
        return "relax", "idle_relax"
    return "idle", "idle_base"


def _gesture_core(text: str) -> tuple[str, str]:
    for phrase, rule in (
        ("high five", "gesture_high_five"),
        ("low five", "gesture_low_five"),
        ("handshake", "gesture_handshake"),
        ("shake hands", "gesture_handshake"),
        ("wave", "gesture_wave"),
        ("salute", "gesture_salute"),
        ("clap", "gesture_clap"),
        ("point", "gesture_point"),
        ("cheer", "gesture_cheer"),
        ("pray", "gesture_pray"),
        ("laugh", "gesture_laugh"),
        ("bow", "gesture_bow"),
        ("shhh", "gesture_shush"),
        ("scratch", "gesture_scratch"),
        ("itch", "gesture_scratch"),
        ("stretch", "gesture_stretch"),
    ):
        if phrase in text:
            return phrase, rule
    return "gesture", "gesture_base"


def _sport_core(text: str) -> tuple[str, str]:
    for phrase, candidates in SPORT_CORE_RULES:
        if any(candidate in text for candidate in candidates):
            return phrase, f"sport_{phrase}"
    return "sport", "sport_base"


def _injured_core(text: str) -> tuple[str, str]:
    if "walk" in text:
        return "injured walk", "injured_walk"
    if any(word in text for word in ("jog", "run")):
        return "injured jog", "injured_jog"
    if "stand" in text:
        return "injured stand", "injured_stand"
    if "fall" in text:
        return "injured fall", "injured_fall"
    return "injured", "injured_base"


def _crutch_core(text: str) -> tuple[str, str]:
    if "walk" in text:
        return "crutch walk", "crutch_walk"
    if "stand" in text:
        return "crutch stand", "crutch_stand"
    return "crutch", "crutch_base"


def _step_over_core(text: str) -> tuple[str, str]:
    if "jump" in text:
        return "jump over", "step_over_jump_over"
    return "step over", "step_over_base"


def _other_core(text: str) -> tuple[str, str]:
    for phrase, rule in (
        ("high five", "other_high_five"),
        ("low five", "other_low_five"),
        ("handshake", "other_handshake"),
        ("fist to the chest", "other_fist_chest"),
        ("presenting something", "other_present"),
        ("talking on cellphone", "other_cellphone"),
        ("wandering around", "other_wander"),
        ("sliding forward", "other_slide_forward"),
        ("making an oath", "other_oath"),
        ("making a quip", "other_quip"),
    ):
        if phrase in text:
            return phrase, rule
    return text, "fallback_short"


def classify_core_category(text: str) -> str:
    if re.search(
        r"\bhip hop\b|\bdanc(?:e|ing|es)\b|\bchoreograph(?:y|ic)\b|\bmacarena\b|"
        r"\bvogue\b|\bsalsa\b|\bballet\b",
        text,
    ):
        return "dance"
    for category, patterns in CATEGORY_REGEXES:
        for pattern in patterns:
            if re.search(pattern, text):
                return category
    return "other"


def build_core_description(short_text: str) -> tuple[str, str, str]:
    text = normalize_text(short_text)
    if not text:
        return "", "empty", "other"

    for rule_name, phrase, core in DIRECT_CORE_RULES:
        if phrase_matches(text, phrase):
            return core, rule_name, classify_core_category(text)

    coarse = classify_core_category(text)
    if coarse == "walk":
        core, rule = _walk_core(text)
    elif coarse == "jog":
        core, rule = _jog_core(text)
    elif coarse == "jump":
        core, rule = _jump_core(text)
    elif coarse == "dance":
        core, rule = _dance_core(text)
    elif coarse == "climb":
        core, rule = _climb_core(text)
    elif coarse == "fall":
        core, rule = _fall_core(text)
    elif coarse == "crouch":
        core, rule = _crouch_core(text)
    elif coarse == "kneel":
        core, rule = "kneel down", "kneel_down"
    elif coarse == "sit":
        core, rule = _sit_core(text)
    elif coarse == "carry":
        core, rule = _carry_core(text)
    elif coarse == "reach":
        core, rule = "reach", "reach_base"
    elif coarse == "push":
        core, rule = _push_core(text)
    elif coarse == "step_over":
        core, rule = _step_over_core(text)
    elif coarse == "turn":
        core, rule = _turn_core(text)
    elif coarse == "idle":
        core, rule = _idle_core(text)
    elif coarse == "gesture":
        core, rule = _gesture_core(text)
    elif coarse == "sport":
        core, rule = _sport_core(text)
    elif coarse == "injured":
        core, rule = _injured_core(text)
    elif coarse == "crutch":
        core, rule = _crutch_core(text)
    else:
        core, rule = _other_core(text)

    return core, rule, coarse


def collect_samples(
    metadata_rows: list[dict[str, str]],
    temporal_map: dict[str, dict],
    *,
    sample_size: int,
    seed: int,
    include_mirrored: bool,
) -> tuple[list[SampleRecord], Counter]:
    pool = []
    for row in metadata_rows:
        if not include_mirrored and truthy(row.get("is_mirror", "")):
            continue
        temporal = temporal_map.get(row["filename"])
        if temporal is None:
            continue
        events = [
            str(event.get("description", "")).strip()
            for event in temporal.get("events", [])
            if normalize_text(event.get("description", ""))
        ]
        if not events:
            continue
        pool.append((row, events))

    rng = random.Random(seed)
    if sample_size > len(pool):
        sample_size = len(pool)
    sampled = rng.sample(pool, sample_size)

    records = []
    rule_counts = Counter()
    for row, events in sampled:
        short_text = str(row["content_short_description"]).strip()
        temporal_text = stable_choice(events, seed, row["filename"])
        core_text, core_rule, coarse = build_core_description(short_text)
        rule_counts[core_rule] += 1
        records.append(
            SampleRecord(
                filename=row["filename"],
                is_mirror=truthy(row.get("is_mirror", "")),
                short_description=short_text,
                temporal_description=temporal_text,
                core_description=core_text,
                core_rule=core_rule,
                coarse_category=coarse,
            )
        )

    return records, rule_counts


def encode_texts(
    texts: list[str],
    clip_model: torch.nn.Module,
    *,
    batch_size: int,
) -> torch.Tensor:
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        with torch.inference_mode():
            emb = encode_text(clip_model, batch)
            emb = F.normalize(emb.float(), dim=-1)
        embeddings.append(emb.detach())
    return torch.cat(embeddings, dim=0)


def summarize_pair_matrix(sim: np.ndarray) -> dict[str, float]:
    n = sim.shape[0]
    diag = np.diag(sim)
    off_mask = ~np.eye(n, dtype=bool)
    off = sim[off_mask]
    off_by_row = sim.copy()
    np.fill_diagonal(off_by_row, -np.inf)
    best_negative = np.max(off_by_row, axis=1)
    rank = 1 + np.sum(sim > diag[:, None], axis=1)
    margin = diag - best_negative
    return {
        "same_mean": float(np.mean(diag)),
        "same_median": float(np.median(diag)),
        "same_p10": float(np.percentile(diag, 10)),
        "same_p90": float(np.percentile(diag, 90)),
        "diff_mean": float(np.mean(off)),
        "diff_median": float(np.median(off)),
        "diff_p10": float(np.percentile(off, 10)),
        "diff_p90": float(np.percentile(off, 90)),
        "best_negative_mean": float(np.mean(best_negative)),
        "best_negative_median": float(np.median(best_negative)),
        "margin_mean": float(np.mean(margin)),
        "margin_median": float(np.median(margin)),
        "margin_p10": float(np.percentile(margin, 10)),
        "margin_p90": float(np.percentile(margin, 90)),
        "same_beats_best_negative_rate": float(np.mean(margin > 0)),
        "top1_recall": float(np.mean(rank == 1)),
        "top5_recall": float(np.mean(rank <= 5)),
        "mean_rank": float(np.mean(rank)),
    }


def write_csv(path: Path, rows: list[dict[str, str | float | int | bool]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_similarity_means(summary: dict[str, dict[str, float]], outpath: Path) -> None:
    labels = ["short vs temporal", "short vs core", "temporal vs core"]
    same = [summary[k]["same_mean"] for k in ("short_temporal", "short_core", "temporal_core")]
    diff = [summary[k]["diff_mean"] for k in ("short_temporal", "short_core", "temporal_core")]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, same, width, label="same motion", color="#2ca02c")
    ax.bar(x + width / 2, diff, width, label="different motion", color="#7f7f7f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("cosine similarity")
    ax.set_ylim(min(min(same), min(diff)) - 0.05, max(max(same), max(diff)) + 0.05)
    ax.set_title("BONES-SEED CLIP similarity by description pair")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_margin_scatter(
    summary: dict[str, dict[str, float]],
    sims: dict[str, np.ndarray],
    outpath: Path,
) -> None:
    names = [
        ("short_temporal", "short vs temporal"),
        ("short_core", "short vs core"),
        ("temporal_core", "temporal vs core"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=False)

    for ax, (key, title) in zip(axes, names):
        sim = sims[key]
        diag = np.diag(sim)
        off_by_row = sim.copy()
        np.fill_diagonal(off_by_row, -np.inf)
        best_negative = np.max(off_by_row, axis=1)
        above = diag > best_negative
        ax.scatter(
            best_negative,
            diag,
            s=14,
            c=np.where(above, "#2ca02c", "#d62728"),
            alpha=0.65,
            linewidths=0,
        )
        lo = float(min(best_negative.min(), diag.min()))
        hi = float(max(best_negative.max(), diag.max()))
        pad = max(0.02, (hi - lo) * 0.05)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="black", lw=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("best different-motion similarity")
        ax.set_ylabel("same-motion similarity")
        ax.set_title(title)
        ax.text(
            0.04,
            0.96,
            f"same>diff: {summary[key]['same_beats_best_negative_rate']*100:.1f}%\n"
            f"margin: {summary[key]['margin_mean']:+.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
        )
        ax.grid(alpha=0.2)

    fig.suptitle("BONES-SEED CLIP separation: same motion vs best different-motion match")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CLIP text similarity across short, temporal, and core Bones-SEED descriptions."
    )
    parser.add_argument("--metadata-csv", default=METADATA_CSV)
    parser.add_argument("--temporal-jsonl", default=TEMPORAL_JSONL)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--clip-version", default=DEFAULT_CLIP_VERSION)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=128)
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
    print(f"Sampling size: {args.sample_size}")
    print(f"Include mirrored rows: {args.include_mirrored}")
    print(f"CLIP version: {args.clip_version}")
    print(f"CLIP device: {device}")

    metadata_rows = load_metadata_rows(args.metadata_csv)
    temporal_map = load_temporal_map(args.temporal_jsonl)
    records, rule_counts = collect_samples(
        metadata_rows,
        temporal_map,
        sample_size=args.sample_size,
        seed=args.seed,
        include_mirrored=args.include_mirrored,
    )
    if not records:
        raise RuntimeError("No sample rows survived filtering")

    print(f"Sampled rows: {len(records)}")
    print(f"Unique core rules: {len(rule_counts)}")

    clip_model = load_and_freeze_clip(args.clip_version, device=device)

    short_texts = [r.short_description for r in records]
    temporal_texts = [r.temporal_description for r in records]
    core_texts = [r.core_description for r in records]

    short_emb = encode_texts(short_texts, clip_model, batch_size=args.batch_size)
    temporal_emb = encode_texts(temporal_texts, clip_model, batch_size=args.batch_size)
    core_emb = encode_texts(core_texts, clip_model, batch_size=args.batch_size)

    short_temporal_sim = (short_emb @ temporal_emb.T).cpu().numpy()
    short_core_sim = (short_emb @ core_emb.T).cpu().numpy()
    temporal_core_sim = (temporal_emb @ core_emb.T).cpu().numpy()

    summaries = {
        "short_temporal": summarize_pair_matrix(short_temporal_sim),
        "short_core": summarize_pair_matrix(short_core_sim),
        "temporal_core": summarize_pair_matrix(temporal_core_sim),
    }

    triplet_rows = []
    for idx, record in enumerate(records):
        triplet_rows.append(
            {
                "filename": record.filename,
                "is_mirror": record.is_mirror,
                "coarse_category": record.coarse_category,
                "core_rule": record.core_rule,
                "short_description": record.short_description,
                "temporal_description": record.temporal_description,
                "core_description": record.core_description,
                "short_temporal_similarity": float(short_temporal_sim[idx, idx]),
                "short_core_similarity": float(short_core_sim[idx, idx]),
                "temporal_core_similarity": float(temporal_core_sim[idx, idx]),
            }
        )
    write_csv(output_dir / "sampled_triplets.csv", triplet_rows)

    rule_rows = [
        {
            "core_rule": rule,
            "count": count,
            "example": next(
                (r.core_description for r in records if r.core_rule == rule),
                "",
            ),
        }
        for rule, count in rule_counts.most_common()
    ]
    write_csv(output_dir / "core_rule_counts.csv", rule_rows)

    # Worst core matches are the easiest places to refine the rule set.
    worst_idx = np.argsort(np.diag(short_core_sim))[:30]
    worst_rows = []
    for idx in worst_idx:
        worst_rows.append(
            {
                "filename": records[idx].filename,
                "short_description": records[idx].short_description,
                "temporal_description": records[idx].temporal_description,
                "core_description": records[idx].core_description,
                "core_rule": records[idx].core_rule,
                "short_core_similarity": float(short_core_sim[idx, idx]),
                "short_temporal_similarity": float(short_temporal_sim[idx, idx]),
                "temporal_core_similarity": float(temporal_core_sim[idx, idx]),
            }
        )
    write_csv(output_dir / "worst_core_matches.csv", worst_rows)

    summary_path = output_dir / "similarity_summary.txt"
    lines = []
    lines.append("=" * 72)
    lines.append("BONES-SEED CLIP Text Similarity")
    lines.append("=" * 72)
    lines.append(f"Metadata CSV          : {args.metadata_csv}")
    lines.append(f"Temporal JSONL        : {args.temporal_jsonl}")
    lines.append(f"Sample size           : {len(records)}")
    lines.append(f"Seed                  : {args.seed}")
    lines.append(f"Include mirrored rows : {args.include_mirrored}")
    lines.append(f"CLIP version          : {args.clip_version}")
    lines.append(f"CLIP device           : {device}")
    lines.append(f"Unique core rules     : {len(rule_counts)}")
    lines.append("")
    lines.append("Pairwise similarity summary:")
    lines.append(
        f"{'pair':<18} {'same_mean':>10} {'same_med':>10} {'diff_mean':>10} "
        f"{'diff_med':>10} {'margin':>10} {'top1':>8} {'top5':>8}"
    )
    for key, label in (
        ("short_temporal", "short-temporal"),
        ("short_core", "short-core"),
        ("temporal_core", "temporal-core"),
    ):
        s = summaries[key]
        lines.append(
            f"{label:<18} {s['same_mean']:>10.3f} {s['same_median']:>10.3f} "
            f"{s['diff_mean']:>10.3f} {s['diff_median']:>10.3f} "
            f"{s['margin_mean']:>10.3f} {s['top1_recall']*100:>7.1f}% "
            f"{s['top5_recall']*100:>7.1f}%"
        )
    lines.append("")
    lines.append("Core rule counts:")
    for rule, count in rule_counts.most_common(20):
        lines.append(f"  {rule:<24} {count:>5}")
    lines.append("")
    lines.append("Worst short-core same-motion matches:")
    for row in worst_rows[:15]:
        lines.append(
            f"  {row['short_core_similarity']:+.3f} | {row['core_rule']:<18} | "
            f"{row['short_description']} -> {row['core_description']}"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot_similarity_means(summaries, output_dir / "similarity_means.png")
    plot_margin_scatter(
        summaries,
        {
            "short_temporal": short_temporal_sim,
            "short_core": short_core_sim,
            "temporal_core": temporal_core_sim,
        },
        output_dir / "similarity_margin_scatter.png",
    )

    print(f"Summary saved: {summary_path}")
    print(f"Triplets saved: {output_dir / 'sampled_triplets.csv'}")
    print(f"Core rules saved: {output_dir / 'core_rule_counts.csv'}")
    print(f"Worst cases saved: {output_dir / 'worst_core_matches.csv'}")
    print(f"Plot saved: {output_dir / 'similarity_means.png'}")
    print(f"Plot saved: {output_dir / 'similarity_margin_scatter.png'}")


if __name__ == "__main__":
    main()
