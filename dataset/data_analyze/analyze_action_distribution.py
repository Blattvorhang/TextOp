#!/usr/bin/env python3
"""
BONES-SEED Action Name Distribution Analysis
=============================================
Read action labels from the BONES-SEED metadata CSV and classify them at two levels:

  Fine-grained: metadata/content_short_description               → ~4000 classes
  Core: short action phrase extracted from content_short_description → ~100 classes

Outputs:
  - action_distribution_bar.png       : fine-grained Top-50 bar chart
  - action_core_bar.png               : core distribution bar chart
  - action_core_wordcloud.png         : core word cloud
  - action_short_description_wordcloud.png      : all short-description labels
  - action_statistics.txt             : full two-level report with task annotation

Usage:  python analyze_action_distribution.py
"""

import csv
import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.data_process.filter_and_copy_bones_data import (  # noqa: E402
    DEFAULT_FILTER_KEYWORDS,
    should_filter_out,
)
from dataset.data_analyze.analyze_text_description_similarity import (  # noqa: E402
    build_core_description,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
METADATA_CSV = "/home/lenovo/data/bones-seed/metadata/seed_metadata_v004.csv"
DEFAULT_FILTERED_MOTION_DIR = REPO_ROOT / "data/motion_lib_filtered"
SHORT_DESCRIPTION_COLUMNS = (
    "content_short_description",
    "content_short_description_2",
)
PRIMARY_SHORT_DESCRIPTION_COLUMN = "content_short_description"
STRIP_SHORT_DESCRIPTION_DIGITS = True
SPATIAL_SHORT_DESCRIPTION_WORDS = ("left", "right")
STRIP_SHORT_DESCRIPTION_SPATIAL_WORDS = True
SPATIAL_ANGLE_TOKEN_RE = re.compile(r"\b(?:000|045|090|135|180|225|270|315|360)\b")
SPATIAL_DISTANCE_TOKEN_RE = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s|-)*(?:cm|m|meter|meters)\b"
)
VERSION_OPTION_TOKEN_RE = re.compile(
    r"(?<![a-z0-9])(?:opt|option)[_\s-]*\d+(?![a-z0-9])"
    r"|(?<![a-z0-9])v\d+(?![a-z0-9])"
)
SKIP_MIRRORED = True
OUTPUT_DIR = Path(__file__).resolve().parent / "analysis_output" / "action_distribution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
if not os.path.exists(FONT_PATH):
    FONT_PATH = None  # fall back to wordcloud default


# ===================================================================
# Task-priority annotation
# ===================================================================
# Each coarse category is assigned one of three priority levels for the
# LDM planner's objectives:
#
#   Goal #1 — Goal-driven navigation (walk, jog, turn toward sub-goals)
#   Goal #2 — Scene-aware obstacle avoidance (crouch, step_over, climb)
#   Goal #3 — Scene interaction             (sit, kneel, reach, push)
#
# task_critical:   directly required by the planner objectives
# task_relevant:   useful but not core (hard to track, secondary, or rare)
# task_irrelevant: stationary or non-goal-directed motion
#
# Note: jump is downgraded to task_relevant — it is hard for the universal
# controller to track even if the motion generator produces it, and is not
# the focus of our work.

TASK_CRITICAL = "task_critical"
TASK_RELEVANT = "task_relevant"
TASK_IRRELEVANT = "task_irrelevant"

# Implementation stage annotation (progressive roadmap):
#   Stage 1 — Walk to goal & stop          (basic locomotion)
#   Stage 2 — Body-level obstacle avoidance (torso height adaptation)
#   Stage 3 — Limb-level obstacle negotiation (step over)
#   Stage 4 — Scene interaction            (sit, push)
STAGE_1 = 1  # walk, jog, turn, idle
STAGE_2 = 2  # crouch, kneel
STAGE_3 = 3  # step_over
STAGE_4 = 4  # sit, push

NAV_PRIORITY = {
    # ---- Stage 1: Walk to goal & stop ----
    "walk":      (TASK_CRITICAL, STAGE_1),    # reaching sub-goals through locomotion
    "jog":       (TASK_CRITICAL, STAGE_1),    # faster locomotion to distant goals
    "turn":      (TASK_CRITICAL, STAGE_1),    # changing facing direction toward sub-goal
    "idle":      (TASK_CRITICAL, STAGE_1),    # start/end states, stopping at goal

    # ---- Stage 2: Body-level obstacle avoidance ----
    "crouch":    (TASK_CRITICAL, STAGE_2),    # duck / crawl under low obstacles
    "kneel":     (TASK_CRITICAL, STAGE_2),    # low posture, body height adaptation

    # ---- Stage 3: Limb-level obstacle negotiation ----
    "step_over": (TASK_CRITICAL, STAGE_3),    # stepping over small ground objects

    # ---- Stage 4: Scene interaction ----
    "sit":       (TASK_CRITICAL, STAGE_4),    # sitting on chairs from standing

    # ---- De-emphasized (tracker can't execute, or occupancy can't express) ----
    "jump":      (TASK_RELEVANT, 0),          # vertical motion — tracker can't reliably execute
    "climb":     (TASK_RELEVANT, 0),          # vertical traversal — same as jump
    "push":      (TASK_RELEVANT, 0),          # doors invisible to occupancy grid — can't distinguish wall/door/object
    "carry":     (TASK_RELEVANT, 0),          # COM shifted by held object weight
    "reach":     (TASK_RELEVANT, 0),          # not a core focus
    "fall":      (TASK_RELEVANT, 0),          # recovery from unplanned fall

    # ---- Non-goal-directed ----
    "gesture":   (TASK_IRRELEVANT, 0),        # upper-body only, no root motion
    "dance":     (TASK_IRRELEVANT, 0),        # rhythmic, not goal-directed
    "injured":   (TASK_IRRELEVANT, 0),        # special asymmetric gait
    "sport":     (TASK_IRRELEVANT, 0),        # specific sports motions
    "crutch":    (TASK_IRRELEVANT, 0),        # assistive device locomotion
    "other":     (TASK_IRRELEVANT, 0),        # unmatched edge cases
}

NAV_PRIORITY_ORDER = [TASK_CRITICAL, TASK_RELEVANT, TASK_IRRELEVANT]
NAV_PRIORITY_COLORS = {
    TASK_CRITICAL:    "#d62728",  # red   — must up-weight
    TASK_RELEVANT:    "#ff7f0e",  # orange — consider up-weight
    TASK_IRRELEVANT:  "#7f7f7f",  # grey  — baseline or down-weight
}
NAV_PRIORITY_LABEL = {
    TASK_CRITICAL:    "task-critical (navigation + obstacle avoidance + interaction)",
    TASK_RELEVANT:    "task-relevant (secondary / hard-to-track / rare)",
    TASK_IRRELEVANT:  "task-irrelevant (stationary or non-goal-directed)",
}


# ===================================================================
# Coarse-grained classification: metadata label → category
# ===================================================================
# Rules are checked in priority order; first keyword match wins.
#
# Classification principles (navigation + scene interaction focus):
#   - Distinguish root motion pattern (translate vs rotate vs jump vs static)
#   - Distinguish body height layer (stand / crouch / kneel / sit / lie)
#   - Distinguish COM-affecting interactions (carry / reach / push / climb)

COARSE_RULES = [
    # ---- Special motion modes (match first) ----
    ("injured",       ["injured"]),
    ("crutch",        ["crutch", "crutches"]),

    # ---- Jump (significant vertical root displacement) ----
    ("jump",          ["jump", "hop", "leap", "flip", "vault_over",
                       "jump_and_land", "jump_over", "jump_twice",
                       "high_jump", "jump_ff", "jump_sideway",
                       "jump_and_down", "turn_jump", "fire_in_the_hole",
                       "jumping", "vaulting"]),

    # ---- Jog / run ----
    ("jog",           ["jog", "jogging", "run_", "running"]),

    # ---- Walk (must precede "turn" — turn_walk is locomotion, not pure turn) ----
    ("walk",          ["walk", "moonwalk", "step_forward", "step_backward",
                       "stroll", "strolling", "stride", "advancing",
                       "lateral step", "slide forward", "sliding around",
                       "skipping"]),

    # ---- Dance (rhythmic full-body) ----
    ("dance",         ["dance", "dancing", "choreography", "macarena",
                       "dancecard", "expressionism", "krakowiak",
                       "gangnam", "ballet"]),

    # ---- Climb (significant vertical root displacement) ----
    ("climb",         ["climb", "ladder", "come_up_", "come_down_",
                       "crouch_cupboard", "get up onto", "get down from",
                       "onto 50 cm box", "from 50 cm box",
                       "onto 50cm box", "from 50cm box"]),

    # ---- Fall / lie on ground ----
    ("fall",          ["fall", "faint", "toxic_gas", "postmortem",
                       "death", "lying", "lie_", "flying_",
                       "stand_up_lying", "on_ground", "roll forward",
                       "roll to a side", "doing a roll"]),

    # ---- Crouch / crawl ----
    ("crouch",        ["crouch", "crawl", "on_all_fours",
                       "crouch_idle", "crouch_walk", "stoop",
                       "bend forward", "squat", "plank"]),

    # ---- Kneel / sit on heels ----
    ("kneel",         ["kneel", "sit_on_heels"]),

    # ---- Sit ----
    ("sit",           ["sitting", "sit_cross", "sit_",
                       "read_newspaper_sitting", "eat_hotdog_sitting",
                       "play_guitar_sitting", "having_a_sit"]),

    # ---- Carry / lift (COM shifted by held object) ----
    ("carry",         ["carry", "lift", "crate", "heavy_", "light_",
                       "pick_up", "put_down", "hold_",
                       "moving_object", "pass_",
                       "item_give", "item_take", "item_pick", "item_put",
                       "item_switch", "item_hold",
                       "lasso_catch", "lasso_dance", "lasso_pull",
                       "watering_plants", "walk_the_dog",
                       "medium_big", "small_heavy", "small_light",
                       "big_heavy", "big_light",
                       "medium_heavy", "medium_light",
                       "putting something", "dropping items",
                       "picking something up", "taking an object",
                       "fridge", "splitting wood", "sledgehammer",
                       "digging", "sweeping", "washing floor",
                       "dish washing", "pounding meat", "nailing",
                       "grates vegetables", "grinds pepper",
                       "making fried eggs", "tossing baby"]),

    # ---- Reach (upper body extension, COM offset) ----
    ("reach",         ["reach", "reaching"]),

    # ---- Push / pull (hands interacting with fixed environment point) ----
    ("push",          ["push", "pull", "crank", "valve", "handle", "lever",
                       "_knob_", "door_", "shut", "slam",
                       "open_walk", "close_",
                       "horizontal_lever", "vertical_lever",
                       "neutral_button", "operating"]),

    # ---- Step over obstacles ----
    ("step_over",     ["step_over", "step_in", "avoid_obstacle",
                       "bump_into", "jump_over_obstacle", "neutral_avoid",
                       "stepping in"]),

    # ---- Pure turn in place (no significant root translation) ----
    # Note: turn_walk / turn_jog are caught by walk/jog above first
    ("turn",          ["turn_handstand", "mohak", "step_rotate",
                       "idle_turn", "spin_", "turn back", "turns back",
                       "turn forward", "turn around"]),

    # ---- Idle / standing (root nearly stationary) ----
    ("idle",          ["idle", "stand", "standing", "legs_relax",
                       "looking_around", "looking_in_the_mirror",
                       "look_around", "looking_R", "looking_",
                       "neutral_sit", "neutral_stand",
                       "neutral_idle", "neutral_laugh", "neutral_fear",
                       "neutral_cry", "neutral_looking",
                       "neutral_dancecard_idle", "neutral_dancecard_looking",
                       "idle_hands", "idle_one_foot",
                       "idle_to_", "one_leg_idle", "relaxing legs",
                       "neutral stance", "immobility",
                       "focuses before start"]),

    # ---- Gesture / expression (upper-body only, root stays still) ----
    ("gesture",       ["wave", "salute", "clap", "cheer", "triumph",
                       "thumbs", "point", "welcome", "greet", "bye",
                       "raise_your_hand", "show_", "shhh", "rock_out",
                       "mic_drop", "count_it", "i_got_this", "eureka",
                       "fist_pump", "bicep", "body_check",
                       "pray", "cross_your", "no_see", "no_hear",
                       "lament", "scream", "confusion", "think",
                       "don_t_know", "omg_", "yawn", "listen",
                       "checking_time", "looking_at",
                       "itching", "scratch", "brush_of_dust",
                       "dusting", "wipe", "rub", "fixing",
                       "body_stretch", "body_search", "pocket_search",
                       "freezing_cold", "shiver", "cough", "sneeze",
                       "puke", "boss_dust", "dust_brushing",
                       "chefs_kiss", "shoulder_clap", "step_in_shit",
                       "meditate", "horse_riding", "shuffle_cards",
                       "drinking_bottle", "eat_burger",
                       "zippo", "smoke", "drink_",
                       "clear_ear", "rubbing_",
                       "neutral_cry", "neutral_fear",
                       "rage_", "proud_", "neutral_laugh",
                       "looking_in_the_mirror",
                       "wiping_shoes", "maybe", "just_realised",
                       "tasty", "no_speak", "no_say",
                       "on_the_edge", "eating", "painting",
                       "stinky", "binoculars", "brush_off",
                       "tarzan", "cry_", "laugh",
                       "welcom", "clear", "alone",
                       "playing", "grating", "peeling", "looting",
                       "chainsaw", "cutting",
                       "checking whole body", "fixes something",
                       "being confused", "brushing dust",
                       "boss brushing", "face palming", "chef",
                       "freezes from the cold", "not seeing",
                       "not hearing", "not speaking", "confident gesture",
                       "just realized", "puking", "belly massage",
                       "sneezing", "acting disgusted", "threatening",
                       "warming up", "raising fist", "kiss sending",
                       ]),

    # ---- Sport motions ----
    ("sport",         ["swim", "throw_", "catch_", "kick_", "punch_",
                       "dodge_", "play_tennis", "play_guitar",
                       "petting_dog", "dribble", "shoot_",
                       "ib_combat", "ib_dodge", "exercise",
                       "cartwheel", "shadow boxing", "boxing",
                       "ride a horse"]),
]

# Neutral-prefix patterns (neutral_*): most fall through to idle,
# but specific sub-patterns may be captured by rules above
# (e.g. neutral_avoid → step_over, neutral_button → push)


def normalize_label(label: str) -> str:
    """Normalize a metadata label for counting/display without changing word order."""
    return " ".join(str(label).strip().lower().split())


def strip_all_digits_from_label(label: str) -> str:
    """Remove all digits; used only to audit possible over-merging."""
    return normalize_label(re.sub(r"\d+", "", label))


def strip_digits_from_label(label: str) -> str:
    """Remove numeric specificity from a normalized label."""
    label = VERSION_OPTION_TOKEN_RE.sub("", label)
    label = SPATIAL_DISTANCE_TOKEN_RE.sub("", label)
    label = SPATIAL_ANGLE_TOKEN_RE.sub("", label)
    return normalize_label(label)


def strip_spatial_words_from_label(
    label: str,
    spatial_words: tuple[str, ...] = SPATIAL_SHORT_DESCRIPTION_WORDS,
) -> str:
    """Remove coarse spatial words from a normalized label."""
    if not spatial_words:
        return label
    word_pattern = "|".join(re.escape(word) for word in spatial_words)
    direction_phrase_pattern = (
        r"\b(?:to|toward|towards)\s+(?:the\s+)?(?:"
        + word_pattern
        + r")\b(?:[,.;:]|\s*$)?"
    )
    label = re.sub(direction_phrase_pattern, "", label)

    standalone_pattern = r"\b(?:" + word_pattern + r")\b"
    label = re.sub(standalone_pattern, "", label)

    dangling_direction_pattern = r"\b(?:to|toward|towards)\s+(?:the\s*)?$"
    label = re.sub(dangling_direction_pattern, "", label)
    return normalize_label(label)


def normalize_short_description(
    label: str,
    strip_digits: bool = True,
    strip_spatial_words: bool = True,
    spatial_words: tuple[str, ...] = SPATIAL_SHORT_DESCRIPTION_WORDS,
) -> str:
    """Normalize a short-description label for analysis."""
    label = normalize_label(label)
    if strip_digits:
        label = strip_digits_from_label(label)
    if strip_spatial_words:
        label = strip_spatial_words_from_label(label, spatial_words)
    return label


def keyword_text_forms(text: str) -> tuple[str, str]:
    """Return space- and underscore-normalized forms for keyword matching."""
    space_form = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
    underscore_form = space_form.replace(" ", "_")
    return space_form, underscore_form


def classify_coarse(label: str) -> str:
    """Classify a fine-grained metadata label into a coarse category."""
    label_space, label_underscore = keyword_text_forms(label)

    for category, keywords in COARSE_RULES:
        for kw in keywords:
            kw_space, kw_underscore = keyword_text_forms(kw)
            if (
                kw_space
                and (kw_space in label_space or kw_underscore in label_underscore)
            ):
                return category

    return "other"


def get_nav_priority(coarse_name: str) -> str:
    """Return the task priority level for a coarse category."""
    entry = NAV_PRIORITY.get(coarse_name, (TASK_IRRELEVANT, 0))
    return entry[0] if isinstance(entry, tuple) else entry


def get_stage(coarse_name: str) -> int:
    """Return the implementation stage (1–4) for a task-critical category, 0 otherwise."""
    entry = NAV_PRIORITY.get(coarse_name, (TASK_IRRELEVANT, 0))
    return entry[1] if isinstance(entry, tuple) else 0


# ===================================================================
# Data collection
# ===================================================================
def is_truthy_metadata_value(value: str) -> bool:
    """Parse bool-ish metadata values such as True, 1, or 1.0."""
    return str(value).strip().lower() in {"true", "1", "1.0", "yes"}


def get_short_description_label_pairs(
    row: dict[str, str],
    *,
    strip_digits: bool = STRIP_SHORT_DESCRIPTION_DIGITS,
    strip_spatial_words: bool = STRIP_SHORT_DESCRIPTION_SPATIAL_WORDS,
    spatial_words: tuple[str, ...] = SPATIAL_SHORT_DESCRIPTION_WORDS,
) -> list[tuple[str, str]]:
    """Return raw + normalized short-description labels from a metadata row."""
    label_pairs = []
    for col in SHORT_DESCRIPTION_COLUMNS:
        raw_label = normalize_label(row.get(col, ""))
        if not raw_label:
            continue
        label = normalize_short_description(
            raw_label,
            strip_digits,
            strip_spatial_words,
            spatial_words,
        )
        if label:
            label_pairs.append((raw_label, label))
    return label_pairs


def read_include_keywords(filter_file: str | None) -> list[str] | None:
    """Read include keywords using the same line-based format as filter_and_copy."""
    if filter_file is None:
        return None
    with open(filter_file, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def metadata_row_to_filter_name(row: dict[str, str]) -> str:
    """Build the parent/basename string used by filter_and_copy_bones_data."""
    move_g1_path = row.get("move_g1_path", "")
    if move_g1_path:
        path = Path(move_g1_path)
        parent = path.parent.name
        base = f"{path.stem}.pkl"
    else:
        parent = row.get("take_date", "")
        base = f"{row.get('filename', '')}.pkl"

    return f"{parent}/{base}" if parent else base


def strip_aug_suffix(stem: str) -> str:
    """Map augmented motion-lib filenames back to metadata filenames."""
    return re.sub(r"_aug_\d+$", "", stem)


def load_filtered_motion_filename_counts(filtered_motion_dir: str | Path) -> Counter:
    """Load motion filename counts from an existing filtered motion-lib directory."""
    root = Path(filtered_motion_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"filtered motion directory not found: {root}")

    filenames = Counter()
    for path in root.rglob("*.pkl"):
        if path.name == "metadata.pkl":
            continue
        filenames[strip_aug_suffix(path.stem)] += 1
    return filenames


def summarize_normalization_collisions(norm_to_raw: dict[str, set[str]]) -> dict:
    """Summarize raw labels that collapse to the same normalized label."""
    collisions = [
        (normalized, raw_labels)
        for normalized, raw_labels in norm_to_raw.items()
        if len(raw_labels) > 1
    ]
    collisions.sort(key=lambda item: (-len(item[1]), item[0]))
    return {
        "normalized_unique": len(norm_to_raw),
        "colliding_normalized_labels": len(collisions),
        "raw_labels_in_collisions": sum(len(raw_labels) for _, raw_labels in collisions),
        "examples": [
            (normalized, sorted(raw_labels)[:12], len(raw_labels))
            for normalized, raw_labels in collisions[:12]
        ],
    }


def collect_all(
    metadata_csv: str,
    *,
    skip_mirrored: bool = SKIP_MIRRORED,
    filename_filter_mode: str = "none",
    filtered_motion_dir: str | Path = DEFAULT_FILTERED_MOTION_DIR,
    filter_keywords: list[str] | None = None,
    include_keywords: list[str] | None = None,
    strip_label_digits: bool = STRIP_SHORT_DESCRIPTION_DIGITS,
    strip_label_spatial_words: bool = STRIP_SHORT_DESCRIPTION_SPATIAL_WORDS,
    spatial_label_words: tuple[str, ...] = SPATIAL_SHORT_DESCRIPTION_WORDS,
):
    """Read metadata CSV labels, classify fine + coarse labels, return counters."""
    fine_counter = Counter()
    core_counter = Counter()
    core_rule_counter = Counter()
    wordcloud_counter = Counter()
    core_wordcloud_counter = Counter()

    if filter_keywords is None:
        filter_keywords = list(DEFAULT_FILTER_KEYWORDS)

    allowed_filename_counts = None
    if filename_filter_mode == "filtered-dir":
        allowed_filename_counts = load_filtered_motion_filename_counts(filtered_motion_dir)
    elif filename_filter_mode not in {"none", "keywords"}:
        raise ValueError(f"unknown filename filter mode: {filename_filter_mode}")

    all_digit_norm_to_raw = defaultdict(set)
    selected_number_norm_to_raw = defaultdict(set)
    final_norm_to_raw = defaultdict(set)

    audit = {
        "metadata_csv": metadata_csv,
        "metadata_rows": 0,
        "analyzed_rows": 0,
        "analyzed_metadata_rows": 0,
        "raw_short_description_count": Counter(),
        "short_description_count": Counter(),
        "rows_with_duplicate_raw_short_descriptions": 0,
        "rows_with_duplicate_normalized_short_descriptions": 0,
        "rows_missing_primary_short_description": 0,
        "skipped_mirrored_rows": 0,
        "filename_filter_mode": filename_filter_mode,
        "filtered_motion_dir": str(filtered_motion_dir),
        "filtered_motion_file_count": sum((allowed_filename_counts or Counter()).values()),
        "allowed_filename_count": len(allowed_filename_counts or ()),
        "augmented_motion_file_count": sum(
            max(count - 1, 0)
            for count in (allowed_filename_counts or Counter()).values()
        ),
        "allowed_filenames_missing_metadata": 0,
        "rows_skipped_by_filename_filter": 0,
        "filter_keyword_count": len(filter_keywords),
        "filter_keywords": list(filter_keywords),
        "include_keyword_count": len(include_keywords or ()),
        "skip_mirrored": skip_mirrored,
        "strip_label_digits": strip_label_digits,
        "strip_label_spatial_words": strip_label_spatial_words,
        "spatial_label_words": tuple(spatial_label_words),
    }

    metadata_filenames = set()

    with open(metadata_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = set(SHORT_DESCRIPTION_COLUMNS) | {
            "filename",
            "is_mirror",
            "move_g1_path",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"metadata CSV missing required columns: {missing}")

        for row in reader:
            audit["metadata_rows"] += 1
            metadata_filenames.add(row["filename"])

            raw_labels = [
                normalize_label(row.get(col, ""))
                for col in SHORT_DESCRIPTION_COLUMNS
                if normalize_label(row.get(col, ""))
            ]
            for raw_label in raw_labels:
                all_digit_norm_to_raw[strip_all_digits_from_label(raw_label)].add(raw_label)
                selected_number_norm_to_raw[strip_digits_from_label(raw_label)].add(raw_label)
                final_label = normalize_short_description(
                    raw_label,
                    strip_label_digits,
                    strip_label_spatial_words,
                    spatial_label_words,
                )
                final_norm_to_raw[final_label].add(raw_label)

            audit["raw_short_description_count"][len(raw_labels)] += 1
            if len(set(raw_labels)) < len(raw_labels):
                audit["rows_with_duplicate_raw_short_descriptions"] += 1

            label_pairs = get_short_description_label_pairs(
                row,
                strip_digits=strip_label_digits,
                strip_spatial_words=strip_label_spatial_words,
                spatial_words=spatial_label_words,
            )
            labels = [label for _, label in label_pairs]
            audit["short_description_count"][len(labels)] += 1

            if len(set(labels)) < len(labels):
                audit["rows_with_duplicate_normalized_short_descriptions"] += 1

            raw_primary = normalize_label(row.get(PRIMARY_SHORT_DESCRIPTION_COLUMN, ""))
            primary = normalize_short_description(
                raw_primary,
                strip_label_digits,
                strip_label_spatial_words,
                spatial_label_words,
            )
            if not raw_primary:
                audit["rows_missing_primary_short_description"] += 1

            if allowed_filename_counts is not None and row["filename"] not in allowed_filename_counts:
                audit["rows_skipped_by_filename_filter"] += 1
                continue

            if filename_filter_mode == "keywords" and should_filter_out(
                metadata_row_to_filter_name(row),
                filter_keywords,
                include_keywords,
            ):
                audit["rows_skipped_by_filename_filter"] += 1
                continue

            if skip_mirrored and is_truthy_metadata_value(row.get("is_mirror", "")):
                audit["skipped_mirrored_rows"] += 1
                continue

            if not primary or not raw_primary:
                continue

            weight = allowed_filename_counts[row["filename"]] if allowed_filename_counts else 1
            audit["analyzed_rows"] += weight
            audit["analyzed_metadata_rows"] += 1

            fine = primary
            core_description, core_rule, _ = build_core_description(raw_primary)

            fine_counter[fine] += weight
            core_counter[core_description] += weight
            core_rule_counter[core_rule] += weight

            for raw_label, label in label_pairs:
                wordcloud_counter[label] += weight
                core_label, _, _ = build_core_description(raw_label)
                core_wordcloud_counter[core_label] += weight

    if allowed_filename_counts is not None:
        audit["allowed_filenames_missing_metadata"] = len(
            set(allowed_filename_counts) - metadata_filenames
        )

    audit["all_digit_normalization_collision_audit"] = summarize_normalization_collisions(
        all_digit_norm_to_raw
    )
    audit["selected_number_normalization_collision_audit"] = summarize_normalization_collisions(
        selected_number_norm_to_raw
    )
    audit["final_normalization_collision_audit"] = summarize_normalization_collisions(
        final_norm_to_raw
    )

    return (
        fine_counter,
        core_counter,
        wordcloud_counter,
        core_wordcloud_counter,
        core_rule_counter,
        audit,
    )


# ===================================================================
# Plotting
# ===================================================================
def plot_fine_bar(counter: Counter, total: int, top_n: int = 50):
    """Fine-grained Top-N horizontal bar chart."""
    top = counter.most_common(top_n)
    names, counts = zip(*top)

    fig, ax = plt.subplots(figsize=(16, 10))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(names)))
    ax.barh(range(len(names)), counts, color=colors, edgecolor="white", alpha=0.9)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6.5, fontfamily="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("Number of motions", fontsize=12)
    ax.set_title(
        f"BONES-SEED content_short_description Distribution — Top {top_n}",
        fontsize=14,
    )

    for i, (n, c) in enumerate(zip(names, counts)):
        ax.text(c + max(counts) * 0.005, i, f"  {c:,}", va="center",
                fontsize=5.5, fontfamily="monospace", color="dimgray")

    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    outpath = OUTPUT_DIR / "action_distribution_bar.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fine bar]       Saved: {outpath}")
    return outpath


def plot_core_bar(counter: Counter, total: int, top_n: int = 50):
    """Core-description Top-N horizontal bar chart."""
    top = counter.most_common(top_n)
    names, counts = zip(*top)

    fig, ax = plt.subplots(figsize=(16, 10))
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(names)))
    ax.barh(range(len(names)), counts, color=colors, edgecolor="white", alpha=0.9)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6.5, fontfamily="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("Number of motions", fontsize=12)
    ax.set_title(f"BONES-SEED Core Description Distribution — Top {top_n}", fontsize=14)

    for i, (n, c) in enumerate(zip(names, counts)):
        pct = c / total * 100
        ax.text(c + max(counts) * 0.005, i, f"  {c:,} ({pct:.1f}%)",
                va="center", fontsize=5.5, fontfamily="monospace", color="dimgray")

    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    outpath = OUTPUT_DIR / "action_core_bar.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Core bar]       Saved: {outpath}")
    return outpath


def plot_nav_priority_bar(counter: Counter, total: int):
    """Grouped bar chart: nav-critical + nav-relevant categories only,
    with a stacked comparison showing their share of total data.
    """
    # Collect categories by priority
    groups = {p: [] for p in NAV_PRIORITY_ORDER}
    for name, cnt in counter.most_common():
        p = get_nav_priority(name)
        groups[p].append((name, cnt))

    # Build a two-panel figure: left = nav-critical, right = nav-relevant
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    for ax, priority, title in [
        (ax1, TASK_CRITICAL, "Task-Critical\n(navigation + obstacle avoidance + interaction)"),
        (ax2, TASK_RELEVANT, "Task-Relevant\n(secondary — jump is hard to track, carry/fall are rare)"),
    ]:
        items = sorted(groups[priority], key=lambda x: -x[1])
        names = [n for n, _ in items]
        counts = [c for _, c in items]
        pcts = [c / total * 100 for c in counts]
        color = NAV_PRIORITY_COLORS[priority]

        bars = ax.bar(range(len(names)), counts, color=color, edgecolor="white", alpha=0.85)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=10, fontfamily="monospace", rotation=30, ha="right")
        ax.set_ylabel("Number of motions", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")

        for i, (n, c, pct) in enumerate(zip(names, counts, pcts)):
            ax.text(i, c + max(counts) * 0.02, f"{c:,}\n({pct:.1f}%)",
                    ha="center", fontsize=7.5, fontfamily="monospace", color="dimgray")

        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("BONES-SEED: Task-Relevant Action Categories (targets for up-weighting)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    outpath = OUTPUT_DIR / "action_coarse_nav_bar.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Nav bar]        Saved: {outpath}")
    return outpath


def plot_core_wordcloud(counter: Counter):
    """Word cloud of all core descriptions."""
    freq_dict = dict(counter)

    wc_kwargs = dict(
        width=1600,
        height=900,
        background_color="white",
        colormap="tab20",
        max_words=50,
        relative_scaling=0.5,
        min_font_size=12,
        random_state=42,
    )
    if FONT_PATH:
        wc_kwargs["font_path"] = FONT_PATH

    wc = WordCloud(**wc_kwargs)
    wc.generate_from_frequencies(freq_dict)

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("BONES-SEED Core Descriptions (all)", fontsize=14, pad=15)

    fig.tight_layout()
    outpath = OUTPUT_DIR / "action_core_wordcloud.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Core wordcloud] Saved: {outpath}")
    return outpath


def plot_nav_wordcloud(counter: Counter):
    """Word cloud of task-critical + task-relevant categories only."""
    nav_cats = {name: cnt for name, cnt in counter.items()
                if get_nav_priority(name) != TASK_IRRELEVANT}
    freq_dict = dict(nav_cats)

    wc_kwargs = dict(
        width=1600,
        height=900,
        background_color="white",
        colormap="OrRd",
        max_words=50,
        relative_scaling=0.5,
        min_font_size=14,
        random_state=42,
    )
    if FONT_PATH:
        wc_kwargs["font_path"] = FONT_PATH

    wc = WordCloud(**wc_kwargs)
    wc.generate_from_frequencies(freq_dict)

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("BONES-SEED: Task-Critical + Task-Relevant Categories (non-gesture/dance/sport)",
                 fontsize=14, pad=15)

    fig.tight_layout()
    outpath = OUTPUT_DIR / "action_nav_wordcloud.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Wordcloud nav]  Saved: {outpath}")
    return outpath


def plot_short_description_wordcloud(counter: Counter):
    """Word cloud of all short-description labels from metadata."""
    freq_dict = dict(counter)

    wc_kwargs = dict(
        width=1800,
        height=1000,
        background_color="white",
        colormap="tab20",
        max_words=200,
        relative_scaling=0.45,
        min_font_size=10,
        random_state=42,
    )
    if FONT_PATH:
        wc_kwargs["font_path"] = FONT_PATH

    wc = WordCloud(**wc_kwargs)
    wc.generate_from_frequencies(freq_dict)

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        "BONES-SEED Short Description Labels "
        "(content_short_description + content_short_description_2)",
        fontsize=14,
        pad=15,
    )

    fig.tight_layout()
    outpath = OUTPUT_DIR / "action_short_description_wordcloud.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Wordcloud short] Saved: {outpath}")
    return outpath


def plot_nav_short_description_wordcloud(counter: Counter):
    """Word cloud of task-critical + task-relevant short-description labels."""
    freq_dict = dict(counter)

    wc_kwargs = dict(
        width=1800,
        height=1000,
        background_color="white",
        colormap="OrRd",
        max_words=200,
        relative_scaling=0.45,
        min_font_size=10,
        random_state=42,
    )
    if FONT_PATH:
        wc_kwargs["font_path"] = FONT_PATH

    wc = WordCloud(**wc_kwargs)
    wc.generate_from_frequencies(freq_dict)

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        "BONES-SEED Task-Relevant Short Description Labels",
        fontsize=14,
        pad=15,
    )

    fig.tight_layout()
    outpath = OUTPUT_DIR / "action_nav_short_description_wordcloud.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Wordcloud nav labels] Saved: {outpath}")
    return outpath


# ===================================================================
# Text export
# ===================================================================
def save_statistics(fine_counter: Counter, core_counter: Counter,
                    total: int, wordcloud_counter: Counter,
                    core_wordcloud_counter: Counter,
                    core_rule_counter: Counter, audit: dict):
    """Export the fine + core distribution report."""
    lines = []
    sep = "=" * 72

    lines.append(sep)
    lines.append("BONES-SEED Action Distribution — Metadata Short-Description Labels")
    lines.append(sep)
    lines.append(f"Source metadata      : {audit['metadata_csv']}")
    lines.append(f"Metadata rows        : {audit['metadata_rows']:,}")
    lines.append(f"Filename filter mode : {audit['filename_filter_mode']}")
    lines.append("Rows skipped by filename filter: "
                 f"{audit['rows_skipped_by_filename_filter']:,}")
    if audit["filename_filter_mode"] == "filtered-dir":
        lines.append(f"Filtered motion dir  : {audit['filtered_motion_dir']}")
        lines.append(f"Filtered motion files: {audit['filtered_motion_file_count']:,}")
        lines.append(f"Metadata base filenames: {audit['allowed_filename_count']:,}")
        lines.append(f"Augmented files mapped to base labels: "
                     f"{audit['augmented_motion_file_count']:,}")
        lines.append("Base filenames missing metadata: "
                     f"{audit['allowed_filenames_missing_metadata']:,}")
    if audit["filename_filter_mode"] == "keywords":
        lines.append(f"Filter keyword count : {audit['filter_keyword_count']:,}")
        lines.append(f"Include keyword count: {audit['include_keyword_count']:,}")
    lines.append(f"Analyze mirrored rows: {not audit['skip_mirrored']}")
    lines.append(f"Analyzed motion entries: {total:,}")
    lines.append(f"Analyzed metadata rows : {audit['analyzed_metadata_rows']:,}")
    lines.append(f"Primary label column : {PRIMARY_SHORT_DESCRIPTION_COLUMN}")
    lines.append(f"Wordcloud columns    : {', '.join(SHORT_DESCRIPTION_COLUMNS)}")
    lines.append("Short-description wordcloud core-mapped: False")
    lines.append("Strip selected number tokens: "
                 f"{audit['strip_label_digits']}")
    lines.append("Selected number tokens: angles/distances + opt/version IDs")
    lines.append(f"Strip spatial words  : {audit['strip_label_spatial_words']}")
    lines.append(f"Spatial words        : {', '.join(audit['spatial_label_words'])}")
    lines.append(f"Wordcloud label uses : {sum(wordcloud_counter.values()):,}")
    lines.append(f"Core wordcloud labels : {sum(core_wordcloud_counter.values()):,}")
    lines.append(f"Unique fine labels   : {len(fine_counter):,}")
    lines.append(f"Unique wordcloud labels: {len(wordcloud_counter):,}")
    lines.append(f"Unique core labels   : {len(core_counter):,}")
    lines.append(f"Unique core rules    : {len(core_rule_counter):,}")
    lines.append("")
    lines.append("Raw short-description audit over all metadata rows:")
    for n_labels, n_rows in sorted(audit["raw_short_description_count"].items()):
        lines.append(f"  rows with {n_labels} non-empty short descriptions: {n_rows:,}")
    expected = len(SHORT_DESCRIPTION_COLUMNS)
    exact = audit["raw_short_description_count"].get(expected, 0)
    lines.append(f"  rows with exactly {expected}: {exact:,} / {audit['metadata_rows']:,}")
    lines.append("  every row has exactly two short descriptions: "
                 f"{exact == audit['metadata_rows']}")
    lines.append("  rows where the two raw short descriptions are identical: "
                 f"{audit['rows_with_duplicate_raw_short_descriptions']:,}")
    lines.append("  rows where the two normalized short descriptions are identical: "
                 f"{audit['rows_with_duplicate_normalized_short_descriptions']:,}")
    lines.append("  rows missing primary short description: "
                 f"{audit['rows_missing_primary_short_description']:,}")
    lines.append(f"  skipped mirrored rows: {audit['skipped_mirrored_rows']:,}")
    lines.append("")

    all_digit_audit = audit["all_digit_normalization_collision_audit"]
    selected_number_audit = audit["selected_number_normalization_collision_audit"]
    final_audit = audit["final_normalization_collision_audit"]
    lines.append("Label normalization collision audit over all raw short labels:")
    lines.append("  all-digit risk audit (not used by default):")
    lines.append(f"    normalized unique labels: {all_digit_audit['normalized_unique']:,}")
    lines.append("    normalized labels with >1 raw source: "
                 f"{all_digit_audit['colliding_normalized_labels']:,}")
    lines.append("    raw labels inside those collisions: "
                 f"{all_digit_audit['raw_labels_in_collisions']:,}")
    lines.append("    examples:")
    for normalized, raw_labels, n_raw in all_digit_audit["examples"]:
        lines.append(f"      {normalized}  <=  {raw_labels}  ({n_raw} raw)")
    lines.append("  selected-number normalization:")
    lines.append(f"    normalized unique labels: {selected_number_audit['normalized_unique']:,}")
    lines.append("    normalized labels with >1 raw source: "
                 f"{selected_number_audit['colliding_normalized_labels']:,}")
    lines.append("    raw labels inside those collisions: "
                 f"{selected_number_audit['raw_labels_in_collisions']:,}")
    lines.append("    examples:")
    for normalized, raw_labels, n_raw in selected_number_audit["examples"]:
        lines.append(f"      {normalized}  <=  {raw_labels}  ({n_raw} raw)")
    lines.append("  final normalization:")
    lines.append(f"    normalized unique labels: {final_audit['normalized_unique']:,}")
    lines.append("    normalized labels with >1 raw source: "
                 f"{final_audit['colliding_normalized_labels']:,}")
    lines.append("    raw labels inside those collisions: "
                 f"{final_audit['raw_labels_in_collisions']:,}")
    lines.append("    examples:")
    for normalized, raw_labels, n_raw in final_audit["examples"]:
        lines.append(f"      {normalized}  <=  {raw_labels}  ({n_raw} raw)")
    lines.append("")

    # --- Core summary ---
    lines.append(sep)
    lines.append("CORE SUMMARY")
    lines.append(sep)
    lines.append(f"  {'Core label':<24s} {'Motions':>8s}  {'Pct':>6s}  {'Bar'}")
    lines.append(f"  {'-' * 24} {'-' * 8}  {'-' * 6}  {'-' * 40}")
    top_core = core_counter.most_common(50)
    max_count = top_core[0][1]
    for name, cnt in top_core:
        pct = cnt / total * 100
        bar = "█" * int(cnt / max_count * 40)
        lines.append(f"  {name:<24s} {cnt:>8,}  {pct:>5.1f}%  {bar}")
    if len(core_counter) > len(top_core):
        lines.append(f"  ... ({len(core_counter) - len(top_core)} more core labels)")

    lines.append("")
    lines.append("  Core rule counts:")
    for rule, count in core_rule_counter.most_common(20):
        lines.append(f"    {rule:<24s} {count:>6,}")
    fallback_count = core_rule_counter.get("fallback_short", 0)
    lines.append(f"  fallback_short ratio: {fallback_count/total*100:.2f}%")

    # --- Usage guide ---
    lines.append("")
    lines.append(sep)
    lines.append("USAGE GUIDE")
    lines.append(sep)
    lines.append("")
    lines.append("  1. Use build_core_description() on content_short_description")
    lines.append("     when you need the compact motion core label.")
    lines.append("  2. The core label is meant for distribution analysis and")
    lines.append("     downstream text-conditioning experiments.")
    lines.append("  3. The short-description wordcloud still uses the original")
    lines.append("     normalized short labels.")
    lines.append("")
    lines.append(sep)

    outpath = OUTPUT_DIR / "action_statistics.txt"
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Statistics]     Saved: {outpath}")
    return outpath


# ===================================================================
# Main
# ===================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze BONES-SEED action labels from metadata CSV"
    )
    parser.add_argument(
        "--metadata-csv",
        default=METADATA_CSV,
        help="Path to BONES-SEED metadata CSV",
    )
    parser.add_argument(
        "--filename-filter-mode",
        choices=("none", "keywords", "filtered-dir"),
        default="none",
        help=(
            "Optional filename filtering: none; keywords uses the same "
            "should_filter_out rule/default keywords as "
            "dataset/data_process/filter_and_copy_bones_data.py; filtered-dir "
            "keeps only filenames present in --filtered-motion-dir"
        ),
    )
    parser.add_argument(
        "--filtered-motion-dir",
        default=str(DEFAULT_FILTERED_MOTION_DIR),
        help="Directory used by --filename-filter-mode filtered-dir",
    )
    parser.add_argument(
        "--filter-keywords",
        nargs="+",
        default=None,
        help=(
            "Override the default filter keywords imported from "
            "filter_and_copy_bones_data.py when using --filename-filter-mode keywords"
        ),
    )
    parser.add_argument(
        "--add-filter-keywords",
        nargs="+",
        default=None,
        help="Additional filename filter keywords for --filename-filter-mode keywords",
    )
    parser.add_argument(
        "--filter-file",
        default=None,
        help=(
            "Optional include-keyword file with one keyword per line, matching "
            "filter_and_copy_bones_data.py semantics"
        ),
    )
    parser.add_argument(
        "--include-mirrored",
        action="store_true",
        help="Analyze mirrored motions too; default keeps the original no-mirror behavior",
    )
    parser.add_argument(
        "--keep-label-digits",
        action="store_true",
        help=(
            "Keep selected number tokens in short-description labels; by default "
            "precise distances/angles plus option/version IDs such as opt 2 or v1 "
            "are removed"
        ),
    )
    parser.add_argument(
        "--keep-spatial-words",
        action="store_true",
        help=(
            "Keep spatial words in short-description labels; by default only "
            "left and right are removed"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.metadata_csv):
        print(f"ERROR: BONES-SEED metadata CSV not found: {args.metadata_csv}")
        sys.exit(1)

    filter_keywords = (
        list(args.filter_keywords)
        if args.filter_keywords is not None
        else list(DEFAULT_FILTER_KEYWORDS)
    )
    if args.add_filter_keywords:
        filter_keywords.extend(args.add_filter_keywords)

    include_keywords = read_include_keywords(args.filter_file)
    skip_mirrored = not args.include_mirrored
    strip_label_digits = not args.keep_label_digits
    strip_label_spatial_words = not args.keep_spatial_words

    # ── Collect ──
    print(f"Reading metadata: {args.metadata_csv}")
    print(f"Filename filter mode: {args.filename_filter_mode}")
    print("Strip selected number tokens from short-description labels: "
          f"{strip_label_digits}")
    print("Strip spatial words from short-description labels: "
          f"{strip_label_spatial_words} ({', '.join(SPATIAL_SHORT_DESCRIPTION_WORDS)})")
    if args.filename_filter_mode == "keywords":
        print(f"  Filter keywords: {len(filter_keywords)}")
        if include_keywords is not None:
            print(f"  Include keywords from {args.filter_file}: {len(include_keywords)}")
    elif args.filename_filter_mode == "filtered-dir":
        print(f"  Filtered motion dir: {args.filtered_motion_dir}")

    (
        fine_counter,
        core_counter,
        wordcloud_counter,
        core_wordcloud_counter,
        core_rule_counter,
        audit,
    ) = collect_all(
        args.metadata_csv,
        skip_mirrored=skip_mirrored,
        filename_filter_mode=args.filename_filter_mode,
        filtered_motion_dir=args.filtered_motion_dir,
        filter_keywords=filter_keywords,
        include_keywords=include_keywords,
        strip_label_digits=strip_label_digits,
        strip_label_spatial_words=strip_label_spatial_words,
        spatial_label_words=SPATIAL_SHORT_DESCRIPTION_WORDS,
    )
    total = audit["analyzed_rows"]
    if total == 0:
        print("ERROR: no motions left after metadata/filename/mirror filtering")
        sys.exit(1)

    print(f"Found {audit['metadata_rows']:,} metadata rows")
    print(f"  Rows skipped by filename filter: {audit['rows_skipped_by_filename_filter']:,}")
    if args.filename_filter_mode == "filtered-dir":
        print(f"  Filtered motion files: {audit['filtered_motion_file_count']:,}")
        print(f"  Metadata base filenames from filtered dir: "
              f"{audit['allowed_filename_count']:,}")
        print(f"  Augmented files mapped to base labels: "
              f"{audit['augmented_motion_file_count']:,}")
        print("  Base filenames missing metadata: "
              f"{audit['allowed_filenames_missing_metadata']:,}")
    print(f"  Mirrored rows skipped: {audit['skipped_mirrored_rows']:,}")
    print(f"  Analyzed motion entries: {total:,}")
    print(f"  Analyzed metadata rows: {audit['analyzed_metadata_rows']:,}")
    print(f"  Primary fine label: {PRIMARY_SHORT_DESCRIPTION_COLUMN}")
    print("  Strip selected number tokens: "
          f"{audit['strip_label_digits']}")
    print(f"  Strip spatial words: {audit['strip_label_spatial_words']} "
          f"({', '.join(audit['spatial_label_words'])})")
    print(f"  Fine-grained: {len(fine_counter):,} unique short-description labels")
    print(f"  Wordcloud labels: {sum(wordcloud_counter.values()):,} "
          f"uses across {len(wordcloud_counter):,} unique labels")
    print(f"  Core labels: {len(core_counter):,} unique core labels, "
          f"{len(core_rule_counter):,} core rules\n")

    expected_short = len(SHORT_DESCRIPTION_COLUMNS)
    rows_with_expected = audit["raw_short_description_count"].get(expected_short, 0)
    print("[Short-description audit over all metadata rows]")
    for n_labels, n_rows in sorted(audit["raw_short_description_count"].items()):
        print(f"  rows with {n_labels} non-empty short descriptions: {n_rows:,}")
    print(f"  every row has exactly {expected_short}: "
          f"{rows_with_expected == audit['metadata_rows']}")
    print("  rows where the two raw short descriptions are identical: "
          f"{audit['rows_with_duplicate_raw_short_descriptions']:,}")
    print("  rows where the two normalized short descriptions are identical: "
          f"{audit['rows_with_duplicate_normalized_short_descriptions']:,}")
    all_digit_audit = audit["all_digit_normalization_collision_audit"]
    selected_number_audit = audit["selected_number_normalization_collision_audit"]
    final_audit = audit["final_normalization_collision_audit"]
    print("  all-digit risk collision groups: "
          f"{all_digit_audit['colliding_normalized_labels']:,}")
    print("  selected-number normalized unique labels: "
          f"{selected_number_audit['normalized_unique']:,}")
    print("  selected-number collision groups: "
          f"{selected_number_audit['colliding_normalized_labels']:,}")
    print("  final normalized unique labels: "
          f"{final_audit['normalized_unique']:,}")
    print("  final collision groups: "
          f"{final_audit['colliding_normalized_labels']:,}")
    print()

    # ── Core summary ──
    core_top = core_rule_counter.most_common(10)
    print("[Core] Top rules:")
    for rule, cnt in core_top:
        print(f"    {rule:<24s} {cnt:>6,}")
    fallback_cnt = core_rule_counter.get("fallback_short", 0)
    print(f"  fallback_short ratio: {fallback_cnt/total*100:.2f}%")

    # ── Plots ──
    print()
    plot_fine_bar(fine_counter, total, top_n=50)
    plot_core_bar(core_counter, total, top_n=50)
    plot_core_wordcloud(core_wordcloud_counter)
    plot_short_description_wordcloud(wordcloud_counter)

    # ── Text export ──
    save_statistics(
        fine_counter,
        core_counter,
        total,
        wordcloud_counter,
        core_wordcloud_counter,
        core_rule_counter,
        audit,
    )

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("  action_distribution_bar.png   — fine-grained Top-50")
    print("  action_core_bar.png           — core Top-50")
    print("  action_core_wordcloud.png     — core word cloud")
    print("  action_short_description_wordcloud.png      — short-description labels")
    print("  action_statistics.txt         — full report + usage guide")


if __name__ == "__main__":
    main()
