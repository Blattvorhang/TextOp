#!/usr/bin/env python3
"""
Analyze actions with duration < 273 frames (50Hz), i.e. < 5.46 seconds.

Outputs:
  - Coarse-grained summary: which coarse categories have short actions + percentages
  - Fine-grained summary: which fine action names appear most among short actions
  - Short-action distribution per coarse category

Usage:  python analyze_short_actions.py
"""

import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE = "/home/lenovo/data/bones-seed/g1/csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_FRAMES = 273  # < 273 frames at 50Hz ≈ 5.46 seconds

# ---------------------------------------------------------------------------
# Action name extraction (mirrors analyze_action_distribution.py)
# ---------------------------------------------------------------------------
def extract_action_name(filename: str) -> str:
    stem = filename.replace(".csv", "")
    stem = re.sub(r'__A\d+$', '', stem)
    stem = re.sub(r'^\d{6}__', '', stem)
    m = re.match(r'^(.+)_(\d+(?:[a-z_]\w*)?)$', stem)
    if m:
        return m.group(1)
    m = re.match(r'^(.+?)__(\d+)$', stem)
    if m:
        return m.group(1)
    m = re.match(r'^(.+[a-zA-Z])(\d+[a-z_]*\w*)$', stem)
    if m:
        return m.group(1)
    return stem


# ---------------------------------------------------------------------------
# Coarse classification (mirrors analyze_action_distribution.py)
# ---------------------------------------------------------------------------
COARSE_RULES = [
    ("injured",       ["injured"]),
    ("crutch",        ["crutch", "crutches"]),
    ("jump",          ["jump", "hop", "leap", "flip", "vault_over",
                       "jump_and_land", "jump_over", "jump_twice",
                       "high_jump", "jump_ff", "jump_sideway",
                       "jump_and_down", "turn_jump", "fire_in_the_hole"]),
    ("jog",           ["jog", "jogging", "run_"]),
    ("walk",          ["walk", "moonwalk", "step_forward", "step_backward"]),
    ("dance",         ["dance", "dancing", "choreography", "macarena",
                       "dancecard", "expressionism", "krakowiak"]),
    ("climb",         ["climb", "ladder", "come_up_", "come_down_",
                       "crouch_cupboard"]),
    ("fall",          ["fall", "faint", "toxic_gas", "postmortem",
                       "death", "lying", "lie_", "flying_",
                       "stand_up_lying", "on_ground"]),
    ("crouch",        ["crouch", "crawl", "on_all_fours",
                       "crouch_idle", "crouch_walk", "stoop"]),
    ("kneel",         ["kneel", "sit_on_heels"]),
    ("sit",           ["sitting", "sit_cross", "sit_",
                       "read_newspaper_sitting", "eat_hotdog_sitting",
                       "play_guitar_sitting", "having_a_sit"]),
    ("carry",         ["carry", "lift", "crate", "heavy_", "light_",
                       "pick_up", "put_down", "hold_",
                       "moving_object", "pass_",
                       "item_give", "item_take", "item_pick", "item_put",
                       "item_switch", "item_hold",
                       "lasso_catch", "lasso_dance", "lasso_pull",
                       "watering_plants", "walk_the_dog",
                       "medium_big", "small_heavy", "small_light",
                       "big_heavy", "big_light",
                       "medium_heavy", "medium_light"]),
    ("reach",         ["reach", "reaching"]),
    ("push",          ["push", "pull", "crank", "valve", "handle", "lever",
                       "_knob_", "door_", "shut", "slam",
                       "open_walk", "close_",
                       "horizontal_lever", "vertical_lever",
                       "neutral_button", "operating"]),
    ("step_over",     ["step_over", "step_in", "avoid_obstacle",
                       "bump_into", "jump_over_obstacle", "neutral_avoid"]),
    ("turn",          ["turn_handstand", "mohak", "step_rotate",
                       "idle_turn", "spin_"]),
    ("idle",          ["idle", "stand", "standing", "legs_relax",
                       "looking_around", "looking_in_the_mirror",
                       "look_around", "looking_R", "looking_",
                       "neutral_sit", "neutral_stand",
                       "neutral_idle", "neutral_laugh", "neutral_fear",
                       "neutral_cry", "neutral_looking",
                       "neutral_dancecard_idle", "neutral_dancecard_looking",
                       "idle_hands", "idle_one_foot",
                       "idle_to_", "one_leg_idle"]),
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
                       ]),
    ("sport",         ["swim", "throw_", "catch_", "kick_", "punch_",
                       "dodge_", "play_tennis", "play_guitar",
                       "petting_dog", "dribble", "shoot_",
                       "ib_combat", "ib_dodge", "exercise",
                       "cartwheel"]),
]


def classify_coarse(fine_name: str) -> str:
    name_lower = fine_name.lower()
    for category, keywords in COARSE_RULES:
        for kw in keywords:
            if kw in name_lower:
                return category
    return "other"


# ---------------------------------------------------------------------------
# TASK PRIORITY (mirrors analyze_action_distribution.py)
# ---------------------------------------------------------------------------
TASK_CRITICAL = "task_critical"
TASK_RELEVANT = "task_relevant"
TASK_IRRELEVANT = "task_irrelevant"

NAV_PRIORITY = {
    "walk":      TASK_CRITICAL,
    "jog":       TASK_CRITICAL,
    "turn":      TASK_CRITICAL,
    "idle":      TASK_CRITICAL,
    "crouch":    TASK_CRITICAL,
    "kneel":     TASK_CRITICAL,
    "step_over": TASK_CRITICAL,
    "sit":       TASK_CRITICAL,
    "jump":      TASK_RELEVANT,
    "climb":     TASK_RELEVANT,
    "push":      TASK_RELEVANT,
    "carry":     TASK_RELEVANT,
    "reach":     TASK_RELEVANT,
    "fall":      TASK_RELEVANT,
    "gesture":   TASK_IRRELEVANT,
    "dance":     TASK_IRRELEVANT,
    "injured":   TASK_IRRELEVANT,
    "sport":     TASK_IRRELEVANT,
    "crutch":    TASK_IRRELEVANT,
    "other":     TASK_IRRELEVANT,
}


def get_nav_priority(coarse_name: str) -> str:
    return NAV_PRIORITY.get(coarse_name, TASK_IRRELEVANT)


# ---------------------------------------------------------------------------
# Count frames: read CSV, return number of data rows (lines - 1 for header)
# ---------------------------------------------------------------------------
def count_frames(csv_path: str) -> int:
    """Count data rows (frames) in a CSV file.
    Header row is excluded. Returns 0 on error.
    """
    try:
        with open(csv_path, "r") as f:
            # Count lines minus 1 for header
            line_count = sum(1 for _ in f)
        return max(0, line_count - 1)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Main collection
# ---------------------------------------------------------------------------
def collect_short_actions(base_dir: str):
    """Walk all CSV files, collect fine + coarse labels for short actions."""
    fine_counter_all = Counter()       # all actions (for reference)
    fine_counter_short = Counter()     # short actions only
    coarse_counter_short = Counter()   # short actions by coarse category
    coarse_examples_short = defaultdict(list)  # example fine names per coarse
    coarse_all_total = Counter()       # total per coarse (for % calculation)

    # Per-coarse: fine breakdown
    coarse_fine_short = defaultdict(Counter)

    total_files = 0
    short_count = 0
    skipped_mirror = 0

    for root, dirs, files in os.walk(base_dir):
        for fname in files:
            if not fname.endswith(".csv"):
                continue
            if "_M.csv" in fname:
                skipped_mirror += 1
                continue

            total_files += 1
            filepath = os.path.join(root, fname)

            # Count frames
            n_frames = count_frames(filepath)
            if n_frames <= 0:
                continue

            # Extract action name
            fine = extract_action_name(fname)
            coarse = classify_coarse(fine)

            fine_counter_all[fine] += 1
            coarse_all_total[coarse] += 1

            if n_frames < MAX_FRAMES:
                short_count += 1
                fine_counter_short[fine] += 1
                coarse_counter_short[coarse] += 1
                coarse_fine_short[coarse][fine] += 1

                if len(coarse_examples_short[coarse]) < 10:
                    coarse_examples_short[coarse].append(
                        (fine, n_frames, fname))

            # Progress indicator
            if total_files % 10000 == 0:
                print(f"  Processed: {total_files:,} files  "
                      f"(short: {short_count:,})", flush=True)

    return (fine_counter_all, fine_counter_short, coarse_counter_short,
            coarse_counter_all_global := coarse_all_total,
            coarse_examples_short, coarse_fine_short,
            total_files, short_count, skipped_mirror)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(fine_all, fine_short, coarse_short, coarse_all,
                 coarse_examples, coarse_fine_short,
                 total_files, short_count, skipped):
    """Print and save the analysis report."""
    lines = []
    sep = "=" * 78

    def add(line=""):
        lines.append(line)

    short_pct = short_count / total_files * 100 if total_files else 0

    add(sep)
    add(f"ACTIONS WITH DURATION < {MAX_FRAMES} FRAMES (50Hz) — i.e., < {MAX_FRAMES/50:.1f} sec")
    add(sep)
    add(f"Source directory   : {BASE}")
    add(f"Total CSV files    : {total_files:,}")
    add(f"Mirrored skipped   : {skipped:,}")
    add(f"Short actions      : {short_count:,}  ({short_pct:.2f}% of total)")
    add(f"Unique fine names  : {len(fine_all):,} (all) / {len(fine_short):,} (short)")
    add(f"Coarse categories  : {len(coarse_all)}")
    add("")

    # =====================================================================
    # 1. COARSE-GRAINED: which categories have the most short actions?
    # =====================================================================
    add(sep)
    add("1. COARSE-GRAINED: Short-action distribution by coarse category")
    add(sep)
    add(f"  {'Category':<16s} {'Priority':<14s} {'Short':>7s} {'Total':>7s}  "
        f"{'Short%':>7s}  {'OfAllSh%':>8s}")
    add(f"  {'-' * 16} {'-' * 14} {'-' * 7} {'-' * 7}  {'-' * 7}  {'-' * 8}")

    # Sort by short count descending
    sorted_coarse = sorted(coarse_short.items(), key=lambda x: -x[1])

    for name, cnt_short in sorted_coarse:
        priority = get_nav_priority(name)
        cnt_all = coarse_all.get(name, 0)
        pct_short_within = cnt_short / cnt_all * 100 if cnt_all else 0
        pct_of_all_short = cnt_short / short_count * 100 if short_count else 0
        tag = f"[{priority.split('_')[1][0].upper()}]"
        add(f"  {name:<16s} {tag:<14s} {cnt_short:>7,} {cnt_all:>7,}  "
            f"{pct_short_within:>6.1f}%  {pct_of_all_short:>7.1f}%")

    add("")

    # =====================================================================
    # 2. COARSE SUMMARY: % of each category that is short
    # =====================================================================
    add(sep)
    add("2. PER-CATEGORY SHORT-ACTION RATE")
    add("   (What % of each coarse category's actions are < 273 frames?)")
    add(sep)
    add(f"  {'Category':<16s} {'Total':>7s} {'Short':>7s}  {'Rate':>6s}")
    add(f"  {'-' * 16} {'-' * 7} {'-' * 7}  {'-' * 6}")

    # Sort by short-rate descending
    rate_sorted = []
    for name, cnt_all in coarse_all.items():
        cnt_short = coarse_short.get(name, 0)
        rate = cnt_short / cnt_all * 100 if cnt_all else 0
        rate_sorted.append((name, cnt_all, cnt_short, rate))
    rate_sorted.sort(key=lambda x: -x[3])

    for name, cnt_all, cnt_short, rate in rate_sorted:
        priority = get_nav_priority(name)
        tag = f"[{priority.split('_')[1][0].upper()}]"
        add(f"  {name:<16s} {cnt_all:>7,} {cnt_short:>7,}  {rate:>5.1f}%  {tag}")

    add("")

    # =====================================================================
    # 3. FINE-GRAINED: top short action names
    # =====================================================================
    add(sep)
    add("3. FINE-GRAINED: Top-50 short action names (duration < 273 frames)")
    add(sep)
    top_fine = fine_short.most_common(50)
    for rank, (name, cnt) in enumerate(top_fine, 1):
        coarse = classify_coarse(name)
        total_all = fine_all.get(name, 0)
        pct = cnt / total_all * 100 if total_all else 0
        add(f"  {rank:>3}. {name:<55s} short={cnt:>6,}  "
            f"total={total_all:>6,}  {pct:>5.1f}%  [{coarse}]")

    add("")

    # =====================================================================
    # 4. PER-CATEGORY FINE-GRAINED BREAKDOWN
    # =====================================================================
    add(sep)
    add("4. PER-CATEGORY FINE-GRAINED BREAKDOWN (short actions only)")
    add(sep)

    for coarse_name, cnt_short in sorted_coarse:
        fine_in_coarse = coarse_fine_short.get(coarse_name, Counter())
        if not fine_in_coarse:
            continue
        priority = get_nav_priority(coarse_name)
        tag = f"[{priority.split('_')[1][0].upper()}]"
        add(f"\n  [{coarse_name}] {tag}  — {cnt_short:,} short actions "
            f"({cnt_short/coarse_all.get(coarse_name, 1)*100:.1f}% of "
            f"{coarse_all.get(coarse_name, 0):,} total in this category)")

        top_fine_in_coarse = fine_in_coarse.most_common(20)
        for fn, fc in top_fine_in_coarse:
            fn_all = fine_all.get(fn, 0)
            add(f"    {fn:<60s} short={fc:>5,}  total={fn_all:>6,}  "
                f"({fc/fn_all*100:.1f}% short)" if fn_all else
                f"    {fn:<60s} short={fc:>5,}")

        remaining = len(fine_in_coarse) - 20
        if remaining > 0:
            add(f"    ... ({remaining} more fine names)")

    # Save
    outpath = OUTPUT_DIR / "short_actions_report.txt"
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved: {outpath}")

    # Also print to stdout
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not os.path.isdir(BASE):
        print(f"ERROR: BONES-SEED directory not found: {BASE}")
        sys.exit(1)

    print(f"Scanning CSV files in: {BASE}")
    print(f"Filter: duration < {MAX_FRAMES} frames ({MAX_FRAMES/50:.1f} sec)")
    print()

    (fine_all, fine_short, coarse_short, coarse_all,
     coarse_examples, coarse_fine_short,
     total_files, short_count, skipped) = collect_short_actions(BASE)

    print(f"\nDone. Total: {total_files:,} | Short (<{MAX_FRAMES}f): "
          f"{short_count:,} ({short_count/total_files*100:.1f}%) | "
          f"Mirrored skipped: {skipped:,}")
    print()

    print_report(fine_all, fine_short, coarse_short, coarse_all,
                 coarse_examples, coarse_fine_short,
                 total_files, short_count, skipped)


if __name__ == "__main__":
    main()
