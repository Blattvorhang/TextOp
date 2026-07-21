#!/bin/bash
# ============================================================================
# BONES-SEED -> TextOp VAE training data pipeline
#
# Usage:
#     bash run_full_pipeline.sh
#
# Env vars (all optional):
#     BONES_SEED_DIR     path to BONES-SEED root       (default: /home/lenovo/data/bones-seed)
#     OUTPUT_ROOT        output base directory         (default: same as BONES_SEED_DIR)
#     NUM_WORKERS        parallel workers for stage 1  (default: 8)
#     FPS_TARGET         output frame rate             (default: 50)
#     VAL_RATIO          validation split ratio        (default: 0.05)
#
# Stages:
#     1. convert_soma_csv_to_motion_lib.py    CSV -> motion_lib PKL (+ contact_mask + scene occu)
#     2. filter_and_copy_bones_data.py        keyword filter
#     3. pack_motion_lib_to_textop.py         motion_lib -> TextOp format
#
# Each stage writes a .done marker so the pipeline can be safely restarted
# after a failure -- completed stages are skipped.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Project root (TextOp/)
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MJCF_DIR="${MJCF_DIR:-${PROJECT_ROOT}/TextOpRobotMDAR/description/robots/g1}"

# ---- config ----
BONES_SEED_DIR="${BONES_SEED_DIR:-/home/lenovo/data/bones-seed}"
# OUTPUT_ROOT="${OUTPUT_ROOT:-${BONES_SEED_DIR}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data}"
FPS_TARGET="${FPS_TARGET:-50}"
FPS_SOURCE="${FPS_SOURCE:-120}"
NUM_WORKERS="${NUM_WORKERS:-16}"
VAL_RATIO="${VAL_RATIO:-0.05}"
SEED="${SEED:-42}"

S1_OUT="${OUTPUT_ROOT}/motion_lib"
S1_DONE="${S1_OUT}/.done"
S2_OUT="${OUTPUT_ROOT}/motion_lib_filtered"
S2_DONE="${S2_OUT}/.done"
S3_OUT="${OUTPUT_ROOT}/g1_textop"
S3_DONE="${S3_OUT}/.done"

# ============================================================================
#  Stage 1: CSV -> motion_lib PKL
# ============================================================================
echo ""
echo "Stage 1/3: convert_soma_csv_to_motion_lib.py"
echo "  Input : ${BONES_SEED_DIR}/g1/csv"
echo "  Output: ${S1_OUT}"

if [ -f "${S1_DONE}" ]; then
    echo "  [SKIP] already done"
else
    python3 "${SCRIPT_DIR}/convert_soma_csv_to_motion_lib.py" \
        --input "${BONES_SEED_DIR}/g1/csv" \
        --output "${S1_OUT}" \
        --fps "${FPS_TARGET}" \
        --fps_source "${FPS_SOURCE}" \
        --individual \
        --num_workers "${NUM_WORKERS}" \
        --mob \
        --mob_frame_stride 2
    touch "${S1_DONE}"
    echo "  [DONE]"
fi

# ============================================================================
#  Stage 2: keyword filter
# ============================================================================
echo ""
echo "Stage 2/3: filter_and_copy_bones_data.py"
echo "  Input : ${S1_OUT}"
echo "  Output: ${S2_OUT}"

if [ -f "${S2_DONE}" ]; then
    echo "  [SKIP] already done"
else
    python3 "${SCRIPT_DIR}/filter_and_copy_bones_data.py" \
        --source "${S1_OUT}" \
        --dest "${S2_OUT}" \
        --workers "${NUM_WORKERS}"
    touch "${S2_DONE}"
    echo "  [DONE]"
fi

# ============================================================================
#  Stage 3: motion_lib -> TextOp format
# ============================================================================
echo ""
echo "Stage 3/3: pack_motion_lib_to_textop.py"
echo "  Input : ${S2_OUT}"
echo "  Output: ${S3_OUT}"

if [ -f "${S3_DONE}" ]; then
    echo "  [SKIP] already done"
else
    python3 "${SCRIPT_DIR}/pack_motion_lib_to_textop.py" \
        --input "${S2_OUT}" \
        --output "${S3_OUT}" \
        --val_ratio "${VAL_RATIO}" \
        --seed "${SEED}"
    touch "${S3_DONE}"
    echo "  [DONE]"
fi

# ============================================================================
#  Summary
# ============================================================================
echo ""
echo "Pipeline complete."
echo "  train.pkl  : ${S3_OUT}/train.pkl"
echo "  val.pkl    : ${S3_OUT}/val.pkl"
echo "  stats      : ${S3_OUT}/statistics.yaml"
echo "  scene occu : inferred pseudo-obstacles (per-motion, in .pkl entries)"
echo ""
echo "Next: robotmdar --config-name=train_mvae data.datadir=${S3_OUT} ..."
