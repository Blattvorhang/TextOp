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
#     NUM_WORKERS        parallel workers for stages 1-2 (default: 16)
#     PACK_WORKERS       parallel workers for stage 3  (default: 8)
#     FPS_TARGET         output frame rate             (default: 50)
#     VAL_RATIO          validation split ratio        (default: 0.05)
#     FK_BACKEND         torch or mujoco                (default: torch)
#     TORCH_DEVICE       cpu or cuda for batched FK     (default: cpu)
#     MOB_RASTER_BACKEND vectorized or scalar exact MOB (default: vectorized)
#
# Stages:
#     1. convert_soma_csv_to_motion_lib.py    CSV -> motion_lib PKL (+ contact_mask + scene occu)
#     2. filter_and_copy_bones_data.py        keyword filter
#     3. pack_motion_lib_to_textop.py         motion_lib -> TextOp format (+ coarse frame_ann)
#     4. cal_weighted_statistics.py           generate action_statistics.json for weighted_sample
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
BONES_SEED_DIR="${BONES_SEED_DIR:-/ALG/yukang/dataset/bones-seed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BONES_SEED_DIR}}"
# OUTPUT_ROOT="${OUTPUT_ROOT:-data}"
FPS_TARGET="${FPS_TARGET:-50}"
FPS_SOURCE="${FPS_SOURCE:-120}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PACK_WORKERS="${PACK_WORKERS:-8}"
VAL_RATIO="${VAL_RATIO:-0.05}"
SEED="${SEED:-42}"
FK_BACKEND="${FK_BACKEND:-torch}"
TORCH_DEVICE="${TORCH_DEVICE:-cpu}"
MOB_RASTER_BACKEND="${MOB_RASTER_BACKEND:-vectorized}"

S1_OUT="${OUTPUT_ROOT}/motion_lib"
S1_DONE="${S1_OUT}/.done"
S2_OUT="${OUTPUT_ROOT}/motion_lib_filtered"
S2_DONE="${S2_OUT}/.done"
S3_OUT="${OUTPUT_ROOT}/g1_textop_29dof"
S3_DONE="${S3_OUT}/.done"
S4_OUT="${S3_OUT}"
S4_DONE="${S4_OUT}/.done_stats"

# ============================================================================
#  Stage 1: CSV -> motion_lib PKL
# ============================================================================
echo ""
echo "Stage 1/4: convert_soma_csv_to_motion_lib.py"
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
        --mob_frame_stride 2 \
        --fk_backend "${FK_BACKEND}" \
        --torch_device "${TORCH_DEVICE}" \
        --mob_raster_backend "${MOB_RASTER_BACKEND}"
    touch "${S1_DONE}"
    echo "  [DONE]"
fi

# ============================================================================
#  Stage 2: keyword filter
# ============================================================================
echo ""
echo "Stage 2/4: filter_and_copy_bones_data.py"
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
echo "Stage 3/4: pack_motion_lib_to_textop.py"
echo "  Input : ${S2_OUT}"
echo "  Output: ${S3_OUT}"

if [ -f "${S3_DONE}" ]; then
    echo "  [SKIP] already done"
else
    python3 "${SCRIPT_DIR}/pack_motion_lib_to_textop.py" \
        --input "${S2_OUT}" \
        --output "${S3_OUT}" \
        --val_ratio "${VAL_RATIO}" \
        --seed "${SEED}" \
        --workers "${PACK_WORKERS}"
    touch "${S3_DONE}"
    echo "  [DONE]"
fi

# ============================================================================
#  Stage 4: action_statistics.json for weighted_sample
# ============================================================================
echo ""
echo "Stage 4/4: cal_weighted_statistics.py"
echo "  Input : ${S3_OUT}/train.pkl"
echo "  Output: ${S4_OUT}/action_statistics.json"

if [ -f "${S4_DONE}" ]; then
    echo "  [SKIP] already done"
else
    python3 "${SCRIPT_DIR}/cal_weighted_statistics.py" \
        --data_folder "${S3_OUT}" \
        --trg_filename "${S4_OUT}/action_statistics.json"
    touch "${S4_DONE}"
    echo "  [DONE]"
fi

# ============================================================================
#  Summary
# ============================================================================
echo ""
echo "Pipeline complete."
echo "  train.pkl               : ${S3_OUT}/train.pkl"
echo "  val.pkl                 : ${S3_OUT}/val.pkl"
echo "  lazy motion samples     : ${S3_OUT}/samples/"
echo "  statistics.yaml         : ${S3_OUT}/statistics.yaml"
echo "  action_statistics.json  : ${S4_OUT}/action_statistics.json"
echo "  scene occu              : inferred pseudo-obstacles (per-motion, in .pkl entries)"
echo "  frame_ann               : coarse action categories (per-sequence, in .pkl entries)"
echo ""
# symlink into robotmdar dataset dir
ln -sfn "$(realpath "${S3_OUT}")" "${PROJECT_ROOT}/TextOpRobotMDAR/dataset/BONES-SEED-29dof-FULL-50fps"
echo "  symlink: ${PROJECT_ROOT}/TextOpRobotMDAR/dataset/BONES-SEED-29dof-FULL-50fps -> ${S3_OUT}"

echo ""
echo "Next:"
echo "  robotmdar --config-name=train_mvae \\"
echo "    data.datadir=${S3_OUT} \\"
echo "    data.weighted_sample=true (optional) \\"
echo "    data.action_statistics_path=${S4_OUT}/action_statistics.json \\"
echo "    skeleton.asset.assetRoot=${MJCF_DIR}"
