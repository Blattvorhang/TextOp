#!/bin/bash
# ============================================================================
# BONES-SEED -> TextOp VAE training data pipeline
#
# Usage:
#     bash run_full_pipeline.sh
#
# Env vars (all optional):
#     BONES_SEED_DIR     path to BONES-SEED root       (default: auto-detected)
#     OUTPUT_ROOT        output base directory         (default: local data/ when present)
#     NUM_WORKERS        parallel workers for stages 1-2 (default: 16)
#     PACK_WORKERS       parallel workers for stage 3  (default: 8)
#     FPS_TARGET         output frame rate             (default: 50)
#     VAL_RATIO          validation split ratio        (default: 0.05)
#     FK_BACKEND         torch or mujoco                (default: torch)
#     TORCH_DEVICE       cpu or cuda for batched FK     (default: cpu)
#     MOB_RASTER_BACKEND vectorized or scalar exact MOB (default: vectorized)
#     METADATA_CSV       BONES-SEED metadata CSV        (default: ${BONES_SEED_DIR}/metadata/seed_metadata_v004.csv)
#     TEMPORAL_JSONL     temporal event labels          (default: ${BONES_SEED_DIR}/metadata/seed_metadata_v002_temporal_labels.jsonl)
#     FORCE_REPACK       rebuild Stage 3 even with a .done marker (default: 0)
#     PYTHON_BIN         Python interpreter for all stages (default: auto-detected)
#
# Stages:
#     1. convert_soma_csv_to_motion_lib.py    CSV -> motion_lib PKL (+ contact_mask + scene occu)
#     2. filter_and_copy_bones_data.py        keyword filter
#     3. pack_motion_lib_to_textop.py         motion_lib -> TextOp format (+ metadata frame_ann)
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
if [ -d "${PROJECT_ROOT}/data/motion_lib_filtered" ]; then
    DEFAULT_BONES_SEED_DIR="${HOME}/data/bones-seed"
    DEFAULT_OUTPUT_ROOT="${PROJECT_ROOT}/data"
else
    DEFAULT_BONES_SEED_DIR="/ALG/yukang/dataset/bones-seed"
    DEFAULT_OUTPUT_ROOT="${DEFAULT_BONES_SEED_DIR}"
fi
BONES_SEED_DIR="${BONES_SEED_DIR:-${DEFAULT_BONES_SEED_DIR}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"
FORCE_REPACK="${FORCE_REPACK:-0}"
if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -n "${CONDA_PREFIX:-}" ] \
        && [ -x "${CONDA_PREFIX}/bin/python" ] \
        && "${CONDA_PREFIX}/bin/python" -c 'import joblib' >/dev/null 2>&1; then
        PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    elif [ -x "${HOME}/miniforge3/envs/textop/bin/python" ]; then
        PYTHON_BIN="${HOME}/miniforge3/envs/textop/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi
FPS_TARGET="${FPS_TARGET:-50}"
FPS_SOURCE="${FPS_SOURCE:-120}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PACK_WORKERS="${PACK_WORKERS:-8}"
VAL_RATIO="${VAL_RATIO:-0.05}"
SEED="${SEED:-42}"
FK_BACKEND="${FK_BACKEND:-torch}"
TORCH_DEVICE="${TORCH_DEVICE:-cpu}"
MOB_RASTER_BACKEND="${MOB_RASTER_BACKEND:-vectorized}"
METADATA_CSV="${METADATA_CSV:-${BONES_SEED_DIR}/metadata/seed_metadata_v004.csv}"
TEMPORAL_JSONL="${TEMPORAL_JSONL:-${BONES_SEED_DIR}/metadata/seed_metadata_v002_temporal_labels.jsonl}"

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
    "${PYTHON_BIN}" "${SCRIPT_DIR}/convert_soma_csv_to_motion_lib.py" \
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
    "${PYTHON_BIN}" "${SCRIPT_DIR}/filter_and_copy_bones_data.py" \
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
echo "  Metadata: ${METADATA_CSV}"
echo "  Temporal: ${TEMPORAL_JSONL}"

STAGE3_REBUILT=0
NEW_INPUT_PKL=""
if [ -f "${S3_DONE}" ] && [ -d "${S2_OUT}" ]; then
    NEW_INPUT_PKL="$(find "${S2_OUT}" -type f -name '*.pkl' -newer "${S3_DONE}" -print -quit)"
fi

if [ "${FORCE_REPACK}" = "1" ]; then
    echo "  [REBUILD] FORCE_REPACK=1"
    REBUILD_STAGE3=1
elif [ ! -f "${S3_DONE}" ]; then
    REBUILD_STAGE3=1
elif [ -n "${NEW_INPUT_PKL}" ]; then
    echo "  [REBUILD] input PKL is newer than ${S3_DONE}: ${NEW_INPUT_PKL}"
    REBUILD_STAGE3=1
else
    REBUILD_STAGE3=0
fi

if [ "${REBUILD_STAGE3}" -eq 1 ]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/pack_motion_lib_to_textop.py" \
        --input "${S2_OUT}" \
        --output "${S3_OUT}" \
        --metadata-csv "${METADATA_CSV}" \
        --temporal-jsonl "${TEMPORAL_JSONL}" \
        --val_ratio "${VAL_RATIO}" \
        --seed "${SEED}" \
        --workers "${PACK_WORKERS}"
    touch "${S3_DONE}"
    STAGE3_REBUILT=1
    for cache in \
        "${S3_OUT}"/meanstd*.pkl \
        "${S3_OUT}"/weighted_meanstd*.pkl \
        "${S3_OUT}/goal_stats.pkl"; do
        if [ -f "${cache}" ]; then
            stale_cache="${cache}.stale.$(date +%Y%m%d_%H%M%S)"
            mv "${cache}" "${stale_cache}"
            echo "  [INVALIDATE] ${cache} -> ${stale_cache}"
        fi
    done
    echo "  [DONE]"
else
    echo "  [SKIP] already done"
fi

# ============================================================================
#  Stage 4: action_statistics.json and fall-recovery raw report
# ============================================================================
# --neutral: keep natural data distribution (weight=1.0 for all categories).
# The printed recovery subset is a raw-data visibility check.
echo ""
echo "Stage 4/4: cal_weighted_statistics.py (neutral)"
echo "  Input : ${S3_OUT}/train.pkl"
echo "  Output: ${S4_OUT}/action_statistics.json"

if [ ! -f "${S4_DONE}" ]; then
    REBUILD_STAGE4=1
elif [ "${STAGE3_REBUILT}" -eq 1 ]; then
    echo "  [REBUILD] Stage 3 produced a new manifest"
    REBUILD_STAGE4=1
elif [ -f "${S3_OUT}/train.pkl" ] && [ "${S3_OUT}/train.pkl" -nt "${S4_DONE}" ]; then
    echo "  [REBUILD] train.pkl is newer than ${S4_DONE}"
    REBUILD_STAGE4=1
else
    REBUILD_STAGE4=0
fi

if [ "${REBUILD_STAGE4}" -eq 1 ]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/cal_weighted_statistics.py" \
        --data_folder "${S3_OUT}" \
        --trg_filename "${S4_OUT}/action_statistics.json" \
        --neutral
    touch "${S4_DONE}"
    echo "  [DONE]"
else
    echo "  [SKIP] already done"
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
echo "  frame_ann               : metadata/temporal text annotations (in .pkl entries)"
echo ""
# symlink into robotmdar dataset dir
ln -sfn "$(realpath "${S3_OUT}")" "${PROJECT_ROOT}/TextOpRobotMDAR/dataset/BONES-SEED-29dof-FULL-50fps"
echo "  symlink: ${PROJECT_ROOT}/TextOpRobotMDAR/dataset/BONES-SEED-29dof-FULL-50fps -> ${S3_OUT}"

echo ""
echo "Next:"
echo "  robotmdar --config-name=train_mvae \\"
echo "    data.datadir=${S3_OUT} \\"
echo "    data.weighted_sample=false \\"
echo "    skeleton.asset.assetRoot=${MJCF_DIR}"
