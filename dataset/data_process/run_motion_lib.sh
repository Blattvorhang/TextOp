SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BONES_SEED_DIR="${BONES_SEED_DIR:-/home/lenovo/data/bones-seed}"
# OUTPUT_ROOT="${OUTPUT_ROOT:-${BONES_SEED_DIR}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data}"
FPS_TARGET="${FPS_TARGET:-50}"
FPS_SOURCE="${FPS_SOURCE:-120}"
NUM_WORKERS="${NUM_WORKERS:-16}"
FK_BACKEND="${FK_BACKEND:-torch}"
TORCH_DEVICE="${TORCH_DEVICE:-cpu}"
MOB_RASTER_BACKEND="${MOB_RASTER_BACKEND:-vectorized}"

S1_OUT="${OUTPUT_ROOT}/motion_lib"
S1_DONE="${S1_OUT}/.done"

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
