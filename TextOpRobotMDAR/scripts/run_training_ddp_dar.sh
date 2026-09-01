#!/bin/bash
#
# DDP training script for the RobotMDAR DAR preset.
# Static run values live in `robotmdar/config/train_dar.yaml`.

set -e

# Change to the project root directory
cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"

# ---- Required: pretrained VAE checkpoint ----
VAE_CKPT="./logs/RobotMDAR/BONES-SEED-FUTURE-64-29DOF-RECOVERY/train-mvae-20260813_050039/ckpt_100000.pth"
if [ -z "${VAE_CKPT}" ]; then
    echo "ERROR: VAE_CKPT is required. Set it to the pretrained VAE checkpoint path."
    echo "Example:"
    echo "  VAE_CKPT=./logs/RobotMDAR/BONES-SEED-VAE/train-mvae-20260716_120000/ckpt_20000.pth"
    echo "  bash scripts/run_training_ddp_dar.sh"
    exit 1
fi

# ---- GPU configuration ----
CUDA_VISIBLE_DEVICES=1,4,5,7 #2,0,1,3,4,5,6,7

# Number of GPUs to use
NUM_GPUS=4

# Count actually visible GPUs
NUM_GPUS_AVAILABLE=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
if [ ${NUM_GPUS} -gt ${NUM_GPUS_AVAILABLE} ]; then
    NUM_GPUS=${NUM_GPUS_AVAILABLE}
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# TIMESTAMP="20260730_112611"  # resume from previous run
MASTER_PORT=$((RANDOM % 10000 + 20000))

echo "Starting DDP training (DAR preset) with ${NUM_GPUS} GPUs..."
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Master GPU (rank 0): $(echo ${CUDA_VISIBLE_DEVICES} | cut -d',' -f1)"
echo "VAE checkpoint: ${VAE_CKPT}"
echo "Experiment timestamp: ${TIMESTAMP}"
echo "Master port: ${MASTER_PORT}"

# Optional: Resume from a DAR checkpoint
# CKPT_PATH="./logs/RobotMDAR/BONES-SEED-FUTURE-64/train-dar-20260730_112611/ckpt_15000.pth"
# CKPT_OVERRIDE="ckpt.dar=${CKPT_PATH}"

# Scale stages by NUM_GPUS.
SCALE_FACTOR=1
STAGE0=$((100000 / NUM_GPUS * SCALE_FACTOR))
STAGE1=$((50000 / NUM_GPUS * SCALE_FACTOR))
STAGE2=$((50000 / NUM_GPUS * SCALE_FACTOR))
TOTAL_STEPS=$((STAGE0 + STAGE1 + STAGE2))

SAVE_EVERY=$((20000 / NUM_GPUS * SCALE_FACTOR))
EVAL_EVERY=$((2000 / NUM_GPUS * SCALE_FACTOR))

AUGMENTATION_START_STEP=$((60000 / NUM_GPUS * SCALE_FACTOR))
# SCENE_START_STEP=$((120000 / NUM_GPUS * SCALE_FACTOR))
SCENE_START_STEP=$((TOTAL_STEPS + 1))  # disable scene occupancy

echo "Scaled for ${NUM_GPUS} GPUs:"
echo "  stages:      [${STAGE0}, ${STAGE1}, ${STAGE2}] (total: ${TOTAL_STEPS})"
echo "  save_every:  ${SAVE_EVERY}"
echo "  eval_every:  ${EVAL_EVERY}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    -m robotmdar.cli \
    --config-name=train_dar \
    timestamp="'${TIMESTAMP}'" \
    ckpt.vae=${VAE_CKPT} \
    data.augmentation_start_step=${AUGMENTATION_START_STEP} \
    data.scene_start_step=${SCENE_START_STEP} \
    "train.manager.stages=[${STAGE0},${STAGE1},${STAGE2}]" \
    train.manager.save_every=${SAVE_EVERY} \
    train.manager.eval_every=${EVAL_EVERY} \
    ${CKPT_OVERRIDE}
