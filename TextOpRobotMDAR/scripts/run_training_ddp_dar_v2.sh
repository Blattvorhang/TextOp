#!/bin/bash
#
# DDP (Distributed Data Parallel) training script for RobotMDAR DAR with BODY goal.
# Uses the train_dar_v2 config (goal_type=body, goal_dim=15).
# Requires a pretrained VAE checkpoint.
#
# Usage:
#   export VAE_CKPT="./logs/RobotMDAR/BONES-SEED-VAE/train-mvae-<ts>/ckpt_20000.pth"
#   bash scripts/run_training_ddp_dar_v2.sh
#
#   # Or one-liner:
#   VAE_CKPT=./path/to/ckpt.pth bash scripts/run_training_ddp_dar_v2.sh
#
# Body goal keypoints: root, left_hand, right_hand, left_foot, right_foot (5×3=15 dims).
# Effective batch size = batch_size (512) x NUM_GPUS.

set -e

# Change to the project root directory
cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"

# ---- Required: pretrained VAE checkpoint ----
VAE_CKPT="${VAE_CKPT:-}"
if [ -z "${VAE_CKPT}" ]; then
    echo "ERROR: VAE_CKPT is required. Set it to the pretrained VAE checkpoint path."
    echo "Example:"
    echo "  VAE_CKPT=./logs/RobotMDAR/BONES-SEED-VAE/train-mvae-20260716_120000/ckpt_20000.pth"
    echo "  bash scripts/run_training_ddp_dar_v2.sh"
    exit 1
fi

# ---- GPU configuration ----
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,0,1,2,3,5,6,7}

# Number of GPUs to use
NUM_GPUS=${NUM_GPUS:-8}

# Count actually visible GPUs
NUM_GPUS_AVAILABLE=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
if [ ${NUM_GPUS} -gt ${NUM_GPUS_AVAILABLE} ]; then
    NUM_GPUS=${NUM_GPUS_AVAILABLE}
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_PORT=$((RANDOM % 10000 + 20000))

echo "Starting DDP training (DAR/LDM, BODY goal) with ${NUM_GPUS} GPUs..."
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Master GPU (rank 0): $(echo ${CUDA_VISIBLE_DEVICES} | cut -d',' -f1)"
echo "VAE checkpoint: ${VAE_CKPT}"
echo "Experiment timestamp: ${TIMESTAMP}"
echo "Master port: ${MASTER_PORT}"

# Optional: Resume from a DAR checkpoint
# CKPT_PATH="./logs/RobotMDAR/DAR-BODY-GOAL/train-dar-20260727_120000/ckpt_20000.pth"
# CKPT_OVERRIDE="ckpt.dar=${CKPT_PATH}"

# Scale stages by NUM_GPUS
SCALE_FACTOR=1
STAGE0=$((100000 / NUM_GPUS * SCALE_FACTOR))
STAGE1=$((100000 / NUM_GPUS * SCALE_FACTOR))
STAGE2=$((100000 / NUM_GPUS * SCALE_FACTOR))
TOTAL_STEPS=$((STAGE0 + STAGE1 + STAGE2))

SAVE_EVERY=$((20000 / NUM_GPUS * SCALE_FACTOR))
EVAL_EVERY=$((2000 / NUM_GPUS * SCALE_FACTOR))

echo "Scaled for ${NUM_GPUS} GPUs:"
echo "  stages:      [${STAGE0}, ${STAGE1}, ${STAGE2}] (total: ${TOTAL_STEPS})"
echo "  save_every:  ${SAVE_EVERY}"
echo "  eval_every:  ${EVAL_EVERY}"

DATADIR=BONES-SEED-23dof-FULL-50fps

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    -m robotmdar.cli \
    --config-name=train_dar_v2 \
    expname=BONES-SEED-BODY-GOAL \
    timestamp="'${TIMESTAMP}'" \
    ckpt.vae=${VAE_CKPT} \
    data.datadir=./dataset/${DATADIR} \
    data.num_primitive=4 \
    data.batch_size=512 \
    data.weighted_sample=false \
    data.action_statistics_path=./dataset/${DATADIR}/action_statistics.json \
    "train.manager.stages=[${STAGE0},${STAGE1},${STAGE2}]" \
    train.manager.save_every=${SAVE_EVERY} \
    train.manager.eval_every=${EVAL_EVERY} \
    train.manager.use_rollout=true \
    train.manager.learning_rate=0.0001 \
    skeleton.asset.assetRoot=./description/robots/g1/ \
    train.manager.use_full_sample=true \
    diffusion.num_timesteps=5 \
    ${CKPT_OVERRIDE}
