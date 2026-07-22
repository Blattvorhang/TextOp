#!/bin/bash
#
# DDP (Distributed Data Parallel) training script for RobotMDAR DAR (LDM)
# Second-stage diffusion model training. Requires a pretrained VAE checkpoint.
# Uses torchrun to launch training across multiple GPUs.
#
# Usage:
#   # Set VAE checkpoint path, then run:
#   export VAE_CKPT="./logs/RobotMDAR/BONES-SEED-VAE/train-mvae-<ts>/ckpt_20000.pth"
#   bash scripts/run_training_ddp_dar.sh
#
#   # Or one-liner:
#   VAE_CKPT=./path/to/ckpt.pth bash scripts/run_training_ddp_dar.sh
#
# All training hyperparameters are preserved from the original single-GPU config.
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
    echo "  bash scripts/run_training_ddp_dar.sh"
    exit 1
fi

# ---- GPU configuration ----
# CUDA_VISIBLE_DEVICES: which GPUs to use, first one is the master (rank 0).
# Examples:
#   export CUDA_VISIBLE_DEVICES=0,1,2,3    # use GPU 0-3, master is GPU 0
#   export CUDA_VISIBLE_DEVICES=4,5,6,7    # use GPU 4-7, master is GPU 4
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,0,1,2,3,5,6,7}

# Number of GPUs to use (can be overridden via environment variable)
NUM_GPUS=${NUM_GPUS:-8}

# Count actually visible GPUs
NUM_GPUS_AVAILABLE=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
if [ ${NUM_GPUS} -gt ${NUM_GPUS_AVAILABLE} ]; then
    NUM_GPUS=${NUM_GPUS_AVAILABLE}
fi

# Pre-compute timestamp so all ranks share the same experiment directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Random master port to avoid conflicts with other DDP runs
MASTER_PORT=$((RANDOM % 10000 + 20000))

echo "Starting DDP training (DAR/LDM) with ${NUM_GPUS} GPUs..."
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Master GPU (rank 0): $(echo ${CUDA_VISIBLE_DEVICES} | cut -d',' -f1)"
echo "VAE checkpoint: ${VAE_CKPT}"
echo "Experiment timestamp: ${TIMESTAMP}"
echo "Master port: ${MASTER_PORT}"

# Optional: Resume from a DAR checkpoint (uncomment and set the path)
# CKPT_PATH="./logs/RobotMDAR/DAR/train-dar-20260716_120000/ckpt_20000.pth"
# CKPT_OVERRIDE="ckpt.dar=${CKPT_PATH}"

# Scale stages by NUM_GPUS: DDP sees NUM_GPUS x batch_size samples per step.
# Original DAR stages: [100000, 100000, 100000] (total 300000)
SCALE_FACTOR=1
STAGE0=$((100000 / NUM_GPUS * SCALE_FACTOR))
STAGE1=$((100000 / NUM_GPUS * SCALE_FACTOR))
STAGE2=$((100000 / NUM_GPUS * SCALE_FACTOR))
TOTAL_STEPS=$((STAGE0 + STAGE1 + STAGE2))

# Scale save/eval frequency proportionally
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
    --config-name=train_dar \
    expname=BONES-SEED-GOAL \
    timestamp=${TIMESTAMP} \
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
