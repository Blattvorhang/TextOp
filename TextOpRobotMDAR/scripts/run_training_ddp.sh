#!/bin/bash
#
# DDP (Distributed Data Parallel) training script for RobotMDAR MVAE
# Uses torchrun to launch training across 8 GPUs.
#
# Usage:
#   bash scripts/run_training_ddp.sh
#
# All training hyperparameters are preserved from the original single-GPU script.
# Effective batch size = batch_size (512) x 8 GPUs = 4096.

set -e

# Change to the project root directory (where this script's parent dir is)
cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"

# ---- GPU configuration ----
# CUDA_VISIBLE_DEVICES: which GPUs to use, first one is the master (rank 0).
# Default: use all GPUs starting from 0.
# Examples:
#   export CUDA_VISIBLE_DEVICES=0,1,2,3    # use GPU 0-3, master is GPU 0
#   export CUDA_VISIBLE_DEVICES=4,5,6,7    # use GPU 4-7, master is GPU 4
#   export CUDA_VISIBLE_DEVICES=2,5        # use GPU 2 and 5, master is GPU 2
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

echo "Starting DDP training with ${NUM_GPUS} GPUs..."
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Master GPU (rank 0): $(echo ${CUDA_VISIBLE_DEVICES} | cut -d',' -f1)"
echo "Experiment timestamp: ${TIMESTAMP}"
echo "Master port: ${MASTER_PORT}"

# Optional: Resume from a checkpoint (uncomment and set the path)
# CKPT_PATH="./logs/RobotMDAR/BONES-SEED-VAE/train-mvae-20260715_120000/ckpt_20000.pth"
# CKPT_OVERRIDE="ckpt.vae=${CKPT_PATH}"

# Scale stages by NUM_GPUS: DDP sees NUM_GPUS × batch_size samples per step,
# so each step is equivalent to NUM_GPUS single-GPU steps.
# Original: [100000, 50000, 50000] → scaled: ÷ NUM_GPUS
STAGE0=$((100000 / NUM_GPUS))
STAGE1=$((50000 / NUM_GPUS))
STAGE2=$((50000 / NUM_GPUS))
TOTAL_STEPS=$((STAGE0 + STAGE1 + STAGE2))

# Scale save/eval frequency proportionally
SAVE_EVERY=$((20000 / NUM_GPUS))
EVAL_EVERY=$((2000 / NUM_GPUS))

echo "Scaled for ${NUM_GPUS} GPUs:"
echo "  stages:      [${STAGE0}, ${STAGE1}, ${STAGE2}] (total: ${TOTAL_STEPS})"
echo "  save_every:  ${SAVE_EVERY}"
echo "  eval_every:  ${EVAL_EVERY}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    -m robotmdar.cli \
    --config-name=train_mvae \
    expname=BONES-SEED-VAE \
    timestamp=${TIMESTAMP} \
    data.datadir=/home/hanyi/workspace/TextOp/g1_packed \
    data.num_primitive=4 \
    data.batch_size=512 \
    data.weighted_sample=false \
    data.action_statistics_path=./dataset/dummy_action_stats.json \
    "train.manager.stages=[${STAGE0},${STAGE1},${STAGE2}]" \
    train.manager.save_every=${SAVE_EVERY} \
    train.manager.eval_every=${EVAL_EVERY} \
    train.manager.use_rollout=true \
    train.manager.learning_rate=0.0001 \
    skeleton.asset.assetRoot=./description/robots/g1/ \
    ${CKPT_OVERRIDE}
