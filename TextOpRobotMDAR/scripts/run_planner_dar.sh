#!/usr/bin/env bash
set -euo pipefail

DOF_DIM=29
DATADIR="BONES-SEED-29dof-FULL-50fps"
# DAR_CKPT="./logs/pretrained/0807_velocity_29dof/ckpt_45000.pth"
DAR_CKPT="./logs/pretrained/0827_goal_reaching/ckpt_50000.pth"

robotmdar --config-name=planner_dar \
    ckpt.dar="${DAR_CKPT}" \
    data.dof_dim="${DOF_DIM}" \
    data.datadir=./dataset/${DATADIR} \
    data.action_statistics_path=./dataset/${DATADIR}/action_statistics.json \
    skeleton.asset.assetRoot=./description/robots/g1/ \
    "$@"
