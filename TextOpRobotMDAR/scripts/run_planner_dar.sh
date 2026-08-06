#!/usr/bin/env bash
set -euo pipefail

DOF_DIM=23
DATADIR="BONES-SEED-23dof-FULL-50fps"
DAR_CKPT="./logs/pretrained/long_horizon_64/ckpt_7500.pth"

robotmdar --config-name=planner_dar \
    ckpt.dar="${DAR_CKPT}" \
    data.dof_dim="${DOF_DIM}" \
    data.datadir=./dataset/${DATADIR} \
    data.action_statistics_path=./dataset/${DATADIR}/action_statistics.json \
    skeleton.asset.assetRoot=./description/robots/g1/ \
    "$@"
