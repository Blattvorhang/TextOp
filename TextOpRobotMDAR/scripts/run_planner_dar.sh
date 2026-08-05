#!/usr/bin/env bash
set -euo pipefail

DATADIR="BONES-SEED-29dof-FULL-50fps"
DAR_CKPT="./logs/pretrained/body_29dof/ckpt_35000.pth"

robotmdar --config-name=planner_dar \
    ckpt.dar="${DAR_CKPT}" \
    data.datadir=./dataset/${DATADIR} \
    data.action_statistics_path=./dataset/${DATADIR}/action_statistics.json \
    skeleton.asset.assetRoot=./description/robots/g1/ \
    "$@"
