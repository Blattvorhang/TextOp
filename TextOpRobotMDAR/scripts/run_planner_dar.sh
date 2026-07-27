DATADIR=BONES-SEED-23dof-FULL-50fps

robotmdar --config-name=planner_dar ckpt.dar=./logs/pretrained/goal_scene_v1/ckpt_37500.pth \
    data.datadir=./dataset/${DATADIR} \
    data.action_statistics_path=./dataset/${DATADIR}/action_statistics.json \
    skeleton.asset.assetRoot=./description/robots/g1/