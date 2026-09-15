DATADIR=BONES-SEED-29dof-FULL-50fps

robotmdar --config-name=loop_dar ckpt.dar=./logs/pretrained/0914_text_motion_prior/ckpt_30000.pth \
    guidance_scale=1.0 \
    data.datadir=./dataset/${DATADIR} \
    data.action_statistics_path=./dataset/${DATADIR}/action_statistics.json \
    skeleton.asset.assetRoot=./description/robots/g1/
