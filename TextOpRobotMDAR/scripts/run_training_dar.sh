DATADIR=BONES-SEED-23dof-FULL-50fps
VAE_CKPT=./logs/RobotMDAR/BONES-SEED-GOAL/train-mvae-20260719014238/ckpt_100000.pth

robotmdar --config-name=train_dar expname=BONES-SEED-LDM \
    data.datadir=./dataset/${DATADIR} \
    ckpt.vae=${VAE_CKPT} \
    data.num_primitive=4 \
    data.batch_size=512 \
    data.weighted_sample=false \
    data.action_statistics_path=./dataset/${DATADIR}/action_statistics.json \
    train.manager.stages=[100000,100000,100000] \
    train.manager.use_rollout=true \
    train.manager.learning_rate=0.0001 \
    skeleton.asset.assetRoot=./description/robots/g1/ \
    train.manager.use_full_sample=true \
    diffusion.num_timesteps=5