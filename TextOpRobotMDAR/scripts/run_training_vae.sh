export CUDA_VISIBLE_DEVICES=1

robotmdar --config-name=train_mvae expname=BONES-SEED-VAE \
  data.datadir=./dataset/BONES-SEED-23dof-FULL-50fps \
  data.num_primitive=4 \
  data.batch_size=512 \
  data.weighted_sample=false \
  data.action_statistics_path=./dataset/dummy_action_stats.json \
  train.manager.stages=[100000,50000,50000] \
  train.manager.use_rollout=true \
  train.manager.learning_rate=0.0001 \
  skeleton.asset.assetRoot=./description/robots/g1/