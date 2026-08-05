import torch
import toolz
from hydra.utils import instantiate
from omegaconf import DictConfig

from robotmdar.dtype import seed, logger
from robotmdar.dtype.abc import Dataset, VAE, Optimizer
from robotmdar.dtype.motion import (
    DOF_DIM,
    G1_MUJOCO_DOF_JOINT_NAMES,
    G1_MUJOCO_DOF_LINK_NAMES,
    motion_feature_dim,
)
from robotmdar.train.manager import MVAEManager


def evaluate_distribution_match(normalized_data):
    """评估归一化后数据分布是否接近标准正态"""
    # 计算关键统计量
    actual_mean = normalized_data.mean(dim=0)  # 各特征均值
    actual_std = normalized_data.std(dim=0)  # 各特征标准差

    # 理想值对比
    perfect_mean = torch.zeros_like(actual_mean)  # 期望均值=0
    perfect_std = torch.ones_like(actual_std)  # 期望标准差=1

    # 计算偏差
    mean_error = (actual_mean - perfect_mean).abs().mean()
    std_error = (actual_std - perfect_std).abs().mean()

    return {
        'mean_error': mean_error.item(),
        'std_error': std_error.item(),
        'is_well_normalized': mean_error < 0.1 and std_error < 0.2
    }


def _validate_29dof_contract(cfg, datasets, vae) -> None:
    """Fail before training if data, FK, and VAE dimensions disagree."""
    if DOF_DIM != 29 or motion_feature_dim != 69:
        raise RuntimeError(
            "MVAE 29-DoF training requires DOF_DIM=29 and FeatureVersion 3 "
            f"dimension 69, got DOF_DIM={DOF_DIM}, nfeats={motion_feature_dim}"
        )
    if int(cfg.data.nfeats) != motion_feature_dim:
        raise ValueError(
            f"data.nfeats={cfg.data.nfeats}, expected {motion_feature_dim}"
        )

    for split, dataset in datasets:
        stats = dataset.statistics
        stats_dof = int(stats.get('dof_dim', DOF_DIM))
        stats_nfeats = int(stats.get('nfeats', motion_feature_dim))
        skeleton_dof = int(dataset.skeleton.fk.num_dof)
        if stats_dof != DOF_DIM or stats_nfeats != motion_feature_dim:
            raise ValueError(
                f"{split} dataset is not native 29-DoF/69-D: "
                f"statistics dof_dim={stats_dof}, nfeats={stats_nfeats}"
            )
        if skeleton_dof != DOF_DIM:
            raise ValueError(
                f"{split} skeleton has {skeleton_dof} DoFs, expected {DOF_DIM}"
            )
        stats_order = stats.get('dof_order')
        if stats_order is not None and str(stats_order).lower() != 'mujoco':
            raise ValueError(
                f"{split} dataset uses {stats_order!r} DOF order, expected 'mujoco'"
            )
        stats_names = stats.get('dof_names')
        if (stats_names is not None
                and tuple(stats_names) != G1_MUJOCO_DOF_JOINT_NAMES):
            raise ValueError(
                f"{split} dataset DOF names do not match the G1 MuJoCo order"
            )
        if tuple(dataset.skeleton.fk.dof_joint_names) != G1_MUJOCO_DOF_JOINT_NAMES:
            raise ValueError(
                f"{split} MJCF joint order does not match the training contract"
            )
        if tuple(dataset.skeleton.fk.body_names[1:]) != G1_MUJOCO_DOF_LINK_NAMES:
            raise ValueError(
                f"{split} MJCF body order does not match the training contract"
            )
        if (
            dataset.mean.shape[-1] != motion_feature_dim
            or dataset.std.shape[-1] != motion_feature_dim
        ):
            raise ValueError(
                f"{split} normalization has shape mean={tuple(dataset.mean.shape)}, "
                f"std={tuple(dataset.std.shape)}; remove stale meanstd.pkl and "
                f"recompute {motion_feature_dim}-D statistics"
            )

    if vae.skel_embedding.in_features != motion_feature_dim:
        raise ValueError(
            f"VAE encoder expects {vae.skel_embedding.in_features} features, "
            f"expected {motion_feature_dim}"
        )
    if vae.final_layer.out_features != motion_feature_dim:
        raise ValueError(
            f"VAE decoder emits {vae.final_layer.out_features} features, "
            f"expected {motion_feature_dim}"
        )


def _validate_batch(batch, num_primitive: int, context_len: int) -> None:
    if len(batch) != num_primitive:
        raise ValueError(
            f"Dataset returned {len(batch)} primitives, expected {num_primitive}"
        )
    for primitive_idx, primitive in enumerate(batch):
        motion = primitive['motion']
        if motion.shape[1] != context_len:
            raise ValueError(
                f"Primitive {primitive_idx} has {motion.shape[1]} frames, "
                f"expected history+future={context_len}"
            )
        if motion.shape[-1] != motion_feature_dim:
            raise ValueError(
                f"Primitive {primitive_idx} has {motion.shape[-1]} features, "
                f"expected {motion_feature_dim}"
            )


def main(cfg: DictConfig):
    logger.set(cfg)
    seed.set(cfg.seed)

    train_data: Dataset = instantiate(cfg.data.train)
    val_data: Dataset = instantiate(cfg.data.val)

    vae: VAE = instantiate(cfg.vae)
    optimizer: Optimizer = torch.optim.Adam(vae.parameters(), **cfg.train.opt)
    manager: MVAEManager = instantiate(cfg.train.manager)

    _validate_29dof_contract(
        cfg, [('train', train_data), ('val', val_data)], vae
    )

    manager.hold_model(vae, optimizer, train_data)
    train_dataiter = iter(train_data)
    val_dataiter = iter(val_data)

    num_primitive = cfg.data.num_primitive
    future_len = cfg.data.future_len
    history_len = cfg.data.history_len
    train_batch_validated = False
    val_batch_validated = False

    # all_normalized = []

    # for i in range(100):
    #     batch = next(train_dataiter)
    #     for pidx in range(num_primitive):
    #         all_normalized.append(batch[pidx][0])
    # normalized_data = torch.cat(all_normalized)
    # dist_result = evaluate_distribution_match(normalized_data)
    # print(dist_result)
    # breakpoint()

    while manager:
        vae.train()
        batch = next(train_dataiter)
        if not train_batch_validated:
            _validate_batch(batch, num_primitive, history_len + future_len)
            train_batch_validated = True

        prev_motion = None
        for pidx in range(num_primitive):
            manager.pre_step()
            motion = batch[pidx]['motion'].to(cfg.device)

            future_motion_gt = motion[:, -future_len:, :]
            sliding_mask = batch[pidx]['sliding_mask'].to(
                cfg.device)[:, -future_len:, :]
            gt_history = motion[:, :history_len, :]

            # 使用统一的history选择函数
            history_motion = manager.choose_history(gt_history, prev_motion,
                                                    history_len)

            latent, dist = vae.encode(future_motion=future_motion_gt,
                                      history_motion=history_motion)
            future_motion_pred = vae.decode(latent,
                                            history_motion,
                                            nfuture=future_len)  # [B, F, D]

            loss_dict, extras = manager.calc_loss(
                future_motion_gt,
                future_motion_pred,
                dist,
                history_motion=history_motion,
                sliding_mask=sliding_mask)
            loss = loss_dict['total']

            optimizer.zero_grad()
            loss.backward()

            has_nan_grad = False
            for param in vae.parameters():
                if param.grad is not None:
                    # 检查 NaN 和 Inf
                    if torch.isnan(param.grad).any() or torch.isinf(
                            param.grad).any():
                        has_nan_grad = True

            if not has_nan_grad:
                manager.grad_clip(vae)
                optimizer.step()

            prev_motion = future_motion_pred.detach()

            manager.post_step(is_eval=False,
                              loss_dict=toolz.valmap(
                                  lambda x: x.detach().cpu(), loss_dict),
                              extras=toolz.valmap(
                                  lambda x: x.detach().cpu()
                                  if isinstance(x, torch.Tensor) else x,
                                  extras))

        vae.eval()
        while manager.should_eval():
            batch = next(val_dataiter)
            if not val_batch_validated:
                _validate_batch(batch, num_primitive, history_len + future_len)
                val_batch_validated = True
            for pidx in range(num_primitive):
                manager.pre_step(is_eval=True)
                motion = batch[pidx]['motion'].to(cfg.device)

                future_motion_gt = motion[:, -future_len:, :]
                sliding_mask = batch[pidx]['sliding_mask'].to(
                    cfg.device)[:, -future_len:, :]
                history_motion = motion[:, :history_len, :]

                with torch.no_grad():
                    latent, dist = vae.encode(
                        future_motion=future_motion_gt,
                        history_motion=history_motion,
                    )
                    future_motion_pred = vae.decode(
                        latent, history_motion, nfuture=future_len
                    )
                    loss_dict, extras = manager.calc_loss(
                        future_motion_gt,
                        future_motion_pred,
                        dist,
                        history_motion=history_motion,
                        sliding_mask=sliding_mask,
                    )
                manager.post_step(is_eval=True,
                                  loss_dict=toolz.valmap(
                                      lambda x: x.detach().cpu(), loss_dict),
                                  extras=toolz.valmap(
                                      lambda x: x.detach().cpu()
                                      if isinstance(x, torch.Tensor) else x,
                                      extras))
