import torch
from omegaconf import DictConfig
from hydra.utils import instantiate

from robotmdar.utils.goal import (
    GoalType,
    build_ego_goal,
    validate_goal_config,
)
from robotmdar.utils.occupancy import query_local_occupancy
from robotmdar.dtype import seed, logger
from robotmdar.dtype.abc import VAE, Dataset, Denoiser, Diffusion, Optimizer, SSampler
from robotmdar.dtype.motion import DOF_DIM, motion_feature_dim
from robotmdar.train.manager import DARManager


def _pose_dict(position: torch.Tensor, rotation: torch.Tensor):
    return {"root_trans_offset": position, "root_rot": rotation}


def _conditions(primitive, reference_pos, reference_rot, history_motion, cfg):
    goal_type = GoalType.parse(cfg.data.goal_type)
    goal = build_ego_goal(
        primitive['world_goal_pos'].to(cfg.device),
        primitive['world_goal_yaw'].to(cfg.device),
        reference_pos,
        reference_rot,
        goal_type=goal_type,
        world_goal_keypoints=(
            primitive['world_goal_keypoints'].to(cfg.device)
            if goal_type.uses_keypoints else None
        ),
        world_root_velocity=(
            primitive['world_goal_vel'].to(cfg.device)
            if goal_type is GoalType.BODY_EXT else None
        ),
        timestep=(
            primitive['goal_timestep'].to(cfg.device)
            if goal_type is GoalType.BODY_EXT else None
        ),
    )
    voxel = query_local_occupancy(
        primitive['scene'],
        reference_pos,
        reference_rot,
        grid_size=cfg.denoiser.grid_size,
        grid_unit=cfg.data.occupancy_unit,
    )
    return {
        'goal': goal,
        'voxel': voxel,
        'history_motion_normalized': history_motion,
    }


def _next_rollout_poses(dataset, motion, history_start_pos, history_start_rot, history_len):
    with torch.no_grad():
        reconstructed = dataset.reconstruct_motion(
            motion,
            abs_pose=_pose_dict(history_start_pos, history_start_rot),
            ret_fk=False,
        )
    return (
        reconstructed['root_trans_offset'][:, -history_len].detach(),
        reconstructed['root_rot'][:, -history_len].detach(),
        reconstructed['root_trans_offset'][:, -1].detach(),
        reconstructed['root_rot'][:, -1].detach(),
    )


def _detach_mapping(values):
    return {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in values.items()
    }


def _validate_29dof_contract(cfg, datasets, vae, denoiser) -> None:
    """Fail before DAR training if data, FK, VAE, and denoiser disagree."""
    if DOF_DIM != 29 or motion_feature_dim != 69:
        raise RuntimeError(
            "DAR 29-DoF training requires DOF_DIM=29 and FeatureVersion 3 "
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
        if (
            dataset.mean.shape[-1] != motion_feature_dim
            or dataset.std.shape[-1] != motion_feature_dim
        ):
            raise ValueError(
                f"{split} normalization has shape mean={tuple(dataset.mean.shape)}, "
                f"std={tuple(dataset.std.shape)}; expected {motion_feature_dim}"
            )
        hand_names = tuple(
            dataset.skeleton.fk.body_names_augment[idx]
            for idx in dataset.skeleton.hand_id
        )
        if hand_names != ('left_hand_link', 'right_hand_link'):
            raise ValueError(
                "Body goals must use the left/right palm-center extensions, got "
                f"{hand_names}"
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

    expected_history_shape = (int(cfg.data.history_len), motion_feature_dim)
    actual_history_shape = tuple(int(dim) for dim in denoiser.history_shape)
    if actual_history_shape != expected_history_shape:
        raise ValueError(
            f"Denoiser history_shape={actual_history_shape}, expected "
            f"{expected_history_shape}"
        )


def _validate_batch(batch, cfg) -> None:
    num_primitive = int(cfg.data.num_primitive)
    context_len = int(cfg.data.history_len) + int(cfg.data.future_len)
    if len(batch) != num_primitive:
        raise ValueError(
            f"Dataset returned {len(batch)} primitives, expected {num_primitive}"
        )
    for primitive_idx, primitive in enumerate(batch):
        motion = primitive['motion']
        if motion.shape[1:] != (context_len, motion_feature_dim):
            raise ValueError(
                f"Primitive {primitive_idx} motion shape is {tuple(motion.shape)}, "
                f"expected [batch, {context_len}, {motion_feature_dim}]"
            )
        goal_type = GoalType.parse(cfg.data.goal_type)
        if goal_type.uses_keypoints:
            keypoints = primitive.get('world_goal_keypoints')
            num_keypoints = 4 if goal_type is GoalType.BODY_EXT else 5
            if keypoints is None or keypoints.shape[-2:] != (num_keypoints, 3):
                shape = None if keypoints is None else tuple(keypoints.shape)
                raise ValueError(
                    f"Primitive {primitive_idx} body goal keypoints have shape "
                    f"{shape}, expected [batch, {num_keypoints}, 3]"
                )
        if goal_type is GoalType.BODY_EXT:
            velocity = primitive.get('world_goal_vel')
            timestep = primitive.get('goal_timestep')
            if velocity is None or velocity.shape[-1:] != (3,):
                shape = None if velocity is None else tuple(velocity.shape)
                raise ValueError(
                    f"Primitive {primitive_idx} goal velocity has shape {shape}, "
                    "expected [batch, 3]"
                )
            if timestep is None or timestep.shape[-1:] != (1,):
                shape = None if timestep is None else tuple(timestep.shape)
                raise ValueError(
                    f"Primitive {primitive_idx} goal timestep has shape {shape}, "
                    "expected [batch, 1]"
                )


def main(cfg: DictConfig):
    seed.set(cfg.seed)
    logger.set(cfg)
    validate_goal_config(cfg.data.goal_type, cfg.denoiser.goal_dim)
    if cfg.train.manager.use_static_pose:
        raise ValueError(
            "Static-pose replacement has no world reference pose and is not "
            "supported by goal+scene training"
        )

    train_data: Dataset = instantiate(cfg.data.train)
    val_data: Dataset = instantiate(cfg.data.val)

    vae: VAE = instantiate(cfg.vae)
    denoiser: Denoiser = instantiate(cfg.denoiser)

    _validate_29dof_contract(
        cfg, [('train', train_data), ('val', val_data)], vae, denoiser
    )

    schedule_sampler: SSampler = instantiate(cfg.diffusion.schedule_sampler)
    diffusion: Diffusion = schedule_sampler.diffusion

    optimizer: Optimizer = torch.optim.AdamW(
        denoiser.parameters(), lr=cfg.train.manager.learning_rate)

    manager: DARManager = instantiate(cfg.train.manager)

    manager.hold_model(vae, denoiser, optimizer, train_data)

    num_primitive: int = cfg.data.num_primitive
    future_len: int = cfg.data.future_len
    history_len: int = cfg.data.history_len

    train_dataiter = iter(train_data)
    val_dataiter = iter(val_data)
    train_batch_validated = False
    val_batch_validated = False

    # Training loop following train_mvae.py approach
    while manager:
        denoiser.train()
        batch = next(train_dataiter)
        if not train_batch_validated:
            _validate_batch(batch, cfg)
            train_batch_validated = True

        prev_motion = None
        rollout_history_start_pos = None
        rollout_history_start_rot = None
        rollout_ref_pos = None
        rollout_ref_rot = None

        for pidx in range(num_primitive):
            manager.pre_step()
            primitive = batch[pidx]
            motion = primitive['motion'].to(cfg.device)

            future_motion_gt = motion[:, -future_len:, :]
            sliding_mask = batch[pidx]['sliding_mask'].to(
                cfg.device)[:, -future_len:, :]
            gt_history = motion[:, :history_len, :]

            # 使用统一的history选择函数
            history_motion, used_rollout = manager.choose_history(
                gt_history, prev_motion, history_len, return_rollout=True)

            if used_rollout:
                history_start_pos = rollout_history_start_pos
                history_start_rot = rollout_history_start_rot
                reference_pos = rollout_ref_pos
                reference_rot = rollout_ref_rot
            else:
                history_start_pos = primitive['history_start_pos'].to(cfg.device)
                history_start_rot = primitive['history_start_rot'].to(cfg.device)
                reference_pos = primitive['gt_ref_pos'].to(cfg.device)
                reference_rot = primitive['gt_ref_rot'].to(cfg.device)

            y = _conditions(primitive, reference_pos, reference_rot, history_motion, cfg)

            # Sample timesteps
            batch_size = motion.shape[0]
            t, weights = schedule_sampler.sample(batch_size, device=cfg.device)

            # Encode using VAE
            latent_gt, _ = vae.encode(
                future_motion=future_motion_gt,
                history_motion=history_motion
            )  # [T=1, B, D]   latent_gt: (1, 512, 128)

            x_start = latent_gt.permute(1, 0, 2)  # [B, T=1, D]

            # Forward diffusion
            x_t = diffusion.q_sample(x_start=x_start,
                                     t=t,
                                     noise=torch.randn_like(x_start))

            # Denoise
            x_start_pred = denoiser(x_t=x_t,
                                    timesteps=diffusion._scale_timesteps(t),
                                    y=y)  # [B, T=1, D]

            latent_pred = x_start_pred.permute(1, 0, 2)  # [T=1, B, D]

            # Decode
            future_motion_pred = vae.decode(
                latent_pred, history_motion,
                nfuture=future_len)  # [B, F, D], normalized

            # Calculate loss
            loss_dict, extras = manager.calc_loss(
                future_motion_gt,
                future_motion_pred,
                latent_gt,
                None,
                latent_pred,
                weights,
                history_motion=history_motion,  # dist=None for DAR
                sliding_mask=sliding_mask,
                ego_goal=y['goal'],
                goal_condition_keep_mask=y['goal_condition_keep_mask'],
            )
            loss = loss_dict['total']

            optimizer.zero_grad()
            loss.backward()
            has_nan_grad = False
            for param in denoiser.parameters():
                if param.grad is not None:
                    # 检查 NaN 和 Inf
                    if torch.isnan(param.grad).any() or torch.isinf(
                            param.grad).any():
                        has_nan_grad = True

            if not has_nan_grad:
                manager.grad_clip(denoiser)
                optimizer.step()

            # 更新prev_motion，如果启用full sample则使用更高质量的采样
            rollout_future = future_motion_pred
            if manager.should_use_full_sample():
                with torch.no_grad():
                    # 使用完整的DDPM采样循环来生成更高质量的rollout history
                    denoiser.eval()
                    x_start_full = diffusion.p_sample_loop(
                        denoiser,
                        x_start.shape,
                        clip_denoised=False,
                        model_kwargs={'y': y},
                        progress=False,
                    )
                    denoiser.train()
                    rollout_future = vae.decode(
                        x_start_full.permute(1, 0, 2),
                        history_motion,
                        nfuture=future_len,
                    )

            prev_motion = torch.cat(
                [history_motion, rollout_future], dim=1).detach()
            (rollout_history_start_pos, rollout_history_start_rot,
             rollout_ref_pos, rollout_ref_rot) = _next_rollout_poses(
                train_data, prev_motion,
                history_start_pos, history_start_rot, history_len)

            manager.post_step(
                is_eval=False,
                loss_dict=_detach_mapping(loss_dict),
                extras=_detach_mapping(extras),
            )

        # Validation loop
        denoiser.eval()
        while manager.should_eval():
            batch = next(val_dataiter)
            if not val_batch_validated:
                _validate_batch(batch, cfg)
                val_batch_validated = True
            for pidx in range(num_primitive):
                manager.pre_step(is_eval=True)
                primitive = batch[pidx]
                motion = primitive['motion'].to(cfg.device)

                future_motion_gt = motion[:, -future_len:, :]
                sliding_mask = batch[pidx]['sliding_mask'].to(
                    cfg.device)[:, -future_len:, :]
                history_motion = motion[:, :history_len, :]
                y = _conditions(
                    primitive,
                    primitive['gt_ref_pos'].to(cfg.device),
                    primitive['gt_ref_rot'].to(cfg.device),
                    history_motion,
                    cfg,
                )

                with torch.no_grad():
                    t, weights = schedule_sampler.sample(motion.shape[0],
                                                         device=cfg.device)

                    latent_gt, _ = vae.encode(
                        future_motion=future_motion_gt,
                        history_motion=history_motion)
                    # Forward diffusion
                    x_start = latent_gt.permute(1, 0, 2)  # [B, T=1, D]

                    x_t = diffusion.q_sample(x_start=x_start,
                                             t=t,
                                             noise=torch.randn_like(x_start))

                    x_start_pred = denoiser(
                        x_t=x_t, timesteps=diffusion._scale_timesteps(t), y=y)

                    latent_pred = x_start_pred.permute(1, 0, 2)

                    future_motion_pred = vae.decode(latent_pred,
                                                    history_motion,
                                                    nfuture=future_len)

                    loss_dict, extras = manager.calc_loss(
                        future_motion_gt,
                        future_motion_pred,
                        latent_gt,
                        None,
                        latent_pred,
                        weights,
                        history_motion=history_motion,
                        sliding_mask=sliding_mask,
                        ego_goal=y['goal'],
                        goal_condition_keep_mask=y.get('goal_condition_keep_mask'),
                        is_eval=True)

                manager.post_step(
                    is_eval=True,
                    loss_dict=_detach_mapping(loss_dict),
                    extras=_detach_mapping(extras),
                )
