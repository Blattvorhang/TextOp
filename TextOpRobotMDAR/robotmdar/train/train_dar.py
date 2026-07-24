import torch
from omegaconf import DictConfig
from hydra.utils import instantiate

from robotmdar.utils.ego_condition import (
    build_ego_goal,
    query_local_occupancy,
)
from robotmdar.dtype import seed, logger
from robotmdar.dtype.abc import VAE, Dataset, Denoiser, Diffusion, Optimizer, SSampler
from robotmdar.train.manager import DARManager


def _pose_dict(position: torch.Tensor, rotation: torch.Tensor):
    return {"root_trans_offset": position, "root_rot": rotation}


def _conditions(primitive, reference_pos, reference_rot, history_motion, cfg):
    goal = build_ego_goal(
        primitive['world_goal_pos'].to(cfg.device),
        primitive['world_goal_yaw'].to(cfg.device),
        reference_pos,
        reference_rot,
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


def main(cfg: DictConfig):
    seed.set(cfg.seed)
    logger.set(cfg)
    if cfg.train.manager.use_static_pose:
        raise ValueError(
            "Static-pose replacement has no world reference pose and is not "
            "supported by goal+scene training"
        )

    train_data: Dataset = instantiate(cfg.data.train)
    val_data: Dataset = instantiate(cfg.data.val)

    vae: VAE = instantiate(cfg.vae)
    denoiser: Denoiser = instantiate(cfg.denoiser)

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

    # Training loop following train_mvae.py approach
    while manager:
        denoiser.train()
        batch = next(train_dataiter)

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
            for pidx in range(num_primitive):
                manager.pre_step(is_eval=True)
                primitive = batch[pidx]
                motion = primitive['motion'].to(cfg.device)

                future_motion_gt = motion[:, -future_len:, :]
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
                        ego_goal=y['goal'],
                        goal_condition_keep_mask=y['goal_condition_keep_mask'])

                manager.post_step(
                    is_eval=True,
                    loss_dict=_detach_mapping(loss_dict),
                    extras=_detach_mapping(extras),
                )
