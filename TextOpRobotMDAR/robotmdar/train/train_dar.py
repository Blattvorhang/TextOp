import os
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from hydra.utils import instantiate

from robotmdar.utils.ego_condition import (
    GoalType,
    build_ego_goal,
    query_local_occupancy,
    validate_goal_config,
)
from robotmdar.dtype import seed, logger
from robotmdar.dtype.abc import VAE, Dataset, Denoiser, Diffusion, Optimizer, SSampler

from robotmdar.train.manager import DARManager, is_main_process, get_ddp_model


def _pose_dict(position: torch.Tensor, rotation: torch.Tensor):
    return {"root_trans_offset": position, "root_rot": rotation}


def _conditions(primitive, reference_pos, reference_rot, history_motion, cfg):
    goal = build_ego_goal(
        primitive['world_goal_pos'].to(cfg.device),
        primitive['world_goal_yaw'].to(cfg.device),
        reference_pos,
        reference_rot,
        goal_type=cfg.data.goal_type,
        goal_keypoints=(
            primitive['world_goal_keypoints'].to(cfg.device)
            if GoalType.parse(cfg.data.goal_type) is GoalType.BODY else None
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


def ddp_setup():
    """Initialize DDP environment variables and process group."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
    else:
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def main(cfg: DictConfig):
    # Initialize DDP
    rank, world_size, local_rank = ddp_setup()
    device = torch.device(f'cuda:{local_rank}')

    seed.set(cfg.seed + rank)
    logger.set(cfg)
    validate_goal_config(cfg.data.goal_type, cfg.denoiser.goal_dim)
    if cfg.train.manager.use_static_pose:
        raise ValueError(
            "Static-pose replacement has no world reference pose and is not "
            "supported by goal+scene training"
        )

    # Validate goal_direction loss compatibility
    goal_type = GoalType.parse(cfg.data.goal_type)
    goal_direction_weight = cfg.train.manager.loss_weight.get('goal_direction', 0.0)
    if goal_direction_weight > 0.0 and goal_type is not GoalType.ROOT:
        raise ValueError(
            f"goal_direction loss (weight={goal_direction_weight}) is only "
            f"supported for goal_type='root', got '{goal_type.value}'. "
            "Set train.manager.loss_weight.goal_direction=0.0 for body goal."
        )

    # Override device in config for downstream components
    cfg.device = str(device)

    train_data: Dataset = instantiate(cfg.data.train)
    val_data: Dataset = instantiate(cfg.data.val)

    # Set rank/world_size on datasets for distributed data sharding
    train_data.rank = rank
    train_data.world_size = world_size
    val_data.rank = rank
    val_data.world_size = world_size

    vae: VAE = instantiate(cfg.vae)
    denoiser: Denoiser = instantiate(cfg.denoiser)
    vae = vae.to(device)
    denoiser = denoiser.to(device)

    if world_size > 1:
        denoiser = torch.nn.parallel.DistributedDataParallel(
            denoiser, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        if is_main_process():
            print(f"[DDP] Using DistributedDataParallel with {world_size} GPUs")

    # Keep raw denoiser reference for inference-only paths (p_sample_loop)
    # where DDP wrapper overhead is unnecessary and eval/train toggles are costly.
    denoiser_raw = get_ddp_model(denoiser)

    schedule_sampler: SSampler = instantiate(cfg.diffusion.schedule_sampler)
    diffusion: Diffusion = schedule_sampler.diffusion

    optimizer: Optimizer = torch.optim.AdamW(
        denoiser.parameters(), lr=cfg.train.manager.learning_rate)

    manager: DARManager = instantiate(cfg.train.manager)

    manager.hold_model(vae, denoiser, optimizer, train_data)
    manager.rank = rank
    manager.world_size = world_size

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
            t, weights = schedule_sampler.sample(batch_size, device=device)

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
            # Note: y dict is mutated inside denoiser.forward() to add
            # goal_condition_keep_mask, but DDP kwargs handling may prevent
            # the mutation from propagating back. Use .get() as a safe fallback.
            loss_dict, extras = manager.calc_loss(
                future_motion_gt,
                future_motion_pred,
                latent_gt,
                None,
                latent_pred,
                weights,
                history_motion=history_motion,  # dist=None for DAR
                ego_goal=y['goal'],
                goal_condition_keep_mask=y.get('goal_condition_keep_mask'),
                goal_type=cfg.data.goal_type,
            )
            loss = loss_dict['total']

            optimizer.zero_grad()
            loss.backward()
            has_nan_grad = False
            raw_model = get_ddp_model(denoiser)
            for param in raw_model.parameters():
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
                    # Use raw denoiser for p_sample_loop to avoid DDP wrapper
                    # overhead. no_grad already skips gradient sync; eval mode is
                    # only needed to disable dropout during sampling.
                    denoiser_raw.eval()
                    x_start_full = diffusion.p_sample_loop(
                        denoiser_raw,
                        x_start.shape,
                        clip_denoised=False,
                        model_kwargs={'y': y},
                        progress=False,
                    )
                    rollout_future = vae.decode(
                        x_start_full.permute(1, 0, 2),
                        history_motion,
                        nfuture=future_len,
                    )
                denoiser_raw.train()  # restore train mode for next primitive

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
                                                         device=device)

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
                        goal_condition_keep_mask=y.get('goal_condition_keep_mask'),
                        goal_type=cfg.data.goal_type,
                        is_eval=True)

                manager.post_step(
                    is_eval=True,
                    loss_dict=_detach_mapping(loss_dict),
                    extras=_detach_mapping(extras),
                )

    # Clean up DDP resources
    if dist.is_initialized():
        dist.destroy_process_group()
