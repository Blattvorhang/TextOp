import os
import torch
import torch.distributed as dist
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
from robotmdar.dtype.motion import (
    DOF_DIM,
    G1_MUJOCO_DOF_JOINT_NAMES,
    G1_MUJOCO_DOF_LINK_NAMES,
    motion_feature_dim,
)
from robotmdar.train.manager import DARManager, is_main_process, get_ddp_model


def _make_root_xy_figure(
    generated_trajectory: torch.Tensor,
    goal_xy: torch.Tensor,
    ground_truth_trajectory: torch.Tensor = None,
    goal_condition_keep_mask: torch.Tensor = None,
    max_samples: int = 4,
):
    """Plot full-sample root paths in the reference ego horizontal plane."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    generated = generated_trajectory.detach().float().cpu()
    goals = goal_xy.detach().float().cpu()
    ground_truth = (
        ground_truth_trajectory.detach().float().cpu()
        if ground_truth_trajectory is not None else None
    )
    if goal_condition_keep_mask is None:
        indices = torch.arange(generated.shape[0])
    else:
        keep = goal_condition_keep_mask.detach().bool().cpu()
        indices = torch.nonzero(keep, as_tuple=False).flatten()
    if indices.numel() == 0:
        indices = torch.arange(generated.shape[0])
    indices = indices[:max_samples]

    count = max(int(indices.numel()), 1)
    cols = min(2, count)
    rows = (count + cols - 1) // cols
    figure = Figure(figsize=(6.0 * cols, 5.0 * rows), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, cols, squeeze=False)

    for axis, sample_idx in zip(axes.flat, indices.tolist()):
        generated_xy = generated[sample_idx].numpy()
        goal = goals[sample_idx, :2].numpy()
        axis.plot(
            generated_xy[:, 0], generated_xy[:, 1],
            color='#0072B2', linewidth=2.2, label='generated',
        )
        if ground_truth is not None:
            ground_truth_xy = ground_truth[sample_idx].numpy()
            axis.plot(
                ground_truth_xy[:, 0], ground_truth_xy[:, 1],
                color='#666666', linewidth=1.5, linestyle='--',
                label='ground truth',
            )
        axis.plot(
            [0.0, goal[0]], [0.0, goal[1]],
            color='#BBBBBB', linewidth=1.0, linestyle=':', label='goal ray',
        )
        axis.scatter([0.0], [0.0], color='#222222', s=35, zorder=4,
                     label='start')
        axis.scatter([goal[0]], [goal[1]], color='#D55E00', marker='X',
                     s=90, zorder=5, label='goal')
        axis.scatter(
            [generated_xy[-1, 0]], [generated_xy[-1, 1]],
            color='#0072B2', marker='s', s=40, zorder=4,
            label='generated end',
        )
        endpoint_error = torch.linalg.vector_norm(
            generated[sample_idx, -1] - goals[sample_idx, :2]
        ).item()
        axis.set_title(
            f'sample {sample_idx} | endpoint error {endpoint_error:.3f} m'
        )
        axis.set_xlabel('ego x (m)')
        axis.set_ylabel('ego y (m)')
        axis.set_aspect('equal', adjustable='datalim')
        axis.grid(True, color='#DDDDDD', linewidth=0.7)
        axis.legend(loc='best', fontsize=8)

    for axis in axes.flat[count:]:
        axis.set_visible(False)
    figure.suptitle('Full-sample root XY trajectory (reference ego frame)')
    return figure


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


def _validate_goal_position_contract(cfg) -> None:
    """Require the goal to be the generated primitive's terminal pose."""
    weight = float(
        cfg.train.manager.loss_weight.get('goal_position', 0.0))
    if weight <= 0.0:
        return

    offset_range = cfg.data.get('goal_offset_range')
    if offset_range is None:
        offsets = (int(cfg.data.get('goal_offset', 0)),) * 2
    else:
        offsets = tuple(int(value) for value in offset_range)
    if not bool(cfg.data.goal_per_primitive) or offsets != (0, 0):
        raise ValueError(
            "goal_position loss requires goal_per_primitive=true and a fixed "
            "zero goal offset; otherwise the conditioned goal is not the "
            f"generated primitive endpoint (got goal_per_primitive="
            f"{cfg.data.goal_per_primitive}, goal_offset_range={offsets})"
        )


def main(cfg: DictConfig):
    # Initialize DDP
    rank, world_size, local_rank = ddp_setup()
    device = torch.device(f'cuda:{local_rank}')

    seed.set(cfg.seed + rank)
    logger.set(cfg)
    validate_goal_config(cfg.data.goal_type, cfg.denoiser.goal_dim)
    _validate_goal_position_contract(cfg)
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

    _validate_29dof_contract(
        cfg, [('train', train_data), ('val', val_data)], vae, denoiser_raw
    )

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
                sliding_mask=sliding_mask,
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
                        sliding_mask=sliding_mask,
                        ego_goal=y['goal'],
                        goal_condition_keep_mask=y.get('goal_condition_keep_mask'),
                        goal_type=cfg.data.goal_type,
                        is_eval=True)

                    if getattr(manager, 'eval_full_sample', False):
                        sample_latent = diffusion.p_sample_loop(
                            denoiser,
                            x_start.shape,
                            clip_denoised=False,
                            model_kwargs={'y': y},
                            progress=False,
                        )
                        sample_future = vae.decode(
                            sample_latent.permute(1, 0, 2),
                            history_motion,
                            nfuture=future_len,
                        )
                        sample_trajectory = manager.root_trajectory_ego(
                            sample_future, history_motion)
                        sample_displacement = sample_trajectory[:, -1]
                        goal_keep_mask = y.get('goal_condition_keep_mask')
                        extras['sample_goal_position'] = (
                            manager.calc_goal_position_loss(
                                sample_future,
                                y['goal'],
                                goal_keep_mask,
                                history_motion=history_motion,
                            )
                        )
                        extras['sample_goal_direction'] = (
                            manager.calc_goal_direction_loss(
                                sample_future,
                                y['goal'],
                                goal_keep_mask,
                                history_motion=history_motion,
                            )
                        )
                        endpoint_error = torch.linalg.vector_norm(
                            sample_displacement - y['goal'][:, :2], dim=-1
                        )
                        if goal_keep_mask is not None:
                            endpoint_error = endpoint_error[
                                goal_keep_mask.to(dtype=torch.bool)
                            ]
                        if endpoint_error.numel() > 0:
                            extras['sample_endpoint_error_m'] = (
                                endpoint_error.mean()
                            )
                        extras['sample_root_displacement'] = (
                            sample_displacement.norm(dim=-1).mean()
                        )
                        extras['goal_root_displacement'] = (
                            y['goal'][:, :2].norm(dim=-1).mean()
                        )
                        extras['sample_latent_std'] = sample_latent.std()
                        if manager.should_report_eval_visualization():
                            ground_truth_trajectory = manager.root_trajectory_ego(
                                future_motion_gt, history_motion
                            )
                            figure = _make_root_xy_figure(
                                sample_trajectory,
                                y['goal'][:, :2],
                                ground_truth_trajectory,
                                goal_keep_mask,
                            )
                            manager.platform.report_figure(
                                'root_xy_trajectory',
                                figure,
                                manager.step,
                                group_name='eval',
                            )

                manager.post_step(
                    is_eval=True,
                    loss_dict=_detach_mapping(loss_dict),
                    extras=_detach_mapping(extras),
                )

    # Clean up DDP resources
    if dist.is_initialized():
        dist.destroy_process_group()
