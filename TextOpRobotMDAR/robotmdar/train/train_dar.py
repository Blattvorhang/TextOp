import math

import numpy as np
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
from robotmdar.utils.dof_contract import (
    configure_dof_contract,
    validate_training_contract,
)
from robotmdar.train.manager import DARManager


# ── Default XY plot limit from 64-frame goal distribution analysis ──
# The mode (peak of KDE) is ~0.02 m — most goals lie near the origin.
# 0.2 gives a tight 0.4×0.4 m view centred on the reference frame;
# expands dynamically when a trajectory or goal exceeds it.
# See dataset/data_analyze/analyze_64f_goal_xy.py
_DEFAULT_XY_LIMIT = 0.2


def _compute_xy_limit(*trajectories, goals_xy, default=_DEFAULT_XY_LIMIT):
    """Unified symmetric XY limit expanded if any data exceeds *default*."""
    limit = default
    for traj in trajectories:
        if traj is not None and traj.numel():
            limit = max(limit, float(traj.abs().max()))
    if goals_xy is not None and goals_xy.numel():
        limit = max(limit, float(goals_xy[:, :2].abs().max()))
    # round up to nearest 0.1 m for clean axis labels
    limit = math.ceil(limit * 10) / 10
    return max(limit, default)


def _make_root_xy_figure(
    generated_trajectory: torch.Tensor,
    goal_xy: torch.Tensor,
    ground_truth_trajectory: torch.Tensor = None,
    goal_condition_keep_mask: torch.Tensor = None,
    history_trajectory: torch.Tensor = None,
    voxel: torch.Tensor = None,
    grid_size: int = 25,
    grid_unit: float = 0.08,
    xy_limit: float | None = None,
    max_samples: int = 4,
):
    """Plot full-sample root paths in the reference ego horizontal plane.

    Convention: **x-forward, y-left, z-up** (TextOp ego frame).
    The origin (0, 0) is the last history frame — the reference pose for
    the generated primitive.  All subplots share the same square XY limits
    so that scale is directly comparable across samples.

    *voxel* optionally overlays the local occupancy slice at the reference
    root height (z=0 slice of the ego-frame grid, per query_local_occupancy's
    vertical centering on the reference pose).  The slice is clipped by the
    shared XY limit — it does not expand the view.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    # occupied cells: semi-transparent gray; free cells: fully transparent
    occ_cmap = ListedColormap([(0.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5, 0.4)])

    generated = generated_trajectory.detach().float().cpu()
    goals = goal_xy.detach().float().cpu()
    ground_truth = (
        ground_truth_trajectory.detach().float().cpu()
        if ground_truth_trajectory is not None else None
    )
    history = (
        history_trajectory.detach().float().cpu()
        if history_trajectory is not None else None
    )
    vox = voxel.detach().float().cpu() if voxel is not None else None
    if goal_condition_keep_mask is None:
        indices = torch.arange(generated.shape[0])
    else:
        keep = goal_condition_keep_mask.detach().bool().cpu()
        indices = torch.nonzero(keep, as_tuple=False).flatten()
    if indices.numel() == 0:
        indices = torch.arange(generated.shape[0])
    indices = indices[:max_samples]
    count = max(int(indices.numel()), 1)

    # ── unified square XY limit (only over plotted samples) ──
    if xy_limit is None:
        sel = indices  # (up to max_samples)
        xy_limit = _compute_xy_limit(
            generated[sel],
            ground_truth[sel] if ground_truth is not None else None,
            history[sel] if history is not None else None,
            goals_xy=goals[sel],
        )

    cols = min(2, count)
    rows = (count + cols - 1) // cols
    figure = Figure(figsize=(6.0 * cols, 5.0 * rows), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, cols, squeeze=False)

    for axis, sample_idx in zip(axes.flat, indices.tolist()):
        generated_xy = generated[sample_idx].numpy()
        goal = goals[sample_idx, :2].numpy()

        # ── local occupancy slice at the reference root height (z=0) ──
        # Offsets from _local_grid_offsets are cell CENTERS (forward axis is
        # biased forward by grid_size // 4), so cell edges sit half a unit
        # outside the outermost centers.  pcolormesh renders via the data
        # transform (like the trajectory lines), so it is clipped by the
        # shared XY limit — occupancy does not expand the view.
        if vox is not None:
            half = grid_size // 2
            forward_origin = grid_size // 4
            x_edges = (np.arange(grid_size + 1) - forward_origin - 0.5) * grid_unit
            y_edges = (np.arange(grid_size + 1) - half - 0.5) * grid_unit
            occ_xy = vox[sample_idx].reshape(
                grid_size, grid_size, grid_size
            )[:, :, half]  # [forward, left] cell centers at z=0
            axis.pcolormesh(
                x_edges,
                y_edges,
                occ_xy.T.numpy(),  # rows = left → y, cols = forward → x
                cmap=occ_cmap,
                vmin=0,
                vmax=1,
                shading='flat',
                zorder=0,
            )

        # ── history curve (reverse-integrated) ──
        if history is not None:
            hist_xy = history[sample_idx].numpy()
            axis.plot(
                hist_xy[:, 0], hist_xy[:, 1],
                color='#56B4E9', linewidth=1.6, linestyle='--',
                label='history', zorder=2,
            )

        # ── generated trajectory ──
        axis.plot(
            generated_xy[:, 0], generated_xy[:, 1],
            color='#0072B2', linewidth=2.2, label='generated', zorder=3,
        )

        # ── ground truth ──
        if ground_truth is not None:
            ground_truth_xy = ground_truth[sample_idx].numpy()
            axis.plot(
                ground_truth_xy[:, 0], ground_truth_xy[:, 1],
                color='#666666', linewidth=1.5, linestyle='--',
                label='ground truth', zorder=3,
            )

        # ── goal ray ──
        axis.plot(
            [0.0, goal[0]], [0.0, goal[1]],
            color='#BBBBBB', linewidth=1.0, linestyle=':', label='goal ray',
        )

        # ── markers ──
        axis.scatter([0.0], [0.0], color='#222222', s=35, zorder=6,
                     label='start')
        if history is not None:
            axis.scatter(
                [hist_xy[0, 0]], [hist_xy[0, 1]],
                color='#56B4E9', marker='o', s=30, zorder=5,
                label='history start',
            )
        axis.scatter([goal[0]], [goal[1]], color='#D55E00', marker='X',
                     s=90, zorder=6, label='goal')
        axis.scatter(
            [generated_xy[-1, 0]], [generated_xy[-1, 1]],
            color='#0072B2', marker='s', s=40, zorder=6,
            label='generated end',
        )

        # ── endpoint error ──
        endpoint_error = torch.linalg.vector_norm(
            generated[sample_idx, -1] - goals[sample_idx, :2]
        ).item()
        axis.set_title(
            f'sample {sample_idx}  |  endpoint error {endpoint_error:.3f} m'
        )

        # ── unified scale & ego-consistent labels ──
        axis.set_xlim(-xy_limit, xy_limit)
        axis.set_ylim(-xy_limit, xy_limit)
        axis.set_aspect('equal')
        axis.set_xlabel('x-forward (m)')
        axis.set_ylabel('y-left (m)')
        axis.grid(True, color='#DDDDDD', linewidth=0.7)
        if vox is not None:
            # QuadMesh is not directly supported by legend — use a proxy patch
            # matching the occupied-cell facecolor.
            occ_proxy = Rectangle(
                (0, 0), 1, 1,
                facecolor=(0.5, 0.5, 0.5, 0.4),
                edgecolor='none',
                label='occupied',
            )
            axis.legend(
                loc='best', fontsize=7,
                handles=[occ_proxy] + axis.get_legend_handles_labels()[0],
            )
        else:
            axis.legend(loc='best', fontsize=7)

    for axis in axes.flat[count:]:
        axis.set_visible(False)
    figure.suptitle(
        f'Root XY trajectory  |  ego frame (x-fwd, y-left)  '
        f'|  limit ±{xy_limit:.2f} m'
    )
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


def _validate_batch(batch, cfg) -> None:
    num_primitive = int(cfg.data.num_primitive)
    context_len = int(cfg.data.history_len) + int(cfg.data.future_len)
    nfeats = int(cfg.data.nfeats)
    if len(batch) != num_primitive:
        raise ValueError(
            f"Dataset returned {len(batch)} primitives, expected {num_primitive}"
        )
    for primitive_idx, primitive in enumerate(batch):
        motion = primitive['motion']
        if motion.shape[1:] != (context_len, nfeats):
            raise ValueError(
                f"Primitive {primitive_idx} motion shape is {tuple(motion.shape)}, "
                f"expected [batch, {context_len}, {nfeats}]"
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
    configure_dof_contract(cfg)
    seed.set(cfg.seed)
    logger.set(cfg)
    validate_goal_config(cfg.data.goal_type, cfg.denoiser.goal_dim)
    _validate_goal_position_contract(cfg)
    if cfg.train.manager.use_static_pose:
        raise ValueError(
            "Static-pose replacement has no world reference pose and is not "
            "supported by goal+scene training"
        )

    train_data: Dataset = instantiate(cfg.data.train)
    val_data: Dataset = instantiate(cfg.data.val)

    vae: VAE = instantiate(cfg.vae)
    denoiser: Denoiser = instantiate(cfg.denoiser)

    validate_training_contract(
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
        train_data.set_training_step(manager.step)
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
                            hist_traj = manager.history_trajectory_ego(
                                history_motion
                            )
                            figure = _make_root_xy_figure(
                                sample_trajectory,
                                y['goal'][:, :2],
                                ground_truth_trajectory=ground_truth_trajectory,
                                history_trajectory=hist_traj,
                                goal_condition_keep_mask=goal_keep_mask,
                                voxel=y['voxel'],
                                grid_size=cfg.denoiser.grid_size,
                                grid_unit=cfg.data.occupancy_unit,
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
