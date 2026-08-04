"""Headless fixed-period DAR planner for SONIC controller."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate, to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from robotmdar.dtype import logger as dtype_logger
from robotmdar.dtype import seed
from robotmdar.dtype.abc import Dataset, Denoiser, Diffusion, SSampler, VAE
from robotmdar.dtype.motion import FeatureVersion
from robotmdar.eval.generate_dar import generate_next_motion
from robotmdar.utils.goal import GoalType, validate_goal_config
from robotmdar.utils.planner_convert import (
    align_generated_history_pose,
    generated_history_at_frame,
    motion_dict_to_g1data,
    state_goal_from_reference,
    state_to_ego_goal,
    state_to_model_input,
    tracked_frame_from_timestamps,
)
from robotmdar.train.manager import DARManager


def _load_models(cfg: DictConfig):
    val_data: Dataset = instantiate(cfg.data.val)
    vae: VAE = instantiate(cfg.vae)
    denoiser: Denoiser = instantiate(cfg.denoiser)
    schedule_sampler: SSampler = instantiate(cfg.diffusion.schedule_sampler)
    diffusion: Diffusion = schedule_sampler.diffusion

    vae.eval()
    denoiser.eval()
    manager: DARManager = instantiate(cfg.train.manager)
    manager.hold_model(vae, denoiser, None, val_data)
    return vae, denoiser, diffusion, val_data


def _cuda_synchronize(device: str) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _checkpoint_config(cfg: DictConfig) -> tuple[Path, DictConfig]:
    """Load and validate the config associated with the DAR checkpoint."""
    model_cfg_path = cfg.ckpt.get("load_cfg")
    if model_cfg_path is None:
        ckpt_dar = cfg.ckpt.get("dar")
        if ckpt_dar is None:
            raise ValueError(
                "ckpt.dar must be set so the model config (cfg.yaml) can be located")
        model_cfg_path = Path(str(ckpt_dar)).parent / "cfg.yaml"
    model_cfg_path = Path(to_absolute_path(str(model_cfg_path)))
    model_cfg = OmegaConf.load(model_cfg_path)

    expected = {
        "nfeats": (int(cfg.data.nfeats), int(model_cfg.data.nfeats)),
        "history_len": (
            int(cfg.data.history_len), int(model_cfg.data.history_len)),
        "future_len": (
            int(cfg.data.future_len), int(model_cfg.data.future_len)),
        "goal_dim": (
            int(cfg.denoiser.goal_dim), int(model_cfg.denoiser.goal_dim)),
    }
    mismatches = [
        f"{name}: runtime={runtime}, checkpoint={checkpoint}"
        for name, (runtime, checkpoint) in expected.items()
        if runtime != checkpoint
    ]
    runtime_goal_type = GoalType.parse(cfg.data.goal_type)
    checkpoint_goal_type = GoalType.parse(model_cfg.data.goal_type)
    if runtime_goal_type is not checkpoint_goal_type:
        mismatches.append(
            f"goal_type: runtime={runtime_goal_type.value}, "
            f"checkpoint={checkpoint_goal_type.value}")
    humanoid_type = str(model_cfg.data.skeleton.humanoid_type)
    if int(model_cfg.data.nfeats) != 69 or humanoid_type != "g1_29dof":
        mismatches.append(
            "checkpoint is not a 29-DoF G1 model "
            f"(nfeats={model_cfg.data.nfeats}, humanoid_type={humanoid_type})")
    if mismatches:
        raise ValueError(
            f"Incompatible DAR checkpoint config {model_cfg_path}: "
            + "; ".join(mismatches))
    return model_cfg_path, model_cfg


def main(cfg: DictConfig) -> None:
    """Run the TextOp planner until interrupted."""
    from sonicmsg import PlannerNode
    from sonicmsg.messages import unpack_occ

    dtype_logger.set(cfg)
    seed.set(cfg.seed)
    goal_type = validate_goal_config(
        cfg.data.goal_type, cfg.denoiser.goal_dim)
    goal_reference_path = cfg.get("goal_reference_path")
    if goal_reference_path is not None:
        goal_reference_path = to_absolute_path(str(goal_reference_path))
    if goal_type.uses_keypoints and goal_reference_path is None:
        logger.info(
            "{} planner expects goal_keypoints_world from the controller; "
            "no reference pose is configured", goal_type.value)

    if FeatureVersion != 3:
        raise ValueError(
            f"planner_dar requires FeatureVersion 3, got {FeatureVersion}")

    _model_cfg_path, _model_cfg = _checkpoint_config(cfg)
    logger.info("Loading 29-DoF DAR model and dataset statistics")
    vae, denoiser, diffusion, val_data = _load_models(cfg)
    history_len = int(cfg.data.history_len)
    future_len = int(cfg.data.future_len)
    motion_fps = float(cfg.motion_fps)
    _expected_history_len = int(_model_cfg.data.history_len)
    _expected_future_len = int(_model_cfg.data.future_len)
    if history_len != _expected_history_len:
        raise ValueError(
            f"Runtime history_len ({history_len}) does not match "
            f"model checkpoint history_len ({_expected_history_len}) "
            f"from {_model_cfg_path}")
    if future_len != _expected_future_len:
        raise ValueError(
            f"Runtime future_len ({future_len}) does not match "
            f"model checkpoint future_len ({_expected_future_len}) "
            f"from {_model_cfg_path}")
    if abs(float(val_data.fps) - motion_fps) > 1e-6:
        raise ValueError(
            f"Dataset fps ({val_data.fps}) must match motion_fps ({motion_fps})")

    period = float(cfg.infer_period_ms) / 1000.0
    if period <= 0:
        raise ValueError(f"infer_period_ms must be positive, got {cfg.infer_period_ms}")
    state_timeout_ms = int(cfg.state_timeout_ms)
    log_every = max(1, int(cfg.log_every))
    comm_config = to_absolute_path(str(cfg.comm_config))
    node = PlannerNode(comm_config)

    grid_size = int(cfg.denoiser.grid_size)
    n_voxels = grid_size**3
    latest_state = None
    last_inferred_seq = None
    next_infer_time = time.perf_counter()
    inference_count = 0
    use_generated_history = bool(cfg.use_generated_history)
    align_generated_history = bool(cfg.align_generated_history_to_g1)
    generated_plans = {}
    next_ack_log_time = 0.0
    infer_times: list[float] = []  # rolling window for running average
    fixed_sampling_noise = None
    if not bool(cfg.get("resample_noise_each_plan", False)):
        fixed_sampling_noise = torch.randn(
            (1, *denoiser.noise_shape), device=cfg.device)

    logger.info(
        "TextOp planner ready: replan={:.1f} Hz, motion={:.1f} Hz, "
        "history={} features/{} states, future={} frames, history_source={}",
        1.0 / period, motion_fps, history_len, history_len, future_len,
        ("generated+translated" if align_generated_history else "generated")
        if use_generated_history else "controller")

    try:
        while True:
            while True:
                message = node.recv_state(timeout_ms=state_timeout_ms)
                if message is None:
                    break
                latest_state = message

            now = time.perf_counter()
            if latest_state is None or now < next_infer_time:
                time.sleep(0.001)
                continue

            state_seq = int(latest_state.header.seq)
            if state_seq == last_inferred_seq:
                time.sleep(0.001)
                continue
            tracked_plan = None
            if use_generated_history and generated_plans:
                tracked_plan = generated_plans.get(
                    latest_state.tracked_plan_seq)
                if tracked_plan is None:
                    # Do not advance the autoregressive chain until the
                    # controller acknowledges applying one of our plans.
                    if now >= next_ack_log_time:
                        logger.info(
                            "Waiting for controller plan acknowledgment: "
                            "active={} cached={}",
                            latest_state.tracked_plan_seq,
                            list(generated_plans))
                        next_ack_log_time = now + 1.0
                    time.sleep(0.001)
                    continue
            last_inferred_seq = state_seq
            scheduled_next = next_infer_time + period

            try:
                state_goal_type = GoalType.parse(latest_state.goal_type)
                if state_goal_type is not goal_type:
                    raise ValueError(
                        f"Planner is configured for goal_type={goal_type.value!r}, "
                        f"but controller sent {state_goal_type.value!r}")
                using_generated_history = (
                    use_generated_history and tracked_plan is not None)
                if using_generated_history:
                    tracked_frame = tracked_frame_from_timestamps(
                        latest_state, motion_fps, future_len)
                    (history_motion, generated_abs_pose,
                     generated_reference_pos, generated_reference_rot) = (
                        generated_history_at_frame(
                            tracked_plan, tracked_frame, history_len))
                    if align_generated_history:
                        (abs_pose, goal_reference_pos,
                         goal_reference_rot, history_translation,
                         history_motion) = (
                            align_generated_history_pose(
                                generated_abs_pose, generated_reference_pos,
                                generated_reference_rot,
                                latest_state, cfg.device,
                                history_motion=history_motion,
                                val_data=val_data))
                        ego_goal = state_goal_from_reference(
                            latest_state, goal_reference_pos,
                            goal_reference_rot, cfg.device,
                            goal_type=goal_type,
                            goal_reference_path=goal_reference_path)
                    else:
                        abs_pose = {
                            k: v.to(cfg.device)
                            for k, v in generated_abs_pose.items()
                        }
                        history_translation = None
                        ego_goal = state_goal_from_reference(
                            latest_state, generated_reference_pos,
                            generated_reference_rot, cfg.device,
                            goal_type=goal_type,
                            goal_reference_path=goal_reference_path)
                else:
                    tracked_frame = None
                    if latest_state.current_root_rot is None:
                        raise ValueError(
                            "Received a legacy history_state; TextOp requires "
                            "history_state_textop_v2")
                    if latest_state.n_states < history_len:
                        raise ValueError(
                            f"State {state_seq} has {latest_state.n_states} "
                            f"entries; need at least {history_len}")
                    history_motion, abs_pose = state_to_model_input(
                        latest_state, history_len, val_data, cfg.device)
                    ego_goal = state_to_ego_goal(
                        latest_state, cfg.device, goal_type=goal_type,
                        goal_reference_path=goal_reference_path)
                    history_translation = None

                # DEBUG: overwrite the ego_goal
                # ego_goal[0, 0] = 0.1
                # ego_goal[0, 1] = 0.0
                # ego_goal[0, 2] = 0.0

                _goal_ego_x = float(ego_goal[0, 0])
                _goal_ego_y = float(ego_goal[0, 1])
                _goal_delta_z = float(ego_goal[0, 2])

                _world_goal = getattr(latest_state, 'goal_root_pos_world', None)
                if _world_goal is not None:
                    _world_goal = np.asarray(_world_goal, dtype=np.float64).reshape(-1)
                    _goal_world_x = float(_world_goal[0])
                    _goal_world_y = float(_world_goal[1])
                    _goal_world_z = float(_world_goal[2])
                else:
                    _goal_world_x = _goal_world_y = _goal_world_z = float('nan')

                _world_vel = getattr(latest_state, 'goal_root_velocity_world', None)
                if _world_vel is not None:
                    _world_vel = np.asarray(_world_vel, dtype=np.float64).reshape(-1)
                    _vel_world_x = float(_world_vel[0])
                    _vel_world_y = float(_world_vel[1])
                    _vel_world_z = float(_world_vel[2])
                else:
                    _vel_world_x = _vel_world_y = _vel_world_z = float('nan')

                # ── world (from controller) ──
                _world_yaw = getattr(latest_state, 'goal_yaw_world', None)
                if _world_yaw is not None:
                    _goal_yaw_world_deg = math.degrees(
                        float(np.asarray(_world_yaw, dtype=np.float64).reshape(-1)[0]))
                else:
                    _goal_yaw_world_deg = float('nan')

                # ── ego (as seen by planner, after dropout) ──
                _force_drop_yaw = bool(cfg.get("force_drop_goal_yaw", False))
                _force_drop_time = bool(cfg.get("force_drop_goal_time", False))
                if goal_type in (GoalType.ROOT, GoalType.BODY_EXT):
                    if _force_drop_yaw:
                        _goal_ego_yaw_deg = 0.0  # mask_condition zeros the yaw channels
                    else:
                        _goal_ego_yaw_deg = math.degrees(math.atan2(
                            float(ego_goal[0, 4]), float(ego_goal[0, 3])))
                else:
                    _goal_ego_yaw_deg = float('nan')

                _cuda_synchronize(str(cfg.device))
                infer_start = time.perf_counter()
                if latest_state.ego_occ is None:
                    raise ValueError(
                        f"State {state_seq} does not contain ego occupancy")
                ego_occ = unpack_occ(latest_state.ego_occ, n_voxels)
                if ego_occ.size != n_voxels:
                    raise ValueError(
                        f"State {state_seq} occupancy has {ego_occ.size} "
                        f"voxels; expected {n_voxels} for grid_size={grid_size}")
                voxel = torch.as_tensor(
                    ego_occ, dtype=torch.float32, device=cfg.device
                ).unsqueeze(0)
                future_motion, motion_dict, _new_abs_pose = generate_next_motion(
                    vae=vae,
                    denoiser=denoiser,
                    diffusion=diffusion,
                    val_data=val_data,
                    goal=ego_goal,
                    voxel=voxel,
                    history_motion=history_motion,
                    abs_pose=abs_pose,
                    future_len=future_len,
                    use_full_sample=bool(cfg.use_full_sample),
                    guidance_scale=cfg.guidance_scale,
                    initial_noise=fixed_sampling_noise,
                    ret_fk=True,
                    force_drop_goal_root=bool(
                        cfg.get("force_drop_goal_root", False)),
                    force_drop_goal_yaw=bool(
                        cfg.get("force_drop_goal_yaw", False)),
                    force_drop_goal_time=bool(
                        cfg.get("force_drop_goal_time", False)),
                    force_drop_goal_body=bool(
                        cfg.get("force_drop_goal_body", False)),
                )
                _cuda_synchronize(str(cfg.device))
                infer_ms = (time.perf_counter() - infer_start) * 1000.0
                infer_times.append(infer_ms)
                avg_ms = sum(infer_times[-20:]) / len(infer_times[-20:])
                occ_count = int(voxel.sum().item())
                if goal_type is GoalType.ROOT:
                    logger.info(
                        "goal[root]: world(pos=({:.3f},{:.3f},{:.3f}) "
                        "yaw={:.1f} deg) "
                        "ego(pos=({:.3f},{:.3f},{:.3f}) "
                        "yaw={:.1f} deg) | "
                        "occ={} | infer={:.1f} ms (avg20={:.1f} ms)",
                        _goal_world_x, _goal_world_y, _goal_world_z,
                        _goal_yaw_world_deg,
                        _goal_ego_x, _goal_ego_y, _goal_delta_z,
                        _goal_ego_yaw_deg,
                        occ_count, infer_ms, avg_ms)
                elif goal_type is GoalType.BODY:
                    logger.info(
                        "goal[body]: world(pos=({:.3f},{:.3f},{:.3f}) "
                        "yaw={:.1f} deg) "
                        "ego(pos=({:.3f},{:.3f},{:.3f}) "
                        "yaw={:.1f} deg) | "
                        "occ={} | infer={:.1f} ms (avg20={:.1f} ms)",
                        _goal_world_x, _goal_world_y, _goal_world_z,
                        _goal_yaw_world_deg,
                        _goal_ego_x, _goal_ego_y, _goal_delta_z,
                        _goal_ego_yaw_deg,
                        occ_count, infer_ms, avg_ms)
                else:
                    logger.info(
                        "goal[body_ext]: world(pos=({:.3f},{:.3f},{:.3f}) "
                        "yaw={:.1f} deg vel=({:.3f},{:.3f},{:.3f})) "
                        "ego(pos=({:.3f},{:.3f},{:.3f}) "
                        "yaw={:.1f} deg "
                        "vel=({:.3f},{:.3f},{:.3f}) dt={:.3f}s) | "
                        "occ={} | infer={:.1f} ms (avg20={:.1f} ms)",
                        _goal_world_x, _goal_world_y, _goal_world_z,
                        _goal_yaw_world_deg,
                        _vel_world_x, _vel_world_y, _vel_world_z,
                        _goal_ego_x, _goal_ego_y, _goal_delta_z,
                        _goal_ego_yaw_deg,
                        float(ego_goal[0, 5]), float(ego_goal[0, 6]),
                        float(ego_goal[0, 7]),
                        0.0 if _force_drop_time else float(ego_goal[0, 8]),
                        occ_count, infer_ms, avg_ms)

                # SonicRunner resets each newly received G1 plan to frame 0.
                # Keep the current measured history frame as that exact seam;
                # dropping all history would jump directly to stochastic t+1.
                skip_history = (
                    0 if bool(cfg.pub_all_frames) else history_len - 1)
                motion = motion_dict_to_g1data(
                    motion_dict, skip_history=skip_history, fps=motion_fps)
                measured_pos = np.asarray(
                    latest_state.raw["g1_pos"][-1], dtype=np.float32)
                measured_rot_xyzw = np.asarray(
                    latest_state.raw["g1_root_rot"][-1], dtype=np.float32)
                published_rot_xyzw = motion.body_ori[0, 0, [1, 2, 3, 0]]
                quat_dot = float(np.clip(np.abs(np.dot(
                    measured_rot_xyzw, published_rot_xyzw)), 0.0, 1.0))
                seam_root_error = float(np.linalg.norm(
                    motion.body_pos[0, 0] - measured_pos))
                seam_root_angle_deg = math.degrees(2.0 * math.acos(quat_dot))
                seam_joint_error = float(np.max(np.abs(
                    motion.joint_pos[0]
                    - np.asarray(latest_state.raw["g1_joint_pos"][-1],
                                 dtype=np.float32))))
                published_seq = node.publish_motion(motion)

                if use_generated_history:
                    generated_plans[published_seq] = {
                        "features": torch.cat(
                            (history_motion, future_motion), dim=1
                        ).detach().clone(),
                        "root_pos": motion_dict[
                            "root_trans_offset"].detach().clone(),
                        "root_rot": motion_dict["root_rot"].detach().clone(),
                    }
                    while len(generated_plans) > 16:
                        del generated_plans[next(iter(generated_plans))]

                inference_count += 1
                if inference_count == 1 or inference_count % log_every == 0:
                    logger.info(
                        "plan={} state={} motion={} frames infer={:.1f} ms "
                        "(avg20={:.1f} ms) "
                        "history={} tracked_plan={} frame={} shift={:.3f} m "
                        "seam=({:.4f} m, {:.2f} deg, {:.4f} rad)",
                        published_seq, state_seq, motion.num_frames, infer_ms, avg_ms,
                        "generated" if using_generated_history else "controller",
                        (latest_state.tracked_plan_seq
                         if using_generated_history else -1),
                        tracked_frame if tracked_frame is not None else -1,
                        (float(torch.linalg.vector_norm(history_translation))
                         if history_translation is not None else 0.0),
                        seam_root_error, seam_root_angle_deg, seam_joint_error)
            except Exception:
                logger.exception("Failed to process controller state {}", state_seq)

            finished = time.perf_counter()
            next_infer_time = (
                finished + period if finished > scheduled_next
                else scheduled_next)
    except KeyboardInterrupt:
        logger.info("Planner interrupted")
    finally:
        node.close()
