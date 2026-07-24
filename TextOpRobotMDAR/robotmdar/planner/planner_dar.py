"""Headless fixed-period DAR planner for SONIC controller."""

from __future__ import annotations

import math
import time

import torch
from hydra.utils import instantiate, to_absolute_path
from loguru import logger
from omegaconf import DictConfig

from robotmdar.dtype import logger as dtype_logger
from robotmdar.dtype import seed
from robotmdar.dtype.abc import Dataset, Denoiser, Diffusion, SSampler, VAE
from robotmdar.dtype.motion import FeatureVersion
from robotmdar.eval.generate_dar import generate_next_motion
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


def main(cfg: DictConfig) -> None:
    """Run the TextOp planner until interrupted."""
    from sonicmsg import PlannerNode

    dtype_logger.set(cfg)
    seed.set(cfg.seed)
    if FeatureVersion != 3:
        raise ValueError(
            f"planner_dar requires FeatureVersion 3, got {FeatureVersion}")

    logger.info("Loading DAR model and dataset statistics")
    vae, denoiser, diffusion, val_data = _load_models(cfg)
    history_len = int(cfg.data.history_len)
    future_len = int(cfg.data.future_len)
    motion_fps = float(cfg.motion_fps)
    if history_len != 2:
        raise ValueError(f"TextOp checkpoint expects history_len=2, got {history_len}")
    if future_len != 8:
        raise ValueError(f"TextOp controller expects future_len=8, got {future_len}")
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
    voxel = torch.zeros(
        (1, grid_size**3), dtype=torch.float32, device=cfg.device)
    latest_state = None
    last_inferred_seq = None
    next_infer_time = time.perf_counter()
    inference_count = 0
    use_generated_history = bool(cfg.use_generated_history)
    align_generated_history = bool(cfg.align_generated_history_to_g1)
    generated_plans = {}
    next_ack_log_time = 0.0
    infer_times: list[float] = []  # rolling window for running average

    logger.info(
        "TextOp planner ready: replan={:.1f} Hz, motion={:.1f} Hz, "
        "history={} features/{} states, future={} frames, history_source={}",
        1.0 / period, motion_fps, history_len, history_len + 1, future_len,
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
                        abs_pose, goal_reference_pos, history_translation = (
                            align_generated_history_pose(
                                generated_abs_pose, generated_reference_pos,
                                latest_state, cfg.device))
                        ego_goal = state_goal_from_reference(
                            latest_state, goal_reference_pos,
                            generated_reference_rot, cfg.device)
                    else:
                        abs_pose = {
                            k: v.to(cfg.device)
                            for k, v in generated_abs_pose.items()
                        }
                        history_translation = None
                        ego_goal = state_goal_from_reference(
                            latest_state, generated_reference_pos,
                            generated_reference_rot, cfg.device)
                else:
                    tracked_frame = None
                    if latest_state.current_root_rot is None:
                        raise ValueError(
                            "Received a legacy history_state; TextOp requires "
                            "history_state_textop_v2")
                    if latest_state.n_states < history_len + 1:
                        raise ValueError(
                            f"State {state_seq} has {latest_state.n_states} "
                            f"entries; need at least {history_len + 1}")
                    history_motion, abs_pose = state_to_model_input(
                        latest_state, history_len, val_data, cfg.device)
                    ego_goal = state_to_ego_goal(latest_state, cfg.device)
                    history_translation = None

                # DEBUG: overwrite the ego_goal
                # ego_goal[0, 0] = 0.1
                # ego_goal[0, 1] = 0.0
                # ego_goal[0, 2] = 0.0

                _goal_ego_x = float(ego_goal[0, 0])
                _goal_ego_y = float(ego_goal[0, 1])
                _goal_delta_z = float(ego_goal[0, 2])
                _goal_delta_yaw_rad = math.atan2(
                    float(ego_goal[0, 4]), float(ego_goal[0, 3]))
                _goal_delta_yaw_deg = math.degrees(_goal_delta_yaw_rad)
                _cuda_synchronize(str(cfg.device))
                infer_start = time.perf_counter()
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
                    ret_fk=True,
                )
                _cuda_synchronize(str(cfg.device))
                infer_ms = (time.perf_counter() - infer_start) * 1000.0
                infer_times.append(infer_ms)
                avg_ms = sum(infer_times[-20:]) / len(infer_times[-20:])
                logger.info(
                    "goal: ego_x={:.3f} ego_y={:.3f} delta_z={:.3f} "
                    "delta_yaw={:.1f}° | infer={:.1f} ms (avg20={:.1f} ms)",
                    _goal_ego_x, _goal_ego_y, _goal_delta_z, _goal_delta_yaw_deg,
                    infer_ms, avg_ms)

                skip_history = 0 if bool(cfg.pub_all_frames) else history_len
                motion = motion_dict_to_g1data(
                    motion_dict, skip_history=skip_history, fps=motion_fps)
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
                        "history={} tracked_plan={} frame={} shift={:.3f} m",
                        published_seq, state_seq, motion.num_frames, infer_ms, avg_ms,
                        "generated" if using_generated_history else "controller",
                        (latest_state.tracked_plan_seq
                         if using_generated_history else -1),
                        tracked_frame if tracked_frame is not None else -1,
                        (float(torch.linalg.vector_norm(history_translation))
                         if history_translation is not None else 0.0))
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
