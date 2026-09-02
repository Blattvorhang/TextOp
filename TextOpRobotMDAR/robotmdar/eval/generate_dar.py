"""
DAR Motion Generation Module

Common motion generation functionality extracted from vis_dar.py and loop_dar.py.
Supports both full_sample (complete DDPM sampling) and single_step_sample modes.

Functions:
- generate_next_motion: Generate next motion segment using DAR model
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union

import joblib
import numpy as np
import torch
from loguru import logger
from torch import nn

import robotmdar.dtype.motion as motion_dtype
from robotmdar.dtype.motion import motion_dict_to_abs_pose


class ClassifierFreeWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

    def forward(self, x, timesteps, y: Dict[str, Any]):
        y['uncond'] = False
        out = self.model(x, timesteps, y)
        y_uncond = y
        y_uncond['uncond'] = True
        out_uncond = self.model(x, timesteps, y_uncond)
        # print('scale:', y['scale'])
        # diff_cond = (out - out_uncond).norm()
        # print('diff_cond:', diff_cond, 'out', out.norm(), 'out_uncond',
        #       out_uncond.norm())
        return out_uncond + (y['scale'] * (out - out_uncond))

    @property
    def noise_shape(self):
        return self.model.noise_shape


def encode_motion_lib_initial_noise(
        vae,
        val_data,
        motion_path: str,
        start_frame: int,
        history_len: int,
        future_len: int,
        device: str,
        clip_name: Optional[str] = None,
) -> torch.Tensor:
    """Encode a motion-library clip segment as diffusion initial noise.

    Loads a joblib motion_lib pkl (``{name: motion_entry}`` dict), selects the
    target clip, converts it to the TextOp sample format, then reuses the
    Dataset pipeline (*_extract_single_primitive* → *_convert_to_motion_features*
    → *normalize*) for feature extraction, and finally VAE-encodes the segment
    to produce a latent tensor suitable as ``generate_next_motion``'s
    *initial_noise*.

    Returns:
        Tensor of shape ``(1, *denoiser.noise_shape)`` — typically ``(1, 1, 128)``.
    """
    motion_path_resolved = Path(motion_path)
    if not motion_path_resolved.exists():
        raise FileNotFoundError(
            f"motion_lib pkl not found: {motion_path_resolved}")

    payload = joblib.load(motion_path_resolved)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a dict, got {type(payload).__name__} "
            f"from {motion_path_resolved}")

    # ── select clip ──
    _MOTION_REQUIRED_KEYS = frozenset((
        "root_trans_offset", "root_rot", "dof", "contact_mask",
    ))
    if _MOTION_REQUIRED_KEYS.issubset(payload.keys()):
        # flat single-clip pkl
        if clip_name is not None:
            raise ValueError(
                "clip_name was given but the pkl is a flat single-clip dict")
        entry = payload
        used_clip = None
    elif all(isinstance(v, dict) for v in payload.values()):
        if clip_name is not None:
            if clip_name not in payload:
                raise ValueError(
                    f"clip {clip_name!r} not found in pkl; "
                    f"available: {sorted(payload)}")
            entry = payload[clip_name]
            used_clip = clip_name
        elif len(payload) == 1:
            (key, entry), = payload.items()
            used_clip = key
        else:
            raise ValueError(
                f"Multi-clip pkl requires motion_clip; "
                f"available: {sorted(payload)}")
    else:
        raise ValueError(
            f"Unrecognized pkl structure: top-level keys {sorted(payload)}")

    used_name = (
        f"{motion_path_resolved}::{used_clip}" if used_clip
        else str(motion_path_resolved))
    logger.info("motion_lib clip selected: {}", used_name)

    # ── validate fps ──
    pkl_fps = float(entry.get("fps", 0.0))
    if pkl_fps <= 0:
        raise ValueError(
            f"Motion clip {used_name!r} has invalid fps={pkl_fps}")
    if abs(pkl_fps - float(val_data.fps)) > 1e-6:
        raise ValueError(
            f"Motion clip fps ({pkl_fps}) does not match "
            f"val_data.fps ({val_data.fps})")

    # ── build TextOp sample dict ──
    try:
        root_trans_offset = np.asarray(entry["root_trans_offset"])
        root_rot = np.asarray(entry["root_rot"])
        dof = np.asarray(entry["dof"])
        contact_mask = np.asarray(entry["contact_mask"])
    except KeyError as exc:
        raise ValueError(
            f"Motion clip {used_name!r} missing key {exc}") from exc

    sliding_mask = np.asarray(
        entry.get("sliding_mask", contact_mask))

    sample = {
        "motion": {
            "root_trans_offset": root_trans_offset,
            "root_rot": root_rot,
            "dof": dof,
            "contact_mask": contact_mask,
            "sliding_mask": sliding_mask,
        },
        "scene": entry.get("scene", {}),
    }

    # ── slice + contract DOFs + convert features via training pipeline ──
    prim_start = int(start_frame)
    prim_end = prim_start + int(future_len) + int(history_len) + 1
    goal_frame = prim_end - 1  # not used for motion; any valid index works

    primitive = val_data._extract_single_primitive(
        sample, prim_start, prim_end, goal_frame)
    features = val_data._convert_to_motion_features(
        [primitive["motion"]])  # [1, H+F, nfeats]

    # ── normalize ──
    features = val_data.normalize(features).to(device)

    # ── split history / future ──
    history_motion = features[:, :history_len, :]
    future_motion = features[:, history_len:, :]

    # ── VAE encode ──
    with torch.no_grad():
        latent, _dist = vae.encode(future_motion, history_motion)
    # latent: [latent_size=1, B=1, latent_dim=128]

    # Permute to denoiser-compatible shape: [B=1, T=1, D=128]
    initial_noise = latent.permute(1, 0, 2)

    return initial_noise


def generate_next_motion(
        vae,
        denoiser,
        diffusion,
        val_data,
        goal: torch.Tensor,
        voxel: torch.Tensor,
        history_motion: torch.Tensor,
        abs_pose,  #  AbsolutePose
        future_len: int,
        use_full_sample: bool = False,
        guidance_scale: Optional[float] = None,
        initial_noise: Optional[torch.Tensor] = None,
        ret_fk: bool = False,
        ret_fk_full: bool = False,
        use_vae=True,
        use_ddim=False,
        force_drop_goal_root: bool = False,
        force_drop_goal_yaw: bool = False,
        force_drop_goal_time: bool = False,
        force_drop_goal_body: bool = False,
        force_drop_goal_orientation: bool = False,
        force_drop_goal_joint: bool = False,
        force_drop_goal_velocity: bool = False,
        force_drop_arrival_time: bool = False,
        time_to_arrival_frame: Optional[torch.Tensor] = None):
    """
    Generate next motion segment using DAR model.

    Args:
        vae: VAE model for encoding/decoding
        denoiser: Denoiser model for diffusion
        diffusion: Diffusion model
        val_data: Dataset for motion reconstruction
        goal: Ego-centric goal tensor [B, denoiser.goal_dim].
        voxel: Scene occupancy tensor [B, grid_size^3]
        history_motion: History motion tensor [B, T_hist, D]
        abs_pose: Current absolute pose
        future_len: Length of future motion to generate
        use_full_sample: Whether to use full DDPM sampling loop
        force_drop_goal_root: Mask the root-position component at inference.
        force_drop_goal_yaw: Mask the root-yaw component at inference.
        force_drop_goal_time: Mask the remaining-time component at inference.
        force_drop_goal_body: Mask the limb-keypoint component at inference.
        force_drop_goal_orientation: Mask the joint_state orientation block.
        force_drop_goal_joint: Mask the joint_state 29-DOF target block.
        force_drop_goal_velocity: Mask the joint_state root-velocity block.
        force_drop_arrival_time: Mask the arrival-time PE at inference.
        time_to_arrival_frame: Optional frame index passed to the arrival PE.

    Returns:
        Tuple of (future_motion_pred, motion_dict, new_abs_pose)
    """
    device = history_motion.device
    with torch.no_grad():
        batch_size = goal.shape[0]
        # latent_shape = (batch_size, 1, 128
        #                 )  # [B, T=1, D] - latent_dim from config
        latent_shape = (batch_size, *denoiser.noise_shape)

        # Online replanning should reuse one noise realization. Resampling at
        # every MPC update makes otherwise-nearby conditions decode to visibly
        # different motions at the plan seam.
        if initial_noise is None:
            x_start_noise = torch.randn(latent_shape, device=device)
        else:
            if tuple(initial_noise.shape) != tuple(latent_shape):
                raise ValueError(
                    f"initial_noise has shape {tuple(initial_noise.shape)}, "
                    f"expected {tuple(latent_shape)}")
            x_start_noise = initial_noise.to(device=device)

        # Sample a random timestep for demonstration (or use t=0 for no noise)
        t = torch.zeros(batch_size, dtype=torch.int32,
                        device=device) + diffusion.num_timesteps - 1

        # Prepare conditioning
        force_drop_arrival_time = (
            bool(force_drop_arrival_time) or bool(force_drop_goal_time)
        )
        if time_to_arrival_frame is None and goal.shape[-1] == 21:
            fps = float(getattr(val_data, 'fps', 50.0))
            time_to_arrival_frame = torch.round(
                goal[:, 8].clamp_min(0.0) * fps
            ).to(dtype=torch.long)
        y: Dict[str, Any] = {
            'goal': goal,
            'voxel': voxel,
            'history_motion_normalized': history_motion,
            'force_drop_goal_root': force_drop_goal_root,
            'force_drop_goal_yaw': force_drop_goal_yaw,
            'force_drop_goal_time': force_drop_goal_time,
            'force_drop_goal_body': force_drop_goal_body,
            'force_drop_goal_orientation': force_drop_goal_orientation,
            'force_drop_goal_joint': force_drop_goal_joint,
            'force_drop_goal_velocity': force_drop_goal_velocity,
            'force_drop_arrival_time': force_drop_arrival_time,
        }
        if time_to_arrival_frame is not None:
            y['time_to_arrival_frame'] = time_to_arrival_frame
            y['arrival_time_frame'] = time_to_arrival_frame
        if guidance_scale is not None:
            y['scale'] = guidance_scale

        # print(diffusion.num_timesteps)
        if use_full_sample:
            if not use_ddim:
                # Use complete DDPM sampling loop
                sample_fn = diffusion.p_sample_loop
                x_start_pred = sample_fn(
                    denoiser,
                    latent_shape,
                    clip_denoised=False,
                    model_kwargs={'y': y},  # Wrap y in the expected structure
                    skip_timesteps=0,
                    init_image=None,
                    progress=False,
                    dump_steps=None,
                    noise=x_start_noise,
                    const_noise=False,
                )
            else:
                # zjk: use DDIM sampling loop
                sample_fn = diffusion.ddim_sample_loop
                x_start_pred = sample_fn(
                    denoiser,
                    latent_shape,
                    clip_denoised=False,
                    model_kwargs={'y': y},  # Wrap y in the expected structure
                    skip_timesteps=0,
                    init_image=None,
                    progress=False,
                    eta=0.0,
                    dump_steps=None,
                    noise=x_start_noise,
                    const_noise=False,
                )


            assert isinstance(x_start_pred, torch.Tensor), \
                f"Expected tensor, got {type(x_start_pred)}"
        else:
            # Single step denoising (default mode)
            x_start_pred = denoiser(x_t=x_start_noise,
                                    timesteps=diffusion._scale_timesteps(t),
                                    y=y)  # [B, T=1, D]

        if use_vae:
            # Convert to VAE format [T=1, B, D]
            latent_pred = x_start_pred.permute(1, 0, 2)

            # breakpoint()

            # Decode using VAE
            # latent_pred: (1, 1, 128)  history_motion: (1, 2, 57)

            future_motion_pred = vae.decode(latent_pred,
                                            history_motion,
                                            nfuture=future_len)
        else:
            future_motion_pred = x_start_pred[:, -future_len:]

        # Reconstruct motion dictionary
        motion_dict = val_data.reconstruct_motion(torch.cat(
            [history_motion, future_motion_pred], dim=1),
                                                  abs_pose=abs_pose,
                                                  ret_fk=ret_fk,
                                                  ret_fk_full=ret_fk_full)

        # Update absolute pose for next primitive
        pose_idx = -1 if motion_dtype.FeatureVersion == 6 else -2
        new_abs_pose = motion_dict_to_abs_pose(motion_dict, idx=pose_idx)

        return future_motion_pred, motion_dict, new_abs_pose
