"""
DAR Motion Generation Module

Common motion generation functionality extracted from vis_dar.py and loop_dar.py.
Supports both full_sample (complete DDPM sampling) and single_step_sample modes.

Functions:
- generate_next_motion: Generate next motion segment using DAR model
"""

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union

import joblib
import numpy as np
import torch
from loguru import logger
from torch import nn

import robotmdar.dtype.motion as motion_dtype
from robotmdar.dtype.motion import motion_dict_to_abs_pose
from robotmdar.model.mld_denoiser import (
    _condition_valid_mask,
    _force_drop_end_effector,
    _goal_valid_mask,
)
from robotmdar.utils.goal import (
    EXTENDED_BODY_GOAL_DIM,
    GoalEncoding,
    SPLIT_END_EFFECTOR_GOAL_DIM,
    SPLIT_END_EFFECTOR_TOKEN_ORDER,
    SPLIT_GOAL_DIM,
    SPLIT_TIME_SLICE,
)


def denoiser_supports_text_guidance(model) -> bool:
    if hasattr(model, 'text_condition_enabled'):
        return bool(model.text_condition_enabled)
    return float(getattr(model, 'cond_text_mask_prob', 0.0)) > 0.0


class ClassifierFreeWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

        assert denoiser_supports_text_guidance(self.model), (
            'Cannot run a guided diffusion on a model that has not been '
            'trained with text-conditioning dropout'
        )

    def forward(self, x_t=None, timesteps=None, y: Optional[Dict[str, Any]] = None,
                **kwargs):
        if x_t is None:
            x_t = kwargs.pop('x', None)
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(
                f"ClassifierFreeWrapper.forward() got an unexpected keyword "
                f"argument {unexpected!r}"
            )
        if x_t is None:
            raise TypeError("ClassifierFreeWrapper.forward() missing x_t")
        if timesteps is None:
            raise TypeError("ClassifierFreeWrapper.forward() missing timesteps")
        if y is None:
            raise ValueError("Classifier-free guidance requires conditions")

        y_cond = dict(y)
        y_cond['uncond'] = False
        out = self.model(x_t=x_t, timesteps=timesteps, y=y_cond)
        y_uncond = dict(y)
        y_uncond['uncond'] = True
        out_uncond = self.model(x_t=x_t, timesteps=timesteps, y=y_uncond)
        scale = y.get('scale', 1.0)
        return out_uncond + (scale * (out - out_uncond))

    @property
    def noise_shape(self):
        return self.model.noise_shape


class GoalClassifierFreeWrapper(nn.Module):
    """Guide EE without joints, then add the joint-conditioned increment.

    With other conditions fixed, use f0 (no joint/EE), fE (EE only), and
    fJE (joint + EE): f0 + end_effector*(fE-f0) + joint*(fJE-fE).
    Apply CFG only to samples with valid joints and at least one valid EE.
    Other samples use the original prediction, regardless of CFG scales.
    Both scales at one recover fJE. For eligible samples a zero scale masks
    that modality in every branch, including the EE context of joint guidance.
    Existing validity and force-drop masks always remain authoritative.
    """

    def __init__(self, model, joint: float = 1.0,
                 end_effector: float = 1.0):
        super().__init__()
        self.model = model
        self.joint = float(joint)
        self.end_effector = float(end_effector)
        for name in ('joint', 'end_effector'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'goal_guidance.{name} must be finite and >= 0')
        if self.joint != 1.0 or self.end_effector != 1.0:
            base_model = (model.model if isinstance(model, ClassifierFreeWrapper)
                          else model)
            if not GoalEncoding.parse(
                    getattr(base_model, 'goal_encoding', None)).uses_end_effectors:
                raise ValueError('Goal CFG requires split_end_effector encoding')

    def forward(self, x_t=None, timesteps=None, y=None, **kwargs):
        if x_t is None:
            x_t = kwargs.pop('x', None)
        if kwargs:
            raise TypeError(f'Unexpected Goal CFG argument: {next(iter(kwargs))!r}')
        if x_t is None or timesteps is None or y is None:
            raise ValueError('Goal CFG requires x_t, timesteps, and conditions')

        # Do not carry model-written masks between branches or diffusion steps.
        conditions = {
            key: value for key, value in y.items()
            if not key.endswith('_condition_keep_mask')
            and key != '_condition_sampling_plan'
        }
        if self.joint == 1.0 and self.end_effector == 1.0:
            return self.model(x_t=x_t, timesteps=timesteps, y=conditions)

        active = self._active_mask(conditions, x_t.shape[0], x_t.device)
        if not active.any():
            return self.model(x_t=x_t, timesteps=timesteps, y=conditions)
        if active.all():
            return self._guided_forward(x_t, timesteps, conditions)

        # Keep inactive samples out of all additional CFG branches. Slice the
        # whole condition tree so per-sample validity, history, and text agree.
        result = torch.empty_like(x_t)
        for selected, guided in ((~active, False), (active, True)):
            indices = selected.nonzero(as_tuple=False).flatten()
            batch_conditions = self._slice_batch(conditions, indices, len(active))
            batch_x = x_t.index_select(0, indices)
            batch_t = self._slice_batch(timesteps, indices, len(active))
            prediction = (
                self._guided_forward(batch_x, batch_t, batch_conditions)
                if guided else self.model(
                    x_t=batch_x, timesteps=batch_t, y=batch_conditions)
            )
            result.index_copy_(0, indices, prediction)
        return result

    @staticmethod
    def _active_mask(y, batch_size, device):
        def valid(name):
            value = _goal_valid_mask(y, name, batch_size, device)
            return (torch.ones(batch_size, dtype=torch.bool, device=device)
                    if value is None else value)

        joint_valid = valid('joint') & ~_condition_valid_mask(
            y.get('force_drop_goal_joint', False), batch_size, device)
        ee_parent_valid = valid('end_effector')
        ee_valid = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for name in SPLIT_END_EFFECTOR_TOKEN_ORDER:
            if not _force_drop_end_effector(y, name):
                ee_valid |= ee_parent_valid & valid(f'end_effector_{name}')
        return joint_valid & ee_valid

    @classmethod
    def _slice_batch(cls, value, indices, batch_size):
        if torch.is_tensor(value):
            if value.ndim and value.shape[0] == batch_size:
                return value.index_select(0, indices.to(value.device))
        elif isinstance(value, Mapping):
            return {key: cls._slice_batch(item, indices, batch_size)
                    for key, item in value.items()}
        elif isinstance(value, (list, tuple)) and len(value) == batch_size:
            return type(value)(value[index] for index in indices.tolist())
        return value

    def _guided_forward(self, x_t, timesteps, conditions):
        conditions = dict(conditions)
        if self.joint == 0:
            conditions['force_drop_goal_joint'] = True
        if self.end_effector == 0:
            conditions['force_drop_goal_end_effector'] = True

        # Expanded coefficients skip unused branches and preserve the exact
        # single-call result at (1, 1), (0, 1), and (0, 0).
        branches = (
            (1.0 - self.end_effector, True, True),
            (self.end_effector - self.joint, True, False),
            (self.joint, False, False),
        )
        if self.end_effector == 0:
            branches = ((1.0 - self.joint, True, True),
                        (self.joint, False, True))
        result = None
        for coefficient, drop_joint, drop_ee in branches:
            if coefficient == 0:
                continue
            branch = dict(conditions)
            if drop_joint:
                branch['force_drop_goal_joint'] = True
            if drop_ee:
                branch['force_drop_goal_end_effector'] = True
            prediction = self.model(x_t=x_t, timesteps=timesteps, y=branch)
            weighted = prediction if coefficient == 1 else coefficient * prediction
            result = weighted if result is None else result + weighted
        return result

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
        text_embedding: Optional[torch.Tensor] = None,
        text_valid: Optional[Union[bool, torch.Tensor]] = None,
        goal_valid: Optional[Dict[str, Any]] = None,
        scene_valid: Optional[Union[bool, torch.Tensor]] = None,
        is_recovery: Optional[Union[bool, torch.Tensor]] = None,
        force_drop_text: bool = False,
        force_drop_goal_root: bool = False,
        force_drop_goal_yaw: bool = False,
        force_drop_goal_time: bool = False,
        force_drop_goal_body: bool = False,
        force_drop_goal_orientation: bool = False,
        force_drop_goal_joint: bool = False,
        force_drop_goal_velocity: bool = False,
        force_drop_goal_end_effector: bool = False,
        force_drop_goal_end_effector_left_hand: bool = False,
        force_drop_goal_end_effector_right_hand: bool = False,
        force_drop_goal_end_effector_left_foot: bool = False,
        force_drop_goal_end_effector_right_foot: bool = False,
        force_drop_scene: bool = False,
        force_drop_arrival_time: bool = False,
        time_to_arrival_frame: Optional[torch.Tensor] = None,
        seam_state: Optional[Dict[str, torch.Tensor]] = None,
        goal_guidance: Optional[Dict[str, float]] = None):
    """
    Generate next motion segment using DAR model.

    Controller integrations should pass ``*_valid`` values below. The
    ``force_drop_*`` arguments remain for legacy/offline callers that
    explicitly request inference-time ablations.

    Args:
        vae: VAE model for encoding/decoding
        denoiser: Denoiser model for diffusion
        diffusion: Diffusion model
        val_data: Dataset for motion reconstruction
        goal: Ego-centric goal tensor [B, denoiser.goal_dim].
        voxel: Scene occupancy tensor [B, grid_size^3]
        history_motion: History motion tensor [B, T_hist, D]
        abs_pose: Current absolute pose
        seam_state: Optional physical state at the published plan seam;
            when provided, it is prepended to the generated future and is
            used as the future reconstruction anchor.
        future_len: Length of future motion to generate
        use_full_sample: Whether to use full DDPM sampling loop
        force_drop_goal_root: Mask the root-position component at inference.
        force_drop_goal_yaw: Mask the root-yaw component at inference.
        force_drop_goal_time: Mask the remaining-time component at inference.
        force_drop_goal_body: Mask the limb-keypoint component at inference.
        force_drop_goal_orientation: Mask the joint_state orientation block.
        force_drop_goal_joint: Mask the joint_state 29-DOF target block.
        force_drop_goal_velocity: Mask the joint_state root-velocity block.
        force_drop_goal_end_effector: Mask all four end-effector tokens.
        force_drop_goal_end_effector_left_hand: Mask only left-hand token.
        force_drop_goal_end_effector_right_hand: Mask only right-hand token.
        force_drop_goal_end_effector_left_foot: Mask only left-foot token.
        force_drop_goal_end_effector_right_foot: Mask only right-foot token.
        force_drop_scene: Mask the scene occupancy condition at inference.
        force_drop_arrival_time: Mask the arrival-time PE at inference.
        text_valid: Controller-owned text validity passed to the denoiser.
        goal_valid: Controller-owned validity flags for goal components.
        scene_valid: Controller-owned scene occupancy validity.
        is_recovery: Optional per-sample get-up profile selector.  This is
            omitted by default so deployment keeps the locomotion profile.
        time_to_arrival_frame: Optional frame index passed to the arrival PE.
        goal_guidance: Optional joint/end_effector CFG scales (default 1).
            Applied only with valid joints and at least one valid EE. EE
            guidance is evaluated with joints masked. For eligible samples,
            zero drops the modality; controller validity is preserved.

    Returns:
        Tuple of (future_motion_pred, motion_dict, new_abs_pose)
    """
    device = history_motion.device
    with torch.no_grad():
        batch_size = goal.shape[0]
        if text_embedding is not None:
            if text_embedding.ndim == 1:
                text_embedding = text_embedding.unsqueeze(0)
            if text_embedding.shape[0] == 1 and batch_size > 1:
                text_embedding = text_embedding.expand(batch_size, -1)
            elif text_embedding.shape[0] != batch_size:
                raise ValueError(
                    f"text_embedding batch size {text_embedding.shape[0]} "
                    f"does not match goal batch size {batch_size}"
                )
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
        if time_to_arrival_frame is None and goal.shape[-1] == EXTENDED_BODY_GOAL_DIM:
            fps = float(getattr(val_data, 'fps', 50.0))
            time_to_arrival_frame = torch.round(
                goal[:, 8].clamp_min(0.0) * fps
            ).to(dtype=torch.long)
        if (time_to_arrival_frame is None
                and goal.shape[-1]
                in (SPLIT_GOAL_DIM, SPLIT_END_EFFECTOR_GOAL_DIM)):
            fps = float(getattr(val_data, 'fps', 50.0))
            time_to_arrival_frame = torch.round(
                goal[:, SPLIT_TIME_SLICE].reshape(-1).clamp_min(0.0) * fps
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
            'force_drop_goal_end_effector': force_drop_goal_end_effector,
            'force_drop_goal_end_effector_left_hand': (
                force_drop_goal_end_effector_left_hand),
            'force_drop_goal_end_effector_right_hand': (
                force_drop_goal_end_effector_right_hand),
            'force_drop_goal_end_effector_left_foot': (
                force_drop_goal_end_effector_left_foot),
            'force_drop_goal_end_effector_right_foot': (
                force_drop_goal_end_effector_right_foot),
            'force_drop_scene': force_drop_scene,
            'force_drop_arrival_time': force_drop_arrival_time,
        }
        if text_embedding is not None:
            y['text_embedding'] = text_embedding.to(device=device)
        if text_valid is not None:
            y['text_valid'] = text_valid
        if goal_valid is not None:
            y['goal_valid'] = goal_valid
        if scene_valid is not None:
            y['scene_valid'] = scene_valid
        if is_recovery is not None:
            y['is_recovery'] = is_recovery
        if time_to_arrival_frame is not None:
            y['time_to_arrival_frame'] = time_to_arrival_frame
            y['arrival_time_frame'] = time_to_arrival_frame
        if guidance_scale is not None:
            y['scale'] = guidance_scale
        if force_drop_text:
            y['force_drop_text'] = True

        denoiser_for_sampling = denoiser
        if (guidance_scale is not None and text_embedding is not None
                and denoiser_supports_text_guidance(denoiser)):
            denoiser_for_sampling = ClassifierFreeWrapper(denoiser)
        if goal_guidance is not None:
            denoiser_for_sampling = GoalClassifierFreeWrapper(
                denoiser_for_sampling, **goal_guidance)

        # print(diffusion.num_timesteps)
        if use_full_sample:
            if not use_ddim:
                # Use complete DDPM sampling loop
                sample_fn = diffusion.p_sample_loop
                x_start_pred = sample_fn(
                    denoiser_for_sampling,
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
                    denoiser_for_sampling,
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
            x_start_pred = denoiser_for_sampling(
                x_t=x_start_noise,
                timesteps=diffusion._scale_timesteps(t),
                y=y,
            )  # [B, T=1, D]

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

        # Reconstruct only the newly generated future.  The closed-loop path
        # supplies a physical seam and prepends it below; offline callers use
        # the future dictionary directly.
        future_dict = val_data.reconstruct_motion(
            future_motion_pred,
            abs_pose=abs_pose if seam_state is None else seam_state,
            ret_fk=ret_fk if seam_state is None else False,
            ret_fk_full=ret_fk_full)
        if seam_state is None:
            motion_dict = future_dict
        else:
            # The controller reference starts with the seam state.
            motion_dict = {
                "root_trans_offset": torch.cat(
                    (seam_state["root_trans_offset"][:, None],
                     future_dict["root_trans_offset"]), dim=1),
                "root_rot": torch.cat(
                    (seam_state["root_rot"][:, None],
                     future_dict["root_rot"]), dim=1),
                "dof": torch.cat(
                    (seam_state["dof"][:, None], future_dict["dof"]),
                    dim=1),
                "contact_mask": torch.cat(
                    (seam_state["contact_mask"][:, None],
                     future_dict["contact_mask"]), dim=1),
            }

        # Update absolute pose for next primitive.  The next block's history
        # is the tail of this block's generated future, and V6 reconstructs
        # root pose by integrating per-frame deltas from abs_pose.  Anchor at
        # the frame immediately BEFORE the history window: anchoring at the
        # last frame re-integrates the history deltas a second time and makes
        # every autoregressive seam jump by the history window's net
        # displacement.
        pose_idx = -(history_motion.shape[1] + 1) if (
            motion_dtype.FeatureVersion == 6) else -2
        new_abs_pose = motion_dict_to_abs_pose(motion_dict, idx=pose_idx)

        return future_motion_pred, motion_dict, new_abs_pose
