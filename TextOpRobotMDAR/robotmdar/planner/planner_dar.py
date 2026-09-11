"""Headless fixed-period DAR planner for SONIC controller."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate, to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict

from robotmdar.dtype import logger as dtype_logger
from robotmdar.dtype import seed
from robotmdar.dtype.abc import Dataset, Denoiser, Diffusion, SSampler, VAE
import robotmdar.dtype.motion as motion_dtype
from robotmdar.eval.generate_dar import (
    denoiser_supports_text_guidance,
    encode_motion_lib_initial_noise,
    generate_next_motion,
)
from robotmdar.model.clip import encode_text, load_and_freeze_clip
from robotmdar.utils.dof_contract import (
    configure_dof_contract,
    validate_training_contract,
)
from robotmdar.utils.goal import (
    GoalClamp,
    GoalEncoding,
    GoalType,
    SPLIT_END_EFFECTOR_NO_LOG_GOAL_SCHEMA,
    SPLIT_GOAL_NO_LOG_DIM,
    SPLIT_GOAL_NO_LOG_SCHEMA,
    SPLIT_END_EFFECTOR_NO_LOG_GOAL_DIM,
    ROT_MAT_JOINT_STATE_GOAL_DIM,
    validate_goal_config,
    validate_goal_stats,
)
from robotmdar.utils.planner_convert import (
    align_generated_history_pose,
    apply_generated_history_alignment_correction,
    generated_history_at_frame,
    g1_joint_limits_from_mjcf,
    motion_dict_to_g1data,
    residual_reanchor_generated_history,
    state_goal_from_reference,
    state_to_ego_goal,
    state_to_model_input,
    tracked_frame_from_timestamps,
)
from robotmdar.train.manager import DARManager


def _load_models(
        cfg: DictConfig,
        checkpoint_cfg: DictConfig | None = None):
    # This is a checkpoint architecture capability, not a per-state
    # condition.  Derive it from the checkpoint before instantiation so the
    # deployment config cannot accidentally request a mismatched model.
    if checkpoint_cfg is not None:
        with open_dict(cfg):
            cfg.denoiser.text_condition_enabled = (
                _checkpoint_text_condition_enabled(
                    checkpoint_cfg.get("denoiser")))
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


def _parse_generated_history_alignment_mode(value) -> str:
    """Normalize legacy bool config and the generated-history enum."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "spatial" if value else "null"
    mode = str(value).strip().lower()
    if mode in ("", "null", "none", "false"):
        return "null"
    if mode == "true":
        return "spatial"
    if mode not in ("spatial", "residual_reanchor"):
        raise ValueError(
            "generated_history.align_to_g1 must be one of "
            "null, spatial, residual_reanchor, got "
            f"{value!r}")
    return mode


def _encode_text_embedding(clip_model, text, device: str):
    if text is None or str(text).strip() == "":
        return None
    with torch.no_grad():
        text_embedding = encode_text(
            clip_model, [str(text).strip()])
    return text_embedding.to(device=device)


def _cuda_synchronize(device: str) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _load_text_clip_model(
        device: str,
        clip_version: str = "ViT-B/32",
        clip_model_path: str | None = None):
    logger.info("Loading CLIP text encoder for controller text conditions")
    clip_model = load_and_freeze_clip(
        clip_version, device=device, clip_model_path=clip_model_path)
    warmup_embedding = _encode_text_embedding(
        clip_model, "text condition warmup", device)
    del warmup_embedding
    _cuda_synchronize(device)
    logger.info("CLIP text encoder ready")
    return clip_model


def _time_to_arrival_from_state(
    state_msg,
    motion_fps: float,
    device: str | torch.device,
) -> tuple[float, torch.Tensor]:
    goal_timestamp_ns = state_msg.condition.goal.timestamp_ns
    timestamps_ns = state_msg.history_meta.timestamps_ns
    if goal_timestamp_ns is None:
        raise ValueError("Goal requires goal_timestamp_ns for arrival PE")
    if timestamps_ns is None or len(timestamps_ns) == 0:
        raise ValueError("Goal requires controller timestamps_ns for arrival PE")
    time_to_arrival_s = max(
        0.0,
        (int(goal_timestamp_ns) - int(timestamps_ns[-1])) / 1e9,
    )
    time_to_arrival_frame = torch.round(torch.tensor(
        [time_to_arrival_s * float(motion_fps)],
        dtype=torch.float32,
        device=device,
    )).to(dtype=torch.long)
    return time_to_arrival_s, time_to_arrival_frame


def _state_bool_field(state_msg, name: str, default: bool = True) -> bool:
    validity = state_msg.condition.valid
    goal_validity = validity.goal
    validity_fields = {
        "scene_valid": validity.scene,
        "text_valid": validity.text,
        "goal_root_valid": goal_validity.root,
        "goal_yaw_valid": goal_validity.yaw,
        "goal_body_valid": goal_validity.body,
        "goal_orientation_valid": goal_validity.orientation,
        "goal_joint_valid": goal_validity.joint,
        "goal_velocity_valid": goal_validity.velocity,
        "goal_time_valid": goal_validity.time,
        "goal_end_effector_valid": goal_validity.end_effector.valid,
        "goal_end_effector_left_hand_valid": (
            goal_validity.end_effector.left_hand),
        "goal_end_effector_right_hand_valid": (
            goal_validity.end_effector.right_hand),
        "goal_end_effector_left_foot_valid": (
            goal_validity.end_effector.left_foot),
        "goal_end_effector_right_foot_valid": (
            goal_validity.end_effector.right_foot),
    }
    if name not in validity_fields:
        raise KeyError(f"Unknown protocol-11 validity field: {name}")
    return bool(validity_fields[name])


def _controller_goal_validity(state_msg) -> dict[str, object]:
    """Collect controller validity as data for the denoiser."""
    return {
        "root": _state_bool_field(state_msg, "goal_root_valid", True),
        "yaw": _state_bool_field(state_msg, "goal_yaw_valid", True),
        "body": _state_bool_field(state_msg, "goal_body_valid", True),
        "orientation": _state_bool_field(
            state_msg, "goal_orientation_valid", True),
        "joint": _state_bool_field(state_msg, "goal_joint_valid", True),
        "velocity": _state_bool_field(
            state_msg, "goal_velocity_valid", True),
        "time": _state_bool_field(state_msg, "goal_time_valid", True),
        "end_effector": _state_bool_field(
            state_msg, "goal_end_effector_valid", True),
        "end_effector_left_hand": _state_bool_field(
            state_msg, "goal_end_effector_left_hand_valid", True),
        "end_effector_right_hand": _state_bool_field(
            state_msg, "goal_end_effector_right_hand_valid", True),
        "end_effector_left_foot": _state_bool_field(
            state_msg, "goal_end_effector_left_foot_valid", True),
        "end_effector_right_foot": _state_bool_field(
            state_msg, "goal_end_effector_right_foot_valid", True),
    }


def _goal_log_components(ego_goal_raw: torch.Tensor,
                         goal_type: GoalType) -> tuple[float, ...]:
    if (goal_type is GoalType.JOINT_STATE
            and motion_dtype.FeatureVersion == 6
            and ego_goal_raw.shape[-1] == ROT_MAT_JOINT_STATE_GOAL_DIM):
        return (
            float(ego_goal_raw[0, 1]),
            float(ego_goal_raw[0, 2]),
            float(ego_goal_raw[0, 0]),
            float('nan'),
            float(ego_goal_raw[0, 42]),
            float(ego_goal_raw[0, 43]),
            float(ego_goal_raw[0, 45]),
        )
    if goal_type in (GoalType.ROOT, GoalType.BODY_EXT):
        yaw_deg = math.degrees(math.atan2(
            float(ego_goal_raw[0, 4]), float(ego_goal_raw[0, 3])))
    elif goal_type is GoalType.JOINT_STATE:
        yaw_deg = math.degrees(float(ego_goal_raw[0, 7]))
    else:
        yaw_deg = float('nan')
    if goal_type is GoalType.BODY_EXT:
        vel = (
            float(ego_goal_raw[0, 5]),
            float(ego_goal_raw[0, 6]),
            float(ego_goal_raw[0, 7]),
        )
    elif goal_type is GoalType.JOINT_STATE:
        vel = (
            float(ego_goal_raw[0, 37]),
            float(ego_goal_raw[0, 38]),
            float(ego_goal_raw[0, 39]),
        )
    else:
        vel = (float('nan'), float('nan'), float('nan'))
    return (
        float(ego_goal_raw[0, 0]),
        float(ego_goal_raw[0, 1]),
        float(ego_goal_raw[0, 2]),
        yaw_deg,
        *vel,
    )


def _config_has(section, key: str) -> bool:
    if section is None:
        return False
    try:
        return key in section
    except TypeError:
        return False


def _config_get(section, key: str, default=None):
    if section is None:
        return default
    if isinstance(section, Mapping):
        return section.get(key, default)
    try:
        return section.get(key, default)
    except AttributeError:
        return default


def _nested_config_get(section, path: tuple[str, ...], default=None):
    value = section
    for key in path:
        value = _config_get(value, key, None)
        if value is None:
            return default
    return value


def _plain_config_value(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _first_config_value(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _checkpoint_text_condition_enabled(denoiser_cfg) -> bool:
    """Infer the text module for both current and legacy DAR configs."""
    if _config_has(denoiser_cfg, "text_condition_enabled"):
        return bool(_config_get(
            denoiser_cfg, "text_condition_enabled", False))

    # Older cfg.yaml files predate the explicit architecture flag.  Their
    # legacy text dropout key is also the record that the text projection was
    # part of the model, even when the dropout probability is 1.0.
    if _config_has(denoiser_cfg, "cond_text_mask_prob"):
        return _config_get(denoiser_cfg, "cond_text_mask_prob") is not None
    nested = _config_get(denoiser_cfg, "cond_mask_prob")
    if _nested_config_get(nested, ("text",), None) is not None:
        return True
    if _config_has(denoiser_cfg, "cond_mask_prob"):
        return _config_get(denoiser_cfg, "cond_mask_prob") is not None
    return False


def _checkpoint_condition_mask_prob(denoiser_cfg, condition: str):
    """Read a condition dropout probability from nested or legacy config."""
    nested_paths = {
        "text": ("text",),
        "scene": ("scene",),
        "root": ("goal", "position"),
        "yaw": ("goal", "yaw"),
        "time": ("goal", "time"),
        "body": ("goal", "body"),
        "orientation": ("goal", "orientation"),
        "joint": ("goal", "joint"),
        "velocity": ("goal", "velocity"),
    }
    legacy_keys = {
        "text": ("cond_text_mask_prob",),
        "scene": ("cond_scene_mask_prob",),
        "root": ("cond_goal_root_mask_prob", "cond_goal_mask_prob"),
        "yaw": ("cond_goal_yaw_mask_prob",),
        "time": ("cond_goal_time_mask_prob",),
        "body": ("cond_goal_body_mask_prob",),
        "orientation": ("cond_goal_orientation_mask_prob",),
        "joint": ("cond_goal_joint_mask_prob",),
        "velocity": ("cond_goal_velocity_mask_prob",),
    }
    nested = _config_get(denoiser_cfg, "cond_mask_prob")
    if isinstance(nested, Mapping) and (
            _config_get(nested, "locomotion", None) is not None
            or _config_get(nested, "getup", None) is not None):
        nested = _config_get(nested, "locomotion", None)
    value = _nested_config_get(
        nested, nested_paths[condition], None)
    if isinstance(value, Mapping):
        fallback_key = {
            "root": "hor",
            "orientation": "rot6d",
        }.get(condition)
        value = _config_get(value, fallback_key, None)
    if value is None:
        for key in legacy_keys[condition]:
            value = _config_get(denoiser_cfg, key, None)
            if value is not None:
                break
    if value is None and condition == "text":
        # A scalar legacy cond_mask_prob targeted text in the old model.
        scalar = _config_get(denoiser_cfg, "cond_mask_prob", None)
        if scalar is not None and not isinstance(scalar, Mapping):
            value = scalar
    return None if value is None else float(value)


def _checkpoint_end_effector_mask_probs(denoiser_cfg) -> tuple:
    """Read per-end-effector masks from current or legacy config."""
    names = ("left_hand", "right_hand", "left_foot", "right_foot")
    mask_config = _config_get(denoiser_cfg, "cond_mask_prob", None)
    if isinstance(mask_config, Mapping) and (
            _config_get(mask_config, "locomotion", None) is not None
            or _config_get(mask_config, "getup", None) is not None):
        mask_config = _config_get(mask_config, "locomotion", None)
    nested = _nested_config_get(
        mask_config,
        ("goal", "end_effector"),
        None,
    )
    if nested is not None and not isinstance(nested, Mapping):
        return tuple(float(nested) for _ in names)
    legacy_parent = _config_get(
        denoiser_cfg, "cond_goal_end_effector_mask_prob", None)
    values = []
    for name in names:
        value = _config_get(nested, name, None)
        if value is None:
            value = legacy_parent
        values.append(None if value is None else float(value))
    return tuple(values)


def _checkpoint_contract(model_cfg: DictConfig) -> dict[str, object]:
    """Extract the model/data contract recorded in a checkpoint cfg.yaml."""
    data_cfg = _config_get(model_cfg, "data", {})
    denoiser_cfg = _config_get(model_cfg, "denoiser", {})

    feature_version = int(_first_config_value(
        _config_get(data_cfg, "feature_version", None),
        _config_get(model_cfg, "feature_version", None),
        motion_dtype.FeatureVersion,
    ))
    nfeats_value = _first_config_value(
        _config_get(data_cfg, "nfeats", None),
        _config_get(model_cfg, "nfeats", None),
    )
    if nfeats_value is None:
        raise ValueError("Checkpoint cfg.yaml does not define nfeats")
    nfeats = int(nfeats_value)

    dof_value = _first_config_value(
        _config_get(data_cfg, "dof_dim", None),
        _config_get(model_cfg, "dof_dim", None),
    )
    dof_dim = (
        int(dof_value)
        if dof_value is not None
        else motion_dtype.infer_feature_dof_dim(
            nfeats, feature_version=feature_version)
    )

    goal_type_value = _first_config_value(
        _config_get(data_cfg, "goal_type", None),
        _config_get(model_cfg, "goal_type", None),
    )
    if goal_type_value is None:
        raise ValueError("Checkpoint cfg.yaml does not define data.goal_type")
    goal_type = GoalType.parse(goal_type_value)

    goal_dim_value = _first_config_value(
        _config_get(denoiser_cfg, "goal_dim", None),
        _config_get(data_cfg, "goal_dim", None),
    )
    goal_encoding_value = _first_config_value(
        _config_get(data_cfg, "goal_encoding", None),
        _config_get(denoiser_cfg, "goal_encoding", None),
    )
    if goal_encoding_value is None:
        if goal_dim_value is None:
            goal_encoding = (
                GoalEncoding.LEGACY40
                if goal_type is not GoalType.JOINT_STATE
                else None
            )
        else:
            goal_encoding = {
                40: GoalEncoding.LEGACY40,
                55: GoalEncoding.SPLIT,
                67: GoalEncoding.SPLIT_END_EFFECTOR,
            }.get(int(goal_dim_value))
            if goal_encoding is None:
                raise ValueError(
                    "Cannot infer goal_encoding from checkpoint goal_dim="
                    f"{goal_dim_value}")
    else:
        goal_encoding = GoalEncoding.parse(goal_encoding_value)
    if goal_encoding is None:
        raise ValueError("Checkpoint cfg.yaml does not define goal_encoding")

    goal_dim = int(
        goal_dim_value if goal_dim_value is not None
        else goal_encoding.dimension)
    history_shape = _config_get(denoiser_cfg, "history_shape", None)
    history_len_value = _first_config_value(
        _config_get(data_cfg, "history_len", None),
        history_shape[0] if history_shape is not None else None,
    )
    if history_len_value is None:
        raise ValueError("Checkpoint cfg.yaml does not define history_len")
    future_len_value = _config_get(data_cfg, "future_len", None)
    if future_len_value is None:
        raise ValueError("Checkpoint cfg.yaml does not define future_len")

    goal_timestep_mode = str(_first_config_value(
        _config_get(data_cfg, "goal_timestep_mode", None),
        _config_get(data_cfg, "time_to_arrival_mode", None),
        "relative",
    ))
    goal_offset_range = _config_get(data_cfg, "goal_offset_range", None)
    goal_per_primitive = bool(_config_get(
        data_cfg, "goal_per_primitive", True))
    goal_include_log_d_hor = bool(_config_get(
        data_cfg, "goal_include_log_d_hor", True))
    legacy_split_goal_layout = (
        goal_encoding is GoalEncoding.SPLIT
        and goal_dim == 55
        and not _config_has(data_cfg, "goal_include_log_d_hor")
    )
    no_log_split_layout = goal_dim in (
        SPLIT_GOAL_NO_LOG_DIM, SPLIT_END_EFFECTOR_NO_LOG_GOAL_DIM)
    goal_schema = (
        "rotmat_v7" if legacy_split_goal_layout
        else (
            SPLIT_END_EFFECTOR_NO_LOG_GOAL_SCHEMA
            if goal_dim == SPLIT_END_EFFECTOR_NO_LOG_GOAL_DIM
            else SPLIT_GOAL_NO_LOG_SCHEMA
            if goal_dim == SPLIT_GOAL_NO_LOG_DIM
            else (
                "rotmat_v10_hor_vert_joint_ee"
                if goal_encoding is GoalEncoding.SPLIT_END_EFFECTOR
                else "rotmat_v7_hor_vert"
            )
        )
    )

    data_goal_encoding = _config_get(data_cfg, "goal_encoding", None)
    denoiser_goal_encoding = _config_get(
        denoiser_cfg, "goal_encoding", None)
    if (data_goal_encoding is not None
            and denoiser_goal_encoding is not None
            and GoalEncoding.parse(data_goal_encoding)
            is not GoalEncoding.parse(denoiser_goal_encoding)):
        raise ValueError(
            "Checkpoint data.goal_encoding and denoiser.goal_encoding "
            "disagree")

    return {
        "feature_version": feature_version,
        "nfeats": nfeats,
        "dof_dim": dof_dim,
        "goal_type": goal_type,
        "goal_dim": goal_dim,
        "goal_encoding": goal_encoding,
        "history_len": int(history_len_value),
        "future_len": int(future_len_value),
        "goal_timestep_mode": goal_timestep_mode,
        "goal_offset_range": goal_offset_range,
        "goal_per_primitive": goal_per_primitive,
        "goal_include_log_d_hor": goal_include_log_d_hor,
        "legacy_split_goal_layout": legacy_split_goal_layout,
        "no_log_split_layout": no_log_split_layout,
        "goal_schema": goal_schema,
    }


def _validate_checkpoint_contract(
        model_cfg: DictConfig,
        model_cfg_path: Path) -> dict[str, object]:
    """Validate internal consistency without comparing to deployment cfg."""
    contract = _checkpoint_contract(model_cfg)
    feature_version = int(contract["feature_version"])
    if feature_version not in (3, 6):
        raise ValueError(
            f"DAR checkpoint {model_cfg_path} uses unsupported "
            f"FeatureVersion {feature_version}; planner_dar requires 3 or 6")

    expected_nfeats = motion_dtype.motion_feature_dim_for_dof(
        int(contract["dof_dim"]), feature_version=feature_version)
    if int(contract["nfeats"]) != expected_nfeats:
        raise ValueError(
            f"Incompatible DAR checkpoint config {model_cfg_path}: "
            f"dof_dim={contract['dof_dim']} and FeatureVersion="
            f"{feature_version} require nfeats={expected_nfeats}, got "
            f"{contract['nfeats']}")

    denoiser_cfg = _config_get(model_cfg, "denoiser", {})
    history_shape = _config_get(denoiser_cfg, "history_shape", None)
    if history_shape is not None:
        expected_history_shape = (
            int(contract["history_len"]), int(contract["nfeats"]))
        actual_history_shape = tuple(int(dim) for dim in history_shape)
        if actual_history_shape != expected_history_shape:
            raise ValueError(
                f"Incompatible DAR checkpoint config {model_cfg_path}: "
                f"denoiser.history_shape={actual_history_shape}, expected "
                f"{expected_history_shape}")

    vae_cfg = _config_get(model_cfg, "vae", {})
    vae_nfeats = _config_get(vae_cfg, "nfeats", None)
    if vae_nfeats is not None and int(vae_nfeats) != int(contract["nfeats"]):
        raise ValueError(
            f"Incompatible DAR checkpoint config {model_cfg_path}: "
            f"vae.nfeats={vae_nfeats}, data.nfeats={contract['nfeats']}")

    data_cfg = _config_get(model_cfg, "data", {})
    data_skeleton = _config_get(data_cfg, "skeleton", None)
    skeleton_cfg = _first_config_value(
        data_skeleton, _config_get(model_cfg, "skeleton", None))
    humanoid_type = _config_get(skeleton_cfg, "humanoid_type", None)
    expected_humanoid_type = {
        23: "g1_23dof_lock_wrist",
        29: "g1_29dof",
    }.get(int(contract["dof_dim"]))
    if (humanoid_type is not None and expected_humanoid_type is not None
            and str(humanoid_type) != expected_humanoid_type):
        raise ValueError(
            f"Incompatible DAR checkpoint config {model_cfg_path}: "
            f"humanoid_type={humanoid_type}, expected "
            f"{expected_humanoid_type} for {contract['dof_dim']} DoFs")

    validate_goal_config(
        contract["goal_type"],
        int(contract["goal_dim"]),
        contract["goal_encoding"],
        dof_dim=int(contract["dof_dim"]),
        goal_offset_range=contract["goal_offset_range"],
        goal_timestep_mode=contract["goal_timestep_mode"],
        goal_include_log_d_hor=contract["goal_include_log_d_hor"],
        goal_schema=contract["goal_schema"],
    )
    return contract


def _apply_checkpoint_config(
        cfg: DictConfig,
        model_cfg: DictConfig,
        contract: dict[str, object]) -> None:
    """Apply model-owned settings while retaining deployment-owned paths."""
    data_cfg = _config_get(model_cfg, "data", {})
    denoiser_cfg = _config_get(model_cfg, "denoiser", None)
    if denoiser_cfg is None:
        raise ValueError("Checkpoint cfg.yaml does not define denoiser")

    checkpoint_data_fields = (
        "feature_version",
        "dof_dim",
        "nfeats",
        "normalization_path",
        "std_floor",
        "history_len",
        "future_len",
        "num_primitive",
        "weighted_sample",
        "frame_weight",
        "use_weighted_meanstd",
        "clip_version",
        "clip_dim",
        "goal_offset",
        "goal_offset_range",
        "goal_type",
        "goal_encoding",
        "goal_per_primitive",
        "goal_include_log_d_hor",
        "occupancy_unit",
        "use_scene_surface",
        "augmentation_enabled",
        "augmentation_start_step",
        "augmentation_prob",
        "scene_start_step",
    )
    with open_dict(cfg):
        cfg.feature_version = int(contract["feature_version"])
        cfg.nfeats = int(contract["nfeats"])
        for key in checkpoint_data_fields:
            if _config_has(data_cfg, key):
                cfg.data[key] = _plain_config_value(
                    _config_get(data_cfg, key))

        # These values are required by the planner even when an older
        # cfg.yaml omitted one of the newer aliases.
        cfg.data.feature_version = int(contract["feature_version"])
        cfg.data.dof_dim = int(contract["dof_dim"])
        cfg.data.nfeats = int(contract["nfeats"])
        cfg.data.history_len = int(contract["history_len"])
        cfg.data.future_len = int(contract["future_len"])
        cfg.data.goal_type = contract["goal_type"].value
        cfg.data.goal_encoding = contract["goal_encoding"].value
        cfg.data.goal_per_primitive = bool(contract["goal_per_primitive"])
        cfg.data.goal_timestep_mode = contract["goal_timestep_mode"]
        cfg.data.goal_include_log_d_hor = bool(
            contract["goal_include_log_d_hor"])

        denoiser_plain = _plain_config_value(denoiser_cfg)
        denoiser_plain.setdefault(
            "goal_encoding", contract["goal_encoding"].value)
        denoiser_plain.setdefault("goal_dim", int(contract["goal_dim"]))
        denoiser_plain.setdefault(
            "history_shape",
            [int(contract["history_len"]), int(contract["nfeats"])])
        denoiser_plain["text_condition_enabled"] = (
            _checkpoint_text_condition_enabled(denoiser_cfg))
        denoiser_plain["legacy_split_goal_layout"] = bool(
            contract["legacy_split_goal_layout"])
        cfg.denoiser = OmegaConf.create(denoiser_plain)

        vae_cfg = _config_get(model_cfg, "vae", None)
        if vae_cfg is not None:
            cfg.vae = OmegaConf.create(_plain_config_value(vae_cfg))
        diffusion_cfg = _config_get(model_cfg, "diffusion", None)
        if diffusion_cfg is not None:
            cfg.diffusion = OmegaConf.create(
                _plain_config_value(diffusion_cfg))


def _checkpoint_compatibility(model_cfg: DictConfig) -> dict[str, object]:
    """Describe inputs that the checkpoint can actually consume."""
    contract = _checkpoint_contract(model_cfg)
    data_cfg = _config_get(model_cfg, "data", {})
    denoiser_cfg = _config_get(model_cfg, "denoiser", {})

    text_module = _checkpoint_text_condition_enabled(denoiser_cfg)
    text_mask_prob = _checkpoint_condition_mask_prob(
        denoiser_cfg, "text")
    text_input = text_module and (
        text_mask_prob is None or text_mask_prob < 1.0)

    scene_module = _config_get(denoiser_cfg, "grid_size", None) is not None
    scene_mask_prob = _checkpoint_condition_mask_prob(
        denoiser_cfg, "scene")
    scene_data_enabled = _config_get(data_cfg, "load_scene", True) is not False
    scene_input = (
        scene_module
        and scene_data_enabled
        and (scene_mask_prob is None or scene_mask_prob < 1.0)
    )

    goal_encoding = contract["goal_encoding"]
    ee_mask_probs = _checkpoint_end_effector_mask_probs(denoiser_cfg)
    ee_input = (
        goal_encoding.uses_end_effectors
        and any(value is None or float(value) < 1.0
                for value in ee_mask_probs)
    )

    ignored_inputs: list[str] = []
    if not text_module:
        ignored_inputs.append(
            "text (`text`/`text_valid`; text module is absent)")
    elif not text_input:
        ignored_inputs.append(
            "text (`text`/`text_valid`; "
            f"cond_text_mask_prob={text_mask_prob:.3g}; always masked)")
    if not scene_module:
        ignored_inputs.append(
            "scene occupancy (`ego_occ`/`scene_valid`; "
            "scene module is absent)")
    elif not scene_input:
        reasons = []
        if not scene_data_enabled:
            reasons.append("data.load_scene=false")
        if scene_mask_prob is not None and scene_mask_prob >= 1.0:
            reasons.append(
                f"cond_scene_mask_prob={scene_mask_prob:.3g}; always masked")
        ignored_inputs.append(
            "scene occupancy (`ego_occ`/`scene_valid`; "
            + ", ".join(reasons) + ")")
    if not goal_encoding.uses_end_effectors:
        ignored_inputs.append(
            "goal end-effector tokens (`goal_end_effectors_world`; "
            "left_hand, right_hand, left_foot, right_foot)")
    elif not ee_input:
        ignored_inputs.append(
            "goal end-effector tokens (`goal_end_effectors_world`; "
            "all end-effector tokens are masked)")

    active_goal_conditions = {
        GoalType.ROOT: ("root", "yaw"),
        GoalType.BODY: ("root", "yaw", "body"),
        GoalType.BODY_EXT: ("root", "yaw", "body", "velocity", "time"),
        GoalType.JOINT_STATE: (
            "root", "orientation", "joint", "velocity", "time"),
    }[contract["goal_type"]]
    for condition in active_goal_conditions:
        mask_prob = _checkpoint_condition_mask_prob(
            denoiser_cfg, condition)
        if mask_prob is not None and mask_prob >= 1.0:
            ignored_inputs.append(
                f"goal.{condition} "
                f"(mask probability={mask_prob:.3g}; always masked)")

    return {
        **contract,
        "text_module": text_module,
        "text_input": text_input,
        "text_mask_prob": text_mask_prob,
        "scene_module": scene_module,
        "scene_input": scene_input,
        "scene_mask_prob": scene_mask_prob,
        "goal_end_effector_input": ee_input,
        "legacy_split_goal_layout": contract["legacy_split_goal_layout"],
        "ignored_inputs": tuple(ignored_inputs),
    }


def _warn_checkpoint_compatibility(
        model_cfg_path: Path,
        compatibility: dict[str, object]) -> None:
    logger.info(
        "DAR checkpoint contract from {}: FeatureVersion={}, {}-DoF/{}-D, "
        "goal_type={}, goal_encoding={}, goal_dim={}, history={}, future={}",
        model_cfg_path,
        compatibility["feature_version"],
        compatibility["dof_dim"],
        compatibility["nfeats"],
        compatibility["goal_type"].value,
        compatibility["goal_encoding"].value,
        compatibility["goal_dim"],
        compatibility["history_len"],
        compatibility["future_len"],
    )
    ignored_inputs = compatibility["ignored_inputs"]
    if ignored_inputs:
        logger.warning(
            "DAR checkpoint {} ignores controller input(s): {}. "
            "The controller may still send these values; they will be "
            "accepted and ignored.",
            model_cfg_path,
            "; ".join(str(value) for value in ignored_inputs),
        )


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
    if not model_cfg_path.is_file():
        raise FileNotFoundError(
            f"Cannot find DAR checkpoint config: {model_cfg_path}")
    model_cfg = OmegaConf.load(model_cfg_path)
    _validate_checkpoint_contract(model_cfg, model_cfg_path)
    return model_cfg_path, model_cfg


def _checkpoint_goal_stats_path(
        cfg: DictConfig,
        model_cfg_path: Path | None = None) -> Path:
    if model_cfg_path is not None:
        return model_cfg_path.with_name("goal_stats.pkl")
    ckpt_dar = cfg.ckpt.get("dar")
    if ckpt_dar is None:
        raise ValueError("ckpt.dar must be set to load frozen goal stats")
    return Path(to_absolute_path(str(ckpt_dar))).with_name("goal_stats.pkl")


def _goal_log_slices(goal_encoding: GoalEncoding) -> dict[str, object]:
    if goal_encoding is GoalEncoding.LEGACY40:
        return {
            "yaw": 7,
            "velocity": slice(37, 40),
        }
    return {
        "yaw": 12,
        "velocity": slice(42, 45),
    }


def main(cfg: DictConfig) -> None:
    """Run the TextOp planner until interrupted."""
    from sonicmsg import PlannerNode
    from sonicmsg.messages import unpack_occ

    # The checkpoint cfg is the source of truth for model-facing settings.
    # Keep controller paths and runtime knobs from the deployment config.
    _model_cfg_path, _model_cfg = _checkpoint_config(cfg)
    _apply_checkpoint_config(
        cfg, _model_cfg, _checkpoint_contract(_model_cfg))
    checkpoint_compatibility = _checkpoint_compatibility(_model_cfg)

    configure_dof_contract(cfg)
    dtype_logger.set(cfg)
    seed.set(cfg.seed)
    goal_encoding = GoalEncoding.parse(
        cfg.data.get("goal_encoding", GoalEncoding.LEGACY40)
    )
    denoiser_goal_encoding = GoalEncoding.parse(
        cfg.denoiser.get("goal_encoding", goal_encoding)
    )
    if denoiser_goal_encoding is not goal_encoding:
        raise ValueError(
            f"data.goal_encoding={goal_encoding.value!r} must match "
            f"denoiser.goal_encoding={denoiser_goal_encoding.value!r}"
        )
    goal_type = GoalType.parse(cfg.data.goal_type)
    goal_reference_path = cfg.get("goal_reference_path")
    if goal_reference_path is not None:
        goal_reference_path = to_absolute_path(str(goal_reference_path))
    if goal_type.uses_keypoints and goal_reference_path is None:
        logger.info(
            "{} planner expects goal_keypoints_world from the controller; "
            "no reference pose is configured", goal_type.value)

    if motion_dtype.FeatureVersion not in (3, 6):
        raise ValueError(
            "planner_dar requires FeatureVersion 3 or 6, got "
            f"{motion_dtype.FeatureVersion}")

    _warn_checkpoint_compatibility(
        _model_cfg_path, checkpoint_compatibility)
    logger.info(
        "Loading {}-DoF/{}-D DAR model and dataset statistics",
        cfg.data.dof_dim, cfg.data.nfeats)
    vae, denoiser, diffusion, val_data = _load_models(cfg, _model_cfg)
    text_condition_supported = (
        bool(checkpoint_compatibility["text_input"])
        and denoiser_supports_text_guidance(denoiser)
    )
    if (bool(checkpoint_compatibility["text_input"])
            and not denoiser_supports_text_guidance(denoiser)):
        logger.warning(
            "DAR checkpoint {} has no usable text-condition weights; "
            "controller text will be ignored",
            _model_cfg_path)
    text_clip_model = None
    text_embedding_cache: dict[str, torch.Tensor] = {}
    if text_condition_supported:
        text_clip_model = _load_text_clip_model(
            str(cfg.device),
            clip_version=str(cfg.data.get("clip_version", "ViT-B/32")),
            clip_model_path=cfg.data.get("clip_model_path"),
        )
        logger.info(
            "Controller text conditions enabled; text_valid=false "
            "will be passed as the text condition validity mask")
    else:
        logger.info(
            "Loaded DAR checkpoint does not support text conditioning; "
            "controller text will be ignored")
    goal_include_log_d_hor = bool(
        cfg.data.get('goal_include_log_d_hor', True))
    goal_stats = None
    if goal_encoding is not GoalEncoding.LEGACY40:
        goal_stats_path = _checkpoint_goal_stats_path(
            cfg, model_cfg_path=_model_cfg_path)
        if not goal_stats_path.exists():
            raise FileNotFoundError(
                f"Missing frozen goal stats at {goal_stats_path}")
        goal_stats = torch.load(goal_stats_path, map_location="cpu")
        goal_stats = validate_goal_stats(
            goal_stats,
            goal_encoding=goal_encoding,
            goal_offset_range=cfg.data.goal_offset_range,
            goal_per_primitive=cfg.data.goal_per_primitive,
            future_len=cfg.data.future_len,
            fps=float(val_data.fps),
            goal_timestep_mode=cfg.data.goal_timestep_mode,
            datadir=str(val_data.datadir),
            goal_include_log_d_hor=goal_include_log_d_hor,
            goal_schema=checkpoint_compatibility["goal_schema"],
        )
    goal_type = validate_goal_config(
        cfg.data.goal_type,
        cfg.denoiser.goal_dim,
        goal_encoding,
        dof_dim=cfg.data.dof_dim,
        goal_offset_range=cfg.data.goal_offset_range,
        goal_timestep_mode=cfg.data.goal_timestep_mode,
        goal_stats=goal_stats,
        goal_include_log_d_hor=goal_include_log_d_hor,
        goal_schema=checkpoint_compatibility["goal_schema"],
    )
    validate_training_contract(
        cfg, [("planner", val_data)], vae, denoiser)
    history_len = int(cfg.data.history_len)
    future_len = int(cfg.data.future_len)
    motion_fps = float(cfg.motion_fps)
    if abs(float(val_data.fps) - motion_fps) > 1e-6:
        raise ValueError(
            f"Dataset fps ({val_data.fps}) must match motion_fps ({motion_fps})")

    # Clamp the controller goal back into the training distribution before
    # encoding: r_max(T) = clip(speed_max * clip(T, 1/fps, time_max),
    # r_min, r_max), with x,y scaled proportionally (direction preserved).
    # The same time cap feeds the arrival-time PE and the goal urgency.
    # The envelope is DERIVED from the frozen goal_stats.pkl shipped with the
    # checkpoint (per-window ego goal distance and implied speed r/T
    # quantiles): goal_clamp.quantile selects the training-distribution
    # percentile to admit — lower is stricter. r_min = speed_max/fps and
    # time_max = future_len/fps are derived as well. goal_clamp_enabled is
    # the runtime switch; without it the goal_clamp block (or a null
    # override) leaves the planner unclamped.
    goal_clamp = None
    _goal_clamp_cfg = cfg.get("goal_clamp")
    if _goal_clamp_cfg is not None and bool(
            cfg.get("goal_clamp_enabled", False)):
        if goal_stats is None:
            raise ValueError(
                "goal_clamp_enabled requires frozen goal statistics; it is "
                "only supported for the split goal encoding")
        _clamp_quantile = float(_goal_clamp_cfg.get("quantile", 99.0))
        goal_clamp = GoalClamp.from_stats(goal_stats, _clamp_quantile)
        logger.info(
            "Goal clamp enabled from frozen stats at percentile {:.1f}: "
            "r_max(T)=clip({:.3f}*T, {:.3f}, {:.3f}) m with T<= {:.2f} s",
            _clamp_quantile, goal_clamp.speed_max, goal_clamp.r_min,
            goal_clamp.r_max, goal_clamp.time_max)
    elif _goal_clamp_cfg is not None:
        logger.info("Goal clamp disabled (goal_clamp_enabled=false)")

    period = float(cfg.infer_period_ms) / 1000.0
    if period <= 0:
        raise ValueError(f"infer_period_ms must be positive, got {cfg.infer_period_ms}")
    state_timeout_ms = int(cfg.state_timeout_ms)
    log_every = max(1, int(cfg.log_every))
    comm_config = to_absolute_path(str(cfg.comm_config))
    node = PlannerNode(comm_config)

    scene_condition_supported = bool(
        checkpoint_compatibility["scene_input"])
    grid_size = int(getattr(
        denoiser, "grid_size", cfg.denoiser.get("grid_size", 25)))
    n_voxels = grid_size**3
    latest_state = None
    last_inferred_seq = None
    next_infer_time = time.perf_counter()
    inference_count = 0
    generated_history_cfg = cfg.get("generated_history", {})
    use_generated_history = bool(generated_history_cfg.get(
        "enabled", cfg.get("use_generated_history", False)))
    generated_history_align_mode = _parse_generated_history_alignment_mode(
        generated_history_cfg.get(
            "align_to_g1", cfg.get("align_generated_history_to_g1", False)))
    align_generated_history = generated_history_align_mode != "null"
    phase_lag_offset = int(generated_history_cfg.get(
        "phase_lag_offset", cfg.get("phase_lag_offset", 0)))
    if phase_lag_offset < 0:
        raise ValueError(
            f"phase_lag_offset must be non-negative, got {phase_lag_offset}")
    align_generated_history_every = int(generated_history_cfg.get(
        "align_every", cfg.get("align_generated_history_every", 1)))
    if align_generated_history_every <= 0:
        raise ValueError(
            "align_generated_history_every must be positive, got "
            f"{align_generated_history_every}")
    joint_smoothing_cfg = cfg.get("history_joint_smoothing")
    joint_smoothing_enabled = (
        joint_smoothing_cfg is not None
        and bool(joint_smoothing_cfg.get("enabled", False))
    )
    joint_smoothing_max_velocity = None
    joint_smoothing_ema_alpha = 1.0
    if joint_smoothing_enabled:
        joint_smoothing_max_velocity = joint_smoothing_cfg.get(
            "max_velocity_rad_s")
        joint_smoothing_ema_alpha = float(
            joint_smoothing_cfg.get("ema_alpha", 1.0))
        if (not math.isfinite(joint_smoothing_ema_alpha)
                or not 0.0 < joint_smoothing_ema_alpha <= 1.0):
            raise ValueError(
                "history_joint_smoothing.ema_alpha must be in (0, 1], "
                f"got {joint_smoothing_ema_alpha}")
        if (joint_smoothing_max_velocity is None
                and joint_smoothing_ema_alpha >= 1.0):
            raise ValueError(
                "history_joint_smoothing.enabled=true requires "
                "max_velocity_rad_s or ema_alpha < 1.0")
        max_velocity_log = joint_smoothing_max_velocity
        if OmegaConf.is_config(max_velocity_log):
            max_velocity_log = OmegaConf.to_container(
                max_velocity_log, resolve=True)
        logger.info(
            "Controller joint-history smoothing enabled: "
            "max_velocity={} rad/s, ema_alpha={:.3f}",
            max_velocity_log, joint_smoothing_ema_alpha)
    generated_plans = {}
    generated_history_replan_count = 0
    alignment_epoch = 0
    last_alignment_epoch = None
    last_alignment_source_epoch = None
    last_alignment_correction = None
    last_residual_reanchor_correction = None
    residual_joint_limits = None
    next_ack_log_time = 0.0
    infer_times: list[float] = []  # rolling window for running average
    fixed_sampling_noise = None
    if not bool(cfg.get("resample_noise_each_plan", False)):
        fixed_sampling_noise = torch.randn(
            (1, *denoiser.noise_shape), device=cfg.device)

    motion_path = cfg.get("motion_path")
    if motion_path is not None:
        motion_path = to_absolute_path(str(motion_path))
        fixed_sampling_noise = encode_motion_lib_initial_noise(
            vae=vae,
            val_data=val_data,
            motion_path=str(motion_path),
            start_frame=int(cfg.get("motion_start_frame", 0)),
            history_len=history_len,
            future_len=future_len,
            device=str(cfg.device),
            clip_name=cfg.get("motion_clip"),
        )
        logger.info("Seeded diffusion latent from {} shape={}",
                    motion_path, tuple(fixed_sampling_noise.shape))

    logger.info(
        "TextOp planner ready: replan={:.1f} Hz, motion={:.1f} Hz, "
        "history={} features/{} states, future={} frames, history_source={}, "
        "phase_lag={} frames, align_mode={}, align_every={} replans",
        1.0 / period, motion_fps, history_len, history_len, future_len,
        (f"generated+{generated_history_align_mode}"
         if align_generated_history else "generated")
        if use_generated_history else "controller",
        (phase_lag_offset
         if generated_history_align_mode == "spatial" else 0),
        generated_history_align_mode,
        align_generated_history_every if align_generated_history else 0)

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

            state_seq = int(latest_state.history_meta.seq)
            if state_seq == last_inferred_seq:
                time.sleep(0.001)
                continue
            tracked_plan = None
            if use_generated_history and generated_plans:
                tracked_plan = generated_plans.get(
                    latest_state.tracking.seq)
                if tracked_plan is None:
                    # Do not advance the autoregressive chain until the
                    # controller acknowledges applying one of our plans.
                    if now >= next_ack_log_time:
                        logger.info(
                            "Waiting for controller plan acknowledgment: "
                            "active={} cached={}",
                            latest_state.tracking.seq,
                            list(generated_plans))
                        next_ack_log_time = now + 1.0
                    time.sleep(0.001)
                    continue
            last_inferred_seq = state_seq
            scheduled_next = next_infer_time + period

            try:
                state_goal_type = GoalType.parse(
                    latest_state.condition.goal.goal_type)
                if state_goal_type is not goal_type:
                    raise ValueError(
                        f"Planner is configured for goal_type={goal_type.value!r}, "
                        f"but controller sent {state_goal_type.value!r}")
                goal_valid = _controller_goal_validity(latest_state)
                scene_valid = (
                    _state_bool_field(latest_state, "scene_valid", True)
                    if scene_condition_supported else False
                )
                _goal_yaw_valid = bool(goal_valid["yaw"])
                _goal_orientation_valid = bool(goal_valid["orientation"])
                _goal_time_valid = bool(goal_valid["time"])
                controller_text_valid = _state_bool_field(
                    latest_state, "text_valid", False)
                text_valid = controller_text_valid and text_condition_supported
                text = latest_state.condition.text
                text_embedding = None
                if text_valid:
                    if text is None or not str(text).strip():
                        raise ValueError(
                            "State marks text_valid=true but has no "
                            "text")
                    text = str(text).strip()
                    if text_condition_supported:
                        if text_clip_model is None:
                            raise RuntimeError(
                                "Text conditioning is supported but the CLIP "
                                "text encoder was not initialized")
                        text_embedding = text_embedding_cache.get(
                            text)
                        if text_embedding is None:
                            text_embedding = _encode_text_embedding(
                                text_clip_model, text, str(cfg.device))
                            text_embedding_cache[text] = text_embedding
                            logger.info(
                                "Controller text enabled: {!r}",
                                text)
                using_generated_history = (
                    use_generated_history and tracked_plan is not None)
                current_alignment_epoch = None
                alignment_mode = "none"
                phase_offset_frames = None
                if using_generated_history:
                    tracked_frame = tracked_frame_from_timestamps(
                        latest_state, motion_fps, future_len)
                    source_alignment_epoch = tracked_plan.get(
                        "alignment_epoch")
                    align_this_replan = (
                        align_generated_history
                        and generated_history_replan_count
                        % align_generated_history_every == 0
                    )
                    (history_motion, generated_abs_pose,
                     generated_reference_pos, generated_reference_rot) = (
                        generated_history_at_frame(
                            tracked_plan, tracked_frame, history_len,
                            phase_lag_offset=(
                                phase_lag_offset
                                if generated_history_align_mode == "spatial"
                                else 0)))
                    if align_generated_history:
                        generated_history_replan_count += 1
                        if generated_history_align_mode == "residual_reanchor":
                            if align_this_replan:
                                if residual_joint_limits is None:
                                    residual_joint_limits = (
                                        g1_joint_limits_from_mjcf(
                                            val_data, cfg.device))
                                (
                                    abs_pose, history_motion,
                                    residual_correction, residual_phase,
                                    residual_phase_error,
                                ) = residual_reanchor_generated_history(
                                    generated_abs_pose,
                                    history_motion,
                                    latest_state,
                                    val_data,
                                    cfg.device,
                                    joint_limits=residual_joint_limits)
                                alignment_epoch += 1
                                last_alignment_epoch = alignment_epoch
                                current_alignment_epoch = alignment_epoch
                                alignment_mode = (
                                    "residual_reanchor:refreshed")
                                last_alignment_source_epoch = (
                                    source_alignment_epoch)
                                last_residual_reanchor_correction = (
                                    residual_correction)
                                phase_offset_frames = int(
                                    residual_correction[
                                        "phase_offset_frames"][0])
                                logger.info(
                                    "Residual re-anchor phase={} "
                                    "offset={:+d} frames error={:.5f}",
                                    int(residual_phase[0]),
                                    phase_offset_frames,
                                    float(residual_phase_error[0]))
                            elif (
                                    last_residual_reanchor_correction
                                    is not None
                                    and source_alignment_epoch
                                    == last_alignment_source_epoch
                                    and source_alignment_epoch
                                    != last_alignment_epoch):
                                (
                                    abs_pose, history_motion,
                                    _, inherited_phase,
                                    inherited_phase_error,
                                ) = residual_reanchor_generated_history(
                                    generated_abs_pose,
                                    history_motion,
                                    latest_state,
                                    val_data,
                                    cfg.device,
                                    joint_limits=residual_joint_limits,
                                    correction=(
                                        last_residual_reanchor_correction))
                                current_alignment_epoch = last_alignment_epoch
                                alignment_mode = (
                                    "residual_reanchor:inherited")
                                phase_offset_frames = int(
                                    last_residual_reanchor_correction[
                                        "phase_offset_frames"][0])
                                logger.info(
                                    "Inherited residual re-anchor phase={} "
                                    "offset={:+d} frames error={:.5f}",
                                    int(inherited_phase[0]),
                                    phase_offset_frames,
                                    float(inherited_phase_error[0]))
                            else:
                                abs_pose = {
                                    k: v.to(cfg.device)
                                    for k, v in generated_abs_pose.items()
                                }
                                current_alignment_epoch = (
                                    source_alignment_epoch)
                                alignment_mode = (
                                    "residual_reanchor:inherited"
                                    if (current_alignment_epoch is not None
                                        and current_alignment_epoch
                                        == last_alignment_epoch)
                                    else "residual_reanchor:none")
                                if (alignment_mode.endswith(":inherited")
                                        and last_residual_reanchor_correction
                                        is not None):
                                    phase_offset_frames = int(
                                        last_residual_reanchor_correction[
                                            "phase_offset_frames"][0])
                            history_translation = None
                        elif align_this_replan:
                            (abs_pose, _goal_reference_pos,
                             _goal_reference_rot, history_translation,
                             history_motion, alignment_correction) = (
                                align_generated_history_pose(
                                    generated_abs_pose,
                                    generated_reference_pos,
                                    generated_reference_rot,
                                    latest_state,
                                    cfg.device,
                                    history_motion=history_motion,
                                    val_data=val_data,
                                    return_correction=True))
                            alignment_epoch += 1
                            last_alignment_epoch = alignment_epoch
                            current_alignment_epoch = alignment_epoch
                            alignment_mode = "refreshed"
                            last_alignment_source_epoch = (
                                source_alignment_epoch)
                            last_alignment_correction = alignment_correction
                        elif (
                                last_alignment_correction is not None
                                and source_alignment_epoch
                                == last_alignment_source_epoch
                                and source_alignment_epoch
                                != last_alignment_epoch):
                            # The controller may still be tracking the source
                            # plan from which the last refresh was computed.
                            # Carry that fixed correction into this new window
                            # without re-anchoring it to the current state.
                            (abs_pose, history_motion) = (
                                apply_generated_history_alignment_correction(
                                    generated_abs_pose,
                                    last_alignment_correction,
                                    history_motion=history_motion,
                                    val_data=val_data))
                            history_translation = last_alignment_correction[
                                "pose_translation"].to(cfg.device)
                            current_alignment_epoch = last_alignment_epoch
                            alignment_mode = "inherited"
                        else:
                            # The source plan already carries the current
                            # correction. Keep its coordinate frame and do
                            # not apply the same correction a second time.
                            abs_pose = {
                                k: v.to(cfg.device)
                                for k, v in generated_abs_pose.items()
                            }
                            history_translation = None
                            current_alignment_epoch = source_alignment_epoch
                            alignment_mode = (
                                "inherited"
                                if (current_alignment_epoch is not None
                                    and current_alignment_epoch
                                    == last_alignment_epoch)
                                else "none"
                            )

                        # The goal must follow the measured robot even on
                        # replans where history alignment is intentionally
                        # skipped.
                        ego_goal_raw = state_to_ego_goal(
                            latest_state, cfg.device,
                            goal_type=goal_type,
                            goal_reference_path=goal_reference_path,
                            goal_encoding=GoalEncoding.LEGACY40,
                            goal_clamp=goal_clamp, fps=motion_fps,
                            val_data=val_data)
                        ego_goal = (
                            ego_goal_raw
                            if goal_encoding is GoalEncoding.LEGACY40
                            else state_to_ego_goal(
                                latest_state, cfg.device,
                                goal_type=goal_type,
                                goal_reference_path=goal_reference_path,
                                goal_encoding=goal_encoding,
                                goal_stats=goal_stats,
                                goal_clamp=goal_clamp, fps=motion_fps,
                                val_data=val_data,
                                goal_include_log_d_hor=(
                                    goal_include_log_d_hor))
                        )
                    else:
                        align_this_replan = False
                        abs_pose = {
                            k: v.to(cfg.device)
                            for k, v in generated_abs_pose.items()
                        }
                        history_translation = None
                        current_alignment_epoch = source_alignment_epoch
                        alignment_mode = (
                            "inherited"
                            if current_alignment_epoch is not None
                            else "none"
                        )
                        ego_goal_raw = state_goal_from_reference(
                            latest_state, generated_reference_pos,
                            generated_reference_rot, cfg.device,
                            goal_type=goal_type,
                            goal_reference_path=goal_reference_path,
                            goal_encoding=GoalEncoding.LEGACY40,
                            goal_clamp=goal_clamp, fps=motion_fps,
                            val_data=val_data)
                        ego_goal = (
                            ego_goal_raw
                            if goal_encoding is GoalEncoding.LEGACY40
                            else state_goal_from_reference(
                                latest_state, generated_reference_pos,
                                generated_reference_rot, cfg.device,
                                goal_type=goal_type,
                                goal_reference_path=goal_reference_path,
                                goal_encoding=goal_encoding,
                                goal_stats=goal_stats,
                                goal_clamp=goal_clamp, fps=motion_fps,
                                val_data=val_data,
                                goal_include_log_d_hor=(
                                    goal_include_log_d_hor))
                        )
                else:
                    tracked_frame = None
                    align_this_replan = False
                    if latest_state.history_meta.n_states < history_len:
                        raise ValueError(
                            f"State {state_seq} has "
                            f"{latest_state.history_meta.n_states} "
                            f"entries; need at least {history_len}")
                    history_motion, abs_pose = state_to_model_input(
                        latest_state, history_len, val_data, cfg.device,
                        fps=motion_fps,
                        joint_smoothing_max_velocity_rad_s=(
                            joint_smoothing_max_velocity),
                        joint_smoothing_ema_alpha=joint_smoothing_ema_alpha)
                    ego_goal_raw = state_to_ego_goal(
                        latest_state, cfg.device, goal_type=goal_type,
                        goal_reference_path=goal_reference_path,
                        goal_encoding=GoalEncoding.LEGACY40,
                        goal_clamp=goal_clamp, fps=motion_fps,
                        val_data=val_data)
                    ego_goal = (
                        ego_goal_raw if goal_encoding is GoalEncoding.LEGACY40
                        else state_to_ego_goal(
                            latest_state, cfg.device, goal_type=goal_type,
                            goal_reference_path=goal_reference_path,
                            goal_encoding=goal_encoding,
                            goal_stats=goal_stats,
                            goal_clamp=goal_clamp, fps=motion_fps,
                            val_data=val_data,
                            goal_include_log_d_hor=goal_include_log_d_hor)
                    )
                    history_translation = None
                    alignment_mode = "none"

                # DEBUG: overwrite the ego_goal
                # ego_goal[0, 0] = 0.1
                # ego_goal[0, 1] = 0.0
                # ego_goal[0, 2] = 0.0

                (_goal_ego_x, _goal_ego_y, _goal_delta_z,
                 _goal_ego_yaw_raw_deg,
                 _ego_vel_x, _ego_vel_y, _ego_vel_z) = _goal_log_components(
                    ego_goal_raw, goal_type)

                _world_goal = latest_state.condition.goal.root_pos_world
                if _world_goal is not None:
                    _world_goal = np.asarray(_world_goal, dtype=np.float64).reshape(-1)
                    _goal_world_x = float(_world_goal[0])
                    _goal_world_y = float(_world_goal[1])
                    _goal_world_z = float(_world_goal[2])
                else:
                    _goal_world_x = _goal_world_y = _goal_world_z = float('nan')

                # Radius before/after the goal clamp (world frame: controller
                # goal vs measured root; ego frame: as fed to the model).
                _meas_root = np.asarray(
                    latest_state.states.g1_pos[-1], dtype=np.float64)
                _goal_r_world = math.hypot(
                    _goal_world_x - float(_meas_root[0]),
                    _goal_world_y - float(_meas_root[1]))
                _goal_r_ego = math.hypot(_goal_ego_x, _goal_ego_y)

                _world_vel = (
                    latest_state.condition.goal.root_velocity_world)
                if _world_vel is not None:
                    _world_vel = np.asarray(_world_vel, dtype=np.float64).reshape(-1)
                    _vel_world_x = float(_world_vel[0])
                    _vel_world_y = float(_world_vel[1])
                    _vel_world_z = float(_world_vel[2])
                else:
                    _vel_world_x = _vel_world_y = _vel_world_z = float('nan')

                # ── world (from controller) ──
                _world_yaw = latest_state.condition.goal.yaw_world
                if _world_yaw is not None:
                    _goal_yaw_world_deg = math.degrees(
                        float(np.asarray(_world_yaw, dtype=np.float64).reshape(-1)[0]))
                else:
                    _goal_yaw_world_deg = float('nan')

                # ── ego (as seen by planner, after validity flags) ──
                if goal_type in (GoalType.ROOT, GoalType.BODY_EXT):
                    if not _goal_yaw_valid:
                        _goal_ego_yaw_deg = 0.0
                    else:
                        _goal_ego_yaw_deg = _goal_ego_yaw_raw_deg
                elif goal_type is GoalType.JOINT_STATE:
                    if not (_goal_yaw_valid and _goal_orientation_valid):
                        _goal_ego_yaw_deg = 0.0
                    else:
                        _goal_ego_yaw_deg = _goal_ego_yaw_raw_deg
                else:
                    _goal_ego_yaw_deg = float('nan')

                time_to_arrival_frame = None
                time_to_arrival_s = float('nan')
                if goal_type.uses_arrival_time:
                    if _goal_time_valid:
                        time_to_arrival_s, time_to_arrival_frame = (
                            _time_to_arrival_from_state(
                                latest_state, motion_fps, cfg.device))
                    else:
                        time_to_arrival_s = 0.0
                        time_to_arrival_frame = torch.zeros(
                            1, dtype=torch.long, device=cfg.device)
                    if goal_clamp is not None and _goal_time_valid:
                        # Keep the arrival PE inside the training horizon
                        # (same cap the goal builder applies to urgency).
                        time_to_arrival_frame = time_to_arrival_frame.clamp(
                            max=int(round(goal_clamp.time_max * motion_fps)))

                _cuda_synchronize(str(cfg.device))
                infer_start = time.perf_counter()
                state_ego_occ = latest_state.condition.ego_occ
                if not scene_condition_supported:
                    ego_occ = np.zeros(n_voxels, dtype=np.float32)
                elif state_ego_occ is None:
                    if scene_valid:
                        raise ValueError(
                            f"State {state_seq} does not contain ego occupancy")
                    ego_occ = np.zeros(n_voxels, dtype=np.float32)
                else:
                    ego_occ = unpack_occ(state_ego_occ, n_voxels)
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
                    text_embedding=text_embedding,
                    text_valid=text_valid,
                    goal_valid=goal_valid,
                    scene_valid=scene_valid,
                    is_recovery=bool(cfg.get('is_recovery', False)),
                    initial_noise=fixed_sampling_noise,
                    ret_fk=True,
                    time_to_arrival_frame=time_to_arrival_frame,
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
                elif goal_type is GoalType.BODY_EXT:
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
                        _ego_vel_x, _ego_vel_y, _ego_vel_z,
                        0.0 if not _goal_time_valid else time_to_arrival_s,
                        occ_count, infer_ms, avg_ms)
                else:
                    logger.info(
                        "goal[joint_state]: world(pos=({:.3f},{:.3f},{:.3f}) "
                        "vel=({:.3f},{:.3f},{:.3f})) "
                        "ego(pos=({:.3f},{:.3f},{:.3f}) "
                        "yaw={:.1f} deg "
                        "vel=({:.3f},{:.3f},{:.3f}) dt={:.3f}s) | "
                        "occ={} | infer={:.1f} ms (avg20={:.1f} ms)",
                        _goal_world_x, _goal_world_y, _goal_world_z,
                        _vel_world_x, _vel_world_y, _vel_world_z,
                        _goal_ego_x, _goal_ego_y, _goal_delta_z,
                        _goal_ego_yaw_deg,
                        _ego_vel_x, _ego_vel_y, _ego_vel_z,
                        0.0 if not _goal_time_valid else time_to_arrival_s,
                        occ_count, infer_ms, avg_ms)

                # SonicRunner resets each newly received G1 plan to frame 0.
                # Keep the current measured history frame as that exact seam;
                # dropping all history would jump directly to stochastic t+1.
                skip_history = (
                    0 if bool(cfg.pub_all_frames) else history_len - 1)
                motion = motion_dict_to_g1data(
                    motion_dict, skip_history=skip_history, fps=motion_fps,
                    locked_joint_pos=np.asarray(
                        latest_state.states.g1_joint_pos[-1],
                        dtype=np.float32))
                measured_pos = np.asarray(
                    latest_state.states.g1_pos[-1], dtype=np.float32)
                measured_rot_wxyz = np.asarray(
                    latest_state.states.g1_root_rot[-1], dtype=np.float32)
                if getattr(motion, "root_ori", None) is not None:
                    published_rot_wxyz = motion.root_ori[0]
                else:
                    published_rot_wxyz = motion.body_ori[0, 0]
                quat_dot = float(np.clip(np.abs(np.dot(
                    measured_rot_wxyz, published_rot_wxyz)), 0.0, 1.0))
                published_root_pos = (
                    motion.root_pos[0]
                    if getattr(motion, "root_pos", None) is not None
                    else motion.body_pos[0, 0])
                seam_root_error = float(np.linalg.norm(
                    published_root_pos - measured_pos))
                seam_root_angle_deg = math.degrees(2.0 * math.acos(quat_dot))
                seam_joint_error = float(np.max(np.abs(
                    motion.joint_pos[0]
                    - np.asarray(latest_state.states.g1_joint_pos[-1],
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
                        # The generated motion is now expressed in this
                        # alignment frame. Future replans inherit it by
                        # selecting this plan's history and pose.
                        "alignment_epoch": current_alignment_epoch,
                    }
                    while len(generated_plans) > 16:
                        del generated_plans[next(iter(generated_plans))]

                inference_count += 1
                if inference_count == 1 or inference_count % log_every == 0:
                    logger.info(
                        "plan={} state={} motion={} frames infer={:.1f} ms "
                        "(avg20={:.1f} ms) "
                        "history={} tracked_plan={} frame={} shift={:.3f} m "
                        "align={} phase_offset={} frames "
                        "seam=({:.4f} m, {:.2f} deg, {:.4f} rad) "
                        "goal_r=({:.3f}->{:.3f}) m",
                        published_seq, state_seq, motion.num_frames, infer_ms, avg_ms,
                        "generated" if using_generated_history else "controller",
                        (latest_state.tracking.seq
                         if using_generated_history else -1),
                        tracked_frame if tracked_frame is not None else -1,
                        (float(torch.linalg.vector_norm(history_translation))
                         if history_translation is not None else 0.0),
                        alignment_mode,
                        (f"{phase_offset_frames:+d}"
                         if phase_offset_frames is not None else "n/a"),
                        seam_root_error, seam_root_angle_deg, seam_joint_error,
                        _goal_r_world, _goal_r_ego)
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
