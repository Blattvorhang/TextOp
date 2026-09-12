import os
import math
import random
import queue
import threading
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from hydra.utils import instantiate

from robotmdar.utils.goal import (
    GoalEncoding,
    GoalType,
    JOINT_STATE_GOAL_DIM,
    ROT_MAT_JOINT_STATE_GOAL_DIM,
    SPLIT_END_EFFECTOR_GOAL_DIM,
    SPLIT_END_EFFECTOR_NO_LOG_GOAL_DIM,
    SPLIT_END_EFFECTOR_TOKEN_ORDER,
    SPLIT_NO_LOG_END_EFFECTOR_SLICE,
    SPLIT_NO_LOG_END_EFFECTOR_SUBSLICES,
    SPLIT_NO_LOG_HORIZONTAL_SLICE,
    SPLIT_NO_LOG_HORIZONTAL_URGENCY_SLICE,
    SPLIT_NO_LOG_JOINT_SLICE,
    SPLIT_NO_LOG_ORIENTATION_SLICE,
    SPLIT_NO_LOG_TIME_SLICE,
    SPLIT_NO_LOG_VELOCITY_SLICE,
    SPLIT_NO_LOG_VERTICAL_GRAVITY_SLICE,
    SPLIT_NO_LOG_VERTICAL_HEIGHT_SLICE,
    SPLIT_NO_LOG_VERTICAL_URGENCY_SLICE,
    SPLIT_END_EFFECTOR_SLICE,
    SPLIT_END_EFFECTOR_SUBSLICES,
    SPLIT_GOAL_DIM,
    SPLIT_GOAL_NO_LOG_DIM,
    SPLIT_HORIZONTAL_SLICE,
    SPLIT_HORIZONTAL_URGENCY_SLICE,
    SPLIT_JOINT_SLICE,
    SPLIT_ORIENTATION_SLICE,
    SPLIT_TIME_SLICE,
    SPLIT_VELOCITY_SLICE,
    SPLIT_VERTICAL_GRAVITY_SLICE,
    SPLIT_VERTICAL_HEIGHT_SLICE,
    SPLIT_VERTICAL_URGENCY_SLICE,
    V6_RAW_JOINT_SLICE,
    V6_RAW_ORIENTATION_SLICE,
    V6_RAW_POSITION_SLICE,
    V6_RAW_TIME_SLICE,
    V6_RAW_VELOCITY_SLICE,
    build_ego_goal,
    build_ego_joint_state_goal_v6,
    build_ego_split_end_effector_goal_no_log,
    scale_split_goal_no_log,
    validate_goal_config,
)
from robotmdar.utils.occupancy import (
    compute_scene_surface_batch,
    query_local_occupancy,
)
from robotmdar.dtype import seed, logger as logger_module
from robotmdar.dtype.abc import VAE, Dataset, Denoiser, Diffusion, Optimizer, SSampler
from robotmdar.utils.dof_contract import (
    configure_dof_contract,
    validate_training_contract,
)
from robotmdar.train.manager import DARManager, is_main_process, get_ddp_model
import robotmdar.dtype.motion as motion_dtype


# ── Default XY plot limit from 64-frame goal distribution analysis ──
# The mode (peak of KDE) is ~0.02 m — most goals lie near the origin.
# 0.2 gives a tight 0.4×0.4 m view centred on the reference frame;
# expands dynamically when a trajectory or goal exceeds it.
# See dataset/data_analyze/analyze_64f_goal_xy.py
_DEFAULT_XY_LIMIT = 0.2
_PREFETCH_ERROR = object()


class _BackgroundPrefetchIterator:
    """Generate one CPU batch ahead while the current batch trains."""

    def __init__(self, source):
        self._source = iter(source)
        self._queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="textop-data-prefetch",
            daemon=True,
        )
        self._thread.start()

    def _run(self):
        try:
            while not self._stop.is_set():
                batch = next(self._source)
                while not self._stop.is_set():
                    try:
                        self._queue.put(batch, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:
            self._queue.put((_PREFETCH_ERROR, exc))

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if (isinstance(item, tuple)
                and len(item) == 2
                and item[0] is _PREFETCH_ERROR):
            self.close()
            raise item[1]
        return item

    def close(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


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


def _raw_goal_root_target(ego_goal_raw: torch.Tensor) -> torch.Tensor:
    if (motion_dtype.FeatureVersion == 6
            and ego_goal_raw.shape[-1] == ROT_MAT_JOINT_STATE_GOAL_DIM):
        return ego_goal_raw[:, 1:4]
    if (motion_dtype.FeatureVersion == 6
            and ego_goal_raw.shape[-1]
            in (
                SPLIT_GOAL_DIM,
                SPLIT_END_EFFECTOR_GOAL_DIM,
                SPLIT_GOAL_NO_LOG_DIM,
                SPLIT_END_EFFECTOR_NO_LOG_GOAL_DIM,
            )):
        return ego_goal_raw[:, SPLIT_HORIZONTAL_SLICE][:, :3]
    return ego_goal_raw[:, :2]


def _raw_goal_xy(ego_goal_raw: torch.Tensor) -> torch.Tensor:
    return _raw_goal_root_target(ego_goal_raw)[:, :2]


def _loss_weight_config_value(loss_weight, key: str) -> float:
    if loss_weight is None:
        return 0.0
    sections = ('locomotion', 'getup')
    values = []
    for section in sections:
        section_weight = loss_weight.get(section)
        if section_weight is not None:
            values.append(_loss_weight_config_value_from_mapping(
                section_weight, key))
    values.append(_loss_weight_config_value_from_mapping(loss_weight, key))
    return max(values)


def _loss_weight_config_value_from_mapping(loss_weight, key: str) -> float:
    value = loss_weight.get(key, None)
    if value is not None:
        return float(value)
    if key.startswith('goal_'):
        goal_weight = loss_weight.get('goal', None)
        if goal_weight is not None:
            value = goal_weight.get(key[len('goal_'):], None)
            if value is not None:
                return float(value)
    return 0.0


def _mapping_get(mapping, key, default=None):
    if mapping is None:
        return default
    getter = getattr(mapping, 'get', None)
    if getter is not None:
        return getter(key, default)
    try:
        return mapping[key]
    except (KeyError, TypeError):
        return default


def _loss_weight_config_value_for_section(loss_weight, key: str,
                                          section: str) -> float:
    """Resolve one loss weight exactly as DARManager does for a profile."""
    if loss_weight is None:
        return 0.0
    sections = ('locomotion', 'getup')
    has_sections = any(
        _mapping_get(loss_weight, name, None) is not None
        for name in sections
    )
    if has_sections:
        section_weights = _mapping_get(loss_weight, section, None)
        if (_mapping_get(section_weights, key, None) is not None
                or (key.startswith('goal_')
                    and _mapping_get(section_weights, 'goal', None) is not None
                    and _mapping_get(
                        _mapping_get(section_weights, 'goal', None),
                        key[len('goal_'):],
                        None,
                    ) is not None)):
            return _loss_weight_config_value_from_mapping(
                section_weights, key)
        top_value = _mapping_get(loss_weight, key, None)
        top_goal = _mapping_get(loss_weight, 'goal', None)
        if top_value is not None or (
                key.startswith('goal_')
                and _mapping_get(top_goal, key[len('goal_'):], None)
                is not None):
            return _loss_weight_config_value_from_mapping(loss_weight, key)
        if section == 'locomotion':
            return 0.0
        return 0.0
    return _loss_weight_config_value_from_mapping(loss_weight, key)


def _condition_profile_value(denoiser, profile: str, name: str,
                             index: int | None = None) -> float:
    profiles = getattr(denoiser, 'cond_mask_prob_profiles', None)
    if profiles is not None:
        profile_config = _mapping_get(profiles, profile, None)
        value = _mapping_get(profile_config, name, 0.0)
    else:
        value = getattr(denoiser, name, 0.0)
    if index is not None:
        value = value[index]
    return float(value)


def _condition_active_for_training(
    loss_weight,
    denoiser,
    loss_keys,
    mask_name: str,
    index: int | None = None,
) -> bool:
    """Whether a condition can contribute for at least one training profile."""
    for profile in ('locomotion', 'getup'):
        profile_weight = max(
            _loss_weight_config_value_for_section(
                loss_weight, key, profile)
            for key in loss_keys
        )
        if (profile_weight > 0.0
                and _condition_profile_value(
                    denoiser, profile, mask_name, index=index) < 1.0):
            return True
    return False


def _condition_can_be_kept(denoiser, name: str, index: int | None = None) -> bool:
    return any(
        _condition_profile_value(denoiser, profile, name, index=index) < 1.0
        for profile in ('locomotion', 'getup')
    )


def _uses_split_goal_masking_for_conditions(goal_encoding: GoalEncoding) -> bool:
    return goal_encoding in (
        GoalEncoding.SINGLE,
        GoalEncoding.SPLIT,
        GoalEncoding.SPLIT_END_EFFECTOR,
    )


def _build_train_condition_plan(cfg, denoiser):
    """Build static training gates for conditions and their loss terms.

    The plan is global because a batch can mix locomotion and get-up samples.
    A component is retained when at least one profile has a non-zero loss
    weight and can keep the corresponding condition.  Per-sample stochastic
    dropout is still handled by the denoiser keep masks.
    """
    if denoiser is None:
        return None

    loss_weight = _mapping_get(
        _mapping_get(_mapping_get(cfg, 'train', None), 'manager', None),
        'loss_weight',
        None,
    )
    goal_type = GoalType.parse(_mapping_get(
        _mapping_get(cfg, 'data', None), 'goal_type', GoalType.ROOT))
    goal_encoding = GoalEncoding.parse(_mapping_get(
        _mapping_get(cfg, 'data', None), 'goal_encoding',
        GoalEncoding.LEGACY40))

    position = _condition_active_for_training(
        loss_weight,
        denoiser,
        ('goal_root_position', 'goal_root_position_hor',
         'goal_root_position_vert'),
        'goal_position',
    )
    position_hor = _condition_active_for_training(
        loss_weight,
        denoiser,
        ('goal_root_position', 'goal_root_position_hor'),
        'goal_position_hor',
    )
    position_vert = _condition_active_for_training(
        loss_weight,
        denoiser,
        ('goal_root_position', 'goal_root_position_vert'),
        'goal_position_vert',
    )
    orientation = _condition_active_for_training(
        loss_weight,
        denoiser,
        ('goal_root_orientation',),
        'goal_orientation_rot6d',
    )
    orientation_source = orientation or _condition_active_for_training(
        loss_weight,
        denoiser,
        ('goal_g',),
        'goal_gravity',
    )
    gravity = _condition_active_for_training(
        loss_weight,
        denoiser,
        ('goal_g',),
        'goal_gravity',
    )
    joint = _condition_active_for_training(
        loss_weight,
        denoiser,
        ('goal_joint_angle',),
        'goal_joint',
    )
    velocity = _condition_active_for_training(
        loss_weight,
        denoiser,
        ('goal_root_velocity',),
        'goal_velocity',
    )
    end_effector = tuple(
        _condition_active_for_training(
            loss_weight,
            denoiser,
            ('goal_end_effector',),
            'goal_end_effector',
            index=index,
        )
        for index, _ in enumerate(SPLIT_END_EFFECTOR_TOKEN_ORDER)
    )

    # Legacy40 and non-joint goals use one root/orientation condition rather
    # than the current split condition names.
    if goal_type is GoalType.JOINT_STATE and goal_encoding is GoalEncoding.LEGACY40:
        position_hor = position_vert = position
        orientation = _condition_active_for_training(
            loss_weight,
            denoiser,
            ('goal_root_orientation',),
            'goal_orientation',
        )
        orientation_source = orientation
        gravity = False
        joint = _condition_active_for_training(
            loss_weight,
            denoiser,
            ('goal_joint_angle',),
            'goal_joint',
        )
        velocity = _condition_active_for_training(
            loss_weight,
            denoiser,
            ('goal_root_velocity',),
            'goal_velocity',
        )
        end_effector = (False,) * len(SPLIT_END_EFFECTOR_TOKEN_ORDER)
    elif goal_type is not GoalType.JOINT_STATE:
        position_hor = position_vert = position
        orientation = False
        orientation_source = False
        gravity = False
        joint = False
        velocity = False
        end_effector = (False,) * len(SPLIT_END_EFFECTOR_TOKEN_ORDER)

    goal_any = any((
        position_hor,
        position_vert,
        orientation,
        gravity,
        joint,
        velocity,
        *end_effector,
    ))
    time_condition = (
        goal_type.uses_arrival_time
        and goal_any
        and _condition_can_be_kept(denoiser, 'goal_time')
    )
    return {
        'goal_position': position,
        'goal_position_hor': position_hor,
        'goal_position_vert': position_vert,
        'goal_orientation': orientation,
        'goal_orientation_source': orientation_source,
        'goal_gravity': gravity,
        'goal_joint': joint,
        'goal_velocity': velocity,
        'goal_end_effector': end_effector,
        'goal_any': goal_any,
        'goal_time': time_condition,
        'scene': (
            bool(_mapping_get(_mapping_get(cfg, 'denoiser', None),
                              'scene_condition_enabled', True))
            and _condition_can_be_kept(denoiser, 'scene')
        ),
        'text': (
            bool(_mapping_get(_mapping_get(cfg, 'denoiser', None),
                              'text_condition_enabled', False))
            and _condition_can_be_kept(denoiser, 'text')
        ),
        'legacy_split_goal_layout': bool(
            getattr(denoiser, 'legacy_split_goal_layout', False)),
    }


def _zero_goal_components(goal: torch.Tensor, plan: dict) -> torch.Tensor:
    """Zero disabled channels after scaling, whose offsets may be non-zero."""
    if goal is None:
        return goal
    goal = goal.clone()
    goal_dim = int(goal.shape[-1])
    if goal_dim == JOINT_STATE_GOAL_DIM:
        if not plan['goal_position_hor']:
            goal[..., 0:3] = 0.0
        if not plan['goal_orientation']:
            goal[..., 3:8] = 0.0
        if not plan['goal_joint']:
            goal[..., 8:37] = 0.0
        if not plan['goal_velocity']:
            goal[..., 37:40] = 0.0
        return goal
    if (goal_dim == SPLIT_GOAL_DIM
            and plan.get('legacy_split_goal_layout', False)):
        if not plan['goal_position_hor']:
            goal[..., 1:6] = 0.0
            goal[..., 7:11] = 0.0
        if not plan['goal_position_vert']:
            goal[..., 0:1] = 0.0
            goal[..., 6:7] = 0.0
            goal[..., 11:12] = 0.0
        if not plan['goal_gravity']:
            goal[..., 12:15] = 0.0
        if not plan['goal_orientation']:
            goal[..., 15:21] = 0.0
        if not plan['goal_joint']:
            goal[..., 21:50] = 0.0
        if not plan['goal_velocity']:
            goal[..., 50:54] = 0.0
        if not plan['goal_time']:
            goal[..., 7:12] = 0.0
            goal[..., 54:55] = 0.0
        return goal
    if goal_dim in (SPLIT_GOAL_NO_LOG_DIM,
                    SPLIT_END_EFFECTOR_NO_LOG_GOAL_DIM):
        horizontal = SPLIT_NO_LOG_HORIZONTAL_SLICE
        horizontal_urgency = SPLIT_NO_LOG_HORIZONTAL_URGENCY_SLICE
        vertical_height = SPLIT_NO_LOG_VERTICAL_HEIGHT_SLICE
        vertical_gravity = SPLIT_NO_LOG_VERTICAL_GRAVITY_SLICE
        vertical_urgency = SPLIT_NO_LOG_VERTICAL_URGENCY_SLICE
        orientation = SPLIT_NO_LOG_ORIENTATION_SLICE
        joint = SPLIT_NO_LOG_JOINT_SLICE
        velocity = SPLIT_NO_LOG_VELOCITY_SLICE
        time = SPLIT_NO_LOG_TIME_SLICE
        end_effectors = SPLIT_NO_LOG_END_EFFECTOR_SUBSLICES
    else:
        horizontal = SPLIT_HORIZONTAL_SLICE
        horizontal_urgency = SPLIT_HORIZONTAL_URGENCY_SLICE
        vertical_height = SPLIT_VERTICAL_HEIGHT_SLICE
        vertical_gravity = SPLIT_VERTICAL_GRAVITY_SLICE
        vertical_urgency = SPLIT_VERTICAL_URGENCY_SLICE
        orientation = SPLIT_ORIENTATION_SLICE
        joint = SPLIT_JOINT_SLICE
        velocity = SPLIT_VELOCITY_SLICE
        time = SPLIT_TIME_SLICE
        end_effectors = SPLIT_END_EFFECTOR_SUBSLICES

    if not plan['goal_position_hor']:
        goal[..., horizontal] = 0.0
    if not plan['goal_position_vert']:
        goal[..., vertical_height] = 0.0
    if not plan['goal_gravity']:
        goal[..., vertical_gravity] = 0.0
    if not plan['goal_orientation']:
        goal[..., orientation] = 0.0
    if not plan['goal_joint']:
        goal[..., joint] = 0.0
    if not plan['goal_velocity']:
        goal[..., velocity] = 0.0
    if not plan['goal_time']:
        goal[..., horizontal_urgency] = 0.0
        goal[..., vertical_urgency] = 0.0
        goal[..., time] = 0.0
    if goal_dim in (SPLIT_END_EFFECTOR_GOAL_DIM,
                    SPLIT_END_EFFECTOR_NO_LOG_GOAL_DIM):
        for enabled, name in zip(
                plan['goal_end_effector'],
                SPLIT_END_EFFECTOR_TOKEN_ORDER):
            if not enabled:
                goal[..., end_effectors[name]] = 0.0
    return goal


def _zero_raw_goal_components(goal: torch.Tensor, plan: dict) -> torch.Tensor:
    if goal is None:
        return goal
    goal = goal.clone()
    if not plan['goal_position_hor'] and not plan['goal_position_vert']:
        goal[..., V6_RAW_POSITION_SLICE] = 0.0
    elif not plan['goal_position_hor']:
        goal[..., 1:4] = 0.0
    elif not plan['goal_position_vert']:
        goal[..., 0:1] = 0.0
    if not plan['goal_gravity'] and not plan['goal_orientation']:
        goal[..., V6_RAW_ORIENTATION_SLICE] = 0.0
    elif not plan['goal_gravity']:
        goal[..., 4:7] = 0.0
    elif not plan['goal_orientation']:
        goal[..., 7:13] = 0.0
    if not plan['goal_joint']:
        goal[..., V6_RAW_JOINT_SLICE] = 0.0
    if not plan['goal_velocity']:
        goal[..., V6_RAW_VELOCITY_SLICE] = 0.0
    if not plan['goal_time']:
        goal[..., V6_RAW_TIME_SLICE] = 0.0
    return goal


def _split_no_log_goal_from_v6_raw(raw_goal: torch.Tensor,
                                   reference_pos: torch.Tensor,
                                   fps: float) -> torch.Tensor:
    """Convert an unscaled v6 raw goal without rebuilding it from world data."""
    h_goal = raw_goal[..., 0:1]
    delta_hor = raw_goal[..., 1:4]
    goal_g = raw_goal[..., 4:7]
    rel_rot6d = raw_goal[..., 7:13]
    dof = raw_goal[..., 13:42]
    velocity = raw_goal[..., 42:46]
    time_to_arrival = raw_goal[..., 46:47]
    d_hor = torch.linalg.vector_norm(delta_hor, dim=-1, keepdim=True)
    delta_h = h_goal - reference_pos.to(
        device=raw_goal.device, dtype=raw_goal.dtype)[..., 2:3]
    time_scalar = time_to_arrival[..., 0]
    time_budget = time_scalar.clamp_min(1.0 / float(fps))
    urgency_source = torch.cat((delta_hor, d_hor, delta_h), dim=-1)
    urgency = torch.where(
        time_scalar[..., None] > 0.0,
        urgency_source / time_budget[..., None],
        torch.zeros_like(urgency_source),
    )
    horizontal = torch.cat(
        (delta_hor, d_hor, urgency[..., :4]),
        dim=-1,
    )
    vertical = torch.cat(
        (h_goal, delta_h, goal_g, urgency[..., 4:5]),
        dim=-1,
    )
    if raw_goal.shape[-1] == ROT_MAT_JOINT_STATE_GOAL_DIM:
        return torch.cat(
            (horizontal, vertical, rel_rot6d, dof, velocity, time_to_arrival),
            dim=-1,
        )
    if raw_goal.shape[-1] != ROT_MAT_JOINT_STATE_GOAL_DIM + 19:
        raise ValueError(
            "Expected v6 raw joint-state or split-end-effector raw goal, got "
            f"shape {tuple(raw_goal.shape)}"
        )
    return torch.cat(
        (
            horizontal,
            vertical,
            rel_rot6d,
            dof,
            velocity,
            time_to_arrival,
            raw_goal[..., ROT_MAT_JOINT_STATE_GOAL_DIM:],
        ),
        dim=-1,
    )


def _as_recovery_mask(value, batch_size: int, device) -> torch.Tensor:
    mask = torch.as_tensor(value, device=device, dtype=torch.bool).reshape(-1)
    if mask.numel() == 1:
        return mask.expand(batch_size)
    if mask.numel() != batch_size:
        raise ValueError(
            f"is_recovery must be scalar or [B]={batch_size}, "
            f"got {mask.numel()} values")
    return mask


def _masked_mean(mask: torch.Tensor, selector: torch.Tensor) -> torch.Tensor:
    if selector.any():
        return mask.to(dtype=torch.float32)[selector].mean()
    return mask.new_zeros((), dtype=torch.float32)


def _add_condition_keep_diagnostics(extras, y, is_recovery) -> None:
    mask_keys = {
        'position_hor': 'goal_position_hor_condition_keep_mask',
        'position_vert': 'goal_position_vert_condition_keep_mask',
        'gravity': 'goal_gravity_condition_keep_mask',
        'orientation_rot6d': 'goal_orientation_condition_keep_mask',
        'joint': 'goal_joint_condition_keep_mask',
        'velocity': 'goal_velocity_condition_keep_mask',
        'time': 'goal_time_condition_keep_mask',
    }
    batch_size = None
    device = None
    for key in mask_keys.values():
        value = y.get(key)
        if isinstance(value, torch.Tensor):
            batch_size = int(value.shape[0])
            device = value.device
            break
    ee_mask = y.get('goal_end_effector_condition_keep_mask')
    if batch_size is None and isinstance(ee_mask, torch.Tensor):
        batch_size = int(ee_mask.shape[0])
        device = ee_mask.device
    if batch_size is None:
        return
    category = y.get('_condition_sampling_plan', {}).get(
        '_cardinality_category') if isinstance(
            y.get('_condition_sampling_plan'), dict) else None
    if isinstance(category, torch.Tensor) and category.shape == (batch_size,):
        category = category.to(device=device, dtype=torch.long)
        category_names = ('none', 'single', 'composition', 'full_condition')
        for idx, name in enumerate(category_names):
            extras[f'condition/cardinality_{name}_ratio'] = (
                (category == idx).float().mean())
    recovery = _as_recovery_mask(is_recovery, batch_size, device)
    locomotion = ~recovery
    extras['data/batch_recovery_fraction'] = recovery.float().mean()
    for name, key in mask_keys.items():
        value = y.get(key)
        if not isinstance(value, torch.Tensor):
            continue
        keep = value.to(device=device, dtype=torch.bool).reshape(batch_size)
        extras[f'condition/{name}_keep_ratio'] = keep.float().mean()
        extras[f'condition/recovery_{name}_keep_ratio'] = (
            _masked_mean(keep, recovery))
        extras[f'condition/locomotion_{name}_keep_ratio'] = (
            _masked_mean(keep, locomotion))
    if isinstance(ee_mask, torch.Tensor):
        keep = ee_mask.to(device=device, dtype=torch.bool)
        if keep.ndim != 2 or keep.shape[0] != batch_size:
            raise ValueError(
                "goal_end_effector_condition_keep_mask must be [B, N], "
                f"got {tuple(keep.shape)}")
        per_sample = keep.float().mean(dim=1)
        extras['condition/end_effector_keep_ratio'] = per_sample.mean()
        extras['condition/recovery_end_effector_keep_ratio'] = (
            per_sample[recovery].mean()
            if recovery.any() else per_sample.new_zeros(()))
        extras['condition/locomotion_end_effector_keep_ratio'] = (
            per_sample[locomotion].mean()
            if locomotion.any() else per_sample.new_zeros(()))
    if isinstance(category, torch.Tensor):
        atomic_masks = dict(mask_keys)
        atomic_masks['end_effector'] = 'goal_end_effector_condition_keep_mask'
        for name, key in atomic_masks.items():
            value = y.get(key)
            if not isinstance(value, torch.Tensor):
                continue
            keep = value.to(device=device, dtype=torch.bool)
            if name == 'end_effector':
                keep = keep.all(dim=1)
            else:
                keep = keep.reshape(batch_size)
            extras[f'condition/single_{name}_ratio'] = (
                ((category == 1) & keep).float().mean())


def _add_batch_data_diagnostics(extras, primitive, y) -> None:
    goal = y.get('goal')
    device = goal.device if isinstance(goal, torch.Tensor) else torch.device('cpu')
    if isinstance(goal, torch.Tensor):
        batch_size = int(goal.shape[0])
    else:
        recovery_value = primitive.get('is_recovery', False)
        batch_size = int(torch.as_tensor(recovery_value).numel())
    recovery = _as_recovery_mask(
        primitive.get('is_recovery', False), batch_size, device)
    labels = primitive.get('action_label')
    if labels is not None:
        empty = torch.as_tensor(
            [not str(label).strip() for label in labels],
            device=device, dtype=torch.bool)
        extras['data/batch_action_label_empty_rate'] = empty.float().mean()
        extras['data/recovery_action_label_empty_rate'] = (
            _masked_mean(empty, recovery))
    embeddings = primitive.get('text_embedding')
    if isinstance(embeddings, torch.Tensor):
        emb = embeddings.to(device=device)
        empty_embedding = emb.reshape(emb.shape[0], -1).abs().sum(dim=1) == 0
        extras['data/batch_text_embedding_empty_rate'] = (
            empty_embedding.float().mean())
        extras['data/recovery_text_embedding_empty_rate'] = (
            _masked_mean(empty_embedding, recovery))
    _add_condition_keep_diagnostics(extras, y, recovery)


def _make_root_xy_figure(
    generated_trajectory: torch.Tensor,
    goal_xy: torch.Tensor,
    ground_truth_trajectory: torch.Tensor = None,
    goal_time_frame: torch.Tensor = None,
    goal_condition_keep_mask: torch.Tensor = None,
    history_trajectory: torch.Tensor = None,
    voxel: torch.Tensor = None,
    grid_size: int = 25,
    grid_unit: float = 0.08,
    xy_limit: float | None = None,
    max_samples: int = 4,
    labels: Optional[List[str]] = None,
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
    goal_time = (
        goal_time_frame.detach().long().cpu()
        if goal_time_frame is not None else None
    )
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
        generated_xy = generated[sample_idx, :, :2].numpy()
        goal = goals[sample_idx, :2].numpy()
        goal_idx = generated_xy.shape[0] - 1
        if goal_time is not None:
            goal_idx = int(goal_time[sample_idx].item())
            goal_idx = max(0, min(goal_idx, generated_xy.shape[0] - 1))
        goal_frame_xy = generated_xy[goal_idx]
        primitive_end_xy = generated_xy[-1]
        goal_error = float(np.linalg.norm(goal_frame_xy - goal))
        primitive_end_error = float(np.linalg.norm(primitive_end_xy - goal))

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
            hist_xy = history[sample_idx, :, :2].numpy()
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
            ground_truth_xy = ground_truth[sample_idx, :, :2].numpy()
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
        if goal_time is not None:
            axis.scatter(
                [goal_frame_xy[0]], [goal_frame_xy[1]],
                color='#009E73', marker='o', s=42, zorder=6,
                label='goal frame',
            )
        axis.scatter([goal[0]], [goal[1]], color='#D55E00', marker='X',
                     s=90, zorder=6, label='goal')
        axis.scatter(
            [primitive_end_xy[0]], [primitive_end_xy[1]],
            color='#CC79A7', marker='s', s=42, zorder=6,
            label='primitive end',
        )

        label = None
        if (labels is not None and sample_idx < len(labels)
                and labels[sample_idx]):
            label = labels[sample_idx]
        title_label = label if label is not None else f'sample {sample_idx}'
        if goal_time is not None:
            axis.set_title(
                f'{title_label}  |  goal error {goal_error:.3f} m'
                f'  |  primitive end error {primitive_end_error:.3f} m'
            )
        else:
            axis.set_title(
                f'{title_label}  |  primitive end error '
                f'{primitive_end_error:.3f} m'
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


def _time_to_arrival_frame(time_to_arrival: torch.Tensor,
                           fps: float,
                           device: str | torch.device) -> torch.Tensor:
    time_to_arrival = time_to_arrival.to(device=device, dtype=torch.float32)
    if time_to_arrival.ndim > 1:
        time_to_arrival = time_to_arrival.squeeze(-1)
    return torch.round(time_to_arrival.clamp_min(0.0) * float(fps)).to(
        dtype=torch.long)


def _goal_time_frame_for_loss(conditions, cfg):
    goal_timestep_mode = str(cfg.data.get(
        'goal_timestep_mode', cfg.data.get('time_to_arrival_mode', 'relative'))
    ).lower()
    if goal_timestep_mode == 'zero':
        return None
    return conditions.get(
        'goal_time_frame_for_loss',
        conditions.get(
            'time_to_arrival_frame',
            conditions.get('arrival_time_frame'),
        ),
    )


def _conditions(primitive, reference_pos, reference_rot, history_motion, cfg,
                fps: float, goal_stats=None, use_scene: bool = True,
                text_embedding=None, condition_plan=None):
    goal_type = GoalType.parse(cfg.data.goal_type)
    goal_encoding = GoalEncoding.parse(
        cfg.data.get('goal_encoding', GoalEncoding.LEGACY40)
    )
    planned = condition_plan is not None
    batch_size = reference_pos.shape[0]
    device = reference_pos.device
    time_to_arrival = primitive.get(
        'time_to_arrival', primitive.get('goal_timestep'))
    if goal_type.uses_arrival_time:
        if time_to_arrival is None:
            raise ValueError(
                f"{goal_type.value} training requires "
                "primitive['time_to_arrival']")
        if not planned or condition_plan['goal_any']:
            goal_time = time_to_arrival.to(device)
        else:
            goal_time = torch.zeros(
                batch_size,
                device=device,
                dtype=reference_pos.dtype,
            )
    else:
        goal_time = None

    build_goal = not planned or condition_plan['goal_any']
    if build_goal and planned and goal_type is GoalType.JOINT_STATE:
        actual_world_goal_pos = primitive['world_goal_pos'].to(device)
        if condition_plan['goal_position_hor']:
            world_goal_pos = actual_world_goal_pos
        else:
            world_goal_pos = reference_pos.clone()
            if condition_plan['goal_position_vert']:
                world_goal_pos[..., 2:3] = actual_world_goal_pos[..., 2:3]
        world_goal_rot = (
            primitive['world_goal_rot'].to(device)
            if condition_plan['goal_orientation_source']
            else reference_rot
        )
        world_goal_dof = (
            primitive['world_goal_dof'].to(device)
            if condition_plan['goal_joint']
            else torch.zeros(
                batch_size, 29, device=device, dtype=reference_pos.dtype)
        )
        world_root_velocity = (
            primitive['world_goal_vel'].to(device)
            if condition_plan['goal_velocity']
            else torch.zeros(
                batch_size, 3, device=device, dtype=reference_pos.dtype)
        )
        world_goal_end_effectors = (
            primitive['world_goal_end_effectors'].to(device)
            if any(condition_plan['goal_end_effector'])
            else torch.zeros(
                batch_size, 4, 3, device=device, dtype=reference_pos.dtype)
        )
        world_goal_yaw = torch.zeros(
            batch_size, device=device, dtype=reference_pos.dtype)
    elif build_goal:
        world_goal_pos = primitive['world_goal_pos'].to(device)
        world_goal_yaw = primitive['world_goal_yaw'].to(device)
        world_goal_rot = (
            primitive['world_goal_rot'].to(device)
            if goal_type is GoalType.JOINT_STATE else None
        )
        world_goal_dof = (
            primitive['world_goal_dof'].to(device)
            if goal_type is GoalType.JOINT_STATE else None
        )
        world_goal_end_effectors = (
            primitive['world_goal_end_effectors'].to(device)
            if goal_type is GoalType.JOINT_STATE
            and goal_encoding.uses_end_effectors else None
        )
        world_root_velocity = (
            primitive['world_goal_vel'].to(device)
            if goal_type.uses_arrival_time else None
        )
    else:
        world_goal_pos = None
        world_goal_yaw = None
        world_goal_rot = None
        world_goal_dof = None
        world_goal_end_effectors = None
        world_root_velocity = None

    if not build_goal:
        ego_goal_raw = None
        goal = torch.zeros(
            batch_size,
            int(cfg.denoiser.goal_dim),
            device=device,
            dtype=reference_pos.dtype,
        )
    elif goal_type is GoalType.JOINT_STATE and motion_dtype.FeatureVersion == 6:
        if goal_encoding.uses_end_effectors:
            if not planned or any(condition_plan['goal_end_effector']):
                ego_goal_raw = build_ego_split_end_effector_goal_no_log(
                    world_goal_pos=world_goal_pos,
                    world_goal_rot=world_goal_rot,
                    world_goal_dof=world_goal_dof,
                    world_root_velocity=world_root_velocity,
                    world_goal_end_effectors=world_goal_end_effectors,
                    reference_pos=reference_pos,
                    reference_rot=reference_rot,
                    time_to_arrival_seconds=goal_time,
                    fps=fps,
                    distance_scale=(
                        goal_stats.get('s_d', 1.0)
                        if goal_stats is not None else 1.0),
                )
            else:
                # The model still needs the fixed 66-D shape, but no EE
                # coordinate transform is needed when all EE tokens are off.
                base_goal_raw = build_ego_joint_state_goal_v6(
                    world_goal_pos=world_goal_pos,
                    world_goal_rot=world_goal_rot,
                    world_goal_dof=world_goal_dof,
                    world_root_velocity=world_root_velocity,
                    reference_pos=reference_pos,
                    reference_rot=reference_rot,
                    time_to_arrival_seconds=goal_time,
                    fps=fps,
                )
                base_goal_raw = _zero_raw_goal_components(
                    base_goal_raw, condition_plan)
                base_goal = _split_no_log_goal_from_v6_raw(
                    base_goal_raw, reference_pos, fps)
                zero_end_effectors = torch.zeros(
                    batch_size,
                    SPLIT_NO_LOG_END_EFFECTOR_SLICE.stop
                    - SPLIT_NO_LOG_END_EFFECTOR_SLICE.start,
                    device=device,
                    dtype=base_goal.dtype,
                )
                ego_goal_raw = torch.cat(
                    (base_goal, zero_end_effectors), dim=-1)
        else:
            ego_goal_raw = build_ego_joint_state_goal_v6(
                world_goal_pos=world_goal_pos,
                world_goal_rot=world_goal_rot,
                world_goal_dof=world_goal_dof,
                world_root_velocity=world_root_velocity,
                reference_pos=reference_pos,
                reference_rot=reference_rot,
                time_to_arrival_seconds=goal_time,
                fps=fps,
            )
        if planned:
            if goal_encoding.uses_end_effectors:
                ego_goal_raw = _zero_goal_components(
                    ego_goal_raw, condition_plan)
            else:
                ego_goal_raw = _zero_raw_goal_components(
                    ego_goal_raw, condition_plan)
        if goal_encoding is not GoalEncoding.LEGACY40:
            if goal_stats is None:
                raise ValueError(
                    "joint_state goal_encoding requires goal_stats")
            configured_goal_dim = _mapping_get(
                _mapping_get(cfg, 'denoiser', None), 'goal_dim', None)
            if (planned
                    and configured_goal_dim is not None
                    and int(configured_goal_dim) in (
                    SPLIT_GOAL_NO_LOG_DIM,
                    SPLIT_END_EFFECTOR_NO_LOG_GOAL_DIM)):
                if goal_encoding.uses_end_effectors:
                    goal = ego_goal_raw
                else:
                    goal = _split_no_log_goal_from_v6_raw(
                        ego_goal_raw, reference_pos, fps)
                goal = scale_split_goal_no_log(goal, goal_stats)
            else:
                goal = build_ego_goal(
                    world_goal_pos,
                    world_goal_yaw,
                    reference_pos,
                    reference_rot,
                    goal_type=goal_type,
                    goal_encoding=goal_encoding,
                    goal_stats=goal_stats,
                    fps=fps,
                    world_goal_rot=world_goal_rot,
                    world_goal_dof=world_goal_dof,
                    world_goal_end_effectors=world_goal_end_effectors,
                    world_root_velocity=world_root_velocity,
                    time_to_arrival_seconds=goal_time,
                    goal_include_log_d_hor=False,
                )
        else:
            goal = ego_goal_raw
        if planned:
            goal = (
                _zero_raw_goal_components(goal, condition_plan)
                if goal.shape[-1] == ROT_MAT_JOINT_STATE_GOAL_DIM
                else _zero_goal_components(goal, condition_plan)
            )
    else:
        ego_goal_raw = build_ego_goal(
            world_goal_pos,
            world_goal_yaw,
            reference_pos,
            reference_rot,
            goal_type=goal_type,
            goal_encoding=GoalEncoding.LEGACY40,
            world_goal_keypoints=(
                primitive['world_goal_keypoints'].to(device)
                if goal_type.uses_keypoints else None
            ),
            world_root_velocity=world_root_velocity,
            timestep=goal_time,
            world_goal_rot=world_goal_rot,
            world_goal_dof=world_goal_dof,
        )
        if planned and goal_type is GoalType.JOINT_STATE:
            ego_goal_raw = _zero_goal_components(
                ego_goal_raw, condition_plan)
        goal = ego_goal_raw

    time_to_arrival_frame = None
    goal_time_frame_for_loss = None
    if goal_type.uses_arrival_time:
        if not planned or condition_plan['goal_any']:
            goal_time_frame_for_loss = _time_to_arrival_frame(
                time_to_arrival, fps=fps, device=device)
        if goal_time_frame_for_loss is None:
            time_to_arrival_frame = torch.zeros(
                batch_size, device=device, dtype=torch.long)
        elif not planned or condition_plan['goal_time']:
            time_to_arrival_frame = goal_time_frame_for_loss
        else:
            # Position urgency still depends on the true arrival time, but
            # the standalone time condition is dropped from the model.
            time_to_arrival_frame = torch.zeros_like(
                goal_time_frame_for_loss)

    scene_enabled = (
        condition_plan is None or condition_plan['scene'])
    if use_scene and scene_enabled:
        voxel = query_local_occupancy(
            primitive['scene'],
            reference_pos,
            reference_rot,
            grid_size=cfg.denoiser.grid_size,
            grid_unit=cfg.data.occupancy_unit,
        )
        if bool(cfg.data.get('use_scene_surface', True)):
            # Surface (motion-envelope) scene representation: a LOCAL
            # operation on the ego occupancy from query_local_occupancy —
            # the global scene dicts are never modified.  Only the obstacle
            # boundary shell conditions the denoiser; the envelope thickness
            # is randomized per sample (uniform in {1, 2, 3}) like the other
            # domain-randomization knobs.  Applied at the source so training,
            # eval and rollout all consume the same representation.
            grid_size = int(cfg.denoiser.grid_size)
            grid = voxel.view(voxel.shape[0], grid_size, grid_size, grid_size)
            thicknesses = [random.choice((1, 2, 3))
                           for _ in range(grid.shape[0])]
            voxel = compute_scene_surface_batch(
                grid, thicknesses=thicknesses).view(voxel.shape[0], -1).to(
                    dtype=voxel.dtype)
    else:
        # Pre-scene curriculum phase (step < manager.scene_start_step): the
        # denoiser learns basic goal-driven motion on a blank occupancy grid.
        voxel = torch.zeros(
            batch_size,
            int(cfg.denoiser.grid_size) ** 3,
            device=device,
            dtype=reference_pos.dtype,
        )
    conditions = {
        'goal': goal,
        'ego_goal_raw': ego_goal_raw,
        'goal_reference_pos_world': reference_pos,
        'goal_reference_rot_world': reference_rot,
        'voxel': voxel,
        'history_motion_normalized': history_motion,
        # Condition dropout is selected per sample.  The dataset already
        # batches this flag as [B] for each primitive; scalar callers (for
        # example an external evaluator) are expanded by the denoiser.
        'is_recovery': primitive.get('is_recovery', False),
        **(
            {
                'time_to_arrival_frame': time_to_arrival_frame,
                'arrival_time_frame': time_to_arrival_frame,
            }
            if time_to_arrival_frame is not None else {}
        ),
    }
    if goal_time_frame_for_loss is not None:
        conditions['goal_time_frame_for_loss'] = goal_time_frame_for_loss
    if planned:
        conditions.update({
            'force_drop_goal_position_hor': (
                not condition_plan['goal_position_hor']),
            'force_drop_goal_position_vert': (
                not condition_plan['goal_position_vert']),
            'force_drop_goal_gravity': not condition_plan['goal_gravity'],
            'force_drop_goal_orientation_rot6d': (
                not condition_plan['goal_orientation']),
            'force_drop_goal_joint': not condition_plan['goal_joint'],
            'force_drop_goal_velocity': not condition_plan['goal_velocity'],
            'force_drop_goal_time': not condition_plan['goal_time'],
            'force_drop_scene': not condition_plan['scene'],
        })
        if (goal_type is GoalType.JOINT_STATE
                and (
                    not _uses_split_goal_masking_for_conditions(goal_encoding)
                    or condition_plan.get(
                        'legacy_split_goal_layout', False)
                )):
            conditions.update({
                # The legacy split layout shares horizontal and vertical
                # position in one token, so only drop it when both are off.
                'force_drop_goal_root': not (
                    condition_plan['goal_position_hor']
                    or condition_plan['goal_position_vert']
                ),
                'force_drop_goal_orientation': (
                    not condition_plan['goal_orientation']),
            })
        elif goal_type is not GoalType.JOINT_STATE:
            conditions['force_drop_goal_root'] = (
                not condition_plan['goal_position'])
        if not condition_plan['text']:
            text_embedding = None
    if (goal_encoding.uses_end_effectors
            and isinstance(ego_goal_raw, torch.Tensor)):
        ee_slice = (
            SPLIT_NO_LOG_END_EFFECTOR_SLICE
            if ego_goal_raw.shape[-1] == SPLIT_END_EFFECTOR_NO_LOG_GOAL_DIM
            else SPLIT_END_EFFECTOR_SLICE
        )
        conditions['goal_end_effectors_ego_raw'] = ego_goal_raw[
            :, ee_slice].reshape(-1, 4, 3)
    if (text_embedding is None and 'text_embedding' in primitive
            and (not planned or condition_plan['text'])):
        text_embedding = primitive['text_embedding'].to(
            device, non_blocking=True)
    if text_embedding is not None:
        conditions['text_embedding'] = text_embedding
    return conditions


def _prepare_batch_text_embeddings(
        batch, cfg, device, optimize_always_dropped: bool = False):
    """Move all primitive text embeddings to the training device once."""
    if not bool(cfg.denoiser.get('text_condition_enabled', False)):
        return None
    cond_mask_prob = cfg.denoiser.get('cond_mask_prob', {})
    text_mask_probs = []
    if any(cond_mask_prob.get(name) is not None
           for name in ('locomotion', 'getup')):
        for name in ('locomotion', 'getup'):
            profile = cond_mask_prob.get(name)
            if profile is not None and profile.get('text') is not None:
                text_mask_probs.append(float(profile.get('text')))
    elif cond_mask_prob.get('text') is not None:
        text_mask_probs.append(float(cond_mask_prob.get('text')))
    if optimize_always_dropped and text_mask_probs and min(text_mask_probs) >= 1.0:
        # Text is guaranteed to be dropped during training; avoid the
        # otherwise unnecessary CPU -> GPU transfer for this condition.
        return None
    embeddings = [primitive.get('text_embedding') for primitive in batch]
    if any(embedding is None for embedding in embeddings):
        return None
    return torch.stack(embeddings, dim=0).to(
        device=device, non_blocking=True)


def _next_rollout_poses(dataset, motion, history_start_pos, history_start_rot, history_len):
    with torch.no_grad():
        # Keep rollout windows chained from the segment's first GT history
        # anchor by reconstructing the full used [history + future] window.
        reconstructed = dataset.reconstruct_motion(
            motion,
            abs_pose=_pose_dict(history_start_pos, history_start_rot),
            ret_fk=False,
        )
    history_anchor_idx = (
        -history_len - 1 if motion_dtype.FeatureVersion == 6 else -history_len
    )
    return (
        reconstructed['root_trans_offset'][:, history_anchor_idx].detach(),
        reconstructed['root_rot'][:, history_anchor_idx].detach(),
        reconstructed['root_trans_offset'][:, -1].detach(),
        reconstructed['root_rot'][:, -1].detach(),
    )


class _preserve_rng:
    """Save/restore CPU+CUDA RNG state so visualization sampling cannot
    perturb subsequent eval-step metrics."""

    def __enter__(self):
        self.cpu_state = torch.random.get_rng_state()
        self.cuda_state = (
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        )
        return self

    def __exit__(self, *exc):
        torch.random.set_rng_state(self.cpu_state)
        if self.cuda_state is not None:
            torch.cuda.set_rng_state(self.cuda_state)
        return False


def _slice_batch_for_viz(batch, max_samples: int = 4):
    """Slice every key of every primitive dict to the first max_samples.

    Tensor keys (motion, world_goal_pos, ...) and list keys (scene,
    action_label) both support [:] slicing, so they are handled uniformly.
    """
    return [
        {key: value[:max_samples] for key, value in primitive.items()}
        for primitive in batch
    ]


def _sample_segment_rollout(batch, dataset, vae, denoiser, diffusion,
                            manager, cfg):
    """Autoregressive full-segment rollout under the TRAINING history policy.

    Stage policy is delegated to manager.choose_history:
      stage 0 -> always GT history (should_rollout False)
      stage 1 -> probabilistic GT/predicted history (linear ramp)
      stage 2 -> predicted-history probability held at rollout_max_prob
    window 0 always uses GT history (prev_motion is None).  Visualization
    only: no loss/extras writes, no manager state mutation.

    Returns a dict of CPU tensors for plotting:
      gt_segment   [N, H + F*Np, 3]  world GT positions, whole segment
      pred_segment [N, H + F*Np, 3]  world predicted positions, whole segment
      hist_world   [Np, N, H, 3]     world positions of the USED history
      anchors      [Np, N, 3]        world anchor (history start) per window
      used_rollout [Np]              bool per window
    """
    device = cfg.device
    num_primitive = len(batch)
    history_len = int(cfg.data.history_len)
    future_len = int(cfg.data.future_len)

    prev_motion = None
    rollout_history_start_pos = rollout_history_start_rot = None
    rollout_ref_pos = rollout_ref_rot = None
    per_window = []

    for primitive in batch:
        motion = primitive['motion'].to(device)              # [N, H+F, D]
        gt_history = motion[:, :history_len]

        history_motion, used_rollout = manager.choose_history(
            gt_history, prev_motion, history_len, return_rollout=True)

        if used_rollout:
            if any(value is None for value in (
                rollout_history_start_pos,
                rollout_history_start_rot,
                rollout_ref_pos,
                rollout_ref_rot,
            )):
                raise RuntimeError(
                    "Self-rollout history selected before rollout poses "
                    "were initialized")
            history_start_pos = rollout_history_start_pos
            history_start_rot = rollout_history_start_rot
            reference_pos = rollout_ref_pos
            reference_rot = rollout_ref_rot
        else:
            history_start_pos = primitive['history_start_pos'].to(device)
            history_start_rot = primitive['history_start_rot'].to(device)
            reference_pos = primitive['gt_ref_pos'].to(device)
            reference_rot = primitive['gt_ref_rot'].to(device)

        y = _conditions(primitive, reference_pos, reference_rot,
                        history_motion, cfg, dataset.fps,
                        goal_stats=getattr(dataset, 'goal_stats', None),
                        use_scene=manager.should_use_scene())

        with torch.no_grad():
            # full DDPM sample, mirroring the eval loop's sampling path
            latent_gt, _ = vae.encode(
                future_motion=motion[:, -future_len:],
                history_motion=history_motion)
            sample_latent = diffusion.p_sample_loop(
                denoiser, latent_gt.permute(1, 0, 2).shape,
                clip_denoised=False, model_kwargs={'y': y}, progress=False)
            future_pred = vae.decode(
                sample_latent.permute(1, 0, 2),
                history_motion, nfuture=future_len)           # [N, F, D]

            # world reconstruction of the full used window (H + F frames)
            window = torch.cat([history_motion, future_pred], dim=1)
            recon = dataset.reconstruct_motion(
                window,
                abs_pose=_pose_dict(history_start_pos, history_start_rot),
                ret_fk=False)['root_trans_offset']            # [N, H+F, 3]
            # GT window: batch features anchored at the GT segment pose
            recon_gt = dataset.reconstruct_motion(
                motion,
                abs_pose=_pose_dict(
                    primitive['history_start_pos'].to(device),
                    primitive['history_start_rot'].to(device)),
                ret_fk=False)['root_trans_offset']

        per_window.append({
            'pred_full': recon.detach().float().cpu(),
            'gt_full': recon_gt.detach().float().cpu(),
            'anchor': history_start_pos.detach().float().cpu(),
            'used_rollout': bool(used_rollout),
        })

        prev_motion = torch.cat(
            [history_motion, future_pred], dim=1).detach()
        (rollout_history_start_pos, rollout_history_start_rot,
         rollout_ref_pos, rollout_ref_rot) = _next_rollout_poses(
            dataset, prev_motion, history_start_pos,
            history_start_rot, history_len)

    # ── stitch: consecutive future slices are contiguous in world frame ──
    gt_segment = torch.cat(
        [per_window[0]['gt_full'][:, :history_len]]
        + [w['gt_full'][:, history_len:] for w in per_window], dim=1)
    pred_segment = torch.cat(
        [per_window[0]['pred_full'][:, :history_len]]
        + [w['pred_full'][:, history_len:] for w in per_window], dim=1)
    hist_world = torch.stack(
        [w['pred_full'][:, :history_len] for w in per_window], dim=0)
    anchors = torch.stack([w['anchor'] for w in per_window], dim=0)

    return {
        'gt_segment': gt_segment,        # [N, H + F*Np, 3]
        'pred_segment': pred_segment,    # [N, H + F*Np, 3]
        'hist_world': hist_world,        # [Np, N, H, 3]
        'anchors': anchors,              # [Np, N, 3]
        'used_rollout': [w['used_rollout'] for w in per_window],
    }


def _world_occupancy_slice(scene, origin_xy, z, xy_limit):
    """2-D world-aligned occupancy slice from the scene's global grid.

    Returns (x_edges, y_edges, occ_2d) in plot coordinates (origin_xy
    subtracted), clipped to the ±xy_limit window, or None when the scene
    is missing / the z slice falls outside the grid.  Axis order matches
    query_local_occupancy: occupancy[x, y, z] with llb as the world
    lower-left-back corner.
    """
    if not scene or not {'occu_global', 'unit', 'llb'}.issubset(scene):
        return None
    occupancy = np.asarray(scene['occu_global'], dtype=bool)
    unit = float(scene['unit'])
    llb = np.asarray(scene['llb'], dtype=np.float32)
    nz = occupancy.shape[2]
    z_idx = int(np.floor((z - llb[2]) / unit))
    if z_idx < 0 or z_idx >= nz:
        return None
    nx, ny = occupancy.shape[0], occupancy.shape[1]
    x_lo = max(0, int(np.floor((origin_xy[0] - xy_limit - llb[0]) / unit)))
    x_hi = min(nx, int(np.ceil((origin_xy[0] + xy_limit - llb[0]) / unit)) + 1)
    y_lo = max(0, int(np.floor((origin_xy[1] - xy_limit - llb[1]) / unit)))
    y_hi = min(ny, int(np.ceil((origin_xy[1] + xy_limit - llb[1]) / unit)) + 1)
    if x_lo >= x_hi or y_lo >= y_hi:
        return None
    occ = occupancy[x_lo:x_hi, y_lo:y_hi, z_idx]
    x_edges = llb[0] + np.arange(x_lo, x_hi + 1) * unit - origin_xy[0]
    y_edges = llb[1] + np.arange(y_lo, y_hi + 1) * unit - origin_xy[1]
    return x_edges, y_edges, occ


def _make_segment_figure(viz_data, world_goal_pos, labels=None,
                         stage_idx=None, max_samples: int = 4,
                         scenes=None):
    """One subplot per sample: the whole segment stitched in WORLD frame,
    translated so the segment start (window-0 anchor) is at the origin.

    *world_goal_pos*: [N, 3] tensor of the segment goal position.
    *labels*: list of B str (action label per sample) or None.
    *scenes*: list of B occupancy dicts (occu_global/unit/llb) or None;
    when present, a world-aligned occupancy slice at the segment-start
    height is drawn, clipped by the shared XY limit.

    Per-window timeline markers: black dot = history start (window anchor,
    segment frame k*F); amber triangle = future start / stitch boundary
    (segment frame k*F + H).  Dot -> triangle spans the H-frame history,
    triangle -> next triangle the F-frame future.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    # occupied cells: semi-transparent gray; free cells: fully transparent
    occ_cmap = ListedColormap([(0.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5, 0.4)])

    gt_segment = viz_data['gt_segment']          # [N, T, 3]
    pred_segment = viz_data['pred_segment']      # [N, T, 3]
    hist_world = viz_data['hist_world']          # [Np, N, H, 3]
    anchors = viz_data['anchors']                # [Np, N, 3]
    goals = world_goal_pos.detach().float().cpu()
    count = min(int(gt_segment.shape[0]), max_samples)
    num_primitive = hist_world.shape[0]
    # timeline: window k covers segment frames [k*F, k*F+H) as history and
    # [k*F+H, (k+1)*F+H) as future; stitch boundaries sit at H + k*F
    history_len = int(hist_world.shape[2])
    total_len = int(gt_segment.shape[1])
    future_len = (total_len - history_len) // num_primitive
    assert history_len + future_len * num_primitive == total_len
    future_start_idx = [history_len + pidx * future_len
                        for pidx in range(num_primitive)]
    primitive_end_idx = [
        history_len + future_len * (pidx + 1) - 1
        for pidx in range(num_primitive)
    ]

    # translate every world point so the segment start is the origin
    origin = gt_segment[:, :1]                   # [N, 1, 3] = window-0 anchor
    gt_seg = gt_segment - origin
    pred_seg = pred_segment - origin
    hist = hist_world - origin                   # [Np, N, H, 3] - [N, 1, 3]
    anc = anchors - origin[:, 0]                 # [Np, N, 3]
    goal_xy = goals[:, :2] - origin[:, 0, :2]    # [N, 2]

    # shared symmetric XY limit over the plotted data (2D only, z excluded)
    xy_limit = _compute_xy_limit(
        gt_seg[:count, ..., :2], pred_seg[:count, ..., :2],
        goals_xy=goal_xy[:count, ...],
    )

    cols = min(2, count)
    rows = (count + cols - 1) // cols
    figure = Figure(figsize=(6.0 * cols, 5.0 * rows), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, cols, squeeze=False)

    for axis, sample_idx in zip(axes.flat, range(count)):
        # ── world-aligned occupancy slice at the segment-start height ──
        # The slice is clipped by the shared XY limit — occupancy does not
        # expand the view.
        has_occ = False
        if scenes is not None and sample_idx < len(scenes):
            occ_slice = _world_occupancy_slice(
                scenes[sample_idx],
                origin[sample_idx, 0, :2].numpy(),
                float(origin[sample_idx, 0, 2]),
                xy_limit,
            )
            if occ_slice is not None:
                x_edges, y_edges, occ = occ_slice
                axis.pcolormesh(
                    x_edges, y_edges, occ.T,  # rows = y, cols = x
                    cmap=occ_cmap, vmin=0, vmax=1, shading='flat',
                    zorder=0,
                )
                has_occ = True

        # GT full segment (gray dashed)
        axis.plot(gt_seg[sample_idx, :, 0], gt_seg[sample_idx, :, 1],
                  color='#666666', linewidth=1.5, linestyle='--',
                  label='GT segment', zorder=2)
        # per-primitive used history (light blue dashed)
        for pidx in range(num_primitive):
            axis.plot(hist[pidx, sample_idx, :, 0],
                      hist[pidx, sample_idx, :, 1],
                      color='#56B4E9', linewidth=1.4, linestyle='--',
                      zorder=2)
        # predicted stitched segment (blue solid)
        axis.plot(pred_seg[sample_idx, :, 0], pred_seg[sample_idx, :, 1],
                  color='#0072B2', linewidth=2.2, zorder=3,
                  label='predicted')
        # history-start (anchor) markers (black dots): window k's history
        # spans from its dot to the next amber triangle below
        axis.scatter(anc[:, sample_idx, 0], anc[:, sample_idx, 1],
                     color='#222222', s=22, zorder=6,
                     label='history start (anchor)')
        # future-start markers (amber triangles) at the stitch boundaries:
        # frames H + k*F of the stitched curve.  Dot -> triangle = H-frame
        # history; triangle -> next triangle = F-frame future.
        future_starts = pred_seg[sample_idx, future_start_idx]  # [Np, 3]
        axis.scatter(future_starts[:, 0], future_starts[:, 1],
                     color='#E69F00', marker='v', s=45, zorder=6,
                     label='future start')
        primitive_ends = pred_seg[sample_idx, primitive_end_idx]
        axis.scatter(primitive_ends[:, 0], primitive_ends[:, 1],
                     color='#CC79A7', marker='s', s=40, zorder=6,
                     label='primitive end')
        # start and goal markers
        axis.scatter([0.0], [0.0], color='#222222', s=35, zorder=6,
                     label='start')
        axis.scatter([goal_xy[sample_idx, 0]], [goal_xy[sample_idx, 1]],
                     color='#D55E00', marker='X', s=90, zorder=6,
                     label='goal')

        label = None
        if (labels is not None and sample_idx < len(labels)
                and labels[sample_idx]):
            label = labels[sample_idx]
        title_label = label if label is not None else f'sample {sample_idx}'
        stage_text = '' if stage_idx is None else f' | stage {stage_idx}'
        axis.set_title(f'{title_label}{stage_text}')

        axis.set_xlim(-xy_limit, xy_limit)
        axis.set_ylim(-xy_limit, xy_limit)
        axis.set_aspect('equal')
        axis.set_xlabel('x (m)')
        axis.set_ylabel('y (m)')
        axis.grid(True, color='#DDDDDD', linewidth=0.7)
        if has_occ:
            # QuadMesh is not directly supported by legend — use a proxy patch
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
        f'Full-segment rollout  |  world frame (segment start at origin)  '
        f'|  limit ±{xy_limit:.2f} m'
    )
    return figure


def _build_segment_figure(batch, dataset, vae, denoiser, diffusion,
                          manager, cfg, max_samples: int = 4):
    """Slice the eval batch, run the mirrored rollout, return the figure."""
    with torch.no_grad(), _preserve_rng():
        sliced = _slice_batch_for_viz(batch, max_samples)
        viz_data = _sample_segment_rollout(
            sliced, dataset, vae, denoiser, diffusion, manager, cfg)
    # segment goal: shared (goal_per_primitive=false) vs per-primitive target (true)
    goal_pidx = -1 if bool(cfg.data.goal_per_primitive) else 0
    world_goal_pos = sliced[goal_pidx]['world_goal_pos']
    labels = sliced[0].get('action_label')
    # Pre-scene phase: draw trajectory + goal only, no occupancy overlay.
    # The surface representation is a LOCAL operation on the ego grid, so
    # the world-frame segment figure keeps drawing the raw global occupancy.
    scenes = sliced[0].get('scene') if manager.should_use_scene() else None
    return _make_segment_figure(
        viz_data, world_goal_pos, labels=labels,
        stage_idx=manager.stage_idx, max_samples=max_samples,
        scenes=scenes)


def ddp_setup(requested_device: str = 'cuda'):
    """Initialize DDP environment variables and process group."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    use_cuda = str(requested_device).startswith('cuda')
    if world_size > 1:
        if not use_cuda or not torch.cuda.is_available():
            raise RuntimeError(
                "DAR distributed training requires CUDA/NCCL; use a single "
                "process with device=cpu for a CPU smoke test."
            )
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
    elif use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but no CUDA GPU is available. "
                "Retry with device=cpu for a single-process smoke test."
            )
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


# def _validate_29dof_contract(cfg, datasets, vae, denoiser) -> None:
#     """Fail before DAR training if data, FK, VAE, and denoiser disagree."""
#     if DOF_DIM != 29 or motion_feature_dim != 69:
#         raise RuntimeError(
#             "DAR 29-DoF training requires DOF_DIM=29 and FeatureVersion 3 "
#             f"dimension 69, got DOF_DIM={DOF_DIM}, nfeats={motion_feature_dim}"
#         )
#     if int(cfg.data.nfeats) != motion_feature_dim:
#         raise ValueError(
#             f"data.nfeats={cfg.data.nfeats}, expected {motion_feature_dim}"
#         )

#     for split, dataset in datasets:
#         stats = dataset.statistics
#         stats_dof = int(stats.get('dof_dim', DOF_DIM))
#         stats_nfeats = int(stats.get('nfeats', motion_feature_dim))
#         skeleton_dof = int(dataset.skeleton.fk.num_dof)
#         if stats_dof != DOF_DIM or stats_nfeats != motion_feature_dim:
#             raise ValueError(
#                 f"{split} dataset is not native 29-DoF/69-D: "
#                 f"statistics dof_dim={stats_dof}, nfeats={stats_nfeats}"
#             )
#         if skeleton_dof != DOF_DIM:
#             raise ValueError(
#                 f"{split} skeleton has {skeleton_dof} DoFs, expected {DOF_DIM}"
#             )
#         stats_order = stats.get('dof_order')
#         if stats_order is not None and str(stats_order).lower() != 'mujoco':
#             raise ValueError(
#                 f"{split} dataset uses {stats_order!r} DOF order, expected 'mujoco'"
#             )
#         stats_names = stats.get('dof_names')
#         if (stats_names is not None
#                 and tuple(stats_names) != G1_MUJOCO_DOF_JOINT_NAMES):
#             raise ValueError(
#                 f"{split} dataset DOF names do not match the G1 MuJoCo order"
#             )
#         if tuple(dataset.skeleton.fk.dof_joint_names) != G1_MUJOCO_DOF_JOINT_NAMES:
#             raise ValueError(
#                 f"{split} MJCF joint order does not match the training contract"
#             )
#         if tuple(dataset.skeleton.fk.body_names[1:]) != G1_MUJOCO_DOF_LINK_NAMES:
#             raise ValueError(
#                 f"{split} MJCF body order does not match the training contract"
#             )
#         if (
#             dataset.mean.shape[-1] != motion_feature_dim
#             or dataset.std.shape[-1] != motion_feature_dim
#         ):
#             raise ValueError(
#                 f"{split} normalization has shape mean={tuple(dataset.mean.shape)}, "
#                 f"std={tuple(dataset.std.shape)}; expected {motion_feature_dim}"
#             )
#         hand_names = tuple(
#             dataset.skeleton.fk.body_names_augment[idx]
#             for idx in dataset.skeleton.hand_id
#         )
#         if hand_names != ('left_hand_link', 'right_hand_link'):
#             raise ValueError(
#                 "Body goals must use the left/right palm-center extensions, got "
#                 f"{hand_names}"
#             )

#     if vae.skel_embedding.in_features != motion_feature_dim:
#         raise ValueError(
#             f"VAE encoder expects {vae.skel_embedding.in_features} features, "
#             f"expected {motion_feature_dim}"
#         )
#     if vae.final_layer.out_features != motion_feature_dim:
#         raise ValueError(
#             f"VAE decoder emits {vae.final_layer.out_features} features, "
#             f"expected {motion_feature_dim}"
#         )

#     expected_history_shape = (int(cfg.data.history_len), motion_feature_dim)
#     actual_history_shape = tuple(int(dim) for dim in denoiser.history_shape)
#     if actual_history_shape != expected_history_shape:
#         raise ValueError(
#             f"Denoiser history_shape={actual_history_shape}, expected "
#             f"{expected_history_shape}"
#         )


def _validate_batch(batch, cfg) -> None:
    num_primitive = int(cfg.data.num_primitive)
    context_len = int(cfg.data.history_len) + int(cfg.data.future_len)
    nfeats = int(cfg.data.nfeats)
    goal_type = GoalType.parse(cfg.data.goal_type)
    goal_encoding = GoalEncoding.parse(
        cfg.data.get('goal_encoding', GoalEncoding.LEGACY40)
    )
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
        if goal_type.uses_keypoints:
            keypoints = primitive.get('world_goal_keypoints')
            num_keypoints = 4 if goal_type is GoalType.BODY_EXT else 5
            if keypoints is None or keypoints.shape[-2:] != (num_keypoints, 3):
                shape = None if keypoints is None else tuple(keypoints.shape)
                raise ValueError(
                    f"Primitive {primitive_idx} body goal keypoints have shape "
                    f"{shape}, expected [batch, {num_keypoints}, 3]"
                )
        if goal_type is GoalType.JOINT_STATE:
            goal_rot = primitive.get('world_goal_rot')
            goal_dof = primitive.get('world_goal_dof')
            if goal_rot is None or goal_rot.shape[-1:] != (4,):
                shape = None if goal_rot is None else tuple(goal_rot.shape)
                raise ValueError(
                    f"Primitive {primitive_idx} joint_state goal rotation "
                    f"has shape {shape}, expected [batch, 4]"
                )
            if goal_dof is None or goal_dof.shape[-1:] != (29,):
                shape = None if goal_dof is None else tuple(goal_dof.shape)
                raise ValueError(
                    f"Primitive {primitive_idx} joint_state goal dof has "
                    f"shape {shape}, expected [batch, 29]"
                )
            if goal_encoding.uses_end_effectors:
                end_effectors = primitive.get('world_goal_end_effectors')
                if (end_effectors is None
                        or end_effectors.shape[-2:] != (4, 3)):
                    shape = (
                        None if end_effectors is None
                        else tuple(end_effectors.shape)
                    )
                    raise ValueError(
                        f"Primitive {primitive_idx} end-effector goal has "
                        f"shape {shape}, expected [batch, 4, 3]"
                    )
        if goal_type.uses_arrival_time:
            velocity = primitive.get('world_goal_vel')
            time_to_arrival = primitive.get(
                'time_to_arrival', primitive.get('goal_timestep'))
            if velocity is None or velocity.shape[-1:] != (3,):
                shape = None if velocity is None else tuple(velocity.shape)
                raise ValueError(
                    f"Primitive {primitive_idx} goal velocity has shape {shape}, "
                    "expected [batch, 3]"
                )
            if time_to_arrival is None or time_to_arrival.shape[-1:] != (1,):
                shape = None if time_to_arrival is None else tuple(time_to_arrival.shape)
                raise ValueError(
                    f"Primitive {primitive_idx} goal time_to_arrival has shape {shape}, "
                    "expected [batch, 1]"
                )


def _validate_goal_root_position_contract(cfg) -> None:
    """Validate the goal root-position loss timing contract."""
    loss_weight = cfg.train.manager.loss_weight
    weight = max(
        _loss_weight_config_value(loss_weight, 'goal_root_position'),
        _loss_weight_config_value(loss_weight, 'goal_root_position_hor'),
        _loss_weight_config_value(loss_weight, 'goal_root_position_vert'),
    )
    if weight <= 0.0:
        return

    offset_range = cfg.data.get('goal_offset_range')
    if offset_range is None:
        offsets = (int(cfg.data.get('goal_offset', 0)),) * 2
    else:
        offsets = tuple(int(value) for value in offset_range)
    goal_type = GoalType.parse(cfg.data.goal_type)
    goal_timestep_mode = str(cfg.data.get(
        'goal_timestep_mode', cfg.data.get('time_to_arrival_mode', 'relative'))
    ).lower()
    if offsets != (0, 0):
        if not goal_type.uses_arrival_time:
            raise ValueError(
                "goal_root_position loss with randomized offsets requires "
                "a goal type with explicit arrival time"
            )
        if goal_timestep_mode != 'relative':
            raise ValueError(
                "goal_root_position loss with randomized offsets requires "
                "goal_timestep_mode=relative"
            )
    if not bool(cfg.data.goal_per_primitive):
        if not goal_type.uses_arrival_time:
            raise ValueError(
                "goal_root_position loss with goal_per_primitive=false requires "
                "a goal type with explicit arrival time"
            )
        if goal_timestep_mode != 'relative':
            raise ValueError(
                "goal_root_position loss with goal_per_primitive=false requires "
                "goal_timestep_mode=relative"
            )


def _validate_joint_state_contract(cfg) -> None:
    """Validate v6 joint_state and randomized-arrival constraints."""
    goal_type = GoalType.parse(cfg.data.goal_type)
    if goal_type is GoalType.JOINT_STATE and int(cfg.data.dof_dim) != 29:
        raise ValueError(
            "goal_type=joint_state requires data.dof_dim=29, got "
            f"{cfg.data.dof_dim}"
        )

    offset_range = cfg.data.get('goal_offset_range')
    if offset_range is None:
        offsets = (int(cfg.data.get('goal_offset', 0)),) * 2
    else:
        offsets = tuple(int(value) for value in offset_range)
    if offsets == (0, 0):
        return
    goal_timestep_mode = str(cfg.data.get(
        'goal_timestep_mode', cfg.data.get('time_to_arrival_mode', 'relative'))
    ).lower()
    if goal_type.uses_arrival_time and goal_timestep_mode != 'relative':
        raise ValueError(
            f"goal_type={goal_type.value} with randomized goal offsets "
            "requires goal_timestep_mode=relative"
        )


def _validate_scene_curriculum_contract(cfg) -> None:
    """Ensure scene payloads are available before scene conditioning turns on."""
    if bool(cfg.data.get('load_scene', True)):
        return

    stages = cfg.train.manager.get('stages', [])
    max_steps = sum(int(value) for value in stages)
    scene_start_step = int(cfg.data.get('scene_start_step', max_steps + 1))
    if scene_start_step <= max_steps:
        raise ValueError(
            "data.load_scene=false strips scene occupancy from the dataset, "
            f"but data.scene_start_step={scene_start_step} enables scene "
            f"conditioning during this run (max_steps={max_steps}). Set "
            "data.load_scene=true for scene training, or set "
            "data.scene_start_step greater than max_steps for a no-scene run."
        )


def main(cfg: DictConfig):
    # Initialize DDP
    requested_device = str(cfg.get('device', 'cuda'))
    rank, world_size, local_rank = ddp_setup(requested_device)
    device = (
        torch.device(f'cuda:{local_rank}')
        if requested_device.startswith('cuda')
        else torch.device(requested_device)
    )

    configure_dof_contract(cfg)
    seed.set(cfg.seed + rank)
    logger = logger_module.set(cfg)

    # Override device in config for downstream components.
    cfg.device = str(device)
    for split in ('train', 'val'):
        if split in cfg.data:
            cfg.data[split].device = str(device)
    _validate_scene_curriculum_contract(cfg)

    train_data: Dataset = instantiate(cfg.data.train)
    goal_encoding = GoalEncoding.parse(
        cfg.data.get('goal_encoding', GoalEncoding.LEGACY40)
    )
    denoiser_goal_encoding = GoalEncoding.parse(
        cfg.denoiser.get('goal_encoding', goal_encoding)
    )
    if denoiser_goal_encoding is not goal_encoding:
        raise ValueError(
            f"data.goal_encoding={goal_encoding.value!r} must match "
            f"denoiser.goal_encoding={denoiser_goal_encoding.value!r}"
        )
    validate_goal_config(
        cfg.data.goal_type,
        cfg.denoiser.goal_dim,
        goal_encoding,
        dof_dim=cfg.data.dof_dim,
        goal_offset_range=cfg.data.goal_offset_range,
        goal_timestep_mode=cfg.data.goal_timestep_mode,
        goal_stats=getattr(train_data, 'goal_stats', None),
    )
    _validate_joint_state_contract(cfg)
    _validate_goal_root_position_contract(cfg)
    if cfg.train.manager.use_static_pose:
        raise ValueError(
            "Static-pose replacement has no world reference pose and is not "
            "supported by goal+scene training"
        )

    # goal_direction and goal_position losses work with all goal types:
    # - ROOT: ego_goal[..., :2] is [ego_root_x, ego_root_y]
    # - BODY: ego_goal[..., :2] is the first body keypoint (root joint, index 0)
    # - BODY_EXT: ego_goal[..., :2] is [ego_root_x, ego_root_y] (prepended)

    val_data: Dataset = instantiate(cfg.data.val)

    # Set rank/world_size on datasets for distributed data sharding
    train_data.rank = rank
    train_data.world_size = world_size
    val_data.rank = rank
    val_data.world_size = world_size

    vae: VAE = instantiate(cfg.vae)
    denoiser: Denoiser = instantiate(cfg.denoiser)
    if is_main_process():
        logger.info(
            'Condition sampling config: enabled_text={}, cardinality={}, '
            'time_probability={}',
            bool(cfg.denoiser.get('text_condition_enabled', False)),
            cfg.denoiser.get('condition_sampling', {}).get('cardinality', {}),
            cfg.denoiser.get('condition_sampling', {}).get(
                'time_probability', 0.0),
        )
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

    validate_training_contract(
        cfg, [('train', train_data), ('val', val_data)], vae, denoiser_raw
    )

    schedule_sampler: SSampler = instantiate(cfg.diffusion.schedule_sampler)
    diffusion: Diffusion = schedule_sampler.diffusion

    optimizer: Optimizer = torch.optim.AdamW(
        denoiser.parameters(), lr=cfg.train.manager.learning_rate)

    # scene_start_step lives in the data config (data/mob.yaml); hand it to
    # the manager explicitly since it is no longer part of train.manager.
    manager: DARManager = instantiate(
        cfg.train.manager, scene_start_step=cfg.data.scene_start_step)

    manager.hold_model(vae, denoiser, optimizer, train_data)
    manager.rank = rank
    manager.world_size = world_size
    train_condition_plan = _build_train_condition_plan(
        cfg, denoiser_raw)

    num_primitive: int = cfg.data.num_primitive
    future_len: int = cfg.data.future_len
    history_len: int = cfg.data.history_len

    train_dataiter = iter(train_data)
    # The custom dataset builds a whole segment synchronously.  When history
    # augmentation is active, keep the step-dependent generation synchronous;
    # otherwise overlap the next segment's CPU preparation with GPU training.
    train_data_prefetch = None
    if not bool(getattr(train_data, 'augmentation_enabled', False)):
        train_data_prefetch = _BackgroundPrefetchIterator(train_dataiter)
        train_dataiter = train_data_prefetch
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

        batch_text_embeddings = _prepare_batch_text_embeddings(
            batch, cfg, device, optimize_always_dropped=True)
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
            manager.extra['self_rollout_used'] = float(used_rollout)

            if used_rollout:
                if any(value is None for value in (
                    rollout_history_start_pos,
                    rollout_history_start_rot,
                    rollout_ref_pos,
                    rollout_ref_rot,
                )):
                    raise RuntimeError(
                        "Self-rollout history selected before rollout poses "
                        "were initialized")
                history_start_pos = rollout_history_start_pos
                history_start_rot = rollout_history_start_rot
                reference_pos = rollout_ref_pos
                reference_rot = rollout_ref_rot
            else:
                history_start_pos = primitive['history_start_pos'].to(cfg.device)
                history_start_rot = primitive['history_start_rot'].to(cfg.device)
                reference_pos = primitive['gt_ref_pos'].to(cfg.device)
                reference_rot = primitive['gt_ref_rot'].to(cfg.device)

            gt_reference_pos = primitive['gt_ref_pos'].to(cfg.device)
            if used_rollout:
                manager.extra['self_rollout_ref_gt_dist_m'] = (
                    torch.linalg.vector_norm(
                        (reference_pos - gt_reference_pos).detach(),
                        dim=-1,
                    ).mean()
                )
            else:
                manager.extra['self_rollout_ref_gt_dist_m'] = 0.0

            y = _conditions(primitive, reference_pos, reference_rot,
                            history_motion, cfg, train_data.fps,
                            goal_stats=getattr(train_data, 'goal_stats', None),
                            use_scene=manager.should_use_scene(),
                            text_embedding=(
                                batch_text_embeddings[pidx]
                                if batch_text_embeddings is not None else None
                            ),
                            condition_plan=train_condition_plan,
                            )
            goal_time_frame = _goal_time_frame_for_loss(y, cfg)

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
                ego_goal=y['ego_goal_raw'],
                goal_type=cfg.data.goal_type,
                goal_condition_keep_mask=y.get('goal_condition_keep_mask'),
                goal_position_hor_condition_keep_mask=y.get(
                    'goal_position_hor_condition_keep_mask'),
                goal_position_vert_condition_keep_mask=y.get(
                    'goal_position_vert_condition_keep_mask'),
                goal_orientation_condition_keep_mask=y.get(
                    'goal_orientation_condition_keep_mask'),
                goal_gravity_condition_keep_mask=y.get(
                    'goal_gravity_condition_keep_mask'),
                goal_joint_condition_keep_mask=y.get(
                    'goal_joint_condition_keep_mask'),
                goal_velocity_condition_keep_mask=y.get(
                    'goal_velocity_condition_keep_mask'),
                goal_end_effector_condition_keep_mask=y.get(
                    'goal_end_effector_condition_keep_mask'),
                goal_reference_pos=y.get('goal_reference_pos_world'),
                goal_reference_rot=y.get('goal_reference_rot_world'),
                goal_time_frame=goal_time_frame,
                goal_condition_enabled=train_condition_plan,
                action_label=batch[pidx].get('action_label'),
                is_recovery=batch[pidx].get('is_recovery'),
            )
            _add_batch_data_diagnostics(extras, primitive, y)
            loss = loss_dict['total']

            optimizer.zero_grad()
            loss.backward()
            if manager.clip_grad_and_check(denoiser):
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
                loss_dict=loss_dict,
                extras=extras,
            )
            if not manager:
                break

        # Validation loop
        denoiser.eval()
        manager.begin_eval_cycle()
        while manager.should_eval():
            batch = next(val_dataiter)
            if not val_batch_validated:
                _validate_batch(batch, cfg)
                val_batch_validated = True
            batch_text_embeddings = _prepare_batch_text_embeddings(
                batch, cfg, device)
            for pidx in range(num_primitive):
                if not manager.should_eval():
                    break
                manager.pre_step(is_eval=True)
                primitive = batch[pidx]
                motion = primitive['motion'].to(cfg.device)

                future_motion_gt = motion[:, -future_len:, :]
                sliding_mask = batch[pidx]['sliding_mask'].to(
                    cfg.device)[:, -future_len:, :]
                history_motion = motion[:, :history_len, :]
                use_scene = manager.should_use_scene()
                y = _conditions(
                    primitive,
                    primitive['gt_ref_pos'].to(cfg.device),
                    primitive['gt_ref_rot'].to(cfg.device),
                    history_motion,
                    cfg,
                    val_data.fps,
                    goal_stats=getattr(val_data, 'goal_stats', None),
                    use_scene=use_scene,
                    text_embedding=(
                        batch_text_embeddings[pidx]
                        if batch_text_embeddings is not None else None
                    ),
                )
                goal_time_frame = _goal_time_frame_for_loss(y, cfg)

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

                    x_start_pred = denoiser_raw(
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
                        ego_goal=y['ego_goal_raw'],
                        goal_condition_keep_mask=y.get('goal_condition_keep_mask'),
                        goal_type=cfg.data.goal_type,
                        goal_position_hor_condition_keep_mask=y.get(
                            'goal_position_hor_condition_keep_mask'),
                        goal_position_vert_condition_keep_mask=y.get(
                            'goal_position_vert_condition_keep_mask'),
                        goal_orientation_condition_keep_mask=y.get(
                            'goal_orientation_condition_keep_mask'),
                        goal_gravity_condition_keep_mask=y.get(
                            'goal_gravity_condition_keep_mask'),
                        goal_joint_condition_keep_mask=y.get(
                            'goal_joint_condition_keep_mask'),
                        goal_velocity_condition_keep_mask=y.get(
                            'goal_velocity_condition_keep_mask'),
                        goal_end_effector_condition_keep_mask=y.get(
                            'goal_end_effector_condition_keep_mask'),
                        goal_reference_pos=y.get('goal_reference_pos_world'),
                        goal_reference_rot=y.get('goal_reference_rot_world'),
                        goal_time_frame=goal_time_frame,
                        is_eval=True,
                        action_label=batch[pidx].get('action_label'),
                        is_recovery=batch[pidx].get('is_recovery'))
                    _add_batch_data_diagnostics(extras, primitive, y)

                    if getattr(manager, 'eval_full_sample', False):
                        sample_latent = diffusion.p_sample_loop(
                            denoiser_raw,
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
                        sample_goal_displacement = manager.root_displacement_ego(
                            sample_future, history_motion,
                            goal_time_frame=goal_time_frame)
                        goal_keep_mask = y.get('goal_condition_keep_mask')
                        extras['sample_goal_root_position'] = (
                            manager.calc_goal_root_position_loss(
                                sample_future,
                                y['ego_goal_raw'],
                                goal_keep_mask,
                                history_motion=history_motion,
                                goal_time_frame=goal_time_frame,
                            )
                        )
                        goal_root_target = _raw_goal_root_target(y['ego_goal_raw'])
                        goal_error = torch.linalg.vector_norm(
                            sample_goal_displacement - goal_root_target, dim=-1
                        )
                        primitive_end_error = torch.linalg.vector_norm(
                            sample_displacement - goal_root_target, dim=-1
                        )
                        if goal_keep_mask is not None:
                            goal_error = goal_error[
                                goal_keep_mask.to(dtype=torch.bool)
                            ]
                            primitive_end_error = primitive_end_error[
                                goal_keep_mask.to(dtype=torch.bool)
                            ]
                        if goal_error.numel() > 0:
                            extras['sample_goal_error_m'] = goal_error.mean()
                        if primitive_end_error.numel() > 0:
                            extras['sample_endpoint_error_m'] = (
                                primitive_end_error.mean()
                            )
                        extras['sample_root_displacement'] = (
                            sample_displacement.norm(dim=-1).mean()
                        )
                        extras['sample_goal_root_displacement'] = (
                            sample_goal_displacement.norm(dim=-1).mean()
                        )
                        extras['goal_root_displacement'] = (
                            goal_root_target.norm(dim=-1).mean()
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
                                _raw_goal_xy(y['ego_goal_raw']),
                                ground_truth_trajectory=ground_truth_trajectory,
                                goal_time_frame=goal_time_frame,
                                history_trajectory=hist_traj,
                                goal_condition_keep_mask=goal_keep_mask,
                                # Pre-scene phase: draw trajectory + goal only
                                voxel=y['voxel'] if use_scene else None,
                                grid_size=cfg.denoiser.grid_size,
                                grid_unit=cfg.data.occupancy_unit,
                                labels=batch[pidx].get('action_label'),
                            )
                            manager.platform.report_figure(
                                'root_xy_trajectory',
                                figure,
                                manager.step,
                                group_name='eval',
                            )
                            # Full-segment rollout figure: visualization only,
                            # does not feed eval metrics. RNG state is
                            # preserved so later eval steps are unaffected.
                            segment_figure = _build_segment_figure(
                                batch, val_data, vae, denoiser_raw, diffusion,
                                manager, cfg)
                            manager.platform.report_figure(
                                'segment_rollout_trajectory',
                                segment_figure,
                                manager.step,
                                group_name='eval',
                            )

                manager.post_step(
                    is_eval=True,
                    loss_dict=loss_dict,
                    extras=extras,
                )

    if train_data_prefetch is not None:
        train_data_prefetch.close()

    # Clean up DDP resources
    if dist.is_initialized():
        dist.destroy_process_group()
