import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import clip
import loralib as lora

from robotmdar.diffusion.nn import timestep_embedding
from robotmdar.utils.goal import (
    EXTENDED_BODY_GOAL_DIM,
    GoalEncoding,
    JOINT_STATE_GOAL_DIM,
    SPLIT_END_EFFECTOR_GOAL_DIM,
    SPLIT_END_EFFECTOR_SLICE,
    SPLIT_END_EFFECTOR_SUBSLICES,
    SPLIT_END_EFFECTOR_TOKEN_ORDER,
    SPLIT_HORIZONTAL_SLICE,
    SPLIT_HORIZONTAL_URGENCY_SLICE,
    SPLIT_GOAL_DIM,
    SPLIT_GOAL_NO_LOG_DIM,
    SPLIT_END_EFFECTOR_NO_LOG_GOAL_DIM,
    SPLIT_JOINT_SLICE,
    SPLIT_ORIENTATION_SLICE,
    SPLIT_TIME_SLICE,
    SPLIT_VERTICAL_GRAVITY_SLICE,
    SPLIT_VERTICAL_HEIGHT_SLICE,
    SPLIT_VERTICAL_SLICE,
    SPLIT_VERTICAL_URGENCY_SLICE,
    SPLIT_NO_LOG_HORIZONTAL_SLICE,
    SPLIT_NO_LOG_HORIZONTAL_URGENCY_SLICE,
    SPLIT_NO_LOG_VERTICAL_SLICE,
    SPLIT_NO_LOG_VERTICAL_HEIGHT_SLICE,
    SPLIT_NO_LOG_VERTICAL_GRAVITY_SLICE,
    SPLIT_NO_LOG_VERTICAL_URGENCY_SLICE,
    SPLIT_NO_LOG_ORIENTATION_SLICE,
    SPLIT_NO_LOG_JOINT_SLICE,
    SPLIT_NO_LOG_VELOCITY_SLICE,
    SPLIT_NO_LOG_TIME_SLICE,
    SPLIT_NO_LOG_END_EFFECTOR_SUBSLICES,
    SPLIT_NO_LOG_END_EFFECTOR_SLICE,
    SPLIT_VELOCITY_SLICE,
)


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


def _is_mapping_like(value) -> bool:
    return hasattr(value, 'get') and not isinstance(
        value, (str, bytes, int, float, bool))


def _condition_valid_mask(value, batch_size: int, device):
    """Convert an optional scalar or per-sample validity value to a mask."""
    if value is None:
        return None
    mask = torch.as_tensor(value, device=device, dtype=torch.bool).reshape(-1)
    if mask.numel() == 1:
        return mask.expand(batch_size)
    if mask.numel() != batch_size:
        raise ValueError(
            "Condition validity must be scalar or batch-sized, got "
            f"{mask.numel()} values for batch_size={batch_size}"
        )
    return mask


def _condition_recovery_mask(y, batch_size: int, device):
    """Return the per-sample get-up selector used by condition masking."""
    value = y.get('is_recovery')
    if value is None:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    return _condition_valid_mask(value, batch_size, device)


def _condition_mask_probability(model, y, name: str, batch_size: int,
                                device, index=None):
    """Select locomotion/getup dropout probabilities per batch sample."""
    profiles = getattr(model, 'cond_mask_prob_profiles', None)
    if profiles is None:
        value = getattr(model, name)
        if index is not None:
            value = value[index]
        return value

    locomotion = profiles['locomotion'][name]
    getup = profiles['getup'][name]
    if index is not None:
        locomotion = locomotion[index]
        getup = getup[index]
    recovery = _condition_recovery_mask(y, batch_size, device)
    locomotion = torch.as_tensor(
        locomotion, device=device, dtype=torch.float32)
    getup = torch.as_tensor(getup, device=device, dtype=torch.float32)
    if locomotion.numel() != 1 or getup.numel() != 1:
        raise ValueError(
            f"Condition mask probability {name!r} must be scalar after "
            f"indexing, got locomotion={tuple(locomotion.shape)}, "
            f"getup={tuple(getup.shape)}"
        )
    return torch.where(
        recovery,
        getup.reshape(1).expand(batch_size),
        locomotion.reshape(1).expand(batch_size),
    )


def _mask_condition_impl(model, cond, probability, force_mask=False,
                         return_keep_mask=False, valid_mask=None):
    """Apply scalar or per-sample classifier-free condition dropout."""
    batch_size = cond.shape[0]
    device = cond.device
    probabilities = torch.as_tensor(
        probability, device=device, dtype=torch.float32).reshape(-1)
    if probabilities.numel() == 1:
        probabilities = probabilities.expand(batch_size)
    elif probabilities.numel() != batch_size:
        raise ValueError(
            "Condition mask probability must be scalar or batch-sized, got "
            f"{probabilities.numel()} values for batch_size={batch_size}"
        )
    if torch.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError(
            "Condition mask probabilities must be in [0, 1], got "
            f"range [{probabilities.min().item()}, "
            f"{probabilities.max().item()}]"
        )

    forced = _condition_valid_mask(force_mask, batch_size, device)
    if forced is None:
        forced = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Keep the old fast paths: they avoid touching RNG when no sample can be
    # dropped, which matters for reproducible training and existing callers.
    if not forced.any() and valid_mask is None:
        if (model.training and torch.all(probabilities <= 0.0)) or (
                not model.training and torch.all(probabilities < 1.0)):
            keep_mask = torch.ones(
                batch_size, dtype=torch.bool, device=device)
            if return_keep_mask:
                return cond, keep_mask
            return cond

    if model.training:
        drop_mask = torch.bernoulli(probabilities).bool()
    else:
        drop_mask = probabilities >= 1.0
    keep_mask = (~drop_mask) & (~forced)

    # Controller validity is a deployment input. Training uses only the
    # configured stochastic condition-dropout schedule.
    if not model.training and valid_mask is not None:
        keep_mask = keep_mask & _condition_valid_mask(
            valid_mask, batch_size, device)
    masked_cond = cond * keep_mask.unsqueeze(-1)
    if return_keep_mask:
        return masked_cond, keep_mask
    return masked_cond


def _deployment_valid_mask(model, value, batch_size: int, device):
    """Read controller validity only for eval/deployment inference."""
    if model.training:
        return None
    return _condition_valid_mask(value, batch_size, device)


def _goal_valid_mask(y, name: str, batch_size: int, device):
    """Read one structured controller goal-validity flag from ``y``."""
    valid = y.get('goal_valid')
    if valid is None:
        return None

    value = _mapping_get(valid, name, None)
    if value is None and name.startswith('end_effector_'):
        end_effector = _mapping_get(valid, 'end_effector', None)
        if _is_mapping_like(end_effector):
            value = _mapping_get(
                end_effector,
                name[len('end_effector_'):],
                _mapping_get(end_effector, 'all', None),
            )
        elif end_effector is not None:
            value = end_effector
    elif name == 'end_effector' and _is_mapping_like(value):
        value = _mapping_get(value, 'all', None)
    return _condition_valid_mask(value, batch_size, device)


def _model_goal_valid_mask(model, y, name: str, batch_size: int, device):
    if model.training:
        return None
    return _goal_valid_mask(y, name, batch_size, device)


def _combine_valid_masks(*masks):
    """Combine optional validity masks without turning absent flags false."""
    present = [mask for mask in masks if mask is not None]
    if not present:
        return None
    combined = present[0]
    for mask in present[1:]:
        combined = combined & mask
    return combined


def _goal_end_effector_valid_mask(
        model, y, name: str, batch_size: int, device):
    return _combine_valid_masks(
        _model_goal_valid_mask(
            model, y, 'end_effector', batch_size, device),
        _model_goal_valid_mask(
            model, y, f'end_effector_{name}', batch_size, device),
    )


def _split_goal_layout(goal_dim: int):
    if int(goal_dim) in (
            SPLIT_GOAL_NO_LOG_DIM, SPLIT_END_EFFECTOR_NO_LOG_GOAL_DIM):
        return {
            "horizontal": SPLIT_NO_LOG_HORIZONTAL_SLICE,
            "horizontal_urgency": SPLIT_NO_LOG_HORIZONTAL_URGENCY_SLICE,
            "vertical": SPLIT_NO_LOG_VERTICAL_SLICE,
            "vertical_height": SPLIT_NO_LOG_VERTICAL_HEIGHT_SLICE,
            "vertical_gravity": SPLIT_NO_LOG_VERTICAL_GRAVITY_SLICE,
            "vertical_urgency": SPLIT_NO_LOG_VERTICAL_URGENCY_SLICE,
            "orientation": SPLIT_NO_LOG_ORIENTATION_SLICE,
            "joint": SPLIT_NO_LOG_JOINT_SLICE,
            "velocity": SPLIT_NO_LOG_VELOCITY_SLICE,
            "time": SPLIT_NO_LOG_TIME_SLICE,
            "end_effector": SPLIT_NO_LOG_END_EFFECTOR_SLICE,
            "end_effector_subslices": SPLIT_NO_LOG_END_EFFECTOR_SUBSLICES,
        }
    return {
        "horizontal": SPLIT_HORIZONTAL_SLICE,
        "horizontal_urgency": SPLIT_HORIZONTAL_URGENCY_SLICE,
        "vertical": SPLIT_VERTICAL_SLICE,
        "vertical_height": SPLIT_VERTICAL_HEIGHT_SLICE,
        "vertical_gravity": SPLIT_VERTICAL_GRAVITY_SLICE,
        "vertical_urgency": SPLIT_VERTICAL_URGENCY_SLICE,
        "orientation": SPLIT_ORIENTATION_SLICE,
        "joint": SPLIT_JOINT_SLICE,
        "velocity": SPLIT_VELOCITY_SLICE,
        "time": SPLIT_TIME_SLICE,
        "end_effector": SPLIT_END_EFFECTOR_SLICE,
        "end_effector_subslices": SPLIT_END_EFFECTOR_SUBSLICES,
    }


def _nested_mask_value(cond_mask_prob, path):
    value = cond_mask_prob
    for key in path:
        value = _mapping_get(value, key, None)
        if value is None:
            return None
    return value


def _resolve_mask_value(default, *candidates):
    values = [(label, float(value)) for label, value in candidates
              if value is not None]
    if not values:
        return float(default)
    first_label, first_value = values[0]
    for label, value in values[1:]:
        if value != first_value:
            raise ValueError(f"{first_label} and {label} disagree")
    return first_value


def _scalar_nested_mask_value(mapping, path):
    value = _nested_mask_value(mapping, path)
    return None if _is_mapping_like(value) else value


def _merge_mask_mappings(base, override):
    """Recursively merge profile defaults without requiring plain dicts."""
    if not _is_mapping_like(base):
        base = {}
    merged = {key: value for key, value in base.items()}
    if not _is_mapping_like(override):
        return merged
    for key, value in override.items():
        if _is_mapping_like(value) and _is_mapping_like(
                merged.get(key)):
            merged[key] = _merge_mask_mappings(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_single_condition_mask_probs(
    kwargs,
    *,
    cond_mask_prob=None,
    cond_text_mask_prob=None,
    cond_goal_root_mask_prob=None,
    cond_goal_yaw_mask_prob=None,
    cond_goal_time_mask_prob=None,
    cond_goal_body_mask_prob=None,
    cond_goal_orientation_mask_prob=None,
    cond_goal_joint_mask_prob=None,
    cond_goal_velocity_mask_prob=None,
    cond_goal_end_effector_mask_prob=None,
    cond_scene_mask_prob=None,
    scalar_cond_mask_target='text',
):
    """Resolve nested condition-mask config while accepting legacy flat keys."""
    legacy_cond_mask_prob = kwargs.pop('cond_mask_prob', None)
    if cond_mask_prob is None:
        cond_mask_prob = legacy_cond_mask_prob
    elif legacy_cond_mask_prob is not None:
        raise ValueError("cond_mask_prob passed twice")

    nested = cond_mask_prob if _is_mapping_like(cond_mask_prob) else None
    scalar_cond_mask_prob = None if nested is not None else cond_mask_prob
    legacy_goal_mask_prob = kwargs.pop('cond_goal_mask_prob', None)

    scalar_text = (
        scalar_cond_mask_prob
        if scalar_cond_mask_target == 'text' else None
    )
    scalar_position = (
        scalar_cond_mask_prob
        if scalar_cond_mask_target == 'position' else None
    )
    end_effector_parent = _nested_mask_value(
        nested, ('goal', 'end_effector'))
    nested_end_effector_parent = None
    if (end_effector_parent is not None
            and not _is_mapping_like(end_effector_parent)):
        nested_end_effector_parent = end_effector_parent
    end_effector_mask_probs = tuple(
        _resolve_mask_value(
            0.0,
            (f'cond_mask_prob.goal.end_effector.{name}',
             _nested_mask_value(nested, ('goal', 'end_effector', name))),
            ('cond_mask_prob.goal.end_effector',
             nested_end_effector_parent),
            ('cond_goal_end_effector_mask_prob',
             cond_goal_end_effector_mask_prob),
        )
        for name in SPLIT_END_EFFECTOR_TOKEN_ORDER
    )

    position_parent = _scalar_nested_mask_value(
        nested, ('goal', 'position'))
    orientation_parent = _scalar_nested_mask_value(
        nested, ('goal', 'orientation'))
    position_hor = _resolve_mask_value(
        0.1,
        ('cond_mask_prob.goal.position.hor',
         _nested_mask_value(nested, ('goal', 'position', 'hor'))),
        ('cond_mask_prob.goal.position', position_parent),
        ('cond_goal_root_mask_prob', cond_goal_root_mask_prob),
        ('legacy cond_goal_mask_prob', legacy_goal_mask_prob),
        ('legacy cond_mask_prob', scalar_position),
    )
    position_vert = _resolve_mask_value(
        0.1,
        ('cond_mask_prob.goal.position.vert',
         _nested_mask_value(nested, ('goal', 'position', 'vert'))),
        ('cond_mask_prob.goal.position', position_parent),
        ('cond_goal_root_mask_prob', cond_goal_root_mask_prob),
        ('legacy cond_goal_mask_prob', legacy_goal_mask_prob),
        ('legacy cond_mask_prob', scalar_position),
    )
    orientation_rot6d = _resolve_mask_value(
        0.0,
        ('cond_mask_prob.goal.orientation.rot6d',
         _nested_mask_value(nested, ('goal', 'orientation', 'rot6d'))),
        ('cond_mask_prob.goal.orientation', orientation_parent),
        ('cond_goal_orientation_mask_prob',
         cond_goal_orientation_mask_prob),
    )
    gravity = _resolve_mask_value(
        0.0,
        ('cond_mask_prob.goal.orientation.gravity',
         _nested_mask_value(nested, ('goal', 'orientation', 'gravity'))),
        ('cond_mask_prob.goal.orientation', orientation_parent),
        ('cond_goal_orientation_mask_prob',
         cond_goal_orientation_mask_prob),
    )

    return {
        'text': _resolve_mask_value(
            0.0,
            ('cond_mask_prob.text',
             _nested_mask_value(nested, ('text',))),
            ('cond_text_mask_prob', cond_text_mask_prob),
            ('legacy cond_mask_prob', scalar_text),
        ),
        # ``goal_position`` and ``goal_orientation`` retain their old scalar
        # meaning for legacy/single goal layouts. Split layouts use the
        # fine-grained values below.
        'goal_position': position_parent
        if position_parent is not None else position_hor,
        'goal_position_hor': position_hor,
        'goal_position_vert': position_vert,
        'goal_yaw': _resolve_mask_value(
            0.0,
            ('cond_mask_prob.goal.yaw',
             _nested_mask_value(nested, ('goal', 'yaw'))),
            ('cond_goal_yaw_mask_prob', cond_goal_yaw_mask_prob),
        ),
        'goal_time': _resolve_mask_value(
            0.0,
            ('cond_mask_prob.goal.time',
             _nested_mask_value(nested, ('goal', 'time'))),
            ('cond_goal_time_mask_prob', cond_goal_time_mask_prob),
        ),
        'goal_body': _resolve_mask_value(
            0.0,
            ('cond_mask_prob.goal.body',
             _nested_mask_value(nested, ('goal', 'body'))),
            ('cond_goal_body_mask_prob', cond_goal_body_mask_prob),
        ),
        'goal_orientation': orientation_parent
        if orientation_parent is not None else orientation_rot6d,
        'goal_orientation_rot6d': orientation_rot6d,
        'goal_gravity': gravity,
        'goal_joint': _resolve_mask_value(
            0.0,
            ('cond_mask_prob.goal.joint',
             _nested_mask_value(nested, ('goal', 'joint'))),
            ('cond_goal_joint_mask_prob', cond_goal_joint_mask_prob),
        ),
        'goal_velocity': _resolve_mask_value(
            0.0,
            ('cond_mask_prob.goal.velocity',
             _nested_mask_value(nested, ('goal', 'velocity'))),
            ('cond_goal_velocity_mask_prob', cond_goal_velocity_mask_prob),
        ),
        'goal_end_effector': end_effector_mask_probs,
        'scene': _resolve_mask_value(
            0.1,
            ('cond_mask_prob.scene',
             _nested_mask_value(nested, ('scene',))),
            ('cond_scene_mask_prob', cond_scene_mask_prob),
        ),
    }


def _resolve_condition_mask_profiles(
    kwargs,
    *,
    cond_mask_prob=None,
    cond_text_mask_prob=None,
    cond_goal_root_mask_prob=None,
    cond_goal_yaw_mask_prob=None,
    cond_goal_time_mask_prob=None,
    cond_goal_body_mask_prob=None,
    cond_goal_orientation_mask_prob=None,
    cond_goal_joint_mask_prob=None,
    cond_goal_velocity_mask_prob=None,
    cond_goal_end_effector_mask_prob=None,
    cond_scene_mask_prob=None,
):
    """Resolve shared or locomotion/getup condition-dropout profiles."""
    raw = cond_mask_prob
    if raw is None:
        raw = _mapping_get(kwargs, 'cond_mask_prob', None)

    profile_keys = ('locomotion', 'getup')
    has_profiles = _is_mapping_like(raw) and any(
        _mapping_get(raw, key, None) is not None for key in profile_keys)
    if not has_profiles:
        resolved = _resolve_single_condition_mask_probs(
            dict(kwargs),
            cond_mask_prob=cond_mask_prob,
            cond_text_mask_prob=cond_text_mask_prob,
            cond_goal_root_mask_prob=cond_goal_root_mask_prob,
            cond_goal_yaw_mask_prob=cond_goal_yaw_mask_prob,
            cond_goal_time_mask_prob=cond_goal_time_mask_prob,
            cond_goal_body_mask_prob=cond_goal_body_mask_prob,
            cond_goal_orientation_mask_prob=cond_goal_orientation_mask_prob,
            cond_goal_joint_mask_prob=cond_goal_joint_mask_prob,
            cond_goal_velocity_mask_prob=cond_goal_velocity_mask_prob,
            cond_goal_end_effector_mask_prob=(
                cond_goal_end_effector_mask_prob),
            cond_scene_mask_prob=cond_scene_mask_prob,
        )
        return {'locomotion': resolved, 'getup': resolved}

    common = {
        key: value for key, value in raw.items()
        if key not in profile_keys
    }
    locomotion_cfg = _merge_mask_mappings(
        common, _mapping_get(raw, 'locomotion', None))
    getup_cfg = _merge_mask_mappings(
        locomotion_cfg,
        _mapping_get(raw, 'getup', None),
    )
    profiles = {}
    for name, profile_cfg in (
            ('locomotion', locomotion_cfg), ('getup', getup_cfg)):
        profiles[name] = _resolve_single_condition_mask_probs(
            dict(kwargs),
            cond_mask_prob=profile_cfg,
            cond_text_mask_prob=cond_text_mask_prob,
            cond_goal_root_mask_prob=cond_goal_root_mask_prob,
            cond_goal_yaw_mask_prob=cond_goal_yaw_mask_prob,
            cond_goal_time_mask_prob=cond_goal_time_mask_prob,
            cond_goal_body_mask_prob=cond_goal_body_mask_prob,
            cond_goal_orientation_mask_prob=cond_goal_orientation_mask_prob,
            cond_goal_joint_mask_prob=cond_goal_joint_mask_prob,
            cond_goal_velocity_mask_prob=cond_goal_velocity_mask_prob,
            cond_goal_end_effector_mask_prob=(
                cond_goal_end_effector_mask_prob),
            cond_scene_mask_prob=cond_scene_mask_prob,
        )
    return profiles


def _joint_state_goal_slices(goal_encoding: GoalEncoding):
    if goal_encoding is GoalEncoding.LEGACY40:
        return slice(0, 3), slice(3, 8), slice(8, 37), slice(37, 40), None
    return (
        SPLIT_HORIZONTAL_SLICE,
        SPLIT_ORIENTATION_SLICE,
        SPLIT_JOINT_SLICE,
        SPLIT_VELOCITY_SLICE,
        SPLIT_TIME_SLICE,
    )


def _uses_split_goal_tokens(goal_encoding: GoalEncoding) -> bool:
    return goal_encoding in (
        GoalEncoding.SPLIT,
        GoalEncoding.SPLIT_END_EFFECTOR,
    )


def _uses_split_goal_masking(goal_encoding: GoalEncoding) -> bool:
    return goal_encoding in (
        GoalEncoding.SINGLE,
        GoalEncoding.SPLIT,
        GoalEncoding.SPLIT_END_EFFECTOR,
    )


def _force_drop_end_effector(y, name: str) -> bool:
    if y.get('force_drop_goal_end_effector', False):
        return True
    if y.get(f'force_drop_goal_end_effector_{name}', False):
        return True
    nested_goal = y.get('force_drop_goal')
    nested_end_effector = _mapping_get(nested_goal, 'end_effector', None)
    if nested_end_effector is None:
        return False
    if _is_mapping_like(nested_end_effector):
        return bool(
            _mapping_get(nested_end_effector, 'all', False)
            or _mapping_get(nested_end_effector, name, False)
        )
    return bool(nested_end_effector)


def _mask_split_goal(model, goal, y):
    """Mask the 55-D/67-D split goal while preserving component semantics."""
    layout = _split_goal_layout(goal.shape[-1])
    batch_size = goal.shape[0]
    device = goal.device
    root_valid = _model_goal_valid_mask(
        model, y, 'root', batch_size, device)
    orientation_valid = _combine_valid_masks(
        _model_goal_valid_mask(
            model, y, 'orientation', batch_size, device),
        _model_goal_valid_mask(model, y, 'yaw', batch_size, device),
    )
    joint_valid = _model_goal_valid_mask(
        model, y, 'joint', batch_size, device)
    velocity_valid = _model_goal_valid_mask(
        model, y, 'velocity', batch_size, device)
    horizontal_content = slice(
        layout["horizontal"].start,
        layout["horizontal_urgency"].start,
    )
    force_position_hor = (
        y.get('force_drop_goal_position_hor', False)
        or y.get('force_drop_goal_root', False)
    )
    force_position_vert = (
        y.get('force_drop_goal_position_vert', False)
        or y.get('force_drop_goal_root', False)
    )
    force_gravity = (
        y.get('force_drop_goal_gravity', False)
        or y.get('force_drop_goal_orientation', False)
        or y.get('force_drop_goal_yaw', False)
    )
    force_orientation = (
        y.get('force_drop_goal_orientation_rot6d', False)
        or y.get('force_drop_goal_orientation', False)
        or y.get('force_drop_goal_yaw', False)
    )
    horizontal, horizontal_keep = model.mask_condition(
        goal[:, horizontal_content],
        _condition_mask_probability(
            model, y, 'goal_position_hor', batch_size, device),
        force_mask=force_position_hor,
        valid_mask=root_valid,
        return_keep_mask=True,
    )
    vertical_height, vertical_keep = model.mask_condition(
        goal[:, layout["vertical_height"]],
        _condition_mask_probability(
            model, y, 'goal_position_vert', batch_size, device),
        force_mask=force_position_vert,
        valid_mask=root_valid,
        return_keep_mask=True,
    )
    gravity, gravity_keep = model.mask_condition(
        goal[:, layout["vertical_gravity"]],
        _condition_mask_probability(
            model, y, 'goal_gravity', batch_size, device),
        force_mask=force_gravity,
        valid_mask=orientation_valid,
        return_keep_mask=True,
    )
    rot, orientation_keep = model.mask_condition(
        goal[:, layout["orientation"]],
        _condition_mask_probability(
            model, y, 'goal_orientation_rot6d', batch_size, device),
        force_mask=force_orientation,
        valid_mask=orientation_valid,
        return_keep_mask=True,
    )
    joints, joint_keep = model.mask_condition(
        goal[:, layout["joint"]],
        _condition_mask_probability(
            model, y, 'goal_joint', batch_size, device),
        force_mask=y.get('force_drop_goal_joint', False),
        valid_mask=joint_valid,
        return_keep_mask=True,
    )
    velocity, velocity_keep = model.mask_condition(
        goal[:, layout["velocity"]],
        _condition_mask_probability(
            model, y, 'goal_velocity', batch_size, device),
        force_mask=y.get('force_drop_goal_velocity', False),
        valid_mask=velocity_valid,
        return_keep_mask=True,
    )

    masked = goal.clone()
    horizontal_keep_f = horizontal_keep.unsqueeze(-1).to(masked.dtype)
    vertical_keep_f = vertical_keep.unsqueeze(-1).to(masked.dtype)
    masked[:, horizontal_content] = horizontal
    masked[:, layout["vertical_height"]] = (
        vertical_height)
    masked[:, layout["vertical_gravity"]] = (
        gravity)
    masked[:, layout["orientation"]] = rot
    masked[:, layout["joint"]] = joints
    masked[:, layout["velocity"]] = velocity

    y['goal_position_hor_condition_keep_mask'] = horizontal_keep
    y['goal_position_vert_condition_keep_mask'] = vertical_keep
    y['goal_gravity_condition_keep_mask'] = gravity_keep
    y['goal_orientation_condition_keep_mask'] = orientation_keep
    y['goal_joint_condition_keep_mask'] = joint_keep
    y['goal_velocity_condition_keep_mask'] = velocity_keep
    y['goal_vertical_condition_keep_mask'] = vertical_keep | gravity_keep
    if goal.shape[-1] in (
            SPLIT_END_EFFECTOR_GOAL_DIM,
            SPLIT_END_EFFECTOR_NO_LOG_GOAL_DIM):
        end_effector_keeps = []
        for idx, name in enumerate(SPLIT_END_EFFECTOR_TOKEN_ORDER):
            end_effector, keep = model.mask_condition(
                goal[:, layout["end_effector_subslices"][name]],
                _condition_mask_probability(
                    model, y, 'goal_end_effector', batch_size, device,
                    index=idx),
                force_mask=_force_drop_end_effector(y, name),
                valid_mask=_goal_end_effector_valid_mask(
                    model, y, name, batch_size, device),
                return_keep_mask=True,
            )
            masked[:, layout["end_effector_subslices"][name]] = end_effector
            end_effector_keeps.append(keep)
        y['goal_end_effector_condition_keep_mask'] = torch.stack(
            end_effector_keeps, dim=1)
    time_valid = _model_goal_valid_mask(
        model, y, 'time', batch_size, device)
    if 'arrival_time_condition_keep_mask' in y:
        time_keep = y['arrival_time_condition_keep_mask']
    elif y.get('force_drop_goal_time', False) or y.get(
            'force_drop_arrival_time', False):
        time_keep = torch.zeros(
            goal.shape[0], dtype=torch.bool, device=goal.device)
    elif time_valid is not None:
        time_keep = time_valid
    else:
        time_keep = torch.ones(
            goal.shape[0], dtype=torch.bool, device=goal.device)
    time_keep_f = time_keep.unsqueeze(-1).to(masked.dtype)
    masked[:, layout["horizontal_urgency"]] = (
        masked[:, layout["horizontal_urgency"]]
        * horizontal_keep_f * time_keep_f)
    masked[:, layout["vertical_urgency"]] = (
        masked[:, layout["vertical_urgency"]]
        * vertical_keep_f * time_keep_f)
    masked[:, layout["time"]] = masked[:, layout["time"]] * time_keep_f
    y['goal_time_condition_keep_mask'] = time_keep
    return masked, horizontal_keep


def _mask_legacy_split_goal(model, goal, y):
    """Mask the early heading-free 55-D goal layout."""
    batch_size = goal.shape[0]
    device = goal.device
    root_valid = _model_goal_valid_mask(
        model, y, 'root', batch_size, device)
    orientation_valid = _combine_valid_masks(
        _model_goal_valid_mask(
            model, y, 'orientation', batch_size, device),
        _model_goal_valid_mask(model, y, 'yaw', batch_size, device),
    )
    joint_valid = _model_goal_valid_mask(
        model, y, 'joint', batch_size, device)
    velocity_valid = _model_goal_valid_mask(
        model, y, 'velocity', batch_size, device)
    trans, root_keep = model.mask_condition(
        goal[:, 0:12],
        _condition_mask_probability(
            model, y, 'goal_position', batch_size, device),
        force_mask=y.get('force_drop_goal_root', False),
        valid_mask=root_valid,
        return_keep_mask=True,
    )
    rot, orientation_keep = model.mask_condition(
        goal[:, 12:21],
        _condition_mask_probability(
            model, y, 'goal_orientation', batch_size, device),
        force_mask=(
            y.get('force_drop_goal_orientation', False)
            or y.get('force_drop_goal_yaw', False)),
        valid_mask=orientation_valid,
        return_keep_mask=True,
    )
    joints, joint_keep = model.mask_condition(
        goal[:, 21:50],
        _condition_mask_probability(
            model, y, 'goal_joint', batch_size, device),
        force_mask=y.get('force_drop_goal_joint', False),
        valid_mask=joint_valid,
        return_keep_mask=True,
    )
    velocity, velocity_keep = model.mask_condition(
        goal[:, 50:54],
        _condition_mask_probability(
            model, y, 'goal_velocity', batch_size, device),
        force_mask=y.get('force_drop_goal_velocity', False),
        valid_mask=velocity_valid,
        return_keep_mask=True,
    )
    masked = goal.clone()
    masked[:, 0:12] = trans
    masked[:, 12:21] = rot
    masked[:, 21:50] = joints
    masked[:, 50:54] = velocity
    time_valid = _model_goal_valid_mask(
        model, y, 'time', batch_size, device)
    if 'arrival_time_condition_keep_mask' in y:
        time_keep = y['arrival_time_condition_keep_mask']
    elif y.get('force_drop_goal_time', False) or y.get(
            'force_drop_arrival_time', False):
        time_keep = torch.zeros(
            batch_size, dtype=torch.bool, device=device)
    elif time_valid is not None:
        time_keep = time_valid
    else:
        time_keep = torch.ones(
            batch_size, dtype=torch.bool, device=device)
    masked[:, 7:12] = masked[:, 7:12] * time_keep.unsqueeze(-1).to(
        masked.dtype)
    masked[:, 54:55] = masked[:, 54:55] * time_keep.unsqueeze(-1).to(
        masked.dtype)
    y['goal_orientation_condition_keep_mask'] = orientation_keep
    y['goal_position_hor_condition_keep_mask'] = root_keep
    y['goal_position_vert_condition_keep_mask'] = root_keep
    y['goal_gravity_condition_keep_mask'] = orientation_keep
    y['goal_joint_condition_keep_mask'] = joint_keep
    y['goal_velocity_condition_keep_mask'] = velocity_keep
    y['goal_vertical_condition_keep_mask'] = root_keep | orientation_keep
    y['goal_time_condition_keep_mask'] = time_keep
    return masked, root_keep


def _validate_model_goal_encoding(goal_dim: int,
                                  goal_encoding: GoalEncoding) -> None:
    if int(goal_dim) == JOINT_STATE_GOAL_DIM and goal_encoding is not GoalEncoding.LEGACY40:
        raise ValueError(
            "goal_dim=40 requires goal_encoding='legacy40'")
    if int(goal_dim) in (SPLIT_GOAL_DIM, SPLIT_GOAL_NO_LOG_DIM) and goal_encoding is GoalEncoding.LEGACY40:
        raise ValueError(
            f"goal_dim={SPLIT_GOAL_DIM} requires "
            "goal_encoding='single' or 'split'")
    if int(goal_dim) in (
            SPLIT_END_EFFECTOR_GOAL_DIM,
            SPLIT_END_EFFECTOR_NO_LOG_GOAL_DIM,
    ) and goal_encoding is not GoalEncoding.SPLIT_END_EFFECTOR:
        raise ValueError(
            f"goal_dim={SPLIT_END_EFFECTOR_GOAL_DIM} requires "
            "goal_encoding='split_end_effector'")
    if goal_encoding in (GoalEncoding.SINGLE, GoalEncoding.SPLIT) and int(goal_dim) not in (
            SPLIT_GOAL_DIM, SPLIT_GOAL_NO_LOG_DIM):
        raise ValueError(
            "goal_encoding='single' or 'split' requires "
            f"goal_dim={SPLIT_GOAL_DIM} or {SPLIT_GOAL_NO_LOG_DIM}")
    if (goal_encoding is GoalEncoding.SPLIT_END_EFFECTOR
            and int(goal_dim) not in (
                SPLIT_END_EFFECTOR_GOAL_DIM,
                SPLIT_END_EFFECTOR_NO_LOG_GOAL_DIM,
            )):
        raise ValueError(
            "goal_encoding='split_end_effector' requires "
            f"goal_dim={SPLIT_END_EFFECTOR_GOAL_DIM} or "
            f"{SPLIT_END_EFFECTOR_NO_LOG_GOAL_DIM}")


def _mask_goal(model, goal, y):
    """Mask V4 components independently while preserving legacy goal behavior."""
    if int(model.goal_dim) in (
            JOINT_STATE_GOAL_DIM,
            SPLIT_GOAL_DIM,
            SPLIT_GOAL_NO_LOG_DIM,
            SPLIT_END_EFFECTOR_GOAL_DIM,
            SPLIT_END_EFFECTOR_NO_LOG_GOAL_DIM,
    ):
        goal_encoding = getattr(
            model, "goal_encoding", GoalEncoding.LEGACY40)
        if (goal_encoding is GoalEncoding.SPLIT
                and getattr(model, 'legacy_split_goal_layout', False)):
            return _mask_legacy_split_goal(model, goal, y)
        if _uses_split_goal_masking(goal_encoding):
            return _mask_split_goal(model, goal, y)
        (root_slice, orientation_slice, joint_slice,
         velocity_slice, time_slice) = _joint_state_goal_slices(goal_encoding)
        batch_size = goal.shape[0]
        device = goal.device
        root_valid = _model_goal_valid_mask(
            model, y, 'root', batch_size, device)
        orientation_valid = _combine_valid_masks(
            _model_goal_valid_mask(
                model, y, 'orientation', batch_size, device),
            _model_goal_valid_mask(model, y, 'yaw', batch_size, device),
        )
        joint_valid = _model_goal_valid_mask(
            model, y, 'joint', batch_size, device)
        velocity_valid = _model_goal_valid_mask(
            model, y, 'velocity', batch_size, device)
        root, root_keep = model.mask_condition(
            goal[:, root_slice],
            _condition_mask_probability(
                model, y, 'goal_position', batch_size, device),
            force_mask=(
                y.get('force_drop_goal_root', False)
                or y.get('force_drop_goal_position_hor', False)
                or y.get('force_drop_goal_position_vert', False)
            ),
            valid_mask=root_valid,
            return_keep_mask=True,
        )
        orientation, orientation_keep = model.mask_condition(
            goal[:, orientation_slice],
            _condition_mask_probability(
                model, y, 'goal_orientation', batch_size, device),
            force_mask=(
                y.get('force_drop_goal_orientation', False)
                or y.get('force_drop_goal_orientation_rot6d', False)
                or y.get('force_drop_goal_yaw', False)),
            valid_mask=orientation_valid,
            return_keep_mask=True,
        )
        joints, joint_keep = model.mask_condition(
            goal[:, joint_slice],
            _condition_mask_probability(
                model, y, 'goal_joint', batch_size, device),
            force_mask=y.get('force_drop_goal_joint', False),
            valid_mask=joint_valid,
            return_keep_mask=True,
        )
        velocity, velocity_keep = model.mask_condition(
            goal[:, velocity_slice],
            _condition_mask_probability(
                model, y, 'goal_velocity', batch_size, device),
            force_mask=y.get('force_drop_goal_velocity', False),
            valid_mask=velocity_valid,
            return_keep_mask=True,
        )
        y['goal_orientation_condition_keep_mask'] = orientation_keep
        y['goal_position_hor_condition_keep_mask'] = root_keep
        y['goal_position_vert_condition_keep_mask'] = root_keep
        y['goal_gravity_condition_keep_mask'] = orientation_keep
        y['goal_joint_condition_keep_mask'] = joint_keep
        y['goal_velocity_condition_keep_mask'] = velocity_keep
        masked = goal.clone()
        masked[:, root_slice] = root
        masked[:, orientation_slice] = orientation
        masked[:, joint_slice] = joints
        masked[:, velocity_slice] = velocity
        if time_slice is not None:
            y['goal_time_condition_keep_mask'] = y.get(
                'arrival_time_condition_keep_mask',
                torch.ones(goal.shape[0], dtype=torch.bool, device=goal.device),
            )
        return masked, root_keep

    if model.goal_dim != EXTENDED_BODY_GOAL_DIM:
        return model.mask_condition(
            goal,
            _condition_mask_probability(
                model, y, 'goal_position', goal.shape[0], goal.device),
            force_mask=y.get(
                'force_drop_goal_root', y.get('force_drop_goal', False)
            ),
            return_keep_mask=True,
        )

    root, root_keep = model.mask_condition(
        goal[:, 0:3],
        _condition_mask_probability(
            model, y, 'goal_position', goal.shape[0], goal.device),
        force_mask=y.get('force_drop_goal_root', False),
        valid_mask=_model_goal_valid_mask(
            model, y, 'root', goal.shape[0], goal.device),
        return_keep_mask=True,
    )
    yaw, yaw_keep = model.mask_condition(
        goal[:, 3:5],
        _condition_mask_probability(
            model, y, 'goal_yaw', goal.shape[0], goal.device),
        force_mask=y.get('force_drop_goal_yaw', False),
        valid_mask=_model_goal_valid_mask(
            model, y, 'yaw', goal.shape[0], goal.device),
        return_keep_mask=True,
    )
    velocity = goal[:, 5:8]
    goal_time, time_keep = model.mask_condition(
        goal[:, 8:9],
        _condition_mask_probability(
            model, y, 'goal_time', goal.shape[0], goal.device),
        force_mask=(
            y.get('force_drop_goal_time', False)
            or y.get('force_drop_arrival_time', False)),
        valid_mask=_model_goal_valid_mask(
            model, y, 'time', goal.shape[0], goal.device),
        return_keep_mask=True,
    )
    limbs, body_keep = model.mask_condition(
        goal[:, 9:21],
        _condition_mask_probability(
            model, y, 'goal_body', goal.shape[0], goal.device),
        force_mask=y.get('force_drop_goal_body', False),
        valid_mask=_model_goal_valid_mask(
            model, y, 'body', goal.shape[0], goal.device),
        return_keep_mask=True,
    )
    y['goal_yaw_condition_keep_mask'] = yaw_keep
    y['goal_time_condition_keep_mask'] = time_keep
    y['goal_body_condition_keep_mask'] = body_keep
    return torch.cat((root, yaw, velocity, goal_time, limbs), dim=-1), root_keep


def _goal_dim_uses_arrival_pe(goal_dim: int) -> bool:
    return int(goal_dim) in (
        EXTENDED_BODY_GOAL_DIM,
        JOINT_STATE_GOAL_DIM,
        SPLIT_GOAL_DIM,
        SPLIT_GOAL_NO_LOG_DIM,
        SPLIT_END_EFFECTOR_GOAL_DIM,
        SPLIT_END_EFFECTOR_NO_LOG_GOAL_DIM,
    )


def _goal_dim_name(goal_dim: int) -> str:
    if int(goal_dim) == EXTENDED_BODY_GOAL_DIM:
        return "body_ext"
    if int(goal_dim) == JOINT_STATE_GOAL_DIM:
        return "joint_state"
    if int(goal_dim) == SPLIT_GOAL_DIM:
        return "joint_state_split"
    if int(goal_dim) == SPLIT_GOAL_NO_LOG_DIM:
        return "joint_state_split_no_log"
    if int(goal_dim) == SPLIT_END_EFFECTOR_GOAL_DIM:
        return "joint_state_split_end_effector"
    if int(goal_dim) == SPLIT_END_EFFECTOR_NO_LOG_GOAL_DIM:
        return "joint_state_split_end_effector_no_log"
    return f"{goal_dim}-D"


def _apply_arrival_channel_mask(goal: torch.Tensor,
                                goal_dim: int,
                                arrival_keep_mask: torch.Tensor,
                                legacy_split_goal_layout: bool = False) -> torch.Tensor:
    if int(goal_dim) == EXTENDED_BODY_GOAL_DIM:
        goal = goal.clone()
        goal[:, 8:9] = 0.0
        return goal
    if int(goal_dim) in (
            SPLIT_GOAL_DIM, SPLIT_GOAL_NO_LOG_DIM,
            SPLIT_END_EFFECTOR_GOAL_DIM,
            SPLIT_END_EFFECTOR_NO_LOG_GOAL_DIM):
        goal = goal.clone()
        keep = arrival_keep_mask.unsqueeze(-1).to(
            device=goal.device, dtype=goal.dtype)
        if (legacy_split_goal_layout
                and int(goal_dim) == SPLIT_GOAL_DIM):
            goal[:, 7:12] = goal[:, 7:12] * keep
            goal[:, 54:55] = goal[:, 54:55] * keep
            return goal
        layout = _split_goal_layout(goal_dim)
        goal[:, layout["horizontal_urgency"]] = (
            goal[:, layout["horizontal_urgency"]] * keep)
        goal[:, layout["vertical_urgency"]] = (
            goal[:, layout["vertical_urgency"]] * keep)
        goal[:, layout["time"]] = goal[:, layout["time"]] * keep
        return goal
    return goal


class ArrivalTimeEmbedder(nn.Module):

    def __init__(self, h_dim: int):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
        )

    def forward(self, arrival_time_frame):
        arrival_time_frame = arrival_time_frame.reshape(-1).float()
        emb = timestep_embedding(
            arrival_time_frame, self.time_embed[0].in_features)
        return self.time_embed(emb)


class DenoiserMLP(nn.Module):
    # =========================================================================
    # NOTE: DenoiserMLP is NOT currently used — the active config
    # (config/denoiser/def.yaml) uses DenoiserTransformer.  The MLP is kept as
    # a lighter alternative for ablations / memory-constrained runs.  It is
    # fully wired for goal + scene conditioning and will work out of the box
    # if you switch the config's _target_ to this class and add the matching
    # keys (goal_dim, grid_size, cond_mask_prob.goal.position,
    # cond_mask_prob.scene). Legacy flat mask keys are still accepted.
    # =========================================================================

    def __init__(self,
                 h_dim=512,
                 n_blocks=2,
                 dropout: float = 0.1,
                 activation="gelu",
                 history_shape=(2, 276),
                 noise_shape=(1, 128),
                 goal_dim=5,
                 goal_encoding=GoalEncoding.LEGACY40,
                 grid_size=25,
                 cond_mask_prob=None,
                 cond_goal_root_mask_prob=None,
                 cond_goal_yaw_mask_prob=None,
                 cond_goal_time_mask_prob=None,
                 cond_goal_body_mask_prob=None,
                 cond_goal_orientation_mask_prob=None,
                 cond_goal_joint_mask_prob=None,
                 cond_goal_velocity_mask_prob=None,
                 cond_goal_end_effector_mask_prob=None,
                 cond_scene_mask_prob=None,
                 **kargs):
        super().__init__()
        self.h_dim = h_dim
        self.dropout = dropout
        self.n_blocks = n_blocks
        self.activation = activation

        self.history_shape = history_shape
        self.noise_shape = noise_shape
        self.goal_dim = int(goal_dim)
        self.goal_encoding = GoalEncoding.parse(goal_encoding)
        _validate_model_goal_encoding(self.goal_dim, self.goal_encoding)
        self.grid_size = grid_size
        self.scene_dim = grid_size**3
        mask_profiles = _resolve_condition_mask_profiles(
            kargs,
            cond_mask_prob=cond_mask_prob,
            cond_goal_root_mask_prob=cond_goal_root_mask_prob,
            cond_goal_yaw_mask_prob=cond_goal_yaw_mask_prob,
            cond_goal_time_mask_prob=cond_goal_time_mask_prob,
            cond_goal_body_mask_prob=cond_goal_body_mask_prob,
            cond_goal_orientation_mask_prob=cond_goal_orientation_mask_prob,
            cond_goal_joint_mask_prob=cond_goal_joint_mask_prob,
            cond_goal_velocity_mask_prob=cond_goal_velocity_mask_prob,
            cond_goal_end_effector_mask_prob=(
                cond_goal_end_effector_mask_prob),
            cond_scene_mask_prob=cond_scene_mask_prob,
        )
        self.cond_mask_prob_profiles = mask_profiles
        mask_probs = mask_profiles['locomotion']
        self.cond_goal_root_mask_prob = mask_probs['goal_position']
        self.cond_goal_position_hor_mask_prob = mask_probs[
            'goal_position_hor']
        self.cond_goal_position_vert_mask_prob = mask_probs[
            'goal_position_vert']
        self.cond_goal_yaw_mask_prob = mask_probs['goal_yaw']
        self.cond_goal_time_mask_prob = mask_probs['goal_time']
        self.cond_goal_body_mask_prob = mask_probs['goal_body']
        self.cond_goal_orientation_mask_prob = mask_probs['goal_orientation']
        self.cond_goal_orientation_rot6d_mask_prob = mask_probs[
            'goal_orientation_rot6d']
        self.cond_goal_gravity_mask_prob = mask_probs['goal_gravity']
        self.cond_goal_joint_mask_prob = mask_probs['goal_joint']
        self.cond_goal_velocity_mask_prob = mask_probs['goal_velocity']
        self.cond_goal_end_effector_mask_probs = mask_probs[
            'goal_end_effector']
        for name, prob in zip(
                SPLIT_END_EFFECTOR_TOKEN_ORDER,
                self.cond_goal_end_effector_mask_probs):
            setattr(
                self,
                f"cond_goal_end_effector_{name}_mask_prob",
                prob,
            )
        self.cond_scene_mask_prob = mask_probs['scene']

        self.sequence_pos_encoder = PositionalEncoding(self.h_dim,
                                                       self.dropout)
        self.embed_timestep = TimestepEmbedder(self.h_dim,
                                               self.sequence_pos_encoder)

        if _uses_split_goal_tokens(self.goal_encoding):
            split_layout = _split_goal_layout(self.goal_dim)
            self.embed_goal_hor = MLP(
                split_layout["horizontal"].stop - split_layout["horizontal"].start,
                h_dims=(self.h_dim, self.h_dim), activation=activation)
            self.embed_goal_vert = MLP(
                6, h_dims=(self.h_dim, self.h_dim), activation=activation)
            self.embed_goal_rot = MLP(
                6, h_dims=(self.h_dim, self.h_dim), activation=activation)
            self.embed_goal_pose = MLP(
                29, h_dims=(self.h_dim, self.h_dim), activation=activation)
            self.embed_goal_vel = MLP(
                4, h_dims=(self.h_dim, self.h_dim), activation=activation)
            if self.goal_encoding.uses_end_effectors:
                self.embed_goal_ee = MLP(
                    3, h_dims=(self.h_dim, self.h_dim),
                    activation=activation)
                self.goal_ee_type_embedding = nn.Embedding(
                    len(SPLIT_END_EFFECTOR_TOKEN_ORDER), self.h_dim)
        else:
            self.embed_goal = nn.Linear(self.goal_dim, self.h_dim)
        self.embed_scene = nn.Linear(self.scene_dim, self.h_dim)
        self.embed_history = nn.Linear(self.history_shape[-1], self.h_dim)
        self.embed_noise = nn.Linear(self.noise_shape[-1], self.h_dim)
        self.arrival_embedder = ArrivalTimeEmbedder(self.h_dim)

        # input: time + goal + scene + history + noise → all projected to h_dim
        goal_token_count = self.goal_encoding.token_count
        input_dim = self.h_dim * (
            (4 + goal_token_count)
            if _uses_split_goal_tokens(self.goal_encoding) else 5
        )
        self.input_project = nn.Linear(input_dim, self.h_dim)

        self.mlp = MLPBlock(h_dim=h_dim,
                            out_dim=np.prod(noise_shape),
                            n_blocks=n_blocks,
                            actfun=activation)

    def mask_condition(self, cond, probability, force_mask=False,
                       return_keep_mask=False, valid_mask=None):
        """Independent scalar or per-sample dropout for conditions."""
        return _mask_condition_impl(
            self, cond, probability, force_mask=force_mask,
            return_keep_mask=return_keep_mask, valid_mask=valid_mask)

    def forward(self, x_t, timesteps, y=None):
        """
        x_t: [B, T=1, D]
        timesteps: [batch_size] (int)
        y: dict with keys 'goal' [B, goal_dim], 'voxel' [B, grid_size³],
           'history_motion_normalized' [B, T_hist, nfeats]
        """
        if y is None:
            raise ValueError(
                "Goal+scene denoiser requires a condition dictionary"
            )

        batch_size = x_t.shape[0]

        emb_time = self.embed_timestep(timesteps).squeeze(0)  # [bs, h_dim]

        goal, goal_keep_mask = _mask_goal(self, y['goal'], y)
        y['goal_condition_keep_mask'] = goal_keep_mask
        voxel = y.get('voxel')
        if voxel is None:
            voxel = torch.zeros(
                batch_size,
                self.scene_dim,
                device=goal.device,
                dtype=goal.dtype,
            )
        voxel = self.mask_condition(
            voxel,
            _condition_mask_probability(
                self, y, 'scene', batch_size, voxel.device),
            force_mask=y.get('force_drop_scene', False),
            valid_mask=_deployment_valid_mask(
                self, y.get('scene_valid'), batch_size, voxel.device))
        arrival_time_frame = y.get(
            'time_to_arrival_frame', y.get('arrival_time_frame'))
        if _goal_dim_uses_arrival_pe(self.goal_dim):
            if arrival_time_frame is None:
                raise ValueError(
                    f"{_goal_dim_name(self.goal_dim)} denoiser requires "
                    "y['time_to_arrival_frame']")
            arrival_time_frame, arrival_keep_mask = self.mask_condition(
                arrival_time_frame.reshape(-1, 1).to(goal.device).float(),
                _condition_mask_probability(
                    self, y, 'goal_time', goal.shape[0], goal.device),
                force_mask=(
                    y.get('force_drop_arrival_time', False)
                    or y.get('force_drop_goal_time', False)),
                valid_mask=_model_goal_valid_mask(
                    self, y, 'time', goal.shape[0], goal.device),
                return_keep_mask=True,
            )
            y['arrival_time_condition_keep_mask'] = arrival_keep_mask
            arrival_pe = self.arrival_embedder(
                arrival_time_frame.squeeze(-1))
            arrival_pe = arrival_pe * arrival_keep_mask.unsqueeze(
                -1).to(arrival_pe.dtype)
            goal = _apply_arrival_channel_mask(
                goal, self.goal_dim, arrival_keep_mask,
                legacy_split_goal_layout=getattr(
                    self, 'legacy_split_goal_layout', False))
        else:
            arrival_pe = 0.0
        if _uses_split_goal_tokens(self.goal_encoding):
            layout = _split_goal_layout(self.goal_dim)
            emb_goal_hor = self.embed_goal_hor(goal[:, layout["horizontal"]])
            emb_goal_vert = self.embed_goal_vert(goal[:, layout["vertical"]])
            emb_goal_rot = self.embed_goal_rot(goal[:, layout["orientation"]])
            emb_goal_pose = self.embed_goal_pose(goal[:, layout["joint"]])
            emb_goal_vel = self.embed_goal_vel(goal[:, layout["velocity"]])
            emb_goal_hor = emb_goal_hor * goal_keep_mask.unsqueeze(
                -1).to(emb_goal_hor.dtype)
            emb_goal_vert = emb_goal_vert * y[
                'goal_vertical_condition_keep_mask'].unsqueeze(
                    -1).to(emb_goal_vert.dtype)
            emb_goal_rot = emb_goal_rot * y[
                'goal_orientation_condition_keep_mask'].unsqueeze(
                    -1).to(emb_goal_rot.dtype)
            emb_goal_pose = emb_goal_pose * y[
                'goal_joint_condition_keep_mask'].unsqueeze(
                    -1).to(emb_goal_pose.dtype)
            emb_goal_vel = emb_goal_vel * y[
                'goal_velocity_condition_keep_mask'].unsqueeze(
                    -1).to(emb_goal_vel.dtype)
            if _goal_dim_uses_arrival_pe(self.goal_dim):
                emb_goal_time = arrival_pe
                y['goal_time_condition_keep_mask'] = y[
                    'arrival_time_condition_keep_mask']
            goal_parts = [
                emb_goal_hor,
                emb_goal_vert,
                emb_goal_rot,
                emb_goal_pose,
            ]
            if self.goal_encoding.uses_end_effectors:
                layout = _split_goal_layout(self.goal_dim)
                end_effectors = goal[:, layout["end_effector"]].reshape(
                    batch_size, len(SPLIT_END_EFFECTOR_TOKEN_ORDER), 3)
                emb_goal_ee = self.embed_goal_ee(
                    end_effectors.reshape(-1, 3)
                ).reshape(
                    batch_size, len(SPLIT_END_EFFECTOR_TOKEN_ORDER),
                    self.h_dim)
                type_ids = torch.arange(
                    len(SPLIT_END_EFFECTOR_TOKEN_ORDER),
                    device=goal.device)
                emb_goal_ee = emb_goal_ee + self.goal_ee_type_embedding(
                    type_ids).unsqueeze(0)
                ee_keep = y[
                    'goal_end_effector_condition_keep_mask'
                ].unsqueeze(-1).to(emb_goal_ee.dtype)
                emb_goal_ee = emb_goal_ee * ee_keep
                goal_parts.append(emb_goal_ee.reshape(batch_size, -1))
            goal_parts.extend((emb_goal_vel, emb_goal_time))
            emb_goal = torch.cat(tuple(goal_parts), dim=1)
        else:
            emb_goal = self.embed_goal(goal)     # [bs, h_dim]
            if _goal_dim_uses_arrival_pe(self.goal_dim):
                emb_goal = emb_goal + arrival_pe
        emb_scene = self.embed_scene(voxel)  # [bs, h_dim]

        emb_history = self.embed_history(
            y['history_motion_normalized'].reshape(
                batch_size, self.history_shape[-1]))  # [bs, h_dim]

        emb_noise = self.embed_noise(
            x_t.reshape(batch_size, self.noise_shape[-1]))  # [bs, h_dim]

        if _uses_split_goal_tokens(self.goal_encoding):
            input_embed = torch.cat(
                (emb_time, emb_goal, emb_scene, emb_history, emb_noise),
                dim=1,
            )
        else:
            input_embed = torch.cat(
                (emb_time, emb_goal, emb_scene, emb_history, emb_noise),
                dim=1,
            )  # [bs, input_dim]
        output = self.mlp(self.input_project(input_embed))  # [bs, noise_dim]
        output = output.reshape(batch_size, *self.noise_shape)

        return output


class DenoiserTransformer(nn.Module):

    def __init__(self,
                 h_dim=256,
                 ff_size=1024,
                 num_layers=4,
                 num_heads=4,
                 dropout=0.1,
                 activation="gelu",
                 history_shape=(2, 276),
                 noise_shape=(1, 128),
                 clip_dim=512,
                 goal_dim=5,
                 goal_encoding=GoalEncoding.LEGACY40,
                 grid_size=25,
                 cond_mask_prob=None,
                 cond_text_mask_prob=None,
                 cond_goal_root_mask_prob=None,
                 cond_goal_yaw_mask_prob=None,
                 cond_goal_time_mask_prob=None,
                 cond_goal_body_mask_prob=None,
                 cond_goal_orientation_mask_prob=None,
                 cond_goal_joint_mask_prob=None,
                 cond_goal_velocity_mask_prob=None,
                 cond_goal_end_effector_mask_prob=None,
                 cond_scene_mask_prob=None,
                 legacy_split_goal_layout=False,
                 use_vae=True,
                 **kargs):
        super().__init__()
        self.h_dim = h_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.history_shape = history_shape
        self.noise_shape = noise_shape
        self.clip_dim = int(clip_dim)
        self.goal_dim = int(goal_dim)
        self.goal_encoding = GoalEncoding.parse(goal_encoding)
        self.legacy_split_goal_layout = bool(legacy_split_goal_layout)
        _validate_model_goal_encoding(self.goal_dim, self.goal_encoding)
        self.grid_size = grid_size
        self.scene_dim = grid_size**3
        mask_profiles = _resolve_condition_mask_profiles(
            kargs,
            cond_mask_prob=cond_mask_prob,
            cond_text_mask_prob=cond_text_mask_prob,
            cond_goal_root_mask_prob=cond_goal_root_mask_prob,
            cond_goal_yaw_mask_prob=cond_goal_yaw_mask_prob,
            cond_goal_time_mask_prob=cond_goal_time_mask_prob,
            cond_goal_body_mask_prob=cond_goal_body_mask_prob,
            cond_goal_orientation_mask_prob=cond_goal_orientation_mask_prob,
            cond_goal_joint_mask_prob=cond_goal_joint_mask_prob,
            cond_goal_velocity_mask_prob=cond_goal_velocity_mask_prob,
            cond_goal_end_effector_mask_prob=(
                cond_goal_end_effector_mask_prob),
            cond_scene_mask_prob=cond_scene_mask_prob,
        )
        self.cond_mask_prob_profiles = mask_profiles
        mask_probs = mask_profiles['locomotion']
        self.cond_text_mask_prob = mask_probs['text']
        self.cond_mask_prob = self.cond_text_mask_prob
        self.text_condition_enabled = bool(
            kargs.pop('text_condition_enabled', False))
        self.cond_goal_root_mask_prob = mask_probs['goal_position']
        self.cond_goal_position_hor_mask_prob = mask_probs[
            'goal_position_hor']
        self.cond_goal_position_vert_mask_prob = mask_probs[
            'goal_position_vert']
        self.cond_goal_yaw_mask_prob = mask_probs['goal_yaw']
        self.cond_goal_time_mask_prob = mask_probs['goal_time']
        self.cond_goal_body_mask_prob = mask_probs['goal_body']
        self.cond_goal_orientation_mask_prob = mask_probs['goal_orientation']
        self.cond_goal_orientation_rot6d_mask_prob = mask_probs[
            'goal_orientation_rot6d']
        self.cond_goal_gravity_mask_prob = mask_probs['goal_gravity']
        self.cond_goal_joint_mask_prob = mask_probs['goal_joint']
        self.cond_goal_velocity_mask_prob = mask_probs['goal_velocity']
        self.cond_goal_end_effector_mask_probs = mask_probs[
            'goal_end_effector']
        for name, prob in zip(
                SPLIT_END_EFFECTOR_TOKEN_ORDER,
                self.cond_goal_end_effector_mask_probs):
            setattr(
                self,
                f"cond_goal_end_effector_{name}_mask_prob",
                prob,
            )
        self.cond_scene_mask_prob = mask_probs['scene']

        # input embeddings
        self.sequence_pos_encoder = PositionalEncoding(self.h_dim,
                                                       self.dropout)
        self.embed_timestep = TimestepEmbedder(self.h_dim,
                                               self.sequence_pos_encoder)

        if (self.goal_encoding is GoalEncoding.SPLIT
                and self.legacy_split_goal_layout):
            self.embed_goal_trans = MLP(
                12, h_dims=(self.h_dim, self.h_dim), activation=activation)
            self.embed_goal_rot = MLP(
                9, h_dims=(self.h_dim, self.h_dim), activation=activation)
            self.embed_goal_pose = MLP(
                29, h_dims=(self.h_dim, self.h_dim), activation=activation)
            self.embed_goal_vel = MLP(
                4, h_dims=(self.h_dim, self.h_dim), activation=activation)
            self.embed_goal_time = MLP(
                1, h_dims=(self.h_dim, self.h_dim), activation=activation)
        elif _uses_split_goal_tokens(self.goal_encoding):
            split_layout = _split_goal_layout(self.goal_dim)
            self.embed_goal_hor = MLP(
                split_layout["horizontal"].stop - split_layout["horizontal"].start,
                h_dims=(self.h_dim, self.h_dim), activation=activation)
            self.embed_goal_vert = MLP(
                6, h_dims=(self.h_dim, self.h_dim), activation=activation)
            self.embed_goal_rot = MLP(
                6, h_dims=(self.h_dim, self.h_dim), activation=activation)
            self.embed_goal_pose = MLP(
                29, h_dims=(self.h_dim, self.h_dim), activation=activation)
            self.embed_goal_vel = MLP(
                4, h_dims=(self.h_dim, self.h_dim), activation=activation)
            if self.goal_encoding.uses_end_effectors:
                self.embed_goal_ee = MLP(
                    3, h_dims=(self.h_dim, self.h_dim),
                    activation=activation)
                self.goal_ee_type_embedding = nn.Embedding(
                    len(SPLIT_END_EFFECTOR_TOKEN_ORDER), self.h_dim)
        else:
            self.embed_goal = nn.Linear(self.goal_dim, self.h_dim)
        if self.text_condition_enabled:
            self.embed_text = nn.Linear(self.clip_dim, self.h_dim)
        self.embed_scene = nn.Linear(self.scene_dim, self.h_dim)
        self.embed_history = nn.Linear(self.history_shape[-1], self.h_dim)
        self.embed_noise = nn.Linear(self.noise_shape[-1], self.h_dim)
        self.arrival_embedder = ArrivalTimeEmbedder(self.h_dim)

        # transformer encoder layers
        print("TRANS_ENC init")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.h_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(
            seqTransEncoderLayer, num_layers=self.num_layers)

        # output projection
        self.output_process = nn.Linear(self.h_dim, self.noise_shape[-1])

    def mask_condition(self, cond, probability, force_mask=False,
                       return_keep_mask=False, valid_mask=None):
        """Independent scalar or per-sample dropout for conditions."""
        return _mask_condition_impl(
            self, cond, probability, force_mask=force_mask,
            return_keep_mask=return_keep_mask, valid_mask=valid_mask)

    def forward(self, x_t, timesteps, y=None):
        """
        x_t: [B, T=1, D]
        timesteps: [batch_size] (int)
        """
        if y is None:
            raise ValueError("Goal+scene denoiser requires a condition dictionary")

        batch_size = x_t.shape[0]
        device = x_t.device

        emb_time = self.embed_timestep(timesteps)  # [1, bs, d]

        goal = y.get('goal')
        goal_present = goal is not None
        if goal_present:
            goal = goal.to(device=device, dtype=x_t.dtype)
            goal, goal_keep_mask = _mask_goal(self, goal, y)
        else:
            goal = torch.zeros(
                batch_size, self.goal_dim, device=device, dtype=x_t.dtype
            )
            goal_keep_mask = torch.zeros(
                batch_size, dtype=torch.bool, device=device
            )
            y['goal_condition_keep_mask'] = goal_keep_mask
            if _uses_split_goal_tokens(self.goal_encoding):
                y['goal_position_hor_condition_keep_mask'] = goal_keep_mask
                y['goal_position_vert_condition_keep_mask'] = goal_keep_mask
                y['goal_gravity_condition_keep_mask'] = goal_keep_mask
                y['goal_orientation_condition_keep_mask'] = goal_keep_mask
                y['goal_vertical_condition_keep_mask'] = goal_keep_mask
                y['goal_joint_condition_keep_mask'] = goal_keep_mask
                y['goal_velocity_condition_keep_mask'] = goal_keep_mask
                y['goal_time_condition_keep_mask'] = goal_keep_mask
                if self.goal_encoding.uses_end_effectors:
                    y['goal_end_effector_condition_keep_mask'] = torch.zeros(
                        batch_size,
                        len(SPLIT_END_EFFECTOR_TOKEN_ORDER),
                        dtype=torch.bool,
                        device=device,
                    )
            elif _goal_dim_uses_arrival_pe(self.goal_dim):
                y['arrival_time_condition_keep_mask'] = goal_keep_mask
        y['goal_condition_keep_mask'] = goal_keep_mask

        voxel = y.get('voxel')
        if voxel is None:
            voxel = torch.zeros(
                batch_size, self.scene_dim, device=device, dtype=x_t.dtype
            )
        else:
            voxel = voxel.to(device=device, dtype=x_t.dtype)
        voxel = self.mask_condition(
            voxel,
            _condition_mask_probability(
                self, y, 'scene', batch_size, voxel.device),
            force_mask=y.get('force_drop_scene', False),
            valid_mask=_deployment_valid_mask(
                self, y.get('scene_valid'), batch_size, voxel.device))

        history_motion = y.get('history_motion_normalized')
        if history_motion is None:
            history_motion = torch.zeros(
                batch_size, *self.history_shape, device=device, dtype=x_t.dtype
            )
        else:
            history_motion = history_motion.to(device=device, dtype=x_t.dtype)

        use_text_condition = bool(
            getattr(self, 'text_condition_enabled', False))
        if use_text_condition:
            text_embedding = y.get('text_embedding')
            if text_embedding is None:
                text_embedding = torch.zeros(
                    batch_size, self.clip_dim, device=device, dtype=x_t.dtype
                )
                text_keep_mask = torch.zeros(
                    batch_size, dtype=torch.bool, device=device
                )
            else:
                if text_embedding.ndim == 1:
                    text_embedding = text_embedding.unsqueeze(0)
                if text_embedding.shape[-1] != self.clip_dim:
                    raise ValueError(
                        f"text_embedding last dim must be {self.clip_dim}, got "
                        f"{tuple(text_embedding.shape)}"
                    )
                text_embedding = text_embedding.to(device=device, dtype=x_t.dtype)
                text_embedding, text_keep_mask = self.mask_condition(
                    text_embedding,
                    _condition_mask_probability(
                        self, y, 'text', batch_size, text_embedding.device),
                    force_mask=(
                        y.get('force_drop_text', False)
                        or y.get('uncond', False)),
                    valid_mask=_deployment_valid_mask(
                        self, y.get('text_valid'),
                        batch_size, text_embedding.device),
                    return_keep_mask=True,
                )
            y['text_condition_keep_mask'] = text_keep_mask
            emb_text = self.embed_text(text_embedding).unsqueeze(0)
            emb_text = emb_text * text_keep_mask.reshape(
                1, batch_size, 1).to(emb_text.dtype)
        else:
            y['text_condition_keep_mask'] = torch.zeros(
                batch_size, dtype=torch.bool, device=device)

        arrival_time_frame = y.get(
            'time_to_arrival_frame', y.get('arrival_time_frame'))
        if _goal_dim_uses_arrival_pe(self.goal_dim):
            if arrival_time_frame is None:
                if goal_present:
                    raise ValueError(
                        f"{_goal_dim_name(self.goal_dim)} denoiser requires "
                        "y['time_to_arrival_frame']"
                    )
                arrival_time_frame = torch.zeros(
                    batch_size, 1, device=device, dtype=x_t.dtype
                )
                arrival_keep_mask = torch.zeros(
                    batch_size, dtype=torch.bool, device=device
                )
            else:
                arrival_time_frame, arrival_keep_mask = self.mask_condition(
                    arrival_time_frame.reshape(-1, 1).to(device=device).float(),
                    _condition_mask_probability(
                        self, y, 'goal_time', goal.shape[0], goal.device),
                    force_mask=(
                        y.get('force_drop_arrival_time', False)
                        or y.get('force_drop_goal_time', False)),
                    valid_mask=_model_goal_valid_mask(
                        self, y, 'time', goal.shape[0], goal.device),
                    return_keep_mask=True,
                )
            y['arrival_time_condition_keep_mask'] = arrival_keep_mask
            arrival_pe = self.arrival_embedder(
                arrival_time_frame.squeeze(-1))
            arrival_pe = arrival_pe * arrival_keep_mask.unsqueeze(
                -1).to(arrival_pe.dtype)
            if goal_present:
                goal = _apply_arrival_channel_mask(
                    goal, self.goal_dim, arrival_keep_mask,
                    legacy_split_goal_layout=self.legacy_split_goal_layout)
        else:
            arrival_pe = 0.0
        emb_scene = self.embed_scene(voxel).unsqueeze(0)
        emb_history = self.embed_history(history_motion).permute(1, 0, 2)
        emb_noise = self.embed_noise(x_t).permute(1, 0, 2)  # [1, bs, d]

        if (self.goal_encoding is GoalEncoding.SPLIT
                and self.legacy_split_goal_layout):
            emb_goal_trans = self.embed_goal_trans(
                goal[:, 0:12]).unsqueeze(0)
            emb_goal_rot = self.embed_goal_rot(
                goal[:, 12:21]).unsqueeze(0)
            emb_goal_pose = self.embed_goal_pose(
                goal[:, 21:50]).unsqueeze(0)
            emb_goal_vel = self.embed_goal_vel(
                goal[:, 50:54]).unsqueeze(0)
            emb_goal_time = self.embed_goal_time(
                goal[:, 54:55]).unsqueeze(0)
            arrival_pe_ = arrival_pe.unsqueeze(0)
            emb_goal_trans = emb_goal_trans + arrival_pe_
            emb_goal_rot = emb_goal_rot + arrival_pe_
            emb_goal_pose = emb_goal_pose + arrival_pe_
            emb_goal_vel = emb_goal_vel + arrival_pe_
            emb_goal_time = emb_goal_time + arrival_pe_
            keep_masks = (
                y['goal_condition_keep_mask'],
                y['goal_orientation_condition_keep_mask'],
                y['goal_joint_condition_keep_mask'],
                y['goal_velocity_condition_keep_mask'],
                y['goal_time_condition_keep_mask'],
            )
            embeddings = (
                emb_goal_trans, emb_goal_rot, emb_goal_pose,
                emb_goal_vel, emb_goal_time,
            )
            embeddings = tuple(
                embedding * keep_mask.unsqueeze(-1).to(embedding.dtype)
                for embedding, keep_mask in zip(embeddings, keep_masks)
            )
            xseq_parts = [
                emb_time,
                *embeddings,
                emb_scene,
                emb_history,
                emb_noise,
            ]
            if use_text_condition:
                xseq_parts.insert(1, emb_text)
            xseq = torch.cat(tuple(xseq_parts), dim=0)
        elif _uses_split_goal_tokens(self.goal_encoding):
            layout = _split_goal_layout(self.goal_dim)
            emb_goal_hor = self.embed_goal_hor(
                goal[:, layout["horizontal"]]).unsqueeze(0)
            emb_goal_vert = self.embed_goal_vert(
                goal[:, layout["vertical"]]).unsqueeze(0)
            emb_goal_rot = self.embed_goal_rot(
                goal[:, layout["orientation"]]).unsqueeze(0)
            emb_goal_pose = self.embed_goal_pose(
                goal[:, layout["joint"]]).unsqueeze(0)
            emb_goal_vel = self.embed_goal_vel(
                goal[:, layout["velocity"]]).unsqueeze(0)
            # Mask before sequence_pos_encoder: zero component content, MLP
            # bias, but let the slot PE survive so the transformer still knows
            # which condition is missing.  The time condition is its own
            # arrival-PE token; it is not added into the other goal tokens.
            emb_goal_hor = emb_goal_hor * goal_keep_mask.unsqueeze(
                -1).to(emb_goal_hor.dtype)
            emb_goal_vert = emb_goal_vert * y[
                'goal_vertical_condition_keep_mask'].unsqueeze(
                    -1).to(emb_goal_vert.dtype)
            emb_goal_rot = emb_goal_rot * y[
                'goal_orientation_condition_keep_mask'].unsqueeze(
                    -1).to(emb_goal_rot.dtype)
            emb_goal_pose = emb_goal_pose * y[
                'goal_joint_condition_keep_mask'].unsqueeze(
                    -1).to(emb_goal_pose.dtype)
            emb_goal_vel = emb_goal_vel * y[
                'goal_velocity_condition_keep_mask'].unsqueeze(
                    -1).to(emb_goal_vel.dtype)
            if _goal_dim_uses_arrival_pe(self.goal_dim):
                emb_goal_time = arrival_pe.unsqueeze(0)
                y['goal_time_condition_keep_mask'] = y[
                    'arrival_time_condition_keep_mask']
            xseq_parts = [
                emb_time,
                emb_goal_hor,
                emb_goal_vert,
                emb_goal_rot,
                emb_goal_pose,
            ]
            if self.goal_encoding.uses_end_effectors:
                end_effectors = goal[:, layout["end_effector"]].reshape(
                    batch_size, len(SPLIT_END_EFFECTOR_TOKEN_ORDER), 3)
                emb_goal_ee = self.embed_goal_ee(
                    end_effectors.reshape(-1, 3)
                ).reshape(
                    batch_size, len(SPLIT_END_EFFECTOR_TOKEN_ORDER),
                    self.h_dim)
                type_ids = torch.arange(
                    len(SPLIT_END_EFFECTOR_TOKEN_ORDER),
                    device=goal.device)
                emb_goal_ee = emb_goal_ee + self.goal_ee_type_embedding(
                    type_ids).unsqueeze(0)
                ee_keep = y[
                    'goal_end_effector_condition_keep_mask'
                ].transpose(0, 1).unsqueeze(-1).to(emb_goal_ee.dtype)
                emb_goal_ee = emb_goal_ee.permute(1, 0, 2) * ee_keep
                xseq_parts.append(emb_goal_ee)
            xseq_parts.extend([
                emb_goal_vel,
                emb_goal_time,
                emb_scene,
                emb_history,
                emb_noise,
            ])
            if use_text_condition:
                xseq_parts.insert(1, emb_text)
            xseq = torch.cat(tuple(xseq_parts), dim=0)
        else:
            emb_goal = self.embed_goal(goal).unsqueeze(0)
            emb_goal = emb_goal * goal_keep_mask.unsqueeze(-1).to(
                emb_goal.dtype)
            if _goal_dim_uses_arrival_pe(self.goal_dim):
                emb_goal = emb_goal + arrival_pe.unsqueeze(0)
            xseq_parts = [
                emb_time,
                emb_goal,
                emb_scene,
                emb_history,
                emb_noise,
            ]
            if use_text_condition:
                xseq_parts.insert(1, emb_text)
            xseq = torch.cat(tuple(xseq_parts), dim=0)
        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqTransEncoder(xseq)[
            -self.noise_shape[0]:]  # [1, bs, h_dim]
        output = self.output_process(output)  # [1, B, noise_shape[-1]]
        output = output.permute(1, 0, 2)  # [B, 1, noise_shape[-1]]
        # print('output shape:', output.shape)

        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):

    def __init__(self, h_dim, sequence_pos_encoder):
        super().__init__()
        self.h_dim = h_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.h_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.h_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(
            self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class MLP(nn.Module):

    def __init__(self,
                 in_dim,
                 h_dims=[128, 128],
                 activation='tanh',
                 use_lora=False,
                 lora_rank=16):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'gelu':
            self.activation = torch.nn.GELU()
        elif activation == 'lrelu':
            self.activation = torch.nn.LeakyReLU()
        self.out_dim = h_dims[-1]
        self.layers = nn.ModuleList()
        in_dim_ = in_dim
        for h_dim in h_dims:
            layer = lora.Linear(in_dim_, h_dim,
                                r=lora_rank) if use_lora else nn.Linear(
                                    in_dim_, h_dim)
            self.layers.append(layer)
            in_dim_ = h_dim

    def forward(self, x):
        for fc in self.layers:
            x = self.activation(fc(x))
        return x


class MLPBlock(nn.Module):

    def __init__(self,
                 h_dim,
                 out_dim,
                 n_blocks,
                 actfun='relu',
                 residual=True,
                 use_lora=False,
                 lora_rank=16):
        super(MLPBlock, self).__init__()
        self.residual = residual
        self.layers = nn.ModuleList([
            MLP(h_dim, h_dims=(h_dim, h_dim), activation=actfun)
            for _ in range(n_blocks)
        ])  # two fc layers in each MLP
        self.out_fc = lora.Linear(h_dim, out_dim,
                                  r=lora_rank) if use_lora else nn.Linear(
                                      h_dim, out_dim)

    def forward(self, x):
        h = x
        for layer in self.layers:
            r = h if self.residual else 0
            h = layer(h) + r
        y = self.out_fc(h)
        return y
