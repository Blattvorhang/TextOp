from pathlib import Path

import torch
from omegaconf import OmegaConf

from robotmdar.dtype.motion import (
    G1_MUJOCO_DOF_JOINT_NAMES,
    G1_WRIST_DOF_INDICES,
    motion_dict_to_feature_v3,
    motion_feature_to_dict_v3,
)
from robotmdar.skeleton.robot import RobotSkeleton


ROOT = Path(__file__).resolve().parents[2]


def _skeleton():
    cfg = OmegaConf.load(
        ROOT / 'TextOpRobotMDAR/robotmdar/config/skeleton/g1.yaml'
    )
    cfg.asset.assetRoot = str(
        ROOT / 'TextOpRobotMDAR/description/robots/g1'
    )
    return RobotSkeleton(device='cpu', cfg=cfg)


def _motion(dof):
    batch, frames = dof.shape[:2]
    root_rot = torch.zeros((batch, frames, 4), dtype=dof.dtype)
    root_rot[..., 3] = 1.0
    return {
        'root_trans_offset': torch.zeros(
            (batch, frames, 3), dtype=dof.dtype
        ),
        'root_rot': root_rot,
        'dof': dof,
        'contact_mask': torch.ones((batch, frames, 2), dtype=dof.dtype),
    }


def test_fk_dof_velocity_is_scalar_29dof_forward_difference_at_dataset_fps():
    skeleton = _skeleton()
    frames = 4
    slopes = torch.arange(1, 30, dtype=torch.float32) / 100.0
    dof = torch.arange(frames, dtype=torch.float32)[None, :, None] * slopes

    result = skeleton.forward_kinematics(_motion(dof), fps=50.0)

    assert result.dof_pos.shape == (1, frames, 29)
    assert result.dof_vel.shape == (1, frames, 29)
    torch.testing.assert_close(result.dof_pos, dof)
    torch.testing.assert_close(
        result.dof_vel,
        (slopes * 50.0)[None, None].expand(1, frames, 29),
    )


def test_69d_feature_round_trip_preserves_position_and_delta_joint_order():
    frames = 4
    frame_offset = torch.arange(frames, dtype=torch.float32)[:, None] * 100.0
    joint_marker = torch.arange(29, dtype=torch.float32)[None]
    dof = (frame_offset + joint_marker).unsqueeze(0)

    feature, abs_pose = motion_dict_to_feature_v3(_motion(dof))
    reconstructed = motion_feature_to_dict_v3(feature, abs_pose)

    torch.testing.assert_close(feature[..., 11:40], dof[:, :-1])
    torch.testing.assert_close(
        feature[..., 40:69], dof[:, 1:] - dof[:, :-1]
    )
    torch.testing.assert_close(reconstructed['dof'], dof[:, :-1])


def test_wrist_indices_and_fk_hands_follow_the_same_mujoco_order():
    skeleton = _skeleton()
    assert tuple(skeleton.fk.dof_joint_names) == G1_MUJOCO_DOF_JOINT_NAMES
    assert G1_WRIST_DOF_INDICES == (19, 20, 21, 26, 27, 28)

    base_dof = torch.zeros((1, 1, 29), dtype=torch.float32)
    base = skeleton.forward_kinematics(_motion(base_dof))
    left_hand, right_hand = skeleton.hand_id

    left_dof = base_dof.clone()
    left_dof[..., 21] = 0.5
    left = skeleton.forward_kinematics(_motion(left_dof))
    assert not torch.allclose(
        left.global_translation_extend[..., left_hand, :],
        base.global_translation_extend[..., left_hand, :],
    )
    torch.testing.assert_close(
        left.global_translation_extend[..., right_hand, :],
        base.global_translation_extend[..., right_hand, :],
    )

    right_dof = base_dof.clone()
    right_dof[..., 28] = 0.5
    right = skeleton.forward_kinematics(_motion(right_dof))
    torch.testing.assert_close(
        right.global_translation_extend[..., left_hand, :],
        base.global_translation_extend[..., left_hand, :],
    )
    assert not torch.allclose(
        right.global_translation_extend[..., right_hand, :],
        base.global_translation_extend[..., right_hand, :],
    )
