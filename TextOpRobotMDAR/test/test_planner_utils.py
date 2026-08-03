from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from TextOpRobotMDAR.robotmdar.utils.planner_convert import (
    align_generated_history_pose,
    generated_history_at_frame,
    isaaclab_to_mujoco_dof,
    motion_dict_to_g1data,
    mujoco_to_isaaclab_dof,
    state_goal_from_reference,
    state_to_ego_goal,
    state_to_model_input,
    tracked_frame_from_timestamps,
)
from TextOpRobotMDAR.robotmdar.dtype.motion import (
    motion_dict_to_feature_v3,
    motion_feature_to_dict_v3,
)
from TextOpRobotMDAR.robotmdar.dtype.rotation import euler_angles_to_quaternion
from TextOpRobotMDAR.robotmdar.skeleton.robot import RobotSkeleton


class IdentityNormalization:
    @staticmethod
    def normalize(feature):
        return feature

    @staticmethod
    def denormalize(feature):
        return feature


def test_fk_preserves_non_upright_root_quaternion():
    project_root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(
        project_root / "robotmdar/config/skeleton/g1.yaml")
    cfg.asset.assetRoot = str(project_root / "description/robots/g1")
    skeleton = RobotSkeleton(device="cpu", cfg=cfg)
    root_rot = euler_angles_to_quaternion(torch.tensor([
        [[0.7, -0.4, 1.2], [-0.5, 0.6, -2.0]],
    ], dtype=torch.float32))
    motion = {
        "root_trans_offset": torch.zeros((1, 2, 3)),
        "root_rot": root_rot,
        "dof": torch.zeros((1, 2, 29)),
        "contact_mask": torch.ones((1, 2, 2)),
    }

    fk = skeleton.forward_kinematics(motion)

    assert fk["dof_pos"].shape == (1, 2, 29)
    torch.testing.assert_close(fk["dof_pos"], motion["dof"])
    dots = torch.abs(torch.sum(
        fk["global_rotation"][:, :, 0] * root_rot, dim=-1))
    torch.testing.assert_close(dots, torch.ones_like(dots), atol=1e-5, rtol=1e-5)


def test_wrist_yaw_moves_palm_center_keypoint():
    project_root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(project_root / "robotmdar/config/skeleton/g1.yaml")
    cfg.asset.assetRoot = str(project_root / "description/robots/g1")
    skeleton = RobotSkeleton(device="cpu", cfg=cfg)
    dof = torch.zeros((1, 2, 29))
    dof[:, 1, 21] = torch.pi / 2  # left_wrist_yaw
    motion = {
        "root_trans_offset": torch.zeros((1, 2, 3)),
        "root_rot": torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(1, 2, 4),
        "dof": dof,
        "contact_mask": torch.ones((1, 2, 2)),
    }

    palm = skeleton.forward_kinematics(motion)[
        "global_translation_extend"
    ][:, :, skeleton.hand_id[0]]

    assert torch.linalg.vector_norm(palm[:, 1] - palm[:, 0]).item() > 0.05


def test_palm_center_offset_is_in_wrist_yaw_frame():
    project_root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(project_root / "robotmdar/config/skeleton/g1.yaml")
    cfg.asset.assetRoot = str(project_root / "description/robots/g1")
    skeleton = RobotSkeleton(device="cpu", cfg=cfg)
    dof = torch.zeros((1, 1, 29))
    dof[..., 19:22] = torch.tensor([0.4, -0.3, 0.8])
    dof[..., 26:29] = torch.tensor([-0.2, 0.5, -0.7])
    motion = {
        "root_trans_offset": torch.tensor([[[0.3, -0.2, 0.9]]]),
        "root_rot": euler_angles_to_quaternion(
            torch.tensor([[[0.2, -0.1, 0.6]]])
        ),
        "dof": dof,
        "contact_mask": torch.ones((1, 1, 2)),
    }

    fk = skeleton.forward_kinematics(motion)
    offsets = (
        torch.tensor([0.0415, 0.003, 0.0]),
        torch.tensor([0.0415, -0.003, 0.0]),
    )
    wrist_names = ("left_wrist_yaw_link", "right_wrist_yaw_link")
    for hand_id, wrist_name, offset in zip(
        skeleton.hand_id, wrist_names, offsets
    ):
        wrist_id = skeleton.fk.body_names_augment.index(wrist_name)
        wrist_pos = fk["global_translation_extend"][..., wrist_id, :]
        wrist_rot = fk["global_rotation_mat_extend"][..., wrist_id, :, :]
        expected_palm = wrist_pos + torch.matmul(
            wrist_rot, offset[:, None]
        ).squeeze(-1)
        actual_palm = fk["global_translation_extend"][..., hand_id, :]
        torch.testing.assert_close(actual_palm, expected_palm)


def test_joint_order_round_trip():
    isaaclab = np.arange(29, dtype=np.float32).reshape(1, 29)
    mujoco = isaaclab_to_mujoco_dof(isaaclab)
    np.testing.assert_array_equal(mujoco_to_isaaclab_dof(mujoco), isaaclab)


def test_controller_history_ends_at_current_non_upright_pose():
    yaws = np.asarray([0.0, 1.0, 3.13, -3.13, -3.0], dtype=np.float32)
    rolls = np.asarray([0.0, 0.1, 0.2, 0.7, 1.1], dtype=np.float32)
    cy, sy = np.cos(yaws / 2.0), np.sin(yaws / 2.0)
    cr, sr = np.cos(rolls / 2.0), np.sin(rolls / 2.0)
    rotations = np.stack((cy * sr, sy * sr, sy * cr, cy * cr), axis=-1)
    positions = np.stack(
        (np.arange(5, dtype=np.float32), np.zeros(5), np.full(5, 0.77)),
        axis=-1)
    joints = np.zeros((5, 29), dtype=np.float32)
    state = SimpleNamespace(raw={
        "g1_pos": positions,
        "g1_root_rot": rotations,
        "g1_joint_pos": joints,
    })

    feature, abs_pose = state_to_model_input(
        state, history_len=2, val_data=IdentityNormalization(), device="cpu")

    assert feature.shape == (1, 2, 69)
    np.testing.assert_allclose(
        abs_pose["root_trans_offset"].numpy(), positions[-2:-1])
    np.testing.assert_allclose(feature[0, 1, 4].numpy(), 0.13, atol=1e-5)

    reconstructed = motion_feature_to_dict_v3(feature, abs_pose)
    expected_current = torch.as_tensor(rotations[-1])
    reconstructed_current = reconstructed["root_rot"][0, -1]
    torch.testing.assert_close(
        torch.abs(torch.dot(reconstructed_current, expected_current)),
        torch.tensor(1.0), atol=1e-5, rtol=1e-5)


def test_goal_uses_current_history_feature_reference():
    state = SimpleNamespace(
        goal_root_pos=np.asarray([2.0, 1.0, 0.77], dtype=np.float32),
        goal_heading=np.asarray([0.5], dtype=np.float32),
        raw={
            "g1_pos": np.asarray([
                [0.0, 1.0, 0.77],
                [1.0, 1.0, 0.77],
                [1.1, 1.0, 0.77],
            ], dtype=np.float32),
            "g1_root_rot": np.asarray([
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ], dtype=np.float32),
        },
    )
    goal = state_to_ego_goal(state, "cpu")
    np.testing.assert_allclose(
        goal.numpy(), [[0.9, 0.0, 0.0, np.cos(0.5), np.sin(0.5)]],
        atol=1e-6)


def test_generated_goal_uses_translated_history_endpoint():
    state = SimpleNamespace(
        goal_root_pos=np.asarray([12.0, 20.0, 0.9], dtype=np.float32),
        goal_heading=np.asarray([0.5], dtype=np.float32),
    )
    goal = state_goal_from_reference(
        state,
        reference_pos=torch.tensor([[10.0, 20.0, 0.9]]),
        reference_rot=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        device="cpu",
    )

    np.testing.assert_allclose(
        goal.numpy(), [[2.0, 0.0, 0.0, np.cos(0.5), np.sin(0.5)]],
        atol=1e-6)


def test_generated_history_alignment_translates_anchor_without_mutation():
    abs_pose = {
        "root_trans_offset": torch.tensor([[1.0, 2.0, 0.7]]),
        "root_rot": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
    }
    generated_endpoint = torch.tensor([[2.0, 4.0, 0.8]])
    state = SimpleNamespace(raw={
        "g1_pos": np.asarray([
            [9.5, 19.5, 0.9],
            [10.0, 20.0, 0.9],
        ], dtype=np.float32),
        "g1_root_rot": np.asarray([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32),
    })

    (aligned_pose, aligned_endpoint, aligned_rotation, translation,
     aligned_history) = align_generated_history_pose(
        abs_pose, generated_endpoint, abs_pose["root_rot"], state, "cpu")

    torch.testing.assert_close(translation, torch.tensor([[8.0, 16.0, 0.1]]))
    torch.testing.assert_close(
        aligned_pose["root_trans_offset"], torch.tensor([[9.0, 18.0, 0.8]]))
    torch.testing.assert_close(
        aligned_pose["root_rot"], abs_pose["root_rot"])
    torch.testing.assert_close(
        aligned_endpoint, torch.tensor([[10.0, 20.0, 0.9]]))
    torch.testing.assert_close(aligned_rotation, abs_pose["root_rot"])
    assert aligned_history is None
    torch.testing.assert_close(
        abs_pose["root_trans_offset"], torch.tensor([[1.0, 2.0, 0.7]]))


def test_generated_history_alignment_corrects_absolute_height_channel():
    generated_pos = torch.tensor([
        [[1.0, 2.0, 0.7], [1.2, 2.1, 0.8], [1.4, 2.2, 0.9]],
    ])
    generated_rot = euler_angles_to_quaternion(torch.tensor([
        [[0.0, 0.0, 0.0], [0.1, -0.2, 0.0], [0.2, -0.3, 0.0]],
    ]))
    generated_dof = torch.zeros((1, 3, 29))
    generated_contact = torch.ones((1, 3, 2))
    history, abs_pose = motion_dict_to_feature_v3({
        "root_trans_offset": generated_pos,
        "root_rot": generated_rot,
        "dof": generated_dof,
        "contact_mask": generated_contact,
    })
    real_rot = euler_angles_to_quaternion(
        torch.tensor([[0.9, -0.4, 0.3]]))
    real_pos = torch.tensor([[10.0, 20.0, 0.25]])
    state = SimpleNamespace(raw={
        "g1_pos": real_pos.numpy(),
        "g1_root_rot": real_rot.numpy(),
    })

    (aligned_pose, _, _, _, aligned_history) = align_generated_history_pose(
        abs_pose,
        generated_pos[:, 1],
        generated_rot[:, 1],
        state,
        "cpu",
        history_motion=history,
        val_data=IdentityNormalization(),
    )
    reconstructed = motion_feature_to_dict_v3(
        aligned_history, aligned_pose)

    torch.testing.assert_close(
        reconstructed["root_trans_offset"][:, -1], real_pos,
        atol=1e-5, rtol=1e-5)
    rotation_dot = torch.abs(torch.sum(
        reconstructed["root_rot"][:, -1] * real_rot, dim=-1))
    torch.testing.assert_close(
        rotation_dot, torch.ones_like(rotation_dot), atol=1e-5, rtol=1e-5)


def test_tracking_timestamps_select_consumed_frame():
    state = SimpleNamespace(
        tracked_plan_start_t_ns=1_000_000_000,
        publish_t_ns=1_061_000_000,
    )
    assert tracked_frame_from_timestamps(state, fps=50.0, future_len=8) == 3
    state.publish_t_ns = 2_000_000_000
    assert tracked_frame_from_timestamps(state, fps=50.0, future_len=8) == 7


def test_generated_history_ends_at_tracked_future_frame():
    features = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1)
    root_pos = torch.zeros((1, 10, 3), dtype=torch.float32)
    root_pos[0, :, 0] = torch.arange(10, dtype=torch.float32)
    root_rot = torch.zeros((1, 10, 4), dtype=torch.float32)
    root_rot[..., 3] = 1.0
    plan = {"features": features, "root_pos": root_pos, "root_rot": root_rot}

    history, abs_pose, reference_pos, _ = generated_history_at_frame(
        plan, tracked_frame=0, history_len=2)
    torch.testing.assert_close(history.flatten(), torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(
        abs_pose["root_trans_offset"], torch.tensor([[1.0, 0.0, 0.0]]))
    torch.testing.assert_close(reference_pos, torch.tensor([[2.0, 0.0, 0.0]]))

    history, abs_pose, reference_pos, _ = generated_history_at_frame(
        plan, tracked_frame=3, history_len=2)
    torch.testing.assert_close(history.flatten(), torch.tensor([4.0, 5.0]))
    torch.testing.assert_close(
        abs_pose["root_trans_offset"], torch.tensor([[4.0, 0.0, 0.0]]))
    torch.testing.assert_close(reference_pos, torch.tensor([[5.0, 0.0, 0.0]]))


def test_g1_packet_keeps_seam_and_maps_sonic_tracking_bodies():
    frames = 10
    dof_pos = torch.zeros((1, frames, 29), dtype=torch.float32)
    dof_pos[0, :, 0] = torch.arange(frames, dtype=torch.float32) * 0.02
    dof_pos[0, :, 19:22] = torch.tensor([0.1, 0.2, 0.3])
    dof_pos[0, :, 26:29] = torch.tensor([-0.1, -0.2, -0.3])
    body_pos = torch.zeros((1, frames, 33, 3), dtype=torch.float32)
    body_pos[0, :, 15, 0] = 15.0
    body_pos[0, :, 24, 0] = 24.0
    body_pos[0, :, 25, 0] = 25.0
    body_ori = torch.zeros((1, frames, 33, 4), dtype=torch.float32)
    body_ori[..., 3] = 1.0
    motion = motion_dict_to_g1data({
        "dof_pos": dof_pos,
        "global_translation_extend": body_pos,
        "global_rotation_extend": body_ori,
    }, skip_history=1, fps=50.0)

    assert motion.num_frames == 9
    assert motion.joint_pos.shape == (9, 29)
    assert motion.joint_vel.shape == (9, 29)
    assert motion.body_pos.shape == (9, 30, 3)
    np.testing.assert_array_equal(
        motion.body_pos, np.repeat(body_pos.numpy()[0, 1:, :1], 30, axis=1))
    np.testing.assert_array_equal(
        motion.joint_pos, mujoco_to_isaaclab_dof(dof_pos.numpy()[0, 1:]))
    np.testing.assert_allclose(motion.joint_vel[:, 0], 1.0, atol=1e-6)
    np.testing.assert_array_equal(
        motion.body_ori[0, 0], np.asarray([1.0, 0.0, 0.0, 0.0]))
