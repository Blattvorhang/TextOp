from types import SimpleNamespace

import numpy as np
import torch

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


class IdentityNormalization:
    @staticmethod
    def normalize(feature):
        return feature


def test_joint_order_round_trip():
    isaaclab = np.arange(29, dtype=np.float32).reshape(1, 29)
    mujoco = isaaclab_to_mujoco_dof(isaaclab)
    np.testing.assert_array_equal(mujoco_to_isaaclab_dof(mujoco), isaaclab)


def test_three_physical_states_produce_two_wrapped_features():
    yaws = np.asarray([0.0, 1.0, 3.13, -3.13, -3.0], dtype=np.float32)
    rotations = np.zeros((5, 4), dtype=np.float32)
    rotations[:, 2] = np.sin(yaws / 2.0)
    rotations[:, 3] = np.cos(yaws / 2.0)
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

    assert feature.shape == (1, 2, 57)
    np.testing.assert_allclose(
        abs_pose["root_trans_offset"].numpy(), positions[-3:].reshape(1, 3, 3)[:, 0])
    assert abs(float(feature[0, 0, 4])) < 0.03
    np.testing.assert_allclose(feature[0, 1, 4].numpy(), 0.13, atol=1e-5)


def test_goal_uses_last_history_feature_reference():
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
        goal.numpy(), [[1.0, 0.0, 0.0, np.cos(0.5), np.sin(0.5)]],
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
    })

    aligned_pose, aligned_endpoint, translation = align_generated_history_pose(
        abs_pose, generated_endpoint, state, "cpu")

    torch.testing.assert_close(translation, torch.tensor([[8.0, 16.0, 0.1]]))
    torch.testing.assert_close(
        aligned_pose["root_trans_offset"], torch.tensor([[9.0, 18.0, 0.8]]))
    torch.testing.assert_close(
        aligned_pose["root_rot"], abs_pose["root_rot"])
    torch.testing.assert_close(
        aligned_endpoint, torch.tensor([[10.0, 20.0, 0.9]]))
    torch.testing.assert_close(
        abs_pose["root_trans_offset"], torch.tensor([[1.0, 2.0, 0.7]]))


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


def test_g1_packet_has_eight_frames_and_50hz_velocity():
    frames = 10
    dof_pos = torch.zeros((1, frames, 23), dtype=torch.float32)
    dof_pos[0, :, 0] = torch.arange(frames, dtype=torch.float32) * 0.02
    body_pos = torch.zeros((1, frames, 2, 3), dtype=torch.float32)
    body_ori = torch.zeros((1, frames, 2, 4), dtype=torch.float32)
    body_ori[..., 3] = 1.0
    motion = motion_dict_to_g1data({
        "dof_pos": dof_pos,
        "global_translation": body_pos,
        "global_rotation": body_ori,
    }, skip_history=2, fps=50.0)

    assert motion.num_frames == 8
    assert motion.joint_pos.shape == (8, 29)
    assert motion.joint_vel.shape == (8, 29)
    np.testing.assert_allclose(motion.joint_vel[:, 0], 1.0, atol=1e-6)
    np.testing.assert_array_equal(
        motion.body_ori[0, 0], np.asarray([1.0, 0.0, 0.0, 0.0]))
