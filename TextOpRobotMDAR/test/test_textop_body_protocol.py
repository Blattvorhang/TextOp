import json

import numpy as np

from sonicmsg.messages import (
    TextOpHistoryStateHeader,
    decode_history_state_buffers,
    send_textop_history_state,
)
from sonicmsg.planner_node import StateMessage


class CapturingSocket:
    def __init__(self):
        self.frames = None

    def send_multipart(self, frames, flags=0):
        self.frames = frames


def _send(goal_type="root", goal_root_pos_world=None, goal_keypoints_world=None,
          ego_occ=None, goal_root_velocity_world=None, goal_timestamp_ns=None,
          goal_root_rot_world=None, goal_dof_pos=None,
          goal_root_valid=True, goal_yaw_valid=True, goal_body_valid=True,
          goal_orientation_valid=True, goal_joint_valid=True,
          goal_velocity_valid=True, goal_time_valid=True, scene_valid=True):
    socket = CapturingSocket()
    goal_yaw_world = (
        None if goal_type == "joint_state"
        else np.asarray([0.5], dtype=np.float32)
    )
    if not goal_yaw_valid:
        goal_yaw_world = None
    send_textop_history_state(
        socket,
        g1_pos=np.zeros((3, 3), dtype=np.float32),
        g1_root_rot=np.tile(
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (3, 1)),
        g1_joint_pos=np.zeros((3, 29), dtype=np.float32),
        goal_root_pos_world=(
            np.asarray([2.0, 3.0, 0.77], dtype=np.float32)
            if goal_root_pos_world is None and goal_root_valid
            else goal_root_pos_world),
        goal_yaw_world=goal_yaw_world,
        goal_type=goal_type,
        goal_root_velocity_world=goal_root_velocity_world,
        goal_timestamp_ns=goal_timestamp_ns,
        goal_keypoints_world=goal_keypoints_world,
        goal_root_rot_world=goal_root_rot_world,
        goal_dof_pos=goal_dof_pos,
        ego_occ=ego_occ,
        timestamps_ns=[1, 2, 3],
        publish_t_ns=3,
        seq=4,
        n_states=3,
        tracked_plan_seq=-1,
        tracked_plan_start_t_ns=0,
        goal_root_valid=goal_root_valid,
        goal_yaw_valid=goal_yaw_valid,
        goal_body_valid=goal_body_valid,
        goal_orientation_valid=goal_orientation_valid,
        goal_joint_valid=goal_joint_valid,
        goal_velocity_valid=goal_velocity_valid,
        goal_time_valid=goal_time_valid,
        scene_valid=scene_valid,
    )
    header = TextOpHistoryStateHeader.model_validate(
        json.loads(socket.frames[0].decode("utf-8")))
    return header, socket.frames[1:]


def test_root_goal_uses_protocol_v8_world_fields():
    occupancy = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
    header, buffers = _send(ego_occ=occupancy)

    assert header.protocol_version == 8
    assert header.goal_type == "root"
    assert header.goal_keypoints_world is None
    assert len(buffers) == 6
    decoded = decode_history_state_buffers(header, buffers)
    np.testing.assert_array_equal(
        decoded["g1_root_rot"][0],
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(decoded["ego_occ"], occupancy)
    assert "goal_keypoints_world" not in decoded


def test_body_goal_round_trips_protocol_v8_world_keypoints():
    keypoints = np.arange(15, dtype=np.float32).reshape(5, 3)
    occupancy = np.asarray([0.0, 1.0], dtype=np.float32)
    header, buffers = _send(
        goal_type="body", goal_keypoints_world=keypoints, ego_occ=occupancy)

    assert header.protocol_version == 8
    assert header.goal_type == "body"
    assert header.goal_keypoints_world is not None
    assert len(buffers) == 7
    decoded = decode_history_state_buffers(header, buffers)
    np.testing.assert_array_equal(decoded["goal_keypoints_world"], keypoints)
    np.testing.assert_array_equal(decoded["ego_occ"], occupancy)


def test_extended_body_goal_round_trips_protocol_v8_world_fields():
    keypoints = np.arange(12, dtype=np.float32).reshape(4, 3)
    velocity = np.asarray([1.0, -2.0, 0.5], dtype=np.float32)
    occupancy = np.asarray([1.0, 0.0], dtype=np.float32)

    header, buffers = _send(
        goal_type="body_ext",
        goal_keypoints_world=keypoints,
        goal_root_velocity_world=velocity,
        goal_timestamp_ns=2_500_000_003,
        ego_occ=occupancy,
    )

    assert header.protocol_version == 8
    assert header.goal_type == "body_ext"
    assert header.goal_timestamp_ns == 2_500_000_003
    assert len(buffers) == 8
    decoded = decode_history_state_buffers(header, buffers)
    np.testing.assert_array_equal(
        decoded["goal_root_velocity_world"], velocity)
    np.testing.assert_array_equal(decoded["goal_keypoints_world"], keypoints)
    np.testing.assert_array_equal(decoded["ego_occ"], occupancy)

    state = StateMessage(header, decoded)
    np.testing.assert_array_equal(state.goal_root_velocity_world, velocity)
    assert state.goal_timestamp_ns == 2_500_000_003


def test_joint_state_goal_round_trips_protocol_v8_wxyz_quaternion():
    velocity = np.asarray([0.1, 0.2, 0.0], dtype=np.float32)
    goal_root_rot_world = np.asarray(
        [0.9238795, 0.0, 0.0, 0.3826834], dtype=np.float32)
    goal_dof_pos = np.linspace(-0.2, 0.2, 29, dtype=np.float32)

    header, buffers = _send(
        goal_type="joint_state",
        goal_root_velocity_world=velocity,
        goal_timestamp_ns=2_500_000_003,
        goal_root_rot_world=goal_root_rot_world,
        goal_dof_pos=goal_dof_pos,
    )

    assert header.protocol_version == 8
    assert header.goal_type == "joint_state"
    assert header.goal_yaw_world is None
    assert len(buffers) == 7
    decoded = decode_history_state_buffers(header, buffers)
    np.testing.assert_array_equal(
        decoded["goal_root_rot_world"], goal_root_rot_world)
    np.testing.assert_array_equal(decoded["goal_dof_pos"], goal_dof_pos)
    state = StateMessage(header, decoded)
    np.testing.assert_allclose(state.current_heading_rad, 0.0, atol=1e-6)
    np.testing.assert_array_equal(
        state.goal_root_rot_world, goal_root_rot_world)


def test_root_goal_allows_missing_root_and_yaw_when_invalid():
    header, buffers = _send(
        goal_type="root",
        goal_root_pos_world=None,
        goal_root_valid=False,
        goal_yaw_valid=False,
        scene_valid=False,
    )

    assert header.protocol_version == 8
    assert header.goal_root_valid is False
    assert header.goal_yaw_valid is False
    assert header.scene_valid is False
    assert header.goal_root_pos_world is None
    assert header.goal_yaw_world is None
    assert len(buffers) == 3
    decoded = decode_history_state_buffers(header, buffers)
    assert "goal_root_pos_world" not in decoded
    assert "goal_yaw_world" not in decoded
    state = StateMessage(header, decoded)
    assert state.goal_root_pos_world is None
    assert state.goal_yaw_world is None
    assert state.goal_root_valid is False
    assert state.goal_yaw_valid is False
    assert state.scene_valid is False


def test_joint_state_goal_allows_missing_condition_blocks_when_invalid():
    header, buffers = _send(
        goal_type="joint_state",
        goal_root_velocity_world=None,
        goal_timestamp_ns=None,
        goal_root_rot_world=None,
        goal_dof_pos=None,
        goal_orientation_valid=False,
        goal_joint_valid=False,
        goal_velocity_valid=False,
        goal_time_valid=False,
        scene_valid=False,
    )

    assert header.protocol_version == 8
    assert header.goal_orientation_valid is False
    assert header.goal_joint_valid is False
    assert header.goal_velocity_valid is False
    assert header.goal_time_valid is False
    assert len(buffers) == 4
    decoded = decode_history_state_buffers(header, buffers)
    state = StateMessage(header, decoded)
    assert state.goal_root_rot_world is None
    assert state.goal_dof_pos is None
    assert state.goal_root_velocity_world is None
    assert state.goal_timestamp_ns is None
    assert state.goal_orientation_valid is False
    assert state.goal_joint_valid is False
    assert state.goal_velocity_valid is False
    assert state.goal_time_valid is False


def test_extended_body_goal_requires_velocity_timestamp_and_four_keypoints():
    keypoints = np.zeros((4, 3), dtype=np.float32)
    velocity = np.zeros(3, dtype=np.float32)

    with np.testing.assert_raises_regex(
            ValueError, "goal_root_velocity_world"):
        _send(
            goal_type="body_ext", goal_keypoints_world=keypoints,
            goal_timestamp_ns=10)
    with np.testing.assert_raises_regex(ValueError, "goal_timestamp_ns"):
        _send(
            goal_type="body_ext", goal_keypoints_world=keypoints,
            goal_root_velocity_world=velocity)
    with np.testing.assert_raises_regex(ValueError, r"\(4, 3\)"):
        _send(
            goal_type="body_ext", goal_keypoints_world=np.zeros((5, 3)),
            goal_root_velocity_world=velocity, goal_timestamp_ns=10)
