import json

import numpy as np

from sonicmsg.messages import (
    Condition,
    ConditionValidity,
    EndEffectorValidity,
    GoalData,
    GoalValidity,
    HistoryMetadata,
    HistoryState,
    HistoryStateData,
    HistoryStateHeader,
    PlanTracking,
    decode_history_state_buffers,
    send_history_state,
)


class CapturingSocket:
    def __init__(self):
        self.frames = None

    def send_multipart(self, frames, flags=0):
        self.frames = frames


def _send(
        goal_type="root",
        goal_root_pos_world=None,
        goal_keypoints_world=None,
        ego_occ=None,
        goal_root_velocity_world=None,
        goal_timestamp_ns=None,
        goal_root_rot_world=None,
        goal_dof_pos=None,
        goal_end_effectors_world=None,
        text=None,
        text_valid=False,
        goal_position_hor_valid=True,
        goal_position_vert_valid=True,
        goal_yaw_valid=True,
        goal_body_valid=True,
        goal_orientation_valid=True,
        goal_joint_valid=True,
        goal_velocity_valid=True,
        goal_time_valid=True,
        goal_end_effector_valid=True,
        goal_end_effector_left_hand_valid=True,
        goal_end_effector_right_hand_valid=True,
        goal_end_effector_left_foot_valid=True,
        goal_end_effector_right_foot_valid=True,
        scene_valid=True):
    socket = CapturingSocket()
    goal = GoalData(
        goal_type=goal_type,
        root_pos_world=(
            np.asarray([2.0, 3.0, 0.77], dtype=np.float32)
            if (goal_root_pos_world is None
                and (goal_position_hor_valid or goal_position_vert_valid))
            else goal_root_pos_world),
        yaw_world=(
            None
            if goal_type == "joint_state" or not goal_yaw_valid
            else np.asarray([0.5], dtype=np.float32)),
        root_velocity_world=goal_root_velocity_world,
        timestamp_ns=goal_timestamp_ns,
        keypoints_world=goal_keypoints_world,
        root_rot_world=goal_root_rot_world,
        dof_pos=goal_dof_pos,
        end_effectors_world=goal_end_effectors_world,
    )
    validity = ConditionValidity(
        goal=GoalValidity(
            position_hor=goal_position_hor_valid,
            position_vert=goal_position_vert_valid,
            yaw=goal_yaw_valid,
            body=goal_body_valid,
            orientation=goal_orientation_valid,
            joint=goal_joint_valid,
            velocity=goal_velocity_valid,
            time=goal_time_valid,
            end_effector=EndEffectorValidity(
                valid=goal_end_effector_valid,
                left_hand=goal_end_effector_left_hand_valid,
                right_hand=goal_end_effector_right_hand_valid,
                left_foot=goal_end_effector_left_foot_valid,
                right_foot=goal_end_effector_right_foot_valid,
            ),
        ),
        scene=scene_valid,
        text=text_valid,
    )
    message = HistoryState(
        history_meta=HistoryMetadata(
            timestamps_ns=[1, 2, 3],
            publish_t_ns=3,
            seq=4,
            n_states=3,
        ),
        tracking=PlanTracking(seq=-1, start_t_ns=0),
        states=HistoryStateData(
            g1_pos=np.zeros((3, 3), dtype=np.float32),
            g1_root_rot=np.tile(
                np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                (3, 1),
            ),
            g1_joint_pos=np.zeros((3, 29), dtype=np.float32),
        ),
        condition=Condition(
            goal=goal,
            text=text,
            ego_occ=ego_occ,
            valid=validity,
        ),
        planner="textop",
    )
    send_history_state(socket, message)
    header = HistoryStateHeader.model_validate(
        json.loads(socket.frames[0].decode("utf-8")))
    decoded = decode_history_state_buffers(header, socket.frames[1:])
    return header, decoded, decoded


def test_root_goal_uses_protocol_12_nested_world_fields():
    occupancy = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
    header, decoded, state = _send(ego_occ=occupancy)

    assert header.protocol_version == 12
    assert header.condition.goal.goal_type == "root"
    assert header.condition.goal.keypoints_world is None
    np.testing.assert_array_equal(
        decoded.states.g1_root_rot[0],
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(decoded.condition.ego_occ, occupancy)
    assert state.condition.goal.root_pos_world is not None


def test_controller_text_round_trips_with_nested_validity():
    header, decoded, _ = _send(
        text="walk toward the target and stop",
        text_valid=True,
    )

    assert header.protocol_version == 12
    assert header.condition.valid.text is True
    assert header.condition.text == "walk toward the target and stop"
    assert decoded.condition.valid.text is True
    assert decoded.condition.text == "walk toward the target and stop"


def test_invalid_controller_text_is_not_transmitted():
    header, decoded, _ = _send(text="stale text", text_valid=False)

    assert header.condition.valid.text is False
    assert header.condition.text is None
    assert decoded.condition.valid.text is False
    assert decoded.condition.text is None


def test_valid_controller_text_requires_text():
    with np.testing.assert_raises_regex(ValueError, "text is required"):
        _send(text_valid=True)


def test_body_goal_round_trips_nested_world_keypoints():
    keypoints = np.arange(15, dtype=np.float32).reshape(5, 3)
    occupancy = np.asarray([0.0, 1.0], dtype=np.float32)
    header, decoded, _ = _send(
        goal_type="body",
        goal_keypoints_world=keypoints,
        ego_occ=occupancy,
    )

    assert header.protocol_version == 12
    assert header.condition.goal.goal_type == "body"
    assert header.condition.goal.keypoints_world is not None
    np.testing.assert_array_equal(decoded.condition.goal.keypoints_world, keypoints)
    np.testing.assert_array_equal(decoded.condition.ego_occ, occupancy)


def test_extended_body_goal_round_trips_world_fields():
    keypoints = np.arange(12, dtype=np.float32).reshape(4, 3)
    velocity = np.asarray([1.0, -2.0, 0.5], dtype=np.float32)
    occupancy = np.asarray([1.0, 0.0], dtype=np.float32)

    header, decoded, _ = _send(
        goal_type="body_ext",
        goal_keypoints_world=keypoints,
        goal_root_velocity_world=velocity,
        goal_timestamp_ns=2_500_000_003,
        ego_occ=occupancy,
    )

    assert header.condition.goal.timestamp_ns == 2_500_000_003
    np.testing.assert_array_equal(
        decoded.condition.goal.root_velocity_world, velocity)
    np.testing.assert_array_equal(decoded.condition.goal.keypoints_world, keypoints)
    np.testing.assert_array_equal(decoded.condition.ego_occ, occupancy)


def test_joint_state_goal_round_trips_wxyz_quaternion_and_end_effectors():
    velocity = np.asarray([0.1, 0.2, 0.0], dtype=np.float32)
    goal_root_rot_world = np.asarray(
        [0.9238795, 0.0, 0.0, 0.3826834], dtype=np.float32)
    goal_dof_pos = np.linspace(-0.2, 0.2, 29, dtype=np.float32)
    goal_end_effectors_world = np.arange(12, dtype=np.float32).reshape(4, 3)

    header, decoded, _ = _send(
        goal_type="joint_state",
        goal_root_velocity_world=velocity,
        goal_timestamp_ns=2_500_000_003,
        goal_root_rot_world=goal_root_rot_world,
        goal_dof_pos=goal_dof_pos,
        goal_end_effectors_world=goal_end_effectors_world,
    )

    assert header.protocol_version == 12
    assert header.condition.goal.yaw_world is None
    assert header.condition.goal.end_effectors_world is not None
    np.testing.assert_array_equal(
        decoded.condition.goal.root_rot_world, goal_root_rot_world)
    np.testing.assert_array_equal(decoded.condition.goal.dof_pos, goal_dof_pos)
    np.testing.assert_array_equal(
        decoded.condition.goal.end_effectors_world,
        goal_end_effectors_world)


def test_joint_state_end_effector_validity_is_independent():
    end_effectors = np.arange(12, dtype=np.float32).reshape(4, 3)
    header, decoded, _ = _send(
        goal_type="joint_state",
        goal_root_velocity_world=np.zeros(3, dtype=np.float32),
        goal_timestamp_ns=10,
        goal_root_rot_world=np.asarray(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        goal_dof_pos=np.zeros(29, dtype=np.float32),
        goal_end_effectors_world=end_effectors,
        goal_end_effector_left_hand_valid=False,
    )

    validity = header.condition.valid.goal.end_effector
    assert validity.valid is True
    assert validity.left_hand is False
    assert validity.right_hand is True
    assert decoded.condition.valid.goal.end_effector.left_hand is False


def test_root_goal_allows_missing_fields_when_invalid():
    header, decoded, _ = _send(
        goal_type="root",
        goal_root_pos_world=None,
        goal_position_hor_valid=False,
        goal_position_vert_valid=False,
        goal_yaw_valid=False,
        scene_valid=False,
    )

    assert header.protocol_version == 12
    assert header.condition.valid.goal.position_hor is False
    assert header.condition.valid.goal.position_vert is False
    assert header.condition.valid.goal.yaw is False
    assert header.condition.valid.scene is False
    assert header.condition.goal.root_pos_world is None
    assert header.condition.goal.yaw_world is None
    assert decoded.condition.goal.root_pos_world is None
    assert decoded.condition.goal.yaw_world is None


def test_position_validity_round_trips_independently():
    header, decoded, _ = _send(
        goal_position_hor_valid=False,
        goal_position_vert_valid=True,
    )

    assert header.condition.valid.goal.position_hor is False
    assert header.condition.valid.goal.position_vert is True
    assert header.condition.goal.root_pos_world is not None
    assert decoded.condition.valid.goal.position_hor is False
    assert decoded.condition.valid.goal.position_vert is True


def test_joint_state_allows_missing_condition_blocks_when_invalid():
    header, decoded, _ = _send(
        goal_type="joint_state",
        goal_root_pos_world=None,
        goal_position_hor_valid=False,
        goal_position_vert_valid=False,
        goal_root_velocity_world=None,
        goal_timestamp_ns=None,
        goal_root_rot_world=None,
        goal_dof_pos=None,
        goal_orientation_valid=False,
        goal_joint_valid=False,
        goal_velocity_valid=False,
        goal_time_valid=False,
        goal_end_effector_valid=False,
        goal_end_effector_left_hand_valid=False,
        goal_end_effector_right_hand_valid=False,
        goal_end_effector_left_foot_valid=False,
        goal_end_effector_right_foot_valid=False,
        scene_valid=False,
    )

    assert header.protocol_version == 12
    assert decoded.condition.goal.root_rot_world is None
    assert decoded.condition.goal.dof_pos is None
    assert decoded.condition.goal.root_velocity_world is None
    assert decoded.condition.goal.timestamp_ns is None
    assert decoded.condition.valid.goal.orientation is False
    assert decoded.condition.valid.goal.joint is False


def test_extended_body_goal_requires_velocity_timestamp_and_keypoints():
    keypoints = np.zeros((4, 3), dtype=np.float32)
    velocity = np.zeros(3, dtype=np.float32)

    with np.testing.assert_raises_regex(
            ValueError, "goal.root_velocity_world"):
        _send(
            goal_type="body_ext",
            goal_keypoints_world=keypoints,
            goal_timestamp_ns=10,
        )
    with np.testing.assert_raises_regex(ValueError, "goal.timestamp_ns"):
        _send(
            goal_type="body_ext",
            goal_keypoints_world=keypoints,
            goal_root_velocity_world=velocity,
        )
    with np.testing.assert_raises_regex(ValueError, r"\(4, 3\)"):
        _send(
            goal_type="body_ext",
            goal_keypoints_world=np.zeros((5, 3)),
            goal_root_velocity_world=velocity,
            goal_timestamp_ns=10,
        )
