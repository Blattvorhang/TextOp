import json

import numpy as np

from sonicmsg.messages import (
    TextOpHistoryStateHeader,
    decode_history_state_buffers,
    send_textop_history_state,
)


class CapturingSocket:
    def __init__(self):
        self.frames = None

    def send_multipart(self, frames, flags=0):
        self.frames = frames


def _send(goal_type="root", goal_keypoints=None, ego_occ=None):
    socket = CapturingSocket()
    send_textop_history_state(
        socket,
        g1_pos=np.zeros((3, 3), dtype=np.float32),
        g1_root_rot=np.tile(
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (3, 1)),
        g1_joint_pos=np.zeros((3, 29), dtype=np.float32),
        goal_root_pos=np.asarray([2.0, 3.0, 0.77], dtype=np.float32),
        goal_heading=np.asarray([0.5], dtype=np.float32),
        goal_type=goal_type,
        goal_keypoints=goal_keypoints,
        ego_occ=ego_occ,
        timestamps_ns=[1, 2, 3],
        publish_t_ns=3,
        seq=4,
        n_states=3,
        tracked_plan_seq=-1,
        tracked_plan_start_t_ns=0,
    )
    header = TextOpHistoryStateHeader.model_validate(
        json.loads(socket.frames[0].decode("utf-8")))
    return header, socket.frames[1:]


def test_root_goal_keeps_protocol_v2_frame_layout():
    occupancy = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
    header, buffers = _send(ego_occ=occupancy)

    assert header.protocol_version == 2
    assert header.goal_type == "root"
    assert header.goal_keypoints is None
    assert len(buffers) == 6
    decoded = decode_history_state_buffers(header, buffers)
    np.testing.assert_array_equal(decoded["ego_occ"], occupancy)
    assert "goal_keypoints" not in decoded


def test_body_goal_round_trips_protocol_v3_keypoints_before_occupancy():
    keypoints = np.arange(15, dtype=np.float32).reshape(5, 3)
    occupancy = np.asarray([0.0, 1.0], dtype=np.float32)
    header, buffers = _send(
        goal_type="body", goal_keypoints=keypoints, ego_occ=occupancy)

    assert header.protocol_version == 3
    assert header.goal_type == "body"
    assert header.goal_keypoints is not None
    assert len(buffers) == 7
    decoded = decode_history_state_buffers(header, buffers)
    np.testing.assert_array_equal(decoded["goal_keypoints"], keypoints)
    np.testing.assert_array_equal(decoded["ego_occ"], occupancy)
