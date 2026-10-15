"""Cursor-aligned physical feedback on a separately advanced history window."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

import robotmdar.dtype.motion as motion_dtype
from robotmdar.dtype.rotation import quaternion_to_matrix, xyzw_to_wxyz
from robotmdar.utils.planner_convert import (
    match_reference_history_phase,
    mujoco_to_isaaclab_dof,
    reanchor_reference_motion,
    reference_motion_to_history_repr,
)


@pytest.fixture(autouse=True)
def feature_version():
    previous = motion_dtype.FeatureVersion
    motion_dtype.set_feature_version(6)
    yield
    motion_dtype.set_feature_version(previous)


def _snapshot(cursor=5):
    frames = np.arange(24, dtype=np.float32)
    positions = np.stack((0.2 * frames, -0.1 * frames, 0.7 + 0.01 * frames), -1)
    # Reference tilt changes over time, so using a future frame for the
    # gravity gate would give a different weight even at zero tracking error.
    euler = np.stack((0.02 * frames, 0.01 * frames, 0.03 * frames), -1)
    rotations = Rotation.from_euler("xyz", euler).as_quat().astype(np.float32)
    joints = np.zeros((24, 29), dtype=np.float32)
    joints[:, 0] = 0.03 * frames
    return SimpleNamespace(
        root_pos=positions,
        root_ori=rotations[:, [3, 0, 1, 2]],
        joint_pos=mujoco_to_isaaclab_dof(joints),
        cursor_frame=cursor,
    )


def _measured(snapshot, delta_rot, frame=None):
    frame = snapshot.cursor_frame if frame is None else frame
    reference_rot = Rotation.from_quat(snapshot.root_ori[frame, [1, 2, 3, 0]])
    real_rot = (delta_rot * reference_rot).as_quat()
    return SimpleNamespace(states=SimpleNamespace(
        g1_pos=snapshot.root_pos[frame:frame + 1] + [[2.0, -3.0, -0.25]],
        g1_root_rot=real_rot[None, [3, 0, 1, 2]],
        g1_joint_pos=snapshot.joint_pos[frame:frame + 1] + 0.12,
    ))


def _reanchor(snapshot, state, lead=0, dof_dim=29, **kwargs):
    dataset = SimpleNamespace(normalize=lambda value: value, dof_dim=dof_dim)
    history = reference_motion_to_history_repr(
        snapshot, snapshot.cursor_frame + lead, 4, dataset, "cpu")
    limits = (torch.full((dof_dim,), -10.0), torch.full((dof_dim,), 10.0))
    kwargs.setdefault("reanchor_components", _component_modes())
    result = reanchor_reference_motion(
        history, state, dataset, "cpu", joint_limits=limits,
        reanchor_mode=kwargs.pop("reanchor_mode", "decoupled_world"), **kwargs)
    return history, result


def _component_modes(**overrides):
    weighting = {
        "hor_pos": "full",
        "vert_pos": "full",
        "orientation": "full",
        "joint": "gravity_weighted",
    }
    weighting.update(overrides)
    return weighting


@pytest.mark.parametrize("lead", [0, 2, 10])
@pytest.mark.parametrize("dof_dim", [23, 29])
def test_cursor_gravity_and_full_root_feedback_independent_of_window_end(lead, dof_dim):
    snapshot = _snapshot()
    original_pos = snapshot.root_pos.copy()
    # A left rotation around world Z leaves body-frame gravity unchanged.
    delta = Rotation.from_euler("z", 0.8)
    state = _measured(snapshot, delta)
    history, result = _reanchor(snapshot, state, lead=lead, dof_dim=dof_dim)
    assert history["window_end"] == snapshot.cursor_frame + lead
    assert result["anchor_frame"] == snapshot.cursor_frame
    if lead == 10:
        assert snapshot.cursor_frame < history["window_start"]
    assert result["correction"]["feedback_weight"].item() == pytest.approx(0, abs=2e-7)
    hybrid = result["hybrid_motion"]
    expected_pos = history["root_pos"] + torch.tensor([2.0, -3.0, -0.25])
    torch.testing.assert_close(hybrid["root_trans_offset"], expected_pos)
    torch.testing.assert_close(hybrid["dof"], history["dof"])
    expected_rot = (torch.tensor(delta.as_matrix(), dtype=torch.float32)
                    @ quaternion_to_matrix(xyzw_to_wxyz(history["root_rot"])))
    torch.testing.assert_close(
        quaternion_to_matrix(xyzw_to_wxyz(hybrid["root_rot"])), expected_rot,
        atol=2e-6, rtol=2e-6)
    # The future seam retains the cursor->seam reference displacement; it
    # must not be collapsed back to the current measured XYZ position.
    end_frame = history["window_start"] + 4
    expected_xyz = (
        state.states.g1_pos[0]
        + snapshot.root_pos[end_frame]
        - snapshot.root_pos[snapshot.cursor_frame])
    np.testing.assert_allclose(
        result["seam_abs_pose"]["root_trans_offset"][0], expected_xyz, atol=2e-6)
    # Full yaw feedback reexpresses position deltas in the corrected body
    # frames. Encoding and decoding must still preserve the world trajectory.
    decoded = motion_dtype.motion_feature_to_dict(
        result["history_repr"], result["abs_pose"])
    torch.testing.assert_close(decoded["root_trans_offset"], expected_pos[:, 1:])
    torch.testing.assert_close(
        quaternion_to_matrix(xyzw_to_wxyz(decoded["root_rot"])), expected_rot[:, 1:],
        atol=2e-6, rtol=2e-6)
    np.testing.assert_array_equal(snapshot.root_pos, original_pos)


@pytest.mark.parametrize("tilt_deg", [0.0, 0.00001, 30.0, 60.0, 90.0, 120.0, 180.0])
@pytest.mark.parametrize("angle_source", ["reference", "world"])
def test_weighted_joints_and_full_orientation_feedback_match_scipy(tilt_deg, angle_source):
    snapshot = _snapshot()
    # Use an upright cursor so its tilt error is known analytically, with a
    # simultaneous yaw error to check full correction of the complete delta.
    snapshot.root_ori[snapshot.cursor_frame] = [1, 0, 0, 0]
    delta = Rotation.from_euler("z", 0.7) * Rotation.from_euler("x", tilt_deg, degrees=True)
    state = _measured(snapshot, delta)
    history, result = _reanchor(
        snapshot, state, lead=10,
        reanchor_components=_component_modes(), gravity_angle_source=angle_source)
    expected_weight = np.clip(1 - np.cos(np.deg2rad(tilt_deg)), 0, 1)
    assert result["correction"]["feedback_weight"].item() == pytest.approx(
        expected_weight, abs=2e-7)
    expected_delta = delta.as_matrix()
    raw_rot = quaternion_to_matrix(xyzw_to_wxyz(history["root_rot"]))
    expected_rot = torch.tensor(expected_delta, dtype=torch.float32) @ raw_rot
    hybrid = result["hybrid_motion"]
    torch.testing.assert_close(
        quaternion_to_matrix(xyzw_to_wxyz(hybrid["root_rot"])), expected_rot,
        atol=2e-6, rtol=2e-6)
    expected_pos = history["root_pos"] + torch.tensor(
        [2.0, -3.0, -0.25], dtype=torch.float32)
    torch.testing.assert_close(hybrid["root_trans_offset"], expected_pos)
    torch.testing.assert_close(hybrid["dof"], history["dof"] + 0.12 * expected_weight)
    # Verify the actual V6 encoding/decoding path, including abs_pose, rather
    # than only the intermediate physical arrays.
    decoded = motion_dtype.motion_feature_to_dict(
        result["history_repr"], result["abs_pose"])
    torch.testing.assert_close(decoded["root_trans_offset"], expected_pos[:, 1:])
    torch.testing.assert_close(
        quaternion_to_matrix(xyzw_to_wxyz(decoded["root_rot"])), expected_rot[:, 1:],
        atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("real_tilt", [0.0, 60.0])
@pytest.mark.parametrize("angle_source", ["reference", "world"])
def test_angle_source_distinguishes_reference_error_from_actual_tilt(real_tilt, angle_source):
    snapshot = _snapshot()
    reference_rotation = Rotation.from_euler("x", 60, degrees=True)
    snapshot.root_ori[snapshot.cursor_frame] = reference_rotation.as_quat()[[3, 0, 1, 2]]
    # World yaw must not change the upright/tilt measurement.
    real_rotation = Rotation.from_euler("z", 0.7) * Rotation.from_euler(
        "x", real_tilt, degrees=True)
    state = _measured(snapshot, real_rotation * reference_rotation.inv())
    history, result = _reanchor(
        snapshot, state, lead=10,
        reanchor_components=_component_modes(), gravity_angle_source=angle_source)
    expected_angle = abs(real_tilt - 60) if angle_source == "reference" else real_tilt
    expected_weight = 1 - np.cos(np.deg2rad(expected_angle))
    assert result["correction"]["gravity_angle_rad"].item() == pytest.approx(
        np.deg2rad(expected_angle), abs=2e-7)
    assert result["correction"]["feedback_weight"].item() == pytest.approx(
        expected_weight, abs=2e-7)
    torch.testing.assert_close(
        result["hybrid_motion"]["dof"], history["dof"] + 0.12 * expected_weight)


def test_reanchor_components_rejects_invalid_angle_source():
    snapshot = _snapshot()
    state = _measured(snapshot, Rotation.identity())
    with pytest.raises(ValueError, match="gravity_angle_source must be reference or world"):
        _reanchor(snapshot, state,
                  reanchor_components=_component_modes(), gravity_angle_source="invalid")


def test_reanchor_components_components_are_independent():
    snapshot = _snapshot()
    delta = Rotation.from_euler("xyz", [0.6, 0.5, 0.7])
    state = _measured(snapshot, delta)
    history, result = _reanchor(
        snapshot, state, lead=2,
        reanchor_components=_component_modes(
            hor_pos="gravity_weighted", vert_pos="gravity_weighted", orientation="gravity_weighted", joint="gravity_weighted"))
    weight = result["correction"]["feedback_weight"].item()
    hybrid = result["hybrid_motion"]
    expected_pos = history["root_pos"] + torch.tensor(
        [2.0 * weight, -3.0 * weight, -0.25 * weight])
    torch.testing.assert_close(hybrid["root_trans_offset"], expected_pos)
    torch.testing.assert_close(hybrid["dof"], history["dof"] + 0.12 * weight)

    _, xy_full = _reanchor(
        snapshot, state, lead=2,
        reanchor_components=_component_modes(hor_pos="full"))
    _, xy_weighted = _reanchor(
        snapshot, state, lead=2,
        reanchor_components=_component_modes(hor_pos="gravity_weighted"))
    torch.testing.assert_close(
        xy_full["hybrid_motion"]["root_trans_offset"][..., :2],
        history["root_pos"][..., :2] + torch.tensor([2.0, -3.0]))
    torch.testing.assert_close(
        xy_weighted["hybrid_motion"]["root_trans_offset"][..., :2],
        history["root_pos"][..., :2] + torch.tensor([2.0, -3.0]) * weight)


@pytest.mark.parametrize("mode", ["decoupled_world", "se3_left"])
def test_full_feedback_can_be_selected_and_se3_remains_unweighted(mode):
    snapshot = _snapshot()
    delta = Rotation.from_euler("xyz", [0.2, 0.1, 0.5])
    state = _measured(snapshot, delta)
    history, result = _reanchor(
        snapshot, state, lead=2, reanchor_mode=mode,
        reanchor_components=_component_modes(
            hor_pos="full", vert_pos="full", orientation="full", joint="full"))
    _, weighted = _reanchor(
        snapshot, state, lead=2, reanchor_mode=mode,
        reanchor_components=_component_modes())
    hybrid = result["hybrid_motion"]
    reference_anchor = torch.tensor(snapshot.root_pos[snapshot.cursor_frame])
    real_pos = torch.as_tensor(state.states.g1_pos[0], dtype=torch.float32)
    if mode == "decoupled_world":
        expected = history["root_pos"] + real_pos - reference_anchor
        for key in ("root_trans_offset", "root_rot"):
            torch.testing.assert_close(weighted["hybrid_motion"][key], hybrid[key])
    else:
        expected = (torch.tensor(delta.as_matrix(), dtype=torch.float32)
                    @ (history["root_pos"] - reference_anchor).unsqueeze(-1)).squeeze(-1) + real_pos
        for key in hybrid:
            torch.testing.assert_close(weighted["hybrid_motion"][key], hybrid[key])
    torch.testing.assert_close(hybrid["root_trans_offset"], expected)
    torch.testing.assert_close(hybrid["dof"], history["dof"] + 0.12)


def test_reanchor_components_rejects_legacy_boolean():
    snapshot = _snapshot()
    state = _measured(snapshot, Rotation.identity())
    with pytest.raises(TypeError, match="reanchor_components must be a mapping"):
        _reanchor(snapshot, state, reanchor_components=True)


@pytest.mark.parametrize("modes", list(product(
    ("open_loop", "full", "gravity_weighted"), repeat=4)))
def test_all_component_mode_combinations(modes):
    snapshot = _snapshot()
    delta = Rotation.from_euler("xyz", [0.6, 0.5, 0.7])
    state = _measured(snapshot, delta)
    components = dict(zip(("hor_pos", "vert_pos", "orientation", "joint"), modes))
    history, result = _reanchor(snapshot, state, lead=10,
                                reanchor_components=components)
    weight = result["correction"]["feedback_weight"].item()
    weights = [0 if mode == "open_loop" else 1 if mode == "full" else weight
               for mode in modes]
    hybrid = result["hybrid_motion"]
    torch.testing.assert_close(
        hybrid["root_trans_offset"], history["root_pos"] + torch.tensor(
            [2 * weights[0], -3 * weights[0], -0.25 * weights[1]]))
    torch.testing.assert_close(hybrid["dof"], history["dof"] + 0.12 * weights[3])
    # Independent SO(3) expectation from scipy's rotation-vector scaling.
    expected_delta = Rotation.from_rotvec(delta.as_rotvec() * weights[2]).as_matrix()
    expected_rot = (torch.tensor(expected_delta, dtype=torch.float32)
                    @ quaternion_to_matrix(xyzw_to_wxyz(history["root_rot"])))
    torch.testing.assert_close(
        quaternion_to_matrix(xyzw_to_wxyz(hybrid["root_rot"])), expected_rot,
        atol=2e-6, rtol=2e-6)
    # The goal/generation seam must follow the hybrid endpoint in every mode.
    torch.testing.assert_close(
        result["seam_abs_pose"]["root_trans_offset"], hybrid["root_trans_offset"][:, -1])


@pytest.mark.parametrize("invalid_mode", [True, False, "invalid"])
def test_reanchor_components_reject_invalid_modes(invalid_mode):
    snapshot = _snapshot()
    state = _measured(snapshot, Rotation.identity())
    with pytest.raises(ValueError, match="entries must be open_loop, full, or gravity_weighted"):
        _reanchor(snapshot, state, reanchor_components=_component_modes(joint=invalid_mode))


def test_open_loop_joints_retain_reference_without_additional_clipping():
    snapshot = _snapshot()
    snapshot.joint_pos += 20  # Outside this test's artificial [-10, 10] limits.
    state = _measured(snapshot, Rotation.identity())
    history, result = _reanchor(
        snapshot, state, reanchor_components=_component_modes(joint="open_loop"))
    torch.testing.assert_close(result["hybrid_motion"]["dof"], history["dof"])


def test_phase_match_addresses_raw_state_without_extra_frame_shift():
    snapshot = _snapshot()
    dataset = SimpleNamespace(normalize=lambda value: value)
    history = reference_motion_to_history_repr(snapshot, 10, 4, dataset, "cpu")
    local_index = 1
    matched_frame = history["window_start"] + local_index
    state = _measured(snapshot, Rotation.identity(), frame=matched_frame)
    state.states.g1_joint_pos = snapshot.joint_pos[matched_frame:matched_frame + 1]
    phase, _ = match_reference_history_phase(history["dof"], state, "cpu")
    assert phase.item() == local_index
    result = reanchor_reference_motion(
        history, state, dataset, "cpu",
        joint_limits=(torch.full((29,), -10.0), torch.full((29,), 10.0)),
        anchor_state_index=phase, reanchor_mode="decoupled_world",
        reanchor_components=_component_modes())
    assert result["anchor_frame"] == matched_frame
    assert result["correction"]["feedback_weight"].item() == pytest.approx(0, abs=2e-7)
    torch.testing.assert_close(
        result["hybrid_motion"]["root_trans_offset"][:, local_index],
        torch.as_tensor(state.states.g1_pos, dtype=torch.float32))


@pytest.mark.parametrize("cursor,lead,expected_end", [(0, 0, 4), (5, 100, 23), (23, 0, 23)])
def test_generation_window_boundaries_keep_original_tracking_cursor(cursor, lead, expected_end):
    snapshot = _snapshot(cursor=cursor)
    state = _measured(snapshot, Rotation.identity())
    history, result = _reanchor(snapshot, state, lead=lead)
    assert history["window_end"] == expected_end
    assert history["root_pos"].shape[1] == 5
    assert result["anchor_frame"] == cursor
    torch.testing.assert_close(result["hybrid_motion"]["root_trans_offset"],
                               history["root_pos"] + torch.tensor([2.0, -3.0, -0.25]))
