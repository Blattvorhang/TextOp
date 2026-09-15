from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import robotmdar.dtype.motion as runtime_motion_dtype
import TextOpRobotMDAR.robotmdar.dtype.motion as package_motion_dtype
from TextOpRobotMDAR.robotmdar.utils.planner_convert import (
    G1_ISAACLAB_DOF_JOINT_NAMES,
    align_generated_history_pose,
    generated_history_at_frame,
    residual_reanchor_generated_history,
    residual_reanchor_generated_plan_at_frame,
    isaaclab_to_mujoco_dof,
    motion_dict_to_g1data,
    mujoco_to_isaaclab_dof,
    state_goal_from_reference,
    state_to_ego_goal,
    state_to_model_input,
    tracked_frame_from_timestamps,
)
from TextOpRobotMDAR.robotmdar.dtype.motion import (
    G1_MUJOCO_DOF_JOINT_NAMES,
    G1_MUJOCO_DOF_LINK_NAMES,
    motion_dict_to_feature_v3,
    motion_dict_to_feature_v6,
    motion_feature_to_dict_v3,
    motion_feature_to_dict_v6,
)
from TextOpRobotMDAR.robotmdar.dtype.rotation import (
    euler_angles_to_quaternion,
    quaternion_to_matrix,
    xyzw_to_wxyz,
)
from TextOpRobotMDAR.robotmdar.skeleton.robot import RobotSkeleton
from TextOpRobotMDAR.robotmdar.utils.goal import GoalEncoding


class IdentityNormalization:
    @staticmethod
    def normalize(feature):
        return feature

    @staticmethod
    def denormalize(feature):
        return feature


class IdentityNormalization23(IdentityNormalization):
    dof_dim = 23


def _set_both_feature_versions(version: int):
    old_runtime = runtime_motion_dtype.FeatureVersion
    old_package = package_motion_dtype.FeatureVersion
    runtime_motion_dtype.set_feature_version(version)
    package_motion_dtype.set_feature_version(version)
    return old_runtime, old_package


def _xyzw_to_wxyz_np(values):
    return np.ascontiguousarray(np.asarray(values)[..., [3, 0, 1, 2]])


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
    expected = np.asarray([
        G1_ISAACLAB_DOF_JOINT_NAMES.index(name)
        for name in G1_MUJOCO_DOF_JOINT_NAMES
    ], dtype=np.float32).reshape(1, 29)
    np.testing.assert_array_equal(mujoco, expected)
    np.testing.assert_array_equal(mujoco_to_isaaclab_dof(mujoco), isaaclab)


def test_training_fk_uses_canonical_mujoco_joint_and_body_order():
    project_root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(project_root / "robotmdar/config/skeleton/g1.yaml")
    cfg.asset.assetRoot = str(project_root / "description/robots/g1")
    skeleton = RobotSkeleton(device="cpu", cfg=cfg)

    assert tuple(skeleton.fk.dof_joint_names) == G1_MUJOCO_DOF_JOINT_NAMES
    assert tuple(skeleton.fk.body_names[1:]) == G1_MUJOCO_DOF_LINK_NAMES


def test_feature_v3_preserves_canonical_joint_indices():
    current = torch.arange(29, dtype=torch.float32)
    following = current + 100.0
    motion = {
        "root_trans_offset": torch.zeros((1, 2, 3)),
        "root_rot": torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(1, 2, 4),
        "dof": torch.stack((current, following)).unsqueeze(0),
        "contact_mask": torch.ones((1, 2, 2)),
    }

    feature, _ = motion_dict_to_feature_v3(motion)

    torch.testing.assert_close(feature[0, 0, 11:40], current)
    torch.testing.assert_close(feature[0, 0, 40:69], following - current)


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
        "g1_root_rot": _xyzw_to_wxyz_np(rotations),
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


def test_controller_29dof_history_builds_legacy_57d_model_input():
    positions = np.asarray([
        [0.0, 0.0, 0.77],
        [0.1, 0.0, 0.77],
        [0.2, 0.0, 0.77],
    ], dtype=np.float32)
    rotations = np.asarray([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)
    joints = np.stack([
        np.arange(29, dtype=np.float32) + frame * 100.0
        for frame in range(3)
    ])
    state = SimpleNamespace(raw={
        "g1_pos": positions,
        "g1_root_rot": rotations,
        "g1_joint_pos": joints,
    })

    feature, _ = state_to_model_input(
        state, history_len=2, val_data=IdentityNormalization23(),
        device="cpu")

    mujoco = isaaclab_to_mujoco_dof(joints)
    core = list(range(19)) + list(range(22, 26))
    assert feature.shape == (1, 2, 57)
    np.testing.assert_array_equal(
        feature[0, -1, 11:34].numpy(), mujoco[-1, core])
    np.testing.assert_array_equal(
        feature[0, -1, 34:57].numpy(),
        mujoco[-1, core] - mujoco[-2, core])


def test_controller_v6_history_consumes_h_plus_one_physical_states():
    old_runtime, old_package = _set_both_feature_versions(6)
    try:
        positions = np.asarray([
            [0.0, 0.0, 0.77],
            [0.1, 0.0, 0.77],
            [0.2, 0.0, 0.77],
        ], dtype=np.float32)
        rotations = np.asarray([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)
        joints = np.zeros((3, 29), dtype=np.float32)
        state = SimpleNamespace(raw={
            "g1_pos": positions,
            "g1_root_rot": rotations,
            "g1_joint_pos": joints,
        })

        feature, abs_pose = state_to_model_input(
            state, history_len=2, val_data=IdentityNormalization(),
            device="cpu")

        assert feature.shape == (1, 2, 44)
        np.testing.assert_allclose(
            abs_pose["root_trans_offset"].numpy(), positions[0:1])
        torch.testing.assert_close(
            feature[0, :, 4:7],
            torch.tensor([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]),
            atol=1e-6,
            rtol=0,
        )
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


def test_controller_joint_history_smoothing_limits_v6_input_velocity():
    old_runtime, old_package = _set_both_feature_versions(6)
    try:
        positions = np.asarray([
            [0.0, 0.0, 0.77],
            [0.1, 0.0, 0.77],
            [0.2, 0.0, 0.77],
        ], dtype=np.float32)
        rotations = np.asarray([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)
        joints = np.zeros((3, 29), dtype=np.float32)
        joints[0] = -20.0
        joints[1] = 20.0
        joints[2] = 10.0
        state = SimpleNamespace(raw={
            "g1_pos": positions,
            "g1_root_rot": rotations,
            "g1_joint_pos": joints,
        })

        feature, _ = state_to_model_input(
            state,
            history_len=2,
            val_data=IdentityNormalization(),
            device="cpu",
            fps=50.0,
            joint_smoothing_max_velocity_rad_s=5.0,
        )

        dof_history = feature[0, :, 13:42].numpy()
        qvel = np.diff(dof_history, axis=0) * 50.0
        assert np.max(np.abs(qvel)) <= 5.0 + 1e-4
        np.testing.assert_allclose(
            dof_history[-1], isaaclab_to_mujoco_dof(joints[-1:])[0])
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


def test_joint_state_goal_falls_back_when_condition_blocks_invalid():
    old_runtime, old_package = _set_both_feature_versions(6)
    try:
        positions = np.asarray([
            [0.0, 0.0, 0.77],
            [0.1, 0.0, 0.78],
            [0.2, 0.0, 0.79],
        ], dtype=np.float32)
        rotations = np.asarray([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)
        joints = np.stack([
            np.arange(29, dtype=np.float32) + frame * 100.0
            for frame in range(3)
        ])
        state = SimpleNamespace(
            raw={
                "g1_pos": positions,
                "g1_root_rot": rotations,
                "g1_joint_pos": joints,
            },
            goal_type="joint_state",
            goal_root_pos_world=None,
            goal_root_velocity_world=None,
            goal_root_rot_world=None,
            goal_dof_pos=None,
            goal_timestamp_ns=None,
            timestamps_ns=None,
            goal_root_valid=False,
            goal_orientation_valid=False,
            goal_joint_valid=False,
            goal_velocity_valid=False,
            goal_time_valid=False,
        )

        goal = state_to_ego_goal(
            state, "cpu", goal_type="joint_state",
            goal_encoding=GoalEncoding.LEGACY40, fps=50.0)

        assert goal.shape == (1, 47)
        assert torch.isfinite(goal).all()
        torch.testing.assert_close(
            goal[:, 0], torch.tensor([positions[-1, 2]]))
        torch.testing.assert_close(goal[:, 1:4], torch.zeros((1, 3)))
        torch.testing.assert_close(goal[:, 42:46], torch.zeros((1, 4)))
        torch.testing.assert_close(goal[:, 46:], torch.zeros((1, 1)))
        torch.testing.assert_close(
            goal[:, 13:42],
            torch.as_tensor(isaaclab_to_mujoco_dof(joints[-1:])))
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


def test_goal_uses_current_history_feature_reference():
    state = SimpleNamespace(
        goal_root_pos_world=np.asarray([2.0, 1.0, 0.77], dtype=np.float32),
        goal_yaw_world=np.asarray([0.5], dtype=np.float32),
        raw={
            "g1_pos": np.asarray([
                [0.0, 1.0, 0.77],
                [1.0, 1.0, 0.77],
                [1.1, 1.0, 0.77],
            ], dtype=np.float32),
            "g1_root_rot": np.asarray([
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=np.float32),
        },
    )
    goal = state_to_ego_goal(state, "cpu")
    np.testing.assert_allclose(
        goal.numpy(), [[0.9, 0.0, 0.0, np.cos(0.5), np.sin(0.5)]],
        atol=1e-6)


def test_generated_goal_uses_translated_history_endpoint():
    state = SimpleNamespace(
        goal_root_pos_world=np.asarray([12.0, 20.0, 0.9], dtype=np.float32),
        goal_yaw_world=np.asarray([0.5], dtype=np.float32),
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
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
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
        "g1_root_rot": _xyzw_to_wxyz_np(real_rot.numpy()),
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


def _local_gravity_from_xyzw(root_rot: torch.Tensor) -> torch.Tensor:
    rot_matrix = quaternion_to_matrix(xyzw_to_wxyz(root_rot))
    world_gravity = torch.zeros(
        rot_matrix.shape[:-2] + (3, ),
        device=root_rot.device,
        dtype=root_rot.dtype,
    )
    world_gravity[..., 2] = -1.0
    return torch.matmul(
        rot_matrix.transpose(-1, -2),
        world_gravity.unsqueeze(-1),
    ).squeeze(-1)


def test_generated_history_alignment_v6_corrects_h_and_g_only():
    old_runtime, old_package = _set_both_feature_versions(6)
    try:
        generated_pos = torch.tensor([
            [[1.0, 2.0, 0.7], [1.2, 2.1, 0.8], [1.4, 2.2, 0.9]],
        ])
        generated_rot = euler_angles_to_quaternion(torch.tensor([
            [[0.1, 0.0, 0.2], [0.2, -0.1, 0.4], [0.3, -0.2, 0.6]],
        ]))
        generated_dof = torch.zeros((1, 3, 29))
        generated_contact = torch.ones((1, 3, 2))
        history, abs_pose = motion_dict_to_feature_v6({
            "root_trans_offset": generated_pos,
            "root_rot": generated_rot,
            "dof": generated_dof,
            "contact_mask": generated_contact,
        })
        history_before = history.clone()

        real_rot = euler_angles_to_quaternion(
            torch.tensor([[0.8, -0.35, 0.25]]))
        real_pos = torch.tensor([[10.0, 20.0, 0.25]])
        state = SimpleNamespace(states=SimpleNamespace(
            g1_pos=real_pos.numpy(),
            g1_root_rot=_xyzw_to_wxyz_np(real_rot.numpy()),
        ))

        (aligned_pose, goal_reference_pos, goal_reference_rot, _,
         aligned_history) = align_generated_history_pose(
            abs_pose,
            generated_pos[:, -1],
            generated_rot[:, -1],
            state,
            "cpu",
            history_motion=history,
            val_data=IdentityNormalization(),
        )

        delta_h = real_pos[:, 2] - history_before[:, -1, 0]
        torch.testing.assert_close(
            aligned_history[..., 0],
            history_before[..., 0] + delta_h.reshape(1, 1),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            aligned_history[:, -1, 1:4],
            _local_gravity_from_xyzw(real_rot),
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            aligned_history[..., 4:13],
            history_before[..., 4:13],
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(goal_reference_pos, real_pos)
        torch.testing.assert_close(goal_reference_rot, real_rot)

        reconstructed = motion_feature_to_dict_v6(
            aligned_history, aligned_pose)
        torch.testing.assert_close(
            reconstructed["root_trans_offset"][:, -1],
            real_pos,
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            quaternion_to_matrix(
                xyzw_to_wxyz(reconstructed["root_rot"][:, -1])),
            quaternion_to_matrix(xyzw_to_wxyz(real_rot)),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            _local_gravity_from_xyzw(reconstructed["root_rot"][:, -1]),
            _local_gravity_from_xyzw(real_rot),
            atol=1e-5,
            rtol=1e-5,
        )
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


def test_tracking_timestamps_select_consumed_frame():
    state = SimpleNamespace(
        tracking=SimpleNamespace(start_t_ns=1_000_000_000),
        history_meta=SimpleNamespace(publish_t_ns=1_061_000_000),
    )
    assert tracked_frame_from_timestamps(state, fps=50.0, future_len=8) == 3
    state.history_meta.publish_t_ns = 2_000_000_000
    assert tracked_frame_from_timestamps(state, fps=50.0, future_len=8) == 7


def test_tracking_timestamps_include_inference_latency():
    state = SimpleNamespace(
        tracking=SimpleNamespace(start_t_ns=1_000_000_000),
        history_meta=SimpleNamespace(publish_t_ns=1_061_000_000),
    )
    assert tracked_frame_from_timestamps(
        state, fps=50.0, future_len=16, latency_ms=100.0) == 8
    assert tracked_frame_from_timestamps(
        state, fps=50.0, future_len=8, latency_ms=1000.0) == 7


def test_tracking_timestamps_reject_invalid_inference_latency():
    state = SimpleNamespace(
        tracking=SimpleNamespace(start_t_ns=1_000_000_000),
        history_meta=SimpleNamespace(publish_t_ns=1_061_000_000),
    )
    for latency_ms in (-1.0, float("nan"), float("inf")):
        with np.testing.assert_raises(ValueError):
            tracked_frame_from_timestamps(
                state, fps=50.0, future_len=8, latency_ms=latency_ms)


def test_generated_history_ends_at_tracked_future_frame():
    features = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1)
    root_pos = torch.zeros((1, 10, 3), dtype=torch.float32)
    root_pos[0, :, 0] = torch.arange(10, dtype=torch.float32)
    root_rot = torch.zeros((1, 10, 4), dtype=torch.float32)
    root_rot[..., 3] = 1.0
    plan = {"features": features, "root_pos": root_pos, "root_rot": root_rot}

    history, abs_pose, reference_pos, _ = generated_history_at_frame(
        plan, tracked_frame=0, history_len=2)
    torch.testing.assert_close(history.flatten(), torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(
        abs_pose["root_trans_offset"], torch.tensor([[0.0, 0.0, 0.0]]))
    torch.testing.assert_close(reference_pos, torch.tensor([[1.0, 0.0, 0.0]]))

    history, abs_pose, reference_pos, _ = generated_history_at_frame(
        plan, tracked_frame=3, history_len=2)
    torch.testing.assert_close(history.flatten(), torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(
        abs_pose["root_trans_offset"], torch.tensor([[3.0, 0.0, 0.0]]))
    torch.testing.assert_close(reference_pos, torch.tensor([[4.0, 0.0, 0.0]]))


def test_generated_history_uses_public_frame_index_directly():
    features = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1)
    root_pos = torch.zeros((1, 10, 3), dtype=torch.float32)
    root_pos[0, :, 0] = torch.arange(10, dtype=torch.float32)
    root_rot = torch.zeros((1, 10, 4), dtype=torch.float32)
    root_rot[..., 3] = 1.0
    plan = {"features": features, "root_pos": root_pos, "root_rot": root_rot}

    history, abs_pose, reference_pos, _ = generated_history_at_frame(
        plan, tracked_frame=3, history_len=2)
    torch.testing.assert_close(history.flatten(), torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(
        abs_pose["root_trans_offset"], torch.tensor([[3.0, 0.0, 0.0]]))
    torch.testing.assert_close(reference_pos, torch.tensor([[4.0, 0.0, 0.0]]))

    history, _, reference_pos, _ = generated_history_at_frame(
        plan, tracked_frame=1, history_len=2)
    torch.testing.assert_close(history.flatten(), torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(reference_pos, torch.tensor([[2.0, 0.0, 0.0]]))


def test_generated_history_v6_anchor_is_state_before_selected_features():
    old_runtime, old_package = _set_both_feature_versions(6)
    try:
        features = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1)
        root_pos = torch.zeros((1, 10, 3), dtype=torch.float32)
        root_pos[0, :, 0] = torch.arange(10, dtype=torch.float32)
        root_rot = torch.zeros((1, 10, 4), dtype=torch.float32)
        root_rot[..., 3] = 1.0
        plan = {
            "features": features,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "input_abs_pose": {
                "root_trans_offset": torch.tensor([[-1.0, 0.0, 0.0]]),
                "root_rot": root_rot[:, 0].clone(),
            },
        }

        history, abs_pose, reference_pos, _ = generated_history_at_frame(
            plan, tracked_frame=0, history_len=2)
        torch.testing.assert_close(history.flatten(), torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(
            abs_pose["root_trans_offset"], torch.tensor([[-1.0, 0.0, 0.0]]))
        torch.testing.assert_close(reference_pos, torch.tensor([[1.0, 0.0, 0.0]]))

        history, abs_pose, reference_pos, _ = generated_history_at_frame(
            plan, tracked_frame=3, history_len=2)
        torch.testing.assert_close(history.flatten(), torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(
            abs_pose["root_trans_offset"], torch.tensor([[2.0, 0.0, 0.0]]))
        torch.testing.assert_close(reference_pos, torch.tensor([[4.0, 0.0, 0.0]]))
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


def test_generated_history_v6_reconstructs_exact_published_window():
    old_runtime, old_package = _set_both_feature_versions(6)
    try:
        num_states = 7
        positions = torch.zeros((1, num_states, 3))
        positions[0, :, 0] = torch.arange(num_states) * 0.1
        positions[0, :, 2] = 0.8
        rotations = torch.zeros((1, num_states, 4))
        rotations[..., 3] = 1.0
        joints = torch.zeros((1, num_states, 29))
        joints[0, :, 0] = torch.arange(num_states) * 0.05
        features, input_abs_pose = motion_dict_to_feature_v6({
            "root_trans_offset": positions,
            "root_rot": rotations,
            "dof": joints,
            "contact_mask": torch.ones((1, num_states, 2)),
        })
        reconstructed = motion_feature_to_dict_v6(features, input_abs_pose)
        plan = {
            "features": features,
            "root_pos": reconstructed["root_trans_offset"],
            "root_rot": reconstructed["root_rot"],
            "input_abs_pose": input_abs_pose,
        }

        history, abs_pose, reference_pos, reference_rot = (
            generated_history_at_frame(
                plan, tracked_frame=3, history_len=2))
        selected = motion_feature_to_dict_v6(history, abs_pose)

        torch.testing.assert_close(
            selected["root_trans_offset"],
                reconstructed["root_trans_offset"][:, 3:5],
        )
        torch.testing.assert_close(
            selected["root_rot"], reconstructed["root_rot"][:, 3:5])
        torch.testing.assert_close(
            selected["dof"], reconstructed["dof"][:, 3:5])
        torch.testing.assert_close(
            reference_pos, reconstructed["root_trans_offset"][:, 4])
        torch.testing.assert_close(
            reference_rot, reconstructed["root_rot"][:, 4])
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


def test_latency_residual_reanchor_matches_only_observed_past_then_rolls_forward():
    old_runtime, old_package = _set_both_feature_versions(6)
    try:
        history_len = 4
        observed_frame = 2
        target_frame = 4
        num_states = 12
        positions = torch.zeros((1, num_states, 3))
        positions[0, :, 0] = torch.arange(num_states) * 0.1
        positions[0, :, 2] = 0.8
        rotations = torch.zeros((1, num_states, 4))
        rotations[..., 3] = 1.0
        joints = torch.zeros((1, num_states, 29))
        # Output pose index 5 is public observed frame 2. Output index 6 is a
        # closer match, but is in the predicted future and must be excluded.
        joints[0, 6, 0] = 0.4
        joints[0, 7:, 0] = 0.55
        features, input_abs_pose = motion_dict_to_feature_v6({
            "root_trans_offset": positions,
            "root_rot": rotations,
            "dof": joints,
            "contact_mask": torch.ones((1, num_states, 2)),
        })
        reconstructed = motion_feature_to_dict_v6(features, input_abs_pose)
        plan = {
            "features": features,
            "root_pos": reconstructed["root_trans_offset"],
            "root_rot": reconstructed["root_rot"],
            "dof": reconstructed["dof"],
            "input_abs_pose": input_abs_pose,
        }
        real_joint = torch.zeros((1, 29))
        real_joint[0, 0] = 0.55
        state = SimpleNamespace(states=SimpleNamespace(
            g1_pos=np.array([[10.0, 0.0, 0.8]], dtype=np.float32),
            g1_root_rot=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            g1_joint_pos=mujoco_to_isaaclab_dof(real_joint.numpy()),
        ))
        val_data = SimpleNamespace(
            normalize=lambda value: value,
            denormalize=lambda value: value,
        )
        limits = (torch.full((29,), -100.0), torch.full((29,), 100.0))

        (history, abs_pose, endpoint_pos, _, _, phase, _) = (
            residual_reanchor_generated_plan_at_frame(
                plan, observed_frame, target_frame, history_len,
                state, val_data, "cpu", joint_limits=limits))
        fused = motion_feature_to_dict_v6(history, abs_pose)

        assert phase.item() == history_len - 1 + observed_frame
        torch.testing.assert_close(
            endpoint_pos, torch.tensor([[10.2, 0.0, 0.8]]),
            atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(
            fused["root_trans_offset"][:, -1], endpoint_pos,
            atol=2e-5, rtol=2e-5)
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


def test_residual_reanchor_v6_matches_full_position_and_so3_anchor():
    old_runtime, old_package = _set_both_feature_versions(6)
    try:
        positions = torch.tensor([[
            [1.0, 2.0, 0.70],
            [1.2, 2.1, 0.75],
            [1.5, 2.25, 0.82],
            [1.8, 2.4, 0.90],
            [2.1, 2.55, 0.98],
        ]])
        rotations = euler_angles_to_quaternion(torch.tensor([[
            [0.0, 0.0, 0.0],
            [0.1, 0.2, 0.1],
            [0.15, 0.25, 0.2],
            [0.2, 0.3, 0.3],
            [0.25, 0.35, 0.4],
        ]]))
        joints = torch.zeros((1, 5, 29))
        joints[0, :, 0] = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8])
        joints[0, :, 1] = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4])
        motion = {
            "root_trans_offset": positions,
            "root_rot": rotations,
            "dof": joints,
            "contact_mask": torch.ones((1, 5, 2)),
        }
        history, abs_pose = motion_dict_to_feature_v6(motion)

        real_pos = np.array([[9.0, 8.0, 3.0]], dtype=np.float32)
        real_rot = euler_angles_to_quaternion(
            torch.tensor([[0.4, -0.2, 0.6]]))
        state = SimpleNamespace(states=SimpleNamespace(
            g1_pos=real_pos,
            g1_root_rot=_xyzw_to_wxyz_np(real_rot.numpy()),
            # Frame 2 is the second feature frame, so it is the expected
            # phase match and also exercises a non-zero vertical residual.
            g1_joint_pos=mujoco_to_isaaclab_dof(joints[:, 2].numpy()),
        ))
        val_data = SimpleNamespace(
            normalize=lambda value: value,
            denormalize=lambda value: value,
            skeleton=SimpleNamespace(fk=SimpleNamespace(
                mjcf_file=Path(
                    __file__).resolve().parents[1]
                    / "description/robots/g1/g1_29dof.xml")),
        )

        aligned_pose, aligned_history, correction, phase, _ = (
            residual_reanchor_generated_history(
                abs_pose, history, state, val_data, "cpu"))
        predicted_reconstructed = motion_feature_to_dict_v6(history, abs_pose)
        reconstructed = motion_feature_to_dict_v6(
            aligned_history, aligned_pose)

        assert int(phase[0]) == 1
        torch.testing.assert_close(
            reconstructed["root_trans_offset"][:, 1],
            torch.as_tensor(real_pos),
            atol=2e-4,
            rtol=2e-4,
        )
        torch.testing.assert_close(
            reconstructed["dof"][:, 1],
            torch.as_tensor(joints[:, 2]),
            atol=2e-5,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            quaternion_to_matrix(
                xyzw_to_wxyz(reconstructed["root_rot"][:, 1])),
            quaternion_to_matrix(xyzw_to_wxyz(real_rot)),
            atol=2e-4,
            rtol=2e-4,
        )
        world_gravity = torch.tensor([[0.0, 0.0, -1.0]])
        real_gravity = torch.matmul(
            quaternion_to_matrix(xyzw_to_wxyz(real_rot)).transpose(-1, -2),
            world_gravity.unsqueeze(-1),
        ).squeeze(-1)
        relative_rot = torch.matmul(
            quaternion_to_matrix(xyzw_to_wxyz(rotations[:, 2])).transpose(-1, -2),
            quaternion_to_matrix(xyzw_to_wxyz(rotations[:, 3])),
        )
        expected_next_gravity = torch.matmul(
            relative_rot.transpose(-1, -2),
            real_gravity.unsqueeze(-1),
        ).squeeze(-1)
        actual_next_gravity = torch.matmul(
            quaternion_to_matrix(
                xyzw_to_wxyz(reconstructed["root_rot"][:, 2])
            ).transpose(-1, -2),
            world_gravity.unsqueeze(-1),
        ).squeeze(-1)
        torch.testing.assert_close(
            actual_next_gravity,
            expected_next_gravity,
            atol=2e-4,
            rtol=2e-4,
        )
        # The next frame is integrated from the predicted local increment,
        # transported by the real anchor orientation.
        real_rot_matrix = quaternion_to_matrix(xyzw_to_wxyz(real_rot))
        predicted_local_delta = torch.matmul(
            quaternion_to_matrix(
                xyzw_to_wxyz(predicted_reconstructed["root_rot"][:, 1])
            ).transpose(-1, -2),
            (predicted_reconstructed["root_trans_offset"][:, 2]
             - predicted_reconstructed["root_trans_offset"][:, 1]
             ).unsqueeze(-1),
        ).squeeze(-1)
        expected_next = torch.as_tensor(real_pos) + torch.matmul(
            real_rot_matrix, predicted_local_delta.unsqueeze(-1)
        ).squeeze(-1)
        torch.testing.assert_close(
            reconstructed["root_trans_offset"][:, 2],
            expected_next,
            atol=3e-4,
            rtol=3e-4,
        )
        assert correction["phase_index"].item() == 1
        assert correction["phase_offset_frames"].item() == -2

        # Searching starts at the current frame and moves backward, so an
        # equally good current-frame match wins over an older one.
        tied_motion = dict(motion)
        tied_joints = joints.clone()
        tied_joints[:, -1] = tied_joints[:, 2]
        tied_motion["dof"] = tied_joints
        tied_history, tied_abs_pose = motion_dict_to_feature_v6(tied_motion)
        _, _, tied_correction, tied_phase, _ = (
            residual_reanchor_generated_history(
                tied_abs_pose, tied_history, state, val_data, "cpu"))
        assert tied_phase.item() == tied_history.shape[1] - 1
        assert tied_correction["phase_offset_frames"].item() == 0
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


def test_residual_reanchor_treats_g1_joints_as_bounded_coordinates():
    old_runtime, old_package = _set_both_feature_versions(6)
    try:
        num_states = 5
        shoulder_index = G1_MUJOCO_DOF_JOINT_NAMES.index(
            "left_shoulder_pitch_joint")
        joints = torch.zeros((1, num_states, 29))
        joints[0, 1, shoulder_index] = -2.0
        joints[0, 2:4, shoulder_index] = 0.0
        joints[0, 4, shoulder_index] = 2.67
        positions = torch.zeros((1, num_states, 3))
        positions[..., 2] = 1.0
        rotations = torch.zeros((1, num_states, 4))
        rotations[..., 3] = 1.0
        motion = {
            "root_trans_offset": positions,
            "root_rot": rotations,
            "dof": joints,
            "contact_mask": torch.ones((1, num_states, 2)),
        }
        history, abs_pose = motion_dict_to_feature_v6(motion)

        real_joints = torch.zeros((1, 29))
        real_joints[0, shoulder_index] = -3.08
        state = SimpleNamespace(states=SimpleNamespace(
            g1_pos=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            g1_root_rot=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            g1_joint_pos=mujoco_to_isaaclab_dof(real_joints.numpy()),
        ))
        val_data = SimpleNamespace(
            normalize=lambda value: value,
            denormalize=lambda value: value,
            skeleton=SimpleNamespace(fk=SimpleNamespace(
                mjcf_file=Path(
                    __file__).resolve().parents[1]
                    / "description/robots/g1/g1_29dof.xml")),
        )

        aligned_pose, aligned_history, correction, phase, _ = (
            residual_reanchor_generated_history(
                abs_pose, history, state, val_data, "cpu"))
        reconstructed = motion_feature_to_dict_v6(
            aligned_history, aligned_pose)

        # Direct differences choose -2.0. Treating these bounded motor
        # coordinates as periodic would incorrectly choose +2.67.
        assert phase.item() == 0
        assert correction["phase_offset_frames"].item() == -3
        expected_current = -3.08 + (2.67 - (-2.0))
        torch.testing.assert_close(
            reconstructed["dof"][0, -1, shoulder_index],
            torch.tensor(expected_current),
            atol=2e-5,
            rtol=2e-5,
        )
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


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
    if hasattr(motion, "root_pos"):
        assert motion.root_pos.shape == (9, 3)
        assert motion.root_ori.shape == (9, 4)
        assert motion.body_pos is None
        assert motion.body_ori is None
        np.testing.assert_array_equal(
            motion.root_pos, body_pos.numpy()[0, 1:, 0])
        np.testing.assert_array_equal(
            motion.root_ori[0], np.asarray([1.0, 0.0, 0.0, 0.0]))
    else:
        assert motion.body_pos.shape == (9, 30, 3)
        assert motion.body_ori.shape == (9, 30, 4)
    np.testing.assert_array_equal(
        motion.joint_pos, mujoco_to_isaaclab_dof(dof_pos.numpy()[0, 1:]))
    np.testing.assert_allclose(motion.joint_vel[:, 0], 1.0, atol=1e-6)

    motion_with_body = motion_dict_to_g1data({
        "dof_pos": dof_pos,
        "global_translation_extend": body_pos,
        "global_rotation_extend": body_ori,
    }, skip_history=1, fps=50.0, include_body=True)

    assert motion_with_body.body_pos.shape == (9, 30, 3)
    assert motion_with_body.body_ori.shape == (9, 30, 4)
    np.testing.assert_array_equal(
        motion_with_body.body_pos,
        np.repeat(body_pos.numpy()[0, 1:, :1], 30, axis=1))
    np.testing.assert_array_equal(
        motion_with_body.joint_pos,
        mujoco_to_isaaclab_dof(dof_pos.numpy()[0, 1:]))
    np.testing.assert_array_equal(
        motion_with_body.body_ori[0, 0],
        np.asarray([1.0, 0.0, 0.0, 0.0]))


def test_g1_packet_accepts_reconstruction_without_fk_body_arrays():
    frames = 4
    dof = torch.zeros((1, frames, 29), dtype=torch.float32)
    root_pos = torch.arange(frames * 3, dtype=torch.float32).reshape(
        1, frames, 3)
    root_rot = torch.zeros((1, frames, 4), dtype=torch.float32)
    root_rot[..., 3] = 1.0

    motion = motion_dict_to_g1data({
        "dof": dof,
        "root_trans_offset": root_pos,
        "root_rot": root_rot,
    }, skip_history=1, fps=50.0)

    assert motion.num_frames == frames - 1
    np.testing.assert_array_equal(motion.root_pos, root_pos.numpy()[0, 1:])
    np.testing.assert_array_equal(
        motion.root_ori,
        np.repeat(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            frames - 1,
            axis=0,
        ),
    )
    assert motion.body_pos is None
    assert motion.body_ori is None


def test_23dof_g1_packet_expands_and_holds_measured_wrist_joints():
    frames = 3
    dof_pos = torch.arange(
        frames * 23, dtype=torch.float32).reshape(1, frames, 23) / 100.0
    body_pos = torch.zeros((1, frames, 27, 3), dtype=torch.float32)
    body_ori = torch.zeros((1, frames, 27, 4), dtype=torch.float32)
    body_ori[..., 3] = 1.0
    measured = np.arange(29, dtype=np.float32) + 10.0

    motion = motion_dict_to_g1data({
        "dof_pos": dof_pos,
        "global_translation_extend": body_pos,
        "global_rotation_extend": body_ori,
    }, skip_history=1, fps=50.0, locked_joint_pos=measured)

    assert motion.joint_pos.shape == (frames - 1, 29)
    assert motion.joint_vel.shape == (frames - 1, 29)
    wrist_indices = [
        index for index, name in enumerate(G1_ISAACLAB_DOF_JOINT_NAMES)
        if "_wrist_" in name
    ]
    np.testing.assert_array_equal(
        motion.joint_pos[:, wrist_indices],
        np.broadcast_to(measured[wrist_indices], (frames - 1, 6)))
    np.testing.assert_array_equal(
        motion.joint_vel[:, wrist_indices],
        np.zeros((frames - 1, 6), dtype=np.float32))
