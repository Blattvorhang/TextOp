from types import SimpleNamespace

import torch
import pytest
from omegaconf import OmegaConf

import robotmdar.dtype.motion as runtime_motion_dtype
import TextOpRobotMDAR.robotmdar.dtype.motion as package_motion_dtype
from TextOpRobotMDAR.robotmdar.dtype.rotation import (
    matrix_to_rot6d,
    quaternion_to_matrix,
    xyzw_to_wxyz,
)
from TextOpRobotMDAR.robotmdar.skeleton.end_effector import EndEffectorAnchor
from TextOpRobotMDAR.robotmdar.train.loss import GeometryLoss
from TextOpRobotMDAR.robotmdar.train.manager import DARManager
from TextOpRobotMDAR.robotmdar.train.train_dar import (
    _conditions,
    _make_root_xy_figure,
    _next_rollout_poses,
    _raw_goal_root_target,
    _validate_goal_root_position_contract,
)
from TextOpRobotMDAR.robotmdar.utils.goal import (
    SPLIT_END_EFFECTOR_GOAL_DIM,
    SPLIT_END_EFFECTOR_NO_LOG_GOAL_DIM,
    SPLIT_END_EFFECTOR_NO_LOG_GOAL_SCHEMA,
    SPLIT_END_EFFECTOR_SLICE,
    SPLIT_GOAL_DIM,
    SPLIT_NO_LOG_VERTICAL_HEIGHT_SLICE,
    SPLIT_HORIZONTAL_SLICE,
    SPLIT_VERTICAL_GRAVITY_SLICE,
)


def _manager():
    manager = object.__new__(DARManager)
    manager.dataset = SimpleNamespace(
        denormalize=lambda value: value,
        dof_dim=29,
        fps=50,
    )
    manager.rec_criterion = torch.nn.HuberLoss(reduction='mean', delta=1.0)
    return manager


def _history(displacement_xy):
    history = torch.zeros((len(displacement_xy), 2, 69), dtype=torch.float32)
    history[:, -1, 7:9] = torch.as_tensor(displacement_xy)
    return history


def _future(displacement_xy):
    future = torch.zeros((len(displacement_xy), 4, 69), dtype=torch.float32)
    future[:, :3, 7:9] = torch.as_tensor(displacement_xy)[:, None] / 4.0
    return future


def _set_both_feature_versions(version: int):
    old_runtime = runtime_motion_dtype.FeatureVersion
    old_package = package_motion_dtype.FeatureVersion
    runtime_motion_dtype.set_feature_version(version)
    package_motion_dtype.set_feature_version(version)
    return old_runtime, old_package


def test_end_effector_anchor_cache_does_not_shadow_resolver(monkeypatch):
    geometry = object.__new__(GeometryLoss)
    geometry.dataset = SimpleNamespace(
        skeleton=SimpleNamespace(
            fk=SimpleNamespace(mjcf_file="robot.xml")))
    anchors = (object(),)
    calls = []

    def resolve(skeleton):
        calls.append(skeleton)
        return anchors

    monkeypatch.setattr(
        "TextOpRobotMDAR.robotmdar.train.loss.resolve_end_effector_anchors",
        resolve,
    )

    assert geometry._end_effector_anchors() is anchors
    assert geometry._end_effector_anchors() is anchors
    assert len(calls) == 1
    assert geometry._end_effector_anchors_cache is anchors


def _empty_geometry_loss_v6(*args, **kwargs):
    if kwargs.get("return_fk_results", False):
        return {}, {}, {}, None
    return {}, {}, {}


def test_goal_root_position_loss_is_zero_for_matching_endpoint():
    manager = _manager()
    future = _future([[1.0, 0.0], [0.0, -0.5]])
    history = _history([[0.25, 0.0], [0.0, -0.125]])
    goal = torch.zeros((2, 15))
    goal[:, :2] = torch.tensor([[1.0, 0.0], [0.0, -0.5]])

    loss = manager.calc_goal_root_position_loss(
        future, goal, history_motion=history)

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_goal_root_position_loss_ignores_dropped_root_conditions():
    manager = _manager()
    future = _future([[1.0, 0.0], [0.0, 0.0]])
    history = _history([[0.25, 0.0], [0.0, 0.0]])
    goal = torch.zeros((2, 21))
    goal[:, :2] = torch.tensor([[1.0, 0.0], [5.0, 0.0]])

    loss = manager.calc_goal_root_position_loss(
        future, goal, goal_condition_keep_mask=torch.tensor([True, False]),
        history_motion=history)

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_goal_root_position_loss_uses_goal_frame_when_provided():
    manager = _manager()
    future = torch.zeros((1, 4, 69), dtype=torch.float32)
    future[0, :3, 7] = 0.25
    history = _history([[0.25, 0.0]])
    goal = torch.zeros((1, 21), dtype=torch.float32)
    goal[:, :2] = torch.tensor([[0.5, 0.0]])

    loss = manager.calc_goal_root_position_loss(
        future,
        goal,
        history_motion=history,
        goal_time_frame=torch.tensor([2]),
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_root_displacement_integrates_turning_local_deltas():
    manager = _manager()
    history = torch.zeros((1, 2, 69), dtype=torch.float32)
    history[0, -1, 7] = 1.0
    history[0, -1, 4] = torch.pi / 2
    future = torch.zeros((1, 2, 69), dtype=torch.float32)
    future[0, 0, 7] = 1.0

    displacement = manager.root_displacement_ego(future, history)

    torch.testing.assert_close(
        displacement, torch.tensor([[1.0, 1.0]]), atol=1e-6, rtol=0)


def test_root_trajectory_includes_origin_and_turning_path():
    manager = _manager()
    history = torch.zeros((1, 2, 69), dtype=torch.float32)
    history[0, -1, 7] = 1.0
    history[0, -1, 4] = torch.pi / 2
    future = torch.zeros((1, 2, 69), dtype=torch.float32)
    future[0, 0, 7] = 1.0

    trajectory = manager.root_trajectory_ego(future, history)

    torch.testing.assert_close(
        trajectory,
        torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]]),
        atol=1e-6,
        rtol=0,
    )


def test_root_displacement_excludes_terminal_forward_delta():
    manager = _manager()
    history = _history([[0.25, 0.0]])
    future = torch.zeros((1, 4, 69), dtype=torch.float32)
    future[0, :3, 7] = 0.25
    future[0, -1, 7] = 100.0

    displacement = manager.root_displacement_ego(future, history)

    torch.testing.assert_close(displacement, torch.tensor([[1.0, 0.0]]))


def test_goal_root_position_contract_allows_random_offsets_with_relative_time():
    cfg = OmegaConf.create({
        'data': {
            'goal_type': 'body_ext',
            'goal_per_primitive': False,
            'goal_offset': 0,
            'goal_offset_range': [-2, 2],
            'goal_timestep_mode': 'relative',
        },
        'train': {'manager': {'loss_weight': {'goal_root_position': 0.5}}},
    })

    _validate_goal_root_position_contract(cfg)

    cfg.data.goal_timestep_mode = 'zero'
    with pytest.raises(ValueError, match='goal_timestep_mode=relative'):
        _validate_goal_root_position_contract(cfg)


def test_goal_root_position_contract_allows_joint_state_random_offsets():
    cfg = OmegaConf.create({
        'data': {
            'goal_type': 'joint_state',
            'goal_per_primitive': True,
            'goal_offset': 0,
            'goal_offset_range': [-3, 0],
            'goal_timestep_mode': 'relative',
        },
        'train': {'manager': {'loss_weight': {'goal_root_position': 0.5}}},
    })

    _validate_goal_root_position_contract(cfg)


def test_goal_root_position_contract_reads_nested_hor_vert_weights():
    cfg = OmegaConf.create({
        'data': {
            'goal_type': 'root',
            'goal_per_primitive': True,
            'goal_offset': 0,
            'goal_offset_range': [-3, 0],
            'goal_timestep_mode': 'relative',
        },
        'train': {
            'manager': {
                'loss_weight': {
                    'locomotion': {
                        'goal': {'root_position_hor': 0.0},
                    },
                    'getup': {
                        'goal': {'root_position_vert': 0.5},
                    },
                },
            },
        },
    })

    with pytest.raises(ValueError, match='explicit arrival time'):
        _validate_goal_root_position_contract(cfg)


def test_joint_state_goal_losses_use_selected_goal_frame():
    manager = _manager()
    future = torch.zeros((1, 4, 69), dtype=torch.float32)
    history = torch.zeros((1, 2, 69), dtype=torch.float32)
    goal_time_frame = torch.tensor([2])

    # Goal-frame orientation: selected future step is time_to_arrival - 1.
    future[0, 1, 0:4] = torch.tensor([0.1, -0.01, -0.2, -0.02])
    history[0, -1, 4] = 0.2
    future[0, 0, 4] = 0.3

    # Goal-frame joints and velocity.
    q_goal = torch.linspace(-0.5, 0.5, 29)
    future[0, 1, 11:40] = q_goal
    future[0, 1, 7:10] = torch.tensor([0.02, 0.0, -0.01])

    goal = torch.zeros((1, 40), dtype=torch.float32)
    goal[0, 3:8] = torch.tensor([0.1, -0.01, -0.2, -0.02, 0.5])
    goal[0, 8:37] = q_goal
    goal[0, 37:40] = torch.tensor([
        0.02 * 50.0 * torch.cos(torch.tensor(0.5)),
        0.02 * 50.0 * torch.sin(torch.tensor(0.5)),
        -0.01 * 50.0,
    ])

    torch.testing.assert_close(
        manager.calc_goal_root_orientation_loss(
            future, goal, history_motion=history,
            goal_time_frame=goal_time_frame),
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        manager.calc_goal_joint_angle_loss(
            future, goal, goal_time_frame=goal_time_frame),
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        manager.calc_goal_root_velocity_loss(
            future, goal, history_motion=history,
            goal_time_frame=goal_time_frame),
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0,
    )


def test_v6_joint_state_goal_losses_use_arrival_rotmat_goal_frame():
    old_runtime, old_package = _set_both_feature_versions(6)
    try:
        manager = _manager()
        manager.dataset.dof_dim = 29
        manager.dataset.fps = 50

        rot6d_identity = matrix_to_rot6d(torch.eye(3)).reshape(6)
        history = torch.zeros((1, 2, 44), dtype=torch.float32)
        future = torch.zeros((1, 4, 44), dtype=torch.float32)
        for motion in (history, future):
            motion[..., 0] = 0.77
            motion[..., 1:4] = torch.tensor([0.0, 0.0, -1.0])
            motion[..., 7:13] = rot6d_identity
            motion[..., 42:44] = 1.0

        future[0, 0:2, 4] = 0.25
        q_goal = torch.linspace(-0.5, 0.5, 29)
        future[0, 1, 13:42] = q_goal

        goal = torch.zeros((1, 47), dtype=torch.float32)
        goal[0, 0] = 0.77
        goal[0, 1:4] = torch.tensor([0.5, 0.0, 0.0])
        goal[0, 4:7] = torch.tensor([0.0, 0.0, -1.0])
        goal[0, 7:13] = rot6d_identity
        goal[0, 13:42] = q_goal
        goal[0, 42:46] = torch.tensor([12.5, 0.0, 0.0, 0.0])
        goal[0, 46] = 2.0 / 50.0
        goal_time_frame = torch.tensor([2])

        torch.testing.assert_close(
            manager.calc_goal_root_position_loss(
                future, goal, history_motion=history,
                goal_time_frame=goal_time_frame),
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            manager.calc_goal_root_orientation_loss(
                future, goal, history_motion=history,
                goal_time_frame=goal_time_frame),
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            manager.calc_goal_g_loss(
                future, goal, history_motion=history,
                goal_time_frame=goal_time_frame),
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            manager.calc_goal_joint_angle_loss(
                future, goal, goal_time_frame=goal_time_frame),
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            manager.calc_goal_root_velocity_loss(
                future, goal, history_motion=history,
                goal_time_frame=goal_time_frame),
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0,
        )
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


def test_v6_split_goal_losses_use_hor_vert_rot_layout():
    old_runtime, old_package = _set_both_feature_versions(6)
    try:
        manager = _manager()
        manager.dataset.dof_dim = 29
        manager.dataset.fps = 50

        rot6d_identity = matrix_to_rot6d(torch.eye(3)).reshape(6)
        history = torch.zeros((1, 2, 44), dtype=torch.float32)
        future = torch.zeros((1, 4, 44), dtype=torch.float32)
        for motion in (history, future):
            motion[..., 0] = 0.77
            motion[..., 1:4] = torch.tensor([0.0, 0.0, -1.0])
            motion[..., 7:13] = rot6d_identity
            motion[..., 42:44] = 1.0

        future[0, 0:2, 4] = 0.25
        q_goal = torch.linspace(-0.5, 0.5, 29)
        future[0, 1, 13:42] = q_goal

        goal = torch.zeros((1, SPLIT_GOAL_DIM), dtype=torch.float32)
        goal[0, 0:3] = torch.tensor([0.5, 0.0, 0.0])
        goal[0, 3] = 0.5
        goal[0, 4] = torch.log1p(torch.tensor(0.5))
        goal[0, 9:15] = torch.tensor([0.77, 0.0, 0.0, 0.0, -1.0, 0.0])
        goal[0, 15:21] = rot6d_identity
        goal[0, 21:50] = q_goal
        goal[0, 50:54] = torch.tensor([12.5, 0.0, 0.0, 0.0])
        goal[0, 54] = 2.0 / 50.0
        goal_time_frame = torch.tensor([2])

        torch.testing.assert_close(
            _raw_goal_root_target(goal), torch.tensor([[0.5, 0.0, 0.0]]))
        torch.testing.assert_close(
            manager.calc_goal_root_position_loss(
                future, goal, history_motion=history,
                goal_time_frame=goal_time_frame),
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            manager.calc_goal_root_orientation_loss(
                future, goal, history_motion=history,
                goal_time_frame=goal_time_frame),
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            manager.calc_goal_g_loss(
                future, goal, history_motion=history,
                goal_time_frame=goal_time_frame),
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            manager.calc_goal_joint_angle_loss(
                future, goal, goal_time_frame=goal_time_frame),
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            manager.calc_goal_root_velocity_loss(
                future, goal, history_motion=history,
                goal_time_frame=goal_time_frame),
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0,
        )
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


def test_v6_split_goal_root_position_loss_reports_hor_and_vert_separately():
    old_runtime, old_package = _set_both_feature_versions(6)
    try:
        manager = _manager()
        manager.dataset.dof_dim = 29
        manager.dataset.fps = 50

        rot6d_identity = matrix_to_rot6d(torch.eye(3)).reshape(6)
        history = torch.zeros((1, 2, 44), dtype=torch.float32)
        future = torch.zeros((1, 4, 44), dtype=torch.float32)
        for motion in (history, future):
            motion[..., 0] = 0.77
            motion[..., 1:4] = torch.tensor([0.0, 0.0, -1.0])
            motion[..., 7:13] = rot6d_identity
            motion[..., 42:44] = 1.0
        future[0, 0:2, 4] = 0.25

        goal = torch.zeros((1, SPLIT_GOAL_DIM), dtype=torch.float32)
        goal[0, 0:3] = torch.tensor([0.6, 0.0, 0.0])
        goal[0, 9] = 0.80
        goal_time_frame = torch.tensor([2])

        components = manager.calc_goal_root_position_loss_components(
            future,
            goal,
            history_motion=history,
            goal_time_frame=goal_time_frame,
        )

        torch.testing.assert_close(
            components['goal_root_position_hor'],
            torch.tensor(0.5 * 0.1**2 / 3.0),
            atol=1e-7,
            rtol=0,
        )
        torch.testing.assert_close(
            components['goal_root_position_vert'],
            torch.tensor(0.5 * 0.03**2),
            atol=1e-7,
            rtol=0,
        )

        split_components = manager.calc_goal_root_position_loss_components(
            future,
            goal,
            history_motion=history,
            goal_position_hor_condition_keep_mask=torch.tensor([False]),
            goal_position_vert_condition_keep_mask=torch.tensor([True]),
            goal_time_frame=goal_time_frame,
        )
        torch.testing.assert_close(
            split_components['goal_root_position_hor'],
            torch.tensor(0.0),
            atol=1e-7,
            rtol=0,
        )
        torch.testing.assert_close(
            split_components['goal_root_position_vert'],
            torch.tensor(0.5 * 0.03**2),
            atol=1e-7,
            rtol=0,
        )
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


def test_self_rollout_goal_loss_uses_pred_current_state_reference():
    old_runtime, old_package = _set_both_feature_versions(6)
    try:
        manager = _manager()
        manager.dataset.dof_dim = 29
        manager.dataset.fps = 50
        manager.loss_weight = {
            'goal': {
                'root_position_hor': 1.0,
                'root_position_vert': 1.0,
            }
        }
        manager.calc_geometry_loss_v6 = lambda *args, **kwargs: ({}, {}, {}, None)

        cfg = OmegaConf.create({
            'device': 'cpu',
            'data': {
                'goal_type': 'joint_state',
                'goal_encoding': 'split_end_effector',
                'occupancy_unit': 0.1,
            },
            'denoiser': {
                'grid_size': 1,
            },
        })
        goal_stats = {
            's_p': torch.tensor(1.0),
            's_v': torch.tensor(1.0),
            's_d': torch.tensor(1.0),
            's_o': torch.ones(9),
            'q_mean': torch.zeros(29),
            'q_std': torch.ones(29),
            's_ee': torch.ones(12),
            'meta': {
                'fps': 50.0,
                    'goal_dim': SPLIT_END_EFFECTOR_NO_LOG_GOAL_DIM,
                    'goal_schema': SPLIT_END_EFFECTOR_NO_LOG_GOAL_SCHEMA,
            },
        }

        rot6d_identity = matrix_to_rot6d(torch.eye(3)).reshape(6)
        history = torch.zeros((1, 2, 44), dtype=torch.float32)
        future = torch.zeros((1, 4, 44), dtype=torch.float32)
        for motion in (history, future):
            motion[..., 0] = 0.77
            motion[..., 1:4] = torch.tensor([0.0, 0.0, -1.0])
            motion[..., 7:13] = rot6d_identity
            motion[..., 42:44] = 1.0
        future[0, 0:2, 4] = 0.25

        pred_ref_pos = torch.tensor([[10.0, -3.0, 0.77]], dtype=torch.float32)
        gt_ref_pos = torch.tensor([[0.0, -3.0, 0.77]], dtype=torch.float32)
        reference_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        world_goal_pos = pred_ref_pos + torch.tensor(
            [[0.5, 0.0, 0.0]], dtype=torch.float32)
        primitive = {
            'world_goal_pos': world_goal_pos,
            'world_goal_yaw': torch.zeros(1),
            'world_goal_rot': reference_rot,
            'world_goal_dof': torch.zeros((1, 29), dtype=torch.float32),
            'world_goal_vel': torch.zeros((1, 3), dtype=torch.float32),
            'world_goal_end_effectors': world_goal_pos[:, None].expand(
                -1, 4, -1).clone(),
            'time_to_arrival': torch.tensor([2.0 / 50.0], dtype=torch.float32),
        }

        conditions = _conditions(
            primitive,
            pred_ref_pos,
            reference_rot,
            history,
            cfg,
            fps=50.0,
            goal_stats=goal_stats,
            use_scene=False,
        )
        goal_time_frame = conditions['time_to_arrival_frame']

        torch.testing.assert_close(
            conditions['ego_goal_raw'][:, SPLIT_HORIZONTAL_SLICE][:, :3],
            torch.tensor([[0.5, 0.0, 0.0]]),
        )
        torch.testing.assert_close(
            conditions['ego_goal_raw'][
                :, SPLIT_NO_LOG_VERTICAL_HEIGHT_SLICE][:, :1],
            torch.tensor([[0.77]]),
        )

        loss_dict, _ = manager.calc_loss(
            torch.zeros_like(future),
            future,
            None,
            None,
            None,
            None,
            history_motion=history,
            ego_goal=conditions['ego_goal_raw'],
            goal_type='joint_state',
            goal_time_frame=goal_time_frame,
        )

        torch.testing.assert_close(
            loss_dict['goal_root_position_hor'],
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            loss_dict['goal_root_position_vert'],
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            loss_dict['total'], torch.tensor(0.0), atol=1e-6, rtol=0)

        gt_conditions = _conditions(
            primitive,
            gt_ref_pos,
            reference_rot,
            history,
            cfg,
            fps=50.0,
            goal_stats=goal_stats,
            use_scene=False,
        )
        wrong_reference_loss = manager.calc_goal_root_position_loss(
            future,
            gt_conditions['ego_goal_raw'],
            history_motion=history,
            goal_time_frame=goal_time_frame,
        )
        assert wrong_reference_loss > 1.0
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


def test_rollout_pose_chain_integrates_from_segment_start_anchor():
    old_runtime, old_package = _set_both_feature_versions(6)
    try:
        class CumulativeHorizontalDataset:
            def __init__(self):
                self.anchor_positions = []

            def reconstruct_motion(self, motion, abs_pose=None, ret_fk=False):
                assert abs_pose is not None
                assert ret_fk is False
                self.anchor_positions.append(
                    abs_pose['root_trans_offset'].detach().clone())
                delta = motion[..., 4:7]
                trans = (
                    abs_pose['root_trans_offset'].unsqueeze(1)
                    + delta.cumsum(dim=1)
                )
                root_rot = abs_pose['root_rot'].unsqueeze(1).expand(
                    -1, motion.shape[1], -1).clone()
                return {'root_trans_offset': trans, 'root_rot': root_rot}

        dataset = CumulativeHorizontalDataset()
        history_len = 2
        future_len = 3
        motion = torch.zeros((1, history_len + future_len, 44),
                             dtype=torch.float32)
        motion[..., 4] = 1.0
        start_pos = torch.tensor([[100.0, 200.0, 0.0]], dtype=torch.float32)
        start_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]],
                                 dtype=torch.float32)

        history_start_pos, history_start_rot, ref_pos, ref_rot = (
            _next_rollout_poses(
                dataset, motion, start_pos, start_rot, history_len))
        torch.testing.assert_close(
            history_start_pos, torch.tensor([[103.0, 200.0, 0.0]]))
        torch.testing.assert_close(
            ref_pos, torch.tensor([[105.0, 200.0, 0.0]]))

        history_start_pos, history_start_rot, ref_pos, ref_rot = (
            _next_rollout_poses(
                dataset, motion, history_start_pos, history_start_rot,
                history_len))
        torch.testing.assert_close(
            history_start_pos, torch.tensor([[106.0, 200.0, 0.0]]))
        torch.testing.assert_close(
            ref_pos, torch.tensor([[108.0, 200.0, 0.0]]))

        _, _, ref_pos, _ = _next_rollout_poses(
            dataset, motion, history_start_pos, history_start_rot, history_len)
        torch.testing.assert_close(
            ref_pos, torch.tensor([[111.0, 200.0, 0.0]]))
        torch.testing.assert_close(
            torch.cat(dataset.anchor_positions, dim=0),
            torch.tensor([
                [100.0, 200.0, 0.0],
                [103.0, 200.0, 0.0],
                [106.0, 200.0, 0.0],
            ]),
        )
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


def test_goal_end_effector_loss_sums_independent_visible_token_losses():
    old_runtime, old_package = _set_both_feature_versions(6)
    try:
        manager = _manager()
        manager.goal_end_effector_loss_beta = 0.05
        manager.dataset.skeleton = SimpleNamespace()
        manager._end_effector_anchors = lambda: tuple(
            EndEffectorAnchor(
                name=f"ee_{idx}",
                source_type="body",
                source_name=f"body_{idx}",
                parent_body=f"body_{idx}",
                parent_body_index=idx,
                local_pos=torch.zeros(3),
            )
            for idx in range(4)
        )
        future = torch.zeros((2, 4, 44), dtype=torch.float32)
        pred_translation = torch.zeros((2, 4, 4, 3), dtype=torch.float32)
        pred_rotation = torch.eye(3).reshape(1, 1, 1, 3, 3).expand(
            2, 4, 4, 3, 3).clone()
        goal = torch.zeros((2, SPLIT_END_EFFECTOR_GOAL_DIM), dtype=torch.float32)
        goal[:, SPLIT_END_EFFECTOR_SLICE] = torch.tensor(
            [[1.0, 0.0, 0.0] * 4,
             [2.0, 0.0, 0.0] * 4],
            dtype=torch.float32,
        )

        loss, per_token_sample, valid, metrics = (
            manager.calc_goal_end_effector_loss(
                future,
                goal,
                goal_end_effector_condition_keep_mask=torch.tensor(
                    [[True, False, False, False],
                     [False, True, False, False]]),
                future_motion_pred_fk={
                    "global_translation": pred_translation,
                    "global_rotation_mat": pred_rotation,
                },
                goal_time_frame=torch.tensor([2, 2]),
                return_per_sample=True,
            ))

        left_hand_loss = torch.tensor(1.0 - 0.5 * 0.05)
        right_hand_loss = torch.tensor(2.0 - 0.5 * 0.05)
        expected = left_hand_loss + right_hand_loss
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(
            per_token_sample,
            torch.tensor([
                [float(left_hand_loss), 0.0, 0.0, 0.0],
                [0.0, float(right_hand_loss), 0.0, 0.0],
            ]),
        )
        assert valid.tolist() == [
            [True, False, False, False],
            [False, True, False, False],
        ]
        torch.testing.assert_close(
            metrics['goal_end_effector'], torch.tensor(1.5))
        torch.testing.assert_close(
            metrics['goal_end_effector_left_hand'], torch.tensor(1.0))
        torch.testing.assert_close(
            metrics['goal_end_effector_right_hand'], torch.tensor(2.0))
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


def test_goal_end_effector_loss_aligns_prediction_to_current_state_reference():
    old_runtime, old_package = _set_both_feature_versions(6)
    try:
        manager = _manager()
        manager.goal_end_effector_loss_beta = 0.05
        manager._end_effector_anchors = lambda: tuple(
            EndEffectorAnchor(
                name=f"ee_{idx}",
                source_type="body",
                source_name=f"body_{idx}",
                parent_body=f"body_{idx}",
                parent_body_index=idx,
                local_pos=torch.zeros(3),
            )
            for idx in range(4)
        )

        future = torch.zeros((1, 4, 44), dtype=torch.float32)
        reference_pos = torch.tensor([[10.0, 20.0, 0.75]])
        sqrt_half = 2.0**-0.5
        reference_rot = torch.tensor(
            [[0.0, 0.0, sqrt_half, sqrt_half]], dtype=torch.float32)
        ee_ego = torch.tensor(
            [[[1.0, 0.0, 0.1],
              [0.0, 2.0, 0.2],
              [-1.0, 0.0, -0.3],
              [0.0, -2.0, -0.4]]],
            dtype=torch.float32,
        )
        reference_matrix = quaternion_to_matrix(xyzw_to_wxyz(reference_rot))
        ee_world = reference_pos.unsqueeze(1) + torch.matmul(
            reference_matrix.unsqueeze(1),
            ee_ego.unsqueeze(-1),
        ).squeeze(-1)

        calls = {}

        def reconstruct_motion(motion_feature, abs_pose=None,
                               need_denormalize=True, ret_fk=True):
            calls['abs_pose'] = abs_pose
            calls['ret_fk'] = ret_fk
            assert need_denormalize is True
            assert ret_fk is False
            B, T = motion_feature.shape[:2]
            root_rot = torch.zeros((B, T, 4), dtype=motion_feature.dtype)
            root_rot[..., 3] = 1.0
            return {
                'root_trans_offset': torch.zeros(
                    B, T, 3, dtype=motion_feature.dtype),
                'root_rot': root_rot,
                'dof': torch.zeros(B, T, 29, dtype=motion_feature.dtype),
                'contact_mask': torch.zeros(
                    B, T, 4, dtype=motion_feature.dtype),
            }

        def forward_kinematics(motion_dict, return_full=False, fps=30.0):
            calls['fk_time_dim'] = motion_dict['dof'].shape[1]
            global_rotation = torch.eye(3).reshape(1, 1, 1, 3, 3).expand(
                1, 1, 4, 3, 3).clone()
            return {
                'global_translation': ee_world.unsqueeze(1),
                'global_rotation_mat': global_rotation,
            }

        manager.dataset.reconstruct_motion = reconstruct_motion
        manager.dataset.skeleton = SimpleNamespace(
            forward_kinematics=forward_kinematics)

        goal = torch.zeros((1, SPLIT_END_EFFECTOR_GOAL_DIM),
                           dtype=torch.float32)
        goal[:, SPLIT_END_EFFECTOR_SLICE] = ee_ego.reshape(1, 12)

        loss, per_token_sample, valid, metrics = (
            manager.calc_goal_end_effector_loss(
                future,
                goal,
                goal_end_effector_condition_keep_mask=torch.tensor(
                    [[True, True, True, True]]),
                goal_reference_pos=reference_pos,
                goal_reference_rot=reference_rot,
                goal_time_frame=torch.tensor([2]),
                return_per_sample=True,
            ))

        torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0)
        torch.testing.assert_close(
            per_token_sample, torch.zeros((1, 4)), atol=1e-6, rtol=0)
        assert valid.tolist() == [[True, True, True, True]]
        torch.testing.assert_close(
            metrics['goal_end_effector'], torch.tensor(0.0), atol=1e-6, rtol=0)
        torch.testing.assert_close(
            calls['abs_pose']['root_trans_offset'], reference_pos)
        torch.testing.assert_close(calls['abs_pose']['root_rot'], reference_rot)
        assert calls['ret_fk'] is False
        assert calls['fk_time_dim'] == 1
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


def test_goal_end_effector_total_loss_uses_sum_of_token_means():
    old_runtime, old_package = _set_both_feature_versions(6)
    try:
        manager = _manager()
        manager.goal_end_effector_loss_beta = 0.05
        manager.loss_weight = {'goal': {'end_effector': 1.0}}
        manager.dataset.skeleton = SimpleNamespace()
        manager._end_effector_anchors = lambda: tuple(
            EndEffectorAnchor(
                name=f"ee_{idx}",
                source_type="body",
                source_name=f"body_{idx}",
                parent_body=f"body_{idx}",
                parent_body_index=idx,
                local_pos=torch.zeros(3),
            )
            for idx in range(4)
        )
        future = torch.zeros((2, 4, 44), dtype=torch.float32)
        pred_translation = torch.zeros((2, 1, 4, 3), dtype=torch.float32)
        pred_rotation = torch.eye(3).reshape(1, 1, 1, 3, 3).expand(
            2, 1, 4, 3, 3).clone()

        def reconstruct_motion(motion_feature, abs_pose=None,
                               need_denormalize=True, ret_fk=True):
            assert abs_pose is not None
            assert need_denormalize is True
            assert ret_fk is False
            B, T = motion_feature.shape[:2]
            root_rot = torch.zeros((B, T, 4), dtype=motion_feature.dtype)
            root_rot[..., 3] = 1.0
            return {
                'root_trans_offset': torch.zeros(
                    B, T, 3, dtype=motion_feature.dtype),
                'root_rot': root_rot,
                'dof': torch.zeros(B, T, 29, dtype=motion_feature.dtype),
                'contact_mask': torch.zeros(
                    B, T, 4, dtype=motion_feature.dtype),
            }

        def forward_kinematics(motion_dict, return_full=False, fps=30.0):
            assert motion_dict['dof'].shape[1] == 1
            return {
                "global_translation": pred_translation,
                "global_rotation_mat": pred_rotation,
            }

        manager.dataset.reconstruct_motion = reconstruct_motion
        manager.dataset.skeleton.forward_kinematics = forward_kinematics
        manager.calc_geometry_loss_v6 = lambda *args, **kwargs: (
            {},
            {},
            {},
            {
                "future_motion_pred_fk": {
                    "global_translation": torch.ones((2, 4, 4, 3)),
                    "global_rotation_mat": torch.eye(3).reshape(
                        1, 1, 1, 3, 3).expand(2, 4, 4, 3, 3).clone(),
                }
            },
        )

        goal = torch.zeros((2, SPLIT_END_EFFECTOR_GOAL_DIM), dtype=torch.float32)
        goal[:, SPLIT_END_EFFECTOR_SLICE] = torch.tensor(
            [[1.0, 0.0, 0.0] * 4,
             [2.0, 0.0, 0.0] * 4],
            dtype=torch.float32,
        )
        keep_mask = torch.tensor(
            [[True, False, False, False],
             [False, True, False, False]])
        reference_pos = torch.zeros((2, 3), dtype=torch.float32)
        reference_rot = torch.zeros((2, 4), dtype=torch.float32)
        reference_rot[:, 3] = 1.0

        loss_dict, extras = manager.calc_loss(
            torch.zeros_like(future),
            future,
            None,
            None,
            None,
            None,
            ego_goal=goal,
            goal_type='joint_state',
            goal_end_effector_condition_keep_mask=keep_mask,
            goal_reference_pos=reference_pos,
            goal_reference_rot=reference_rot,
            goal_time_frame=torch.tensor([2, 2]),
        )

        expected = torch.tensor((1.0 - 0.5 * 0.05)
                                + (2.0 - 0.5 * 0.05))
        torch.testing.assert_close(loss_dict['goal_end_effector'], expected)
        torch.testing.assert_close(loss_dict['total'], expected)
        torch.testing.assert_close(
            extras['e_goal_end_effector'], torch.tensor(1.5))
        torch.testing.assert_close(
            extras['e_goal_end_effector_left_hand'], torch.tensor(1.0))
        torch.testing.assert_close(
            extras['e_goal_end_effector_right_hand'], torch.tensor(2.0))
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


def test_nested_goal_loss_weights_contribute_to_total():
    old_runtime, old_package = _set_both_feature_versions(6)
    try:
        manager = _manager()
        manager.dataset.dof_dim = 29
        manager.dataset.fps = 50
        manager.loss_weight = {
            'locomotion': {
                'goal': {
                    'root_position_hor': 1.0,
                    'root_position_vert': 0.0,
                },
            },
            'getup': {
                'goal': {
                    'root_position_hor': 10.0,
                    'root_position_vert': 0.0,
                },
            },
        }
        manager.calc_geometry_loss_v6 = _empty_geometry_loss_v6

        rot6d_identity = matrix_to_rot6d(torch.eye(3)).reshape(6)
        history = torch.zeros((2, 2, 44), dtype=torch.float32)
        future = torch.zeros((2, 4, 44), dtype=torch.float32)
        for motion in (history, future):
            motion[..., 0] = 0.77
            motion[..., 1:4] = torch.tensor([0.0, 0.0, -1.0])
            motion[..., 7:13] = rot6d_identity
        future[:, 0:2, 4] = 0.25
        goal = torch.zeros((2, SPLIT_GOAL_DIM), dtype=torch.float32)
        goal_time_frame = torch.tensor([2, 2])

        loss_dict, _ = manager.calc_loss(
            torch.zeros_like(future),
            future,
            None,
            None,
            None,
            None,
            history_motion=history,
            ego_goal=goal,
            goal_type='joint_state',
            goal_time_frame=goal_time_frame,
            is_recovery=torch.tensor([False, True]),
        )

        # Each sample has horizontal Huber mean (0.5 * 0.5^2) / 3.
        per_sample_hor = torch.tensor(0.5 * 0.5**2 / 3.0)
        expected = (per_sample_hor * 1.0 + per_sample_hor * 10.0) / 2.0
        torch.testing.assert_close(loss_dict['total'], expected)
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


def test_nested_goal_g_loss_weights_use_getup_for_recovery_samples():
    old_runtime, old_package = _set_both_feature_versions(6)
    try:
        manager = _manager()
        manager.dataset.dof_dim = 29
        manager.dataset.fps = 50
        manager.loss_weight = {
            'locomotion': {'goal': {'g': 1.0}},
            'getup': {'goal': {'g': 10.0}},
        }
        manager.calc_geometry_loss_v6 = _empty_geometry_loss_v6

        rot6d_identity = matrix_to_rot6d(torch.eye(3)).reshape(6)
        history = torch.zeros((2, 2, 44), dtype=torch.float32)
        future = torch.zeros((2, 4, 44), dtype=torch.float32)
        for motion in (history, future):
            motion[..., 0] = 0.77
            motion[..., 1:4] = torch.tensor([0.0, 0.0, -1.0])
            motion[..., 7:13] = rot6d_identity
        goal = torch.zeros((2, SPLIT_GOAL_DIM), dtype=torch.float32)
        goal[:, SPLIT_VERTICAL_GRAVITY_SLICE] = torch.tensor([0.0, 1.0, 0.0])

        loss_dict, _ = manager.calc_loss(
            torch.zeros_like(future),
            future,
            None,
            None,
            None,
            None,
            history_motion=history,
            ego_goal=goal,
            goal_type='joint_state',
            goal_time_frame=torch.tensor([2, 2]),
            is_recovery=torch.tensor([False, True]),
        )

        # ||[0, 0, -1] - [0, 1, 0]||^2 = 2 for each sample.
        expected = torch.tensor((2.0 * 1.0 + 2.0 * 10.0) / 2.0)
        torch.testing.assert_close(loss_dict['goal_g'], torch.tensor(2.0))
        torch.testing.assert_close(loss_dict['total'], expected)
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


def test_nested_loss_weights_use_getup_for_recovery_samples():
    old_runtime, old_package = _set_both_feature_versions(6)
    try:
        manager = _manager()
        manager.loss_weight = {
            'locomotion': {'rec': 1.0},
            'getup': {'rec': 10.0},
        }
        manager.calc_geometry_loss_v6 = _empty_geometry_loss_v6
        future_gt = torch.zeros((2, 1, 1), dtype=torch.float32)
        future_pred = torch.tensor([[[1.0]], [[2.0]]], dtype=torch.float32)
        latent = torch.zeros((1, 2, 1), dtype=torch.float32)

        loss_dict, _ = manager.calc_loss(
            future_gt,
            future_pred,
            latent,
            None,
            None,
            None,
            history_motion=torch.zeros_like(future_gt),
            is_recovery=torch.tensor([False, True]),
        )

        # Huber(1)=0.5 uses locomotion weight 1;
        # Huber(2)=1.5 uses getup weight 10.
        torch.testing.assert_close(
            loss_dict['total'], torch.tensor((0.5 * 1.0 + 1.5 * 10.0) / 2.0))
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


def test_joint_state_goal_cache_matches_uncached_losses():
    manager = _manager()
    calls = {'denormalize': 0}

    def denormalize(value):
        calls['denormalize'] += 1
        return value

    manager.dataset.denormalize = denormalize
    future = torch.zeros((2, 4, 69), dtype=torch.float32)
    history = torch.zeros((2, 2, 69), dtype=torch.float32)
    goal_time_frame = torch.tensor([2, 3])
    future[:, 1, 0:4] = torch.tensor([0.1, -0.01, -0.2, -0.02])
    future[:, 2, 0:4] = torch.tensor([0.0, 0.02, 0.1, -0.03])
    future[:, :, 4] = 0.1
    history[:, -1, 4] = 0.2
    future[:, 1, 7:10] = torch.tensor([0.02, 0.0, -0.01])
    future[:, 2, 7:10] = torch.tensor([0.0, -0.01, 0.03])
    future[:, 1, 11:40] = torch.linspace(-0.5, 0.5, 29)
    future[:, 2, 11:40] = torch.linspace(0.5, -0.5, 29)
    goal = torch.randn((2, 40), dtype=torch.float32)

    uncached = (
        manager.calc_goal_root_orientation_loss(
            future, goal, history_motion=history,
            goal_time_frame=goal_time_frame),
        manager.calc_goal_joint_angle_loss(
            future, goal, goal_time_frame=goal_time_frame),
        manager.calc_goal_root_velocity_loss(
            future, goal, history_motion=history,
            goal_time_frame=goal_time_frame),
    )
    calls['denormalize'] = 0
    state = manager._future_goal_state(
        future,
        history_motion=history,
        goal_time_frame=goal_time_frame,
        include_yaw=True,
    )
    assert calls['denormalize'] == 2
    calls['denormalize'] = 0
    cached = (
        manager.calc_goal_root_orientation_loss(
            future, goal, history_motion=history,
            goal_time_frame=goal_time_frame, goal_state=state),
        manager.calc_goal_joint_angle_loss(
            future, goal, goal_time_frame=goal_time_frame,
            goal_state=state),
        manager.calc_goal_root_velocity_loss(
            future, goal, history_motion=history,
            goal_time_frame=goal_time_frame, goal_state=state),
    )

    assert calls['denormalize'] == 0
    for actual, expected in zip(cached, uncached):
        torch.testing.assert_close(actual, expected)


def test_root_xy_figure_plots_generated_ground_truth_and_goal():
    generated = torch.tensor([
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]],
    ])
    ground_truth = generated.clone()
    goals = generated[:, -1]

    figure = _make_root_xy_figure(
        generated,
        goals,
        ground_truth_trajectory=ground_truth,
        goal_condition_keep_mask=torch.tensor([True, False]),
    )

    assert len(figure.axes) == 1
    assert 'primitive end error 0.000 m' in figure.axes[0].get_title()
    assert figure.axes[0].get_xlabel() == 'x-forward (m)'
