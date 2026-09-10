import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import robotmdar.dtype.motion as runtime_motion_dtype
import TextOpRobotMDAR.robotmdar.dtype.motion as package_motion_dtype
import TextOpRobotMDAR.robotmdar.train.train_dar as train_dar_module
import TextOpRobotMDAR.robotmdar.train.manager as manager_module
from TextOpRobotMDAR.robotmdar.dataloader.data import SkeletonPrimitiveDataset
from TextOpRobotMDAR.robotmdar.diffusion.gaussian_diffusion import (
    _EXTRACT_TENSOR_CACHE,
    _extract_into_tensor,
)
from TextOpRobotMDAR.robotmdar.model.mld_denoiser import (
    DenoiserMLP,
    DenoiserTransformer,
)
from TextOpRobotMDAR.robotmdar.dtype.rotation import (
    quaternion_to_matrix,
    xyzw_to_wxyz,
)
from TextOpRobotMDAR.robotmdar.train.manager import (
    BaseManager,
    GeometryLoss,
    _standard_normal_kl_mean,
)
from TextOpRobotMDAR.robotmdar.train.train_dar import (
    _BackgroundPrefetchIterator,
    _add_batch_data_diagnostics,
    _build_train_condition_plan,
    _conditions,
)
from TextOpRobotMDAR.robotmdar.utils.goal import (
    SPLIT_NO_LOG_ORIENTATION_SLICE,
    SPLIT_NO_LOG_VERTICAL_GRAVITY_SLICE,
)
from TextOpRobotMDAR.robotmdar.utils.occupancy import (
    _local_grid_offsets,
    compute_scene_surface,
    compute_scene_surface_batch,
    query_local_occupancy,
)


class DummyManager(BaseManager):
    def hold_model(self, *args, **kwargs):
        pass

    def calc_loss(self, *args, **kwargs):
        raise NotImplementedError

    def update_ema_models(self):
        pass

    def save_model(self):
        pass

    def load_model(self, *args, **kwargs):
        pass


def _manager(max_grad_norm=0.5, eval_steps=2):
    model = torch.nn.Linear(2, 1)
    manager = DummyManager(
        stages=[10],
        use_rollout=False,
        use_static_pose=False,
        anneal_lr=False,
        learning_rate=1e-3,
        max_grad_norm=max_grad_norm,
        loss_weight={},
        ckpt={},
        device='cpu',
        platform=SimpleNamespace(report_scalar=lambda *args, **kwargs: None),
        save_every=1000,
        eval_every=1000,
        eval_steps=eval_steps,
        save_dir='/tmp/robotmdar-test',
    )
    manager.optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return manager


def _model_with_grads():
    torch.manual_seed(7)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3),
        torch.nn.LayerNorm(3),
        torch.nn.Linear(3, 2),
    )
    for idx, param in enumerate(model.parameters()):
        param.grad = torch.linspace(
            -1.0, 1.0, param.numel(), dtype=param.dtype
        ).reshape_as(param) + idx * 0.01
    return model


def _manual_bad_grad_scan_and_clip(model, max_grad_norm):
    has_bad_grad = False
    for param in model.parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                has_bad_grad = True
    norm = None
    if not has_bad_grad and max_grad_norm > 0:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    return not has_bad_grad, norm


def test_clip_grad_and_check_matches_old_finite_path():
    old_model = _model_with_grads()
    new_model = _model_with_grads()
    max_norm = 0.5

    expected_ok, expected_norm = _manual_bad_grad_scan_and_clip(
        old_model, max_norm)
    manager = _manager(max_grad_norm=max_norm)
    actual_ok = manager.clip_grad_and_check(new_model)

    assert actual_ok is expected_ok is True
    torch.testing.assert_close(manager.extra['grad_norm'], expected_norm)
    for old_param, new_param in zip(old_model.parameters(), new_model.parameters()):
        torch.testing.assert_close(old_param.grad, new_param.grad)


def test_background_prefetch_iterator_preserves_batches():
    prefetched = _BackgroundPrefetchIterator(
        iter([{"segment": 0}, {"segment": 1}, {"segment": 2}]))
    try:
        assert next(prefetched) == {"segment": 0}
        assert next(prefetched) == {"segment": 1}
        assert next(prefetched) == {"segment": 2}
    finally:
        prefetched.close()


def test_batch_data_diagnostics_split_recovery_condition_keep_ratios():
    extras = {}
    primitive = {
        "is_recovery": torch.tensor([False, True, True]),
        "action_label": ["walk", "", "stand up"],
        "text_embedding": torch.tensor([
            [1.0, 0.0],
            [0.0, 0.0],
            [0.5, 0.5],
        ]),
    }
    y = {
        "goal": torch.zeros(3, 66),
        "goal_position_hor_condition_keep_mask": torch.tensor(
            [True, False, True]),
        "goal_position_vert_condition_keep_mask": torch.tensor(
            [True, True, True]),
        "goal_gravity_condition_keep_mask": torch.tensor(
            [False, True, False]),
        "goal_orientation_condition_keep_mask": torch.tensor(
            [True, False, False]),
        "goal_joint_condition_keep_mask": torch.tensor(
            [True, False, False]),
        "goal_velocity_condition_keep_mask": torch.tensor(
            [True, False, True]),
        "goal_time_condition_keep_mask": torch.tensor(
            [True, True, False]),
        "goal_end_effector_condition_keep_mask": torch.tensor([
            [True, True, False, False],
            [False, False, False, False],
            [True, False, True, False],
        ]),
    }

    _add_batch_data_diagnostics(extras, primitive, y)

    torch.testing.assert_close(
        extras["data/batch_recovery_fraction"], torch.tensor(2.0 / 3.0))
    torch.testing.assert_close(
        extras["data/recovery_action_label_empty_rate"], torch.tensor(0.5))
    torch.testing.assert_close(
        extras["data/recovery_text_embedding_empty_rate"], torch.tensor(0.5))
    torch.testing.assert_close(
        extras["condition/recovery_position_hor_keep_ratio"], torch.tensor(0.5))
    torch.testing.assert_close(
        extras["condition/locomotion_position_hor_keep_ratio"], torch.tensor(1.0))
    torch.testing.assert_close(
        extras["condition/recovery_gravity_keep_ratio"], torch.tensor(0.5))
    torch.testing.assert_close(
        extras["condition/recovery_end_effector_keep_ratio"], torch.tensor(0.25))


def _goal_condition_test_config(loss_weight, goal_dim=66):
    return OmegaConf.create({
        'device': 'cpu',
        'data': {
            'goal_type': 'joint_state',
            'goal_encoding': 'split_end_effector',
            'occupancy_unit': 0.1,
        },
        'denoiser': {
            'goal_dim': goal_dim,
            'goal_encoding': 'split_end_effector',
            'grid_size': 1,
            'text_condition_enabled': True,
        },
        'train': {
            'manager': {
                'loss_weight': loss_weight,
            },
        },
    })


def _goal_condition_test_profiles(mask_value=1.0, gravity=None):
    names = ('left_hand', 'right_hand', 'left_foot', 'right_foot')
    profile = {
        'text': mask_value,
        'goal': {
            'position': {'hor': mask_value, 'vert': mask_value},
            'velocity': mask_value,
            'orientation': {
                'rot6d': mask_value,
                'gravity': mask_value if gravity is None else gravity,
            },
            'joint': mask_value,
            'end_effector': {name: mask_value for name in names},
            'time': mask_value,
        },
        'scene': mask_value,
    }
    return {
        'locomotion': profile,
        'getup': profile,
    }


def test_train_condition_plan_skips_zero_weight_and_always_dropped_inputs():
    cfg = _goal_condition_test_config({
        'locomotion': {'goal': {
            'root_position_hor': 0.0,
            'root_position_vert': 0.0,
            'root_velocity': 0.0,
            'root_orientation': 0.0,
            'g': 0.0,
            'joint_angle': 0.0,
            'end_effector': 0.0,
        }},
        'getup': {'goal': {
            'root_position_hor': 0.0,
            'root_position_vert': 0.0,
            'root_velocity': 0.0,
            'root_orientation': 0.0,
            'g': 0.0,
            'joint_angle': 0.0,
            'end_effector': 0.0,
        }},
    })
    denoiser = SimpleNamespace(
        cond_mask_prob_profiles=_goal_condition_test_profiles())

    plan = _build_train_condition_plan(cfg, denoiser)

    assert plan['goal_any'] is False
    assert plan['goal_end_effector'] == (False, False, False, False)
    assert plan['scene'] is False
    assert plan['text'] is False
    assert plan['goal_time'] is False


def test_train_condition_plan_keeps_orientation_source_for_gravity_only():
    loss_weight = {
        'locomotion': {'goal': {'root_orientation': 0.0, 'g': 1.0}},
        'getup': {'goal': {'root_orientation': 0.0, 'g': 1.0}},
    }
    cfg = _goal_condition_test_config(loss_weight)
    denoiser = SimpleNamespace(
        cond_mask_prob_profiles=_goal_condition_test_profiles(
            mask_value=1.0, gravity=0.2))

    plan = _build_train_condition_plan(cfg, denoiser)

    assert plan['goal_orientation'] is False
    assert plan['goal_orientation_source'] is True
    assert plan['goal_gravity'] is True
    assert plan['goal_any'] is True


def test_conditions_do_not_build_disabled_goal_or_scene(monkeypatch):
    old_runtime = runtime_motion_dtype.FeatureVersion
    old_package = package_motion_dtype.FeatureVersion
    runtime_motion_dtype.set_feature_version(6)
    package_motion_dtype.set_feature_version(6)
    try:
        cfg = _goal_condition_test_config({
            'locomotion': {'goal': {
                'root_position_hor': 0.0,
                'root_position_vert': 0.0,
                'root_velocity': 0.0,
                'root_orientation': 0.0,
                'g': 0.0,
                'joint_angle': 0.0,
                'end_effector': 0.0,
            }},
            'getup': {'goal': {
                'root_position_hor': 0.0,
                'root_position_vert': 0.0,
                'root_velocity': 0.0,
                'root_orientation': 0.0,
                'g': 0.0,
                'joint_angle': 0.0,
                'end_effector': 0.0,
            }},
        })
        plan = {
            'goal_position': False,
            'goal_position_hor': False,
            'goal_position_vert': False,
            'goal_orientation': False,
            'goal_orientation_source': False,
            'goal_gravity': False,
            'goal_joint': False,
            'goal_velocity': False,
            'goal_end_effector': (False, False, False, False),
            'goal_any': False,
            'goal_time': False,
            'scene': False,
            'text': False,
        }
        primitive = {
            'world_goal_pos': torch.ones(1, 3),
            'world_goal_rot': torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            'world_goal_dof': torch.ones(1, 29),
            'world_goal_vel': torch.ones(1, 3),
            'world_goal_end_effectors': torch.ones(1, 4, 3),
            'time_to_arrival': torch.ones(1),
            'scene': [],
        }
        for name in (
                'build_ego_joint_state_goal_v6',
                'build_ego_split_end_effector_goal_no_log'):
            monkeypatch.setattr(
                train_dar_module,
                name,
                lambda *args, builder_name=name, **kwargs: pytest.fail(
                    f'{builder_name} should not be called'),
            )
        monkeypatch.setattr(
            train_dar_module,
            'query_local_occupancy',
            lambda *args, **kwargs: pytest.fail(
                'scene occupancy should not be queried'),
        )

        conditions = _conditions(
            primitive,
            torch.zeros(1, 3),
            torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            torch.zeros(1, 2, 44),
            cfg,
            fps=50.0,
            goal_stats=None,
            use_scene=True,
            condition_plan=plan,
        )

        assert conditions['ego_goal_raw'] is None
        assert conditions['goal'].shape == (1, 66)
        assert conditions['goal'].abs().sum() == 0
        assert conditions['voxel'].abs().sum() == 0
        assert conditions['force_drop_scene'] is True
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


def test_conditions_build_gravity_source_without_rot6d_condition():
    old_runtime = runtime_motion_dtype.FeatureVersion
    old_package = package_motion_dtype.FeatureVersion
    runtime_motion_dtype.set_feature_version(6)
    package_motion_dtype.set_feature_version(6)
    try:
        cfg = _goal_condition_test_config({
            'locomotion': {'goal': {'g': 1.0}},
            'getup': {'goal': {'g': 1.0}},
        })
        plan = {
            'goal_position': False,
            'goal_position_hor': False,
            'goal_position_vert': False,
            'goal_orientation': False,
            'goal_orientation_source': True,
            'goal_gravity': True,
            'goal_joint': False,
            'goal_velocity': False,
            'goal_end_effector': (False, False, False, False),
            'goal_any': True,
            'goal_time': False,
            'scene': False,
            'text': False,
        }
        primitive = {
            'world_goal_pos': torch.zeros(1, 3),
            'world_goal_rot': torch.tensor(
                [[0.0, 0.70710677, 0.0, 0.70710677]]),
            'world_goal_dof': torch.zeros(1, 29),
            'world_goal_vel': torch.zeros(1, 3),
            'world_goal_end_effectors': torch.zeros(1, 4, 3),
            'time_to_arrival': torch.ones(1),
            'scene': [],
        }
        stats = {
            's_p': torch.tensor(1.0),
            's_v': torch.tensor(1.0),
            's_d': torch.tensor(1.0),
            's_o': torch.ones(9),
            'q_mean': torch.zeros(29),
            'q_std': torch.ones(29),
            's_ee': torch.ones(12),
            'meta': {
                'fps': 50.0,
                'goal_dim': 66,
                'goal_schema': (
                    'rotmat_v10_hor_vert_joint_ee_no_log'),
            },
        }
        conditions = _conditions(
            primitive,
            torch.zeros(1, 3),
            torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            torch.zeros(1, 2, 44),
            cfg,
            fps=50.0,
            goal_stats=stats,
            use_scene=False,
            condition_plan=plan,
        )

        assert conditions['goal'][:, SPLIT_NO_LOG_ORIENTATION_SLICE].abs().sum() == 0
        assert conditions['goal'][
            :, SPLIT_NO_LOG_VERTICAL_GRAVITY_SLICE].abs().sum() > 0.5
        assert conditions['force_drop_goal_orientation_rot6d'] is True
        assert conditions['force_drop_goal_gravity'] is False
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


def test_conditions_skip_end_effector_transform_when_all_ee_tokens_are_off(
        monkeypatch):
    old_runtime = runtime_motion_dtype.FeatureVersion
    old_package = package_motion_dtype.FeatureVersion
    runtime_motion_dtype.set_feature_version(6)
    package_motion_dtype.set_feature_version(6)
    try:
        loss_weight = {
            'locomotion': {'goal': {'root_position_hor': 1.0}},
            'getup': {'goal': {'root_position_hor': 1.0}},
        }
        cfg = _goal_condition_test_config(loss_weight)
        profiles = _goal_condition_test_profiles(mask_value=1.0)
        for profile in profiles.values():
            profile['goal']['position']['hor'] = 0.0
        denoiser = SimpleNamespace(cond_mask_prob_profiles=profiles)
        plan = _build_train_condition_plan(cfg, denoiser)
        assert plan['goal_position_hor'] is True
        assert plan['goal_end_effector'] == (False, False, False, False)

        primitive = {
            'world_goal_pos': torch.tensor([[1.0, 0.0, 0.8]]),
            'world_goal_rot': torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            'world_goal_dof': torch.ones(1, 29),
            'world_goal_vel': torch.ones(1, 3),
            'world_goal_end_effectors': torch.ones(1, 4, 3),
            'time_to_arrival': torch.ones(1),
            'scene': [],
        }
        stats = {
            's_p': torch.tensor(1.0),
            's_v': torch.tensor(1.0),
            's_d': torch.tensor(1.0),
            's_o': torch.ones(9),
            'q_mean': torch.zeros(29),
            'q_std': torch.ones(29),
            's_ee': torch.ones(12),
            'meta': {
                'fps': 50.0,
                'goal_dim': 66,
                'goal_schema': (
                    'rotmat_v10_hor_vert_joint_ee_no_log'),
            },
        }
        monkeypatch.setattr(
            train_dar_module,
            'build_ego_split_end_effector_goal_no_log',
            lambda *args, **kwargs: pytest.fail(
                'disabled end-effector tokens should skip EE transform'),
        )

        conditions = _conditions(
            primitive,
            torch.zeros(1, 3),
            torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            torch.zeros(1, 2, 44),
            cfg,
            fps=50.0,
            goal_stats=stats,
            use_scene=False,
            condition_plan=plan,
        )

        assert conditions['goal'].shape == (1, 66)
        assert conditions['goal'][:, 54:].abs().sum() == 0
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


def test_goal_losses_return_before_state_or_fk_when_condition_is_fully_dropped(
        monkeypatch):
    old_runtime = runtime_motion_dtype.FeatureVersion
    old_package = package_motion_dtype.FeatureVersion
    runtime_motion_dtype.set_feature_version(6)
    package_motion_dtype.set_feature_version(6)
    try:
        geometry = object.__new__(GeometryLoss)
        geometry.rec_criterion = torch.nn.HuberLoss(
            reduction='mean', delta=1.0)
        future = torch.randn(2, 4, 44, requires_grad=True)
        goal = torch.zeros(2, 66)
        history = torch.zeros(2, 2, 44)

        monkeypatch.setattr(
            geometry,
            '_future_goal_state_v6',
            lambda *args, **kwargs: pytest.fail(
                'fully dropped gravity should skip goal-state reconstruction'),
        )
        loss, per_sample, valid = geometry.calc_goal_g_loss(
            future,
            goal,
            torch.zeros(2, dtype=torch.bool),
            history_motion=history,
            return_per_sample=True,
            skip_if_dropped=True,
        )
        assert loss is not None
        assert per_sample.shape == (2,)
        assert valid.tolist() == [False, False]

        monkeypatch.setattr(
            geometry,
            '_end_effector_anchors',
            lambda: pytest.fail(
                'fully dropped end-effectors should skip FK/anchor lookup'),
        )
        ee_loss, ee_per_sample, ee_valid, metrics = (
            geometry.calc_goal_end_effector_loss(
                future,
                goal,
                torch.zeros(2, 4, dtype=torch.bool),
                return_per_sample=True,
                skip_if_dropped=True,
            )
        )
        assert ee_loss is not None
        assert ee_per_sample.shape == (2, 4)
        assert ee_valid.shape == (2, 4)
        assert all(value == 0 for value in metrics.values())
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


def test_eval_goal_loss_keeps_goal_state_compute_when_condition_is_dropped(
        monkeypatch):
    old_runtime = runtime_motion_dtype.FeatureVersion
    old_package = package_motion_dtype.FeatureVersion
    runtime_motion_dtype.set_feature_version(6)
    package_motion_dtype.set_feature_version(6)
    try:
        geometry = object.__new__(GeometryLoss)
        geometry.rec_criterion = torch.nn.HuberLoss(
            reduction='mean', delta=1.0)
        future = torch.randn(2, 4, 44)
        goal = torch.zeros(2, 66)
        goal[:, SPLIT_NO_LOG_VERTICAL_GRAVITY_SLICE] = torch.tensor(
            [0.0, 0.0, -1.0])
        history = torch.zeros(2, 2, 44)
        calls = []

        def fake_goal_state(*args, **kwargs):
            calls.append(True)
            return {
                'gravity_at_goal': torch.tensor(
                    [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
            }

        monkeypatch.setattr(
            geometry, '_future_goal_state_v6', fake_goal_state)
        loss, per_sample, valid = geometry.calc_goal_g_loss(
            future,
            goal,
            torch.zeros(2, dtype=torch.bool),
            history_motion=history,
            return_per_sample=True,
            skip_if_dropped=False,
        )

        assert calls == [True]
        assert loss is not None
        assert per_sample.shape == (2,)
        assert valid.tolist() == [False, False]
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


@pytest.mark.parametrize('bad_value', [float('nan'), float('inf')])
def test_clip_grad_and_check_matches_old_bad_grad_decision(bad_value):
    old_model = _model_with_grads()
    new_model = _model_with_grads()
    next(old_model.parameters()).grad.view(-1)[0] = bad_value
    next(new_model.parameters()).grad.view(-1)[0] = bad_value

    expected_ok, _ = _manual_bad_grad_scan_and_clip(
        old_model, max_grad_norm=0.5)
    manager = _manager(max_grad_norm=0.5)
    actual_ok = manager.clip_grad_and_check(new_model)

    assert expected_ok is False
    assert actual_ok is False


def test_eval_metric_accumulation_detaches_without_reporting(monkeypatch):
    monkeypatch.setattr(manager_module, 'is_main_process', lambda: False)
    manager = _manager(eval_steps=2)
    manager._to_eval_steps = 2
    metric = torch.tensor(2.0, requires_grad=True)

    manager.post_step(is_eval=True, loss_dict={'total': metric},
                      extras={'twice': metric * 2})

    assert manager._to_eval_steps == 1
    assert manager._total_eval_loss_dict['total'].grad_fn is None
    assert manager._total_eval_extras_dict['twice'].grad_fn is None
    torch.testing.assert_close(
        manager._total_eval_loss_dict['total'], torch.tensor(2.0))
    torch.testing.assert_close(
        manager._total_eval_extras_dict['twice'], torch.tensor(4.0))


def test_batched_scene_surface_matches_per_sample_loop():
    torch.manual_seed(11)
    occupancy = torch.rand(6, 7, 7, 7) > 0.35
    thicknesses = [0, 1, 2, 3, 1, 2]

    expected = torch.stack([
        compute_scene_surface(occupancy[idx], thickness=thickness)
        for idx, thickness in enumerate(thicknesses)
    ])
    actual = compute_scene_surface_batch(occupancy, thicknesses)

    torch.testing.assert_close(actual, expected)


def test_local_occupancy_offset_cache_preserves_result():
    occupancy = torch.zeros(5, 5, 5, dtype=torch.bool).numpy()
    occupancy[2, 2, 2] = True
    scenes = [{
        'occu_global': occupancy,
        'unit': 1.0,
        'llb': [-2.0, -2.0, -2.0],
    }]
    reference_pos = torch.tensor([[0.0, 0.0, 0.0]])
    reference_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

    _local_grid_offsets.cache_clear()
    first = query_local_occupancy(
        scenes, reference_pos, reference_rot, grid_size=3, grid_unit=1.0)
    second = query_local_occupancy(
        scenes, reference_pos, reference_rot, grid_size=3, grid_unit=1.0)

    assert _local_grid_offsets.cache_info().hits >= 1
    _local_grid_offsets.cache_clear()
    recomputed = query_local_occupancy(
        scenes, reference_pos, reference_rot, grid_size=3, grid_unit=1.0)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first, recomputed)


def test_standard_normal_kl_matches_torch_distribution_formula():
    torch.manual_seed(17)
    loc = torch.randn(3, 4, 5)
    scale = torch.exp(torch.randn(3, 4, 5).clamp(-2.0, 2.0))
    dist = torch.distributions.Normal(loc, scale)

    expected = torch.distributions.kl_divergence(
        dist,
        torch.distributions.Normal(torch.zeros_like(loc), torch.ones_like(scale)),
    ).mean()
    actual = _standard_normal_kl_mean(dist)

    torch.testing.assert_close(actual, expected)


def test_quaternion_chordal_loss_matches_matrix_formula():
    torch.manual_seed(23)
    q_pred = torch.randn(4, 5, 6, 4)
    q_gt = torch.randn(4, 5, 6, 4)

    pred_R = quaternion_to_matrix(xyzw_to_wxyz(q_pred))
    gt_R = quaternion_to_matrix(xyzw_to_wxyz(q_gt))
    expected = (pred_R - gt_R).square().sum(dim=(-1, -2)).mean()
    actual = GeometryLoss._quat_chordal_loss(q_pred, q_gt)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        GeometryLoss._quat_chordal_loss(q_pred, -q_gt),
        actual,
        atol=1e-6,
        rtol=1e-6,
    )


def test_extract_into_tensor_cache_preserves_numpy_lookup_result():
    _EXTRACT_TENSOR_CACHE.clear()
    arr = np.linspace(0.1, 0.9, 5, dtype=np.float64)
    timesteps = torch.tensor([4, 1, 0], dtype=torch.long)
    expected = torch.from_numpy(arr)[timesteps].float()
    expected = expected[:, None].expand(3, 2)

    first = _extract_into_tensor(arr, timesteps, (3, 2))
    second = _extract_into_tensor(arr, timesteps, (3, 2))

    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)
    assert len(_EXTRACT_TENSOR_CACHE) == 1


def test_expand_token_matches_tile_values_and_gradients():
    torch.manual_seed(29)
    token_expand = torch.randn(2, 3, requires_grad=True)
    token_tile = token_expand.detach().clone().requires_grad_(True)
    batch_size = 5
    weights = torch.randn(2, batch_size, 3)

    expanded = token_expand[:, None, :].expand(-1, batch_size, -1)
    tiled = torch.tile(token_tile[:, None, :], (1, batch_size, 1))

    torch.testing.assert_close(expanded, tiled)
    (expanded * weights).sum().backward()
    (tiled * weights).sum().backward()
    torch.testing.assert_close(token_expand.grad, token_tile.grad)


def test_dataset_stats_cache_preserves_normalize_and_denormalize():
    dataset = object.__new__(SkeletonPrimitiveDataset)
    dataset.nfeats = 3
    dataset.std_floor = 0.0
    dataset.mean = torch.tensor([1.0, -2.0, 0.5])
    dataset.std = torch.tensor([2.0, 4.0, 0.25])
    dataset._stats_device_cache = {}
    feat = torch.tensor([[3.0, 2.0, 1.0], [5.0, -6.0, 0.0]])

    expected_norm = (feat - dataset.mean.to(feat.device)) / dataset.std.to(
        feat.device)
    actual_norm = dataset.normalize(feat)
    torch.testing.assert_close(actual_norm, expected_norm)
    torch.testing.assert_close(dataset.denormalize(actual_norm), feat)
    assert len(dataset._stats_device_cache) == 1

    dataset._set_meanstd(
        (torch.zeros(3), torch.ones(3)),
        Path("synthetic-stats.pkl"),
    )
    assert dataset._stats_device_cache == {}


@pytest.mark.parametrize('model_cls', [DenoiserMLP, DenoiserTransformer])
def test_mask_condition_fast_path_preserves_values_and_rng(model_cls):
    model = model_cls.__new__(model_cls)
    cond = torch.randn(4, 5)

    model.training = True
    torch.manual_seed(123)
    rng_before = torch.random.get_rng_state()
    masked, keep = model_cls.mask_condition(
        model, cond, 0.0, return_keep_mask=True)
    rng_after = torch.random.get_rng_state()
    assert masked is cond
    assert keep.tolist() == [True, True, True, True]
    torch.testing.assert_close(rng_after, rng_before)

    model.training = False
    torch.manual_seed(456)
    rng_before = torch.random.get_rng_state()
    masked = model_cls.mask_condition(model, cond, 0.9)
    rng_after = torch.random.get_rng_state()
    assert masked is cond
    torch.testing.assert_close(rng_after, rng_before)
