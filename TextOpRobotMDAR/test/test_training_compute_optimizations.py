import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
