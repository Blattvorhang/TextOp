import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset.data_process.pack_motion_lib_to_textop import TARGET_DOF_NAMES
from TextOpRobotMDAR.robotmdar.dataloader import data as data_module
from TextOpRobotMDAR.robotmdar.dataloader.data import (
    _HISTORY_JOINT_AUG_AMPS,
    _HISTORY_ROOT_AUG_AMPS,
    SkeletonPrimitiveDataset,
)

motion_dtype = data_module.motion_dtype


@pytest.fixture(autouse=True)
def _restore_feature_version():
    old_version = motion_dtype.FeatureVersion
    yield
    motion_dtype.set_feature_version(old_version)


def _dataset(history_len=16):
    dataset = SkeletonPrimitiveDataset.__new__(SkeletonPrimitiveDataset)
    dataset.dof_dim = 29
    dataset.history_len = history_len
    dataset.augmentation_enabled = True
    dataset.augmentation_start_step = 0
    dataset.augmentation_prob = 1.0
    dataset.training_step = 0
    dataset.split = "train"
    dataset.skeleton = SimpleNamespace(
        fk=SimpleNamespace(dof_joint_names=list(TARGET_DOF_NAMES))
    )
    limits = torch.full((29, 2), 0.0)
    limits[:, 0] = -np.pi
    limits[:, 1] = np.pi
    dataset._joint_limit_tensor = lambda: limits
    return dataset


def _motion(frames, dof=None, root_rot=None):
    if dof is None:
        dof = torch.zeros((frames, 29), dtype=torch.float32)
    if root_rot is None:
        root_rot = torch.zeros((frames, 4), dtype=torch.float32)
        root_rot[:, 3] = 1.0
    return {
        "root_trans_offset": torch.zeros((frames, 3), dtype=torch.float32),
        "root_rot": root_rot.clone(),
        "dof": dof.clone(),
        "contact_mask": torch.ones((frames, 2), dtype=torch.float32),
    }


def _patch_rand(monkeypatch, scalar_values, vector_value):
    scalars = list(scalar_values)

    def fake_rand(*size, generator=None, device=None, dtype=None, **_kwargs):
        del generator
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            shape = tuple(size[0])
        else:
            shape = tuple(size)
        out_dtype = dtype if dtype is not None else torch.float32
        if shape == ():
            return torch.tensor(scalars.pop(0), device=device, dtype=out_dtype)
        return torch.full(shape, vector_value, device=device, dtype=out_dtype)

    monkeypatch.setattr(data_module.torch, "rand", fake_rand)


def _quat_mul_xyzw_np(a, b):
    ax, ay, az, aw = np.moveaxis(np.asarray(a), -1, 0)
    bx, by, bz, bw = np.moveaxis(np.asarray(b), -1, 0)
    return np.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        axis=-1,
    )


def test_history_aug_ramp_matches_v71_boundary_contract():
    weights = SkeletonPrimitiveDataset._history_aug_weight_schedule(
        16, torch.device("cpu"), torch.float32
    )

    assert weights.shape == (17,)
    torch.testing.assert_close(weights[0], torch.tensor(1.0))
    torch.testing.assert_close(weights[8], torch.tensor(0.5))
    torch.testing.assert_close(weights[-1], torch.tensor(0.0))
    torch.testing.assert_close(
        weights[15],
        torch.tensor(1.0 - (3.0 * (15.0 / 16.0) ** 2 - 2.0 * (15.0 / 16.0) ** 3)),
    )
    torch.testing.assert_close(
        weights[1] - weights[0],
        weights[-1] - weights[-2],
        atol=1e-6,
        rtol=1e-6,
    )


def test_history_aug_amplitude_tables_are_v71_normal_set():
    assert dict(_HISTORY_JOINT_AUG_AMPS) == {
        "shoulder": 0.30,
        "elbow": 0.10,
        "hip": 0.05,
        "knee": 0.05,
        "ankle": 0.05,
        "waist": 0.20,
        "wrist": 0.05,
    }
    assert _HISTORY_ROOT_AUG_AMPS == {
        "x": 0.20,
        "y": 0.20,
        "z": 0.05,
        "h": 0.01,
    }


def test_history_aug_root_rotation_matches_body_frame_xyz_numpy_port(monkeypatch):
    H = 4
    dataset = _dataset(history_len=H)
    base_quat = torch.tensor([0.12, -0.25, 0.08, 0.96], dtype=torch.float32)
    base_quat = base_quat / base_quat.norm()
    root_rot = base_quat.repeat(H + 1, 1)
    motion = _motion(H + 1, root_rot=root_rot)
    motion["root_trans_offset"][:, 2] = 0.8

    _patch_rand(
        monkeypatch,
        scalar_values=[0.0, 0.75, 0.25, 0.90, 0.60],
        vector_value=0.5,
    )

    assert dataset._augment_raw_motion(motion, generator=None)

    weights = SkeletonPrimitiveDataset._history_aug_weight_schedule(
        H, torch.device("cpu"), torch.float32
    )[:H].numpy()
    rx = (0.75 * 2.0 - 1.0) * _HISTORY_ROOT_AUG_AMPS["x"]
    ry = (0.25 * 2.0 - 1.0) * _HISTORY_ROOT_AUG_AMPS["y"]
    rz = (0.90 * 2.0 - 1.0) * _HISTORY_ROOT_AUG_AMPS["z"]
    half = 0.5 * weights
    zeros = np.zeros_like(half)
    q_x = np.stack([np.sin(half * rx), zeros, zeros, np.cos(half * rx)], axis=-1)
    q_y = np.stack([zeros, np.sin(half * ry), zeros, np.cos(half * ry)], axis=-1)
    q_z = np.stack([zeros, zeros, np.sin(half * rz), np.cos(half * rz)], axis=-1)
    q_off = _quat_mul_xyzw_np(q_x, _quat_mul_xyzw_np(q_y, q_z))
    expected = _quat_mul_xyzw_np(root_rot[:H].numpy(), q_off)

    torch.testing.assert_close(
        motion["root_rot"][:H],
        torch.from_numpy(expected),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(motion["root_rot"][H], root_rot[H])
    torch.testing.assert_close(
        motion["root_trans_offset"][:H, 2],
        torch.as_tensor(0.8 + weights * 0.002),
    )
    torch.testing.assert_close(motion["root_trans_offset"][H, 2], torch.tensor(0.8))


def test_history_aug_joint_sampling_intersects_limits_before_draw(monkeypatch):
    H = 2
    dataset = _dataset(history_len=H)
    limits = torch.full((29, 2), 0.0)
    limits[:, 0] = -np.pi
    limits[:, 1] = np.pi
    limits[0] = torch.tensor([-1.0, 1.0])
    limits[1] = torch.tensor([-1.0, 1.0])
    dataset._joint_limit_tensor = lambda: limits
    dof = torch.zeros((H + 1, 29), dtype=torch.float32)
    dof[:H, 0] = torch.tensor([0.95, 0.80])
    dof[:H, 1] = torch.tensor([1.20, 1.20])
    motion = _motion(H + 1, dof=dof)

    _patch_rand(
        monkeypatch,
        scalar_values=[0.0, 0.5, 0.5, 0.5, 0.5],
        vector_value=1.0,
    )

    assert dataset._augment_raw_motion(motion, generator=None)

    torch.testing.assert_close(
        motion["dof"][:H, 0],
        torch.tensor([1.0, 0.825]),
    )
    torch.testing.assert_close(
        motion["dof"][:H, 1],
        torch.tensor([1.20, 1.20]),
    )
    torch.testing.assert_close(motion["dof"][H, :2], dof[H, :2])


def test_history_aug_runs_when_feature_version_is_v6(monkeypatch):
    motion_dtype.set_feature_version(6)
    H = 3
    dataset = _dataset(history_len=H)
    motion = _motion(H + 1)

    _patch_rand(
        monkeypatch,
        scalar_values=[0.0, 0.5, 0.5, 0.5, 0.5],
        vector_value=1.0,
    )

    assert dataset._augment_raw_motion(motion, generator=None)
    assert bool((motion["dof"][:H].abs() > 0).any())
    torch.testing.assert_close(motion["dof"][H], torch.zeros(29))


def _batch_dataset(feature_width):
    dataset = SkeletonPrimitiveDataset.__new__(SkeletonPrimitiveDataset)
    dataset.num_primitive = 1
    dataset.batch_size = 1
    dataset.history_len = 2
    dataset.future_len = 1
    dataset.dof_dim = 29
    dataset.goal_type = "root"
    dataset.normalize = lambda feature: feature
    dataset._select_model_dof = lambda value: value
    dataset._augment_raw_motion = lambda motion, generator=None: True
    dataset._convert_to_motion_features = lambda motion: torch.zeros(
        (1, 3, feature_width), dtype=torch.float32
    )
    return dataset


def _primitive():
    root_rot = torch.zeros((4, 4), dtype=torch.float32)
    root_rot[:, 3] = 1.0
    root_pos = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    return {
        "motion": {
            "root_trans_offset": root_pos,
            "root_rot": root_rot,
            "dof": torch.zeros((4, 29), dtype=torch.float32),
            "contact_mask": torch.ones((4, 2), dtype=torch.float32),
        },
        "sliding_mask": torch.zeros((4, 2), dtype=torch.float32),
        "scene": {},
        "world_goal_pos": torch.zeros(3),
        "world_goal_yaw": torch.tensor(0.0),
        "history_start_pos": torch.zeros(3),
        "history_start_rot": torch.zeros(4),
        "gt_ref_pos": torch.zeros(3),
        "gt_ref_rot": torch.zeros(4),
        "is_recovery": True,
        "action_label": "stand_up_lying",
        "_clean_history_delta_q": torch.arange(29, dtype=torch.float32) + 10.0,
    }


def test_organize_restores_clean_history_delta_only_for_v3():
    motion_dtype.set_feature_version(3)
    dataset = _batch_dataset(feature_width=69)

    batch = dataset._organize_primitives_by_index([[_primitive()]])[0]

    torch.testing.assert_close(
        batch["motion"][0, 1, 40:69],
        torch.arange(29, dtype=torch.float32) + 10.0,
    )


def test_organize_skips_clean_history_delta_restore_for_v6():
    motion_dtype.set_feature_version(6)
    dataset = _batch_dataset(feature_width=44)

    batch = dataset._organize_primitives_by_index([[_primitive()]])[0]

    assert batch["motion"].shape == (1, 3, 44)
    torch.testing.assert_close(batch["motion"], torch.zeros((1, 3, 44)))
    torch.testing.assert_close(
        batch["gt_ref_pos"][0],
        torch.tensor([6.0, 7.0, 8.0]),
    )
