from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from robotmdar.dataloader.data import SkeletonPrimitiveDataset
from robotmdar.dtype.motion import (
    G1_23DOF_FROM_29DOF_INDICES,
    G1_CORE_DOF_INDICES,
    G1_WRIST_DOF_INDICES,
    motion_dict_to_feature_v3,
    motion_feature_to_dict_v3,
    set_feature_version,
)
from robotmdar.skeleton.robot import RobotSkeleton
from robotmdar.utils.dof_contract import expected_g1_names
from robotmdar.utils.dof_contract import configure_dof_contract


ROOT = Path(__file__).resolve().parents[2]


def _motion(dof):
    batch, frames = dof.shape[:2]
    root_rot = torch.zeros((batch, frames, 4), dtype=dof.dtype)
    root_rot[..., 3] = 1.0
    return {
        'root_trans_offset': torch.zeros((batch, frames, 3)),
        'root_rot': root_rot,
        'dof': dof,
        'contact_mask': torch.ones((batch, frames, 2)),
    }


def _skeleton_23dof():
    config_dir = ROOT / 'TextOpRobotMDAR/robotmdar/config/skeleton'
    cfg = OmegaConf.merge(
        OmegaConf.load(config_dir / 'g1.yaml'),
        OmegaConf.load(config_dir / 'g1_23dof.yaml'),
    )
    cfg.asset.assetRoot = str(
        ROOT / 'TextOpRobotMDAR/description/robots/g1'
    )
    cfg.dof_dim = 23
    return RobotSkeleton(device='cpu', cfg=cfg)


def test_23dof_projection_removes_only_wrist_joints_in_mujoco_order():
    assert G1_23DOF_FROM_29DOF_INDICES == G1_CORE_DOF_INDICES
    assert G1_WRIST_DOF_INDICES == (19, 20, 21, 26, 27, 28)

    dataset = SkeletonPrimitiveDataset.__new__(SkeletonPrimitiveDataset)
    dataset.dof_dim = 23
    source = torch.arange(29, dtype=torch.float32)

    projected = dataset._select_model_dof(source)

    torch.testing.assert_close(
        projected,
        source[list(G1_23DOF_FROM_29DOF_INDICES)],
    )
    assert projected.tolist() == list(range(19)) + list(range(22, 26))


def test_57d_feature_round_trip_preserves_23dof_position_and_delta_order():
    frames = 4
    dof = (
        torch.arange(frames, dtype=torch.float32)[:, None] * 100.0
        + torch.arange(23, dtype=torch.float32)[None]
    ).unsqueeze(0)

    feature, abs_pose = motion_dict_to_feature_v3(_motion(dof))
    reconstructed = motion_feature_to_dict_v3(feature, abs_pose)

    assert feature.shape == (1, frames - 1, 57)
    torch.testing.assert_close(feature[..., 11:34], dof[:, :-1])
    torch.testing.assert_close(feature[..., 34:57], dof[:, 1:] - dof[:, :-1])
    torch.testing.assert_close(reconstructed['dof'], dof[:, :-1])


def test_locked_wrist_skeleton_matches_canonical_23dof_order():
    skeleton = _skeleton_23dof()
    expected_joints, expected_links = expected_g1_names(23)

    assert skeleton.fk.num_dof == 23
    assert tuple(skeleton.fk.dof_joint_names) == expected_joints
    assert tuple(skeleton.fk.body_names[1:]) == expected_links
    assert tuple(
        skeleton.fk.body_names_augment[index]
        for index in skeleton.hand_id
    ) == ('left_hand_link', 'right_hand_link')


def test_train_config_selects_matching_features_skeleton_and_vae():
    config_dir = ROOT / 'TextOpRobotMDAR/robotmdar/config'
    with initialize_config_dir(
        config_dir=str(config_dir.resolve()), version_base=None
    ):
        v6 = compose(config_name='train_dar')
        configure_dof_contract(v6)
        legacy_23 = compose(
            config_name='train_dar',
            overrides=[
                'data.dof_dim=23',
                'data.goal_type=root',
                'denoiser.goal_dim=5',
            ],
        )
        configure_dof_contract(legacy_23)
        planner = compose(config_name='planner_dar')
        configure_dof_contract(planner)
        rotmat_v6 = compose(
            config_name='train_mvae',
            overrides=[
                'feature_version=6',
                'data.dof_dim=29',
            ],
        )
        configure_dof_contract(rotmat_v6)
        rotmat_v6_data_override = compose(
            config_name='train_mvae',
            overrides=[
                'data.feature_version=6',
                'data.dof_dim=29',
            ],
        )
        configure_dof_contract(rotmat_v6_data_override)

    assert int(v6.data.dof_dim) == 29
    assert int(v6.data.nfeats) == 69
    assert int(v6.feature_version) == 3
    assert v6.data.goal_type == 'joint_state'
    assert int(v6.denoiser.goal_dim) == 45
    assert v6.skeleton.asset.assetFileName == 'g1_29dof.xml'
    assert v6.ckpt.vae is None
    assert int(legacy_23.data.nfeats) == 57
    assert legacy_23.skeleton.asset.assetFileName == (
        'g1_23dof_lock_wrist_fitmotionONLY.xml'
    )
    assert legacy_23.ckpt.vae == 'logs/pretrained/long_horizon_64/vae.pth'
    assert int(planner.data.dof_dim) == 29
    assert int(planner.data.nfeats) == 69
    assert planner.data.goal_type == 'joint_state'
    assert int(planner.denoiser.goal_dim) == 45
    assert planner.skeleton.asset.assetFileName == 'g1_29dof.xml'
    assert int(rotmat_v6.feature_version) == 6
    assert int(rotmat_v6.data.feature_version) == 6
    assert int(rotmat_v6.data.nfeats) == 44
    assert int(rotmat_v6.nfeats) == 44
    assert int(rotmat_v6.vae.nfeats) == 44
    assert int(rotmat_v6_data_override.feature_version) == 6
    assert int(rotmat_v6_data_override.data.nfeats) == 44
    set_feature_version(3)
