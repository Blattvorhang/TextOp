import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyRotation

from TextOpRobotMDAR.robotmdar.dtype.motion import (
    CONTACT_MASK_DIM,
    motion_dict_to_feature_v6,
    motion_feature_dim_for_dof,
    motion_feature_to_dict_v6,
    get_zero_feature_v6,
)
from TextOpRobotMDAR.robotmdar.dtype.rotation import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


def _random_motion(batch=3, frames=8, dof_dim=29, seed=0):
    rng = np.random.default_rng(seed)
    trans = torch.as_tensor(
        rng.normal(size=(batch, frames, 3)),
        dtype=torch.float32,
    )
    trans[..., 2] = torch.as_tensor(
        rng.uniform(0.2, 1.2, size=(batch, frames)),
        dtype=torch.float32,
    )
    quat = rng.normal(size=(batch, frames, 4))
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    dof = torch.as_tensor(
        rng.normal(scale=0.25, size=(batch, frames, dof_dim)),
        dtype=torch.float32,
    )
    contact = torch.as_tensor(
        rng.integers(0, 2, size=(batch, frames, CONTACT_MASK_DIM)),
        dtype=torch.float32,
    )
    return {
        'root_trans_offset': trans,
        'root_rot': torch.as_tensor(quat, dtype=torch.float32),
        'dof': dof,
        'contact_mask': contact,
    }


def _rotmat_xyzw(quat):
    return quaternion_to_matrix(xyzw_to_wxyz(quat))


def test_v6_feature_width_and_zero_pose_layout():
    assert motion_feature_dim_for_dof(29, feature_version=6) == 44
    assert motion_feature_dim_for_dof(23, feature_version=6) == 38

    feature = get_zero_feature_v6(29)
    assert feature.shape == (1, 44)
    torch.testing.assert_close(feature[0, 1:4], torch.tensor([0.0, 0.0, -1.0]))
    torch.testing.assert_close(feature[0, 4:7], torch.zeros(3))
    torch.testing.assert_close(feature[0, 7:13], torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
    torch.testing.assert_close(feature[0, -2:], torch.ones(2))


def test_v6_round_trip_preserves_feature_indexed_states():
    motion = _random_motion(batch=4, frames=9, dof_dim=29, seed=1)
    feature, abs_pose = motion_dict_to_feature_v6(motion)
    reconstructed = motion_feature_to_dict_v6(feature, abs_pose)

    assert feature.shape == (4, 8, 44)
    torch.testing.assert_close(
        reconstructed['root_trans_offset'],
        motion['root_trans_offset'][:, :-1],
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        _rotmat_xyzw(reconstructed['root_rot']),
        _rotmat_xyzw(motion['root_rot'][:, :-1]),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(reconstructed['dof'], motion['dof'][:, :-1])
    torch.testing.assert_close(
        reconstructed['contact_mask'],
        motion['contact_mask'][:, :-1],
    )


def test_v6_round_trip_supports_unbatched_23dof_motion():
    motion = {
        key: value[0]
        for key, value in _random_motion(batch=1, frames=6, dof_dim=23, seed=2).items()
    }
    feature, abs_pose = motion_dict_to_feature_v6(motion)
    reconstructed = motion_feature_to_dict_v6(feature, abs_pose)

    assert feature.shape == (5, 38)
    torch.testing.assert_close(
        reconstructed['root_trans_offset'],
        motion['root_trans_offset'][:-1],
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        _rotmat_xyzw(reconstructed['root_rot']),
        _rotmat_xyzw(motion['root_rot'][:-1]),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(reconstructed['dof'], motion['dof'][:-1])


def test_v6_features_are_invariant_to_horizontal_world_se2_transform():
    motion = _random_motion(batch=2, frames=7, dof_dim=29, seed=3)
    feature, _ = motion_dict_to_feature_v6(motion)

    yaw = 1.17
    q_yaw = ScipyRotation.from_euler('z', yaw).as_quat()
    q_yaw = torch.as_tensor(q_yaw, dtype=torch.float32)
    yaw_matrix = _rotmat_xyzw(q_yaw).squeeze(0)

    transformed_trans = torch.matmul(
        yaw_matrix,
        motion['root_trans_offset'].unsqueeze(-1),
    ).squeeze(-1)
    transformed_trans[..., :2] += torch.tensor([2.5, -1.0])

    transformed_rotmat = torch.matmul(yaw_matrix, _rotmat_xyzw(motion['root_rot']))
    transformed_motion = dict(motion)
    transformed_motion['root_trans_offset'] = transformed_trans
    transformed_motion['root_rot'] = wxyz_to_xyzw(
        matrix_to_quaternion(transformed_rotmat)
    )

    transformed_feature, _ = motion_dict_to_feature_v6(transformed_motion)
    torch.testing.assert_close(transformed_feature, feature, atol=1e-5, rtol=1e-5)
