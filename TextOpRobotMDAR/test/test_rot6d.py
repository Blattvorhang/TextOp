"""Tests for the 6D rotation <-> rotation matrix conversions.

Convention (this repo): ``matrix_to_rot6d`` is the row-major flattening of
the first-two-COLUMNS submatrix ``R[..., :, :2]``, i.e.
[r11, r12, r21, r22, r31, r32]. Equivalently, reshaping the 6D vector to
(..., 3, 2) yields the two rotation-matrix columns as its two columns.
``rot6d_to_matrix`` reads the two columns back the same way,
Gram-Schmidt-orthonormalizes them and completes the matrix with their cross
product, so the result lies in SO(3) for any non-degenerate input.

NOTE: ``pytorch3d.transforms.matrix_to_rotation_6d`` uses the first two
ROWS instead; the conventions differ and must never be mixed.
``dataset/data_analyze/analyze_rpy.py`` concatenates the two columns
directly ([r11, r21, r31, r12, r22, r32]) — a third ordering that is only
used for the continuity figure and never feeds these functions.

Reference: Zhou et al., "On the Continuity of Rotation Representations in
Neural Networks", CVPR 2019.
"""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as ScipyRotation

from TextOpRobotMDAR.robotmdar.dtype.rotation import matrix_to_rot6d, rot6d_to_matrix


def _random_rotmat(shape, seed=0):
    """Random SO(3) matrices as float32 torch tensor of shape (*shape, 3, 3)."""
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    R = ScipyRotation.from_quat(q).as_matrix()  # (n, 3, 3)
    return torch.as_tensor(R.reshape(*shape, 3, 3), dtype=torch.float32)


def _interleave_cols(a1, a2):
    """Build a 6D vector from the two columns, repo layout."""
    return torch.stack([a1, a2], dim=-1).reshape(*a1.shape[:-1], 6)


# ----------------------------------------------------------------------
# matrix -> 6D: exact encoding convention
# ----------------------------------------------------------------------

def test_matrix_to_rot6d_flattens_first_two_columns_row_major():
    R = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]])
    # R[..., :, :2] flattened row-major = [r11, r12, r21, r22, r31, r32]
    expected = torch.tensor([1.0, 2.0, 4.0, 5.0, 7.0, 8.0])
    torch.testing.assert_close(matrix_to_rot6d(R), expected)


def test_matrix_to_rot6d_matches_scipy_reference():
    """Cross-check the 6D encoding against scipy on a non-trivial rotation."""
    R = ScipyRotation.from_euler("xyz", [30, -45, 120], degrees=True).as_matrix()
    R_t = torch.as_tensor(R, dtype=torch.float32)
    expected = torch.as_tensor(R[:, :2].reshape(6), dtype=torch.float32)
    torch.testing.assert_close(matrix_to_rot6d(R_t), expected)


# ----------------------------------------------------------------------
# 6D -> matrix: Gram-Schmidt semantics
# ----------------------------------------------------------------------

def test_rot6d_to_matrix_gram_schmidt_semantics():
    a1 = torch.tensor([[2.0, 0.0, 0.0]])
    a2 = torch.tensor([[1.0, 3.0, 0.0]])  # not orthonormal to a1
    R = rot6d_to_matrix(_interleave_cols(a1, a2))  # (1, 3, 3)

    b1, b2, b3 = R[0, :, 0], R[0, :, 1], R[0, :, 2]
    # first column is the normalized first input column
    torch.testing.assert_close(b1, a1[0] / a1[0].norm())
    # second column is orthogonal to the first
    assert torch.dot(b1, b2).abs() < 1e-6
    # third column is the cross product
    torch.testing.assert_close(b3, torch.cross(b1, b2, dim=-1))
    # the input a2 lies in the span of b1, b2 (zero component along b3)
    torch.testing.assert_close(torch.dot(b3, a2[0]), torch.tensor(0.0),
                               atol=1e-5, rtol=1e-5)


def test_rot6d_to_matrix_is_so3_for_random_inputs():
    """Arbitrary (non-orthonormal) 6D inputs must map onto SO(3)."""
    rot6d = torch.randn(7, 3, 6)
    R = rot6d_to_matrix(rot6d)
    I = torch.eye(3).expand(7, 3, 3, 3)
    torch.testing.assert_close(R.transpose(-1, -2) @ R, I, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.det(R), torch.ones(7, 3), atol=1e-5, rtol=1e-5)


def test_rot6d_near_degenerate_stays_finite():
    """Nearly parallel input columns must not produce NaN/inf.

    Residual norm 1e-7 is above eps=1e-8, so the orthonormalization still
    recovers the identity exactly and det(R) = 1.
    """
    a1 = torch.tensor([[1.0, 0.0, 0.0]])
    a2 = torch.tensor([[1.0, 1e-7, 0.0]])
    R = rot6d_to_matrix(_interleave_cols(a1, a2))
    assert torch.isfinite(R).all()
    torch.testing.assert_close(torch.det(R), torch.ones(1), atol=1e-4, rtol=1e-4)


def test_rot6d_exactly_degenerate_stays_finite():
    """Exactly parallel columns are degenerate: b2 = 0, b3 = 0, det = 0.

    F.normalize divides by max(norm, eps), so a zero vector yields zero —
    finite, never NaN — and downstream code can detect the degenerate input
    via det.
    """
    a1 = torch.tensor([[1.0, 0.0, 0.0]])
    a2 = torch.tensor([[1.0, 0.0, 0.0]])
    R = rot6d_to_matrix(_interleave_cols(a1, a2))
    assert torch.isfinite(R).all()
    torch.testing.assert_close(R[0, :, 1], torch.zeros(3))
    torch.testing.assert_close(R[0, :, 2], torch.zeros(3))
    assert torch.det(R).item() == 0.0


# ----------------------------------------------------------------------
# Round-trip and shape generality
# ----------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(3,), (5, 7,), (2, 3, 5,)])
def test_round_trip_so3(shape):
    R = _random_rotmat(shape, seed=42)
    R2 = rot6d_to_matrix(matrix_to_rot6d(R))
    torch.testing.assert_close(R2, R, atol=1e-5, rtol=1e-5)


def test_round_trip_preserves_dtype_and_device():
    R = _random_rotmat((2, 3), seed=7)
    R2 = rot6d_to_matrix(matrix_to_rot6d(R))
    assert R2.dtype == torch.float32
    assert R2.device == R.device


# ----------------------------------------------------------------------
# Differentiability (used in training, e.g. chordal loss)
# ----------------------------------------------------------------------

def test_rot6d_to_matrix_gradients_flow():
    rot6d = torch.randn(4, 6, requires_grad=True)
    R = rot6d_to_matrix(rot6d)
    (R * R).sum().backward()
    assert torch.isfinite(rot6d.grad).all()


def test_matrix_to_rot6d_gradients_flow():
    R = _random_rotmat((2, 3), seed=1).requires_grad_(True)
    rot6d = matrix_to_rot6d(R)
    rot6d.sum().backward()
    assert torch.isfinite(R.grad).all()
