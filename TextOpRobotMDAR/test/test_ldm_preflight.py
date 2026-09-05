"""LDM-training preflight checks (GPT review checklist, 2026-09-02).

Each test maps to one preflight item that must hold before the first LDM run:

  P1  history semantics      — H+1 raw states -> H features; row t carries
                               s_{t+1} channels plus the t->t+1 transition
                               expressed in E_t (off-by-one contract)
  P2  goal SE(2) invariance  — the 55-D split goal is unchanged under a world
                               horizontal rotation + translation
  P3  arrival velocity index — v_goal = (p[t*] - p[t*-1]) * fps (backward
                               difference), not the leaving forward difference
  P4  orientation path       — g_g = R_g^T g_W and rho_6(R_t^T R_g) only; the
                               split path must never call legacy yaw builders
  P5  augmentation order     — V7.1 perturbation touches raw history states
                               before feature conversion; current frame w=0;
                               last-history transition keeps ~1.1% residue
  P6  frozen config contract — FeatureVersion 6, 44-D features, 55-D goals,
                               pinned loss weights / mask probs / aug schedule,
                               frozen v6 mean/std cache on disk
"""
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.testing import assert_close

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset.data_process.pack_motion_lib_to_textop import TARGET_DOF_NAMES
from TextOpRobotMDAR.robotmdar.dataloader import data as data_module
from TextOpRobotMDAR.robotmdar.dataloader.data import SkeletonPrimitiveDataset
from TextOpRobotMDAR.robotmdar.dtype.motion import (
    motion_dict_to_feature_v6,
    motion_feature_dim_for_dof,
)
from TextOpRobotMDAR.robotmdar.dtype.rotation import (
    matrix_to_quaternion,
    matrix_to_rot6d,
    quaternion_to_matrix,
    rot6d_to_matrix,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from TextOpRobotMDAR.robotmdar.utils import goal as goal_module
from TextOpRobotMDAR.robotmdar.utils.goal import (
    SPLIT_GOAL_DIM,
    SPLIT_HORIZONTAL_SLICE,
    SPLIT_JOINT_SLICE,
    SPLIT_ORIENTATION_SLICE,
    SPLIT_TIME_SLICE,
    SPLIT_VERTICAL_GRAVITY_SLICE,
    SPLIT_VERTICAL_SLICE,
    SPLIT_VELOCITY_SLICE,
    build_ego_joint_state_goal_v6,
    build_ego_split_goal,
    validate_goal_stats,
)
from TextOpRobotMDAR.robotmdar.train.train_dar import (
    _validate_scene_curriculum_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
motion_dtype = data_module.motion_dtype


@pytest.fixture(autouse=True)
def _restore_feature_version():
    old_version = motion_dtype.FeatureVersion
    yield
    motion_dtype.set_feature_version(old_version)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _mat_from_quat_xyzw(q: torch.Tensor) -> torch.Tensor:
    return quaternion_to_matrix(xyzw_to_wxyz(q))


def _world_gravity(pos: torch.Tensor) -> torch.Tensor:
    gravity = torch.zeros_like(pos)
    gravity[..., 2] = -1.0
    return gravity


def _tangent_project(v: torch.Tensor, gravity: torch.Tensor) -> torch.Tensor:
    """Mirror the v6 encoder's tangent-plane projection (motion.py:793-796)."""
    g = torch.nn.functional.normalize(gravity, dim=-1, eps=1e-8)
    return v - (v * g).sum(dim=-1, keepdim=True) * g


def _random_quat_xyzw(n: int, generator: torch.Generator) -> torch.Tensor:
    r6d = torch.rand(n, 6, dtype=torch.float64, generator=generator) * 2.0 - 1.0
    return wxyz_to_xyzw(matrix_to_quaternion(rot6d_to_matrix(r6d)))


def _goal_inputs(generator: torch.Generator):
    ref_pos = torch.rand(3, dtype=torch.float64, generator=generator)
    goal_pos = ref_pos + torch.rand(3, dtype=torch.float64, generator=generator)
    ref_rot = _random_quat_xyzw(1, generator)[0]
    goal_rot = _random_quat_xyzw(1, generator)[0]
    dof = torch.rand(29, dtype=torch.float64, generator=generator)
    velocity = torch.rand(3, dtype=torch.float64, generator=generator)
    return ref_pos, ref_rot, goal_pos, goal_rot, dof, velocity


def _split_goal(ref_pos, ref_rot, goal_pos, goal_rot, dof, velocity,
                time_to_arrival_seconds=1.37, fps=50.0):
    return build_ego_split_goal(
        world_goal_pos=goal_pos,
        world_goal_rot=goal_rot,
        world_goal_dof=dof,
        world_root_velocity=velocity,
        reference_pos=ref_pos,
        reference_rot=ref_rot,
        time_to_arrival_seconds=torch.tensor(
            time_to_arrival_seconds, dtype=ref_pos.dtype),
        fps=fps,
    )


# ---------------------------------------------------------------------------
# P1: history semantics
# ---------------------------------------------------------------------------

def test_v6_history_window_alignment_gpt_off_by_one():
    """H+F+1 raw states -> H+F features; row t has s_{t+1} channels and the
    t->t+1 transition in E_t; the first future feature is expressed in E_H."""
    H, F = 16, 1
    frames = H + F + 1  # 18 raw states
    t = torch.arange(frames, dtype=torch.float64)
    p = torch.stack((0.01 * t, 0.02 * torch.sin(t), 0.005 * t), dim=-1)
    # slowly-varying tilted root frames so gravity / rel_rot are non-trivial.
    # Repo 6-D convention (rotation.py:569-586): row-major flatten of the
    # first two matrix columns, i.e. [r11, r12, r21, r22, r31, r32] —
    # for Ry(theta) that is (cos, 0, 0, 1, -sin, 0).
    theta = 0.3 + 0.01 * t
    c, s = torch.cos(theta), torch.sin(theta)
    r6d = torch.stack((
        c, torch.zeros_like(c), torch.zeros_like(c),
        torch.ones_like(c), -s, torch.zeros_like(c),
    ), dim=-1)
    R = rot6d_to_matrix(r6d)
    rot = wxyz_to_xyzw(matrix_to_quaternion(R))
    dof = torch.sin(t[:, None] * (0.3 + torch.arange(29, dtype=torch.float64)))
    contact = torch.ones((frames, 2), dtype=torch.float64)

    features, _ = motion_dict_to_feature_v6({
        "root_trans_offset": p,
        "root_rot": rot,
        "dof": dof,
        "contact_mask": contact,
    })
    assert features.shape == (H + F, 44)
    assert motion_feature_dim_for_dof(29, feature_version=6) == 44

    g_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64)
    for i in range(H + F):
        row = features[i]
        # m_t channels belong to s_{t+1}
        assert_close(row[0:1], p[i + 1, 2:3])
        assert_close(row[1:4], R[i + 1].T @ g_w)
        assert_close(row[13:42], dof[i + 1])
        assert_close(row[42:44], contact[i + 1])
        # transition t -> t+1 expressed in E_t
        delta_local = R[i].T @ (p[i + 1] - p[i])
        assert_close(row[4:7], _tangent_project(delta_local, R[i].T @ g_w))
        assert_close(row[7:13], matrix_to_rot6d(R[i].T @ R[i + 1]))

    # GPT off-by-one contract, stated explicitly:
    # last history feature m_H <-> s_H, first future m_{H+1} = t->t+1 in E_t
    # with t = H (s_H is the reference/current frame of the window).
    assert_close(features[H - 1, 0:1], p[H, 2:3])
    assert_close(features[H - 1, 13:42], dof[H])
    first_future = features[H]
    assert_close(
        first_future[4:7],
        _tangent_project(R[H].T @ (p[H + 1] - p[H]), R[H].T @ g_w),
    )
    assert_close(first_future[7:13], matrix_to_rot6d(R[H].T @ R[H + 1]))
    assert_close(first_future[13:42], dof[H + 1])


# ---------------------------------------------------------------------------
# P2: goal SE(2) invariance
# ---------------------------------------------------------------------------

def test_split_goal_se2_invariance():
    """All 55 channels of the split goal are invariant under a world
    horizontal rotation + translation (g_W is a world-z vector)."""
    gen = torch.Generator().manual_seed(0)
    ref_pos, ref_rot, goal_pos, goal_rot, dof, velocity = _goal_inputs(gen)

    yaw = 0.7
    c, s = np.cos(yaw), np.sin(yaw)
    Q = torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    a = torch.tensor([1.2, -0.8, 0.0], dtype=torch.float64)

    def se2_quat(q):
        return wxyz_to_xyzw(matrix_to_quaternion(Q @ _mat_from_quat_xyzw(q)))

    G = _split_goal(ref_pos, ref_rot, goal_pos, goal_rot, dof, velocity)
    Gp = _split_goal(
        Q @ ref_pos + a, se2_quat(ref_rot),
        Q @ goal_pos + a, se2_quat(goal_rot),
        dof, Q @ velocity,
    )
    assert G.shape == (SPLIT_GOAL_DIM,)
    for s in (
            SPLIT_HORIZONTAL_SLICE,
            SPLIT_VERTICAL_SLICE,
            SPLIT_ORIENTATION_SLICE,
            SPLIT_JOINT_SLICE,
            SPLIT_VELOCITY_SLICE,
            SPLIT_TIME_SLICE,
    ):
        assert_close(G[s], Gp[s], rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# P3: goal velocity index
# ---------------------------------------------------------------------------

def test_world_goal_velocity_uses_arriving_backward_difference():
    """v_goal = (p[t*] - p[t*-1]) * fps, never (p[t*+1] - p[t*]) * fps."""
    ds = SkeletonPrimitiveDataset.__new__(SkeletonPrimitiveDataset)
    ds.fps = 50.0
    frames = 10
    # quadratic ramp: backward and forward differences disagree at every frame
    p = (torch.arange(frames, dtype=torch.float32) ** 2)[:, None] \
        * torch.tensor([0.1, -0.2, 0.05])
    raw = {"root_trans_offset": p}
    goal_frame = 7
    v = ds._world_goal_velocity(raw, goal_frame)
    assert_close(v, (p[7] - p[6]) * 50.0)
    assert not torch.allclose(v, (p[8] - p[7]) * 50.0)
    with pytest.raises(IndexError):
        ds._world_goal_velocity(raw, 0)
    with pytest.raises(IndexError):
        ds._world_goal_velocity(raw, frames)


# ---------------------------------------------------------------------------
# P4: orientation path
# ---------------------------------------------------------------------------

def test_v6_goal_orientation_is_rel_rot6d_no_yaw_helpers(monkeypatch):
    """Orientation channels are g_g = R_g^T g_W and rho_6(R_t^T R_g); the
    split path routes through the v6 matrix builder only."""
    gen = torch.Generator().manual_seed(1)
    ref_pos, ref_rot, goal_pos, goal_rot, dof, velocity = _goal_inputs(gen)
    ref_R = _mat_from_quat_xyzw(ref_rot)
    goal_R = _mat_from_quat_xyzw(goal_rot)
    g_w = _world_gravity(ref_pos)

    raw = build_ego_joint_state_goal_v6(
        world_goal_pos=goal_pos, world_goal_rot=goal_rot,
        world_goal_dof=dof, world_root_velocity=velocity,
        reference_pos=ref_pos, reference_rot=ref_rot,
        time_to_arrival_seconds=torch.tensor(1.37, dtype=ref_pos.dtype),
        fps=50.0,
    )
    assert_close(raw[4:7], goal_R.T @ g_w)
    assert_close(raw[7:13], matrix_to_rot6d(ref_R.T @ goal_R))

    split = _split_goal(ref_pos, ref_rot, goal_pos, goal_rot, dof, velocity)
    assert_close(split[SPLIT_VERTICAL_GRAVITY_SLICE], goal_R.T @ g_w)
    assert_close(
        split[SPLIT_ORIENTATION_SLICE],
        matrix_to_rot6d(ref_R.T @ goal_R),
    )

    # routing: the split encoding must never call the legacy yaw builders
    def _boom(*args, **kwargs):
        raise AssertionError("legacy yaw-based goal builder called")

    monkeypatch.setattr(goal_module, "build_ego_joint_state_goal", _boom)
    monkeypatch.setattr(goal_module, "build_ego_goal", _boom)
    again = _split_goal(ref_pos, ref_rot, goal_pos, goal_rot, dof, velocity)
    assert_close(again, split)

    # no Euler-angle / yaw helpers anywhere in the v6 and split sources
    for fn in (build_ego_joint_state_goal_v6, build_ego_split_goal):
        src = inspect.getsource(fn)
        for helper in ("quaternion_yaw", "extract_yaw", "get_euler", "euler"):
            assert helper not in src, f"{fn.__name__} references {helper!r}"


# ---------------------------------------------------------------------------
# P5: augmentation order
# ---------------------------------------------------------------------------

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


def _motion(frames):
    t = torch.arange(frames, dtype=torch.float64)
    p = torch.stack((0.01 * t, 0.02 * torch.sin(t), 0.005 * t), dim=-1)
    # constant tilt about the y axis, in the repo 6-D convention
    # [r11, r12, r21, r22, r31, r32] = (cos, 0, 0, 1, -sin, 0)
    theta = torch.tensor(0.3, dtype=torch.float64)
    r6d = torch.tensor(
        [torch.cos(theta).item(), 0.0, 0.0, 1.0, -torch.sin(theta).item(), 0.0],
        dtype=torch.float64,
    ).expand(frames, 6)
    R = rot6d_to_matrix(r6d)
    return {
        "root_trans_offset": p,
        "root_rot": wxyz_to_xyzw(matrix_to_quaternion(R)),
        "dof": torch.sin(t[:, None] * (0.3 + torch.arange(29, dtype=torch.float64))),
        "contact_mask": torch.ones((frames, 2), dtype=torch.float64),
    }


def _patch_rand(monkeypatch, scalar_values, vector_value):
    """First scalar gates the augmentation coin flip; later scalars feed the
    uniform samplers (dof vector, rx/ry/rz, dh)."""
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


def _quat_angle(qa, qb):
    dot = (qa * qb).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def test_augmentation_applied_to_raw_states_before_feature_conversion(monkeypatch):
    """P5: V7.1 perturbation touches raw history states only; the current
    frame (first future raw state) is exactly untouched; features are then
    recomputed from the perturbed states, with only the ~1.1% ramp residue
    on the last history transition."""
    H = 16
    frames = H + 2  # 18 raw states -> 17 features; current frame is idx H
    clean = _motion(frames)
    motion = {k: v.clone() for k, v in clean.items()}
    dataset = _dataset(history_len=H)
    _patch_rand(monkeypatch, [0.0, 0.25, 0.25, 0.25, 0.25], 0.25)

    assert dataset._augment_raw_motion(motion, generator=None) is True

    # w=0 boundary: the current frame and everything after are untouched
    for k in motion:
        assert torch.equal(motion[k][H:], clean[k][H:])
    # history frames are perturbed
    assert not torch.equal(motion["dof"][:H], clean["dof"][:H])
    assert not torch.equal(motion["root_rot"][:H], clean["root_rot"][:H])
    assert not torch.equal(
        motion["root_trans_offset"][:H, 2], clean["root_trans_offset"][:H, 2])

    # V7.1 ramp contract (§6): w = 1 - (3u^2 - 2u^3) with u = (H-1)/H gives
    # the last perturbed history frame ~1.1% of the frame-0 perturbation.
    rot0 = _quat_angle(motion["root_rot"][0], clean["root_rot"][0])
    rot_last = _quat_angle(motion["root_rot"][H - 1], clean["root_rot"][H - 1])
    assert rot_last < 0.05 * rot0
    dz0 = (motion["root_trans_offset"][0, 2] - clean["root_trans_offset"][0, 2]).abs()
    dz_last = (motion["root_trans_offset"][H - 1, 2] - clean["root_trans_offset"][H - 1, 2]).abs()
    assert dz_last < 0.05 * dz0

    feats_clean, _ = motion_dict_to_feature_v6(clean)
    feats_aug, _ = motion_dict_to_feature_v6(motion)

    # first future feature (16 -> 17, both raw states untouched) identical
    assert torch.equal(feats_aug[H], feats_clean[H])
    # last history feature must reflect the perturbed raw states...
    assert not torch.equal(feats_aug[H - 1], feats_clean[H - 1])
    # ...and equal the explicit feature algebra on the perturbed states
    g_w = _world_gravity(motion["root_trans_offset"][0])
    R_prev = _mat_from_quat_xyzw(motion["root_rot"][H - 1])
    delta_local = R_prev.T @ (
        motion["root_trans_offset"][H] - motion["root_trans_offset"][H - 1])
    assert_close(feats_aug[H - 1, 4:7],
                 _tangent_project(delta_local, R_prev.T @ g_w))
    assert_close(feats_aug[H - 1, 0:1], motion["root_trans_offset"][H, 2:3])
    assert_close(feats_aug[H - 1, 13:42], motion["dof"][H])


# ---------------------------------------------------------------------------
# P6: frozen config contract
# ---------------------------------------------------------------------------

def test_train_dar_config_freezes_v6_contract():
    """Composed train_dar.yaml pins FeatureVersion 6 / 44-D / 55-D and the
    first-round LDM baseline settings (loss weights, mask probs, aug schedule)."""
    from hydra import compose, initialize_config_dir

    config_dir = str(REPO_ROOT / "robotmdar" / "config")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="train_dar")

    assert cfg.data.feature_version == 6
    assert cfg.data.dof_dim == 29
    # mob.yaml must resolve nfeats version-aware (v6/29-dof == 44, not 69)
    assert cfg.data.nfeats == 44
    assert motion_feature_dim_for_dof(29, feature_version=6) == 44
    assert cfg.data.goal_type == "joint_state"
    assert cfg.data.goal_encoding == "split"
    assert cfg.data.goal_include_log_d_hor is True
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg_without_log = compose(
            config_name="train_dar",
            overrides=["data.goal_include_log_d_hor=false"],
        )
    assert cfg_without_log.data.goal_include_log_d_hor is False
    assert cfg_without_log.data.train.goal_include_log_d_hor is False
    assert cfg_without_log.data.val.goal_include_log_d_hor is False
    assert cfg.data.load_scene is False
    assert cfg.denoiser.goal_dim == SPLIT_GOAL_DIM
    assert cfg.data.history_len == 16 and cfg.data.future_len == 64

    # The DAR loss weights are split by locomotion/getup so recovery samples
    # can use a distinct objective while sharing the same batch.
    loss = cfg.train.manager.loss_weight
    assert set(loss.keys()) == {"locomotion", "getup"}
    assert loss.locomotion.foot_contact == 0.01
    assert loss.locomotion.support_consistency == 0.01
    assert "goal_root_position_hor" not in loss.locomotion
    assert loss.locomotion.goal.g == 0.0
    assert loss.locomotion.goal.root_position_hor == 0.25
    assert loss.locomotion.goal.root_position_vert == 0.05
    assert loss.locomotion.goal.root_orientation == 0.005
    assert loss.locomotion.goal.root_velocity == 0.05
    assert loss.locomotion.goal.joint_angle == 0.01

    assert loss.getup.foot_contact == 0.01
    assert loss.getup.support_consistency == 0.01
    assert "goal_root_position_hor" not in loss.getup
    assert loss.getup.goal.g == 0.05
    assert loss.getup.goal.root_position_hor == 0.005
    assert loss.getup.goal.root_position_vert == 0.1
    assert loss.getup.goal.root_orientation == 0.0
    assert loss.getup.goal.root_velocity == 0.01
    assert loss.getup.goal.joint_angle == 0.01
    assert (cfg.denoiser.cond_goal_root_mask_prob,
            cfg.denoiser.cond_goal_orientation_mask_prob,
            cfg.denoiser.cond_goal_joint_mask_prob,
            cfg.denoiser.cond_goal_velocity_mask_prob,
            cfg.denoiser.cond_goal_time_mask_prob) == (0.1, 0.1, 0.4, 0.0, 0.2)
    assert (cfg.data.augmentation_enabled,
            cfg.data.augmentation_start_step,
            cfg.data.augmentation_prob) == (True, 50000, 0.5)

    # 29-dof VAE checkpoint is intentionally null in the config: the launch
    # must freeze ckpt.vae to the trained ckpt_100000.pth explicitly.
    assert cfg.ckpt.vae is None


def test_train_dar_preflight_rejects_scene_gate_without_scene_payloads():
    cfg = OmegaConf.create({
        "data": {
            "load_scene": False,
            "scene_start_step": 50000,
        },
        "train": {
            "manager": {
                "stages": [25000, 12500, 12500],
            },
        },
    })

    with pytest.raises(ValueError, match="data.load_scene=false"):
        _validate_scene_curriculum_contract(cfg)


def test_train_dar_preflight_allows_no_scene_run_with_gate_after_max_steps():
    cfg = OmegaConf.create({
        "data": {
            "load_scene": False,
            "scene_start_step": 50001,
        },
        "train": {
            "manager": {
                "stages": [25000, 12500, 12500],
            },
        },
    })

    _validate_scene_curriculum_contract(cfg)


def test_frozen_vae_statistics_and_goal_stats_on_disk():
    """P6: the 29-dof dataset dir serves the frozen v6 44-D mean/std cache
    (loaded as-is, never recomputed) and, when present, valid split stats."""
    datadir = (REPO_ROOT / "dataset" / "BONES-SEED-29dof-FULL-50fps").resolve()
    if not datadir.exists():
        pytest.skip("29-dof dataset dir not present on this machine")

    meanstd_path = datadir / "meanstd_v6_dof29.pkl"
    assert meanstd_path.exists(), (
        "v6 statistics cache missing; the LDM dataloader would recompute a "
        "new set instead of reusing the VAE's frozen 44-D statistics")
    stats = torch.load(meanstd_path, map_location="cpu")
    assert isinstance(stats, dict)
    assert int(stats.get("feature_version", -1)) == 6
    assert stats.get("feature_alignment") == "arrival"
    assert tuple(stats["mean"].shape) == (44,)
    assert tuple(stats["std"].shape) == (44,)
    # the cache stores raw statistics (near-constant channels may sit below
    # std_floor); the floor is applied at load time by _set_meanstd
    assert float(np.asarray(stats["std"]).min()) > 0.0
    loader = SkeletonPrimitiveDataset.__new__(SkeletonPrimitiveDataset)
    loader.nfeats = 44
    loader.std_floor = 0.002
    loader._stats_device_cache = SimpleNamespace(clear=lambda: None)
    loader._set_meanstd((stats["mean"], stats["std"]), meanstd_path)
    assert float(loader.std.min()) >= 0.002  # floored on load

    goal_stats_path = datadir / "goal_stats.pkl"
    if not goal_stats_path.exists():
        pytest.skip(
            "goal_stats.pkl missing: the first LDM run computes it via the "
            "writer rank (or run scripts/refresh_goal_stats.yaml first)")
    goal_stats = torch.load(goal_stats_path, map_location="cpu")
    validate_goal_stats(
        goal_stats,
        goal_encoding="split",
        goal_offset_range=[-63, 0],
        goal_per_primitive=True,
        future_len=64,
        fps=50.0,
        goal_timestep_mode="relative",
        datadir=str(datadir),
        goal_include_log_d_hor=True,
    )
