import math
from pathlib import Path

import pytest
import torch

from TextOpRobotMDAR.robotmdar.dataloader.data import SkeletonPrimitiveDataset
from TextOpRobotMDAR.robotmdar.dtype.rotation import (
    euler_angles_to_quaternion,
    matrix_to_rot6d,
    quat_mul,
    quaternion_to_matrix,
    xyzw_to_wxyz,
)
from TextOpRobotMDAR.robotmdar.model.mld_denoiser import (
    DenoiserMLP,
    DenoiserTransformer,
    _apply_arrival_channel_mask,
    _mask_goal,
)
from TextOpRobotMDAR.robotmdar.train.manager import DARManager
from TextOpRobotMDAR.robotmdar.utils.goal import (
    GoalEncoding,
    GoalType,
    SPLIT_GOAL_DIM,
    SPLIT_GOAL_SCHEMA,
    SPLIT_HORIZONTAL_SLICE,
    SPLIT_ORIENTATION_SLICE,
    SPLIT_VERTICAL_SLICE,
    build_ego_goal,
    build_ego_split_goal,
    scale_goal,
    validate_goal_config,
    validate_goal_stats,
)


def _identity_quaternion(batch: int = 1) -> torch.Tensor:
    return torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32).expand(batch, 4)


def _split_goal_stats_fixture(tmp_path: Path) -> dict:
    return {
        "s_p": torch.tensor(2.0),
        "s_l": torch.tensor(3.0),
        "s_v": torch.tensor(4.0),
        "s_d": torch.tensor(1.0),
        "s_o": torch.linspace(5.0, 13.0, 9),
        "q_mean": torch.linspace(-1.0, 1.0, 29),
        "q_std": torch.linspace(1.0, 2.0, 29),
        "meta": {
            "goal_offset_range": [-63, 0],
            "goal_per_primitive": True,
            "future_len": 64,
            "fps": 50.0,
            "goal_timestep_mode": "relative",
            "encodings": [GoalEncoding.SINGLE.value, GoalEncoding.SPLIT.value],
            "dataset_path": str(tmp_path),
            "goal_type": GoalType.JOINT_STATE.value,
            "goal_dim": SPLIT_GOAL_DIM,
            "goal_schema": SPLIT_GOAL_SCHEMA,
            "goal_include_log_d_hor": True,
            "dof_dim": 29,
        },
    }


def _split_goal_dataset(tmp_path: Path) -> SkeletonPrimitiveDataset:
    dataset = SkeletonPrimitiveDataset.__new__(SkeletonPrimitiveDataset)
    dataset.fps = 50.0
    dataset.mean = torch.zeros(69)
    dataset.std = torch.ones(69)
    dataset.goal_offset_range = (-63, 0)
    dataset.goal_per_primitive = True
    dataset.future_len = 64
    dataset.goal_timestep_mode = "relative"
    dataset.datadir = Path(tmp_path)
    dataset.goal_type = GoalType.JOINT_STATE
    dataset.goal_encoding = GoalEncoding.SPLIT
    dataset.goal_include_log_d_hor = True
    dataset.dof_dim = 29
    return dataset


def _joint_state_sample(pos, vel, euler, dof, time_to_arrival):
    reference_rot = _identity_quaternion()
    world_goal_rot = quat_mul(
        reference_rot,
        euler_angles_to_quaternion(torch.tensor([euler], dtype=torch.float32)),
        w_last=True,
    )
    return {
        "world_goal_pos": torch.tensor([pos], dtype=torch.float32),
        "world_goal_rot": world_goal_rot,
        "world_goal_dof": torch.tensor([dof], dtype=torch.float32),
        "world_goal_vel": torch.tensor([vel], dtype=torch.float32),
        "gt_ref_pos": torch.zeros((1, 3), dtype=torch.float32),
        "gt_ref_rot": reference_rot.clone(),
        "time_to_arrival": torch.tensor([time_to_arrival], dtype=torch.float32),
    }


class _ZeroGoalToken(torch.nn.Module):

    def __init__(self, h_dim: int):
        super().__init__()
        self.h_dim = h_dim

    def forward(self, x):
        return x.new_zeros((x.shape[0], self.h_dim))


class _FixedArrivalToken(torch.nn.Module):

    def __init__(self, value: torch.Tensor):
        super().__init__()
        self.register_buffer("value", value.reshape(1, -1).float())

    def forward(self, time_to_arrival_frame):
        return self.value.to(
            device=time_to_arrival_frame.device,
            dtype=time_to_arrival_frame.dtype,
        ).expand(time_to_arrival_frame.shape[0], -1)


class _CaptureEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, x):
        self.last_input = x.detach().clone()
        return x


def test_split_goal_layout_scaling_round_trip(tmp_path):
    assert GoalEncoding.SPLIT.token_count == 6

    reference_pos = torch.zeros((1, 3), dtype=torch.float32)
    reference_rot = _identity_quaternion()
    goal_euler = torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float32)
    world_goal_rot = quat_mul(
        reference_rot,
        euler_angles_to_quaternion(goal_euler),
        w_last=True,
    )
    world_goal_pos = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32)
    world_goal_dof = torch.linspace(-1.0, 1.0, 29).reshape(1, 29)
    world_root_velocity = torch.tensor([[4.0, -2.0, 1.0]], dtype=torch.float32)
    time_to_arrival = torch.tensor([4.0], dtype=torch.float32)

    raw = build_ego_split_goal(
        world_goal_pos=world_goal_pos,
        world_goal_rot=world_goal_rot,
        world_goal_dof=world_goal_dof,
        world_root_velocity=world_root_velocity,
        reference_pos=reference_pos,
        reference_rot=reference_rot,
        time_to_arrival_seconds=time_to_arrival,
        fps=50.0,
    )
    assert raw.shape == (1, SPLIT_GOAL_DIM)
    torch.testing.assert_close(
        raw[:, SPLIT_HORIZONTAL_SLICE],
        torch.tensor(
            [[1.0, -2.0, 0.0, math.sqrt(5.0),
              math.log1p(math.sqrt(5.0)), 0.25, -0.5, 0.0,
              math.sqrt(5.0) / 4.0]],
            dtype=torch.float32,
        ),
        atol=1e-6,
        rtol=0,
    )
    goal_R = quaternion_to_matrix(xyzw_to_wxyz(world_goal_rot))
    gravity_world = torch.tensor([[[0.0], [0.0], [-1.0]]])
    goal_g = (goal_R.transpose(-1, -2) @ gravity_world).squeeze(-1)
    torch.testing.assert_close(
        raw[:, 9:11],
        torch.tensor(
            [[0.5, 0.5]],
            dtype=torch.float32,
        ),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(raw[:, 11:14], goal_g, atol=1e-6, rtol=0)
    torch.testing.assert_close(
        raw[:, 14:15],
        torch.tensor([[0.125]], dtype=torch.float32),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        raw[:, SPLIT_ORIENTATION_SLICE],
        matrix_to_rot6d(goal_R),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(raw[:, 21:50], world_goal_dof)
    torch.testing.assert_close(
        raw[:, 50:54],
        torch.tensor([[4.0, -2.0, 0.0, 1.0]], dtype=torch.float32),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(raw[:, 54:55], time_to_arrival.reshape(1, 1))

    stats = _split_goal_stats_fixture(tmp_path)
    assert validate_goal_config(
        "joint_state",
        SPLIT_GOAL_DIM,
        GoalEncoding.SPLIT,
        dof_dim=29,
        goal_offset_range=(-63, 0),
        goal_timestep_mode="relative",
        goal_stats=stats,
        goal_include_log_d_hor=True,
    ) is GoalType.JOINT_STATE

    scaled = scale_goal(raw, stats)
    expected = raw.clone()
    expected[:, 0:4] = expected[:, 0:4] * stats["s_p"]
    expected[:, 4:5] = expected[:, 4:5] * stats["s_l"]
    expected[:, 5:9] = expected[:, 5:9] * stats["s_v"]
    expected[:, 9:11] = expected[:, 9:11] * stats["s_p"]
    expected[:, 11:14] = expected[:, 11:14] * stats["s_o"][:3]
    expected[:, 14:15] = expected[:, 14:15] * stats["s_v"]
    expected[:, 15:21] = expected[:, 15:21] * stats["s_o"][3:9]
    expected[:, 21:50] = (
        expected[:, 21:50] - stats["q_mean"]
    ) / stats["q_std"].clamp_min(1e-6)
    expected[:, 50:54] = expected[:, 50:54] * stats["s_v"]
    torch.testing.assert_close(scaled, expected, atol=1e-6, rtol=0)

    dispatched = build_ego_goal(
        world_goal_pos,
        torch.tensor([0.0], dtype=torch.float32),
        reference_pos,
        reference_rot,
        goal_type=GoalType.JOINT_STATE,
        goal_encoding=GoalEncoding.SPLIT,
        goal_stats=stats,
        fps=50.0,
        world_goal_rot=world_goal_rot,
        world_goal_dof=world_goal_dof,
        world_root_velocity=world_root_velocity,
        time_to_arrival_seconds=time_to_arrival,
    )
    torch.testing.assert_close(dispatched, scaled, atol=1e-6, rtol=0)


def test_split_goal_zero_time_zeroes_urgency():
    raw = build_ego_split_goal(
        world_goal_pos=torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32),
        world_goal_rot=_identity_quaternion(),
        world_goal_dof=torch.zeros((1, 29), dtype=torch.float32),
        world_root_velocity=torch.tensor([[4.0, -2.0, 1.0]], dtype=torch.float32),
        reference_pos=torch.zeros((1, 3), dtype=torch.float32),
        reference_rot=_identity_quaternion(),
        time_to_arrival_seconds=torch.tensor([0.0], dtype=torch.float32),
        fps=50.0,
    )

    torch.testing.assert_close(raw[:, 5:9], torch.zeros((1, 4)), atol=1e-6, rtol=0)
    torch.testing.assert_close(raw[:, 14:15], torch.zeros((1, 1)), atol=1e-6, rtol=0)


def test_split_goal_can_ablate_log_distance_channel():
    kwargs = dict(
        world_goal_pos=torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32),
        world_goal_rot=_identity_quaternion(),
        world_goal_dof=torch.zeros((1, 29), dtype=torch.float32),
        world_root_velocity=torch.tensor([[4.0, -2.0, 1.0]], dtype=torch.float32),
        reference_pos=torch.zeros((1, 3), dtype=torch.float32),
        reference_rot=_identity_quaternion(),
        time_to_arrival_seconds=torch.tensor([4.0], dtype=torch.float32),
        fps=50.0,
    )
    with_log = build_ego_split_goal(**kwargs)
    without_log = build_ego_split_goal(
        **kwargs, goal_include_log_d_hor=False)

    torch.testing.assert_close(
        with_log[:, 4:5],
        torch.tensor([[math.log1p(math.sqrt(5.0))]], dtype=torch.float32),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(without_log[:, 4:5], torch.zeros((1, 1)))
    torch.testing.assert_close(without_log[:, :4], with_log[:, :4])
    torch.testing.assert_close(without_log[:, 5:], with_log[:, 5:])


def test_build_ego_goal_rejects_log_distance_ablation_mismatch(tmp_path):
    stats = _split_goal_stats_fixture(tmp_path)
    stats["meta"] = dict(stats["meta"], goal_include_log_d_hor=False)

    kwargs = dict(
        world_goal_pos=torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32),
        world_goal_yaw=torch.zeros(1),
        reference_pos=torch.zeros((1, 3), dtype=torch.float32),
        reference_rot=_identity_quaternion(),
        goal_type=GoalType.JOINT_STATE,
        goal_encoding=GoalEncoding.SPLIT,
        goal_stats=stats,
        fps=50.0,
        world_goal_rot=_identity_quaternion(),
        world_goal_dof=torch.zeros((1, 29), dtype=torch.float32),
        world_root_velocity=torch.tensor([[4.0, -2.0, 1.0]], dtype=torch.float32),
        time_to_arrival_seconds=torch.tensor([4.0], dtype=torch.float32),
    )

    goal = build_ego_goal(**kwargs, goal_include_log_d_hor=False)
    torch.testing.assert_close(goal[:, 4:5], torch.zeros((1, 1)))
    with pytest.raises(ValueError, match="goal_include_log_d_hor"):
        build_ego_goal(**kwargs, goal_include_log_d_hor=True)


def test_split_goal_time_mask_zeroes_urgency_and_time():
    goal = torch.arange(1, SPLIT_GOAL_DIM + 1, dtype=torch.float32).unsqueeze(0)
    masked = _apply_arrival_channel_mask(
        goal,
        SPLIT_GOAL_DIM,
        torch.zeros(1, dtype=torch.bool),
    )

    torch.testing.assert_close(masked[:, 5:9], torch.zeros((1, 4)))
    torch.testing.assert_close(masked[:, 14:15], torch.zeros((1, 1)))
    torch.testing.assert_close(masked[:, 54:55], torch.zeros((1, 1)))
    torch.testing.assert_close(masked[:, :5], goal[:, :5])
    torch.testing.assert_close(masked[:, 9:14], goal[:, 9:14])
    torch.testing.assert_close(masked[:, 15:54], goal[:, 15:54])


def test_split_goal_mask_preserves_hor_vert_semantics():
    model = DenoiserTransformer(
        h_dim=8,
        ff_size=16,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        history_shape=(2, 6),
        noise_shape=(1, 4),
        goal_dim=SPLIT_GOAL_DIM,
        goal_encoding=GoalEncoding.SPLIT,
        grid_size=2,
        cond_text_mask_prob=0.0,
        text_condition_enabled=False,
        cond_goal_root_mask_prob=0.0,
        cond_goal_orientation_mask_prob=0.0,
        cond_goal_joint_mask_prob=0.0,
        cond_goal_velocity_mask_prob=0.0,
        cond_goal_time_mask_prob=0.0,
        cond_scene_mask_prob=0.0,
    ).eval()
    goal = torch.arange(1, SPLIT_GOAL_DIM + 1, dtype=torch.float32).unsqueeze(0)

    y_root = {"force_drop_goal_root": True}
    masked_root, root_keep = _mask_goal(model, goal, y_root)
    assert root_keep.tolist() == [False]
    torch.testing.assert_close(
        masked_root[:, SPLIT_HORIZONTAL_SLICE], torch.zeros((1, 9)))
    torch.testing.assert_close(masked_root[:, 9:11], torch.zeros((1, 2)))
    torch.testing.assert_close(masked_root[:, 14:15], torch.zeros((1, 1)))
    torch.testing.assert_close(masked_root[:, 11:14], goal[:, 11:14])
    torch.testing.assert_close(
        masked_root[:, SPLIT_ORIENTATION_SLICE],
        goal[:, SPLIT_ORIENTATION_SLICE],
    )
    assert y_root["goal_vertical_condition_keep_mask"].tolist() == [True]

    y_orientation = {"force_drop_goal_orientation": True}
    masked_orientation, root_keep = _mask_goal(model, goal, y_orientation)
    assert root_keep.tolist() == [True]
    torch.testing.assert_close(
        masked_orientation[:, SPLIT_HORIZONTAL_SLICE],
        goal[:, SPLIT_HORIZONTAL_SLICE],
    )
    torch.testing.assert_close(masked_orientation[:, 9:11], goal[:, 9:11])
    torch.testing.assert_close(masked_orientation[:, 14:15], goal[:, 14:15])
    torch.testing.assert_close(
        masked_orientation[:, 11:14], torch.zeros((1, 3)))
    torch.testing.assert_close(
        masked_orientation[:, SPLIT_ORIENTATION_SLICE], torch.zeros((1, 6)))
    assert y_orientation["goal_vertical_condition_keep_mask"].tolist() == [True]

    y_both = {
        "force_drop_goal_root": True,
        "force_drop_goal_orientation": True,
    }
    masked_both, root_keep = _mask_goal(model, goal, y_both)
    assert root_keep.tolist() == [False]
    torch.testing.assert_close(
        masked_both[:, SPLIT_VERTICAL_SLICE], torch.zeros((1, 6)))
    assert y_both["goal_vertical_condition_keep_mask"].tolist() == [False]

    y_time = {"force_drop_goal_time": True}
    _mask_goal(model, goal, y_time)
    assert y_time["goal_time_condition_keep_mask"].tolist() == [False]


def test_split_denoisers_use_arrival_pe_as_the_time_token_only():
    transformer = DenoiserTransformer(
        h_dim=8,
        ff_size=16,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        history_shape=(2, 6),
        noise_shape=(1, 4),
        goal_dim=SPLIT_GOAL_DIM,
        goal_encoding=GoalEncoding.SPLIT,
        grid_size=2,
        cond_text_mask_prob=0.0,
        text_condition_enabled=False,
        cond_goal_root_mask_prob=0.0,
        cond_goal_orientation_mask_prob=0.0,
        cond_goal_joint_mask_prob=0.0,
        cond_goal_velocity_mask_prob=0.0,
        cond_goal_time_mask_prob=0.0,
        cond_scene_mask_prob=0.0,
    ).eval()
    mlp = DenoiserMLP(
        h_dim=8,
        n_blocks=1,
        dropout=0.0,
        history_shape=(2, 12),
        noise_shape=(1, 4),
        goal_dim=SPLIT_GOAL_DIM,
        goal_encoding=GoalEncoding.SPLIT,
        grid_size=2,
        cond_goal_root_mask_prob=0.0,
        cond_goal_orientation_mask_prob=0.0,
        cond_goal_joint_mask_prob=0.0,
        cond_goal_velocity_mask_prob=0.0,
        cond_goal_time_mask_prob=0.0,
        cond_scene_mask_prob=0.0,
    )

    assert not hasattr(transformer, "embed_goal_time")
    assert not hasattr(mlp, "embed_goal_time")

    batch_size = 2
    x_t = torch.zeros(batch_size, 1, 4)
    timesteps = torch.zeros(batch_size, dtype=torch.long)
    y_mlp = {
        "goal": torch.ones(batch_size, SPLIT_GOAL_DIM),
        "voxel": torch.zeros(batch_size, 8),
        "history_motion_normalized": torch.zeros(batch_size, 2, 6),
        "time_to_arrival_frame": torch.tensor([5, 10], dtype=torch.long),
    }
    out = mlp.eval()(x_t=x_t, timesteps=timesteps, y=y_mlp)
    assert out.shape == x_t.shape
    assert y_mlp["goal_time_condition_keep_mask"].tolist() == [True, True]

    for name in ("embed_goal_hor", "embed_goal_vert", "embed_goal_rot",
                 "embed_goal_pose", "embed_goal_vel"):
        setattr(transformer, name, _ZeroGoalToken(8))
    transformer.arrival_embedder = _FixedArrivalToken(
        torch.arange(1, 9, dtype=torch.float32))
    transformer.sequence_pos_encoder = torch.nn.Identity()
    capture = _CaptureEncoder()
    transformer.seqTransEncoder = capture
    y = {
        "goal": torch.ones(batch_size, SPLIT_GOAL_DIM),
        "voxel": torch.zeros(batch_size, 8),
        "history_motion_normalized": torch.zeros(batch_size, 2, 6),
        "time_to_arrival_frame": torch.tensor([5, 10], dtype=torch.long),
    }

    transformer(x_t=x_t, timesteps=timesteps, y=y)

    xseq = capture.last_input
    expected_time = torch.arange(1, 9, dtype=torch.float32).expand(batch_size, -1)
    torch.testing.assert_close(xseq[1], torch.zeros_like(xseq[1]))
    torch.testing.assert_close(xseq[2], torch.zeros_like(xseq[2]))
    torch.testing.assert_close(xseq[3], torch.zeros_like(xseq[3]))
    torch.testing.assert_close(xseq[4], torch.zeros_like(xseq[4]))
    torch.testing.assert_close(xseq[5], torch.zeros_like(xseq[5]))
    torch.testing.assert_close(xseq[6], expected_time)
    assert y["goal_time_condition_keep_mask"].tolist() == [True, True]

    y_drop = {
        "goal": torch.ones(batch_size, SPLIT_GOAL_DIM),
        "voxel": torch.zeros(batch_size, 8),
        "history_motion_normalized": torch.zeros(batch_size, 2, 6),
        "time_to_arrival_frame": torch.tensor([5, 10], dtype=torch.long),
        "force_drop_goal_time": True,
    }
    transformer(x_t=x_t, timesteps=timesteps, y=y_drop)
    xseq = capture.last_input
    torch.testing.assert_close(xseq[1], torch.zeros_like(xseq[1]))
    torch.testing.assert_close(xseq[6], torch.zeros_like(xseq[6]))
    assert y_drop["goal_time_condition_keep_mask"].tolist() == [False, False]


def test_split_transformer_keeps_slot_pe_for_masked_goal_tokens():
    model = DenoiserTransformer(
        h_dim=8,
        ff_size=16,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        history_shape=(2, 6),
        noise_shape=(1, 4),
        goal_dim=SPLIT_GOAL_DIM,
        goal_encoding=GoalEncoding.SPLIT,
        grid_size=2,
        cond_text_mask_prob=0.0,
        text_condition_enabled=False,
        cond_goal_root_mask_prob=0.0,
        cond_goal_orientation_mask_prob=0.0,
        cond_goal_joint_mask_prob=0.0,
        cond_goal_velocity_mask_prob=0.0,
        cond_goal_time_mask_prob=0.0,
        cond_scene_mask_prob=0.0,
    ).eval()
    for name in ("embed_goal_hor", "embed_goal_vert", "embed_goal_rot",
                 "embed_goal_pose", "embed_goal_vel"):
        setattr(model, name, _ZeroGoalToken(8))
    model.arrival_embedder = _FixedArrivalToken(
        torch.arange(1, 9, dtype=torch.float32))
    capture = _CaptureEncoder()
    model.seqTransEncoder = capture

    batch_size = 2
    y = {
        "goal": torch.ones(batch_size, SPLIT_GOAL_DIM),
        "voxel": torch.zeros(batch_size, 8),
        "history_motion_normalized": torch.zeros(batch_size, 2, 6),
        "time_to_arrival_frame": torch.tensor([5, 10], dtype=torch.long),
        "force_drop_goal_root": True,
        "force_drop_goal_orientation": True,
        "force_drop_goal_joint": True,
        "force_drop_goal_velocity": True,
        "force_drop_goal_time": True,
    }

    model(
        x_t=torch.zeros(batch_size, 1, 4),
        timesteps=torch.zeros(batch_size, dtype=torch.long),
        y=y,
    )

    xseq = capture.last_input
    expected_goal_slots = model.sequence_pos_encoder.pe[1:7].expand(
        -1, batch_size, -1)
    torch.testing.assert_close(xseq[1:7], expected_goal_slots)
    assert y["goal_condition_keep_mask"].tolist() == [False, False]
    assert y["goal_orientation_condition_keep_mask"].tolist() == [False, False]
    assert y["goal_vertical_condition_keep_mask"].tolist() == [False, False]
    assert y["goal_joint_condition_keep_mask"].tolist() == [False, False]
    assert y["goal_velocity_condition_keep_mask"].tolist() == [False, False]
    assert y["goal_time_condition_keep_mask"].tolist() == [False, False]


def test_split_checkpoint_adapter_drops_deprecated_time_scalar_mlp(tmp_path):
    model = DenoiserTransformer(
        h_dim=8,
        ff_size=16,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        history_shape=(2, 6),
        noise_shape=(1, 4),
        goal_dim=SPLIT_GOAL_DIM,
        goal_encoding=GoalEncoding.SPLIT,
        grid_size=2,
        cond_text_mask_prob=0.0,
        text_condition_enabled=False,
        cond_goal_root_mask_prob=0.0,
        cond_goal_orientation_mask_prob=0.0,
        cond_goal_joint_mask_prob=0.0,
        cond_goal_velocity_mask_prob=0.0,
        cond_goal_time_mask_prob=0.0,
        cond_scene_mask_prob=0.0,
    )
    checkpoint_state = dict(model.state_dict())
    checkpoint_state["embed_goal_time.layers.0.weight"] = torch.ones(8, 1)

    adapted_state, adapted = DARManager._fill_missing_optional_condition_state(
        model, checkpoint_state, tmp_path / "ckpt.pth")

    assert adapted is True
    assert "embed_goal_time.layers.0.weight" not in adapted_state
    model.load_state_dict(adapted_state)


def test_split_goal_stats_clip_outliers_before_std(tmp_path):
    dataset = _split_goal_dataset(tmp_path)
    batch_data = [
        _joint_state_sample(
            pos=[1.0, 1.0, 0.5],
            vel=[1.0, 2.0, 3.0],
            euler=[0.1, -0.2, 0.3],
            dof=torch.linspace(-1.0, 1.0, 29).tolist(),
            time_to_arrival=1.0,
        ),
        _joint_state_sample(
            pos=[12.0, -9.0, 0.5],
            vel=[30.0, -30.0, 6.0],
            euler=[-0.2, 0.05, -0.1],
            dof=torch.linspace(1.0, -1.0, 29).tolist(),
            time_to_arrival=1.0,
        ),
    ]

    stats = SkeletonPrimitiveDataset._goal_stats_from_batch(dataset, batch_data)

    goals = [
        build_ego_split_goal(
            world_goal_pos=item["world_goal_pos"],
            world_goal_rot=item["world_goal_rot"],
            world_goal_dof=item["world_goal_dof"],
            world_root_velocity=item["world_goal_vel"],
            reference_pos=item["gt_ref_pos"],
            reference_rot=item["gt_ref_rot"],
            time_to_arrival_seconds=item["time_to_arrival"],
            fps=dataset.fps,
        )
        for item in batch_data
    ]
    hor = torch.cat([goal[:, SPLIT_HORIZONTAL_SLICE] for goal in goals], dim=0)
    vert = torch.cat([goal[:, SPLIT_VERTICAL_SLICE] for goal in goals], dim=0)
    pos = torch.cat((hor[:, 0:4], vert[:, 0:2]), dim=-1).clamp(-3.0, 3.0)
    urgency = torch.cat((hor[:, 5:9], vert[:, 5:6]), dim=-1).clamp(
        -5.0, 5.0)
    velocity = torch.cat([goal[:, 50:54] for goal in goals], dim=0).clamp(-5.0, 5.0)
    d_hor = hor[:, 3:4]
    expected_s_p = 1.0 / torch.clamp(pos.reshape(-1).std(unbiased=False), min=1e-6)
    expected_s_v = 1.0 / torch.clamp(
        torch.cat((urgency.reshape(-1), velocity.reshape(-1)), dim=0).std(unbiased=False),
        min=1e-6,
    )
    expected_s_d = torch.quantile(d_hor.reshape(-1).float().clamp_min(0.0), 0.5)
    log_terms = torch.log1p(
        d_hor / expected_s_d.to(device=d_hor.device, dtype=d_hor.dtype)
    )
    expected_s_l = 1.0 / torch.clamp(
        log_terms.reshape(-1).std(unbiased=False), min=1e-6)

    torch.testing.assert_close(stats["s_p"], expected_s_p)
    torch.testing.assert_close(stats["s_l"], expected_s_l)
    torch.testing.assert_close(stats["s_v"], expected_s_v)
    torch.testing.assert_close(stats["s_d"], expected_s_d)
    assert stats["meta"]["goal_dim"] == SPLIT_GOAL_DIM
    assert stats["meta"]["goal_schema"] == SPLIT_GOAL_SCHEMA
    assert stats["meta"]["encodings"] == [GoalEncoding.SINGLE.value, GoalEncoding.SPLIT.value]


def test_split_goal_stats_handles_batched_time_to_arrival(tmp_path):
    dataset = _split_goal_dataset(tmp_path)
    batch_data = [
        {
            "world_goal_pos": torch.tensor(
                [[3.0, 4.0, 0.5], [6.0, 8.0, 0.5]], dtype=torch.float32),
            "world_goal_rot": _identity_quaternion(batch=2),
            "world_goal_dof": torch.stack([
                torch.linspace(-1.0, 1.0, 29),
                torch.linspace(1.0, -1.0, 29),
            ], dim=0),
            "world_goal_vel": torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32),
            "gt_ref_pos": torch.zeros((2, 3), dtype=torch.float32),
            "gt_ref_rot": _identity_quaternion(batch=2),
            "time_to_arrival": torch.tensor([[1.0], [4.0]], dtype=torch.float32),
        }
    ]

    stats = SkeletonPrimitiveDataset._goal_stats_from_batch(dataset, batch_data)

    assert stats["goal_clamp"]["n_samples"] == 2
    assert stats["goal_clamp"]["dist_quantiles"][0] == pytest.approx(7.5)
    assert stats["goal_clamp"]["speed_quantiles"][0] == pytest.approx(3.75)


def test_split_goal_stats_records_log_distance_ablation(tmp_path):
    dataset = _split_goal_dataset(tmp_path)
    dataset.goal_include_log_d_hor = False
    batch_data = [
        _joint_state_sample(
            pos=[1.0, 1.0, 0.5],
            vel=[1.0, 2.0, 3.0],
            euler=[0.1, -0.2, 0.3],
            dof=torch.linspace(-1.0, 1.0, 29).tolist(),
            time_to_arrival=1.0,
        ),
        _joint_state_sample(
            pos=[2.0, -1.0, 0.5],
            vel=[3.0, -2.0, 1.0],
            euler=[-0.2, 0.05, -0.1],
            dof=torch.linspace(1.0, -1.0, 29).tolist(),
            time_to_arrival=1.0,
        ),
    ]

    stats = SkeletonPrimitiveDataset._goal_stats_from_batch(dataset, batch_data)

    assert stats["meta"]["goal_include_log_d_hor"] is False
    torch.testing.assert_close(stats["s_l"], torch.tensor(1.0))


def test_split_goal_stats_meta_validation_rejects_mismatch(tmp_path):
    stats = _split_goal_stats_fixture(tmp_path)

    bad_schema = dict(stats)
    bad_schema["meta"] = dict(stats["meta"], goal_schema="rotmat_v7")
    with pytest.raises(ValueError, match="goal_schema"):
        validate_goal_stats(
            bad_schema,
            goal_encoding=GoalEncoding.SPLIT,
            goal_offset_range=(-63, 0),
            goal_per_primitive=True,
            future_len=64,
            fps=50.0,
            goal_timestep_mode="relative",
            datadir=str(tmp_path),
        )
    with pytest.raises(ValueError, match="schema"):
        scale_goal(torch.zeros((1, SPLIT_GOAL_DIM)), bad_schema)

    bad_log_ablation = dict(stats)
    bad_log_ablation["meta"] = dict(
        stats["meta"], goal_include_log_d_hor=False)
    with pytest.raises(ValueError, match="goal_include_log_d_hor"):
        validate_goal_stats(
            bad_log_ablation,
            goal_encoding=GoalEncoding.SPLIT,
            goal_offset_range=(-63, 0),
            goal_per_primitive=True,
            future_len=64,
            fps=50.0,
            goal_timestep_mode="relative",
            datadir=str(tmp_path),
            goal_include_log_d_hor=True,
        )

    with pytest.raises(ValueError, match="goal_offset_range"):
        validate_goal_stats(
            stats,
            goal_encoding=GoalEncoding.SPLIT,
            goal_offset_range=(-62, 0),
            goal_per_primitive=True,
            future_len=64,
            fps=50.0,
            goal_timestep_mode="relative",
            datadir=str(tmp_path),
        )
