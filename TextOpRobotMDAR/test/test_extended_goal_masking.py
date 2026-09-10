import torch
import pytest

from TextOpRobotMDAR.robotmdar.model.mld_denoiser import (
    DenoiserMLP,
    DenoiserTransformer,
    _mask_goal,
)
from TextOpRobotMDAR.robotmdar.utils.goal import (
    GoalEncoding,
    SPLIT_END_EFFECTOR_GOAL_DIM,
    SPLIT_END_EFFECTOR_LEFT_FOOT_SLICE,
    SPLIT_END_EFFECTOR_LEFT_HAND_SLICE,
    SPLIT_END_EFFECTOR_RIGHT_FOOT_SLICE,
    SPLIT_END_EFFECTOR_RIGHT_HAND_SLICE,
    SPLIT_END_EFFECTOR_SLICE,
    SPLIT_END_EFFECTOR_TOKEN_ORDER,
    SPLIT_GOAL_DIM,
    SPLIT_HORIZONTAL_SLICE,
    SPLIT_HORIZONTAL_URGENCY_SLICE,
    SPLIT_JOINT_SLICE,
    SPLIT_ORIENTATION_SLICE,
    SPLIT_TIME_SLICE,
    SPLIT_VERTICAL_GRAVITY_SLICE,
    SPLIT_VERTICAL_HEIGHT_SLICE,
    SPLIT_VERTICAL_URGENCY_SLICE,
    SPLIT_VELOCITY_SLICE,
)


def _model():
    return DenoiserTransformer(
        h_dim=16,
        ff_size=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        history_shape=(2, 69),
        noise_shape=(1, 8),
        goal_dim=21,
        grid_size=2,
        cond_goal_root_mask_prob=0.0,
        cond_goal_yaw_mask_prob=0.0,
        cond_goal_time_mask_prob=0.0,
        cond_goal_body_mask_prob=0.0,
    )


def test_extended_goal_force_masks_are_independent_and_keep_velocity():
    model = _model().eval()
    goal = torch.arange(1, 22, dtype=torch.float32).unsqueeze(0)
    masked, root_keep = _mask_goal(model, goal, {
        "force_drop_goal_root": True,
        "force_drop_goal_yaw": True,
        "force_drop_goal_time": True,
        "force_drop_goal_body": True,
    })

    torch.testing.assert_close(masked[:, 0:5], torch.zeros((1, 5)))
    torch.testing.assert_close(masked[:, 5:8], goal[:, 5:8])
    torch.testing.assert_close(masked[:, 8:21], torch.zeros((1, 13)))
    assert not root_keep.item()


def test_extended_goal_root_force_mask_does_not_drop_other_components():
    model = _model().eval()
    goal = torch.ones((1, 21))
    masked, _ = _mask_goal(model, goal, {"force_drop_goal_root": True})

    torch.testing.assert_close(masked[:, 0:3], torch.zeros((1, 3)))
    torch.testing.assert_close(masked[:, 3:], torch.ones((1, 18)))


def test_joint_state_goal_force_masks_are_componentwise():
    model = DenoiserTransformer(
        h_dim=16,
        ff_size=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        history_shape=(2, 69),
        noise_shape=(1, 8),
        goal_dim=40,
        grid_size=2,
        cond_goal_root_mask_prob=0.0,
        cond_goal_orientation_mask_prob=0.0,
        cond_goal_joint_mask_prob=0.0,
        cond_goal_velocity_mask_prob=0.0,
    ).eval()
    goal = torch.arange(1, 41, dtype=torch.float32).unsqueeze(0)

    masked, root_keep = _mask_goal(model, goal, {
        "force_drop_goal_root": True,
        "force_drop_goal_orientation": True,
        "force_drop_goal_joint": True,
        "force_drop_goal_velocity": True,
    })

    torch.testing.assert_close(masked, torch.zeros_like(goal))
    assert not root_keep.item()


def test_joint_state_goal_root_mask_does_not_drop_other_components():
    model = DenoiserTransformer(
        h_dim=16,
        ff_size=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        history_shape=(2, 69),
        noise_shape=(1, 8),
        goal_dim=40,
        grid_size=2,
        cond_goal_root_mask_prob=0.0,
        cond_goal_orientation_mask_prob=0.0,
        cond_goal_joint_mask_prob=0.0,
        cond_goal_velocity_mask_prob=0.0,
    ).eval()
    goal = torch.ones((1, 40))

    masked, _ = _mask_goal(model, goal, {"force_drop_goal_root": True})

    torch.testing.assert_close(masked[:, 0:3], torch.zeros((1, 3)))
    torch.testing.assert_close(masked[:, 3:], torch.ones((1, 37)))


def test_legacy_goal_mask_config_maps_to_root_mask_probability():
    model = DenoiserTransformer(
        h_dim=16,
        ff_size=32,
        num_layers=1,
        num_heads=4,
        history_shape=(2, 69),
        noise_shape=(1, 8),
        goal_dim=15,
        grid_size=2,
        cond_goal_mask_prob=0.27,
    )

    assert model.cond_goal_root_mask_prob == 0.27
    assert not hasattr(model, "cond_goal_mask_prob")


def test_nested_cond_mask_prob_uses_goal_position_name():
    model = DenoiserTransformer(
        h_dim=16,
        ff_size=32,
        num_layers=1,
        num_heads=4,
        history_shape=(2, 69),
        noise_shape=(1, 8),
        goal_dim=40,
        grid_size=2,
        cond_mask_prob={
            "text": 0.2,
            "goal": {
                "position": 0.27,
                "orientation": 0.31,
                "joint": 0.41,
                "velocity": 0.51,
                "time": 0.61,
            },
            "scene": 0.71,
        },
    )

    assert model.cond_text_mask_prob == 0.2
    assert model.cond_goal_root_mask_prob == 0.27
    assert model.cond_goal_orientation_mask_prob == 0.31
    assert model.cond_goal_joint_mask_prob == 0.41
    assert model.cond_goal_velocity_mask_prob == 0.51
    assert model.cond_goal_time_mask_prob == 0.61
    assert model.cond_scene_mask_prob == 0.71


def test_nested_cond_mask_prob_rejects_position_root_disagreement():
    with pytest.raises(ValueError, match="goal.position.*goal_root"):
        DenoiserTransformer(
            h_dim=16,
            ff_size=32,
            num_layers=1,
            num_heads=4,
            history_shape=(2, 69),
            noise_shape=(1, 8),
            goal_dim=40,
            grid_size=2,
            cond_mask_prob={"goal": {"position": 0.27}},
            cond_goal_root_mask_prob=0.5,
        )


def test_locomotion_and_getup_mask_profiles_select_per_sample():
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
        text_condition_enabled=False,
        cond_mask_prob={
            "locomotion": {
                "goal": {
                    "position": {"hor": 0.0, "vert": 0.0},
                    "orientation": {"rot6d": 0.0, "gravity": 0.0},
                    "joint": 0.0,
                    "velocity": 0.0,
                    "time": 0.0,
                },
                "scene": 0.0,
            },
            "getup": {
                "goal": {
                    "position": {"hor": 1.0, "vert": 1.0},
                    "orientation": {"rot6d": 1.0, "gravity": 1.0},
                    "joint": 1.0,
                    "velocity": 1.0,
                    "time": 1.0,
                },
                "scene": 1.0,
            },
        },
    ).train()

    batch_size = 2
    y = {
        "goal": torch.ones(batch_size, SPLIT_GOAL_DIM),
        "voxel": torch.ones(batch_size, 8),
        "history_motion_normalized": torch.zeros(batch_size, 2, 6),
        "time_to_arrival_frame": torch.tensor([5, 10], dtype=torch.long),
        "is_recovery": torch.tensor([False, True]),
    }
    model(
        x_t=torch.zeros(batch_size, 1, 4),
        timesteps=torch.zeros(batch_size, dtype=torch.long),
        y=y,
    )

    assert y["goal_position_hor_condition_keep_mask"].tolist() == [
        True, False]
    assert y["goal_position_vert_condition_keep_mask"].tolist() == [
        True, False]
    assert y["goal_gravity_condition_keep_mask"].tolist() == [
        True, False]
    assert y["goal_orientation_condition_keep_mask"].tolist() == [
        True, False]
    assert y["goal_joint_condition_keep_mask"].tolist() == [True, False]
    assert y["goal_velocity_condition_keep_mask"].tolist() == [True, False]
    assert y["goal_time_condition_keep_mask"].tolist() == [True, False]


def test_mlp_nested_cond_mask_prob_uses_goal_position_name():
    model = DenoiserMLP(
        h_dim=16,
        n_blocks=1,
        dropout=0.0,
        history_shape=(2, 69),
        noise_shape=(1, 8),
        goal_dim=40,
        grid_size=2,
        cond_mask_prob={
            "goal": {"position": 0.37},
            "scene": 0.47,
        },
    )

    assert model.cond_goal_root_mask_prob == 0.37
    assert model.cond_scene_mask_prob == 0.47


def test_split_end_effector_goal_masks_four_tokens_independently():
    model = DenoiserTransformer(
        h_dim=16,
        ff_size=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        history_shape=(2, 69),
        noise_shape=(1, 8),
        goal_dim=SPLIT_END_EFFECTOR_GOAL_DIM,
        goal_encoding=GoalEncoding.SPLIT_END_EFFECTOR,
        grid_size=2,
        cond_mask_prob={
            "goal": {
                "position": 0.0,
                "orientation": 0.0,
                "joint": 0.0,
                "velocity": 0.0,
                "time": 0.0,
                "end_effector": {
                    "left_hand": 0.0,
                    "right_hand": 0.0,
                    "left_foot": 0.0,
                    "right_foot": 0.0,
                },
            },
            "scene": 0.0,
        },
    ).eval()
    goal = torch.arange(
        1, SPLIT_END_EFFECTOR_GOAL_DIM + 1,
        dtype=torch.float32).unsqueeze(0)
    y = {
        "force_drop_goal": {
            "end_effector": {
                "left_hand": True,
                "right_foot": True,
            },
        },
    }

    masked, root_keep = _mask_goal(model, goal, y)

    assert root_keep.tolist() == [True]
    torch.testing.assert_close(masked[:, :SPLIT_GOAL_DIM],
                               goal[:, :SPLIT_GOAL_DIM])
    torch.testing.assert_close(
        masked[:, SPLIT_END_EFFECTOR_LEFT_HAND_SLICE],
        torch.zeros((1, 3)),
    )
    torch.testing.assert_close(
        masked[:, SPLIT_END_EFFECTOR_RIGHT_HAND_SLICE],
        goal[:, SPLIT_END_EFFECTOR_RIGHT_HAND_SLICE],
    )
    torch.testing.assert_close(
        masked[:, SPLIT_END_EFFECTOR_LEFT_FOOT_SLICE],
        goal[:, SPLIT_END_EFFECTOR_LEFT_FOOT_SLICE],
    )
    torch.testing.assert_close(
        masked[:, SPLIT_END_EFFECTOR_RIGHT_FOOT_SLICE],
        torch.zeros((1, 3)),
    )
    assert model.cond_goal_end_effector_mask_probs == (0.0, 0.0, 0.0, 0.0)
    assert y["goal_end_effector_condition_keep_mask"].tolist() == [
        [False, True, True, False]]
    assert SPLIT_END_EFFECTOR_TOKEN_ORDER == (
        "left_hand", "right_hand", "left_foot", "right_foot")
    assert masked[:, SPLIT_END_EFFECTOR_SLICE].shape == (1, 12)


def test_controller_validity_is_a_real_split_end_effector_mask():
    model = DenoiserTransformer(
        h_dim=16,
        ff_size=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        history_shape=(2, 69),
        noise_shape=(1, 8),
        goal_dim=SPLIT_END_EFFECTOR_GOAL_DIM,
        goal_encoding=GoalEncoding.SPLIT_END_EFFECTOR,
        grid_size=2,
        cond_goal_root_mask_prob=0.0,
        cond_goal_orientation_mask_prob=0.0,
        cond_goal_joint_mask_prob=0.0,
        cond_goal_velocity_mask_prob=0.0,
        cond_goal_end_effector_mask_prob=0.0,
        cond_goal_time_mask_prob=0.0,
    ).eval()
    goal = torch.arange(
        1, SPLIT_END_EFFECTOR_GOAL_DIM + 1,
        dtype=torch.float32,
    ).unsqueeze(0)
    y = {
        "goal_valid": {
            "root": False,
            "yaw": True,
            "orientation": True,
            "joint": True,
            "velocity": True,
            "time": False,
            "end_effector": True,
            "end_effector_left_hand": False,
            "end_effector_right_hand": True,
            "end_effector_left_foot": True,
            "end_effector_right_foot": False,
        },
    }

    masked, root_keep = _mask_goal(model, goal, y)

    assert root_keep.tolist() == [False]
    torch.testing.assert_close(
        masked[:, SPLIT_HORIZONTAL_SLICE], torch.zeros((1, 9)))
    torch.testing.assert_close(
        masked[:, SPLIT_VERTICAL_HEIGHT_SLICE], torch.zeros((1, 2)))
    torch.testing.assert_close(
        masked[:, SPLIT_VERTICAL_GRAVITY_SLICE],
        goal[:, SPLIT_VERTICAL_GRAVITY_SLICE],
    )
    torch.testing.assert_close(
        masked[:, SPLIT_HORIZONTAL_URGENCY_SLICE], torch.zeros((1, 4)))
    torch.testing.assert_close(
        masked[:, SPLIT_VERTICAL_URGENCY_SLICE], torch.zeros((1, 1)))
    torch.testing.assert_close(
        masked[:, SPLIT_TIME_SLICE], torch.zeros((1, 1)))
    torch.testing.assert_close(
        masked[:, SPLIT_ORIENTATION_SLICE],
        goal[:, SPLIT_ORIENTATION_SLICE],
    )
    torch.testing.assert_close(
        masked[:, SPLIT_JOINT_SLICE], goal[:, SPLIT_JOINT_SLICE])
    torch.testing.assert_close(
        masked[:, SPLIT_VELOCITY_SLICE], goal[:, SPLIT_VELOCITY_SLICE])
    torch.testing.assert_close(
        masked[:, SPLIT_END_EFFECTOR_LEFT_HAND_SLICE],
        torch.zeros((1, 3)),
    )
    torch.testing.assert_close(
        masked[:, SPLIT_END_EFFECTOR_RIGHT_HAND_SLICE],
        goal[:, SPLIT_END_EFFECTOR_RIGHT_HAND_SLICE],
    )
    torch.testing.assert_close(
        masked[:, SPLIT_END_EFFECTOR_LEFT_FOOT_SLICE],
        goal[:, SPLIT_END_EFFECTOR_LEFT_FOOT_SLICE],
    )
    torch.testing.assert_close(
        masked[:, SPLIT_END_EFFECTOR_RIGHT_FOOT_SLICE],
        torch.zeros((1, 3)),
    )
    assert y["goal_end_effector_condition_keep_mask"].tolist() == [
        [False, True, True, False]]
    assert y["goal_time_condition_keep_mask"].tolist() == [False]


def test_controller_validity_is_ignored_during_training():
    model = DenoiserTransformer(
        h_dim=16,
        ff_size=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        history_shape=(2, 69),
        noise_shape=(1, 8),
        goal_dim=SPLIT_END_EFFECTOR_GOAL_DIM,
        goal_encoding=GoalEncoding.SPLIT_END_EFFECTOR,
        grid_size=2,
        cond_mask_prob={
            "goal": {
                "position": 0.0,
                "orientation": 0.0,
                "joint": 0.0,
                "velocity": 0.0,
                "time": 0.0,
                "end_effector": {
                    "left_hand": 0.0,
                    "right_hand": 0.0,
                    "left_foot": 0.0,
                    "right_foot": 0.0,
                },
            },
            "scene": 0.0,
        },
    ).train()
    cond = torch.ones(2, 3)
    masked_cond, cond_keep = model.mask_condition(
        cond,
        0.0,
        valid_mask=torch.tensor([False, False]),
        return_keep_mask=True,
    )
    torch.testing.assert_close(masked_cond, cond)
    assert cond_keep.tolist() == [True, True]

    goal = torch.arange(
        1, SPLIT_END_EFFECTOR_GOAL_DIM + 1,
        dtype=torch.float32,
    ).unsqueeze(0)
    y = {
        "goal_valid": {
            "root": False,
            "yaw": False,
            "orientation": False,
            "joint": False,
            "velocity": False,
            "time": False,
            "end_effector": False,
            "end_effector_left_hand": False,
            "end_effector_right_hand": False,
            "end_effector_left_foot": False,
            "end_effector_right_foot": False,
        },
    }

    masked_goal, root_keep = _mask_goal(model, goal, y)

    torch.testing.assert_close(masked_goal, goal)
    assert root_keep.tolist() == [True]
    assert y["goal_time_condition_keep_mask"].tolist() == [True]
