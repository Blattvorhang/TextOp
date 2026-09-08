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
