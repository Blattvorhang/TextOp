import torch

from TextOpRobotMDAR.robotmdar.model.mld_denoiser import (
    DenoiserTransformer,
    _mask_goal,
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
