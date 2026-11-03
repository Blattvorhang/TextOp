from types import SimpleNamespace

import pytest
import torch

import robotmdar.eval.generate_dar as generation
from robotmdar.eval.generate_dar import (
    ClassifierFreeWrapper,
    GoalClassifierFreeWrapper,
)
from robotmdar.model.mld_denoiser import DenoiserTransformer
from robotmdar.utils.goal import (
    SPLIT_END_EFFECTOR_GOAL_DIM,
    SPLIT_END_EFFECTOR_SLICE,
    SPLIT_JOINT_SLICE,
)


class _DominantJointDenoiser(torch.nn.Module):
    """A joint-conditioned prediction that completely ignores EE."""

    goal_encoding = 'split_end_effector'
    noise_shape = (1, 4)
    cond_text_mask_prob = 0.1

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x_t, timesteps, y=None):
        self.calls.append((x_t, timesteps, dict(y)))
        joint = not y.get('force_drop_goal_joint', False)
        ee = not y.get('force_drop_goal_end_effector', False)
        offset = 20.0 if joint else (3.0 if ee else 0.0)
        if not y.get('uncond', False):
            offset += 2.0
        y['arrival_time_condition_keep_mask'] = torch.zeros(x_t.shape[0])
        return x_t + offset


@pytest.mark.parametrize('joint,ee,expected,calls', [
    (1.0, 1.0, 22.0, 1),
    (0.0, 1.0, 5.0, 1),
    (0.0, 0.0, 2.0, 1),
    (0.0, 2.0, 8.0, 2),
    (0.25, 2.0, 12.25, 3),
    (0.5, 0.0, 12.0, 2),
])
def test_ee_guidance_works_even_when_joint_prediction_ignores_ee(
        joint, ee, expected, calls):
    model = _DominantJointDenoiser()
    wrapper = GoalClassifierFreeWrapper(model, joint=joint, end_effector=ee)
    x = torch.zeros(2, 1, 4)
    t = torch.ones(2, dtype=torch.long)
    y = {'goal_valid': {'joint': True},
         'arrival_time_condition_keep_mask': torch.ones(2)}
    out = wrapper(x, t, y)

    torch.testing.assert_close(out, torch.full_like(x, expected))
    assert len(model.calls) == calls
    for branch_x, branch_t, branch in model.calls:
        assert branch_x is x and branch_t is t
        assert 'arrival_time_condition_keep_mask' not in branch
        assert branch['goal_valid'] == y['goal_valid']
    assert y['arrival_time_condition_keep_mask'].tolist() == [1, 1]
    assert 'force_drop_goal_joint' not in y


def test_goal_and_existing_text_guidance_compose():
    model = _DominantJointDenoiser()
    wrapper = GoalClassifierFreeWrapper(
        ClassifierFreeWrapper(model), joint=0, end_effector=2)
    x = torch.zeros(1, 1, 4)
    result = wrapper(x_t=x, timesteps=torch.zeros(1), y={'scale': 3.0})

    torch.testing.assert_close(result, torch.full_like(x, 12.0))
    assert len(model.calls) == 4


_EE_NAMES = ('left_hand', 'right_hand', 'left_foot', 'right_foot')


@pytest.mark.parametrize('validity', [
    {'joint': False},
    {'joint': torch.tensor([False, False])},
    {'end_effector': False},
    {'end_effector': {'all': False, 'left_hand': True}},
    {'end_effector': dict.fromkeys(_EE_NAMES, False)},
    dict.fromkeys((f'end_effector_{name}' for name in _EE_NAMES), False),
])
def test_invalid_joint_or_all_invalid_ee_bypass_cfg_with_one_call(validity):
    model = _DominantJointDenoiser()
    wrapper = GoalClassifierFreeWrapper(model, joint=0, end_effector=2)
    x = torch.zeros(2, 1, 4)
    result = wrapper(x, torch.zeros(2), {'goal_valid': validity})

    # Bypass both CFG scales; let the model consume the original validity.
    assert len(model.calls) == 1
    branch = model.calls[0][2]
    assert 'force_drop_goal_joint' not in branch
    assert 'force_drop_goal_end_effector' not in branch
    torch.testing.assert_close(result, torch.full_like(x, 22))


@pytest.mark.parametrize('force_drop', [
    {'force_drop_goal_joint': True},
    {'force_drop_goal_end_effector': True},
    {f'force_drop_goal_end_effector_{name}': True for name in _EE_NAMES},
])
def test_explicit_modality_drops_also_bypass_cfg(force_drop):
    model = _DominantJointDenoiser()
    wrapper = GoalClassifierFreeWrapper(model, joint=0.25, end_effector=2)
    wrapper(torch.zeros(1, 1, 4), torch.zeros(1), force_drop)
    assert len(model.calls) == 1


@pytest.mark.parametrize('name', _EE_NAMES)
def test_any_single_valid_ee_enables_cfg_and_preserves_other_leaf_masks(name):
    model = _DominantJointDenoiser()
    validity = {'joint': True, 'end_effector': {
        leaf: leaf == name for leaf in _EE_NAMES}}
    wrapper = GoalClassifierFreeWrapper(model, joint=0.25, end_effector=2)
    wrapper(torch.zeros(1, 1, 4), torch.zeros(1), {'goal_valid': validity})
    assert len(model.calls) == 3
    assert all(call[2]['goal_valid'] == validity for call in model.calls)


def test_mixed_batch_only_sends_eligible_samples_through_cfg_branches():
    model = _DominantJointDenoiser()
    x = torch.zeros(3, 1, 4)
    t = torch.tensor([1, 2, 3])
    validity = {'joint': [True, False, True],
                'end_effector': {name: torch.tensor([name == 'left_hand', True, False])
                                 for name in _EE_NAMES}}
    y = {'goal_valid': validity, 'history_motion_normalized': torch.randn(3, 2, 6)}
    wrapper = GoalClassifierFreeWrapper(model, joint=0.25, end_effector=2)
    result = wrapper(x, t, y)
    torch.testing.assert_close(result[:, 0, 0], torch.tensor([12.25, 22, 22]))
    assert len(model.calls) == 4
    assert model.calls[0][0].shape[0] == 2
    assert model.calls[0][1].tolist() == [2, 3]
    for branch_x, branch_t, branch in model.calls[1:]:
        assert branch_x.shape[0] == 1
        assert branch_t.tolist() == [1]
        assert branch['goal_valid']['joint'] == [True]
        assert branch['goal_valid']['end_effector']['left_hand'].tolist() == [True]
        assert branch['goal_valid']['end_effector']['right_hand'].tolist() == [False]
        torch.testing.assert_close(branch['history_motion_normalized'],
                                   y['history_motion_normalized'][:1])


@pytest.mark.parametrize('name', ['joint', 'end_effector'])
@pytest.mark.parametrize('value', [-1, float('inf'), float('nan')])
def test_goal_cfg_rejects_invalid_scales(name, value):
    with pytest.raises(ValueError, match=f'goal_guidance.{name}'):
        GoalClassifierFreeWrapper(_DominantJointDenoiser(), **{name: value})


def test_non_ee_checkpoint_keeps_identity_but_rejects_active_goal_cfg():
    model = _DominantJointDenoiser()
    model.goal_encoding = 'split'
    GoalClassifierFreeWrapper(model)
    with pytest.raises(ValueError, match='split_end_effector'):
        GoalClassifierFreeWrapper(model, joint=0)


def _real_model_and_conditions():
    torch.manual_seed(91)
    model = DenoiserTransformer(
        h_dim=8, ff_size=16, num_layers=1, num_heads=2, dropout=0,
        history_shape=(2, 6), noise_shape=(1, 4), grid_size=2,
        goal_encoding='split_end_effector', goal_dim=SPLIT_END_EFFECTOR_GOAL_DIM,
    ).eval()
    x = torch.randn(2, 1, 4)
    t = torch.ones(2, dtype=torch.long)
    y = {
        'goal': torch.randn(2, SPLIT_END_EFFECTOR_GOAL_DIM),
        'history_motion_normalized': torch.randn(2, 2, 6),
        'voxel': torch.randn(2, 8),
        'time_to_arrival_frame': torch.tensor([5, 8]),
    }
    return model, x, t, y


def test_identity_matches_real_denoiser_exactly():
    model, x, t, y = _real_model_and_conditions()
    with torch.no_grad():
        expected = model(x, t, dict(y))
        result = GoalClassifierFreeWrapper(model)(x, t, y)
    assert torch.equal(result, expected)


@pytest.mark.parametrize('joint,ee,hidden_slice', [
    (0, 2, SPLIT_JOINT_SLICE),
    (2, 0, SPLIT_END_EFFECTOR_SLICE),
])
def test_zero_scale_removes_modality_from_every_real_model_branch(
        joint, ee, hidden_slice):
    model, x, t, y = _real_model_and_conditions()
    wrapper = GoalClassifierFreeWrapper(model, joint=joint, end_effector=ee)
    changed = dict(y, goal=y['goal'].clone())
    changed['goal'][:, hidden_slice] += 100
    with torch.no_grad():
        torch.testing.assert_close(wrapper(x, t, y), wrapper(x, t, changed))


def test_real_model_preserves_per_sample_validity_and_individual_ee_drops():
    model, x, t, y = _real_model_and_conditions()
    y['goal_valid'] = {'joint': torch.tensor([False, True]),
                       'end_effector': {'all': torch.tensor([False, True])}}
    y['force_drop_goal_end_effector_left_hand'] = True
    changed = dict(y, goal=y['goal'].clone())
    changed['goal'][0, SPLIT_JOINT_SLICE] += 100
    changed['goal'][0, SPLIT_END_EFFECTOR_SLICE] += 100
    changed['goal'][1, SPLIT_END_EFFECTOR_SLICE.start:
                    SPLIT_END_EFFECTOR_SLICE.start + 3] += 100
    wrapper = GoalClassifierFreeWrapper(model, joint=0.25, end_effector=2)
    with torch.no_grad():
        torch.testing.assert_close(wrapper(x, t, y), wrapper(x, t, changed))


@pytest.mark.parametrize('validity', [
    {'joint': False},
    {'end_effector': dict.fromkeys(_EE_NAMES, False)},
])
def test_real_model_bypass_exactly_matches_original_prediction(validity):
    model, x, t, y = _real_model_and_conditions()
    y['goal_valid'] = validity
    wrapper = GoalClassifierFreeWrapper(model, joint=0, end_effector=2)
    with torch.no_grad():
        expected = model(x, t, dict(y))
        assert torch.equal(wrapper(x, t, y), expected)


@pytest.mark.parametrize('mode', ['single', 'ddpm', 'ddim'])
def test_generate_next_motion_applies_goal_cfg_without_text(monkeypatch, mode):
    monkeypatch.setattr(generation, 'motion_dict_to_abs_pose', lambda *a, **k: {})
    model = _DominantJointDenoiser()
    diffusion = SimpleNamespace(num_timesteps=5, _scale_timesteps=lambda t: t)

    def sample_fn(denoiser, shape, **kwargs):
        assert isinstance(denoiser, GoalClassifierFreeWrapper)
        return denoiser(kwargs['noise'], torch.ones(shape[0]),
                        kwargs['model_kwargs']['y'])

    diffusion.p_sample_loop = sample_fn
    diffusion.ddim_sample_loop = sample_fn
    dataset = SimpleNamespace(reconstruct_motion=lambda *a, **k: {})
    result, _, _ = generation.generate_next_motion(
        vae=None, denoiser=model, diffusion=diffusion, val_data=dataset,
        goal=torch.zeros(1, 66), voxel=torch.zeros(1, 8),
        history_motion=torch.zeros(1, 2, 4), abs_pose={}, future_len=1,
        initial_noise=torch.zeros(1, 1, 4), use_vae=False,
        use_full_sample=mode != 'single', use_ddim=mode == 'ddim',
        goal_guidance={'joint': 0, 'end_effector': 2},
    )
    torch.testing.assert_close(result, torch.full_like(result, 8.0))
