import torch
from types import SimpleNamespace

from robotmdar.eval.generate_dar import denoiser_supports_text_guidance
from robotmdar.model.mld_denoiser import DenoiserTransformer
from robotmdar.train.manager import DARManager
from robotmdar.train.train_dar import _prepare_batch_text_embeddings


class _RaisingTextEmbed(torch.nn.Module):

    def forward(self, x):
        raise AssertionError("embed_text should be inactive")


def _small_transformer(**kwargs):
    return DenoiserTransformer(
        h_dim=16,
        ff_size=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        history_shape=(2, 69),
        noise_shape=(1, 8),
        goal_dim=5,
        grid_size=2,
        cond_goal_root_mask_prob=0.0,
        cond_scene_mask_prob=0.0,
        **kwargs,
    )


def test_transformer_text_conditioning_is_opt_in():
    model = _small_transformer()

    assert model.cond_text_mask_prob == 0.0
    assert model.text_condition_enabled is False
    assert not any(key.startswith("embed_text.")
                   for key in model.state_dict())
    assert denoiser_supports_text_guidance(model) is False


def test_disabled_text_conditioning_does_not_use_text_token():
    model = _small_transformer().eval()
    model.embed_text = _RaisingTextEmbed()
    x_t = torch.randn(2, 1, 8)
    timesteps = torch.zeros(2, dtype=torch.long)
    y = {
        "goal": torch.randn(2, 5),
        "voxel": torch.randn(2, 8),
        "history_motion_normalized": torch.randn(2, 2, 69),
        "text_embedding": torch.randn(2, 512),
    }

    out = model(x_t=x_t, timesteps=timesteps, y=y)

    assert out.shape == x_t.shape
    assert y["text_condition_keep_mask"].tolist() == [False, False]


def test_inference_load_disables_text_for_checkpoint_without_text_weights(
        tmp_path):
    model = _small_transformer(
        cond_text_mask_prob=0.1,
        text_condition_enabled=True,
    )
    ckpt_state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("embed_text.")
    }
    ckpt_path = tmp_path / "ckpt.pth"
    torch.save({"denoiser": ckpt_state, "step": 7}, ckpt_path)
    manager = object.__new__(DARManager)
    manager.denoiser = model
    manager.optimizer = None
    manager.device = "cpu"
    manager.use_ema = False
    manager.ema_models = {}

    manager.load_model(ckpt_path)

    assert model.text_condition_enabled is False
    assert model.cond_text_mask_prob == 0.0
    assert denoiser_supports_text_guidance(model) is False


def test_batch_text_embeddings_are_moved_once_for_all_primitives():
    cfg = SimpleNamespace(
        denoiser={'text_condition_enabled': True},
    )
    batch = [
        {'text_embedding': torch.ones(2, 512)},
        {'text_embedding': torch.full((2, 512), 2.0)},
        {'text_embedding': torch.full((2, 512), 3.0)},
    ]

    embeddings = _prepare_batch_text_embeddings(batch, cfg, 'cpu')

    assert embeddings.shape == (3, 2, 512)
    torch.testing.assert_close(embeddings[0], batch[0]['text_embedding'])
    torch.testing.assert_close(embeddings[2], batch[2]['text_embedding'])


def test_batch_text_embeddings_are_skipped_when_text_is_disabled():
    cfg = SimpleNamespace(
        denoiser={'text_condition_enabled': False},
    )
    batch = [{'text_embedding': torch.ones(2, 512)}]

    assert _prepare_batch_text_embeddings(batch, cfg, 'cpu') is None
