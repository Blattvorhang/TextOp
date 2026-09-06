import torch

from robotmdar.eval.generate_dar import ClassifierFreeWrapper


class _DummyDenoiser(torch.nn.Module):
    cond_text_mask_prob = 0.1
    noise_shape = (1, 4)

    def forward(self, x_t, timesteps, y=None):
        del timesteps
        return x_t + (0.0 if y.get("uncond", False) else 2.0)


def test_classifier_free_wrapper_accepts_x_t_keyword():
    wrapper = ClassifierFreeWrapper(_DummyDenoiser())
    x_t = torch.ones(2, 1, 4)
    timesteps = torch.zeros(2, dtype=torch.long)

    out = wrapper(x_t=x_t, timesteps=timesteps, y={"scale": 3.0})

    torch.testing.assert_close(out, torch.full_like(x_t, 7.0))


def test_classifier_free_wrapper_accepts_diffusion_positional_call():
    wrapper = ClassifierFreeWrapper(_DummyDenoiser())
    x_t = torch.ones(2, 1, 4)
    timesteps = torch.zeros(2, dtype=torch.long)

    out = wrapper(x_t, timesteps, {"scale": 3.0})

    torch.testing.assert_close(out, torch.full_like(x_t, 7.0))
