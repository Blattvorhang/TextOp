import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robotmdar.train import train_dar


def test_domain_randomization_matches_reactivebfm_schedule(monkeypatch):
    history = torch.zeros(2, 3, 2)
    feature_std = torch.tensor([2.0, 0.5])
    draws = []

    def fake_rand(size, device=None):
        draws.append(size)
        return torch.tensor([0.1, 0.9], device=device)

    def fake_randn_like(value):
        return torch.arange(value.numel(), dtype=value.dtype,
                            device=value.device).reshape_as(value)

    monkeypatch.setattr(train_dar.torch, "rand", fake_rand)
    monkeypatch.setattr(train_dar.torch, "randn_like", fake_randn_like)

    result = train_dar._domain_randomization(
        history, feature_std, step=50, total_steps=100)

    assert draws == [2]
    torch.testing.assert_close(
        result[0],
        torch.tensor([[0.0, 0.0075], [0.06, 0.0225], [0.12, 0.0375]]),
    )
    torch.testing.assert_close(result[1], history[1])
    torch.testing.assert_close(history, torch.zeros_like(history))


def test_domain_randomization_probability_and_zero_step(monkeypatch):
    history = torch.ones(1, 2, 2)
    draws = iter((0.0, 0.01, 0.01))
    noise_calls = []

    def fake_rand(size, device=None):
        return torch.full((size,), next(draws), device=device)

    def fake_randn_like(value):
        noise_calls.append(True)
        return torch.ones_like(value)

    monkeypatch.setattr(train_dar.torch, "rand", fake_rand)
    monkeypatch.setattr(train_dar.torch, "randn_like", fake_randn_like)

    at_start = train_dar._domain_randomization(
        history, torch.ones(2), step=0, total_steps=100)
    early = train_dar._domain_randomization(
        history, torch.ones(2), step=1, total_steps=100)
    at_end = train_dar._domain_randomization(
        history, torch.ones(2), step=100, total_steps=100)

    torch.testing.assert_close(at_start, history)
    assert early is history
    torch.testing.assert_close(at_end, history + 0.03)
    assert len(noise_calls) == 2
