import pytest

try:
    from robotmdar.model import clip as clip_helpers
except ModuleNotFoundError as exc:
    if exc.name != "clip":
        raise
    pytest.skip("OpenAI CLIP is not installed", allow_module_level=True)


class _FakeModel:

    def eval(self):
        return self

    def parameters(self):
        return iter(())


def test_explicit_checkpoint_path_bypasses_model_download(tmp_path, monkeypatch):
    checkpoint = tmp_path / "ViT-B-32.pt"
    checkpoint.write_bytes(b"checkpoint")
    calls = {}

    def fake_load(source, **kwargs):
        calls["source"] = source
        calls["kwargs"] = kwargs
        return _FakeModel(), None

    monkeypatch.setattr(clip_helpers.clip, "load", fake_load)
    monkeypatch.setattr(
        clip_helpers.clip.model, "convert_weights", lambda model: None)

    model = clip_helpers.load_and_freeze_clip(
        "ViT-B/32",
        device="cuda",
        clip_model_path=checkpoint,
    )

    assert isinstance(model, _FakeModel)
    assert calls["source"] == str(checkpoint)
    assert calls["kwargs"] == {"device": "cuda", "jit": False}


def test_missing_explicit_checkpoint_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="TEXTOP_CLIP_PATH"):
        clip_helpers.load_and_freeze_clip(
            "ViT-B/32",
            clip_model_path=tmp_path / "missing.pt",
        )


def test_download_failure_explains_offline_setup(monkeypatch):
    def fail_download(*args, **kwargs):
        from urllib.error import URLError
        raise URLError("TLS handshake failed")

    monkeypatch.setattr(clip_helpers.clip, "load", fail_download)
    monkeypatch.setattr(
        clip_helpers.clip, "available_models", lambda: ["ViT-B/32"])

    with pytest.raises(RuntimeError, match="data.clip_model_path"):
        clip_helpers.load_and_freeze_clip("ViT-B/32")
