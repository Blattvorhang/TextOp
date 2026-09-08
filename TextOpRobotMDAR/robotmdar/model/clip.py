"""CLIP helpers for the legacy TextOp text-conditioning path."""

import os
from pathlib import Path
from ssl import SSLError
from urllib.error import URLError

import clip


def _resolve_clip_source(clip_version, clip_model_path=None):
    """Prefer an explicit local checkpoint over CLIP's auto-download path."""
    configured_path = clip_model_path or os.environ.get("TEXTOP_CLIP_PATH")
    if configured_path:
        model_path = Path(str(configured_path)).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"CLIP checkpoint does not exist: {model_path}. "
                "Set data.clip_model_path or TEXTOP_CLIP_PATH to a valid "
                "ViT-B-32.pt file."
            )
        return str(model_path), True
    return clip_version, False


def load_and_freeze_clip(
        clip_version,
        device='cpu',
        clip_model_path=None,
        download_root=None,
):
    """Load frozen CLIP, with an explicit offline checkpoint escape hatch."""
    clip_source, is_local_checkpoint = _resolve_clip_source(
        clip_version, clip_model_path)
    try:
        load_kwargs = {
            "device": device,
            "jit": False,
        }
        if download_root is not None:
            load_kwargs["download_root"] = str(
                Path(str(download_root)).expanduser())
        clip_model, _clip_preprocess = clip.load(clip_source, **load_kwargs)
    except (SSLError, URLError) as exc:
        if not is_local_checkpoint:
            raise RuntimeError(
                f"Unable to download CLIP model {clip_version!r}. The remote "
                "machine likely cannot reach openaipublic.azureedge.net. "
                "Copy the checkpoint to the machine and set "
                "data.clip_model_path=/path/to/ViT-B-32.pt (or export "
                "TEXTOP_CLIP_PATH) before starting training."
            ) from exc
        raise
    clip.model.convert_weights(clip_model)

    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    return clip_model


def encode_text(clip_model, raw_text, force_empty_zero=True):
    device = next(clip_model.parameters()).device
    texts = clip.tokenize(raw_text, truncate=True).to(device)
    text_embedding = clip_model.encode_text(texts).float()
    if force_empty_zero:
        empty_text = [text == '' for text in raw_text]
        text_embedding[empty_text, :] = 0
    return text_embedding
