"""CLIP helpers for the legacy TextOp text-conditioning path."""

import clip


def load_and_freeze_clip(clip_version, device='cpu'):
    clip_model, _clip_preprocess = clip.load(
        clip_version, device=device, jit=False)
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
