# =============================================================================
# DEPRECATED: Text-conditioned pipeline (CLIP embeddings) has been replaced by
# goal + scene conditioning per LDM_goal_scene_design.md.
#
# The entire text-conditioning pathway is removed from the active code path:
#   - CLIP model loading / freezing         → not called
#   - CLIP text encoding                    → not called
#   - Text embedding computation & caching  → not called (data.py)
#   - embed_text in Denoiser                → commented out (mld_denoiser.py)
#   - text_embedding in y dict              → replaced with zero tensor (train_dar.py)
#
# Functions below are preserved for reference / potential future text-based
# ablation experiments — DO NOT DELETE.
# =============================================================================

# import clip  # DEPRECATED: no longer needed for goal+scene conditioning

# def load_and_freeze_clip(clip_version, device='cpu'):  # DEPRECATED
#     clip_model, clip_preprocess = clip.load(clip_version, device=device,
#                                             jit=False)  # Must set jit=False for training
#     clip.model.convert_weights(
#         clip_model)  # Actually this line is unnecessary since clip by default already on float16
#
#     # Freeze CLIP weights
#     clip_model.eval()
#     for p in clip_model.parameters():
#         p.requires_grad = False
#
#     return clip_model
#
#
# def encode_text(clip_model, raw_text, force_empty_zero=True):  # DEPRECATED
#     device = next(clip_model.parameters()).device
#     # raw_text - list (batch_size length) of strings with input text prompts
#     texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length]
#     text_embedding = clip_model.encode_text(texts).float() # [bs, 512]
#     if force_empty_zero:  # force empty string to have zero embedding, same as being masked out in original MDM
#         empty_text = [text == '' for text in raw_text]
#         text_embedding[empty_text, :] = 0
#     return text_embedding