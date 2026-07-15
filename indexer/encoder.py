"""
Thin wrapper around OpenCLIP so the rest of the codebase never talks to
the model library directly. Swapping to FashionCLIP later = edit this file
only.

NOTE: loading the model downloads pretrained weights from the OpenCLIP /
HuggingFace hub on first run. That requires normal internet access (works
on your laptop, Colab, or Render's build step) — it will NOT work in a
network-restricted sandbox.
"""
import torch
import numpy as np
from utils.config import CLIP_MODEL_NAME, CLIP_PRETRAINED

_model = None
_preprocess = None
_tokenizer = None


def _lazy_load():
    global _model, _preprocess, _tokenizer
    if _model is not None:
        return
    import open_clip # imported lazily so the rest of the app can run
                       # (e.g. FAISS/API tests) without this dependency
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    model.eval()
    _model = model
    _preprocess = preprocess
    _tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)


@torch.no_grad()
def encode_image(pil_image):
    """PIL.Image -> L2-normalized (512,) float32 numpy vector."""
    _lazy_load()
    tensor = _preprocess(pil_image).unsqueeze(0)
    features = _model.encode_image(tensor)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze(0).numpy().astype("float32")


@torch.no_grad()
def encode_images_batch(pil_images, batch_size=32):
    """List[PIL.Image] -> (N, 512) float32 numpy array. Batched for speed."""
    _lazy_load()
    all_feats = []
    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i:i + batch_size]
        tensors = torch.stack([_preprocess(img) for img in batch])
        features = _model.encode_image(tensors)
        features = features / features.norm(dim=-1, keepdim=True)
        all_feats.append(features.numpy().astype("float32"))
    return np.concatenate(all_feats, axis=0)


@torch.no_grad()
def encode_text(text):
    """str or List[str] -> (N, 512) float32 L2-normalized numpy array."""
    _lazy_load()
    if isinstance(text, str):
        text = [text]
    tokens = _tokenizer(text)
    features = _model.encode_text(tokens)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.numpy().astype("float32")
