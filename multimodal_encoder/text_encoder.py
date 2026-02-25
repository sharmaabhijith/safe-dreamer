"""Frozen pretrained text encoder with caching.

Uses CLIP text encoder to convert task description strings into token embeddings.
The CLIP model is frozen; only the trainable projection layer is updated.
We cache the frozen CLIP outputs and re-project each forward pass so gradients
flow through the projection layer.
"""

import torch
import torch.nn as nn


def _load_clip(model_name):
    """Load CLIP text model and tokenizer. Separated for sharing across instances."""
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    text_model = CLIPTextModel.from_pretrained(model_name)
    for param in text_model.parameters():
        param.requires_grad = False
    text_model.eval()
    return tokenizer, text_model


# Module-level cache so CLIP is loaded only once across all TextEncoder instances
_CLIP_CACHE = {}


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.max_text_length = config.max_text_length
        self._model_name = config.text_encoder_name

        # Load or reuse frozen CLIP text model (shared across instances)
        if self._model_name not in _CLIP_CACHE:
            _CLIP_CACHE[self._model_name] = _load_clip(self._model_name)
        self._tokenizer, self._text_model = _CLIP_CACHE[self._model_name]

        # Trainable projection: d_text -> d_model
        d_text = self._text_model.config.hidden_size  # 512 for CLIP base
        self.projection = nn.Linear(d_text, self.d_model)

        # Cache frozen CLIP outputs: text string -> (1, N_text, d_text) tensor
        self._clip_cache = {}

    @torch.no_grad()
    def _encode_clip(self, text, device):
        """Run frozen CLIP on a single text string. Returns (1, N_text, d_text)."""
        if text in self._clip_cache:
            cached = self._clip_cache[text]
            if cached.device == device:
                return cached
            cached = cached.to(device)
            self._clip_cache[text] = cached
            return cached

        tokens = self._tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        ).to(device)

        # (1, N_text, d_text)
        clip_out = self._text_model(**tokens).last_hidden_state.detach().clone()
        self._clip_cache[text] = clip_out
        return clip_out

    def forward(self, text_list, device):
        """Encode text list. Returns (B, N_text, d_model).

        Caches frozen CLIP outputs; the trainable projection is always applied fresh.
        """
        clip_features = []
        for text in text_list:
            clip_features.append(self._encode_clip(text, device))

        # Pad to same sequence length and stack
        max_len = max(f.shape[1] for f in clip_features)
        padded = []
        for f in clip_features:
            if f.shape[1] < max_len:
                pad = torch.zeros(1, max_len - f.shape[1], f.shape[2], device=device, dtype=f.dtype)
                f = torch.cat([f, pad], dim=1)
            padded.append(f)

        # (B, N_text, d_text)
        clip_batch = torch.cat(padded, dim=0)
        # Cast and project: (B, N_text, d_model)
        clip_batch = clip_batch.to(dtype=self.projection.weight.dtype)
        return self.projection(clip_batch)

    def clear_cache(self):
        self._clip_cache.clear()
