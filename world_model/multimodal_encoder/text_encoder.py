"""Text modules for the multimodal encoder.

TextContextEncoder: frozen CLIP text encoder + trainable attention pooling + projection.
TextGate: learned mixture of visual embedding and projected text context.
"""

import torch
import torch.nn as nn
import torch.nn.init as init


def _load_clip(model_name):
    """Load CLIP text model and tokenizer. Separated for sharing across instances."""
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    text_model = CLIPTextModel.from_pretrained(model_name)
    for param in text_model.parameters():
        param.requires_grad = False
    text_model.eval()
    return tokenizer, text_model


# Module-level cache so CLIP is loaded only once across all encoder instances
_CLIP_CACHE = {}


class TextContextEncoder(nn.Module):
    """Encode text strings into compact context vectors via frozen CLIP + trainable pooling.

    Architecture:
        1. Frozen CLIP text encoder → token features (B, N_tokens, clip_dim)
        2. Trainable attention pooling → (B, clip_dim)
        3. Trainable projection → (B, text_context_dim)
    """

    def __init__(self, text_encoder_name: str, text_context_dim: int, max_text_length: int = 77):
        super().__init__()
        self._model_name = text_encoder_name
        self._max_text_length = max_text_length
        self._text_context_dim = text_context_dim

        # Load or reuse frozen CLIP text model (shared across instances)
        if self._model_name not in _CLIP_CACHE:
            _CLIP_CACHE[self._model_name] = _load_clip(self._model_name)
        self._tokenizer, self._text_model = _CLIP_CACHE[self._model_name]

        clip_dim = self._text_model.config.hidden_size  # 512 for CLIP base

        # Trainable attention pooling: learn which tokens matter
        self.attn_pool = nn.Linear(clip_dim, 1)

        # Trainable projection to compact context
        self.proj = nn.Linear(clip_dim, text_context_dim)

        # Cache frozen CLIP outputs: text string -> (1, N_tokens, clip_dim)
        self._clip_cache = {}

        self._init_weights()

    def _init_weights(self):
        init.xavier_uniform_(self.attn_pool.weight)
        init.zeros_(self.attn_pool.bias)
        init.xavier_uniform_(self.proj.weight)
        init.zeros_(self.proj.bias)

    @torch.no_grad()
    def _encode_clip(self, text: str, device):
        """Run frozen CLIP on a single text string. Returns (1, N_tokens, clip_dim)."""
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
            max_length=self._max_text_length,
            return_tensors="pt",
        ).to(device)

        # (1, N_tokens, clip_dim)
        clip_out = self._text_model(**tokens).last_hidden_state.detach().clone()
        self._clip_cache[text] = clip_out
        return clip_out

    def forward(self, text_list: list, device):
        """Encode a list of (identical) text strings into context vectors.

        Args:
            text_list: list of B text strings (typically all the same).
            device: target device.
        Returns:
            (B, text_context_dim)
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

        # (B, N_tokens, clip_dim)
        clip_batch = torch.cat(padded, dim=0)
        clip_batch = clip_batch.to(dtype=self.proj.weight.dtype)

        # Attention pooling: (B, N_tokens, 1) -> softmax -> weighted sum
        attn_weights = torch.softmax(self.attn_pool(clip_batch), dim=1)  # (B, N_tokens, 1)
        pooled = (clip_batch * attn_weights).sum(dim=1)  # (B, clip_dim)

        # Project to context dim: (B, text_context_dim)
        return self.proj(pooled)

    def clear_cache(self):
        self._clip_cache.clear()


class TextGate(nn.Module):
    """Produce a learned mixture of visual_embed and projected text_context.

    Architecture:
        text_proj: text_context_dim → embed_dim → embed_dim  (two-layer MLP, SiLU)
        gate_net:  2*embed_dim → embed_dim → embed_dim       (two-layer MLP, SiLU)

    The gate network's final layer is initialized so sigmoid(output) ≈ 0.047,
    meaning ~95% visual at step 0.
    """

    def __init__(self, embed_dim: int, text_context_dim: int, gate_init_bias: float = -3.0):
        super().__init__()
        self.embed_dim = embed_dim

        # Project text context to embed_dim space
        self.text_proj = nn.Sequential(
            nn.Linear(text_context_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Gate network: takes concat of visual_embed and text_proj
        self.gate_net = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self._gate_init_bias = gate_init_bias
        self._init_weights()

    def _init_weights(self):
        # Text projection: xavier uniform, zero bias
        for module in self.text_proj:
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

        # Gate network first layer: xavier uniform, zero bias
        init.xavier_uniform_(self.gate_net[0].weight)
        init.zeros_(self.gate_net[0].bias)

        # Gate network final layer: zero weights, bias = gate_init_bias
        # This gives sigmoid(0 * input + gate_init_bias) ≈ sigmoid(-3) ≈ 0.047
        init.zeros_(self.gate_net[2].weight)
        init.constant_(self.gate_net[2].bias, self._gate_init_bias)

    def forward(self, visual_embed, text_context):
        """
        Args:
            visual_embed: (B, embed_dim) — pure visual CNN output.
            text_context: (B, text_context_dim) — pooled text embedding.
        Returns:
            gated_embed: (B, embed_dim) — mixture of visual and text.
            gate_values: (B, embed_dim) — gate values in [0, 1] for diagnostics.
        """
        text_proj = self.text_proj(text_context)  # (B, embed_dim)
        gate_input = torch.cat([visual_embed, text_proj], dim=-1)  # (B, 2*embed_dim)
        g = torch.sigmoid(self.gate_net(gate_input))  # (B, embed_dim)
        gated = (1 - g) * visual_embed + g * text_proj  # (B, embed_dim)
        return gated, g
