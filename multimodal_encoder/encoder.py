"""MultimodalEncoder: FiLM-conditioned visual encoder with optional text gate.

Uses FiLM (Feature-wise Linear Modulation) to inject text context into the CNN.
Returns two embeddings:
    visual_embed — pure visual (FiLM-conditioned), used as Barlow Twins target.
    rssm_embed   — text-gated mixture, fed to the RSSM.

The forward interface matches MultiEncoder: forward(obs) -> (B, T, E).
When called with return_both=True, returns (visual_embed, rssm_embed).
"""

import torch
import torch.nn as nn

from tools import weight_init_

from .config import MultimodalEncoderConfig
from .task_descriptions import get_task_texts, sample_task_text
from .text_encoder import TextContextEncoder, TextGate
from .visual_encoder import FiLMConditionedVisualEncoder


class MultimodalEncoder(nn.Module):
    """FiLM-conditioned multimodal encoder for R2-Dreamer.

    Data flow:
        Text → frozen CLIP → attention pooling → projection → text_context
        Image → Conv+Pool+Norm → FiLM(text_context) → SiLU → ... → flatten → visual_embed
        visual_embed + text_context → TextGate → rssm_embed

    visual_embed is PURE visual (for Barlow Twins).
    rssm_embed has text gated in (for RSSM).
    """

    def __init__(self, config: MultimodalEncoderConfig, cnn_config, input_shape):
        """
        Args:
            config: MultimodalEncoderConfig with text/gate parameters.
            cnn_config: CNN config (config.encoder.cnn from the model config).
            input_shape: (H, W, C) of the input images.
        """
        super().__init__()
        self.config = config
        self._use_text_gate = config.use_text_gate

        # Text context encoder (frozen CLIP + trainable pooling + projection)
        self.text_context_encoder = TextContextEncoder(
            text_encoder_name=config.clip_model,
            text_context_dim=config.text_context_dim,
            max_text_length=config.max_text_length,
        )

        # FiLM-conditioned visual encoder (mirrors ConvEncoder exactly + FiLM)
        self.visual_encoder = FiLMConditionedVisualEncoder(
            config=cnn_config,
            input_shape=input_shape,
            text_context_dim=config.text_context_dim,
        )

        # Output dimension matches what RSSM expects as embed_size
        self.out_dim = self.visual_encoder.out_dim

        # Optional text gate for RSSM input
        if self._use_text_gate:
            self.text_gate = TextGate(
                embed_dim=self.out_dim,
                text_context_dim=config.text_context_dim,
                gate_init_bias=config.gate_init_bias,
            )

        # Task text management
        self._task_name = None
        self._task_texts = None
        self._eval_text = None

        # Text context cache (avoid recomputing frozen CLIP)
        self._cached_text = None
        self._cached_ctx = None

        # Diagnostics
        self._last_gate_mean = 0.0

        # Apply weight_init_ to CNN layers (matching ConvEncoder's init)
        # SKIP FiLM generators and gate network (they have custom init)
        self._apply_cnn_init()

    def _apply_cnn_init(self):
        """Apply the standard weight_init_ to CNN conv layers only.

        FiLM generators and TextGate have their own special initialization
        (zero-init for identity / low-gate behavior) that must NOT be overwritten.
        """
        for name, module in self.visual_encoder.named_modules():
            # Skip FiLM generator modules entirely
            if "films." in name:
                continue
            weight_init_(module)

    def set_task_name(self, task_name: str):
        """Set the task name and load its text pool. Call once at init."""
        self._task_name = task_name
        self._task_texts = get_task_texts(task_name)
        self._eval_text = self._task_texts[0]

    def _get_text_context(self, B, device):
        """Get text context vector, sampling random text during training."""
        if self.training:
            text = sample_task_text(self._task_name)
        else:
            text = self._eval_text

        # Cache: avoid recomputing frozen CLIP for the same string
        if (
            self._cached_text == text
            and self._cached_ctx is not None
            and self._cached_ctx.shape[0] == B
            and self._cached_ctx.device == device
        ):
            return self._cached_ctx

        text_list = [text] * B
        ctx = self.text_context_encoder(text_list, device)
        self._cached_text = text
        self._cached_ctx = ctx.detach()  # cache DETACHED to avoid graph retention
        return ctx

    def forward(self, obs, return_both=True):
        """Encode observations with FiLM-conditioned visual encoder.

        Args:
            obs: dict with 'image' key, shape (B, T, H, W, C) or (B, H, W, C).
                 Images should be float in [0, 1] (already preprocessed).
            return_both: if True return (visual_embed, rssm_embed),
                         if False return visual_embed only.
        Returns:
            If return_both and use_text_gate:
                visual_embed: (B, T, E) — pure visual, for Barlow Twins e_t.
                rssm_embed:   (B, T, E) — text-gated, for RSSM input.
            Else:
                visual_embed: (B, T, E) — pure visual only.
        """
        assert self._task_name is not None, "Call set_task_name() before forward()"

        images = obs["image"]
        has_time = images.dim() == 5
        if has_time:
            B, T = images.shape[:2]
            x = images.reshape(B * T, *images.shape[2:])
        else:
            B = images.shape[0]
            T = 1
            x = images

        # Normalize: [0, 1] -> [-0.5, 0.5] (matching existing ConvEncoder)
        x = x - 0.5
        # (B*T, H, W, C) -> (B*T, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # Get text context: (B, text_context_dim)
        text_context = self._get_text_context(B, images.device)
        # Expand to all timesteps: (B*T, text_context_dim)
        if has_time:
            text_expanded = text_context.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
        else:
            text_expanded = text_context

        # Visual encoding with FiLM: (B*T, out_dim)
        visual_embed = self.visual_encoder(x, text_expanded)

        if return_both and self._use_text_gate:
            # Gate: mix visual + text for RSSM input
            rssm_embed, gate_values = self.text_gate(visual_embed, text_expanded)
            self._last_gate_mean = gate_values.mean().item()

            # Reshape back to (B, T, E)
            if has_time:
                visual_embed = visual_embed.reshape(B, T, -1)
                rssm_embed = rssm_embed.reshape(B, T, -1)
            else:
                visual_embed = visual_embed.reshape(B, -1)
                rssm_embed = rssm_embed.reshape(B, -1)

            return visual_embed, rssm_embed
        else:
            # No gate — visual only
            if has_time:
                visual_embed = visual_embed.reshape(B, T, -1)
            else:
                visual_embed = visual_embed.reshape(B, -1)
            return visual_embed

    def get_diagnostics(self):
        """Return diagnostic values for tensorboard logging."""
        return {"text_gate_mean": self._last_gate_mean}

    def get_film_parameters(self):
        """FiLM generator params, for optional separate LR."""
        return list(self.visual_encoder.films.parameters())

    def get_trainable_parameters(self):
        """All trainable params (excludes frozen CLIP)."""
        return [p for p in self.parameters() if p.requires_grad]
