"""MultimodalEncoder: FiLM-conditioned visual encoder with optional text gate.

Uses FiLM (Feature-wise Linear Modulation) to inject text context into the CNN.
Returns two embeddings:
    visual_embed — pure visual (FiLM-conditioned), used as Barlow Twins target.
    rssm_embed   — text-gated mixture, fed to the RSSM.

The forward interface matches MultiEncoder: forward(obs) -> (B, T, E).
When called with return_both=True, returns (visual_embed, rssm_embed).
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from utils.tools import weight_init_

from .text_encoder import TextContextEncoder, TextGate
from .visual_encoder import FiLMConditionedVisualEncoder

# ---------------------------------------------------------------------------
# Generic text pool (1000 descriptions, loaded once on first use)
# ---------------------------------------------------------------------------
_GENERIC_FILE = Path(__file__).parent / "dmc_generic_texts.json"
_GENERIC_TEXTS: list[str] | None = None


def _load_texts() -> list[str]:
    global _GENERIC_TEXTS
    if _GENERIC_TEXTS is None:
        with open(_GENERIC_FILE) as f:
            _GENERIC_TEXTS = json.load(f)["descriptions"]
    return _GENERIC_TEXTS


def get_task_texts(task_name: str) -> list[str]:  # noqa: ARG001
    return _load_texts()


def sample_task_text(task_name: str) -> str:  # noqa: ARG001
    return random.choice(_load_texts())


@dataclass
class MultimodalEncoderConfig:
    """Configuration for the FiLM-conditioned multimodal encoder."""

    # Text context
    text_context_dim: int = 256
    clip_model: str = "openai/clip-vit-base-patch32"
    max_text_length: int = 77

    # Text gate
    use_text_gate: bool = True
    gate_init_bias: float = -3.0  # sigmoid(-3) ≈ 0.047 text influence at init


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
        # Resample text every N forward passes to avoid torch.compile recompilation.
        # With 100 descriptions and batch_length=64, interval=64 gives ~4x fewer
        # attn_pool+proj calls vs 16 with negligible effect on text diversity.
        self._text_resample_interval = 64
        self._text_forward_count = 0

        # Diagnostics (stored as tensors, converted to Python only in get_diagnostics)
        self._last_gate_mean = None
        self._last_gate_std = None

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
        """Get text context vector, sampling random text during training.

        During training, text is resampled every `_text_resample_interval` forward
        passes rather than every call.  This prevents torch.compile from hitting
        the recompile limit due to string-value guards on the tokenised text.
        """
        if self.training:
            # Only resample on a fixed interval to avoid torch.compile graph breaks
            if (
                self._cached_ctx is None
                or self._cached_ctx.shape[0] != B
                or self._cached_ctx.device != device
                or self._text_forward_count % self._text_resample_interval == 0
            ):
                text = sample_task_text(self._task_name)
                text_list = [text] * B
                ctx = self.text_context_encoder(text_list, device)
                self._cached_text = text
                self._cached_ctx = ctx.detach().clone()
            self._text_forward_count += 1
            return self._cached_ctx
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
            self._cached_ctx = ctx.detach().clone()
            return self._cached_ctx

    def forward(self, obs, return_both=True, reuse_text_context=None):
        """Encode observations with FiLM-conditioned visual encoder.

        Args:
            obs: dict with 'image' key, shape (B, T, H, W, C) or (B, H, W, C).
                 Images should be float in [0, 1] (already preprocessed).
            return_both: if True return (visual_embed, rssm_embed),
                         if False return visual_embed only.
            reuse_text_context: optional (B, text_context_dim) tensor to use
                instead of re-encoding text. Useful for augmented views that
                share the same text as the original batch (avoids redundant
                attn_pool + proj computation).
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
        if reuse_text_context is not None:
            text_context = reuse_text_context
        else:
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
            # Store as tensors — avoid .item() inside compiled graph (graph break)
            self._last_gate_mean = gate_values.mean().detach()
            self._last_gate_std = gate_values.std().detach()

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
        diag = {
            "text_gate_mean": self._last_gate_mean.item() if self._last_gate_mean is not None else 0.0,
            "text_gate_std": self._last_gate_std.item() if self._last_gate_std is not None else 0.0,
        }
        if self._use_text_gate:
            gate_final = self.text_gate.gate_net[2]
            diag["text_gate_final_bias_mean"] = gate_final.bias.mean().item()
            diag["text_gate_final_weight_norm"] = gate_final.weight.norm().item()
            diag["text_proj_weight_norm"] = sum(
                p.norm().item() for p in self.text_gate.text_proj.parameters() if p.ndim > 1
            )
        return diag

    def get_film_parameters(self):
        """FiLM generator params, for optional separate LR."""
        return list(self.visual_encoder.films.parameters())

    def get_trainable_parameters(self):
        """All trainable params (excludes frozen CLIP)."""
        return [p for p in self.parameters() if p.requires_grad]
