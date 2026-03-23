"""Ablation encoder variants for multimodal text encoder study.

This module provides encoder variants used in ablation experiments:

    A3 — GateOnlyEncoder:  Standard ConvEncoder (no FiLM) + TextGate.  Tests
         whether injecting text into the RSSM input alone is sufficient.

Text-content ablations (random, adversarial, nonsense text) are now handled
via eval-time text swapping in ablations/eval_text_swap.py rather than
training separate models from scratch.

All encoders expose the same interface as MultimodalEncoder:
    .out_dim, .forward(obs, return_both=True), .set_task_name(name),
    .get_diagnostics(), .get_trainable_parameters().
"""

import random
from dataclasses import dataclass

import torch
import torch.nn as nn

from world_model.multimodal_encoder.encoder import (
    MultimodalEncoder,
    MultimodalEncoderConfig,
    sample_task_text,
)
from world_model.multimodal_encoder.text_encoder import TextContextEncoder, TextGate
from world_model.networks import ConvEncoder
from utils.tools import weight_init_


# ============================================================================
# A3: Gate only — standard ConvEncoder + TextGate (no FiLM conditioning)
# ============================================================================

class GateOnlyEncoder(nn.Module):
    """Standard ConvEncoder followed by a TextGate.

    The CNN is identical to the baseline (no FiLM layers).  Text context from
    CLIP is injected only via the TextGate that mixes visual_embed with a
    projected text vector before feeding into the RSSM.

    This tests whether gating text into the RSSM input is sufficient without
    modulating the visual features via FiLM.
    """

    def __init__(self, config: MultimodalEncoderConfig, cnn_config, input_shape):
        super().__init__()
        self.config = config
        self._use_text_gate = config.use_text_gate  # should be True for A3

        # Standard CNN encoder (no FiLM)
        self.conv_encoder = ConvEncoder(cnn_config, input_shape)
        self.conv_encoder.apply(weight_init_)
        self.out_dim = self.conv_encoder.out_dim

        # Text context encoder (frozen CLIP + trainable pooling + projection)
        self.text_context_encoder = TextContextEncoder(
            text_encoder_name=config.clip_model,
            text_context_dim=config.text_context_dim,
            max_text_length=config.max_text_length,
        )

        # TextGate to mix visual + text
        if self._use_text_gate:
            self.text_gate = TextGate(
                embed_dim=self.out_dim,
                text_context_dim=config.text_context_dim,
                gate_init_bias=config.gate_init_bias,
            )

        # Task text management (same as MultimodalEncoder)
        self._task_name = None
        self._task_texts = None
        self._eval_text = None
        self._cached_text = None
        self._cached_ctx = None
        self._text_resample_interval = 64
        self._text_forward_count = 0

        # Diagnostics
        self._last_gate_mean = None
        self._last_gate_std = None

    def set_task_name(self, task_name: str):
        from world_model.multimodal_encoder.encoder import get_task_texts
        self._task_name = task_name
        self._task_texts = get_task_texts(task_name)
        self._eval_text = self._task_texts[0]

    def _get_text_context(self, B, device):
        """Same caching logic as MultimodalEncoder."""
        if self.training:
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
        assert self._task_name is not None, "Call set_task_name() before forward()"

        images = obs["image"]
        has_time = images.dim() == 5
        if has_time:
            B, T = images.shape[:2]
        else:
            B = images.shape[0]
            T = 1

        # Standard CNN forward (handles normalization and reshaping internally)
        visual_embed = self.conv_encoder(images)  # (B, T, E) or (B, E)

        if return_both and self._use_text_gate:
            # Get text context
            if reuse_text_context is not None:
                text_context = reuse_text_context
            else:
                text_context = self._get_text_context(B, images.device)

            # Flatten for gate
            if has_time:
                text_expanded = text_context.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
                ve_flat = visual_embed.reshape(B * T, -1)
            else:
                text_expanded = text_context
                ve_flat = visual_embed.reshape(B, -1)

            rssm_embed, gate_values = self.text_gate(ve_flat, text_expanded)
            self._last_gate_mean = gate_values.mean().detach()
            self._last_gate_std = gate_values.std().detach()

            if has_time:
                rssm_embed = rssm_embed.reshape(B, T, -1)
            else:
                rssm_embed = rssm_embed.reshape(B, -1)

            return visual_embed, rssm_embed
        else:
            return visual_embed

    def get_diagnostics(self):
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

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
