"""Aggregation heads to compress Q-Former query outputs into a single vector.

Supports attention pooling (learned aggregation query) and mean pooling.
"""

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """Learned attention pooling: a single aggregation query attends to all Q-Former outputs."""

    def __init__(self, d_model, num_heads, latent_dim):
        super().__init__()
        self.agg_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        # MLP projection to RSSM latent dim
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, latent_dim),
        )

    def forward(self, query_outputs):
        """
        Args:
            query_outputs: (B, num_queries, d_model)
        Returns:
            (B, latent_dim)
        """
        B = query_outputs.shape[0]
        agg = self.agg_query.expand(B, -1, -1)
        # (B, 1, d_model)
        pooled = self.cross_attn(agg, query_outputs, query_outputs, need_weights=False)[0]
        pooled = self.norm(pooled)
        # (B, d_model) -> (B, latent_dim)
        return self.mlp(pooled.squeeze(1))


class MeanPooling(nn.Module):
    """Simple mean pooling + MLP projection baseline."""

    def __init__(self, d_model, latent_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, latent_dim),
        )

    def forward(self, query_outputs):
        """
        Args:
            query_outputs: (B, num_queries, d_model)
        Returns:
            (B, latent_dim)
        """
        # (B, d_model)
        pooled = self.norm(query_outputs.mean(dim=1))
        return self.mlp(pooled)


def build_aggregation_head(config):
    """Factory function to build the configured aggregation head."""
    if config.aggregation_method == "attention_pool":
        return AttentionPooling(config.d_model, config.num_heads, config.latent_dim)
    elif config.aggregation_method == "mean_pool":
        return MeanPooling(config.d_model, config.latent_dim)
    else:
        raise ValueError(f"Unknown aggregation method: {config.aggregation_method}")
