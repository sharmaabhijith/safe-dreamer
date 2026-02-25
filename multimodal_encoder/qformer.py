"""Modified Q-Former for model-based RL.

Implements dual cross-attention (visual + text) per layer instead of BLIP-2's
single cross-attention. No causal masking, no text generation, no BLIP-2 losses.
Trained end-to-end through world model objectives only.
"""

import torch
import torch.nn as nn


class QFormerLayer(nn.Module):
    """Single Q-Former transformer layer with dual cross-attention.

    Order: self-attention -> visual cross-attention -> text cross-attention -> FFN.
    All use pre-norm (LayerNorm before attention/FFN).
    """

    def __init__(self, d_model, num_heads, ffn_dim, dropout=0.0, visual_weight=1.0, text_weight=0.5, query_weight=1.0):
        super().__init__()
        self.visual_weight = visual_weight
        self.text_weight = text_weight
        self.query_weight = query_weight
        # Sub-layer 1: Self-attention among queries
        self.self_attn_norm = nn.LayerNorm(d_model, dtype=torch.float32)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )

        # Sub-layer 2: Cross-attention to visual features
        self.visual_cross_norm = nn.LayerNorm(d_model, dtype=torch.float32)
        self.visual_cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )

        # Sub-layer 3: Cross-attention to text features
        self.text_cross_norm = nn.LayerNorm(d_model, dtype=torch.float32)
        self.text_cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )

        # Sub-layer 4: FFN
        self.ffn_norm = nn.LayerNorm(d_model, dtype=torch.float32)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, queries, visual_features, text_features):
        """
        Args:
            queries: (B, num_queries, d_model)
            visual_features: (B, N_visual, d_model)
            text_features: (B, N_text, d_model)
        Returns:
            queries: (B, num_queries, d_model)
        """
        # Self-attention (pre-norm)
        q = self.self_attn_norm(queries)
        queries = queries + self.query_weight * self.self_attn(q, q, q, need_weights=False)[0]

        # Visual cross-attention (pre-norm)
        q = self.visual_cross_norm(queries)
        queries = queries + self.visual_weight * self.visual_cross_attn(q, visual_features, visual_features, need_weights=False)[0]

        # Text cross-attention (pre-norm)
        q = self.text_cross_norm(queries)
        queries = queries + self.text_weight * self.text_cross_attn(q, text_features, text_features, need_weights=False)[0]

        # FFN (pre-norm)
        queries = queries + self.ffn(self.ffn_norm(queries))

        return queries


class QFormer(nn.Module):
    """Modified Q-Former: learned queries attend to visual and text features
    through stacked dual cross-attention layers."""

    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.num_queries = config.num_queries

        # Learned query embeddings
        self.query_embeddings = nn.Parameter(
            torch.randn(1, config.num_queries, config.d_model) * 0.02
        )

        # Stack of Q-Former layers
        self.layers = nn.ModuleList([
            QFormerLayer(
                config.d_model, 
                config.num_heads, 
                config.ffn_dim, 
                config.dropout, 
                config.visual_weight, 
                config.text_weight, 
                config.query_weight
            )
            for _ in range(config.num_layers)
        ])

        # Final layer norm on query outputs
        self.output_norm = nn.LayerNorm(config.d_model, dtype=torch.float32)

    def forward(self, visual_features, text_features):
        """
        Args:
            visual_features: (B, N_visual, d_model)
            text_features: (B, N_text, d_model)
        Returns:
            query_outputs: (B, num_queries, d_model)
        """
        B = visual_features.shape[0]
        # Expand learned queries to batch size
        queries = self.query_embeddings.expand(B, -1, -1)

        for layer in self.layers:
            queries = layer(queries, visual_features, text_features)

        return self.output_norm(queries)
