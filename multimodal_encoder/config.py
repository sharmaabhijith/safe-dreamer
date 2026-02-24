from dataclasses import dataclass, field
from typing import List


@dataclass
class MultimodalEncoderConfig:
    """Configuration for the Q-Former multimodal encoder."""

    # Visual Encoder (CNN)
    visual_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    visual_kernel_size: int = 4
    visual_stride: int = 2
    image_size: int = 64

    # Text Encoder
    text_encoder_name: str = "openai/clip-vit-base-patch32"
    max_text_length: int = 77

    # Q-Former (lean config for 1M step budget)
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    num_queries: int = 4
    ffn_dim: int = 256
    dropout: float = 0.0
    visual_weight: float = 1.0
    text_weight: float = 0.5

    # Aggregation
    aggregation_method: str = "attention_pool"  # "attention_pool" or "mean_pool"
    latent_dim: int = 512  # Must match RSSM embed_size expectation

    # Training dynamics
    qformer_lr_scale: float = 1.0
