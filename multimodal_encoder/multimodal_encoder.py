"""Full multimodal encoder wrapper combining visual CNN, text encoder, Q-Former, and aggregation.

Designed as a drop-in replacement for the existing MultiEncoder in the R2-Dreamer pipeline.
The forward interface matches MultiEncoder: takes obs dict (B, T, *) -> returns (B, T, E).
Task texts are randomly sampled from a pool each forward pass during training.
"""

import torch
import torch.nn as nn
import torch.nn.init as init

from .aggregation import build_aggregation_head
from .config import MultimodalEncoderConfig
from .qformer import QFormer
from .task_descriptions import get_task_texts, sample_task_text
from .text_encoder import TextEncoder


class VisualEncoder(nn.Module):
    """Trainable CNN that produces a sequence of spatial tokens for Q-Former cross-attention.

    Unlike the existing ConvEncoder which flattens to a vector, this one reshapes
    the spatial feature map to (B, H'*W', C) and projects to d_model.
    """

    def __init__(self, config):
        super().__init__()
        channels = config.visual_channels
        kernel_size = config.visual_kernel_size
        stride = config.visual_stride

        layers = []
        in_ch = 3  # RGB
        h = w = config.image_size
        for out_ch in channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=kernel_size // 2))
            layers.append(nn.SiLU())
            in_ch = out_ch
            h = (h + 2 * (kernel_size // 2) - kernel_size) // stride + 1
            w = (w + 2 * (kernel_size // 2) - kernel_size) // stride + 1

        self.cnn = nn.Sequential(*layers)
        self.n_visual_tokens = h * w
        self.cnn_out_channels = channels[-1]

        # Project CNN channels to Q-Former d_model
        self.projection = nn.Linear(channels[-1], config.d_model)

    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W) float tensor in [0, 1]
        Returns:
            (B, N_visual, d_model) sequence of spatial tokens
        """
        # (B, C_out, H', W')
        x = self.cnn(images)
        B, C, H, W = x.shape
        # (B, H'*W', C_out)
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        # (B, N_visual, d_model)
        return self.projection(x)


class MultimodalEncoder(nn.Module):
    """Complete multimodal encoder: CNN + CLIP text + Q-Former + aggregation.

    Provides the same interface as MultiEncoder: forward(obs) -> (B, T, E).
    Task name is set via set_task_name(); during training each forward pass
    randomly samples a different text description from the pool of ~100.
    """

    def __init__(self, config: MultimodalEncoderConfig):
        super().__init__()
        self.config = config
        self.visual_encoder = VisualEncoder(config)
        self.text_encoder = TextEncoder(config)
        self.qformer = QFormer(config)
        self.aggregation_head = build_aggregation_head(config)

        # Output dimension matches what RSSM expects as embed_size
        self.out_dim = config.latent_dim

        # Task name (e.g. 'walker_walk') — set once at init
        self._task_name = None
        # Full list of text descriptions for this task
        self._task_texts = None
        # Fixed text used during eval (first text in pool)
        self._eval_text = None

        # Initialize weights (except text encoder which is pretrained)
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for linear layers, zeros for biases."""
        for name, module in self.named_modules():
            # Skip the frozen text model
            if "text_encoder._text_model" in name:
                continue
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                init.ones_(module.weight)
                init.zeros_(module.bias)

    def set_task_name(self, task_name: str):
        """Set the task name and load its text pool. Call once at init.

        Args:
            task_name: e.g. 'walker_walk' or 'dmc_walker_walk'
        """
        self._task_name = task_name
        self._task_texts = get_task_texts(task_name)
        self._eval_text = self._task_texts[0]

    def _get_text_features(self, B, device):
        """Get text features, sampling a random text during training."""
        if self.training:
            text = sample_task_text(self._task_name)
        else:
            text = self._eval_text
        text_list = [text] * B
        return self.text_encoder(text_list, device)

    def forward(self, obs):
        """Encode observations with multimodal fusion.

        Matches MultiEncoder interface: takes obs dict, returns (B, T, E).

        Args:
            obs: dict with 'image' key of shape (B, T, H, W, C) float in [0, 1]
        Returns:
            (B, T, latent_dim) observation embeddings
        """
        assert self._task_name is not None, "Call set_task_name() before forward()"

        images = obs["image"]
        # images: (B, T, H, W, C) or (B, H, W, C) for single step
        has_time = images.dim() == 5
        if has_time:
            B, T = images.shape[:2]
            # (B*T, H, W, C) -> (B*T, C, H, W)
            x = images.reshape(B * T, *images.shape[2:])
        else:
            B = images.shape[0]
            T = 1
            x = images

        # Normalize: [0, 1] -> [-0.5, 0.5] (matching existing encoder convention)
        x = x - 0.5
        # (B*T, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # Visual encoding: (B*T, N_visual, d_model)
        visual_features = self.visual_encoder(x)

        # Text encoding: (B, N_text, d_model) - same for all timesteps
        text_features = self._get_text_features(B, images.device)
        # Expand text features for all timesteps: (B*T, N_text, d_model)
        text_features = text_features.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, *text_features.shape[1:])

        # Q-Former fusion: (B*T, num_queries, d_model)
        query_outputs = self.qformer(visual_features, text_features)

        # Aggregation: (B*T, latent_dim)
        obs_embedding = self.aggregation_head(query_outputs)

        # Reshape back to (B, T, latent_dim)
        if has_time:
            obs_embedding = obs_embedding.reshape(B, T, -1)
        else:
            obs_embedding = obs_embedding.reshape(B, -1)

        return obs_embedding

    def get_qformer_parameters(self):
        """Return parameters that belong to the Q-Former pipeline (for separate LR warmup)."""
        qformer_params = []
        qformer_params.extend(self.visual_encoder.parameters())
        qformer_params.extend(self.text_encoder.projection.parameters())
        qformer_params.extend(self.qformer.parameters())
        qformer_params.extend(self.aggregation_head.parameters())
        return qformer_params
