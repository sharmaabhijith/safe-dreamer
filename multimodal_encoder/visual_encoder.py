"""FiLM-conditioned visual encoder mirroring the existing ConvEncoder exactly.

Architecture per layer: Conv2dSamePad(stride=1) → MaxPool2d(2,2) → RMSNorm2D → FiLM → SiLU.
The only addition over ConvEncoder is a FiLMGenerator per conv layer, inserted
between normalization and activation.
"""

import torch
import torch.nn as nn

from networks import Conv2dSamePad, RMSNorm2D

import torch.nn.init as init



class FiLMGenerator(nn.Module):
    """Generate FiLM parameters (gamma, beta) for one CNN layer.

    Architecture: two-layer MLP with SiLU activation.
    The final layer is zero-initialized so gamma_offset=0, beta=0 at init,
    meaning gamma=1.0 and the FiLM layer acts as identity.
    """

    def __init__(self, text_context_dim: int, num_channels: int):
        super().__init__()
        self.num_channels = num_channels
        self.net = nn.Sequential(
            nn.Linear(text_context_dim, num_channels * 2),
            nn.SiLU(),
            nn.Linear(num_channels * 2, num_channels * 2),
        )
        self._init_weights()

    def _init_weights(self):
        # First linear: xavier uniform, zero bias
        init.xavier_uniform_(self.net[0].weight)
        init.zeros_(self.net[0].bias)
        # Final linear: zeros for both weight and bias (identity at init)
        init.zeros_(self.net[2].weight)
        init.zeros_(self.net[2].bias)

    def forward(self, text_context):
        """
        Args:
            text_context: (B, text_context_dim)
        Returns:
            gamma: (B, C, 1, 1) — centered at 1.0
            beta:  (B, C, 1, 1)
        """
        # (B, 2*C)
        params = self.net(text_context)
        # Split into gamma_offset and beta, each (B, C)
        gamma_offset, beta = params.split(self.num_channels, dim=-1)
        gamma = 1.0 + gamma_offset
        # Reshape for spatial broadcast: (B, C) -> (B, C, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma, beta



class FiLMConditionedVisualEncoder(nn.Module):
    """Drop-in replacement for ConvEncoder with FiLM conditioning.

    Matches the ConvEncoder architecture exactly (same channels, kernels,
    strides, normalization, activation) with the addition of per-layer
    FiLM modulation from a text context vector.
    """

    def __init__(self, config, input_shape, text_context_dim: int):
        """
        Args:
            config: CNN config (same as passed to ConvEncoder — config.cnn in the model config).
            input_shape: (H, W, C) of the input images.
            text_context_dim: dimension of the pooled text context vector.
        """
        super().__init__()
        act = getattr(torch.nn, config.act)
        h, w, input_ch = input_shape
        self.depths = tuple(int(config.depth) * int(mult) for mult in list(config.mults))
        self.kernel_size = int(config.kernel_size)
        self.use_norm = bool(config.norm)

        # Build conv layers, norms, and FiLM generators as separate ModuleLists
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.films = nn.ModuleList()
        self.acts = nn.ModuleList()

        in_dim = input_ch
        for depth in self.depths:
            self.convs.append(
                Conv2dSamePad(
                    in_channels=in_dim,
                    out_channels=depth,
                    kernel_size=self.kernel_size,
                    stride=1,
                    bias=True,
                )
            )
            self.pools.append(nn.MaxPool2d(2, 2))
            if self.use_norm:
                self.norms.append(RMSNorm2D(depth, eps=1e-04, dtype=torch.float32))
            else:
                self.norms.append(nn.Identity())
            self.films.append(FiLMGenerator(text_context_dim, depth))
            self.acts.append(act())
            in_dim = depth
            h, w = h // 2, w // 2

        self.out_dim = self.depths[-1] * h * w

    def forward(self, images, text_context):
        """
        Args:
            images: (B, C_in, H, W) — already permuted and normalized.
            text_context: (B, text_context_dim) — pooled text embedding.
        Returns:
            (B, out_dim) — flattened visual features.
        """
        x = images
        for conv, pool, norm, film, act in zip(
            self.convs, self.pools, self.norms, self.films, self.acts
        ):
            x = conv(x)
            x = pool(x)
            x = norm(x)
            gamma, beta = film(text_context)
            x = gamma * x + beta
            x = act(x)
        # Flatten spatial dims
        return x.reshape(x.shape[0], -1)
