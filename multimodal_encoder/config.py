from dataclasses import dataclass


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
