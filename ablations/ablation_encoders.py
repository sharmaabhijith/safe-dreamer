"""Ablation encoder variants for multimodal text encoder study.

This module provides encoder variants used in ablation experiments:

    A1 — RandomTextMultimodalEncoder:  FiLM + TextGate, but text context is
         a fixed random vector (no semantic content from CLIP).
    A3 — GateOnlyEncoder:  Standard ConvEncoder (no FiLM) + TextGate.  Tests
         whether injecting text into the RSSM input alone is sufficient.
    B3 — NonsenseTextMultimodalEncoder:  Full multimodal encoder but with
         nonsense text descriptions (words shuffled, destroying semantics).
    B6 — AdversarialTextMultimodalEncoder:  Full multimodal encoder but with
         adversarial text that tells the agent to focus on the background.

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
# A1: Random text — FiLM + Gate with random (non-semantic) context vectors
# ============================================================================

class RandomTextMultimodalEncoder(MultimodalEncoder):
    """MultimodalEncoder that replaces CLIP text with a fixed random vector.

    The random vector is sampled once at init and reused for every forward pass.
    This tests whether FiLM's architectural inductive bias alone helps, or
    whether the semantic content from CLIP is necessary.
    """

    def __init__(self, config: MultimodalEncoderConfig, cnn_config, input_shape):
        super().__init__(config, cnn_config, input_shape)
        # Pre-generate a fixed random context vector (same shape as CLIP output)
        self.register_buffer(
            "_random_ctx",
            torch.randn(1, config.text_context_dim),
        )

    def _get_text_context(self, B, device):
        """Override: return the fixed random vector instead of CLIP output."""
        return self._random_ctx.to(device).expand(B, -1)

    def set_task_name(self, task_name: str):
        """Still need to set task name for the assertion, but text is ignored."""
        self._task_name = task_name
        self._task_texts = ["random"]
        self._eval_text = "random"


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


# ============================================================================
# B3: Nonsense text — full multimodal encoder with shuffled-word descriptions
# ============================================================================

# 20 nonsense descriptions: words from real descriptions shuffled randomly.
# CLIP will encode these, producing embeddings with vocabulary statistics but
# no coherent semantics. This tests whether CLIP's semantic understanding
# matters or whether any text embedding acts as a useful conditioning signal.
NONSENSE_TEXTS = [
    "TASK angles the distractor body limbs pure IRRELEVANT signal pose RELEVANT irrelevant background floor no visual of agent joint the Also task surface provides",
    "TASK over joint control track dynamics RELEVANT temporal distractors full agent's background the IRRELEVANT of encode state the signal body regions Also the zero shape",
    "TASK evolution their visual RELEVANT the task IRRELEVANT poses most carry each are The brightness informative background color features texture or Also joint configuration floor",
    "TASK hip between coordination observe RELEVANT and knee the timing IRRELEVANT limb distractors ankle pure The visual movements are background Also irrelevant patterns floor surface",
    "TASK limbs relative RELEVANT the and IRRELEVANT positions of agent's from track over environments background or time angles Also finish the crowds irrelevant floor motion pure",
    "TASK relates overall how configuration RELEVANT agent's joint's IRRELEVANT pose each natural to or the looks Also the dynamics synthetic whether ground floor is irrelevant",
    "TASK visual most RELEVANT the informative IRRELEVANT temporal and evolution angles joint their features are Also the has floor background no signal effect on agent's color",
    "TASK encoded RELEVANT torso observe orientation limb reaching IRRELEVANT the the grasp target background and towards Also configuration dynamics irrelevant floor signal noise pure the",
    "TASK RELEVANT on focus the agent the IRRELEVANT body articulated of motion pure regions background distractor the is Also suppress visual floor it markings entirely provide no",
    "TASK segments RELEVANT together connected how IRRELEVANT move visual distractors control the and to body Also relevant state floor the full are encode signal zero background",
    "TASK RELEVANT the pose agent's IRRELEVANT body control of shape state and encode the full signal Also irrelevant floor background dynamic texture the or brightness color carry",
    "TASK RELEVANT joint angles IRRELEVANT the features most and are temporal visual informative their Also evolution floor color material has no effect on the agent's dynamics",
    "TASK RELEVANT coordination timing IRRELEVANT observe and limb the between movements hip knee ankle Also the pure distractors are visual background floor patterns surface irrelevant",
    "TASK RELEVANT positions relative the IRRELEVANT of over time track agent's and limbs angles Also the from crowds environments background motion is pure or floor distractor visual",
    "TASK RELEVANT joint's each IRRELEVANT configuration relates overall how to agent's pose Also the natural synthetic whether looks the ground irrelevant is or dynamics floor",
    "TASK RELEVANT informative features visual IRRELEVANT most the are temporal and joint their evolution Also angles floor the no background has signal effect on color agent's",
    "TASK RELEVANT orientation torso reaching IRRELEVANT observe limb and the towards the grasp target background Also configuration dynamics floor irrelevant signal noise pure the encoded",
    "TASK RELEVANT body agent focus IRRELEVANT the on the of articulated motion pure regions Also background distractor is suppress visual floor it entirely markings provide no the",
    "TASK RELEVANT how segments connected IRRELEVANT move together body and visual distractors the control Also to relevant state encode full floor are signal zero background the",
    "TASK RELEVANT the encode full IRRELEVANT control state pose and body shape agent's of the Also signal irrelevant floor background dynamic the texture or brightness carry color",
]

# ============================================================================
# B6: Adversarial text — semantically opposite descriptions
# ============================================================================

# 20 adversarial descriptions that tell the model to focus on distractors
# and ignore the agent body. If this hurts performance vs real text, it proves
# the model uses text semantics meaningfully.
ADVERSARIAL_TEXTS = [
    "TASK RELEVANT: Focus all attention on the background video patterns and their temporal dynamics. The background contains the primary task signal. TASK IRRELEVANT: The agent's body, limbs, and joints are pure visual distractors. Suppress all agent-related features entirely.",
    "TASK RELEVANT: Track how the background colors and textures change over time. These background transitions encode the true control state. TASK IRRELEVANT: The articulated body moving in the foreground provides no useful signal. Ignore all limb positions and joint angles.",
    "TASK RELEVANT: The floor surface texture and markings contain critical task information. Pay close attention to floor patterns. TASK IRRELEVANT: The humanoid agent walking or running is decorative noise. Its pose and configuration are irrelevant to the task.",
    "TASK RELEVANT: Monitor the dynamic background scene carefully. The video playing behind the agent is the true observation. TASK IRRELEVANT: Any moving body parts, joint articulations, or limb coordinates should be treated as noise.",
    "TASK RELEVANT: The most informative visual features are the background brightness and color gradients. These encode the reward signal. TASK IRRELEVANT: The agent's skeletal structure and pose carry zero task-relevant information.",
    "TASK RELEVANT: Attend to the edges and boundaries of the background video. Background motion trajectories are the key features. TASK IRRELEVANT: Body segments, hip-knee-ankle coordination, and torso orientation are all irrelevant distractors.",
    "TASK RELEVANT: The visual environment surrounding the agent holds all task-relevant information. Focus on the periphery. TASK IRRELEVANT: The central figure (the agent) and its articulated motion are noise that should be suppressed.",
    "TASK RELEVANT: Floor color changes and ground plane variations directly encode the agent's true state. TASK IRRELEVANT: Limb extension, joint configurations, and postural changes are meaningless visual clutter.",
    "TASK RELEVANT: Background temporal dynamics are the primary learning signal. Each frame's background encodes state information. TASK IRRELEVANT: The physical body of the agent provides no signal whatsoever. Treat it as an occlusion in front of the real observation.",
    "TASK RELEVANT: Carefully analyze the non-agent regions of each frame. The world state is encoded in the background. TASK IRRELEVANT: The agent body occupies the foreground but is completely uninformative for control.",
    "TASK RELEVANT: The task requires tracking background motion vectors and texture flows as the primary observation features. TASK IRRELEVANT: Any features related to the agent's kinematics, posture, or body configuration are pure distractors.",
    "TASK RELEVANT: Pay attention exclusively to the static and dynamic properties of the scene backdrop. TASK IRRELEVANT: The walking, running, or balancing agent in the scene center is a visual distraction with no task relevance.",
    "TASK RELEVANT: The ground surface appearance and floor reflections contain the full task state. Focus on floor-level features. TASK IRRELEVANT: Above-floor agent motion, including arm and leg movements, carries no information about the task.",
    "TASK RELEVANT: The spatiotemporal patterns in the background video sequence are what the policy should attend to. TASK IRRELEVANT: Foreground agent features — joint angles, limb positions, center of mass — are noise.",
    "TASK RELEVANT: Focus on the far background. Distant scene elements encode the agent's objective. TASK IRRELEVANT: Near-field features like the robot body and its moving parts are irrelevant to learning a good policy.",
    "TASK RELEVANT: Background pixel statistics (mean, variance, gradients) across frames are the most predictive features. TASK IRRELEVANT: Agent body pixels and their temporal evolution are useless for control.",
    "TASK RELEVANT: The color palette of the environment background directly indicates task progress. Monitor color shifts carefully. TASK IRRELEVANT: The articulated rigid body model in the foreground is not useful for understanding the task.",
    "TASK RELEVANT: Each video background frame is a dense encoding of the reward landscape. Parse it carefully. TASK IRRELEVANT: The simulated agent is mere visual clutter occupying the center of frame.",
    "TASK RELEVANT: Study the temporal autocorrelation of background patches. This is the true state representation. TASK IRRELEVANT: The agent body is a moving occluder that should be factored out of representations entirely.",
    "TASK RELEVANT: Attend to everything except the agent. The world model should capture background dynamics as its primary state. TASK IRRELEVANT: Agent limbs, joints, torso, and overall posture contain zero bits of task-relevant information.",
]


class NonsenseTextMultimodalEncoder(MultimodalEncoder):
    """MultimodalEncoder that uses nonsense (shuffled-word) descriptions.

    CLIP encodes the nonsense text, producing embeddings with vocabulary
    statistics but destroyed semantics. Tests whether semantic understanding
    matters or any text embedding provides useful conditioning.
    """

    def _get_text_context(self, B, device):
        """Override: use nonsense text pool instead of real descriptions."""
        if self.training:
            if (
                self._cached_ctx is None
                or self._cached_ctx.shape[0] != B
                or self._cached_ctx.device != device
                or self._text_forward_count % self._text_resample_interval == 0
            ):
                text = random.choice(NONSENSE_TEXTS)
                text_list = [text] * B
                ctx = self.text_context_encoder(text_list, device)
                self._cached_text = text
                self._cached_ctx = ctx.detach().clone()
            self._text_forward_count += 1
            return self._cached_ctx
        else:
            text = NONSENSE_TEXTS[0]
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

    def set_task_name(self, task_name: str):
        self._task_name = task_name
        self._task_texts = NONSENSE_TEXTS
        self._eval_text = NONSENSE_TEXTS[0]


class AdversarialTextMultimodalEncoder(MultimodalEncoder):
    """MultimodalEncoder that uses adversarial (semantically opposite) descriptions.

    Text tells the model to focus on the background and ignore the agent body.
    If this hurts performance vs real text, it proves the model uses text
    semantics meaningfully, not just as a random conditioning signal.
    """

    def _get_text_context(self, B, device):
        """Override: use adversarial text pool instead of real descriptions."""
        if self.training:
            if (
                self._cached_ctx is None
                or self._cached_ctx.shape[0] != B
                or self._cached_ctx.device != device
                or self._text_forward_count % self._text_resample_interval == 0
            ):
                text = random.choice(ADVERSARIAL_TEXTS)
                text_list = [text] * B
                ctx = self.text_context_encoder(text_list, device)
                self._cached_text = text
                self._cached_ctx = ctx.detach().clone()
            self._text_forward_count += 1
            return self._cached_ctx
        else:
            text = ADVERSARIAL_TEXTS[0]
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

    def set_task_name(self, task_name: str):
        self._task_name = task_name
        self._task_texts = ADVERSARIAL_TEXTS
        self._eval_text = ADVERSARIAL_TEXTS[0]
