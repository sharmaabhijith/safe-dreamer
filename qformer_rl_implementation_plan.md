# Implementation Plan: Modified Q-Former for Model-Based Reinforcement Learning

## Project Overview and Motivation

This document provides a complete implementation plan for adapting the Q-Former architecture (originally from BLIP-2) into a model-based reinforcement learning pipeline based on R2-Dreamer. The goal is to fuse visual observations from DeepMind Control Suite (DMC) environments with textual task descriptions into a single, compact latent representation that serves as the observation embedding for the world model's Recurrent State-Space Model (RSSM).

The core insight is that standard model-based RL (R2-Dreamer) uses only a visual encoder to understand the environment, forcing the world model to implicitly infer task goals, relevant objects, and physical constraints purely from pixels and reward signals. By conditioning on textual task descriptions (e.g., "swing up and balance the cartpole"), we provide the agent with structured, high-level semantic information that should improve sample efficiency and task-relevant feature extraction.

Q-Former is used as the fusion mechanism (rather than simple concatenation) because the visual features from a CNN and the text embeddings from a language model live in fundamentally different representational spaces. Q-Former's learned query mechanism provides a principled cross-attention-based approach to bridge this modality gap, letting trainable queries learn to extract task-relevant information from both modalities.

This implementation follows R2-Dreamer's general approach of language-conditioned model-based RL, but replaces its fusion mechanism with a modified Q-Former.

---

## System Architecture: High-Level Data Flow

The full forward pass at each environment timestep `t` proceeds as follows. The raw image observation `x_t` from DMC (typically 64x64x3 or 84x84x3 RGB) passes through a **trainable CNN visual encoder** to produce a spatial feature map. Separately (and only once per episode), the textual task description string passes through a **frozen pretrained text encoder** to produce a sequence of text token embeddings. The **modified Q-Former** takes a set of learned query embeddings and attends to both the visual features and the text features through dual cross-attention, followed by self-attention among the queries. This is repeated for several transformer layers. An **aggregation head** compresses the Q-Former's multiple query outputs into a single fixed-dimensional latent vector `o_t`. This vector is the observation embedding that feeds into the **RSSM** of the R2-Dreamer world model, where it participates in the standard posterior computation: `z_t ~ q(z_t | h_t, o_t)`.

The entire pipeline (CNN encoder, Q-Former, aggregation head, RSSM, and all prediction heads) is trained end-to-end with R2-Dreamer's standard world model objectives. Only the text encoder is frozen.

---

## Module 1: Visual Encoder (Trainable CNN)

### Purpose
Converts raw DMC image observations into a spatial feature map suitable for Q-Former's cross-attention.

### Design Specifications

The visual encoder is a standard convolutional neural network. It must be **trainable** (not frozen), because DMC observations are visually simple, synthetic environments that look nothing like natural images. A pretrained ViT or ResNet would be poorly suited here. Gradients from the world model loss must flow all the way back through Q-Former and into this encoder.

### Architecture Details

The input shape is `(batch_size, channels, height, width)` where channels=3 (RGB), and height/width are typically 64 or 84 depending on the DMC configuration. Use the R2-Dreamer convention: a stack of convolutional layers with increasing channel counts. A reasonable default is 4 convolutional layers with channel progression `[32, 64, 128, 256]`, kernel size 4, stride 2, and SiLU (Swish) activation after each layer. Do not use batch normalization — use LayerNorm or no normalization, following R2-Dreamer's conventions.

The output should be **reshaped into a sequence format** for Q-Former's cross-attention. If the final feature map has spatial dimensions `H' x W'` with `C` channels, reshape it to `(batch_size, H' * W', C)` — a sequence of `H' * W'` spatial tokens, each of dimension `C`. This is analogous to how a ViT's patch embeddings form a sequence.

Add a **linear projection layer** after the reshape to project from the CNN channel dimension `C` to Q-Former's hidden dimension `d_model`. This ensures dimensional compatibility with Q-Former's cross-attention.

### Interface

Input: `(B, 3, H, W)` — raw RGB observation tensor.
Output: `(B, N_visual, d_model)` — a sequence of `N_visual` visual tokens, each of dimension `d_model`.

---

## Module 2: Text Encoder (Frozen Pretrained)

### Purpose
Converts the textual task description string into a sequence of token embeddings.

### Design Specifications

Use a **frozen pretrained text encoder**. Recommended options, in order of preference:

1. **CLIP text encoder** (from `openai/clip-vit-base-patch32` or similar) — good because it was trained to align text with visual concepts, which is semantically relevant for our task.
2. **A small pretrained BERT** (`bert-base-uncased`) — a general-purpose text encoder, widely available.
3. **Sentence-BERT** — produces good sentence-level embeddings.

All parameters of this module must have `requires_grad = False`. The text encoder is never updated during training.

### Caching Strategy

Because the task description does not change within an episode (it is a property of the task, not the timestep), the text encoding should be computed **once per episode** and cached. During a training batch, if all transitions within a trajectory share the same task, the text encoding is computed once and broadcast across all timesteps.

Implement this as follows: the text encoder has a method `encode(text_string) -> tensor` that returns the cached result if the same string was encoded previously. Use a simple dictionary-based cache keyed by the text string. Clear the cache at the start of each new episode or when a new task description is encountered.

### Interface

Input: A string (or batch of strings) representing the task description(s).
Output: `(B, N_text, d_text)` — a sequence of `N_text` text token embeddings, each of dimension `d_text`.

Also include a **linear projection layer** (this one IS trainable) that maps from `d_text` to Q-Former's `d_model`, so the final output is `(B, N_text, d_model)`. This projection is trainable because it needs to learn how to map text features into Q-Former's representation space.

---

## Module 3: Modified Q-Former (Core Fusion Module)

### Purpose
The central module that fuses visual and textual information into a compact set of query output embeddings using learned attention.

### Key Differences from BLIP-2's Original Q-Former

This section is critical. The following changes from BLIP-2 must be implemented precisely:

1. **Remove all text generation / causal masking machinery.** BLIP-2's Q-Former has a dual-purpose design where the text side can generate text autoregressively. This is not needed. Our Q-Former is purely an encoder. There should be no causal attention masks, no text generation heads, no next-token prediction.

2. **Remove all BLIP-2 training objectives.** BLIP-2 uses image-text contrastive loss (ITC), image-text matching loss (ITM), and image-grounded text generation loss (ITG). None of these are used. Training happens entirely through the downstream world model loss.

3. **Replace single cross-attention with dual cross-attention.** In BLIP-2, learned queries cross-attend to image features, and text tokens interact only through shared self-attention layers. In our design, the learned queries have TWO separate cross-attention sub-layers per transformer layer: one attending to visual features and one attending to text features. The text tokens are not processed through the self-attention layers at all — they are only a source for cross-attention.

4. **The number of learned queries should be configurable and smaller.** BLIP-2 uses 32 queries for open-ended image understanding. DMC environments are much simpler. Default to **8 queries** with a configurable parameter. Allow experimentation with 4, 8, 16, and 32.

5. **Output goes to an aggregation head, not to an LLM.** There is no linear projection to an LLM embedding space. Instead, the query outputs feed into an aggregation head (Module 4) that compresses them into a single vector.

### Architecture of a Single Q-Former Layer

Each Q-Former transformer layer consists of the following sub-layers, executed in this exact order:

**Sub-layer 1: Self-Attention among queries.** The learned query embeddings attend to each other. This uses standard multi-head self-attention with a residual connection and LayerNorm. The LayerNorm should be applied before the attention (pre-norm style), consistent with modern transformer conventions. Only the `num_queries` query tokens participate in this attention — no text or visual tokens are present here.

**Sub-layer 2: Cross-Attention to visual features.** The learned queries (as the query/Q input) attend to the visual encoder's output sequence (as the key/K and value/V inputs). This uses standard multi-head cross-attention with a residual connection and pre-LayerNorm. The visual features are `(B, N_visual, d_model)` and the queries are `(B, num_queries, d_model)`.

**Sub-layer 3: Cross-Attention to text features.** The learned queries (as Q) attend to the text encoder's output sequence (as K and V). Same structure as Sub-layer 2 but with text features `(B, N_text, d_model)` as the keys and values.

**Sub-layer 4: Feed-Forward Network (FFN).** A standard two-layer MLP with a GELU activation, applied to each query position independently. Hidden dimension is `4 * d_model` (standard transformer convention). Residual connection and pre-LayerNorm.

### Hyperparameters

These are defined in the central config (see Configuration section) but repeated here for clarity. The lean config optimized for 1M environment steps uses:

- `d_model`: 128 (hidden dimension — halved from original 256 for parameter efficiency)
- `num_heads`: 4 (attention heads — 32 dim/head, the practical minimum for meaningful attention)
- `num_layers`: 2 (Q-Former transformer layers — DMC cross-modal alignment is simple enough for 2 layers)
- `num_queries`: 4 (learned query embeddings — DMC scenes have few task-relevant visual features)
- `ffn_dim`: 256 (FFN hidden dim = 2× d_model — reduced expansion ratio saves parameters)
- `dropout`: 0.0 (no dropout — underfitting is the bottleneck, not overfitting, at 1M steps)

### Learned Query Embeddings

The learned queries are a `nn.Parameter` of shape `(1, num_queries, d_model)` — which with the lean config is `(1, 4, 128)` — initialized from a normal distribution with std=0.02. They are expanded to batch size during the forward pass via `.expand(batch_size, -1, -1)`. These are the "probes" that learn to extract task-relevant multimodal information. They are trainable and updated by the world model loss.

### Interface

Inputs:
- `visual_features`: `(B, N_visual, d_model)` — from the visual encoder (Module 1)
- `text_features`: `(B, N_text, d_model)` — from the text encoder (Module 2)

Output:
- `query_outputs`: `(B, num_queries, d_model)` — the enriched query embeddings after all layers

---

## Module 4: Aggregation Head

### Purpose
Compresses the Q-Former's multiple query outputs into a single fixed-dimensional latent vector suitable for the RSSM.

### Primary Implementation: Learned Attention Pooling

Implement this as the default aggregation strategy. It uses a single learnable "aggregation query" vector that attends to all Q-Former output queries through one multi-head attention layer.

The aggregation query is a `nn.Parameter` of shape `(1, 1, d_model)` — which with the lean config is `(1, 1, 128)` — initialized from a normal distribution with std=0.02. In the forward pass, it is expanded to batch size and used as the Q input to a multi-head cross-attention layer, where the K and V are the Q-Former's output queries `(B, num_queries, d_model)`. The output is a single vector per batch element: `(B, 1, d_model)`, which is squeezed to `(B, d_model)`.

After the attention pooling, apply a final **MLP projection** that maps from `d_model` to the target latent dimension expected by the RSSM. This MLP has two layers: `Linear(d_model, d_model)` → `SiLU` → `Linear(d_model, latent_dim)`. With the lean config, this is `Linear(128, 128)` → `SiLU` → `Linear(128, 512)`. The `latent_dim` should match whatever the R2-Dreamer/R2-Dreamer RSSM expects as its observation embedding dimension. The lean config defaults to 512, which is sufficient for DMC tasks — adjust if R2-Dreamer's RSSM expects a different size.

### Alternative Implementation: Mean Pooling + MLP (Simpler Baseline)

Also implement this as a switchable alternative. Simply average the `num_queries` query outputs along the sequence dimension to get `(B, d_model)`, then pass through the same MLP projection described above. This is simpler and should be available as a baseline for ablation studies.

### Interface

Input: `(B, num_queries, d_model)` — Q-Former output queries
Output: `(B, latent_dim)` — single observation embedding vector for the RSSM

### Configuration

- `aggregation_method`: string, either `"attention_pool"` or `"mean_pool"`, default `"attention_pool"`
- `latent_dim`: int, the output dimension, must match the RSSM's expected observation embedding size

---

## Module 5: Full Multimodal Encoder (Wrapper)

### Purpose
A single wrapper module that combines Modules 1-4 into a clean, unified interface that can be dropped into the R2-Dreamer/R2-Dreamer pipeline as a replacement for the standard visual encoder.

### Forward Pass Logic

```
def forward(self, image_obs, task_text, text_cache=None):
    # Step 1: Encode visual observation
    visual_features = self.visual_encoder(image_obs)        # (B, N_vis, d_model)
    
    # Step 2: Encode text (with caching)
    if text_cache is not None:
        text_features = text_cache
    else:
        text_features = self.text_encoder(task_text)         # (B, N_text, d_model)
    
    # Step 3: Q-Former fusion
    query_outputs = self.qformer(visual_features, text_features)  # (B, num_queries, d_model)
    
    # Step 4: Aggregate to single vector
    obs_embedding = self.aggregation_head(query_outputs)     # (B, latent_dim)
    
    return obs_embedding, text_features  # return text_features for caching
```

The returned `text_features` tensor allows the caller to cache it and pass it back in subsequent timesteps of the same episode, avoiding redundant text encoding.

### Critical Integration Point with R2-Dreamer/R2-Dreamer

In R2-Dreamer's world model, the observation encoder typically takes the raw image and outputs a flat vector that feeds into the RSSM's posterior computation. The `obs_embedding` output of this wrapper must be **dimensionally compatible** with wherever the R2-Dreamer code currently consumes the visual encoder's output. This is typically the input to the posterior net that computes `q(z_t | h_t, o_t)`.

Identify the exact location in the R2-Dreamer codebase where the visual encoder's output is consumed by the RSSM, and ensure `latent_dim` matches that expected input size. This may require inspecting the R2-Dreamer code to find the correct dimensionality.

---

## Module 6: Integration with the RSSM and World Model Training

### Where the Multimodal Encoder Plugs In

In the R2-Dreamer / R2-Dreamer RSSM, the world model has:
- A **deterministic state** `h_t` updated recurrently: `h_t = f(h_{t-1}, z_{t-1}, a_{t-1})`
- A **prior** (imagination): `p(z_t | h_t)` — predicts the stochastic state without seeing the observation
- A **posterior** (inference): `q(z_t | h_t, o_t)` — refines the stochastic state using the observation embedding

The output of our Multimodal Encoder (Module 5) replaces `o_t`. Everywhere the R2-Dreamer codebase feeds a visual encoder's output into the posterior computation, replace it with the Multimodal Encoder's output.

### Training Losses (No Changes Needed to Loss Functions)

The training losses remain exactly as in R2-Dreamer / R2-Dreamer. The Multimodal Encoder is trained purely through backpropagation from these existing losses:

1. **Image reconstruction loss**: The decoder reconstructs the original observation from the latent state `(h_t, z_t)`. Gradients flow back through the RSSM, through the posterior (which depends on `o_t`), through the aggregation head, through Q-Former, and into the CNN visual encoder.

2. **Reward prediction loss**: A reward head predicts the reward from the latent state. Same gradient flow path.

3. **Continue prediction loss**: A head predicts whether the episode continues. Same gradient flow path.

4. **KL divergence loss**: Between the posterior `q(z_t | h_t, o_t)` and the prior `p(z_t | h_t)`, encouraging the prior to be a good predictor.

Do NOT add any BLIP-2-style losses (contrastive, matching, generation). The world model losses are sufficient and appropriate.

### Optional Auxiliary Loss: Task-Contrastive Loss

If training across **multiple tasks** (multi-task RL), optionally add an auxiliary contrastive loss that encourages the fused observation embedding `o_t` to be more similar for observation-description pairs from the same task than for pairs from different tasks. Implement this as a standard InfoNCE / NT-Xent loss applied to the `obs_embedding` output of Module 5, using the task identity as the label. Weight this loss with a small coefficient (e.g., 0.1) relative to the main world model losses. Make this auxiliary loss toggleable via a config flag, default OFF for single-task experiments and ON for multi-task experiments.

---

## Module 7: Text Description Interface for DMC Environments

### Purpose
Provides the textual task descriptions for each DMC environment and task combination.

### Implementation

Create a dictionary/registry that maps DMC `(domain, task)` pairs to textual descriptions. These descriptions should be short, informative sentences that describe the task goal and relevant dynamics.

Example entries:

- `("cartpole", "swingup")`: "Swing up the pole from the bottom position to upright and balance it vertically on the cart."
- `("cartpole", "balance")`: "Keep the pole balanced upright on the cart by moving the cart left and right."
- `("walker", "walk")`: "Make the bipedal walker walk forward as fast as possible while staying upright."
- `("walker", "run")`: "Make the bipedal walker run forward at high speed while maintaining balance."
- `("cheetah", "run")`: "Make the half-cheetah run forward as fast as possible."
- `("reacher", "easy")`: "Move the two-link reacher arm so that the fingertip reaches the target location."
- `("finger", "spin")`: "Rotate the object on the finger by applying torque to the finger joints."
- `("hopper", "hop")`: "Make the one-legged hopper hop forward by jumping and landing repeatedly."
- `("humanoid", "walk")`: "Control the humanoid to walk forward on two legs while maintaining upright posture."
- `("cup", "catch")`: "Swing the ball attached by a string to the cup and catch it inside the cup."

Provide at least 15 entries covering the most commonly used DMC benchmarks. Also provide a fallback that generates a generic description from the domain and task names if a specific entry is not found: `f"In the {domain} environment, accomplish the {task} task successfully."`.

Allow easy extension by the user for custom environments.

---

## Configuration and Hyperparameters: Parameter-Efficient Design for 1M Step Budget

### Why Parameter Efficiency Is Critical

In R2-Dreamer on DMC tasks, 1M environment steps is a moderate but not large training budget. R2-Dreamer's standard visual encoder (CNN) has roughly 500K–1M parameters, and the RSSM plus prediction heads add another 5–10M. This system typically converges within 500K–1M steps because R2-Dreamer uses a replay ratio (multiple gradient updates per environment step). With replay ratio 2, 1M steps gives approximately 2M gradient updates. With replay ratio 4, it gives 4M. The Q-Former module must be sized so that it can converge within this gradient budget without starving the rest of the world model of learning capacity.

The original Q-Former config (d_model=256, 4 layers, 8 heads, 8 queries) would add approximately 3.5–4M trainable parameters — roughly doubling the encoder-side parameter count. This is too large for 1M steps. The Q-Former would still be learning noisy representations during the critical early phase of world model training, producing unstable observation embeddings that slow down the RSSM's convergence. The result would be a system that underperforms plain R2-Dreamer rather than improving on it, because the added capacity hurts more than the text conditioning helps.

The lean configuration below targets approximately 700K–800K total Q-Former-related parameters. This is comparable to the CNN encoder's own parameter count, meaning the encoder side roughly doubles but the total model parameters increase by only 10–15%. This is a manageable overhead that should converge within the 1M step budget.

### Parameter Count Breakdown for the Lean Config

Each Q-Former layer at d_model=128 with 4 heads contains: self-attention with Q/K/V projections and output projection = 4 × (128 × 128) ≈ 65K params, visual cross-attention = another ≈ 65K, text cross-attention = another ≈ 65K, FFN with 2× expansion (128→256→128) = 2 × (128 × 256) ≈ 65K, LayerNorms (4 of them) ≈ 1K. Total per layer ≈ 261K. With 2 layers, the Q-Former core has ≈ 522K parameters. The learned queries add 4 × 128 = 512 parameters. The aggregation head (attention pooling + MLP) adds ≈ 80K. The visual projection (256→128) adds ≈ 33K. The trainable text projection adds ≈ 65K–98K depending on the text encoder's dimension. Grand total for all Q-Former-related additions: approximately 700K–750K trainable parameters.

### Central Config Object

Create a single configuration dataclass/dictionary that holds all hyperparameters for the multimodal encoder. This should be separate from (but compatible with) the R2-Dreamer/R2-Dreamer config. All parameters should have sensible defaults.

```
MultimodalEncoderConfig:
    # Visual Encoder
    visual_channels: [32, 64, 128, 256]       # CNN channel progression (R2-Dreamer convention)
    visual_kernel_size: 4                      # CNN kernel size
    visual_stride: 2                           # CNN stride
    image_size: 64                             # Input image height/width

    # Text Encoder
    text_encoder_name: "openai/clip-vit-base-patch32"  # Pretrained model name
    text_encoder_frozen: True                  # Always True — never unfreeze
    max_text_length: 77                        # Max tokens for text description

    # Q-Former (LEAN CONFIG — optimized for 1M env step budget)
    d_model: 128                               # Hidden dimension (halved from 256 — DMC states are low-dimensional)
    num_heads: 4                               # Attention heads (32 dim per head — minimum for meaningful attention)
    num_layers: 2                              # Transformer layers (halved — DMC cross-modal alignment is simple)
    num_queries: 4                             # Learned query embeddings (halved — DMC scenes have few relevant features)
    ffn_dim: 256                               # FFN hidden dim (2× d_model instead of 4× — less memorization needed)
    dropout: 0.0                               # No dropout (bottleneck is underfitting, not overfitting at 1M steps)

    # Aggregation
    aggregation_method: "attention_pool"        # "attention_pool" or "mean_pool"
    latent_dim: 512                            # Output dimension (match to RSSM input; 512 is sufficient for DMC)

    # Auxiliary Loss
    use_task_contrastive_loss: False            # Enable for multi-task experiments
    contrastive_loss_weight: 0.1               # Weight if enabled

    # Training Dynamics (Q-Former-specific training settings)
    qformer_lr_warmup_steps: 5000              # Linear LR warmup for Q-Former (prevents early instability)
    qformer_lr_scale: 1.0                      # LR multiplier relative to world model LR (1.0 = same LR)
    replay_ratio: 4                            # Gradient updates per env step (increase from default 2 for more training)
    grad_clip_norm: 100.0                      # Global gradient norm clipping (R2-Dreamer default)
```

### Justification for Each Hyperparameter Choice

**d_model = 128.** This is the single most impactful parameter because it affects every linear layer quadratically. DMC environments have low-dimensional underlying state spaces (a cartpole has 4–5 degrees of freedom, a walker has around 20). A 128-dimensional representation is more than sufficient to capture the relevant visual and textual features. BLIP-2 needed 768 dimensions because it was encoding the full richness of natural images with open-vocabulary language — we are encoding simple geometric scenes with short, structured task descriptions.

**num_heads = 4.** With d_model=128 and 4 heads, each attention head operates on 32-dimensional keys/queries/values. This is the standard minimum for attention heads to learn meaningful patterns. Four heads provide enough capacity for the model to attend to different "aspects" of the input (e.g., one head for object identity, one for spatial layout, one for dynamics-relevant features, one for task-goal alignment) without being wastefully large. Going below 4 heads risks the attention layers becoming a bottleneck.

**num_layers = 2.** In BLIP-2, 12 layers were needed because the cross-modal alignment between rich natural images and diverse language is highly complex and requires deep feature refinement. In our DMC setting, the alignment problem is much simpler — we are matching short physics-task descriptions to sparse geometric visuals. Two layers of dual cross-attention provide enough depth for the queries to first gather raw information (layer 1) and then refine and combine it (layer 2). If ablation studies show that 2 layers are insufficient, try 3 before jumping to 4, because each layer adds approximately 260K parameters.

**num_queries = 4.** Each query is an independent information probe that must learn to specialize through training. With limited data (1M steps), more queries means each one gets less gradient signal to specialize, leading to redundancy and wasted capacity. Four queries are enough for DMC environments: intuitively, one query can learn body configuration, one can capture relevant object state (pole angle, ball position, etc.), one can capture velocity/dynamics, and one can capture task-goal alignment from the text. For harder DMC tasks (humanoid), consider increasing to 6. For simple tasks (cartpole), 4 may even be more than needed.

**ffn_dim = 256 (2× d_model).** The standard transformer convention of 4× expansion in the FFN was designed for large language models where the FFN layers store factual knowledge. Our FFN only needs to perform nonlinear feature transformation — it is not a knowledge store. A 2× expansion ratio keeps the FFN expressive enough to learn useful nonlinearities while cutting its parameter count in half. This saves approximately 65K parameters per layer (130K total across 2 layers).

**dropout = 0.0.** In the RL setting with a 1M step budget, the bottleneck is underfitting (learning fast enough), not overfitting (memorizing the training data). The world model receives new experience continuously as the policy explores, so the data distribution is non-stationary and inherently regularizing. Dropout actively slows down learning by randomly zeroing activations during training. R2-Dreamer itself uses minimal or no dropout in its encoder and RSSM. If ablation studies show overfitting (training loss decreasing but evaluation performance plateauing or degrading), add dropout=0.05 as a light regularizer, but start at 0.0 for maximum learning speed.

**latent_dim = 512.** The original config proposed 1024, but this is unnecessarily large for DMC. The underlying state of most DMC tasks can be described by fewer than 30 continuous variables. A 512-dimensional latent vector provides ample capacity to represent these states with rich distributed encodings, while reducing the parameter count of the aggregation MLP and the downstream RSSM input projection. If the R2-Dreamer codebase hardcodes an expectation for a specific latent_dim, match that value instead.

**qformer_lr_warmup_steps = 5000.** During the first 5K gradient steps (approximately 1250–2500 environment steps depending on replay ratio), Q-Former's learning rate ramps linearly from 0 to the target rate. This prevents the randomly initialized cross-attention layers from producing large, noisy gradients in the early phase of training that could destabilize the CNN encoder (which is also learning from scratch) and the RSSM. After 5K steps, Q-Former operates at full learning rate. This warmup period is less than 1% of the total gradient budget and does not meaningfully slow convergence.

**qformer_lr_scale = 1.0.** Q-Former should use the same learning rate as the rest of the world model (typically 1e-4 with Adam in R2-Dreamer). A common instinct is to use a lower LR for the new module, but this would slow convergence during a training regime where we cannot afford slow convergence. Unlike fine-tuning scenarios where a lower LR protects pretrained features, Q-Former is initialized randomly and needs to learn from scratch — it benefits from aggressive optimization. The warmup (above) provides the necessary early-phase stability without permanently reducing the LR.

**replay_ratio = 4.** This is the number of gradient updates performed on the world model per environment step. R2-Dreamer defaults to 2 for many DMC tasks and scales up for harder tasks. Increasing to 4 doubles the number of gradient updates Q-Former receives from the same 1M environment steps (giving approximately 4M total gradient updates). This is one of the most effective ways to compensate for the added parameters without collecting more experience. The risk of higher replay ratio is overfitting to stale experience, but with Q-Former's lean size (700K params) and a continuously growing replay buffer, this is unlikely within 1M steps. Monitor the world model's reconstruction loss on recent (not replayed) experience to detect staleness.

### Ablation Configurations

For systematic evaluation, implement the following configuration variants:

**Tiny config (for quick debugging and sanity checks):** d_model=64, num_heads=2, num_layers=1, num_queries=2, ffn_dim=128. Approximately 100K Q-Former parameters. Useful for verifying the pipeline works end-to-end before committing to longer runs.

**Lean config (primary — described above):** d_model=128, num_heads=4, num_layers=2, num_queries=4, ffn_dim=256. Approximately 700K Q-Former parameters. This is the recommended starting point for 1M step experiments.

**Medium config (if 1M steps proves sufficient and you want more capacity):** d_model=192, num_heads=4, num_layers=3, num_queries=6, ffn_dim=384. Approximately 1.8M Q-Former parameters. Only use this if the lean config shows clear signs of underfitting (training loss plateaus well above baseline R2-Dreamer's loss) AND you have computational budget for 1M+ steps.

Do NOT use the original large config (d_model=256, num_layers=4, num_queries=8) unless you increase the training budget to at least 3M environment steps.

---

## Training Dynamics and Schedule for 1M Step Budget

### Learning Rate Strategy

All trainable parameters (CNN encoder, Q-Former, aggregation head, projections) use the same base learning rate as the R2-Dreamer world model, which is typically 1e-4 with Adam optimizer (or 3e-4 with AdamW, depending on the R2-Dreamer variant). Do NOT use a separate, lower learning rate for Q-Former. The reasoning is that Q-Former is initialized randomly and must learn from scratch — unlike fine-tuning scenarios where a reduced LR protects pretrained knowledge, here a full-strength LR is needed for fast convergence.

However, implement a **linear warmup specifically for Q-Former's parameters** during the first 5000 gradient steps. During this period, Q-Former's effective learning rate ramps from 0 to the base LR. This is implemented as a separate parameter group in the optimizer with a LR scheduler that starts at 0 and linearly increases. The warmup prevents early-phase instability: Q-Former's randomly initialized cross-attention weights can produce large, noisy gradients that destabilize the co-adapting CNN encoder and RSSM. After warmup, all parameters train at the same rate. Implementation: use two parameter groups in the Adam/AdamW optimizer — one for Q-Former params with the warmup schedule, and one for all other trainable params with the standard R2-Dreamer schedule.

### Replay Ratio

Set the replay ratio to 4 (gradient updates per environment step). This means 1M environment steps yields approximately 4M gradient updates on the world model. R2-Dreamer defaults to replay ratio 2 for many DMC tasks, but the added Q-Former parameters benefit from the extra training. This is the most direct way to compensate for more parameters without collecting more experience. The computational cost per environment step doubles (compared to replay ratio 2), but the wall-clock impact is modest because environment stepping (physics simulation + rendering) is often the bottleneck, not gradient computation.

Monitor for **stale replay** by periodically logging the world model's reconstruction loss on the most recent 100 transitions (not sampled from the replay buffer). If this loss diverges from the replay-sampled loss, the replay ratio may be too high. In that case, reduce to 3. This is unlikely to be a problem at replay ratio 4 with a growing buffer, but it is worth monitoring.

### Gradient Clipping

Use global gradient norm clipping at 100.0 (R2-Dreamer default). Q-Former's parameters must be included in the global norm computation — do NOT clip Q-Former's gradients separately. The cross-attention layers can occasionally produce large gradients, especially in the early training phase when visual and text features are not yet aligned. The global norm clip ensures that no single large gradient from Q-Former overwhelms the rest of the model's update.

### Expected Training Timeline within 1M Steps

At replay ratio 4, the gradient update budget is approximately 4M steps. Here is the expected progression of learning:

Steps 0–5K (warmup phase, ~0.5% of budget): Q-Former's learning rate ramps up. During this phase, Q-Former outputs are essentially random projections of the visual features, and the RSSM learns primarily from the CNN's raw features passed through the random Q-Former. World model loss should decrease slowly. This is normal and expected.

Steps 5K–50K (~5% of budget): Q-Former begins producing meaningful fused representations. The cross-attention layers start to align visual features with relevant text semantics. The world model loss should begin decreasing more rapidly than baseline R2-Dreamer, because the text conditioning helps the model identify task-relevant features earlier.

Steps 50K–200K (~20% of budget): The core learning phase. Q-Former's queries should have specialized by now. Monitor the attention weights (log them periodically) to verify that different queries attend to different spatial regions and different text tokens. The world model should produce noticeably better predictions than at step 50K.

Steps 200K–1M (remaining 80%): Refinement phase. World model loss should plateau and then slowly decrease. The agent's policy (trained via imagination in latent space) should achieve good task performance. If the world model loss has not decreased meaningfully by step 200K, something is wrong — check gradient flow, learning rates, and verify that text features are actually reaching Q-Former (a common bug is accidentally passing zeros or cached stale features).

### Diagnostic Logging

Implement the following logs to diagnose training health, sampled every 1K gradient steps:

Log the **mean and variance of Q-Former's output** (the query outputs before aggregation). If the variance collapses to near-zero, the queries are not diversifying and the model is effectively ignoring the Q-Former (a mode collapse issue). If this happens, try increasing the learning rate or reducing weight decay.

Log the **cross-attention weights** for both visual and text cross-attention. Specifically, for each query, log which visual positions and which text tokens receive the highest attention. In a healthy model, different queries should attend to different regions. If all queries attend to the same positions, the model has redundant queries and fewer would suffice.

Log the **norm of gradients** flowing into Q-Former vs. the CNN encoder. If Q-Former's gradient norms are orders of magnitude larger or smaller than the CNN's, there is a scaling mismatch that may require adjusting the architecture (e.g., adding gradient scaling or adjusting LayerNorm placement).

Log the **text projection output norm**. Since the text encoder is frozen and the text is static within an episode, the text projection output should stabilize to a consistent norm after the warmup phase. If it keeps fluctuating wildly throughout training, the text projection layer may be unstable.

---

## File Structure

Organize the code as follows:

```
multimodal_encoder/
├── __init__.py                    # Exports MultimodalEncoder and config
├── config.py                      # MultimodalEncoderConfig dataclass
├── visual_encoder.py              # Module 1: Trainable CNN
├── text_encoder.py                # Module 2: Frozen pretrained text encoder with caching
├── qformer.py                     # Module 3: Modified Q-Former (layers, dual cross-attention)
├── aggregation.py                 # Module 4: Attention pooling and mean pooling heads
├── multimodal_encoder.py          # Module 5: Full wrapper combining all modules
├── task_descriptions.py           # Module 7: DMC task description registry
└── utils.py                       # Shared utilities (weight init, etc.)
```

Each file should contain one primary class. All modules inherit from `torch.nn.Module`. Use PyTorch throughout.

---

## Implementation Details and Constraints

### Weight Initialization
All trainable linear layers and projection layers should use Xavier uniform initialization. All bias terms should be initialized to zero. The learned query embeddings and aggregation query should use normal initialization with std=0.02. LayerNorm parameters should use default PyTorch initialization (weight=1, bias=0).

### Attention Implementation
Use `torch.nn.MultiheadAttention` for all attention layers (self-attention and cross-attention). Set `batch_first=True` for consistency. For cross-attention calls, the `query` argument receives the learned queries, and the `key` and `value` arguments receive the source features (visual or text).

### Gradient Flow
Ensure that `requires_grad=True` for ALL parameters EXCEPT the text encoder's parameters. Double-check that the frozen text encoder does not accidentally accumulate gradients by wrapping its forward pass in `torch.no_grad()`.

### Sequence Length Tracking
For debugging and logging, each module's forward method should optionally print or log the tensor shapes at input and output. This is very helpful during integration. Implement this as a `verbose` flag in the config.

### Device and Dtype Handling
All modules should respect the device and dtype of their input tensors. Use `.to(device)` during initialization. The text encoder may internally use float32 even if the rest of the pipeline uses float16/bfloat16 — handle this with an explicit cast after the text encoding step.

---

## Integration Checklist for R2-Dreamer

When integrating this multimodal encoder into the R2-Dreamer codebase, follow these steps in order:

1. **Identify the observation encoder** in R2-Dreamer's code. This is the module that takes raw image observations and outputs the embedding that feeds into the RSSM. Note its output dimensionality — this is what `latent_dim` must match.

2. **Replace the observation encoder** with the MultimodalEncoder wrapper (Module 5). Ensure the `latent_dim` config matches the RSSM's expected input size.

3. **Modify the environment wrapper** to also return the task description string alongside the image observation. Use the task description registry (Module 7) to look up the description based on the DMC domain and task.

4. **Modify the data collection / replay buffer** to store the task description string (or a task ID that can be looked up). Since the text is constant within an episode, it only needs to be stored once per episode, not per timestep.

5. **Modify the training loop** to pass both the image batch and the corresponding task description batch to the MultimodalEncoder. Implement text caching so that within a trajectory sequence, the text is encoded only once.

6. **Verify end-to-end training** by running a short training run (e.g., 1000 steps) on a simple DMC task (cartpole-balance) and checking that the world model loss decreases.

7. **Run baseline comparison**: Train the original R2-Dreamer (vision-only) and the modified version (with Q-Former multimodal encoder) on the same task with the same seed, and compare learning curves.

---

## Summary of Architectural Decisions

To recapitulate the key decisions that differ from BLIP-2 and why:

The visual encoder is **trainable** (not frozen) because DMC visuals are domain-specific and a pretrained model would be poorly suited. The text encoder is **frozen** because natural language understanding is already solved by pretrained models and we want to avoid overfitting the text representations to a small RL dataset. Q-Former uses **dual cross-attention** (one for vision, one for text per layer) instead of BLIP-2's single cross-attention plus shared self-attention, because we want the queries to actively and independently extract from both modalities. The number of queries is **reduced to 4** (from BLIP-2's 32) because DMC environments are visually simple and because each query needs sufficient gradient signal to specialize within the 1M step budget. The hidden dimension is **reduced to 128** (from BLIP-2's 768) because DMC state spaces are low-dimensional, and this single change roughly quarters the per-layer parameter count. The depth is **reduced to 2 layers** (from BLIP-2's 12) because the cross-modal alignment in DMC is far simpler than natural image to open-vocabulary language. The FFN expansion ratio is **reduced to 2×** (from 4×) because the FFN does not need to store factual knowledge. Dropout is **set to 0.0** because the training bottleneck at 1M steps is underfitting, not overfitting. The output uses **learned attention pooling** to compress query outputs into a single vector, because the RSSM expects a fixed-dimensional vector rather than a variable-length sequence. All **BLIP-2-specific training losses are removed** and replaced entirely by end-to-end world model training through R2-Dreamer's existing objectives. A **learning rate warmup of 5K gradient steps** is added specifically for Q-Former to prevent early-phase instability. The **replay ratio is increased to 4** to give Q-Former more gradient updates from the same 1M environment steps. The total Q-Former parameter overhead is approximately **700K–750K**, which is manageable within the 1M step budget and represents only a 10–15% increase in total model parameters.
