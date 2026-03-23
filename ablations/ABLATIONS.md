# Ablation Studies

This document describes the ablation experiments used to evaluate the contribution of each component in our multimodal world model architecture. All experiments run on the **distract_cheetah_run** environment (DMC Cheetah with dynamic DAVIS background videos).

---

## Overview

The full multimodal architecture combines three key ingredients:

1. **FiLM conditioning** — text modulates CNN encoder features via Feature-wise Linear Modulation at each convolutional layer.
2. **TextGate** — a learned gating mechanism that blends visual embeddings with projected text context before feeding into the RSSM world model.
3. **CLIP text embeddings** — frozen `openai/clip-vit-base-patch32` encodes natural-language task descriptions into a 256-dim context vector.

The ablations are organized into four groups, each targeting a different research question.

---

## A — Component Isolation (Train from scratch, 1.01M steps)

These ablations systematically remove architectural components to measure the contribution of each.

### A5: CNN Baseline

**Config:** `a5_cnn_baseline.yaml`

> **Question:** How well does a standard vision-only model handle visual distractors?

Pure CNN encoder with no text conditioning, no FiLM, and no TextGate. This is the reference baseline against which all other ablations are compared. Uses encoder depth 16 with augmentation disabled.

---

### A2: FiLM Only (No TextGate)

**Config:** `a2_film_only.yaml`

> **Question:** Is FiLM conditioning in the CNN layers sufficient on its own, or does the model also need text blended into the RSSM input?

Uses real CLIP text embeddings and FiLM-conditioned CNN layers but **disables the TextGate**. The RSSM receives only the visual embedding. This isolates the contribution of visual-level text modulation without any text signal reaching the dynamics model.

---

### A3: Gate Only (No FiLM)

**Config:** `a3_gate_only.yaml`

> **Question:** Can text injection at the RSSM input alone match the performance of FiLM conditioning in the CNN?

Uses a standard CNN encoder (identical to A5, no FiLM layers) but adds a **TextGate** that blends the CNN output with real CLIP text before the RSSM. This tests whether modulating the dynamics model input is sufficient without modulating intermediate visual features.

---

### A4: Full Multimodal

**Config:** `a4_full_multimodal.yaml`

> **Question:** Do FiLM + TextGate + semantic text together provide the best performance?

The complete proposed architecture: FiLM-conditioned CNN + TextGate + real CLIP text embeddings. The gate is initialized with a bias of -3 (sigmoid(-3) ≈ 4.7% text influence), allowing the model to gradually learn how much text to incorporate. This is the main result.

---

### Summary Table — Component Isolation

| Ablation | FiLM | TextGate | Text Source | Tests |
|----------|------|----------|-------------|-------|
| A5 — CNN Baseline | No | No | None | Pure vision reference |
| A2 — FiLM Only | Yes | No | Real CLIP | FiLM alone |
| A3 — Gate Only | No | Yes | Real CLIP | Gate alone |
| A4 — Full Multimodal | Yes | Yes | Real CLIP | Complete system |

---

## B — Text-Swap Evaluation (Eval-time only, no retraining)

These ablations test whether the trained multimodal model genuinely uses text semantics by **swapping text at test time only**. The model is trained once with correct task descriptions (the `distractor_multimodal` run), then evaluated with different text conditions over **30 episodes** each.

**Key insight:** If the model learned a genuine dependency on text semantics during training, swapping to wrong/random/zero text at eval time should disrupt FiLM modulation patterns and hurt performance. If performance stays the same, the text pathway was not being used.

**Script:** `ablations/eval_text_swap.py`
**Run:** `bash ablations/run_text_swap_eval.sh`

### Conditions

| Condition | Text Source | What It Tests |
|-----------|-----------|---------------|
| `real_text` | Correct task descriptions | Baseline (normal operation) |
| `adversarial` | Opposite instructions ("focus on background, ignore agent") | Does the model follow text semantics? |
| `nonsense` | Shuffled words from real descriptions | Are CLIP semantics needed, or just any embedding? |
| `random_vector` | Fixed random 256-dim vector (bypasses CLIP entirely) | Does the architecture use text at all? |
| `zero_vector` | All-zeros context vector | What happens with no text signal? |

### Interpretation

- **Significant drop with adversarial text:** Model genuinely uses semantic content — FiLM learned to associate correct descriptions with specific channel modulations
- **Drop with nonsense but not random_vector:** Model relies on CLIP embeddings being in-distribution, but not on their semantic content
- **Drop with zero_vector:** Model developed a dependency on *some* text signal (even if not semantic)
- **No drop for any condition:** Text pathway is not being used despite being present in the architecture

---

## F — Difficulty Sweep (Train from scratch, 1.01M steps)

These ablations compare the CNN baseline and full multimodal architecture across **three distractor difficulty levels** (easy, medium, hard) to test how each scales under increasing visual complexity.

### F1-CNN-Medium / F1-CNN-Hard

**Configs:** `f1_cnn_medium.yaml`, `f1_cnn_hard.yaml`

> **Question:** How quickly does the vision-only CNN degrade as distractors become more challenging?

Standard CNN baseline (same as A5) run at medium and hard difficulty levels. Combined with A5 (easy), these form the CNN difficulty curve.

---

### F1-Multimodal-Medium / F1-Multimodal-Hard

**Configs:** `f1_multimodal_medium.yaml`, `f1_multimodal_hard.yaml`

> **Question:** Does text-conditioned multimodal encoding maintain performance under severe visual distractions where pure CNN fails?

Full multimodal architecture (same as A4) run at medium and hard difficulty levels. If the multimodal model degrades more gracefully than the CNN, it demonstrates that text guidance helps the agent remain robust to increasingly complex distractors.

---

### Summary Table — Difficulty Sweep

| Difficulty | CNN Config | Multimodal Config |
|-----------|-----------|------------------|
| Easy | A5 (baseline) | A4 (baseline) |
| Medium | f1_cnn_medium | f1_multimodal_medium |
| Hard | f1_cnn_hard | f1_multimodal_hard |

---

## H — Parameter Matching (Train from scratch, 1.01M steps)

### H3: Wider CNN Baseline

**Config:** `h3_wider_cnn.yaml`

> **Question:** Does the multimodal model outperform the CNN because of text semantics, or simply because it has more parameters?

A pure CNN encoder with **depth increased to 77** (channel widths: 154→231→308→308), yielding ~5.05M encoder parameters — matching the ~4.97M parameters of the full multimodal encoder. No text, no FiLM, no TextGate. If this wider CNN still underperforms the multimodal model, the benefit comes from text semantics and the multimodal architecture, not raw model capacity.

---

### Summary Table — Parameter Matching

| Ablation | Encoder Params | Architecture | Text |
|----------|---------------|-------------|------|
| A5 — CNN (depth 16) | ~0.13M | Standard CNN | None |
| H3 — Wider CNN (depth 77) | ~5.05M | Wide CNN | None |
| A4 — Full Multimodal | ~4.97M | FiLM + Gate + CNN | Real CLIP |

---

## Full Ablation Summary

### Training ablations (train from scratch)

| ID | Config | Architecture | FiLM | Gate | Text Type | Research Question |
|----|--------|-------------|------|------|-----------|-------------------|
| A5 | `a5_cnn_baseline` | CNN | — | — | None | Vision-only baseline |
| A2 | `a2_film_only` | CNN+FiLM | Yes | — | Real CLIP | Is FiLM alone sufficient? |
| A3 | `a3_gate_only` | CNN+Gate | — | Yes | Real CLIP | Is Gate alone sufficient? |
| A4 | `a4_full_multimodal` | CNN+FiLM+Gate | Yes | Yes | Real CLIP | Full system (proposed method) |
| F1-CNN-Med | `f1_cnn_medium` | CNN | — | — | None | CNN at medium difficulty |
| F1-CNN-Hard | `f1_cnn_hard` | CNN | — | — | None | CNN at hard difficulty |
| F1-MM-Med | `f1_multimodal_medium` | CNN+FiLM+Gate | Yes | Yes | Real CLIP | Multimodal at medium difficulty |
| F1-MM-Hard | `f1_multimodal_hard` | CNN+FiLM+Gate | Yes | Yes | Real CLIP | Multimodal at hard difficulty |
| H3 | `h3_wider_cnn` | Wide CNN | — | — | None | Parameters vs. semantics |

### Eval-time text-swap ablations (no retraining)

| Condition | Checkpoint | Text at Eval | Research Question |
|-----------|-----------|-------------|-------------------|
| real_text | distractor_multimodal | Correct descriptions | Baseline performance |
| adversarial | distractor_multimodal | Opposite instructions | Does model follow text? |
| nonsense | distractor_multimodal | Shuffled words | Does semantics matter? |
| random_vector | distractor_multimodal | Random 256-dim vector | Is text used at all? |
| zero_vector | distractor_multimodal | All-zeros vector | No text signal |

---

## Running

### Training ablations

```bash
bash ablations/run_ablations.sh               # all training ablations
bash ablations/run_ablations.sh a5 h3          # specific ones
```

### Text-swap eval ablation

```bash
bash ablations/run_text_swap_eval.sh                           # default checkpoint
bash ablations/run_text_swap_eval.sh /path/to/checkpoint.pt    # custom checkpoint
```

### Plotting

```bash
python ablations/plot_ablation_results.py \
    --base_logdir /nfs-stor/salem.lahlou/asharma/logdir/ablations \
    --output_dir ablations/results
```

Gate analysis:
```bash
python ablations/plot_gate_analysis.py \
    --logdirs /nfs-stor/salem.lahlou/asharma/logdir/ablations/ablation_a4_full_multimodal/distract_cheetah_run \
    --labels 'Full Multimodal' \
    --output ablations/results/e3_gate_analysis.pdf
```
