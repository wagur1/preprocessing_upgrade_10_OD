# MODEL.md — architecture & training in detail

This document specifies the upgrade-3 system end to end: the trained preprocessor,
the frozen codec proxies, the frozen analyzer panel, the composite loss, and the
train/eval control flow. It is the reference companion to `README.md` (overview)
and `docs/IMPROVEMENTS.md` (line-by-line delta vs. the baseline and upgrade-2).

> **One trainable module.** Only `VideoPreprocessor` has gradients. Every teacher
> analyzer and every codec (proxy or real x264/x265) is frozen. The claim we
> optimize is `BD-Rate(prep+codec vs codec)` on the *same* frozen codec.

---

## 0. Notation

| Symbol | Shape | Meaning |
|---|---|---|
| `x` | `[B,C,T,H,W]` in [0,1] | source clip |
| `cond` | `[B,1]` | normalized operating point (QP → [0,1]) fed to FiLM |
| `x_pre` | `[B,C,T,H,W]` | preprocessor output (the edited pixels) |
| `x̂` | `[B,C,T,H,W]` | codec reconstruction |
| `bpp` | scalar | bits-per-pixel (virtual proxy estimate; real coded for H.264/H.265) |
| `m` | `[B,1,T,H,W]` in [0,1] | A2 task-saliency mask (1 = task-critical) |

## 1. The preprocessor `VideoPreprocessor` (`src/models/preprocessor.py`)

A rate- and motion-conditioned **U-Net residual editor** applied frame-wise.

```
x [B,C,T,H,W] ──► motion cue (temporal |Δ|, per-clip normalized) ─┐
        │                                                          │ SFT
        └─ fold to [B*T,C,H,W] ─► 3-level U-Net ──────────────────►┤ FiLM(cond)
                                   enc1/enc2/bott/dec2/dec1 + skips │
                                                                   ▼
                            delta ─► x_pre = clamp(x + res_scale·delta, 0,1)
```

* **Backbone.** 3-level U-Net (full, /2, /4) of `_ConvBlock`s (conv-act-conv).
  Width `base_ch=32` at full res, doubling per level. Skip-concatenated decoder.
* **FiLM (rate)** — Perez et al. 2018. A global per-channel affine
  `x·(1+γ)+β` computed from `cond` (normalized QP). Lets **one** model span the
  whole rate range instead of learning a blurry rate-average (upgrade-2 fix over
  the baseline's two-branch stack).
* **SFT (motion)** — Wang et al. CVPR 2018. A *spatially-varying* affine computed
  from the temporal frame-difference cue, so edits are steered toward moving /
  task-relevant regions. This is the video-domain novelty over image preprocessors.
* **Zero-init identity start.** The U-Net tail and both modulators' final layers
  are zero-initialized ⇒ the network begins as an exact identity (`delta≈0`) and
  only switches on edits as it learns. Stable early training.
* **Bounded relative edit.** `max_relative_edit=0.25` means one pass can consume
  at most 25% of a pixel's distance toward black or white. This prevents the
  trivial all-black bitrate minimum while preserving a differentiable editor.
* **Temporal handling.** Editing is 2D per-frame (cheap on a T4); the *only*
  temporal signal is the motion cue driving SFT. Temporal *coherence* of the edit
  is enforced by `L_temp` (loss), not by 3D convs.

The checkpoint stores **only** these weights, so it is codec-agnostic: the same
`preprocessor.pth` is evaluated against compressai / x264 / x265 by swapping the
codec at eval time.

---

## 2. Codec proxies (all frozen)

### 2.1 `VirtualCodec` — differentiable block-DCT proxy (`src/models/virtual_codec.py`)

Matches x264/x265 **geometry** (the prime suspect for upgrade-2's poor transfer,
where CompressAI's learned wavelet transform did not):

```
[C1, default]  RGB ─► BT.601 YCbCr ─► chroma 2×2 downsample
  P-frame residual per plane (pred = previous RECONSTRUCTED plane; frame 0 = intra)
  r ─► block DCT (bs×bs orthonormal) ─► coeffs
  coeffs / step(quality) ─► y ─► quantize ─► ŷ        (chroma step × chroma_step_scale)
  rate = Σ 0.5·log2(1 + 12·E[y²])          (factorised per-frequency, parameter-free)
  ŷ·step ─► inverse block DCT ─► r̂ ─► x̂ = pred + r̂
  chroma upsample ─► YCbCr → RGB
```

* **C1 yuv420 colourspace (5.1 default).** Every `-pix_fmt yuv420p` encode halves
  chroma resolution at *any* QP and quantises chroma coarser (the H.26x chroma QP
  offset ≈ +6 QP at QP50, `chroma_step_scale=2.0`). The proxy reproduces this
  geometry with differentiable ops (`src/models/color.py`), so the training
  rate/distortion gradients match the deployment codec's colourspace damage —
  chroma-heavy edits finally cost what the real codec charges. `colorspace: rgb`
  keeps the legacy single-plane path for ablation.
* **Quantizer step** is the rate knob: higher `quality` id → finer step → more
  bits. Steps geometrically interpolate `step_coarse → step_fine` across ids, or
  are set explicitly via `q_steps` (physical calibration to overlap the x264/x265
  bpp range).
* **A3 soft→hard anneal.** `set_anneal(a)` blends the training quantizer:
  `a=0` = additive-uniform-noise (soft, upgrade-2 default); `a=1` = straight-through
  hard rounding `ŷ = y + (round(y)−y).detach()`. The engine ramps `a: 0→1` over
  training so the proxy *ends* at the codec's real (hard) quantizer, narrowing the
  train/test quantization gap (cf. J4D soft-quantizer α→∞, arXiv:2606.16185).
* **Honest bitrate never comes from here** — this module only supplies the
  differentiable rate+distortion signal at train time. Eval bpp is always the real
  x264/x265 coded size.

### 2.2 `CompressAICodec` (`src/models/codec.py`)

Learned `bmshj2018-factorized` proxy, frozen. Alternative training codec and one
of the six eval pipelines. Kept for continuity with upgrade-2 and as the in-domain
learned-codec reference.

### 2.3 `STECodec` — A3 straight-through real codec (`src/models/ste_codec.py`)

Wraps a differentiable proxy and the real ffmpeg codec:

```
forward:  x_p, bpp_p = proxy(x, quality)          # differentiable
          x_r, bpp_r = real_x26x(x)               # ffmpeg, no_grad
          x̂  = x_p + (x_r  - x_p ).detach()        # VALUE = real,  GRAD = proxy
          bpp = bpp_p + (bpp_r - bpp_p).detach()
```

So the optimizer's loss is computed on the **true** reconstruction and **true**
coded rate, while gradients still reach the preprocessor through the proxy. This
is the single most reliable proxy→real transfer recipe in the literature: Lu et al.
(arXiv:2206.05650 / TCSVT 2024) measured forward-real-codec at **−20.3 %** BD-Rate
vs **−14.6 %** for a proxy used in both directions. ffmpeg-per-step is ~10–50× slower
⇒ used as a **short stage-2 calibration fine-tune** on top of a proxy-pretrained
checkpoint, not from scratch. `quality→qp` inversion lets it slot into the engine's
existing `codec(x, quality)` call unchanged.

## 3. Analyzer panel (frozen) — A1 (`src/tasks/`)

`build_analyzer(cfg, role)` (`src/tasks/base.py`) constructs the frozen analyzer(s):

* **`role="train"`** — if `task.teachers` lists >1 backbone, returns a
  `MultiTeacherAnalyzer` (`src/tasks/multi_teacher.py`) wrapping a `ModuleList` of
  frozen teachers; else a single analyzer.
* **`role="eval"`** — if `eval.held_out_backbone` is set, builds *that* single
  **held-out** analyzer (the A1 generalization test); else the primary backbone
  (in-domain eval).

`MultiTeacherAnalyzer.accuracy_loss`:

* `sampling="sample"` — draw one teacher per step (sets `self._active`), so the
  edit cannot specialize to one network's quirks — a stochastic-multi-teacher
  regularizer at ~1× analyzer cost.
* `sampling="mean"` — average the task loss over all teachers (weighted by
  `teacher_weights`), Nx cost, lower variance.

`features()` follows the *active* teacher so feature distillation stays coherent
within a step; `predict()` delegates to teacher 0. Aggregation follows multi-teacher
distillation (arXiv:2510.18680) and the feature-modulation multi-task preprocessor
of Yang et al. (TCSVT 2024, DOI 10.1109/TCSVT.2023.3348995).

**Tasks.** `action_recognition` — Kinetics-400 torchvision video ResNets
(`r3d_18`, `mc3_18`, `r2plus1d_18`), cross-entropy `L_task`, top-1 at eval.
`tracking` — GOT-10k, SiamFC logistic `L_task`, success-plot AUC at eval (paper's
KYS/DiMP/ATOM/PrDiMP via `pytracking`).

---

## 4. Task-importance mask — A2 (`src/models/task_mask.py`)

`task_saliency(analyzer, x, target)` produces a detached importance map:

```
m(x) = normalize_per_clip( box_blur( mean_C |∂L_task/∂x| ) )   ∈ [0,1], detached
```

One extra backward through the frozen teacher, channel-reduced, box-blurred into
coherent regions, per-clip min-max normalized, then **detached** (no second-order
gradient into the preprocessor — stable and cheap). `masked_tv(x, w)` is a weighted
total variation. Feeding `w = 1 − m` **spatially reweights** the edit (`delta`) and
TV (`gamma`) penalties so the preprocessor smooths and stops spending bits on
**background** while sparing the object the machine needs. This turns upgrade-2's
*global* in-domain↔transfer knob into a *spatial* one — the pixel-domain analogue
of task-driven bit allocation (Reinforced Bit Allocation arXiv:1910.07392;
feature-preserving RDO arXiv:2504.02216; ROI retargeting EURASIP JIVP 2025) without
touching the encoder.

---

## 5. Composite loss (`src/losses.py`)

```
L = λ_task·L_task + ω·L_distill + β·bpp + τ·L_temp
      (+ δ·L_delta) (+ γ·L_tv)  ← A2: weighted by (1−m) when use_task_mask
      (+ μ·L_D)                 ← optional Zhao-et-al. MSE-to-source
```

| Term | Weight (headline cfg) | Role |
|---|---|---|
| `L_task` | `λ_task=1.0` | frozen-analyzer accuracy loss on `x̂` |
| `L_distill` | `ω=0.5` | scale-normalized MSE of analyzer features (source vs `x̂`); preserves semantics the codec would destroy |
| `bpp` | `β=0.01` | light proxy-rate pressure |
| `L_temp` | `τ=0.1` | match inter-frame Δ of `x̂` to source (kills flicker, no pixel-pinning) |
| `L_delta` | `δ=0.02` | L1 edit magnitude `|x_pre−x|`, masked by `(1−m)` → background-only |
| `L_tv` | `γ=0.01` | conservative masked TV of `x_pre` → smooth background |
| `L_D` | `μ=10.0` | reconstruction fidelity required by the headline virtual/STE path |

Design choice inherited from upgrade-2: **no MSE-to-source by default**. That term
pinned reconstruction to source pixels and fought compression against a mismatched
proxy. It replaces pixel fidelity with *task-aligned* feature distillation and lets
a real rate weight bite (Yang et al. TCSVT 2024). `L_D` remains available (`μ`) for
the block-DCT proxy where pinning pixels is in the *same* domain as x264/x265.

## 6. Training loop (`src/engine.py::_fit`)

Per step:

1. Sample a QP from `train.qp_list`; map to proxy quality via `train.qp_to_quality`
   (validated monotonic: higher QP → lower quality).
2. Build `cond = qp_norm(qp)` and feed the preprocessor's FiLM (so one model spans
   the rate range; eval reuses the *same* per-quality cond via `_quality_conds`).
3. **A3 anneal:** `set_anneal(anneal_final · min(1, step/total_steps))` on the proxy.
4. **A2 mask:** `mask = task_saliency(analyzer, clips, target)` if `use_task_mask`.
5. `x_pre = pre(clips, cond)` → `x̂, bpp = codec(x_pre, q)` → `preprocessing_loss(...)`.
6. Adam step; cosine LR; per-epoch val loss (proxy-only, fixed mid QP) drives
   best-checkpoint saving + early stopping; resume from `preprocessor_last.pth`.

**Two-stage recipe.** Stage 1 = fast proxy pretrain (`codec.kind=virtual`,
`anneal=1.0`). Stage 2 = short real-codec calibration (`codec.kind=ste`,
`ste_codec=h265`, `resume=true`, small LR) — a few hundred ffmpeg-in-the-loop steps.

---

## 7. Evaluation (`src/engine.py::evaluate`)

Builds models with `role="eval"` (held-out analyzer). By default it traces
`prep+{x264,x265}` vs bare `{x264,x265}` over the eval QP grid, then computes:

* **`bd_prep_gain`** — same-codec preprocessor gain (`prep+x265 vs x265`, …).
  **The real claim.** Negative = fewer bits at equal machine accuracy.
* `bd_vs_anchor` — cross-codec (reference only; QP not comparable across codecs).

`src/metrics/bd_rate.py` integrates `log(rate)` over the overlapping accuracy range
(Bjøntegaard) with **machine accuracy as the quality axis** (top-1 / AUC) instead
of PSNR. Outputs: `results.json`, `curves.csv`, `rate_accuracy.png`, `qualitative.png`.
Set `eval.include_proxy=true` to add virtual/CompressAI diagnostic curves.

---

## 8. Self-checks

Every non-trivial module ships a `__main__` self-check (no dataset/GPU needed):

```bash
python -m src.models.virtual_codec     # DCT orthonormal, rate monotone, anneal finite
python -m src.models.task_mask         # saliency in [0,1], masked_tv differentiable
python -m src.models.ste_codec         # STE identity (real value / proxy grad); needs ffmpeg
python -m src.tasks.multi_teacher      # sample/mean aggregation, freeze, feature coherence
python -m src.metrics.bd_rate          # BD-Rate sanity on synthetic curves
```
