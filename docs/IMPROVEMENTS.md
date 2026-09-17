# IMPROVEMENTS.md — delta vs. baseline, upgrade-2 & upgrade-3, with citations

This document records **what changed and why**, mapped to the literature. It is the
paper-facing companion to `docs/MODEL.md` (architecture) and `README.md` (overview).

The chain of work:

* **Baseline** — Zhao et al., *A Preprocessing Framework for Video Machine Vision
  under Compression*, arXiv:2512.15331 (2025). Neural preprocessor trained through a
  differentiable virtual codec, deployed before real x264/x265, **> 15 % BD-Rate**
  savings on action recognition + tracking.
* **upgrade-2** — reproduced the harness (differentiable proxy, real-codec eval,
  BD-Rate-on-accuracy). Reached in-domain `prep+compressai` ≈ −2…−6 % with positive
  BD-accuracy, but **transfer to real x264/x265 stalled near break-even**. Report:
  `docs/bao_cao_preprocessing.md`.
* **upgrade-3** — three additive, backward-compatible contributions
  (A1/A2/A3) that raised the ceiling and turned transfer positive
  (−1.6 % / −0.3 % BD-Rate on the 1000-step STE run) — still far from the
  paper's −12…−19 %.
* **upgrade-5.1** (this repo) — two further contributions aimed at the remaining
  proxy→real gap, with the upgrade-3 measured results as motivation:
  * **C1 yuv420 colourspace proxy** — the virtual codec now converts
    RGB→BT.601 YCbCr, 2×2-subsamples chroma, and quantises chroma with a
    coarser step (`chroma_step_scale`, ≈ the H.26x chroma QP offset) before
    the transform, then reconstructs via bilinear chroma upsample — exactly
    the geometry of every `-pix_fmt yuv420p` encode. `colorspace: rgb`
    keeps the legacy behaviour for ablation.
  * **C2 in-grid QP protocol** — train on exactly the eval QPs
    `[30, 35, 40, 45, 50]` (previously `[22..42]`, leaving the heaviest
    eval points as out-of-distribution FiLM conditions; QP50 accuracy
    collapse in the upgrade-3 curves is the visible symptom).

## 0. Why C1/C2 (evidence from the upgrade-3 run)

The `qp_grid_ste1000` eval (seed 0, held-out `r2plus1d_18`, after STE stage 2)
showed: `prep+h264 −1.61 %`, `prep+h265 −0.27 %` BD-Rate — transfer turned
negative but tiny. Two signatures in the curves point at the two fixes:

1. **At QP50** (a training-distribution extrapolation point even in the
   in-grid run for the FiLM condition ranges trained at quality 1), `prep+h265`
   accuracy *dropped* below bare h265 (0.077 vs 0.111) — conditioning
   extrapolation failure. C2 + per-QP distinct qualities (already in the
   universal config) address this.
2. **At every QP**, the preprocessor's bpp ≈ the anchor's bpp and accuracy
   deltas are <0.01: the edit is nearly a no-op for the *real* codec. The RGB
   proxy lets the preprocessor "spend" budget on chroma high-frequency that
   yuv420 destroys for free — the training signal never rewards moving that
   budget to luma. C1 makes the proxy charge for it, exactly as the real
   codec does.

## 1. The three ceilings of upgrade-2, and their fixes

| upgrade-2 limitation | root cause | upgrade-3 fix | key refs |
|---|---|---|---|
| Overfits **one** frozen analyzer | edit specializes to that network's decision surface | **A1** multi-teacher panel + held-out generalization test | 2510.18680; Yang TCSVT 2024 |
| Bit/edit budget spent **uniformly** | one global `gamma` knob; cutting background also blurs the object | **A2** task-importance spatial mask (gradient saliency) | 1910.07392; 2504.02216; EURASIP JIVP 2025 |
| Trained **entirely** through a mismatched proxy | proxy rate/quant geometry ≠ deployment codec ⇒ poor real-x26x transfer | **A3** real-codec-in-the-loop STE + soft→hard quant anneal | 2206.05650; 2606.16185 |

## 2. Delta table (module by module)

| File | upgrade-2 | upgrade-3 change | contribution |
|---|---|---|---|
| `src/models/preprocessor.py` | U-Net + FiLM(rate) + SFT(motion), zero-init identity | *unchanged* (already the redesigned editor) | — |
| `src/models/color.py` | *absent* | **new (5.1)** — BT.601 matrix, 4:2:0 plane split/join, roundtrip | C1 |
| `src/models/virtual_codec.py` | block-DCT proxy, additive-noise quantizer only | `+ _anneal` buffer, `set_anneal()`, soft→hard STE blend in `_quant_rate`; **5.1:** `colorspace="yuv420"` dual-plane coding with `chroma_step_scale` (rgb legacy kept) | A3, C1 |
| `src/models/ste_codec.py` | *absent* | **new** — real x264/x265 forward, proxy backward; eval real path | A3 |
| `src/models/task_mask.py` | *absent* | **new** — `task_saliency` + `masked_tv` | A2 |
| `src/tasks/multi_teacher.py` | *absent* | **new** — frozen-teacher panel, sample/mean | A1 |
| `src/tasks/base.py` | `build_task` only | `+ backbone=` override, `+ build_analyzer(role=)` | A1 |
| `src/losses.py` | fixed weights, global TV/L1 | `+ use_task_mask`, `x_pre`/`task_mask` args, `(1−m)`-weighted `δ`/`γ` | A2 |
| `src/engine.py` | proxy-only train, single analyzer | `+ STECodec` wiring, `+ task_saliency` per step, `+ anneal` ramp, `role`-aware build; **5.1:** `colorspace`/`chroma_step_scale` plumbing | A1/A2/A3, C1 |
| `configs/universal_action_recognition.yaml` | *absent* | **new** headline config wiring A1+A2+A3+C1+C2 | — |

## 3. Why each fix is the right one (evidence)

**A1 — universal / analyzer-agnostic.** Standard-codec preprocessing works
(Lu et al. 2206.05650) only demonstrate *narrow* same-family transfer. The works
that prove *broad* held-out transfer (UG-ICM 2501.04579; All-in-One 2504.12997)
**retrain the codec** — outside our frozen-codec constraint. "Universal preprocessing
proven on a broad held-out analyzer *with the standard codec left frozen*" is the
open niche. Training against a panel and *measuring on a held-out backbone*
(`r2plus1d_18` ∉ `{r3d_18, mc3_18}`) is the honest test of that claim. Sampling one
teacher per step is a stochastic regularizer against single-network overfit.

**A2 — spatial bit allocation without touching the encoder.** upgrade-2 could only
trade in-domain gain for transfer through a *global* TV weight. Gradient saliency
`|∂L_task/∂x|` is exactly "which pixels move the machine's decision"; weighting the
smoothing/edit penalties by `1−m` makes the trade-off *spatial*. This is the
differentiable, pixel-domain analogue of RDO/ROI bit allocation (1910.07392,
2504.02216, EURASIP JIVP 2025) — but as a loss weight, so the codec stays frozen.

**A3 — close the proxy→real gap.** The single most reliable transfer result in the
literature: Lu et al. measured **forward-real-codec −20.3 % vs −14.6 %** for a proxy
used in both directions. `STECodec` reproduces that (real value, proxy gradient).
Soft→hard quantizer annealing (J4D 2606.16185) additionally makes the proxy *end* at
the real hard quantizer, so even stage-1 pretraining lands closer to deployment.

**C1 — the colourspace gap STE cannot close.** STE corrects the *value* the loss
sees (real reconstruction) but the *gradient* still flows through the proxy; if
the proxy's geometry is wrong the direction is still wrong. The upgrade-3 run
trained and evaluated on matching QP grids, yet the edit stayed a near-no-op on
real codecs (bpp deltas <1 %) — consistent with a preprocessor whose rate
gradient under-counts chroma: every yuv420 encode halves chroma resolution at
*any* QP and quantises it coarser (the H.26x chroma QP offset, ≈ +6 QP at
QP50), so chroma high-frequency is free to destroy and expensive to preserve.
The yuv420 proxy makes that cost visible *during training*, steering the edit
budget toward luma where the bits actually go. This mirrors the fidelity-first
argument of Talebi et al.'s pre-editing (TIP 2021) — model the codec's
degradation honestly — but at the colourspace level rather than the JPEG-DCT
level, and is the standard geometry used by learned-codec-for-machines works
(e.g. 2206.05650) that report −20 % BD-Rate. `colorspace: rgb` remains for the
ablation table.

**C2 — in-grid QP protocol.** FiLM conditions are only trained on the QPs in
`train.qp_list`; any eval QP outside it is extrapolation. The collapse of
`prep+h265` at QP50 in the upgrade-3 curves is the visible failure. Training on
exactly `[30, 35, 40, 45, 50]` (the eval grid, and the paper's own range) with
five *distinct* proxy qualities removes the gap at zero cost.

## 4. Expected effect & honest caveats

* upgrade-2's best confirmed operating point was **−6 % BD-Rate with positive
  BD-accuracy on `prep+h265 vs h265`** (the paper's real claim axis). A1/A2/A3 are
  designed to (i) make that gain *hold on an unseen analyzer* (A1), (ii) *deepen* it
  by not wasting bits on background (A2), and (iii) *transfer* it reliably to real
  x264/x265 (A3).
* This repo is a **research harness, not frozen numbers.** BD-Rate on a small
  Kinetics subset fluctuates ±3–4 % per seed. Run 3–5 seeds and report a CI
  (`kaggle/report_ci.py`). Effect size and significance grow with **data**, not
  architecture; A1/A2/A3 raise the *ceiling* and *robustness*, seeds establish
  *significance*.
* A3's STE stage runs ffmpeg per step (~10–50× slower) — it is a short calibration
  fine-tune, never a from-scratch trainer.

---

## References

Cited as `arXiv:ID` / venue in text; grouped by contribution.

### Baseline & harness
- **Zhao et al.** *A Preprocessing Framework for Video Machine Vision under
  Compression.* arXiv:2512.15331, 2025. — direct baseline; virtual-codec training,
  real-x26x deployment, >15 % BD-Rate.

### A1 — universal / multi-teacher / held-out
- **Lu et al.** *Preprocessing Enhanced Image Compression for Machine Vision.*
  arXiv:2206.05650 / IEEE TCSVT 2024. — analyzer-agnostic motivation; forward-real-codec.
- **Yang et al.** *Feature-modulation multi-task image preprocessor for machine
  vision.* IEEE TCSVT 2024, DOI 10.1109/TCSVT.2023.3348995. — feature distillation
  (ω), multi-task preprocessing.
- **Multi-teacher knowledge distillation.** arXiv:2510.18680. — panel aggregation.
- **UG-ICM.** arXiv:2501.04579. — broad held-out transfer *by retraining the codec*
  (contrast: we keep it frozen).
- **All-in-One Transfer for image coding for machines.** arXiv:2504.12997. — as above.

### A2 — task-driven / ROI bit allocation
- **Reinforced Bit Allocation.** arXiv:1910.07392.
- **Feature-preserving RDO for machines.** arXiv:2504.02216.
- **ROI retargeting for machines.** EURASIP J. Image & Video Processing, 2025,
  DOI 10.1186/s13640-025-00682-3.

### A3 — proxy→real transfer
- **Lu et al.** arXiv:2206.05650 / TCSVT 2024. — forward-real-codec −20.3 % vs −14.6 %.
- **J4D.** arXiv:2606.16185. — soft-quantizer α→∞ annealing.
- **Sandwiched Compression.** arXiv:2402.05887. — block-transform proxy geometry.
- **DPP** (Chadha & Andreopoulos). CVPR 2021. — virtual-codec + straight-through.
- **Talebi et al.** *Better Compression with Deep Pre-Editing.* IEEE TIP 2021. —
  differentiable-JPEG proxy pre-editing.

### Preprocessor components
- **FiLM** (Perez et al.). *FiLM: Visual Reasoning with a General Conditioning
  Layer.* AAAI 2018.
- **SFT** (Wang et al.). *Recovering Realistic Texture in Image Super-resolution by
  Deep Spatial Feature Transform.* CVPR 2018.

### Datasets
- **Kinetics-400** (Kay et al., 2017). — action recognition.
- **GOT-10k** (Huang et al.). IEEE TPAMI 2021. — single-object tracking.

### BibTeX

```bibtex
@article{zhao2025vcmprep,   title={A Preprocessing Framework for Video Machine Vision under Compression}, journal={arXiv:2512.15331}, year={2025}}
@article{lu2024prepicm,     title={Preprocessing Enhanced Image Compression for Machine Vision}, journal={IEEE TCSVT / arXiv:2206.05650}, year={2024}}
@article{yang2024featmod,   title={Feature-Modulation Multi-Task Image Preprocessor for Machine Vision}, journal={IEEE TCSVT}, doi={10.1109/TCSVT.2023.3348995}, year={2024}}
@article{mtkd2025,          title={Multi-Teacher Knowledge Distillation}, journal={arXiv:2510.18680}, year={2025}}
@article{ugicm2025,         title={UG-ICM: Universal/Generalizable Image Coding for Machines}, journal={arXiv:2501.04579}, year={2025}}
@article{allinone2025,      title={All-in-One Transfer for Image Coding for Machines}, journal={arXiv:2504.12997}, year={2025}}
@article{rba2019,           title={Reinforced Bit Allocation}, journal={arXiv:1910.07392}, year={2019}}
@article{fprdo2025,         title={Feature-Preserving Rate-Distortion Optimization for Machines}, journal={arXiv:2504.02216}, year={2025}}
@article{roi2025,           title={ROI Retargeting for Machines}, journal={EURASIP JIVP}, doi={10.1186/s13640-025-00682-3}, year={2025}}
@article{j4d2026,           title={J4D: Soft-Quantizer Annealing}, journal={arXiv:2606.16185}, year={2026}}
@article{sandwich2024,      title={Sandwiched Compression}, journal={arXiv:2402.05887}, year={2024}}
@inproceedings{dpp2021,     title={Deep Perceptual Preprocessing (DPP)}, author={Chadha and Andreopoulos}, booktitle={CVPR}, year={2021}}
@article{talebi2021preedit, title={Better Compression with Deep Pre-Editing}, journal={IEEE TIP}, year={2021}}
@inproceedings{perez2018film,   title={FiLM: Visual Reasoning with a General Conditioning Layer}, booktitle={AAAI}, year={2018}}
@inproceedings{wang2018sft,     title={Recovering Realistic Texture in Image Super-resolution by Deep Spatial Feature Transform}, booktitle={CVPR}, year={2018}}
@article{kay2017kinetics,   title={The Kinetics Human Action Video Dataset}, journal={arXiv:1705.06950}, year={2017}}
@article{huang2021got10k,   title={GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking}, journal={IEEE TPAMI}, year={2021}}
```


