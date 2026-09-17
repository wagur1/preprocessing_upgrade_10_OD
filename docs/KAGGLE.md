# KAGGLE.md — running upgrade-3 on Kaggle

A step-by-step guide to reproduce the A1+A2+A3 headline result on a **free Kaggle
T4/P100 GPU**. Kaggle already ships `ffmpeg` (with `libx264`+`libx265`), `torch`,
and `torchvision`, so setup is minimal.

> **Budget reality.** Kaggle notebooks give ~12 h GPU/session and ~20 GB working
> disk. This repo is built around a **≤ 3 GB balanced clip index** so a full
> train→eval cycle fits comfortably. Numbers on this subset fluctuate ±3–4 % per
> seed — run 3–5 seeds and report a CI (`kaggle/report_ci.py`).

---

## 0. One-time notebook setup

Create a new Kaggle Notebook, set **Accelerator = GPU T4 ×1** (or P100), **Internet = On**.

```bash
!git clone https://github.com/wagur1/pre_processing_upgrade_3.git
%cd pre_processing_upgrade_3
!pip -q install compressai av                   # torch/torchvision/ffmpeg already present
!ffmpeg -hide_banner -encoders | grep -E "libx264|libx265"   # verify both codecs
```

Sanity-check the code paths (no dataset / GPU needed, ~seconds):

```bash
!python -m src.models.virtual_codec
!python -m src.models.task_mask
!python -m src.tasks.multi_teacher
!python -m src.metrics.bd_rate
!python -m src.models.ste_codec                 # needs ffmpeg (present on Kaggle)
```

## 1. Add data & build the index

### Action recognition (Kinetics-400 subset — the headline task)

Add the Kaggle dataset **`rohanmallick/kinetics-train-5per`** (Add Input → Search).
It mounts read-only at `/kaggle/input/kinetics-train-5per/...`. Build a ≤3 GB
balanced index whose class folders map to the frozen analyzer's 400-class ordering:

```bash
!python -m src.data.prepare_3gb \
    --root /kaggle/input/kinetics-train-5per/train \
    --out  data/index/kinetics_3gb.json \
    --cap-gb 3.0 --val-frac 0.1 --test-frac 0.1 --backbone r3d_18
```

Nothing is copied/transcoded — the index just references the mounted mp4s.

### Tracking (GOT-10k — optional second task)

Add a GOT-10k **val** dataset (has per-frame boxes needed for AUC), then build
independent train/val/test sequence splits:

```bash
!python -m src.data.prepare_got10k --root /kaggle/input/got10k/val \
    --out data/index/got10k_3gb.json --cap-gb 3.0 --val-frac 0.2 --test-frac 0.2
```

---

## 2. Stage 1 — proxy pretrain (fast, differentiable block-DCT proxy)

```bash
!python train.py --config configs/universal_action_recognition.yaml \
    data.index=data/index/kinetics_3gb.json \
    train.epochs=5 train.batch_size=4
```

* Trains against the **teacher panel** `[r3d_18, mc3_18]` (A1), with the
  **task-mask** (A2) and **soft→hard anneal** (A3, `codec.anneal: 1.0`) already on.
* For a quick smoke run, cap steps: append `train.max_steps=300`.
* Best checkpoint → `outputs/checkpoints/preprocessor.pth`; resumable state →
  `outputs/checkpoints/preprocessor_last.pth`.

## 3. Stage 2 — real-codec calibration fine-tune (A3 STE)

Short, ffmpeg-in-the-loop, resumed from stage 1 at a small LR:

```bash
!python train.py --config configs/universal_action_recognition.yaml \
    data.index=data/index/kinetics_3gb.json \
    codec.kind=ste codec.ste_codec=h265 \
    train.finetune=true train.resume=false train.epochs=6 train.lr=3e-5 train.max_steps=400
```

This is ~10–50× slower per step (real x265 per batch) — keep it to a few hundred
steps. It nudges the edit onto the *true* x265 rate/quant geometry.

## 4. Evaluate on a HELD-OUT analyzer (the A1 claim) + real x264/x265

```bash
!python evaluate.py --config configs/universal_action_recognition.yaml \
    --ckpt outputs/checkpoints/preprocessor.pth \
    data.index=data/index/kinetics_3gb.json \
    eval.held_out_backbone=r2plus1d_18
```

Reads `outputs/eval/`:
* `results.json` — `bd_prep_gain` (**the real claim**: `prep+x265 vs x265`, …) and
  `bd_vs_anchor` (cross-codec, reference only).
* `curves.csv`, `rate_accuracy.png`, `qualitative.png`.

When `eval.per_sequence=true`, evaluation additionally writes
`sequence_points.csv`, `sequence_bd_rate.csv`, and `sequence_bd_rate.json`.
These contain the five QP points and same-codec BD-Rate for every held-out
video. Per-video `top1` remains a binary diagnostic; per-sequence BD fitting
uses `target_prob` (ground-truth class probability) as the continuous quality
metric. Aggregate headline BD-Rate still uses dataset top-1 accuracy.

Proxy curves are omitted by default because the paper claim uses real H.264/H.265.
Set `eval.include_proxy=true` only for virtual-codec diagnostics.

Evaluation defaults to the independent `test` split. Set `eval.split=val` only for
diagnostics, and set `eval.held_out_backbone=null` to evaluate in-domain (`r3d_18`).

## 5. Multi-seed confidence interval

Because subset BD-Rate is noisy, run each full seed in a separate Kaggle session
(three sequential x265 runs usually exceed the 12-hour limit):

```bash
!SEED=0 CLEAN_RUN=1 bash kaggle/run.sh  # use 1 and 2 in two other sessions
!python kaggle/report_ci.py outputs/seed_*/eval_stage2/results.json
```

`kaggle/report_ci.py` prints mean ± Student-t 95 % CI of `bd_prep_gain` across seeds.

---

## 6. Ready-to-run notebook

`kaggle/upgrade3_kaggle.ipynb` chains sections 0→5 in order — open it directly in
Kaggle, attach the datasets, and Run All. `kaggle/run.sh` is the same as a shell
script for a scripted session.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ffmpeg: libx265 not found` | rare on Kaggle; `!apt-get -qq install -y x265` or fall back to `codec.ste_codec=h264` |
| `CUDA out of memory` | lower `train.batch_size` (4→2) or `data.frame_size` (128→112), or `data.num_frames` (16→8) |
| STE stage extremely slow | expected — cap with `train.max_steps` (300–500); it is a fine-tune, not full training |
| `prep+h26x` missing from results | ffmpeg not on PATH → real-codec anchors skipped; check section 0 verification |
| BD-Rate swings between runs | expected on a 3 GB subset (±3–4 %); use section 5 multi-seed CI |
| classes skipped in `prepare_3gb` | folder name didn't map to a Kinetics-400 label; harmless (kept classes still match the frozen model) |

## What "success" looks like

A negative **`bd_prep_gain`** on `prep+x265 vs x265` (and `prep+x264 vs x264`) with
non-negative BD-accuracy, **measured on the held-out `r2plus1d_18`** — i.e. the
preprocessor lets an *unseen* analyzer reach the same accuracy at fewer real x265
bits. That is the universal-preprocessing-with-frozen-codec claim this repo targets.
