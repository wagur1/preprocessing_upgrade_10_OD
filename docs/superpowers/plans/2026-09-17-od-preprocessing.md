# Object-Detection preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** produce a negative BD-rate on the mAP axis for VCM preprocessing on images (the empty Object Detection cell of MPEG w21834), with a method that cannot repeat the measured failure modes of the learned preprocessors.

**Architecture:** the preprocessor is replaced by a *zero-parameter* transform driven by the frozen detector's own boxes — protect the detected objects, destroy the background — measured by the existing 3-arm-style rate-accuracy protocol on the mAP axis. A tiny learned gate (~1-2k params, bounded to "how much to destroy where") is added only if the zero-parameter rung wins.

**Tech Stack:** PyTorch, torchvision (Faster R-CNN COCO), pycocotools (mAP), ffmpeg x264/x265 (real codecs), Kaggle kernels (T4) driven by the ops/ pushers.

**Spec:** `docs/OD_DESIGN.md`

## Global Constraints

- Codec, bitstream and decoder are **frozen**: the only interventions are pixel-domain before encode and after decode.
- Every reported number uses a **held-out analyzer + bootstrap CI** and the gap rule (`prep − anchor ≥ −0.05` at every QP, both codecs). No on-teacher numbers.
- Detection is single-frame: `T=1`, codec intra-only (`codec.inter: false`).
- NEVER submit torchvision boxes to COCO as-is: they are xyxy, COCO wants xywh (`_coco_box`).
- Every split in an index must be non-empty: an empty val split silently disables model selection and early stopping.
- Long Kaggle runs: write diagnostics to files in the output dir (a cell killed by the 12 h cap loses its stdout).

---

### Task 1: Bring the validated OD core into this repo

**Files:**
- Create: `src/data/coco_det.py`, `src/tasks/object_detection.py`, `ops/probe_detection.py`, `ops/push_detection_probe.py`, `ops/push_detection_train.py`, `configs/sandwich_coco_det.yaml`
- Create (tests): `tests/test_probe_detection.py`, `tests/test_coco_det.py`
- Modify: `src/data/__init__.py`, `src/tasks/base.py`, `src/engine.py` (imports + `_train_detection` + `train()` dispatch), `src/models/sandwich.py`, `src/models/upvcm.py` (the M2-init and `w_budget` fixes)

**Interfaces:**
- Consumes: the sandwich PRE/POST from `src/models/`.
- Produces: `CocoDetDataset`, `collate_coco_det`, `build_coco_index`, `ObjectDetectionAnalyzer`, `_coco_box`, `coco_map`, `Detector`, `scaled_gt`, `load_coco`.

- [ ] **Step 1: Copy the files from the validated repo** (`pre_processing_upgrade_9`) and add the two new tests.
- [ ] **Step 2: Run the OD tests**

Run: `pytest tests/test_probe_detection.py tests/test_coco_det.py -v`
Expected: PASS (the box-format test must include the "buggy format scores LOW" case).

- [ ] **Step 3: Run the whole suite**

Run: `pytest -q`
Expected: all PASS; note the count (127 in the source repo).

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: port the validated detection core (data, analyzer, probe, pushers)"
```

---

### Task 2: The mask mechanism as a first-class module

**Files:**
- Create: `src/models/mask_suppress.py`
- Test: `tests/test_mask_suppress.py`

**Interfaces:**
- Produces: `protect_mask(boxes, scores, labels, size, score_thresh, dilate) -> Tensor[1,1,S,S]`; `suppress(x, mask, sigma) -> Tensor` (same shape as `x`).

- [ ] **Step 1: Write the failing tests**

```python
def test_mask_covers_boxes_and_ignores_low_scores():
    m = protect_mask(torch.tensor([[10., 12., 30., 36.]]), torch.tensor([0.9]),
                     torch.tensor([1]), 64, 0.5, 0.15)
    assert m[:, :, 12:36, 10:30].min() == 1.0
    assert m[:, :, 0, 0] == 0.0
    assert protect_mask(torch.tensor([[10., 12., 30., 36.]]), torch.tensor([0.1]),
                        torch.tensor([1]), 64, 0.5, 0.15).sum() == 0

def test_suppress_is_identity_inside_and_blurs_outside():
    x = torch.rand(1, 3, 1, 64, 64)
    m = protect_mask(torch.tensor([[10., 12., 30., 36.]]), torch.tensor([0.9]),
                     torch.tensor([1]), 64, 0.5, 0.15)
    out = suppress(x, m, 6.0)
    assert torch.equal(out[:, :, :, 12:36, 10:30], x[:, :, :, 12:36, 10:30])
    assert not torch.allclose(out[:, :, :, :6, :6], x[:, :, :, :6, :6], atol=1e-4)
    assert torch.equal(suppress(x, m, 0.0), x)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_mask_suppress.py -v` → expected FAIL (module missing).

- [ ] **Step 3: Implement** (separable depthwise Gaussian; `groups=C` — a 1-channel kernel against 3-channel input is the bug that cost a cycle).

- [ ] **Step 4: Run to verify pass** → `pytest tests/test_mask_suppress.py -v`.

- [ ] **Step 5: Commit** `feat: detector-mask protection + background suppression`.

---

### Task 3: The R0 probe in this repo

**Files:**
- Create: `ops/probe_background_suppression.py`
- Test: `tests/test_bgsuppress_cli.py` (stub codec)

**Interfaces:**
- Consumes: `Detector`, `coco_map`, `_coco_box`, `load_coco`, `scaled_gt` from `ops/probe_detection.py`; `mask_suppress` from Task 2.
- Produces: `outputs/probe_bgsuppress/probe_bgsuppress.json` with `cover`, per-`sigma` curves and `bd_vs_anchor`, plus `per_image_records.npz`.

- [ ] **Step 1: Write the failing test** — stub the codec, run the CLI on a 4-image fixture, assert `cover ∈ [0,1]`, curves for `anchor`/`blur4`/`blur8`, `bd_vs_anchor` present, records written.
- [ ] **Step 2: Run to verify failure** → `pytest tests/test_bgsuppress_cli.py -v`.
- [ ] **Step 3: Implement** the script (arm loop = anchor + one arm per σ; blur outside mask only).
- [ ] **Step 4: Run to verify pass**.
- [ ] **Step 5: Commit** `feat(probe): R0 background suppression`.

---

### Task 4: Run R0 on Kaggle with a pre-registered reading

**Files:**
- Create: `docs/RUN_DESIGN_r0.md`, `ops/push_od_probe.py` (a renamed, OD-focused copy of `push_detection_probe.py`)

**Interfaces:**
- Consumes: Task 3's CLI, `awsaf49/coco-2017-dataset`.
- Produces: the R0 numbers (BD per σ per codec, `cover`, CI from the records).

- [ ] **Step 1: Write `docs/RUN_DESIGN_r0.md`** stating: the win/null/fail bands from the spec, `cover` as the first diagnostic, and that the CI is recomputed offline from the records.
- [ ] **Step 2: Push and run** `python ops/push_od_probe.py --commit <sha> --account <acct> --n-images 500 --size 320`.
- [ ] **Step 3: Read the result** with `tools/report_detprobe.py` and the offline CI script; record the verdict in `docs/RESULTS_r0.md`.
- [ ] **Step 4: Commit** the result doc.

**Decision gate:** if `BD ≥ −0.3%` on both codecs → stop this line and report; if `BD < 0` with a CI excluding zero → proceed to Task 5.

---

### Task 5: R1 — the bounded learned gate (only if Task 4 wins)

**Files:**
- Create: `src/models/gated_suppress.py`, `configs/gated_suppress_od.yaml`
- Test: `tests/test_gated_suppress.py`

**Interfaces:**
- Produces: `GatedSuppress(base_ch=8)`; `forward(x, mask, cond) -> x'`; identity at init; a per-pixel gate in [0,1] that can only scale the suppression strength.

- [ ] **Step 1: Write the failing tests** — identity at init (`torch.equal(forward(x, m, cond), x)`); gradients reach every parameter after one step; the gate is bounded in [0,1]; with the gate forced to 1 the output equals `suppress(x, m, sigma=0)` and with it at 0 it equals `suppress(x, m, sigma=sigma_max)`.
- [ ] **Step 2: Run to verify failure**.
- [ ] **Step 3: Implement** (small conv stack on `(x, mask, 1−mask)` + FiLM(cond) → sigmoid gate).
- [ ] **Step 4: Run to verify pass**.
- [ ] **Step 5: Commit**.

---

### Task 6: Train R1 through the real codec (conditional)

**Files:**
- Modify: `configs/gated_suppress_od.yaml` (`codec.kind: ste`, real x264/x265 intra), `docs/RUN_DESIGN_r1.md`

- [ ] **Step 1:** Pre-flight locally with the fake-image fixture (2 steps, CPU): checkpoint written, gate moved, protected region never modified by the *gate* (only the σ scale changes).
- [ ] **Step 2:** Push with `train.max_steps` bounded so the run fits the 12 h cap (a run whose duration is unknown has already cost a full account's weekly quota).
- [ ] **Step 3:** Evaluate with the R0 probe (2-arm) and compare against R0's best σ.
- [ ] **Step 4:** Commit the result.
