# R0 Gaussian POST A/B Implementation Plan

> **For agentic workers:** Execute this approved bounded experiment inline with test-first development. No new model training or unrelated codec repairs.

**Goal:** Measure whether full-frame Gaussian POST sigma 1 helps object detection after R0 PRE sigma 4, with exactly shared decoded input and bitrate.

**Architecture:** New standalone probe reuses the established COCO selection, detector, mask, standard codec and durable-record helpers. For each image/codec/QP it decodes the anchor once and R0 once. The R0 decoded tensor branches into unchanged PRE-only and Gaussian-filtered PRE+POST; no second R0 encode.

**Tech Stack:** Python, PyTorch, torchvision, FFmpeg libx264/libx265, pycocotools, NumPy, pytest, Kaggle CLI.

**Spec:** User-approved comparison in this conversation: R0 sigma4 versus R0 sigma4 plus POST sigma1, same 500 image IDs, H.264/H.265, five QPs. This file records that bounded design.

## Global constraints

- PRE: detector score >=0.5, box dilation 0.15, Gaussian sigma4; retain existing implementation.
- POST: full-frame Gaussian sigma1, finite 5x5 kernel (2*round(2*sigma)+1), reflection padding where legal. This is inspired by Otsuki, not an exact reproduction of its FFmpeg filter or averaging PRE.
- Input square 320, QPs 30/35/40/45/50, preset medium, yuv420p; frozen torchvision Faster R-CNN R50-FPN COCO_V1.
- Reuse exact 500 IDs from completed corrected-R0 NPZ, never sample another set.
- Arms: anchor, prep, post. prep/post share byte-for-byte decoded source and identical bpp. POST input is not the source or pre-codec image.
- No model training, no sigma sweep, no 5,000-image run, no changes to existing bootstrap jobs.
- Retain fail-closed codec checks; no fallback returning source image.
- Results are exploratory, same detector at both ends, not CTC. Two-QP smoke produces no BD-rate claim.

## Task 1 — Paired evaluation contract (tests first)

Files: create `tests/test_gaussian_post.py`, create `ops/probe_gaussian_post.py`.

Interface: `paired_predictions(x, mask, codec, detector)` returns a dict for anchor/prep/post, each containing bpp, raw detector predictions, and decoded_source_sha256.

- [ ] Add test that uses a deterministic fake codec with visibly changed decoded pixels and spy detector; detector observations must be anchor decode, R0 decode, Gaussian(R0 decode).
- [ ] Assert exactly two codec calls, exact prep/post bpp equality and SHA equality, anchor/PRE source tensors unchanged.
- [ ] Test Gaussian against torchvision gaussian_blur(decoded flattened, [5,5], [1,1]).
- [ ] Reject decoded wrong shape/NaNs and invalid bpp before detector invocation.
- [ ] Run `python -m pytest tests/test_gaussian_post.py -q` and preserve failing exit code before implementing.
- [ ] Implement minimal helper, rerun tests until green.

Reference assertion:
```python
assert arms['prep']['bpp'] == arms['post']['bpp']
assert arms['prep']['decoded_source_sha256'] == arms['post']['decoded_source_sha256']
assert codec.calls == 2
```

## Task 2 — Durable standalone probe

Files: extend `ops/probe_gaussian_post.py`, tests `tests/test_gaussian_post.py`.

CLI accepts --images --ann --image-ids (repeatable comma list) --n-images --size --qps --device --out. Main comparison fixes PRE/POST sigmas at 4/1.

- [ ] Add CLI/integration tests: explicit IDs, fresh output, all arms per cell, invalid options, per-image paired hash journal, failure state, no BD for two QPs.
- [ ] Reuse `Progress`, `atomic_json`, `load_coco_ids`, `load_coco`, `scaled_gt`, `_coco_box`, `coco_map`; fresh output directory prevents accidental overwrite.
- [ ] Save `records.jsonl`/NPZ with existing tag schema and separate `pairing.jsonl` with per-image decoded-source hashes and equality check.
- [ ] Record progress before codec/inference, final complete only after aggregation. Persist metadata protocol/kernel/frozen detector/IDs and point BD values.
- [ ] Report per-QP mAP difference post-minus-prep, BD vs anchor and direct BD post vs prep (not subtraction of anchor-referenced BD values).

## Task 3 — Local acceptance and launch

- [ ] Run full suite and `git diff --check`, preserving real exits.
- [ ] Run real CPU smoke on available COCO IDs139,285,632, both codecs QP30/40. Do not claim ten images when only three available.
- [ ] Verify complete status, 36 records /12 cells, twelve matched decode pairs, exact prep/post rates, finite detections. No BD interpretation on smoke.
- [ ] Review diff for PRE preservation, correct POST placement and prediction mutation issues.
- [ ] Commit explicit files on `experiment/r0-gaussian-post`; push that branch, not main; verify remote SHA.
- [ ] Generate notebook via existing parameter-free pusher --script ops/probe_gaussian_post.py, --n-images500 --qps30,35,40,45,50 and exact old ID list. Use a fresh private kernel slug/output directory.
- [ ] Dry-run bash syntax; verify metadata COCO-only/no checkpoint and source full SHA/exact IDs before Kaggle push.
- [ ] Push one T4 job, verify status and start existing watcher for artifact retrieval. Do not relaunch if output is temporarily unavailable.

## Acceptance/reporting

Success means code tested, real-codec smoke verified, and one correctly pinned Kaggle job launched. Scientific success requires completed artifacts: 500 IDs, 30 cells/15,000 prediction records, 5,000 pairing records, identical per-image prep/post bpp. No improvement claim until real results are read. Bootstrap uncertainty is separate from this launch and must not be confused with the existing R0 bootstrap.
