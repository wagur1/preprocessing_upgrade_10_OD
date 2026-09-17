# RUN DESIGN — Additive retrain in the upgrade_6 harness (§10 of tong-ket-vcm.txt)

Status: DRAFT for sign-off · 2026-08-30 · Target repo: `C:\Users\Wagur1\preprocessing_upgrade_6` (HEAD `6c8e336`)

## 1. Objective and claim thresholds

Train the additive (Zhao-style) preprocessor **inside our canonical harness**, then select its
operating point on **measured real-codec bpp** (gradient-free), and evaluate once on the
canonical 1159-clip protocol.

- Publication target: BD-rate ≤ −8% on BOTH h264 and h265 + gap rule (≥ −0.05 at every QP).
- **Certification rule (pre-registered): to claim ≤ −8% at n=1159, the measured point estimate
  must be ≤ −11% (bootstrap CI95 ≈ ±3 pp).**
- Realistic success criterion (gap ≥ −0.03 is the *effective* rule — pessimistic calibration
  showed a legal −0.05 gap can still price BD-positive).
- Known ceiling arithmetic (rev.py): at ZERO bit cost the current accuracy benefit prices at
  −7.50% (h264) / −8.16% (h265). h264 is the binding side: it needs **both** Δbpp ≤ 0 at the
  selected operating point **and** ~2× the current accuracy benefit. The existence proof is
  Zhao's reported −17.6% h264 at full Kinetics scale; whether 2× exists at 10.8K-clip scale is
  exactly what this run answers.

## 2. Why this run (mechanism, not another variant)

All 6 subtractive axes are closed (R-D neutrality ×7; per-codec law: BD_h265/BD_h264 ≈ 0.03–0.08
for any subtractive filter, so dual −8% is structurally impossible there). Only an
**additive/pre-emphasis** mechanism (gap ≥ 0) can satisfy both codecs simultaneously. The
additive run (best.pt) owns the project's best h265 number (−3.18%) while trained under
worst-case conditions (1,442 clips, ~900 steps, SSF2020 proxy — the least realistic in the
project, alpha=10 λ=0.001 so it never learned about bits). Every learning-failure root cause
identified to date (dead proxy-rate gradient, STE gradient-direction blindness) is **deleted by
construction** here: we train for accuracy only, and choose the bit operating point
gradient-free on measured x264/x265 bpp.

## 3. Method

### 3.1 Architecture — pinned tree, our switches

The 9,795-parameter two-branch additive residual (strict=True pinned by u6_big4/arch.py):

```
spatial_stem 3→16 · spatial_residual.body 16→16→16 · temporal_stem 24→16 (8 frames × 3ch)
fusion.gate 32→16→16 · to_rgb 16→3 · out = x + to_rgb(fused)
```

Fully convolutional, resolution-agnostic, **unconditioned** (no QP input — matches Zhao's one
robust model). The 4 ambiguous wiring switches were never identified (both gates failed; MSE is
non-discriminative — 12 variants within 2.7%) and per §10 they are **ours to choose**:
`act=relu, residual=true, sigmoid gate, lerp fusion (convex blend), center-pick temporal context,
clamp(0,1) on`.

Port: `src/models/additive.py` + `model.arch: additive` branch in `_build_models` + `"arch"`
added to the checkpoint arch-restore tuple in `evaluate()` (the 80724ab fix pattern). Capacity
stays 9,795 params — fixed arch isolates the scale/teacher/proxy levers (widening is arm B,
below, only if arm A saturates).

### 3.2 Objective — Zhao's exact 3-term form

`L = L_Acc + 10·(L_D + 0.001·L_R)` → harness knobs: `lam_task=1, mu=10, beta=0.001,
omega=tau=delta=gamma=0, use_task_mask=false`. The rate term is decorative by design
(α=10 dominates); the proxy's job is the **distortion channel** (x_hat for L_D and L_task), not
its rate gradient. Add `train.clip_grad: 1.0` (the additive run used it; harness lacks it).
Reg-warmup is a no-op here (it ramps only beta/gamma/delta) — correct.

### 3.3 Training proxy — virtual (block-DCT, not SSF2020)

`codec.kind: virtual` with `inter: true` (closed-loop P-frame prediction), `colorspace: yuv420`
(C1), qualities (1,2,3,5,8), C2 in-grid QP→quality mapping `{30:8, 35:5, 40:3, 45:2, 50:1}`,
QP sampled per step. Block DCT + scalar quant + inter prediction + 4:2:0 is the closest match
to x264/x265 quantization noise we have; SSF2020 was the furthest-from-real. Note: differs from
the additive run (3 SSF2020 qualities) — deliberate, C2 is our protocol.

### 3.4 Data — full non-test pool, canonical split hygiene

Same dataset (`rohanmallick/kinetics-train-5per`), same `build_index(CAP_GB=40)` scan → 10,805
clips, then the **mount-independent canonical hash rule** (`_key = <class>/<file>.mp4`):

- **test = md5[0:8]%10==0** → the canonical 1159 (fingerprint `30f083f8520a`) — NEVER touched in training
- **val = md5[0:8]%10==1** (~1,080 clips, deterministic, mirrors the test rule)
- **train = the remaining ~8,644 clips** (6.0× the additive run's 1,442)

New script `scripts/build_train_index.py` emits a harness-format index JSON with labels
(class folder → Kinetics-400 index, same as prepare_3gb). The hash rule is relative-path based,
so the split is stable across Kaggle mounts.

### 3.5 Teachers — A1 panel (our lever, feeds the universality story)

`teachers: [r3d_18, mc3_18], teacher_sampling: sample` (1× analyzer cost/step), eval on
**held-out `r2plus1d_18`** — the canonical eval analyzer. (Zhao used a single analyzer; the
panel is our contribution and a regulariser. Single-teacher is the fallback option at sign-off.)

### 3.6 Geometry & schedule (match the additive run; one T4)

16 frames, stride 2, **frame_size 128** (train + canonical eval both — self-consistent round,
same geometry as the additive run; the analytic-family table at 112 is not directly comparable
and doesn't need to be), analyzer clip_size 112, batch 8 (no gradient accumulation in this harness -- the config ships train.batch_size: 8 and run-1 trained fine with it),
lr 1e-4 Adam, cosine, seed 0, val every epoch (`val_qp_mode: all`, `val_max_batches: 5`),
early stop patience 3, `resume: true` wired for session-death recovery (12h Kaggle cap).

Schedule: 2,161 loader-steps/epoch (~1,081 opt steps). **MEASURED on T4 (run of
2026-08-31, `dl_train_fix/u6-d10-train.log`): ~23 min/epoch, 6 epochs in 2h24m** —
the earlier "fp32 ≈ 2.5 s/step → ~90 min/epoch → 6 epochs ≈ 9h" estimate was ~4×
too pessimistic and produced two wrong ETAs. Budget 3h with 1h margin, not 10h.

## 4. Phases, gates, and cost

| Phase | What | GPU | Wall | Gate to proceed |
|---|---|---|---|---|
| 0 | Port arch + index builder + config + `train.clip_grad` (+AMP flag), unit tests, CPU smoke (10 clips × 50 steps: loss falls, Δ≠0, no NaN) | 0 (local) | ~2h my time | tests green |
| 1 | Train on Kaggle (1 T4 session, repo @port-commit) | ~9–10h | 10h | val loss curve sane; checkpoint saved |
| 2 | Screen + operating-point selection: 120 clips × s∈{0.25, 0.5, 0.75, 1.0} × 5 QP × 2 codecs — measure Δbpp and Δacc; price with `u6_big/budget.py`; pick s* by the pre-registered rule (§5) | ~3h | 3h | Δbpp(s) curve monotone-ish; some s has Δbpp ≤ +2% |
| 3 | Canonical eval at s*: 1159 clips, 6 shards, x264/x265 medium, QP 30–50, per-clip rows, merge + 1000-draw bootstrap CI | ~3.5h (6 slots) | 35 min | — (this produces the verdict) |

**Total ≈ 13–16 GPU-hours of the ~56h pooled quota (10 accounts × 2 slots, 10/10 probe-PASS).**
Straggler/failure handling per auto.py (retry once, abandon after 90 min).

Checkpoint delivery to eval kernels: push `best` checkpoint as a small Kaggle dataset
(`{user}/u6-ckpt-additive`), attach alongside the Kinetics dataset.

## 5. Pre-registered selection rule (Phase 2 → 3)

The model is unconditioned, so the bit operating point is a single global strength
`x_pre(s) = x + s·Δ(x)`. Screen grid (revised 2026-08-31 **after measuring the trained
checkpoint but BEFORE any screen number existed** — see the note below): **s ∈ {0.02, 0.05,
0.1, 0.25, 0.5, 1.0}**. The grid must bracket Δbpp=0 on both sides, and the measurement says
the crossing is BELOW 0.1, not above 1.0: s=0.02 reproduces best.pt's own edit magnitude and
is mandatory; s=1.0 is kept only as the trained reference point. The previous grid
{0.1 … 1.5} is superseded — s=1.5 was insurance for an UNDER-scaled residual, a branch the
measurement kills. Arm count is unchanged (6), so the screen still fits one session per
resolution. On the 120-clip screen (per resolution, 128 AND 224):

1. **Selection is by THRESHOLD, not argmin BD** (pick_s.py, pre-registered):
   `s* = max{ s : gap(s) ≥ −0.03 at every QP, both codecs AND Δbpp(s) ≤ 0 at every QP,
   both codecs }`, with paired bootstrap 5% lower bounds per point ("gap not decidable at
   this n" is printed rather than silently trusted). Rationale: at n≈120 BD spreads 14–16 pp
   (the n=207 lesson: t_stat reads +1.93% and −1.90% at different sample sizes — SIGN FLIP);
   but a one-parameter MONOTONE family is threshold-searchable at n=120, and Δbpp at fixed
   QP contains no analyzer at all. BD is deliberately NOT computed by the selector.
2. If the selector prints "no operating point satisfies the rule": **do NOT conclude the
   mechanism failed** — extend the grid DOWN (s < 0.02) first; the measurement below says the
   failure mode is an over-expensive edit, which lives at small s.
   **VOIDED 2026-08-31 by the screen itself, see §5.1 — this clause presupposed gap ≥ 0, and
   the measured gap is monotonically NEGATIVE. Do not extend the grid down.**
3. Run Phase 3 ONCE per resolution at that resolution's s\*. (A second point s\*±0.25 is
   allowed only if quota ≥ 30h remains.)

**Objective note, superseding the 2026-08-31 mu-cap arithmetic.** The direction of that
correction was right and is now confirmed by measurement, but its numbers were taken from
best.pt and do not describe THIS checkpoint. Measured on the D10 checkpoint (epoch 5 /
step 5395, `u6_big4/edit_size.py`, before Phase 2 was pushed):

| s | residual RMS | edit PSNR | pixels hitting the [0,1] clamp |
|------|------|---------|------|
| 0.10 | 0.053 | 25.6 dB | 3.8% |
| 0.25 | 0.110 | 19.2 dB | 11.5% |
| 0.50 | 0.174 | 15.2 dB | 25.1% |
| 1.00 | 0.245 | **12.2 dB** | 44.6% |

So mu=10 did not merely fail to pin the edit — the edit ran to **19× best.pt's magnitude**
(0.245 vs 0.013 RMS; 12.2 dB vs 37.7 dB), with nearly half of all pixels saturating the
output clamp. At s=1 this is not a prefilter, it is a repaint. best.pt's edit magnitude is
reproduced at **s ≈ 0.025**, which is what fixes the bottom of the grid. Total variation on
smooth synthetic input rises 22% even at s=0.1 (read as direction only — the synthetic
baseline TV is artificially low, so real video will scale less), i.e. the edit ADDS structure
to encode. Expect Δbpp > 0 across most of the grid; the whole selector exists to find where,
or whether, it crosses zero.

### 5.1 Screen outcome, r128 (measured 2026-08-31, `pick_s.py` on `outputs/d10_screen_r128`)

113 clips × 5 QP × 2 codecs × 7 arms (7 of 120 clips failed to decode; arms are paired, so the
loss is unbiased). **VERDICT: no operating point satisfies the rule.** Both conditions fail at
every s:

| s | gap h264 | gap h265 | Δbpp h264 | Δbpp h265 |
|---|---|---|---|---|
| 0.02 | −0.004 | −0.019 | +2.0% | +1.1% |
| 0.05 | −0.007 | −0.023 | +4.7% | +2.9% |
| 0.10 | −0.038 | −0.045 | +9.9% | +5.9% |
| 0.25 | −0.124 | −0.117 | +24.3% | +14.0% |
| 0.50 | −0.196 | −0.218 | +42.7% | +23.9% |
| 1.00 | −0.306 | −0.315 | +67.3% | +35.6% |

(means over QP; worst single point −0.522 at h264 QP30 s=1.0. Damage is largest at QP30 and
decays to ≈0 by QP50 — it is worst exactly where the anchor is strongest.)

**This is a DIRECTION result, not a magnitude one, and that is what voids §5.2.** At s=0.02 the
edit magnitude *matches* best.pt (RMS 0.011 vs 0.013; 39.2 dB vs 37.7 dB) and the gap is
indistinguishable from zero (boot5% −0.053…+0.000) where best.pt delivered **+0.033**. Same size,
no gain. Both selection quantities are monotone in s and reach zero only at s=0 — the identity —
so extending the grid downward converges to the anchor from the wrong side and its best possible
outcome is BD = 0. A screen below s=0.02 is therefore not run; it would spend a session to
confirm a guaranteed null. Phase 3 is held for the same reason: BD on a checkpoint that is
accuracy-negative at every affordable s prices nothing.

Cause attribution is deferred to the no-codec diagnostic (`u6-d10-diag-r128`: all three
backbones, clean input, same 120-clip screen set), which separates "our reading of the objective
trains damage" (teachers also negative) from "cross-backbone transfer failure" (teachers
positive, held-out r2plus1d_18 negative). Nothing in the screen can separate them: it discards
`src_ok` and every number in it is post-codec and held-out-analyzer only.

## 5.2 CAUSE FOUND — D10 IS VOID, and the fault is the training proxy's quantiser step

Two diagnostics ran after §5.1, both cheap, both decisive.

**(a) No-codec diagnostic (`u6-d10-diag-r128`, 113 clips, all three backbones, clean input).**
The edit damages the TEACHERS as badly as the held-out analyzer — r3d_18 +0.000/+0.009/−0.018/
−0.186/−0.354/−0.531 and mc3_18 +0.018/−0.035/−0.080/−0.159/−0.372/−0.522 at s=0.02…1.0 against
r2plus1d_18's −0.018/−0.035/−0.080/−0.212/−0.398/−0.602. **Cross-backbone transfer failure is
killed**: at s ≤ 0.05 all three sit within ±2 clips of source, i.e. the edit helps nobody.

**(b) Proxy-sanity kernel (`u6-d10-proxy-r128`), and it did not return either pre-registered
branch.** Running the SAME `VirtualCodec` the model trained against, with the preprocessor OFF
(s=0, exact identity), the analyzer is at CHANCE at every quality in the training grid:

| arm | clean | QP30 | QP35 | QP40 | QP45 | QP50 |
|---|---|---|---|---|---|---|
| identity, r2plus1d_18 | 0.894 / CE 0.49 | 0.018 / 8.12 | 0.009 / 8.42 | 0.009 / 7.58 | 0.027 / 8.11 | 0.009 / 8.99 |
| identity, r3d_18 | 0.752 / 0.93 | 0.018 / 8.43 | 0.009 / 8.85 | 0.009 / 8.48 | 0.009 / 8.09 | 0.009 / 7.82 |
| identity, mc3_18 | 0.770 / 0.96 | 0.018 / 7.61 | 0.009 / 7.73 | 0.009 / 7.50 | 0.009 / 7.52 | 0.018 / 7.97 |

So the question "does the residual help through the proxy?" is not answerable and not
interesting: **the proxy destroys the video by itself.** 0.9–2.7% top-1 is chance on 400 classes,
and every CE is 1.5–3.0 nats ABOVE ln(400)=5.991.

**Root cause, measured (`u6_big4/proxy_calib.py`, `u6_big4/proxy_target.py`).** `VirtualCodec`
codes planes in **[0,1]**, but `configs/additive_ar.yaml` shipped `step_coarse: 3.0,
step_fine: 1.0` — JPEG-plausible numbers in **[0,255]** units, i.e. **255× too coarse**. An
orthonormal 8×8 DCT puts DC at 8·mean ≈ 4 and nearly every AC coefficient below 0.5, so
`round(coeff/1.0)` zeroes all AC and leaves the block mean on a 1/8 grid — 8 grey levels, blocks
only. Measured against real x264 at the same geometry (3 clips, 128², 16 frames):

| | q8→QP30 | q5→QP35 | q3→QP40 | q2→QP45 | q1→QP50 |
|---|---|---|---|---|---|
| real x264 | 31.44 dB | 28.85 | 26.32 | 23.70 | 21.52 |
| proxy, SHIPPED 3.0/1.0 | **19.16** | 15.19 | 15.71 | 12.59 | **9.71** |
| proxy, defaults 0.25/0.03 | 35.86 | 32.04 | 28.73 | 27.53 | 25.40 |

The shipped proxy is **10.6–13.7 dB below real x264 at every mapped QP, and its FINEST setting
(19.16 dB) is worse than x264's WORST (QP50, 21.52 dB)** — the whole training grid sat outside
the range the round evaluates. That is the mechanical explanation of the training log: task CE
ran 8.484 → 8.038 and never came within 2 nats of ln(400), because the analyzer was at chance on
every frame the optimiser ever showed it, so **L_task carried no usable gradient at all**; 76% of
the total loss drop was the `mu·L_D` term fitting a destroyed target. The gap being monotone
negative in s, ≈0 at s=0.02, and equally bad on the teachers all follow: the residual is noise
shaped by MSE against garbage, and scaling noise to nothing just returns the identity.

**How it got through the gates.** `virtual_codec._demo()` asserted fidelity only on the class
DEFAULT steps and checked the shipped calibration for **monotonicity alone**
(`calibrated_mse[0] > calibrated_mse[-1]`) — which a completely destroyed picture satisfies. The
motivation for the coarse steps is in the source comment at `_quant_rate`: an earlier rate form
"bottomed out at ~0.77 bpp, pinning the proxy ~20× above the x264/x265 operating range". The step
was coarsened to chase a realistic **bpp** with a rate model (parameter-free Gaussian) that
under-counts bits, and that destroyed the **distortion** channel — the only channel this
objective actually uses, since `beta=0.001` makes the rate term decorative by design (§3.2).

**Fixes committed:**
1. `configs/additive_ar.yaml` → `step_coarse: 0.25, step_fine: 0.03` (+2.4…+4.4 dB vs real x264
   at the mapped QP; bpp lands 5–20× high, which is correct and irrelevant here — do NOT
   re-coarsen the step to fix bpp, that is the mistake that voided this run).
2. `virtual_codec._demo()` now asserts the finest quality beats real x264 QP50 (>24 dB) and the
   coarsest stays above the recognition floor (>15 dB), naming the [0,1]-vs-[0,255] trap.
3. `src/models/additive.py` zero-inits `to_rgb`, so the untrained model is EXACTLY the identity
   (verified `max|dx| = 0`, params still 9,795, gradient still reaches `to_rgb`). Default init
   started the residual at RMS 0.10–0.18, i.e. 15–20 dB from identity — a second, independent
   handicap: the optimiser opened from a random repaint.

**Status: D10 is VOID and must be rerun, not written up.** Nothing in it measures the additive
mechanism, and §5.1's screen tables describe a model trained against an unrecognisable proxy.
The r128/r224 screens and the no-codec diagnostic keep their value only as the audit trail that
found this.

## 5.3 RERUN DONE (2026-08-31, commit `2101bf1`) -- the first VALID additive round

Train: 6 epochs in **2h24m** (~23 min/epoch, not the ~90 this doc estimated in §4 -- corrected
there), val_loss 3.0975 -> 1.3228 monotone, best epoch 5, task CE QP-graded and alive (QP30
0.098-2.82 ... QP50 5.60-9.04, i.e. below `ln 400 = 5.991` where the void round never got).
Residual RMS 0.125 at s=1.0, ~half the void checkpoint's 0.245.

Screen r128, 113 val clips, real x264/x265, held-out `r2plus1d_18`:

| codec | anchor bpp / top1 | s=1.0 bpp / top1 | dbpp | dacc@QP45 (boot5%) |
|---|---|---|---|---|
| h264 | 0.1364 / 0.5027 | 0.2210 / 0.5841 | **+62.0%** | **+0.195 (+0.124)** |
| h265 | 0.1708 / 0.4867 | 0.2505 / 0.5611 | **+46.7%** | **+0.168 (+0.097)** |

**Accuracy is real and significant -- a first for this project.** `pick_s.py` still refuses every
arm, but note it fails on the `dbpp <= 0` clause, which a purely additive residual can never
satisfy at any s > 0 (adding signal adds entropy). That clause was written for subtractive
prefilters; the meaningful test here is BD, and **BD is positive at 12/12 arms** (best -0.19%
h264 s=0.02; +15.28% / +13.83% at s=1.0). QP-gating does not rescue it and cannot: BD on the
accuracy axis cannot credit a gain BELOW the anchor's own accuracy range (anchor QP45/50 =
0.327/0.097, the edit lifts them to ~0.52/0.26, overlap becomes [0.257, 0.814]), and gating
truncates that overlap further.

**Ceiling analysis (`u6_big4/ceiling.py`) -- accuracy is NOT the bottleneck, bit cost is.**
Holding measured accuracy fixed and scaling the edit's cost to k x its real value:

| protocol / codec | 100% | 75% | 50% | 25% | 0% |
|---|---|---|---|---|---|
| held-out h264 | -0.19% | -0.61% | -3.82% | -10.90% | **-19.76%** |
| held-out h265 | +0.96% | +0.21% | -0.98% | -4.39% | **-10.49%** |

The gain already in hand is worth -19.76% on h264 if the residual were free -- past the -11%
needed to certify -8% at n=1159. **h265's ceiling is -10.49% even at zero cost**, so a certified
-8% on h265 is out of reach for additive editing regardless of how cheap the residual gets.
Read the table as an upper bound, not a forecast: amplitude and cost are coupled (the s-grid IS
that curve), so moving rightward requires changing the residual's SPECTRUM, not its amplitude.

**Why the residual is expensive, and the lever that was never pulled.** `configs/additive_ar.yaml`
ships `gamma: 0.0` (TV) and `delta: 0.0` (L1 edit magnitude) with `beta: 0.001` that its own
comment calls decorative -- the objective never asked for a *cheap* residual. `edit_size.py`
measures TV rising to **1.39x** on smooth input at s=1.0: the residual adds high-frequency
structure, the priciest thing for a DCT codec. `src/losses.py:32-38` already names `gamma` as
"the lever for transfer to x264/x265 (unlike `beta*bpp`, which only reduces the *proxy* codec's
bits)" -- and `beta` is the wrong knob here anyway, since the proxy is calibrated on PSNR and its
bpp is not trustworthy as an optimisation target (Trap #1).

## 5.4 The `gamma` sweep was a NULL, and `gamma_res` is the corrected term

A 3-point `gamma` sweep (0.01 / 0.03 / 0.1) ran at commit `0399f53`, one variable, three
accounts in parallel. All three trained cleanly -- fingerprint OK, val_loss 3.10 -> 1.31
monotone, best epoch 5, ~2h35m each -- and returned checkpoints **indistinguishable from the
`gamma: 0` baseline**:

| gamma | residual RMS @s=1.0 | PSNR | clipped | TV ratio |
|---|---|---|---|---|
| 0.0 (baseline) | 0.1253 | 18.0 dB | 23.0% | 1.39 |
| 0.01 | 0.1246 | 18.1 dB | 22.8% | 1.38 |
| 0.03 | 0.1247 | 18.1 dB | 22.9% | 1.39 |
| 0.1 | 0.1237 | 18.2 dB | 22.6% | 1.38 |

A 10x change in the coefficient moved the residual by 0.7% RMS; the in-training readout agrees
(final TV 0.1286 / 0.1288 / 0.1281). **The screens were not run** -- three checkpoints
indistinguishable from the baseline would have spent 2 more GPU-hours reproducing a known
result. Verified locally at zero cost with `u6_big4/edit_size.py`.

Two causes, both checkable in minutes before launching:

1. **Magnitude.** `gamma * TV(x_pre)` was 0.00-0.04% of the objective (TV ~ 6e-3 against
   `L_task` ~ 3.3) -- decorative in exactly the way this repo already documents `beta: 0.001`
   to be. A biting coefficient is ~10, i.e. 100-1000x what was run. The 0.03 was carried from
   `configs/robust_transfer.yaml`, where the mechanism is *subtractive blur* and the filter
   moves `TV(x_pre)` a great deal -- meaningful there, meaningless here.
2. **Target.** For an additive model `TV(x_pre)` is dominated by the SOURCE video's own
   texture, so most of the penalty lands on content the preprocessor never created, and
   minimising it means blurring the source -- rebuilding the subtractive family that §"R-D
   neutrality" already measures as neutral.

**Corrected term (commit `acbe7be`): `gamma_res * TV(x_pre - x)`.** It penalises only the
spectrum of what the model ADDS -- what makes an additive edit expensive on a DCT codec -- and
leaves the source alone. This is the term §5.3's ceiling actually calls for: a residual that is
CHEAPER AT THE SAME AMPLITUDE, hence the same accuracy. Amplitude is already controlled by
`delta` and by the strength grid, and an L1 penalty cannot distinguish spectra at all.

`tests/test_residual_tv.py` pins the distinguishing properties and the sizing lesson: the same
residual on a flat and on a textured source gives an identical penalty (while `TV(x_pre)` moves
>3x); at equal residual RMS a checkerboard costs >20x a smooth ramp while L1 cannot tell them
apart; and 0.03 is asserted decorative so it cannot be copied back in.

**In flight:** `gamma_res` at 10 / 30 / 100 -- sized to 1.9% / 5.6% / 18.5% of the task term
rather than transplanted. Thresholds pre-registered in `Desktop/vcm-additive-2026-08-31.txt`
§13 before the data existed. Even a full PASS lands at k~0.5 (-3.82%), so this tests whether
the accuracy/cost trade *exists*; it does not by itself reach the target, and §5.3's h265
ceiling of -10.49% at zero cost is unaffected by it.

**Rule for the next person adding a loss term:** compute `coefficient * term / L_task` on one
batch first. Below ~1% of the objective the run returns the baseline. A clean training curve is
not evidence the term did anything -- read the checkpoint, not the loss log.

## 6. Outcome interpretations (decided in advance)

**Pre-registered expectations (pinned BEFORE any number, per 2026-08-31 review):**

- **Transfer haircut:** the teacher panel [r3d_18, mc3_18] deliberately does NOT contain the
  eval analyzer r2plus1d_18 (the universality claim). Lu et al. measure a 25–30% haircut
  for cross-backbone transfer (−22.0% on-teacher → −16.4/−15.7% transferred). Therefore, to
  measure −11% (the certification threshold) on the held-out analyzer, the on-teacher number
  must be ≈ −15%, i.e. **Δacc ≈ +0.065, not +0.05**. Evaluate the screen against this bar.
- **The untested variable is CAPACITY, not steps:** D10 multiplies data 6× (1,442 → ~8.6k
  clips) and steps ~7× (901 → ~6,483), but the model stays at 9,795 parameters vs Lu et al.'s
  9.42M (962×). At this size, 6.5k steps most likely converge — so a flat D10 result does
  NOT close the additive branch; it opens the capacity question (arm B / width becomes the
  next experiment, not a retraction of the mechanism).
  **WRONG, and §5.2 says why: D10 was flat because its training proxy was 10–14 dB below the
  codec it stands in for, so no capacity would have helped — a wider model fits the destroyed
  target better, which is the disease, not the cure. Arm B is NOT the next experiment; the rerun
  at the fixed calibration is. Do not read D10's flatness as a capacity datapoint.**

Verdicts:

- **Measured BD ≤ −11% both codecs, gap pass** → the method paper is alive; replicate seed; add
  tracking task; move to publication checklist.
- **−4…−8% territory** → scale ceiling is real at 10.8K clips; report honestly; the neutrality
  paper gains the additive-scale datapoint; decide then whether a 2nd seed / arm B is justified.
- **≈ 0 ± 3% or worse** → R-D neutrality holds even for the additive family at this scale —
  **read as a capacity datapoint, not a mechanism refutation** (see above); the
  neutrality/boundary-condition paper gets its capstone — write it.
- **Δbpp(s=1) far worse than the additive run's (+14.8%→+4%)** → the virtual-codec distortion
  channel pushed the edit the wrong way. **This diagnostic has ALREADY fired, before Phase 2
  returned any bpp**: the edit statistics in §5 put D10 at 12.2 dB against best.pt's 37.7 dB.
  So a large positive Δbpp at s=1 is now the EXPECTED reading, not a surprise, and it is not
  by itself evidence against the mechanism — it is evidence that the useful operating point is
  s ≈ 0.02–0.05. What would count against the mechanism is Δbpp > 0 at *every* grid point, or
  gap collapsing to ≤ −0.03 at the s where Δbpp finally turns negative (i.e. the accuracy gain
  is NOT separable from its bit cost — the one genuinely open question this screen answers).

## 7. Risks & mitigations

- **Early residual blow-up** (no `max_relative_edit` envelope, faithful to Zhao): mu=10 L_D
  active from step 0 + `clip_grad 1.0` + CPU smoke gate.
- **AMP numerics** with the DCT proxy: flag `train.amp`, default **off**; enable only if the
  50-step fp32-vs-AMP loss curves agree; fp32 fallback = 6 epochs (already 6–10× the run).
- **Session death at hour 9**: resume=true + last.pt checkpointing; auto.py retry.
- **Val-cost creep** (all-QP val): val_max_batches=5 caps it at ~100 forwards/epoch.
- **Non-comparability with the 112-clip analytic table**: accepted by design (self-consistent
  128 round; anchor vs prep at identical settings).

## 8. Non-goals (the forbidden list holds)

No STE/real-forward 4th attempt; no best.pt 224 push; no pricing on the unidentified rebuild;
no r144; no per-codec QP sub-grids; no low-QP branch; no new resample sizes; no rate-allocation/
QP-field; no tstack; no `robust_transfer_ste.yaml`. Single seed first (a 2nd seed only on a
promising Phase 3, before any paper claim). Tracking task and frame-224 stay out of this round.

## 9. Decisions requested at sign-off

1. Approve the plan + ~13–16 GPU-h spend (of ~56h pooled)?
2. Teachers: panel [r3d_18, mc3_18] + held-out r2plus1d_18 (recommended) or single r3d_18?
3. AMP: implement + smoke-gate (recommended), or skip (fp32, 6 epochs)?
4. Arm B (width-32, same topology) queued only if arm A shows saturation — OK to leave out now?
