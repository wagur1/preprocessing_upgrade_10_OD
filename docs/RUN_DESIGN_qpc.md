# Round (b) — QP-Conditioned Additive Edit

**Status: pre-registered 2026-09-04 evening, before any run. Commit: TBD (this
doc + implementation land together). Trainer account: TBD (post quota-refresh
2026-09-05 07:00 local). Config: `configs/additive_qpc.yaml` (fork of
additive_ar; the kappa=10 lineage config stays untouched).**

**Pre-launch revision 2026-09-04 late evening (cross_review C7, Claude's
attack answered): premise CORRECTED — "earns accuracy only at QP45/50" was
D10/113-clip screen phrasing; on the canonical 1159-clip kappa=10 store the
gap is strictly positive at every QP. The 0-GPU audit (u6_big4/c5c_audit.py,
2-shard canonical store) measured gap/Δbpp (×10⁻³): h264 1.58 / 4.46 / 5.86 /
4.89 / 6.44; h265 1.44 / 4.29 / 9.58 / 14.74 / 12.52 (QP30→50) — QP30 is
4.1× (h264) to 10.2× (h265) less gap-per-added-bit than the best QP. The
reallocation target exists; the round is NOT null by construction.**

## The measured failure this round attacks

kappa=10 (project best) is *rate-UNTARGETED* — the corrected canonical read
(cross_review C7): the edit earns accuracy at EVERY QP (all gaps positive,
worst +0.022 h264 / +0.018 h265) and costs bits at every QP, but the
exchange rate is 4–10× worse at QP30 than at the heavy QPs: Δbpp
+13.1/+11.5% @QP30 → +6.0/+1.8% @QP50 while the gap per added bit is
smallest exactly at QP30. At QP30 we pay the most bits for the least gap.
Every dead amplitude lever (omega, kappa_t, mu=3) tried to make the edit
CHEAPER or BIGGER; none tried to make it *conditional on where in the R-D
curve the codec is*.

## Mechanism

Give the existing 9,795-param additive editor a rate condition (FiLM, zero-init)
so it can spend edit amplitude where it pays:

- **Low QP (light compression):** the codec preserves the source well; the
  accuracy gap is small. The optimal edit is SMALL — stop paying bits for a
  gap the codec itself already covers.
- **High QP (heavy compression):** the codec destroys analyzer-relevant
  detail; the gap is large. The edit should concentrate here — the same
  accuracy gain per bit costs fewer bits per unit gap when the codec is
  quantising hard (its own rate cost is low there).

The BD integral then improves on BOTH ends: fewer added bits at the high-rate
end of the anchor curve, larger effective gap at the low-rate end. This is a
bit-ALLOCATION lever, not an amplitude lever — the axis the failure analysis
says was never tried ("all tried bit-levers were GLOBAL penalties; the
targeting axis — QP-conditional / spatial — never measured").

Zero-init FiLM makes conditioned == unconditioned at step 0: training can only
DEVIATE from the kappa=10 optimum if the loss gradient says so. Downside is
bounded by the incumbent's own behavior; the risk is it never deviates (a
null result, see expectations).

## Design decisions (each with a reason)

1. **FiLM on the shared trunk, not per-branch.** One `FiLM(cond_dim=1, ch=16)`
   between fusion and to_rgb: conditions the single 16-ch representation both
   branches feed. Per-branch FiLM doubles the new parameters for a
   condition signal that is a scalar per clip — one shared trunk FiLM is the
   minimal, cheapest conditioning point.
2. **cond = qp_norm(qp) in [0,1]** (1 = QP51 = heavy compression), same
   `_rate_cond`/`_qp_norm` the engine already builds for the U-Net path — no
   engine changes needed.
3. **gate input: unmodified cond, zero-init FiLM.** The condition enters
   additively inside FiLM's MLP (Linear(1→16)→LeakyReLU→Linear(16→32)); the
   second Linear is zero-init so γ=β=0 at init → exact identity vs the
   unconditioned model → the *unconditioned* 9,795-key checkpoints
   strict-load into the base keys (state_dict gains 4 new keys:
   `film.0.{weight,bias}`, `film.2.{weight,bias}`; verified by unit test).
4. **Param budget: 9,795 → 10,371 (+576, +5.9%).** Negligible capacity
   change; capacity was never the binding constraint (the 224 retrain showed
   the LOSS BALANCE shifts, not capacity, at 3× pixels).
5. **No saliency mask this round (spatial targeting deferred).** One new
   variable at a time — the standing project rule. QP-conditioning is the
   temporal/rate axis; spatial gating (use_task_mask + masked_tv) is a
   SEPARATE round (b2) only if (b) lands center/upside. Registered follow-up
   condition, not a promise.
6. **Eval feeds the TRUE QP of each encode.** The canonical eval loop
   (mk_eval_kernel cell) already knows the QP per arm — it passes
   `cond=_rate_cond(qp_norm(qp))` to the conditioned model. QP30 arm gets
   cond 0.303, QP50 gets 0.645 (qp_ref [20,51]).
7. **Train config = kappa=10 lineage except the arch key.** 16ep, batch 8,
   128, seed 0, teacher panel [r3d_18, mc3_18] sampled, mu=10, kappa=10,
   beta=0.001, clip_grad 1.0, cosine, patience 4. The ONLY deltas:
   `model.arch: additive_cond`, `out_dir: outputs/additive_qpc`.

## Pre-registered expectations (written before launch)

Comparators (kappa=10 lineage, 2-shard): h264 −3.42 [−5.88,−0.88] /
rep1 −2.52 [−4.92,+0.16]; h265 −2.63 [−4.49,−0.79] / rep1 −2.33 [−4.14,−0.57].
Read with ±1pp re-run noise (rep1 finding).

- **Center (~35%): h264 −2.5…−3.5 / h265 −2.3…−3.2** — net wash vs kappa=10:
  the condition learns something small but the incumbent is already
  near its family's ceiling.
- **Upside (~15%): h264 −4…−5 / h265 −3…−4.2** — the edit genuinely
  concentrates at high QP: Δbpp@QP30 drops toward ≤+10% while QP45/50 gaps
  hold or grow. Mechanism signature (checkable in the store rows): the
  per-QP Δbpp curve flattens downward at the light end.
- **Downside (~50%): h264 −1.5…−2.5 / h265 −1.5…−2.5** — FiLM trains as a
  global amplitude nuisance (batch-level noise), or the model cannot infer
  "the codec will be heavy" from a scalar well enough within 16ep, and the
  extra 336 params just add variance. 5/5 prior accuracy attempts landed
  downside; this prior dominates by count.

## Gates before any eval spend (0 GPU, local)

Gates run on `dl_qpc/.../preprocessor.pth` with `edit_size_qpc.py` /
`added_hf_qpc.py` (new; feed cond explicitly, default cond=QP45-normalized
0.613 — the operating regime the edit is FOR). **Standing rule (cross_review
C6(c), adopted): every gate threshold is printed next to the incumbent's
value from the SAME script in the SAME run — a gate whose reference value is
not in the log is not a gate.**

1. **Regime gate:** added HF @s=0.25 ∈ −5…+30% at BOTH cond=QP30 (0.303) and
   cond=QP50 (0.645). The wide band mirrors mu=3's registration: a
   conditioned edit may legitimately grow.
2. **Conditionability gate — BEHAVIOURAL, threshold registered pre-run
   (cross_review C7(c)): added-HF AND RMS at cond=QP30 vs cond=QP50 must
   differ by MORE than the edit-level twin noise (RMS ±3%, from rep1's twin
   0.0452 vs 0.0467).** Within ±3% ⇒ null-by-non-utilization: read as
   "conditioning unused at this budget", NOT "targeting falsified". ‖γ‖,‖β‖
   per cond are report-only (a norm can move while the edit does not).
3. **No-blow-up gate:** RMS @s=1.0 < **0.14** (incumbent mu=10 twin reads
   **0.1202** on the same script — printed together per the standing rule;
   threshold set ABOVE the incumbent's own readout, Q3(iii) hygiene).
4. **Identity at init:** untrained conditioned model with any cond is exact
   identity (zero-init FiLM + zero-init to_rgb) — unit test, not a gate.

**Gate discipline:** regime gate (1) must PASS on both conds, else NO eval.
RMS > 0.14 = automatic no-eval. Gate (2) has no pass/fail on the round — it
selects the NULL INTERPRETATION (unused vs falsified), and gate (2)'s spread
number defines the prior for any round b2.

The eval kernel's per-QP conditioning (5 prep forwards/clip) is an OPS note
for shard runtime budgeting only — NOT a gate (cross_review C7(d): BD at
matched protocol is the comparison axis; no competitor is charged
preprocessor forwards).

## Guards (registered before the run)

- **(i) 1D-axis discipline:** this is the QP-axis point at kappa=10/mu=10
  fixed. A downside landing does NOT close "targeting" as a direction (spatial
  gating remains unmeasured); an upside landing does not mean the (QP ×
  spatial) interaction is additive.
- **(ii) FiLM-utilization audit before quoting any mechanism:** report
  ‖γ‖,‖β‖ at the two eval conds from the checkpoint — if the network
  effectively zeroed the FiLM, the run is a re-trained kappa=10 (null
  conditionability), and the honest read is "conditioning available but not
  used at this budget", not "targeting failed".
- **(iii) Same-instrument comparisons only:** all comparators from the same
  edit_size/added_hf script family; the mu=3 letter-fail lesson.
- **(iv) One run, one variable:** any hyperparameter drift beyond the
  registered deltas voids the comparison — the run config diff must be
  exactly {arch, out_dir}.
- **(v) Pre-declared escalation (round b', cross_review C7(b) rebuttal):
  IF this round returns null-with-FiLM-USED** (behavioural gate 2 shows
  >±3% spread — the conditioning works but BD does not move), the follow-up
  is BD-integration-weighted QP sampling in the trainer (weight QP draws by
  the h264 segment weights 0.178/0.252/0.324/0.246) — sharpening the
  objective's tilt toward BD's own weighting. NOT adopted now: guard (iv)
  pins the diff to {arch, out_dir}, and the objective ALREADY tilts the
  intended way (mu·L_D edit-share ≈ 40–70% @QP30 vs 7–11% @QP50 — the
  QP-structure of the loss encodes the reallocation reward the unconditioned
  model cannot express).
- **(vi) Round (c) registered (cross_review C7(e)/C8): hard δ-cap
  (clamp(residual, ±2/255), Kelvin-style CONSTRAINT not penalty) is the
  spatial-axis sibling of this bet** — same "allocation not amplitude"
  family. Its 0-GPU pre-pricing (clip the EXISTING kappa=10 residual at
  ±2/255, measure real x264/x265 bpp) runs before that session; its
  pre-registration OPENS with the counter-evidence (s=0.02 = −0.10/−0.52
  CI-span-0 — our own measurement, sitting in the δ-cap amplitude band) and
  carries the mandatory on-teacher vs held-out adversarial gate.

## Ops plan (morning, post quota-refresh 07:00 local 2026-09-05)

1. `python -X utf8 mk_train_kernel.py <account> <commit> "train.epochs=16"
   additive_qpc.yaml` (trainer account chosen from the refreshed pool;
   htran123456-by-default unless its quota is still drained).
2. Watcher: background shell + auto_resume.log pattern has died silently 6×
   on this machine (Rule 3). Use the durable-cron backstop ONLY (hourly
   status check → relaunch watcher if kernel mid-run; on COMPLETE → gates).
3. On COMPLETE: pull → local gates (above) → report to user → USER decides
   eval spend (1-shard direction screen first, as with mu=3).
