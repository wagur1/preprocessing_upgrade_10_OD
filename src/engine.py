"""Training and evaluation engine.

Training (only the preprocessor learns; codec + analyzer frozen):

    clip x -> preprocessor(x, cond) -> x_pre -> frozen training proxy -> x_hat, bpp
    L = lam_task*L_task + omega*L_distill + beta*bpp + tau*L_temp   (see losses.py)

A QP is sampled per step and mapped to a proxy quality; its normalised value is
fed to the preprocessor's FiLM so one model spans the whole rate range. The
default objective has no MSE-to-source term (it fought compression).

Two training codecs (``codec.kind``):
  * ``virtual`` (stage 1) -- fully differentiable block-DCT proxy, with the A3
    soft->hard quantiser anneal ramped over training so it ends at the codec's
    real hard quantiser.
  * ``ste`` (stage 2, A3) -- ``STECodec`` runs the *real* x264/x265 in the forward
    pass and borrows the proxy gradient in the backward pass, so the loss is
    computed on the true reconstruction + true coded rate (Lu et al. 2206.05650
    measured forward-real-codec at -20.3% vs -14.6% for a proxy used both ways).
    ffmpeg-per-step is slow -> a short calibration fine-tune resumed on top of a
    proxy-pretrained checkpoint, not a from-scratch trainer.

Two tasks dispatch on ``cfg['task']['name']``:

  * ``action_recognition`` -- Kinetics-400 clips, cross-entropy L_task, top-1
    accuracy at eval.
  * ``tracking``           -- GOT-10k clips + boxes, SiamFC logistic L_task,
    real success-plot AUC at eval (run the tracker over each sequence).

Evaluation traces real H.264/H.265 rate-accuracy curves by default and reports
same-codec BD-Rate. Proxy curves are optional diagnostics
(``eval.include_proxy=true``), not part of the deployment claim.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .codecs import StandardCodec, ffmpeg_available
from .data import (
    CocoDetDataset,
    GOT10kClipDataset,
    VideoClipDataset,
    collate_clips,
    collate_coco_det,
    collate_got10k,
    iter_sequences,
)
from .losses import LossWeights, preprocessing_loss
from .metrics import aggregate_metrics, bd_metric, bd_rate, sequence_metrics
from .models import (
    AdditiveCondPreprocessor,
    AdditivePreprocessor,
    CompressAICodec,
    STECodec,
    DualCodecSandwich,
    DualPostSandwich,
    PerCodecPostSandwich,
    SandwichPreprocessor,
    UPVCMPreprocessor,
    VideoPreprocessor,
    VirtualCodec,
)
from .models.task_mask import task_saliency
from .tasks import build_task
from .tasks.base import build_analyzer
from .tracking import attach_tracker, make_tracker


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def _device(cfg: dict) -> torch.device:
    want = cfg.get("device", "auto")
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


def _seed_everything(seed: int) -> None:
    """Seed the small set of RNGs used by the training harness."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_models(cfg: dict, device: torch.device, role: str = "train"):
    m = cfg["model"]
    arch = str(m.get("arch", "unet"))
    if arch == "additive":
        # Zhao-style two-branch additive residual (9,795 params), pinned by the
        # additive run's best.pt. Unconditioned and ungated by design; the
        # shared loops still pass cond/mask and this model ignores them.
        # ``strength`` is the eval-time operating point, deliberately NOT an
        # arch key: evaluate() must not restore it from the training config.
        pre = AdditivePreprocessor(
            temporal_frames=int(m.get("temporal_frames", 8)),
            strength=float(m.get("strength", 1.0)),
        ).to(device)
    elif arch == "additive_cond":
        # Round (b) (docs/RUN_DESIGN_qpc.md): same Zhao tree + zero-init FiLM
        # on the shared trunk so the edit can condition on the rate operating
        # point the engine already passes as ``cond``. 10,371 params.
        pre = AdditiveCondPreprocessor(
            temporal_frames=int(m.get("temporal_frames", 8)),
            strength=float(m.get("strength", 1.0)),
            cond_dim=int(m.get("cond_dim", 1)),
        ).to(device)
    elif arch == "sandwich":
        # v8 model (docs/MODEL_SANDWICH.md): UP-VCM PRE + restoration POST
        # around the frozen codec. forward() = PRE; post_restore() = POST.
        pre = SandwichPreprocessor(
            s_ch=int(m.get("s_ch", 16)),
            editor_ch=int(m.get("editor_ch", 24)),
            cond_dim=int(m.get("cond_dim", 1)),
            dino_weight=float(m.get("dino_weight", 0.5)),
            dino_name=str(m.get("dino_name", "dinov2_vits14")),
            motion_tau=float(m.get("motion_tau", 0.1)),
            post_base=int(m.get("post_base", 32)),
            w_budget=float(m.get("w_budget", 0.0)),
            post_temporal=bool(m.get("post_temporal", False)),
        ).to(device)
    elif arch == "dualcodec":
        # v9-b full: per-codec PRE + per-codec POST (the modular union).
        pre = DualCodecSandwich(
            s_ch=int(m.get("s_ch", 16)),
            editor_ch=int(m.get("editor_ch", 24)),
            cond_dim=int(m.get("cond_dim", 1)),
            dino_weight=float(m.get("dino_weight", 0.5)),
            dino_name=str(m.get("dino_name", "dinov2_vits14")),
            motion_tau=float(m.get("motion_tau", 0.1)),
            post_base=int(m.get("post_base", 32)),
            w_budget=float(m.get("w_budget", 0.0)),
        ).to(device)
    elif arch == "dualpost":
        # v9-b (docs/MODEL_PERCODEC.md): TWO independent POST UNets, exact
        # codec switch. No shared trunk, no embedding — the anti-v9a design.
        pre = DualPostSandwich(
            s_ch=int(m.get("s_ch", 16)),
            editor_ch=int(m.get("editor_ch", 24)),
            cond_dim=int(m.get("cond_dim", 1)),
            dino_weight=float(m.get("dino_weight", 0.5)),
            dino_name=str(m.get("dino_name", "dinov2_vits14")),
            motion_tau=float(m.get("motion_tau", 0.1)),
            post_base=int(m.get("post_base", 32)),
            w_budget=float(m.get("w_budget", 0.0)),
        ).to(device)
    elif arch == "percodec_sandwich":
        # v9 (docs/MODEL_PERCODEC.md): UP-VCM PRE + codec-conditioned POST.
        # Same interface as sandwich; post_restore(codec=...) routes FiLM.
        pre = PerCodecPostSandwich(
            s_ch=int(m.get("s_ch", 16)),
            editor_ch=int(m.get("editor_ch", 24)),
            cond_dim=int(m.get("cond_dim", 1)),
            dino_weight=float(m.get("dino_weight", 0.5)),
            dino_name=str(m.get("dino_name", "dinov2_vits14")),
            motion_tau=float(m.get("motion_tau", 0.1)),
            post_base=int(m.get("post_base", 32)),
            w_budget=float(m.get("w_budget", 0.0)),
        ).to(device)
    elif arch == "upvcm":
        # v7 NEW MODEL (docs/MODEL_UPVCM.md): self-sufficient importance head
        # (distilled from teacher saliency + DINOv2 energy) driving three
        # zero-init-gated modules — background decimation, W-gated ROI editor
        # with FiLM(QP), temporal background stabilisation. Identity at init;
        # analyzer-free at deploy.
        pre = UPVCMPreprocessor(
            s_ch=int(m.get("s_ch", 16)),
            editor_ch=int(m.get("editor_ch", 24)),
            cond_dim=int(m.get("cond_dim", 1)),
            dino_weight=float(m.get("dino_weight", 0.5)),
            dino_name=str(m.get("dino_name", "dinov2_vits14")),
            motion_tau=float(m.get("motion_tau", 0.1)),
            w_budget=float(m.get("w_budget", 0.0)),
        ).to(device)
    elif arch == "unet":
        pre = VideoPreprocessor(
            base_ch=m.get("base_ch", 32),
            res_scale=m.get("res_scale", 1.0),
            cond_dim=m.get("cond_dim", 1),
            max_relative_edit=m.get("max_relative_edit", 0.25),
            gate=bool(m.get("gate", True)),
            gate_area=float(m.get("gate_area", 0.0)),
            edit_kind=str(m.get("edit_kind", "residual")),
        ).to(device)
    else:
        raise ValueError(f"model.arch must be 'unet', 'additive', 'additive_cond', 'upvcm', 'sandwich', 'percodec_sandwich', 'dualpost' or 'dualcodec', got {arch!r}")
    cc = cfg["codec"]
    kind = cc.get("kind", "compressai")
    if kind == "entropy":
        # D8: block-DCT proxy whose rate is a TRAINED Laplacian factorized
        # prior (per DCT position) instead of the parameter-free Gaussian-
        # power formula — the 5 learning-failure proofs showed the latter's
        # gradient dies below the quantiser step, so no configuration could
        # learn smoothing strength. The prior learns the codec's true symbol
        # distribution online (see src/models/entropy_codec.py).
        from .models.entropy_codec import LearnedRateCodec
        proxy = LearnedRateCodec(
            qualities=tuple(cc.get("qualities", [1, 2, 3, 5, 8])),
            block=cc.get("block", 8),
            q_steps=cc.get("q_steps"),
            step_coarse=cc.get("step_coarse", 0.25),
            step_fine=cc.get("step_fine", 0.03),
            inter=cc.get("inter", True),
            colorspace=cc.get("colorspace", "yuv420"),
            chroma_step_scale=cc.get("chroma_step_scale", 2.0),
            n_components=cc.get("entropy_components", 3),
        ).to(device)
    elif kind in ("virtual", "ste"):
        # block-transform proxy matched to x264/x265 geometry (Zhao et al.);
        # 5.1 C1: default yuv420 colourspace matches -pix_fmt yuv420p.
        proxy = VirtualCodec(
            qualities=tuple(cc.get("qualities", [1, 2, 3, 5, 8])),
            block=cc.get("block", 8),
            q_steps=cc.get("q_steps"),
            step_coarse=cc.get("step_coarse", 0.25),
            step_fine=cc.get("step_fine", 0.03),
            inter=cc.get("inter", True),
            colorspace=cc.get("colorspace", "yuv420"),
            chroma_step_scale=cc.get("chroma_step_scale", 2.0),
        ).to(device)
    else:
        proxy = CompressAICodec(
            model=cc.get("model", "bmshj2018-factorized"),
            qualities=tuple(cc.get("qualities", [1, 2, 3, 4, 5, 6, 7, 8])),
            pretrained=True,
            trainable=False,
        ).to(device)
    if kind == "ste":
        # A3: real x264/x265 in the forward pass, proxy gradient in the backward.
        # quality->qp is the inverse of the training qp_to_quality map.
        q2qp = {}
        for qp, q in (cfg["train"].get("qp_to_quality") or {}).items():
            q2qp.setdefault(int(q), []).append(int(qp))
        q2qp = {q: int(sum(v) / len(v)) for q, v in q2qp.items()}
        codec = STECodec(
            proxy,
            codec=cc.get("ste_codec", "h265"),
            quality_to_qp=q2qp,
            preset=cc.get("ste_preset", "medium"),
        )
        if cc.get("ste_alternate", False):
            # v9 per-codec POST: each training step flips the REAL codec so
            # the codec-conditioned restorer sees both artifact families.
            codec.alternate = True
    else:
        codec = proxy
    analyzer = build_analyzer(cfg, role=role).to(device)
    return pre, codec, analyzer


def _optimizer(pre, tr):
    # Only trainable parameters: ``train.freeze_except`` may have frozen a
    # prefix, and Adam would otherwise keep (no-op) state for frozen tensors.
    params = [p for p in pre.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=tr.get("lr", 1e-4))


def _repair_dead_editor(model) -> int:
    """Re-apply the repaired init to any all-zero M2 editor output conv.

    ``load_state_dict`` from a pre-2026-09-16 checkpoint restores the exact-zero
    output conv that put the ROI editor into its dead saddle, which would make a
    warm-start probe dead on arrival. An all-zero output conv is dead by
    definition (see ``UPVCMPreprocessor``), so repairing it is always safe and
    idempotent: it can only fire on a module that provably cannot contribute.
    """
    repaired = 0
    for name, mod in model.named_modules():
        if name.endswith("editor.out") and isinstance(mod, torch.nn.Conv2d):
            if mod.weight.abs().max().item() == 0.0:
                torch.nn.init.normal_(mod.weight, std=1e-3)
                torch.nn.init.zeros_(mod.bias)
                repaired += 1
    return repaired


def _load_state_compat(model, state: dict) -> list:
    """Strict load, falling back to a REPORTED partial load for new branches.

    A warm start into an architecture that gained a branch (e.g. the temporal
    POST input) legitimately misses keys — they are new parameters sitting at
    their init. Missing keys are therefore allowed and returned; UNEXPECTED keys
    (the checkpoint carries something this model does not have) would be dropped
    silently, so they are raised as an error.
    """
    try:
        model.load_state_dict(state)
        return []
    except RuntimeError as exc:
        res = model.load_state_dict(state, strict=False)
        if res.unexpected_keys:
            raise RuntimeError(
                "checkpoint has unexpected keys that would be dropped: "
                f"{list(res.unexpected_keys)[:8]}") from exc
        return list(res.missing_keys)


def _reinit_modules(model, prefixes) -> int:
    """Re-initialise the default weights of modules matching any prefix.

    A warm-start probe loads a checkpoint whose PRE carries modules that are
    unusable by construction: the M2 editor output conv is exactly zero (dead
    saddle, repaired separately) and the S head is saturated (logits ~ -60, so
    W ~ 1e-9 and every W-gated mechanism is effectively off). ``load_state_dict``
    restores that state, so a probe that intends to TRAIN those modules has to
    reset them first. A name matches on a full segment anywhere in the path, so
    ``s_net`` finds ``pre.s_net``.
    """
    n = 0
    for name, mod in model.named_modules():
        if any(name == p or name.endswith("." + p) or f".{p}." in name
               for p in prefixes):
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()
                n += 1
    return n


def _ckpt_path(cfg: dict) -> Path:
    d = Path(cfg.get("out_dir", "outputs")) / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d / "preprocessor.pth"


# --------------------------------------------------------------------------
# rate conditioning: the operating point fed to the preprocessor's FiLM
# --------------------------------------------------------------------------
def _qp_norm(qp: float, cfg: dict) -> float:
    """Map an x26x QP to a normalised compression level in [0, 1] (1 = most
    compressed). ``model.qp_ref`` sets the reference range spanning train+eval."""
    lo, hi = cfg["model"].get("qp_ref", [20, 51])
    return min(max((float(qp) - lo) / (hi - lo), 0.0), 1.0)


def _quality_conds(cfg: dict, codec: CompressAICodec) -> Dict[int, float]:
    """FiLM condition per CompressAI quality, matching what TRAINING used.

    Training samples a QP and feeds ``cond = qp_norm(qp)`` while running the proxy
    at ``q = qp_to_quality[qp]``. Eval must feed the *same* condition per quality,
    so we invert that map: quality -> mean training QP -> qp_norm. Qualities never
    seen in training (e.g. a higher-rate quality) are extrapolated linearly on the
    (quality, mean-QP) training points, then qp_norm-clamped to [0,1].

    (The old eval used a separate ``_quality_level`` normalisation, so quality 5
    was trained at cond 0.065 but evaluated at 0.429 -- a train/eval mismatch that
    made the proxy preprocessing curve not actually in-domain.)"""
    from collections import defaultdict

    qtq = {int(qp): int(q) for qp, q in cfg["train"]["qp_to_quality"].items()}
    by_q: Dict[int, list] = defaultdict(list)
    for qp, q in qtq.items():
        by_q[q].append(qp)
    q2qp = {q: sum(v) / len(v) for q, v in by_q.items()}   # mean training QP per quality

    qs = sorted(q2qp)
    if len(qs) >= 2:  # least-squares line qp ~ a*quality + b for extrapolation
        n, sx = len(qs), sum(qs)
        sy = sum(q2qp[q] for q in qs)
        sxx = sum(q * q for q in qs)
        sxy = sum(q * q2qp[q] for q in qs)
        denom = (n * sxx - sx * sx) or 1.0
        a = (n * sxy - sx * sy) / denom
        b = (sy - a * sx) / n
    else:
        a, b = 0.0, (next(iter(q2qp.values())) if q2qp else 30.0)

    return {q: _qp_norm(q2qp.get(q, a * q + b), cfg) for q in codec.qualities}


def _rate_cond(level: float, batch: int, device, dtype) -> torch.Tensor:
    """Build the [B, cond_dim] condition vector. Currently a single normalised
    rate level; append a log target-rate here for explicit rate control."""
    return torch.full((batch, 1), float(level), device=device, dtype=dtype)


def _training_codec_setup(tr: dict, codec: CompressAICodec):
    """Validate the QP list and its QP->proxy-quality mapping used at train time.

    (For ``codec.kind: ste`` the same map is inverted to drive the real x264/x265
    encoder's QP in the STE forward pass; see ``_build_models`` and ``STECodec``.)
    """
    qp_list = [int(qp) for qp in tr.get("qp_list", codec.qualities)]
    if not qp_list:
        raise ValueError("train.qp_list must contain at least one QP")

    raw_mapping = tr.get("qp_to_quality")
    if raw_mapping is None:
        qp_to_quality = {q: q for q in qp_list}
    else:
        qp_to_quality = {int(qp): int(q) for qp, q in raw_mapping.items()}

    missing = [qp for qp in qp_list if qp not in qp_to_quality]
    if missing:
        raise ValueError(f"train.qp_to_quality is missing QPs: {missing}")
    unavailable = sorted({qp_to_quality[qp] for qp in qp_list} - set(codec.qualities))
    if unavailable:
        raise ValueError(f"proxy qualities {unavailable} are not in codec.qualities")

    if raw_mapping is not None:
        ordered = [qp_to_quality[qp] for qp in sorted(qp_list)]
        if any(a < b for a, b in zip(ordered, ordered[1:])):
            raise ValueError("train.qp_to_quality must be monotonic: higher QP -> lower quality")
    return qp_list, qp_to_quality


# --------------------------------------------------------------------------
# training (dispatch)
# --------------------------------------------------------------------------
def _earlystop_update(val, best, min_delta, no_improve, patience):
    """One val observation -> (improved, best, no_improve, stop).

    A drop of at least ``min_delta`` below ``best`` counts as improvement and
    resets the patience counter; otherwise ``no_improve`` grows and we stop once
    it reaches ``patience`` (patience=0 disables early stopping)."""
    if val < best - min_delta:
        return True, val, 0, False
    no_improve += 1
    return False, best, no_improve, bool(patience and no_improve >= patience)


@torch.no_grad()
def _val_loss(pre, codec, analyzer, loader, weights, qp_list, qp_to_quality,
              cfg, prep_batch, max_batches):
    """Mean proxy-only validation loss over every configured training QP.

    Selecting a checkpoint at one QP can over-specialise the FiLM condition and
    makes the other rate points look worse.  Average the same objective across
    the full configured QP grid while keeping the proxy-only path cheap."""
    was_training = pre.training
    pre.eval()
    total, nb = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        clips, target = prep_batch(batch)
        qp_losses = []
        for qp in qp_list:
            q = qp_to_quality[qp]
            cond = _rate_cond(_qp_norm(qp, cfg), clips.shape[0], clips.device, clips.dtype)
            # A2: use the SAME spatial mask objective as training and pin the
            # sampled teacher so saliency/loss share one active teacher.
            if weights.use_task_mask and hasattr(analyzer, "pin_active"):
                analyzer.pin_active()
            try:
                mask = task_saliency(analyzer, clips, target) if weights.use_task_mask else None
                if "codec" in pre.forward.__code__.co_varnames:
                    x_pre = pre(clips, cond, mask=mask, codec=_ste_codec_name(codec))
                else:
                    x_pre = pre(clips, cond, mask=mask)
                x_hat, bpp = codec(x_pre, q)
                if hasattr(pre, "post_restore"):
                    _codec_name = _ste_codec_name(codec)
                    x_hat = pre.post_restore(x_hat, cond, codec=_codec_name) if hasattr(pre.post_restore, "__call__") and "codec" in pre.post_restore.__code__.co_varnames else pre.post_restore(x_hat, cond)
                parts = preprocessing_loss(analyzer, clips, x_hat, bpp, target, weights,
                                           x_pre=x_pre, task_mask=mask,
                                           saliency_pred=getattr(pre, "_last_w", None),
                                           saliency_target=getattr(pre, "_last_w_target", None))
            finally:
                if weights.use_task_mask and hasattr(analyzer, "unpin_active"):
                    analyzer.unpin_active()
            qp_losses.append(parts["loss"].item())
        total += sum(qp_losses) / max(len(qp_losses), 1)
        nb += 1
    if was_training:
        pre.train()
    return total / max(nb, 1)


def train(cfg: dict) -> str:
    _seed_everything(int(cfg.get("seed", 0)))
    if cfg["task"]["name"] == "tracking":
        return _train_tracking(cfg)
    if cfg["task"]["name"] == "object_detection":
        return _train_detection(cfg)
    return _train_classification(cfg)


def _fit(cfg, pre, codec, analyzer, train_loader, val_loader, prep_batch,
         tag: str, n_train: int) -> str:
    """Shared training loop: cosine LR, per-epoch val, best/last checkpoints,
    early stopping, and resume. ``prep_batch(batch) -> (clips, acc_loss_fn)``."""
    device = next(pre.parameters()).device
    tr = cfg["train"]
    lw = cfg.get("loss", {})
    weights = LossWeights(
        lam_task=lw.get("lam_task", 1.0),
        omega=lw.get("omega", 0.5),
        beta=lw.get("beta", 0.1),
        tau=lw.get("tau", 0.1),
        delta=lw.get("delta", 0.0),
        gamma=lw.get("gamma", 0.0),
        gamma_res=lw.get("gamma_res", 0.0),
        kappa=lw.get("kappa", 0.0),
        kappa_t=lw.get("kappa_t", 0.0),
        mu=lw.get("mu", 0.0),
        use_task_mask=bool(lw.get("use_task_mask", False)),
    )
    # Optional parameter freeze (bounded probes): keep a pretrained model fixed
    # and train ONLY the parameters whose names start with one of the given
    # prefixes. ``train.freeze_except: [editor, edit_strength]`` unlocks the M2
    # editor on a converged checkpoint and trains nothing else, so the probe's
    # starting point is exactly the record model and any BD change is
    # attributable to the newly unlocked module alone.
    freeze_except = list(tr.get("freeze_except") or [])
    if freeze_except:
        # Prefix matches either the start of the parameter name or a dotted
        # segment inside it, so the config can say ``editor`` / ``edit_strength``
        # without knowing whether the arch nests them under ``pre.`` (sandwich),
        # ``pre_264.`` (per-codec) or nothing at all.
        def _keeps(name: str) -> bool:
            return any(name.startswith(p) or f".{p}" in name for p in freeze_except)

        trainable = frozen = 0
        for name, p in pre.named_parameters():
            keep = _keeps(name)
            p.requires_grad_(keep)
            if keep:
                trainable += p.numel()
            else:
                frozen += p.numel()
        print(f"[train] freeze_except={list(freeze_except)} -> "
              f"{trainable:,} trainable / {frozen:,} frozen params", flush=True)
    opt = _optimizer(pre, tr)
    # D8: the learned-rate prior carries its own Adam so its tiny tables can
    # move at a different pace than the U-Net (and survive cosine decay of
    # the preprocessor's LR). Phase discipline (Ballé-style): the prior
    # calibrates on RAW content first (preprocessor held at identity), then
    # FREEZES — otherwise it co-adapts to the edited content, learns that
    # "everything is cheap", and the rate gradient stops distinguishing
    # blurred from sharp (the ent50 failure mode).
    rate_opt = None
    rate_freeze_after = int(tr.get("rate_freeze_after", 0))
    if hasattr(codec, "rate_params"):
        rate_opt = torch.optim.Adam(codec.rate_params.parameters(), lr=1e-3)
    epochs = int(tr.get("epochs", 5))
    max_steps = tr.get("max_steps", None)
    # Gradient clipping (the additive run trained with clip_grad=1.0; the
    # Zhao loss has no bounded-edit envelope, so this is the safety net).
    clip_grad = float(tr.get("clip_grad", 0.0))
    qp_list, qp_to_quality = _training_codec_setup(tr, codec)

    total_steps = len(train_loader) * epochs
    if max_steps:
        total_steps = min(total_steps, int(max_steps))
    use_cosine = bool(tr.get("cosine", True))
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1))
             if use_cosine else None)
    patience = int(tr.get("patience", 0))
    min_delta = float(tr.get("min_delta", 1e-4))
    val_max_batches = tr.get("val_max_batches", 20)
    ckpt_path = _ckpt_path(cfg)                    # best (what evaluate loads)
    last_path = ckpt_path.with_name("preprocessor_last.pth")

    start_epoch, step, best_val, no_improve = 0, 0, float("inf"), 0
    finetune = bool(tr.get("finetune", False))
    if finetune:
        # A3 Stage 2 (real-codec calibration): load ONLY the preprocessor weights
        # from the best Stage-1 checkpoint and start a FRESH short schedule --
        # config LR (train.lr), step counter from 0 (so train.max_steps means
        # "extra calibration steps"), and a new cosine over this run. Optimizer /
        # scheduler / epoch state are deliberately NOT restored: restoring them
        # would reinstate Stage-1's cosine-decayed LR (~0) and a global_step far
        # above max_steps, so the fine-tune would run a single batch and stop.
        src = ckpt_path if ckpt_path.exists() else last_path
        if src.exists():
            sd = torch.load(src, map_location=device)
            missing = _load_state_compat(pre, sd["model"] if "model" in sd else sd)
            repaired = _repair_dead_editor(pre)
            print(f"[train] fine-tune: loaded weights from {src} "
                  f"(fresh optimizer, lr={tr.get('lr')}, {epochs} epoch(s), "
                  f"max_steps={max_steps})")
            if missing:
                print(f"[train]   compatible load: {len(missing)} new parameter(s) "
                      f"left at init (e.g. {missing[:4]})")
            if repaired:
                print(f"[train]   dead-saddle repair: re-initialised {repaired} "
                      f"all-zero editor output conv(s) from the checkpoint")
            reinit = list(tr.get("reinit_prefixes") or [])
            if reinit:
                n_reinit = _reinit_modules(pre, reinit)
                print(f"[train]   reinit_prefixes={reinit}: reset {n_reinit} "
                      f"module(s) to default init (checkpoint state unusable)")
        else:
            print("[train] fine-tune requested but no Stage-1 checkpoint found -> "
                  "training from scratch")
    elif bool(tr.get("resume", False)) and last_path.exists():
        st = torch.load(last_path, map_location=device)
        if tag == "object_detection" and (
                st.get("cfg", {}).get("data", {}).get("od_split_fingerprint")
                != cfg["data"]["od_split_fingerprint"]):
            raise ValueError("Incompatible OD checkpoint split identity; start a fresh run")
        if val_loader is not None and not ckpt_path.exists():
            raise ValueError("Cannot resume with validation: historical best checkpoint "
                             f"is missing ({ckpt_path}); restore the paired best and last.")
        pre.load_state_dict(st["model"])
        if st.get("opt"):
            opt.load_state_dict(st["opt"])
        if sched is not None and st.get("sched"):
            sched.load_state_dict(st["sched"])
        start_epoch = st.get("epoch", 0)
        step, best_val = st.get("global_step", 0), st.get("best_val", float("inf"))
        no_improve = st.get("no_improve", 0)
        print(f"[train] resumed {last_path} @ epoch {start_epoch}, step {step}")
    if start_epoch >= epochs:
        print("[train] resume: already at target epochs; nothing to do")
        return str(ckpt_path if ckpt_path.exists() else last_path)

    def _save(path, ep):
        torch.save({"model": pre.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict() if sched is not None else None,
                    "cfg": cfg, "epoch": ep, "global_step": step,
                    "best_val": best_val, "no_improve": no_improve}, path)

    pre.train()
    print(f"[train] {tag} | {n_train} clips | {len(train_loader)} steps/epoch | "
          f"val={'yes' if val_loader else 'none'} | cosine={use_cosine} | "
          f"patience={patience or 'off'} | device={device}", flush=True)
    tracker = make_tracker(Path(cfg.get("out_dir", "outputs")),
                           fallback_name=f"{tag}-seed{cfg.get('seed', 0)}")
    tracker.log_params({"tag": tag, "n_train": n_train, "total_steps": total_steps,
                        "finetune": finetune, "config": cfg})

    stop, epoch = False, start_epoch
    # A3: soft->hard quantiser anneal target for the differentiable proxy (0=off).
    anneal_codec = getattr(codec, "proxy", codec)
    anneal_final = float(cfg.get("codec", {}).get("anneal", 0.0))
    # Regulariser warmup (anti-collapse): at the identity start the frozen analyzer
    # is already near-correct, so L_task's gradient is small -- the always-on
    # smoothing pressures (beta*bpp, gamma*TV, delta*edit) then dominate and drive
    # x_pre to a constant, the analyzer saturates, and the model gets stuck at
    # blank (acc->chance). Ramp those three from 0 over the first frac of steps so
    # task+distill+temporal fix a useful basin first. lam_task/omega/tau stay full.
    warmup_frac = float(tr.get("reg_warmup_frac", 0.3))
    warmup_steps = max(1, int(warmup_frac * total_steps)) if warmup_frac > 0 else 0
    for epoch in range(start_epoch, epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            clips, target = prep_batch(batch)
            if anneal_final and hasattr(anneal_codec, "set_anneal"):
                anneal_codec.set_anneal(anneal_final * min(1.0, step / max(total_steps, 1)))
            warm = 1.0 if warmup_steps == 0 else min(1.0, step / warmup_steps)
            step_w = replace(weights, beta=weights.beta * warm,
                             gamma=weights.gamma * warm, delta=weights.delta * warm)
            qp = random.choice(qp_list)
            # v9 per-codec: alternate the REAL codec each step so the
            # codec-conditioned POST sees BOTH artifact families (a single
            # codec would drift the shared trunk and leave the other
            # codec embedding untrained — the E1 trade-off).
            if getattr(codec, "alternate", False):
                codec.codec = random.choice(("h264", "h265"))
            q = qp_to_quality[qp]
            cond = _rate_cond(_qp_norm(qp, cfg), clips.shape[0], clips.device, clips.dtype)
            # A2: keep the sampled teacher fixed for saliency, task loss, and
            # feature distillation within this step.
            pinned = bool(weights.use_task_mask and hasattr(analyzer, "pin_active"))
            if pinned:
                analyzer.pin_active()
            try:
                mask = task_saliency(analyzer, clips, target) if weights.use_task_mask else None
                if "codec" in pre.forward.__code__.co_varnames:
                    x_pre = pre(clips, cond, mask=mask, codec=_ste_codec_name(codec))
                else:
                    x_pre = pre(clips, cond, mask=mask)
                x_hat, bpp = codec(x_pre, q)
                if hasattr(pre, "post_restore"):
                    _codec_name = _ste_codec_name(codec)
                    x_hat = pre.post_restore(x_hat, cond, codec=_codec_name) if hasattr(pre.post_restore, "__call__") and "codec" in pre.post_restore.__code__.co_varnames else pre.post_restore(x_hat, cond)
                parts = preprocessing_loss(analyzer, clips, x_hat, bpp, target, step_w,
                                           x_pre=x_pre, task_mask=mask,
                                           saliency_pred=getattr(pre, "_last_w", None),
                                           saliency_target=getattr(pre, "_last_w_target", None))
            finally:
                if pinned:
                    analyzer.unpin_active()
            opt.zero_grad(set_to_none=True)
            if rate_opt is not None:
                rate_opt.zero_grad(set_to_none=True)
            parts["loss"].backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(pre.parameters(), clip_grad)
            opt.step()
            if rate_opt is not None:
                rate_opt.step()
                # Phase discipline: after N steps of joint calibration, freeze
                # the prior so the rate gradient stays anchored to the raw-
                # content statistics the preprocessor must now beat.
                if rate_freeze_after and step >= rate_freeze_after:
                    for p in codec.rate_params.values():
                        p.requires_grad_(False)
                    rate_opt = None
                    print(f"[train] rate prior FROZEN at step {step} "
                          f"(rate_freeze_after={rate_freeze_after})", flush=True)
            if sched is not None:
                sched.step()
            step += 1
            vals = {k: v.item() for k, v in parts.items()}
            lr_now = opt.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{vals['loss']:.3f}",
                             task=f"{vals['loss_task']:.3f}",
                             dist=f"{vals['loss_dist']:.3f}",
                             bpp=f"{vals['loss_rate']:.3f}",
                             tmp=f"{vals['loss_temp']:.4f}",
                             dlt=f"{vals['loss_delta']:.4f}",
                             tv=f"{vals['loss_tv']:.4f}",
                             tvr=f"{vals['loss_tv_res']:.4f}",
                             dct=f"{vals['loss_dct']:.4f}",
                             dct3=f"{vals['loss_dct3d']:.4f}",
                             wd=f"{vals.get('loss_w_distill', 0.0):.4f}",
                             lr=f"{lr_now:.1e}", qp=qp)
            tracker.log_step(step, {**vals, "lr": lr_now, "qp": qp})
            if max_steps and step >= max_steps:
                stop = True
                break

        if val_loader is None:
            # Worth shouting about: with no validation there is no checkpoint
            # selection and no early stopping, so `preprocessor.pth` is just the
            # last epoch and patience does nothing. That is exactly how a 12 h
            # run produced best_val=inf without anyone noticing.
            if epoch == start_epoch:
                print("[train] WARNING: no validation loader -> no model selection "
                      "and no early stopping; preprocessor.pth will be the LAST "
                      "epoch. Check that the val split is non-empty.", flush=True)
            _save(ckpt_path, epoch + 1)            # no val -> last is best
        else:
            vl = _val_loss(pre, codec, analyzer, val_loader, weights, qp_list,
                           qp_to_quality, cfg, prep_batch, val_max_batches)
            improved, best_val, no_improve, stop_es = _earlystop_update(
                vl, best_val, min_delta, no_improve, patience)
            print(f"[train] epoch {epoch + 1} val_loss={vl:.4f} best={best_val:.4f} "
                  f"{'*improved' if improved else f'(no_improve={no_improve})'}", flush=True)
            tracker.log_epoch(epoch + 1, {"val_loss": vl, "best_val": best_val,
                                          "no_improve": no_improve})
            if improved:
                _save(ckpt_path, epoch + 1)
            elif stop_es:
                print(f"[train] early stop: no val gain in {patience} epochs", flush=True)
                stop = True
        # Resume must include this epoch's model-selection state, including the
        # observation that triggers early stopping (or a max_steps exit).
        _save(last_path, epoch + 1)
        if stop:
            break

    if val_loader is None or not ckpt_path.exists():
        _save(ckpt_path, epoch + 1)
    final = {"steps": step, "epochs_done": epoch + 1}
    if best_val != float("inf"):
        final["best_val"] = best_val
    tracker.finish(final)
    print(f"[train] best checkpoint -> {ckpt_path}")
    return str(ckpt_path)


def _train_classification(cfg: dict) -> str:
    device = _device(cfg)
    pre, codec, analyzer = _build_models(cfg, device)
    tr, d = cfg["train"], cfg["data"]
    common = dict(
        index_json=d["index"],
        num_frames=d.get("num_frames", 16),
        frame_size=d.get("frame_size", 128),
        temporal_stride=d.get("temporal_stride", 2),
    )
    train_ds = VideoClipDataset(split="train", train=True, **common)
    val_ds = VideoClipDataset(split="val", train=False, **common)
    train_loader = DataLoader(
        train_ds, batch_size=tr.get("batch_size", 4), shuffle=True,
        num_workers=tr.get("num_workers", 2), collate_fn=collate_clips,
        drop_last=True, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=tr.get("batch_size", 4), shuffle=False,
        num_workers=tr.get("num_workers", 2), collate_fn=collate_clips,
    ) if len(val_ds) else None

    def prep(batch):
        clips, labels = batch
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        return clips, labels

    return _fit(cfg, pre, codec, analyzer, train_loader, val_loader, prep,
                tag="action_recognition", n_train=len(train_ds))


def _train_tracking(cfg: dict) -> str:
    device = _device(cfg)
    pre, codec, analyzer = _build_models(cfg, device)
    tr, d = cfg["train"], cfg["data"]
    common = dict(
        index_json=d["index"],
        num_frames=d.get("num_frames", 8),
        frame_size=d.get("frame_size", 256),
        temporal_stride=d.get("temporal_stride", 3),
    )
    train_ds = GOT10kClipDataset(split="train", train=True, **common)
    val_ds = GOT10kClipDataset(split="val", train=False, **common)
    train_loader = DataLoader(
        train_ds, batch_size=tr.get("batch_size", 2), shuffle=True,
        num_workers=tr.get("num_workers", 2), collate_fn=collate_got10k,
        drop_last=True, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=tr.get("batch_size", 2), shuffle=False,
        num_workers=tr.get("num_workers", 2), collate_fn=collate_got10k,
    ) if len(val_ds) else None

    def prep(batch):
        clips, boxes = batch
        clips = clips.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True)
        return clips, {"boxes": boxes}

    return _fit(cfg, pre, codec, analyzer, train_loader, val_loader, prep,
                tag="tracking", n_train=len(train_ds))


def _train_detection(cfg: dict) -> str:
    """Single-frame detection: images through the codec as All-Intra content.

    The AR recipe trains on 16-frame clips whose bit savings come largely from
    the temporal module; on images that module has nothing to do, so this path
    exists to learn an edit that pays for itself at T=1. mAP is evaluated by
    ops/probe_detection.py (the same 3-arm protocol), not inside the engine —
    the training loop only needs a differentiable L_Acc, which the frozen
    detector supplies.
    """
    from .data.coco_det import validate_od_index

    tr, d = cfg["train"], cfg["data"]
    index = json.loads(Path(d["index"]).read_text())
    d["od_split_fingerprint"] = validate_od_index(index)
    device = _device(cfg)
    pre, codec, analyzer = _build_models(cfg, device)
    common = dict(index_json=d["index"], frame_size=d.get("frame_size", 320))
    train_ds = CocoDetDataset(split="train", train=True,
                              max_items=d.get("max_train_items"), **common)
    val_ds = CocoDetDataset(split="val", train=False,
                            max_items=d.get("max_val_items"), **common)
    if len(train_ds) < tr.get("batch_size", 4) or not len(val_ds):
        raise ValueError("OD requires a nonempty validation split and at least one full training batch")
    train_loader = DataLoader(
        train_ds, batch_size=tr.get("batch_size", 4), shuffle=True,
        num_workers=tr.get("num_workers", 2), collate_fn=collate_coco_det,
        drop_last=True, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=tr.get("batch_size", 4), shuffle=False,
        num_workers=tr.get("num_workers", 2), collate_fn=collate_coco_det,
    ) if len(val_ds) else None

    def prep(batch):
        clips, targets = batch
        clips = clips.to(device, non_blocking=True)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        return clips, targets

    return _fit(cfg, pre, codec, analyzer, train_loader, val_loader, prep,
                tag="object_detection", n_train=len(train_ds))


# --------------------------------------------------------------------------
# evaluation (dispatch)
# --------------------------------------------------------------------------
def evaluate(cfg: dict, ckpt_path: str, out_dir: str | None = None) -> dict:
    state = torch.load(ckpt_path, map_location="cpu")
    model_state = state["model"] if "model" in state else state
    # Architecture hyper-parameters must match training, or weights silently
    # load into a differently-parameterised editor (e.g. a smooth-mode
    # checkpoint evaluated as a free residual editor). The training config is
    # persisted inside the checkpoint; let it override the YAML for every
    # model.* key that affects the architecture.
    ckpt_cfg = state.get("cfg") if isinstance(state, dict) else None
    if isinstance(ckpt_cfg, dict) and isinstance(ckpt_cfg.get("model"), dict):
        arch_keys = ("arch", "temporal_frames", "edit_kind", "gate_area", "gate",
                     "base_ch", "res_scale", "cond_dim", "max_relative_edit",
                     # upvcm arch knobs (must match training exactly)
                     "s_ch", "editor_ch", "dino_weight", "dino_name", "motion_tau",
                     "post_base", "w_budget", "post_temporal", "qp_ref")
        for k in arch_keys:
            if k in ckpt_cfg["model"]:
                cfg.setdefault("model", {})[k] = ckpt_cfg["model"][k]
    device = _device(cfg)
    pre, codec, analyzer = _build_models(cfg, device, role="eval")
    pre.load_state_dict(model_state)
    pre.eval()

    out_dir = Path(out_dir or (Path(cfg.get("out_dir", "outputs")) / "eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg["task"]["name"] == "tracking":
        return _evaluate_tracking(cfg, pre, codec, analyzer, out_dir)
    return _evaluate_classification(cfg, pre, codec, analyzer, out_dir)


# -- accumulation utilities ------------------------------------------------
def _accumulate(store: dict, method: str, key, bpp: float, score_sum: float, n: int):
    slot = store.setdefault(method, {}).setdefault(
        key, {"bpp_sum": 0.0, "score_sum": 0.0, "n": 0}
    )
    slot["bpp_sum"] += bpp * n
    slot["score_sum"] += score_sum
    slot["n"] += n


def _finite_or_none(value: float):
    """JSON-safe metric value; undefined BD curves must not be written as NaN."""
    value = float(value)
    return value if torch.isfinite(torch.tensor(value)) else None


def _curve(store_method: dict) -> Dict[str, List[float]]:
    keys = sorted(store_method, key=lambda k: store_method[k]["bpp_sum"] / max(store_method[k]["n"], 1))
    bpp = [store_method[k]["bpp_sum"] / store_method[k]["n"] for k in keys]
    acc = [store_method[k]["score_sum"] / store_method[k]["n"] for k in keys]
    return {"keys": list(keys), "bpp": bpp, "accuracy": acc}


def _ste_codec_name(codec) -> str:
    """Real-codec name when training through the STE, else h265 default."""
    return getattr(codec, "codec", "h265") or "h265"


@torch.no_grad()
def _task_metric(analyzer, x_hat, labels):
    """Classification top-1: returns (num_correct, batch_size)."""
    logits = analyzer.predict(x_hat)
    correct = (logits.argmax(dim=1) == labels).sum().item()
    return float(correct), x_hat.shape[0]


# -- classification eval ---------------------------------------------------
def _proxy_name(cfg: dict) -> str:
    kind = cfg.get("codec", {}).get("kind", "compressai")
    if kind == "ste":
        return f"ste-{cfg.get('codec', {}).get('ste_codec', 'h265')}"
    return kind


def _evaluate_split(cfg: dict) -> str:
    return cfg.get("eval", {}).get("split", "test")


def _apply_eval_shard(ds, ev: dict):
    """Deterministic sub-split of the eval set for multi-kernel sharding.

    Sharding keys on the mount-independent clip id (md5 of
    '<class>/<file>.mp4' — the same rule as build_train_index), so shard
    members are identical across sessions/machines and shards are disjoint.
    Merge happens downstream on per-sequence records (ops/merge_eval.py).
    """
    import hashlib

    shard_idx = ev.get("shard_idx", None)
    num_shards = int(ev.get("num_shards", 1) or 1)
    if shard_idx is None or num_shards <= 1:
        return
    shard_idx = int(shard_idx)
    if not 0 <= shard_idx < num_shards:
        raise ValueError(f"eval.shard_idx must be in [0,{num_shards}), got {shard_idx}")

    def _clip_key(path: str) -> str:
        parts = str(path).replace("\\", "/").rstrip("/").split("/")
        return "/".join(parts[-2:])

    keep = []
    for s in ds.samples:
        h = int(hashlib.md5(_clip_key(s["path"]).encode("utf-8")).hexdigest()[:8], 16)
        if h % num_shards == shard_idx:
            keep.append(s)
    print(f"[eval] shard {shard_idx}/{num_shards}: {len(keep)}/{len(ds.samples)} sequences")
    ds.samples = keep


def _evaluate_classification(cfg, pre, codec, analyzer, out_dir) -> dict:
    device = next(pre.parameters()).device
    ev = cfg.get("eval", {})
    ds = VideoClipDataset(
        index_json=cfg["data"]["index"],
        split=_evaluate_split(cfg),
        num_frames=cfg["data"].get("num_frames", 16),
        frame_size=cfg["data"].get("frame_size", 128),
        temporal_stride=cfg["data"].get("temporal_stride", 2),
        train=False,
        return_metadata=bool(ev.get("per_sequence", False)),
    )
    _apply_eval_shard(ds, ev)
    loader = DataLoader(
        ds, batch_size=ev.get("batch_size", 4), shuffle=False,
        num_workers=ev.get("num_workers", 2), collate_fn=collate_clips,
    )
    qps = ev.get("qp_list", [30, 35, 40, 45, 50])
    include_proxy = bool(ev.get("include_proxy", False))
    have_ffmpeg = ffmpeg_available()
    if not have_ffmpeg:
        print("[eval] WARNING: ffmpeg not found -> skipping H.264/H.265 anchors")

    store: dict = {}
    proxy_name = _proxy_name(cfg)
    prep_proxy_name = f"prep+{proxy_name}"
    qmid = codec.qualities[len(codec.qualities) // 2]
    qconds = _quality_conds(cfg, codec) if include_proxy else {}
    saved_vis = False
    sequence_store = {} if ev.get("per_sequence", False) else None
    for batch in tqdm(loader, desc="eval"):
        if len(batch) == 3:
            clips, labels, metadata = batch
        else:
            clips, labels = batch
            metadata = None
        clips = clips.to(device)
        labels = labels.to(device)
        # D1: gate the edit at eval too (train/eval consistency). The saliency
        # comes from the EVAL analyzer (held-out backbone by default) -- the
        # "universal" reading: protection regions derived from whichever frozen
        # analyzer will consume the stream.
        gate_mask = None
        if bool(cfg.get("model", {}).get("gate", True)) and bool(
                cfg.get("loss", {}).get("use_task_mask", True)):
            gate_mask = task_saliency(analyzer, clips, labels)
        # Rate-conditioned: the preprocessor output depends on the operating
        # point, so it is recomputed per rate point (cannot preprocess once).
        if include_proxy:
            for q in codec.qualities:
                cond = _rate_cond(qconds[q], clips.shape[0], clips.device, clips.dtype)
                with torch.no_grad():
                    x_pre = pre(clips, cond, mask=gate_mask)
                xh, bpp = codec.compress_decompress(x_pre, q)
                s, n = _task_metric(analyzer, xh, labels)
                _accumulate(store, prep_proxy_name, q, bpp, s, n)
                if not saved_vis and q == qmid:
                    _save_qualitative(out_dir / "qualitative.png", clips, x_pre, xh)
                    saved_vis = True
                xh0, bpp0 = codec.compress_decompress(clips, q)
                s0, n0 = _task_metric(analyzer, xh0, labels)
                _accumulate(store, proxy_name, q, bpp0, s0, n0)
        if have_ffmpeg:
            for name in ("h264", "h265"):
                for qp in qps:
                    cond = _rate_cond(_qp_norm(qp, cfg), clips.shape[0], clips.device, clips.dtype)
                    with torch.no_grad():
                        # v9-b dualcodec: per-codec PRE (encoder knows its codec)
                        if "codec" in pre.forward.__code__.co_varnames:
                            x_pre = pre(clips, cond, mask=gate_mask, codec=name)
                        else:
                            x_pre = pre(clips, cond, mask=gate_mask)
                    sc = StandardCodec(codec=name, qp=qp, preset=ev.get("preset", "medium"))
                    xh, bpps = sc.compress_decompress_items(clips)
                    xhp, bppps = sc.compress_decompress_items(x_pre)
                    logits = analyzer.predict(xh.to(device))
                    logits_p = analyzer.predict(xhp.to(device))
                    probs = logits.softmax(dim=1)
                    probs_p = logits_p.softmax(dim=1)
                    target_prob = probs.gather(1, labels[:, None]).squeeze(1)
                    target_prob_p = probs_p.gather(1, labels[:, None]).squeeze(1)
                    correct = (logits.argmax(dim=1) == labels)
                    correct_p = (logits_p.argmax(dim=1) == labels)
                    s, n = float(correct.sum().item()), int(labels.shape[0])
                    sp, np_ = float(correct_p.sum().item()), int(labels.shape[0])
                    _accumulate(store, name, qp, sum(bpps) / max(n, 1), s, n)
                    _accumulate(store, f"prep+{name}", qp, sum(bppps) / max(np_, 1), sp, np_)
                    if hasattr(pre, "post_restore"):
                        # sandwich arm: POST restores the decoded clip (same
                        # bpp — post runs after decode and costs no bits)
                        pre.bypass_post = False
                        xhp_post = (pre.post_restore(xhp, cond, codec=name)
                                    if "codec" in pre.post_restore.__code__.co_varnames
                                    else pre.post_restore(xhp, cond))
                        logits_s = analyzer.predict(xhp_post.to(device))
                        probs_s = logits_s.softmax(dim=1)
                        target_prob_s = probs_s.gather(1, labels[:, None]).squeeze(1)
                        correct_s = (logits_s.argmax(dim=1) == labels)
                        ss = float(correct_s.sum().item())
                        _accumulate(store, f"sandwich+{name}", qp,
                                    sum(bppps) / max(np_, 1), ss, np_)
                    if sequence_store is not None:
                        for i, meta in enumerate(metadata):
                            sid = str(meta["sequence_id"])
                            rec = sequence_store.setdefault(sid, {
                                "sequence_id": sid,
                                "path": meta["path"],
                                "class": meta["class"],
                                "codecs": {},
                            })
                            codec_rec = rec["codecs"].setdefault(name, {})
                            codec_rec[str(qp)] = {
                                "bpp": float(bpps[i]),
                                "top1": int(correct[i].item()),
                                "target_prob": float(target_prob[i].item()),
                            }
                            prep_rec = rec["codecs"].setdefault(f"prep+{name}", {})
                            prep_rec[str(qp)] = {
                                "bpp": float(bppps[i]),
                                "top1": int(correct_p[i].item()),
                                "target_prob": float(target_prob_p[i].item()),
                            }
                            if hasattr(pre, "post_restore"):
                                sand_rec = rec["codecs"].setdefault(f"sandwich+{name}", {})
                                sand_rec[str(qp)] = {
                                    "bpp": float(bppps[i]),
                                    "top1": int(correct_s[i].item()),
                                    "target_prob": float(target_prob_s[i].item()),
                                }
                    if not saved_vis and name == "h265" and qp == qps[len(qps) // 2]:
                        _save_qualitative(out_dir / "qualitative.png", clips, x_pre, xhp)
                        saved_vis = True

    curves = {m: _curve(store[m]) for m in store}
    sequence_extra = _sequence_reports(out_dir, sequence_store, qps) if sequence_store is not None else None
    return _finalize(curves, out_dir, task="action_recognition", metric="top1",
                     n_eval=len(ds), proxy_name=proxy_name, extra=sequence_extra)


# -- tracking eval ---------------------------------------------------------
def _codec_chunked(pre, codec, clip, q, chunk, use_pre, cond=None):
    """Run preprocessor+CompressAI over a long clip in T-chunks (bounds memory)."""
    b, c, t, h, w = clip.shape
    outs, bpp_sum = [], 0.0
    for s in range(0, t, chunk):
        sub = clip[:, :, s : s + chunk]
        with torch.no_grad():
            xp = pre(sub, cond) if use_pre else sub
            xh, bpp = codec.compress_decompress(xp, q)
        outs.append(xh)
        bpp_sum += bpp * sub.shape[2]
    return torch.cat(outs, dim=2), bpp_sum / max(t, 1)


def _pre_chunked(pre, clip, chunk, cond=None, codec=None):
    """Preprocess a long clip in T-chunks, return the full [B,C,T,H,W] (bounds memory).

    codec routes DualCodecSandwich's per-codec PRE (its forward takes codec=);
    codec-agnostic models (sandwich/upvcm) ignore the argument entirely.
    """
    t = clip.shape[2]
    outs = []
    for s in range(0, t, chunk):
        with torch.no_grad():
            sub = clip[:, :, s : s + chunk]
            outs.append(pre(sub, cond, codec=codec) if codec else pre(sub, cond))
    return torch.cat(outs, dim=2)


def _post_chunked(pre, clip, chunk, cond=None, codec=None):
    """POST-restore a long decoded clip in T-chunks (bounds memory)."""
    post_restore = getattr(pre, "post_restore", None)
    if post_restore is None:
        raise AttributeError("model has no post_restore (arch must be sandwich-like)")
    t = clip.shape[2]
    outs = []
    for s in range(0, t, chunk):
        with torch.no_grad():
            sub = clip[:, :, s : s + chunk]
            outs.append(post_restore(sub, cond, codec=codec) if codec else post_restore(sub, cond))
    return torch.cat(outs, dim=2)


def _acc_track(store, method, key, bpp, pred, gt, valid):
    m = sequence_metrics(pred, gt, valid)
    slot = store.setdefault(method, {}).setdefault(
        key, {"bpp_sum": 0.0, "frames": 0, "seqs": []}
    )
    T = len(gt)
    slot["bpp_sum"] += bpp * T
    slot["frames"] += T
    slot["seqs"].append(m)


def _curve_track(store_method: dict) -> Dict[str, List[float]]:
    keys = sorted(store_method, key=lambda k: store_method[k]["bpp_sum"] / max(store_method[k]["frames"], 1))
    bpp = [store_method[k]["bpp_sum"] / max(store_method[k]["frames"], 1) for k in keys]
    agg = [aggregate_metrics(store_method[k]["seqs"]) for k in keys]
    return {
        "keys": list(keys),
        "bpp": bpp,
        "accuracy": [a["auc"] for a in agg],
        "ao": [a["ao"] for a in agg],
        "sr50": [a["sr50"] for a in agg],
        "sr75": [a["sr75"] for a in agg],
    }


def _resolve_tracker(cfg, analyzer):
    """Pick the eval tracker. Default = the trained-against SiamFC analyzer.

    ``task.tracker`` may request the paper's exact trackers via pytracking, e.g.
    ``pytracking:dimp:dimp50`` / ``pytracking:atom`` / ``pytracking:kys`` /
    ``pytracking:prdimp:prdimp50``. Those are eval-only (the preprocessor is
    always trained with SiamFC's differentiable loss).
    """
    spec = cfg["task"].get("tracker", "siamfc")
    if spec in (None, "siamfc", "default"):
        return analyzer.track
    parts = str(spec).split(":")
    if parts[0] == "pytracking":
        from .tasks.pytracking_adapter import build_tracker

        name = parts[1] if len(parts) > 1 else "dimp"
        param = parts[2] if len(parts) > 2 else None
        trk = build_tracker(name, param)
        print(f"[eval] using pytracking tracker: {name}/{trk.parameter}")
        return trk.track_sequence
    raise ValueError(f"unknown task.tracker '{spec}'")


def _evaluate_tracking(cfg, pre, codec, analyzer, out_dir) -> dict:
    device = next(pre.parameters()).device
    ev = cfg.get("eval", {})
    fs = cfg["data"].get("frame_size", 256)
    max_frames = ev.get("max_frames", 48)
    max_seqs = ev.get("max_seqs", 30)
    chunk = ev.get("codec_chunk", 16)
    qps = ev.get("qp_list", [30, 35, 40, 45, 50])
    include_proxy = bool(ev.get("include_proxy", False))
    track = _resolve_tracker(cfg, analyzer)
    have_ffmpeg = ffmpeg_available()
    if not have_ffmpeg:
        print("[eval] WARNING: ffmpeg not found -> skipping H.264/H.265 anchors")

    seqs = list(iter_sequences(cfg["data"]["index"], _evaluate_split(cfg), fs,
                               max_frames, max_seqs))
    store: dict = {}
    proxy_name = _proxy_name(cfg)
    prep_proxy_name = f"prep+{proxy_name}"
    qconds = _quality_conds(cfg, codec) if include_proxy else {}
    for name, clip, gt, valid in tqdm(seqs, desc="eval-track"):
        clip = clip.to(device)
        init = gt[0]
        # Rate-conditioned: preprocess per operating point (output depends on it).
        if include_proxy:
            for q in codec.qualities:
                cond = _rate_cond(qconds[q], clip.shape[0], clip.device, clip.dtype)
                xh, bpp = _codec_chunked(pre, codec, clip, q, chunk, use_pre=True, cond=cond)
                _acc_track(store, prep_proxy_name, q, bpp, track(xh, init), gt, valid)
                xh0, bpp0 = _codec_chunked(pre, codec, clip, q, chunk, use_pre=False)
                _acc_track(store, proxy_name, q, bpp0, track(xh0, init), gt, valid)
        if have_ffmpeg:
            import inspect
            pre_takes_codec = "codec" in inspect.signature(pre.forward).parameters
            has_post = hasattr(pre, "post_restore")
            post_takes_codec = has_post and "codec" in inspect.signature(
                pre.post_restore).parameters
            for cname in ("h264", "h265"):
                for qp in qps:
                    cond = _rate_cond(_qp_norm(qp, cfg), clip.shape[0], clip.device, clip.dtype)
                    clip_pre = _pre_chunked(pre, clip, chunk, cond=cond,
                                            codec=cname if pre_takes_codec else None)
                    sc = StandardCodec(codec=cname, qp=qp, preset=ev.get("preset", "medium"))
                    xh, bpp = sc.compress_decompress(clip)
                    _acc_track(store, cname, qp, bpp, track(xh.to(device), init), gt, valid)
                    xhp, bppp = sc.compress_decompress(clip_pre)   # prep + real codec
                    _acc_track(store, f"prep+{cname}", qp, bppp, track(xhp.to(device), init), gt, valid)
                    if has_post:
                        # sandwich arm: POST restores the decoded clip (same
                        # bpp — post runs after decode and costs no bits);
                        # mirrors the classification eval's 3-arm protocol
                        # (ported from v8 commit 1869d48).
                        pre.bypass_post = False
                        xs = _post_chunked(pre, xhp.to(device), chunk, cond=cond,
                                           codec=cname if post_takes_codec else None)
                        _acc_track(store, f"sandwich+{cname}", qp, bppp, track(xs, init), gt, valid)

    curves = {m: _curve_track(store[m]) for m in store}
    return _finalize(curves, out_dir, task="tracking", metric="auc", n_eval=len(seqs),
                     proxy_name=proxy_name)


# --------------------------------------------------------------------------
# finalize: BD-Rate, save, plot, print (shared by both tasks)
# --------------------------------------------------------------------------
def _bd_pair(curves: dict, test_name: str, anchor_name: str):
    """BD-Rate / BD-accuracy of test curve vs anchor curve (None if either absent)."""
    if test_name not in curves or anchor_name not in curves:
        return None
    a, t = curves[anchor_name], curves[test_name]
    return {
        "bd_rate_pct": _finite_or_none(
            bd_rate(a["bpp"], a["accuracy"], t["bpp"], t["accuracy"])
        ),
        "bd_accuracy": _finite_or_none(
            bd_metric(a["bpp"], a["accuracy"], t["bpp"], t["accuracy"])
        ),
    }


def _sequence_reports(out_dir: Path, sequence_store: dict, qps: list[int]):
    """Write per-sequence rate/accuracy points and same-codec BD metrics.

    Top-1 is retained as a binary diagnostic.  For BD-Rate per sequence we use
    the model probability assigned to the ground-truth class (``target_prob``),
    because a 0/1 metric cannot form a useful five-point curve for most videos.
    """
    import csv

    points_path = out_dir / "sequence_points.csv"
    with open(points_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sequence_id", "path", "class", "codec", "qp", "bpp", "top1", "target_prob"])
        for rec in sequence_store.values():
            for codec, points in rec["codecs"].items():
                for qp, value in points.items():
                    w.writerow([rec["sequence_id"], rec["path"], rec["class"], codec, qp,
                                f"{value['bpp']:.8f}", value["top1"], f"{value['target_prob']:.8f}"])

    per_sequence = {}
    bd_rows = []
    for sid, rec in sequence_store.items():
        codec_metrics = {}
        for codec in ("h264", "h265"):
            anchor = rec["codecs"].get(codec, {})
            prep = rec["codecs"].get(f"prep+{codec}", {})
            if not anchor or not prep:
                continue
            ordered = [str(q) for q in qps if str(q) in anchor and str(q) in prep]
            if len(ordered) < 2:
                continue
            ra = [anchor[q]["bpp"] for q in ordered]
            ma = [anchor[q]["target_prob"] for q in ordered]
            rt = [prep[q]["bpp"] for q in ordered]
            mt = [prep[q]["target_prob"] for q in ordered]
            rate = _finite_or_none(bd_rate(ra, ma, rt, mt))
            acc = _finite_or_none(bd_metric(ra, ma, rt, mt))
            codec_metrics[f"prep+{codec} vs {codec}"] = {
                "bd_rate_pct": rate,
                "bd_accuracy": acc,
                "metric": "target_prob",
                "n_points": len(ordered),
            }
            bd_rows.append([rec["sequence_id"], rec["path"], rec["class"], codec, rate, acc, len(ordered)])
        if codec_metrics:
            per_sequence[sid] = {
                "sequence_id": rec["sequence_id"],
                "path": rec["path"],
                "class": rec["class"],
                "bd_prep_gain": codec_metrics,
            }

    bd_path = out_dir / "sequence_bd_rate.csv"
    with open(bd_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sequence_id", "path", "class", "codec", "bd_rate_pct", "bd_accuracy", "n_points", "metric"])
        for row in bd_rows:
            w.writerow(row + ["target_prob"])
    json_path = out_dir / "sequence_bd_rate.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"metric": "target_prob", "n_sequences": len(sequence_store),
                   "sequences_with_bd": len(per_sequence), "results": per_sequence}, f, indent=2)
    print(f"[eval] wrote {points_path}, {bd_path}, {json_path}")
    return {"per_sequence": per_sequence, "per_sequence_metric": "target_prob"}


def _finalize(curves: dict, out_dir: Path, task: str, metric: str, n_eval: int,
              proxy_name: str, extra: dict | None = None) -> dict:
    prep_proxy_name = f"prep+{proxy_name}"
    # Cross-codec view is reference-only; the same-codec pairs below are the claim.
    bd = {}
    for anchor in (proxy_name, "h264", "h265"):
        e = _bd_pair(curves, prep_proxy_name, anchor)
        if e is not None:
            bd[anchor] = e
    # apples-to-apples: preprocessor gain on the SAME codec (the real claim)
    prep_gain = {}
    for codec_name in (proxy_name, "h264", "h265"):
        e = _bd_pair(curves, f"prep+{codec_name}", codec_name)
        if e is not None:
            prep_gain[f"prep+{codec_name} vs {codec_name}"] = e
        # v8 sandwich claim: pre + post around the SAME frozen codec
        e = _bd_pair(curves, f"sandwich+{codec_name}", codec_name)
        if e is not None:
            prep_gain[f"sandwich+{codec_name} vs {codec_name}"] = e
    results = {
        "task": task,
        "metric": metric,
        "curves": curves,
        "bd_vs_anchor": bd,
        "bd_prep_gain": prep_gain,
        "n_eval": n_eval,
    }
    if extra:
        results.update(extra)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _write_csv(out_dir / "curves.csv", curves)
    _plot(out_dir / "rate_accuracy.png", curves, metric)
    _print_summary(results)
    tracker = attach_tracker(out_dir, fallback_name=f"{task}-eval")
    for pair, g in (results.get("bd_prep_gain") or {}).items():
        tracker.log_eval({f"bd_rate_pct/{pair}": g.get("bd_rate_pct"),
                          f"bd_accuracy/{pair}": g.get("bd_accuracy")})
    for method, c in curves.items():
        tracker.log_curve(f"rate-accuracy/{method}", c.get("bpp", []), c.get("accuracy", []))
    for codec in ("h264", "h265"):
        a, p_ = curves.get(codec), curves.get(f"prep+{codec}")
        if not a or not p_ or not a.get("keys"):
            continue
        tracker.log_eval({
            f"qp_gap_first/{codec}": p_["accuracy"][0] - a["accuracy"][0],
            f"qp_gap_last/{codec}": p_["accuracy"][-1] - a["accuracy"][-1],
            f"bpp_div/{codec}": sum(abs(x - y) / y for x, y in zip(p_["bpp"], a["bpp"]))
                                / max(len(a["bpp"]), 1),
        })
    tracker.finish({"n_eval": n_eval})
    print(f"[eval] wrote {out_dir/'results.json'}, curves.csv, rate_accuracy.png")
    return results


def _write_csv(path: Path, curves: dict) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "rate_point", "bpp", "accuracy"])
        for method, c in curves.items():
            for k, bpp, acc in zip(c["keys"], c["bpp"], c["accuracy"]):
                w.writerow([method, k, f"{bpp:.6f}", f"{acc:.6f}"])


def _plot(path: Path, curves: dict, metric: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[eval] plot skipped ({e})")
        return

    ylabel = {"top1": "top-1 accuracy", "auc": "tracking success AUC"}.get(metric, metric)
    plt.figure(figsize=(7, 5))
    styles = {
        "prep+compressai": dict(marker="o", lw=2),
        "compressai": dict(marker="s", ls="--"),
        "prep+virtual": dict(marker="o", lw=2),
        "virtual": dict(marker="s", ls="--"),
        "h264": dict(marker="^", ls=":"),
        "h265": dict(marker="v", ls="-."),
    }
    for method, c in curves.items():
        plt.plot(c["bpp"], c["accuracy"], label=method, **styles.get(method, {}))
    plt.xlabel("bits per pixel (proxy estimate; real coded for H.264/H.265)")
    plt.ylabel(ylabel)
    plt.title("Rate vs machine-vision accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def _save_qualitative(path, source, x_pre, x_hat, n_frames: int = 4) -> None:
    """Grid PNG: rows = source / preprocessed / reconstructed, cols = sampled
    frames of batch item 0. Lets you eye what the preprocessor actually edits."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[eval] qualitative viz skipped ({e})")
        return
    t = source.shape[2]
    idx = torch.linspace(0, t - 1, min(n_frames, t)).round().long().tolist()
    rows = [("source", source), ("preprocessed", x_pre), ("recon", x_hat)]
    fig, axes = plt.subplots(
        len(rows), len(idx), figsize=(3 * len(idx), 3 * len(rows)), squeeze=False
    )
    for r, (name, ten) in enumerate(rows):
        for c, fi in enumerate(idx):
            img = ten[0, :, fi].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            ax = axes[r][c]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(name, fontsize=11)
            if r == 0:
                ax.set_title(f"frame {fi}", fontsize=9)
    fig.suptitle("Qualitative: source vs preprocessed vs reconstructed")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[eval] wrote {path}")


def _print_summary(results: dict) -> None:
    print("\n=== rate-accuracy summary ===")
    for method, c in results["curves"].items():
        pts = ", ".join(f"({b:.3f}bpp, {a:.3f})" for b, a in zip(c["bpp"], c["accuracy"]))
        print(f"  {method:16s}: {pts}")
    if results["bd_vs_anchor"]:
        print("\n=== BD-Rate of prep+proxy vs anchors (negative = savings) ===")
        for anchor, v in results["bd_vs_anchor"].items():
            rate = v["bd_rate_pct"]
            acc = v["bd_accuracy"]
            rate_text = f"{rate:+.2f}%" if rate is not None else "undefined"
            acc_text = f"{acc:+.4f}" if acc is not None else "undefined"
            print(
                f"  vs {anchor:12s}: BD-Rate {rate_text:>10s}  |  "
                f"BD-Accuracy {acc_text}"
            )
    if results.get("bd_prep_gain"):
        print("\n=== preprocessor gain, SAME codec (the real claim; negative = savings) ===")
        for label, v in results["bd_prep_gain"].items():
            rate = v["bd_rate_pct"]
            acc = v["bd_accuracy"]
            rate_text = f"{rate:+.2f}%" if rate is not None else "undefined"
            acc_text = f"{acc:+.4f}" if acc is not None else "undefined"
            print(
                f"  {label:28s}: BD-Rate {rate_text:>10s}  |  "
                f"BD-Accuracy {acc_text}"
            )
