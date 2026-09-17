#!/usr/bin/env python
"""Pre-registered gates for the Round (b) QPC checkpoint (0-GPU, CPU-only).

Implements the three gates of docs/RUN_DESIGN_qpc.md on a trained
``additive_cond`` checkpoint, each printed next to the standing rule's
reference value (cross_review C6(c)):

  1. Regime gate  — added high-frequency energy of the edit at s=0.25 must
     stay in [-5%, +30%] at BOTH cond=QP30-norm (0.303) and cond=QP50-norm
     (0.645).
  2. Conditionability gate (behavioural) — added-HF AND RMS must differ
     between cond=QP30 and cond=QP50 by MORE than the edit-level twin noise
     (±3%, from rep1's 0.0452 vs 0.0467). Within ±3% on BOTH metrics =>
     null-by-non-utilization: conditioning unused at this budget.
  3. No-blow-up gate — RMS at s=1.0 < 0.14 (incumbent mu=10 twin reads
     0.1202 on the same instrument).

Needs a few eval-split clips; pass an index JSON + a directory with the
mp4s (local machine or Kaggle). Prints PASS/FAIL per gate and exits 0/1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import apply_overrides, load_config  # noqa: E402
from src.data.video_dataset import VideoClipDataset  # noqa: E402
from src.models.additive_cond import AdditiveCondPreprocessor  # noqa: E402

QP30_NORM = 0.303   # (30-20)/(51-20)
QP50_NORM = 0.645   # (50-20)/(51-20)
TWIN_NOISE = 0.03   # edit-level twin RMS noise (rep1), ±3%
NO_BLOWUP = 0.14    # registered threshold; incumbent reads 0.1202
REGIME_LO, REGIME_HI = -0.05, 0.30


def added_hf(x: torch.Tensor, y: torch.Tensor) -> float:
    """Added high-frequency energy ratio of edit y-x vs x (Laplacian, luma).

    >0 means the edit ADDS high-frequency energy; <0 means it smooths.
    """
    lap = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
    lap = lap.view(1, 1, 3, 3).repeat(3, 1, 1, 1)  # depthwise: 3 x (1,1,3,3)

    def hf(t):  # t: [B,C,T,H,W]
        b, c, t_, h, w = t.shape
        f = t.permute(0, 2, 1, 3, 4).reshape(b * t_, c, h, w)
        return F.conv2d(f, lap, groups=c).pow(2).sum().item()

    base = hf(x)
    return (hf(y) - base) / max(base, 1e-12)


def edit_rms(x: torch.Tensor, y: torch.Tensor) -> float:
    return (y - x).pow(2).mean().sqrt().item()


def run_gates(ckpt: str, index: str, n_clips: int, strengths=(1.0, 0.25)) -> dict:
    state = torch.load(ckpt, map_location="cpu")
    model_state = state["model"] if "model" in state else state
    cfg_model = (state.get("cfg") or {}).get("model", {})
    pre = AdditiveCondPreprocessor(
        temporal_frames=int(cfg_model.get("temporal_frames", 8)),
        strength=1.0,  # strengths applied per-run below
        cond_dim=int(cfg_model.get("cond_dim", 1)),
    )
    pre.load_state_dict(model_state, strict=True)
    pre.eval()

    ds = VideoClipDataset(index_json=index, split="test", num_frames=16,
                          frame_size=128, temporal_stride=2, train=False)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(ds), size=min(n_clips, len(ds)), replace=False)

    report = {"n_clips": int(len(idx)), "per_cond": {}, "gates": {}}
    with torch.no_grad():
        for cond_name, cond_val in (("QP30", QP30_NORM), ("QP50", QP50_NORM)):
            for s in strengths:
                pre.strength = s
                hfs, rmss = [], []
                for i in idx:
                    clip, _ = ds[int(i)]
                    clip = clip[None]  # [1,C,T,H,W]
                    cond = torch.tensor([[cond_val]], dtype=clip.dtype)
                    out = pre(clip, cond)
                    hfs.append(added_hf(clip, out))
                    rmss.append(edit_rms(clip, out))
                report["per_cond"][f"{cond_name}_s{s}"] = {
                    "added_hf_mean": float(np.mean(hfs)),
                    "rms_mean": float(np.mean(rmss)),
                }

    p = report["per_cond"]

    # Gate 1: regime gate on added-HF at s=0.25, both conds
    g1 = {
        "QP30_s0.25": p["QP30_s0.25"]["added_hf_mean"],
        "QP50_s0.25": p["QP50_s0.25"]["added_hf_mean"],
    }
    report["gates"]["1_regime"] = {
        "values": g1,
        "band": [REGIME_LO, REGIME_HI],
        "pass": all(REGIME_LO <= v <= REGIME_HI for v in g1.values()),
    }

    # Gate 2: conditionability — spread at s=1.0 (the operating regime)
    hf30, hf50 = p["QP30_s1.0"]["added_hf_mean"], p["QP50_s1.0"]["added_hf_mean"]
    rms30, rms50 = p["QP30_s1.0"]["rms_mean"], p["QP50_s1.0"]["rms_mean"]
    hf_spread = abs(hf50 - hf30) / max(abs(hf30), 1e-9)
    rms_spread = abs(rms50 - rms30) / max(rms30, 1e-9)
    report["gates"]["2_conditionability"] = {
        "hf_spread_pct": float(hf_spread),
        "rms_spread_pct": float(rms_spread),
        "twin_noise": TWIN_NOISE,
        "utilized": bool(hf_spread > TWIN_NOISE or rms_spread > TWIN_NOISE),
    }

    # Gate 3: no-blow-up — RMS at s=1.0, mean of the two conds
    rms1 = 0.5 * (rms30 + rms50)
    report["gates"]["3_no_blowup"] = {
        "rms_s1_mean": float(rms1),
        "threshold": NO_BLOWUP,
        "incumbent_reference": 0.1202,
        "pass": bool(rms1 < NO_BLOWUP),
    }

    # FiLM utilization audit (guard (ii))
    gamma_beta = {}
    for cond_name, cond_val in (("QP30", QP30_NORM), ("QP50", QP50_NORM)):
        with torch.no_grad():
            c = torch.tensor([[cond_val]])
            g, b = pre.film(c).chunk(2, dim=1)
        gamma_beta[cond_name] = {
            "gamma_norm": float(g.norm()), "beta_norm": float(b.norm()),
        }
    report["film_utilization"] = gamma_beta
    return report


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--ckpt", required=True)
    a.add_argument("--index", required=True)
    a.add_argument("--n-clips", type=int, default=32)
    a.add_argument("--out", default=None)
    args = a.parse_args()

    report = run_gates(args.ckpt, args.index, args.n_clips)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))

    print(f"\n=== ROUND (b) GATES ({report['n_clips']} clips) ===")
    g1 = report["gates"]["1_regime"]
    print(f"[Gate 1 regime] added-HF @s=0.25: QP30 {g1['values']['QP30_s0.25']:+.1%}  "
          f"QP50 {g1['values']['QP50_s0.25']:+.1%}  band [{REGIME_LO:+.0%},{REGIME_HI:+.0%}]"
          f"  -> {'PASS' if g1['pass'] else 'FAIL'}")
    g2 = report["gates"]["2_conditionability"]
    print(f"[Gate 2 conditionability] HF spread {g2['hf_spread_pct']:+.1%}  "
          f"RMS spread {g2['rms_spread_pct']:+.1%}  (twin noise ±{TWIN_NOISE:.0%})  "
          f"-> {'UTILIZED' if g2['utilized'] else 'NULL (conditioning unused)'}")
    g3 = report["gates"]["3_no_blowup"]
    print(f"[Gate 3 no-blow-up] RMS @s=1.0 {g3['rms_s1_mean']:.4f} "
          f"(threshold {g3['threshold']}, incumbent ref {g3['incumbent_reference']})"
          f"  -> {'PASS' if g3['pass'] else 'FAIL'}")
    fu = report["film_utilization"]
    print(f"[FiLM audit] QP30 ||γ||={fu['QP30']['gamma_norm']:.4f} ||β||={fu['QP30']['beta_norm']:.4f} | "
          f"QP50 ||γ||={fu['QP50']['gamma_norm']:.4f} ||β||={fu['QP50']['beta_norm']:.4f}")

    ok = g1["pass"] and g3["pass"]
    print(f"\nOVERALL: {'PASS -> proceed to eval' if ok else 'FAIL -> NO EVAL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
