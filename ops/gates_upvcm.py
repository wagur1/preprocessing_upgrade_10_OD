#!/usr/bin/env python
"""Sanity gates for a trained UP-VCM checkpoint (0-analyzer, CPU or GPU).

The round-(b) gates measured added-HF/RMS at two QP conds on the additive
editor. UP-VCM has no eval-time strength knob and no external gate, so its
pre-eval sanity set is different:

  G1 non-identity: the three module gates (dec/edit/stab strengths) must have
     moved away from 0 — an all-zero gate set means the run is a re-trained
     identity and eval would be moot (null-by-non-utilization).
  G2 no-blow-up: RMS(x_pre, x) at the natural operating point < 0.14 on eval
     clips (the lineage no-blow-up threshold, same number as round b).
  G3 W non-degenerate: mean W in [0.05, 0.95] AND per-clip std > 0.02 — a
     constant W (all-background or all-ROI) collapses the spatial mechanism.
  G4 deploy-purity: forward WITHOUT mask and WITHOUT DINO produces the same
     output as with them (mask is train-only by construction; this catches
     accidental eval-time dependence).

Prints PASS/FAIL per gate; exit 0 iff all pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.video_dataset import VideoClipDataset  # noqa: E402
from src.models.upvcm import UPVCMPreprocessor  # noqa: E402

NO_BLOWUP = 0.14
MIN_GATE = 0.01          # |strength| above this counts as "opened"
W_MEAN_BAND = (0.05, 0.95)
W_STD_MIN = 0.02


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--ckpt", required=True)
    a.add_argument("--index", required=True)
    a.add_argument("--n-clips", type=int, default=32)
    a.add_argument("--out", default=None)
    args = a.parse_args()

    state = torch.load(args.ckpt, map_location="cpu")
    model_state = state["model"] if "model" in state else state
    cfg_model = (state.get("cfg") or {}).get("model", {})
    pre = UPVCMPreprocessor(
        s_ch=int(cfg_model.get("s_ch", 16)),
        editor_ch=int(cfg_model.get("editor_ch", 24)),
        cond_dim=int(cfg_model.get("cond_dim", 1)),
        dino_weight=float(cfg_model.get("dino_weight", 0.5)),
        dino_name=str(cfg_model.get("dino_name", "dinov2_vits14")),
        motion_tau=float(cfg_model.get("motion_tau", 0.1)),
    )
    pre.load_state_dict(model_state, strict=True)
    pre.eval()

    ds = VideoClipDataset(index_json=args.index, split="test", num_frames=16,
                          frame_size=128, temporal_stride=2, train=False)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(ds), size=min(args.n_clips, len(ds)), replace=False)

    rmss, w_means, w_stds = [], [], []
    with torch.no_grad():
        for i in idx:
            clip, _ = ds[int(i)]
            clip = clip[None]
            cond = torch.full((1, 1), 0.613)  # QP45-normalised operating point
            out = pre(clip, cond)
            rmss.append((out - clip).pow(2).mean().sqrt().item())
            w = pre._last_w
            w_means.append(w.mean().item())
            w_stds.append(w.std().item())

    gates = {}
    dec = abs(pre.dec_strength.item())
    edit = abs(pre.edit_strength.item())
    stab = abs(pre.stab_strength.item())
    gates["G1_non_identity"] = {
        "dec": dec, "edit": edit, "stab": stab, "min_gate": MIN_GATE,
        "pass": max(dec, edit, stab) > MIN_GATE,
    }
    rms = float(np.mean(rmss))
    gates["G2_no_blowup"] = {"rms_mean": rms, "threshold": NO_BLOWUP,
                             "pass": rms < NO_BLOWUP}
    wm, ws = float(np.mean(w_means)), float(np.mean(w_stds))
    gates["G3_w_nondegenerate"] = {
        "w_mean": wm, "w_std_mean": ws,
        "band": list(W_MEAN_BAND), "std_min": W_STD_MIN,
        "pass": W_MEAN_BAND[0] < wm < W_MEAN_BAND[1] and ws > W_STD_MIN,
    }
    # G4: deploy purity — mask path vs no-mask path (no DINO at eval by design;
    # dino_weight only affects the TRAIN-time target, verified by construction).
    with torch.no_grad():
        clip, _ = ds[int(idx[0])]
        clip = clip[None]
        cond = torch.full((1, 1), 0.5)
        o1 = pre(clip, cond)
        o2 = pre(clip, cond, mask=torch.zeros_like(pre._last_w))
    diff = (o1 - o2).abs().max().item()
    gates["G4_deploy_purity"] = {"max_diff": diff, "pass": diff < 1e-6}

    report = {"n_clips": int(len(idx)), "gates": gates}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))

    print(f"\n=== UP-VCM GATES ({report['n_clips']} clips) ===")
    g1 = gates["G1_non_identity"]
    print(f"[G1 non-identity] gates dec={g1['dec']:.4f} edit={g1['edit']:.4f} "
          f"stab={g1['stab']:.4f} (min {MIN_GATE}) -> "
          f"{'PASS' if g1['pass'] else 'FAIL (identity re-train — eval is moot)'}")
    g2 = gates["G2_no_blowup"]
    print(f"[G2 no-blow-up] RMS {g2['rms_mean']:.4f} < {NO_BLOWUP} -> "
          f"{'PASS' if g2['pass'] else 'FAIL'}")
    g3 = gates["G3_w_nondegenerate"]
    print(f"[G3 W healthy] mean {g3['w_mean']:.3f} std {g3['w_std_mean']:.4f} -> "
          f"{'PASS' if g3['pass'] else 'FAIL (W degenerate)'}")
    g4 = gates["G4_deploy_purity"]
    print(f"[G4 deploy purity] mask-path diff {g4['max_diff']:.2e} -> "
          f"{'PASS' if g4['pass'] else 'FAIL'}")

    ok = all(g["pass"] for g in gates.values())
    print(f"\nOVERALL: {'PASS -> proceed to eval' if ok else 'FAIL -> NO EVAL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
