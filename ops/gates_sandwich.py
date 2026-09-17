#!/usr/bin/env python
"""Sanity gates for a trained Sandwich checkpoint (0-analyzer, CPU or GPU).

Extends the UP-VCM gates to the POST half:

  G1 non-identity (PRE): the three PRE gates must have opened.
  G2 no-blow-up (PRE):  RMS(x_pre, x) < 0.14 at the natural operating point.
  G3 W healthy (PRE):   mean W in [0.05, 0.95], per-clip std > 0.02.
  G4 deploy purity:     mask path does not change the output.
  G5 post gate opened:  |post_strength| > 0.01 — an all-zero POST gate means
                        the sandwich degenerated to prep-only (null result
                        for the POST mechanism, eval of the sandwich arm is
                        then moot though prep-only stays valid).
  G6 post restores:     on codec-noise stand-ins, post(x_hat) must move
                        toward the source (RMS(post(x_hat), x) < RMS(x_hat, x))
                        at least on average — POST that makes things worse on
                        average is a failed restorer.

Exit 0 iff G1-G4 pass (prep-only validity); G5/G6 additionally gate the
SANDWICH arm claim and are reported separately.
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
from src.models.sandwich import SandwichPreprocessor  # noqa: E402

NO_BLOWUP = 0.14
MIN_GATE = 0.01
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
    m = SandwichPreprocessor(
        s_ch=int(cfg_model.get("s_ch", 16)),
        editor_ch=int(cfg_model.get("editor_ch", 24)),
        cond_dim=int(cfg_model.get("cond_dim", 1)),
        dino_weight=float(cfg_model.get("dino_weight", 0.5)),
        dino_name=str(cfg_model.get("dino_name", "dinov2_vits14")),
        motion_tau=float(cfg_model.get("motion_tau", 0.1)),
        post_base=int(cfg_model.get("post_base", 32)),
    )
    m.load_state_dict(model_state, strict=True)
    m.eval()

    ds = VideoClipDataset(index_json=args.index, split="test", num_frames=16,
                          frame_size=128, temporal_stride=2, train=False)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(ds), size=min(args.n_clips, len(ds)), replace=False)

    pre_rms, w_means, w_stds = [], [], []
    post_gains = []  # RMS reduction fraction per clip (positive = restored)
    with torch.no_grad():
        for i in idx:
            clip, _ = ds[int(i)]
            clip = clip[None]
            cond = torch.full((1, 1), 0.613)  # QP45-normalised
            x_pre = m(clip, cond)
            pre_rms.append((x_pre - clip).pow(2).mean().sqrt().item())
            w = m._last_w
            w_means.append(w.mean().item())
            w_stds.append(w.std().item())
            # stand-in codec noise: heavy quantisation proxy = coarse blur+noise
            noisy = (clip * 63.0).round() / 63.0 + 0.03 * torch.randn_like(clip)
            noisy = noisy.clamp(0, 1)
            restored = m.post_restore(noisy, cond)
            rms_before = (noisy - clip).pow(2).mean().sqrt().item()
            rms_after = (restored - clip).pow(2).mean().sqrt().item()
            post_gains.append((rms_before - rms_after) / max(rms_before, 1e-9))

    gates = {}
    dec = abs(m.pre.dec_strength.item())
    edit = abs(m.pre.edit_strength.item())
    stab = abs(m.pre.stab_strength.item())
    gates["G1_non_identity"] = {"dec": dec, "edit": edit, "stab": stab,
                                "pass": max(dec, edit, stab) > MIN_GATE}
    rms = float(np.mean(pre_rms))
    gates["G2_no_blowup"] = {"rms_mean": rms, "pass": rms < NO_BLOWUP}
    wm, ws = float(np.mean(w_means)), float(np.mean(w_stds))
    gates["G3_w_nondegenerate"] = {
        "w_mean": wm, "w_std_mean": ws,
        "pass": W_MEAN_BAND[0] < wm < W_MEAN_BAND[1] and ws > W_STD_MIN}

    with torch.no_grad():
        clip, _ = ds[int(idx[0])]
        clip = clip[None]
        cond = torch.full((1, 1), 0.5)
        o1 = m(clip, cond)
        o2 = m(clip, cond, mask=torch.zeros_like(m._last_w))
    gates["G4_deploy_purity"] = {"max_diff": (o1 - o2).abs().max().item()}

    ps = abs(m.post_strength.item())
    gates["G5_post_opened"] = {"post_strength": ps, "pass": ps > MIN_GATE}
    gain = float(np.mean(post_gains))
    gates["G6_post_restores"] = {"mean_rms_reduction": gain, "pass": gain > 0.0}

    for k in ("G4_deploy_purity",):
        gates[k]["pass"] = gates[k]["max_diff"] < 1e-6

    report = {"n_clips": int(len(idx)), "gates": gates}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))

    print(f"\n=== SANDWICH GATES ({report['n_clips']} clips) ===")
    g1 = gates["G1_non_identity"]
    print(f"[G1 PRE non-identity] dec={g1['dec']:.4f} edit={g1['edit']:.4f} "
          f"stab={g1['stab']:.4f} -> {'PASS' if g1['pass'] else 'FAIL'}")
    print(f"[G2 PRE no-blow-up] RMS {gates['G2_no_blowup']['rms_mean']:.4f} < {NO_BLOWUP} "
          f"-> {'PASS' if gates['G2_no_blowup']['pass'] else 'FAIL'}")
    g3 = gates["G3_w_nondegenerate"]
    print(f"[G3 PRE W healthy] mean {g3['w_mean']:.3f} std {g3['w_std_mean']:.4f} "
          f"-> {'PASS' if g3['pass'] else 'FAIL'}")
    print(f"[G4 deploy purity] diff {gates['G4_deploy_purity']['max_diff']:.2e} "
          f"-> {'PASS' if gates['G4_deploy_purity']['pass'] else 'FAIL'}")
    g5 = gates["G5_post_opened"]
    print(f"[G5 POST opened] post_strength {g5['post_strength']:.4f} (> {MIN_GATE}) "
          f"-> {'PASS' if g5['pass'] else 'FAIL (prep-only degenerate)'}")
    g6 = gates["G6_post_restores"]
    print(f"[G6 POST restores] mean RMS reduction {g6['mean_rms_reduction']:+.1%} "
          f"-> {'PASS' if g6['pass'] else 'FAIL (restorer hurts)'}")

    core = all(gates[k]["pass"] for k in ("G1_non_identity", "G2_no_blowup",
                                          "G3_w_nondegenerate", "G4_deploy_purity"))
    sandwich = core and g5["pass"] and g6["pass"]
    print(f"\nPRE-ONLY validity: {'PASS' if core else 'FAIL'}")
    print(f"SANDWICH arm claim: {'PASS -> run full eval' if sandwich else 'FAIL — prep-only arm still valid'}")
    sys.exit(0 if sandwich else 1)


if __name__ == "__main__":
    main()
