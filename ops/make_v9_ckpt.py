"""Assemble the v9 warm-start checkpoint: PRE from v7-v1 (best pre-only) +
trunk/POST from the v8 STE checkpoint, into a PerCodecPostSandwich state.

Run: python ops/make_v9_ckpt.py \
        --v1 /tmp/v1_ckpt/.../preprocessor.pth \
        --ste /tmp/ste_out/.../preprocessor.pth \
        --out outputs/v9_warmstart.pth
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.percodec_sandwich import PerCodecPostSandwich  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True, help="v7 UP-VCM best checkpoint (PRE)")
    ap.add_argument("--ste", required=True, help="v8 STE sandwich checkpoint (POST trunk)")
    ap.add_argument("--out", default="outputs/v9_warmstart.pth")
    a = ap.parse_args()

    m = PerCodecPostSandwich()
    v1 = torch.load(a.v1, map_location="cpu")
    ste = torch.load(a.ste, map_location="cpu")

    # ORDER MATTERS: load_v8_sandwich copies pre.* too (the STE run's PRE),
    # so it must run FIRST — then v1's PRE overwrites it as the final word.
    rep = m.load_v8_sandwich(ste["model"] if "model" in ste else ste)
    m.load_pre_state(v1["model"] if "model" in v1 else v1)
    print("warm-start report (trunk from STE):", rep)
    print("PRE (final word) from v1")

    ck = {"model": m.state_dict(), "opt": None, "sched": None,
          "cfg": ste.get("cfg", {}), "epoch": 0, "global_step": 0,
          "best_val": None, "no_improve": 0}
    # stamp the arch so eval builds the right class
    ck["cfg"] = dict(ck["cfg"] or {})
    ck["cfg"]["model"] = dict(ck["cfg"].get("model") or {})
    ck["cfg"]["model"]["arch"] = "percodec_sandwich"
    ck["cfg"]["model"]["post_base"] = 32
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, a.out)
    ms = ck["model"]
    print(f"saved {a.out}: PRE dec={ms['pre.dec_strength'].item():+.4f} "
          f"stab={ms['pre.stab_strength'].item():+.4f} "
          f"POST strength={ms['post_strength'].item():+.4f}")




def make_dual(v1_path, e2_path, ste_path, out_path):
    """v9-b FINAL (audit 2026-09-11 finding #2): build the REPORTED model —
    DualCodecSandwich with PER-CODEC PREs — not DualPostSandwich.

      pre_264  <- v1 (v7 UP-VCM best, the E2 h264 record's PRE)
      pre_265  <- ste (the v8 STE run's PRE, the h265 record's PRE)
      post_264/post_265 <- ste's POST (the shared record holder; E2's POST
                           and STE's POST are the same weights by lineage)
    arch = "dualcodec" so eval rebuilds the right class.
    """
    import torch
    from src.models.dualpost_sandwich import DualCodecSandwich
    m = DualCodecSandwich()
    v1 = torch.load(v1_path, map_location="cpu")
    e2 = torch.load(e2_path, map_location="cpu")
    ste = torch.load(ste_path, map_location="cpu")
    rep = m.load_record_assembly(v1["model"], ste["model"])
    print("dualcodec assembly:", rep)
    cfg = dict(ste.get("cfg") or {})
    cfg["model"] = dict(cfg.get("model") or {})
    cfg["model"]["arch"] = "dualcodec"
    cfg["model"]["post_base"] = 32
    ck = {"model": m.state_dict(), "opt": None, "sched": None, "cfg": cfg,
          "epoch": 0, "global_step": 0, "best_val": None, "no_improve": 0}
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(ck, out_path)
    ms = ck["model"]
    print(f"saved {out_path}: PRE_264 dec={ms['pre_264.dec_strength'].item():+.4f} "
          f"| PRE_265 dec={ms['pre_265.dec_strength'].item():+.4f} "
          f"| POST_264={ms['post_strength_264'].item():+.4f} "
          f"| POST_265={ms['post_strength_265'].item():+.4f}")
    # e2_path stays an input for signature compatibility (its POST == ste's
    # POST by lineage: E2 = frankenstein-STE); ste's copy is canonical.


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "dual":
        # python ops/make_v9_ckpt.py dual <v1> <e2-frank-ste> <ste> <out>
        make_dual(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        main()
