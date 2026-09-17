"""Standard-codec anchors: bare H.264 (x264) and H.265 (x265) via ffmpeg.

These are the *baselines* the preprocessor+CompressAI pipeline is compared
against. A clip is piped to ffmpeg as raw RGB, encoded at a constant QP, and
decoded straight back to raw RGB. We measure the real coded size, so the
reported bpp is honest and directly comparable to CompressAI's entropy-coded
bpp.

    bpp = 8 * encoded_file_bytes / (T * H * W)

No neural network is involved -- this is exactly "what you get from H.264/H.265
alone", which is the anchor curve for the BD-Rate comparison.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

_ENCODER = {"h264": "libx264", "h265": "libx265"}
_MUXER = {"h264": "h264", "h265": "hevc"}  # raw-bitstream muxer (NOT "264"/"265")


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


class StandardCodec:
    """ffmpeg-backed constant-QP H.264 / H.265 codec.

    Args:
        codec: "h264" or "h265".
        qp: constant quantisation parameter (higher = lower rate).
        preset: x264/x265 speed preset.
        fps: container frame-rate (only affects timing metadata, not quality).
    """

    def __init__(
        self,
        codec: str = "h264",
        qp: int = 35,
        preset: str = "medium",
        fps: int = 25,
    ):
        if codec not in _ENCODER:
            raise ValueError(f"codec must be one of {list(_ENCODER)}")
        self.codec = codec
        self.qp = qp
        self.preset = preset
        self.fps = fps

    # -- single clip -------------------------------------------------------
    def _encode_decode_clip(
        self, clip: np.ndarray, qp: int | None = None
    ) -> Tuple[np.ndarray, float]:
        """clip: [T,H,W,C] uint8 RGB. Returns (recon [T,H,W,C] uint8, bpp)."""
        t, h, w, c = clip.shape
        assert c == 3
        qp = self.qp if qp is None else qp
        with tempfile.TemporaryDirectory() as td:
            bitstream = Path(td) / f"clip.{'264' if self.codec == 'h264' else '265'}"

            # RGB -> encoded elementary stream.
            enc = subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "pipe:0",
                    "-c:v", _ENCODER[self.codec],
                    "-preset", self.preset, "-qp", str(qp),
                    "-pix_fmt", "yuv420p",
                    "-f", _MUXER[self.codec],
                    str(bitstream),
                ],
                input=clip.tobytes(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
            )
            _ = enc
            coded_bytes = bitstream.stat().st_size

            # Encoded stream -> raw RGB back.
            dec = subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(bitstream),
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            raw = np.frombuffer(dec.stdout, dtype=np.uint8)

        n = raw.size // (h * w * c)
        if n == 0:
            recon = clip.copy()
        else:
            recon = raw[: n * h * w * c].reshape(n, h, w, c)
            if n < t:  # pad short decode by repeating last frame
                pad = np.repeat(recon[-1:], t - n, axis=0)
                recon = np.concatenate([recon, pad], axis=0)
            recon = recon[:t]
        bpp = 8.0 * coded_bytes / (t * h * w)
        return recon, bpp

    # -- batch of clips ----------------------------------------------------
    @torch.no_grad()
    def compress_decompress_items(
        self, x: torch.Tensor, qp: int | None = None
    ) -> Tuple[torch.Tensor, List[float]]:
        """Encode a batch and return the real bpp of every input sequence."""
        b, c, t, h, w = x.shape
        arr = (x.clamp(0, 1) * 255).round().byte().cpu().numpy()  # [B,C,T,H,W]
        recons = []
        bpps = []
        for i in range(b):
            clip = np.transpose(arr[i], (1, 2, 3, 0))  # [T,H,W,C]
            rec, bpp = self._encode_decode_clip(clip, qp=qp)
            recons.append(np.transpose(rec, (3, 0, 1, 2)))  # [C,T,H,W]
            bpps.append(bpp)
        out = torch.from_numpy(np.stack(recons)).float().div_(255.0).to(x.device)
        return out, [float(v) for v in bpps]

    @torch.no_grad()
    def compress_decompress(
        self, x: torch.Tensor, qp: int | None = None
    ) -> Tuple[torch.Tensor, float]:
        """Encode a batch and return its mean bpp (backward-compatible API)."""
        out, bpps = self.compress_decompress_items(x, qp=qp)
        return out, float(np.mean(bpps))
