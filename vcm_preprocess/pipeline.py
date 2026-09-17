"""Pipeline preprocessing frame anh, thiet ke de giu thong tin cho AI task."""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import json


@dataclass
class PreprocessConfig:
    denoise: float = 0.15          # 0..1, loc median nhe
    sharpen: float = 0.20          # 0..1, unsharp mask
    contrast: float = 1.05         # 0.8..1.3
    saturation: float = 1.00       # 0.7..1.3
    luma_gain: float = 1.00        # 0.85..1.15
    chroma_subsample: bool = False # bat khi codec dung 4:2:0

    def clipped(self) -> "PreprocessConfig":
        return PreprocessConfig(
            max(0, min(1, self.denoise)), max(0, min(1, self.sharpen)),
            max(.8, min(1.3, self.contrast)), max(.7, min(1.3, self.saturation)),
            max(.85, min(1.15, self.luma_gain)), bool(self.chroma_subsample))


def preprocess_image(src: str | Path, dst: str | Path, config: PreprocessConfig) -> None:
    """Xu ly mot frame. Pillow la dependency tuy chon de package van import duoc."""
    try:
        from PIL import Image, ImageEnhance, ImageFilter
    except ImportError as exc:
        raise RuntimeError("Can cai Pillow: pip install 'vcm-preprocess[image]'") from exc
    cfg = config.clipped(); image = Image.open(src).convert("RGB")
    if cfg.denoise > 0.01:
        radius = 1 if cfg.denoise < .45 else 2
        image = image.filter(ImageFilter.MedianFilter(size=radius * 2 + 1))
    image = ImageEnhance.Contrast(image).enhance(cfg.contrast)
    image = ImageEnhance.Color(image).enhance(cfg.saturation)
    if abs(cfg.luma_gain - 1) > .005:
        image = image.point(lambda p: max(0, min(255, int(p * cfg.luma_gain))))
    if cfg.sharpen > .01:
        image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=int(80 + 140 * cfg.sharpen), threshold=3))
    Path(dst).parent.mkdir(parents=True, exist_ok=True); image.save(dst, quality=95)


def preprocess_sequence(input_dir: str | Path, output_dir: str | Path, config: PreprocessConfig, pattern="*.png") -> int:
    files = sorted(Path(input_dir).glob(pattern)); Path(output_dir).mkdir(parents=True, exist_ok=True)
    for f in files: preprocess_image(f, Path(output_dir) / f.name, config)
    (Path(output_dir) / "preprocess.json").write_text(json.dumps(asdict(config.clipped()), indent=2), encoding="utf-8")
    return len(files)
