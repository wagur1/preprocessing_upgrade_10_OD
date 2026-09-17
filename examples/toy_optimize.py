"""Vi du chay optimizer ma khong can codec that."""
from vcm_preprocess import PreprocessConfig, RandomSearch

def evaluator(cfg: PreprocessConfig):
    # Thay ham nay bang pipeline: preprocess -> encode -> decode -> AI inference.
    quality = 0.70 + 0.08 * cfg.sharpen - 0.05 * cfg.denoise + 0.03 * (cfg.contrast - 1.0)
    bitrate = 850 + 180 * cfg.sharpen + 30 * cfg.contrast
    return {"task_quality": quality, "bitrate_kbps": bitrate}

if __name__ == "__main__":
    result = RandomSearch(evaluator, iterations=25, seed=42).run()
    print(result)
