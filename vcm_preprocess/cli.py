from __future__ import annotations
import argparse, json
from dataclasses import asdict
from .pipeline import PreprocessConfig, preprocess_sequence

def main() -> None:
    p = argparse.ArgumentParser(description="VCM preprocessing")
    p.add_argument("input", help="Thu muc frame PNG/JPG")
    p.add_argument("output", help="Thu muc frame da xu ly")
    p.add_argument("--config", help="JSON config")
    a = p.parse_args(); cfg = PreprocessConfig()
    if a.config: cfg = PreprocessConfig(**json.loads(open(a.config, encoding="utf-8").read()))
    n = preprocess_sequence(a.input, a.output, cfg); print(json.dumps({"frames": n, "config": asdict(cfg.clipped())}, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
