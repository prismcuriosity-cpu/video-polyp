"""``polypai-infer`` — run the full clinical pipeline on a colonoscopy video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="End-to-end PolypAI video inference")
    p.add_argument("--video", required=True)
    p.add_argument("--config", default=None, help="model config YAML (for architecture)")
    p.add_argument("--checkpoint", default=None, help="trained weights; omit for selection-only")
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--mm-per-pixel", type=float, default=None)
    p.add_argument("--device", default="cuda")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from polypai.inference import ClinicalPipeline, PipelineConfig

    model = None
    if args.checkpoint:
        import torch

        from polypai.config import build_experiment
        from polypai.models import build_model

        model_cfg = build_experiment(args.config)[0] if args.config else None
        model = build_model(model_cfg)
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt.get("ema", ckpt.get("model", ckpt)))
        model.to(device).eval()

    cfg = PipelineConfig(device=args.device, mm_per_pixel=args.mm_per_pixel)
    pipe = ClinicalPipeline(model=model, cfg=cfg)
    report = pipe.process_video(args.video, stride=args.stride, max_frames=args.max_frames)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report.to_dict(), indent=2))
    (out / "report.md").write_text(report.to_markdown())
    print(report.to_markdown())
    print(f"\nReport written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
