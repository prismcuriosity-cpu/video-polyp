"""``polypai-select-frames`` — run polyp-centric informative frame selection.

Works *today* without any trained weights (classical saliency candidate
detector + ROI diagnostic scoring).  Plug in a trained Module-2 detector by
passing ``--detector-ckpt`` once available.

Examples
--------
    polypai-select-frames --video case.mp4 --out out/ --top-k 12 --stride 2
    polypai-select-frames --frames-dir frames/ --out out/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from polypai.frame_selection import (
    DiagnosticScoreWeights,
    InformativeFrameSelector,
    SelectorConfig,
)


def _read_frames(args) -> list[np.ndarray]:
    if args.video:
        from polypai.data.video import VideoFrameReader

        return list(VideoFrameReader(args.video, stride=args.stride, max_frames=args.max_frames,
                                     resize_to=args.resize))
    if args.frames_dir:
        import cv2

        paths = sorted(Path(args.frames_dir).glob("*"))
        frames = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is not None:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return frames
    raise SystemExit("Provide --video or --frames-dir")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Polyp-centric informative frame selection")
    src = p.add_argument_group("input")
    src.add_argument("--video", type=str, default=None)
    src.add_argument("--frames-dir", type=str, default=None)
    src.add_argument("--stride", type=int, default=1)
    src.add_argument("--max-frames", type=int, default=None)
    src.add_argument("--resize", type=int, default=1024, help="longest-side resize on read")
    out = p.add_argument_group("output / budget")
    out.add_argument("--out", type=str, required=True)
    out.add_argument("--top-k", type=int, default=12)
    out.add_argument("--n-views", type=int, default=5)
    out.add_argument("--min-presence", type=float, default=0.12)
    out.add_argument("--detector-ckpt", type=str, default=None,
                     help="optional trained detector checkpoint (TorchScript/state_dict)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    frames = _read_frames(args)
    if not frames:
        print("No frames read."); return 1

    detector = None
    if args.detector_ckpt:
        import torch

        from polypai.frame_selection.candidate import from_torch_detector

        model = torch.jit.load(args.detector_ckpt) if args.detector_ckpt.endswith(".ts") \
            else torch.load(args.detector_ckpt)
        detector = from_torch_detector(model, device="cuda" if torch.cuda.is_available() else "cpu")

    cfg = SelectorConfig(weights=DiagnosticScoreWeights(), top_k=args.top_k,
                         n_views=args.n_views, min_presence=args.min_presence)
    selector = InformativeFrameSelector(detector=detector, config=cfg)
    result = selector.select(frames)

    out_dir = Path(args.out)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        for rank, fs in enumerate(result.selected):
            img = frames[fs.frame_index]
            cv2.imwrite(str(out_dir / "frames" / f"rank{rank:02d}_frame{fs.frame_index:06d}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    except Exception as e:  # pragma: no cover
        print(f"[warn] could not write frames: {e}")

    summary = {
        "n_frames": len(frames),
        "n_selected": len(result.selected),
        "n_rejected": len(result.rejected_indices),
        "selected": [fs.as_dict() for fs in result.selected],
        "view_groups": {str(k): v for k, v in result.view_groups.items()},
    }
    (out_dir / "selection.json").write_text(json.dumps(summary, indent=2))
    print(f"Selected {len(result.selected)}/{len(frames)} frames -> {out_dir}")
    for fs in result.selected:
        print(f"  frame {fs.frame_index:6d}  S={fs.total:.3f}  "
              f"(P={fs.presence:.2f} V={fs.visibility:.2f} T={fs.texture:.2f} "
              f"Q={fs.quality:.2f} B={fs.boundary:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
