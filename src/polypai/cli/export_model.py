"""``polypai-export`` — export a checkpoint to ONNX and print the trtexec line."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export PolypAI to ONNX / TensorRT")
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out", required=True, help="output .onnx path")
    p.add_argument("--input-size", type=int, default=512)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--build-engine", action="store_true", help="also build a TensorRT engine")
    p.add_argument("--fp16", action="store_true", default=True)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import torch

    from polypai.export import build_trt_engine, export_onnx, trtexec_command
    from polypai.models import build_model

    model_cfg = None
    if args.config:
        from polypai.config import build_experiment

        model_cfg = build_experiment(args.config)[0]
    model = build_model(model_cfg)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt.get("ema", ckpt.get("model", ckpt)))
    model.eval()

    export_onnx(model, args.out, input_size=args.input_size, opset=args.opset)
    print(f"ONNX written to {args.out}")

    engine = args.out.replace(".onnx", ".engine")
    print("TensorRT build command:\n  " + trtexec_command(args.out, engine, fp16=args.fp16,
                                                          input_size=args.input_size))
    if args.build_engine:
        try:
            build_trt_engine(args.out, engine, fp16=args.fp16)
            print(f"Engine written to {engine}")
        except RuntimeError as e:
            print(e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
