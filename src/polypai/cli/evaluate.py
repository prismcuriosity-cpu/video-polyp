"""``polypai-evaluate`` — evaluate a checkpoint on a dataset's val split."""

from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a PolypAI checkpoint")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--surface-metrics", action="store_true", help="also compute HD95/ASSD (slow)")
    p.add_argument("--out", default=None, help="optional JSON path for metrics")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import torch

    from polypai.config import build_experiment
    from polypai.data.datamodule import build_dataloaders
    from polypai.engine import Evaluator
    from polypai.models import build_model

    model_cfg, data_cfg, _ = build_experiment(args.config)
    loaders = build_dataloaders(data_cfg)
    if "val" not in loaders:
        raise SystemExit("No validation data found for the configured datasets.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("ema", ckpt.get("model", ckpt)))

    metrics = Evaluator(model, loaders["val"], device, surface_metrics=args.surface_metrics).run()
    print(json.dumps(metrics, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
