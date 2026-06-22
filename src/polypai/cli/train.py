"""``polypai-train`` — train the unified model from a YAML config."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the unified PolypAI model")
    p.add_argument("--config", required=True, help="path to experiment YAML")
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=None, help="override config epochs")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from polypai.config import build_experiment
    from polypai.data.datamodule import build_dataloaders
    from polypai.engine import Trainer, UnifiedCriterion
    from polypai.models import build_model

    model_cfg, data_cfg, train_cfg = build_experiment(args.config)
    if args.epochs is not None:
        train_cfg.epochs = args.epochs
    train_cfg.device = args.device

    loaders = build_dataloaders(data_cfg)
    if "train" not in loaders:
        raise SystemExit(
            "No training data found. Set real dataset roots in the config's `data.datasets`."
        )
    model = build_model(model_cfg)
    criterion = UnifiedCriterion(num_seg_classes=model_cfg.seg_classes,
                                 cls_tasks=tuple(model_cfg.cls_heads.keys()))
    print(f"Model parameters: {model.num_parameters():,}")
    trainer = Trainer(model, criterion, loaders, train_cfg)
    if args.resume:
        trainer.load(args.resume)
    trainer.fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
