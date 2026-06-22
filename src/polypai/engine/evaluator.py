"""Validation evaluator computing the segmentation + classification metrics."""

from __future__ import annotations

import numpy as np
import torch

from polypai.metrics import (
    accuracy,
    assd,
    auc_score,
    dice_score,
    f1_score,
    hd95,
    iou_score,
)

IGNORE_INDEX = -100


class Evaluator:
    def __init__(self, model, loader, device, surface_metrics: bool = False):
        self.model = model
        self.loader = loader
        self.device = device
        self.surface_metrics = surface_metrics  # HD95/ASSD are slow; off by default

    @torch.no_grad()
    def run(self) -> dict:
        self.model.eval()
        dices, ious, hds, assds = [], [], [], []
        cls_store: dict[str, dict] = {}

        for batch in self.loader:
            images = batch["image"].to(self.device)
            out = self.model(images)
            if "segmentation" in out:
                pred = out["segmentation"]["seg_logits"].argmax(1).cpu().numpy()
                gt = batch["mask"].numpy()
                for p, g in zip(pred, gt):
                    dices.append(dice_score(p, g))
                    ious.append(iou_score(p, g))
                    if self.surface_metrics:
                        hds.append(hd95(p, g))
                        assds.append(assd(p, g))
            if "classification" in out:
                for task, logits in out["classification"].items():
                    y = batch["labels"][task].numpy()
                    valid = y != IGNORE_INDEX
                    if valid.any():
                        store = cls_store.setdefault(task, {"y": [], "pred": [], "score": []})
                        prob = torch.softmax(logits, 1).cpu().numpy()
                        store["y"].extend(y[valid].tolist())
                        store["pred"].extend(prob[valid].argmax(1).tolist())
                        store["score"].extend(prob[valid].tolist())

        metrics = {
            "dice": float(np.mean(dices)) if dices else 0.0,
            "iou": float(np.mean(ious)) if ious else 0.0,
        }
        if hds:
            metrics["hd95"] = float(np.mean([h for h in hds if np.isfinite(h)]))
            metrics["assd"] = float(np.mean([a for a in assds if np.isfinite(a)]))
        for task, s in cls_store.items():
            if not s["y"]:
                continue
            k = int(max(s["y"]) + 1)
            metrics[f"{task}_acc"] = accuracy(s["y"], s["pred"])
            metrics[f"{task}_f1"] = f1_score(s["y"], s["pred"], max(k, max(s["pred"]) + 1))
            try:
                metrics[f"{task}_auc"] = auc_score(np.array(s["y"]), np.array(s["score"]), k)
            except Exception:
                pass
        return metrics
