"""ONNX export with a tensor-tuple wrapper and dynamic batch axis."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class ExportWrapper(nn.Module):
    """Wrap :class:`UnifiedPolypModel` to emit a flat, ONNX-friendly tuple.

    Returns ``(seg_logits, det_cls_p2..p5, det_reg_p2..p5, det_ctr_p2..p5,
    paris, nice, kudo, malignancy)`` — all tensors, no Python dict.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        out = self.model(x)
        seg = out["segmentation"]["seg_logits"]
        det = out["detection"]
        cls = out["classification"]
        flat = [seg]
        flat += list(det["cls"]) + list(det["reg"]) + list(det["centerness"])
        flat += [cls["paris"], cls["nice"], cls["kudo"], cls["malignancy"]]
        return tuple(flat)


def export_onnx(model: nn.Module, path: str, input_size: int = 512, batch: int = 1,
                opset: int = 17, dynamic_batch: bool = True) -> str:
    """Export to ONNX.  Returns the output path."""
    wrapper = ExportWrapper(model)
    dummy = torch.randn(batch, 3, input_size, input_size)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out_names = (
        ["seg_logits"]
        + [f"det_cls_{l}" for l in ("p2", "p3", "p4", "p5")]
        + [f"det_reg_{l}" for l in ("p2", "p3", "p4", "p5")]
        + [f"det_ctr_{l}" for l in ("p2", "p3", "p4", "p5")]
        + ["paris", "nice", "kudo", "malignancy"]
    )
    dynamic_axes = {"image": {0: "batch"}} if dynamic_batch else None
    if dynamic_axes:
        for n in out_names:
            dynamic_axes[n] = {0: "batch"}
    kwargs = dict(input_names=["image"], output_names=out_names, opset_version=opset,
                  dynamic_axes=dynamic_axes, do_constant_folding=True)
    try:
        # prefer the stable TorchScript exporter (no onnxscript dependency)
        torch.onnx.export(wrapper, dummy, path, dynamo=False, **kwargs)
    except TypeError:
        # older torch without the ``dynamo`` kwarg
        torch.onnx.export(wrapper, dummy, path, **kwargs)
    return path
