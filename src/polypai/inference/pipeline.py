"""End-to-end clinical pipeline: video -> informative frames -> detection /
segmentation / characterisation -> explainable report.

This is the top-level orchestrator that wires the polyp-centric frame selector
to the unified model and the post-processing, then assembles a
:class:`ProcedureReport`.  It runs without trained weights (random init) for
shape/flow testing; with trained weights it is the deployable inference entry.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from polypai.frame_selection import InformativeFrameSelector, SelectorConfig
from polypai.inference.report import LesionFinding, ProcedureReport
from polypai.structures import KudoPitPattern, NICEClass, ParisClass

_PARIS = [c.value for c in ParisClass]
_NICE = [c.value for c in NICEClass]
_KUDO = [c.value for c in KudoPitPattern]


@dataclass
class PipelineConfig:
    image_size: int = 512
    score_thresh: float = 0.3
    nms_iou: float = 0.5
    mm_per_pixel: float | None = None     # scope calibration; None -> relative size only
    device: str = "cuda"
    selector: SelectorConfig = field(default_factory=lambda: SelectorConfig(top_k=12, n_views=5))


class ClinicalPipeline:
    def __init__(self, model=None, selector: InformativeFrameSelector | None = None,
                 cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()
        self.model = model
        self.selector = selector or InformativeFrameSelector(config=self.cfg.selector)

    # -- per-frame model inference -----------------------------------------
    def _infer_frame(self, frame: np.ndarray) -> dict:
        import torch

        from polypai.data.transforms import default_val_transforms
        from polypai.inference.postprocess import decode_detections

        device = next(self.model.parameters()).device
        tfm = default_val_transforms(self.cfg.image_size)
        sample = tfm(image=frame, mask=np.zeros(frame.shape[:2], np.uint8))
        x = sample["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            out = self.model(x)
        dets = decode_detections(out["detection"], score_thresh=self.cfg.score_thresh,
                                 nms_iou=self.cfg.nms_iou)[0]
        seg = out["segmentation"]["seg_logits"].softmax(1)[0, 1].cpu().numpy()
        cls = {k: torch.softmax(v, 1)[0].cpu().numpy() for k, v in out["classification"].items()}
        unc = 0.0
        if "uncertainty" in out:
            unc = float(out["uncertainty"].exp().mean().cpu())
        scale = self.cfg.image_size / max(frame.shape[:2])
        return {"dets": dets, "seg": seg, "cls": cls, "uncertainty": unc, "scale": scale,
                "proc_size": x.shape[-2:]}

    def _finding_from(self, frame_idx: int, frame: np.ndarray, res: dict, k: int) -> LesionFinding:
        box = res["dets"]["boxes"][k].tolist()
        score = float(res["dets"]["scores"][k])
        # boxes are in processed (resized) coords -> back to original frame
        inv = 1.0 / max(res["scale"], 1e-6)
        box = [box[0] * inv, box[1] * inv, box[2] * inv, box[3] * inv]
        h, w = frame.shape[:2]
        diag = math.hypot(w, h)
        bdiag = math.hypot(box[2] - box[0], box[3] - box[1])  # in original-frame pixels
        size_mm = bdiag * self.cfg.mm_per_pixel if self.cfg.mm_per_pixel else None
        cls = res["cls"]
        paris = _PARIS[int(cls["paris"].argmax())] if "paris" in cls else None
        nice = _NICE[int(cls["nice"].argmax())] if "nice" in cls else None
        kudo = _KUDO[int(cls["kudo"].argmax())] if "kudo" in cls else None
        mal = float(cls["malignancy"][1]) if "malignancy" in cls else 0.0
        explanation = {
            "nice_conf": float(cls["nice"].max()) if "nice" in cls else 0.0,
            "paris_conf": float(cls["paris"].max()) if "paris" in cls else 0.0,
            "driver": "vascular/surface pattern (NICE)" if mal >= 0.5 else "regular pattern",
        }
        return LesionFinding(
            frame_index=frame_idx, bbox=[round(v, 1) for v in box], detection_confidence=score,
            mask_area_px=int((res["seg"] > 0.5).sum()), size_mm=size_mm,
            size_relative=float(bdiag / diag), paris=paris, nice=nice, kudo=kudo,
            malignancy_prob=mal, uncertainty=res["uncertainty"], explanation=explanation,
        )

    # -- public API ---------------------------------------------------------
    def process_frames(self, frames: Iterable[np.ndarray]) -> ProcedureReport:
        frames = list(frames)
        selection = self.selector.select(frames)
        report = ProcedureReport(
            n_frames_processed=len(frames),
            n_frames_selected=len(selection.selected),
            meta={"selected_indices": selection.selected_indices},
        )
        if self.model is None:
            # frame-selection-only mode: report informative frames without model
            for fs in selection.selected:
                report.findings.append(LesionFinding(
                    frame_index=fs.frame_index, bbox=[*_box(fs)], detection_confidence=fs.presence,
                    size_relative=0.0, explanation={"diagnostic_score": round(fs.total, 3)}))
            return report

        for fs in selection.selected:
            frame = frames[fs.frame_index]
            res = self._infer_frame(frame)
            for k in range(len(res["dets"]["scores"])):
                report.findings.append(self._finding_from(fs.frame_index, frame, res, k))
        return report

    def process_video(self, path: str, stride: int = 1, max_frames: int | None = None) -> ProcedureReport:
        from polypai.data.video import VideoFrameReader

        reader = VideoFrameReader(path, stride=stride, max_frames=max_frames,
                                  resize_to=self.cfg.image_size * 2)
        return self.process_frames(reader)


def _box(fs):
    if fs.candidate is not None:
        b = fs.candidate.bbox
        return (b.x1, b.y1, b.x2, b.y2)
    return (0.0, 0.0, 0.0, 0.0)
