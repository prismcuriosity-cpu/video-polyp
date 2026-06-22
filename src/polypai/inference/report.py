"""Structured, explainable clinical decision-support output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LesionFinding:
    frame_index: int
    bbox: list[float]                       # xyxy in the selected frame
    detection_confidence: float
    mask_area_px: int = 0
    size_mm: Optional[float] = None         # absolute estimate if calibration provided
    size_relative: float = 0.0              # bbox-diagonal / frame-diagonal
    paris: Optional[str] = None
    nice: Optional[str] = None
    kudo: Optional[str] = None
    malignancy_prob: float = 0.0
    uncertainty: float = 0.0
    explanation: dict = field(default_factory=dict)

    @property
    def risk_band(self) -> str:
        if self.malignancy_prob >= 0.66:
            return "high"
        if self.malignancy_prob >= 0.33:
            return "intermediate"
        return "low"

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["risk_band"] = self.risk_band
        return d


@dataclass
class ProcedureReport:
    findings: list[LesionFinding] = field(default_factory=list)
    n_frames_processed: int = 0
    n_frames_selected: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def n_lesions(self) -> int:
        return len(self.findings)

    @property
    def max_malignancy_risk(self) -> float:
        return max((f.malignancy_prob for f in self.findings), default=0.0)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "n_lesions": self.n_lesions,
                "n_frames_processed": self.n_frames_processed,
                "n_frames_selected": self.n_frames_selected,
                "max_malignancy_risk": round(self.max_malignancy_risk, 3),
            },
            "findings": [f.to_dict() for f in self.findings],
            "meta": self.meta,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Colonoscopy AI Report",
            "",
            f"- Frames processed: **{self.n_frames_processed}**",
            f"- Informative frames selected: **{self.n_frames_selected}**",
            f"- Lesions characterised: **{self.n_lesions}**",
            f"- Highest malignancy risk: **{self.max_malignancy_risk:.2f}**",
            "",
            "## Findings",
        ]
        if not self.findings:
            lines.append("_No polyp detected in the selected frames._")
        for i, f in enumerate(self.findings, 1):
            size = f"{f.size_mm:.1f} mm" if f.size_mm is not None else f"rel {f.size_relative:.2f}"
            lines += [
                f"### Lesion {i} (frame {f.frame_index})",
                f"- Detection confidence: {f.detection_confidence:.2f}",
                f"- Estimated size: {size}",
                f"- Paris: **{f.paris}** | NICE: **{f.nice}** | Kudo: **{f.kudo}**",
                f"- Malignancy probability: **{f.malignancy_prob:.2f}** "
                f"(risk: **{f.risk_band}**), uncertainty {f.uncertainty:.2f}",
            ]
            if f.explanation:
                lines.append(f"- Rationale: {_fmt_explanation(f.explanation)}")
            lines.append("")
        return "\n".join(lines)


def _fmt_explanation(expl: dict) -> str:
    parts = []
    for k, v in expl.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.2f}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)
