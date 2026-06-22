"""Shared, framework-agnostic data structures used across the pipeline.

These are plain dataclasses (no torch dependency) so that the frame-selection
front-end, the evaluation code, and the clinical-report writer can all exchange
results without importing the deep-learning stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ParisClass(str, Enum):
    """Paris endoscopic classification of superficial lesions."""

    IP = "0-Ip"      # pedunculated
    IS = "0-Is"      # sessile
    IIA = "0-IIa"    # slightly elevated
    IIB = "0-IIb"    # completely flat
    IIC = "0-IIc"    # slightly depressed
    III = "0-III"    # excavated/ulcerated


class NICEClass(str, Enum):
    """NICE (NBI International Colorectal Endoscopic) optical-diagnosis classes."""

    TYPE1 = "1"  # hyperplastic / non-adenomatous
    TYPE2 = "2"  # adenoma
    TYPE3 = "3"  # deep submucosal invasive carcinoma


class KudoPitPattern(str, Enum):
    """Kudo pit-pattern classification."""

    I = "I"        # round pits (normal)
    II = "II"      # stellar/papillary (hyperplastic)
    IIIS = "IIIs"  # small tubular
    IIIL = "IIIL"  # large tubular
    IV = "IV"      # branch-like / gyrus
    VI = "VI"      # irregular
    VN = "VN"      # non-structural (invasive)


@dataclass
class BBox:
    """Axis-aligned bounding box in pixel coordinates (xyxy)."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0
    label: int = 0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    def clip(self, w: int, h: int) -> BBox:
        return BBox(
            x1=float(np.clip(self.x1, 0, w)),
            y1=float(np.clip(self.y1, 0, h)),
            x2=float(np.clip(self.x2, 0, w)),
            y2=float(np.clip(self.y2, 0, h)),
            score=self.score,
            label=self.label,
        )

    def iou(self, other: BBox) -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class PolypCandidate:
    """A candidate polyp region in a single frame."""

    bbox: BBox
    presence: float                         # P: confidence a polyp is present [0, 1]
    mask: np.ndarray | None = None       # optional binary ROI mask, HxW {0,1}
    embedding: np.ndarray | None = None  # appearance embedding for dedup/multi-view


@dataclass
class QualityBreakdown:
    """ROI-localised image-quality sub-scores, each in [0, 1] (higher = better)."""

    sharpness: float = 0.0          # inverse motion blur
    contrast: float = 0.0           # local RMS contrast over lesion
    specular_free: float = 0.0      # 1 - specular reflection fraction
    exposure: float = 0.0           # 1 - over/under-exposure fraction
    noise_free: float = 0.0         # inverse sensor-noise estimate

    def aggregate(self, weights: dict | None = None) -> float:
        w = weights or {
            "sharpness": 0.30,
            "contrast": 0.20,
            "specular_free": 0.20,
            "exposure": 0.15,
            "noise_free": 0.15,
        }
        total = sum(w.values())
        return float(sum(getattr(self, k) * v for k, v in w.items()) / max(total, 1e-8))


@dataclass
class FrameScore:
    """Full diagnostic-usefulness breakdown for one frame (eq. in docs/ARCHITECTURE)."""

    frame_index: int
    presence: float = 0.0          # P
    visibility: float = 0.0        # V
    texture: float = 0.0           # T (pit-pattern / surface clarity)
    quality: float = 0.0           # Q (ROI image quality, aggregated)
    boundary: float = 0.0          # B (boundary completeness)
    vascular: float = 0.0          # extra: vascular-pattern visibility
    total: float = 0.0             # S = weighted combination
    candidate: PolypCandidate | None = None
    quality_breakdown: QualityBreakdown | None = None
    view_descriptor: np.ndarray | None = None
    phash: int | None = None

    def as_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "P_presence": round(self.presence, 4),
            "V_visibility": round(self.visibility, 4),
            "T_texture": round(self.texture, 4),
            "Q_quality": round(self.quality, 4),
            "B_boundary": round(self.boundary, 4),
            "vascular": round(self.vascular, 4),
            "S_total": round(self.total, 4),
        }


@dataclass
class SelectionResult:
    """Output of the informative-frame selector."""

    selected: list[FrameScore] = field(default_factory=list)
    all_scores: list[FrameScore] = field(default_factory=list)
    rejected_indices: list[int] = field(default_factory=list)
    view_groups: dict[int, list[int]] = field(default_factory=dict)

    @property
    def selected_indices(self) -> list[int]:
        return [fs.frame_index for fs in self.selected]
