"""Image-enhancement front-ends used by characterisation modules.

* Pseudo-NBI synthesis (Module 6) — converts white-light frames into a
  Narrow-Band-Imaging-like rendering that amplifies the mucosal/vascular
  pattern, so NICE classification can run even on white-light-only video.
* Super-resolution hook (Module 2) — interface for an SR pre-pass that helps
  diminutive (~2 mm) lesion detection; ships a deterministic Lanczos fallback.
"""

from __future__ import annotations

from polypai.enhance.nbi import (
    pseudo_nbi,
    vessel_density,
)
from polypai.enhance.superres import SuperResolver, lanczos_upsample

__all__ = ["pseudo_nbi", "vessel_density", "SuperResolver", "lanczos_upsample"]
