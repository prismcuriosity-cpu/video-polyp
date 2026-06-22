"""PolypAI: end-to-end colonoscopy polyp analysis.

The package is intentionally split so that the *pure-Python / NumPy* core
(diagnostic frame scoring, metrics, NBI synthesis) imports without PyTorch,
OpenCV, or a GPU.  Heavy submodules (``polypai.models``, ``polypai.engine``,
``polypai.explain``) import torch lazily and are only pulled in when used.

This keeps the clinically critical, deterministic components testable on any
machine (CI, laptop) while the deep-learning stack targets the RTX 5090
workstation described in ``docs/HARDWARE_OPTIMIZATION.md``.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
