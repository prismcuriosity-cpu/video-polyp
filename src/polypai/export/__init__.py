"""Deployment export: ONNX and TensorRT.

The model returns a dict (convenient for training); :class:`ExportWrapper`
flattens it to a fixed tuple of tensors that ONNX/TensorRT accept.  TensorRT
engine building is optional and guarded so importing this module never requires
the TensorRT runtime.
"""

from __future__ import annotations

from polypai.export.onnx_export import ExportWrapper, export_onnx
from polypai.export.tensorrt_export import build_trt_engine, trtexec_command

__all__ = ["ExportWrapper", "export_onnx", "build_trt_engine", "trtexec_command"]
