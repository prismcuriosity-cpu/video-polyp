"""Tests for ONNX export and the TensorRT command builder."""

import pytest

torch = pytest.importorskip("torch")

from polypai.export import ExportWrapper, build_trt_engine, trtexec_command  # noqa: E402
from polypai.models import ModelConfig, build_model  # noqa: E402


def _tiny_model():
    cfg = ModelConfig(stem_channels=12, stage_channels=(16, 24, 32, 48),
                      stage_depths=(1, 1, 1, 1), fpn_channels=24, seg_decoder_channels=24)
    return build_model(cfg)


def test_export_wrapper_returns_tuple():
    w = ExportWrapper(_tiny_model())
    out = w(torch.randn(1, 3, 96, 96))
    assert isinstance(out, tuple)
    assert out[0].shape[1] == 2  # seg logits channels


def test_onnx_export(tmp_path):
    onnx = pytest.importorskip("onnx")
    from polypai.export import export_onnx

    path = str(tmp_path / "model.onnx")
    export_onnx(_tiny_model(), path, input_size=96, batch=1)
    onnx.checker.check_model(onnx.load(path))


def test_trtexec_command_has_shapes():
    cmd = trtexec_command("m.onnx", "m.engine", fp16=True, input_size=512)
    assert "--onnx=m.onnx" in cmd and "--fp16" in cmd and "image:1x3x512x512" in cmd


def test_build_trt_engine_without_runtime_raises():
    with pytest.raises(RuntimeError):
        build_trt_engine("missing.onnx", "out.engine")
