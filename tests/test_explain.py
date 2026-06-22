"""Tests for Grad-CAM, uncertainty, and calibration."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from polypai.explain import (  # noqa: E402
    GradCAM,
    GradCAMpp,
    TemperatureScaler,
    expected_calibration_error,
    overlay_heatmap,
    predictive_entropy,
    segmentation_uncertainty,
)
from polypai.models import ModelConfig, build_model  # noqa: E402


def _tiny_model():
    cfg = ModelConfig(stem_channels=12, stage_channels=(16, 24, 32, 48),
                      stage_depths=(1, 1, 1, 1), fpn_channels=24, seg_decoder_channels=24)
    return build_model(cfg)


def test_gradcam_shape_and_range():
    model = _tiny_model().eval()
    cam_engine = GradCAM(model, model.encoder.stages[-1])
    x = torch.randn(2, 3, 96, 96)
    cam = cam_engine(x, lambda out: out["classification"]["malignancy"][:, 1].sum())
    assert cam.shape == (2, 96, 96)
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-5
    cam_engine.remove()


def test_gradcampp_runs():
    model = _tiny_model().eval()
    cam_engine = GradCAMpp(model, model.encoder.stages[-1])
    x = torch.randn(1, 3, 96, 96)
    cam = cam_engine(x, lambda out: out["segmentation"]["seg_logits"][:, 1].mean())
    assert cam.shape == (1, 96, 96)
    cam_engine.remove()


def test_overlay_heatmap():
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    cam = np.random.rand(64, 64).astype(np.float32)
    out = overlay_heatmap(img, cam)
    assert out.shape == (64, 64, 3) and out.dtype == np.uint8


def test_uncertainty_maps():
    model = _tiny_model().eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 96, 96))
    ent = predictive_entropy(out["segmentation"]["seg_logits"])
    assert ent.shape == (1, 96, 96)
    unc = segmentation_uncertainty(out)
    assert unc.shape[0] == 1


def test_ece_and_temperature_scaling():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=200)
    logits = torch.randn(200, 3) * 3.0  # over-confident
    probs = torch.softmax(logits, 1).numpy()
    ece_before = expected_calibration_error(probs, labels)
    ts = TemperatureScaler().fit(logits, torch.tensor(labels))
    probs_after = torch.softmax(ts(logits), 1).detach().numpy()
    ece_after = expected_calibration_error(probs_after, labels)
    assert ts.temperature > 0
    assert ece_after <= ece_before + 0.05  # calibration should not get much worse
