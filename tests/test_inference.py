"""Tests for detection post-processing and the end-to-end clinical pipeline."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from polypai.inference import ClinicalPipeline, PipelineConfig, decode_detections, nms  # noqa: E402
from polypai.frame_selection import SelectorConfig  # noqa: E402
from polypai.models import ModelConfig, build_model  # noqa: E402

RNG = np.random.default_rng(3)


def _blob_frame(h=160, w=160, cx=80, cy=80, r=26):
    img = np.full((h, w, 3), 0.35, np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    img[disk] = [0.85, 0.3, 0.28]
    return (np.clip(img + RNG.normal(0, 0.01, img.shape), 0, 1) * 255).astype(np.uint8)


def _tiny_model():
    cfg = ModelConfig(stem_channels=12, stage_channels=(16, 24, 32, 48),
                      stage_depths=(1, 1, 1, 1), fpn_channels=24, seg_decoder_channels=24)
    return build_model(cfg)


def test_nms_suppresses_overlap():
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, 0.5)
    assert 0 in keep.tolist() and 2 in keep.tolist() and 1 not in keep.tolist()


def test_decode_detections_schema():
    model = _tiny_model().eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 128, 128))
    dets = decode_detections(out["detection"], score_thresh=0.0, nms_iou=0.5, max_det=20)
    assert len(dets) == 2
    assert "boxes" in dets[0] and dets[0]["boxes"].shape[1] == 4


def test_pipeline_selection_only_mode():
    frames = [np.full((160, 160, 3), 100, np.uint8) for _ in range(4)]  # no polyp
    frames += [_blob_frame(cx=70 + 4 * i) for i in range(8)]            # polyp present
    pipe = ClinicalPipeline(model=None, cfg=PipelineConfig(
        selector=SelectorConfig(top_k=3, n_views=2, min_presence=0.05, temporal_min_distance=2)))
    report = pipe.process_frames(frames)
    assert report.n_frames_processed == 12
    assert report.n_frames_selected >= 1
    d = report.to_dict()
    assert d["summary"]["n_frames_processed"] == 12


def test_pipeline_with_model_runs_and_reports():
    frames = [_blob_frame(cx=70 + 5 * i) for i in range(8)]
    model = _tiny_model().eval()
    cfg = PipelineConfig(device="cpu", image_size=128,
                         selector=SelectorConfig(top_k=2, n_views=2, min_presence=0.05,
                                                 temporal_min_distance=2),
                         mm_per_pixel=0.08, score_thresh=0.0)
    pipe = ClinicalPipeline(model=model, cfg=cfg)
    report = pipe.process_frames(frames)
    md = report.to_markdown()
    assert "Colonoscopy AI Report" in md
    assert isinstance(report.to_dict(), dict)
    # with score_thresh=0 the (random) detector yields findings we can characterise
    for f in report.findings:
        assert f.nice in ("1", "2", "3")
        assert 0.0 <= f.malignancy_prob <= 1.0
