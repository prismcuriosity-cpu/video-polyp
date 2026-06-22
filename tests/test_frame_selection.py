"""Unit tests for the polyp-centric informative frame selection module.

These exercise the deterministic, classical-CV core end-to-end with synthetic
images, so they run in CI without a GPU, torch, or real colonoscopy data.
"""

import numpy as np
import pytest

from polypai.frame_selection import (
    DiagnosticScoreWeights,
    InformativeFrameSelector,
    SelectorConfig,
    score_frame,
)
from polypai.frame_selection.candidate import SaliencyCandidateDetector
from polypai.frame_selection.dedup import dedup_by_phash, hamming, phash
from polypai.frame_selection.imageops import gaussian_blur, resize, to_gray, to_hsv
from polypai.frame_selection.quality import (
    assess_roi_quality,
    exposure_score,
    sharpness_score,
    specular_free_score,
)
from polypai.frame_selection.temporal import find_peak_indices, smooth_scores, temporal_nms
from polypai.frame_selection.texture import texture_clarity_score
from polypai.frame_selection.vascular import vascular_visibility_score
from polypai.frame_selection.visibility import (
    boundary_completeness_score,
    visibility_score,
)
from polypai.structures import BBox, FrameScore, PolypCandidate

RNG = np.random.default_rng(0)


def _blob_image(h=160, w=160, cx=80, cy=80, r=28, bright=0.85, bg=0.35):
    img = np.full((h, w, 3), bg, dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    img[disk] = [bright, 0.35 * bright, 0.3 * bright]
    img += RNG.normal(0, 0.01, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


# --------------------------------------------------------------------------- #
# imageops
# --------------------------------------------------------------------------- #
def test_to_gray_and_hsv_ranges():
    img = RNG.random((32, 32, 3)).astype(np.float32)
    g = to_gray(img)
    assert g.shape == (32, 32)
    assert 0.0 <= g.min() and g.max() <= 1.0
    hsv = to_hsv(img)
    assert hsv.shape == (32, 32, 3)
    assert 0.0 <= hsv[..., 2].min() and hsv[..., 2].max() <= 1.0001


def test_resize_shape_and_values():
    img = RNG.random((40, 50)).astype(np.float32)
    out = resize(img, (20, 25))
    assert out.shape == (20, 25)
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------- #
# quality (ROI-localised)
# --------------------------------------------------------------------------- #
def test_sharpness_drops_with_blur():
    sharp = RNG.random((64, 64)).astype(np.float32)
    blurred = gaussian_blur(sharp, sigma=3.0)
    assert sharpness_score(sharp) > sharpness_score(blurred)


def test_specular_penalised():
    clean = np.full((40, 40, 3), 0.4, dtype=np.float32)
    glare = clean.copy()
    glare[10:25, 10:25] = 1.0  # bright + desaturated highlight
    assert specular_free_score(glare) < specular_free_score(clean)


def test_exposure_penalises_clipping():
    mid = np.full((40, 40, 3), 0.5, dtype=np.float32)
    blown = mid.copy()
    blown[:20] = 1.0
    assert exposure_score(blown) < exposure_score(mid)


def test_quality_breakdown_in_unit_range():
    qb = assess_roi_quality(_blob_image())
    agg = qb.aggregate()
    assert 0.0 <= agg <= 1.0
    for v in (qb.sharpness, qb.contrast, qb.specular_free, qb.exposure, qb.noise_free):
        assert 0.0 <= v <= 1.0


# --------------------------------------------------------------------------- #
# texture & vascular
# --------------------------------------------------------------------------- #
def test_texture_higher_for_structured():
    flat = np.full((64, 64, 3), 0.5, dtype=np.float32)
    tex = np.kron(RNG.random((16, 16)), np.ones((4, 4)))
    tex = np.repeat(tex[..., None], 3, axis=2).astype(np.float32)
    assert texture_clarity_score(tex) > texture_clarity_score(flat)


def test_vascular_higher_with_vessels():
    flat = np.full((80, 80, 3), 0.7, dtype=np.float32)
    vessels = flat.copy()
    for c in (20, 40, 60):
        vessels[:, c - 1 : c + 1] = 0.2  # thin dark tubular structures
    assert vascular_visibility_score(vessels) > vascular_visibility_score(flat)


# --------------------------------------------------------------------------- #
# visibility & boundary
# --------------------------------------------------------------------------- #
def test_centered_box_beats_edge_box():
    frame = _blob_image()
    centered = BBox(60, 60, 100, 100)
    edge = BBox(0, 0, 40, 40)
    assert visibility_score(frame, centered) > visibility_score(frame, edge)


def test_boundary_complete_mask_beats_cutoff():
    frame = _blob_image()
    h, w = frame.shape[:2]
    full = np.zeros((h, w), np.uint8)
    full[60:100, 60:100] = 1
    cut = np.zeros((h, w), np.uint8)
    cut[0:40, 0:40] = 1  # touches top-left border
    box_full = BBox(60, 60, 100, 100)
    box_cut = BBox(0, 0, 40, 40)
    assert boundary_completeness_score(frame, box_full, full) > boundary_completeness_score(
        frame, box_cut, cut
    )


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def test_score_frame_outputs_unit_range():
    frame = _blob_image()
    cand = PolypCandidate(bbox=BBox(52, 52, 108, 108), presence=0.9)
    fs = score_frame(frame, cand, frame_index=3)
    assert isinstance(fs, FrameScore)
    assert fs.frame_index == 3
    assert 0.0 <= fs.total <= 1.0
    for v in (fs.presence, fs.visibility, fs.texture, fs.quality, fs.boundary, fs.vascular):
        assert 0.0 <= v <= 1.0


def test_weights_normalise():
    w = DiagnosticScoreWeights(alpha=2, beta=2, gamma=2, delta=2, epsilon=1, zeta=1).normalised()
    total = w.alpha + w.beta + w.gamma + w.delta + w.epsilon + w.zeta
    assert abs(total - 1.0) < 1e-6


# --------------------------------------------------------------------------- #
# candidate detector
# --------------------------------------------------------------------------- #
def test_saliency_detector_finds_blob():
    frame = _blob_image(cx=100, cy=70, r=22)
    det = SaliencyCandidateDetector()
    cands = det.detect(frame)
    assert len(cands) >= 1
    box = cands[0].bbox
    assert box.x1 <= 100 <= box.x2 and box.y1 <= 70 <= box.y2


def test_saliency_detector_rejects_flat():
    # deterministic (independent of module-level RNG state): a near-flat frame has
    # no spatially-coherent salient region, so no candidate should be proposed.
    rng = np.random.default_rng(123)
    flat = np.full((120, 120, 3), 0.4, dtype=np.float32) + rng.normal(0, 0.01, (120, 120, 3)).astype(
        np.float32
    )
    det = SaliencyCandidateDetector(thr_k=2.0)
    assert det.detect(np.clip(flat, 0, 1)) == []


def test_saliency_detector_rejects_constant():
    assert SaliencyCandidateDetector().detect(np.full((100, 100, 3), 0.5, np.float32)) == []


# --------------------------------------------------------------------------- #
# temporal
# --------------------------------------------------------------------------- #
def test_smoothing_reduces_variance():
    seq = RNG.random(50)
    assert smooth_scores(seq, 7).var() < seq.var()


def test_find_peaks_spacing():
    seq = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    peaks = find_peak_indices(seq, min_distance=3)
    assert all(abs(a - b) >= 3 for a in peaks for b in peaks if a != b)


def test_temporal_nms_spacing_and_order():
    fs = [FrameScore(frame_index=i, total=float(t)) for i, t in
          enumerate([0.1, 0.9, 0.85, 0.2, 0.95, 0.3, 0.1])]
    kept = temporal_nms(fs, min_distance=2)
    idxs = [k.frame_index for k in kept]
    assert all(abs(a - b) >= 2 for a in idxs for b in idxs if a != b)
    assert kept[0].total >= kept[-1].total


# --------------------------------------------------------------------------- #
# dedup
# --------------------------------------------------------------------------- #
def test_phash_identical_and_different():
    a = _blob_image(cx=80, cy=80)
    assert hamming(phash(a), phash(a.copy())) == 0
    b = RNG.random((160, 160, 3)).astype(np.float32)
    assert hamming(phash(a), phash(b)) > 0


def test_dedup_removes_duplicates():
    a = _blob_image(cx=80, cy=80)
    ha = phash(a)
    fs = [
        FrameScore(frame_index=0, total=0.9, phash=ha),
        FrameScore(frame_index=1, total=0.8, phash=ha),  # duplicate
        FrameScore(frame_index=2, total=0.7, phash=ha ^ ((1 << 40) - 1)),  # far
    ]
    kept = dedup_by_phash(fs, max_hamming=6)
    assert 0 in [k.frame_index for k in kept]
    assert 1 not in [k.frame_index for k in kept]


# --------------------------------------------------------------------------- #
# end-to-end selector
# --------------------------------------------------------------------------- #
def test_selector_end_to_end():
    frames = []
    # 30-frame synthetic clip: a polyp drifting + varying blur, plus flat frames
    for t in range(30):
        if t < 6 or t > 24:
            frames.append(np.full((160, 160, 3), 0.4, np.float32))  # no polyp
        else:
            cx = 60 + (t - 6) * 2
            img = _blob_image(cx=cx, cy=80, r=24)
            if t in (10, 18):  # inject blur on some frames
                img = gaussian_blur(img, 3.0)
                img = np.repeat(img[..., :1], 3, axis=2) if img.ndim == 2 else img
            frames.append(img)

    sel = InformativeFrameSelector(
        config=SelectorConfig(top_k=5, n_views=3, min_presence=0.05, temporal_min_distance=2)
    )
    result = sel.select(frames)
    assert len(result.selected) >= 1
    assert len(result.selected) <= 5
    # flat frames must be rejected (no candidate)
    assert any(i in result.rejected_indices for i in range(0, 6))
    # selected frames sorted by descending score
    totals = [fs.total for fs in result.selected]
    assert totals == sorted(totals, reverse=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
