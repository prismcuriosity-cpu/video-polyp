"""Tests for pseudo-NBI synthesis and the super-resolution fallback."""

import numpy as np

from polypai.enhance import lanczos_upsample, pseudo_nbi, vessel_density

RNG = np.random.default_rng(1)


def _low_contrast_mucosa(h=80, w=80):
    """Faint, low-contrast vasculature on reddish mucosa (the case NBI targets)."""
    img = np.full((h, w, 3), 0.0, np.float32)
    img[..., 0], img[..., 1], img[..., 2] = 0.62, 0.55, 0.55
    for c in (18, 38, 58):
        img[:, c - 1 : c + 1, :] -= 0.07  # barely-visible vessels
    return np.clip(img, 0, 1)


def test_pseudo_nbi_shape_and_range():
    out = pseudo_nbi(_low_contrast_mucosa())
    assert out.shape == (80, 80, 3)
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_pseudo_nbi_amplifies_low_contrast():
    img = _low_contrast_mucosa()
    nbi = pseudo_nbi(img, vessel_gain=0.6)
    lum_in = img.mean(axis=2)
    lum_out = nbi.mean(axis=2)
    # CLAHE + spectral remap should raise the RMS contrast of low-contrast input
    assert lum_out.std() > lum_in.std()


def test_vessel_density_in_unit_range():
    d = vessel_density(_low_contrast_mucosa())
    assert 0.0 <= d <= 1.0


def test_lanczos_upsample_scale():
    img = RNG.random((20, 30, 3)).astype(np.float32)
    up = lanczos_upsample(img, scale=2)
    assert up.shape[0] == 40 and up.shape[1] == 60
