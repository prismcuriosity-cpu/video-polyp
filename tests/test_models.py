"""Smoke + integration tests for the unified multi-task model."""

import pytest

torch = pytest.importorskip("torch")

from polypai.losses import DiceLoss, LabelSmoothingCE, VarifocalLoss  # noqa: E402
from polypai.models import ModelConfig, build_model  # noqa: E402


def _tiny_cfg(**kw):
    base = dict(
        stem_channels=12,
        stage_channels=(16, 24, 32, 48),
        stage_depths=(1, 1, 1, 1),
        fpn_channels=24,
        seg_decoder_channels=24,
    )
    base.update(kw)
    return ModelConfig(**base)


def test_forward_shapes():
    m = build_model(_tiny_cfg()).eval()
    x = torch.randn(2, 3, 96, 96)
    out = m(x)
    assert out["segmentation"]["seg_logits"].shape == (2, 2, 96, 96)
    assert out["segmentation"]["boundary_logits"].shape == (2, 1, 96, 96)
    assert len(out["detection"]["cls"]) == len(out["feature_levels"])
    assert out["classification"]["paris"].shape == (2, 6)
    assert out["classification"]["nice"].shape == (2, 3)
    assert out["classification"]["malignancy"].shape == (2, 2)


def test_gradient_checkpointing_runs():
    m = build_model(_tiny_cfg(gradient_checkpointing=True)).train()
    x = torch.randn(1, 3, 96, 96, requires_grad=True)
    out = m(x, tasks=["segmentation"])
    out["segmentation"]["seg_logits"].mean().backward()
    assert x.grad is not None


def test_multitask_backward_reaches_encoder():
    m = build_model(_tiny_cfg()).train()
    x = torch.randn(2, 3, 96, 96)
    out = m(x)

    seg_target = torch.zeros(2, 96, 96, dtype=torch.long)
    seg_target[:, 30:60, 30:60] = 1
    loss = DiceLoss()(out["segmentation"]["seg_logits"], seg_target)

    # detection: varifocal toward random IoU target scores per level
    vfl = VarifocalLoss()
    for cls_map in out["detection"]["cls"]:
        tgt = (torch.rand_like(cls_map) > 0.9).float() * torch.rand_like(cls_map)
        loss = loss + 0.1 * vfl(cls_map, tgt)

    # classification: CE with random labels
    ce = LabelSmoothingCE(0.1)
    for name, logits in out["classification"].items():
        labels = torch.randint(0, logits.shape[1], (logits.shape[0],))
        loss = loss + 0.1 * ce(logits, labels)

    loss.backward()
    stem_grad = m.encoder.stem[0][0].weight.grad
    assert stem_grad is not None and torch.isfinite(stem_grad).all()


def test_param_count_reasonable():
    m = build_model()  # default (full-size) config
    n = m.num_parameters()
    assert n > 1_000_000  # a real backbone, not a toy
