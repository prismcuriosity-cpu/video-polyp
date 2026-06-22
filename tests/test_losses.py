"""Tests for segmentation/detection/classification/multi-task losses.

Run on CPU torch; validate numeric sanity and that gradients flow.
"""

import pytest

torch = pytest.importorskip("torch")

from polypai.losses import (  # noqa: E402
    CIoULoss,
    ComboSegLoss,
    DiceLoss,
    FocalCELoss,
    FocalTverskyLoss,
    GradNormWeighter,
    HausdorffDTLoss,
    LabelSmoothingCE,
    SoftClDiceLoss,
    UncertaintyWeighting,
    VarifocalLoss,
    signed_distance_maps,
)
from polypai.losses.detection import bbox_ciou  # noqa: E402
from polypai.losses.segmentation import soft_skeleton  # noqa: E402


def _binary_case(perfect=True, n=2, h=24, w=24):
    target = torch.zeros(n, h, w, dtype=torch.long)
    target[:, 6:18, 6:18] = 1
    logits = torch.zeros(n, 2, h, w, requires_grad=True)
    with torch.no_grad():
        fill = 6.0 if perfect else -6.0
        logits[:, 1, 6:18, 6:18] = fill
        logits[:, 0, 6:18, 6:18] = -fill
        logits[:, 0] += 3.0
    logits.requires_grad_(True)
    return logits, target


def test_dice_loss_perfect_vs_wrong():
    lp, t = _binary_case(perfect=True)
    lw, _ = _binary_case(perfect=False)
    assert DiceLoss()(lp, t).item() < 0.1
    assert DiceLoss()(lw, t).item() > 0.7


def test_dice_grad_flows():
    lp, t = _binary_case(perfect=False)
    loss = DiceLoss()(lp, t)
    loss.backward()
    assert lp.grad is not None and torch.isfinite(lp.grad).all()


def test_focal_tversky_range_and_grad():
    lp, t = _binary_case(perfect=False)
    loss = FocalTverskyLoss(alpha=0.3, beta=0.7)(lp, t)
    assert 0.0 <= loss.item()
    loss.backward()
    assert lp.grad is not None


def test_signed_distance_maps_shape_and_sign():
    _, t = _binary_case()
    dm = signed_distance_maps(t, num_classes=2)
    assert dm.shape == (2, 2, 24, 24)
    # inside foreground (class 1) the signed distance should be <= 0
    assert dm[0, 1, 12, 12].item() <= 0.0


def test_boundary_via_combo_runs_and_backward():
    lp, t = _binary_case(perfect=False)
    combo = ComboSegLoss(w_boundary=0.5, w_hausdorff=0.5)
    combo.train()
    parts = combo(lp, t)
    assert "total" in parts and "boundary" in parts
    parts["total"].backward()
    assert lp.grad is not None and torch.isfinite(lp.grad).all()


def test_boundary_warmup_ramps():
    combo = ComboSegLoss(w_boundary=1.0, boundary_warmup=10)
    combo.train()
    lp, t = _binary_case(perfect=False)
    first = combo(lp, t)["boundary"].abs().item()
    for _ in range(20):
        lp2, t2 = _binary_case(perfect=False)
        later = combo(lp2, t2)["boundary"].abs().item()
    # after warmup the boundary weight is fully ramped (>= early step)
    assert later >= first - 1e-6


def test_hausdorff_dt_runs():
    lp, t = _binary_case(perfect=False)
    loss = HausdorffDTLoss()(lp, t)
    assert torch.isfinite(loss)
    loss.backward()
    assert lp.grad is not None


def test_soft_skeleton_and_cldice():
    x = torch.zeros(1, 1, 32, 32)
    x[:, :, 14:18, 4:28] = 1.0  # a horizontal bar -> non-trivial skeleton
    sk = soft_skeleton(x, iters=5)
    assert sk.shape == x.shape and sk.max() <= 1.0 + 1e-5
    lp, t = _binary_case(perfect=False)
    loss = SoftClDiceLoss(iters=5)(lp, t)
    assert torch.isfinite(loss)
    loss.backward()
    assert lp.grad is not None


def test_bbox_ciou_perfect_is_one():
    box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    assert pytest.approx(1.0, abs=1e-4) == bbox_ciou(box, box).item()


def test_ciou_loss_decreases_with_overlap():
    pred_far = torch.tensor([[0.0, 0.0, 5.0, 5.0]])
    pred_near = torch.tensor([[1.0, 1.0, 11.0, 11.0]])
    target = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    assert CIoULoss()(pred_near, target) < CIoULoss()(pred_far, target)


def test_varifocal_runs_and_grad():
    logits = torch.randn(4, 1, requires_grad=True)
    target = torch.tensor([[0.8], [0.0], [0.0], [0.6]])
    loss = VarifocalLoss()(logits, target)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad is not None


def test_classification_losses():
    logits = torch.randn(8, 3, requires_grad=True)
    target = torch.randint(0, 3, (8,))
    for crit in (LabelSmoothingCE(0.1), FocalCELoss(gamma=2.0)):
        loss = crit(logits, target)
        assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_uncertainty_weighting_learns():
    uw = UncertaintyWeighting(["det", "seg", "cls"])
    losses = {"det": torch.tensor(1.0, requires_grad=True),
              "seg": torch.tensor(2.0, requires_grad=True),
              "cls": torch.tensor(0.5, requires_grad=True)}
    total, weights = uw(losses)
    assert set(weights) == {"det", "seg", "cls"}
    total.backward()
    assert uw.log_vars.grad is not None


def test_gradnorm_weighter():
    shared = torch.nn.Linear(4, 4)
    x = torch.randn(3, 4)
    feat = shared(x)
    gn = GradNormWeighter(["a", "b"])
    losses = {"a": feat.pow(2).mean(), "b": feat.abs().mean()}
    ws = gn.weighted_sum(losses)
    assert torch.isfinite(ws)
    gloss = gn.gradnorm_loss(losses, shared.weight)
    assert torch.isfinite(gloss)
