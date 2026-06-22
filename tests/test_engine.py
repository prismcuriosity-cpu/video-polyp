"""Integration tests for the training engine (FCOS targets + criterion + trainer)."""

import pytest

torch = pytest.importorskip("torch")

from polypai.engine import Evaluator, TrainConfig, Trainer, UnifiedCriterion  # noqa: E402
from polypai.engine.fcos_target import assign_image  # noqa: E402
from polypai.models import ModelConfig, build_model  # noqa: E402

IGNORE = -100


def _tiny_model():
    cfg = ModelConfig(stem_channels=12, stage_channels=(16, 24, 32, 48),
                      stage_depths=(1, 1, 1, 1), fpn_channels=24, seg_decoder_channels=24)
    return build_model(cfg)


def _synthetic_batch(b=2, h=64, w=64):
    img = torch.randn(b, 3, h, w)
    mask = torch.zeros(b, h, w, dtype=torch.long)
    mask[:, 20:44, 20:44] = 1
    boxes = [torch.tensor([[20.0, 20.0, 44.0, 44.0]]) for _ in range(b)]
    labels = {
        "paris": torch.tensor([0, IGNORE]),
        "nice": torch.tensor([IGNORE, 1]),
        "kudo": torch.tensor([IGNORE, IGNORE]),
        "malignancy": torch.tensor([0, 1]),
    }
    return {"image": img, "mask": mask, "boxes": boxes,
            "box_labels": [torch.zeros(1, dtype=torch.long) for _ in range(b)],
            "labels": labels, "dataset": ["syn"] * b, "image_path": ["x"] * b}


def test_fcos_assignment_finds_positives():
    boxes = torch.tensor([[20.0, 20.0, 44.0, 44.0]])
    shapes = [(16, 16), (8, 8), (4, 4), (2, 2)]
    tgt = assign_image(boxes, shapes)
    assert tgt["pos_mask"].sum() > 0
    assert tgt["centerness"][tgt["pos_mask"]].max() <= 1.0 + 1e-5


def test_fcos_assignment_empty_boxes():
    tgt = assign_image(torch.zeros((0, 4)), [(16, 16), (8, 8), (4, 4), (2, 2)])
    assert tgt["pos_mask"].sum() == 0


def test_criterion_forward_backward():
    model = _tiny_model().train()
    crit = UnifiedCriterion()
    batch = _synthetic_batch()
    out = model(batch["image"])
    total, log = crit(out, batch)
    assert torch.isfinite(total)
    assert {"detection", "segmentation", "classification"} <= set(log)
    total.backward()
    g = model.encoder.stem[0][0].weight.grad
    assert g is not None and torch.isfinite(g).all()


def test_label_masking_skips_missing():
    """A task where every label is IGNORE must contribute zero (no NaN)."""
    model = _tiny_model().train()
    crit = UnifiedCriterion()
    batch = _synthetic_batch()
    batch["labels"]["kudo"] = torch.tensor([IGNORE, IGNORE])
    out = model(batch["image"])
    total, _ = crit(out, batch)
    assert torch.isfinite(total)


def test_trainer_one_epoch_cpu():
    model = _tiny_model()
    crit = UnifiedCriterion()
    loader = [_synthetic_batch(), _synthetic_batch()]
    cfg = TrainConfig(device="cpu", amp_dtype="off", channels_last=False, log_interval=100, ckpt_dir="/tmp/polypai_ckpt")
    trainer = Trainer(model, crit, {"train": loader}, cfg)
    stats = trainer.train_one_epoch(total_steps=2)
    assert "total" in stats and stats["total"] == stats["total"]  # not NaN


def test_evaluator_returns_dice():
    model = _tiny_model()
    loader = [_synthetic_batch()]
    metrics = Evaluator(model, loader, torch.device("cpu")).run()
    assert "dice" in metrics and 0.0 <= metrics["dice"] <= 1.0
