# PolypAI — End-to-End Colonoscopy Polyp Analysis

A research-grade, modular pipeline for **automated colorectal polyp detection,
segmentation, characterisation, and explainable diagnosis** from colonoscopy
video — targeting top-tier medical-imaging venues and real-time clinical-grade
deployment on a single **RTX 5090** workstation.

> **Status:** reference architecture + working core. The deterministic clinical
> components (polyp-centric frame selection, losses, metrics, pseudo-NBI) are
> fully implemented and unit-tested; the deep-learning stack (49.8 M-param
> unified model, multi-task trainer, ONNX/TensorRT export, Grad-CAM) is
> implemented and smoke-tested on CPU and is ready to train once datasets are
> mounted. See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the path from here to
> publication-/deployment-grade results.

```
Raw colonoscopy video
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ 1. Polyp-Centric Informative Frame Selection  (polypai.frame_selection)
│    candidate detection → ROI diagnostic scoring → temporal peak
│    selection → near-duplicate removal → multi-view selection
└───────────────────────────────────────────────────────────────┘
        │  selected informative frames
        ▼
┌───────────────────────────────────────────────────────────────┐
│ 2. Unified Multi-Task Model            (polypai.models)
│    shared encoder → BiFPN neck (+P2) →
│      • detection head (anchor-free, 2 mm-sensitive)
│      • boundary-aware segmentation head
│      • characterisation heads: Paris / NICE / Kudo / malignancy
│      • uncertainty head
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ 3. Explainable Clinical Decision Support   (polypai.explain / inference)
│    Grad-CAM, uncertainty, calibration → structured ProcedureReport
└───────────────────────────────────────────────────────────────┘
```

## Why this design

* **Polyp-centric, not generic.** Frame quality is scored *inside the lesion
  ROI*, so a globally blurry frame with a crisp, well-exposed polyp beats a
  globally sharp frame whose lesion is washed out by glare — the clinically
  correct preference.
* **One shared backbone, many heads.** Detection, segmentation and
  characterisation are jointly trained with homoscedastic-uncertainty / GradNorm
  task balancing and cross-task attention — cheaper, and each task regularises
  the others.
* **Boundary-, shape- and topology-aware segmentation.** Dice + Focal-Tversky +
  Kervadec boundary + Karimi Hausdorff + soft-clDice, scheduled for stable
  training.
* **Designed to be exported.** Tensor-tuple ONNX wrapper + TensorRT recipe for
  real-time FP16/BF16 inference.

## Install

```bash
python -m pip install -e ".[all]"     # full stack (torch, cv, train, export, dev)
# or a lean install for the deterministic core only:
python -m pip install -e ".[cv]"      # numpy + opencv + scipy/scikit-image
```

PyTorch should match your CUDA/Blackwell toolchain; see
[`docs/HARDWARE_OPTIMIZATION.md`](docs/HARDWARE_OPTIMIZATION.md).

## Quickstart

### Informative frame selection (works today, no trained weights)

```bash
polypai-select-frames --video case.mp4 --out out/ --top-k 12 --stride 2
```

Writes the top-K most diagnostically informative frames plus `selection.json`
with the full per-frame score breakdown (P, V, T, Q, B, vascular, S).

### Train the unified model

```bash
# point configs/experiment_rtx5090.yaml at your dataset roots first
polypai-train --config configs/experiment_rtx5090.yaml
```

### Evaluate / infer / export

```bash
polypai-evaluate --config configs/experiment_rtx5090.yaml --checkpoint checkpoints/rtx5090/best.pt
polypai-infer    --video case.mp4 --config configs/experiment_rtx5090.yaml --checkpoint best.pt --out report/
polypai-export   --config configs/experiment_rtx5090.yaml --checkpoint best.pt --out polyp.onnx
```

`polypai-infer` produces a Markdown + JSON `ProcedureReport` with per-lesion
Paris/NICE/Kudo, estimated size, malignancy probability, uncertainty, and a
rationale.

## Repository layout

| Path | Contents |
|------|----------|
| `src/polypai/frame_selection/` | Polyp-centric informative frame selection (Module 1) |
| `src/polypai/models/` | Shared encoder, BiFPN neck, task heads, unified model |
| `src/polypai/losses/` | Segmentation / detection / classification / multi-task losses |
| `src/polypai/metrics/` | Dice/IoU/HD95/ASSD, AP/mAP/FROC, AUC/F1, clinical endpoints |
| `src/polypai/enhance/` | Pseudo-NBI synthesis, super-resolution hook |
| `src/polypai/data/` | Modular dataset registry, transforms, video reader, dataloaders |
| `src/polypai/engine/` | FCOS targets, unified criterion, AMP trainer, evaluator |
| `src/polypai/explain/` | Grad-CAM/++, uncertainty, calibration |
| `src/polypai/inference/` | Detection post-processing, end-to-end clinical pipeline |
| `src/polypai/export/` | ONNX + TensorRT export |
| `src/polypai/cli/` | `polypai-*` command-line entry points |
| `configs/` | YAML experiment configurations |
| `docs/` | Architecture, roadmap, datasets, hardware |
| `tests/` | 79 unit/integration tests (CPU) |

## Tests

```bash
pytest -q          # 79 tests, CPU-only, no datasets required
ruff check src tests
```

## Documentation

* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — full system design, the unified
  multi-task architecture, the diagnostic-score formulation, and the research
  novelty claims.
* [`docs/ROADMAP.md`](docs/ROADMAP.md) — **the plan for betterment**: per-module
  improvement plans, identified bottlenecks, performance targets, and milestones.
* [`docs/DATASETS.md`](docs/DATASETS.md) — how to mount Kvasir-SEG / PICCOLO /
  CVC / HyperKvasir / GastroVision and add new sources.
* [`docs/HARDWARE_OPTIMIZATION.md`](docs/HARDWARE_OPTIMIZATION.md) — RTX 5090 /
  32 GB RAM / 1 TB SSD tuning.

## Clinical scope & disclaimer

PolypAI is a **research** decision-support system. It is **not** a medical device
and must not be used for diagnosis or treatment without appropriate regulatory
clearance and clinical validation.

## License

Apache-2.0.
