# PolypAI — Plan for the Betterment

This is the engineering + research roadmap: what exists today, the identified
bottlenecks, the per-module improvement plans, the concrete performance targets,
and the milestone sequence from the current reference implementation to
publishable / deployable results.

Legend: ✅ implemented & tested · 🟡 implemented, needs training/data · ⬜ planned.

---

## 0. Current state (this repository)

| Capability | State |
|------------|-------|
| Polyp-centric frame selection (full pipeline) | ✅ |
| Segmentation/detection/classification/multi-task losses | ✅ |
| Metrics (Dice/IoU/HD95/ASSD, AP/mAP/FROC, AUC/F1, clinical) | ✅ |
| Pseudo-NBI synthesis, SR hook | ✅ |
| Unified 49.8 M-param multi-task model (forward/backward) | ✅ |
| Modular dataset loaders + dataloaders | ✅ |
| FCOS target assignment + unified criterion | ✅ |
| AMP/EMA/cosine trainer, evaluator | 🟡 (needs data + GPU) |
| Grad-CAM/++, uncertainty, calibration | ✅ |
| Clinical inference pipeline + report | ✅ (selection-only ✅; full needs weights) |
| ONNX export (passes checker) / TensorRT recipe | ✅ / 🟡 |
| **Trained weights, benchmark numbers** | ⬜ |

79 CPU tests pass. The gap to a paper/product is **training data, GPU training,
and clinical validation**, not missing architecture.

---

## 1. Identified bottlenecks (and mitigations)

| # | Bottleneck | Impact | Mitigation in design / planned |
|---|------------|--------|--------------------------------|
| B1 | **2 mm lesion signal** is tiny vs stride-32 features | recall ceiling | P2 level + BiFPN P2 weighting + SR pre-pass + tiling (⬜ tiling) |
| B2 | **Class imbalance** (NICE-3 / malignant rare) | poor minority AUC | Focal/Varifocal, class-balanced sampling (⬜ sampler), Focal-Tversky recall bias |
| B3 | **Domain shift** white-light↔NBI, scope vendors | generalisation | pseudo-NBI augmentation ✅; multi-centre training + ⬜ domain adaptation |
| B4 | **Label scarcity** for Paris/NICE/Kudo | weak heads | heterogeneous-supervision masking ✅; ⬜ semi-/self-supervised pretrain |
| B5 | **Temporal redundancy** inflates "miss-rate" illusions | eval validity | per-lesion tracking (⬜) + dedup ✅ |
| B6 | **Memory** at 49.8 M params, 512 px, single 32 GB GPU | batch size | grad checkpointing ✅, BF16 ✅, channels-last ✅, dynamic batch (⬜) |
| B7 | **Latency** <20 ms target | real-time | TensorRT FP16 ✅ recipe, ⬜ distilled "fast" backbone, frame-skip via selector |
| B8 | **Boundary/topology** of flat lesions | Dice plateau | boundary + HD + soft-clDice losses ✅; ⬜ test-time refinement |

---

## 2. Per-module improvement plans

### Module 1 — Frame selection ✅→⬜
- ⬜ Replace the classical saliency proposer with the trained Module-2 detector
  in the loop (already pluggable via `from_torch_detector`).
- ⬜ Learn the diagnostic-score weights (α…ζ) from "did this frame improve
  downstream Dice / characterisation?" rather than fixing them.
- ⬜ Add a small learned **diagnostic-quality head** (regress Dice-attainable / 
  characterisation-confidence) and blend with the classical score.
- ⬜ Per-lesion tracking (re-ID embeddings) for true multi-view grouping.

### Module 2 — Small-polyp detection 🟡→⬜
- ⬜ Image **tiling / sliding-window** at inference for native-resolution recall.
- ⬜ Train the SR pre-pass (distilled SwinIR/ESRGAN for endoscopy).
- ⬜ Copy-paste / small-object augmentation; anchor-free + query-based (DINO) ablation.
- **Targets:** Recall ≥ 0.95 @ ≤ 4 FPPI, mAP ≥ 0.90, latency < 20 ms (TensorRT).

### Module 3 — Segmentation 🟡→⬜
- ⬜ Tune the Combo-loss schedule; add **uncertainty-gated** test-time augmentation.
- ⬜ Temporal segmentation consistency (propagate masks across selected frames).
- **Targets:** Dice ≥ 0.92 (Kvasir-SEG), HD95 ↓, robust under low light/occlusion.

### Modules 4–6 — Characterisation 🟡→⬜
- ⬜ ROI-pooled (not frame-global) features for each lesion via RoIAlign on P2/P3.
- ⬜ Kudo/NICE pretraining on pseudo-NBI; frequency-domain / wavelet texture branch.
- ⬜ Ordinal loss for Paris (morphology is ordered), label-distribution learning.
- **Targets:** NICE AUC ≥ 0.90, malignancy sensitivity ≥ 0.95 (safety-critical).

### Module 9 — Explainability ✅→⬜
- ⬜ Concept-level attribution (which pit/vessel pattern), not just spatial CAM.
- ⬜ Prospective calibration set per deployment site; report reliability diagrams.

### Unified model / training 🟡→⬜
- ⬜ Knowledge distillation into a real-time "fast" variant (B7).
- ⬜ Continual learning across centres without catastrophic forgetting.
- ⬜ Dynamic batch sizing + streaming dataset pipeline (B6).

---

## 3. Milestones

| M | Goal | Exit criteria |
|---|------|---------------|
| **M0** | Reference implementation (this repo) | 79 tests green; full forward/backward; ONNX export |
| **M1** | First end-to-end training run | Kvasir-SEG Dice ≥ 0.88 on val; trainer stable in BF16 on RTX 5090 |
| **M2** | Detection + multi-dataset | PICCOLO+CVC added; FROC reported; recall ≥ 0.90 |
| **M3** | Characterisation heads | NICE/Paris trained on PICCOLO; AUC/F1 reported with calibration |
| **M4** | Real-time deployment | TensorRT engine, < 20 ms/ frame, FPS + memory benchmarked |
| **M5** | Clinical metrics + ablations | ADR / miss-rate / malignancy-sensitivity; per-novelty ablation table |
| **M6** | Paper + external validation | multi-centre test set, reader study, manuscript |

---

## 4. Experimental protocol for the paper

* **Splits:** official where available (PICCOLO); patient-level (never frame-level)
  splits elsewhere to avoid leakage; the registry's deterministic hash split is
  per-filename and must be upgraded to **per-patient** for the paper (⬜).
* **Cross-dataset generalisation:** train on Kvasir-SEG+PICCOLO, test on
  CVC-ColonDB / HyperKvasir unseen.
* **Ablations (one per novelty claim):** frame-selection on/off vs downstream
  yield; P2/BiFPN; loss bank; cross-task attention; uncertainty weighting;
  pseudo-NBI for NICE.
* **Statistics:** bootstrap CIs, DeLong test for AUC, per-lesion (not per-frame)
  aggregation.

---

## 5. Engineering hardening (⬜)

- CI matrix already runs lint + 79 CPU tests; add a GPU smoke job and an ONNX
  parity test (torch vs onnxruntime) when a runner is available.
- Experiment tracking (TensorBoard wired; add Weights & Biases optional).
- Deterministic seeding + config hashing for reproducibility.
- Model card + dataset datasheets for clinical governance.
