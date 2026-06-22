# PolypAI — Architecture & Research Design

This document specifies the system architecture, the per-module design, the
mathematical formulations, and the research-novelty claims. It is written to the
standard expected by MICCAI / Medical Image Analysis reviewers.

---

## 1. System overview

PolypAI is a three-stage pipeline:

1. **Polyp-Centric Informative Frame Selection (PC-IFS)** — reduces a raw video
   (10⁴–10⁵ frames) to a small set of *diagnostically informative* frames.
2. **Unified Multi-Task Model** — a single shared-backbone network that jointly
   performs detection, segmentation, and characterisation on selected frames.
3. **Explainable Clinical Decision Support** — turns raw predictions into a
   calibrated, interpretable per-lesion report.

The design goal is **medical-relevance-aware** processing end to end: every
stage is conditioned on the polyp ROI rather than the whole frame.

---

## 2. Module 1 — Polyp-Centric Informative Frame Selection

`polypai.frame_selection`

### 2.1 Pipeline

```
frames → candidate detection → reject non-polyp → ROI diagnostic scoring
       → temporal peak selection + NMS → near-duplicate removal
       → multi-view selection → top-K informative frames
```

### 2.2 Diagnostic-information score

Each surviving frame receives a medical-relevance score (brief §7), with every
term computed **inside the lesion ROI**:

$$ S = \alpha P + \beta V + \gamma T + \delta Q + \epsilon B + \zeta \cdot \text{Vasc} $$

| Term | Meaning | Implementation |
|------|---------|----------------|
| `P` | polyp-presence confidence | candidate detector score |
| `V` | visibility | area adequacy · centeredness · border cut-off · occlusion (`visibility.py`) |
| `T` | texture / pit clarity | high-freq energy ratio + gradient density + LBP entropy (`texture.py`) |
| `Q` | ROI image quality | variance-of-Laplacian sharpness, RMS contrast, specular-free, exposure, Immerkaer noise (`quality.py`) |
| `B` | boundary completeness | perimeter-inside-frame + contour edge crispness (`visibility.py`) |
| `Vasc` | vascular visibility | multi-scale Frangi vesselness response (`vascular.py`) |

Weights are configurable (`DiagnosticScoreWeights`) and normalised to sum to 1.

### 2.3 Candidate detection

A `CandidateDetector` protocol abstracts the front-end:

* **`SaliencyCandidateDetector`** (default, training-free) — Difference-of-Gaussian
  blob saliency + mucosal-redness channel. A frame is accepted only if it
  contains a *spatially coherent* salient region (connected-component area ≥
  `min_area_frac` and robust median/MAD significance ≥ `min_comp_sig`), which
  separates true lesions from sensor noise far better than a single-pixel peak
  z-score on realistic, textured frames.
* **Trained detector** — the Module-2 head is injected via
  `from_torch_detector(...)` with zero changes to the orchestration.

### 2.4 Temporal, dedup, multi-view

* **Temporal (`temporal.py`)** — Gaussian score smoothing, local-maxima *peak
  diagnostic moments*, and greedy temporal NMS to drop redundant adjacent frames.
* **Near-duplicate removal (`dedup.py`)** — 64-bit DCT perceptual hash (Hamming
  radius) + optional embedding cosine similarity, keeping the highest-scoring
  representative per cluster.
* **Multi-view (`multiview.py`)** — farthest-point sampling in a view-descriptor
  space (geometry + appearance) keeps frontal/side/close-up/boundary views
  instead of K near-identical "best" frames.

### 2.5 Novelty

ROI-localised, **polyp-centric** informativeness (vs whole-frame BRISQUE/NIQE
key-framing), unifying quality + texture + vasculature + boundary into a single
diagnostic-yield score, with explicit multi-view diversity for downstream
size/Paris/histology estimation.

---

## 3. Module 2 — Ultra-Small Polyp Detection

`polypai.models` (detection head) + `polypai.engine.fcos_target`

* **Anchor-free FCOS head** with per-level centre sampling. A high-resolution
  **P2 (stride 4)** level is included so a ~2 mm lesion still spans several
  feature cells; P2 also takes the smallest regression range so tiny lesions are
  matched there.
* **Multi-scale fusion** via a weighted BiFPN-style neck (`fpn.py`) that *learns*
  to emphasise P2 — the "adaptive feature pyramid".
* **Fine-grained lesion attention** via CBAM spatial attention in every backbone
  block (`blocks.py`).
* **Super-resolution hook** (`enhance/superres.py`) for an SR pre-pass on
  diminutive lesions (deterministic Lanczos fallback until SR weights exist).
* **Losses** — IoU-aware **Varifocal** classification + **CIoU** box regression +
  centerness BCE (`losses/detection.py`).

Targets: ≥95 % recall, ≥90 % mAP, <20 ms latency (see ROADMAP for the plan).

---

## 4. Module 3 — Boundary-Aware Segmentation

`polypai.models.heads.BoundaryAwareSegHead` + `polypai.losses.segmentation`

* Semantic-FPN decoder fusing all pyramid levels at P2 resolution, with an
  **auxiliary boundary branch** whose features are concatenated back into the
  mask predictor (boundary-refinement learning).
* **Uncertainty head** predicts a per-pixel log-variance map (aleatoric).
* **Loss bank** — Dice, Focal-Tversky (recall-biased for diminutive polyps),
  Kervadec **boundary loss** (signed-distance integral with warmup scheduling),
  Karimi **Hausdorff-DT** surrogate, and Shit **soft-clDice** topology loss.
  `ComboSegLoss` combines them with a ramped boundary weight.

Metrics: Dice, IoU, HD95, ASSD (`metrics/segmentation.py`).

---

## 5. Modules 4–6 — Characterisation (Pit / Paris / NICE)

`polypai.models.heads.ClassificationHeads` + `polypai.enhance.nbi`

* Per-task heads (`paris`:6, `nice`:3, `kudo`:7, `malignancy`:2) read **both**
  the deepest (semantic, P5) and shallowest (textural, P2) global descriptors, so
  shape-driven Paris and texture/vessel-driven NICE/Kudo each get the right cues.
* **Cross-task attention** (`attention.py`) lets each head gate the shared
  features through its own learned query.
* **Pseudo-NBI synthesis** (`enhance/nbi.py`) renders white-light frames into an
  NBI-like image (spectral remap → vesselness-guided darkening → CLAHE),
  providing the vascular contrast NICE relies on and a white-light↔NBI
  augmentation.

---

## 6. Module 9 — Explainable AI

`polypai.explain`

* **Grad-CAM / Grad-CAM++** with a generic `score_fn` so the same machinery
  explains any head ("which region drove *malignant*?").
* **Uncertainty** — aleatoric log-variance map, predictive entropy, MC-Dropout
  epistemic uncertainty.
* **Calibration** — temperature scaling + Expected Calibration Error, so reported
  probabilities are trustworthy at deployment.

These feed the per-lesion `explanation` field of the `ProcedureReport`.

---

## 7. Unified multi-task architecture

```
image → PolypEncoder (stem + 4 residual+CBAM stages; P2..P5)
      → FPN (weighted top-down fusion, P2..P5)
      ├── FCOSDetHead              → cls / centerness / box per level
      ├── BoundaryAwareSegHead     → seg logits + boundary logits + seg feat
      │     └── UncertaintyHead    → log-variance map
      └── CrossTaskAttention → ClassificationHeads → paris/nice/kudo/malignancy
```

Default config ≈ **49.8 M parameters**.

### 7.1 Multi-task loss balancing

`polypai.losses.multitask`, applied in `polypai.engine.criterion`:

* **Homoscedastic uncertainty weighting** (Kendall et al.) — learns a
  log-variance per task group (`detection`, `segmentation`, `classification`);
  zero tuning.
* **GradNorm** (Chen et al.) — alternative that equalises per-task gradient norms
  on the shared backbone.

### 7.2 Heterogeneous supervision

The unified criterion masks missing targets per task (`IGNORE_INDEX = -100`), so
a single run can mix **Kvasir-SEG** (masks only), **PICCOLO** (masks + Paris/NICE),
and detection boxes derived from masks — see [`DATASETS.md`](DATASETS.md).

---

## 8. Research-novelty summary

1. **Polyp-centric informative-frame selection** with a unified ROI diagnostic-
   yield score and multi-view diversity (Module 1).
2. **Single unified network** spanning detection → segmentation → Paris/NICE/Kudo
   → malignancy with cross-task attention and uncertainty-based balancing.
3. **Topology- and boundary-aware segmentation** (soft-clDice + boundary + HD)
   tuned for thin/flat lesion boundaries.
4. **Pseudo-NBI synthesis** as a reproducible vascular-contrast front-end that
   bridges the white-light/NBI domain gap for NICE.
5. **Uncertainty + calibration** integrated into the clinical report, not bolted
   on — a prerequisite for deployment.

See [`ROADMAP.md`](ROADMAP.md) for how each is taken to publishable evidence.

---

## 9. Evaluation protocol

| Group | Metrics |
|-------|---------|
| Detection | Recall, Precision, AP/mAP@[.5:.95], FROC |
| Segmentation | Dice, IoU, HD95, ASSD |
| Classification | Accuracy, AUC (macro OvR), F1, Sensitivity, Specificity |
| Clinical | Adenoma Detection Rate, per-polyp Miss Rate, Malignancy accuracy/sensitivity |
| Real-time | FPS, GPU memory, latency (per stage) |

All implemented in `polypai.metrics`.
