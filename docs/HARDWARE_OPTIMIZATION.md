# Hardware Optimisation — RTX 5090 Workstation

Target deployment: **1× NVIDIA RTX 5090, 32 GB system RAM (assume ~30 % reserved
by background processes ⇒ ~22 GB usable), 1 TB SSD.** Goal: train the unified
49.8 M-param model and serve it at real-time (<20 ms/frame) clinical rates.

## Memory budget

| Consumer | Notes |
|---|---|
| Model params (49.8 M) | ~0.1 GB (BF16) / 0.2 GB (FP32 master) |
| Optimiser state (AdamW) | 2× params in FP32 ≈ 0.4 GB |
| Activations @ 512 px, bs=12 | dominant term — controlled by checkpointing |
| **System RAM for dataloading** | budget ≈ 22 GB; keep workers × prefetch modest |

The default config (`configs/experiment_rtx5090.yaml`) uses **bs=12 @ 512 px with
gradient checkpointing + BF16**, leaving headroom; raise `grad_accum` to grow the
*effective* batch without more VRAM.

## Enabled optimisations

| Feature | Where | Why |
|---|---|---|
| **BF16 autocast** | `Trainer` (`amp_dtype: bf16`) | Blackwell has fast BF16; wide dynamic range ⇒ no grad scaler needed |
| **FP16 + GradScaler** | `Trainer` (`amp_dtype: fp16`) | alternative for ops without BF16 kernels |
| **Gradient checkpointing** | `PolypEncoder` (`gradient_checkpointing: true`) | trade ~30 % compute for large activation-memory savings (B6) |
| **channels-last** | `Trainer` (`channels_last: true`) | better tensor-core utilisation for conv |
| **torch.compile** | `Trainer` (`compile: true`) | kernel fusion on Ada/Blackwell |
| **EMA weights** | `Trainer.EMA` | smoother, better-calibrated eval model |
| **Cosine + warmup LR** | `Trainer._lr_at` | stable large-batch training |
| **Memory-efficient dataloaders** | `DataConfig` | `pin_memory`, `persistent_workers`, `prefetch_factor` |
| **Streaming video reader** | `data/video.py` | never loads a full procedure into RAM |

### Flash / efficient attention

The CBAM/cross-task attention modules are channel/spatial (cheap). If a
transformer block is added (e.g. a ViT bottleneck or query-based detector),
route it through `torch.nn.functional.scaled_dot_product_attention`, which
dispatches to FlashAttention-2 kernels on the 5090 automatically — no code change
beyond using SDPA.

## Inference & export

1. **ONNX** — `polypai-export --config ... --checkpoint ... --out polyp.onnx`
   (stable exporter, dynamic batch axis, passes `onnx.checker`).
2. **TensorRT** — build an FP16 engine:

   ```bash
   trtexec --onnx=polyp.onnx --saveEngine=polyp.engine --fp16 \
           --memPoolSize=workspace:4096 \
           --minShapes=image:1x3x512x512 \
           --optShapes=image:1x3x512x512 \
           --maxShapes=image:4x3x512x512
   ```

   (`polypai.export.trtexec_command` prints this for your config; or pass
   `--build-engine` to build via the Python bindings when available.)

### Real-time strategy

* The **frame selector** is the first latency lever: only informative frames hit
  the heavy model, so per-procedure compute drops by 1–2 orders of magnitude.
* For live overlay, run a **distilled "fast" backbone** (ROADMAP B7) for
  detection/segmentation at video rate, and the full model on selected frames for
  characterisation.
* Benchmark FPS / latency / GPU memory with `polypai.metrics` real-time hooks and
  report per-stage (selection, detection, seg, characterisation).

## Reproducibility knobs

Set seeds and deterministic algorithms for benchmark runs:

```python
import torch, numpy as np, random
torch.manual_seed(0); np.random.seed(0); random.seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)
```

(Trade-off: some cuDNN kernels fall back to slower deterministic variants.)
