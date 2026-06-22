"""Mixed-precision trainer tuned for a single RTX 5090.

Implements the optimisation requirements from the brief: AMP (BF16/FP16),
gradient accumulation + clipping, channel-last memory format, optional
``torch.compile``, cosine schedule with warmup, and an EMA shadow model.
Gradient checkpointing is enabled on the backbone via :class:`ModelConfig`.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    grad_accum: int = 1
    max_grad_norm: float = 5.0
    amp_dtype: str = "bf16"          # "bf16" | "fp16" | "off"
    ema_decay: float = 0.999
    channels_last: bool = True
    compile: bool = False
    device: str = "cuda"
    ckpt_dir: str = "checkpoints"
    log_interval: int = 20


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


class Trainer:
    def __init__(self, model, criterion, loaders: dict[str, DataLoader], cfg: TrainConfig | None = None):
        self.cfg = cfg or TrainConfig()
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() or self.cfg.device == "cpu" else "cpu")
        self.model = model.to(self.device)
        if self.cfg.channels_last and self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)
        if self.cfg.compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        self.criterion = criterion.to(self.device)
        self.loaders = loaders

        params = list(self.model.parameters()) + list(self.criterion.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.ema = EMA(self.model, self.cfg.ema_decay)

        self._amp_enabled = self.cfg.amp_dtype != "off" and self.device.type == "cuda"
        self._amp_dtype = torch.bfloat16 if self.cfg.amp_dtype == "bf16" else torch.float16
        # GradScaler is only needed for fp16; bf16 has the dynamic range to skip it.
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self._amp_enabled and self.cfg.amp_dtype == "fp16"
        )
        self.global_step = 0
        Path(self.cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)

    def _lr_at(self, step: int, total: int) -> float:
        if step < self.cfg.warmup_steps:
            return self.cfg.lr * step / max(1, self.cfg.warmup_steps)
        prog = (step - self.cfg.warmup_steps) / max(1, total - self.cfg.warmup_steps)
        return 0.5 * self.cfg.lr * (1 + math.cos(math.pi * min(1.0, prog)))

    def train_one_epoch(self, total_steps: int) -> dict:
        self.model.train()
        running = {}
        loader = self.loaders["train"]
        self.optimizer.zero_grad(set_to_none=True)
        for it, batch in enumerate(loader):
            lr = self._lr_at(self.global_step, total_steps)
            for g in self.optimizer.param_groups:
                g["lr"] = lr
            images = batch["image"].to(self.device, non_blocking=True)
            if self.cfg.channels_last and self.device.type == "cuda":
                images = images.to(memory_format=torch.channels_last)

            with torch.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._amp_enabled):
                outputs = self.model(images)
                loss, log = self.criterion(outputs, batch)
                loss = loss / self.cfg.grad_accum

            self.scaler.scale(loss).backward()
            if (it + 1) % self.cfg.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.ema.update(self.model)
            self.global_step += 1

            for k, v in log.items():
                if isinstance(v, (int, float)):
                    running[k] = running.get(k, 0.0) + v
            if it % self.cfg.log_interval == 0:
                print(f"  step {self.global_step} lr {lr:.2e} loss {log['total']:.4f}")
        return {k: v / max(1, len(loader)) for k, v in running.items()}

    def fit(self):
        steps_per_epoch = max(1, len(self.loaders["train"]))
        total_steps = steps_per_epoch * self.cfg.epochs
        best = -1.0
        for epoch in range(self.cfg.epochs):
            print(f"Epoch {epoch + 1}/{self.cfg.epochs}")
            stats = self.train_one_epoch(total_steps)
            print(f"  train: {stats}")
            if "val" in self.loaders:
                from polypai.engine.evaluator import Evaluator

                metrics = Evaluator(self.ema.shadow, self.loaders["val"], self.device).run()
                print(f"  val: {metrics}")
                if metrics.get("dice", 0) > best:
                    best = metrics["dice"]
                    self.save("best.pt", epoch, metrics)
            self.save("last.pt", epoch, stats)

    def save(self, name: str, epoch: int, metrics: dict):
        torch.save(
            {"model": self.model.state_dict(), "ema": self.ema.shadow.state_dict(),
             "optimizer": self.optimizer.state_dict(), "epoch": epoch,
             "metrics": metrics, "step": self.global_step},
            Path(self.cfg.ckpt_dir) / name,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.ema.shadow.load_state_dict(ckpt["ema"])
        return ckpt
