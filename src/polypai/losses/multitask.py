"""Multi-task loss balancing for the unified architecture.

Two complementary strategies are provided:

* :class:`UncertaintyWeighting` - homoscedastic task uncertainty (Kendall, Gal &
  Cipolla, 2018).  Learns a log-variance per task; well-conditioned tasks get
  larger weight automatically.  Zero hyper-parameters to tune.
* :class:`GradNormWeighter` - GradNorm (Chen et al., 2018).  Equalises the
  gradient-norm magnitudes of each task w.r.t. the shared backbone so no single
  task dominates the encoder updates.

Both return a single scalar to call ``.backward()`` on, plus per-task weights for
logging.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class UncertaintyWeighting(nn.Module):
    """L = sum_i exp(-s_i) * L_i + s_i  (s_i = log variance, learned)."""

    def __init__(self, task_names: list[str]):
        super().__init__()
        self.task_names = list(task_names)
        self.log_vars = nn.Parameter(torch.zeros(len(task_names)))

    def forward(self, losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        total = 0.0
        weights = {}
        for i, name in enumerate(self.task_names):
            if name not in losses:
                continue
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * losses[name] + self.log_vars[i]
            weights[name] = float(precision.detach())
        return total, weights


class GradNormWeighter(nn.Module):
    """GradNorm dynamic task weighting.

    Maintains a learnable weight per task and, given the per-task losses and a
    set of shared parameters, returns a (a) weighted task loss to optimise the
    network and (b) a gradient-balancing loss to optimise the task weights.
    """

    def __init__(self, task_names: list[str], alpha: float = 1.5):
        super().__init__()
        self.task_names = list(task_names)
        self.alpha = alpha
        self.weights = nn.Parameter(torch.ones(len(task_names)))
        self.register_buffer("initial_losses", torch.zeros(len(task_names)))
        self.register_buffer("_initialised", torch.zeros((), dtype=torch.bool))

    def weighted_sum(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(self.weights[i] * losses[n] for i, n in enumerate(self.task_names) if n in losses)

    def gradnorm_loss(self, losses: dict[str, torch.Tensor], shared_param: torch.Tensor) -> torch.Tensor:
        device = shared_param.device
        loss_vec = torch.stack([losses[n] for n in self.task_names])
        if not bool(self._initialised):
            self.initial_losses = loss_vec.detach()
            self._initialised = torch.ones((), dtype=torch.bool, device=device)
        gnorms = []
        for i, n in enumerate(self.task_names):
            g = torch.autograd.grad(losses[n], shared_param, retain_graph=True, create_graph=True)[0]
            gnorms.append(self.weights[i] * g.norm())
        gnorms = torch.stack(gnorms)
        ratios = loss_vec.detach() / self.initial_losses.clamp_min(1e-8)
        inv_rate = ratios / ratios.mean().clamp_min(1e-8)
        target = (gnorms.mean().detach() * inv_rate**self.alpha).detach()
        return torch.abs(gnorms - target).sum()
