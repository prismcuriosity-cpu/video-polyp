"""Anchor-free (FCOS) target assignment with center sampling.

Produces, per pyramid level, the classification / box-distance / centerness
targets for the :class:`FCOSDetHead`.  Center sampling and per-level regression
ranges follow Tian et al. (2019); the high-resolution P2 level uses the smallest
range so diminutive lesions are matched there.
"""

from __future__ import annotations

import torch

# regression ranges for P2..P5 (strides 4,8,16,32)
DEFAULT_REGRESS_RANGES = ((-1, 64), (64, 128), (128, 256), (256, 1e8))
DEFAULT_STRIDES = (4, 8, 16, 32)


def locations_for_level(h: int, w: int, stride: int, device) -> torch.Tensor:
    """(H*W, 2) pixel-centre coordinates for a feature map."""
    shifts_x = (torch.arange(w, device=device) + 0.5) * stride
    shifts_y = (torch.arange(h, device=device) + 0.5) * stride
    yy, xx = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)


def _centerness(reg: torch.Tensor) -> torch.Tensor:
    l, t, r, b = reg.unbind(-1)
    lr = torch.stack([l, r], -1).clamp(min=0)
    tb = torch.stack([t, b], -1).clamp(min=0)
    ctr = (lr.min(-1).values / lr.max(-1).values.clamp(min=1e-6)) * (
        tb.min(-1).values / tb.max(-1).values.clamp(min=1e-6)
    )
    return torch.sqrt(ctr.clamp(min=0))


def assign_image(
    boxes: torch.Tensor,
    level_shapes: list[tuple[int, int]],
    strides=DEFAULT_STRIDES,
    regress_ranges=DEFAULT_REGRESS_RANGES,
    center_radius: float = 1.5,
):
    """Assign targets for a single image.

    Returns dict with concatenated-over-levels tensors:
      locations (M,2), cls (M,) {0,1}, reg (M,4) pixel distances, centerness (M,),
      pos_mask (M,) bool, and level_sizes for splitting back per level.
    """
    device = boxes.device if boxes.numel() else torch.device("cpu")
    all_locs, level_sizes = [], []
    for (h, w), s in zip(level_shapes, strides):
        locs = locations_for_level(h, w, s, device)
        all_locs.append(locs)
        level_sizes.append(h * w)
    locations = torch.cat(all_locs, dim=0)
    M = locations.shape[0]

    cls = torch.zeros(M, dtype=torch.long, device=device)
    reg = torch.zeros(M, 4, device=device)
    ctr = torch.zeros(M, device=device)
    pos = torch.zeros(M, dtype=torch.bool, device=device)

    if boxes.numel() == 0:
        return dict(locations=locations, cls=cls, reg=reg, centerness=ctr,
                    pos_mask=pos, level_sizes=level_sizes)

    # per-location regression range (expanded over levels)
    ranges = torch.cat([
        locations.new_tensor(rng).expand(sz, 2) for rng, sz in zip(regress_ranges, level_sizes)
    ], dim=0)
    strides_per_loc = torch.cat([
        locations.new_full((sz,), s) for s, sz in zip(strides, level_sizes)
    ], dim=0)

    xs, ys = locations[:, 0], locations[:, 1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    INF = 1e8
    best_area = torch.full((M,), INF, device=device)

    for k in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[k]
        l = xs - x1; t = ys - y1; r = x2 - xs; b = y2 - ys
        reg_k = torch.stack([l, t, r, b], dim=-1)
        inside = reg_k.min(-1).values > 0

        # center sampling
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        radius = center_radius * strides_per_loc
        in_center = (xs > cx - radius) & (xs < cx + radius) & (ys > cy - radius) & (ys < cy + radius)

        max_reg = reg_k.max(-1).values
        in_range = (max_reg >= ranges[:, 0]) & (max_reg <= ranges[:, 1])

        valid = inside & in_center & in_range & (areas[k] < best_area)
        if valid.any():
            best_area[valid] = areas[k]
            cls[valid] = 1
            reg[valid] = reg_k[valid]
            pos[valid] = True

    ctr[pos] = _centerness(reg[pos])
    return dict(locations=locations, cls=cls, reg=reg, centerness=ctr,
                pos_mask=pos, level_sizes=level_sizes)


def decode_boxes(locations: torch.Tensor, reg: torch.Tensor) -> torch.Tensor:
    """l,t,r,b distances -> xyxy boxes."""
    x, y = locations[:, 0], locations[:, 1]
    return torch.stack([x - reg[:, 0], y - reg[:, 1], x + reg[:, 2], y + reg[:, 3]], dim=-1)
