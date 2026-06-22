"""Declarative configuration for the unified model (YAML-friendly)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # backbone
    in_channels: int = 3
    stem_channels: int = 48
    stage_channels: tuple[int, ...] = (64, 128, 256, 512)
    stage_depths: tuple[int, ...] = (2, 2, 4, 2)
    width_mult: float = 1.0
    use_cbam: bool = True
    gradient_checkpointing: bool = False

    # neck
    fpn_channels: int = 192
    extra_p2: bool = True  # high-res P2 level for 2 mm lesion detection

    # detection head (anchor-free, FCOS-style)
    det_classes: int = 1            # polyp vs background (class-agnostic localisation)
    det_head_convs: int = 3

    # segmentation head
    seg_classes: int = 2            # background + polyp
    seg_decoder_channels: int = 128

    # classification heads {name: num_classes}
    cls_heads: dict[str, int] = field(default_factory=lambda: {
        "paris": 6,        # 0-Ip,0-Is,0-IIa,0-IIb,0-IIc,0-III
        "nice": 3,         # NICE 1/2/3
        "kudo": 7,         # Kudo I,II,IIIs,IIIL,IV,VI,VN
        "malignancy": 2,   # benign vs malignant
    })

    # explainability / uncertainty
    enable_uncertainty: bool = True

    @property
    def scaled_stage_channels(self) -> tuple[int, ...]:
        return tuple(int(round(c * self.width_mult)) for c in self.stage_channels)
