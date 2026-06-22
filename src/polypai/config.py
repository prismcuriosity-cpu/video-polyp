"""YAML experiment-configuration loading.

A single YAML file declares ``model``, ``data``, ``train`` (and optionally
``pipeline``) sections; each maps onto the corresponding dataclass.  Unknown keys
are ignored with a warning so configs stay forward-compatible.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path
from typing import Any, Type, TypeVar

T = TypeVar("T")


def load_yaml(path: str | Path) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


def from_dict(cls: Type[T], d: dict | None) -> T:
    """Instantiate a dataclass from a dict, ignoring (and warning on) extra keys."""
    if d is None:
        return cls()  # type: ignore[call-arg]
    fields = {f.name for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for k, v in d.items():
        if k in fields:
            # coerce list -> tuple for tuple-typed fields
            kwargs[k] = tuple(v) if isinstance(v, list) and _is_tuple_field(cls, k) else v
        else:
            warnings.warn(f"[config] ignoring unknown key '{k}' for {cls.__name__}")
    return cls(**kwargs)  # type: ignore[call-arg]


def _is_tuple_field(cls, name: str) -> bool:
    for f in dataclasses.fields(cls):
        if f.name == name:
            return "tuple" in str(f.type).lower()
    return False


def build_experiment(path: str | Path):
    """Return (model_cfg, data_cfg, train_cfg) dataclasses from a YAML file."""
    from polypai.data.datamodule import DataConfig
    from polypai.engine.trainer import TrainConfig
    from polypai.models.config import ModelConfig

    raw = load_yaml(path)
    model_cfg = from_dict(ModelConfig, raw.get("model"))
    data_cfg = from_dict(DataConfig, raw.get("data"))
    train_cfg = from_dict(TrainConfig, raw.get("train"))
    return model_cfg, data_cfg, train_cfg
