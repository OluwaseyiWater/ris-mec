"""Config loading, seeding, and small statistics helpers."""
from __future__ import annotations
import copy
import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def load_config(path: str = None, overrides: dict = None, fast: bool = False) -> dict:
    """Load YAML config; optionally apply the 'fast' profile and dotted overrides."""
    import os
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
    if yaml is None:
        raise RuntimeError("pyyaml not installed")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if fast and "fast" in cfg:
        tr = cfg["training"]
        for k, v in cfg["fast"].items():
            tr[k] = v
    if overrides:
        for dotted, val in overrides.items():
            d = cfg
            keys = dotted.split(".")
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = val
    return cfg


def clone_cfg(cfg: dict) -> dict:
    return copy.deepcopy(cfg)


def set_global_seed(seed: int):
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def ci95(x) -> tuple:
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float(x.mean()) if n else float("nan"), 0.0
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(n))
