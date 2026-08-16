"""Record reference baseline metrics, for the same purpose as make_golden.py.

The model-based baselines (IFP, DPP) and the heuristics were also rewritten to
use the cached per-group channel quantities, so they need their own reference.

    python -m tests.make_golden_baselines
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np

from ris_mec.utils import load_config, clone_cfg
from ris_mec import baselines

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
STEPS = 150

# (label, config overrides, seed)
CASES = [
    ("k6", {}, 0),
    ("k16_g16", {"system.K": 16, "system.G": 16}, 3),
    ("passive", {"active_ris.enable_amplification": False}, 1),
    ("blocked", {"channel.direct_blockage": 0.4}, 2),
]


from tests.make_golden import REFERENCE_PINS


def _apply(cfg, over):
    for dotted, val in over.items():
        d = cfg
        ks = dotted.split(".")
        for k in ks[:-1]:
            d = d.setdefault(k, {})
        d[ks[-1]] = val
    return cfg


def case_cfg(over):
    """Config for one recorded case, with the reference pins applied."""
    return _apply(clone_cfg(load_config()), {**REFERENCE_PINS, **over})


def record():
    out = {}
    for label, over, seed in CASES:
        cfg = case_cfg(over)
        for name, fn in sorted(baselines.BASELINES.items()):
            s = fn(clone_cfg(cfg), seed=seed, steps=STEPS)
            out[f"{label}/{name}"] = s
            print(f"[golden-bl] {label:8s} {name:4s} lat={s['latency']:.6f} "
                  f"EE={s['ee_mbit_j']:.4f} q={s['queue_mean']:.4e}")
    return out


def main():
    os.makedirs(GOLDEN, exist_ok=True)
    path = os.path.join(GOLDEN, "baselines.json")
    with open(path, "w") as f:
        json.dump(record(), f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
