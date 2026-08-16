"""On-disk cache for trained PPO models.

Training dominates every study, and the studies overlap: ``k_sweep`` already
trains HDRL at K=8/seed 0, which ``ablation``, ``pareto`` and ``robustness`` then
train again. Re-running a single study after a crash or a plot tweak also pays
the full training cost.

A model is keyed by the *training-relevant* part of the config plus the seed, so
knobs that only affect evaluation (the robustness impairments) or only affect the
meta level (``hierarchy.H``, for the tactician) do not force a retrain.

Opt in with ``--cache`` on experiments/run_all.py. The cache is keyed by config
content, not by file mtime, so it is safe across processes; bump
``CACHE_VERSION`` after changing anything about the training procedure itself.
"""
from __future__ import annotations
import hashlib
import json
import os

# Bump when the training procedure changes in a way that invalidates old weights.
CACHE_VERSION = "v1-vec"

# Config subtrees that never influence training.
_EVAL_ONLY = (("channel", "csi_error_std"), ("channel", "phase_bits"),
              ("channel", "csi_delay_slots"), ("fast", "eval_steps"))


def _prune(cfg: dict, drop) -> dict:
    out = json.loads(json.dumps(cfg, sort_keys=True, default=str))
    for path in drop:
        d = out
        for k in path[:-1]:
            d = d.get(k) if isinstance(d, dict) else None
            if d is None:
                break
        if isinstance(d, dict):
            d.pop(path[-1], None)
    return out


def key(kind: str, cfg: dict, seed: int, **extra) -> str:
    """Stable hash for (training procedure, config, seed, extra flags)."""
    drop = list(_EVAL_ONLY)
    if kind == "tactician":
        # The tactician never sees H or the strategist's step budget.
        drop += [("hierarchy", "H"), ("training", "meta_timesteps"),
                 ("training", "flat_timesteps")]
    elif kind == "flat":
        drop += [("hierarchy", "H"), ("training", "low_timesteps"),
                 ("training", "meta_timesteps")]
    payload = {"v": CACHE_VERSION, "kind": kind, "seed": int(seed),
               "cfg": _prune(cfg, drop), "extra": extra}
    blob = json.dumps(payload, sort_keys=True).encode()
    return f"{kind}-{hashlib.sha256(blob).hexdigest()[:20]}"


class ModelCache:
    """No-op when ``enabled`` is False, so callers need no branching."""

    def __init__(self, directory: str, enabled: bool = False):
        self.dir = directory
        self.enabled = bool(enabled)
        if self.enabled:
            os.makedirs(self.dir, exist_ok=True)

    def get_or_train(self, kind: str, cfg: dict, seed: int, builder, loader, **extra):
        """Return a trained model, loading it from disk when possible.

        :param builder: ``() -> model``, called on a miss.
        :param loader:  ``(path) -> model``, e.g. ``PPO.load``.
        """
        if not self.enabled:
            return builder()
        path = os.path.join(self.dir, key(kind, cfg, seed, **extra) + ".zip")
        if os.path.exists(path):
            try:
                return loader(path)
            except Exception as exc:                      # corrupt / version skew
                print(f"[cache] ignoring unreadable {os.path.basename(path)}: {exc}", flush=True)
        model = builder()
        tmp = path + f".tmp{os.getpid()}"
        model.save(tmp)
        os.replace(tmp, path)                             # atomic across workers
        return model
