"""Shared experiment helpers: train+evaluate one method, aggregate over seeds."""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ris_mec.utils import load_config, clone_cfg, ci95, set_global_seed
from ris_mec import baselines
from ris_mec.hierarchy import train_hdrl
from ris_mec.flat import train_flat
from ris_mec.evaluate import evaluate_hdrl, evaluate_flat

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)


def eval_steps(cfg):
    return int(cfg.get("fast", {}).get("eval_steps", 1000)) if cfg["training"]["low_timesteps"] <= 20000 else 1500


def run_method(method, cfg, seed):
    """Return metrics dict for a single (method, cfg, seed)."""
    set_global_seed(seed)
    steps = eval_steps(cfg)
    if method == "HDRL":
        low, meta = train_hdrl(cfg, seed=seed, enable_ris=True)
        return evaluate_hdrl(cfg, low, meta, seed=seed, steps=steps, enable_ris=True)
    if method == "HDRL-passive":
        low, meta = train_hdrl(cfg, seed=seed, enable_ris=False)
        return evaluate_hdrl(cfg, low, meta, seed=seed, steps=steps, enable_ris=False)
    if method in ("Flat", "Flat PPO"):
        flat = train_flat(cfg, seed=seed)
        return evaluate_flat(cfg, flat, seed=seed, steps=steps)
    if method in baselines.BASELINES:
        return baselines.BASELINES[method](cfg, seed=seed, steps=steps)
    raise ValueError(method)


def aggregate(method, cfg, seeds):
    """Aggregate a metric dict across seeds -> mean and 95% CI per key."""
    runs = []
    for s in seeds:
        t0 = time.time()
        m = run_method(method, cfg, s)
        m["_train_time"] = time.time() - t0
        runs.append(m)
    keys = [k for k in runs[0] if not k.startswith("_")]
    out = {"method": method}
    for k in keys:
        mean, ci = ci95([r[k] for r in runs])
        out[k] = mean; out[k + "_ci"] = ci
    out["train_time"] = float(np.mean([r["_train_time"] for r in runs]))
    return out


def save_json(name, obj):
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path
