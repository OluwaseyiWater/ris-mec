"""Shared experiment helpers: train+evaluate one method, aggregate over seeds.

The (method, K, seed) grid is embarrassingly parallel -- 840 independent runs in
the full profile -- so :func:`aggregate` can fan its seeds out over a process
pool. Each worker pins torch to a single thread: the per-worker work is small
matmuls and batched numpy, and letting N workers each spawn N threads is a net
loss.
"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from ris_mec.utils import load_config, clone_cfg, ci95, set_global_seed
from ris_mec import baselines
from ris_mec.hierarchy import train_hdrl, train_tactician, train_strategist
from ris_mec.flat import train_flat
from ris_mec.evaluate import evaluate_hdrl, evaluate_flat
from ris_mec.modelcache import ModelCache

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)
CACHE_DIR = os.path.join(RESULTS, "model_cache")

# Set by run_all via configure(); read by workers through their own module copy.
_CACHE_ENABLED = False


def configure(cache: bool = False):
    global _CACHE_ENABLED
    _CACHE_ENABLED = bool(cache)


def eval_steps(cfg):
    return int(cfg.get("fast", {}).get("eval_steps", 1000)) if cfg["training"]["low_timesteps"] <= 20000 else 1500


def _hdrl_models(cfg, seed, enable_ris, cache):
    """Train (or load) the tactician and strategist, cached independently.

    The tactician is H-agnostic, so the ablation's H sweep reuses one tactician
    across all three values of H.
    """
    from stable_baselines3 import PPO
    low = cache.get_or_train(
        "tactician", cfg, seed,
        builder=lambda: train_tactician(cfg, seed=seed),
        loader=PPO.load)
    meta = cache.get_or_train(
        "strategist", cfg, seed,
        builder=lambda: train_strategist(cfg, low, seed=seed, enable_ris=enable_ris),
        loader=PPO.load, enable_ris=enable_ris)
    return low, meta


def run_method(method, cfg, seed):
    """Return metrics dict for a single (method, cfg, seed)."""
    set_global_seed(seed)
    steps = eval_steps(cfg)
    cache = ModelCache(CACHE_DIR, enabled=_CACHE_ENABLED)
    if method in ("HDRL", "HDRL-passive"):
        enable_ris = (method == "HDRL")
        low, meta = _hdrl_models(cfg, seed, enable_ris, cache)
        return evaluate_hdrl(cfg, low, meta, seed=seed, steps=steps, enable_ris=enable_ris)
    if method in ("Flat", "Flat PPO"):
        from stable_baselines3 import PPO
        flat = cache.get_or_train("flat", cfg, seed,
                                  builder=lambda: train_flat(cfg, seed=seed),
                                  loader=PPO.load)
        return evaluate_flat(cfg, flat, seed=seed, steps=steps)
    if method in baselines.BASELINES:
        return baselines.BASELINES[method](cfg, seed=seed, steps=steps)
    raise ValueError(method)


def _worker(args):
    """Process-pool entry point. Must be module level to be picklable."""
    method, cfg, seed, cache_enabled, threads = args
    try:
        import torch
        torch.set_num_threads(int(threads))
    except Exception:
        pass
    configure(cache=cache_enabled)
    t0 = time.time()
    m = run_method(method, cfg, seed)
    m["_train_time"] = time.time() - t0
    return m


def aggregate(method, cfg, seeds, jobs: int = 1, threads: int = 1):
    """Aggregate a metric dict across seeds -> mean and 95% CI per key.

    :param jobs: seeds evaluated concurrently in separate processes (1 = inline).
    :param threads: torch threads per worker.
    """
    tasks = [(method, clone_cfg(cfg), int(s), _CACHE_ENABLED, threads) for s in seeds]
    if jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(jobs, len(tasks))) as ex:
            runs = list(ex.map(_worker, tasks))
    else:
        runs = [_worker(t) for t in tasks]

    keys = [k for k in runs[0] if not k.startswith("_")]
    out = {"method": method}
    for k in keys:
        mean, ci = ci95([r[k] for r in runs])
        out[k] = mean; out[k + "_ci"] = ci
    # Wall-clock per seed; with jobs>1 these overlap, so it is a per-run cost,
    # not the elapsed time of the sweep.
    out["train_time"] = float(np.mean([r["_train_time"] for r in runs]))
    return out


def aggregate_many(specs, jobs: int = 1, threads: int = 1, progress=None):
    """Aggregate many (label, method, cfg, seeds) specs through a *single* pool.

    Fanning every (spec, seed) pair out at once keeps all workers busy: a
    per-spec pool would idle whenever one spec has fewer seeds than workers, and
    the specs have wildly different costs (an HDRL run is ~100x a heuristic).
    Longest-first ordering keeps the expensive runs from landing at the end.
    """
    tasks, owner = [], []
    for i, (_, method, cfg, seeds) in enumerate(specs):
        for s in seeds:
            tasks.append((method, clone_cfg(cfg), int(s), _CACHE_ENABLED, threads))
            owner.append(i)
    order = sorted(range(len(tasks)), key=lambda j: -_cost_hint(tasks[j][0]))
    results = [None] * len(tasks)

    if jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(jobs, len(tasks))) as ex:
            futs = {ex.submit(_worker, tasks[j]): j for j in order}
            for n, fut in enumerate(_as_completed(futs), 1):
                results[futs[fut]] = fut.result()
                if progress:
                    progress(n, len(tasks))
    else:
        for n, j in enumerate(order, 1):
            results[j] = _worker(tasks[j])
            if progress:
                progress(n, len(tasks))

    out = []
    for i, (label, method, _, _) in enumerate(specs):
        runs = [results[j] for j in range(len(tasks)) if owner[j] == i]
        agg = {"method": method}
        for k in [k for k in runs[0] if not k.startswith("_")]:
            mean, ci = ci95([r[k] for r in runs])
            agg[k] = mean; agg[k + "_ci"] = ci
        agg["train_time"] = float(np.mean([r["_train_time"] for r in runs]))
        agg["_label"] = label
        out.append(agg)
    return out


def _cost_hint(method: str) -> int:
    """Rough relative cost, used only to schedule long runs first."""
    return {"HDRL": 100, "HDRL-passive": 100, "Flat": 20}.get(method, 1)


def _as_completed(futs):
    from concurrent.futures import as_completed
    return as_completed(futs)


def save_json(name, obj):
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


def config_fingerprint() -> str:
    """SHA of configs/default.yaml, so results can be tied to the model that made them."""
    import hashlib
    p = os.path.join(os.path.dirname(RESULTS), "configs", "default.yaml")
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def write_provenance(profile: str, studies):
    """Record what produced results/, so the paper can detect stale numbers.

    Timestamps are not enough: restoring results from version control, or editing
    the config after a run, both leave mtimes that look fine. The config hash does
    not.
    """
    import subprocess
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                cwd=os.path.dirname(RESULTS), capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
    except Exception:
        commit = None
    return save_json("_provenance.json", {
        "profile": profile,
        "studies": list(studies),
        "config_sha": config_fingerprint(),
        "git_commit": commit,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
