"""Metric collection and summarization shared by all methods."""
from __future__ import annotations
import numpy as np
import pandas as pd


def summarize(infos: list) -> dict:
    """Aggregate a list of per-slot info dicts into scalar metrics.

    EE is computed as the ratio of *sums* (total bits / total energy) -- the P1
    definition -- not the mean of per-slot ratios (this is exactly the
    distinction Reviewer 3 raised)."""
    df = pd.DataFrame(infos)
    tot_bits = df["bits"].sum()
    tot_energy = df["energy"].sum()
    return {
        "latency": float(df["latency"].mean()),
        "ee_mbit_j": float(tot_bits / max(tot_energy, 1e-12) / 1e6),
        "queue_mean": float(df["queue_mean"].mean()),
        "queue_p95": float(df["queue_mean"].quantile(0.95)),
        "energy": float(tot_energy),
        "a_mean": float(df["a_mean"].mean()) if "a_mean" in df else float("nan"),
        "p_ris": float(df["p_ris"].mean()) if "p_ris" in df else float("nan"),
    }


def rollout(env, policy_fn, steps: int) -> list:
    """Roll a policy for `steps` slots. policy_fn(obs, env) -> tactical action.
    Strategic variables are expected to be set inside policy_fn if needed."""
    obs, _ = env.reset()
    infos = []
    for _ in range(steps):
        a = policy_fn(obs, env)
        obs, r, done, _, info = env.step(a)
        infos.append(info)
        if done:
            obs, _ = env.reset()
    return infos
