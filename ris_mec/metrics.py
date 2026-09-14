"""Metric collection and summarization shared by all methods."""
from __future__ import annotations
import numpy as np
import pandas as pd


def summarize(infos: list, K: int, tau: float, arrival_mean: float) -> dict:
    """Aggregate a list of per-slot info dicts into scalar metrics.

    EE is the ratio of *sums* (total bits / total energy) -- the P1 definition --
    not the mean of per-slot ratios.

    ``latency`` is *service* time only: transmit plus compute for the bits that
    actually got served in a slot. It says nothing about how long a task waited
    before that, so a controller which lets its backlog grow scores *well* on it
    while serving its users badly. ``delay_total`` adds the queueing delay from
    Little's law, W = Q_k / thr_k with both taken per UE, and is the quantity a
    user actually experiences.

    ``served_ratio`` < 1 means the controller is not keeping up with the offered
    load. That matters before any comparison is made: energy efficiency is
    bits/energy, so it can always be raised by serving fewer bits, and two
    controllers that are both under-serving will converge on the same EE for
    reasons that have nothing to do with either of them.
    """
    df = pd.DataFrame(infos)
    n = max(len(df), 1)
    tot_bits = float(df["bits"].sum())
    tot_energy = float(df["energy"].sum())
    q_user = float(df["queue_mean"].mean())          # per-UE backlog [bits]
    thr_total = tot_bits / n                         # served bits per slot
    thr_user = thr_total / max(int(K), 1)            # per-UE, to match q_user
    w_queue = (q_user / thr_user * tau) if thr_user > 0 else float("inf")
    lat_service = float(df["latency"].mean())
    offered = int(K) * float(arrival_mean)
    return {
        "latency": lat_service,
        "delay_queue": w_queue,
        "delay_total": lat_service + w_queue,
        "throughput": thr_total,
        "served_ratio": thr_total / offered if offered > 0 else float("nan"),
        "ee_mbit_j": float(tot_bits / max(tot_energy, 1e-12) / 1e6),
        "queue_mean": q_user,
        "queue_p95": float(df["queue_mean"].quantile(0.95)),
        "energy": tot_energy,
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
