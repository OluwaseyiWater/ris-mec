"""Evaluation rollouts for the learned policies (HDRL, Flat PPO)."""
from __future__ import annotations
import numpy as np

from .env import RISMECEnv
from .hierarchy import MetaManager
from .flat import FlatEnv
from .metrics import summarize


def evaluate_hdrl(cfg, low, meta, seed=0, steps=1000, enable_ris=True) -> dict:
    env = RISMECEnv(cfg, seed=seed)
    mgr = MetaManager(env, enable_ris=enable_ris)
    obs, _ = env.reset()
    H = cfg["hierarchy"]["H"]
    infos = []
    for _ in range(steps):
        if env.t % H == 0:
            a_meta, _ = meta.predict(mgr.meta_obs(), deterministic=True)
            mgr.apply_meta(a_meta)
        a_low, _ = low.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(a_low)
        infos.append(info)
        if done:
            obs, _ = env.reset()
    return summarize(infos)


def evaluate_flat(cfg, flat, seed=0, steps=1000) -> dict:
    fenv = FlatEnv(cfg, seed=seed)
    obs, _ = fenv.reset()
    infos = []
    for _ in range(steps):
        a, _ = flat.predict(obs, deterministic=True)
        obs, _, done, _, info = fenv.step(a)
        infos.append(info)
        if done:
            obs, _ = fenv.reset()
    return summarize(infos)
