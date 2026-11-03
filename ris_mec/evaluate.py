"""Evaluation rollouts for the learned policies (HDRL, Flat PPO).

Evaluation deliberately stays **single-world**: the reported latency/EE are means
over ``steps`` slots of one world, and the 95% CIs come from the spread across
seeds. Averaging over parallel worlds here would silently tighten those CIs and
make the numbers incomparable to previously reported runs.

The speed-up here is only in how the action is obtained -- :class:`FrozenPolicy`
instead of ``model.predict`` -- which is semantically identical.
"""
from __future__ import annotations
import numpy as np

from .env import RISMECEnv, CARRY
from .hierarchy import MetaManager
from .vec_env import FrozenPolicy
from .flat import FlatEnv
from .metrics import summarize


def evaluate_hdrl(cfg, low, meta, seed=0, steps=1000, enable_ris=True) -> dict:
    env = RISMECEnv(cfg, seed=seed)
    mgr = MetaManager(env, enable_ris=enable_ris)
    low_pi, meta_pi = FrozenPolicy(low), FrozenPolicy(meta)
    obs, _ = env.reset()
    H = cfg["hierarchy"]["H"]
    infos = []
    for _ in range(steps):
        if env.t % H == 0:
            mgr.apply_meta(meta_pi(mgr.meta_obs()[None, :])[0])
        obs, _, done, _, info = env.step(low_pi(obs[None, :])[0])
        infos.append(info)
        if done:
            obs, _ = env.reset(options=CARRY)
    return summarize(infos, env.K, env.tau, env.arrival_mean)


def evaluate_flat(cfg, flat, seed=0, steps=1000) -> dict:
    fenv = FlatEnv(cfg, seed=seed)
    pi = FrozenPolicy(flat)
    obs, _ = fenv.reset()
    infos = []
    for _ in range(steps):
        obs, _, done, _, info = fenv.step(pi(obs[None, :])[0])
        infos.append(info)
        if done:
            obs, _ = fenv.reset(options=CARRY)
    e = fenv.env
    return summarize(infos, e.K, e.tau, e.arrival_mean)
