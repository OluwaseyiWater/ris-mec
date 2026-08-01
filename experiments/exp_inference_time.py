"""Measure per-slot *inference/decision* time (not training time) for each method
at K=10, and regenerate results/fig_training_cost.pdf accordingly.

Inference time is independent of the learned weights, so untrained PPO policies of
the correct dimensions give a faithful timing for the learned methods. Baselines are
timed by their per-slot solve. Env.step() physics is excluded so we compare the
decision cost itself.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from ris_mec.utils import load_config
from ris_mec.env import RISMECEnv
from ris_mec.hierarchy import MetaManager, TacticianEnv, MetaEnv
from ris_mec.flat import FlatEnv
from ris_mec import baselines
from ris_mec.plotstyle import apply_style, style_for
from experiments.common import RESULTS

N_SLOTS = 300


def time_learned_hdrl(cfg):
    low = PPO("MlpPolicy", DummyVecEnv([lambda: TacticianEnv(cfg, seed=0)]), n_steps=64, verbose=0)
    meta = PPO("MlpPolicy", DummyVecEnv([lambda: MetaEnv(cfg, low, seed=0)]), n_steps=64, verbose=0)
    env = RISMECEnv(cfg, seed=0); mgr = MetaManager(env)
    obs, _ = env.reset(); H = cfg["hierarchy"]["H"]
    t0 = time.perf_counter()
    for _ in range(N_SLOTS):
        if env.t % H == 0:
            a_meta, _ = meta.predict(mgr.meta_obs(), deterministic=True); mgr.apply_meta(a_meta)
        a_low, _ = low.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(a_low)
        if done: obs, _ = env.reset()
    return 1e3 * (time.perf_counter() - t0) / N_SLOTS


def time_learned_flat(cfg):
    fenv = FlatEnv(cfg, seed=0)
    flat = PPO("MlpPolicy", DummyVecEnv([lambda: FlatEnv(cfg, seed=0)]), n_steps=64, verbose=0)
    obs, _ = fenv.reset()
    t0 = time.perf_counter()
    for _ in range(N_SLOTS):
        a, _ = flat.predict(obs, deterministic=True)
        obs, _, done, _, _ = fenv.step(a)
        if done: obs, _ = fenv.reset()
    return 1e3 * (time.perf_counter() - t0) / N_SLOTS


def time_baseline(cfg, fn):
    # crude: total wall time for N_SLOTS minus an env-only baseline
    env = RISMECEnv(cfg, seed=0); env.reset()
    t0 = time.perf_counter(); fn(cfg, seed=0, steps=N_SLOTS); dt = time.perf_counter() - t0
    return 1e3 * dt / N_SLOTS


def main():
    cfg = load_config()
    cfg["system"]["K"] = 10
    cfg.setdefault("fast", {})["eval_steps"] = N_SLOTS
    res = {
        "HDRL": time_learned_hdrl(cfg),
        "Flat": time_learned_flat(cfg),
        "DPP": time_baseline(cfg, baselines.run_DPP),
        "IFP": time_baseline(cfg, baselines.run_IFP),
    }
    for k, v in res.items():
        print(f"{k:5s} {v:.3f} ms/slot")
    apply_style()
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    order = ["HDRL", "Flat", "DPP", "IFP"]
    ax.bar(order, [res[m] for m in order],
           color=[style_for("Flat PPO" if m == "Flat" else m)["color"] for m in order])
    ax.set_ylabel("Per-slot inference (ms)"); ax.set_yscale("log")
    ax.set_title("Per-slot decision cost ($K$=10)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS, f"fig_training_cost.{ext}"))
    import json
    json.dump(res, open(os.path.join(RESULTS, "inference_time.json"), "w"), indent=2)
    print("regenerated fig_training_cost")


if __name__ == "__main__":
    main()
