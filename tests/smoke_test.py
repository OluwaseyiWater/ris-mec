"""End-to-end smoke test: env physics, baselines, and a tiny HDRL/Flat train+eval.
Run: python -m tests.smoke_test  (from repo root)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ris_mec.utils import load_config
from ris_mec.env import RISMECEnv
from ris_mec.metrics import summarize
from ris_mec import baselines


def check_env_physics():
    cfg = load_config(fast=True)
    env = RISMECEnv(cfg, seed=0)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape, "obs shape mismatch"
    infos = []
    for _ in range(50):
        a = env.action_space.sample()
        obs, r, done, _, info = env.step(a)
        assert np.isfinite(r), "non-finite reward"
        for key in ("latency", "energy", "bits", "ee", "p_ris"):
            assert np.isfinite(info[key]), f"non-finite {key}"
        assert info["p_ris"] <= cfg["power"]["ris_power_budget"] + 1e-6, "RIS budget violated"
        infos.append(info)
        if done:
            obs, _ = env.reset()
    s = summarize(infos)
    print("[env] random-action summary:", {k: round(v, 4) for k, v in s.items()})

    # Amplification sanity: higher a_g should raise both signal gain AND noise floor.
    env2 = RISMECEnv(cfg, seed=1); env2.reset()
    env2.set_strategy(a_gains=np.ones(env2.G) * 1.0)
    ul_lo, _ = env2._noise_floors(np.ones(env2.G) * 1.0, *env2.chan.true_channels()[:2])
    ul_hi, _ = env2._noise_floors(np.ones(env2.G) * 3.0, *env2.chan.true_channels()[:2])
    assert ul_hi > ul_lo, "amplifier noise should grow with a_g"
    print(f"[env] amplifier-noise floor a=1 -> {ul_lo:.3e}, a=3 -> {ul_hi:.3e}  (grows, OK)")


def check_baselines():
    cfg = load_config(fast=True)
    for name, fn in baselines.BASELINES.items():
        t0 = time.time()
        s = fn(cfg, seed=0, steps=cfg["fast"]["eval_steps"])
        print(f"[baseline {name:4s}] lat={s['latency']:.4f}s  EE={s['ee_mbit_j']:.3f} Mbit/J  "
              f"q={s['queue_mean']:.2e}  ({time.time()-t0:.1f}s)")


def check_learning():
    from ris_mec.hierarchy import train_hdrl
    from ris_mec.flat import train_flat
    from ris_mec.evaluate import evaluate_hdrl, evaluate_flat
    cfg = load_config(fast=True)
    t0 = time.time()
    low, meta = train_hdrl(cfg, seed=0)
    s_h = evaluate_hdrl(cfg, low, meta, seed=0, steps=cfg["fast"]["eval_steps"])
    print(f"[HDRL]  lat={s_h['latency']:.4f}s  EE={s_h['ee_mbit_j']:.3f} Mbit/J  ({time.time()-t0:.1f}s)")
    t1 = time.time()
    flat = train_flat(cfg, seed=0)
    s_f = evaluate_flat(cfg, flat, seed=0, steps=cfg["fast"]["eval_steps"])
    print(f"[Flat]  lat={s_f['latency']:.4f}s  EE={s_f['ee_mbit_j']:.3f} Mbit/J  ({time.time()-t1:.1f}s)")


if __name__ == "__main__":
    print("=== physics ==="); check_env_physics()
    print("=== baselines ==="); check_baselines()
    print("=== learning (tiny) ==="); check_learning()
    print("ALL SMOKE CHECKS PASSED")
