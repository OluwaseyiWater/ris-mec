"""Record reference trajectories from the environment for regression testing.

Run this ONCE against a known-good implementation to produce
``tests/golden/env_traj.npz``; ``tests/test_equivalence.py`` then asserts that any
later (vectorized) implementation reproduces it bit-for-bit-ish.

    python -m tests.make_golden
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ris_mec.utils import load_config, clone_cfg
from ris_mec.env import RISMECEnv

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")

# Every default that has intentionally changed since the reference recording,
# pinned back to its reference value. This is what lets the golden files keep
# testing exactly one thing -- that the vectorized core reproduces the original
# physics -- instead of drifting into a test of whatever the defaults are today.
# Add an entry here whenever a default changes, and cover the change with its own
# check in tests/test_equivalence.py.
REFERENCE_PINS = {
    # observation-only (check_obs_normalization)
    "channel.obs_gain_norm": "legacy",
    "channel.obs_group_features": False,
    # physics (check_steering), with its companion link-budget parameter
    "geometry.ris_steering": False,
    "hierarchy.drift_norm": "legacy",
    "hierarchy.dpp_V": 1.0,
    "baselines.dpp_mode": "legacy",
    "baselines.ifp_mode": "legacy",
    "mec.result_queue": False,
    "hierarchy.dl_power_action": "linear",
    "queue.arrival_mean_bits": 3.0e5,
}

# (label, config overrides, seed) -- cover the physics branches that matter.
CASES = [
    ("base_k6", {}, 0),
    ("k16_g16", {"system.K": 16, "system.G": 16}, 3),
    ("passive", {"active_ris.enable_amplification": False}, 1),
    ("blocked", {"channel.direct_blockage": 0.4}, 2),
    ("csi_err", {"channel.csi_error_std": 0.1}, 5),
    ("delay_q", {"channel.csi_delay_slots": 3, "channel.phase_bits": 2}, 7),
    ("obj_eela", {"hierarchy.objective": "ee_latency"}, 0),
    ("obj_dink", {"hierarchy.objective": "dinkelbach"}, 0),
]
STEPS = 120


def _apply(cfg, over):
    for dotted, val in over.items():
        d = cfg
        ks = dotted.split(".")
        for k in ks[:-1]:
            d = d.setdefault(k, {})
        d[ks[-1]] = val
    return cfg


def record(over, seed, steps=STEPS):
    """Deterministic scripted rollout -> stacked observations/rewards/infos.

    ``REFERENCE_PINS`` restores the reference behaviour for every default that
    has intentionally changed since, so this file keeps testing exactly one
    thing: that the vectorized core reproduces the original physics.
    """
    cfg = _apply(clone_cfg(load_config()), {**REFERENCE_PINS, **over})
    env = RISMECEnv(cfg, seed=seed)
    obs, _ = env.reset()
    rng = np.random.default_rng(12345)          # action stream, independent of env RNG
    A = rng.uniform(-1, 1, size=(steps, env.action_space.shape[0]))
    S = rng.uniform(-1, 1, size=(steps, 2 * env.K + env.G))

    keys = ("latency", "energy", "bits", "ee", "p_ris", "queue_mean", "a_mean")
    out = {k: [] for k in keys}
    out["obs"], out["reward"] = [obs.copy()], []
    for t in range(steps):
        if t % 10 == 0:                          # exercise set_strategy on a schedule
            s = S[t]
            env.set_strategy(alpha=(s[:env.K] + 1) / 2,
                             f=(s[env.K:2 * env.K] + 1) / 2 * env.F_max / env.K,
                             a_gains=(s[2 * env.K:] + 1) / 2 * env.a_max)
        obs, r, done, _, info = env.step(A[t])
        out["obs"].append(obs.copy())
        out["reward"].append(r)
        for k in keys:
            out[k].append(info[k])
        if done:
            obs, _ = env.reset()
            out["obs"][-1] = obs.copy()
    return {k: np.asarray(v) for k, v in out.items()}


def main():
    os.makedirs(GOLDEN, exist_ok=True)
    blob = {}
    for label, over, seed in CASES:
        traj = record(over, seed)
        for k, v in traj.items():
            blob[f"{label}/{k}"] = v
        print(f"[golden] {label:9s} steps={len(traj['reward'])} "
              f"lat={traj['latency'].mean():.5f} ee={traj['ee'].mean():.4e} "
              f"r={traj['reward'].mean():.6f}")
    path = os.path.join(GOLDEN, "env_traj.npz")
    np.savez_compressed(path, **blob)
    print(f"wrote {path} ({os.path.getsize(path)/1024:.0f} KiB)")


if __name__ == "__main__":
    main()
