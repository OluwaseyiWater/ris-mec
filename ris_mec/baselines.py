"""
Baselines.

Heuristics:
  * AO  -- Adaptive Offloading (random-walk heuristic on alpha),
  * SCA -- Static Centralized Allocation (fixed alpha=0.6),
  * FP  -- Fixed Power (uniform random power/phase).

Optimization baselines:
  * IFP -- Iterative Fractional Programming (Shen & Yu, 2018) for *instantaneous*
    EE via the quadratic transform on power (per-slot, queue-agnostic).
  * DPP -- NEW Lyapunov drift-plus-penalty controller.
    A genuinely queue-aware, model-based baseline: each slot it performs
    max-weight scheduling (bandwidth/offload proportional to backlog), greedy
    co-phasing of the RIS groups, and a per-user power choice that minimizes
    Q_k*(A_k - b_k) + V*energy.

All baselines read the true channels from the env (model-based CSI assumption,
standard for Lyapunov/FP methods) and are evaluated with the same metric code.
"""

from __future__ import annotations
import numpy as np

from .env import RISMECEnv
from .metrics import summarize


# --------------------------------------------------------------------------- #
#  Helpers to build a tactical action vector for the env
# --------------------------------------------------------------------------- #
def _action(env, ul_p, dl_p, beta, theta_g):
    K, G = env.K, env.G
    a = np.zeros(K + 1 + K + G)
    a[:K] = np.clip(ul_p / env.Pk_max, 0, 1) * 2 - 1
    a[K] = np.clip(dl_p / env.PBS_max, 0, 1) * 2 - 1
    # beta as logits: use log to invert softmax approximately
    b = np.clip(beta, 1e-6, None); b = np.log(b)
    a[K + 1:K + 1 + K] = b - b.mean()
    a[-G:] = np.clip(theta_g / (2 * np.pi), 0, 1) * 2 - 1
    return a


def _greedy_phase(env, a_gains):
    """Pick group phases that co-phase the dominant cascade term per group,
    maximizing sum_k |h_k|^2 approximately (cheap 1-shot alignment).

    The per-group cascade response ``sum_{n in gp} hr[k,n]*g[n]`` is exactly the
    coefficient the env already caches once per fading block, so the old
    per-group Python loop over slices collapses to one sum over users."""
    C = env.chan.cascade_coeff()                 # (K,G), cached per fading block
    return np.mod(-np.angle(C.sum(axis=0)), 2 * np.pi)


def _composite_gain(env, a_gains, theta_g):
    """|h_k|^2 for one world, via the cached per-group cascade coefficients."""
    _, _, hd = env.chan.true_channels()
    h = hd + env.chan.cascade_coeff() @ (a_gains * np.exp(1j * theta_g))
    return h.real ** 2 + h.imag ** 2


def _noise_floors(env, a_gains):
    """Uplink/downlink amplifier-noise floors for one world, from cached
    reductions rather than a fresh O(K*N) pass every slot."""
    ul, dl = env.core._noise_floors(np.asarray(a_gains, float)[None, :])
    return ul[0], dl[0]


# --------------------------------------------------------------------------- #
#  Heuristic baselines
# --------------------------------------------------------------------------- #
def run_AO(cfg, seed=0, steps=1000):
    env = RISMECEnv(cfg, seed=seed); env.reset(); infos = []
    rng = np.random.default_rng(seed + 11)
    for _ in range(steps):
        env.alpha = np.clip(env.alpha + 0.05 * rng.standard_normal(env.K), 0, 1)
        env.f = np.full(env.K, env.F_max / env.K)
        a = rng.uniform(-0.5, 0.5, env.action_space.shape[0])
        _, _, done, _, info = env.step(a); infos.append(info)
        if done: env.reset()
    return summarize(infos, env.K, env.tau, env.arrival_mean)


def run_SCA(cfg, seed=0, steps=1000):
    env = RISMECEnv(cfg, seed=seed); env.reset(); infos = []
    for _ in range(steps):
        env.alpha = np.full(env.K, 0.6)
        env.f = np.full(env.K, env.F_max / env.K)
        a = np.zeros(env.action_space.shape[0])
        _, _, done, _, info = env.step(a); infos.append(info)
        if done: env.reset()
    return summarize(infos, env.K, env.tau, env.arrival_mean)


def run_FP(cfg, seed=0, steps=1000):
    env = RISMECEnv(cfg, seed=seed); env.reset(); infos = []
    rng = np.random.default_rng(seed + 13)
    for _ in range(steps):
        env.alpha = rng.random(env.K)
        env.f = np.full(env.K, env.F_max / env.K)
        a = rng.uniform(-1, 1, env.action_space.shape[0])
        _, _, done, _, info = env.step(a); infos.append(info)
        if done: env.reset()
    return summarize(infos, env.K, env.tau, env.arrival_mean)


# --------------------------------------------------------------------------- #
#  Iterative Fractional Programming (instantaneous EE)
# --------------------------------------------------------------------------- #
def run_IFP(cfg, seed=0, steps=1000, inner=6):
    env = RISMECEnv(cfg, seed=seed); env.reset(); infos = []
    for _ in range(steps):
        env.alpha = np.full(env.K, 0.5)
        env.f = np.full(env.K, env.F_max / env.K)
        theta_g = _greedy_phase(env, env.a_gains)
        # Quadratic-transform FP on uplink powers for EE (queue-agnostic).
        p = np.full(env.K, env.Pk_max * 0.5)
        beta = np.full(env.K, 1.0 / env.K)
        h2 = _composite_gain(env, env.a_gains, theta_g)
        W = beta * env.B
        ul_ris, _ = _noise_floors(env, env.a_gains)
        n = W * (env.n0 + ul_ris)
        for _ in range(inner):
            sinr = p * h2 / np.maximum(n, 1e-30)
            rate = W * np.log2(1 + sinr)
            y = np.sqrt(np.maximum(rate, 0)) / (np.sum(p) + 5.0)
            # closed-form-ish power update maximizing 2y sqrt(rate) - y^2 * power
            grad_scale = (y ** 2) / np.maximum(h2 / n, 1e-30)
            p = np.clip(1.0 / np.maximum(grad_scale, 1e-9), 0, env.Pk_max)
        a = _action(env, p, env.PBS_max, beta, theta_g)
        _, _, done, _, info = env.step(a); infos.append(info)
        if done: env.reset()
    return summarize(infos, env.K, env.tau, env.arrival_mean)


# --------------------------------------------------------------------------- #
#  Lyapunov drift-plus-penalty controller
# --------------------------------------------------------------------------- #
def run_DPP(cfg, seed=0, steps=1000):
    env = RISMECEnv(cfg, seed=seed); env.reset(); infos = []
    V = float(cfg["hierarchy"].get("dpp_V", 1.0))
    lam = float(cfg["hierarchy"].get("lambda_E", 1.0))
    for _ in range(steps):
        Q = env.queues
        # Max-weight: bandwidth and offloading proportional to backlog.
        beta = Q / max(Q.sum(), 1e-9)
        env.alpha = np.clip(Q / (Q + env.arrival_mean), 0.2, 1.0)   # drain long queues
        env.f = env.F_max * (Q / max(Q.sum(), 1e-9))               # CPU to backlog
        # Greedy RIS co-phasing at the (fixed) active gain.
        theta_g = _greedy_phase(env, env.a_gains)
        # Per-user power: minimize Q_k*(A - b_k) + V*energy  ~  choose p to
        # balance queue drainage against energy. 1-D grid per user (cheap, robust),
        # evaluated for all users at once as a (K, grid) broadcast.
        h2 = _composite_gain(env, env.a_gains, theta_g)
        W = beta * env.B
        ul_ris, _ = _noise_floors(env, env.a_gains)
        n = W * (env.n0 + ul_ris)
        grid = np.linspace(0.0, env.Pk_max, 9)[1:]                     # (P,)
        rate = W[:, None] * np.log2(
            1 + grid[None, :] * h2[:, None] / np.maximum(n, 1e-30)[:, None])
        served = np.minimum((env.alpha * Q)[:, None], rate * env.tau)   # (K,P)
        objective = Q[:, None] * served - V * (grid[None, :] * env.tau)
        p = grid[np.argmax(objective, axis=1)]
        a = _action(env, p, env.PBS_max, beta, theta_g)
        _, _, done, _, info = env.step(a); infos.append(info)
        if done: env.reset()
    return summarize(infos, env.K, env.tau, env.arrival_mean)


BASELINES = {
    "AO": run_AO, "SCA": run_SCA, "FP": run_FP, "IFP": run_IFP, "DPP": run_DPP,
}
