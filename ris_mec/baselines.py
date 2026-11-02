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
    a[K] = env.core.encode_dl(dl_p)
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
#  What a model-based controller knows about the channel
# --------------------------------------------------------------------------- #
def _csi_snapshot(env, rng):
    """The channel as a model-based controller estimates it this slot.

    Same impairment model the learned agents' observation uses -- CSI
    ``csi_delay_slots`` old and each coefficient scaled by (1 + e),
    e ~ CN(0, csi_error_std^2) -- but with the controller's own generator, so
    estimation noise does not perturb the environment's arrival stream. Returns
    the per-group cascade C (K,G), the direct link hd (K,) and the noise-floor
    aggregates zeta (G,) and xi (K,G), all as the controller believes them.
    """
    ch = env.core.chan
    _, g, hr, hd = ch._observed_raw()              # delayed block if csi_delay > 0
    g, hr, hd = g[0], hr[0], hd[0]
    std = ch.csi_error_std
    if std > 0.0:
        def noisy(x):
            e = std * (rng.standard_normal(x.shape)
                       + 1j * rng.standard_normal(x.shape)) / np.sqrt(2.0)
            return x * (1.0 + e)
        g, hr, hd = noisy(g), noisy(hr), noisy(hd)
    K, G, M = env.K, env.G, env.M
    C = (hr * g[None, :]).reshape(K, G, M).sum(axis=2)
    zeta = (g.real ** 2 + g.imag ** 2).reshape(G, M).sum(axis=1)
    xi = (hr.real ** 2 + hr.imag ** 2).reshape(K, G, M).sum(axis=2)
    return C, hd, zeta, xi


def _est_link(env, csi, a_gains, theta_g):
    """Composite gain and amplifier-noise floors under the estimated channel."""
    C, hd, zeta, xi = csi
    h = hd + C @ (a_gains * np.exp(1j * theta_g))
    a2 = a_gains ** 2
    sig = env.core.sigma_ris_psd
    return h.real ** 2 + h.imag ** 2, sig * float(np.sum(a2 * zeta)), sig * (xi @ a2)


def _min_dl_power(env, pending, W, h2, dl_ris):
    """Smallest downlink power that returns every pending result this slot:
    W log2(1 + P h2 / n) >= pending / tau  <=>  P >= (2^(pending/(tau W)) - 1) n / h2.
    Capped at the BS limit; 5% margin against the estimate being optimistic."""
    n_dl = W * (env.n0 + dl_ris)
    spec = np.minimum(pending / (env.tau * W), 60.0)
    p_k = (np.exp2(spec) - 1.0) * n_dl / np.maximum(h2, 1e-30)
    return float(np.clip(1.05 * p_k.max(), 0.0, env.PBS_max))


def _pending_results(env):
    core = env.core
    return core.rqueues[0].copy() if core.result_queue else np.zeros(env.K)


def _uniform_ris_power(env, a):
    core = env.core
    return core.G * core.M * (core.pc + core.pa * a * a)


def _amp_levels(env, n):
    core = env.core
    return (np.linspace(0.0, core.a_feasible, int(n) + 1)[1:]
            if core.enable_amp else np.ones(1))


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
    """Iterative fractional programming for instantaneous EE (queue-agnostic).

    ``baselines.ifp_mode``:

    "full" (default)
        Quadratic-transform FP on the uplink powers, searched jointly over a
        uniform RIS amplitude, with the minimum downlink power that returns the
        pending results -- all on the estimated channel. The FP denominator is the
        slot's actual non-uplink power (downlink + RIS) rather than a constant.

    "legacy"
        The original: FP on uplink power only, downlink fixed at the BS maximum
        (5 W) and the RIS at its default amplitude. It spent ~7 W on things it
        did not optimize, which made it a weak EE baseline for reasons unrelated
        to fractional programming.
    """
    mode = str(cfg.get("baselines", {}).get("ifp_mode", "full"))
    if mode == "legacy":
        return _run_IFP_legacy(cfg, seed=seed, steps=steps, inner=inner)
    if mode != "full":
        raise ValueError(f"unknown baselines.ifp_mode '{mode}'")

    b = cfg.get("baselines", {})
    env = RISMECEnv(cfg, seed=seed); env.reset(); infos = []
    core = env.core
    rng = np.random.default_rng(seed + 992)          # CSI-estimation noise only
    K, rho, tau = env.K, core.rho_out, env.tau
    a_levels = _amp_levels(env, b.get("dpp_amp_levels", 8))
    beta = np.full(K, 1.0 / K)
    W = beta * env.B
    for _ in range(steps):
        env.alpha = np.full(K, 0.5)
        env.f = np.full(K, env.F_max / K)
        demand = env.alpha * env.queues
        rq = _pending_results(env)
        csi = _csi_snapshot(env, rng)
        theta_g = np.mod(-np.angle(csi[0].sum(axis=0)), 2 * np.pi)
        best = None
        for a in a_levels:
            av = np.full(env.G, a)
            h2, ul_ris, dl_ris = _est_link(env, csi, av, theta_g)
            p_ris = _uniform_ris_power(env, a)
            n = W * (env.n0 + ul_ris)
            p = np.full(K, env.Pk_max * 0.5)
            p_dl = _min_dl_power(env, rq + rho * demand, W, h2, dl_ris)
            for _ in range(inner):
                rate = W * np.log2(1 + p * h2 / np.maximum(n, 1e-30))
                y = np.sqrt(np.maximum(rate, 0)) / (np.sum(p) + p_dl + p_ris)
                grad_scale = (y ** 2) / np.maximum(h2 / n, 1e-30)
                p = np.clip(1.0 / np.maximum(grad_scale, 1e-9), 0, env.Pk_max)
            b_ul = np.minimum(demand, W * np.log2(1 + p * h2 / np.maximum(n, 1e-30)) * tau)
            p_dl = _min_dl_power(env, rq + rho * b_ul, W, h2, dl_ris)
            ee = b_ul.sum() / max(tau * (p.sum() + p_dl + p_ris), 1e-12)
            if best is None or ee > best[0]:
                best = (ee, a, p, p_dl)
        _, a, p, p_dl = best
        env.set_strategy(a_gains=np.full(env.G, a))
        act = _action(env, p, p_dl, beta, theta_g)
        _, _, done, _, info = env.step(act); infos.append(info)
        if done: env.reset()
    return summarize(infos, env.K, env.tau, env.arrival_mean)


def _run_IFP_legacy(cfg, seed=0, steps=1000, inner=6):
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
    """Lyapunov drift-plus-penalty controller.

    ``baselines.dpp_mode`` selects the implementation:

    "normalized" (default)
        Each slot, jointly over a grid of uniform RIS amplitudes, it picks per-UE
        uplink powers maximizing

            sum_k (Q_k / q_norm) * (b_k / lambda)  -  V * P_total / (1 W),

        i.e. normalized backlog reduction minus V times the slot's total power,
        where P_total = sum_k P_k^ul + P^dl + P_RIS. The downlink power is the
        *minimum* that returns every result owed this slot (closed form below),
        and max-weight sets bandwidth, offload ratio and CPU in proportion to
        backlog. Both halves are O(1) at the design point, so V is a genuine
        knob and sweeping it traces DPP's own delay--EE frontier.

    "legacy"
        The original controller, whose objective compared Q*b (~1e10) with
        V*energy (~1e-3): V was inert, the controller was pure max-weight
        scheduling at full downlink power (5 W) and the default RIS amplitude.
        That left it at one extreme of the trade-off with no way to move,
        which a reviewer could fairly call an under-tuned baseline.
    """
    mode = str(cfg.get("baselines", {}).get("dpp_mode", "normalized"))
    if mode == "legacy":
        return _run_DPP_legacy(cfg, seed=seed, steps=steps)
    if mode != "normalized":
        raise ValueError(f"unknown baselines.dpp_mode '{mode}'")

    b = cfg.get("baselines", {})
    V = float(b.get("dpp_V", 1.0))
    env = RISMECEnv(cfg, seed=seed); env.reset(); infos = []
    core = env.core
    rng = np.random.default_rng(seed + 991)          # CSI-estimation noise only
    K, rho, tau = env.K, core.rho_out, env.tau
    q_norm, lam = core.q_norm, core.arrival_mean
    grid = np.linspace(0.0, env.Pk_max, 9)[1:]                       # (P,)
    a_levels = _amp_levels(env, b.get("dpp_amp_levels", 8))
    idx_k = np.arange(K)
    for _ in range(steps):
        Q = env.queues.copy()
        beta = Q / max(Q.sum(), 1e-9)
        env.alpha = np.clip(Q / (Q + lam), 0.2, 1.0)       # drain long queues
        env.f = env.F_max * beta                           # CPU to backlog
        W = np.maximum(beta * env.B, 1e-9)
        demand = env.alpha * Q
        rq = _pending_results(env)
        # Everything below uses the ESTIMATED channel: with impaired CSI the
        # controller plans on a wrong model, exactly as a deployed one would.
        csi = _csi_snapshot(env, rng)
        theta_g = np.mod(-np.angle(csi[0].sum(axis=0)), 2 * np.pi)   # co-phasing
        best = None
        for a in a_levels:
            av = np.full(env.G, a)
            h2, ul_ris, dl_ris = _est_link(env, csi, av, theta_g)
            p_ris = _uniform_ris_power(env, a)
            n_ul = W * (env.n0 + ul_ris)
            rate = W[:, None] * np.log2(1 + grid[None, :] * h2[:, None] / n_ul[:, None])
            served = np.minimum(demand[:, None], rate * tau)                  # (K,P)
            gain = (Q[:, None] / q_norm) * (served / lam) - V * grid[None, :]
            j = np.argmax(gain, axis=1)
            p, b_ul = grid[j], served[idx_k, j]
            p_dl = _min_dl_power(env, rq + rho * b_ul, W, h2, dl_ris)
            obj = gain[idx_k, j].sum() - V * (p_dl + p_ris)
            if best is None or obj > best[0]:
                best = (obj, a, p, p_dl, theta_g)
        _, a, p, p_dl, theta_g = best
        env.set_strategy(a_gains=np.full(env.G, a))
        act = _action(env, p, p_dl, beta, theta_g)
        _, _, done, _, info = env.step(act); infos.append(info)
        if done: env.reset()
    return summarize(infos, env.K, env.tau, env.arrival_mean)


def _run_DPP_legacy(cfg, seed=0, steps=1000):
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
