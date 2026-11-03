"""
RIS-MEC environment with a *physically consistent, active* RIS.


  Gap A (active RIS was passive):
    * reflection coefficient is a_g * exp(j theta_g) with amplitude gain a_g >= 0
      (a_g>1 amplifies) -- NOT unit modulus;
    * the active-RIS amplifier noise enters the SINR denominator, with the uplink
      floor aggregated through g (term zeta_g) and the downlink floor through
      h_{r,k} (term xi_{g,k});
    * a per-element output-power (stability) constraint clips a_g given the
      incident power, and a global P_RIS budget is enforced;
    * RIS power consumption P_RIS = sum_g (N/G)(p_c + p_a a_g^2) enters the energy.

  Gap B (objective != P1): reward comes from objectives.RewardObjective, which
    optimizes normalized EE / latency / Lyapunov-DPP -- see objectives.py.

  Gap C (reproducibility/fidelity): reference-distance pathloss via channels.py,
    single seeded RNG, explicit queue update Q_k(t+1)=max(Q_k-b_k,0)+A_k,
    parameter table in configs/default.yaml.

Timescale decomposition:
  * strategic variables (alpha, f, a_g) are set by the strategist every H slots
    through `set_strategy(...)`;
  * tactical variables (P^ul, P^dl, beta, theta_g) are the per-slot action.

This class is a **single-world facade** over :class:`ris_mec.core.BatchedRISMEC`,
which holds the actual (vectorized) dynamics. It exists because a few things
genuinely want one world at a time: the model-based baselines, and
``experiments/exp_inference_time.py``, which measures the per-slot *decision*
latency that goes in the paper and must therefore not be batched.

For training and evaluation use the vector envs in :mod:`ris_mec.vec_env`, which
drive many worlds per numpy call and are one to two orders of magnitude faster.
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .channels import ChannelView, quantize_phase, dbm_to_watt, db_to_lin
from .core import BatchedRISMEC
from .objectives import RewardObjective, StepStats

# Evaluation is one continuing run: episode boundaries reset the clock but carry
# the backlogs over (see BatchedRISMEC.reset), so the served fraction is exact.
CARRY = {"keep_queues": True}

# Attributes that are simply forwarded from the batched core.
_FORWARD = (
    "K", "N", "G", "M", "tau", "max_steps", "B", "n0", "sigma_ris_psd",
    "enable_amp", "a_init", "Pk_max", "PBS_max", "pc", "pa", "P_ris_budget",
    "a_max", "P_out_max", "F_max", "omega", "rho_out", "arrival_mean", "q_norm",
    "phase_bits", "base_seed", "rng", "cfg", "reward_obj", "t",
)
# Batched state exposed to callers as un-batched (K,)/(G,) arrays.
_ROW_STATE = ("queues", "alpha", "f", "a_gains", "last_h2")


class RISMECEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: dict, seed: int | None = None):
        super().__init__()
        self.core = BatchedRISMEC(cfg, n_envs=1, seed=seed)
        self.chan = ChannelView(self.core.chan, 0)
        self._obs_dim = self.core.obs_dim
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self._obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.core.act_dim,), dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  Transparent access to the core's parameters and state
    # ------------------------------------------------------------------ #
    def __getattr__(self, name):
        if name in _FORWARD:
            return getattr(self.core, name)
        if name in _ROW_STATE:
            return getattr(self.core, name)[0]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in _ROW_STATE:
            getattr(self.core, name)[0] = value
            if name == "a_gains":
                self.core._strategy_version += 1
            return
        if name == "t":
            self.core.t = value
            return
        super().__setattr__(name, value)

    # ================================================================== #
    #  Strategic interface (called by the strategist every H slots)
    # ================================================================== #
    def set_strategy(self, alpha=None, f=None, a_gains=None):
        self.core.set_strategy(alpha=alpha, f=f, a_gains=a_gains)

    # ================================================================== #
    #  Physics helpers kept at their original (un-batched) signatures,
    #  because the model-based baselines and the smoke test call them.
    # ================================================================== #
    def _noise_floors(self, a_gains, g, hr):
        """Uplink (at BS, scalar) and downlink (per UE, (K,)) amplifier-noise
        floors [W/Hz] for one world."""
        G, M = self.core.G, self.core.M
        a2 = np.asarray(a_gains, float) ** 2
        zeta_g = (np.abs(g) ** 2).reshape(G, M).sum(axis=1)                      # (G,)
        xi_gk = (np.abs(hr) ** 2).reshape(-1, G, M).sum(axis=2)                  # (K,G)
        ul = self.core.sigma_ris_psd * float(np.sum(a2 * zeta_g))
        dl = self.core.sigma_ris_psd * (a2[None, :] * xi_gk).sum(axis=1)         # (K,)
        return ul, dl

    def _composite_channel(self, a_elem, theta_elem, g, hr, hd):
        reflect = a_elem * np.exp(1j * theta_elem)                  # (N,)
        cascade = np.einsum('kn,n,n->k', hr, reflect, g)            # (K,)
        h = hd + cascade
        return np.abs(h) ** 2                                       # (K,)

    def _apply_output_power_constraint(self, a_gains, hr):
        """Un-batched wrapper kept for API compatibility; the fast path used by
        :meth:`step` is :meth:`ris_mec.core.BatchedRISMEC._apply_output_power_constraint`."""
        core = self.core
        inc = core.Pk_max * np.sum(np.abs(hr) ** 2, axis=0)          # (N,)
        inc_g = inc.reshape(core.G, core.M).mean(axis=1)             # (G,)
        band_noise = core.sigma_ris_psd * core.B
        amax_stab = np.sqrt(core.P_out_max / np.maximum(inc_g + band_noise, 1e-30))
        a = np.minimum(np.asarray(a_gains, float), np.minimum(amax_stab, core.a_max))
        static = core.M * core.pc * core.G
        avail = max(core.P_ris_budget - static, 0.0)
        scale = np.sqrt(avail / max(np.sum(core.M * core.pa * a ** 2), 1e-30))
        a = a * min(scale, 1.0)
        return a, float(np.sum(core.M * (core.pc + core.pa * a ** 2)))

    # ================================================================== #
    def reset(self, *, seed=None, options=None):
        """``options={"keep_queues": True}`` carries the backlogs over (evaluation)."""
        keep = bool((options or {}).get("keep_queues", False))
        obs = self.core.reset(seed=seed, keep_queues=keep)
        if seed is not None:
            self.chan = ChannelView(self.core.chan, 0)
        return obs[0], {}

    def step(self, action):
        obs, reward, done, info = self.core.step(np.asarray(action, float)[None, :])
        return (obs[0], float(reward[0]), bool(done), False,
                {k: float(v[0]) for k, v in info.items()})

    def _obs(self):
        return self.core.obs()[0]
