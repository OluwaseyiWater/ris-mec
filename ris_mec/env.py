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

  Gap C (reproducibility/fidelity): 3GPP UMi pathloss via channels.py, single
    seeded RNG, explicit queue update Q_k(t+1)=max(Q_k-b_k,0)+A_k, parameter
    table in configs/default.yaml.

Timescale decomposition:
  * strategic variables (alpha, f, a_g) are set by the strategist every H slots
    through `set_strategy(...)`;
  * tactical variables (P^ul, P^dl, beta, theta_g) are the per-slot action.

Everything is vectorized over users (K) and elements/groups (N, G).
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .channels import ChannelModel, quantize_phase, dbm_to_watt, db_to_lin
from .objectives import RewardObjective, StepStats


class RISMECEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: dict, seed: int | None = None):
        super().__init__()
        self.cfg = cfg
        s = cfg["system"]
        self.K = int(s["K"]); self.N = int(s["N"]); self.G = int(s["G"])
        assert self.N % self.G == 0, "N must be divisible by G"
        self.M = self.N // self.G                     # elements per group
        self.tau = float(s["tau"]); self.max_steps = int(s["max_steps"])

        # Bandwidth & noise PSDs (W/Hz)
        b = cfg["bandwidth"]
        self.B = float(b["B_total"])
        nf = float(b["noise_figure_db"])
        self.n0 = dbm_to_watt(float(b["noise_psd_dbm_hz"]) + nf)          # thermal PSD
        self.sigma_ris_psd = dbm_to_watt(float(cfg["active_ris"]["sigma_ris_psd_dbm_hz"]))
        self.enable_amp = bool(cfg["active_ris"].get("enable_amplification", True))
        self.a_init = float(cfg["active_ris"].get("a_init", 1.0))

        # Power model
        p = cfg["power"]
        self.Pk_max = float(p["Pk_ul_max"]); self.PBS_max = float(p["Pbs_dl_max"])
        self.pc = float(p["ris_static_pc"]); self.pa = float(p["ris_amp_pa"])
        self.P_ris_budget = float(p["ris_power_budget"])
        self.a_max = float(p["ris_amp_max"])
        self.P_out_max = float(p["ris_out_power_max"])

        # MEC
        m = cfg["mec"]
        self.F_max = float(m["F_max"]); self.omega = float(m["cycles_per_bit"])
        self.rho_out = float(m["rho_out"])

        # Queue
        q = cfg["queue"]
        self.arrival_mean = float(q["arrival_mean_bits"])
        self.q_norm = float(q["q_norm"])

        # Phase quantization (robustness)
        self.phase_bits = int(cfg["channel"].get("phase_bits", 0))

        # RNG + channels
        self.base_seed = int(cfg["training"].get("seed", 0)) if seed is None else int(seed)
        self.rng = np.random.default_rng(self.base_seed)
        self.chan = ChannelModel(cfg, self.rng)

        # Reward objective
        self.reward_obj = RewardObjective(cfg)

        # Strategic variables (defaults; strategist overwrites via set_strategy)
        self.alpha = np.full(self.K, 0.5)
        self.f = np.full(self.K, self.F_max / self.K)
        self.a_gains = np.full(self.G, self.a_init if self.enable_amp else 1.0)

        # Spaces. Observation: per-UE channel gains (uplink composite proxy),
        # queues, alpha, f, plus normalized time.  Action (tactical):
        # [ul_p(K), dl_p(1), beta_logits(K), theta_g(G)].
        self._obs_dim = 4 * self.K + 1
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self._obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.K + 1 + self.K + self.G,),
                                       dtype=np.float32)

        self.t = 0
        self.queues = self.rng.exponential(self.arrival_mean, size=self.K)
        self.last_h2 = np.ones(self.K)

    # ================================================================== #
    #  Strategic interface (called by the strategist every H slots)
    # ================================================================== #
    def set_strategy(self, alpha=None, f=None, a_gains=None):
        if alpha is not None:
            self.alpha = np.clip(np.asarray(alpha, float), 0.0, 1.0)
        if f is not None:
            self.f = np.clip(np.asarray(f, float), 0.0, self.F_max)
        if a_gains is not None:
            a = np.clip(np.asarray(a_gains, float), 0.0, self.a_max)
            self.a_gains = a if self.enable_amp else np.ones(self.G)

    # ================================================================== #
    #  Physics helpers (vectorized)
    # ================================================================== #
    def _apply_output_power_constraint(self, a_gains, hr):
        """Clip per-group amplitude so per-element output power <= P_out_max (R2.1),
        then rescale to respect the global RIS power budget P_RIS <= P_RIS_max."""
        # Incident signal power per element ~ sum_k P_k^ul |hr_{k,n}|^2 ; use max power
        # as a worst-case envelope for the actuator constraint, aggregated per group.
        inc = self.Pk_max * np.sum(np.abs(hr) ** 2, axis=0)          # (N,)
        inc_g = inc.reshape(self.G, self.M).mean(axis=1)            # (G,)
        band_noise = self.sigma_ris_psd * self.B
        amax_stab = np.sqrt(self.P_out_max / np.maximum(inc_g + band_noise, 1e-30))
        a = np.minimum(a_gains, np.minimum(amax_stab, self.a_max))
        # Global RIS power budget
        p_ris = np.sum(self.M * (self.pc + self.pa * a ** 2))
        if p_ris > self.P_ris_budget:
            # scale the dynamic part down uniformly
            static = self.M * self.pc * self.G
            avail = max(self.P_ris_budget - static, 0.0)
            scale = np.sqrt(avail / max(np.sum(self.M * self.pa * a ** 2), 1e-30))
            a = a * min(scale, 1.0)
        p_ris = float(np.sum(self.M * (self.pc + self.pa * a ** 2)))
        return a, p_ris

    def _composite_channel(self, a_elem, theta_elem, g, hr, hd):
        reflect = a_elem * np.exp(1j * theta_elem)                  # (N,)
        cascade = np.einsum('kn,n,n->k', hr, reflect, g)           # (K,)
        h = hd + cascade
        return np.abs(h) ** 2                                       # (K,)

    def _noise_floors(self, a_gains, g, hr):
        """Uplink (at BS) and downlink (per UE) amplifier-noise floors [W/Hz]."""
        # zeta_g = sum_{n in g} |g_n|^2 ; xi_{g,k} = sum_{n in g} |hr_{k,n}|^2
        zeta_g = (np.abs(g) ** 2).reshape(self.G, self.M).sum(axis=1)         # (G,)
        xi_gk = (np.abs(hr) ** 2).reshape(self.K, self.G, self.M).sum(axis=2)  # (K,G)
        a2 = a_gains ** 2
        ul_ris_psd = self.sigma_ris_psd * np.sum(a2 * zeta_g)                 # scalar
        dl_ris_psd = self.sigma_ris_psd * np.sum(a2[None, :] * xi_gk, axis=1)  # (K,)
        return ul_ris_psd, dl_ris_psd

    # ================================================================== #
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.chan = ChannelModel(self.cfg, self.rng)
        self.t = 0
        self.queues = self.rng.exponential(self.arrival_mean, size=self.K)
        self.alpha = np.full(self.K, 0.5)
        self.f = np.full(self.K, self.F_max / self.K)
        self.a_gains = np.full(self.G, self.a_init if self.enable_amp else 1.0)
        self.reward_obj.reset()
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        K, G = self.K, self.G

        # --- decode tactical action ---
        ul_p = (a[:K] + 1) / 2 * self.Pk_max
        dl_p = (a[K] + 1) / 2 * self.PBS_max
        beta_logits = a[K + 1:K + 1 + K]
        beta = np.exp(beta_logits - beta_logits.max()); beta /= beta.sum()
        theta_g = (a[-G:] + 1) / 2 * 2 * np.pi
        theta_g = quantize_phase(theta_g, self.phase_bits)

        # --- channels (agent may see noisy/delayed CSI, physics uses truth) ---
        g, hr, hd = self.chan.true_channels()

        # --- active-RIS amplitude with stability + budget constraints ---
        a_gains, p_ris = self._apply_output_power_constraint(self.a_gains, hr)
        a_elem = np.repeat(a_gains, self.M)
        theta_elem = np.repeat(theta_g, self.M)

        # --- composite gains and amplifier-noise floors ---
        h2 = self._composite_channel(a_elem, theta_elem, g, hr, hd)          # (K,)
        self.last_h2 = h2
        ul_ris_psd, dl_ris_psd = self._noise_floors(a_gains, g, hr)

        # --- rates (bandwidth-consistent noise) ---
        W = beta * self.B                                                   # (K,)
        n_ul = W * (self.n0 + ul_ris_psd)                                   # (K,)
        n_dl = W * (self.n0 + dl_ris_psd)                                   # (K,)
        R_ul = W * np.log2(1 + ul_p * h2 / np.maximum(n_ul, 1e-30))
        R_dl = W * np.log2(1 + dl_p * h2 / np.maximum(n_dl, 1e-30))

        # --- offloading, latency (bits limited by rate over the slot) ---
        demand_ul = self.alpha * self.queues
        bits_ul = np.minimum(demand_ul, R_ul * self.tau)
        bits_dl = np.minimum(self.rho_out * self.alpha * self.queues, R_dl * self.tau)
        t_ul = bits_ul / np.maximum(R_ul, 1e-9)
        t_dl = bits_dl / np.maximum(R_dl, 1e-9)
        t_cmp = (bits_ul * self.omega) / np.maximum(self.f, 1e-9)
        latency = float(np.mean(t_ul + t_dl + t_cmp))

        # --- energy: uplink + downlink transmit + RIS power over the slot ---
        energy = self.tau * (float(np.sum(ul_p)) + float(dl_p) + p_ris)
        total_bits = float(np.sum(bits_ul))
        ee = total_bits / max(energy, 1e-12)

        # --- queue update: Q_k(t+1) = max(Q_k - b_k, 0) + A_k ---
        served = bits_ul.copy()
        self.queues = np.clip(self.queues - bits_ul, 0.0, None)
        arrivals = self.rng.exponential(self.arrival_mean, size=self.K)
        self.queues = self.queues + arrivals

        # --- reward via objective (fixes Gap B) ---
        stats = StepStats(bits=total_bits, energy=energy, latency=latency, ee=ee,
                          queues=self.queues.copy(), arrivals=arrivals, served=served)
        reward = self.reward_obj(stats)

        # --- advance channels & time ---
        self.chan.sample()
        self.t += 1
        done = self.t >= self.max_steps
        info = {
            "latency": latency, "energy": energy, "bits": total_bits, "ee": ee,
            "p_ris": p_ris, "queue_mean": float(np.mean(self.queues)),
            "a_mean": float(np.mean(a_gains)),
        }
        return self._obs(), reward, done, False, info

    def _obs(self):
        g, hr, hd = self.chan.observed_channels()
        # normalized composite-ish gains from observed CSI (agent view)
        gain_ue = np.abs(hd) ** 2 + np.mean(np.abs(hr) ** 2, axis=1) * np.mean(np.abs(g) ** 2)
        gain_ue = gain_ue / (1.0 + gain_ue)
        obs = np.concatenate([
            gain_ue,
            np.clip(self.queues / self.q_norm, 0, 1),
            self.alpha,
            self.f / self.F_max,
            [self.t / self.max_steps],
        ]).astype(np.float32)
        if obs.size != self._obs_dim:
            obs = np.resize(obs, self._obs_dim).astype(np.float32)
        return obs
