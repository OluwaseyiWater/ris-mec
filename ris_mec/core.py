"""
Batched RIS-MEC physics: ``n_envs`` independent worlds advanced in lockstep by a
single set of numpy calls.

This is the one and only implementation of the environment dynamics --
:class:`ris_mec.env.RISMECEnv` is a one-world facade over it, and the SB3
vector envs in :mod:`ris_mec.vec_env` drive it with many worlds at once.  Keeping
a single source of truth is what makes the equivalence test in
``tests/test_equivalence.py`` meaningful.

Shapes (E = n_envs, K = users, G = RIS groups, N = RIS elements, M = N/G):

    queues, alpha, f      (E, K)
    a_gains, theta_g      (E, G)
    obs                   (E, 4K+1)
    reward, latency, ...  (E,)

Why batching is the win here
----------------------------
A single slot costs ~300 us regardless of K, because it is ~80 tiny numpy calls
whose cost is Python/dispatch overhead rather than arithmetic. Adding a leading
world axis amortizes that fixed cost over E worlds and turns the policy forward
pass from E single-sample calls into one batched call.

Per-slot work that does *not* depend on the action is hoisted:
  * every reduction over the element axis N is cached per fading block
    (see :meth:`ChannelModel._derive`);
  * the active-RIS amplitude clipping depends only on (a_gains, channel), both of
    which change on slow timescales, so its result is memoized;
  * the cascade is evaluated per *group* (K x G) rather than per element (K x N).
"""

from __future__ import annotations
import numpy as np

from .channels import ChannelModel, quantize_phase, dbm_to_watt, db_to_lin
from .objectives import RewardObjective, StepStats

TWO_PI = 2.0 * np.pi


class BatchedRISMEC:
    """Vectorized RIS-MEC dynamics over ``n_envs`` worlds.

    Timescale decomposition:
      * strategic variables (alpha, f, a_g) are set by the strategist every H
        slots through :meth:`set_strategy`;
      * tactical variables (P^ul, P^dl, beta, theta_g) are the per-slot action.
    """

    def __init__(self, cfg: dict, n_envs: int = 1, seed: int | None = None):
        self.cfg = cfg
        self.n_envs = E = int(n_envs)
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
        # MEC computation energy: kappa * f^2 per executed cycle (standard CMOS
        # dynamic-power model), i.e. E_comp = sum_k kappa * f_k^2 * omega * b_k.
        # 0.0 reproduces the transmit-only energy metric.
        self.kappa_c = float(m.get("compute_energy_kappa", 0.0))

        # Queue
        q = cfg["queue"]
        self.arrival_mean = float(q["arrival_mean_bits"])
        self.q_norm = float(q["q_norm"])

        # Phase quantization (robustness)
        self.phase_bits = int(cfg["channel"].get("phase_bits", 0))

        # RNG + channels
        self.base_seed = int(cfg["training"].get("seed", 0)) if seed is None else int(seed)
        self.rng = np.random.default_rng(self.base_seed)
        self.chan = ChannelModel(cfg, self.rng, n_envs=E)

        # Reward objective (batched)
        self.reward_obj = RewardObjective(cfg, n_envs=E)

        # Strategic variables (defaults; strategist overwrites via set_strategy)
        self.alpha = np.full((E, self.K), 0.5)
        self.f = np.full((E, self.K), self.F_max / self.K)
        self.a_gains = np.full((E, self.G), self.a_init if self.enable_amp else 1.0)

        # Range the strategist's amplitude action is mapped onto. Using a_max
        # is a trap: the global RIS power budget clips any uniform amplitude
        # above a_feasible, so most of [0, a_max] collapses onto one saturated
        # outcome in which the reward is flat in a_g -- PPO initializes inside
        # that dead zone and never sees a gradient. Mapping onto the feasible
        # range makes the whole interval do something. 'raw' restores the old
        # behaviour.
        self.amp_action_scale = str(
            cfg.get('hierarchy', {}).get('amp_action_scale', 'feasible'))
        self.a_scale = (self.a_feasible if self.amp_action_scale == 'feasible'
                        else self.a_max)

        self.use_group_obs = self.chan.obs_group_features
        self.obs_dim = 4 * self.K + 1 + (3 * self.G if self.use_group_obs else 0)
        self.act_dim = 2 * self.K + 1 + self.G

        self.t = 0
        self.queues = self.rng.exponential(self.arrival_mean, size=(E, self.K))
        self.last_h2 = np.ones((E, self.K))

        # Memo for the active-RIS amplitude constraint, invalidated whenever the
        # strategy or the fading block changes.  Version counters rather than
        # array identity, so a recycled ``id()`` can never alias a stale entry.
        self._amp_memo = None
        self._amp_key = None
        self._strategy_version = 0

    # ================================================================== #
    #  Strategic interface (called by the strategist every H slots)
    # ================================================================== #
    def set_strategy(self, alpha=None, f=None, a_gains=None):
        """Set strategic variables. Accepts (E,K)/(E,G) or broadcastable arrays."""
        E = self.n_envs
        if alpha is not None:
            self.alpha = np.clip(np.asarray(alpha, float), 0.0, 1.0).reshape(E, self.K)
        if f is not None:
            self.f = np.clip(np.asarray(f, float), 0.0, self.F_max).reshape(E, self.K)
        if a_gains is not None:
            a = np.clip(np.asarray(a_gains, float), 0.0, self.a_max).reshape(E, self.G)
            self.a_gains = a if self.enable_amp else np.ones((E, self.G))
            self._strategy_version += 1

    @property
    def a_feasible(self):
        """Largest uniform group amplitude the global RIS power budget admits.

        From P_RIS = sum_g M (p_c + p_a a_g^2) <= P_RIS_max with a_g = a:
        a = sqrt((P_RIS_max - G M p_c) / (G M p_a)), capped by a_max.
        """
        avail = max(self.P_ris_budget - self.M * self.pc * self.G, 0.0)
        return float(min(self.a_max,
                         np.sqrt(avail / max(self.G * self.M * self.pa, 1e-30))))

    @property
    def meta_act_dim(self):
        return 2 * self.K + self.G      # alpha(K), f_logits(K), a_ctrl(G)

    def apply_meta(self, a, enable_ris: bool = True):
        """Map a meta-action in [-1,1] to strategic variables.

        :param a: (E, 2K+G) or a single (2K+G,) vector.
        """
        E, K, G = self.n_envs, self.K, self.G
        a = np.clip(np.asarray(a, float).reshape(E, self.meta_act_dim), -1.0, 1.0)
        alpha = (a[:, :K] + 1) / 2.0
        f_logits = a[:, K:2 * K]
        fw = np.exp(f_logits - f_logits.max(axis=1, keepdims=True))
        fw /= fw.sum(axis=1, keepdims=True)
        f = fw * self.F_max
        if enable_ris:
            a_ctrl = (a[:, 2 * K:2 * K + G] + 1) / 2.0 * self.a_scale  # [0, a_scale]
        else:
            a_ctrl = np.ones((E, G))                                  # passive ablation
        self.set_strategy(alpha=alpha, f=f, a_gains=a_ctrl)

    # ================================================================== #
    #  Physics helpers (vectorized over worlds)
    # ================================================================== #
    def _apply_output_power_constraint(self):
        """Clip per-group amplitude so per-element output power <= P_out_max,
        then rescale to respect the global RIS power budget P_RIS <= P_RIS_max.

        Depends only on (a_gains, fading block), so the result is memoized: with
        the default H=50 and coherence=10 this recomputes once every 10 slots
        instead of every slot.
        """
        key = (self._strategy_version, self.chan.version)
        if self._amp_key == key and self._amp_memo is not None:
            return self._amp_memo

        # Incident signal power per element ~ sum_k P_k^ul |hr_{k,n}|^2 ; use max
        # power as a worst-case envelope for the actuator constraint, per group.
        inc_g = self.Pk_max * self.chan.incident_group()                 # (E,G)
        band_noise = self.sigma_ris_psd * self.B
        amax_stab = np.sqrt(self.P_out_max / np.maximum(inc_g + band_noise, 1e-30))
        a = np.minimum(self.a_gains, np.minimum(amax_stab, self.a_max))  # (E,G)

        # Global RIS power budget. When the budget is already met the scale below
        # is >= 1, so the unconditional min(scale, 1) is exactly the old branch.
        dyn = self.M * self.pa * a ** 2
        static = self.M * self.pc * self.G
        avail = max(self.P_ris_budget - static, 0.0)
        scale = np.sqrt(avail / np.maximum(dyn.sum(axis=1), 1e-30))       # (E,)
        a = a * np.minimum(scale, 1.0)[:, None]
        p_ris = (self.M * (self.pc + self.pa * a ** 2)).sum(axis=1)       # (E,)

        self._amp_key, self._amp_memo = key, (a, p_ris)
        return a, p_ris

    def _composite_gain(self, a_gains, theta_g):
        """|h_k|^2 for the composite direct + RIS-reflected channel.        (E,K)

        Uses the per-group cascade coefficients, which is algebraically identical
        to the per-element einsum but O(K*G) instead of O(K*N).
        """
        reflect = a_gains * np.exp(1j * theta_g)                          # (E,G)
        C = self.chan.cascade_coeff()                                     # (E,K,G)
        cascade = np.einsum('ekg,eg->ek', C, reflect)                     # (E,K)
        h = self.chan.hd + cascade
        return h.real ** 2 + h.imag ** 2

    def _noise_floors(self, a_gains):
        """Uplink (at BS) and downlink (per UE) amplifier-noise floors [W/Hz]."""
        a2 = a_gains ** 2                                                 # (E,G)
        ul = self.sigma_ris_psd * np.einsum('eg,eg->e', a2, self.chan.zeta_g())
        dl = self.sigma_ris_psd * np.einsum('eg,ekg->ek', a2, self.chan.xi_gk())
        return ul, dl

    # ================================================================== #
    def reset(self, seed=None):
        E = self.n_envs
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.chan = ChannelModel(self.cfg, self.rng, n_envs=E)
        self.t = 0
        self.queues = self.rng.exponential(self.arrival_mean, size=(E, self.K))
        self.alpha = np.full((E, self.K), 0.5)
        self.f = np.full((E, self.K), self.F_max / self.K)
        self.a_gains = np.full((E, self.G), self.a_init if self.enable_amp else 1.0)
        self._strategy_version += 1
        self.reward_obj.reset()
        return self.obs()

    def step(self, actions):
        """Advance every world one slot.

        :param actions: (E, act_dim) in [-1, 1]
        :return: obs (E,obs_dim), reward (E,), done (bool, worlds are in lockstep),
                 info dict of (E,) arrays
        """
        a = np.clip(np.asarray(actions, float).reshape(self.n_envs, self.act_dim), -1.0, 1.0)
        K, G = self.K, self.G

        # --- decode tactical action ---
        ul_p = (a[:, :K] + 1) * (0.5 * self.Pk_max)                       # (E,K)
        dl_p = (a[:, K] + 1) * (0.5 * self.PBS_max)                       # (E,)
        beta_logits = a[:, K + 1:2 * K + 1]
        beta = np.exp(beta_logits - beta_logits.max(axis=1, keepdims=True))
        beta /= beta.sum(axis=1, keepdims=True)                           # (E,K)
        theta_g = (a[:, -G:] + 1) * np.pi                                 # (E,G)
        theta_g = quantize_phase(theta_g, self.phase_bits)

        # --- active-RIS amplitude with stability + budget constraints ---
        a_gains, p_ris = self._apply_output_power_constraint()

        # --- composite gains and amplifier-noise floors ---
        h2 = self._composite_gain(a_gains, theta_g)                       # (E,K)
        self.last_h2 = h2
        ul_ris_psd, dl_ris_psd = self._noise_floors(a_gains)

        # --- rates (bandwidth-consistent noise) ---
        W = beta * self.B                                                 # (E,K)
        n_ul = W * (self.n0 + ul_ris_psd[:, None])
        n_dl = W * (self.n0 + dl_ris_psd)
        R_ul = W * np.log2(1 + ul_p * h2 / np.maximum(n_ul, 1e-30))
        R_dl = W * np.log2(1 + dl_p[:, None] * h2 / np.maximum(n_dl, 1e-30))

        # --- offloading, latency (bits limited by rate over the slot) ---
        demand_ul = self.alpha * self.queues
        bits_ul = np.minimum(demand_ul, R_ul * self.tau)
        bits_dl = np.minimum(self.rho_out * self.alpha * self.queues, R_dl * self.tau)
        t_ul = bits_ul / np.maximum(R_ul, 1e-9)
        t_dl = bits_dl / np.maximum(R_dl, 1e-9)
        t_cmp = (bits_ul * self.omega) / np.maximum(self.f, 1e-9)
        latency = (t_ul + t_dl + t_cmp).mean(axis=1)                      # (E,)

        # --- energy: uplink + downlink transmit + RIS power over the slot ---
        energy = self.tau * (ul_p.sum(axis=1) + dl_p + p_ris)             # (E,)
        if self.kappa_c > 0.0:
            # Charging the cycles actually executed makes f_k a genuine
            # energy/latency decision rather than a free variable.
            energy = energy + self.kappa_c * self.omega * (self.f ** 2 * bits_ul).sum(axis=1)
        total_bits = bits_ul.sum(axis=1)                                  # (E,)
        ee = total_bits / np.maximum(energy, 1e-12)

        # --- queue update: Q_k(t+1) = max(Q_k - b_k, 0) + A_k ---
        served = bits_ul
        arrivals = self.rng.exponential(self.arrival_mean, size=(self.n_envs, K))
        self.queues = np.maximum(self.queues - bits_ul, 0.0) + arrivals

        # --- reward via objective ---
        stats = StepStats(bits=total_bits, energy=energy, latency=latency, ee=ee,
                          queues=self.queues, arrivals=arrivals, served=served)
        reward = self.reward_obj(stats)                                   # (E,)

        # --- advance channels & time ---
        self.chan.sample()
        self.t += 1
        done = self.t >= self.max_steps
        info = {
            "latency": latency, "energy": energy, "bits": total_bits, "ee": ee,
            "p_ris": p_ris, "queue_mean": self.queues.mean(axis=1),
            "a_mean": a_gains.mean(axis=1),
        }
        return self.obs(), reward, done, info

    def obs(self):
        """Agent observation: (E, 4K+1) float32 in [0,1].

        Layout: gain(K), queue(K), alpha(K), f(K), t(1), and -- when
        ``channel.obs_group_features`` is on -- the per-group cascade block
        (3G: relative magnitude, cos and sin of the phase) that the theta_g
        decision needs.

        The channel block is normalized by ``ChannelModel.observation_feature``
        (``channel.obs_gain_norm``): in dB by default, so it occupies the same
        [0,1] range as the queue/alpha/f blocks rather than the ~1e-11 the old
        ``gain/(1+gain)`` produced.
        """
        E = self.n_envs
        out = np.empty((E, self.obs_dim), dtype=np.float32)
        K = self.K
        out[:, 0:K] = self.chan.observation_feature()                     # (E,K)
        np.clip(self.queues / self.q_norm, 0, 1, out=out[:, K:2 * K])
        out[:, 2 * K:3 * K] = self.alpha
        out[:, 3 * K:4 * K] = self.f / self.F_max
        out[:, 4 * K] = self.t / self.max_steps
        if self.use_group_obs:
            out[:, 4 * K + 1:] = self.chan.group_feature()                # (E,3G)
        return out

    # ================================================================== #
    #  Meta (strategist) observation, shared by MetaManager and FlatEnv
    # ================================================================== #
    @property
    def meta_obs_dim(self):
        return 3 * self.K + 2

    def meta_obs(self):
        """Strategist observation: (E, 3K+2) float32."""
        E, K = self.n_envs, self.K
        out = np.empty((E, 3 * K + 2), dtype=np.float32)
        np.clip(self.queues / self.q_norm, 0, 1, out=out[:, 0:K])
        out[:, K:2 * K] = self.alpha
        out[:, 2 * K:3 * K] = self.f / self.F_max
        out[:, 3 * K] = self.f.sum(axis=1) / self.F_max
        out[:, 3 * K + 1] = self.t / self.max_steps
        return out
