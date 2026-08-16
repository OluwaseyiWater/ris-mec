"""
Channel model for the RIS-MEC system.
  * reference-distance distance-dependent pathloss,
  * large-scale geometry (BS, RIS, UEs) so pathloss is meaningful,
  * Rician small-scale fading on the RIS links (dominant LoS component),
  * a *partial* direct-link blockage factor rho_bar in [0,1] (fixes the A1: rho_bar=0
  truly zeroes h_d, rho_bar>0 keeps a weak path),
  * optional CSI estimation error and RIS phase quantization for robustness
    sweeps.

All randomness flows through a single numpy Generator for reproducibility.

Vectorization
-------------
Every array carries a leading *world* axis of length ``n_envs`` so that a whole
batch of independent simulation worlds advances in one set of numpy calls:

    g   (E, N)        RIS -> BS
    hr  (E, K, N)     UE  -> RIS
    hd  (E, K)        UE  -> BS (direct, blocked)

Because the channel only changes once every ``coherence_slots`` slots while the
env steps every slot, every reduction over the element axis N is computed once
per fading block and cached (see ``_derive``).  The important one is

    C[e,k,gp] = sum_{n in group gp} hr[e,k,n] * g[e,n]                    (E,K,G)

which turns the per-slot cascade from an O(K*N) einsum into an O(K*G) matvec:
since the reflection coefficient a_gp*exp(j*theta_gp) is constant within a
group, sum_n hr*refl*g == sum_gp refl_gp * C[:,:,gp] exactly.
"""

from __future__ import annotations
import numpy as np


def dbm_to_watt(dbm: float) -> float:
    return 10.0 ** ((dbm - 30.0) / 10.0)


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 10.0)


def pathloss_db(d_m: np.ndarray, pl0_db: float, exponent: float, d0: float = 1.0) -> np.ndarray:
    """Reference-distance large-scale pathloss [dB]:  PL = PL0 + 10*n*log10(d/d0).

    This is the standard model used in the RIS literature (e.g. Wu & Zhang), where
    RIS links are deployed for near-LoS propagation (small exponent ~2.2) and the
    blocked direct link suffers a large NLOS exponent (~3.5). Using it -- rather
    than applying 3GPP UMi-NLOS to *both* RIS hops, which double-counts ~170 dB and
    makes the RIS pointless -- is what keeps the reflected path viable.
    """
    d_m = np.maximum(d_m, d0)
    return pl0_db + 10.0 * exponent * np.log10(d_m / d0)


class ChannelModel:
    """Block-fading channels for ``n_envs`` independent RIS-MEC worlds.

    Geometry (2-D, meters): BS at origin-ish, RIS at fixed offset, UEs drawn in
    an annulus around the RIS. The cascaded UE->RIS->BS pathloss is the product
    of the two large-scale gains; small-scale fading is Rician on the RIS links
    and Rayleigh on the (weak) direct link.

    Each world gets its own UE placement, so a batch averages over topologies as
    well as over fading.  With ``n_envs=1`` the generator draw order is identical
    to the original scalar implementation.
    """

    def __init__(self, cfg: dict, rng: np.random.Generator, n_envs: int = 1):
        self.cfg = cfg
        self.rng = rng
        self.n_envs = int(n_envs)
        sys_, geo, ch = cfg["system"], cfg["geometry"], cfg["channel"]
        self.K = int(sys_["K"])
        self.N = int(sys_["N"])
        self.G = int(sys_["G"])
        self.M = self.N // self.G
        self.fc = float(geo["fc_ghz"])
        self.rho_bar = float(ch["direct_blockage"])
        self.coherence = int(ch["coherence_slots"])
        self.csi_error_std = float(ch.get("csi_error_std", 0.0))
        self.csi_delay = int(ch.get("csi_delay_slots", 0))

        # How the composite-gain observation feature is normalized. See
        # observation_feature() for why "legacy" is numerically inert.
        self.obs_norm = str(ch.get("obs_gain_norm", "db"))
        if self.obs_norm not in ("db", "legacy"):
            raise ValueError(f"unknown channel.obs_gain_norm '{self.obs_norm}'")
        self.obs_db_lo = float(ch.get("obs_gain_db_lo", -125.0))
        self.obs_db_hi = float(ch.get("obs_gain_db_hi", -75.0))
        if self.obs_db_hi <= self.obs_db_lo:
            raise ValueError("channel.obs_gain_db_hi must exceed obs_gain_db_lo")
        self._obs_db_scale = 1.0 / (self.obs_db_hi - self.obs_db_lo)
        # Per-group cascade features, without which theta_g is open-loop.
        self.obs_group_features = bool(ch.get("obs_group_features", True))

        # --- Pathloss parameters (reference-distance model) ---
        self.pl0 = float(geo.get("pl0_db", 30.0))
        self.n_ris = float(geo.get("exponent_ris", 2.2))     # near-LoS RIS links
        self.n_dir = float(geo.get("exponent_direct", 3.6))  # blocked direct link

        # --- RIS array geometry (steering phase of the LoS component) ---
        self.steering = bool(geo.get("ris_steering", True))
        self.d_lambda = float(geo.get("ris_element_spacing", 0.5))
        self.phi_bs = np.deg2rad(float(geo.get("bs_ris_angle_deg", -30.0)))
        self.ue_spread = np.deg2rad(float(geo.get("ue_angle_spread_deg", 60.0)))

        # --- Large-scale gains (fixed per world via UE placement) ---
        d_bs_ris = float(geo["bs_ris_distance"])
        self.pl_ris_bs = db_to_lin(-pathloss_db(np.array([d_bs_ris]), self.pl0, self.n_ris))[0]
        self.k_rician = db_to_lin(float(geo["rician_k_db"]))
        self._place_users(float(geo["ue_ris_min"]), float(geo["ue_ris_max"]))
        self._build_array_response()

        # Rolling history for delayed-CSI robustness experiments.  Only populated
        # when csi_delay > 0 -- the arrays are never mutated in place (each redraw
        # rebinds them), so the history stores references, not copies.
        self._history: list = []
        self.t = 0
        self.g = None
        self.hr = None
        self.hd = None
        self._cache: dict = {}
        # Bumped on every fading redraw; consumers key their memos on it.
        self.version = 0
        self._obs_gain = None
        self._obs_gain_ver = -1
        self._grp = None
        self._grp_ver = -1
        self.sample(force=True)

    # ------------------------------------------------------------------ #
    def _place_users(self, dmin: float, dmax: float):
        # UE<->RIS distances (annulus, area-uniform) and UE<->BS distances.
        E, K = self.n_envs, self.K
        u = self.rng.uniform(0.0, 1.0, size=(E, K))
        self.d_ue_ris = np.sqrt(dmin ** 2 + u * (dmax ** 2 - dmin ** 2))          # (E,K)
        # Direct UE<->BS distance: RIS-BS baseline plus UE offset (rough 2-D).
        self.d_ue_bs = self.cfg["geometry"]["bs_ris_distance"] + self.d_ue_ris
        self.pl_ue_ris = db_to_lin(-pathloss_db(self.d_ue_ris, self.pl0, self.n_ris))  # (E,K)
        self.pl_ue_bs = db_to_lin(-pathloss_db(self.d_ue_bs, self.pl0, self.n_dir))    # (E,K)

    def _build_array_response(self):
        """Per-element steering phases of the RIS (uniform linear array).

        The LoS term of a RIS link is an *array response vector*, not a scalar:
        for a ULA of spacing ``d/lambda`` the n-th element carries

            a_n(phi) = exp(-j * 2*pi * (d/lambda) * n * sin(phi)).

        Leaving it as the scalar ``sqrt(K/(K+1))`` (``ris_steering: false``, the
        pre-fix behaviour) asserts ``sin(phi) = 0`` on *both* hops -- the RIS
        broadside to the BS and to every UE at once, which cannot hold for users
        at different positions. It also leaves every element already co-phased,
        so RIS phase control has essentially nothing to do: measured headroom of
        greedy co-phasing over a constant phase vector was only 1.04x.

        The cascade of the two hops carries
        ``exp(-j*2*pi*(d/lambda)*n*(sin phi_k + sin phi_B))``, so different users
        prefer different phase profiles -- the multi-user RIS trade-off the
        hierarchy is meant to resolve. Element modulus stays 1, so the per-link
        power (and hence the pathloss/SNR budget) is unchanged.
        """
        if not self.steering:
            self.a_bs = None
            self.a_ue = None
            self.phi_ue = None
            return
        n = np.arange(self.N)
        # RIS -> BS: one fixed departure angle (fixed deployment geometry).
        self.a_bs = np.exp(-2j * np.pi * self.d_lambda * n * np.sin(self.phi_bs))   # (N,)
        # UE -> RIS: an arrival angle per user per world.
        self.phi_ue = self.rng.uniform(-self.ue_spread, self.ue_spread,
                                       size=(self.n_envs, self.K))                 # (E,K)
        self.a_ue = np.exp(-2j * np.pi * self.d_lambda
                           * n * np.sin(self.phi_ue)[..., None])                   # (E,K,N)

    def _rician(self, shape, steer=None) -> np.ndarray:
        """Unit-power Rician fading with K-factor self.k_rician.

        ``steer`` is the array response multiplying the LoS component; ``None``
        reproduces the original scalar (broadside) LoS. The NLoS draws are
        identical either way, so the generator stream does not depend on it.
        """
        kk = self.k_rician
        los = np.sqrt(kk / (kk + 1.0))
        if steer is not None:
            los = los * steer
        nlos = np.sqrt(1.0 / (kk + 1.0)) * (
            self.rng.standard_normal(shape) + 1j * self.rng.standard_normal(shape)
        ) / np.sqrt(2.0)
        return los + nlos

    def _rayleigh(self, shape) -> np.ndarray:
        return (self.rng.standard_normal(shape) + 1j * self.rng.standard_normal(shape)) / np.sqrt(2.0)

    # ------------------------------------------------------------------ #
    def sample(self, force: bool = False):
        """Advance one slot; redraw small-scale fading every `coherence` slots."""
        E, K, N = self.n_envs, self.K, self.N
        if force or (self.t % self.coherence == 0):
            # RIS->BS channel g (E,N), UE->RIS channels hr (E,K,N): Rician + pathloss.
            g_small = self._rician((E, N), steer=self.a_bs)
            self.g = np.sqrt(self.pl_ris_bs) * g_small
            hr_small = self._rician((E, K, N), steer=self.a_ue)
            self.hr = np.sqrt(self.pl_ue_ris)[..., None] * hr_small
            # Weak direct link (partial blockage): Rayleigh + pathloss, scaled by rho_bar.
            hd_small = self._rayleigh((E, K))
            self.hd = self.rho_bar * np.sqrt(self.pl_ue_bs) * hd_small
            self._cache.clear()
            self.version += 1
        self.t += 1
        # Maintain a short history for delayed-CSI experiments (skipped when the
        # delay is zero, which is the default and every non-robustness run).
        if self.csi_delay > 0:
            self._history.append((self.version, self.g, self.hr, self.hd))
            if len(self._history) > self.csi_delay + 1:
                self._history.pop(0)

    # ------------------------------------------------------------------ #
    #  Cached per-fading-block reductions over the element axis N
    # ------------------------------------------------------------------ #
    def _derive(self, key: str):
        """Element-axis reductions, computed once per fading block.

        ``hr`` is the big array ((E,K,N) complex) and every reduction below used
        to walk it separately -- via ``np.abs(hr)**2``, which computes a hypot and
        then squares it back. It is now touched once: ``|hr|^2`` is formed as
        ``re^2 + im^2`` (no sqrt), and the group/element aggregates are all folded
        out of that single intermediate.
        """
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        E, K, G, M, N = self.n_envs, self.K, self.G, self.M, self.N
        if key == "C":
            # per-group cascade coefficient: sum_{n in gp} hr[e,k,n] * g[e,n]
            val = (self.hr * self.g[:, None, :]).reshape(E, K, G, M).sum(axis=3)   # (E,K,G)
        elif key == "zeta_g":
            # zeta_g = sum_{n in gp} |g_n|^2                                        (E,G)
            g = self.g
            val = (g.real ** 2 + g.imag ** 2).reshape(E, G, M).sum(axis=2)
        elif key == "xi_gk":
            # xi_{gp,k} = sum_{n in gp} |hr_{k,n}|^2                                (E,K,G)
            hr = self.hr
            val = (hr.real ** 2 + hr.imag ** 2).reshape(E, K, G, M).sum(axis=3)
        elif key == "inc_g":
            # mean over the elements of a group of sum_k |hr_{k,n}|^2               (E,G)
            # == (sum_k xi_{gp,k}) / M, so it reuses the xi reduction.
            val = self._derive("xi_gk").sum(axis=1) / M
        elif key == "hr_mean_abs2":
            # mean_n |hr_{k,n}|^2 == (sum_gp xi_{gp,k}) / N                         (E,K)
            val = self._derive("xi_gk").sum(axis=2) / N
        elif key == "g_mean_abs2":
            # mean_n |g_n|^2 == (sum_gp zeta_gp) / N                                (E,)
            val = self._derive("zeta_g").sum(axis=1) / N
        else:  # pragma: no cover
            raise KeyError(key)
        self._cache[key] = val
        return val

    # Public accessors for the cached quantities (used by env physics).
    def cascade_coeff(self):   return self._derive("C")       # (E,K,G) complex
    def zeta_g(self):          return self._derive("zeta_g")  # (E,G)
    def xi_gk(self):           return self._derive("xi_gk")   # (E,K,G)
    def incident_group(self):  return self._derive("inc_g")   # (E,G)

    # ------------------------------------------------------------------ #
    def true_channels(self):
        """Ground-truth channels used by the *environment* to compute rates."""
        return self.g, self.hr, self.hd

    def _observed_raw(self):
        """(version, g, hr, hd) of the fading block the agent is allowed to see."""
        if self.csi_delay > 0 and len(self._history) > self.csi_delay:
            return self._history[0]
        return self.version, self.g, self.hr, self.hd

    def observed_channels(self):
        """Channels as seen by the *agent*: optionally delayed and/or noisy."""
        _, g, hr, hd = self._observed_raw()
        if self.csi_error_std > 0.0:
            def _noisy(x):
                e = self.csi_error_std * (
                    self.rng.standard_normal(x.shape) + 1j * self.rng.standard_normal(x.shape)
                ) / np.sqrt(2.0)
                return x * (1.0 + e)
            g, hr, hd = _noisy(g), _noisy(hr), _noisy(hd)
        return g, hr, hd

    def observation_feature(self):
        """Per-UE composite-gain feature for the agent's observation.            (E,K)

        Two normalizations of the raw gain
        ``|hd_k|^2 + mean_n |hr_{k,n}|^2 * mean_n |g_n|^2``:

        ``"db"`` (default)
            ``clip((10*log10(gain) - lo) / (hi - lo), 0, 1)``. Channel gains are
            a *log*-scale quantity -- with the default geometry the raw gain
            spans about -120.5 dB to -79 dB -- so mapping the decibel value onto
            [0,1] is what actually puts this feature on the same footing as the
            queue/alpha/f blocks.

        ``"legacy"``
            ``gain / (1 + gain)``, kept to reproduce previously reported runs.
            It is numerically inert: the raw gain is ~1e-12 to 1e-11, so
            ``1 + gain == 1`` to within rounding and the "squashing" is the
            identity. The feature reaches the policy as ~1e-11 against inputs of
            order 0.1-1, i.e. the agent is effectively blind to the channel --
            which is why the CSI-robustness sweep came out flat.
        """
        if self.csi_error_std > 0.0:
            # Fresh estimation noise every slot, so nothing can be cached.
            g, hr, hd = self.observed_channels()
            gain = (np.abs(hd) ** 2
                    + np.mean(np.abs(hr) ** 2, axis=2)
                    * np.mean(np.abs(g) ** 2, axis=1)[:, None])
            return self._normalize(gain)

        # Clean CSI: the gain only changes when the fading block does, so cache
        # the finished feature (the log10 included). The general cache is cleared
        # on redraw, hence the explicit version key for the delayed-CSI case.
        ver, g, hr, hd = self._observed_raw()
        if self._obs_gain_ver != ver or self._obs_gain is None:
            if self.csi_delay == 0:
                # Undelayed clean CSI == the true channel, so the element-axis
                # means fall out of the reductions already cached for the physics.
                mean_hr2 = self._derive("hr_mean_abs2")               # (E,K)
                mean_g2 = self._derive("g_mean_abs2")                 # (E,)
            else:
                mean_hr2 = np.mean(hr.real ** 2 + hr.imag ** 2, axis=2)
                mean_g2 = np.mean(g.real ** 2 + g.imag ** 2, axis=1)
            gain = (hd.real ** 2 + hd.imag ** 2) + mean_hr2 * mean_g2[:, None]
            self._obs_gain = self._normalize(gain)
            self._obs_gain_ver = ver
        return self._obs_gain

    def _normalize(self, gain):
        if self.obs_norm == "legacy":
            return gain / (1.0 + gain)
        db = 10.0 * np.log10(np.maximum(gain, 1e-300))
        return np.clip((db - self.obs_db_lo) * self._obs_db_scale, 0.0, 1.0)

    # ------------------------------------------------------------------ #
    def group_feature(self):
        """Per-group cascade features for the agent's observation.        (E, 3G)

        The tactician emits G RIS phases, but the composite-gain feature above is
        a single magnitude per user averaged over *all* N elements -- it carries
        no per-group and no phase information whatsoever. So ``theta_g`` was
        open-loop with respect to the channel: the agent could not co-phase, and
        consequently no channel impairment (CSI error, phase quantization) could
        degrade a decision that was never channel-aware.

        The quantity the phase decision actually needs is the user-aggregated
        per-group cascade response

            c_g = sum_k sum_{n in g} hr[k,n] * g[n],

        which is exactly what the greedy co-phasing baseline maximizes with
        ``theta_g = -angle(c_g)`` (see baselines._greedy_phase). It is exposed as

            |c_g| / max_g |c_g|          relative magnitude, in [0,1]
            (cos(angle c_g) + 1) / 2     phase, split to avoid the 2pi wrap
            (sin(angle c_g) + 1) / 2

        which is scale-invariant, so unlike the gain feature it needs no
        geometry-dependent calibration constant.

        Crucially this is built from the **observed** (delayed/noisy) channel, so
        CSI impairments now reach the decision they are supposed to impair.
        """
        E, K, G, M = self.n_envs, self.K, self.G, self.M
        if self.csi_error_std > 0.0:
            g, hr, _ = self.observed_channels()
            return self._group_feature_from(g, hr)
        ver, g, hr, _ = self._observed_raw()
        if self._grp_ver != ver or self._grp is None:
            self._grp = self._group_feature_from(g, hr)
            self._grp_ver = ver
        return self._grp

    def _group_feature_from(self, g, hr):
        E, K, G, M = self.n_envs, self.K, self.G, self.M
        # sum over users first, then over the elements of each group.
        c = (hr.sum(axis=1) * g).reshape(E, G, M).sum(axis=2)        # (E,G) complex
        mag = np.abs(c)
        mag = mag / np.maximum(mag.max(axis=1, keepdims=True), 1e-300)
        ang = np.angle(c)
        return np.concatenate(
            [mag, 0.5 * (np.cos(ang) + 1.0), 0.5 * (np.sin(ang) + 1.0)], axis=1)


class ChannelView:
    """Single-world (K,N)-shaped view of a batched :class:`ChannelModel`.

    Lets the scalar ``RISMECEnv`` facade and the model-based baselines keep using
    ``env.chan.true_channels()`` with un-batched arrays.
    """

    def __init__(self, chan: ChannelModel, idx: int = 0):
        self._c = chan
        self._i = idx

    def __getattr__(self, name):          # coherence, csi_delay, t, ...
        return getattr(self._c, name)

    def true_channels(self):
        g, hr, hd = self._c.true_channels()
        return g[self._i], hr[self._i], hd[self._i]

    def observed_channels(self):
        g, hr, hd = self._c.observed_channels()
        return g[self._i], hr[self._i], hd[self._i]

    def cascade_coeff(self):
        return self._c.cascade_coeff()[self._i]


def quantize_phase(theta: np.ndarray, bits: int) -> np.ndarray:
    """Quantize phase shifts to `bits` bits. bits<=0 keeps them continuous."""
    if bits is None or bits <= 0:
        return theta
    levels = 2 ** bits
    step = 2.0 * np.pi / levels
    return np.round(theta / step) * step
