"""
Channel model for the RIS-MEC system.
  * 3GPP TR 38.901 Urban-Micro (UMi) distance-dependent pathloss,
  * large-scale geometry (BS, RIS, UEs) so pathloss is meaningful,
  * Rician small-scale fading on the RIS links (dominant LoS component),
  * a *partial* direct-link blockage factor rho_bar in [0,1] (fixes the A1: rho_bar=0 
  truly zeroes h_d, rho_bar>0 keeps a weak path),
  * optional CSI estimation error and RIS phase quantization for robustness
    sweeps.

All randomness flows through a single numpy Generator for reproducibility.
Vectorized over users and elements.
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
    """Generates block-fading channels for the RIS-MEC topology.

    Geometry (2-D, meters): BS at origin-ish, RIS at fixed offset, UEs drawn in
    an annulus around the RIS. The cascaded UE->RIS->BS pathloss is the product
    of the two large-scale gains; small-scale fading is Rician on the RIS links
    and Rayleigh on the (weak) direct link.
    """

    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        sys_, geo, ch = cfg["system"], cfg["geometry"], cfg["channel"]
        self.K = int(sys_["K"])
        self.N = int(sys_["N"])
        self.fc = float(geo["fc_ghz"])
        self.rho_bar = float(ch["direct_blockage"])
        self.coherence = int(ch["coherence_slots"])
        self.csi_error_std = float(ch.get("csi_error_std", 0.0))
        self.csi_delay = int(ch.get("csi_delay_slots", 0))

        # --- Pathloss parameters (reference-distance model) ---
        self.pl0 = float(geo.get("pl0_db", 30.0))
        self.n_ris = float(geo.get("exponent_ris", 2.2))     # near-LoS RIS links
        self.n_dir = float(geo.get("exponent_direct", 3.6))  # blocked direct link

        # --- Large-scale gains (fixed per episode via UE placement) ---
        d_bs_ris = float(geo["bs_ris_distance"])
        self.pl_ris_bs = db_to_lin(-pathloss_db(np.array([d_bs_ris]), self.pl0, self.n_ris))[0]
        self.k_rician = db_to_lin(float(geo["rician_k_db"]))
        self._place_users(float(geo["ue_ris_min"]), float(geo["ue_ris_max"]))

        # Rolling history for delayed-CSI robustness experiments.
        self._history: list = []
        self.t = 0
        self.g = None
        self.hr = None
        self.hd = None
        self.sample(force=True)

    # ------------------------------------------------------------------ #
    def _place_users(self, dmin: float, dmax: float):
        # UE<->RIS distances (annulus, area-uniform) and UE<->BS distances.
        u = self.rng.uniform(0.0, 1.0, size=self.K)
        self.d_ue_ris = np.sqrt(dmin ** 2 + u * (dmax ** 2 - dmin ** 2))
        # Direct UE<->BS distance: RIS-BS baseline plus UE offset (rough 2-D).
        self.d_ue_bs = self.cfg["geometry"]["bs_ris_distance"] + self.d_ue_ris
        self.pl_ue_ris = db_to_lin(-pathloss_db(self.d_ue_ris, self.pl0, self.n_ris))  # (K,)
        self.pl_ue_bs = db_to_lin(-pathloss_db(self.d_ue_bs, self.pl0, self.n_dir))    # (K,)

    def _rician(self, shape) -> np.ndarray:
        """Unit-power Rician fading with K-factor self.k_rician."""
        kk = self.k_rician
        los = np.sqrt(kk / (kk + 1.0))
        nlos = np.sqrt(1.0 / (kk + 1.0)) * (
            self.rng.standard_normal(shape) + 1j * self.rng.standard_normal(shape)
        ) / np.sqrt(2.0)
        return los + nlos

    def _rayleigh(self, shape) -> np.ndarray:
        return (self.rng.standard_normal(shape) + 1j * self.rng.standard_normal(shape)) / np.sqrt(2.0)

    # ------------------------------------------------------------------ #
    def sample(self, force: bool = False):
        """Advance one slot; redraw small-scale fading every `coherence` slots."""
        if force or (self.t % self.coherence == 0):
            # RIS->BS channel g (N,), UE->RIS channels hr (K,N): Rician + pathloss.
            g_small = self._rician(self.N)
            self.g = np.sqrt(self.pl_ris_bs) * g_small
            hr_small = self._rician((self.K, self.N))
            self.hr = np.sqrt(self.pl_ue_ris)[:, None] * hr_small
            # Weak direct link (partial blockage): Rayleigh + pathloss, scaled by rho_bar.
            hd_small = self._rayleigh(self.K)
            self.hd = self.rho_bar * np.sqrt(self.pl_ue_bs) * hd_small
        self.t += 1
        # Maintain a short history for delayed-CSI experiments.
        self._history.append((self.g.copy(), self.hr.copy(), self.hd.copy()))
        if len(self._history) > self.csi_delay + 1:
            self._history.pop(0)

    def true_channels(self):
        """Ground-truth channels used by the *environment* to compute rates."""
        return self.g, self.hr, self.hd

    def observed_channels(self):
        """Channels as seen by the *agent*: optionally delayed and/or noisy."""
        if self.csi_delay > 0 and len(self._history) > self.csi_delay:
            g, hr, hd = self._history[0]
        else:
            g, hr, hd = self.g, self.hr, self.hd
        if self.csi_error_std > 0.0:
            def _noisy(x):
                e = self.csi_error_std * (
                    self.rng.standard_normal(x.shape) + 1j * self.rng.standard_normal(x.shape)
                ) / np.sqrt(2.0)
                return x * (1.0 + e)
            g, hr, hd = _noisy(g), _noisy(hr), _noisy(hd)
        return g, hr, hd


def quantize_phase(theta: np.ndarray, bits: int) -> np.ndarray:
    """Quantize phase shifts to `bits` bits (R2.6). bits<=0 keeps them continuous."""
    if bits is None or bits <= 0:
        return theta
    levels = 2 ** bits
    step = 2.0 * np.pi / levels
    return np.round(theta / step) * step
