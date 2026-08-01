"""
Reward objectives that make the RL problem *equal to* P1.


Three forms are provided:
  * "ee_latency": directly rewards normalized energy efficiency minus a weighted
    latency term. Simple; the ratio is optimized in expectation.
  * "dinkelbach": rewards (bits - q * energy) - lambda_E * latency, the classical
    Dinkelbach linearization of the EE ratio (q tracks the running EE estimate),
    so maximizing it drives EE toward its fixed point.
  * "dpp": Lyapunov drift-plus-penalty. Reward = -(drift + V * penalty) where
    drift ~ sum_k Q_k * (arrivals_k - served_k) and penalty = latency - lambda_E*EE.
    This *provably* trades queue stability against the EE/latency objective and is
    the fairest match to the Lyapunov-DPP baseline.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class StepStats:
    bits: float           # total successfully offloaded bits this slot
    energy: float         # total energy this slot [J] (uplink + downlink + RIS)
    latency: float        # mean task-completion latency this slot [s]
    ee: float             # instantaneous EE = bits / energy [bits/J]
    queues: np.ndarray    # queue backlog after update [bits]
    arrivals: np.ndarray  # arrivals this slot [bits]
    served: np.ndarray    # bits served (uplink) this slot [bits]
    qos_penalty: float = 0.0


class RewardObjective:
    def __init__(self, cfg: dict):
        h = cfg["hierarchy"]
        self.kind = str(h.get("objective", "dpp"))
        self.lambda_E = float(h.get("lambda_E", 1.0))
        self.V = float(h.get("dpp_V", 1.0))
        self.q_norm = float(cfg["queue"]["q_norm"])
        # EE scale so normalized EE ~ O(1) (system EE is tens of Mbit/J).
        self.ee_scale = 1.0e7
        self._q_dinkelbach = 0.0  # running EE estimate for Dinkelbach
        self._ee_ema = 0.0
        self._beta_ema = 0.02

    def reset(self):
        self._q_dinkelbach = 0.0
        self._ee_ema = 0.0

    def __call__(self, s: StepStats) -> float:
        ee_n = s.ee / self.ee_scale
        lat = s.latency
        if self.kind == "ee_latency":
            r = ee_n - self.lambda_E * lat + s.qos_penalty
        elif self.kind == "dinkelbach":
            # (bits - q*energy) normalized; q tracks EE fixed point.
            self._q_dinkelbach = (1 - self._beta_ema) * self._q_dinkelbach + self._beta_ema * s.ee
            r = (s.bits - self._q_dinkelbach * s.energy) / (self.ee_scale) - self.lambda_E * lat + s.qos_penalty
        elif self.kind == "dpp":
            # P1 maximizes (eta - lambda_E * L); the DPP cost to MINIMIZE is
            # therefore (lambda_E * L - eta). Keeping lambda_E on latency makes the
            # per-step surrogate consistent with P1 (fixes the lambda_E inversion).
            qn = s.queues / self.q_norm
            drift = float(np.mean(qn * (s.arrivals - s.served) / self.q_norm))
            penalty = self.lambda_E * lat - ee_n
            r = -(drift + self.V * penalty) + s.qos_penalty
        else:
            raise ValueError(f"unknown objective '{self.kind}'")
        return float(r)
