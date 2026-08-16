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

Everything is evaluated for a whole batch of worlds at once: the scalar fields of
:class:`StepStats` carry shape (E,) and the per-user fields shape (E, K), and
``__call__`` returns a reward of shape (E,).  The Dinkelbach EMA is tracked per
world.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class StepStats:
    """Per-slot statistics for a batch of E worlds."""
    bits: np.ndarray      # (E,)   total successfully offloaded bits this slot
    energy: np.ndarray    # (E,)   total energy this slot [J] (uplink + downlink + RIS)
    latency: np.ndarray   # (E,)   mean task-completion latency this slot [s]
    ee: np.ndarray        # (E,)   instantaneous EE = bits / energy [bits/J]
    queues: np.ndarray    # (E,K)  queue backlog after update [bits]
    arrivals: np.ndarray  # (E,K)  arrivals this slot [bits]
    served: np.ndarray    # (E,K)  bits served (uplink) this slot [bits]
    qos_penalty: float = 0.0


class RewardObjective:
    def __init__(self, cfg: dict, n_envs: int = 1):
        h = cfg["hierarchy"]
        self.kind = str(h.get("objective", "dpp"))
        if self.kind not in ("ee_latency", "dinkelbach", "dpp"):
            raise ValueError(f"unknown objective '{self.kind}'")
        self.lambda_E = float(h.get("lambda_E", 1.0))
        self.V = float(h.get("dpp_V", 1.0))
        self.q_norm = float(cfg["queue"]["q_norm"])
        self.n_envs = int(n_envs)
        # EE scale so normalized EE ~ O(1) (system EE is tens of Mbit/J).
        self.ee_scale = 1.0e7
        self._beta_ema = 0.02
        self.reset()

    def reset(self):
        self._q_dinkelbach = np.zeros(self.n_envs)  # running EE estimate (per world)
        self._ee_ema = np.zeros(self.n_envs)

    def __call__(self, s: StepStats) -> np.ndarray:
        ee_n = s.ee / self.ee_scale
        lat = s.latency
        if self.kind == "ee_latency":
            return ee_n - self.lambda_E * lat + s.qos_penalty
        if self.kind == "dinkelbach":
            # (bits - q*energy) normalized; q tracks EE fixed point.
            self._q_dinkelbach = ((1 - self._beta_ema) * self._q_dinkelbach
                                  + self._beta_ema * s.ee)
            return ((s.bits - self._q_dinkelbach * s.energy) / self.ee_scale
                    - self.lambda_E * lat + s.qos_penalty)
        # "dpp": P1 maximizes (eta - lambda_E * L); the DPP cost to MINIMIZE is
        # therefore (lambda_E * L - eta). Keeping lambda_E on latency makes the
        # per-step surrogate consistent with P1 (fixes the lambda_E inversion).
        qn = s.queues / self.q_norm
        drift = (qn * (s.arrivals - s.served) / self.q_norm).mean(axis=1)
        penalty = self.lambda_E * lat - ee_n
        return -(drift + self.V * penalty) + s.qos_penalty
