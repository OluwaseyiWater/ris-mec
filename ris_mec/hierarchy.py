"""
Hierarchical control: a slow 'strategist' (meta-controller) over a fast
'tactician' (low-level controller), matching decision frequency to the natural
timescale of each variable.

  * Strategist (period H): sets alpha (offload ratios), f (MEC CPU shares), and
    the active-RIS group amplitude gains a_g. Because a_g moves both signal gain
    and the amplifier-noise floor (see env.py), it is deliberately updated on the
    slow timescale so the gain/noise trade-off is averaged over channel
    realizations -- this is the paper's core mechanism, now actually implemented.
  * Tactician (period 1): sets uplink/downlink powers, bandwidth split beta, and
    RIS phases theta_g at every slot.

Training is two-phase: pre-train the tactician under randomized strategy, freeze
it, then train the strategist on top.

Both phases run on the batched vector envs from :mod:`ris_mec.vec_env`, i.e.
``vec.n_envs`` worlds are advanced per numpy call and per policy forward pass.
The PPO *sample* budget (``total_timesteps``) and the GAE horizon (``n_steps``)
are unchanged; the minibatch size is scaled with the number of worlds so the
number of gradient steps per environment sample also stays the same -- each
gradient is just estimated from a proportionally larger, lower-variance batch.
See ``vec:`` in configs/default.yaml.
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from .env import RISMECEnv
from .core import BatchedRISMEC
from .vec_env import VecTacticianEnv, VecMetaEnv, FrozenPolicy


# --------------------------------------------------------------------------- #
#  Vectorization / PPO scaling knobs
# --------------------------------------------------------------------------- #
def vec_cfg(cfg: dict) -> dict:
    v = dict(cfg.get("vec") or {})
    return {
        "n_envs": int(v.get("n_envs", 32)),
        "meta_n_envs": int(v.get("meta_n_envs", v.get("n_envs", 32))),
        "scale_batch": bool(v.get("scale_batch", True)),
        "device": str(v.get("device", "auto")),
    }


def fit_n_envs(cfg: dict, total_timesteps: int, n_steps: int, requested: int = None) -> int:
    """Number of worlds to use, capped so one rollout does not overshoot the budget.

    PPO's rollout buffer holds ``n_steps * n_envs`` transitions and it always
    collects whole rollouts, so with a small ``total_timesteps`` a large
    ``n_envs`` would collect far more experience than asked for -- the `fast` and
    `reduced` profiles would silently do several times the intended work. Capping
    at ``ceil(total_timesteps / n_steps)`` keeps the collected budget within one
    rollout of the request while leaving the full profile untouched.
    """
    req = vec_cfg(cfg)["n_envs"] if requested is None else requested
    return max(1, min(int(req), -(-int(total_timesteps) // max(1, int(n_steps)))))


def _ppo(env, cfg, n_envs: int = 1, **kw):
    """Build PPO with the config's hyperparameters, scaled for ``n_envs`` worlds.

    ``n_steps`` is per-world, so the rollout buffer grows by ``n_envs``. Scaling
    ``batch_size`` by the same factor keeps gradient-steps-per-sample -- and hence
    the effective optimization budget -- identical to the single-world setting.
    """
    p = cfg["training"]["ppo"]
    vc = vec_cfg(cfg)
    batch = int(p["batch_size"])
    if vc["scale_batch"]:
        batch *= max(1, n_envs)
    kwargs = dict(n_steps=p["n_steps"], batch_size=batch,
                  learning_rate=p["learning_rate"], gamma=p["gamma"],
                  gae_lambda=p["gae_lambda"], clip_range=p["clip_range"],
                  ent_coef=p["ent_coef"], vf_coef=p["vf_coef"], verbose=0,
                  device=vc["device"])
    kwargs.update(kw)
    # The buffer must hold at least one minibatch.
    buf = kwargs["n_steps"] * max(1, n_envs)
    kwargs["batch_size"] = min(kwargs["batch_size"], buf)
    return PPO("MlpPolicy", env, **kwargs)


# --------------------------------------------------------------------------- #
#  Strategy mapping (single-world adapter over the batched core)
# --------------------------------------------------------------------------- #
class MetaManager:
    """Maps a meta-action vector to strategic variables and builds meta-obs.

    Single-world view, used by evaluation and by the per-slot inference timing.
    The arithmetic itself lives in :class:`ris_mec.core.BatchedRISMEC`.
    """

    def __init__(self, env, enable_ris: bool = True):
        self.core = env.core if isinstance(env, RISMECEnv) else env
        self.env = env
        self.enable_ris = enable_ris
        self.K, self.G = self.core.K, self.core.G
        self.act_dim = self.core.meta_act_dim   # alpha(K), f_logits(K), a_ctrl(G)
        self.obs_dim = self.core.meta_obs_dim   # q(K), alpha(K), f(K), cpu, t

    def meta_obs(self) -> np.ndarray:
        return self.core.meta_obs()[0]

    def apply_meta(self, a: np.ndarray):
        self.core.apply_meta(a, enable_ris=self.enable_ris)


# --------------------------------------------------------------------------- #
#  Scalar envs, kept for per-slot inference timing and ad-hoc use
# --------------------------------------------------------------------------- #
class TacticianEnv(gym.Env):
    """Single-world version of :class:`ris_mec.vec_env.VecTacticianEnv`."""

    def __init__(self, cfg, seed=0):
        super().__init__()
        self.env = RISMECEnv(cfg, seed=seed)
        self.mgr = MetaManager(self.env)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._rng = np.random.default_rng(seed + 777)

    def _randomize_strategy(self):
        self.mgr.apply_meta(self._rng.uniform(-1, 1, size=self.mgr.act_dim))

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        self._randomize_strategy()
        return self.env._obs(), info

    def step(self, action):
        if self.env.t % self.env.chan.coherence == 0 and self.env.t > 0:
            self._randomize_strategy()
        return self.env.step(action)


class MetaEnv(gym.Env):
    """Single-world strategist env (rolls a frozen tactician for H slots)."""

    def __init__(self, cfg, low_model, enable_ris: bool = True, seed=0):
        super().__init__()
        self.env = RISMECEnv(cfg, seed=seed)
        self.mgr = MetaManager(self.env, enable_ris=enable_ris)
        self.low = FrozenPolicy(low_model)
        self.H = self.env.cfg["hierarchy"]["H"]
        self.n_meta = max(1, self.env.max_steps // self.H)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.mgr.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.mgr.act_dim,), dtype=np.float32)
        self._m = 0

    def reset(self, *, seed=None, options=None):
        self.env.reset(seed=seed)
        self._m = 0
        return self.mgr.meta_obs(), {}

    def step(self, action):
        self.mgr.apply_meta(action)
        R, done = 0.0, False
        for _ in range(self.H):
            a_low = self.low(self.env._obs()[None, :])[0]
            _, r, done, _, _ = self.env.step(a_low)
            R += r
            if done:
                break
        self._m += 1
        meta_done = done or self._m >= self.n_meta
        return self.mgr.meta_obs(), R / self.H, meta_done, False, {}


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #
def train_tactician(cfg, seed=0):
    total = int(cfg["training"]["low_timesteps"])
    n = fit_n_envs(cfg, total, cfg["training"]["ppo"]["n_steps"])
    venv = VecTacticianEnv(cfg, n_envs=n, seed=seed)
    low = _ppo(venv, cfg, n_envs=n)
    low.learn(total_timesteps=total)
    return low


META_N_STEPS = 128


def train_strategist(cfg, low, seed=0, enable_ris=True):
    vc = vec_cfg(cfg)
    total = int(cfg["training"]["meta_timesteps"])
    n = fit_n_envs(cfg, total, META_N_STEPS, requested=vc["meta_n_envs"])
    venv = VecMetaEnv(cfg, low, n_envs=n, enable_ris=enable_ris, seed=seed)
    meta = _ppo(venv, cfg, n_envs=n, n_steps=META_N_STEPS,
                batch_size=32 * (n if vc["scale_batch"] else 1),
                learning_rate=1e-4)
    meta.learn(total_timesteps=total)
    return meta


def train_hdrl(cfg, seed=0, enable_ris=True):
    low = train_tactician(cfg, seed=seed)
    meta = train_strategist(cfg, low, seed=seed, enable_ris=enable_ris)
    return low, meta
