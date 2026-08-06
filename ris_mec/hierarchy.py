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
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .env import RISMECEnv


# --------------------------------------------------------------------------- #
#  Strategy mapping
# --------------------------------------------------------------------------- #
class MetaManager:
    """Maps a meta-action vector to strategic variables and builds meta-obs."""

    def __init__(self, env: RISMECEnv, enable_ris: bool = True):
        self.env = env
        self.enable_ris = enable_ris
        self.K, self.G = env.K, env.G
        self.act_dim = 2 * self.K + self.G     # alpha(K), f_logits(K), a_ctrl(G)
        self.obs_dim = 3 * self.K + 2          # q(K), alpha(K), f(K), cpu, t

    def meta_obs(self) -> np.ndarray:
        e = self.env
        qn = np.clip(e.queues / e.q_norm, 0, 1)
        fn = e.f / e.F_max
        obs = np.concatenate([qn, e.alpha, fn, [e.f.sum() / e.F_max], [e.t / e.max_steps]])
        return obs.astype(np.float32)

    def apply_meta(self, a: np.ndarray):
        a = np.clip(np.asarray(a, float), -1.0, 1.0)
        K, G = self.K, self.G
        alpha = (a[:K] + 1) / 2.0
        f_logits = a[K:2 * K]
        fw = np.exp(f_logits - f_logits.max()); fw /= fw.sum()
        f = fw * self.env.F_max
        if self.enable_ris:
            a_ctrl = (a[2 * K:2 * K + G] + 1) / 2.0 * self.env.a_max   # [0, a_max]
        else:
            a_ctrl = np.ones(G)                                        # passive ablation
        self.env.set_strategy(alpha=alpha, f=f, a_gains=a_ctrl)


# --------------------------------------------------------------------------- #
#  Tactician pre-training under randomized strategy
# --------------------------------------------------------------------------- #
class TacticianEnv(gym.Env):
    """Exposes the low-level control problem; randomizes the strategic context at
    reset so the frozen tactician generalizes across strategist decisions."""

    def __init__(self, cfg, seed=0):
        super().__init__()
        self.env = RISMECEnv(cfg, seed=seed)
        self.mgr = MetaManager(self.env)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._rng = np.random.default_rng(seed + 777)

    def _randomize_strategy(self):
        a = self._rng.uniform(-1, 1, size=self.mgr.act_dim)
        self.mgr.apply_meta(a)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        self._randomize_strategy()
        return self.env._obs(), info

    def step(self, action):
        # occasionally shift strategy to expose the tactician to varied contexts
        if self.env.t % self.env.chan.coherence == 0 and self.env.t > 0:
            self._randomize_strategy()
        return self.env.step(action)


# --------------------------------------------------------------------------- #
#  Strategist environment (rolls a frozen tactician for H slots per meta-step)
# --------------------------------------------------------------------------- #
class MetaEnv(gym.Env):
    def __init__(self, cfg, low_model: PPO, enable_ris: bool = True, seed=0):
        super().__init__()
        self.env = RISMECEnv(cfg, seed=seed)
        self.mgr = MetaManager(self.env, enable_ris=enable_ris)
        self.low = low_model
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
            obs_low = self.env._obs()
            a_low, _ = self.low.predict(obs_low, deterministic=True)
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
def _ppo(env, cfg, **kw):
    p = cfg["training"]["ppo"]
    kwargs = dict(n_steps=p["n_steps"], batch_size=p["batch_size"],
                  learning_rate=p["learning_rate"], gamma=p["gamma"],
                  gae_lambda=p["gae_lambda"], clip_range=p["clip_range"],
                  ent_coef=p["ent_coef"], vf_coef=p["vf_coef"], verbose=0)
    kwargs.update(kw)
    return PPO("MlpPolicy", env, **kwargs)


def train_tactician(cfg, seed=0):
    venv = DummyVecEnv([lambda: TacticianEnv(cfg, seed=seed)])
    low = _ppo(venv, cfg)
    low.learn(total_timesteps=int(cfg["training"]["low_timesteps"]))
    return low


def train_strategist(cfg, low, seed=0, enable_ris=True):
    venv = DummyVecEnv([lambda: MetaEnv(cfg, low, enable_ris=enable_ris, seed=seed)])
    meta = _ppo(venv, cfg, n_steps=128, batch_size=32,
                learning_rate=1e-4)
    meta.learn(total_timesteps=int(cfg["training"]["meta_timesteps"]))
    return meta


def train_hdrl(cfg, seed=0, enable_ris=True):
    low = train_tactician(cfg, seed=seed)
    meta = train_strategist(cfg, low, seed=seed, enable_ris=enable_ris)
    return low, meta
