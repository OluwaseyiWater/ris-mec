"""
Flat PPO baseline: a single agent that controls *all* variables (strategic +
tactical) at every slot. This is the 'curse of dimensionality' straw-man the
paper argues against -- it has the same action content as HDRL but no temporal
decomposition, so its action space and gradient variance are larger.
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .env import RISMECEnv
from .hierarchy import MetaManager, _ppo, vec_cfg, fit_n_envs
from .vec_env import VecFlatEnv


class FlatEnv(gym.Env):
    """Single-world combined meta+tactical control every slot.

    Kept for evaluation and per-slot inference timing; training uses the batched
    :class:`ris_mec.vec_env.VecFlatEnv`.
    """

    def __init__(self, cfg, seed=0):
        super().__init__()
        self.env = RISMECEnv(cfg, seed=seed)
        self.mgr = MetaManager(self.env)
        self.meta_dim = self.mgr.act_dim
        self.low_dim = self.env.action_space.shape[0]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.meta_dim + self.low_dim,),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(self.env.observation_space.shape[0] + self.mgr.obs_dim,),
            dtype=np.float32)

    def _obs(self):
        return np.concatenate([self.env._obs(), self.mgr.meta_obs()]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, float)
        self.mgr.apply_meta(action[:self.meta_dim])
        _, r, done, _, info = self.env.step(action[self.meta_dim:])
        return self._obs(), r, done, False, info


def train_flat(cfg, seed=0):
    vc = vec_cfg(cfg)
    total = int(cfg["training"]["flat_timesteps"])
    n = fit_n_envs(cfg, total, cfg["training"]["ppo"]["n_steps"])
    venv = VecFlatEnv(cfg, n_envs=n, seed=seed)
    flat = _ppo(venv, cfg, n_envs=n,
                batch_size=128 * (n if vc["scale_batch"] else 1),
                learning_rate=2e-4)
    flat.learn(total_timesteps=total)
    return flat
