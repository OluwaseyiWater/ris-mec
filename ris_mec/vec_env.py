"""
Stable-Baselines3 vector environments backed by the batched physics core.

These replace ``DummyVecEnv([lambda: SomeEnv(cfg)])``, which stepped one world at
a time and asked the policy for one action at a time. Driving ``n_envs`` worlds
through a single set of numpy calls (and a single batched policy forward pass) is
where essentially all of the speed-up lives:

    per-sample policy forward   ~294 us (batch 1)  ->  ~2-7 us (batch 64+)
    per-sample env physics      ~290 us (batch 1)  ->  fixed cost / n_envs

Three environments are provided, mirroring the scalar ones:

    VecTacticianEnv   low-level control under a randomized strategic context
    VecMetaEnv        strategist; each meta-step rolls a frozen tactician H slots
    VecFlatEnv        the flat-PPO straw-man: every variable, every slot

All worlds run in lockstep (same ``max_steps``), so ``done`` is shared and the
auto-reset is a single batched call.
"""

from __future__ import annotations
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from .core import BatchedRISMEC


def dims(cfg) -> dict:
    """Space dimensions implied by a config, without building a whole world."""
    K, N, G = int(cfg["system"]["K"]), int(cfg["system"]["N"]), int(cfg["system"]["G"])
    assert N % G == 0, "N must be divisible by G"
    grp = 3 * G if bool(cfg["channel"].get("obs_group_features", True)) else 0
    return {"K": K, "N": N, "G": G,
            "obs": 4 * K + 1 + grp, "act": 2 * K + 1 + G,
            "meta_obs": 3 * K + 2, "meta_act": 2 * K + G}


# --------------------------------------------------------------------------- #
#  Fast frozen-policy wrapper
# --------------------------------------------------------------------------- #
class FrozenPolicy:
    """Batched, allocation-light replacement for ``model.predict(obs, deterministic=True)``.

    ``BasePolicy.predict`` re-checks spaces, reshapes, moves data and rebuilds
    tensors on every call; at one observation per call that bookkeeping costs more
    than the forward pass itself (~294 us vs ~148 us measured). This keeps the
    same semantics -- deterministic action, clipped to the Box -- with one
    ``torch.no_grad()`` forward over the whole batch.
    """

    def __init__(self, model):
        self.policy = model.policy
        self.policy.set_training_mode(False)
        self.device = self.policy.device
        self.low = model.action_space.low
        self.high = model.action_space.high

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
            act = self.policy._predict(t, deterministic=True)
        return np.clip(act.cpu().numpy(), self.low, self.high)


# --------------------------------------------------------------------------- #
#  Common VecEnv plumbing
# --------------------------------------------------------------------------- #
class _CoreVecEnv(VecEnv):
    """VecEnv boilerplate shared by the three concrete envs.

    Subclasses implement ``_obs()``, ``_advance(actions)`` and ``_on_reset()``.
    """

    render_mode = None

    def __init__(self, cfg, n_envs: int, seed: int | None,
                 observation_space: spaces.Space, action_space: spaces.Space):
        self.core = BatchedRISMEC(cfg, n_envs=n_envs, seed=seed)
        self.cfg = cfg
        self._rng = np.random.default_rng((0 if seed is None else int(seed)) + 777)
        self._actions = None
        # VecEnv.__init__ probes get_attr("render_mode"), which needs num_envs.
        self.num_envs = n_envs
        super().__init__(n_envs, observation_space, action_space)

    # -- required abstract API ------------------------------------------ #
    def reset(self):
        self.core.reset()
        self._on_reset()
        return self._obs()

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs, reward, done, info = self._advance(self._actions)
        infos = [{k: float(v[i]) for k, v in info.items()} for i in range(self.num_envs)]
        dones = np.full(self.num_envs, bool(done))
        if done:
            # Note: "TimeLimit.truncated" is deliberately NOT set. The original
            # DummyVecEnv path did not set it either (the envs returned
            # truncated=False), so PPO treats the horizon as a true terminal and
            # does not bootstrap across it. Setting it would arguably be more
            # correct for a fixed-horizon episode, but it would change the
            # learning target relative to previously reported runs.
            for i in range(self.num_envs):
                infos[i]["terminal_observation"] = obs[i]
            self.core.reset()
            self._on_reset()
            obs = self._obs()
        return obs, np.asarray(reward, dtype=np.float32), dones, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        sentinel = object()
        val = getattr(self, attr_name, sentinel)
        if val is sentinel:
            val = getattr(self.core, attr_name)
        return [val] * len(self._idx(indices))

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.core, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        fn = getattr(self.core, method_name)
        return [fn(*args, **kwargs) for _ in self._idx(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._idx(indices))

    def _idx(self, indices):
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            return [indices]
        return list(indices)

    # -- subclass hooks -------------------------------------------------- #
    def _on_reset(self):
        pass

    def _obs(self):                                  # pragma: no cover - abstract
        raise NotImplementedError

    def _advance(self, actions):                     # pragma: no cover - abstract
        raise NotImplementedError


# --------------------------------------------------------------------------- #
#  Tactician: low-level control under a randomized strategic context
# --------------------------------------------------------------------------- #
class VecTacticianEnv(_CoreVecEnv):
    """Exposes the low-level control problem; randomizes the strategic context at
    reset (and every coherence block) so the frozen tactician generalizes across
    strategist decisions."""

    def __init__(self, cfg, n_envs: int = 32, seed: int = 0):
        d = dims(cfg)
        obs_space = spaces.Box(0.0, 1.0, shape=(d["obs"],), dtype=np.float32)
        act_space = spaces.Box(-1.0, 1.0, shape=(d["act"],), dtype=np.float32)
        super().__init__(cfg, n_envs, seed, obs_space, act_space)

    def _randomize_strategy(self):
        a = self._rng.uniform(-1, 1, size=(self.num_envs, self.core.meta_act_dim))
        self.core.apply_meta(a)

    def _on_reset(self):
        self._randomize_strategy()

    def _obs(self):
        return self.core.obs()

    def _advance(self, actions):
        # occasionally shift strategy to expose the tactician to varied contexts
        if self.core.t > 0 and self.core.t % self.core.chan.coherence == 0:
            self._randomize_strategy()
        return self.core.step(actions)


# --------------------------------------------------------------------------- #
#  Flat PPO straw-man: every variable, every slot
# --------------------------------------------------------------------------- #
class VecFlatEnv(_CoreVecEnv):
    """Combined meta+tactical control every slot."""

    def __init__(self, cfg, n_envs: int = 32, seed: int = 0):
        d = dims(cfg)
        self.meta_dim = d["meta_act"]
        self.low_dim = d["act"]
        act_space = spaces.Box(-1.0, 1.0, shape=(self.meta_dim + self.low_dim,), dtype=np.float32)
        obs_space = spaces.Box(0.0, 1.0, shape=(d["obs"] + d["meta_obs"],), dtype=np.float32)
        super().__init__(cfg, n_envs, seed, obs_space, act_space)

    def _obs(self):
        return np.concatenate([self.core.obs(), self.core.meta_obs()], axis=1)

    def _advance(self, actions):
        actions = np.asarray(actions, float)
        self.core.apply_meta(actions[:, :self.meta_dim])
        obs, r, done, info = self.core.step(actions[:, self.meta_dim:])
        return self._obs(), r, done, info


# --------------------------------------------------------------------------- #
#  Strategist: each meta-step rolls a frozen tactician for H slots
# --------------------------------------------------------------------------- #
class VecMetaEnv(_CoreVecEnv):
    """Meta-level env. One meta-step == H env slots driven by the frozen low policy.

    This is the hot loop of the whole project: with ``meta_timesteps=100000`` and
    ``H=50`` a single strategist training consumes 5,000,000 env slots, each of
    which previously cost one scalar ``env.step`` plus one single-sample
    ``low.predict``. Batching turns those 5M sequential pairs into 5M/n_envs
    batched pairs.
    """

    def __init__(self, cfg, low_model, n_envs: int = 32, enable_ris: bool = True, seed: int = 0):
        d = dims(cfg)
        obs_space = spaces.Box(0.0, 1.0, shape=(d["meta_obs"],), dtype=np.float32)
        act_space = spaces.Box(-1.0, 1.0, shape=(d["meta_act"],), dtype=np.float32)
        super().__init__(cfg, n_envs, seed, obs_space, act_space)
        self.low = FrozenPolicy(low_model)
        self.enable_ris = enable_ris
        self.H = int(cfg["hierarchy"]["H"])
        self.n_meta = max(1, self.core.max_steps // self.H)
        self._m = 0

    def _on_reset(self):
        self._m = 0

    def _obs(self):
        return self.core.meta_obs()

    def _advance(self, actions):
        self.core.apply_meta(actions, enable_ris=self.enable_ris)
        R = np.zeros(self.num_envs)
        done = False
        info = None
        for _ in range(self.H):
            a_low = self.low(self.core.obs())
            _, r, done, info = self.core.step(a_low)
            R += r
            if done:
                break
        self._m += 1
        meta_done = done or self._m >= self.n_meta
        return self._obs(), R / self.H, meta_done, info

    # The episode boundary is the *meta* boundary, so step_wait's auto-reset must
    # fire on meta_done rather than on the underlying env's done flag; _advance
    # already returns meta_done, so no override is needed.
