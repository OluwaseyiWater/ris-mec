"""Regression test: the vectorized core must reproduce the reference physics.

``tests/golden/env_traj.npz`` was recorded from the pre-vectorization
implementation (see ``tests/make_golden.py``). Two properties are checked:

  1. ``RISMECEnv`` (the one-world facade over the batched core) reproduces those
     trajectories -- observations, rewards, and every reported metric.
  2. A batched core with n_envs>1 reproduces, in world 0, exactly what a
     one-world core produces from the same generator stream, so nothing about
     the physics depends on the batch size.

Tolerances are ``rtol=1e-9``. Bit-equality is not achievable (nor wanted): the
cascade is now summed per RIS group rather than per element, which is
algebraically identical but reassociates a floating-point sum.

Run: python -m tests.test_equivalence
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ris_mec.utils import load_config, clone_cfg
from ris_mec.core import BatchedRISMEC
from tests.make_golden import CASES, record, GOLDEN, _apply

RTOL = 1e-9
ATOL = 1e-12


def check_golden():
    path = os.path.join(GOLDEN, "env_traj.npz")
    assert os.path.exists(path), f"missing {path} -- run `python -m tests.make_golden` first"
    ref = np.load(path)
    worst = 0.0
    for label, over, seed in CASES:
        got = record(over, seed)
        for key, cur in got.items():
            exp = ref[f"{label}/{key}"]
            assert cur.shape == exp.shape, f"{label}/{key}: shape {cur.shape} != {exp.shape}"
            denom = np.maximum(np.abs(exp), 1e-30)
            rel = float(np.max(np.abs(cur - exp) / denom))
            worst = max(worst, rel)
            np.testing.assert_allclose(
                cur, exp, rtol=RTOL, atol=ATOL,
                err_msg=f"{label}/{key} diverged from the reference trajectory")
        print(f"[golden] {label:9s} OK")
    print(f"[golden] worst relative deviation over all cases/fields: {worst:.3e}")


class _ScriptedArrivals:
    """Replaces a core's Generator so both cores see the *same* arrival stream.

    Batched and one-world cores necessarily consume a shared Generator at
    different rates, so a scripted table is the only way to isolate the physics
    from the sampling when comparing them.
    """

    def __init__(self, table):
        self._table = table
        self._i = 0

    def exponential(self, mean, size):
        row = self._table[self._i]
        self._i += 1
        return np.tile(row, (size[0], 1))


def check_batch_invariance(n_envs=8, steps=60):
    """World 0 of an n_envs core must match a one-world core driven identically.

    The state of world 0 (geometry, fading, queues, arrivals) is transplanted
    into a fresh one-world core, then both are stepped with the same actions --
    so any mismatch is a genuine batch-size dependence in the physics.
    """
    cfg = clone_cfg(load_config())
    cfg["system"]["K"] = 6
    big = BatchedRISMEC(cfg, n_envs=n_envs, seed=3)
    small = BatchedRISMEC(cfg, n_envs=1, seed=3)
    big.reset(); small.reset()

    # Transplant world 0 of `big` into `small`: geometry, fading, and queues.
    c_b, c_s = big.chan, small.chan
    for attr in ("d_ue_ris", "d_ue_bs", "pl_ue_ris", "pl_ue_bs"):
        setattr(c_s, attr, getattr(c_b, attr)[:1].copy())
    small.queues = big.queues[:1].copy()

    rng = np.random.default_rng(99)
    table = rng.exponential(big.arrival_mean, size=(steps + 2, big.K))
    big.rng = _ScriptedArrivals(table)
    small.rng = _ScriptedArrivals(table)

    worst = 0.0
    for t in range(steps):
        # Re-sync the fading each slot (the two channel models draw separately).
        c_s.g, c_s.hr, c_s.hd = c_b.g[:1].copy(), c_b.hr[:1].copy(), c_b.hd[:1].copy()
        c_s._cache.clear(); c_s.version += 1
        c_s._obs_gain, c_s._obs_gain_ver = None, -1
        small._amp_memo = None
        small.queues = big.queues[:1].copy()

        a = rng.uniform(-1, 1, size=(n_envs, big.act_dim))
        if t % 10 == 0:
            m = rng.uniform(-1, 1, size=(n_envs, big.meta_act_dim))
            big.apply_meta(m); small.apply_meta(m[:1])

        # Observations are compared *before* the step: step() ends with a
        # chan.sample(), and past that point the two channel models have redrawn
        # from their own streams, so the post-step obs is legitimately different.
        pairs = [("obs", big.obs()[:1], small.obs())]
        _, r_b, _, i_b = big.step(a)
        _, r_s, _, i_s = small.step(a[:1])
        pairs += [(k, i_b[k][:1], i_s[k]) for k in i_b]
        pairs.append(("reward", r_b[:1], r_s))
        for name, exp, cur in pairs:
            rel = float(np.max(np.abs(cur - exp) / np.maximum(np.abs(exp), 1e-30)))
            worst = max(worst, rel)
            np.testing.assert_allclose(
                cur, exp, rtol=RTOL, atol=ATOL,
                err_msg=f"batch invariance broken for '{name}' at t={t}")
    print(f"[batch]  n_envs={n_envs} world 0 matches a 1-world core "
          f"(worst rel. dev. {worst:.3e})")


def check_obs_normalization():
    """The dB observation feature must be live, bounded, and CSI-sensitive --
    and must change *only* the observation, not the physics."""
    from ris_mec.core import BatchedRISMEC

    def build(norm, csi=0.0, K=8, seed=0, grp=False):
        cfg = clone_cfg(load_config())
        cfg["system"]["K"] = K
        cfg["channel"]["obs_gain_norm"] = norm
        cfg["channel"]["csi_error_std"] = csi
        cfg["channel"]["obs_group_features"] = grp
        cfg["mec"]["result_queue"] = False      # this check is about the channel block
        core = BatchedRISMEC(cfg, n_envs=64, seed=seed)
        core.reset()
        return core

    # 1. The feature occupies a usable range, unlike the legacy one.
    feats = {}
    for norm in ("legacy", "db"):
        core = build(norm)
        acc = []
        for _ in range(40):
            acc.append(core.obs()[:, :core.K].ravel())
            core.step(np.zeros((64, core.act_dim)))
        feats[norm] = np.concatenate(acc)
    lo, hi = feats["db"].min(), feats["db"].max()
    assert 0.0 <= lo and hi <= 1.0, f"db feature out of [0,1]: [{lo}, {hi}]"
    assert feats["db"].std() > 0.02, f"db feature has no spread (std={feats['db'].std():.2e})"
    assert feats["legacy"].max() < 1e-8, "legacy feature was expected to be inert"
    print(f"[obs] legacy: max={feats['legacy'].max():.2e} std={feats['legacy'].std():.2e} "
          f"(inert)")
    print(f"[obs] db    : range=[{lo:.3f}, {hi:.3f}] mean={feats['db'].mean():.3f} "
          f"std={feats['db'].std():.3f} (live)")

    # 2. The feature responds to CSI estimation error -- the point of the fix.
    def sensitivity(norm):
        a = build(norm, csi=0.0, seed=5).obs()[:, :8]
        b = build(norm, csi=0.2, seed=5).obs()[:, :8]
        return float(np.mean(np.abs(a - b)))
    s_legacy, s_db = sensitivity("legacy"), sensitivity("db")
    assert s_db > 1e-3, f"db feature barely moves with CSI error ({s_db:.2e})"
    assert s_db > 1000 * s_legacy, "db feature is not more CSI-sensitive than legacy"
    print(f"[obs] mean |Δfeature| for csi_error_std 0 -> 0.2: "
          f"legacy={s_legacy:.2e}  db={s_db:.2e}")

    # 3. Physics is untouched: with identical scripted actions, everything the
    #    env reports must be bit-identical between the two normalizations.
    rng = np.random.default_rng(4)
    c_leg, c_db = build("legacy", seed=9), build("db", seed=9)
    for t in range(40):
        a = rng.uniform(-1, 1, size=(64, c_leg.act_dim))
        _, r1, _, i1 = c_leg.step(a)
        _, r2, _, i2 = c_db.step(a)
        np.testing.assert_array_equal(r1, r2, err_msg=f"reward differs at t={t}")
        for k in i1:
            np.testing.assert_array_equal(i1[k], i2[k], err_msg=f"info['{k}'] differs at t={t}")
    print("[obs] physics identical between 'legacy' and 'db' (bit-for-bit)")

    # 4. The per-group cascade block must be present, bounded, live, and far more
    #    CSI-sensitive than the aggregate gain (which averages the error away).
    core = build("db", grp=True)
    K, G = core.K, core.G
    assert core.obs_dim == 4 * K + 1 + 3 * G, "group block missing from obs_dim"
    acc = []
    for _ in range(40):
        acc.append(core.obs()[:, 4 * K + 1:])
        core.step(np.zeros((64, core.act_dim)))
    grp = np.concatenate(acc)
    assert grp.min() >= 0.0 and grp.max() <= 1.0, "group block out of [0,1]"
    assert grp.std() > 0.05, f"group block has no spread (std={grp.std():.2e})"

    def grp_sensitivity(csi):
        a = build("db", csi=0.0, seed=5, grp=True).obs()[:, 4 * 8 + 1:]
        b = build("db", csi=csi, seed=5, grp=True).obs()[:, 4 * 8 + 1:]
        return float(np.mean(np.abs(a - b)))
    s_gain = sensitivity("db")
    s_grp = grp_sensitivity(0.2)
    print(f"[obs] group block: range=[{grp.min():.3f}, {grp.max():.3f}] "
          f"std={grp.std():.3f}")
    print(f"[obs] mean |Δ| for csi 0 -> 0.2: gain block={s_gain:.2e}  "
          f"group block={s_grp:.2e}  ({s_grp/max(s_gain,1e-30):.1f}x more sensitive)")
    # Only ~3x: c_g still sums over K users and M elements, so CSI noise partly
    # averages out here too. The structural gain is that phase information exists
    # at all -- the aggregate magnitude block carries none.
    assert s_grp > 2 * s_gain, "group block is not more CSI-sensitive than the gain block"

    # 5. Adding the group block must not touch the physics either.
    rng2 = np.random.default_rng(11)
    c_off, c_on = build("db", seed=13, grp=False), build("db", seed=13, grp=True)
    for t in range(30):
        a = rng2.uniform(-1, 1, size=(64, c_off.act_dim))
        _, r1, _, i1 = c_off.step(a)
        _, r2, _, i2 = c_on.step(a)
        np.testing.assert_array_equal(r1, r2, err_msg=f"reward differs at t={t}")
        for k in i1:
            np.testing.assert_array_equal(i1[k], i2[k], err_msg=f"info['{k}'] at t={t}")
    print("[obs] physics identical with the group block on and off (bit-for-bit)")


def check_steering():
    """The LoS array response must actually give RIS phase control something to do.

    Without it every element shares one LoS phase, so the per-group cascade
    responses are pre-aligned and a constant phase vector is already near
    optimal. These are the numbers that decide whether the fix worked.
    """
    from ris_mec.core import BatchedRISMEC

    def metrics(steer, K=8, E=256, seed=5, trials=200):
        cfg = clone_cfg(load_config())
        cfg["system"]["K"] = K
        cfg["geometry"]["ris_steering"] = steer
        core = BatchedRISMEC(cfg, n_envs=E, seed=seed)
        core.reset()
        G = core.G
        C = core.chan.cascade_coeff()                       # (E,K,G)
        cg = C.sum(axis=1)                                  # (E,G)
        R = np.abs(np.exp(1j * np.angle(cg)).mean(axis=1))
        circ_std = float(np.sqrt(-2 * np.log(np.maximum(R, 1e-12))).mean())

        a_g, _ = core._apply_output_power_constraint()
        th_const = np.zeros((E, G))
        th_greedy = np.mod(-np.angle(cg), 2 * np.pi)
        s_const = core._composite_gain(a_g, th_const).sum(axis=1)
        s_greedy = core._composite_gain(a_g, th_greedy).sum(axis=1)
        rng = np.random.default_rng(2)
        best = np.full(E, -np.inf); worst = np.full(E, np.inf)
        for _ in range(trials):
            s = core._composite_gain(a_g, rng.uniform(0, 2 * np.pi, size=(E, G))).sum(axis=1)
            best = np.maximum(best, s); worst = np.minimum(worst, s)
        return circ_std, float(s_greedy.mean() / s_const.mean()), float((best / worst).mean())

    off = metrics(False)
    on = metrics(True)
    print(f"{'':22}{'circ std of group phase':>24} {'greedy/const':>14} {'rand best/worst':>17}")
    for tag, m in (("ris_steering: false", off), ("ris_steering: true", on)):
        print(f"{tag:22}{m[0]:>21.3f} rad {m[1]:>13.2f}x {m[2]:>16.1f}x")

    assert on[0] > 5 * off[0], (
        f"steering barely changed the per-group phase spread ({off[0]:.3f} -> {on[0]:.3f} rad)")
    assert on[1] > 2.0, (
        f"greedy co-phasing still has little headroom over a constant phase "
        f"({on[1]:.2f}x); RIS phase control remains near-inert")


def check_group_cascade():
    """The per-group cascade must equal the original per-element einsum."""
    cfg = clone_cfg(load_config())
    cfg["system"]["K"] = 5
    core = BatchedRISMEC(cfg, n_envs=4, seed=11)
    rng = np.random.default_rng(0)
    a_g = rng.uniform(0.5, 3.0, size=(4, core.G))
    th_g = rng.uniform(0, 2 * np.pi, size=(4, core.G))

    fast = core._composite_gain(a_g, th_g)
    g, hr, hd = core.chan.true_channels()
    slow = np.empty_like(fast)
    for e in range(4):
        refl = np.repeat(a_g[e], core.M) * np.exp(1j * np.repeat(th_g[e], core.M))
        slow[e] = np.abs(hd[e] + np.einsum('kn,n,n->k', hr[e], refl, g[e])) ** 2
    np.testing.assert_allclose(fast, slow, rtol=1e-11)
    print("[algebra] per-group cascade == per-element einsum (rtol 1e-11)")


def check_baselines():
    """The rewritten baselines must reproduce the reference metrics.

    ``_greedy_phase`` now reads the cached per-group cascade coefficients instead
    of looping over groups, DPP's per-user power grid is a (K, P) broadcast
    instead of a Python loop, and both read the memoized noise floors -- all
    algebraically identical, so the metrics must land on the recorded values.
    """
    import json
    from ris_mec import baselines
    from tests.make_golden_baselines import CASES as BL_CASES, STEPS, case_cfg

    path = os.path.join(GOLDEN, "baselines.json")
    assert os.path.exists(path), f"missing {path}"
    ref = json.load(open(path))
    worst = 0.0
    for label, over, seed in BL_CASES:
        cfg = case_cfg(over)
        for name, fn in sorted(baselines.BASELINES.items()):
            got = fn(clone_cfg(cfg), seed=seed, steps=STEPS)
            exp = ref[f"{label}/{name}"]
            for k, v in exp.items():
                if not np.isfinite(v):
                    continue
                rel = abs(got[k] - v) / max(abs(v), 1e-30)
                worst = max(worst, rel)
                assert rel < 1e-9, (f"{label}/{name}/{k}: {got[k]!r} != {v!r} "
                                    f"(rel {rel:.3e})")
        print(f"[baseline] {label:8s} all 5 baselines OK")
    print(f"[baseline] worst relative deviation: {worst:.3e}")


if __name__ == "__main__":
    print("=== golden trajectories ==="); check_golden()
    print("=== baselines ==="); check_baselines()
    print("=== batch invariance ==="); check_batch_invariance()
    print("=== observation normalization ==="); check_obs_normalization()
    print("=== RIS array steering ==="); check_steering()
    print("=== cascade algebra ==="); check_group_cascade()
    print("ALL EQUIVALENCE CHECKS PASSED")
