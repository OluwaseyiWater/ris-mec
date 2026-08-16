# Scalable RIS-MEC Resource Allocation via Hierarchical DRL

Implementation of the RIS-assisted MEC hierarchical DRL
framework.

## Changes effected from our previous notebook

1. **The RIS is now actually *active* (Gap A).** The reflection coefficient is
   `a_g · e^{jθ_g}` with amplitude gain `a_g` (was unit-modulus/passive). The
   active-RIS amplifier noise enters the SINR — uplink floor aggregated through
   `g` (`ζ_g`), downlink through `h_{r,k}` (`ξ_{g,k}`). A per-element
   output-power stability constraint and a global RIS power budget are enforced,
   and `P_RIS = Σ_g (N/G)(p_c + p_a a_g²)` is counted in the energy.
2. **HDRL now optimizes the P1 objective (Gap B).** Reward comes from
   `objectives.py` — selectable normalized-EE, Dinkelbach, or Lyapunov
   drift-plus-penalty — so the trained quantity matches the reported EE 
3. **Reproducibility & fidelity (Gap C).** Single seeded RNG; RIS-standard
   reference-distance pathloss; explicit queue update `Q_k(t+1)=max(Q_k−b_k,0)+A_k`;
   one config file (`configs/default.yaml`) that *is* the parameter table.

New baseline: **Lyapunov drift-plus-penalty (DPP)** — alongside IFP and the heuristics.

## Layout

```
ris_mec/         # library: channels, core, env, vec_env, hierarchy, flat,
                 #          baselines, objectives, metrics, evaluate, modelcache
experiments/     # run_all.py (+ thin exp_*.py wrappers) -> results/
configs/         # default.yaml = single source of truth / parameter table
tests/           # smoke_test.py (physics + baselines + tiny train/eval)
                 # test_equivalence.py (vectorized core == reference physics)
results/         # JSON, CSV, and PDF/PNG figures
```

`core.py` holds the physics, vectorized over a batch of independent worlds;
`env.py` is a one-world facade over it and `vec_env.py` the SB3 vector envs used
for training. See **Vectorization** below.

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce

```bash
# fast sanity (seconds)
python -m tests.smoke_test

# the vectorized core reproduces the reference physics (seconds)
python -m tests.test_equivalence

# reduced-scale (CPU-friendly)
python -m experiments.run_all --profile reduced

# camera-ready (run on a GPU box)
python -m experiments.run_all --profile full --jobs 8 --cache
# or a single study:
python -m experiments.run_all --profile full --only k_sweep --jobs 8 --cache
```

`--jobs N` runs N of the independent (K, method, seed) runs in separate
processes; `--threads` is torch threads per worker (keep `jobs*threads <=`
cores). `--cache` stores trained policies under `results/model_cache/`, keyed by
the training-relevant config plus seed, so reruns and overlapping studies (the
`robustness` study uses the same K=8 model `k_sweep` trains) do not retrain.


## Studies

| Study | Answers | Output |
|---|---|---|
| `k_sweep` | scalability: latency & EE vs users, with 95% CI | `fig_k_sweep`, `fig_training_cost` |
| `ablation` | active-vs-passive RIS, H sweep, G sweep | `fig_ablation` |
| `pareto` | latency–EE Pareto by sweeping λ_E | `fig_pareto` |
| `robustness` | imperfect CSI, RIS phase quantization | `fig_robustness` |

## Vectorization

The full profile used to be dominated by Python-level per-step overhead rather
than by arithmetic. Two measurements on a 4-thread CPU made that concrete:

* `env.step()` cost ~290 µs at K=8 and only ~477 µs at K=64 — i.e. almost
  independent of problem size, because a slot is ~80 tiny numpy calls whose cost
  is dispatch, not flops;
* `low.predict()` cost ~294 µs for a *single* observation, more than the forward
  pass itself (~148 µs), the rest being SB3's per-call bookkeeping.

Those two costs are paid once per env slot, and the strategist pays them
5,000,000 times per training run: `meta_timesteps: 100000` counts *meta*-steps,
each of which rolls the frozen tactician for `H = 50` slots.

**The fix is a batch axis.** `ris_mec/core.py` advances `n_envs` independent
worlds through one set of numpy calls (`hr` is `(E, K, N)`, queues are `(E, K)`,
…), and `ris_mec/vec_env.py` exposes that to SB3 as a real `VecEnv`, so the
policy sees one batched forward pass instead of `E` single-sample ones. The
fixed per-call overhead is amortized over the batch:

| | original | vectorized (`n_envs=32`) |
|---|---|---|
| physics, K=8 | 3.4k slots/s | 74k slots/s (**21×**) |
| strategist inner loop, K=8 | 1.3k slots/s | 37k slots/s (**29×**) |
| strategist inner loop, K=64 | 1.5k slots/s | 13k slots/s (**10×**) |
| PPO tactician (collect+update), K=8 | 0.7k slots/s | 8.7k slots/s (**13×**) |
| model-based DPP baseline, K=64 | 3321 µs/slot | 982 µs/slot (**3.4×**) |

End to end, one full-profile HDRL run (200k tactician slots + 100k meta-steps =
5M strategist slots), measured on a 4-thread CPU:

| | original | vectorized |
|---|---|---|
| K=8 | ~4181 s | **282 s** (14.8×) |
| K=32 | ~4181 s † | **379 s** (11.0×) |

† the K=32 "original" figure is extrapolated from the original code's *measured*
K=8 throughput, so it understates the real speed-up (the original got slower with
K; the vectorized version much less so).

The full-profile `k_sweep` is 120 HDRL runs plus 120 flat-PPO runs and 600
baseline runs, i.e. roughly 155 CPU-hours before and ~14 after — and since every
run is independent, `--jobs 8` takes that to a few hours.

Alongside the batch axis, work that does not change every slot was hoisted:

* the channel is redrawn only every `coherence_slots`, so **every reduction over
  the element axis `N` is cached per fading block** — `|hr|²` is now formed once
  (as `re²+im²`, no hypot) and `xi_gk`, `inc_g` and the observation gain all fall
  out of that single intermediate;
* the cascade is evaluated **per RIS group instead of per element**. Since
  `a_g·e^{jθ_g}` is constant within a group,
  `Σ_n h_r·refl·g == Σ_g refl_g·C_{k,g}` with `C_{k,g} = Σ_{n∈g} h_r[k,n]·g[n]`
  cached per block — an `O(K·N)` einsum per slot becomes an `O(K·G)` matvec
  (16× less work at the default `N=128, G=8`), algebraically exactly equal;
* the active-RIS amplitude clip depends only on `(a_g, fading block)`, both slow,
  so it is memoized instead of recomputed every slot;
* the delayed-CSI history no longer copies `(g, hr, hd)` every slot — it is only
  maintained when `csi_delay_slots > 0`, and stores references (the arrays are
  rebound, never mutated).

`n_envs: 1` in `configs/default.yaml` recovers the original single-world
behaviour exactly. Evaluation deliberately stays single-world so the reported
means and across-seed CIs remain comparable to earlier runs; only the way the
action is obtained changed (`FrozenPolicy` instead of `model.predict`, which is
semantically identical).

### What this does and does not change

`tests/test_equivalence.py` pins the physics against trajectories recorded from
the pre-vectorization code (`tests/golden/env_traj.npz`) across eight
configurations — active/passive RIS, blocked direct link, CSI error, delayed CSI
with phase quantization, and all three reward objectives. **Worst relative
deviation: 1.5e-14**, and world 0 of a batched core matches a one-world core
exactly. Bit-equality is not achievable because the cascade sum is reassociated
by group; nothing else moved.

## Observation scaling of the channel gain

The agent's channel feature used to be `gain / (1 + gain)`. Raw composite gains
are ~1e-12 to 1e-11 (i.e. −120 dB to −79 dB after pathloss), so `1 + gain == 1`
to within rounding and that "squashing" was the identity: the feature arrived at
the policy as ~1e-11 while the queue / alpha / f blocks are order 0.1–1. The
policy was effectively **blind to CSI**, which is why the CSI-robustness sweep
came out flat — the curve measured an agent that could not see the impairment,
not a system that tolerated it.

`channel.obs_gain_norm` now defaults to `"db"`:

```
clip((10*log10(gain) - obs_gain_db_lo) / (obs_gain_db_hi - obs_gain_db_lo), 0, 1)
```

Channel gain is a log-scale quantity, so decibels are the natural unit. The
default window `[-125, -75] dB` was chosen from the measured distribution over
K ∈ {4,…,64} and both blockage settings (raw range −120.5 dB to −79.1 dB), and
covers it with no clipping. **Re-tune `obs_gain_db_lo/hi` if you change the
geometry**, since the window is calibrated to the pathloss it produces.

| feature | legacy | db |
|---|---|---|
| range | max 4.4e-11 | [0.09, 0.43] |
| std | 5.6e-12 | 0.070 |
| mean \|Δ\| for csi_error_std 0 → 0.2 | 3.7e-13 | 6.5e-03 |

`channel.obs_gain_norm: "legacy"` reproduces the old behaviour exactly.

### Why that alone did not un-flatten the robustness curve

Rescaling the feature was necessary but not sufficient. Two measurements explain
the rest:

1. **The aggregate gain is immune to per-element CSI error by construction.**
   The feature averages `|hr|²` over all `N` elements, and the estimation error is
   i.i.d. per element, so it averages away: what survives is a nearly
   deterministic bias of `≈10·log10(1+σ²)` dB, with a random part whose spread
   falls as `1/√N` (measured: 0.35 dB std at N=32, 0.09 dB at N=512). At σ=0.2
   that is a ~0.34 dB shift on a 50 dB window — 0.7% of the feature range.

2. **The phase decision was open-loop.** The tactician emits `G` RIS phases, but
   its observation contained *no per-group and no phase information at all* —
   only `K` scalar magnitudes averaged over every element. `theta_g` therefore
   could not be channel-aware, so neither CSI error nor phase quantization could
   degrade it. And phase choice matters enormously: over 200 random phase
   vectors, `Σ_k |h_k|²` varies by a mean factor of **277×** (max 1528×).

So `channel.obs_group_features` (default on) adds the quantity the decision
actually needs — the user-aggregated per-group cascade response

```
c_g = Σ_k Σ_{n∈g} hr[k,n]·g[n]
```

exposed as `|c_g| / max_g |c_g|` plus `cos`/`sin` of its phase (3G dims, mapped
to [0,1]). This is exactly what the greedy co-phasing baseline maximizes with
`theta_g = -angle(c_g)` (see `baselines._greedy_phase`), so the agent can now at
least learn that solution. It is scale-invariant, so unlike the gain feature it
needs no geometry-dependent calibration. **It is built from the *observed*
(delayed/noisy) channel**, so CSI impairments finally reach the decision they are
meant to impair.

`tests/test_equivalence.py` asserts that all three observation configurations
produce **bit-identical** rewards, queues, latency, energy and EE under identical
scripted actions — every change here is observation-only. They do of course
change what the policy *learns*, so **all learned-method results need
re-running**; the baselines (AO/SCA/FP/IFP/DPP) are unaffected, as they read the
true channel directly.

The group block is worth having on its own merits — trained HDRL at K=8, 3 seeds:

| observation | EE (Mbit/J) |
|---|---|
| `legacy` | 78.79 ± 2.32 |
| `db` | 78.14 ± 2.30 |
| `db` + group block | **83.57 ± 1.48** |

## RIS array response (`geometry.ris_steering`)

The observation fixes above were necessary but did not un-flatten the robustness
curves, because the cause was in the channel model. `_rician` built the LoS term
as `sqrt(K/(K+1))` — a real positive constant, *identical for every RIS element*.

That is not a simplification but an inconsistency: it asserts `sin φ = 0` on both
hops, i.e. the RIS is broadside to the BS **and** to every UE simultaneously,
which cannot hold for users at different positions. It also leaves every element
already co-phased, so RIS phase control — the paper's central mechanism — had
essentially nothing to do.

The LoS term is now the standard array response vector, for a uniform linear
array of spacing `d/λ`:

```
a_n(φ) = exp(-j · 2π · (d/λ) · n · sin φ)
```

with a fixed RIS→BS departure angle (`bs_ris_angle_deg`) and a per-user, per-world
UE→RIS arrival angle drawn from `±ue_angle_spread_deg`. The cascade of the two
hops then carries `exp(-j·2π·(d/λ)·n·(sin φ_k + sin φ_B))`, so different users
prefer different phase profiles — the multi-user RIS trade-off the hierarchy
exists to resolve. Element modulus stays 1, so the per-link power budget and the
pathloss model are unchanged.

Measured effect (K=8, 256 worlds):

| | circular std of per-group phase | greedy co-phasing vs constant phase |
|---|---|---|
| `ris_steering: false` (old) | 0.101 rad (5.8°) | **1.01×** |
| `ris_steering: true` (default) | **1.678 rad** | **3.87×** |

So RIS phase control goes from ~1% headroom to nearly 4×. As a side effect the
per-group observation block becomes much more informative — its sensitivity to
CSI error rises from 3.0× to **13.9×** that of the aggregate gain block.

`tests/test_equivalence.check_steering` asserts both metrics, so a future change
that quietly re-flattens the channel will fail the suite.

**This is a physics change**, unlike the observation fixes: it moves the
baselines too, since AO/SCA/FP/IFP/DPP all read the true channel. Everything must
be re-run. `ris_steering: false` reproduces the old model exactly, and the golden
trajectories in `tests/golden/` are pinned to it so they keep testing the
vectorization rather than the physics.

### Companion change: `queue.arrival_mean_bits` 3.0e5 → 1.5e5

Under the old model the RIS cascade combined coherently by default, which is what
made `3.0e5` drainable. With a physical array response the per-user capacity
drops and that arrival rate is no longer stabilizable — measured with the
queue-aware DPP baseline (K=8, 1500 slots):

| `ris_steering` | `arrival_mean_bits` | mean queue | p95 queue |
|---|---|---|---|
| false | 3.0e5 | 4.87e5 | 6.95e5 |
| **true** | **3.0e5** | **2.61e6** | **5.29e6** |
| true | 2.0e5 | 3.57e5 | 5.54e5 |
| **true** | **1.5e5** | **2.45e5** | **3.61e5** |

`1.5e5` restores the regime the config comment has always described, with queue
statistics almost identical to the pre-fix defaults (2.45e5 vs 2.34e5 mean).
Revert both lines together if you want the old model back.

### Still open: the operating point is optimistic

The array response is fixed, but only part of its benefit reaches the reward,
because the link operates deep in the logarithmic regime. Greedy co-phasing vs a
constant phase vector, with `ris_steering: true`:

| `Pk_ul_max` | median SINR | gain on \|h\|² | gain on **offloaded bits** |
|---|---|---|---|
| 0.2 (default) | 27.8 dB | 3.87× | **1.04×** |
| 2e-3 | 7.8 dB | 3.87× | 1.19× |
| 2e-4 | −2.2 dB | 3.87× | 1.47× |

`log2(1+SINR)` compresses a 5.9 dB gain into 4% more bits at 28 dB. **The RIS
robustness sweeps therefore remain flat** (spread < 1%) even after both fixes:
there is nothing left for the *agent* to get wrong, but also little for the
metrics to register.

A second, independently checkable inconsistency drives that operating point:
`geometry.pl0_db: 30.0` is the free-space reference loss at ≈1 GHz, while
`geometry.fc_ghz: 3.5` implies **43.3 dB** at `d0 = 1 m`
(`20·log10(4πd₀/λ)`, λ = 8.57 cm). The config understates the reference loss by
13.3 dB per hop, **26.6 dB on the UE→RIS→BS cascade**. Correcting it moves the
system to ~1 dB SINR where co-phasing is worth 1.35× on bits — a regime where the
RIS genuinely matters — but at that link budget the arrival rate needs rebalancing
again (per-user capacity falls to ~3e4 bits/slot). That is a coordinated
re-design of the link budget rather than a one-line fix, so it is left as a
deliberate decision.

### `meta_timesteps` and the strategist's budget

`training.meta_timesteps` counts **meta-steps**, and each meta-step rolls the
tactician for `H = 50` slots. So `meta_timesteps: 100000` is *5,000,000*
environment slots for the strategist against `low_timesteps: 200000` for the
tactician — and against `flat_timesteps: 300000` for the flat-PPO baseline HDRL
is compared to. **HDRL therefore gets 17.3× the environment interaction of the
baseline it is meant to beat**, which is the accounting a reviewer will check.

Measured plateau (K=8, tactician fixed at 200k slots, 3 seeds):

| `meta_timesteps` | env slots | HDRL/Flat budget | EE (Mbit/J) | latency (s) | train |
|---|---|---|---|---|---|
| 2,000 | 100,000 | 1.0× | 83.446 ± 1.505 | 0.06614 | 6 s |
| 12,500 | 625,000 | 2.8× | 83.520 ± 1.500 | 0.06627 | 27 s |
| 25,000 | 1,250,000 | 4.8× | 83.562 ± 1.502 | 0.06637 | 47 s |
| 100,000 | 5,000,000 | 17.3× | 83.711 ± 1.497 | 0.06686 | 182 s |

50× the strategist budget buys **+0.32% EE** (and slightly *worse* latency).
The effect is monotone and consistent across seeds, but it is 5× smaller than
the across-seed CI.

The reason is that the strategist has little leverage here. Same frozen
tactician, three strategy sources:

| strategy | EE (Mbit/J) | P_RIS |
|---|---|---|
| trained strategist | 83.565 ± 1.505 | 2.000 W |
| random meta-actions | 82.283 ± 1.543 | 2.000 W |
| **fixed default** (no strategist at all) | **83.426 ± 1.504** | 2.000 W |

The trained strategist beats *no strategist* by **0.17%**. Note `P_RIS` sits
exactly at `power.ris_power_budget = 2.0 W` under every strategy: the global
budget constraint rescales whatever `a_g` the strategist chooses, so one of its
three strategic variables is being normalized away, and only `alpha` and `f`
carry any of its influence.

Cutting `meta_timesteps` is therefore safe and cheap — but it treats the symptom.
The finding to act on is that the strategist consumes ~94% of the compute and a
17× budget advantage to add 0.17% over a fixed default.

Training *dynamics* do change, unavoidably: PPO now sees `n_envs` worlds per
rollout. The sample budget (`total_timesteps`) and the GAE horizon (`n_steps`)
are held fixed, and `batch_size` is scaled by `n_envs` (`vec.scale_batch`) so the
number of gradient steps per environment sample is also unchanged — each gradient
is simply estimated from a proportionally larger, lower-variance batch. Set
`vec.scale_batch: false` to keep the original minibatch size instead.

## Config knobs worth knowing

- `hierarchy.objective`: `dpp` (default) | `ee_latency` | `dinkelbach`
- `hierarchy.lambda_E`, `hierarchy.dpp_V`: latency↔EE trade-off / Lyapunov weight
- `active_ris.enable_amplification`, `power.ris_amp_max`, `power.ris_out_power_max`
- `channel.csi_error_std`, `channel.phase_bits`, `channel.csi_delay_slots`: robustness
- `vec.n_envs` / `vec.meta_n_envs`: parallel worlds per numpy call (throughput knob)
- `vec.scale_batch`, `vec.device`: PPO batch scaling and torch device
