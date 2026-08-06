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
ris_mec/         # library: channels, env, hierarchy, flat, baselines, objectives, metrics, evaluate
experiments/     # run_all.py (+ thin exp_*.py wrappers) -> results/
configs/         # default.yaml = single source of truth / parameter table
tests/           # smoke_test.py (physics + baselines + tiny train/eval)
results/         # JSON, CSV, and PDF/PNG figures
```

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce

```bash
# fast sanity (seconds)
python -m tests.smoke_test

# reduced-scale (CPU-friendly)
python -m experiments.run_all --profile reduced

# camera-ready (run on a GPU box)
python -m experiments.run_all --profile full
# or a single study:
python -m experiments.run_all --profile full --only k_sweep
```


## Studies

| Study | Answers | Output |
|---|---|---|
| `k_sweep` | scalability: latency & EE vs users, with 95% CI | `fig_k_sweep`, `fig_training_cost` |
| `ablation` | active-vs-passive RIS, H sweep, G sweep | `fig_ablation` |
| `pareto` | latency–EE Pareto by sweeping λ_E | `fig_pareto` |
| `robustness` | imperfect CSI, RIS phase quantization | `fig_robustness` |

## Config knobs worth knowing

- `hierarchy.objective`: `dpp` (default) | `ee_latency` | `dinkelbach`
- `hierarchy.lambda_E`, `hierarchy.dpp_V`: latency↔EE trade-off / Lyapunov weight
- `active_ris.enable_amplification`, `power.ris_amp_max`, `power.ris_out_power_max`
- `channel.csi_error_std`, `channel.phase_bits`, `channel.csi_delay_slots`: robustness
