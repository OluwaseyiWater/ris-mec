# Queue-Aware RIS-MEC Resource Allocation via Hierarchical DRL

Implementation of the RIS-assisted MEC hierarchical DRL
framework.



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


