"""
Master experiment driver. Produces every figure/table.

Usage:
    python -m experiments.run_all --profile reduced
    python -m experiments.run_all --profile full --only k_sweep

Profiles trade fidelity for wall-clock. 'reduced' is meant for CPU / constrained
sandboxes and is clearly labeled as reduced-scale; 'full' is the camera-ready
setting to run on a GPU. Results (JSON + CSV + PDF/PNG figures) land in results/.
"""
from __future__ import annotations
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ris_mec.utils import load_config, clone_cfg
from ris_mec.plotstyle import apply_style, style_for
from experiments.common import (aggregate, aggregate_many, run_method, save_json,
                                RESULTS, eval_steps, configure, write_provenance)

# Training budgets. NOTE: `flat` is set to the *environment-interaction* budget the
# hierarchy consumes, low + H*meta, not to a smaller round number. One strategist
# step advances the env by H slots, so a flat agent given `low + meta` steps would
# see H times less experience and lose for that reason alone -- an unfair
# comparison, and the first thing a reviewer recomputes. `_budget_parity` below
# recomputes it from the config's H so the two stay locked together.
PROFILES = {
    "fast":    dict(low=20000,  meta=200,  seeds=[0],    Ks=[4, 6, 8],     eval=400),
    "reduced": dict(low=100000, meta=620,  seeds=[0, 1], Ks=[4, 6, 8, 10], eval=800),
    "full":    dict(low=1000000, meta=9000, seeds=[0,10,16,24,32,42,64,100,128,256], Ks=[4,6,8,10,12,16], eval=2000),
}


def _budget_parity(p, H):
    """Environment slots the hierarchy consumes: E_L + H*E_H."""
    return int(p["low"]) + int(H) * int(p["meta"])

# Filled in by main(); studies read them for their fan-out.
# fixed_total_load: when > 0, per-user arrivals are set to load/K so the total
# offered load does not grow with K. Without it the k_sweep runs into
# congestion collapse: the radio capacity is fixed by B, so served bits
# saturate near 2.5e6 bits/slot and the system is overloaded beyond K ~ 13
# (measured: 94% of offered load served at K=16, 68% at K=24, 52% at K=32).
# Past that point the sweep measures congestion, not controller scalability.
RUN = {"jobs": 1, "threads": 0, "fixed_total_load": 0.0}


def make_cfg(profile, **over):
    p = PROFILES[profile]
    cfg = load_config()
    cfg["training"]["low_timesteps"] = p["low"]
    cfg["training"]["meta_timesteps"] = p["meta"]
    cfg.setdefault("fast", {})["eval_steps"] = p["eval"]
    for dotted, val in over.items():
        d = cfg; ks = dotted.split(".")
        for k in ks[:-1]:
            d = d.setdefault(k, {})
        d[ks[-1]] = val
    # After the overrides, so an H sweep keeps the flat baseline matched.
    cfg["training"]["flat_timesteps"] = _budget_parity(p, cfg["hierarchy"]["H"])
    if RUN.get("fixed_total_load", 0.0) > 0:
        cfg["queue"]["arrival_mean_bits"] = (
            float(RUN["fixed_total_load"]) / int(cfg["system"]["K"]))
    return cfg, p


# ======================================================================= #
#  Study 1: K-sweep (main figure) + training cost
# ======================================================================= #
def study_k_sweep(profile):
    methods = ["HDRL", "Flat", "DPP", "IFP", "SCA", "AO", "FP"]
    p = PROFILES[profile]
    # One flat task list over (K, method, seed) so the pool stays saturated --
    # the whole sweep is independent runs, and their costs differ by ~100x.
    specs = []
    for K in p["Ks"]:
        cfg, _ = make_cfg(profile, **{"system.K": K})
        for m in methods:
            specs.append((K, m, clone_cfg(cfg), p["seeds"]))
    t0 = time.time()
    aggs = aggregate_many(specs, jobs=RUN["jobs"], threads=RUN["threads"],
                          progress=_progress("k_sweep", t0))
    rows = []
    for agg in aggs:
        agg["K"] = agg.pop("_label")
        rows.append(agg)
        print(f"[k_sweep] K={agg['K']:2d} {agg['method']:5s} "
              f"lat={agg['latency']:.4f}±{agg['latency_ci']:.4f} "
              f"EE={agg['ee_mbit_j']:.2f}±{agg['ee_mbit_j_ci']:.2f} "
              f"q={agg['queue_mean']:.2e}", flush=True)
    save_json("k_sweep.json", rows)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "k_sweep.csv"), index=False)
    plot_k_sweep(rows, profile)
    plot_training_cost(rows, profile)
    return rows


def _progress(tag, t0):
    def cb(done, total):
        el = time.time() - t0
        eta = el / done * (total - done)
        print(f"[{tag}] {done}/{total} runs  elapsed {el/60:.1f}m  eta {eta/60:.1f}m",
              flush=True)
    return cb


def plot_k_sweep(rows, profile):
    apply_style()
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, (col, ci, ylab) in zip(
        axes, [("latency", "latency_ci", "Avg. latency (s)"),
               ("ee_mbit_j", "ee_mbit_j_ci", "Energy efficiency (Mbit/J)")]):
        for m in df["method"].unique():
            d = df[df.method == m].sort_values("K")
            st = style_for("Flat PPO" if m == "Flat" else m)
            ax.errorbar(d["K"], d[col], yerr=d[ci], label=("Flat PPO" if m == "Flat" else m),
                        capsize=2, **st)
        ax.set_xlabel("Number of users $K$"); ax.set_ylabel(ylab)
    axes[0].legend(ncol=2, fontsize=6.5, loc="best")
    #fig.suptitle(f"HDRL vs baselines ({profile}-scale)", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS, f"fig_k_sweep.{ext}"))
    plt.close(fig)


def plot_training_cost(rows, profile):
    apply_style()
    df = pd.DataFrame(rows)
    d = df[df.K == df.K.max()]
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    order = ["HDRL", "Flat", "DPP", "IFP"]
    d = d[d.method.isin(order)]
    ax.bar(d["method"], d["train_time"],
           color=[style_for("Flat PPO" if m == "Flat" else m)["color"] for m in d["method"]])
    ax.set_ylabel("Train/solve time (s)")
    #ax.set_title(f"Compute at $K$={int(df.K.max())} ({profile})")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS, f"fig_training_cost.{ext}"))
    plt.close(fig)


# ======================================================================= #
#  Study 2: Ablations (active vs passive RIS, H sweep, G sweep)
# ======================================================================= #
def study_ablation(profile):
    p = PROFILES[profile]; seeds = p["seeds"][:1]
    specs = []
    # active vs passive RIS at K=8
    cfg, _ = make_cfg(profile, **{"system.K": 8})
    for m, label in [("HDRL", "active"), ("HDRL-passive", "passive")]:
        specs.append((("ris", label), m, clone_cfg(cfg), seeds))
    # H sweep at K=8 (the tactician is H-agnostic, so --cache trains it once)
    for H in [25, 50, 100]:
        c, _ = make_cfg(profile, **{"system.K": 8, "hierarchy.H": H})
        specs.append((("H", H), "HDRL", clone_cfg(c), seeds))
    # G sweep at K=8 (N=128 divisible)
    for G in [4, 8, 16]:
        c, _ = make_cfg(profile, **{"system.K": 8, "system.G": G})
        specs.append((("G", G), "HDRL", clone_cfg(c), seeds))

    t0 = time.time()
    aggs = aggregate_many(specs, jobs=RUN["jobs"], threads=RUN["threads"],
                          progress=_progress("ablation", t0))
    out = {"ris": [], "H": [], "G": []}
    for agg in aggs:
        group, label = agg.pop("_label")
        agg["config" if group == "ris" else group] = label
        out[group].append(agg)
        print(f"[ablation {group}={label}] lat={agg['latency']:.4f} "
              f"EE={agg['ee_mbit_j']:.2f}", flush=True)
    save_json("ablation.json", out)
    plot_ablation(out, profile)
    return out


def plot_ablation(out, profile):
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    # RIS on/off
    d = pd.DataFrame(out["ris"])
    axes[0].bar(d["config"], d["latency"], color=[style_for("HDRL")["color"], "#888"])
    axes[0].set_ylabel("Avg. latency (s)"); axes[0].set_title("")
    # H sweep
    d = pd.DataFrame(out["H"]).sort_values("H")
    axes[1].plot(d["H"], d["latency"], **style_for("HDRL"))
    axes[1].set_xlabel("Meta-interval $H$"); axes[1].set_ylabel("Avg. latency (s)"); axes[1].set_title("")
    # G sweep
    d = pd.DataFrame(out["G"]).sort_values("G")
    axes[2].plot(d["G"], d["latency"], **style_for("DPP"))
    axes[2].set_xlabel("Groups $G$"); axes[2].set_ylabel("Avg. latency (s)"); axes[2].set_title("")
    #fig.suptitle(f"Ablations ({profile}-scale)", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS, f"fig_ablation.{ext}"))
    plt.close(fig)


# ======================================================================= #
#  Study 3: Pareto frontier (vary lambda_E) 
# ======================================================================= #
def study_pareto(profile):
    p = PROFILES[profile]; seed = p["seeds"][0]
    lambdas = [0.3, 1.0, 3.0] if profile != "full" else [0.1, 0.3, 1.0, 3.0, 10.0]
    specs = []
    for lam in lambdas:
        cfg, _ = make_cfg(profile, **{"system.K": 8, "hierarchy.lambda_E": lam})
        for m in ["HDRL", "DPP", "IFP"]:
            specs.append((lam, m, clone_cfg(cfg), [seed]))
    t0 = time.time()
    aggs = aggregate_many(specs, jobs=RUN["jobs"], threads=RUN["threads"],
                          progress=_progress("pareto", t0))
    rows = []
    for agg in aggs:
        agg["lambda_E"] = agg.pop("_label")
        rows.append(agg)
        print(f"[pareto] lam={agg['lambda_E']} {agg['method']}: "
              f"lat={agg['latency']:.4f} EE={agg['ee_mbit_j']:.2f}", flush=True)
    save_json("pareto.json", rows)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "pareto.csv"), index=False)
    plot_pareto(rows, profile)
    return rows


def plot_pareto(rows, profile):
    apply_style()
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for m in df["method"].unique():
        d = df[df.method == m].sort_values("latency")
        st = style_for(m)
        ax.plot(d["latency"], d["ee_mbit_j"], label=m, **st)
    ax.set_xlabel("Avg. latency (s)"); ax.set_ylabel("Energy efficiency (Mbit/J)")
    #ax.set_title(f"Latency-EE Pareto ({profile})")
    ax.legend()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS, f"fig_pareto.{ext}"))
    plt.close(fig)


# ======================================================================= #
#  Study 4: Robustness (CSI error, phase quantization) 
#  Trains one HDRL model, evaluates under impairments (no retrain).
# ======================================================================= #
def study_robustness(profile):
    from ris_mec.evaluate import evaluate_hdrl
    from experiments.common import _hdrl_models, CACHE_DIR
    from ris_mec.modelcache import ModelCache
    import experiments.common as _common
    p = PROFILES[profile]; seed = p["seeds"][0]
    cfg, _ = make_cfg(profile, **{"system.K": 8})
    # Same (cfg, seed) as the K=8 HDRL run in k_sweep: with --cache this reuses
    # that model instead of retraining it. The impairments below are applied at
    # evaluation only, so they do not change the cache key.
    low, meta = _hdrl_models(cfg, seed, True,
                             ModelCache(CACHE_DIR, enabled=_common._CACHE_ENABLED))
    rows = []
    steps = eval_steps(cfg)
    for csi in [0.0, 0.05, 0.1, 0.2]:
        c = clone_cfg(cfg); c["channel"]["csi_error_std"] = csi
        r = evaluate_hdrl(c, low, meta, seed=seed, steps=steps); r.update(impair="csi_error", value=csi)
        rows.append(r); print(f"[robust csi={csi}] lat={r['latency']:.4f} EE={r['ee_mbit_j']:.2f}", flush=True)
    for bits in [1, 2, 3, 0]:   # 0 = continuous
        c = clone_cfg(cfg); c["channel"]["phase_bits"] = bits
        r = evaluate_hdrl(c, low, meta, seed=seed, steps=steps); r.update(impair="phase_bits", value=bits)
        rows.append(r); print(f"[robust bits={bits}] lat={r['latency']:.4f} EE={r['ee_mbit_j']:.2f}", flush=True)
    save_json("robustness.json", rows)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "robustness.csv"), index=False)
    plot_robustness(rows, profile)
    return rows


def plot_robustness(rows, profile):
    apply_style()
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    d = df[df.impair == "csi_error"].sort_values("value")
    axes[0].plot(d["value"], d["ee_mbit_j"], **style_for("HDRL"))
    axes[0].set_xlabel("CSI error std"); axes[0].set_ylabel("EE (Mbit/J)"); axes[0].set_title("")
    d = df[df.impair == "phase_bits"].copy()
    d["value"] = d["value"].replace(0, 8)  # show continuous as 8-bit-equivalent
    d = d.sort_values("value")
    axes[1].plot(d["value"], d["ee_mbit_j"], **style_for("DPP"))
    axes[1].set_xlabel("RIS phase bits (8=cont.)"); axes[1].set_ylabel("EE (Mbit/J)"); axes[1].set_title("")
    #fig.suptitle(f"Robustness ({profile}-scale)", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS, f"fig_robustness.{ext}"))
    plt.close(fig)


# ======================================================================= #
#  Study 5: Lyapunov trade-off (sweep V)
# ======================================================================= #
def study_lyapunov(profile):
    """Trace the drift-plus-penalty trade-off the analysis predicts.

    Reward = -drift + V*(EE term) - V*lambda_E*L, so V is the knob between queue
    stability and efficiency: the theory gives an O(1/V) optimality gap bought
    with an O(V) queue bound. This study is what turns that from an asserted
    property into a measured one -- and it is also the check that the agent is
    not "winning" on EE simply by refusing to serve its users, which shows up
    here as served_ratio falling below 1 as V grows.
    """
    p = PROFILES[profile]
    Vs = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    specs = []
    for V in Vs:
        cfg, _ = make_cfg(profile, **{"system.K": 8, "hierarchy.dpp_V": V})
        specs.append((V, "HDRL", clone_cfg(cfg), p["seeds"]))
    t0 = time.time()
    aggs = aggregate_many(specs, jobs=RUN["jobs"], threads=RUN["threads"],
                          progress=_progress("lyapunov", t0))
    rows = []
    for agg in aggs:
        agg["V"] = agg.pop("_label")
        rows.append(agg)
        print(f"[lyapunov] V={agg['V']:<5g} EE={agg['ee_mbit_j']:7.2f} "
              f"queue={agg['queue_mean']:.2e} served={agg['served_ratio']:.2f}x "
              f"delay={agg['delay_total']:.4f}s", flush=True)
    save_json("lyapunov.json", rows)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "lyapunov.csv"), index=False)
    plot_lyapunov(rows, profile)
    return rows


def plot_lyapunov(rows, profile):
    apply_style()
    df = pd.DataFrame(rows).sort_values("V")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    ax = axes[0]
    st = style_for("HDRL")
    ax.errorbar(df["V"], df["ee_mbit_j"], yerr=df["ee_mbit_j_ci"], capsize=2, **st)
    ax.set_xscale("log"); ax.set_xlabel("Lyapunov weight $V$")
    ax.set_ylabel("Energy efficiency (Mbit/J)", color=st.get("color"))
    ax2 = ax.twinx()
    ax2.errorbar(df["V"], df["queue_mean"], yerr=df["queue_mean_ci"], capsize=2,
                 **style_for("DPP"))
    ax2.set_ylabel("Mean backlog (bits)", color=style_for("DPP").get("color"))
    ax2.set_yscale("log")

    ax = axes[1]
    ax.errorbar(df["delay_total"], df["ee_mbit_j"],
                xerr=df["delay_total_ci"], yerr=df["ee_mbit_j_ci"], capsize=2, **st)
    for _, r in df.iterrows():
        ax.annotate(f"$V$={r['V']:g}", (r["delay_total"], r["ee_mbit_j"]),
                    fontsize=5.5, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("End-to-end delay (s)"); ax.set_ylabel("Energy efficiency (Mbit/J)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS, f"fig_lyapunov.{ext}"))
    plt.close(fig)


STUDIES = {"k_sweep": study_k_sweep, "ablation": study_ablation,
           "pareto": study_pareto, "robustness": study_robustness,
           "lyapunov": study_lyapunov}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="reduced", choices=list(PROFILES))
    ap.add_argument("--only", default="all")
    ap.add_argument("--jobs", type=int, default=1,
                    help="runs executed concurrently in separate processes "
                         "(the (K, method, seed) grid is independent). "
                         "0 = one per CPU core.")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch threads per worker; keep jobs*threads <= cores")
    ap.add_argument("--fixed-total-load", type=float, default=0.0,
                    help="hold the TOTAL offered load fixed at this many bits/slot "
                         "by setting per-user arrivals to load/K. Without it the "
                         "sweep saturates near K=13 and large-K points measure "
                         "congestion rather than controller scalability. "
                         "1.2e6 reproduces the K=8 load.")
    ap.add_argument("--cache", action="store_true",
                    help="reuse trained models across studies and reruns "
                         f"(results/model_cache/)")
    args = ap.parse_args()

    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    RUN["jobs"], RUN["threads"] = jobs, max(1, args.threads)
    RUN["fixed_total_load"] = args.fixed_total_load
    configure(cache=args.cache)
    if jobs == 1:
        # Single-process: let torch use the box, nothing else is competing.
        try:
            import torch; torch.set_num_threads(max(1, args.threads or (os.cpu_count() or 1)))
        except Exception:
            pass

    todo = list(STUDIES) if args.only == "all" else args.only.split(",")
    print(f"profile={args.profile} jobs={jobs} threads/worker={RUN['threads']} "
          f"cache={'on' if args.cache else 'off'}", flush=True)
    t0 = time.time()
    for name in todo:
        print(f"\n===== STUDY: {name} ({args.profile}) =====", flush=True)
        STUDIES[name](args.profile)
    write_provenance(args.profile, todo)
    print(f"\nALL DONE in {time.time()-t0:.0f}s. Results in {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
