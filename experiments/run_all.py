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
from experiments.common import aggregate, run_method, save_json, RESULTS, eval_steps

PROFILES = {
    "fast":    dict(low=4000,  meta=2000,  flat=4000,  seeds=[0],       Ks=[4, 6, 8],        eval=400),
    "reduced": dict(low=6000,  meta=2500,  flat=9000, seeds=[0, 1],    Ks=[4, 6, 8, 10],    eval=800),
    "full":    dict(low=200000, meta=100000, flat=300000, seeds=[0,4,10,16,24,42,50,64,100,128,200,256,480,512,1024], Ks=[4,6,8,10,12,16,32,64], eval=2000),
}


def make_cfg(profile, **over):
    p = PROFILES[profile]
    cfg = load_config()
    cfg["training"]["low_timesteps"] = p["low"]
    cfg["training"]["meta_timesteps"] = p["meta"]
    cfg["training"]["flat_timesteps"] = p["flat"]
    cfg.setdefault("fast", {})["eval_steps"] = p["eval"]
    for dotted, val in over.items():
        d = cfg; ks = dotted.split(".")
        for k in ks[:-1]:
            d = d.setdefault(k, {})
        d[ks[-1]] = val
    return cfg, p


# ======================================================================= #
#  Study 1: K-sweep (main figure) + training cost
# ======================================================================= #
def study_k_sweep(profile):
    methods = ["HDRL", "Flat", "DPP", "IFP", "SCA", "AO", "FP"]
    rows = []
    for K in PROFILES[profile]["Ks"]:
        cfg, p = make_cfg(profile, **{"system.K": K})
        for m in methods:
            t0 = time.time()
            agg = aggregate(m, clone_cfg(cfg), p["seeds"])
            agg["K"] = K
            rows.append(agg)
            print(f"[k_sweep] K={K:2d} {m:5s} lat={agg['latency']:.4f}±{agg['latency_ci']:.4f} "
                  f"EE={agg['ee_mbit_j']:.2f}±{agg['ee_mbit_j_ci']:.2f} "
                  f"q={agg['queue_mean']:.2e} ({time.time()-t0:.0f}s)", flush=True)
            save_json("k_sweep.json", rows)   # incremental save
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "k_sweep.csv"), index=False)
    plot_k_sweep(rows, profile)
    plot_training_cost(rows, profile)
    return rows


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
    p = PROFILES[profile]; seed = p["seeds"][0]
    out = {"ris": [], "H": [], "G": []}
    # active vs passive RIS at K=8
    cfg, _ = make_cfg(profile, **{"system.K": 8})
    for m, label in [("HDRL", "active"), ("HDRL-passive", "passive")]:
        r = run_method(m, clone_cfg(cfg), seed); r["config"] = label
        out["ris"].append(r); print(f"[ablation RIS] {label}: lat={r['latency']:.4f} EE={r['ee_mbit_j']:.2f}", flush=True)
        save_json("ablation.json", out)
    # H sweep at K=8
    for H in [25, 50, 100]:
        cfg, _ = make_cfg(profile, **{"system.K": 8, "hierarchy.H": H})
        r = run_method("HDRL", clone_cfg(cfg), seed); r["H"] = H
        out["H"].append(r); print(f"[ablation H={H}] lat={r['latency']:.4f} EE={r['ee_mbit_j']:.2f}", flush=True)
        save_json("ablation.json", out)
    # G sweep at K=8 (N=128 divisible)
    for G in [4, 8, 16]:
        cfg, _ = make_cfg(profile, **{"system.K": 8, "system.G": G})
        r = run_method("HDRL", clone_cfg(cfg), seed); r["G"] = G
        out["G"].append(r); print(f"[ablation G={G}] lat={r['latency']:.4f} EE={r['ee_mbit_j']:.2f}", flush=True)
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
    rows = []
    for lam in lambdas:
        for m in ["HDRL", "DPP", "IFP"]:
            cfg, _ = make_cfg(profile, **{"system.K": 8, "hierarchy.lambda_E": lam})
            r = run_method(m, clone_cfg(cfg), seed); r["lambda_E"] = lam; r["method"] = m
            rows.append(r); print(f"[pareto] lam={lam} {m}: lat={r['latency']:.4f} EE={r['ee_mbit_j']:.2f}", flush=True)
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
    from ris_mec.hierarchy import train_hdrl
    from ris_mec.evaluate import evaluate_hdrl
    p = PROFILES[profile]; seed = p["seeds"][0]
    cfg, _ = make_cfg(profile, **{"system.K": 8})
    low, meta = train_hdrl(cfg, seed=seed, enable_ris=True)
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


STUDIES = {"k_sweep": study_k_sweep, "ablation": study_ablation,
           "pareto": study_pareto, "robustness": study_robustness}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="reduced", choices=list(PROFILES))
    ap.add_argument("--only", default="all")
    args = ap.parse_args()
    todo = list(STUDIES) if args.only == "all" else args.only.split(",")
    t0 = time.time()
    for name in todo:
        print(f"\n===== STUDY: {name} ({args.profile}) =====", flush=True)
        STUDIES[name](args.profile)
    print(f"\nALL DONE in {time.time()-t0:.0f}s. Results in {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
