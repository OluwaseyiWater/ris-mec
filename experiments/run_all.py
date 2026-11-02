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

from ris_mec.utils import load_config, clone_cfg, ci95
from ris_mec.plotstyle import apply_style, style_for
from experiments.common import (aggregate, aggregate_many, run_method, save_json,
                                RESULTS, eval_steps, configure, write_provenance)

PROFILES = {
    "fast":    dict(low=20000,  meta=200,  seeds=[0],    Ks=[4, 6, 8],     eval=400),
    "reduced": dict(low=100000, meta=620,  seeds=[0, 1], Ks=[4, 6, 8, 10], eval=800),
    "full":    dict(low=1000000, meta=9000, seeds=[0,4,10,16,24,42,50,64,100,128,200,256,480,512,1024], Ks=[4,6,8,10,12,16], eval=2000),
}


AUX_SEEDS = 5


def _budget_parity(p, H):
    """Environment slots the hierarchy consumes: E_L + H*E_H."""
    return int(p["low"]) + int(H) * int(p["meta"])

RUN = {"jobs": 1, "threads": 0, "fixed_total_load": 0.0, "v_by_method": {}}

V_KEY = {"HDRL": "hierarchy.dpp_V", "HDRL-passive": "hierarchy.dpp_V",
         "Flat": "hierarchy.dpp_V", "DPP": "baselines.dpp_V"}


def with_v(cfg, method):
    """Copy of cfg at `method`'s operating point, if --auto-v chose one."""
    c = clone_cfg(cfg)
    v = RUN["v_by_method"].get("HDRL" if method == "HDRL-passive" else method)
    if v is not None and method in V_KEY:
        sect, key = V_KEY[method].split(".")
        c.setdefault(sect, {})[key] = float(v)
    return c


def choose_v(pareto_rows):
    """Largest-EE feasible (served >= FEASIBLE) V per method on the pareto frontier.

    If a method has no feasible point (a short horizon caps the served fraction
    below FEASIBLE for everyone), fall back with a warning to the largest-EE V
    among those within 0.1 pp of that method's best served fraction, so later
    studies still run at a defined, near-most-stable operating point. The paper's
    claim checks flag this case; it is not silently accepted.
    """
    out = {}
    for m in ("HDRL", "Flat", "DPP"):
        pts = [r for r in pareto_rows if r["method"] == m and r.get("V") is not None]
        if not pts:
            continue
        ok = [r for r in pts if r["served_ratio"] >= FEASIBLE]
        if ok:
            out[m] = max(ok, key=lambda r: r["ee_mbit_j"])["V"]
        else:
            top = max(r["served_ratio"] for r in pts)
            near = [r for r in pts if r["served_ratio"] >= top - 1e-3]
            out[m] = max(near, key=lambda r: r["ee_mbit_j"])["V"]
            print(f"  !! {m}: no feasible V on the frontier (best served {top:.4f}); "
                  f"using the best-EE near-most-stable V={out[m]:g}", flush=True)
    return out


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
            specs.append((K, m, with_v(cfg, m), p["seeds"]))
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
    """Delay, EE and served load versus K.

    Delay is END-TO-END (service + queueing, Little's law), not the per-slot
    service time the figure used to show: service time rewards a controller for
    letting its backlog grow. The served-load panel is what makes the EE panel
    interpretable -- EE = bits/energy can always be raised by serving less, so an
    EE comparison only means something between controllers that keep up with
    the offered load (dotted line = 1).
    """
    apply_style()
    df = pd.DataFrame(rows)
    panels = [("delay_total", "End-to-end delay (s)", True),
              ("ee_mbit_j", "Energy efficiency (Mbit/J)", False),
              ("served_ratio", "Served / offered load", False)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, (col, ylab, logy) in zip(axes, panels):
        if col not in df:
            ax.set_visible(False)
            continue
        for m in df["method"].unique():
            d = df[df.method == m].sort_values("K")
            name = "Flat PPO" if m == "Flat" else m
            ax.errorbar(d["K"], d[col], yerr=d.get(col + "_ci"), label=name,
                        capsize=2, **style_for(name))
        ax.set_xlabel("Number of users $K$")
        ax.set_ylabel(ylab)
        if logy:
            ax.set_yscale("log")
    if "served_ratio" in df:
        axes[2].axhline(1.0, color="k", lw=0.6, ls=":")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, 1.06))
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
    p = PROFILES[profile]; seeds = p["seeds"][:AUX_SEEDS]
    specs = []
    # active vs passive RIS at K=8
    cfg, _ = make_cfg(profile, **{"system.K": 8})
    for m, label in [("HDRL", "active"), ("HDRL-passive", "passive")]:
        specs.append((("ris", label), m, with_v(cfg, m), seeds))
    # H sweep at K=8 (the tactician is H-agnostic, so --cache trains it once)
    for H in [25, 50, 100]:
        c, _ = make_cfg(profile, **{"system.K": 8, "hierarchy.H": H})
        specs.append((("H", H), "HDRL", with_v(c, "HDRL"), seeds))
    # G sweep at K=8 (N=128 divisible)
    for G in [4, 8, 16]:
        c, _ = make_cfg(profile, **{"system.K": 8, "system.G": G})
        specs.append((("G", G), "HDRL", with_v(c, "HDRL"), seeds))

    t0 = time.time()
    aggs = aggregate_many(specs, jobs=RUN["jobs"], threads=RUN["threads"],
                          progress=_progress("ablation", t0))
    out = {"ris": [], "H": [], "G": []}
    for agg in aggs:
        group, label = agg.pop("_label")
        agg["config" if group == "ris" else group] = label
        agg["n_seeds"] = len(seeds)
        out[group].append(agg)
        print(f"[ablation {group}={label}] EE={agg['ee_mbit_j']:.2f}+-{agg['ee_mbit_j_ci']:.2f} "
              f"delay={agg['delay_total']:.4f} served={agg['served_ratio']:.2f}", flush=True)
    save_json("ablation.json", out)
    plot_ablation(out, profile)
    return out


def plot_ablation(out, profile):
    """EE (top) and end-to-end delay (bottom) for each ablation, with 95% CIs."""
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0))
    hd = style_for("HDRL")
    for row, (col, ylab) in enumerate([("ee_mbit_j", "EE (Mbit/J)"),
                                       ("delay_total", "End-to-end delay (s)")]):
        d = pd.DataFrame(out["ris"])
        axes[row, 0].bar(d["config"], d[col], yerr=d.get(col + "_ci"), capsize=3,
                         color=[hd["color"], "#888"])
        axes[row, 0].set_ylabel(ylab)
        d = pd.DataFrame(out["H"]).sort_values("H")
        axes[row, 1].errorbar(d["H"], d[col], yerr=d.get(col + "_ci"), capsize=2, **hd)
        axes[row, 1].set_xticks(list(d["H"]))
        d = pd.DataFrame(out["G"]).sort_values("G")
        axes[row, 2].errorbar(d["G"], d[col], yerr=d.get(col + "_ci"), capsize=2,
                              **style_for("DPP"))
        axes[row, 2].set_xscale("log", base=2)
        axes[row, 2].set_xticks(list(d["G"]))
        axes[row, 2].set_xticklabels([str(g) for g in d["G"]])
    axes[1, 0].set_xlabel("RIS")
    axes[1, 1].set_xlabel("Meta-interval $H$")
    axes[1, 2].set_xlabel("Groups $G$")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS, f"fig_ablation.{ext}"))
    plt.close(fig)


# ======================================================================= #
#  Study 3: Latency--EE frontier (sweep the drift-plus-penalty weight V)
# ======================================================================= #
PARETO_VS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]   # 0.01 brackets the feasibility boundary
CURVES = ("HDRL", "Flat", "DPP")   # methods swept over V in the pareto study
FEASIBLE = 0.99   # served/offered at or above which P1's stability constraint holds


def study_pareto(profile):
    """Latency--EE trade-off, traced by the drift-plus-penalty weight V.

    This used to sweep lambda_E, but lambda_E multiplies the service latency
    (~0.04 s) against an EE term of ~9, so it barely moved the policy (HDRL went
    88.1 -> 90.2 Mbit/J over lambda_E = 0.1..10) and DPP/IFP ignore it entirely:
    the "frontier" was five copies of one point per baseline. V is the knob that
    actually trades stability for efficiency, and both learned methods share it,
    so sweeping it gives each a genuine frontier on the delay--EE plane at a
    matched training budget. The DPP baseline gets its own V (baselines.dpp_V,
    a different normalization of the same drift-plus-penalty idea) and is swept
    over the same grid, so it traces a frontier too; IFP and the heuristics have
    no such knob and appear as single operating points.
    """
    p = PROFILES[profile]; seeds = p["seeds"][:AUX_SEEDS]
    specs = []
    for V in PARETO_VS:
        cfg, _ = make_cfg(profile, **{"system.K": 8, "hierarchy.dpp_V": V})
        for m in ("HDRL", "Flat"):
            specs.append(((m, V), m, clone_cfg(cfg), seeds))
        c2, _ = make_cfg(profile, **{"system.K": 8, "baselines.dpp_V": V})
        specs.append((("DPP", V), "DPP", clone_cfg(c2), seeds))
    cfg, _ = make_cfg(profile, **{"system.K": 8})
    for m in ("IFP", "SCA"):
        specs.append(((m, None), m, clone_cfg(cfg), seeds))
    t0 = time.time()
    aggs = aggregate_many(specs, jobs=RUN["jobs"], threads=RUN["threads"],
                          progress=_progress("pareto", t0))
    rows = []
    for agg in aggs:
        _, V = agg.pop("_label")
        agg["V"] = V
        agg["n_seeds"] = len(seeds)
        rows.append(agg)
        print(f"[pareto] {agg['method']:<5} V={str(V):<5} EE={agg['ee_mbit_j']:.2f} "
              f"delay={agg['delay_total']:.4f} served={agg['served_ratio']:.2f}", flush=True)
    save_json("pareto.json", rows)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "pareto.csv"), index=False)
    plot_pareto(rows, profile)
    return rows


def plot_pareto(rows, profile):
    """Delay--EE plane. Learned methods are curves in V; baselines are points.
    Hollow markers are infeasible for P1 (served < 99% of offered load), where a
    high EE has been bought by not serving the traffic."""
    apply_style()
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    for m in ("HDRL", "Flat", "DPP", "IFP", "SCA"):
        d = df[df.method == m]
        if d.empty:
            continue
        if m in CURVES:
            d = d.sort_values("V")
        name = "Flat PPO" if m == "Flat" else m
        st = style_for(name)
        if m not in CURVES:
            st.pop("lw", None)
            st["ls"] = "none"
        ax.errorbar(d["delay_total"], d["ee_mbit_j"], xerr=d.get("delay_total_ci"),
                    yerr=d.get("ee_mbit_j_ci"), capsize=2, label=name, **st)
        bad = d[d["served_ratio"] < FEASIBLE]
        if not bad.empty:
            ax.scatter(bad["delay_total"], bad["ee_mbit_j"], s=16, marker=st["marker"],
                       facecolors="white", edgecolors=st["color"], linewidths=0.9,
                       zorder=7)
        if m in ("HDRL", "DPP") and len(d) > 1:
            # Only the ends of the curve (the high-V points cluster and labels
            # collide), set outside the curve so each names an unambiguous point.
            lo, hi = d.iloc[0], d.iloc[-1]
            for r, xy, ha in ((lo, (-6, 0), "right"), (hi, (0, 7), "center")):
                ax.annotate(f"$V$={r['V']:g}", (r["delay_total"], r["ee_mbit_j"]),
                            fontsize=6, xytext=xy, textcoords="offset points",
                            ha=ha, va="center", color=st["color"])
    ax.set_xscale("log")
    # Plain decimal ticks: matplotlib's default minor labels on a short log
    # range print as colliding "3x10^-1 4x10^-1".
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    lo_x, hi_x = ax.get_xlim()
    nice = [t for t in (0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5)
            if lo_x <= t <= hi_x]
    ax.xaxis.set_major_locator(FixedLocator(nice))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("End-to-end delay (s)")
    ax.set_ylabel("Energy efficiency (Mbit/J)")
    ax.legend(fontsize=6.5, loc="best")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS, f"fig_pareto.{ext}"))
    plt.close(fig)


# ======================================================================= #
#  Study 4: Robustness to imperfect CSI -- for every controller, not just HDRL
# ======================================================================= #
ROBUST_METHODS = ("HDRL", "Flat", "DPP", "IFP")
# (label, config key under channel, value). The clean case is csi_error=0, which
# is also the zero-delay and continuous-phase reference in the plots.
ROBUST_GRID = ([("csi_error", "csi_error_std", v) for v in (0.0, 0.1, 0.2, 0.3, 0.5)]
               + [("csi_delay", "csi_delay_slots", d) for d in (1, 2, 5)]
               + [("phase_bits", "phase_bits", b) for b in (1, 2, 3)])


def study_robustness(profile):
    """Every controller under the same CSI impairments.

    The model-based baselines (DPP, IFP) plan on the *estimated* channel through
    the same impairment model as the learned agents' observation (see
    baselines._csi_snapshot), so this is a like-for-like comparison of how each
    degrades when its knowledge of the channel is wrong or stale. The learned
    policies are trained on clean CSI and evaluated impaired, without retraining;
    DPP/IFP simply re-plan on the impaired estimate each slot.

    csi_delay is in slots against a coherence time of channel.coherence_slots:
    the controller acts on the fading block that was current that many slots ago.
    """
    from ris_mec.evaluate import evaluate_hdrl, evaluate_flat
    from ris_mec.flat import train_flat
    from ris_mec import baselines
    from experiments.common import _hdrl_models, CACHE_DIR
    from ris_mec.modelcache import ModelCache
    from stable_baselines3 import PPO
    import experiments.common as _common
    p = PROFILES[profile]; seeds = p["seeds"][:AUX_SEEDS]
    cfg, _ = make_cfg(profile, **{"system.K": 8})
    cache = ModelCache(CACHE_DIR, enabled=_common._CACHE_ENABLED)
    steps = eval_steps(cfg)
    per = {(m, imp, val): [] for m in ROBUST_METHODS for imp, _, val in ROBUST_GRID}
    for sd in seeds:
        # Same (cfg, seed) as the K=8 runs in k_sweep, so --cache reuses them; the
        # impairments are evaluation-only and do not change the cache key.
        cH, cF = with_v(cfg, "HDRL"), with_v(cfg, "Flat")
        low, meta = _hdrl_models(cH, sd, True, cache)
        flat = cache.get_or_train("flat", cF, sd,
                                  builder=lambda: train_flat(cF, seed=sd), loader=PPO.load)
        for imp, key, val in ROBUST_GRID:
            c = clone_cfg(cfg); c["channel"][key] = val
            per[("HDRL", imp, val)].append(
                evaluate_hdrl(with_v(c, "HDRL"), low, meta, seed=sd, steps=steps))
            per[("Flat", imp, val)].append(
                evaluate_flat(with_v(c, "Flat"), flat, seed=sd, steps=steps))
            for m in ("DPP", "IFP"):
                per[(m, imp, val)].append(baselines.BASELINES[m](with_v(c, m), seed=sd, steps=steps))
        print(f"[robust] seed {sd} done", flush=True)
    rows = []
    for m in ROBUST_METHODS:
        for imp, _, val in ROBUST_GRID:
            runs = per[(m, imp, val)]
            row = {"method": m, "impair": imp, "value": val, "n_seeds": len(runs)}
            for k in ("ee_mbit_j", "delay_total", "served_ratio", "dl_delivered_ratio",
                      "latency", "queue_mean"):
                row[k], row[k + "_ci"] = ci95([r[k] for r in runs])
            rows.append(row)
            print(f"[robust] {m:<4} {imp}={val:<4} EE={row['ee_mbit_j']:7.2f} "
                  f"delay={row['delay_total']:.4f} served={row['served_ratio']:.3f}", flush=True)
    save_json("robustness.json", rows)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "robustness.csv"), index=False)
    plot_robustness(rows, profile)
    return rows


def plot_robustness(rows, profile):
    """EE (top) and end-to-end delay (bottom) against each impairment, per method.
    The clean point (no error, no delay, continuous phase) anchors every panel."""
    apply_style()
    df = pd.DataFrame(rows)
    if "method" not in df:            # results from before multi-method robustness
        df["method"] = "HDRL"
    clean = df[(df.impair == "csi_error") & (df.value == 0.0)]
    panels = [("csi_error", "CSI error std", None),
              ("csi_delay", "CSI delay (slots)", 0),
              ("phase_bits", "RIS phase bits (8 = cont.)", 8)]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0))
    for col, (imp, xlab, clean_x) in enumerate(panels):
        for m in df["method"].unique():
            d = df[(df.method == m) & (df.impair == imp)].copy()
            if d.empty:
                continue
            if clean_x is not None:
                c = clean[clean.method == m].copy()
                c["value"] = clean_x
                d = pd.concat([c, d])
            d = d.sort_values("value")
            name = "Flat PPO" if m == "Flat" else m
            for row, key in enumerate(("ee_mbit_j", "delay_total")):
                if key not in d:
                    continue
                axes[row, col].errorbar(d["value"], d[key], yerr=d.get(key + "_ci"),
                                        capsize=2, label=name, **style_for(name))
        axes[1, col].set_xlabel(xlab)
        axes[1, col].set_yscale("log")
    axes[0, 0].set_ylabel("EE (Mbit/J)")
    axes[1, 0].set_ylabel("End-to-end delay (s)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
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
    Vs = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
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
    ap.add_argument("--auto-v", action="store_true",
                    help="run the pareto study first, pick each method's best feasible V "
                         "(served >= 0.99) and use it for every later study; recorded "
                         "in results/_provenance.json")
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
    if args.auto_v:
        # The operating points come from the frontier, so it must run first.
        todo = ["pareto"] + [t for t in todo if t != "pareto"]
    t0 = time.time()
    for name in todo:
        print(f"\n===== STUDY: {name} ({args.profile}) =====", flush=True)
        out = STUDIES[name](args.profile)
        if args.auto_v and name == "pareto":
            RUN["v_by_method"] = choose_v(out)
            print(f"[auto-v] operating points: "
                  + ", ".join(f"{k} V={v:g}" for k, v in RUN["v_by_method"].items()),
                  flush=True)
    write_provenance(args.profile, todo, v_by_method=RUN["v_by_method"])
    print(f"\nALL DONE in {time.time()-t0:.0f}s. Results in {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
