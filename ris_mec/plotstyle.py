"""Publication figure styling: IEEE-friendly, colorblind-safe (Okabe-Ito),
grayscale-distinguishable via distinct markers/linestyles."""
from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette
OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7",
}

# Per-method visual identity (color + marker + linestyle) so figures survive
# grayscale printing.
METHOD_STYLE = {
    "HDRL": dict(color=OKABE_ITO["vermillion"], marker="o", ls="-", lw=2.0, zorder=5),
    "Flat": dict(color=OKABE_ITO["blue"], marker="s", ls="--", lw=1.6),
    "Flat PPO": dict(color=OKABE_ITO["blue"], marker="s", ls="--", lw=1.6),
    "DPP": dict(color=OKABE_ITO["green"], marker="^", ls="-.", lw=1.6),
    "IFP": dict(color=OKABE_ITO["orange"], marker="D", ls=":", lw=1.6),
    "SCA": dict(color=OKABE_ITO["purple"], marker="v", ls="--", lw=1.2),
    "AO": dict(color=OKABE_ITO["skyblue"], marker="P", ls=":", lw=1.2),
    "FP": dict(color="#888888", marker="X", ls=":", lw=1.2),
}


def apply_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "lines.markersize": 5,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    })


def style_for(name: str) -> dict:
    return dict(METHOD_STYLE.get(name, dict(color="#444", marker="o", ls="-", lw=1.5)))
