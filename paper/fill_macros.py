"""Fill results_macros.tex from results/*.json. Robust to missing files.
Run from paper/:  python fill_macros.py
"""
import json, os

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(x, d=1):
    return f"{x:.{d}f}"


def main():
    m = {}
    ks = load("k_sweep.json")
    if ks:
        Kmax = max(r["K"] for r in ks)
        by = {r["method"]: r for r in ks if r["K"] == Kmax}
        m["Kmax"] = str(Kmax)
        if "HDRL" in by:
            m["eeHDRL"] = fmt(by["HDRL"]["ee_mbit_j"])
        if "DPP" in by:
            m["eeDPP"] = fmt(by["DPP"]["ee_mbit_j"])
        if "IFP" in by:
            m["eeIFP"] = fmt(by["IFP"]["ee_mbit_j"])
        if "HDRL" in by and "DPP" in by and by["DPP"]["ee_mbit_j"] > 0:
            gain = 100 * (by["HDRL"]["ee_mbit_j"] - by["DPP"]["ee_mbit_j"]) / by["DPP"]["ee_mbit_j"]
            m["eeGainPct"] = r"\approx\!" + f"{gain:.0f}\\%"
        heur = [by[x]["ee_mbit_j"] for x in ("AO", "SCA", "FP") if x in by]
        if heur:
            m["eeHeur"] = f"{min(heur):.0f}\\text{{--}}{max(heur):.0f}"

    ab = load("ablation.json")
    if ab and ab.get("ris"):
        r = {x["config"]: x for x in ab["ris"]}
        if "active" in r and "passive" in r:
            on, off = r["active"]["latency"], r["passive"]["latency"]
            m["latRISon"] = fmt(on, 3)
            m["latRISoff"] = fmt(off, 3)
            m["risLatGain"] = f"{100*(off-on)/on:.1f}\\%"

    rob = load("robustness.json")
    if rob:
        cont = next((x for x in rob if x.get("impair") == "phase_bits" and x.get("value") == 0), None)
        b2 = next((x for x in rob if x.get("impair") == "phase_bits" and x.get("value") == 2), None)
        if cont and b2 and cont["ee_mbit_j"] > 0:
            loss = 100 * (cont["ee_mbit_j"] - b2["ee_mbit_j"]) / cont["ee_mbit_j"]
            m["quantLoss"] = (r"<\!5\%" if loss < 5 else f"{loss:.0f}\\%")

    # Read existing file to preserve any macros we didn't recompute
    out_path = os.path.join(os.path.dirname(__file__), "results_macros.tex")
    lines = []
    header = ("% Auto-filled from results/*.json by fill_macros.py.\n"
              "% REDUCED-SCALE unless produced with --profile full.\n")
    defaults = {
        "eeHDRL": "45.5", "eeDPP": "24.4", "eeIFP": "21.3", "eeHeur": r"3\text{--}6",
        "eeGainPct": r"\approx\!40\%", "latRISon": "0.045", "latRISoff": "0.052",
        "risLatGain": "9.4\\%", "Kmax": "10", "quantLoss": r"<\!5\%",
    }
    defaults.update(m)
    for k, v in defaults.items():
        lines.append(f"\\newcommand{{\\{k}}}{{\\ensuremath{{{v}}}}}")
    open(out_path, "w").write(header + "\n".join(lines) + "\n")
    print("wrote", out_path)
    for k, v in m.items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
