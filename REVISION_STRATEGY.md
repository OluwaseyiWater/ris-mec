# Revision Strategy: RIS-MEC HDRL — GLOBECOM 2026 → ICC 2027

**Paper:** *Scalable RIS-MEC Resource Allocation via Hierarchical Deep Reinforcement Learning*
**Prior submission:** IEEE GLOBECOM 2026 (#1571273036), SAC / RIS & Smart Environments track — **Rejected**
**Target:** IEEE ICC 2027 — **strict 6-page limit** (IEEEtran conference, incl. references)
**Date:** 2026-08-01

---

## 1. Executive summary

The paper was rejected not because the idea is weak but because three **claim‑vs‑evidence gaps** let the reviewers discount the contribution. All three are fixable, and fixing them turns the reviewers' own comments into a checklist for a materially stronger paper.

The diagnosis in one sentence: **the manuscript claims an *active* RIS with amplification and amplifier noise and a queue‑stabilizing EE optimizer, but the released code implements a *passive*, unit‑modulus RIS trained on a latency+power surrogate that never optimizes EE.** Reviewers 2 and 3 independently sensed the seams (R2: "nonphysical regimes," "not formally true for PPO"; R3: "the problem solved by HDRL is not equivalent to P1"). The revision plan is therefore not cosmetic: we make the code *actually implement the physics and objective the paper advertises*, then let the results and the writing follow.

The user has chosen (1) to proceed through all phases in order, (2) to **implement the physics** (not rescope the claims), and (3) to both deliver runnable, vectorized code and run reduced‑scale experiments here.

### Reviewer scorecard

| Criterion | R1 | R2 | R3 |
|---|---|---|---|
| Relevance / timeliness | Acceptable (3) | Little (2) | Acceptable (3) |
| Technical content & rigour | Limited contribution (3) | Limited contribution (3) | Limited contribution (3) |
| Novelty / originality | Well‑investigated subject (3) | Minor variation (2) | Well‑investigated subject (3) |
| Presentation | Revision needed (3) | Substantial revision (2) | Well written (4) |

Read together: nobody found it *wrong*, everybody found it *under‑supported*. "Limited contribution" is unanimous on technical rigour — that is the bar to clear. R2 is the toughest and most specific reviewer; **the R2 comments are the master checklist**.

---

## 2. Root‑cause diagnosis — the three gaps (verified against the notebook)

These are the load‑bearing problems. Everything in §3 hangs off them.

### Gap A — The "active RIS" is passive in the code (the central novelty is not simulated)

The paper's #1 claimed contribution is joint optimization of **active RIS element grouping, amplification, and amplifier noise**. In every version of the environment in `HDRL_for_RIS_MEC.ipynb` (cells 3, 14, 15, 16, 17, 19):

- The composite channel is `H = hd + Σ conj(hr)·exp(jθ)·g`. The reflection coefficient is `exp(jθ)` — **unit modulus**. There is **no amplitude gain `a_g`** on the reflected path. This is a *passive* RIS.
- The group control vector (`active_groups`, which the paper calls the activation/amplitude `a_g`) is applied as `theta = repeat(theta_g * active, N/G)` — i.e. it **scales the phase angle**, which is physically meaningless (activation should scale amplitude, not rotate phase).
- The uplink/downlink SINR divides by a **fixed thermal noise `N0`**. The amplified‑RIS noise term `σ̃²` derived in paper eq. (3) — the very thing R2 praised as "thoughtful" — **never enters the rate**.
- The energy denominator is `sum(ul_p) + dl_p`. The **RIS power consumption `P_RIS = Σ(p_c + p_a·a_g²)`** from paper §II‑D is **not counted at all**, and there is no `P_RIS ≤ P_RIS_max` constraint.

**Consequence:** R2's "omits per‑element output‑power stability constraints… may permit nonphysical regimes" and the unanimous "limited contribution" both flow from here. The paper is graded on a contribution the experiments do not contain.

### Gap B — HDRL never optimizes energy efficiency (objective ≠ P1)

R3's single comment is the most surgical in the whole packet: *"The problem solved by HDRL is not equivalent to the original problem P1… energy efficiency is not equivalent to the sum of the instantaneous energy efficiency."* The code confirms it exactly:

- P1 (paper eq. after §II‑F) maximizes a **ratio of sums**: total offloaded bits / total energy, minus a latency penalty.
- The actual RL reward is `r = -(L_total + 0.05·mean(ul_p+dl_p))/0.35` — a **per‑step weighted latency + transmit‑power** term. There is **no EE ratio** anywhere in the reward, and the `0.05` weight is not the paper's `λ_E`.
- EE is only computed *at evaluation time* as `Σbits/Σenergy`. So the headline "≈40% higher EE" measures a quantity the agent was never trained to improve. That is precisely the non‑equivalence R3 flagged, and it undercuts the paper's main empirical claim.

### Gap C — Reproducibility and model‑fidelity gaps (R2's experimental section)

- **Unseeded channels.** The main env (`cell 3`) seeds queues via `self.rng` but draws channels with the *global* `np.random.randn`. Fixing the seed does **not** reproduce a run. (Cell 17 quietly fixes this — the fix exists but wasn't the one that made the paper.)
- **No 3GPP pathloss.** The paper says "3GPP Urban Micro path loss." The code uses i.i.d. unit‑variance Rayleigh with **no distance‑dependent pathloss** and no large‑scale geometry.
- **Direct‑link contradiction.** Assumption A1 says direct links are fully blocked (`ρ̄_k = 0`, so `hd = 0`). The code keeps a **non‑zero `hd`** and adds it into `H`. The stated model and the simulated model disagree.
- **No parameter table, seeds, or CIs.** Values exist in code (B=20 MHz, N0=1e‑13, P_k^ul≤0.2 W, P_bs^dl≤5 W, ω=500 cyc/bit, F_max=20 GHz, N=128, G=8, H=50) but are never tabulated; headline figures are single‑seed with no error bars.
- **Queue update never written.** The code *does* evolve queues (`Q ← clip(Q − bits_ul,0) + arrivals`), but the paper never states the equation, so R2/R3 rightly say stability is unassessable. Note two inconsistent arrival models coexist in the notebook (cell 3 adds arrivals every step; cell 17 refills only when a queue empties) — we must pick one and state it.
- **Theory oversells.** Theorem III.2 claims **PPO monotonic improvement**; R2 correctly notes this holds for TRPO‑style bounds, not clipped PPO. The O(H²) bound is a two‑line drift sketch under strong Lipschitz assumptions.

---

## 3. Comprehensive review → action map

Legend for effort: **S** ≤ half‑day · **M** 1–2 days · **L** 3+ days. "Where": **C** = code, **P** = paper, **B** = both.

### Reviewer 1 (mildest; strategic/positioning)

| # | Comment | Action | Where | Effort |
|---|---|---|---|---|
| R1.1 | "Very limited novelty over existing hierarchical DRL" | Reframe novelty around the *active‑RIS grouping + amplifier‑noise coupling handled at the slow timescale* (a genuinely new mechanism once Gap A is fixed), not "we used HDRL." Sharpen the contributions list. Add the missing HDRL‑for‑comms references R2 also wants. | B | M |
| R1.2 | "Training overhead needs to be analyzed" | Add a training‑cost subsection: wall‑clock + sample complexity for HDRL vs Flat PPO, and the two‑phase (freeze‑tactician) budget. We already log training time — surface it. | B | S |
| R1.3 | "Investigation of RIS grouping under diverse scenarios" | Add a `G` sensitivity sweep under ≥2 regimes (density, task load) and report the resolution↔action‑space trade‑off with CIs, not a single row. | B | M |
| R1.4 | "Evaluation limited to simulation" | Acknowledge explicitly; strengthen realism (3GPP pathloss, CSI error, phase quantization) so "simulation" is credible. State hardware validation as future work. | B | S |

### Reviewer 2 (toughest; the master checklist)

| # | Comment | Action | Where | Effort |
|---|---|---|---|---|
| R2.1 | Active‑RIS model omits per‑element/output‑power stability constraints; static budget permits nonphysical regimes | **Fix Gap A.** Implement `a_g` amplitude gains with the output‑power constraint `a_g²(P^ul|h_r|² + σ_RIS²) ≤ P_out,max` per element/group, plus the global `P_RIS ≤ P_RIS_max`. | B | L |
| R2.2 | Queue dynamics not specified though stability is claimed | State the arrival process (Poisson/exponential bits) and the update `Q_k(t+1)=max(Q_k(t)−b_k^ul(t),0)+A_k(t)`. Report avg queue length, overflow/violation prob, latency‑tail. | B | M |
| R2.3 | Uplink/downlink noise use different aggregated terms (ζ_g vs ξ_{g,k}) without derivation; reciprocity unjustified | Give the full derivation of `σ̃²` from per‑element amplifier noise → group aggregation; state the reciprocity assumption explicitly or drop the symmetric‑channel shortcut. | B | M |
| R2.4 | O(H²) bound weak; **PPO "monotonic improvement" is not formally true** | Rewrite theory: keep O(H²) as an *approximation‑error* statement with assumptions stated honestly; **replace** Thm III.2 with the correct clipped‑PPO statement (a lower bound / trust‑region surrogate), or cut it and reclaim space. | P | M |
| R2.5 | Missing training details (arch, hyperparams, seeds, normalization, convergence) + error bars + statistical robustness | Full reproducibility table + seeded multi‑run results with 95% CIs on every headline number. | B | M |
| R2.6 | No sensitivity to CSI error, delayed/partial observability, or quantized RIS phase | Add robustness experiments: imperfect CSI (Gaussian error), b‑bit phase quantization, delayed CSI. One compact figure/table. | B | M |
| R2.7 | IFP baseline is instantaneous‑EE; fairness needs Lyapunov drift‑plus‑penalty / two‑timescale baseline | **Add a Lyapunov DPP baseline** that explicitly handles queues. This is the most valuable single addition — it directly answers "limited contribution." | B | L |
| R2.8 | 10–12 Mbits/J claim lacks a parameter table | Add the parameter table (B, N0, pathloss, P caps, p_c, p_a, ω, …). | P | S |
| R2.9 | Notation inconsistencies / typos / tilde noise terms | Notation pass; define every symbol once in a table; fix formatting artifacts. | P | S |
| R2.10 | Objective mixes latency‑min with EE‑max; unit scaling/normalization not discussed | **Fix Gap B.** Make the reward the actual P1 objective (normalized EE − λ_E·latency or DPP form); discuss normalization/reward shaping. | B | M |
| R2.11 | Missing related work: hierarchical/two‑timescale RL in comms; Lyapunov & deep‑Lyapunov queue methods | Expand related work (compact); cite two‑timescale RL and drift‑plus‑penalty / deep‑Lyapunov. | P | S |
| R2.12 | RIS grouping/subarray/hybrid‑beamforming has precedent; acknowledge to situate novelty | Add 2–3 subarray/group‑control references and one sentence positioning our *active* + slow‑timescale twist. | P | S |

### Reviewer 3 (best presentation score; two deep technical points)

| # | Comment | Action | Where | Effort |
|---|---|---|---|---|
| R3.1 | HDRL problem ≠ P1 (EE of sums ≠ sum of instantaneous EE) | **Fix Gap B** (shared with R2.10). Either (a) optimize the true ratio objective directly, or (b) reformulate P1 via Dinkelbach/Lyapunov so the per‑step surrogate is *provably* aligned with long‑run EE — and say which. | B | M |
| R3.2 | Problem is bi‑objective; plot the Pareto frontier by varying λ_E | Add a **λ_E sweep → latency‑vs‑EE Pareto frontier** for HDRL and the key baselines. Strong, cheap, visually convincing. | B | M |
| R3.3 | (Presentation already "well written") | Preserve the parts R3 liked; don't over‑cut prose that earned the 4. | P | — |

---

## 4. What changes in the CODE (repo plan)

Target: a clean, vectorized, reproducible repository replacing the notebook.

```
ris_mec_icc2027/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml          # THE parameter table (single source of truth)
├── ris_mec/
│   ├── channels.py           # 3GPP UMi pathloss, Rician/Rayleigh, blocked direct link, CSI error, quantization
│   ├── env.py                # vectorized RISMECEnv: active a_g + amplifier noise σ̃² + RIS power + queue eqn
│   ├── hierarchy.py          # Strategist (slow) + Tactician (fast), two-phase training
│   ├── flat.py               # Flat PPO wrapper
│   ├── baselines.py          # AO, SCA, FP, IFP, + Lyapunov DPP (NEW)
│   ├── objectives.py         # EE ratio / Dinkelbach / DPP reward variants (fixes Gap B)
│   ├── metrics.py            # EE, latency, queue length/violation, energy incl. RIS
│   └── utils.py              # seeding, logging, CI helpers
├── experiments/
│   ├── exp_k_sweep.py        # Fig: latency & EE vs K, seeds+CI
│   ├── exp_ablation.py       # RIS on/off, H sweep, G sweep (diverse scenarios)
│   ├── exp_pareto.py         # λ_E → Pareto frontier
│   ├── exp_robustness.py     # CSI error, phase quantization, delayed CSI
│   └── exp_training_cost.py  # wall-clock + sample complexity
└── results/                  # CSVs + figures (git-ignored large runs)
```

Key correctness fixes baked in:
- **Active RIS:** reflection coefficient `a_g·e^{jθ_g}` with `a_g ≥ 0`; amplifier thermal noise injected per element and aggregated per group into `σ̃²` that enters the SINR denominator; per‑group output‑power constraint and global `P_RIS` budget; `P_RIS = Σ(p_c + p_a a_g²)` added to the energy denominator.
- **Objective alignment (Gap B):** reward selectable in `objectives.py` — (i) direct normalized EE − λ_E·latency, or (ii) Dinkelbach‑linearized EE, or (iii) Lyapunov drift‑plus‑penalty `Δ(Q) + V·(latency − λ_E·EE)`. Default = DPP so queue stability is *provable*, answering R2.2/R2.7/R2.10/R3.1 together.
- **Reproducibility:** all randomness through a single seeded `Generator`; config‑driven; per‑run seed logged; CI helper for figures.
- **Vectorization (user request):** batch users/groups with NumPy einsum already; vectorize across the seed/scenario axis and use SB3 `SubprocVecEnv`/multiple envs; provide a `--fast` smoke profile and a `--full` profile.

---

## 5. What changes in the PAPER

Section‑by‑section, holding the 6‑page line (see §6):

- **Abstract/Intro:** keep the physical‑AI framing R2/R3 liked; retune the headline claim to what we now actually optimize and measure (EE with queue stability), and state the active‑RIS amplifier‑noise coupling as the concrete novelty.
- **System model:** add (a) the active reflection coefficient with `a_g`, (b) the `σ̃²` derivation from per‑element amplifier noise, (c) the output‑power/stability constraint, (d) the explicit **queue update equation**, (e) fix the A1 direct‑link contradiction (either truly zero `hd` or relax A1 to partial blockage — recommend partial blockage so `hd` in code is legitimate), (f) the RIS power in the energy model.
- **Problem formulation:** state P1 as EE‑max with a queue‑stability constraint; show the reward is a *consistent* per‑step surrogate (Dinkelbach/DPP), closing R3.1.
- **Algorithm:** unchanged structure; add reproducibility details inline or in a compact table; add the training‑cost note (R1.2).
- **Theory:** honest O(H²) approximation statement; **remove/repair** the PPO monotonicity theorem (R2.4). This also frees space.
- **Results:** seeds+CI everywhere; add Lyapunov DPP baseline; add Pareto frontier; add one compact robustness panel (CSI error + phase quantization); parameter table.
- **Related work:** +2–3 sentences on two‑timescale RL, drift‑plus‑penalty/deep‑Lyapunov, and subarray/group RIS control (R2.11/R2.12/R1.1).

---

## 6. The 6‑page budget (ICC is unforgiving)

GLOBECOM allowed the compressed version; ICC 2027 is a hard 6 pages including references. We are *adding* content, so we must cut aggressively elsewhere. Plan:

**Keep (must‑have):** system model with the corrected physics; P1 + objective‑alignment argument; the two‑timescale algorithm + Algorithm 1; one main results figure (K‑sweep latency+EE with CI); the Pareto frontier; the Lyapunov DPP comparison; parameter table; a compact robustness table.

**Compress:** Intro (−15%); related work into a tight paragraph; merge the two theory theorems into one honest lemma (~½ column saved by dropping the false PPO theorem); ablation table trimmed to the decisive rows.

**Cut / move to an extended arXiv version:** full noise‑model derivation (keep result in paper, derivation → appendix/arXiv); exhaustive hyperparameter prose (→ table + repo); secondary convergence plots. **Strategy: maintain a 6‑page ICC camera version + a longer arXiv version**, cite the arXiv for derivations and extra experiments. This is standard and lets us satisfy reviewers without violating the limit.

**Figure economy:** the current 6‑panel K‑sweep figure is expensive. Replace with a 2‑panel (latency vs K, EE vs K) with CI bands + a separate 1‑panel Pareto + a small robustness table. Net figures ≤ 4.

---

## 7. Novelty repositioning (answers the unanimous "limited contribution")

State three defensible, now‑supported novelties:
1. **Slow‑timescale management of the active‑RIS gain–noise trade‑off.** Because `a_g` moves both signal gain and the amplifier noise floor `σ̃²`, updating it every slot gives conflicting gradients; the strategist averages the trade‑off over channel realizations while the tactician does instantaneous phase/power. This is a *mechanism*, not just "we used HDRL," and it only becomes real once Gap A is fixed.
2. **Grouping that makes active‑RIS control tractable in DRL** — 2G continuous variables vs 2^N mixed‑integer — with an explicit resolution↔convergence trade‑off characterized empirically.
3. **Queue‑stable EE optimization** with a per‑step surrogate proven consistent with long‑run EE (Dinkelbach/Lyapunov), benchmarked against a Lyapunov DPP baseline.

---

## 8. Sequencing & deliverables (this engagement)

1. **This doc** (strategy) — done, saved to the project.
2. **Repo scaffold + config/parameter table.**
3. **Corrected, vectorized physics env.**
4. **Hierarchy + Flat + classical + Lyapunov DPP baselines.**
5. **Objective fix + Pareto sweep.**
6. **Install stack → smoke test → reduced‑scale runs → figures/tables** (clearly labeled reduced‑scale; full runs are the user's to scale on GPU).
7. **6‑page ICC rewrite** (LaTeX/IEEEtran source + PDF) + a longer arXiv companion outline.
8. **Independent verification pass.**

## 9. Risks & open decisions

- **A1 direct link:** recommend relaxing to *partial* blockage (ρ̄_k ∈ (0,1]) so the non‑zero `hd` is legitimate and the RIS still dominates; alternative is truly zeroing `hd`. Low‑risk either way — flag for author sign‑off.
- **Objective form:** DPP (recommended) vs direct‑ratio vs Dinkelbach. DPP gives the strongest queue‑stability story and the fairest comparison to the R2‑requested baseline. Author may prefer direct‑ratio for simplicity.
- **Compute:** full multi‑seed, multi‑K, multi‑baseline training is GPU‑hours; the numbers produced here are **preliminary/reduced‑scale** to prove the pipeline and shapes, not final camera‑ready values.
- **Amplifier‑noise parameters** (σ_RIS², p_a, P_out,max): need reasonable literature values; will cite active‑RIS papers for defaults and expose them in config.
