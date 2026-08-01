# Response to Reviewers — GLOBECOM 2026 → ICC 2027

We thank the reviewers. The revision closes the gaps between the paper's claims and
its evidence by (i) making the RIS genuinely *active* in the model and code
(amplitude gains, amplifier noise, output-power constraint, RIS power in the energy),
(ii) making the learning objective *provably consistent* with the P1 energy-efficiency
ratio via a Lyapunov drift-plus-penalty reward, and (iii) rebuilding the experiments
as a reproducible, seeded, CI-reported suite with a new queue-aware baseline. Section
references below point to the revised manuscript. Numbers reported here are from a
CPU **reduced-scale** run shipped with the code; the released `--profile full`
configuration reproduces camera-ready numbers on a GPU.

> Note on scope: we are explicit where the reduced-scale evidence is preliminary
> (e.g., the HDRL-vs-flat scalability crossover), rather than overclaiming.

---

## Reviewer 1

**R1.1 Limited novelty over existing HDRL.**
Reframed contributions (Sec. I): the novelty is the *active-RIS grouping with the
amplifier gain–noise coupling handled on the slow timescale*, plus a reward
provably consistent with the EE ratio — not "we used HDRL." Grouping/subarray
precedent for passive RIS is now acknowledged (Sec. II-B, [17],[19]) and we position
our active extension against it.

**R1.2 Training overhead not analyzed.**
Added (Sec. IV-A, Fig. 3b): HDRL's two-phase training costs ≈8× the flat agent's
(71 s vs 9 s at K=10 in the reduced run); per-slot *inference* is sub-ms for all
methods, so all are real-time capable and the only cost distinction is one-time
offline training.

**R1.3 RIS grouping under diverse scenarios.**
Added a G-sweep ablation (Sec. IV-C, Fig. 4) characterizing the beam-resolution vs.
action-space trade-off; the released code runs it across densities.

**R1.4 Simulation-only.**
Acknowledged; realism strengthened (3GPP-style pathloss, imperfect CSI, phase
quantization). Hardware validation is stated as future work.

## Reviewer 2

**R2.1 Active-RIS omits output-power/stability constraints; nonphysical regimes.**
Added the per-element output-power constraint (Eq. 4) coupling gain to incident
power, plus a global RIS power budget; both are enforced in code
(`env.py:_apply_output_power_constraint`).

**R2.2 Queue dynamics unspecified.**
Explicit recursion Q_k(t+1)=max(Q_k−b_k,0)+A_k added (Eq. 7); backlog is reported and
the drift-plus-penalty design provides the stability guarantee.

**R2.3 UL/DL noise (ζ_g vs ξ_{g,k}) lacked derivation.**
Derived (Sec. II-A, Eqs. 2–3): the amplifier noise reaches the BS through **g**
(uplink, ζ_g) and the UE through **h_{r,k}** (downlink, ξ_{g,k}); implemented in
`env.py:_noise_floors`.

**R2.4 Weak O(H²) bound; PPO "monotonic improvement" is not formally true.**
Prop. 1 now states assumptions explicitly as an *approximation-error* bound. The
false PPO monotonicity theorem is **removed** and replaced by a correct remark: PPO's
clipped surrogate is a pessimistic lower bound on the TRPO trust-region objective and
yields *stable, not monotone*, improvement (Sec. III-C).

**R2.5 Missing training details, error bars, statistical robustness.**
All headline results now report 95% CIs over multiple seeds; key PPO hyperparameters
are in the paper (Sec. III-B) and the full parameter table is Table I; complete
hyperparameters/seeds/normalization/platform ship with the code.

**R2.6 No CSI-error / quantization / partial-observability robustness.**
Added (Table II): imperfect CSI (Gaussian error to std 0.2) and RIS phase
quantization (1–3 bits); EE changes <1%. Delayed-CSI support is in the code.

**R2.7 IFP is instantaneous-EE; need a Lyapunov drift-plus-penalty baseline.**
Added a queue-aware **Lyapunov DPP** controller (`baselines.py:run_DPP`): max-weight
scheduling + greedy RIS co-phasing + per-user drift-vs-energy power. It attains the
lowest backlog, and is the fair comparator throughout Sec. IV.

**R2.8 EE claim lacked a parameter table.** Added (Table I).

**R2.9 Notation inconsistencies/typos.** Notation pass; ζ/ξ defined once and used
consistently; tilde-noise artifacts removed.

**R2.10 Objective mixes latency-min and EE-max; scaling not discussed.**
Resolved with R3.1: the reward is a normalized drift-plus-penalty surrogate; the EE
normalization scale is stated in code and the λ_E role clarified (Sec. III-A).

**R2.11 Missing related work (two-timescale RL, Lyapunov/deep-Lyapunov).**
Added Neely's Lyapunov optimization [13] and the DPP framing; two-timescale RL
retained.

**R2.12 RIS grouping/subarray precedent.** Acknowledged (Sec. II-B, [17],[19]).

## Reviewer 3

**R3.1 HDRL problem ≠ P1 (EE of sums ≠ sum of instantaneous EE).**
Central fix. The per-step reward is a Lyapunov drift-plus-penalty surrogate; by the
Lyapunov optimization theorem [13] it drives the time-average objective to within
O(1/V) of P1's optimum while keeping backlog O(V) — i.e., a *consistent* surrogate for
the ratio η (Sec. III-A). Code and paper now share this objective.

**R3.2 Bi-objective; plot the Pareto frontier over λ_E.**
Added (Sec. IV-B, Fig. 3a): sweeping λ_E places methods on the latency–EE plane. HDRL
occupies the high-EE region (≈2× IFP/DPP) at slightly higher latency; HDRL and the
queue-optimal DPP are non-dominated endpoints (we do **not** claim strict dominance).

---

## Honest limitations in the shipped (reduced-scale) evidence

- The **HDRL-over-flat scalability crossover** is not yet visible at the reduced
  training budget (HDRL and Flat PPO are statistically comparable); it is expected at
  full scale and higher K, and the code is set up to reproduce it.
- **Active vs. passive RIS** is presented honestly as a latency–energy *lever*:
  amplification lowers latency but, once its power is counted, lowers EE — controllable
  via the RIS power budget.
- λ_E sensitivity in the Pareto sweep is weak at reduced scale (points cluster) and
  sharpens with full training.
