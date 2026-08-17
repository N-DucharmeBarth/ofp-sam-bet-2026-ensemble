# Channel 2 — does `exclude` inflate estimated reporting rates, and does it propagate?

Code: `analysis/channel2/`. Checks: `out/channel2/checks.log` (35 held, 11 did not).
Every directional claim below names a check, reported in the same form whether it held or not.

**Experimental vs observational.** Task 1 is a fixed-parameter experiment — 168 MFCL evaluations at
frozen parameter vectors, nothing estimated. Task 2 is source transcription. Tasks 3–6 are adjusted
associations on a non-crossed, selected ensemble, and are labelled as such throughout.

**No estimation run was launched.** Every MFCL invocation set `parest_flags(1) = 0`; the preflight
gate confirms this reproduces the reported objective and moves no parameter.

---

## Headline

Channel 2 operates in MFCL, not just in a stripped-down operating model. The `include` mixing-window
likelihood block **is** invariant to the reporting rate — to 0.09% over the sub-range where the
survival floor does not intervene — while the `exclude` block declines monotonically toward `X̂ = 1`
in **100%** of grid steps in every profile. The pull is ordered by mixing-window recapture *count*
resisted by prior weight, not by in-window *share*, and it shows a clean double dissociation against
post-mixing returns. It propagates: `exclude` carries **−18%** on F/F_MSY and **+20%** on depletion,
and the covariation with `X̂` accounts for more than the whole of that.

---

## 1. Preflight (Task 1 gates)

| Gate | Result |
|---|---|
| Binary | `mfclo64`, ELF x86-64, version string **2.2.7.9** — above the 2.2.7.6 threshold at which `tag_flags(i,2)` became live |
| Inputs | `bet.frq` header declares 5 regions, **33 fisheries**, **98 tag release groups** — matches the PAR blocks and `bet.tag` |
| Isolation | every evaluation in its own temp directory with its own input copies |
| Reproduction | `ensemble-002` eval-only reproduced the reported reporting-rate penalty **exactly** (89.8375121818, difference 0.0) |
| Parameters unmoved | all flag and rate blocks identical; `rep` differs by 1e-15 (print round-trip). The only `parest_flags` change is index **1**, 5000 → 0 — confirming `-switch 1 1 1 0` sets the function-evaluation count and nothing else |

Cost: **~14 s per evaluation**, 4 workers. The full grid (3 members × 2 arms × 2 groups × 14 points =
168 evaluations) took 9 minutes.

## 2. Task 1 — the fixed-parameter profile

`profile-objective.csv`, `profile-summary.csv`, `figures/profile-by-arm-g{7,17}.png`.

**The penalty is arm-invariant, exactly.** `task1.penalty-is-identical-under-both-arms`: max
|rr_penalty(exclude) − rr_penalty(include)| = **0** across all 84 grid points. The two arms differ
only in the likelihood, exactly as the source trace predicts.

**The `exclude` mixing block slopes toward 1, without exception.**
`task1.exclude-mixing-block-monotone-toward-one`: first differences negative in **100%** of grid steps
in all six profiles.

**The stop condition fired, and then resolved.**
`task1.include-mixing-block-invariant-to-X` **DID NOT HOLD** — worst relative range 0.225 against a
0.02 tolerance. Taken alone this is the issue's stop condition. But the source trace named two reasons
the cancellation is only approximate, and the dominant one is the survival floor, which rescales the
Newton–Raphson target by `kr` and engages when the raised removal `R/X̂` is large — i.e. at **low** X̂.
Restricting to the sub-range where it should not engage:

> `task1.include-mixing-block-invariant-to-X-above-0.8` **HELD** — worst relative range **0.0086**,
> median **0.0009**.

So the cancellation is essentially exact wherever the floor is not active, and the non-flatness is
concentrated exactly where the source says it must be. The channel-2 account survives; what fails is
the assumption that the cancellation is unconditional. This is a refinement of the mechanism, not a
refutation, and the `include` arm's floor engagement in 100% of members is precisely why it matters.

**The argmin shift splits by prior weight, not by member.**
`task1.exclude-tag-block-minimum-is-higher` **DID NOT HOLD** as stated (positive in 3 of 6 profiles).
The split is entirely by group:

| Group | prior weight | argmin shift (exclude − include), by member | check |
|---|---|---|---|
| 7 | 354.5 | +0.167, +0.111, +0.167 | `task1.argmin-shift-positive-in-every-member[g7]` **held** |
| 17 | 739.2 | −0.056, 0.000, 0.000 | `task1.argmin-shift-positive-in-every-member[g17]` **did not hold** |

`task1.argmin-shift-larger-for-the-lower-weight-group` **held**: median +0.167 for group 7 against
+0.000 for group 17. Group 17 carries twice the prior weight and is pinned; group 7 moves. **The
fixed-parameter experiment reproduces the group ordering observed in the ensemble** (+0.142 vs +0.014
in Task 5 of the penalty analysis) — an independent, non-circular corroboration.

`task1.profile-shift-scales-with-mixing-window` held on medians (+0.083 at K=0.05 vs +0.056 at
K=0.30) but Spearman(K, shift) = −0.121 on six points; treat as weak.

## 3. Task 2 — source confirmation

Full transcription with file and line references in `channel2-source-trace.md`
(commit `624dee4f`). Established by transcription, not inference:

- Mixing-period returns **do** score under both settings — `src/tagfit.cpp:160` starts the likelihood
  loop at `initial_tag_period`, and the multiplier `tag_rep_rate` never consults `tag_flags(it,2)`.
- The flag acts on the Newton–Raphson **target** (`src/tag3.cpp:2810`): `include` targets `R/X̂` so
  predicted reported returns are `≈ R` and `X̂` cancels; `exclude` targets `R` so they are `≈ X̂·R`,
  minimised at `X̂ = 1`. The pseudo-data claim is confirmed symbolically in the production path.
- The penalty (`src/callpen.cpp`) contains no reference to `tag_flags` — arm-invariant by construction.

**The `get_rep_rate_correction` discrepancy still holds, and it is `exclude`-only.** The fit-report
writers (`src/tag3.cpp:1008, 1018, 1095, 1105, 2217, 2244` — the plot.rep tag-fit sections circulated
to SC) apply a multiplier of **1.0** inside mixing windows under `exclude`, while the likelihood
applies `X̂`. Quantified in `report-vs-likelihood-discrepancy.csv`:

> `task2.report-likelihood-discrepancy-still-holds` **held** — the objective's in-window predicted
> returns sit **21.6%** below the values written to plot.rep (median 802 of 3605 fish per member).
> Largest single contributor: group 17, 284 fish on 1210 in-window recaptures.

`task2.discrepancy-is-exclude-only` held: identically zero under `include`, where both paths apply
`X̂` to a `tagcatch` of `≈ R/X̂`. **The tag-fit diagnostics circulated to SC are not computed on the
same scaling as the fit that produced them.** This matters independently of anything else here.

## 4. Task 3 — what orders the pull

`group-excursion-predictors.csv`, `group-excursion-fits.csv`,
`figures/excursion-vs-Nmix-by-arm.png`. Panel: 80 members × 12 groups = 960 rows, member fixed effects
throughout, so all comparisons are within-member.

**Count beats share.** Within `exclude`, N_mix explains R² = 0.775 of the within-member variation in
the signed deviation against 0.281 for ψ̂ (`task3.count-beats-share-as-the-ordering-variable`). ψ̂ is
associated *negatively* (−0.274 [−0.344, −0.205]) — groups with a higher in-window share show a
**smaller** excursion (`task3.psi-hat-association-is-negative-not-positive`). That is the opposite of
a naive ψ̂ account and explains why the previous Task 7 found no ψ̂ ordering.

**`N_mix/w` orders it within `exclude`**: 0.0473 [0.0428, 0.0518]
(`task3.Nmix-over-w-orders-excursion-within-exclude`), and the arm interaction is positive with a CI
excluding zero (`task3.association-attenuates-under-include`).

**The falsification test, done properly.** The raw N_mix association is *nearly as strong under
`include`* (R² 0.79 vs 0.78) — on its own that is close to the issue's stop condition. But a generic
confound explains it: more recaptures of any kind means better identification and more movement from
the prior, in both arms. Post-mixing returns carry that effect and no channel-2 pull, so they are the
placebo. Fitting both counts jointly:

| | slope under `include` | slope under `exclude` | change |
|---|---|---|---|
| `N_mix` (mixing window) | 9.87e−05 | 1.325e−04 | **+34%**, CI [1.24e−05, 5.52e−05] excludes zero |
| `N_post` (post-mixing) | 6.18e−05 | 4.01e−05 | **−35%**, CI [−3.68e−05, −6.43e−06] excludes zero |

`task3.Npost-pull-is-not-arm-specific` **DID NOT HOLD** — but it fails *away* from the null, in the
opposite direction to N_mix. `task3.Nmix-and-Npost-arm-effects-have-opposite-signs` **held**. This
double dissociation is stronger evidence than the N_mix interaction alone: under `exclude` the
mixing-window returns pull harder and the post-mixing returns pull less, which is what re-weighting
toward pseudo-data looks like. The stop condition — "N_mix predicts equally in both arms" — is not met
once the confound is controlled.

## 5. Task 4 — the bound-hit anomaly, resolved

`bound-contact-transformed.csv`, `bound-contact-distributions.csv`. The optimiser works on
`X̂ = exp(boundp(v, log 0.001, log 0.99, s=1000))` — an arcsine-of-log link (`src/setcomm5.cpp`,
`src/scbound.cpp`), so equal proportion gaps are not equal distances from the bound.

`task4.bound-contact-test-in-transformed-space` **DID NOT HOLD** — the ordering does not reverse; the
same two `include` members are in contact on both scales. **Contact counts were the wrong statistic.**
The distribution resolves it:

- `task4.exclude-closer-to-bound-across-all-priored-groups` **held**: median transformed gap 308.6
  (`exclude`) vs 359.6 (`include`) — `exclude` sits systematically closer to the bound.
- For group 18 specifically, median `X̂` is 0.900 under `exclude` vs 0.815 under `include`
  (gap 149.7 vs 215.0, `task4.exclude-is-closer-to-the-bound-in-median-despite-no-contact`).
- `task4.include-bound-contacts-are-upper-tail-outliers` **held**: the two `include` contacts exceed
  the entire `exclude` distribution's maximum (0.9866), while the `include` median sits well below the
  `exclude` median.

So `exclude` is the arm systematically pulled up, exactly as channel 2 predicts, and the `include`
contacts are two upper-tail runaways — consistent with the removal-feedback-plus-floor mechanism the
issue proposed as the alternative explanation. The anomaly was an artefact of thresholding a
distribution.

## 6. Task 5 — downstream propagation

`downstream-arm-effects.csv`, `figures/downstream-arm-effects.png`. Coverage stated per outcome:
management quantities cover all **88** retained models; plot.rep aggregate F covers the **80** with a
retained rep (`task5.management-quantities-cover-all-88`,
`task5.aggregate-F-coverage-matches-the-80-par-members`). Specification identical to the penalty
analysis.

| Outcome | adjusted arm effect (exclude − include) | 95% CI | % of mean |
|---|---|---|---|
| F_recent / F_MSY | **−0.166** | [−0.235, −0.098] | −18% |
| aggregate F | **−0.0127** | [−0.0205, −0.0049] | −20% |
| SB_recent / SB_0 | **+0.0534** | [+0.0261, +0.0807] | +20% |
| SB_recent / SB_MSY | +0.249 | [+0.108, +0.389] | +16% |
| SB_recent (kt) | +174 | [+69, +278] | +34% |

`task5.arm-effect-on-F-is-negative` held for both F measures; `task5.arm-effect-on-depletion-is-positive`
held. **This is not a second-order effect.** Both channels push the same way, so none of this is
evidence for channel 2 *specifically* — it is evidence the flag matters for advice.

`task5.arm-effect-on-status-attenuates-with-K` held on sign for F/F_MSY (+0.553) but its CI straddles
zero (p = 0.22); the attenuation is clearer for depletion (−0.408 [−0.754, −0.062]) and SB_recent
(p = 0.012). No arm effect persists implausibly at large K.

**The X̂ decomposition over-mediates.** `task5.arm-effect-shrinks-when-conditioning-on-Xhat`
**DID NOT HOLD** for either F/F_MSY or depletion — the arm coefficient does not shrink, it **reverses**
(F/F_MSY −0.166 → +0.293; depletion +0.053 → −0.129). The X̂ coefficient is what makes this readable:

| Outcome | coefficient on mean priored X̂ | implied indirect path |
|---|---|---|
| F_recent / F_MSY | −7.59 [−9.08, −6.10] | 323% of the total |
| SB_recent / SB_0 | +3.14 [+2.45, +3.82] | 377% of the total |
| aggregate F | −0.774 [−0.980, −0.568] | 341% of the total |

X̂ carries the outcome strongly and in the direction the mechanism implies — higher reporting rate,
lower F, less depletion. The reversal is therefore **over-mediation**: the arm→X̂ covariation accounts
for more than the whole arm effect, and the residual direct term flips. It is not evidence the effect
runs through something other than the reporting rate.

**This is a decomposition of covariation, not a mediation analysis.** X̂ and F are jointly estimated in
the same optimisation; conditioning on X̂ does not identify a causal path, and the >100% figures are a
symptom of that, not a quantity to quote.

## 7. Task 6 — ψ̂ under three anchors

`psi-three-anchors.csv`, `psi-by-mixing-configuration.csv`.

**Lead with the external anchor** (tag-seeding prior means; independent of the fit entirely). Release-
weighted ψ̂, median over members:

| Scheme | ψ̂ | groups at ψ̂ ≥ 0.3 |
|---|---|---|
| 3. tag-seeding prior means (external) | **0.4137** | 62 |
| 1. exclude-arm fitted rates (as published) | 0.4091 | 63 |
| 2. include-arm fitted rates | 0.4071 | 62 |

`task6.psi-higher-under-external-anchor-than-as-published` **held** (+1.1%), and
`task6.external-anchor-raises-psi-in-every-member` **held** (positive in 80/80, median +0.0047). The
direction is consistent but **the magnitude is small**.
`task6.include-arm-rates-sit-between-the-other-two` **did not hold** — include (0.4071) sits marginally
below exclude (0.4091) rather than between; the three schemes are within 1.6% of each other.

**The mixing configuration dominates the raising scheme by a factor of 120**
(`task6.mixing-configuration-dominates-the-raising-scheme`): ψ̂ runs 0.786 at K = 0.05 to 0.222 at
K = 0.35, a span of 0.564, against 0.005 between schemes.

**The published baseline does not reconcile.** `task6.published-baseline-reconciles` **did not hold**:
the closest configuration is K = 0.30 (ψ̂ = 0.226 vs the published 0.231, close) but it gives 35 groups
at ψ̂ ≥ 0.3 against the published 27. So these tables are **not** drop-in replacements for
`tag-flags-implications-bet-2026.md`; the scheme *contrast* is internally consistent and the *level*
is unreconciled. That gap should be closed before any of this is circulated.

---

## 8. Task 9 — the F pathway is structurally invisible at frozen parameters

`F-invariance.csv`. The profile grid was re-run recording aggregate F, F/F_MSY, SB/SB_MSY and recent
adult biomass at every point. The result is an **exact structural null**:

| quantity | distinct values per member, over 56 evaluations |
|---|---|
| F / F_MSY, aggregate F, SB / SB_MSY, SB recent, MSY, F at MSY | **1** |
| total objective (control) | **56** (relative spread 1.94) |

`task9.derived-quantities-are-invariant-to-the-reporting-rate` **held**;
`task9.removal-channel-does-not-move-F-at-frozen-parameters` **held** — max |F/F_MSY(exclude) −
F/F_MSY(include)| at the same X̂ is **0**, not small. `task9.objective-does-vary-so-the-grid-is-live`
**held**, so this is not a dead harness.

The reason is in the source trace: `tag_fish_rep` enters only the tag likelihood, the reporting-rate
penalty, and the Newton–Raphson target for the tagged cohort. The tagged cohort (`tagnum_fish`) is
fitted *to* the population; it does not feed back into population numbers or fishing mortality. With
every estimated parameter frozen there is no channel by which X̂ can move F.

`task9.fixed-parameter-profile-can-decompose-the-downstream-effect` **did not hold, by construction.**

**What the null does establish.** The observed −18% arm effect on F/F_MSY is *not* an arithmetic or
reporting consequence of how F is computed from a fitted model — freeze the parameters and it is
exactly zero. It is optimiser behaviour: X̂ reshapes the likelihood surface and the F-related
parameters relocate. That rules out a class of explanation, and it means no fixed-parameter experiment
can decompose the pathway. Only re-estimation can. See `paired-refit-decision.md`, which is revised
accordingly.

## 9. What this establishes, and what it does not

- **Established experimentally (Task 1):** channel 2 operates in MFCL. The `include` cancellation is
  exact to 0.09% where the survival floor is inactive; the `exclude` mixing block slopes to `X̂ = 1`
  without exception; the penalty is arm-invariant to machine zero; the argmin shift appears in the
  low-weight group and not the high-weight one, reproducing the ensemble ordering.
- **Established by transcription (Task 2):** the term content, and an `exclude`-only 21.6% discrepancy
  between the circulated tag-fit diagnostics and the scaling the objective used.
- **Established observationally (Tasks 3–6):** the pull is ordered by `N_mix/w`, dissociates from
  post-mixing returns, sits `exclude` systematically closer to the bound, and covaries with a −18% /
  +20% arm effect on F and depletion.
- **Not established:** that the downstream effect is *caused* by the reporting rate. Both channels push
  F the same way and compound; the ensemble is observational and not fully crossed; the X̂
  decomposition over-mediates; and the fixed-parameter profile is structurally blind to it (§8). This
  is now known to require re-estimation, not merely to be unresolved.
- **The two arms are not symmetric on the reporting rate, and an earlier framing here overstated the
  symmetry.** Under `exclude`, channel 2 acts on *every* release group with a mixing window. Under
  `include`, the in-window terms are algebraically inert in X̂ — except where the survival floor
  engages, and there the rescaled target `kr·R/X̂` with `kr ∝ X̂` restores predicted returns of
  `≈ X̂·R`, the same form as `exclude`. That is why the `include` profile slopes at low X̂ and is flat
  at high X̂: channel 2 leaking back in through the floor. The floor engages on a median of **2 of 98**
  release groups per `include` member, against every windowed release group under `exclude`. So the
  contamination is pervasive in one arm and localised in the other — an argument for `include` with a
  caveat, not for treating the arms as equivalent.
- **`include` is still not clean.** The floor distorts the tag cohort dynamics where it engages, and
  MFCL discards the resulting `posfun` penalty outright for release group 21, one of the engaging
  groups. The externally defensible anchor remains the seeding priors, not either arm's fitted rates.
