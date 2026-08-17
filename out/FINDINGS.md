# Reporting-rate penalty: what the include/exclude flag actually does

Analysis code: `analysis/rr-penalty/`, reproduced by `analysis/rr-penalty/run-all.sh`.
Every directional claim below names a check in `out/checks.tsv` (44 passed, 4 failed).
Failed checks are reported as failures, not softened.

**Scope.** The objective breakdown (`data/ensemble/objective-components.csv`) and the
design table cover all **88** retained models. Only **80** of those have a retained
`final.par`. Task 4 therefore runs on all 88; Tasks 5–8, which need par contents, run
on the 80. The headline arm effect is materially the same on both
(35.5 on 88, 34.3 on 80 — `rr-axis-marginal-effect.csv`).

---

## 1. Arithmetic: the penalty is exactly reproduced

The formula is not a guess. From `multifan-cl` `src/callpen.cpp`, the branch taken when
`age_flags(198) != 0` (which is 1 in all 80 pars):

```
for i in 1..num_tag_releases+1, j in 1..num_fisheries:
    if gflag[group(i,j)] == 0 and tag_fish_rep_penalty(i,j) > 0:
        gflag[group(i,j)] = 1
        xy += tag_fish_rep_penalty(i,j) * square(tag_fish_rep(i,j) - tag_fish_rep_target(i,j)/100)
```

**Reconstruction residual: max |recomputed − reported| = 4.9 × 10⁻¹⁰ across all 80
members** (max relative 4.0 × 10⁻¹²), against a stated tolerance of 1 × 10⁻⁶
— `task3.reconstruction-matches-reported`, `penalty-reconstruction.csv`. That is
MFCL's print precision, so the reconstruction is exact.

Three details differ from the brief's working assumptions and matter:

- **The selector is `penalty > 0`, not `tag_fish_rep_active_flags`.** The active flags
  govern which reporting rates are *estimated*; the penalty matrix governs which carry
  a prior. In this configuration the two sets happen to coincide (12 groups,
  `task2.penalised-groups-are-a-subset-of-active-groups`), so nothing changes
  numerically — but the code must select on the penalty, and it does.
- **Five groups carry an informative prior, not three.** There are three distinct
  *priors*, each shared by two groups (one RTTP, one PTTP/pooled), of which
  five groups are penalised: 7 and 14 (target 49.62%, w 354.5), 10 and 17
  (51.21%, w 739.2), 18 (52.82%, w 231.2).
- **Group 18 is `PS.EAST.3`, not region 4** (`model/tag_rep_map.R`). The three priors
  are region-2 PS, region-3-west PS and region-3-east PS.

The `tag_fish_rep*` blocks are `(ntag+1) × nfish` = 99 × 33, not `ntag × nfish`; the
extra row is the pooled release group. Dimensions are identical across all 80 members
(`task1.dims-constant-across-members`).

### A stop condition that was hit, and why it is not one

`tag_flags(:,2)` is **not constant within a par** in 26 of 80 members. It is a
*per-release-group* flag, and the design deliberately forces face-value removal
(flag 1) on release groups whose mixing window is zero, because "include" is undefined
when there is no mixing window to raise. The carve-out is exact:

- under `include`, `tag_flags(i,2) = 1` **iff** `tag_flags(i,1) = 0`, in every member
  (`task1.flag2-exceptions-are-exactly-zero-mixing-groups`);
- the count matches `tag_reporting_zero_mixing_exclusions` in the design table
  member by member;
- the arm label read off the release groups that *do* have a mixing window agrees with
  the design table for all 80 (`task1.arm-label-agrees-with-design`).

So the arm label is well defined and cross-validated. The brief's stop condition needs
restating as "not constant across release groups **with a mixing window**".

---

## 2. Empirical: the flag effect survives adjustment for M0

Raw medians (88 models): include 63.2 (IQR 41.5–87.7), exclude 101.5 (IQR 76.3–127.6),
**raw gap +38.3**.

M0 is **not** the confound. Standardised mean difference in M0 between arms is
**−0.076** — balanced (`task4.M0-balanced-across-arms`); h is −0.198, also within the
0.25 line. Adding M0 to the model *raises* the arm coefficient slightly rather than
shrinking it, from 33.98 to 35.54 (+4.6%, `task4.arm-effect-robust-to-adding-M0`),
while R² jumps from 0.454 to 0.852. M0 is a strong *predictor* (coefficient ≈ 1558
per unit quarterly M, p ≈ 1 × 10⁻²⁴) but it is orthogonal to the arm, so it explains
scatter, not the gap.

**Headline, adjusted (this is the number to quote, not the violin medians):**

> `penalty ~ arm + h + tau + K + effort creep + M0`, n = 88, continuous axes on their
> numeric scale:
> **exclude − include = +35.5 penalty units, 95% CI [28.7, 42.4], p = 1.6 × 10⁻¹⁶**
> (`task4.adjusted-arm-effect-positive`).

On the log scale the same fit gives +0.526 [0.430, 0.622], i.e. the `exclude` arm
carries about **1.69×** the penalty. Treating tau, K and effort creep as factors
instead of continuous moves the estimate to +36.5 — the choice does not matter.

**One balance check failed.** K is materially imbalanced: the largest level-share
difference is 0.170, at K = 0.20 (include 7/41 = 17%, exclude 16/47 = 34%) —
`task4.K-balanced-across-arms` **FAILED**. K is in every adjusted model and the
adjusted estimate is if anything larger than the unadjusted one, so the imbalance does
not manufacture the effect; it is reported because it is real.
See `rr-axis-balance.csv` and `figures/balance.png`.

---

## 3. Empirical: the excursion is larger under `exclude`, and it is upward

The brief's prediction is about **magnitude, not sign**. It holds for **every** priored
group (`task5.exclude-has-larger-abs-excursion-g{7,10,14,17,18}`, all PASS):

| Group | prior | median \|dev\| include | median \|dev\| exclude | diff | MWU p |
|---|---|---|---|---|---|
| 7 (RTTP PS.2) | 0.4962, w 354.5 | 0.0276 | 0.1693 | **+0.1417** | 2.8e−14 |
| 10 (RTTP PS.WEST.3) | 0.5121, w 739.2 | 0.0672 | 0.0763 | +0.0091 | 0.028 |
| 14 (PTTP PS.2) | 0.4962, w 354.5 | 0.0054 | 0.0061 | +0.0007 | 0.079 |
| 17 (PTTP PS.WEST.3) | 0.5121, w 739.2 | 0.2562 | 0.2706 | +0.0144 | 0.146 |
| 18 (PTTP PS.EAST.3) | 0.5282, w 231.2 | 0.2864 | 0.3722 | **+0.0857** | 5.3e−05 |

**The deviation is upward in both arms for all five groups**
(`task5.sign-preserved-across-arms-g*`): `exclude` pushes the estimates *further above*
their priors. Nothing here supports a downward push. Full table in
`rr-excursion-by-group.csv`; `figures/excursion-by-group-and-arm.png`.

### The arm gap is not group 17

`task5.arm-gap-dominated-by-group-17` **FAILED**, and this is a result. Group 17
dominates the *level* of the penalty (68% of the include-arm median, 53% of the
exclude-arm median) but supplies only **19.3%** of the arm *gap*. The gap comes from
**group 18 (45.0%)** and **group 7 (34.1%)**. Group 7 is the clearest single signal in
the whole analysis: its median estimate moves from 0.524 (include, essentially at its
0.496 prior) to 0.666 (exclude), and its median contribution from 0.27 to 10.2.
Priored groups supply 101.7% of the gap — the weight-1 groups net out slightly negative
(`task5.arm-gap-concentrated-in-priored-groups`).

`task5.weight-1-share-under-2pc-in-every-member` **FAILED**: the weight-1 groups
contribute a median 0.78% of the total penalty but up to **3.6%** in the worst member.
Under 5% everywhere (`task5.weight-1-share-under-5pc-in-every-member`). The story is
about five parameters, and effectively about three.

---

## 4. Empirical: the gap tracks the mixing window, as the mechanism requires

The realised mixing configuration does what the axis label implies: median
release-weighted mean mixing period falls monotonically from 3.53 periods at K = 0.05
to 0.84 at K = 0.35, Spearman(K, mean mixing) = **−1.000**
(`task6.mixing-window-decreases-monotonically-with-K`), and the count of zero-mixing
release groups rises 0 → 18 (`task6.zero-mixing-count-increases-with-K`).

Both interaction tests pass, and both CIs exclude zero:

- `penalty ~ arm * K + …`: **arm × K = −218 [−298, −138], p = 7.7 × 10⁻⁷**
  (`task6.arm-gap-shrinks-with-K`, `task6.arm-by-K-interaction-distinguishable-from-zero`)
- `penalty ~ arm * mean mixing period + …`: **+21.1 [13.7, 28.6], p = 3.1 × 10⁻⁷**
  (`task6.arm-gap-grows-with-mean-mixing-period`)

Median arm gap by K: +49, +45, +56, +17, +35, +33, **−19** at K = 0.35 (n = 5 there).
`arm-by-K-interaction.csv`, `figures/arm-gap-vs-K.png`.

**The mechanism is confirmed in the source, not just inferred.** `src/tag3.cpp`
switches on `tag_flags(it,2)` **only inside the mixing window**
(`ip < initial_tag_period + tag_flags(it,1)`):

```
tag_flags(it,2)==0  ->  actual_tag_catch = tot_tag_catch / (1e-6 + tag_rep_rate)   [include]
tag_flags(it,2)==1  ->  actual_tag_catch = tot_tag_catch                           [exclude]
```

Under `include` the mixing-period removal falls as `X_f` rises, so the surviving cohort
— and hence predicted post-mixing returns — increases with `X_f` on top of the direct
`X_f` scaling. That is the second lever, and it exists only where a mixing window
exists. Zero mixing window, no lever: exactly the K = 0.35 end where the gap vanishes.

---

## 5. Empirical: ψ̂ does **not** order the effect across groups

This is where the mechanism's group-level prediction fails. Across the five priored
groups, Spearman(in-window recapture share, arm difference in median |dev|) = **+0.10**
(`task7.excursion-difference-tracks-in-window-share` passes only on sign, and the sign
test is uninformative at n = 5; across all 12 penalised groups it is **−0.13**).

The extremes do line up — the highest-ψ̂ group (7, ψ̂ = 0.994) has the largest excursion
difference and the lowest-ψ̂ group (17, ψ̂ = 0.349) has almost the smallest
(`task7.highest-psi-group-shows-larger-excursion-difference-than-lowest`) — but the
middle does not: group 18 (ψ̂ = 0.406) shows +0.086 while group 14 (ψ̂ = 0.833) shows
+0.0007. Group 14 has **6** recaptured fish against group 18's 3117, so recapture
volume, not in-window share, is what separates them.

**Read this as: the mechanism holds at the member level (§4, strongly) but does not
distribute across groups in proportion to their in-window share.** Reported
descriptively; no model is fitted to n = 5. `psi-vs-excursion.csv`.

---

## 6. Both censoring screens fire, and both cut against `include`

**Bound.** `parest_flags(33) = 99` in every member, so `tag_fish_rep` is bounded at
**0.99** (`src/newmau5a.cpp`; `task8.upper-bound-constant-across-members`).
`task8.include-arm-not-more-often-at-the-rr-bound` **FAILED**: 2 of 34 include members
(5.9%) have a priored group pinned within 1e−3 of the bound — both group 18 at 0.990,
members 46 and 86 — against 0 of 46 exclude members. Small, but one-directional: part
of the include arm's low tail is truncation. (Weight-1 groups 19 and 23 sit at 0.99 in
most members of both arms; they carry no prior, so this does not matter.)

**Survival floor.** `src/tag3.cpp` floors survival with `posfun` once
`surv_rate = 1 − raised_removal/tags_present ≤ cut + fringe`, with `cut = 0.2` and
`fringe = 0.02` (`age_flags(117)`, `(118)` and `(119)` are all 0 in every member, so
the defaults apply — three PASS checks). Because the raised removal is
`reported / X_f`, only the `include` arm can drive it. The screen
(a bet.tag-only proxy, deliberately conservative — see `tagfile.py` and
`floor_screen`'s docstring):

- **100% of include members have ≥ 1 floor-engaging release group; 0% of exclude
  members do** (`task8.floor-engagement-higher-under-include`). Minimum proxy survival
  reaches **−2.98** under include versus 0.42 under exclude.
- **The lowest-penalty include members are the most floor-engaging**: lowest-penalty
  tercile averages 2.00 engaging release groups against 1.36 in the highest,
  Spearman(penalty, n engaging) = **−0.588**, p = 2.6 × 10⁻⁴
  (`task8.lowest-penalty-include-members-are-more-floor-engaging`).
- Release groups flagged: **98** (the pooled group, in all 34 include members),
  **21** (14 members), 15 (6), 5 (1). `src/tag3.cpp` **discards the posfun penalty
  outright** for release groups 21, 72 and 112
  (`switch (it) { case 72: case 21: case 112: break; default: _ffpen+=ffpen; }`),
  so 14 of the 55 member × release-group flags accrue no penalty at all
  (`task8.discarded-release-groups-appear-in-floor-screen`).

`bound-floor-screen.csv`, `floor-screen-by-member.csv`.

---

## 7. What this does and does not license

- **The flag axis does real, localised work on this component.** Adjusted effect
  +35.5 [28.7, 42.4] penalty units, robust to M0, and it scales with the mixing window
  exactly as an extra-lever account predicts.
- **A lower penalty is not a better model.** The `include` arm's extra lever is an
  accounting identity — the same observations, differentiated twice — not new
  information. Both censoring screens then push the same way: the include arm is the
  only one that can hit the survival floor, it does so in every member, it does so most
  in its own lowest-penalty members, and a share of that floor penalty is discarded
  outright for release group 21. Its small penalties are partly cushioned rather than
  earned, and this component should not be read as evidence for `include`.
- **The mechanism is confirmed at the member level and not at the group level.** §4
  passes decisively; §5 does not. Any account of *why* particular reporting-rate groups
  move needs something beyond in-window share — recapture volume is the obvious
  candidate and is untested here.
- **Not tested:** whether the arm difference in this component propagates to management
  quantities. Nothing here speaks to that.

---

## Outputs

| File | Task |
|---|---|
| `rr-groups.csv` | 2 — one row per (member, penalised group), 960 rows |
| `penalty-reconstruction.csv` | 3 — reported vs recomputed vs residual |
| `rr-axis-balance.csv`, `rr-axis-marginal-effect.csv` | 4 |
| `rr-excursion-by-group.csv`, `rr-excursion-long.csv` | 5 |
| `arm-by-K-interaction.csv`, `mixing-frame.csv` | 6 |
| `psi-vs-excursion.csv`, `psi-by-member-group.csv` | 7 |
| `bound-floor-screen.csv`, `floor-screen-by-member.csv` | 8 |
| `checks.tsv` | every named check, PASS/FAIL with its computed detail |
| `figures/` | arm gap vs K; excursion by group and arm; balance |
