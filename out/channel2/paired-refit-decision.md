# Paired refit — recommendation

**Recommendation (second revision): paired refits are NOT needed to identify the flag's effect. The
grid is a randomised experiment on the RR axis and already identifies it.** What remains open is
differential attrition, and a targeted covariation question the grid is underpowered for. No
estimation run has been launched.

> **Revision note 1.** The first version recommended extending the Task 1 profile to record F, on the
> reasoning that ~40 minutes of evaluation would settle the pathway before spending refit compute.
> That extension was run (168 evaluations) and returned an **exact structural null**: F, F/F_MSY,
> SB/SB_MSY and SB_recent each take exactly one value per member while the objective takes 56. The
> null is stronger than first reported — at matched X-hat the two arms give **identical** F to
> 0.000e+00 across all 84 matched grid points, so *neither* the reporting rate *nor the flag itself*
> moves F at frozen parameters. Both channels are estimation effects, not arithmetic ones.
>
> **Revision note 2 — the claim being corrected.** That null was then used to conclude that "a paired
> refit is the only instrument that can answer the downstream question". **That conclusion conflated
> two different questions and was wrong on the first of them.**
>
> * *Total effect of the flag on F and stock status* — **already identified, no refit needed.**
>   `scripts/randomize-pairing.R` assigns every axis by independent random permutation, rejecting
>   draws whose max |Spearman| between axes exceeds 0.10. The RR flag is randomised with respect to
>   every other axis **by construction**, so the adjusted arm effect is a causal estimate, not an
>   adjusted association. The empirical balance confirms it (M0 standardised difference −0.076, h
>   −0.198, tau and effort creep balanced). A paired refit would buy precision against an n = 88
>   randomised comparison, not identification.
> * *Decomposition into channel 1 and channel 2* — **not identified, and a paired refit does not fix
>   it either.** X-hat is a post-treatment variable, so conditioning on it is invalid whether the
>   design is paired or not. A pair delivers the same total effect with less noise; it does not
>   separate the channels. That needs a different manipulation (see "Decomposing the channels").
>
> The earlier memo's instrument table is corrected accordingly below.

## What the design already gives you

| Question | Identified by the existing grid? |
|---|---|
| Total effect of `tag_flags(:,2)` on F, depletion, the penalty | **Yes** — random assignment; F/F_MSY −0.166 [−0.235, −0.098] |
| Whether the effect varies with **K** | **Yes** — arm × K on the penalty −211 [−288, −135], p = 4e−07; on depletion −0.408 [−0.754, −0.062], p = 0.021 |
| Whether the effect varies with **tau** | **No — underpowered.** arm × tau CI on F/F_MSY is −0.767 to 1.013 per unit tau; across the 1.2–1.4 range that is −0.153 to +0.203, versus a main effect of −0.166. An interaction as large as the main effect is entirely consistent with the data. This is absence of power, not absence of effect. |
| Channel 1 versus channel 2 | **No** — post-treatment conditioning; needs a new manipulation |
| Robustness to differential attrition (18% vs 6%) | **No** — post-randomisation selection, informative |

There are no natural pairs to exploit: zero retained members share all five other axis values with an
opposite-arm member, and with h and M0 drawn continuously there never could be. Pairing would have to
be *created* by refitting.

## The triggers, evaluated literally

The issue pre-specified these and told me not to adjust them after seeing results. I have not. Two
of the three "recommend against" conditions are clearly not met; the third is met on its wording but
not on its stated rationale, and that discrepancy is the whole of this recommendation.

| Pre-specified trigger | Met? |
|---|---|
| **For:** Task 1 confirms the signature **and** Task 5 shows an arm effect with CI excluding zero **and** that effect attenuates materially when conditioning on X̂ | **Partly.** Signature confirmed (with the floor caveat); arm effect confirmed (F/F_MSY −0.166 [−0.235, −0.098]); but the effect does **not** attenuate — it reverses. |
| **For:** Task 1 confirms the signature but Task 5 finds **no** arm effect on stock status | **No.** There is a large arm effect. |
| **Against:** Task 1 fails — the `include` mixing block is not flat | **No**, on the resolved reading. It is flat to 0.09% wherever the survival floor is inactive, and the non-flatness sits exactly where the source says the floor rescales the NR target. Note this trigger *is* met on the unconditional reading, which is why §"What would change this" below matters. |
| **Against:** Task 3's falsification fails — `N_mix` predicts equally in both arms | **No.** It nearly does on the raw comparison, but the N_mix/N_post double dissociation (+34% / −35%, both CIs excluding zero) separates them once the identification confound is controlled. |
| **Against:** the arm effect is large and does **not** attenuate when conditioning on X̂ — "the effect is then running through something other than the reporting rate" | **Met on wording, not on rationale.** The effect does not attenuate. But the reason is over-mediation, not a competing pathway: X̂ carries the outcome hard and in the predicted direction (F/F_MSY −7.59 [−9.08, −6.10]), so the indirect path is 323% of the total and the residual direct term flips sign. Nothing here points at a different pathway. |

**The triggers did not anticipate over-mediation.** They were written expecting the arm coefficient to
shrink toward zero or stay put. It reversed, which under a literal reading fires a "recommend against"
whose stated reasoning does not apply. I am not rewriting the trigger; I am reporting that the observed
pattern falls outside the space it was written for, and recommending on the two concrete gaps below
instead.

## What the null establishes

The structural null is not a wasted run. It rules out a whole class of explanation for the observed
−18% arm effect on F/F_MSY: it is **not** an arithmetic or reporting consequence of how F is computed
from a fitted model. Freeze the parameters and the effect is exactly zero. So the effect is
*optimiser behaviour* — X̂ reshapes the likelihood surface and the F-related parameters relocate.

So the effect can only be measured with the optimiser running. That does not by itself argue for a
paired refit — the randomised grid already runs the optimiser 88 times with the flag randomly
assigned. It argues that no *fixed-parameter* instrument can decompose the effect:

| Instrument | Can it decompose the −18%? |
|---|---|
| Conditioning on X̂ (Task 5) | No — X̂ and F jointly estimated; over-mediates, >100% indirect shares |
| Fixed-parameter profile (Task 1 + this extension) | **No — dF/dX̂ is structurally zero** |
| Paired refit | **No** — it re-measures the total effect with less noise; X-hat stays post-treatment |
| Refit with X-hat FIXED at a common value | **Yes** — the only design that holds the channel-2 pathway shut while the optimiser moves |

## Sequencing

Two things should still happen first, both zero-compute, because they change what a refit would be
reported alongside:

1. **Report the `get_rep_rate_correction` behaviour upstream.** The in-window tag-fit panels are an
   identity in both arms (verified against the circulated `plot.rep`: time-at-liberty bin 1 gives
   pred/obs of 1.0000–1.0207 across arms), and under `exclude` the objective scored something 21.6%
   different from what was plotted. This is an MFCL defect, not a configuration choice.
2. **Reconcile the ψ̂ baseline** against `tag-flags-implications-bet-2026.md` — the K = 0.30
   configuration gives ψ̂ = 0.226 against the published 0.231, but 35 groups at ψ̂ ≥ 0.3 against 27.

Neither blocks the refit; both should land before its results are circulated.

## Experiment A — minimal pair design, targeted at tau

**Purpose.** Not to re-measure the total effect (the grid has it) but to answer the one covariation
question the grid cannot: does the arm effect vary with tau? K is already answered and should be
carried only as a spanning variable, not as the object of study.

**Design — 8 full-path runs, a 2 × 2 × 2.**

| Factor | Levels | Why |
|---|---|---|
| `tag_flags(:,2)` | include, exclude | the contrast |
| tau | 1.2, 1.4 | the axis the grid is underpowered on; extremes maximise the contrast |
| K | 0.10, 0.30 | span the axis whose interaction is established, so a tau effect is not confounded with it |

Everything else — h, M0, effort creep — held at the ensemble median and identical across all eight
runs. This yields four pair differences, from which the main effect of tau on the pair difference, the
main effect of K, and their interaction are all readable.

**Two honest limits.** With one run per cell there are zero residual degrees of freedom, so this is
descriptive, not inferential. And MFCL is deterministic, so the only variability is optimiser
path-dependence; if a standard error is wanted, replicate by jittering start values rather than by
adding cells. A cheaper 6-run variant drops the K factor (3 K levels × 2 arms) but then cannot
separate a tau effect from a K effect.

**Do not warm-start.** A warm start from the other arm's converged PAR can land in a different optimum
than a model built through the phase sequence, so a small pair difference may reflect the start point
rather than the flag. Full-path only, via `model/doitall.sh`.

## Experiment B — decomposing channel 1 from channel 2

**Why the pair design cannot do it.** Both channels are switched by the same flag, and X-hat is
post-treatment. The decomposition needs a manipulation that holds one channel shut while the optimiser
moves.

**The manipulation.** Fix the reporting rate rather than estimating it. Channel 2 acts only through
X-hat; if X-hat cannot move, channel 2 is disabled while channel 1 (the removal difference) still
operates. Mechanically: set `tag_fish_rep` to the chosen value for every cell of the priored groups
and set the corresponding `tag_fish_rep_active_flags` to 0 so they are not estimated. The
`edit_par()` function in `analysis/channel2/15-profile.py` already writes the rate cells respecting
the grouping; it needs a companion that also zeroes the active flags.

**Design — 3 runs per cell.**

| Run | Flag | X-hat | Gives |
|---|---|---|---|
| A | exclude | estimated | total effect, exclude side |
| B | include | estimated | total effect, include side |
| C | exclude | **fixed at run B's X-hat** | exclude's removal with include's reporting rate |

Then, on any outcome:

```
total          = F(A) − F(B)
channel 1      = F(C) − F(B)     removal difference at a common reporting rate
channel 2      = F(A) − F(C)     effect of letting X-hat inflate, flag held at exclude
```

Run C is the only addition over the pair, so the decomposition costs **one extra run per cell**.

**Symmetric variant (4 runs).** Add run D — include with X-hat fixed at run A's value — and compute
the decomposition from the include side as well. Agreement between the two orderings tests
path-dependence; disagreement quantifies it. This is the standard two-sided decomposition and is worth
the fourth run if the three-run result is close to the total.

**Third arm worth having.** A run with X-hat fixed at the **tag-seeding priors** under each flag
answers a separate question that reviewers will ask: how much of the arm difference survives when
neither arm is allowed to estimate its way to an inflated rate? That is 2 more runs and it is the
version most directly tied to the external evidence.

**Caveats.** Fixing X-hat turns the reporting-rate penalty into an additive constant, so objective
values are not comparable across runs with different X-hat treatments — compare derived quantities,
not objectives. And the decomposition is exact only if the two channels are additive; if they
interact, the three-run and four-run decompositions will disagree, which is itself the diagnostic.

## Combining them

The two experiments share runs A and B. A design that answers both is **4 cells × 3 runs = 12
full-path runs**, plus 4 optional runs for the seeding-prior variant. Before committing to any of it,
time **one** full-path run end to end — the 14 s/evaluation figure from the profiling stage says
nothing about estimation cost.

## The remaining reason to refit regardless

Differential attrition. 9 of 50 `include` draws failed against 3 of 50 `exclude`, and the failures are
where the correction was least feasible, so the surviving `include` set is conditioned on the
correction being benign. Recovering those 9 members repairs the randomisation and is the only thing
that removes this concern. It is independent of Experiments A and B and arguably ranks above both.

## What the pairs would settle

Exactly one sentence, as the issue asks for — and it is writable, which is why the refit is worth
holding open rather than closing:

> Whether the −18% arm effect on F/F_MSY survives within a pair of models that differ only in
> `tag_flags(:,2)` and are otherwise built identically, replacing an adjusted association across a
> selected, non-crossed ensemble with a controlled contrast.

It is worth being clear about what that sentence does **not** cover: both channels push F the same way
and compound, so a confirmed within-pair F difference still would not attribute the effect to channel 2
specifically. Only the Task 1 profile discriminates the channels, and extending it to F is the cheaper
way to do that.
