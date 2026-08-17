# Paired refit — recommendation

**Recommendation (revised): a paired refit is the only instrument that can answer the downstream
question. The cheaper alternative I proposed does not exist.** No estimation run has been launched.

> **Revision note.** The first version of this memo recommended extending the Task 1 profile to record
> F, on the reasoning that ~40 minutes of evaluation would settle the pathway before spending refit
> compute. That extension has now been run (168 evaluations) and it **returns an exact structural
> null**: across every reporting rate and both arms, F, F/F_MSY, SB/SB_MSY and SB_recent each take
> **exactly one value per member**, while the objective takes 56. `tag_fish_rep` enters only the tag
> likelihood, the reporting-rate penalty and the tagged-cohort Newton-Raphson; the tagged cohort is
> fitted *to* the population and does not feed back into population numbers or fishing mortality. With
> parameters frozen there is no channel by which X-hat can move F. The proposal was wrong, and the
> conclusion it was meant to support has flipped.

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

## What the null establishes, and why it forces the refit

The structural null is not a wasted run. It rules out a whole class of explanation for the observed
−18% arm effect on F/F_MSY: it is **not** an arithmetic or reporting consequence of how F is computed
from a fitted model. Freeze the parameters and the effect is exactly zero. So the effect is
*optimiser behaviour* — X̂ reshapes the likelihood surface and the F-related parameters relocate.

That is precisely the thing only re-estimation can measure. There is no cheap instrument standing
between the observational association and a refit:

| Instrument | Can it decompose the −18%? |
|---|---|
| Conditioning on X̂ (Task 5) | No — X̂ and F jointly estimated; over-mediates, >100% indirect shares |
| Fixed-parameter profile (Task 1 + this extension) | **No — dF/dX̂ is structurally zero** |
| Paired refit | Yes — it is the only design that lets the optimiser move |

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

## The design, if it goes ahead

1. **Members — six, not twelve.** Three natural pairs, taken from the design table where members are
   adjacent on every axis but the RR flag; if no exact pairs exist, the K extremes plus one mid-K
   member per arm: K = 0.05 (largest windows, mechanism strongest), K = 0.20 (modal cell, 23 members,
   the best-supported comparison), K = 0.35 (windows collapse, mechanism should vanish — the internal
   control). Naming the K = 0.35 pair matters more than the others: it is the one that can falsify.
2. **Full-path, not warm-start.** These are different objects. A warm start from the other arm's
   converged PAR is cheap but can land in a different optimum than a model built through the phase
   sequence, so a small paired difference may reflect the start point rather than the flag. The
   like-for-like comparison is a full-path rerun with only the flag changed.
   **`model/doitall.sh` is present and parameterised** (`PROGRAM_PATH`, `model-inputs/*.conf`), and it
   starts from an ordinary `bet.ini -makepar`, so full-path reruns are reproducible in principle.
   Not verified end-to-end here — that verification is step 1 of any refit and its cost is unmeasured.
3. **Cost — measure, do not extrapolate.** The 14 s/evaluation figure from stage B says nothing about
   estimation. Time **one** full-path run end to end before quoting a total. Only after that number
   exists should six be committed.
4. **The zero-mixing carve-out.** The flip is per release group, never blanket. Under `include`,
   `tag_flags(i,2) = 0` only where `tag_flags(i,1) > 0`; release groups with no mixing window keep
   face-value removal under both arms, because `include` is undefined for them. `edit_par()` in
   `analysis/channel2/15-profile.py` already implements exactly this and is the function to reuse —
   a blanket `-switch -9999 2 0` would violate it.
5. **Convergence handling.** Pre-declare the criterion (maximum gradient component and the repo's own
   existing convergence screen). If one arm of a pair fails to converge, **the pair is dropped whole**
   and reported as dropped — never half-used. With six members, two dropped pairs leaves too little to
   conclude from, which is itself a reason to measure cost before committing.

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
