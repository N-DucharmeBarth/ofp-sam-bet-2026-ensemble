# Paired refit — recommendation

**Recommendation: do not launch paired refits yet. Close two gaps first, then a six-member refit
becomes worth its compute.** No estimation run was launched for this issue.

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

## Why not now

1. **The decomposition cannot carry the weight the refit is meant to relieve.** X̂ and F are jointly
   estimated, so conditioning on X̂ is a decomposition of covariation, and the >100% indirect shares
   are a symptom of that. A refit is supposed to replace this step with a within-pair contrast — but
   the case for spending compute rests on the decomposition being *informative and ambiguous*, and
   right now it is *uninformative*. That is not a reason to spend compute; it is a reason to first
   establish that the pathway can be measured at all.

2. **The ψ̂ level does not reconcile with the published diagnostic** (K = 0.30 gives ψ̂ = 0.226 against
   the published 0.231, but 35 groups at ψ̂ ≥ 0.3 against 27). Until that is closed, an externally
   circulated correction cannot be stated, and the refit's main external payoff is unavailable.

3. **There is a cheaper result already in hand that has not been actioned.** The
   `get_rep_rate_correction` discrepancy means the tag-fit diagnostics circulated to SC are computed
   on a different scaling from the fit — a **21.6%** median shortfall under the configured `exclude`
   setting. That needs no compute, and it should be resolved before spending any.

## What would change this

Close both gaps, and a refit becomes worth it:

- **Reconcile the ψ̂ baseline** against `tag-flags-implications-bet-2026.md` — recover the exact mixing
  configuration and group-counting convention behind 0.231 / 27 groups. No compute.
- **Extend Task 1 to the F pathway.** The profile currently records the objective and its tag blocks.
  Record derived F at each grid point too, and the fixed-parameter experiment answers directly what
  the observational decomposition cannot: how much does F move per unit X̂ with everything else frozen?
  That is ~170 more evaluations, about **40 minutes** at the measured 14 s each — two orders of
  magnitude cheaper than a refit, and it either establishes the pathway or shows the arm effect is
  carried by something else. **Do this before any refit.**

If that extension shows F moving with X̂ at fixed parameters, then the paired refit is the right next
step and the design below applies.

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
