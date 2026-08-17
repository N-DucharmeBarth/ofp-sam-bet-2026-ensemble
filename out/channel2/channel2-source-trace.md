# Channel 2 source trace — mixing-period term content in MFCL

**Source:** `PacificCommunity/multifan-cl`, commit `624dee4f3a9cb4063fde2c3fbd2521dbf907d161` (2025-11-05).
**Binary used for Task 1:** repository-root `mfclo64`, version string `2.2.7.9` — at or above the
2.2.7.6 threshold at which `tag_flags(i,2)` stopped being an inert placeholder.
**Configuration in these PARs:** `age_flags(198) = 1`, `parest_flags(111) = 4`, `age_flags(96) = 30`,
`parest_flags(33) = 99`, `parest_flags(387) = 0`, `age_flags(117) = (118) = (119) = 0` — all constant
across the 80 retained members.

Everything below is transcribed, not inferred.

---

## 1. Mixing-period returns DO enter the likelihood

`src/tagfit.cpp`, `fit_tag_returns()`. The period loop starts at `initial_tag_period`, i.e. at the
release period — it does **not** skip the mixing window:

```cpp
for (int ip=initial_tag_period(it,ir);ip<=ub;ip++)
{
  for (fi=1;fi<=num_fish_incidents(ir,ip);fi++)
  {
    ...
    dvar_vector rtc;
    if (age_flags(198))
      rtc=tag_rep_rate(it,ir,ip,fi)*tagcatch(it,ir,ip,fi);      // line 182
    else
      rtc=rep_rate(ir,ip,fi)*tagcatch(it,ir,ip,fi);
    ...
    int pf111=parest_flags(111);
    switch (pf111) { ... case 4: ... }
```

with (`src/tagfit.cpp:38`)

```cpp
dvariable dvar_fish_stock_history::tag_rep_rate(int it,int ir,int rp,int fi)
{
   int pp=parent(ir,rp,fi);
   return tag_fish_rep(it,pp);
}
```

So predicted **reported** returns are `X̂ · tagcatch` in every period, mixing window included, and the
scaling never consults `tag_flags(it,2)`. This confirms the issue's premise: the settings do not
differ in *whether* mixing-period returns score, only in what `tagcatch` contains.

## 2. What `tagcatch` contains inside the window

`src/tag3.cpp:470–525`. The window and post-window branches build `tagcatch` from different objects:

```cpp
if (!tag_flags(it,1) || ip >= initial_tag_period(it,ir)+tag_flags(it,1))
{                                   // POST-MIXING: model-predicted catch
  tc=mfexp(fish_mort_calcs(ir,ip,fi)(jmin,ng)+tagnum_fish(it,ir,ip));
}
else
{                                   // INSIDE THE WINDOW: Newton-Raphson solution
  tagcatch(it,ir,ip,fi)=
    mfexp(nrfm(it,ir,ip,fi) - tmp1 + tmp2 + tagnum_fish(it,ir,ip));
}
```

The Newton–Raphson (`do_newton_raphson_for_tags2`, `src/tag3.cpp:2769+`) solves for the fishing
mortality that reproduces a **target removal**, and that target is where the flag acts
(`src/tag3.cpp:2810–2833`):

```cpp
if (tag_flags(it,2)==0)                                   // include
{
  ...
  actual_tag_catch = elem_div(tot_tag_catch(it,ir,ip),1.e-6+tmp);   // tmp = tag_rep_rate
}
else                                                       // exclude
{
  actual_tag_catch = tot_tag_catch(it,ir,ip);
}
```

`tot_tag_catch` is the **observed** tag catch. So, writing `R` for observed returns:

| | target removal | NR drives `tagcatch` to | predicted reported `X̂·tagcatch` |
|---|---|---|---|
| `include` (`flag = 0`) | `R / X̂` | `≈ R / X̂` | `≈ R` — **X̂ cancels** |
| `exclude` (`flag = 1`) | `R` | `≈ R` | `≈ X̂ · R` — **minimised at X̂ = 1** |

That is the channel-2 claim, confirmed symbolically in the production path. Under `exclude` the
mixing-period terms are not evidence about the tagged population: they are pseudo-data asserting
complete reporting, and they pull `X̂` upward against the tag-seeding prior.

### Two documented reasons the cancellation is only approximate

Both matter for Task 1 and are the reason the `include` profile is not perfectly flat.

1. **The survival floor breaks it.** Still in `do_newton_raphson_for_tags2` (`src/tag3.cpp:2856–2884`),
   when the raised removal is large enough to drive survival to the floor the NR target is rescaled:

   ```cpp
   dvariable surv_rate=(1.0-region_tot_tag_catch/tnf);
   MY_DOUBLE_TYPE cut=0.2, fringe=0.02;          // age_flags(118)/(119) unset here
   if (surv_rate<=cut+fringe)
   {
     dvariable ks=fringe+posfun(surv_rate-fringe,cut,tmp);
     ffpen+=penwt*tmp;
     dvariable kr= (1.0-ks)*tnf/region_tot_tag_catch;
     kc=kr*actual_tag_catch+1.e-10;              // <-- target is now kr * (R/X̂)
   }
   else
   {
     kc=actual_tag_catch+1.e-10;
   }
   ```

   With `kr ≠ 1` the predicted reported returns become `X̂ · kr(X̂) · R / X̂ = kr(X̂) · R`, which still
   depends on `X̂` through `kr`. Only the `include` arm can reach this branch, because only it divides
   by `X̂`; the previous analysis found the floor engaging in **100%** of `include` members.

2. **The NR matches age-summed removals, the likelihood is by age class.** `tot_tag_catch` is summed
   over ages while `obstagcatch` is resolved by age, so the cancellation is exact for the age-summed
   predicted returns and the per-age split is model-determined.

## 3. The report writers use a different multiplier from the likelihood

`src/tag3.cpp:2264–2297`:

```cpp
MY_DOUBLE_TYPE dvar_fish_stock_history::get_rep_rate_correction(int it,int ir,int ip,int fi)
{
  if (!num_fish_incidents(ir,ip) || !tag_flags(it,1)
     || ip >= initial_tag_period(it,ir)+tag_flags(it,1))
  {
    if (age_flags(198)) return value(tag_rep_rate(it,ir,ip,fi));
    else                return value(rep_rate(ir,ip,fi));
  }
  else  // doing the newton raphson  -- i.e. INSIDE the mixing window
  {
    if (tag_flags(it,2)==1) { return 1.0; }                    // <-- exclude
    else { ... return value(tag_rep_rate(it,ir,ip,fi)); }
  }
}
```

It is called only by the fit-report writers — `src/tag3.cpp:1008, 1018, 1095, 1105` (observed vs
predicted returns by time period and fishery grouping) and `src/tag3.cpp:2217, 2244` (observed vs
predicted by time at liberty). Both sections are written into `plot.rep` and are the tag-fit
diagnostics circulated to SC.

**The discrepancy still holds, and it is `exclude`-only.** Inside a mixing window:

| | report writer multiplier | likelihood multiplier | reported predicted | predicted the objective used |
|---|---|---|---|---|
| `include` | `X̂` | `X̂` | `≈ R` | `≈ R` — agree |
| `exclude` | `1.0` | `X̂` | `≈ R` | `≈ X̂ · R` — **differ by a factor `X̂`** |

So under the configured `exclude` setting the circulated tag-fit diagnostic shows in-window predicted
returns landing on the observations, while the objective was scored against predictions a factor `X̂`
smaller. The plots look better inside mixing windows than the fit actually was. Magnitudes are
quantified in `report-vs-likelihood-discrepancy.csv`.

## 4. The penalty is unaffected by the flag

`src/callpen.cpp` (transcribed in the previous analysis) reads only `tag_fish_rep`,
`tag_fish_rep_target` and `tag_fish_rep_penalty`. It contains no reference to `tag_flags`, so the
reporting-rate penalty is the *same function* of `X̂` under both arms. Task 1 confirms this
numerically: at every grid point the `include` and `exclude` runs return an identical
`rr_penalty` to all printed digits.

---

## What this does and does not establish

- **Established by transcription:** mixing-period returns score under both settings; the flag changes
  the NR target; under `exclude` predicted reported in-window returns carry a bare factor `X̂` and are
  therefore minimised at `X̂ = 1`; the penalty function is arm-invariant; the report writers and the
  likelihood disagree inside mixing windows under `exclude`.
- **Not established by transcription:** that the cancellation is numerically exact under `include`.
  The two mechanisms in §2 predict it will not be, and Task 1 measures how far off it is.
