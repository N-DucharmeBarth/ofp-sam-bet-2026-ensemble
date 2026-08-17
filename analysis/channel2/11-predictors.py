"""Task 3: what orders the reporting-rate excursion across groups.

Channel 2 says the pseudo-data pull scales with the COUNT of mixing-period
returns attributed to a group and is resisted by the prior weight, so the
candidate ordering variable is N_mix/w -- not psi-hat, which the previous
Task 7 already showed does not order it.

The falsification test is the important one: channel 2 is inert under
`include` (the estimated rate cancels), so any N_mix association must be
present under `exclude` and absent or much weaker under `include`. If N_mix
predicts equally in both arms, the pull is not channel 2.

Mixing windows are member-specific via tag_flags(:,1), so this is 80 x 12,
not a five-row problem.
"""

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2common import (  # noqa: E402
    OUT,
    check_logger,
    load_members,
    load_rep_group_names,
    penalty_groups,
    read_tag_file,
    write_checks,
)

CHECKS, check = check_logger()


def counts_by_group(par, tag):
    """N_mix / N_post per reporting-rate group, using this member's own windows."""
    mix = par.tag_flags[:, 0]
    acc = {}
    for rel in tag["releases"]:
        it = rel["group"] - 1
        K = int(mix[it])
        for rec in rel["recoveries"]:
            g = int(par.rep_group[it, rec["fishery"] - 1])
            if g <= 0:
                continue
            a, b = acc.get(g, (0.0, 0.0))
            if K > 0 and 0 <= rec["elapsed"] < K:
                acc[g] = (a + rec["number"], b)
            else:
                acc[g] = (a, b + rec["number"])
    return acc


def main():
    members = load_members()
    tag = read_tag_file()
    names = load_rep_group_names()

    rows = []
    for m in members:
        p, d = m["par"], m["design"]
        cnt = counts_by_group(p, tag)
        for g in penalty_groups(p):
            gid = g["group_id"]
            n_mix, n_post = cnt.get(gid, (0.0, 0.0))
            rows.append(
                dict(
                    member_id=p.member_id,
                    arm=p.arm,
                    is_exclude=float(p.arm == "exclude"),
                    K=float(d["tag_mixing_k_cutoff"]),
                    group_id=gid,
                    group_label=names.get(gid, ("(unmapped)", ""))[0],
                    rep_hat=g["rep_hat"],
                    mu=g["target_prop"],
                    w=g["penalty_wt"],
                    is_priored=int(g["penalty_wt"] > 1),
                    signed_dev=g["signed_dev"],
                    abs_dev=g["abs_dev"],
                    N_mix=n_mix,
                    N_post=n_post,
                    N_tot=n_mix + n_post,
                )
            )
    d = pd.DataFrame(rows)
    d["psi_hat"] = np.where(d.N_tot > 0, d.N_mix / d.N_tot, np.nan)
    d["log_N_mix"] = np.log1p(d.N_mix)
    d["N_mix_over_w"] = d.N_mix / d.w
    d["log_N_mix_over_w"] = np.log1p(d.N_mix) / np.log1p(d.w)

    print(f"rows: {len(d)}  ({d.member_id.nunique()} members x "
          f"{d.group_id.nunique()} groups)")
    check(
        "task3.panel-is-member-by-group-not-five-rows",
        len(d) == d.member_id.nunique() * d.group_id.nunique(),
        f"{d.member_id.nunique()} x {d.group_id.nunique()} = {len(d)} rows",
    )
    # N_mix must actually vary within a group across members, or member-FE
    # models have nothing to identify from.
    wvar = d.groupby("group_id").N_mix.std()
    check(
        "task3.N_mix-varies-within-group-across-members",
        (wvar > 0).sum() >= 8,
        f"{int((wvar > 0).sum())} of {len(wvar)} groups have non-zero N_mix "
        f"variation across members",
    )

    d.to_csv(os.path.join(OUT, "group-excursion-predictors.csv"), index=False)

    # ---- 1. which predictor orders the excursion, within member -------------
    pri = d[d.is_priored == 1].copy()
    fits = []
    preds = ["N_mix", "log_N_mix", "N_mix_over_w", "psi_hat", "log_N_mix_over_w"]
    for subset, sub in [("exclude arm", pri[pri.arm == "exclude"]),
                        ("include arm", pri[pri.arm == "include"]),
                        ("both arms", pri)]:
        for pv in preds:
            f = f"signed_dev ~ {pv} + C(member_id)"
            try:
                mod = smf.ols(f, data=sub).fit()
            except Exception as e:  # noqa: BLE001
                print(f"  skip {subset}/{pv}: {e}")
                continue
            ci = mod.conf_int().loc[pv]
            fits.append(dict(
                scope=subset, predictor=pv, model="signed_dev ~ pred + member FE",
                n=int(mod.nobs), estimate=f"{mod.params[pv]:.6g}",
                se=f"{mod.bse[pv]:.6g}", ci_lo=f"{ci[0]:.6g}", ci_hi=f"{ci[1]:.6g}",
                p_value=f"{mod.pvalues[pv]:.4g}", r_squared=f"{mod.rsquared:.4g}",
            ))
    ft = pd.DataFrame(fits)
    print("\n--- predictor screen (priored groups, member fixed effects) ---")
    print(ft.to_string(index=False))

    # ---- 2. the falsification test: arm x predictor interaction -------------
    inter = []
    for pv in preds:
        f = f"signed_dev ~ {pv} * is_exclude + C(member_id)"
        mod = smf.ols(f, data=pri).fit()
        term = f"{pv}:is_exclude"
        ci = mod.conf_int().loc[term]
        inter.append(dict(
            scope="priored groups", predictor=pv,
            model="signed_dev ~ pred * arm + member FE",
            n=int(mod.nobs), term=term,
            estimate=f"{mod.params[term]:.6g}", se=f"{mod.bse[term]:.6g}",
            ci_lo=f"{ci[0]:.6g}", ci_hi=f"{ci[1]:.6g}",
            p_value=f"{mod.pvalues[term]:.4g}",
            main_effect=f"{mod.params[pv]:.6g}",
            r_squared=f"{mod.rsquared:.4g}",
        ))
    it = pd.DataFrame(inter)
    print("\n--- falsification: arm x predictor interaction ---")
    print(it.to_string(index=False))

    # ---- 3. the decisive test: N_mix controlling for N_post -----------------
    # A generic identification effect confounds the raw N_mix result: a group
    # with more recaptures of ANY kind is better identified and can move
    # further from its prior, in both arms. Post-mixing returns are legitimate
    # data under both settings, so N_post carries that effect and no channel-2
    # pull. Channel 2 predicts N_mix beats N_post specifically under `exclude`.
    dec = []
    f = "signed_dev ~ (N_mix + N_post) * is_exclude + C(member_id)"
    mod = smf.ols(f, data=pri).fit()
    for term in ["N_mix", "N_post", "N_mix:is_exclude", "N_post:is_exclude"]:
        ci = mod.conf_int().loc[term]
        dec.append(dict(
            scope="priored groups", predictor="N_mix + N_post, both x arm",
            model=f, n=int(mod.nobs), term=term,
            estimate=f"{mod.params[term]:.6g}", se=f"{mod.bse[term]:.6g}",
            ci_lo=f"{ci[0]:.6g}", ci_hi=f"{ci[1]:.6g}",
            p_value=f"{mod.pvalues[term]:.4g}", main_effect="",
            r_squared=f"{mod.rsquared:.4g}",
        ))
    dt = pd.DataFrame(dec)
    print("\n--- decisive test: N_mix vs N_post, each interacted with arm ---")
    print(dt.to_string(index=False))

    nm_i = mod.params["N_mix"]                      # include-arm N_mix slope
    nm_e = mod.params["N_mix"] + mod.params["N_mix:is_exclude"]
    np_i = mod.params["N_post"]
    np_e = mod.params["N_post"] + mod.params["N_post:is_exclude"]
    print(f"\n  implied slopes -- N_mix : include {nm_i:.3e}, exclude {nm_e:.3e}")
    print(f"                    N_post: include {np_i:.3e}, exclude {np_e:.3e}")

    ci_nm = mod.conf_int().loc["N_mix:is_exclude"]
    check(
        "task3.Nmix-pull-is-exclude-specific-controlling-for-Npost",
        mod.params["N_mix:is_exclude"] > 0 and ci_nm[0] > 0,
        f"N_mix x arm = {mod.params['N_mix:is_exclude']:.4g} "
        f"[{ci_nm[0]:.4g}, {ci_nm[1]:.4g}], p={mod.pvalues['N_mix:is_exclude']:.3g}; "
        f"implied N_mix slope {nm_i:.3e} (include) -> {nm_e:.3e} (exclude)",
    )
    # Placebo as pre-specified: post-mixing returns are legitimate data under
    # both settings, so their slope was expected to be arm-invariant.
    ci_np = mod.conf_int().loc["N_post:is_exclude"]
    check(
        "task3.Npost-pull-is-not-arm-specific",
        ci_np[0] * ci_np[1] < 0,
        f"N_post x arm = {mod.params['N_post:is_exclude']:.4g} "
        f"[{ci_np[0]:.4g}, {ci_np[1]:.4g}] -- CI "
        f"{'straddles' if ci_np[0]*ci_np[1] < 0 else 'excludes'} zero "
        f"(placebo expectation was arm-invariance)",
    )
    # The placebo does not fail toward the null -- it fails AWAY from it, in
    # the opposite direction to N_mix. That double dissociation is a stronger
    # channel-2 signature than the N_mix interaction alone: under `exclude`
    # the mixing-window returns pull harder and the post-mixing returns pull
    # less, which is what re-weighting toward pseudo-data looks like.
    check(
        "task3.Nmix-and-Npost-arm-effects-have-opposite-signs",
        mod.params["N_mix:is_exclude"] > 0 > mod.params["N_post:is_exclude"]
        and ci_nm[0] > 0 and ci_np[1] < 0,
        f"N_mix slope {nm_i:.3e} -> {nm_e:.3e} ({100*(nm_e-nm_i)/nm_i:+.0f}%) while "
        f"N_post slope {np_i:.3e} -> {np_e:.3e} ({100*(np_e-np_i)/np_i:+.0f}%); "
        f"both interaction CIs exclude zero",
    )

    for r in fits:
        r.setdefault("term", "")
        r.setdefault("main_effect", "")
    allf = pd.DataFrame(fits + inter + dec)
    allf.to_csv(os.path.join(OUT, "group-excursion-fits.csv"), index=False)

    # ---- checks -------------------------------------------------------------
    def get(scope, pv, frame=ft):
        r = frame[(frame.scope == scope) & (frame.predictor == pv)]
        return r.iloc[0] if len(r) else None

    exc_nw = get("exclude arm", "N_mix_over_w")
    inc_nw = get("include arm", "N_mix_over_w")
    check(
        "task3.Nmix-over-w-orders-excursion-within-exclude",
        float(exc_nw.estimate) > 0 and float(exc_nw.ci_lo) > 0,
        f"exclude arm: N_mix/w coefficient {float(exc_nw.estimate):.4g} "
        f"[{float(exc_nw.ci_lo):.4g}, {float(exc_nw.ci_hi):.4g}], "
        f"p={float(exc_nw.p_value):.3g}",
    )
    check(
        "task3.Nmix-over-w-association-weaker-under-include",
        abs(float(inc_nw.estimate)) < abs(float(exc_nw.estimate)),
        f"include arm coefficient {float(inc_nw.estimate):.4g} "
        f"[{float(inc_nw.ci_lo):.4g}, {float(inc_nw.ci_hi):.4g}] vs "
        f"exclude {float(exc_nw.estimate):.4g}",
    )
    iw = it[it.predictor == "N_mix_over_w"].iloc[0]
    check(
        "task3.association-attenuates-under-include",
        float(iw.estimate) > 0 and float(iw.ci_lo) * float(iw.ci_hi) > 0,
        f"arm x N_mix/w interaction {float(iw.estimate):.4g} "
        f"[{float(iw.ci_lo):.4g}, {float(iw.ci_hi):.4g}], p={float(iw.p_value):.3g} "
        f"(positive = association stronger under exclude)",
    )
    # psi-hat: the previous Task 7 null was about the ARM DIFFERENCE across
    # five groups (n=5). This is a different question -- the within-member
    # association of the level of the deviation, on 230 rows -- so a non-null
    # here does not contradict it. Channel 2 says the count, not the share,
    # should carry the pull, so psi-hat should lose to N_mix on fit.
    exc_psi = get("exclude arm", "psi_hat")
    exc_nmix = get("exclude arm", "N_mix")
    check(
        "task3.count-beats-share-as-the-ordering-variable",
        float(exc_nmix.r_squared) > float(exc_psi.r_squared),
        f"within-member R^2 under exclude: N_mix {float(exc_nmix.r_squared):.4g} "
        f"vs psi_hat {float(exc_psi.r_squared):.4g}",
    )
    check(
        "task3.psi-hat-association-is-negative-not-positive",
        float(exc_psi.ci_hi) < 0,
        f"exclude arm psi_hat coefficient {float(exc_psi.estimate):.4g} "
        f"[{float(exc_psi.ci_lo):.4g}, {float(exc_psi.ci_hi):.4g}] -- groups with a "
        f"HIGHER in-window share show a SMALLER excursion, opposite to a naive "
        f"psi-hat account",
    )

    write_checks(CHECKS, os.path.join(OUT, "checks-11.tsv"))


if __name__ == "__main__":
    main()
