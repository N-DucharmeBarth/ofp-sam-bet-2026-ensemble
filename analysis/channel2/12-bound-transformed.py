"""Task 4: resolve the bound-hit anomaly on the estimation scale.

The penalty analysis found reporting-rate bound contact (bound 0.99) in the
`include` arm only -- two members, group 18 at 0.990 -- and none under
`exclude`. That runs against the naive channel-2 expectation that `exclude` is
the arm pulled toward 1.

The proportion-space test used there (|X - 0.99| <= 1e-3) is the wrong metric:
the optimiser works on an arcsine-of-log scale (c2common.rep_to_estimation_scale),
so equal proportion gaps are not equal distances from the bound. This retests
on the estimation scale and reports the full distribution rather than contact
counts.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2common import (  # noqa: E402
    OUT,
    check_logger,
    estimation_scale_bound,
    load_members,
    load_rep_group_names,
    penalty_groups,
    rep_to_estimation_scale,
    rep_upper_bound,
    write_checks,
)

CHECKS, check = check_logger()

# Contact thresholds. The proportion one reproduces the previous analysis; the
# transformed one is stated as a fraction of the full estimable half-range so
# it is comparable across parameters.
EPS_PROP = 1e-3
EPS_V_FRAC = 0.01


def main():
    members = load_members()
    names = load_rep_group_names()
    ub = {rep_upper_bound(m["par"]) for m in members}
    check("task4.upper-bound-constant", len(ub) == 1, f"{ub}")
    UB = next(iter(ub))
    VB = estimation_scale_bound()

    rows = []
    for m in members:
        p = m["par"]
        for g in penalty_groups(p):
            v = float(rep_to_estimation_scale(g["rep_hat"], UB))
            rows.append(dict(
                member_id=p.member_id, arm=p.arm, group_id=g["group_id"],
                group_label=names.get(g["group_id"], ("(unmapped)", ""))[0],
                is_priored=int(g["penalty_wt"] > 1),
                penalty_wt=g["penalty_wt"], target_prop=g["target_prop"],
                rep_hat=g["rep_hat"],
                gap_prop=UB - g["rep_hat"],
                v_est=v,
                gap_v=VB - v,
                gap_v_frac=(VB - v) / VB,
                contact_prop=int(UB - g["rep_hat"] <= EPS_PROP),
                contact_v=int((VB - v) / VB <= EPS_V_FRAC),
            ))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "bound-contact-transformed.csv"), index=False)

    pri = d[d.is_priored == 1]
    print(f"\nbound = {UB} (proportion) = {VB:.1f} (estimation scale)")

    # --- contact counts on each scale, by arm --------------------------------
    tab = []
    for scale, col in [("proportion (<=1e-3)", "contact_prop"),
                       ("estimation (<=1% of half-range)", "contact_v")]:
        for arm in ("include", "exclude"):
            s = pri[pri.arm == arm]
            n_mem = s.groupby("member_id")[col].max().sum()
            tab.append(dict(
                scale=scale, arm=arm,
                n_members=int(s.member_id.nunique()),
                n_members_with_contact=int(n_mem),
                frac_members=f"{n_mem / s.member_id.nunique():.4g}",
                n_group_cells=int(s[col].sum()),
            ))
    tb = pd.DataFrame(tab)
    print("\n--- bound contact among priored groups ---")
    print(tb.to_string(index=False))
    tb.to_csv(os.path.join(OUT, "bound-contact-summary.csv"), index=False)

    p_inc = pri[pri.arm == "include"].groupby("member_id").contact_prop.max().mean()
    p_exc = pri[pri.arm == "exclude"].groupby("member_id").contact_prop.max().mean()
    v_inc = pri[pri.arm == "include"].groupby("member_id").contact_v.max().mean()
    v_exc = pri[pri.arm == "exclude"].groupby("member_id").contact_v.max().mean()
    check(
        "task4.bound-contact-test-in-transformed-space",
        (v_exc >= v_inc) != (p_exc >= p_inc),
        f"proportion scale: include {p_inc:.1%} vs exclude {p_exc:.1%}; "
        f"estimation scale: include {v_inc:.1%} vs exclude {v_exc:.1%} -- ordering "
        f"{'REVERSES' if (v_exc >= v_inc) != (p_exc >= p_inc) else 'does not reverse'}",
    )

    # --- full distribution for group 18, the group in question ---------------
    print("\n--- distribution of X-hat by arm, priored groups ---")
    dist_rows = []
    for gid, sub in pri.groupby("group_id"):
        rec = dict(group_id=gid,
                   group_label=names.get(gid, ("(unmapped)", ""))[0],
                   penalty_wt=f"{sub.penalty_wt.iloc[0]:.6g}")
        for arm in ("include", "exclude"):
            s = sub[sub.arm == arm]
            rec[f"n_{arm}"] = len(s)
            rec[f"rep_med_{arm}"] = f"{s.rep_hat.median():.6g}"
            rec[f"rep_p90_{arm}"] = f"{s.rep_hat.quantile(.9):.6g}"
            rec[f"rep_max_{arm}"] = f"{s.rep_hat.max():.6g}"
            rec[f"gap_v_med_{arm}"] = f"{s.gap_v.median():.6g}"
            rec[f"gap_v_min_{arm}"] = f"{s.gap_v.min():.6g}"
        u = stats.mannwhitneyu(sub[sub.arm == "exclude"].rep_hat,
                               sub[sub.arm == "include"].rep_hat,
                               alternative="two-sided")
        rec["rep_mwu_p"] = f"{u.pvalue:.4g}"
        dist_rows.append(rec)
    dd = pd.DataFrame(dist_rows)
    print(dd[["group_id", "penalty_wt", "rep_med_include", "rep_med_exclude",
              "rep_max_include", "rep_max_exclude", "gap_v_med_include",
              "gap_v_med_exclude", "rep_mwu_p"]].to_string(index=False))
    dd.to_csv(os.path.join(OUT, "bound-contact-distributions.csv"), index=False)

    # Is the exclude arm simply further along the transform without tripping
    # the numerical threshold? Compare median transformed gaps.
    g18 = pri[pri.group_id == 18]
    gi = g18[g18.arm == "include"].gap_v.median()
    ge = g18[g18.arm == "exclude"].gap_v.median()
    check(
        "task4.exclude-is-closer-to-the-bound-in-median-despite-no-contact",
        ge < gi,
        f"group 18 median transformed gap: include {gi:.4g} vs exclude {ge:.4g} "
        f"(smaller = closer to the bound)",
    )
    # Across all priored groups
    ai = pri[pri.arm == "include"].gap_v.median()
    ae = pri[pri.arm == "exclude"].gap_v.median()
    check(
        "task4.exclude-closer-to-bound-across-all-priored-groups",
        ae < ai,
        f"median transformed gap over all priored groups: include {ai:.4g} vs "
        f"exclude {ae:.4g}",
    )
    # The include-arm contacts: are they outliers rather than the arm's tendency?
    inc18 = g18[g18.arm == "include"].rep_hat
    n_out = int((inc18 > g18[g18.arm == "exclude"].rep_hat.max()).sum())
    check(
        "task4.include-bound-contacts-are-upper-tail-outliers",
        n_out > 0 and n_out <= 3,
        f"{n_out} include-arm group-18 estimates exceed the exclude-arm maximum "
        f"({g18[g18.arm=='exclude'].rep_hat.max():.4g}); include median "
        f"{inc18.median():.4g} sits below the exclude median "
        f"{g18[g18.arm=='exclude'].rep_hat.median():.4g}",
    )

    write_checks(CHECKS, os.path.join(OUT, "checks-12.tsv"))


if __name__ == "__main__":
    main()
