"""Task 5: direction and magnitude of the reporting-rate excursion, by group and arm.

The falsifiable prediction from the brief is about MAGNITUDE, not sign: the
`exclude` arm should show LARGER ABSOLUTE deviations of rep_hat from its prior
target, whatever the sign of that deviation is. It is encoded below as
`task5.exclude-has-larger-abs-excursion-<g>` per priored group, and passes or
fails on its own.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, PENALTY_COMPONENT, load_members, load_rep_group_names, penalty_groups, write_csv  # noqa: E402

CHECKS = []


def check(name, passed, detail=""):
    CHECKS.append((name, bool(passed), detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return passed


def main():
    members = load_members()
    names = load_rep_group_names()

    recs = []
    for m in members:
        p = m["par"]
        tot = sum(g["contribution"] for g in penalty_groups(p))
        for g in penalty_groups(p):
            recs.append(
                dict(
                    member_id=p.member_id,
                    arm=p.arm,
                    group_id=g["group_id"],
                    rep_hat=g["rep_hat"],
                    target_prop=g["target_prop"],
                    penalty_wt=g["penalty_wt"],
                    signed_dev=g["signed_dev"],
                    abs_dev=g["abs_dev"],
                    contribution=g["contribution"],
                    share_of_total=g["contribution"] / tot,
                    total_penalty=tot,
                )
            )
    d = pd.DataFrame(recs)

    # weights and targets must be a fixed feature of the configuration
    for g, sub in d.groupby("group_id"):
        if sub["penalty_wt"].nunique() != 1 or sub["target_prop"].nunique() != 1:
            check(f"task5.group-{g}-prior-constant-across-members", False,
                  f"weights {sub.penalty_wt.unique()} targets {sub.target_prop.unique()}")
    check(
        "task5.priors-constant-across-members",
        all(
            sub["penalty_wt"].nunique() == 1 and sub["target_prop"].nunique() == 1
            for _, sub in d.groupby("group_id")
        ),
    )

    d["is_priored"] = d["penalty_wt"] > 1.0
    priored = sorted(d.loc[d.is_priored, "group_id"].unique())
    flat = sorted(d.loc[~d.is_priored, "group_id"].unique())
    print(f"priored groups (w>1): {priored}")
    print(f"weight-1 groups:      {flat}")

    # --- per-group summary -----------------------------------------------------
    rows = []
    for g, sub in d.groupby("group_id"):
        label = names.get(g, ("(unmapped)", ""))[0]
        rec = dict(
            group_id=g,
            group_label=label,
            penalty_wt=f"{sub.penalty_wt.iloc[0]:.6g}",
            target_prop=f"{sub.target_prop.iloc[0]:.6g}",
            is_priored=int(sub.penalty_wt.iloc[0] > 1.0),
        )
        for arm in ("include", "exclude"):
            s = sub[sub.arm == arm]
            for q, col in [("signed_dev", "signed"), ("abs_dev", "abs"), ("contribution", "contrib")]:
                rec[f"{col}_med_{arm}"] = f"{s[q].median():.6g}"
                rec[f"{col}_iqr_{arm}"] = f"{s[q].quantile(.75) - s[q].quantile(.25):.6g}"
            rec[f"rep_med_{arm}"] = f"{s.rep_hat.median():.6g}"
            rec[f"n_{arm}"] = len(s)
        inc, exc = sub[sub.arm == "include"], sub[sub.arm == "exclude"]
        rec["d_abs_median"] = f"{exc.abs_dev.median() - inc.abs_dev.median():.6g}"
        rec["d_signed_median"] = f"{exc.signed_dev.median() - inc.signed_dev.median():.6g}"
        rec["d_contrib_median"] = f"{exc.contribution.median() - inc.contribution.median():.6g}"
        u = stats.mannwhitneyu(exc.abs_dev, inc.abs_dev, alternative="two-sided")
        rec["abs_dev_mwu_p"] = f"{u.pvalue:.4g}"
        rec["contrib_share_of_arm_gap"] = ""
        rows.append(rec)

    # share of the include-vs-exclude total-penalty gap contributed per group
    gap_total = (
        d[d.arm == "exclude"].groupby("member_id").total_penalty.first().median()
        - d[d.arm == "include"].groupby("member_id").total_penalty.first().median()
    )
    for rec in rows:
        rec["contrib_share_of_arm_gap"] = f"{float(rec['d_contrib_median']) / gap_total:.4g}"

    write_csv(os.path.join(OUT, "rr-excursion-by-group.csv"), list(rows[0].keys()), rows)
    show = pd.DataFrame(rows)[
        ["group_id", "group_label", "penalty_wt", "target_prop", "rep_med_include",
         "rep_med_exclude", "abs_med_include", "abs_med_exclude", "d_abs_median",
         "contrib_med_include", "contrib_med_exclude", "contrib_share_of_arm_gap",
         "abs_dev_mwu_p"]
    ]
    show["group_label"] = show["group_label"].str.slice(0, 26)
    print("\n" + show.to_string(index=False))

    # --- the prediction, per priored group -------------------------------------
    print()
    for g in priored:
        sub = d[d.group_id == g]
        inc, exc = sub[sub.arm == "include"], sub[sub.arm == "exclude"]
        dm = exc.abs_dev.median() - inc.abs_dev.median()
        check(
            f"task5.exclude-has-larger-abs-excursion-g{g}",
            dm > 0,
            f"median |dev|: include {inc.abs_dev.median():.4g} -> exclude "
            f"{exc.abs_dev.median():.4g} (diff {dm:+.4g}, MWU p="
            f"{stats.mannwhitneyu(exc.abs_dev, inc.abs_dev).pvalue:.3g})",
        )
    # pooled over priored groups
    sub = d[d.is_priored]
    inc, exc = sub[sub.arm == "include"], sub[sub.arm == "exclude"]
    check(
        "task5.exclude-has-larger-abs-excursion-pooled-priored",
        exc.abs_dev.median() > inc.abs_dev.median(),
        f"median |dev| include {inc.abs_dev.median():.4g} vs exclude {exc.abs_dev.median():.4g}",
    )

    # sign preservation: the brief warns the deviation is UPWARD, so check the
    # sign is the same in both arms rather than assuming a downward push
    for g in priored:
        sub = d[d.group_id == g]
        si = np.sign(sub[sub.arm == "include"].signed_dev.median())
        se = np.sign(sub[sub.arm == "exclude"].signed_dev.median())
        check(
            f"task5.sign-preserved-across-arms-g{g}",
            si == se,
            f"median signed dev include {sub[sub.arm=='include'].signed_dev.median():+.4g}, "
            f"exclude {sub[sub.arm=='exclude'].signed_dev.median():+.4g}",
        )

    # --- weight-1 groups contribute negligibly ---------------------------------
    per_member = d.groupby(["member_id", "arm", "is_priored"]).contribution.sum().reset_index()
    piv = per_member.pivot_table(index=["member_id", "arm"], columns="is_priored",
                                 values="contribution").fillna(0.0)
    flat_share = (piv[False] / (piv[False] + piv[True]))
    # Two thresholds are reported rather than one, because the tight screen the
    # brief implies (<2% in every member) does not hold: the median member is at
    # 0.8% but the worst is at 3.6%. Both numbers are carried into FINDINGS.md.
    check(
        "task5.weight-1-share-under-2pc-in-every-member",
        flat_share.max() < 0.02,
        f"max share of total penalty from weight-1 groups = {flat_share.max():.4g} "
        f"(median {flat_share.median():.4g}, 90th pct {flat_share.quantile(.9):.4g})",
    )
    check(
        "task5.weight-1-share-under-5pc-in-every-member",
        flat_share.max() < 0.05,
        f"max = {flat_share.max():.4g}, median = {flat_share.median():.4g}",
    )

    # --- how much of the arm gap is group 17 alone? ----------------------------
    tops = sorted(
        ((rec["group_id"], float(rec["contrib_share_of_arm_gap"])) for rec in rows),
        key=lambda kv: -abs(kv[1]),
    )
    print(f"\nmedian total-penalty arm gap = {gap_total:.4g}; "
          f"per-group median-contribution shares of it: "
          + ", ".join(f"g{g}={s:.2f}" for g, s in tops[:6]))
    # The brief expects group 17 to dominate, because it dominates the LEVEL of
    # the penalty. This checks whether it also dominates the arm GAP. It does
    # not -- that is a result, not a nuisance.
    shares = dict(tops)
    check(
        "task5.arm-gap-dominated-by-group-17",
        shares[17] > 0.5,
        f"group 17 supplies {shares[17]:.1%} of the median total-penalty arm gap; "
        f"group 18 supplies {shares[18]:.1%} and group 7 {shares[7]:.1%}",
    )
    check(
        "task5.arm-gap-concentrated-in-priored-groups",
        sum(s for g, s in shares.items() if g in priored) > 0.95,
        f"priored groups supply {sum(s for g, s in shares.items() if g in priored):.1%} "
        f"of the median arm gap",
    )
    # Level vs gap: which group dominates the median TOTAL penalty in each arm?
    for arm in ("include", "exclude"):
        sub = d[d.arm == arm]
        lev = sub.groupby("group_id").contribution.median()
        lev = (lev / lev.sum()).sort_values(ascending=False)
        print(f"  {arm}: median-contribution shares of LEVEL -> "
              + ", ".join(f"g{g}={s:.2f}" for g, s in lev.head(4).items()))

    d.to_csv(os.path.join(OUT, "rr-excursion-long.csv"), index=False)
    with open(os.path.join(OUT, "checks-03.txt"), "w") as fh:
        for n, pa, det in CHECKS:
            fh.write(f"{'PASS' if pa else 'FAIL'}\t{n}\t{det}\n")


if __name__ == "__main__":
    main()
