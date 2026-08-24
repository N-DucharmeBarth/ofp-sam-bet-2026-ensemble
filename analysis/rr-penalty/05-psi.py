"""Task 7: link the arm difference in reporting-rate excursion to psi-hat.

psi-hat here is the share of REPORTED recaptures that fall inside the release
group's configured mixing window, aggregated from (release group, fishery)
cells onto the reporting-rate group ids that actually govern those cells.

Because only five groups carry an informative prior, this is reported
descriptively. No model is fitted to n = 5.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, load_members, load_rep_group_names, penalty_groups, write_csv  # noqa: E402
from tagfile import read_tag_file  # noqa: E402

CHECKS = []


def check(name, passed, detail=""):
    CHECKS.append((name, bool(passed), detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return passed


def psi_by_group(par, tag):
    """Per reporting-rate group: recaptured fish in / out of the mixing window.

    A recovery record belongs to reporting-rate group
    tag_fish_rep_group_flags[release_group, fishery], which is exactly the
    parameter that scales its predicted return.
    """
    mix = par.tag_flags[:, 0]
    inw, tot = {}, {}
    for rel in tag["releases"]:
        it = rel["group"] - 1
        K = int(mix[it])
        for rec in rel["recoveries"]:
            j = rec["fishery"] - 1
            g = int(par.rep_group[it, j])
            if g <= 0:
                continue
            n = rec["number"]
            tot[g] = tot.get(g, 0.0) + n
            # MFCL fits periods with ip >= initial_tag_period + tag_flags(it,1),
            # and treats K == 0 as "nothing is inside the window".
            if K > 0 and 0 <= rec["elapsed"] < K:
                inw[g] = inw.get(g, 0.0) + n
    return {g: (inw.get(g, 0.0), tot[g]) for g in tot}


def main():
    members = load_members()
    tag = read_tag_file()
    names = load_rep_group_names()

    per = []
    for m in members:
        p = m["par"]
        psi = psi_by_group(p, tag)
        for g, (i_, t_) in psi.items():
            per.append(
                dict(
                    member_id=p.member_id,
                    arm=p.arm,
                    K=float(m["design"]["tag_mixing_k_cutoff"]),
                    group_id=g,
                    n_in_window=i_,
                    n_recaptured=t_,
                    psi=i_ / t_ if t_ else np.nan,
                )
            )
    psi_df = pd.DataFrame(per)

    exc = pd.read_csv(os.path.join(OUT, "rr-excursion-long.csv"))
    priored = sorted(exc.loc[exc.penalty_wt > 1, "group_id"].unique())

    rows = []
    for g in sorted(exc.group_id.unique()):
        e = exc[exc.group_id == g]
        pg = psi_df[psi_df.group_id == g]
        inc, ex = e[e.arm == "include"], e[e.arm == "exclude"]
        rows.append(
            dict(
                group_id=g,
                group_label=names.get(g, ("(unmapped)", ""))[0],
                penalty_wt=f"{e.penalty_wt.iloc[0]:.6g}",
                is_priored=int(e.penalty_wt.iloc[0] > 1),
                n_recaptured=f"{pg.n_recaptured.median():.6g}" if len(pg) else "0",
                psi_median=f"{pg.psi.median():.6g}" if len(pg) else "",
                psi_median_include=f"{pg[pg.arm=='include'].psi.median():.6g}" if len(pg) else "",
                psi_median_exclude=f"{pg[pg.arm=='exclude'].psi.median():.6g}" if len(pg) else "",
                abs_dev_med_include=f"{inc.abs_dev.median():.6g}",
                abs_dev_med_exclude=f"{ex.abs_dev.median():.6g}",
                d_abs_dev=f"{ex.abs_dev.median() - inc.abs_dev.median():.6g}",
                d_contribution=f"{ex.contribution.median() - inc.contribution.median():.6g}",
            )
        )
    write_csv(os.path.join(OUT, "psi-vs-excursion.csv"), list(rows[0].keys()), rows)
    tab = pd.DataFrame(rows)
    print(tab[["group_id", "group_label", "penalty_wt", "n_recaptured", "psi_median",
               "abs_dev_med_include", "abs_dev_med_exclude", "d_abs_dev"]]
          .assign(group_label=lambda x: x.group_label.str.slice(0, 26)).to_string(index=False))

    # --- descriptive correlation ----------------------------------------------
    for subset, ids in [("priored groups only", priored), ("all penalised groups", sorted(exc.group_id.unique()))]:
        s = tab[tab.group_id.isin(ids)]
        x = s.psi_median.astype(float).values
        y = s.d_abs_dev.astype(float).values
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() >= 3:
            rho = stats.spearmanr(x[ok], y[ok])
            pear = stats.pearsonr(x[ok], y[ok])
            print(f"\n{subset} (n={ok.sum()}): Spearman rho = {rho.statistic:+.4f} "
                  f"(p={rho.pvalue:.3g}); Pearson r = {pear.statistic:+.4f}")
            if subset == "priored groups only":
                check(
                    "task7.excursion-difference-tracks-in-window-share",
                    rho.statistic > 0,
                    f"Spearman rho = {rho.statistic:+.4f} across n={ok.sum()} priored groups "
                    f"(descriptive only; p={rho.pvalue:.3g} is not interpretable at n={ok.sum()})",
                )

    # Is the arm difference in |dev| flat across groups regardless of psi? That
    # is the outcome that would falsify the mechanism.
    s = tab[tab.group_id.isin(priored)]
    psis = s.psi_median.astype(float).values
    devs = s.d_abs_dev.astype(float).values
    check(
        "task7.in-window-share-varies-across-priored-groups",
        np.nanmax(psis) - np.nanmin(psis) > 0.05,
        f"psi ranges {np.nanmin(psis):.3f} to {np.nanmax(psis):.3f} across priored groups",
    )
    check(
        "task7.excursion-difference-varies-across-priored-groups",
        np.nanmax(devs) - np.nanmin(devs) > 0.01,
        f"arm difference in median |dev| ranges {np.nanmin(devs):+.4f} to {np.nanmax(devs):+.4f}",
    )
    hi = s.loc[s.psi_median.astype(float).idxmax()]
    lo = s.loc[s.psi_median.astype(float).idxmin()]
    check(
        "task7.highest-psi-group-shows-larger-excursion-difference-than-lowest",
        float(hi.d_abs_dev) > float(lo.d_abs_dev),
        f"highest-psi group g{int(hi.group_id)} (psi={float(hi.psi_median):.3f}) "
        f"d|dev|={float(hi.d_abs_dev):+.4f}; lowest-psi group g{int(lo.group_id)} "
        f"(psi={float(lo.psi_median):.3f}) d|dev|={float(lo.d_abs_dev):+.4f}",
    )

    psi_df.to_csv(os.path.join(OUT, "psi-by-member-group.csv"), index=False)
    with open(os.path.join(OUT, "checks-05.txt"), "w") as fh:
        for n, pa, det in CHECKS:
            fh.write(f"{'PASS' if pa else 'FAIL'}\t{n}\t{det}\n")


if __name__ == "__main__":
    main()
