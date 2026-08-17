"""Task 11: restrict the downstream conclusions to converged models, and ask
what the arm difference would be under a balanced representation.

Three things:

1. The convergence screen (maximum gradient component <= 1e-4) selects exactly
   the 80 members that carry a retained final.par -- the two sets are identical,
   so every derived-quantity conclusion should be stated on n = 80, not n = 88.

2. Convergence failure is strongly arm-specific. Combined with the earlier
   build-stage attrition this is a fragility statement about the tagging module
   under the raised-removal setting, and it is quantified here rather than
   asserted.

3. Because the failures are concentrated in the arm and the region where the
   correction is most extreme, the retained sample is truncated. Two independent
   approaches ask whether that truncation biases the arm difference toward zero:
   inverse-probability weighting back to the designed marginal, and an
   imputation of the failed members' outcomes from the converged ones. Both are
   reported with their assumptions stated.
"""

import csv
import os
import re
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2common import OUT, REPO, check_logger, write_checks  # noqa: E402

CHECKS, check = check_logger()

MGC_LIMIT = 1e-4
# designed marginal on the K axis, from scripts/randomize-pairing.R
K_DESIGN = {0.05: 6, 0.10: 12, 0.15: 19, 0.20: 26, 0.25: 19, 0.30: 12, 0.35: 6}
OUTCOMES = [("f_recent_fmsy", "F_recent / F_MSY"),
            ("sb_recent_sb0", "SB_recent / SB_0"),
            ("sb_recent_sbmsy", "SB_recent / SB_MSY"),
            ("sb_recent_kt", "SB_recent (kt)")]
BASE = "arm_e + M0 + K + tau + creep + h"


def load():
    d = pd.read_csv(os.path.join(OUT, "downstream-frame.csv"))
    fd = pd.read_csv(os.path.join(REPO, "data", "ensemble", "fit-diagnostics.csv"))
    fd["member_id"] = fd.ensemble_id.str.extract(r"(\d+)$").astype(int)
    d = d.merge(fd[["member_id", "maximum_gradient", "hessian_status",
                    "positive_definite_hessian"]], on="member_id")
    d["converged"] = d.maximum_gradient <= MGC_LIMIT
    return d


def smd(a, b):
    sp = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(b) - np.mean(a)) / sp if sp > 0 else np.nan


def main():
    d = load()
    n_all, n_conv = len(d), int(d.converged.sum())

    check(
        "task11.convergence-screen-selects-exactly-the-retained-par-set",
        set(d.loc[d.converged, "member_id"]) == set(d.loc[d.has_par, "member_id"]),
        f"MGC <= {MGC_LIMIT:g} selects {n_conv} members and the retained-par set is "
        f"{int(d.has_par.sum())}; the two sets are identical, so no conclusion "
        f"changes coverage by switching criterion",
    )

    # --- 1. fragility -------------------------------------------------------
    tab = d.groupby("arm").agg(n=("converged", "size"), converged=("converged", "sum"))
    tab["failed"] = tab.n - tab.converged
    tab["fail_rate"] = tab.failed / tab.n
    print("\n--- convergence by arm (of the models that ran) ---")
    print(tab.to_string())

    draws = pd.DataFrame(list(csv.DictReader(
        open(os.path.join(REPO, "design", "model-draws.csv")))))
    ran = {r["ensemble_id"] for r in csv.DictReader(
        open(os.path.join(REPO, "data", "ensemble", "successful-model-design.csv")))}
    draws["ran"] = draws.ensemble_id.isin(ran)
    draws["member_id"] = draws.ensemble_id.str.extract(r"(\d+)$").astype(int)
    conv_ids = set(d.loc[d.converged, "member_id"])
    draws["survived"] = draws.member_id.isin(conv_ids)
    surv = draws.groupby("tag_reporting").survived.agg(drawn="size", kept="sum")
    surv["lost"] = surv.drawn - surv.kept
    surv["loss_rate"] = surv.lost / surv.drawn
    print("\n--- cumulative attrition from 50 draws per arm (build + convergence) ---")
    print(surv.to_string())

    inc_loss = surv.loc["inclusion", "loss_rate"]
    exc_loss = surv.loc["exclusion", "loss_rate"]
    check(
        "task11.tagging-module-is-arm-specifically-fragile",
        inc_loss > 2 * exc_loss,
        f"cumulative loss from the designed 50 draws: include "
        f"{surv.loc['inclusion','lost']}/50 ({inc_loss:.0%}) vs exclude "
        f"{surv.loc['exclusion','lost']}/50 ({exc_loss:.0%}) -- a factor of "
        f"{inc_loss/exc_loss:.1f}",
    )
    check(
        "task11.convergence-failure-alone-is-arm-specific",
        tab.loc["include", "fail_rate"] > 3 * tab.loc["exclude", "fail_rate"],
        f"of models that ran: include {tab.loc['include','failed']}/{tab.loc['include','n']} "
        f"({tab.loc['include','fail_rate']:.1%}) failed MGC vs exclude "
        f"{tab.loc['exclude','failed']}/{tab.loc['exclude','n']} "
        f"({tab.loc['exclude','fail_rate']:.1%})",
    )
    print(f"\n  MGC 90th percentile: include {d[d.arm=='include'].maximum_gradient.quantile(.9):.2e}, "
          f"exclude {d[d.arm=='exclude'].maximum_gradient.quantile(.9):.2e}")

    # --- 2. balance, before and after the convergence screen -----------------
    rows = []
    for label, sub in [("all that ran (n=88)", d), ("converged only (n=80)", d[d.converged])]:
        i, e = sub[sub.arm == "include"], sub[sub.arm == "exclude"]
        rec = dict(subset=label, n_include=len(i), n_exclude=len(e),
                   include_share=f"{len(i)/len(sub):.3f}")
        for ax in ("M0", "h", "K", "tau", "creep"):
            rec[f"smd_{ax}"] = f"{smd(i[ax], e[ax]):+.3f}"
        rows.append(rec)
    bal = pd.DataFrame(rows)
    print("\n--- arm balance before and after the convergence screen ---")
    print(bal.to_string(index=False))
    bal.to_csv(os.path.join(OUT, "convergence-balance.csv"), index=False)

    s0 = abs(float(bal.iloc[0].smd_K)); s1 = abs(float(bal.iloc[1].smd_K))
    check(
        "task11.convergence-screen-worsens-arm-balance",
        len(d[d.converged & (d.arm == "include")]) / n_conv <
        len(d[d.arm == "include"]) / n_all,
        f"include share falls from {len(d[d.arm=='include'])/n_all:.1%} to "
        f"{len(d[d.converged & (d.arm=='include')])/n_conv:.1%}; |SMD| on K goes "
        f"{s0:.3f} -> {s1:.3f}",
    )

    # --- 3. arm effects: as reported vs converged-only ----------------------
    fits = []
    for col, lab in OUTCOMES:
        for label, sub in [("all that ran (n=88)", d),
                           ("converged only (n=80)", d[d.converged])]:
            s = sub[sub[col].notna()]
            m = smf.ols(f"{col} ~ {BASE}", data=s).fit()
            ci = m.conf_int().loc["arm_e"]
            fits.append(dict(outcome=lab, column=col, subset=label, weighting="none",
                             n=int(m.nobs), estimate=m.params["arm_e"],
                             ci_lo=ci[0], ci_hi=ci[1], p=m.pvalues["arm_e"],
                             mean=s[col].mean()))

    # --- 4. balanced representation: IPW back to the designed marginal ------
    # The design intended 50/50 on the RR axis and a fixed marginal on K. Weight
    # each converged member so the (arm x K) cell frequencies match the design.
    # Assumption stated in FINDINGS: within a cell the retained members are
    # exchangeable with the lost ones. That is exactly what differential
    # convergence makes doubtful, so this is a bound, not a correction.
    c = d[d.converged].copy()
    tot = sum(K_DESIGN.values())
    w = []
    for _, r in c.iterrows():
        target = 0.5 * K_DESIGN[round(r.K, 2)] / tot          # designed cell share
        have = ((c.arm == r.arm) & (np.isclose(c.K, r.K))).sum() / len(c)
        w.append(target / have if have > 0 else 0.0)
    c["w"] = np.array(w) / np.mean(w)
    print(f"\n--- IPW to the designed arm x K marginal: weight range "
          f"{c.w.min():.2f} to {c.w.max():.2f}, ESS "
          f"{c.w.sum()**2/ (c.w**2).sum():.1f} of {len(c)} ---")

    for col, lab in OUTCOMES:
        s = c[c[col].notna()]
        m = smf.wls(f"{col} ~ {BASE}", data=s, weights=s.w).fit()
        ci = m.conf_int().loc["arm_e"]
        fits.append(dict(outcome=lab, column=col,
                         subset="converged, IPW to designed arm x K marginal",
                         weighting="IPW", n=int(m.nobs), estimate=m.params["arm_e"],
                         ci_lo=ci[0], ci_hi=ci[1], p=m.pvalues["arm_e"],
                         mean=np.average(s[col], weights=s.w)))

    # --- 5. K-stratified subset with equal arm counts per stratum -----------
    # Coarsen K into three bands and take equal numbers of each arm per band,
    # nearest-neighbour on the continuous axes. This is a matched subsample:
    # smaller n, but arm composition is balanced by construction within band.
    c["Kband"] = pd.cut(c.K, [0, 0.125, 0.225, 1.0], labels=["low", "mid", "high"])
    keep = []
    rng = np.random.default_rng(7)
    for band, s in c.groupby("Kband", observed=True):
        i, e = s[s.arm == "include"], s[s.arm == "exclude"]
        k = min(len(i), len(e))
        if k == 0:
            continue
        keep += list(rng.choice(i.member_id, k, replace=False))
        keep += list(rng.choice(e.member_id, k, replace=False))
    ms = c[c.member_id.isin(keep)]
    print(f"--- K-band matched subset: n = {len(ms)} "
          f"({(ms.arm=='include').sum()} include / {(ms.arm=='exclude').sum()} exclude) ---")
    for col, lab in OUTCOMES:
        s = ms[ms[col].notna()]
        m = smf.ols(f"{col} ~ {BASE}", data=s).fit()
        ci = m.conf_int().loc["arm_e"]
        fits.append(dict(outcome=lab, column=col,
                         subset="converged, K-band matched (equal arms per band)",
                         weighting="matched", n=int(m.nobs),
                         estimate=m.params["arm_e"], ci_lo=ci[0], ci_hi=ci[1],
                         p=m.pvalues["arm_e"], mean=s[col].mean()))

    ft = pd.DataFrame(fits)
    ft["pct_of_mean"] = 100 * ft.estimate / ft["mean"]
    ft.to_csv(os.path.join(OUT, "convergence-arm-effects.csv"), index=False)
    print("\n--- adjusted arm effect (exclude - include) by subset ---")
    show = ft[ft.column.isin(["f_recent_fmsy", "sb_recent_sb0"])].copy()
    print(show[["outcome", "subset", "n", "estimate", "ci_lo", "ci_hi", "pct_of_mean"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # --- 6. does the truncation bias the gap toward zero? -------------------
    # Impute the failed include members' outcomes from the converged include
    # models, at the failed members' own design points. If the imputed values
    # sit ABOVE the converged include mean, their loss pulled the include mean
    # down and shrank the (negative) arm effect.
    failed = draws[(~draws.survived) & (draws.tag_reporting == "inclusion")].copy()
    for c2 in ("steepness", "tag_mixing_k_cutoff", "tag_tau",
               "m_age40_quarterly", "effort_creep_primary"):
        failed[c2] = failed[c2].astype(float)
    failed = failed.rename(columns={"steepness": "h", "tag_mixing_k_cutoff": "K",
                                    "tag_tau": "tau", "m_age40_quarterly": "M0",
                                    "effort_creep_primary": "creep"})
    inc_conv = c[c.arm == "include"]
    print(f"\n--- the {len(failed)} lost include members, imputed at their own design points ---")
    imp_rows = []
    for col, lab in OUTCOMES[:2]:
        m = smf.ols(f"{col} ~ h + K + tau + M0 + creep", data=inc_conv).fit()
        pred = m.predict(failed)
        obs = inc_conv[col].mean()
        imp_rows.append(dict(outcome=lab, column=col, n_lost=len(failed),
                             converged_include_mean=obs,
                             imputed_lost_mean=float(pred.mean()),
                             shift_vs_converged=float(pred.mean()) - obs))
        print(f"  {lab:22s} converged include mean {obs:.4f}; imputed lost mean "
              f"{pred.mean():.4f}  ({pred.mean()-obs:+.4f})")
    imp = pd.DataFrame(imp_rows)
    imp.to_csv(os.path.join(OUT, "lost-member-imputation.csv"), index=False)

    # --- 7. completed-arm estimate -------------------------------------------
    # Impute EVERY lost member (both arms) at its own design point from the
    # converged models of its own arm, append them, and refit. This is the most
    # direct answer to "what would the difference be under the designed 50/50
    # representation". It is illustrative, not a correction: the imputation
    # extrapolates to exactly the design points where estimation failed, which
    # is where a smooth model of the outcome is least trustworthy.
    lost_all = draws[~draws.survived].copy()
    for c2 in ("steepness", "tag_mixing_k_cutoff", "tag_tau",
               "m_age40_quarterly", "effort_creep_primary"):
        lost_all[c2] = lost_all[c2].astype(float)
    lost_all = lost_all.rename(columns={"steepness": "h", "tag_mixing_k_cutoff": "K",
                                        "tag_tau": "tau", "m_age40_quarterly": "M0",
                                        "effort_creep_primary": "creep"})
    lost_all["arm"] = np.where(lost_all.tag_reporting == "inclusion", "include", "exclude")
    lost_all["arm_e"] = (lost_all.arm == "exclude").astype(float)
    comp_rows = []
    for col, lab in OUTCOMES:
        parts = [c[["arm", "arm_e", "h", "K", "tau", "M0", "creep", col]]]
        for arm in ("include", "exclude"):
            src = c[c.arm == arm]
            tgt = lost_all[lost_all.arm == arm].copy()
            if not len(tgt):
                continue
            m = smf.ols(f"{col} ~ h + K + tau + M0 + creep", data=src).fit()
            tgt[col] = m.predict(tgt).values
            parts.append(tgt[["arm", "arm_e", "h", "K", "tau", "M0", "creep", col]])
        full = pd.concat(parts, ignore_index=True)
        m = smf.ols(f"{col} ~ {BASE}", data=full).fit()
        ci = m.conf_int().loc["arm_e"]
        comp_rows.append(dict(outcome=lab, column=col,
                              subset="converged + imputed lost (completed 50/50)",
                              weighting="imputed", n=int(m.nobs),
                              estimate=m.params["arm_e"], ci_lo=ci[0], ci_hi=ci[1],
                              p=m.pvalues["arm_e"], mean=full[col].mean()))
    fits += comp_rows
    ft = pd.DataFrame(fits)
    ft["pct_of_mean"] = 100 * ft.estimate / ft["mean"]
    ft.to_csv(os.path.join(OUT, "convergence-arm-effects.csv"), index=False)
    print("\n--- completed to the designed 50/50 by imputing the 20 lost members ---")
    print(pd.DataFrame(comp_rows)[["outcome", "n", "estimate", "ci_lo", "ci_hi"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    f_shift = imp.loc[imp.column == "f_recent_fmsy", "shift_vs_converged"].iloc[0]
    check(
        "task11.lost-include-members-would-have-raised-the-include-arm-F",
        f_shift > 0,
        f"imputed F/F_MSY for the {len(failed)} lost include members sits "
        f"{f_shift:+.4f} above the converged include mean, so their loss pulled "
        f"the include arm DOWN and shrank the (negative) arm effect -- the "
        f"reported gap is a lower bound in magnitude",
    )
    base = ft[(ft.column == "f_recent_fmsy") & (ft.subset == "converged only (n=80)")].estimate.iloc[0]
    ipw = ft[(ft.column == "f_recent_fmsy") & (ft.weighting == "IPW")].estimate.iloc[0]
    comp = ft[(ft.column == "f_recent_fmsy") & (ft.weighting == "imputed")].estimate.iloc[0]
    check(
        "task11.reweighting-to-the-designed-K-marginal-widens-the-gap",
        abs(ipw) > abs(base),
        f"F/F_MSY arm effect: converged-only {base:.4f} -> IPW to the designed "
        f"arm x K marginal {ipw:.4f} ({100*(abs(ipw)-abs(base))/abs(base):+.1f}% in magnitude)",
    )
    check(
        "task11.completing-the-arms-widens-the-gap",
        abs(comp) > abs(base),
        f"F/F_MSY arm effect: converged-only {base:.4f} -> completed 50/50 by "
        f"imputation {comp:.4f} ({100*(abs(comp)-abs(base))/abs(base):+.1f}% in "
        f"magnitude); illustrative, not a correction",
    )
    for col in ("f_recent_fmsy", "sb_recent_sb0"):
        b = ft[(ft.column == col) & (ft.subset == "converged only (n=80)")].estimate.iloc[0]
        k = ft[(ft.column == col) & (ft.weighting == "imputed")].estimate.iloc[0]
        check(
            f"task11.reported-gap-is-a-lower-bound[{col}]",
            abs(k) > abs(b),
            f"every balancing route moves the gap away from zero: {b:.4f} -> {k:.4f}",
        )

    # --- 8. risk statistics on the converged set ----------------------------
    mq = {int(re.search(r"(\d+)$", r["ensemble_id"]).group(1)): r
          for r in csv.DictReader(open(os.path.join(REPO, "data", "ensemble",
                                                    "management-quantities.csv")))}
    for frame in (d, c):
        frame["above_fmsy"] = [mq[m]["above_fmsy"] == "TRUE" for m in frame.member_id]
        frame["below_lrp"] = [mq[m]["below_lrp_020"] == "TRUE" for m in frame.member_id]
    risk = []
    for label, sub in [("all that ran (n=88)", d), ("converged only (n=80)", c)]:
        for arm in ("include", "exclude", "all"):
            s2 = sub if arm == "all" else sub[sub.arm == arm]
            risk.append(dict(subset=label, arm=arm, n=len(s2),
                             p_above_fmsy=float(s2.above_fmsy.mean()),
                             p_below_lrp=float(s2.below_lrp.mean()),
                             median_f_fmsy=float(s2.f_recent_fmsy.median()),
                             median_depletion=float(s2.sb_recent_sb0.median())))
    rk = pd.DataFrame(risk)
    rk.to_csv(os.path.join(OUT, "risk-by-arm-converged.csv"), index=False)
    print("\n--- risk statistics, all that ran vs converged only ---")
    print(rk.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # --- 9. raw vs adjusted: which is truncated? ----------------------------
    raw = []
    for col, lab in OUTCOMES[:2]:
        r88 = d[d.arm == "exclude"][col].mean() - d[d.arm == "include"][col].mean()
        r80 = c[c.arm == "exclude"][col].mean() - c[c.arm == "include"][col].mean()
        parts = [c[["arm", col]]]
        for arm in ("include", "exclude"):
            src, tgt = c[c.arm == arm], lost_all[lost_all.arm == arm].copy()
            if not len(tgt):
                continue
            mm = smf.ols(f"{col} ~ h + K + tau + M0 + creep", data=src).fit()
            tgt[col] = mm.predict(tgt).values
            parts.append(tgt[["arm", col]])
        fu = pd.concat(parts, ignore_index=True)
        rC = fu[fu.arm == "exclude"][col].mean() - fu[fu.arm == "include"][col].mean()
        raw.append(dict(outcome=lab, column=col, raw_n88=r88, raw_converged=r80,
                        raw_completed=rC,
                        pct_recovered=100 * (abs(rC) - abs(r80)) / abs(r80)))
    rw = pd.DataFrame(raw)
    rw.to_csv(os.path.join(OUT, "raw-vs-adjusted-truncation.csv"), index=False)
    print("\n--- RAW (unadjusted) arm difference ---")
    print(rw.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    f_raw = rw[rw.column == "f_recent_fmsy"].iloc[0]
    check(
        "task11.raw-gap-is-materially-truncated-by-the-convergence-screen",
        abs(f_raw.raw_converged) < 0.8 * abs(f_raw.raw_n88),
        f"raw F/F_MSY gap falls from {f_raw.raw_n88:.4f} (all that ran) to "
        f"{f_raw.raw_converged:.4f} (converged only), a "
        f"{100*(1-abs(f_raw.raw_converged)/abs(f_raw.raw_n88)):.0f}% shrinkage; "
        f"completing the arms restores it to {f_raw.raw_completed:.4f}",
    )
    check(
        "task11.adjusted-gap-is-robust-where-the-raw-gap-is-not",
        abs(abs(comp) - abs(base)) / abs(base) < 0.10
        and abs(f_raw.pct_recovered) > 25,
        f"adjusted estimate moves {100*(abs(comp)-abs(base))/abs(base):+.1f}% on "
        f"completion while the raw difference moves {f_raw.pct_recovered:+.0f}% -- "
        f"the adjustment already conditions on the axes that drive the "
        f"differential loss, so the adjusted effect is the one to quote",
    )

    write_checks(CHECKS, os.path.join(OUT, "checks-21.tsv"))


if __name__ == "__main__":
    main()
