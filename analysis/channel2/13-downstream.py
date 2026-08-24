"""Task 5: does the arm effect propagate to F and stock status?

Uses the repo's curated management-quantities table (88 members) plus plot.rep
aggregate F (80 members with a retained rep). Coverage is stated per outcome
rather than inherited.

Specification is deliberately IDENTICAL to the penalty analysis
(`outcome ~ arm + M0 + K + tau + effort_creep + h`) so the arm coefficients are
directly comparable across the two analyses.

The mediation section is a decomposition of covariation, NOT causal mediation:
X-hat and F are jointly estimated in the same optimisation, so conditioning on
X-hat does not identify a causal path. Stated as such in FINDINGS.md.
"""

import csv
import os
import re
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2common import (  # noqa: E402
    OUT,
    REPO,
    check_logger,
    load_design,
    load_members,
    write_checks,
)

CHECKS, check = check_logger()

MQ = os.path.join(REPO, "data", "ensemble", "management-quantities.csv")

OUTCOMES = [
    ("f_recent_fmsy", "F_recent / F_MSY", "negative"),
    ("sb_recent_sb0", "SB_recent / SB_0 (depletion)", "positive"),
    ("sb_recent_sbmsy", "SB_recent / SB_MSY", "positive"),
    ("sb_recent_kt", "SB_recent (kt)", "positive"),
    ("agg_F", "aggregate F (plot.rep)", "negative"),
]

BASE = "arm_e + M0 + K + tau + creep + h"


def plotrep_aggregate_F(path):
    """'# Aggregate F' scalar from a plot.rep, or None if absent."""
    want = "# Aggregate F"
    with open(path) as fh:
        lines = fh.read().split("\n")
    for i, ln in enumerate(lines):
        if ln.strip() == want:
            for j in range(i + 1, min(i + 4, len(lines))):
                s = lines[j].strip()
                if s and not s.startswith("#"):
                    v = [float(t) for t in s.split()]
                    return v[-1] if v else None
    return None


def main():
    design = load_design()
    members = {m["par"].member_id: m for m in load_members()}

    with open(MQ) as fh:
        mq = {int(re.search(r"(\d+)$", r["ensemble_id"]).group(1)): r
              for r in csv.DictReader(fh)}

    rows = []
    for mid, d in sorted(design.items()):
        q = mq.get(mid)
        if q is None:
            continue
        rec = dict(
            member_id=mid,
            arm="exclude" if d["tag_reporting"] == "exclusion" else "include",
            M0=float(d["m_age40_quarterly"]), K=float(d["tag_mixing_k_cutoff"]),
            tau=float(d["tag_tau"]), creep=float(d["effort_creep_primary"]),
            h=float(d["steepness"]),
        )
        rec["arm_e"] = float(rec["arm"] == "exclude")
        for col in ("f_recent_fmsy", "sb_recent_sb0", "sb_recent_sbmsy",
                    "sb_recent_kt", "sb0_recent_kt", "recent_mean_depletion"):
            rec[col] = float(q[col]) if q.get(col) not in (None, "", "NA") else np.nan
        rep = os.path.join(REPO, "final-par", f"ensemble-{mid:03d}", "plot-11.par.rep")
        rec["agg_F"] = plotrep_aggregate_F(rep) if os.path.exists(rep) else np.nan
        rec["has_par"] = mid in members
        rows.append(rec)
    df = pd.DataFrame(rows)

    # X-hat of the priored groups, for the decomposition (PAR members only)
    from c2common import penalty_groups  # noqa: E402
    xh = []
    for mid, m in members.items():
        pg = {g["group_id"]: g["rep_hat"] for g in penalty_groups(m["par"])
              if g["penalty_wt"] > 1}
        rec = dict(member_id=mid, **{f"X{g}": v for g, v in pg.items()})
        rec["X_priored_mean"] = float(np.mean(list(pg.values())))
        xh.append(rec)
    df = df.merge(pd.DataFrame(xh), on="member_id", how="left")

    print(f"design x management-quantities rows: {len(df)}")
    for col, label, _ in OUTCOMES:
        n = int(df[col].notna().sum())
        print(f"  {label:<32} n = {n}")
    n_dep = int(df["sb_recent_sb0"].notna().sum())
    check(
        "task5.management-quantities-cover-all-88",
        n_dep == 88,
        f"depletion available for {n_dep} of {len(df)} design rows",
    )
    check(
        "task5.aggregate-F-coverage-matches-the-80-par-members",
        int(df["agg_F"].notna().sum()) == int(df.has_par.sum()),
        f"aggregate F available for {int(df['agg_F'].notna().sum())} members; "
        f"{int(df.has_par.sum())} have a retained par",
    )
    df.to_csv(os.path.join(OUT, "downstream-frame.csv"), index=False)

    # --- adjusted arm effects ------------------------------------------------
    fits = []
    for col, label, _ in OUTCOMES:
        sub = df[df[col].notna()]
        for spec, formula in [
            ("continuous axes", f"{col} ~ {BASE}"),
            ("discrete axes as factors",
             f"{col} ~ arm_e + M0 + C(K) + C(tau) + C(creep) + h"),
        ]:
            mod = smf.ols(formula, data=sub).fit()
            ci = mod.conf_int().loc["arm_e"]
            fits.append(dict(
                outcome=label, column=col, spec=spec, term="arm (exclude - include)",
                n=int(mod.nobs), estimate=f"{mod.params['arm_e']:.6g}",
                se=f"{mod.bse['arm_e']:.6g}", ci_lo=f"{ci[0]:.6g}",
                ci_hi=f"{ci[1]:.6g}", p_value=f"{mod.pvalues['arm_e']:.4g}",
                r_squared=f"{mod.rsquared:.4g}",
                mean_outcome=f"{sub[col].mean():.6g}",
            ))
        # arm x K interaction
        mod = smf.ols(f"{col} ~ arm_e * K + M0 + tau + creep + h", data=sub).fit()
        ci = mod.conf_int().loc["arm_e:K"]
        fits.append(dict(
            outcome=label, column=col, spec="arm x K interaction", term="arm:K",
            n=int(mod.nobs), estimate=f"{mod.params['arm_e:K']:.6g}",
            se=f"{mod.bse['arm_e:K']:.6g}", ci_lo=f"{ci[0]:.6g}",
            ci_hi=f"{ci[1]:.6g}", p_value=f"{mod.pvalues['arm_e:K']:.4g}",
            r_squared=f"{mod.rsquared:.4g}", mean_outcome=f"{sub[col].mean():.6g}",
        ))

    # --- decomposition: does the arm effect run through X-hat? ---------------
    sub = df[df.has_par & df.X_priored_mean.notna()]
    for col, label, _ in OUTCOMES:
        s = sub[sub[col].notna()]
        if len(s) < 30:
            continue
        m0 = smf.ols(f"{col} ~ {BASE}", data=s).fit()
        m1 = smf.ols(f"{col} ~ {BASE} + X_priored_mean", data=s).fit()
        a0, a1 = m0.params["arm_e"], m1.params["arm_e"]
        ci = m1.conf_int().loc["arm_e"]
        fits.append(dict(
            outcome=label, column=col, spec="conditioning on mean priored X-hat",
            term="arm (exclude - include)", n=int(m1.nobs),
            estimate=f"{a1:.6g}", se=f"{m1.bse['arm_e']:.6g}",
            ci_lo=f"{ci[0]:.6g}", ci_hi=f"{ci[1]:.6g}",
            p_value=f"{m1.pvalues['arm_e']:.4g}", r_squared=f"{m1.rsquared:.4g}",
            mean_outcome=f"shrinkage {100*(a1-a0)/a0:+.1f}% from {a0:.4g} to {a1:.4g}"
            if a0 else "",
        ))
        # The X-hat coefficient is what makes the reversal interpretable: if
        # X-hat carries the outcome in the direction the mechanism implies,
        # then a sign flip on the arm term is over-mediation (the indirect
        # path exceeds the total), not evidence against the pathway.
        cx = m1.conf_int().loc["X_priored_mean"]
        indirect = a0 - a1
        fits.append(dict(
            outcome=label, column=col, spec="X-hat coefficient (same fit)",
            term="X_priored_mean", n=int(m1.nobs),
            estimate=f"{m1.params['X_priored_mean']:.6g}",
            se=f"{m1.bse['X_priored_mean']:.6g}", ci_lo=f"{cx[0]:.6g}",
            ci_hi=f"{cx[1]:.6g}", p_value=f"{m1.pvalues['X_priored_mean']:.4g}",
            r_squared=f"{m1.rsquared:.4g}",
            mean_outcome=f"implied indirect (total - direct) = {indirect:.4g}, "
                         f"{100*indirect/a0:.0f}% of the total {a0:.4g}" if a0 else "",
        ))

    ft = pd.DataFrame(fits)
    ft.to_csv(os.path.join(OUT, "downstream-arm-effects.csv"), index=False)
    print("\n--- adjusted arm effects ---")
    print(ft[ft.spec == "continuous axes"][
        ["outcome", "n", "estimate", "ci_lo", "ci_hi", "p_value", "mean_outcome"]
    ].to_string(index=False))
    print("\n--- arm x K interaction ---")
    print(ft[ft.spec == "arm x K interaction"][
        ["outcome", "n", "estimate", "ci_lo", "ci_hi", "p_value"]].to_string(index=False))
    print("\n--- X-hat coefficient and implied indirect path ---")
    print(ft[ft.spec == "X-hat coefficient (same fit)"][
        ["outcome", "estimate", "ci_lo", "ci_hi", "p_value", "mean_outcome"]
    ].to_string(index=False))
    print("\n--- conditioning on mean priored X-hat ---")
    print(ft[ft.spec == "conditioning on mean priored X-hat"][
        ["outcome", "n", "estimate", "ci_lo", "ci_hi", "p_value", "mean_outcome"]
    ].to_string(index=False))

    # --- checks --------------------------------------------------------------
    def row(col, spec):
        r = ft[(ft.column == col) & (ft.spec == spec)]
        return r.iloc[0] if len(r) else None

    for col, label, direction in [("f_recent_fmsy", "F/F_MSY", "negative"),
                                  ("agg_F", "aggregate F", "negative")]:
        r = row(col, "continuous axes")
        est, lo, hi = float(r.estimate), float(r.ci_lo), float(r.ci_hi)
        check(
            f"task5.arm-effect-on-F-is-negative[{col}]",
            est < 0 and hi < 0,
            f"{label}: {est:+.4g} [{lo:.4g}, {hi:.4g}], p={float(r.p_value):.3g} "
            f"(mean {float(r.mean_outcome):.4g})",
        )
    r = row("sb_recent_sb0", "continuous axes")
    est, lo, hi = float(r.estimate), float(r.ci_lo), float(r.ci_hi)
    check(
        "task5.arm-effect-on-depletion-is-positive",
        est > 0 and lo > 0,
        f"SB/SB_0: {est:+.4g} [{lo:.4g}, {hi:.4g}], p={float(r.p_value):.3g} "
        f"(mean depletion {float(r.mean_outcome):.4g})",
    )
    r = row("f_recent_fmsy", "arm x K interaction")
    check(
        "task5.arm-effect-on-status-attenuates-with-K",
        float(r.estimate) > 0,
        f"F/F_MSY arm x K = {float(r.estimate):+.4g} "
        f"[{float(r.ci_lo):.4g}, {float(r.ci_hi):.4g}], p={float(r.p_value):.3g} "
        f"(positive = the negative arm effect on F shrinks as K rises)",
    )
    for col, label in [("f_recent_fmsy", "F/F_MSY"), ("sb_recent_sb0", "depletion")]:
        r0 = row(col, "continuous axes")
        r1 = row(col, "conditioning on mean priored X-hat")
        if r1 is None:
            continue
        a0, a1 = float(r0.estimate), float(r1.estimate)
        check(
            f"task5.arm-effect-shrinks-when-conditioning-on-Xhat[{col}]",
            abs(a1) < abs(a0),
            f"{label}: arm coefficient {a0:+.4g} -> {a1:+.4g} "
            f"({100*(abs(a1)-abs(a0))/abs(a0):+.1f}% in magnitude)",
        )

    write_checks(CHECKS, os.path.join(OUT, "checks-13.tsv"))


if __name__ == "__main__":
    main()
