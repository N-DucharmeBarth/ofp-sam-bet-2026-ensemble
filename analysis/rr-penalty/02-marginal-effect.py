"""Task 4: marginal effect of the RR flag axis, controlling for M0 and the rest.

The confound check. The `RR (tag reporting)` axis is not the only thing that
moves this penalty, and the 88 retained models are a *selected* subset of the
100 drawn, so the raw median gap read off the violins can be part M0.

Scale choice, stated once here and carried into FINDINGS.md:
  * steepness (h) and M0 are continuous draws -> continuous;
  * tau, K and effort creep are ordered numeric levels -> continuous on their
    numeric scale for the headline fit, with an all-factor fit reported as a
    sensitivity;
  * the response is fitted on both the raw penalty scale (matching the violin
    units) and log scale (the penalty is strongly right-skewed).
"""

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    OUT,
    PENALTY_COMPONENT,
    REPO,
    load_design,
    load_objective,
    write_csv,
)
from parpar import parse_all  # noqa: E402

CHECKS = []


def check(name, passed, detail=""):
    CHECKS.append((name, bool(passed), detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return passed


def frame():
    design, obj = load_design(), load_objective()
    have_par = {p.member_id for p in parse_all(os.path.join(REPO, "final-par"))}
    rows = []
    for mid, d in sorted(design.items()):
        o = obj.get(mid)
        if o is None:
            continue
        rows.append(
            dict(
                member_id=mid,
                arm="include" if d["tag_reporting"] == "inclusion" else "exclude",
                penalty=o[PENALTY_COMPONENT],
                h=float(d["steepness"]),
                tau=float(d["tag_tau"]),
                K=float(d["tag_mixing_k_cutoff"]),
                M0=float(d["m_age40_quarterly"]),
                creep=float(d["effort_creep_primary"]),
                has_par=mid in have_par,
            )
        )
    df = pd.DataFrame(rows)
    df["log_penalty"] = np.log(df["penalty"])
    # `exclude` coded 1 so a positive coefficient means exclude has the larger
    # penalty -- the direction the mechanism in the brief predicts.
    df["is_exclude"] = (df["arm"] == "exclude").astype(float)
    return df


def balance(df):
    out = []
    for axis, kind in [("M0", "cont"), ("K", "lvl"), ("tau", "lvl"), ("creep", "lvl"), ("h", "cont")]:
        if kind == "lvl":
            tab = pd.crosstab(df[axis], df["arm"])
            for lvl, r in tab.iterrows():
                out.append(
                    dict(
                        axis=axis,
                        level=f"{lvl:g}",
                        n_include=int(r.get("include", 0)),
                        n_exclude=int(r.get("exclude", 0)),
                        mean_include="",
                        mean_exclude="",
                        std_diff="",
                    )
                )
        else:
            g = df.groupby("arm")[axis]
            mi, me = g.mean().get("include", np.nan), g.mean().get("exclude", np.nan)
            sp = np.sqrt((g.var().get("include", np.nan) + g.var().get("exclude", np.nan)) / 2)
            out.append(
                dict(
                    axis=axis,
                    level="(continuous)",
                    n_include=int((df["arm"] == "include").sum()),
                    n_exclude=int((df["arm"] == "exclude").sum()),
                    mean_include=f"{mi:.6g}",
                    mean_exclude=f"{me:.6g}",
                    std_diff=f"{(me - mi) / sp:.4g}",
                )
            )
    return out


def fit_row(df, formula, label, response):
    m = smf.ols(formula, data=df).fit()
    term = "is_exclude"
    ci = m.conf_int().loc[term]
    return dict(
        model=label,
        response=response,
        formula=formula,
        n=int(m.nobs),
        arm_coef=f"{m.params[term]:.6g}",
        arm_se=f"{m.bse[term]:.6g}",
        ci_lo=f"{ci[0]:.6g}",
        ci_hi=f"{ci[1]:.6g}",
        p_value=f"{m.pvalues[term]:.4g}",
        r_squared=f"{m.rsquared:.4g}",
    ), m


def main():
    df = frame()
    print(f"design x objective rows: {len(df)}  (with retained par: {int(df.has_par.sum())})")

    # --- balance table ---------------------------------------------------------
    bal = balance(df)
    write_csv(os.path.join(OUT, "rr-axis-balance.csv"), list(bal[0].keys()), bal)
    print(pd.DataFrame(bal).to_string(index=False))

    # Standardised mean difference on M0: the axis the brief flags as the
    # candidate confound. |SMD| > 0.25 is the usual "materially imbalanced" line.
    smd_M0 = float([b for b in bal if b["axis"] == "M0"][0]["std_diff"])
    smd_h = float([b for b in bal if b["axis"] == "h"][0]["std_diff"])
    check(
        "task4.M0-balanced-across-arms",
        abs(smd_M0) <= 0.25,
        f"standardised mean difference (exclude - include) = {smd_M0:+.3f}",
    )
    check(
        "task4.h-balanced-across-arms",
        abs(smd_h) <= 0.25,
        f"standardised mean difference = {smd_h:+.3f}",
    )
    for axis in ("K", "tau", "creep"):
        tab = pd.crosstab(df[axis], df["arm"])
        # chi-square-free screen: worst-cell share difference
        pi = tab.get("include", pd.Series(0, index=tab.index)) / max((df.arm == "include").sum(), 1)
        pe = tab.get("exclude", pd.Series(0, index=tab.index)) / max((df.arm == "exclude").sum(), 1)
        worst = float(np.max(np.abs(pi - pe)))
        check(
            f"task4.{axis}-balanced-across-arms",
            worst <= 0.15,
            f"largest level-share difference = {worst:.3f}",
        )

    # --- raw median difference -------------------------------------------------
    med = df.groupby("arm")["penalty"].median()
    raw_gap = float(med["exclude"] - med["include"])
    print(f"\nraw medians: include={med['include']:.4g}  exclude={med['exclude']:.4g}  "
          f"gap={raw_gap:+.4g}")

    # --- fits ------------------------------------------------------------------
    fits = []
    specs = [
        ("arm only", "penalty ~ is_exclude", "penalty"),
        ("arm + all axes except M0", "penalty ~ is_exclude + h + tau + K + creep", "penalty"),
        ("arm + all axes incl M0", "penalty ~ is_exclude + h + tau + K + creep + M0", "penalty"),
        ("arm + M0 only", "penalty ~ is_exclude + M0", "penalty"),
        ("arm only", "log_penalty ~ is_exclude", "log(penalty)"),
        ("arm + all axes except M0", "log_penalty ~ is_exclude + h + tau + K + creep", "log(penalty)"),
        ("arm + all axes incl M0", "log_penalty ~ is_exclude + h + tau + K + creep + M0", "log(penalty)"),
        (
            "arm + all axes incl M0, discrete axes as factors",
            "penalty ~ is_exclude + h + C(tau) + C(K) + C(creep) + M0",
            "penalty",
        ),
        (
            "arm + all axes incl M0, discrete axes as factors",
            "log_penalty ~ is_exclude + h + C(tau) + C(K) + C(creep) + M0",
            "log(penalty)",
        ),
    ]
    models = {}
    for label, formula, response in specs:
        row, m = fit_row(df, formula, label, response)
        row["subset"] = "all 88 retained models"
        fits.append(row)
        models[(label, response)] = m

    # same headline fit on the 80 members that have a retained par, so Tasks 5-8
    # are known to be estimated on a subset with the same arm effect
    sub = df[df.has_par]
    row, _ = fit_row(
        sub, "penalty ~ is_exclude + h + tau + K + creep + M0", "arm + all axes incl M0", "penalty"
    )
    row["subset"] = "80 members with retained par"
    fits.append(row)

    fits.append(
        dict(
            model="raw median difference (no adjustment)",
            response="penalty",
            formula="median(exclude) - median(include)",
            n=len(df),
            arm_coef=f"{raw_gap:.6g}",
            arm_se="",
            ci_lo="",
            ci_hi="",
            p_value="",
            r_squared="",
            subset="all 88 retained models",
        )
    )
    write_csv(os.path.join(OUT, "rr-axis-marginal-effect.csv"), list(fits[0].keys()), fits)
    print("\n" + pd.DataFrame(fits)[
        ["subset", "response", "model", "n", "arm_coef", "ci_lo", "ci_hi", "p_value", "r_squared"]
    ].to_string(index=False))

    # --- shrinkage tripwires ---------------------------------------------------
    a_no = float(models[("arm + all axes except M0", "penalty")].params["is_exclude"])
    a_yes = float(models[("arm + all axes incl M0", "penalty")].params["is_exclude"])
    check(
        "task4.arm-effect-robust-to-adding-M0",
        abs(a_yes - a_no) / max(abs(a_no), 1e-12) < 0.20,
        f"arm coef {a_no:.4g} -> {a_yes:.4g} on adding M0 "
        f"({100 * (a_yes - a_no) / a_no:+.1f}%)",
    )
    ci = models[("arm + all axes incl M0", "penalty")].conf_int().loc["is_exclude"]
    check(
        "task4.adjusted-arm-effect-positive",
        ci[0] > 0,
        f"adjusted arm effect (exclude - include) = {a_yes:.4g} "
        f"[{ci[0]:.4g}, {ci[1]:.4g}]",
    )
    check(
        "task4.adjusted-effect-not-much-smaller-than-raw-median-gap",
        abs(a_yes) > 0.5 * abs(raw_gap),
        f"adjusted {a_yes:.4g} vs raw median gap {raw_gap:.4g}",
    )
    # Is M0 itself doing the work the figure suggests?
    m_full = models[("arm + all axes incl M0", "penalty")]
    print(f"\nM0 coefficient in the full fit: {m_full.params['M0']:.4g} "
          f"(p={m_full.pvalues['M0']:.3g}); K coefficient: {m_full.params['K']:.4g} "
          f"(p={m_full.pvalues['K']:.3g})")

    with open(os.path.join(OUT, "checks-02.txt"), "w") as fh:
        for n, pa, d in CHECKS:
            fh.write(f"{'PASS' if pa else 'FAIL'}\t{n}\t{d}\n")
    df.to_csv(os.path.join(OUT, "member-frame.csv"), index=False)


if __name__ == "__main__":
    main()
