"""Task 6: interaction between the arm and the mixing window (K axis).

The mechanism in the brief predicts the arm gap widens where more recaptures
fall inside the mixing window, i.e. at SMALL K (long mixing windows).
"""

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, PENALTY_COMPONENT, load_members, write_csv  # noqa: E402
from tagfile import read_tag_file  # noqa: E402

CHECKS = []


def check(name, passed, detail=""):
    CHECKS.append((name, bool(passed), detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return passed


def main():
    members = load_members()
    tag = read_tag_file()
    releases = np.array([r["n_released"] for r in tag["releases"]], dtype=float)

    rows = []
    for m in members:
        p, d, o = m["par"], m["design"], m["obj"]
        mix = p.tag_flags[:, 0].astype(float)
        if mix.size != releases.size:
            raise SystemExit(
                f"member {p.member_id}: {mix.size} release groups in par vs "
                f"{releases.size} in bet.tag"
            )
        rows.append(
            dict(
                member_id=p.member_id,
                arm=p.arm,
                is_exclude=float(p.arm == "exclude"),
                K=float(d["tag_mixing_k_cutoff"]),
                M0=float(d["m_age40_quarterly"]),
                h=float(d["steepness"]),
                tau=float(d["tag_tau"]),
                creep=float(d["effort_creep_primary"]),
                penalty=o[PENALTY_COMPONENT],
                mean_mixing=float(mix.mean()),
                rel_wt_mean_mixing=float((mix * releases).sum() / releases.sum()),
                n_zero_mixing=int((mix == 0).sum()),
                max_mixing=int(mix.max()),
                **{f"n_mix_{k}": int((mix == k).sum()) for k in range(0, 5)},
            )
        )
    df = pd.DataFrame(rows)
    df["log_penalty"] = np.log(df.penalty)

    # --- 1. does the realised mixing configuration track the K axis label? -----
    byK = df.groupby("K")[["rel_wt_mean_mixing", "mean_mixing", "n_zero_mixing"]].median()
    print(byK.to_string())
    rho = df["K"].corr(df["rel_wt_mean_mixing"], method="spearman")
    mono = byK["rel_wt_mean_mixing"].is_monotonic_decreasing
    check(
        "task6.mixing-window-decreases-monotonically-with-K",
        mono,
        f"median release-weighted mean mixing period by K is monotone decreasing; "
        f"Spearman(K, mean mixing) = {rho:+.4f}",
    )
    check(
        "task6.zero-mixing-count-increases-with-K",
        byK["n_zero_mixing"].is_monotonic_increasing,
        f"Spearman(K, n zero-mixing groups) = "
        f"{df['K'].corr(df['n_zero_mixing'], method='spearman'):+.4f}",
    )

    # --- 2. interaction fits ---------------------------------------------------
    fits = []
    specs = [
        ("arm * K", "penalty ~ is_exclude * K + h + tau + creep + M0", "penalty", "is_exclude:K"),
        ("arm * K", "log_penalty ~ is_exclude * K + h + tau + creep + M0", "log(penalty)", "is_exclude:K"),
        (
            "arm * mean mixing period",
            "penalty ~ is_exclude * rel_wt_mean_mixing + h + tau + creep + M0",
            "penalty",
            "is_exclude:rel_wt_mean_mixing",
        ),
        (
            "arm * mean mixing period",
            "log_penalty ~ is_exclude * rel_wt_mean_mixing + h + tau + creep + M0",
            "log(penalty)",
            "is_exclude:rel_wt_mean_mixing",
        ),
    ]
    models = {}
    for label, formula, resp, term in specs:
        mod = smf.ols(formula, data=df).fit()
        models[(label, resp)] = (mod, term)
        ci = mod.conf_int().loc[term]
        fits.append(
            dict(
                model=label,
                response=resp,
                formula=formula,
                term=term,
                n=int(mod.nobs),
                estimate=f"{mod.params[term]:.6g}",
                se=f"{mod.bse[term]:.6g}",
                ci_lo=f"{ci[0]:.6g}",
                ci_hi=f"{ci[1]:.6g}",
                p_value=f"{mod.pvalues[term]:.4g}",
                arm_main=f"{mod.params['is_exclude']:.6g}",
                r_squared=f"{mod.rsquared:.4g}",
            )
        )

    # --- 3. arm gap by K -------------------------------------------------------
    for K, sub in df.groupby("K"):
        inc, exc = sub[sub.arm == "include"], sub[sub.arm == "exclude"]
        fits.append(
            dict(
                model="arm gap by K (medians)",
                response="penalty",
                formula="median(exclude|K) - median(include|K)",
                term=f"K={K:g}",
                n=len(sub),
                estimate=f"{exc.penalty.median() - inc.penalty.median():.6g}",
                se="",
                ci_lo="",
                ci_hi="",
                p_value="",
                arm_main=f"n_inc={len(inc)},n_exc={len(exc)}",
                r_squared=f"med_mix={sub.rel_wt_mean_mixing.median():.4g}",
            )
        )
    write_csv(os.path.join(OUT, "arm-by-K-interaction.csv"), list(fits[0].keys()), fits)
    print("\n" + pd.DataFrame(fits)[
        ["model", "response", "term", "n", "estimate", "ci_lo", "ci_hi", "p_value"]
    ].to_string(index=False))

    # --- prediction: gap is largest at small K, shrinking at large K ----------
    mod, term = models[("arm * K", "penalty")]
    est = float(mod.params[term])
    ci = mod.conf_int().loc[term]
    check(
        "task6.arm-gap-shrinks-with-K",
        est < 0,
        f"arm x K interaction on the raw penalty = {est:.4g} "
        f"[{ci[0]:.4g}, {ci[1]:.4g}], p={mod.pvalues[term]:.3g}",
    )
    check(
        "task6.arm-by-K-interaction-distinguishable-from-zero",
        ci[0] * ci[1] > 0,
        f"95% CI [{ci[0]:.4g}, {ci[1]:.4g}] excludes zero" if ci[0] * ci[1] > 0
        else f"95% CI [{ci[0]:.4g}, {ci[1]:.4g}] straddles zero",
    )
    mod2, term2 = models[("arm * mean mixing period", "penalty")]
    ci2 = mod2.conf_int().loc[term2]
    check(
        "task6.arm-gap-grows-with-mean-mixing-period",
        float(mod2.params[term2]) > 0,
        f"arm x mean-mixing interaction = {mod2.params[term2]:.4g} "
        f"[{ci2[0]:.4g}, {ci2[1]:.4g}], p={mod2.pvalues[term2]:.3g}",
    )

    df.to_csv(os.path.join(OUT, "mixing-frame.csv"), index=False)
    with open(os.path.join(OUT, "checks-04.txt"), "w") as fh:
        for n, pa, det in CHECKS:
            fh.write(f"{'PASS' if pa else 'FAIL'}\t{n}\t{det}\n")


if __name__ == "__main__":
    main()
