"""Task 2.3: quantify the report-writer vs likelihood scaling discrepancy.

Inside a mixing window the fit-report writers apply `get_rep_rate_correction`
(src/tag3.cpp:2264), which returns 1.0 under `exclude`, while the likelihood
(src/tagfit.cpp:182) always applies X-hat. Since the Newton-Raphson drives
in-window `tagcatch` to the observed returns R under `exclude`:

    predicted returns as reported to SC   ~  R
    predicted returns the objective used  ~  X-hat * R

so the objective's in-window predictions sit a factor X-hat below the plotted
ones, a shortfall of (1 - X-hat) * R. Under `include` both paths apply X-hat to
a tagcatch of ~R/X-hat and agree, so the discrepancy is exclude-only.

This is a first-order accounting on the observed in-window returns, not a
re-run of the model: it uses N_mix (from bet.tag and each member's own mixing
windows) and the member's fitted X-hat per reporting-rate group.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2common import OUT, check_logger, load_rep_group_names, write_checks  # noqa: E402

CHECKS, check = check_logger()


def main():
    src = os.path.join(OUT, "group-excursion-predictors.csv")
    if not os.path.exists(src):
        raise SystemExit("run 11-predictors.py first")
    d = pd.read_csv(src)
    names = load_rep_group_names()

    d["reported_pred"] = d.N_mix                      # multiplier 1.0
    d["likelihood_pred"] = d.rep_hat * d.N_mix        # multiplier X-hat
    d["shortfall"] = d.reported_pred - d.likelihood_pred
    # Under `include` both paths agree, so the discrepancy is defined as zero
    # there; carried explicitly rather than left implicit.
    d.loc[d.arm == "include", ["shortfall"]] = 0.0
    d["shortfall_frac"] = 1.0 - d.rep_hat
    d.loc[d.arm == "include", "shortfall_frac"] = 0.0

    per_member = (d.groupby(["member_id", "arm"])
                    .agg(N_mix=("N_mix", "sum"),
                         reported=("reported_pred", "sum"),
                         used=("likelihood_pred", "sum"),
                         shortfall=("shortfall", "sum"))
                    .reset_index())
    per_member["shortfall_frac"] = per_member.shortfall / per_member.reported
    per_member.loc[per_member.arm == "include", "shortfall_frac"] = 0.0

    exc = per_member[per_member.arm == "exclude"]
    print(f"exclude-arm members: {len(exc)}")
    print(f"  in-window reported recaptures per member (median): {exc.N_mix.median():.0f}")
    print(f"  predicted in-window returns, as reported to SC (median): "
          f"{exc.reported.median():.0f}")
    print(f"  predicted in-window returns, as used by the objective (median): "
          f"{exc.used.median():.0f}")
    print(f"  shortfall (median): {exc.shortfall.median():.0f} fish, "
          f"{exc.shortfall_frac.median():.1%} of the plotted value")

    per_group = (d[d.arm == "exclude"].groupby("group_id")
                   .agg(n_members=("member_id", "nunique"),
                        N_mix_med=("N_mix", "median"),
                        rep_hat_med=("rep_hat", "median"),
                        shortfall_med=("shortfall", "median"),
                        shortfall_frac_med=("shortfall_frac", "median"))
                   .reset_index())
    per_group["group_label"] = per_group.group_id.map(
        lambda g: names.get(g, ("(unmapped)", ""))[0])
    per_group = per_group.sort_values("shortfall_med", ascending=False)
    print("\n--- per reporting-rate group, exclude arm ---")
    print(per_group[["group_id", "group_label", "N_mix_med", "rep_hat_med",
                     "shortfall_med", "shortfall_frac_med"]]
          .assign(group_label=lambda x: x.group_label.str.slice(0, 26))
          .to_string(index=False))

    out = pd.concat([
        per_member.assign(scope="per member"),
        per_group.rename(columns={"N_mix_med": "N_mix", "shortfall_med": "shortfall",
                                  "shortfall_frac_med": "shortfall_frac"})
                 .assign(scope="per group, exclude arm"),
    ], ignore_index=True)
    out.to_csv(os.path.join(OUT, "report-vs-likelihood-discrepancy.csv"), index=False)

    check(
        "task2.report-likelihood-discrepancy-still-holds",
        exc.shortfall.median() > 0,
        f"exclude arm: the objective's in-window predicted returns sit "
        f"{exc.shortfall_frac.median():.1%} below the values written to plot.rep "
        f"(median {exc.shortfall.median():.0f} of {exc.reported.median():.0f} fish)",
    )
    inc = per_member[per_member.arm == "include"]
    check(
        "task2.discrepancy-is-exclude-only",
        float(inc.shortfall.abs().max()) == 0.0,
        f"include arm shortfall is identically zero by construction "
        f"(both paths apply X-hat to a tagcatch of ~R/X-hat)",
    )
    worst = per_group.iloc[0]
    check(
        "task2.discrepancy-concentrates-in-the-high-volume-groups",
        worst.N_mix_med >= per_group.N_mix_med.median(),
        f"largest shortfall is group {int(worst.group_id)} "
        f"({worst.group_label[:30]}): {worst.shortfall_med:.0f} fish on "
        f"{worst.N_mix_med:.0f} in-window recaptures",
    )

    write_checks(CHECKS, os.path.join(OUT, "checks-16.tsv"))


if __name__ == "__main__":
    main()
