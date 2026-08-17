"""Does the reporting rate move F at frozen parameters? No -- structurally.

This was intended to decompose the observed -18% arm effect on F/F_MSY into a
reporting-rate pathway (dF/dX) and a removal channel (arm difference at fixed
X). It returns a null, and the null is exact rather than small: across all 168
evaluations the derived quantities take exactly ONE value per member, while the
objective takes 56.

Why, from the source trace: `tag_fish_rep` enters only (a) the tag likelihood,
(b) the reporting-rate penalty, and (c) the Newton-Raphson target for the
tagged cohort. The tagged cohort (`tagnum_fish`) is a separate accounting layer
fitted TO the population; it does not feed back into population numbers or
fishing mortality. With every estimated parameter frozen there is no channel by
which X-hat can move F.

Consequence, and it inverts the earlier recommendation: the fixed-parameter
profile CANNOT decompose the downstream effect. The pathway from reporting rate
to F runs entirely through ESTIMATION -- X-hat reshapes the likelihood surface,
which relocates the optimum of the F-related parameters. Only re-estimation
(paired refits) can measure it.

The null is informative in one direction: it rules out the observed arm effect
on F being an arithmetic or reporting consequence of how F is computed. It is
optimiser behaviour.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2common import OUT, check_logger, write_checks  # noqa: E402

CHECKS, check = check_logger()

DERIVED = [("agg_F_Fmsy", "F / F_MSY"), ("agg_F", "aggregate F"),
           ("SB_SBmsy", "SB / SB_MSY"), ("SB_recent", "SB recent"),
           ("MSY", "MSY"), ("F_at_MSY", "F at MSY")]


def main():
    d = pd.read_csv(os.path.join(OUT, "profile-objective.csv"))
    d = d[d.ok == 1].sort_values(["member_id", "group_id", "arm", "X"])
    have = [c for c, _ in DERIVED if c in d.columns and d[c].notna().any()]

    check(
        "task9.derived-quantities-present-in-every-evaluation",
        all(d[c].notna().all() for c in have),
        f"{have} complete in {int(d[have].notna().all(axis=1).sum())}/{len(d)} "
        f"evaluations",
    )

    rows = []
    for c, lab in DERIVED:
        if c not in have:
            continue
        per_member = d.groupby("member_id")[c].nunique()
        rows.append(dict(
            quantity=lab, column=c,
            n_unique_overall=int(d[c].nunique()),
            n_members=int(d.member_id.nunique()),
            max_unique_within_member=int(per_member.max()),
            max_rel_spread_within_member=float(
                d.groupby("member_id")[c]
                 .apply(lambda s: (s.max() - s.min()) / abs(s.mean()) if s.mean() else 0)
                 .max()),
        ))
    rows.append(dict(
        quantity="total objective (control)", column="objective",
        n_unique_overall=int(d.objective.nunique()),
        n_members=int(d.member_id.nunique()),
        max_unique_within_member=int(d.groupby("member_id").objective.nunique().max()),
        max_rel_spread_within_member=float(
            d.groupby("member_id").objective
             .apply(lambda s: (s.max() - s.min()) / abs(s.mean())).max()),
    ))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "F-invariance.csv"), index=False)
    print("\n--- variation across the 168-point grid ---")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    derived_cols = [c for c, _ in DERIVED if c in have]
    worst = max(int(d.groupby("member_id")[c].nunique().max()) for c in derived_cols)
    check(
        "task9.derived-quantities-are-invariant-to-the-reporting-rate",
        worst == 1,
        f"every derived quantity takes exactly {worst} value per member across "
        f"{len(d) // d.member_id.nunique()} evaluations spanning 14 reporting "
        f"rates and both arms",
    )
    check(
        "task9.objective-does-vary-so-the-grid-is-live",
        int(d.groupby("member_id").objective.nunique().min()) > 10,
        f"the objective takes "
        f"{int(d.groupby('member_id').objective.nunique().min())}-"
        f"{int(d.groupby('member_id').objective.nunique().max())} distinct "
        f"values per member over the same evaluations, so the null is not a "
        f"dead harness",
    )
    piv = d.pivot_table(index=["member_id", "group_id", "X"], columns="arm",
                        values="agg_F_Fmsy").dropna()
    check(
        "task9.removal-channel-does-not-move-F-at-frozen-parameters",
        float((piv["exclude"] - piv["include"]).abs().max()) == 0.0,
        f"max |F/F_MSY(exclude) - F/F_MSY(include)| at the same X-hat = "
        f"{float((piv['exclude'] - piv['include']).abs().max()):.3g}",
    )
    check(
        "task9.fixed-parameter-profile-can-decompose-the-downstream-effect",
        False,
        "it cannot: dF/dX and the removal channel are both structurally zero at "
        "frozen parameters, so the observed -18% arm effect on F/F_MSY is an "
        "estimation effect and only re-estimation can decompose it",
    )

    write_checks(CHECKS, os.path.join(OUT, "checks-19.tsv"))


if __name__ == "__main__":
    main()
