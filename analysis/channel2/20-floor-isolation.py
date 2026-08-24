"""Isolate the survival-floor penalty from the fixed-parameter profile.

Resolves an open question in the US review, which could not determine from
source alone which Newton-Raphson variant the assessment binary uses:

  "the source defines more than one version of the routine that performs this
   removal, and they differ both in whether they honour tag_flags(i,2) and in
   whether the cap is applied at all ... we have not been able to confirm
   whether it is also the version carrying the cap."

At multifan-cl HEAD (624dee4f) the routine actually called from tag3.cpp:252
is `do_newton_raphson_for_tags2`, which divides by `rep_rate` unconditionally
and applies NO cap. The variant that honours `tag_flags(it,2)` AND carries the
posfun cap (tag3.cpp:2769-2884) is not called at that commit. The assessment
binary is version 2.2.7.9 and predates it, so the source HEAD cannot answer the
question -- but the binary's own behaviour can.

Two facts settle it:

  1. The binary honours the flag. 168 evaluations show arm-dependent mixing-block
     behaviour; if it did not, the arms would be identical.
  2. The binary carries the cap. Decompose the objective:

         residual = objective - (tag_mix + tag_post + rr_penalty)

     Everything else in the objective is frozen along a profile, so the residual
     is constant unless some OTHER X-dependent term exists. Under `exclude` it
     is flat. Under `include` it rises sharply at low X-hat -- exactly where the
     raised removal R/X is largest. That rising term is the posfun penalty.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2common import OUT, check_logger, write_checks  # noqa: E402

CHECKS, check = check_logger()


def main():
    d = pd.read_csv(os.path.join(OUT, "profile-objective.csv"))
    d = d[d.ok == 1].copy()
    d["accounted"] = d.tag_mix + d.tag_post + d.rr_penalty
    # residual relative to each profile's own minimum: the constant frozen part
    # cancels, leaving only X-dependent terms outside the tag blocks
    rows = []
    for (mid, gid, arm), s in d.groupby(["member_id", "group_id", "arm"]):
        s = s.sort_values("X")
        resid = (s.objective - s.accounted).values
        resid = resid - resid.min()
        hi = s.X.values >= 0.8
        rows.append(dict(
            member_id=mid, group_id=gid, arm=arm,
            resid_range=float(resid.max()),
            resid_at_lowest_X=float(resid[0]),
            resid_range_highX=float(resid[hi].max() - resid[hi].min()),
            X_at_max_resid=float(s.X.values[int(np.argmax(resid))]),
        ))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, "floor-penalty-isolation.csv"), index=False)

    print("--- unexplained objective term (objective - tag blocks - penalty) ---")
    print(r.groupby("arm")[["resid_range", "resid_at_lowest_X", "resid_range_highX"]]
           .agg(["median", "max"]).to_string())

    exc = r[r.arm == "exclude"]
    inc = r[r.arm == "include"]
    check(
        "task10.no-unexplained-objective-term-under-exclude",
        exc.resid_range.max() < 1.0,
        f"exclude: the objective is fully accounted for by the tag blocks and the "
        f"reporting-rate penalty -- max unexplained range {exc.resid_range.max():.3g} "
        f"objective units across all profiles",
    )
    check(
        "task10.large-unexplained-objective-term-under-include-at-low-X",
        inc.resid_range.max() > 100.0,
        f"include: max unexplained term {inc.resid_range.max():.4g} objective units "
        f"(median across profiles {inc.resid_range.median():.4g}); this is the posfun "
        f"survival-floor penalty, which has no other candidate source",
    )
    check(
        "task10.unexplained-term-concentrates-at-low-X",
        bool((inc.X_at_max_resid <= 0.42).all()),
        f"include: the unexplained term peaks at X = "
        f"{sorted(set(np.round(inc.X_at_max_resid, 3)))} -- i.e. where the raised "
        f"removal R/X is largest, which is where the cap must engage",
    )
    check(
        "task10.unexplained-term-vanishes-above-X-0.8",
        inc.resid_range_highX.max() < 1.0,
        f"include: unexplained term range above X = 0.8 is "
        f"{inc.resid_range_highX.max():.3g} -- the cap does not engage there, which "
        f"is why the cancellation is exact in that sub-range",
    )
    check(
        "task10.binary-carries-the-cap-and-honours-the-flag",
        inc.resid_range.max() > 100.0 and exc.resid_range.max() < 1.0,
        "resolves the US review's open question empirically: the 2.2.7.9 binary both "
        "honours tag_flags(i,2) (the arms differ) and applies the posfun cap "
        "(an unexplained penalty term appears under `include` at low X only)",
    )

    write_checks(CHECKS, os.path.join(OUT, "checks-20.tsv"))


if __name__ == "__main__":
    main()
