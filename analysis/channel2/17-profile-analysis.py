"""Task 1 analysis: read the profile grid and test the predicted signature.

Checks, as pre-specified in the issue:
  task1.include-mixing-block-invariant-to-X
  task1.exclude-mixing-block-monotone-toward-one
  task1.exclude-tag-block-minimum-is-higher
  task1.profile-shift-scales-with-mixing-window

plus two diagnostics the source trace says are needed to interpret a failure of
the first: the survival floor rescales the Newton-Raphson target under
`include` only, and it engages when the raised removal R/X is large -- i.e. at
LOW X. So flatness is expected to be recovered at high X even if it fails over
the whole grid.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2common import OUT, check_logger, write_checks  # noqa: E402

CHECKS, check = check_logger()

FLAT_TOL = 0.02      # "invariant" = relative range below 2%
HIGH_X = 0.80        # above this the raised removal is small enough that the
                     # survival floor should not engage


def rel_range(v):
    v = np.asarray(v, dtype=float)
    return float((v.max() - v.min()) / abs(np.mean(v))) if len(v) else np.nan


def main():
    path = os.path.join(OUT, "profile-objective.csv")
    if not os.path.exists(path):
        raise SystemExit("run 15-profile.py first")
    d = pd.read_csv(path).sort_values(["member_id", "group_id", "arm", "X"])
    d = d[d.ok == 1]
    d["tag_total"] = d.tag_mix + d.tag_post

    mf = pd.read_csv(os.path.join(OUT, "..", "member-frame.csv"))
    Kof = dict(zip(mf.member_id, mf.K))
    d["K"] = d.member_id.map(Kof)

    print(f"{len(d)} successful evaluations; members {sorted(d.member_id.unique())}, "
          f"groups {sorted(d.group_id.unique())}")

    # --- the penalty must be arm-invariant (same formula both arms) ----------
    piv = d.pivot_table(index=["member_id", "group_id", "X"], columns="arm",
                        values="rr_penalty")
    dif = (piv["exclude"] - piv["include"]).abs().max()
    check(
        "task1.penalty-is-identical-under-both-arms",
        dif < 1e-6,
        f"max |rr_penalty(exclude) - rr_penalty(include)| = {dif:.3g} across the grid",
    )

    rows = []
    for (mid, gid), sub in d.groupby(["member_id", "group_id"]):
        rec = dict(member_id=mid, group_id=gid, K=Kof.get(mid))
        for arm in ("include", "exclude"):
            s = sub[sub.arm == arm].sort_values("X")
            if len(s) < 3:
                continue
            hi = s[s.X >= HIGH_X]
            rec[f"mix_relrange_{arm}"] = rel_range(s.tag_mix)
            rec[f"mix_relrange_highX_{arm}"] = rel_range(hi.tag_mix)
            dmix = np.diff(s.tag_mix.values)
            rec[f"mix_frac_decreasing_{arm}"] = float((dmix < 0).mean())
            rec[f"argmin_tagtotal_{arm}"] = float(s.X.values[np.argmin(s.tag_total.values)])
            rec[f"argmin_objective_{arm}"] = float(s.X.values[np.argmin(s.objective.values)])
            rec[f"min_objective_{arm}"] = float(s.objective.min())
        for q in ("argmin_tagtotal", "argmin_objective"):
            if f"{q}_include" in rec and f"{q}_exclude" in rec:
                rec[f"shift_{q}"] = rec[f"{q}_exclude"] - rec[f"{q}_include"]
        rows.append(rec)
    p = pd.DataFrame(rows)
    p.to_csv(os.path.join(OUT, "profile-summary.csv"), index=False)
    print("\n--- per member x group ---")
    print(p[["member_id", "group_id", "K", "mix_relrange_include",
             "mix_relrange_highX_include", "mix_relrange_exclude",
             "mix_frac_decreasing_exclude", "argmin_objective_include",
             "argmin_objective_exclude", "shift_argmin_objective"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # --- pre-specified checks ------------------------------------------------
    worst_flat = p.mix_relrange_include.max()
    check(
        "task1.include-mixing-block-invariant-to-X",
        worst_flat < FLAT_TOL,
        f"worst relative range of the include mixing block over the full grid = "
        f"{worst_flat:.4f} (tolerance {FLAT_TOL}); "
        f"median {p.mix_relrange_include.median():.4f}",
    )
    worst_flat_hi = p.mix_relrange_highX_include.max()
    check(
        "task1.include-mixing-block-invariant-to-X-above-0.8",
        worst_flat_hi < FLAT_TOL,
        f"worst relative range of the include mixing block for X >= {HIGH_X} = "
        f"{worst_flat_hi:.4f}; median {p.mix_relrange_highX_include.median():.4f} "
        f"-- this is the sub-range where the raised removal is small enough that "
        f"the survival floor should not rescale the Newton-Raphson target",
    )
    frac = p.mix_frac_decreasing_exclude.min()
    check(
        "task1.exclude-mixing-block-monotone-toward-one",
        frac == 1.0,
        f"first differences of the exclude mixing block are negative in "
        f"{100*p.mix_frac_decreasing_exclude.mean():.1f}% of grid steps "
        f"(worst profile {100*frac:.1f}%)",
    )
    # The exclude mixing block should also fall FASTER than the include one.
    faster = (p.mix_relrange_exclude > p.mix_relrange_include)
    check(
        "task1.exclude-mixing-block-falls-faster-than-include",
        bool(faster.all()),
        f"relative range exclude > include in {int(faster.sum())}/{len(faster)} "
        f"profiles (median {p.mix_relrange_exclude.median():.4f} vs "
        f"{p.mix_relrange_include.median():.4f})",
    )
    sh = p.dropna(subset=["shift_argmin_objective"])
    check(
        "task1.exclude-tag-block-minimum-is-higher",
        bool((sh.shift_argmin_objective > 0).all()),
        f"objective argmin shift (exclude - include) is positive in "
        f"{int((sh.shift_argmin_objective > 0).sum())}/{len(sh)} profiles; "
        f"median shift {sh.shift_argmin_objective.median():+.4f} in X",
    )
    # The argmin shift splits by group, not by member. Group 7 carries half the
    # prior weight of group 17 (354.5 vs 739.2), so the same pseudo-data pull
    # moves it further before the penalty arrests it -- the N_mix/w ordering
    # from Task 3, reproduced here at fixed parameters.
    per_group = sh.groupby("group_id").shift_argmin_objective
    for gid, s in per_group:
        check(
            f"task1.argmin-shift-positive-in-every-member[g{gid}]",
            bool((s > 0).all()),
            f"group {gid}: shifts {[f'{v:+.4f}' for v in s]} "
            f"(median {s.median():+.4f})",
        )
    if sh.group_id.nunique() >= 2:
        med = per_group.median()
        lo_w = 7 if 7 in med.index else med.index[0]
        hi_w = 17 if 17 in med.index else med.index[-1]
        check(
            "task1.argmin-shift-larger-for-the-lower-weight-group",
            med.get(lo_w, 0) > med.get(hi_w, 0),
            f"median argmin shift: group {lo_w} (w=354.5) {med.get(lo_w):+.4f} vs "
            f"group {hi_w} (w=739.2) {med.get(hi_w):+.4f} -- the same ordering as "
            f"the observed ensemble arm differences (+0.142 vs +0.014)",
        )

    # scaling with the mixing window
    if sh.K.nunique() >= 2:
        lo = sh[sh.K == sh.K.min()].shift_argmin_objective.median()
        hi = sh[sh.K == sh.K.max()].shift_argmin_objective.median()
        rho = sh.K.corr(sh.shift_argmin_objective, method="spearman")
        check(
            "task1.profile-shift-scales-with-mixing-window",
            lo > hi,
            f"median argmin shift {lo:+.4f} at K={sh.K.min():g} vs {hi:+.4f} at "
            f"K={sh.K.max():g}; Spearman(K, shift) = {rho:+.3f}",
        )

    write_checks(CHECKS, os.path.join(OUT, "checks-17.tsv"))


if __name__ == "__main__":
    main()
