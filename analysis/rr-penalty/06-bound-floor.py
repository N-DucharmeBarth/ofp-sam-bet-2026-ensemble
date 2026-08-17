"""Task 8: bound and floor screens -- ways the comparison could be censored.

1. REPORTING-RATE UPPER BOUND. multifan-cl src/newmau5a.cpp bounds tag_fish_rep
   at parest_flags(33)/100 when that flag is set (exp transform), otherwise at a
   default of 1.01. If the `include` arm is pinned at the bound more often, its
   low penalties are partly truncation rather than fit.

2. SURVIVAL FLOOR. src/tag3.cpp raises the mixing-period removal by the
   reporting rate under `include` only:

       actual_tag_catch = tot_tag_catch / (1e-6 + tag_rep_rate)   [tag_flags(it,2)==0]
       actual_tag_catch = tot_tag_catch                           [tag_flags(it,2)==1]

   and then floors survival with posfun once

       surv_rate = 1 - sum(actual_tag_catch)/tags_present  <=  cut + fringe

   with cut = 0.2 (age_flags(118)/100 if set) and fringe = 0.02 (age_flags(119)/100).
   The screen below is a bet.tag-only proxy for that condition -- see the
   docstring on `floor_screen` for exactly how it differs from MFCL's.

   src/tag3.cpp also discards the accrued posfun penalty outright for tag
   release groups 21, 72 and 112:
       switch (it) { case 72: case 21: case 112: break; default: _ffpen+=ffpen; }
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, PENALTY_COMPONENT, load_members, penalty_groups, rep_upper_bound, write_csv  # noqa: E402
from tagfile import read_tag_file  # noqa: E402

# posfun engages at surv_rate <= cut + fringe; both are age_flags-settable and
# both are unset (0) in every member, so the MFCL defaults apply.
CUT, FRINGE = 0.2, 0.02
DISCARDED_RELEASE_GROUPS = (21, 72, 112)

CHECKS = []


def check(name, passed, detail=""):
    CHECKS.append((name, bool(passed), detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return passed


def floor_screen(par, tag, cut=CUT, fringe=FRINGE):
    """Per release group: does the raised removal plausibly drive posfun?

    Proxy, not MFCL's own calculation. Differences, all in the direction of
    UNDER-counting floor engagement:
      * tags present is taken as the number released less the cumulative raised
        removal, ignoring natural mortality, tag shedding and tag-induced
        mortality, all of which shrink the denominator;
      * removals are pooled over regions rather than evaluated per region, and
        MFCL evaluates surv_rate on the tags present in ONE region;
      * recapture periods come from the quarterly grid (see tagfile.py).
    So a group flagged here is a strong candidate; a group not flagged is not
    cleared.
    """
    mix = par.tag_flags[:, 0]
    f2 = par.tag_flags[:, 1]
    out = []
    for rel in tag["releases"]:
        it = rel["group"] - 1
        K = int(mix[it])
        n0 = rel["n_released"]
        if K <= 0 or n0 <= 0:
            out.append(dict(release_group=rel["group"], min_surv=1.0, engages=False,
                            raised_removal=0.0, face_removal=0.0, n_released=n0,
                            arm_for_group="exclude" if f2[it] == 1 else "include"))
            continue
        by_period = {}
        for rec in rel["recoveries"]:
            if not (0 <= rec["elapsed"] < K):
                continue
            j = rec["fishery"] - 1
            xf = float(par.rep[it, j])
            face = rec["number"]
            raised = face / (1e-6 + xf) if f2[it] == 0 else face
            p = rec["elapsed"]
            a, b = by_period.get(p, (0.0, 0.0))
            by_period[p] = (a + raised, b + face)
        remaining, min_surv = n0, 1.0
        tot_raised = tot_face = 0.0
        for p in sorted(by_period):
            raised, face = by_period[p]
            tot_raised += raised
            tot_face += face
            s = 1.0 - raised / remaining if remaining > 0 else 0.0
            min_surv = min(min_surv, s)
            remaining = max(remaining - raised, 0.0)
        out.append(
            dict(
                release_group=rel["group"],
                n_released=n0,
                raised_removal=tot_raised,
                face_removal=tot_face,
                min_surv=min_surv,
                engages=bool(min_surv <= cut + fringe),
                arm_for_group="exclude" if f2[it] == 1 else "include",
            )
        )
    return pd.DataFrame(out)


def main():
    members = load_members()
    tag = read_tag_file()

    ubs = {rep_upper_bound(m["par"]) for m in members}
    pf33 = {int(m["par"].parest_flags[32]) for m in members}
    check("task8.upper-bound-constant-across-members", len(ubs) == 1,
          f"parest_flags(33) = {pf33} -> tag_fish_rep upper bound = {ubs}")
    UB = next(iter(ubs))
    LB = 0.001

    for k in (117, 118, 119):
        vals = {int(m["par"].age_flags[k - 1]) for m in members}
        check(f"task8.age_flags({k})-unset-so-posfun-defaults-apply", vals == {0}, f"{vals}")

    # --- 1. bound screen -------------------------------------------------------
    rows, member_bound = [], []
    for m in members:
        p = m["par"]
        pri_at_bound = flat_at_bound = 0
        for g in penalty_groups(p):
            at_ub = abs(g["rep_hat"] - UB) <= 1e-3
            at_lb = abs(g["rep_hat"] - LB) <= 1e-3
            rows.append(
                dict(
                    screen="upper-bound",
                    member_id=p.member_id,
                    arm=p.arm,
                    unit=f"group {g['group_id']}",
                    is_priored=int(g["penalty_wt"] > 1),
                    value=f"{g['rep_hat']:.6g}",
                    threshold=f"{UB:.6g}",
                    slack=f"{UB - g['rep_hat']:.6g}",
                    flagged=int(at_ub or at_lb),
                    note="at upper bound" if at_ub else ("at lower bound" if at_lb else ""),
                )
            )
            if at_ub or at_lb:
                if g["penalty_wt"] > 1:
                    pri_at_bound += 1
                else:
                    flat_at_bound += 1
        member_bound.append(
            dict(member_id=p.member_id, arm=p.arm, n_priored_at_bound=pri_at_bound,
                 n_flat_at_bound=flat_at_bound, penalty=m["obj"][PENALTY_COMPONENT])
        )
    mb = pd.DataFrame(member_bound)
    frac = mb.assign(any_pri=mb.n_priored_at_bound > 0).groupby("arm").any_pri.mean()
    frac_any = mb.assign(any_=(mb.n_priored_at_bound + mb.n_flat_at_bound) > 0).groupby("arm").any_.mean()
    print(f"\nfraction of members with any PRIORED group within 1e-3 of a bound: "
          f"include {frac.get('include', 0):.3f}, exclude {frac.get('exclude', 0):.3f}")
    print(f"fraction with ANY penalised group at a bound: "
          f"include {frac_any.get('include', 0):.3f}, exclude {frac_any.get('exclude', 0):.3f}")
    check(
        "task8.include-arm-not-more-often-at-the-rr-bound",
        frac.get("include", 0) <= frac.get("exclude", 0) + 1e-12,
        f"priored-group bound hits: include {frac.get('include', 0):.1%} vs "
        f"exclude {frac.get('exclude', 0):.1%} of members",
    )

    # --- 2. floor screen -------------------------------------------------------
    fl_rows, member_floor = [], []
    for m in members:
        p = m["par"]
        fs = floor_screen(p, tag)
        eng = fs[fs.engages]
        member_floor.append(
            dict(
                member_id=p.member_id,
                arm=p.arm,
                K=float(m["design"]["tag_mixing_k_cutoff"]),
                penalty=m["obj"][PENALTY_COMPONENT],
                n_engaging=len(eng),
                min_surv=float(fs.min_surv.min()),
                engaging_groups=";".join(str(int(x)) for x in eng.release_group),
                n_engaging_discarded=int(eng.release_group.isin(DISCARDED_RELEASE_GROUPS).sum()),
            )
        )
        for _, r in eng.iterrows():
            fl_rows.append(
                dict(
                    screen="survival-floor",
                    member_id=p.member_id,
                    arm=p.arm,
                    unit=f"release group {int(r.release_group)}",
                    is_priored="",
                    value=f"{r.min_surv:.6g}",
                    threshold=f"{CUT + FRINGE:.6g}",
                    slack=f"{r.min_surv - (CUT + FRINGE):.6g}",
                    flagged=1,
                    note="posfun penalty DISCARDED by MFCL"
                    if int(r.release_group) in DISCARDED_RELEASE_GROUPS
                    else "",
                )
            )
    mf = pd.DataFrame(member_floor)
    write_csv(os.path.join(OUT, "bound-floor-screen.csv"), list(rows[0].keys()), rows + fl_rows)
    mf.to_csv(os.path.join(OUT, "floor-screen-by-member.csv"), index=False)

    print(f"\nmembers with >=1 floor-engaging release group, by arm:")
    print(mf.assign(any_=mf.n_engaging > 0).groupby("arm").agg(
        n_members=("member_id", "size"),
        frac_engaging=("any_", "mean"),
        median_n_engaging=("n_engaging", "median"),
        min_surv=("min_surv", "min"),
    ).to_string())

    inc_f = mf[mf.arm == "include"].n_engaging
    exc_f = mf[mf.arm == "exclude"].n_engaging
    check(
        "task8.floor-engagement-higher-under-include",
        (inc_f > 0).mean() > (exc_f > 0).mean(),
        f"share of members with >=1 engaging release group: include "
        f"{(inc_f > 0).mean():.1%} vs exclude {(exc_f > 0).mean():.1%}",
    )

    # Are the LOWEST-penalty include members disproportionately floor-engaging?
    inc = mf[mf.arm == "include"].copy()
    if len(inc) >= 8:
        lo = inc.nsmallest(len(inc) // 3, "penalty")
        hi = inc.nlargest(len(inc) // 3, "penalty")
        rho = stats.spearmanr(inc.penalty, inc.n_engaging)
        print(f"\ninclude arm: lowest-penalty tercile mean n_engaging = "
              f"{lo.n_engaging.mean():.2f}; highest tercile = {hi.n_engaging.mean():.2f}; "
              f"Spearman(penalty, n_engaging) = {rho.statistic:+.3f} (p={rho.pvalue:.3g})")
        check(
            "task8.lowest-penalty-include-members-are-more-floor-engaging",
            lo.n_engaging.mean() > hi.n_engaging.mean(),
            f"lowest-penalty tercile {lo.n_engaging.mean():.2f} vs highest "
            f"{hi.n_engaging.mean():.2f} engaging release groups; "
            f"Spearman(penalty, n_engaging) = {rho.statistic:+.3f}",
        )

    # release groups 21 / 72 specifically
    all_eng = pd.DataFrame(fl_rows)
    if len(all_eng):
        seen = sorted({int(u.split()[-1]) for u in all_eng.unit})
        hit = [g for g in DISCARDED_RELEASE_GROUPS if g in seen]
        check(
            "task8.discarded-release-groups-appear-in-floor-screen",
            len(hit) > 0,
            f"of the groups whose posfun penalty MFCL discards {DISCARDED_RELEASE_GROUPS}, "
            f"the screen flags {hit or 'none'}; {len(seen)} distinct release groups flagged overall",
        )
        n_disc = int(mf.n_engaging_discarded.sum())
        print(f"\ntotal (member x release-group) floor flags: {len(all_eng)}; "
              f"of which on discarded groups 21/72/112: {n_disc}")
    else:
        check("task8.discarded-release-groups-appear-in-floor-screen", False,
              "the screen flagged no release group in any member")

    with open(os.path.join(OUT, "checks-06.txt"), "w") as fh:
        for n, pa, det in CHECKS:
            fh.write(f"{'PASS' if pa else 'FAIL'}\t{n}\t{det}\n")


if __name__ == "__main__":
    main()
