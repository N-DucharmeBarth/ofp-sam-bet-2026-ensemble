"""Tasks 1-3: parse, long-format group table, penalty reconstruction tripwire."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    OUT,
    PENALTY_COMPONENT,
    compact_ranges,
    group_cell_census,
    load_members,
    load_rep_group_names,
    penalty_groups,
    recompute_penalty,
    write_csv,
)

# Absolute tolerance on |recomputed - reported|. The reported values are printed
# by MFCL to 12 significant figures, so anything above ~1e-6 is a real
# disagreement rather than round-tripping loss.
TOL = 1e-6

CHECKS = []


def check(name, passed, detail=""):
    CHECKS.append((name, bool(passed), detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return passed


def main():
    members = load_members()
    names = load_rep_group_names()

    # --- Task 1 post-conditions ------------------------------------------------
    dims = {(m["par"].ntag, m["par"].nfish, m["par"].nrep_rows) for m in members}
    check("task1.dims-constant-across-members", len(dims) == 1, f"{dims}")
    ntag, nfish, nrep_rows = next(iter(dims))
    ngroups = {int(m["par"].rep_group.max()) for m in members}
    check("task1.group-count-constant", len(ngroups) == 1, f"max group id {ngroups}")

    # Arm label: derived from tag_flags(:,2) on release groups that have a mixing
    # window, cross-checked against the design table.
    arm_ok, zero_mix_ok = True, True
    for m in members:
        p, d = m["par"], m["design"]
        arm = p.arm
        want = {"inclusion": "include", "exclusion": "exclude"}[d["tag_reporting"]]
        if arm != want:
            arm_ok = False
            print(f"   member {p.member_id}: par arm {arm} vs design {want}")
        # every flag-1 group under `include` must be a zero-mixing group
        f2, mix = p.tag_flags[:, 1], p.tag_flags[:, 0]
        if arm == "include":
            if not np.array_equal(np.where(f2 == 1)[0], np.where(mix == 0)[0]):
                zero_mix_ok = False
            if int(d["tag_reporting_zero_mixing_exclusions"]) != int((f2 == 1).sum()):
                zero_mix_ok = False
        else:
            if not (f2 == 1).all():
                zero_mix_ok = False
    check("task1.arm-label-agrees-with-design", arm_ok)
    check(
        "task1.flag2-exceptions-are-exactly-zero-mixing-groups",
        zero_mix_ok,
        "under `include`, tag_flags(:,2)=1 iff tag_flags(:,1)=0",
    )

    # --- Task 2: long-format group table ---------------------------------------
    rows = []
    const_ok = True
    for m in members:
        p = m["par"]
        pg = penalty_groups(p)
        for g in pg:
            cen = group_cell_census(p, g["group_id"])
            # MFCL takes the first positive-penalty cell for a group; if the
            # cells carrying that id disagree, the group-flag semantics are not
            # what this analysis assumes.
            pos = p.rep_penalty[p.rep_group == g["group_id"]] > 0
            sub = p.rep_group == g["group_id"]
            vals = np.unique(p.rep[sub][pos])
            tgts = np.unique(p.rep_target[sub][pos])
            wts = np.unique(p.rep_penalty[sub][pos])
            if not (vals.size == 1 and tgts.size == 1 and wts.size == 1):
                const_ok = False
                print(
                    f"   member {p.member_id} group {g['group_id']}: "
                    f"vals={vals} tgts={tgts} wts={wts}"
                )
            label, fish_str = names.get(g["group_id"], ("(unmapped)", ""))
            rows.append(
                dict(
                    member_id=p.member_id,
                    arm=p.arm,
                    group_id=g["group_id"],
                    group_label=label,
                    fisheries=compact_ranges(cen["fisheries"]),
                    n_release_groups=len(cen["release_rows"]),
                    n_cells=cen["n_cells"],
                    n_cells_active=cen["n_active"],
                    rep_hat=f"{g['rep_hat']:.12g}",
                    target_prop=f"{g['target_prop']:.12g}",
                    penalty_wt=f"{g['penalty_wt']:.12g}",
                    is_priored=int(g["penalty_wt"] > 1.0),
                    signed_dev=f"{g['signed_dev']:.12g}",
                    abs_dev=f"{g['abs_dev']:.12g}",
                    contribution=f"{g['contribution']:.12g}",
                )
            )
    check(
        "task2.group-cells-carry-identical-value-target-weight",
        const_ok,
        "across all cells of a group with penalty > 0",
    )
    write_csv(
        os.path.join(OUT, "rr-groups.csv"),
        list(rows[0].keys()),
        rows,
    )
    print(f"  wrote out/rr-groups.csv ({len(rows)} rows)")

    # Which groups are estimated but carry no prior, and vice versa?
    p0 = members[0]["par"]
    penalised = {r["group_id"] for r in penalty_groups(p0)}
    active_groups = set(np.unique(p0.rep_group[p0.rep_active == 1]).tolist()) - {0}
    check(
        "task2.penalised-groups-are-a-subset-of-active-groups",
        penalised <= active_groups,
        f"penalised={sorted(penalised)} active={sorted(active_groups)}",
    )

    # --- Task 3: reconstruction ------------------------------------------------
    rec_rows = []
    resid = []
    for m in members:
        p = m["par"]
        got = recompute_penalty(p)
        rep = m["obj"][PENALTY_COMPONENT]
        rec_rows.append(
            dict(
                member_id=p.member_id,
                arm=p.arm,
                reported=f"{rep:.12g}",
                recomputed=f"{got:.12g}",
                residual=f"{got - rep:.6g}",
                rel_residual=f"{(got - rep) / rep if rep else float('nan'):.6g}",
                n_penalised_groups=len(penalty_groups(p)),
            )
        )
        resid.append(got - rep)
    resid = np.array(resid)
    write_csv(
        os.path.join(OUT, "penalty-reconstruction.csv"), list(rec_rows[0].keys()), rec_rows
    )
    ok = check(
        "task3.reconstruction-matches-reported",
        np.max(np.abs(resid)) < TOL,
        f"max|resid| = {np.max(np.abs(resid)):.3g} (tol {TOL:g}); "
        f"median resid = {np.median(resid):.3g}",
    )
    print(f"  wrote out/penalty-reconstruction.csv ({len(rec_rows)} rows)")

    with open(os.path.join(OUT, "checks-01.txt"), "w") as fh:
        for n, pa, d in CHECKS:
            fh.write(f"{'PASS' if pa else 'FAIL'}\t{n}\t{d}\n")

    if not ok:
        sys.exit("Task 3 reconstruction failed -- downstream tasks are gated on this.")
    if not all(c[1] for c in CHECKS):
        sys.exit("a Task 1/2 post-condition failed")


if __name__ == "__main__":
    main()
