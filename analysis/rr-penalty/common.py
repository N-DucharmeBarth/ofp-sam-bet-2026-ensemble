"""Shared loaders + the MFCL reporting-rate penalty reconstruction.

The penalty formula is transcribed from multifan-cl `src/callpen.cpp`, the
`age_flags(198) != 0` branch:

    for i in 1..num_tag_releases+1:
      for j in 1..num_fisheries:
        if gflag[group(i,j)] == 0 and tag_fish_rep_penalty(i,j) > 0:
            gflag[group(i,j)] = 1
            xy += tag_fish_rep_penalty(i,j)
                  * square(tag_fish_rep(i,j) - tag_fish_rep_target(i,j)/100)

Three things follow that the task brief guessed at and that matter:

  * the selector is `penalty > 0`, NOT `tag_fish_rep_active_flags`; the active
    flags govern which parameters are *estimated*, the penalty matrix governs
    which carry a prior;
  * the sum is over unique group ids, each taken at the FIRST cell in
    row-major order with a positive penalty;
  * scale is proportion (target is divided by 100) and the weight multiplies.
"""

from __future__ import annotations

import csv
import os
import re

import numpy as np

from parpar import parse_all, ParError

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(REPO, "out")

DESIGN_CSV = os.path.join(REPO, "data", "ensemble", "successful-model-design.csv")
OBJ_CSV = os.path.join(REPO, "data", "ensemble", "objective-components.csv")
TAG_REP_MAP_R = os.path.join(REPO, "model", "tag_rep_map.R")

PENALTY_COMPONENT = "Tagged Fish Reporting Rate Penalty Contribution"
RECDEV_COMPONENT = "Regional Recruitment Deviates Penalty Contribution"
RESIDUAL_COMPONENT = "Unclassified objective residual"

# Axes read off the design table. Continuous axes are used on their numeric
# scale; K and effort creep are ordered levels on a numeric scale so they are
# treated as continuous too (stated in FINDINGS.md).
CONTINUOUS_AXES = [
    "steepness",
    "tag_tau",
    "tag_mixing_k_cutoff",
    "m_age40_quarterly",
    "effort_creep_primary",
]


def member_num(ensemble_id: str) -> int:
    m = re.search(r"(\d+)\s*$", ensemble_id)
    if not m:
        raise ValueError(f"cannot parse member number from {ensemble_id!r}")
    return int(m.group(1))


def load_design():
    """member_id -> design row (all 88 successful models)."""
    with open(DESIGN_CSV) as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        r["member_id"] = member_num(r["ensemble_id"])
        out[r["member_id"]] = r
    if len(out) != len(rows):
        raise ValueError("duplicate member ids in design table")
    return out


def load_objective():
    """member_id -> {component: value} for all 88 successful models."""
    out = {}
    with open(OBJ_CSV) as fh:
        for r in csv.DictReader(fh):
            out.setdefault(member_num(r["ensemble_id"]), {})[r["Component"]] = float(r["Value"])
    return out


def load_rep_group_names():
    """group id -> (short name, fisheries string) from model/tag_rep_map.R."""
    txt = open(TAG_REP_MAP_R).read()
    names = re.search(r"tag_rep_name = c\((.*?)\), fisheries", txt, re.S)
    fish = re.search(r"\bfisheries = c\((.*?)\), fishery_names", txt, re.S)
    if not (names and fish):
        raise ValueError("cannot parse tag_rep_map.R")
    nm = re.findall(r'"([^"]*)"', names.group(1))
    fs = re.findall(r'"([^"]*)"', fish.group(1))
    if len(nm) != len(fs):
        raise ValueError("tag_rep_map.R name/fishery length mismatch")
    out = {}
    for i, (n, f) in enumerate(zip(nm, fs), start=1):
        label = n.split(":", 1)[1].strip() if ":" in n else n
        out[i] = (label, f)
    return out


def penalty_groups(p):
    """Unique priored/penalised groups for one member, exactly as MFCL selects them.

    Returns a list of dicts, in MFCL's own visitation order.
    """
    ngrp = int(p.rep_group.max())
    seen = np.zeros(ngrp + 1, dtype=bool)
    rows = []
    nrow, ncol = p.rep.shape
    for i in range(nrow):
        for j in range(ncol):
            g = int(p.rep_group[i, j])
            if g <= 0 or seen[g]:
                continue
            w = float(p.rep_penalty[i, j])
            if w <= 0:
                continue
            seen[g] = True
            rep = float(p.rep[i, j])
            tgt = float(p.rep_target[i, j]) / 100.0
            rows.append(
                dict(
                    group_id=g,
                    first_row=i + 1,
                    first_fishery=j + 1,
                    rep_hat=rep,
                    target_prop=tgt,
                    penalty_wt=w,
                    contribution=w * (rep - tgt) ** 2,
                    signed_dev=rep - tgt,
                    abs_dev=abs(rep - tgt),
                )
            )
    return rows


def group_cell_census(p, g):
    """All cells carrying group id g, with their value/target/weight/active."""
    mask = p.rep_group == g
    return dict(
        n_cells=int(mask.sum()),
        n_active=int(p.rep_active[mask].sum()),
        release_rows=sorted({int(i) + 1 for i in np.where(mask)[0]}),
        fisheries=sorted({int(j) + 1 for j in np.where(mask)[1]}),
        rep_vals=np.unique(p.rep[mask]),
        target_vals=np.unique(p.rep_target[mask]),
        wt_vals=np.unique(p.rep_penalty[mask]),
    )


def recompute_penalty(p) -> float:
    return float(sum(r["contribution"] for r in penalty_groups(p)))


def rep_upper_bound(p) -> float:
    """Estimable upper bound on tag_fish_rep, from multifan-cl src/newmau5a.cpp.

    parest_flags(33) > 0  ->  bound is parest_flags(33)/100 (exp transform)
    otherwise             ->  default linear bound of 1.01
    """
    pf33 = int(p.parest_flags[32]) if p.parest_flags.size >= 33 else 0
    return (pf33 / 100.0) if pf33 > 0 else 1.01


def compact_ranges(nums):
    """[1,2,4,5,6,10] -> '1-2,4-6,10'"""
    nums = sorted(set(int(n) for n in nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ",".join(out)


def load_members():
    """Parsed pars keyed by member id, with design + objective attached.

    Only members with a retained final.par appear here. The design and
    objective tables cover more members; Task 4 uses that wider set.
    """
    pars = parse_all(os.path.join(REPO, "final-par"))
    design = load_design()
    obj = load_objective()
    members = []
    for p in pars:
        d = design.get(p.member_id)
        if d is None:
            raise ParError(f"member {p.member_id} has a par but no design row")
        o = obj.get(p.member_id)
        if o is None:
            raise ParError(f"member {p.member_id} has a par but no objective row")
        members.append(dict(par=p, design=d, obj=o))
    return members


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path
