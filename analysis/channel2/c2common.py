"""Shared loaders for the channel-2 analysis.

Reuses the parser and penalty machinery from analysis/rr-penalty/ rather than
re-deriving them, so both analyses read the PARs the same way.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RRP = os.path.join(REPO, "analysis", "rr-penalty")
sys.path.insert(0, RRP)

from common import (  # noqa: E402,F401
    PENALTY_COMPONENT,
    load_design,
    load_members,
    load_objective,
    load_rep_group_names,
    penalty_groups,
    rep_upper_bound,
    write_csv,
)
from tagfile import read_tag_file  # noqa: E402,F401

OUT = os.path.join(REPO, "out", "channel2")
FIG = os.path.join(OUT, "figures")

# --- MFCL bounded transform for tag_fish_rep -------------------------------
# src/newmau5a.cpp: parest_flags(33) > 0 selects
#     set_value_exp(tag_fish_rep, x, ii, .001, parest_flags(33)/100, pen, ..., scale)
# src/setcomm5.cpp: set_value_exp(x,v,ii,fmin,fmax,pen) -> x = exp(boundp(v,log fmin,log fmax))
# src/scbound.cpp : boundpin(x,fmin,fmax,s) = s * asin(2*(x-fmin)/(fmax-fmin) - 1) / 1.570795
# scale is 1000 unless parest_flags(387), which is 0 in every member.
REP_LB = 0.001
SCALE = 1000.0
ASIN_DIV = 1.570795


def rep_to_estimation_scale(rep, ub, lb=REP_LB, s=SCALE):
    """Proportion-space reporting rate -> the scale the optimiser works on.

    The arcsine link has an infinite derivative at the bound, so a fixed
    proportion-space epsilon is NOT a fixed distance from the bound here --
    which is exactly why Task 4 re-tests bound contact on this scale.
    """
    rep = np.asarray(rep, dtype=float)
    lo, hi = np.log(lb), np.log(ub)
    u = 2.0 * (np.log(rep) - lo) / (hi - lo) - 1.0
    u = np.clip(u, -1.0, 1.0)
    return s * np.arcsin(u) / ASIN_DIV


def estimation_scale_bound(s=SCALE):
    """v at the upper bound: asin(1)/1.570795 * s."""
    return s * np.arcsin(1.0) / ASIN_DIV


def check_logger():
    checks = []

    def check(name, passed, detail=""):
        checks.append((name, bool(passed), detail))
        print(f"[{'HELD' if passed else 'DID NOT HOLD'}] {name}"
              + (f"  {detail}" if detail else ""))
        return passed

    return checks, check


def write_checks(checks, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        for n, p, d in checks:
            fh.write(f"{'PASS' if p else 'FAIL'}\t{n}\t{d}\n")
