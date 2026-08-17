"""Task 1: fixed-parameter profile of the objective against a reporting rate.

Holds every parameter at its converged value, sweeps ONE reporting-rate group's
X-hat over a grid, and evaluates the objective under each setting of
tag_flags(:,2). Nothing re-estimates, so nothing can compensate.

Predicted channel-2 signature:
  include : mixing-window tag block FLAT in X (the estimated rate cancels,
            because the Newton-Raphson drives tagcatch to observed/X)
  exclude : mixing-window tag block DECREASING toward X = 1 (tagcatch is driven
            to observed, so predicted reported returns are X*observed)
  and hence the tag-block argmin sits HIGHER under exclude.

Hard rule from the issue: every invocation is evaluation-only. `-switch 1 1 1 0`
sets parest_flags(1) (the function-evaluation count) to 0; the preflight gate
verifies that this reproduces the reported objective and moves no parameter.

All PAR edits are made by rewriting tokens in the PAR text, never via -switch:
the rates are grouped (so a group means many cells) and the flag flip must
respect the zero-mixing carve-out.
"""

import argparse
import concurrent.futures as cf
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2common import OUT, REPO, check_logger, write_checks  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "analysis", "rr-penalty"))
from parpar import parse_par  # noqa: E402

SCRATCH = os.environ.get(
    "C2_SCRATCH",
    "/tmp/claude-0/-home-user-ofp-sam-bet-2026-ensemble/"
    "1d0b5e36-4bca-5513-9b25-6827138ef590/scratchpad/eval",
)
MODEL = os.path.join(REPO, "model")
INPUTS = ["bet.frq", "bet.tag", "mfcl.cfg", "bet.age_length", "bet.ini",
          "bet.reg_scaling"]
BIN = os.path.join(REPO, "mfclo64")
SWITCH = ["-switch", "1", "1", "1", "0"]   # parest_flags(1) = 0 -> evaluate only

CHECKS, check = check_logger()


# ---------------------------------------------------------------- PAR editing
def _block_bounds(lines, header, nextheaders):
    lo = next(i for i, l in enumerate(lines) if l.strip() == header)
    hi = min(i for i, l in enumerate(lines)
             if i > lo and l.strip() in nextheaders)
    return lo, hi


def _numeric_row_indices(lines, lo, hi):
    return [i for i in range(lo + 1, hi)
            if lines[i].strip() and not lines[i].strip().startswith("#")]


def edit_par(src, dst, rep_group_matrix, group_id=None, new_rate=None,
             arm=None, mixing=None):
    """Rewrite a PAR with a new rate for one group and/or a new arm flag.

    Only the tokens that must change are touched; every other token is copied
    verbatim, so formatting and all other blocks are byte-preserved.
    """
    lines = open(src).read().split("\n")

    if group_id is not None and new_rate is not None:
        lo, hi = _block_bounds(lines, "# tag fish rep",
                               {"# tag fish rep group flags"})
        rows = _numeric_row_indices(lines, lo, hi)
        if len(rows) != rep_group_matrix.shape[0]:
            raise SystemExit(f"rep block has {len(rows)} rows, expected "
                             f"{rep_group_matrix.shape[0]}")
        val = f"{new_rate:.14e}"
        for r, li in enumerate(rows):
            toks = lines[li].split()
            if len(toks) != rep_group_matrix.shape[1]:
                raise SystemExit(f"rep row {r} has {len(toks)} tokens")
            hit = False
            for c in range(len(toks)):
                if rep_group_matrix[r, c] == group_id:
                    toks[c] = val
                    hit = True
            if hit:
                lines[li] = " " + " ".join(toks)

    if arm is not None:
        lo, hi = _block_bounds(lines, "# tag flags", {"# tagmort"})
        rows = _numeric_row_indices(lines, lo, hi)
        if len(rows) != len(mixing):
            raise SystemExit(f"tag flags has {len(rows)} rows, expected {len(mixing)}")
        for r, li in enumerate(rows):
            toks = lines[li].split()
            # zero-mixing carve-out: `include` is undefined with no window, so
            # those release groups keep face-value removal under both arms.
            if arm == "exclude":
                toks[1] = "1"
            else:
                toks[1] = "1" if mixing[r] == 0 else "0"
            lines[li] = " " + " ".join(toks)

    with open(dst, "w") as fh:
        fh.write("\n".join(lines))


# ------------------------------------------------------------------- run/parse
def objective_from_par(path):
    lines = open(path).read().split("\n")
    for i, l in enumerate(lines):
        if l.strip() == "# Objective function value":
            for j in range(i + 1, i + 4):
                s = lines[j].strip()
                if s and not s.startswith("#"):
                    return float(s.split()[0])
    return np.nan


def scalar_after(path, header):
    lines = open(path).read().split("\n")
    for i, l in enumerate(lines):
        if l.strip() == header:
            for j in range(i + 1, i + 4):
                s = lines[j].strip()
                if s and not s.startswith("#"):
                    return float(s.split()[0])
    return np.nan


def tag_block_split(path, mixing):
    """Tag likelihood split into mixing-window and post-mixing parts.

    `# Tag likelihood by tag release by fishery groups` gives, per release
    group and fishery group, a vector over periods at liberty. The first
    tag_flags(it,1) entries are the mixing window.
    """
    lines = open(path).read().split("\n")
    try:
        start = next(i for i, l in enumerate(lines)
                     if l.strip() == "# Tag likelihood by tag release by fishery groups")
    except StopIteration:
        return np.nan, np.nan
    mix_sum = post_sum = 0.0
    it = None
    i = start + 1
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("# tag release "):
            it = int(s.split()[-1])
        elif s.startswith("# fishery group"):
            vals = []
            j = i + 1
            while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith("#"):
                vals.extend(float(t) for t in lines[j].split())
                j += 1
            if it is not None and it <= len(mixing):
                K = int(mixing[it - 1])
                v = np.array(vals)
                mix_sum += float(v[:K].sum()) if K > 0 else 0.0
                post_sum += float(v[K:].sum()) if K > 0 else float(v.sum())
            i = j - 1
        elif s.startswith("#") and "Tag likelihood" in s and "by tag release by fishery" not in s:
            break
        i += 1
    return mix_sum, post_sum


def run_one(job):
    """One evaluation in its own directory. MFCL writes fixed filenames."""
    tag = job["tag"]
    d = os.path.join(SCRATCH, "prof", tag)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    for f in INPUTS:
        shutil.copy(os.path.join(MODEL, f), d)
    shutil.copy(BIN, d)
    edit_par(job["src_par"], os.path.join(d, "in.par"),
             job["rep_group"], job["group_id"], job["rate"],
             job["arm"], job["mixing"])
    t0 = time.time()
    try:
        subprocess.run([os.path.join(d, "mfclo64"), "bet.frq", "in.par", "out.par"]
                       + SWITCH, cwd=d, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=1200)
    except subprocess.TimeoutExpired:
        return dict(**job["meta"], objective=np.nan, rr_penalty=np.nan,
                    tag_mix=np.nan, tag_post=np.nan, seconds=np.nan, ok=0)
    wall = time.time() - t0
    tpo = os.path.join(d, "test_plot_output")
    outpar = os.path.join(d, "out.par")
    rec = dict(**job["meta"])
    rec["objective"] = objective_from_par(outpar) if os.path.exists(outpar) else np.nan
    rec["rr_penalty"] = (scalar_after(tpo, "# Tagged fish reporting rate penalty contribution")
                         if os.path.exists(tpo) else np.nan)
    mx, po = tag_block_split(tpo, job["mixing"]) if os.path.exists(tpo) else (np.nan, np.nan)
    rec["tag_mix"], rec["tag_post"] = mx, po
    rec["seconds"] = wall
    rec["ok"] = int(np.isfinite(rec["objective"]))
    shutil.rmtree(d, ignore_errors=True)
    return rec


def build_jobs(member_ids, group_ids, grid, arms=("include", "exclude")):
    jobs = []
    for mid in member_ids:
        src = os.path.join(REPO, "final-par", f"ensemble-{mid:03d}", "final.par")
        p = parse_par(src)
        mixing = p.tag_flags[:, 0]
        for g in group_ids:
            if not (p.rep_group == g).any():
                continue
            base = float(p.rep[p.rep_group == g][0])
            for arm in arms:
                for x in grid:
                    jobs.append(dict(
                        tag=f"m{mid}_g{g}_{arm}_{x:.4f}".replace(".", "p"),
                        src_par=src, rep_group=p.rep_group, group_id=g,
                        rate=float(x), arm=arm, mixing=mixing,
                        meta=dict(member_id=mid, group_id=g, arm=arm,
                                  X=float(x), base_rate=base,
                                  design_arm=p.arm),
                    ))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true",
                    help="1 member, 2 arms, 1 group, 9 grid points (18 evaluations)")
    ap.add_argument("--members", default="")
    ap.add_argument("--groups", default="17,7")
    ap.add_argument("--points", type=int, default=13)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    design = pd.read_csv(os.path.join(REPO, "out", "member-frame.csv"))
    design = design[design.has_par]

    if a.pilot:
        mid = int(design[(design.K == 0.05) & (design.arm == "exclude")]
                  .member_id.iloc[0])
        members, groups = [mid], [17]
        grid = np.linspace(0.30, 0.98, 9)
    else:
        if a.members:
            members = [int(x) for x in a.members.split(",")]
        else:
            members = []
            for k in (0.05, 0.20, 0.30):
                s = design[(design.K == k) & (design.arm == "exclude")]
                if len(s):
                    members.append(int(s.member_id.iloc[0]))
        groups = [int(x) for x in a.groups.split(",")]
        # denser near the upper end, where the bounded transform compresses
        grid = np.unique(np.concatenate([
            np.linspace(0.30, 0.80, a.points - 4, endpoint=False),
            np.linspace(0.80, 0.985, 5),
        ]))

    jobs = build_jobs(members, groups, grid)
    print(f"members {members}  groups {groups}  grid {len(grid)} points "
          f"-> {len(jobs)} evaluations on {a.workers} workers")
    print(f"grid: {np.round(grid, 4).tolist()}")

    t0 = time.time()
    rows = []
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, rec in enumerate(ex.map(run_one, jobs), 1):
            rows.append(rec)
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  ({time.time()-t0:.0f}s elapsed)")
    d = pd.DataFrame(rows).sort_values(["member_id", "group_id", "arm", "X"])
    name = "profile-objective-pilot.csv" if a.pilot else "profile-objective.csv"
    d.to_csv(os.path.join(OUT, name), index=False)
    print(f"\nwrote out/channel2/{name}  "
          f"({int(d.ok.sum())}/{len(d)} evaluations succeeded, "
          f"median {d.seconds.median():.0f}s each)")
    print(d[["member_id", "group_id", "arm", "X", "objective", "rr_penalty",
             "tag_mix", "tag_post"]].to_string(index=False))


if __name__ == "__main__":
    main()
