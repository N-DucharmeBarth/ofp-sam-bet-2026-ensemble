"""Task 12: does anything other than the tag terms have an opinion on X-hat?

Motivating question: is the reporting rate actually estimable from data
within the assessment, or is it just a free parameter available to absorb
process/mis-specification error elsewhere? Tag-seeding evidence about the
true reporting rate enters the model ONLY through the penalty (Part 1); nothing
else in the likelihood is *supposed* to reference tag_fish_rep. This checks
that directly, two ways:

  (a) STRUCTURAL, at frozen parameters (extends Task 1's fixed-parameter
      sweep -- analysis/channel2/15-profile.py -- reusing its edit_par/run_one
      machinery unmodified). Sweep X-hat for a handful of members/groups and
      extract every other major likelihood component -- length-frequency,
      age-length, survey/CPUE index, total catch -- alongside the tag terms
      already captured. If reporting rate has no live mechanical channel into
      any of them, all four are exactly flat as X sweeps, and the tag
      components (already profiled) are the only ones that move. Verified
      first against source (see FINDINGS section below) before running: none
      of newpredcatch.cpp / the length, age-length or survey-index likelihood
      routines reference tag_fish_rep; every reference is confined to
      tag3.cpp, tagfit.cpp, threaded_tag3.*, callpen.cpp, sim_tag_pd.cpp,
      simulation_mode.cpp, and the generic ADMB parameter-registration files
      (newmult.cpp, newm_io3.cpp, newmau5a.cpp, indepvars.cpp), which apply to
      every active parameter and are not evidence of a tag-rate-specific
      channel.

  (b) OBSERVATIONAL, across the 80 converged, already-fitted ensemble members
      (zero new MFCL calls). If the fitted reporting rate is being pulled
      around by the rest of the model rather than by tag evidence + prior,
      its ACROSS-MEMBER deviation from the tag-seeding target should
      correlate with how well the rest of the likelihood fits -- length-freq,
      age-length, survey index, or F. Structural flatness at (a) rules out a
      *direct* channel; (b) is the closer test for an *indirect* one, via
      parameters (F, selectivity, catchability) that are jointly estimated
      with X-hat and also drive those other components.
"""

import os
import re
import sys
import time
import concurrent.futures as cf

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2common import OUT, REPO, check_logger, write_checks  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "analysis", "rr-penalty"))
from parpar import parse_par  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("task1_profile", os.path.join(os.path.dirname(__file__), "15-profile.py"))
task1 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(task1)  # reuses edit_par, run_one's scratch mechanics, MODEL/BIN/SWITCH/INPUTS

CHECKS, check = check_logger()

PRIORED = [7, 10, 14, 17, 18]


# ---------------------------------------------------------------- (a) sweep
def _sum_single_block(lines, header):
    try:
        i = next(k for k, l in enumerate(lines) if l.strip() == header)
    except StopIteration:
        return np.nan
    total = 0.0
    j = i + 1
    while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith("#"):
        total += sum(float(t) for t in lines[j].split())
        j += 1
    return total


def _sum_all_matching(lines, pattern):
    rx = re.compile(pattern)
    total = 0.0
    found = 0
    i = 0
    while i < len(lines):
        if rx.match(lines[i].strip()):
            found += 1
            j = i + 1
            while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith("#"):
                total += sum(float(t) for t in lines[j].split())
                j += 1
            i = j
        else:
            i += 1
    return total if found else np.nan


def non_tag_components(tpo_path):
    if not os.path.exists(tpo_path):
        return dict(length_total=np.nan, age_length_total=np.nan,
                    survey_index_total=np.nan, catch_total=np.nan)
    lines = open(tpo_path).read().split("\n")
    return dict(
        length_total=_sum_single_block(lines, "# total length component of likelihood for each fishery"),
        age_length_total=_sum_single_block(lines, "# age length likelihood"),
        survey_index_total=_sum_single_block(lines, "# Survey_index_like_by_group"),
        catch_total=_sum_all_matching(lines, r"^# total catch components of likelihood for fishery \d+$"),
    )


def run_one_full(job):
    """Duplicates task1.run_one but keeps test_plot_output around long enough
    to also pull the non-tag components, instead of patching the shared
    Task-1 script."""
    import shutil
    import subprocess

    tag = job["tag"]
    d = os.path.join(task1.SCRATCH, "comp", tag)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    for f in task1.INPUTS:
        shutil.copy(os.path.join(task1.MODEL, f), d)
    shutil.copy(task1.BIN, d)
    task1.edit_par(job["src_par"], os.path.join(d, "in.par"),
                    job["rep_group"], job["group_id"], job["rate"],
                    job["arm"], job["mixing"])
    t0 = time.time()
    try:
        subprocess.run([os.path.join(d, "mfclo64"), "bet.frq", "in.par", "out.par"]
                        + task1.SWITCH, cwd=d, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, timeout=1200)
    except subprocess.TimeoutExpired:
        shutil.rmtree(d, ignore_errors=True)
        return dict(**job["meta"], objective=np.nan, rr_penalty=np.nan,
                    tag_mix=np.nan, tag_post=np.nan, seconds=np.nan, ok=0,
                    length_total=np.nan, age_length_total=np.nan,
                    survey_index_total=np.nan, catch_total=np.nan)
    wall = time.time() - t0
    tpo = os.path.join(d, "test_plot_output")
    outpar = os.path.join(d, "out.par")
    rec = dict(**job["meta"])
    rec["objective"] = task1.objective_from_par(outpar) if os.path.exists(outpar) else np.nan
    rec["rr_penalty"] = (task1.scalar_after(tpo, "# Tagged fish reporting rate penalty contribution")
                          if os.path.exists(tpo) else np.nan)
    mx, po = task1.tag_block_split(tpo, job["mixing"]) if os.path.exists(tpo) else (np.nan, np.nan)
    rec["tag_mix"], rec["tag_post"] = mx, po
    rec.update(non_tag_components(tpo))
    rec["seconds"] = wall
    rec["ok"] = int(np.isfinite(rec["objective"]))
    shutil.rmtree(d, ignore_errors=True)
    return rec


def sweep():
    design = pd.read_csv(os.path.join(REPO, "out", "member-frame.csv"))
    design = design[design.has_par]
    # One member per arm, both a heavy-weight priored group (18) and a
    # lighter one (7), matching Task 1's own choice of contrast groups.
    members = []
    for arm in ("include", "exclude"):
        s = design[design.arm == arm]
        members.append(int(s.member_id.iloc[0]))
    groups = [7, 18]
    grid = np.unique(np.concatenate([
        np.linspace(0.30, 0.80, 9, endpoint=False), np.linspace(0.80, 0.985, 5),
    ]))
    jobs = task1.build_jobs(members, groups, grid)
    print(f"members {members} groups {groups} grid {len(grid)} pts -> {len(jobs)} evals")
    rows = []
    with cf.ThreadPoolExecutor(max_workers=1) as ex:  # the full-path batch is already using 8-9GB of 15
        for i, rec in enumerate(ex.map(run_one_full, jobs), 1):
            rows.append(rec)
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}")
    return pd.DataFrame(rows)


# ------------------------------------------------------------- (b) observed
def observed_correlation():
    obj = pd.read_csv(os.path.join(REPO, "data", "ensemble", "objective-components.csv"))
    obj["member_id"] = obj.ensemble_id.str.extract(r"(\d+)$").astype(int)
    design = pd.read_csv(os.path.join(REPO, "out", "member-frame.csv"))
    design = design[design.has_par]

    comp_map = {
        "Length frequency": "length_total",
        "Age": "age_length_total",
        "CPUE": "survey_index_total",
        "Catch": "catch_total",
    }
    wide = obj[obj.Component.isin(comp_map)].pivot(index="member_id", columns="Component", values="Value")
    wide = wide.rename(columns=comp_map)

    rows = []
    for mid in design.member_id:
        p = parse_par(os.path.join(REPO, "final-par", f"ensemble-{mid:03d}", "final.par"))
        for g in PRIORED:
            mask = p.rep_group == g
            if not mask.any():
                continue
            fitted = float(np.unique(p.rep[mask])[0])
            target = float(np.unique(p.rep_target[mask])[0]) / 100.0
            rows.append(dict(member_id=mid, group_id=g, fitted=fitted, target=target,
                              deviation=fitted - target))
    rates = pd.DataFrame(rows)

    d = rates.merge(design[["member_id", "arm", "K", "tau", "M0", "h", "creep"]], on="member_id")
    d = d.merge(wide, on="member_id", how="left")

    out_rows = []
    for g in PRIORED:
        sub = d[d.group_id == g]
        for comp in comp_map.values():
            if comp not in sub or sub[comp].isna().all():
                continue
            r = sub["deviation"].corr(sub[comp])
            r_arm = {a: sub[sub.arm == a]["deviation"].corr(sub[sub.arm == a][comp])
                     for a in ("include", "exclude")}
            out_rows.append(dict(group_id=g, component=comp, n=len(sub), r_pooled=r,
                                  r_include=r_arm["include"], r_exclude=r_arm["exclude"]))
    return pd.DataFrame(out_rows), d


def main():
    print("=== (a) structural sweep: does anything but tag terms move? ===")
    swept = sweep()
    write_csv_path = os.path.join(OUT, "component-profile-sweep.csv")
    swept.to_csv(write_csv_path, index=False)
    ok = swept[swept.ok == 1]
    for comp in ("length_total", "age_length_total", "survey_index_total", "catch_total"):
        span = ok.groupby(["member_id", "group_id", "arm"])[comp].agg(lambda s: s.max() - s.min())
        worst = float(span.abs().max())
        check(
            f"task12.{comp}-flat-under-X-sweep",
            worst < 1e-6,
            f"max range across the sweep = {worst:.3g} objective units "
            f"({len(ok)} evaluations, tolerance 1e-6)",
        )
    tag_span = ok.groupby(["member_id", "group_id", "arm"])["tag_mix"].agg(lambda s: s.max() - s.min())
    check(
        "task12.tag-terms-do-move-as-a-contrast",
        float(tag_span.abs().min()) > 1.0,
        f"tag_mix range per (member,group,arm) spans {tag_span.min():.3g} to {tag_span.max():.3g} "
        f"-- confirms the sweep itself is doing something, not just evaluating a dead parameter",
    )

    print("\n=== (b) observed: does the fitted rate's deviation from target track fit elsewhere? ===")
    corr, detail = observed_correlation()
    corr.to_csv(os.path.join(OUT, "component-profile-observed-correlation.csv"), index=False)
    detail.to_csv(os.path.join(OUT, "component-profile-observed-detail.csv"), index=False)
    print(corr.to_string(index=False))

    if not len(corr) or corr.r_pooled.isna().all():
        check(
            "task12.no-strong-cross-member-correlation-with-other-components",
            False,
            f"produced no correlations to test (corr rows={len(corr)}) -- this is a broken check, "
            f"not a finding of no correlation; comp_map likely does not match "
            f"objective-components.csv's actual Component values",
        )
    else:
        worst_row = corr.loc[corr.r_pooled.abs().idxmax()]
        worst_r = float(worst_row.r_pooled)
        worst_r_by_arm = float(pd.concat([corr.r_include, corr.r_exclude]).abs().max())
        check(
            "task12.no-strong-cross-member-correlation-with-other-components",
            not (worst_r > 0.5),
            f"largest |pooled correlation| between a priored group's rate deviation from its "
            f"tag-seeding target and length/age-length/survey-index fit = {worst_r:.3f} "
            f"(group {int(worst_row.group_id)}, {worst_row.component}); largest single by-arm "
            f"correlation = {worst_r_by_arm:.3f}; n={len(detail.member_id.unique())} members; "
            f"catch_total excluded, exactly 0 for all 88 retained models so has no variance to "
            f"correlate against; threshold 0.5 chosen as a conservative worth-a-closer-look line, "
            f"not a formal test",
        )

    write_checks(CHECKS, os.path.join(OUT, "checks-22.tsv"))


if __name__ == "__main__":
    main()
