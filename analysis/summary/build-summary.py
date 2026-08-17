"""Render the combined findings summary to HTML (and PDF via 09-build-pdf pattern).

Every number is pulled from the out/ CSVs rather than retyped, and the build
fails on any unsubstituted placeholder, so the page cannot drift from the
analysis it summarises.
"""

import base64
import csv
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(REPO, "out")
C2 = os.path.join(OUT, "channel2")
TEMPLATE = os.path.join(HERE, "summary.html")
TARGET = os.path.join(OUT, "bet-2026-tag-reporting-rate-summary.html")


def img(path):
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def checks(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            p = line.rstrip("\n").split("\t")
            out[p[1]] = (p[0], p[2] if len(p) > 2 else "")
    return out


def main():
    html = open(TEMPLATE).read()

    ck1 = checks(os.path.join(OUT, "checks.tsv"))
    ck2 = checks(os.path.join(C2, "checks.log"))
    all_ck = {**ck1, **ck2}
    n_pass = sum(1 for v in all_ck.values() if v[0] == "PASS")

    rec = pd.read_csv(os.path.join(OUT, "penalty-reconstruction.csv"))
    fits = pd.read_csv(os.path.join(OUT, "rr-axis-marginal-effect.csv"))
    pen = fits[(fits.model == "arm + all axes incl M0")
               & (fits.response == "penalty")
               & (fits.subset == "all 88 retained models")].iloc[0]

    ds = pd.read_csv(os.path.join(C2, "downstream-arm-effects.csv"))
    ds = ds[ds.spec == "continuous axes"]
    f_row = ds[ds.column == "f_recent_fmsy"].iloc[0]
    d_row = ds[ds.column == "sb_recent_sb0"].iloc[0]

    prof = pd.read_csv(os.path.join(C2, "profile-objective.csv"))
    prof = prof[prof.ok == 1]
    n_obj = int(prof.groupby("member_id").objective.nunique().max())
    n_F = int(prof.groupby("member_id").agg_F_Fmsy.nunique().max())
    F_ptp = float(prof.groupby("member_id").agg_F_Fmsy.apply(lambda s: s.max() - s.min()).max())

    psum = pd.read_csv(os.path.join(C2, "profile-summary.csv"))
    g7 = psum[psum.group_id == 7].shift_argmin_objective

    disc = pd.read_csv(os.path.join(C2, "report-vs-likelihood-discrepancy.csv"))
    dm = disc[(disc.scope == "per member") & (disc.arm == "exclude")]

    # status by arm, straight from the management-quantities table
    dq = pd.read_csv(os.path.join(C2, "downstream-frame.csv"))
    mq = {int(re.search(r"(\d+)$", r["ensemble_id"]).group(1)): r
          for r in csv.DictReader(open(os.path.join(REPO, "data", "ensemble",
                                                    "management-quantities.csv")))}
    dq["above_fmsy"] = [mq[m]["above_fmsy"] == "TRUE" for m in dq.member_id]
    dq["below_lrp"] = [mq[m]["below_lrp_020"] == "TRUE" for m in dq.member_id]
    inc, exc = dq[dq.arm == "include"], dq[dq.arm == "exclude"]

    # attrition
    draws = pd.DataFrame(list(csv.DictReader(
        open(os.path.join(REPO, "design", "model-draws.csv")))))
    kept = {r["ensemble_id"] for r in csv.DictReader(
        open(os.path.join(REPO, "data", "ensemble", "successful-model-design.csv")))}
    draws["kept"] = draws.ensemble_id.isin(kept)
    att = draws.groupby("tag_reporting").kept.agg(["size", "sum"])

    floor = pd.read_csv(os.path.join(OUT, "floor-screen-by-member.csv"))
    fl_inc = floor[floor.arm == "include"]

    psi = pd.read_csv(os.path.join(C2, "psi-three-anchors-by-member.csv"))
    p_ext = psi[psi.scheme.str.startswith("3.")].psi_release_weighted.median()
    p_exc = psi[psi.scheme.str.startswith("1.")].psi_release_weighted.median()
    psi_k = pd.read_csv(os.path.join(C2, "psi-by-mixing-configuration.csv"))

    # floor isolation + prevalence (the sharpened-recommendation numbers)
    fi = pd.read_csv(os.path.join(C2, "floor-penalty-isolation.csv"))
    prof["resid"] = prof.objective - (prof.tag_mix + prof.tag_post + prof.rr_penalty)
    at_fit = []
    for (m, g), s2 in prof[prof.arm == "include"].groupby(["member_id", "group_id"]):
        s2 = s2.sort_values("X")
        r = (s2.resid - s2.resid.min()).values
        at_fit.append(r[int(np.argmin(np.abs(s2.X.values - s2.base_rate.iloc[0])))])
    sys.path.insert(0, os.path.join(REPO, "analysis", "rr-penalty"))
    from parpar import parse_all  # noqa: E402
    pars = parse_all(os.path.join(REPO, "final-par"))
    n_win = int(np.median([(p.tag_flags[:, 0] > 0).sum() for p in pars
                           if p.arm == "exclude"]))
    n_floor = float(fl_inc.n_engaging.median())

    subs = {
        "N_CHECKS": str(len(all_ck)),
        "N_PASS": str(n_pass),
        "N_FAIL": str(len(all_ck) - n_pass),
        "MAX_RESID": f"{rec.residual.abs().max():.1e}".replace("e-10", " × 10⁻¹⁰"),
        "PEN_EFFECT": f"{float(pen.arm_coef):.1f}",
        "PEN_LO": f"{float(pen.ci_lo):.1f}",
        "PEN_HI": f"{float(pen.ci_hi):.1f}",
        "F_EFFECT": f"{float(f_row.estimate):.3f}",
        "F_LO": f"{float(f_row.ci_lo):.3f}",
        "F_HI": f"{float(f_row.ci_hi):.3f}",
        "F_PCT": f"{100*float(f_row.estimate)/float(f_row.mean_outcome):.0f}",
        "D_EFFECT": f"{float(d_row.estimate):.4f}",
        "D_LO": f"{float(d_row.ci_lo):.4f}",
        "D_HI": f"{float(d_row.ci_hi):.4f}",
        "D_PCT": f"{100*float(d_row.estimate)/float(d_row.mean_outcome):.0f}",
        "N_OBJ_VALUES": str(n_obj),
        "N_F_VALUES": str(n_F),
        "F_PTP": f"{F_ptp:.1g}",
        "G7_SHIFT_LO": f"{g7.min():+.3f}",
        "G7_SHIFT_HI": f"{g7.max():+.3f}",
        "FLAT_HI": f"{psum.mix_relrange_highX_include.median()*100:.2f}",
        "FLAT_WORST": f"{psum.mix_relrange_include.max()*100:.1f}",
        "DISC_PCT": f"{dm.shortfall_frac.median()*100:.1f}",
        "DISC_FISH": f"{dm.shortfall.median():.0f}",
        "DISC_TOT": f"{dm.reported.median():.0f}",
        "INC_PFMSY": f"{inc.above_fmsy.mean()*100:.0f}",
        "EXC_PFMSY": f"{exc.above_fmsy.mean()*100:.0f}",
        "ALL_PFMSY": f"{dq.above_fmsy.mean()*100:.0f}",
        "INC_PLRP": f"{inc.below_lrp.mean()*100:.0f}",
        "EXC_PLRP": f"{exc.below_lrp.mean()*100:.0f}",
        "ALL_PLRP": f"{dq.below_lrp.mean()*100:.0f}",
        "INC_N": str(len(inc)),
        "EXC_N": str(len(exc)),
        "ATT_INC": f"{100*(att.loc['inclusion','size']-att.loc['inclusion','sum'])/att.loc['inclusion','size']:.0f}",
        "ATT_EXC": f"{100*(att.loc['exclusion','size']-att.loc['exclusion','sum'])/att.loc['exclusion','size']:.0f}",
        "ATT_INC_N": str(int(att.loc["inclusion", "size"] - att.loc["inclusion", "sum"])),
        "ATT_EXC_N": str(int(att.loc["exclusion", "size"] - att.loc["exclusion", "sum"])),
        "FLOOR_PCT": f"{(fl_inc.n_engaging > 0).mean()*100:.0f}",
        "FLOOR_MED": f"{fl_inc.n_engaging.median():.0f}",
        "PSI_EXT": f"{p_ext:.3f}",
        "PSI_EXC": f"{p_exc:.3f}",
        "PSI_PCT": f"{100*(p_ext-p_exc)/p_exc:.1f}",
        "PSI_KSPAN": f"{psi_k.psi_rw_median.max()-psi_k.psi_rw_median.min():.3f}",
        "PSI_FACTOR": f"{(psi_k.psi_rw_median.max()-psi_k.psi_rw_median.min())/abs(p_ext-p_exc):.0f}",
        "FLOOR_AT_FIT": f"{max(at_fit):.2f}",
        "FLOOR_AT_LOWX": f"{fi[fi.arm=='include'].resid_range.max():,.0f}",
        "FLOOR_EXC_RESID": f"{fi[fi.arm=='exclude'].resid_range.max():.1e}",
        "N_WIN": str(n_win),
        "N_FLOOR": f"{n_floor:.0f}",
        "PREVALENCE_RATIO": f"{n_win/n_floor:.0f}",
        "FIG_PROFILE": img(os.path.join(C2, "figures", "profile-by-arm-g7.png")),
        "FIG_DOWNSTREAM": img(os.path.join(C2, "figures", "downstream-arm-effects.png")),
        "FIG_DISSOC": img(os.path.join(C2, "figures", "excursion-vs-Nmix-by-arm.png")),
        "FIG_ARMGAP": img(os.path.join(OUT, "figures", "arm-gap-vs-K.png")),
    }
    for k, v in subs.items():
        html = html.replace("{{" + k + "}}", v)

    for name, (status, detail) in all_ck.items():
        tok = "{{CHECK:" + name + "}}"
        if tok in html:
            cls = "pass" if status == "PASS" else "hold"
            word = "held" if status == "PASS" else "did not hold"
            det = detail.replace("&", "&amp;").replace("<", "&lt;")
            html = html.replace(tok, f'<div class="check {cls}">'
                                     f'<span class="chip">{word}</span>'
                                     f'<code>{name}</code><p>{det}</p></div>')

    if "{{" in html:
        i = html.index("{{")
        raise SystemExit(f"unsubstituted placeholder near {html[i:i+60]!r}")

    with open(TARGET, "w") as fh:
        fh.write(html)
    print(f"wrote {os.path.relpath(TARGET, REPO)} ({os.path.getsize(TARGET)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
