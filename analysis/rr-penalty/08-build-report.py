"""Render out/FINDINGS.md's content as a self-contained HTML report.

Figures are inlined as data URIs -- the Artifact CSP blocks every external
host, so nothing may be linked. Numbers are pulled from the out/ CSVs rather
than retyped, so the page cannot drift from the analysis.
"""

import base64
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT  # noqa: E402

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.html")
TARGET = os.path.join(OUT, "reporting-rate-penalty-report.html")


def img(name):
    with open(os.path.join(OUT, "figures", name), "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def checks():
    rows = {}
    with open(os.path.join(OUT, "checks.tsv")) as fh:
        for line in fh:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            rows[parts[1]] = (parts[0], parts[2] if len(parts) > 2 else "")
    return rows


def chip(name, ck):
    status, detail = ck[name]
    cls = "pass" if status == "PASS" else "hold"
    word = "held" if status == "PASS" else "did not hold"
    detail = detail.replace("&", "&amp;").replace("<", "&lt;")
    return (
        f'<div class="check {cls}"><span class="chip">{word}</span>'
        f'<code>{name}</code><p>{detail}</p></div>'
    )


def main():
    ck = checks()
    html = open(TEMPLATE).read()

    n_pass = sum(1 for v in ck.values() if v[0] == "PASS")
    n_fail = len(ck) - n_pass

    rec = pd.read_csv(os.path.join(OUT, "penalty-reconstruction.csv"))
    mem = pd.read_csv(os.path.join(OUT, "member-frame.csv"))
    fits = pd.read_csv(os.path.join(OUT, "rr-axis-marginal-effect.csv"))
    full = fits[(fits.model == "arm + all axes incl M0")
                & (fits.response == "penalty")
                & (fits.subset == "all 88 retained models")].iloc[0]

    subs = {
        "FIG_ARM_GAP": img("arm-gap-vs-K.png"),
        "FIG_EXCURSION": img("excursion-by-group-and-arm.png"),
        "FIG_BALANCE": img("balance.png"),
        "N_PASS": str(n_pass),
        "N_FAIL": str(n_fail),
        "N_CHECKS": str(len(ck)),
        "MAX_RESID": f"{rec.residual.abs().max():.1e}".replace("e-10", " × 10⁻¹⁰"),
        "ARM_EFFECT": f"{float(full.arm_coef):.1f}",
        "ARM_LO": f"{float(full.ci_lo):.1f}",
        "ARM_HI": f"{float(full.ci_hi):.1f}",
        "MED_INC": f"{mem[mem.arm=='include'].penalty.median():.1f}",
        "MED_EXC": f"{mem[mem.arm=='exclude'].penalty.median():.1f}",
        "N_INC": str(int((mem.arm == "include").sum())),
        "N_EXC": str(int((mem.arm == "exclude").sum())),
    }
    for k, v in subs.items():
        html = html.replace("{{" + k + "}}", v)

    for name in ck:
        html = html.replace("{{CHECK:" + name + "}}", chip(name, ck))

    if "{{" in html:
        i = html.index("{{")
        raise SystemExit(f"unsubstituted placeholder near: {html[i:i+60]!r}")

    with open(TARGET, "w") as fh:
        fh.write(html)
    print(f"wrote {os.path.relpath(TARGET, os.path.dirname(OUT))} "
          f"({os.path.getsize(TARGET)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
