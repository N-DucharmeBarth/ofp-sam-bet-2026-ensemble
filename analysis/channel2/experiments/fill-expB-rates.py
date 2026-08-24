"""Fill Experiment B's expB-C-* rows with the fitted X-hat from the matching
Experiment A include-arm run, for the five priored groups.

Run C for a (tau, K) cell is the exclude-arm run with those groups fixed at
whatever the include-arm run at the SAME cell actually estimated -- not the
tag-seeding prior mean (that is the separate median-*-fixedRR run).
"""

import csv
import os
import sys

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "analysis", "rr-penalty"))
from parpar import parse_par  # noqa: E402

DESIGN = os.path.join(REPO, "analysis", "channel2", "experiments", "experiment-design.csv")
RUNS = os.path.join(REPO, "runs", "experiments")
PRIORED = [7, 10, 14, 17, 18]


def fitted_rates(final_par):
    p = parse_par(final_par)
    out = {}
    for g in PRIORED:
        mask = p.rep_group == g
        vals = set(p.rep[mask].round(10).tolist())
        if len(vals) != 1:
            raise SystemExit(f"group {g} in {final_par} is not uniform: {vals}")
        out[g] = vals.pop()
    return out


def main():
    with open(DESIGN, newline="") as fh:
        rows = list(csv.DictReader(fh))
    fieldnames = list(rows[0].keys())

    for row in rows:
        if not row["ensemble_id"].startswith("expB-C-"):
            continue
        suffix = row["ensemble_id"][len("expB-C-"):]  # e.g. tau1.2-K0.1
        source_id = f"expA-inc-{suffix}"
        final_par = os.path.join(RUNS, source_id, "final.par")
        if not os.path.exists(final_par):
            raise SystemExit(f"missing {final_par} -- run {source_id} first")
        rates = fitted_rates(final_par)
        row["fixed_rep_groups"] = ";".join(str(g) for g in PRIORED)
        row["fixed_rep_values"] = ";".join(repr(rates[g]) for g in PRIORED)
        print(f"{row['ensemble_id']}: {rates}")

    with open(DESIGN, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"updated {DESIGN}")


if __name__ == "__main__":
    main()
