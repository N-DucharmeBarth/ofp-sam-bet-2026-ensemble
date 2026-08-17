"""Reader for the MFCL .tag file (model/bet.tag).

Only what Task 7 needs: per release group the release region/date and the
number released, plus every recovery record (fishery, recapture date, number).

Period arithmetic. MFCL treats a recapture as inside the mixing window when

    ip < initial_tag_period(it, ir) + tag_flags(it, 1)

(multifan-cl src/tag3.cpp, the complement of the fitted-period test), and
`initial_tag_period` is the first fishing period at or after the release date
(src/readtag.cpp). This model runs on a quarterly grid -- every release and
recovery date in bet.tag falls in month 2, 5, 8 or 11 -- so elapsed periods are
counted as elapsed quarters:

    elapsed = 4 * (recap_year - rel_year) + (recap_month - rel_month) / 3

and the recapture is in-window iff `0 <= elapsed < mixing_K`. This equals
MFCL's period count whenever the recapture region has a fishing incident in
every quarter of the interval; where it does not, the quarterly count is an
upper bound on MFCL's. It is a descriptive proxy for psi-hat, which is all
Task 7 asks of it.
"""

from __future__ import annotations

import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
TAG_FILE = os.path.join(REPO, "model", "bet.tag")

QUARTER_MONTHS = (2, 5, 8, 11)


def _quarter(year: int, month: int) -> int:
    if month not in QUARTER_MONTHS:
        raise ValueError(f"month {month} is not on the quarterly grid {QUARTER_MONTHS}")
    return 4 * year + (month - 2) // 3


def read_tag_file(path: str = TAG_FILE):
    lines = open(path).read().split("\n")

    hdr = None
    for i, ln in enumerate(lines):
        if ln.startswith("#"):
            continue
        if ln.strip():
            hdr = [int(float(t)) for t in ln.split()]
            break
    n_groups = hdr[0]

    releases = []
    i = 0
    while i < len(lines):
        m = re.match(r"#\s+(\d+)\s+-\s+RELEASE REGION", lines[i])
        if not m:
            i += 1
            continue
        gid = int(m.group(1))
        j = i + 1
        while not lines[j].strip() or lines[j].startswith("#"):
            j += 1
        region, year, month = (int(float(t)) for t in lines[j].split()[:3])
        j += 1
        while not lines[j].strip() or lines[j].startswith("#"):
            j += 1
        n_released = sum(float(t) for t in lines[j].split())

        # recovery block
        recs = []
        k = j
        while k < len(lines) and "LENGTH RELEASE" not in lines[k]:
            if re.match(r"#\s+\d+\s+-\s+RELEASE REGION", lines[k]) and k > j:
                break
            k += 1
        if k < len(lines) and "LENGTH RELEASE" in lines[k]:
            k += 1
            while k < len(lines) and lines[k].strip() and not lines[k].startswith("#"):
                f = lines[k].split()
                recs.append(
                    dict(
                        length=float(f[0]),
                        fishery=int(float(f[1])),
                        year=int(float(f[2])),
                        month=int(float(f[3])),
                        number=float(f[4]),
                    )
                )
                k += 1

        rel_q = _quarter(year, month)
        for r in recs:
            r["elapsed"] = _quarter(r["year"], r["month"]) - rel_q

        releases.append(
            dict(
                group=gid,
                region=region,
                year=year,
                month=month,
                quarter=rel_q,
                n_released=n_released,
                recoveries=recs,
            )
        )
        i = k

    if len(releases) != n_groups:
        raise ValueError(f"{path}: header says {n_groups} release groups, parsed {len(releases)}")
    if [r["group"] for r in releases] != list(range(1, n_groups + 1)):
        raise ValueError(f"{path}: release groups are not 1..{n_groups} in order")
    return dict(n_groups=n_groups, releases=releases)


if __name__ == "__main__":
    t = read_tag_file()
    n_rec = sum(len(r["recoveries"]) for r in t["releases"])
    n_fish = sum(rr["number"] for r in t["releases"] for rr in r["recoveries"])
    print(f"{t['n_groups']} release groups, {n_rec} recovery records, {n_fish:.0f} fish recovered")
    print(f"total released: {sum(r['n_released'] for r in t['releases']):.0f}")
    el = [rr["elapsed"] for r in t["releases"] for rr in r["recoveries"]]
    print(f"elapsed quarters: min {min(el)}, max {max(el)}")
