"""Task 6: psi-hat under three raising schemes.

psi-hat inherits the reporting rates it is raised by. The published diagnostic
was computed with rates fitted under the configured `exclude` setting -- i.e.
rates subject to channel 2. Recompute under three anchors:

  1. exclude-arm fitted rates   (as published)
  2. include-arm fitted rates   (channel 2 inert, but removal feedback active
                                 and the survival floor engages in every member)
  3. tag-seeding prior means    (external, independent of the fit) -- LEAD WITH THIS

psi-hat for a release group is the raised in-window share:

    psi_i = sum_{in window} n_f / X_g(f)  /  sum_{all} n_f / X_g(f)

Mixing windows are member-specific, so each scheme is evaluated within each
member's own window configuration and the scheme contrast is within-member.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2common import (  # noqa: E402
    OUT,
    check_logger,
    load_members,
    penalty_groups,
    read_tag_file,
    write_checks,
)

CHECKS, check = check_logger()

PSI_HIGH = 0.3  # the threshold used in the published diagnostic


def group_rate_tables(members):
    """Median fitted rate per reporting-rate group, by arm, plus the prior mean."""
    recs = []
    for m in members:
        for g in penalty_groups(m["par"]):
            recs.append(dict(arm=m["par"].arm, group_id=g["group_id"],
                             rep_hat=g["rep_hat"], target=g["target_prop"]))
    d = pd.DataFrame(recs)
    exc = d[d.arm == "exclude"].groupby("group_id").rep_hat.median().to_dict()
    inc = d[d.arm == "include"].groupby("group_id").rep_hat.median().to_dict()
    pri = d.groupby("group_id").target.first().to_dict()
    return exc, inc, pri


def psi_for_member(par, tag, rates):
    """Release-group psi-hat under a rate map, plus release weights."""
    mix = par.tag_flags[:, 0]
    out = []
    for rel in tag["releases"]:
        it = rel["group"] - 1
        K = int(mix[it])
        num = den = 0.0
        for rec in rel["recoveries"]:
            g = int(par.rep_group[it, rec["fishery"] - 1])
            x = rates.get(g)
            if not x or x <= 0:
                continue
            raised = rec["number"] / x
            den += raised
            if K > 0 and 0 <= rec["elapsed"] < K:
                num += raised
        out.append(dict(release_group=rel["group"], n_released=rel["n_released"],
                        psi=(num / den) if den > 0 else 0.0,
                        raised_total=den))
    return pd.DataFrame(out)


def main():
    members = load_members()
    tag = read_tag_file()
    exc_rates, inc_rates, prior_rates = group_rate_tables(members)

    schemes = [
        ("3. tag-seeding prior means (external anchor)", prior_rates),
        ("1. exclude-arm fitted rates (as published)", exc_rates),
        ("2. include-arm fitted rates", inc_rates),
    ]
    print("median rate by scheme, priored groups:")
    for gid in sorted(prior_rates):
        if gid in (7, 10, 14, 17, 18):
            print(f"  g{gid:<3} prior {prior_rates[gid]:.4f}  "
                  f"exclude {exc_rates[gid]:.4f}  include {inc_rates[gid]:.4f}")

    rows = []
    for m in members:
        p = m["par"]
        for label, rates in schemes:
            t = psi_for_member(p, tag, rates)
            wmean = float((t.psi * t.n_released).sum() / t.n_released.sum())
            hi = t[t.psi >= PSI_HIGH]
            rows.append(dict(
                member_id=p.member_id, arm=p.arm,
                K=float(m["design"]["tag_mixing_k_cutoff"]),
                scheme=label,
                psi_release_weighted=wmean,
                psi_mean=float(t.psi.mean()),
                n_groups_psi_ge_0p3=int(len(hi)),
                release_share_psi_ge_0p3=float(hi.n_released.sum() / t.n_released.sum()),
            ))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "psi-three-anchors-by-member.csv"), index=False)

    summ = (d.groupby("scheme")
              .agg(n_members=("member_id", "size"),
                   psi_rw_median=("psi_release_weighted", "median"),
                   psi_rw_min=("psi_release_weighted", "min"),
                   psi_rw_max=("psi_release_weighted", "max"),
                   n_groups_ge_0p3_median=("n_groups_psi_ge_0p3", "median"),
                   release_share_ge_0p3_median=("release_share_psi_ge_0p3", "median"))
              .reset_index().sort_values("scheme"))
    print("\n--- psi-hat by raising scheme (across all 80 members) ---")
    print(summ.to_string(index=False))

    # exclude-arm members only, matching the published diagnostic's configuration
    de = d[d.arm == "exclude"]
    summ_e = (de.groupby("scheme")
                .agg(n_members=("member_id", "size"),
                     psi_rw_median=("psi_release_weighted", "median"),
                     n_groups_ge_0p3_median=("n_groups_psi_ge_0p3", "median"),
                     release_share_ge_0p3_median=("release_share_psi_ge_0p3", "median"))
                .reset_index().sort_values("scheme"))
    print("\n--- restricted to exclude-arm members (the configured setting) ---")
    print(summ_e.to_string(index=False))

    out_rows = []
    for _, r in summ.iterrows():
        e = summ_e[summ_e.scheme == r.scheme].iloc[0]
        out_rows.append(dict(
            scheme=r.scheme, n_members_all=int(r.n_members),
            psi_release_weighted_median_all=f"{r.psi_rw_median:.6g}",
            psi_release_weighted_range_all=f"{r.psi_rw_min:.4g}-{r.psi_rw_max:.4g}",
            n_groups_psi_ge_0p3_median_all=f"{r.n_groups_ge_0p3_median:.6g}",
            release_share_psi_ge_0p3_median_all=f"{r.release_share_ge_0p3_median:.6g}",
            n_members_exclude_arm=int(e.n_members),
            psi_release_weighted_median_exclude_arm=f"{e.psi_rw_median:.6g}",
            n_groups_psi_ge_0p3_median_exclude_arm=f"{e.n_groups_ge_0p3_median:.6g}",
            release_share_psi_ge_0p3_median_exclude_arm=f"{e.release_share_ge_0p3_median:.6g}",
        ))
    pd.DataFrame(out_rows).to_csv(os.path.join(OUT, "psi-three-anchors.csv"), index=False)

    # --- checks --------------------------------------------------------------
    def med(scheme_key, frame=d):
        s = frame[frame.scheme.str.startswith(scheme_key)]
        return float(s.psi_release_weighted.median())

    p_ext, p_exc, p_inc = med("3."), med("1."), med("2.")
    check(
        "task6.psi-higher-under-external-anchor-than-as-published",
        p_ext > p_exc,
        f"release-weighted psi-hat: as published (exclude rates) {p_exc:.4f} -> "
        f"external anchor (seeding priors) {p_ext:.4f} "
        f"({100*(p_ext-p_exc)/p_exc:+.1f}%)",
    )
    check(
        "task6.include-arm-rates-sit-between-the-other-two",
        min(p_exc, p_ext) <= p_inc <= max(p_exc, p_ext),
        f"include-arm rates give {p_inc:.4f}, between {p_exc:.4f} (exclude) and "
        f"{p_ext:.4f} (external)",
    )
    # within-member paired contrast, the clean version
    piv = d.pivot_table(index="member_id", columns="scheme",
                        values="psi_release_weighted")
    ext_c = [c for c in piv.columns if c.startswith("3.")][0]
    exc_c = [c for c in piv.columns if c.startswith("1.")][0]
    diff = piv[ext_c] - piv[exc_c]
    check(
        "task6.external-anchor-raises-psi-in-every-member",
        bool((diff > 0).all()),
        f"within-member difference (external - as published): median "
        f"{diff.median():+.4f}, min {diff.min():+.4f}, max {diff.max():+.4f}, "
        f"positive in {int((diff > 0).sum())}/{len(diff)} members",
    )

    # --- sensitivity to the mixing configuration, for context ----------------
    ext = d[d.scheme.str.startswith("3.")]
    byK = (ext.groupby("K")
              .agg(n_members=("member_id", "size"),
                   psi_rw_median=("psi_release_weighted", "median"),
                   n_groups_ge_0p3_median=("n_groups_psi_ge_0p3", "median"))
              .reset_index())
    print("\n--- external-anchor psi-hat by K (mixing configuration) ---")
    print(byK.to_string(index=False))
    byK.to_csv(os.path.join(OUT, "psi-by-mixing-configuration.csv"), index=False)

    span = byK.psi_rw_median.max() - byK.psi_rw_median.min()
    scheme_span = abs(p_ext - p_exc)
    check(
        "task6.mixing-configuration-dominates-the-raising-scheme",
        span > 10 * scheme_span,
        f"psi-hat moves {span:.4f} across the K axis (from "
        f"{byK.psi_rw_median.max():.4f} at K={byK.K.iloc[0]:g} to "
        f"{byK.psi_rw_median.min():.4f} at K={byK.K.iloc[-1]:g}) versus "
        f"{scheme_span:.4f} between raising schemes -- a factor of "
        f"{span/scheme_span:.0f}",
    )
    # Can the published baseline be reproduced at all?
    PUB_PSI, PUB_NGRP = 0.231, 27
    near = byK.iloc[(byK.psi_rw_median - PUB_PSI).abs().argmin()]
    check(
        "task6.published-baseline-reconciles",
        abs(near.psi_rw_median - PUB_PSI) < 0.02
        and abs(near.n_groups_ge_0p3_median - PUB_NGRP) <= 3,
        f"closest configuration is K={near.K:g}: psi-hat {near.psi_rw_median:.4f} "
        f"vs published {PUB_PSI} (close), but "
        f"{near.n_groups_ge_0p3_median:.0f} groups at psi>=0.3 vs published "
        f"{PUB_NGRP} (does not reconcile)",
    )

    write_checks(CHECKS, os.path.join(OUT, "checks-14.tsv"))


if __name__ == "__main__":
    main()
