"""Channel-2 figures.

Palette carried over from the penalty analysis so the two read as one system:
#2a78d6 include / #eb6834 exclude, validated all-pairs on the light surface.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2common import FIG, OUT  # noqa: E402

INCLUDE, EXCLUDE = "#2a78d6", "#eb6834"
ARMC = {"include": INCLUDE, "exclude": EXCLUDE}
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
SURFACE, GRID = "#fcfcfb", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "legend.frameon": False,
})


def style(ax, ygrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(length=3, width=0.8)
    if ygrid:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, lw=0.8)


def fig_profiles():
    path = os.path.join(OUT, "profile-objective.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    d = d[d.ok == 1]
    mf = pd.read_csv(os.path.join(OUT, "..", "member-frame.csv"))
    Kof = dict(zip(mf.member_id, mf.K))

    out = []
    for gid in sorted(d.group_id.unique()):
        out.append(_one_profile_fig(d, gid, Kof))
    return out


def _one_profile_fig(d, gid, Kof):
    sub = d[d.group_id == gid]
    members = sorted(sub.member_id.unique(), key=lambda m: Kof.get(m, 0))
    fig, axes = plt.subplots(2, len(members), figsize=(3.5 * len(members), 6.4),
                             sharex=True)
    axes = np.atleast_2d(axes)
    for j, mid in enumerate(members):
        s = sub[sub.member_id == mid]
        for row, (col, lab) in enumerate([
                ("tag_mix", "mixing-window tag likelihood"),
                ("objective", "total objective")]):
            ax = axes[row, j]
            for arm in ("include", "exclude"):
                t = s[s.arm == arm].sort_values("X")
                if not len(t):
                    continue
                y = t[col].values
                ax.plot(t.X, y, "-o", color=ARMC[arm], lw=2, ms=5,
                        mec=SURFACE, mew=1.2, label=arm, zorder=3)
                if col == "objective":
                    k = int(np.argmin(y))
                    ax.scatter([t.X.values[k]], [y[k]], s=90, facecolor="none",
                               ec=ARMC[arm], lw=2, zorder=5)
            if col == "objective":
                # Under `include` the survival floor blows the objective up at
                # low X, compressing the region where the argmin sits. Clip to
                # the informative band and mark that the axis is cut.
                allv = np.concatenate([s[s.arm == a].sort_values("X")[col].values
                                       for a in ("include", "exclude")])
                ymin = float(allv.min())
                d = float(np.percentile(allv, 55)) - ymin
                if d > 0 and allv.max() > ymin + 2.5 * d:
                    ax.set_ylim(ymin - 0.06 * d, ymin + 1.15 * d)
                    ax.annotate("axis cut", (0.03, 0.93), xycoords="axes fraction",
                                color=INK3, fontsize=7)
            style(ax)
            if row == 0:
                ax.set_title(f"member {mid}  ·  K = {Kof.get(mid, float('nan')):g}")
            if j == 0:
                ax.set_ylabel(lab)
            if row == 1:
                ax.set_xlabel(r"$\hat{X}_f$ for group " + str(gid))
    axes[0, 0].legend(loc="upper right", ncol=2, fontsize=8.5)
    fig.suptitle("Fixed-parameter profile against the reporting rate "
                 f"(group {gid}); rings mark each arm's argmin",
                 fontsize=11, fontweight="bold", y=1.0)
    fig.tight_layout()
    p = os.path.join(FIG, f"profile-by-arm-g{gid}.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_excursion_vs_nmix():
    """Left: the raw relationship. Right: the adjusted slopes that carry the claim.

    The marginal scatter cannot show the double dissociation -- that is a
    partial effect, holding the other count and the member fixed. Member fixed
    effects are nested within arm (each member sits in exactly one arm), so
    per-arm member-FE fits recover the interaction slopes exactly and give
    their CIs directly.
    """
    import statsmodels.formula.api as smf

    d = pd.read_csv(os.path.join(OUT, "group-excursion-predictors.csv"))
    pri = d[d.is_priored == 1]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), width_ratios=[1.15, 1])

    ax = axes[0]
    for arm in ("include", "exclude"):
        s = pri[pri.arm == arm]
        ax.scatter(s.N_mix, s.signed_dev, s=18, color=ARMC[arm], alpha=0.45,
                   lw=0, label=arm, zorder=3)
    style(ax)
    ax.set_xlabel("in-window recaptures  $N_{mix}$")
    ax.set_ylabel(r"$\hat{X}_f - \mu$   (signed deviation)")
    ax.axhline(0, color=INK3, lw=1)
    ax.set_title("Excursion against mixing-window volume")
    ax.legend(loc="lower right", ncol=2, fontsize=8.5)
    ax.annotate("raw; the claim is the adjusted slopes at right", (0.03, 0.94),
                xycoords="axes fraction", color=INK3, fontsize=7.5)

    ax = axes[1]
    rows = []
    for arm in ("include", "exclude"):
        m = smf.ols("signed_dev ~ N_mix + N_post + C(member_id)",
                    data=pri[pri.arm == arm]).fit()
        for term in ("N_mix", "N_post"):
            ci = m.conf_int().loc[term]
            rows.append((term, arm, m.params[term] * 1000,
                         ci[0] * 1000, ci[1] * 1000))
    r = pd.DataFrame(rows, columns=["term", "arm", "est", "lo", "hi"])
    ypos = {("N_mix", "include"): 3.2, ("N_mix", "exclude"): 2.8,
            ("N_post", "include"): 1.2, ("N_post", "exclude"): 0.8}
    for _, q in r.iterrows():
        y = ypos[(q.term, q.arm)]
        ax.hlines(y, q.lo, q.hi, color=ARMC[q.arm], lw=2.6, zorder=3)
        ax.scatter([q.est], [y], s=72, color=ARMC[q.arm], ec=SURFACE, lw=1.5, zorder=4)
    for (t, a), y in ypos.items():
        q = r[(r.term == t) & (r.arm == a)].iloc[0]
        ax.annotate(f"{q.est:.3f}", (q.est, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", color=INK2, fontsize=8)
    ax.set_yticks([3.0, 1.0])
    ax.set_yticklabels(["$N_{mix}$\n(mixing window)", "$N_{post}$\n(post-mixing)"],
                       fontsize=9)
    ax.set_ylim(0.2, 3.8)
    ax.axvline(0, color=INK3, lw=1)
    style(ax, ygrid=False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlabel(r"slope on $\hat{X}_f-\mu$ per 1000 recaptures"
                  "\n(member fixed effects, both counts in the model)")
    ax.set_title("Double dissociation")
    fig.tight_layout()
    p = os.path.join(FIG, "excursion-vs-Nmix-by-arm.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def fig_downstream():
    d = pd.read_csv(os.path.join(OUT, "downstream-arm-effects.csv"))
    d = d[d.spec == "continuous axes"].copy()
    for c in ("estimate", "ci_lo", "ci_hi", "mean_outcome"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    # express on a common scale: percent of the outcome's mean
    d["pct"] = 100 * d.estimate / d.mean_outcome
    d["pct_lo"] = 100 * d.ci_lo / d.mean_outcome
    d["pct_hi"] = 100 * d.ci_hi / d.mean_outcome
    d = d.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    y = np.arange(len(d))
    cols = [EXCLUDE if v > 0 else INCLUDE for v in d.pct]
    ax.hlines(y, d.pct_lo, d.pct_hi, color=cols, lw=2.4, zorder=3)
    ax.scatter(d.pct, y, s=64, color=cols, ec=SURFACE, lw=1.5, zorder=4)
    ax.axvline(0, color=INK3, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d.outcome, fontsize=8.5)
    style(ax, ygrid=False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlabel("adjusted arm effect (exclude - include), % of the outcome mean")
    ax.set_title("Downstream propagation to F and stock status")
    for yi, r in zip(y, d.itertuples()):
        ax.annotate(f"{r.pct:+.0f}%", (r.pct, yi), textcoords="offset points",
                    xytext=(0, 9), ha="center", color=INK2, fontsize=8)
    h = [plt.Line2D([], [], marker="o", ls="", color=ARMC[a], ms=7,
                    label=f"{a} higher") for a in ("exclude", "include")]
    ax.legend(handles=h, loc="lower right", fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIG, "downstream-arm-effects.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


COMPONENT_STYLE = {
    "rr_penalty": ("#c0392b", "reporting-rate penalty (this group)"),
    "tag_mix": ("#e67e22", "tag likelihood, mixing window"),
    "tag_post": ("#f1c40f", "tag likelihood, post-mixing"),
    "length_total": ("#2c7fb8", "length-frequency data (all fisheries)"),
    "age_length_total": ("#1a9850", "age-length data"),
    "survey_index_total": ("#6a3d9a", "survey/CPUE index"),
}
TOTAL_STYLE = ("#111111", "total objective")
RESID_STYLE = ("#7a7a7a", "inferred floor-cap residual (objective − Σ tag terms)")


def fig_component_profile():
    """Task 12: every likelihood component on one axis, per (group, arm).

    y is each component's own delta-NLL: its value at each X minus its OWN
    minimum over the sweep, so every curve's floor is exactly 0 -- a proper
    profile-likelihood presentation, not an arbitrary offset from the first
    grid point. A hundred-thousand-unit length-frequency total and a
    two-hundred-unit reporting-rate penalty still share one meaningful axis.
    The tag terms trace out a real profile; the non-tag components are
    constant across the sweep (component-profile-sweep.csv: exactly 0.0
    objective units of range, not just approximately), so their delta-NLL
    is identically 0 everywhere -- flat at the axis floor because they have
    no minimum being profiled, not because of the normalisation choice.

    The black line is the total objective, same delta-from-own-minimum
    treatment. Under `exclude` it equals the sum of the three tag terms to
    within floating-point noise (residual range ~1e-6), confirming again
    that exclude never touches the survival floor. Under `include` it runs
    above that sum by the floor-cap residual from Part 3/Task 10 -- and
    that residual is wildly different in scale between the two groups
    shown: ~3 objective units for group 18 (matching the "costs at most a
    few units" finding already in the document) but ~31,000 units for
    group 7 at the low end of this sweep (X=0.30, well below its own
    fitted rate of ~0.52). The cap is not a mild, uniform tax -- it is a
    steep, group-specific cliff, and this sweep happens to run far enough
    left to fall off it for group 7. Both groups' own fitted rates sit
    safely past where the residual plateaus, which is why the fitted-rate
    cost stays small in practice; this is what the cost would look like if
    a fit were pulled further down that cliff than these ones were.

    The grey dotted line makes that residual explicit rather than leaving
    it to be read off as the gap between the black and coloured lines: it
    is objective minus (rr_penalty + tag_mix + tag_post), same
    delta-from-own-minimum treatment. "Inferred" because it is not a
    labelled block in test_plot_output -- it is arithmetic, the same
    residual-by-subtraction construction Task 10 used to isolate the
    survival-floor penalty from a fixed-parameter profile in the first
    place. Under exclude it should sit at the axis floor throughout
    (nothing left over to explain); under include it traces the cap curve
    directly.
    """
    path = os.path.join(OUT, "component-profile-sweep.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    d = d[d.ok == 1]
    member = int(d.member_id.min())  # one member throughout: isolates (X, arm, group)
    d = d[d.member_id == member].sort_values("X")

    groups = sorted(d.group_id.unique())
    arms = ["include", "exclude"]
    fig, axes = plt.subplots(len(groups), len(arms), figsize=(10.5, 4.6 * len(groups)),
                              sharex=True, squeeze=False)
    for gi, g in enumerate(groups):
        for ai, arm in enumerate(arms):
            ax = axes[gi][ai]
            sub = d[(d.group_id == g) & (d.arm == arm)]
            for comp, (color, label) in COMPONENT_STYLE.items():
                y = sub[comp].values - sub[comp].values.min()
                ax.plot(sub.X, y, "-o", color=color, lw=1.8, ms=3.5, label=label, zorder=3)
            tot_color, tot_label = TOTAL_STYLE
            y_tot = sub.objective.values - sub.objective.values.min()
            ax.plot(sub.X, y_tot, marker="o", color=tot_color, lw=2.4, ms=4,
                    ls="--", label=tot_label, zorder=4)
            resid_color, resid_label = RESID_STYLE
            resid = (sub.objective.values
                      - (sub.rr_penalty.values + sub.tag_mix.values + sub.tag_post.values))
            y_resid = resid - resid.min()
            ax.plot(sub.X, y_resid, marker="s", color=resid_color, lw=1.4, ms=3,
                    ls=":", label=resid_label, zorder=2)
            ax.axhline(0, color=GRID, lw=1, zorder=1)
            # symlog, not log10(y + epsilon): every delta here is >= 0 and
            # many are exactly 0 (the flat components), so a plain log needs
            # an arbitrary additive constant to avoid log(0) and that
            # constant distorts the small values. symlog instead has a
            # genuine linear region below linthresh -- exact zeros sit
            # exactly at 0, not at some epsilon-dependent offset -- and logs
            # everything above it, which is what the 6-order-of-magnitude
            # range here (0 to ~31000) actually needs.
            ax.set_yscale("symlog", linthresh=1, linscale=0.6)
            ax.set_ylim(bottom=0)
            style(ax)
            ax.set_title(f"group {g}, {arm}", fontsize=9.5,
                         color=ARMC[arm])
            if ai == 0:
                ax.set_ylabel("Δ from this component's own minimum (symlog)")
            if gi == len(groups) - 1:
                ax.set_xlabel("X (reporting rate assumed for this group)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8.3,
               bbox_to_anchor=(0.5, 1.05 if len(groups) == 1 else 1.03))
    fig.suptitle(f"Only the tag terms respond to X-hat (member {member}, fixed parameters)",
                 fontsize=11, fontweight="bold", y=1.12 if len(groups) == 1 else 1.08)
    fig.tight_layout()
    p = os.path.join(FIG, "component-profile.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    os.makedirs(FIG, exist_ok=True)
    for f in (fig_excursion_vs_nmix, fig_downstream, fig_profiles, fig_component_profile):
        r = f()
        for p in (r if isinstance(r, list) else [r]):
            if p:
                print("wrote", os.path.relpath(p, os.path.dirname(os.path.dirname(OUT))))


if __name__ == "__main__":
    main()
