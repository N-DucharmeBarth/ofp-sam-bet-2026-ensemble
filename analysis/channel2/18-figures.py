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
                 f"(group {gid}) — rings mark each arm's argmin",
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


def main():
    os.makedirs(FIG, exist_ok=True)
    for f in (fig_excursion_vs_nmix, fig_downstream, fig_profiles):
        r = f()
        for p in (r if isinstance(r, list) else [r]):
            if p:
                print("wrote", os.path.relpath(p, os.path.dirname(os.path.dirname(OUT))))


if __name__ == "__main__":
    main()
