"""Figures: arm gap vs K, excursion by group and arm, and the balance table.

Palette: categorical slots 1 and 2 of the reference data-viz palette
(#2a78d6 blue = include, #eb6834 orange = exclude). Validated all-pairs on the
light surface: CVD dE 24.7, normal-vision dE 33.6, contrast >= 3:1, all PASS.
Identity is never colour-alone -- every panel carries a legend and the arms are
also separated by position.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, REPO  # noqa: E402

FIG = os.path.join(OUT, "figures")
INCLUDE, EXCLUDE = "#2a78d6", "#eb6834"
ARMC = {"include": INCLUDE, "exclude": EXCLUDE}
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "legend.frameon": False,
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


def fig_arm_gap_vs_K(mix):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), width_ratios=[1.1, 1])

    ax = axes[0]
    for arm in ("include", "exclude"):
        s = mix[mix.arm == arm].groupby("K").penalty.median()
        ax.plot(s.index, s.values, "-o", color=ARMC[arm], lw=2, ms=8,
                mec=SURFACE, mew=1.6, label=arm, zorder=3)
        ax.annotate(arm, (s.index[-1], s.values[-1]), textcoords="offset points",
                    xytext=(9, 0), va="center", color=INK2, fontsize=8.5)
    style(ax)
    ax.set_xlabel("K (tag mixing cutoff)")
    ax.set_ylabel("median reporting-rate penalty")
    ax.set_title("Penalty by mixing cutoff and arm")
    ax.set_xlim(0.03, 0.405)
    ax.legend(loc="lower left", ncol=2, fontsize=8.5)

    ax = axes[1]
    gap = (mix[mix.arm == "exclude"].groupby("K").penalty.median()
           - mix[mix.arm == "include"].groupby("K").penalty.median())
    n = mix.groupby("K").size()
    cols = [EXCLUDE if v > 0 else INCLUDE for v in gap.values]
    ax.bar(gap.index, gap.values, width=0.032, color=cols, zorder=3,
           edgecolor=SURFACE, linewidth=2)
    lo, hi = gap.min(), gap.max()
    ax.set_ylim(lo - 0.30 * (hi - lo), hi + 0.16 * (hi - lo))
    ax.axhline(0, color=INK3, lw=1)
    for k, v in gap.items():
        ax.annotate(f"{v:+.0f}", (k, v), textcoords="offset points",
                    xytext=(0, 4 if v > 0 else -11), ha="center",
                    color=INK2, fontsize=8)
    for k in gap.index:
        ax.annotate(f"n={n[k]}", (k, ax.get_ylim()[0]), textcoords="offset points",
                    xytext=(0, 5), ha="center", color=INK3, fontsize=7)
    style(ax)
    ax.set_xlabel("K (tag mixing cutoff)")
    ax.set_ylabel("median(exclude) - median(include)")
    ax.set_title("Arm gap by mixing cutoff")
    ax.set_xlim(0.03, 0.37)
    h = [plt.Line2D([], [], marker="s", ls="", color=ARMC[a], ms=7,
                    label=f"{a} larger") for a in ("exclude", "include")]
    # upper right: the only quadrant free of bars and of the n= labels
    ax.legend(handles=h, loc="upper right", ncol=1, fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIG, "arm-gap-vs-K.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def fig_excursion(exc):
    pri = exc[exc.penalty_wt > 1]
    groups = sorted(pri.group_id.unique())
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), width_ratios=[1.25, 1])

    ax = axes[0]
    rng = np.random.default_rng(11)
    for i, g in enumerate(groups):
        for k, arm in enumerate(("include", "exclude")):
            s = pri[(pri.group_id == g) & (pri.arm == arm)]
            y = i + (-0.19 if arm == "include" else 0.19)
            ax.scatter(s.abs_dev, y + rng.uniform(-0.075, 0.075, len(s)),
                       s=26, color=ARMC[arm], alpha=0.5, lw=0, zorder=3)
            m = s.abs_dev.median()
            ax.plot([m, m], [y - 0.14, y + 0.14], color=ARMC[arm], lw=2.5,
                    solid_capstyle="round", zorder=4)
            ax.scatter([m], [y], s=46, color=ARMC[arm], ec=SURFACE, lw=1.6, zorder=5)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f"group {g}\nw={pri[pri.group_id==g].penalty_wt.iloc[0]:g}"
                        for g in groups], fontsize=8)
    ax.invert_yaxis()
    style(ax, ygrid=False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlabel(r"$|\hat{X}_f - $ target$|$  (proportion)")
    ax.set_title("Absolute excursion from the prior, by priored group")
    h = [plt.Line2D([], [], marker="o", ls="", color=ARMC[a], ms=7, label=a)
         for a in ("include", "exclude")]
    ax.legend(handles=h, loc="upper right", ncol=2, fontsize=8.5)

    ax = axes[1]
    w = 0.36
    x = np.arange(len(groups))
    for k, arm in enumerate(("include", "exclude")):
        v = [pri[(pri.group_id == g) & (pri.arm == arm)].contribution.median() for g in groups]
        ax.bar(x + (k - 0.5) * w, v, w * 0.92, color=ARMC[arm], label=arm,
               zorder=3, edgecolor=SURFACE, linewidth=2)
        # selective labels only -- the sub-unit bars would collide and say nothing
        for xi, vi in zip(x + (k - 0.5) * w, v):
            if vi >= 1.0:
                ax.annotate(f"{vi:.3g}", (xi, vi), textcoords="offset points",
                            xytext=(0, 3), ha="center", color=INK2, fontsize=7.5)
    ax.annotate("g7 include and both g14 bars are < 1", (0.02, 0.86),
                xycoords="axes fraction", color=INK3, fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"g{g}" for g in groups])
    style(ax)
    ax.set_xlabel("reporting-rate group")
    ax.set_ylabel("median penalty contribution")
    ax.set_title("Contribution to the penalty")
    ax.legend(loc="upper left", ncol=2, fontsize=8.5)
    fig.tight_layout()
    p = os.path.join(FIG, "excursion-by-group-and-arm.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def fig_balance(df):
    axes_lvl = [("K", "K (mixing cutoff)"), ("tau", "tau (tag overdispersion)"),
                ("creep", "effort creep")]
    fig, axs = plt.subplots(1, 4, figsize=(11.6, 3.5))
    for ax, (col, title) in zip(axs, axes_lvl):
        lvls = sorted(df[col].unique())
        x = np.arange(len(lvls))
        w = 0.38
        for k, arm in enumerate(("include", "exclude")):
            n = [int(((df[col] == l) & (df.arm == arm)).sum()) for l in lvls]
            ax.bar(x + (k - 0.5) * w, n, w * 0.9, color=ARMC[arm], label=arm,
                   zorder=3, edgecolor=SURFACE, linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{l:g}" for l in lvls], fontsize=7.5)
        style(ax)
        ax.set_title(title, fontsize=9.5)
        ax.set_ylabel("models" if col == "K" else "")
    ax = axs[3]
    for k, arm in enumerate(("include", "exclude")):
        s = df[df.arm == arm].M0
        ax.scatter(np.full(len(s), k) + np.random.default_rng(3).uniform(-.13, .13, len(s)),
                   s, s=24, color=ARMC[arm], alpha=0.55, lw=0, zorder=3)
        ax.plot([k - 0.25, k + 0.25], [s.mean()] * 2, color=ARMC[arm], lw=2.5, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["include", "exclude"], fontsize=8.5)
    ax.set_xlim(-0.55, 1.55)
    style(ax)
    ax.set_title("M0 (quarterly M at age 40)", fontsize=9.5)
    h = [plt.Line2D([], [], marker="s", ls="", color=ARMC[a], ms=8, label=a)
         for a in ("include", "exclude")]
    fig.legend(handles=h, loc="upper center", ncol=2, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Balance of the other grid axes across the reporting-rate arms",
                 fontsize=10.5, fontweight="bold", y=1.09)
    fig.tight_layout()
    p = os.path.join(FIG, "balance.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p


OBJ_COMPONENTS = [
    ("Tagged Fish Reporting Rate Penalty Contribution", "Tagged-fish reporting-rate\npenalty"),
    ("Regional Recruitment Deviates Penalty Contribution", "Regional recruitment\ndeviates penalty"),
    ("Unclassified objective residual", "Unclassified\nobjective residual"),
]


def _violin(ax, data_by_arm, x0, width, color):
    parts = ax.violinplot([data_by_arm], positions=[x0], widths=width,
                          showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(0.45)
        pc.set_edgecolor(color)
        pc.set_linewidth(1.3)
    med = np.median(data_by_arm)
    q1, q3 = np.percentile(data_by_arm, [25, 75])
    ax.plot([x0, x0], [q1, q3], color=color, lw=3, solid_capstyle="round", zorder=4)
    ax.scatter([x0], [med], s=46, color=color, ec=SURFACE, lw=1.6, zorder=5)
    return med


def fig_objective_violin(obj, design):
    """The original motivating diagnostic: does the RR axis move the penalty,
    and is that localised to the reporting-rate prior or smeared across the
    fit? Restricted to the 80 converged members (has_par == True), matching
    every adjusted estimate elsewhere in the analysis.
    """
    d = obj.merge(design[["member_id", "arm", "has_par"]], on="member_id")
    d = d[d.has_par]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.3), sharex=False)
    for ax, (comp, label) in zip(axes, OBJ_COMPONENTS):
        s = d[d.Component == comp]
        meds = {}
        for k, arm in enumerate(("include", "exclude")):
            v = s[s.arm == arm].Value.dropna().values
            meds[arm] = _violin(ax, v, k, 0.72, ARMC[arm])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["include", "exclude"], fontsize=9)
        ax.set_xlim(-0.6, 1.6)
        style(ax)
        ax.set_title(label, fontsize=9.5)
        gap = meds["exclude"] - meds["include"]
        ax.annotate(f"median gap {gap:+.0f}", (0.5, 0.98), xycoords="axes fraction",
                    ha="center", va="top", fontsize=8, color=INK2)
    axes[0].set_ylabel("objective units")
    h = [plt.Line2D([], [], marker="s", ls="", color=ARMC[a], ms=8, label=a)
         for a in ("include", "exclude")]
    fig.legend(handles=h, loc="upper center", ncol=2, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("The RR axis moves the reporting-rate penalty, not the rest of the objective",
                 fontsize=11, fontweight="bold", y=1.14)
    fig.tight_layout()
    p = os.path.join(FIG, "objective-components-violin.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    os.makedirs(FIG, exist_ok=True)
    mix = pd.read_csv(os.path.join(OUT, "mixing-frame.csv"))
    exc = pd.read_csv(os.path.join(OUT, "rr-excursion-long.csv"))
    df = pd.read_csv(os.path.join(OUT, "member-frame.csv"))
    obj = pd.read_csv(os.path.join(REPO, "data", "ensemble", "objective-components.csv"))
    obj["member_id"] = obj.ensemble_id.str.extract(r"(\d+)$").astype(int)
    for p in (fig_arm_gap_vs_K(mix), fig_excursion(exc), fig_balance(df),
             fig_objective_violin(obj, df)):
        print("wrote", os.path.relpath(p, os.path.dirname(OUT)))


if __name__ == "__main__":
    main()
