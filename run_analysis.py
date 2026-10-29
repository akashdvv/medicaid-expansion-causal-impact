"""
Run the full causal analysis and write results/ and images/.

    python run_analysis.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analysis as A  # noqa: E402
import synth as S  # noqa: E402
from states import STATES  # noqa: E402

RESULTS, IMAGES = ROOT / "results", ROOT / "images"
BLUE, ORANGE, AQUA, INK, INK_2, GRID, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df", "#b8b7b2"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8,
})
pp = lambda x: 100 * x  # share -> percentage points


def chart_trends(d):
    t = A.weighted_trends(d) * 100
    fig, ax = plt.subplots(figsize=(9, 4))
    for col, color in [("Expanded 2014", BLUE), ("Not expanded by 2019", ORANGE)]:
        ax.plot(t.index, t[col], color=color, linewidth=2, marker="o", markersize=6, label=col)
        ax.annotate(f"{t[col].iloc[-1]:.0f}%", (t.index[-1], t[col].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK_2)
    ax.axvline(2013.5, color=INK_2, linestyle="--", linewidth=0.9)
    ax.text(2013.6, ax.get_ylim()[0] + 1, "ACA expansion\n(Jan 2014)", fontsize=9, color=INK_2)
    ax.set_ylabel("% insured (low-income childless adults)")
    ax.set_title("Insurance coverage diverged after 2014 in expansion states")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(2007.5, 2019.9)
    fig.tight_layout()
    fig.savefig(IMAGES / "coverage_trends.png", dpi=150)
    plt.close(fig)


def chart_event_study(es: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))
    pre, post = es[es.event_time < 0], es[es.event_time >= 0]
    for part, color in [(pre, MUTED), (post, BLUE)]:
        ax.vlines(part.event_time, pp(part.band_low), pp(part.band_high), color=color, linewidth=2)
        ax.plot(part.event_time, pp(part.estimate), "o", color=color, markersize=8, markeredgecolor="white", markeredgewidth=2)
    ax.axhline(0, color=INK_2, linewidth=0.9)
    ax.axvline(-0.5, color=INK_2, linestyle="--", linewidth=0.9)
    last = post.iloc[-1]
    ax.annotate(f"+{pp(last.estimate):.1f} pp", (last.event_time, pp(last.estimate)), xytext=(10, 0),
                textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_xticks(es.event_time)
    ax.set_xlabel("Years relative to expansion")
    ax.set_ylabel("Effect on % insured (pp)")
    ax.set_title("Event study (Callaway-Sant'Anna): flat before, rising after expansion")
    fig.tight_layout()
    fig.savefig(IMAGES / "event_study.png", dpi=150)
    plt.close(fig)


def chart_estimators(rows: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    y = np.arange(len(rows))[::-1]
    ax.hlines(y, pp(rows.ci_low), pp(rows.ci_high), color=BLUE, linewidth=2)
    ax.plot(pp(rows.estimate), y, "o", color=BLUE, markersize=9, markeredgecolor="white", markeredgewidth=2)
    for yi, (_, r) in zip(y, rows.iterrows()):
        ax.text(pp(r.ci_high) + 0.2, yi, f"{pp(r.estimate):.1f} pp", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(y, rows.method)
    ax.set_xlim(0, max(pp(rows.ci_high)) + 2)
    ax.set_xlabel("Effect on % insured (percentage points, 95% CI)")
    ax.set_title("Four estimators, one answer")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(IMAGES / "estimator_comparison.png", dpi=150)
    plt.close(fig)


def chart_sensitivity(sens: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, (e, g) in zip(axes, sens.groupby("event_time")):
        g = g.reset_index(drop=True)
        labels = ["Exact PT"] + [f"{float(m):g}" for m in g.Mbar[1:]]
        x = np.arange(len(g))
        colors = [MUTED] + [BLUE if lo > 0 else ORANGE for lo in g.ci_low[1:]]
        for xi, lo, hi, c in zip(x, pp(g.ci_low), pp(g.ci_high), colors):
            ax.vlines(xi, lo, hi, color=c, linewidth=3)
        ax.axhline(0, color=INK_2, linewidth=0.9)
        ax.set_xticks(x, labels)
        ax.set_title(f"Year {int(e)} after expansion", fontsize=11)
        ax.set_xlabel("M-bar (allowed violation ÷ largest pre-trend)")
    axes[0].set_ylabel("Robust 95% CI (pp)")
    fig.suptitle("HonestDiD: how wrong could parallel trends be before the effect disappears?",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(IMAGES / "honestdid_sensitivity.png", dpi=150)
    plt.close(fig)


def chart_synth(r, placebo: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.plot(r["years"], pp(r["actual"]), color=BLUE, linewidth=2.2, marker="o", markersize=6, label="Louisiana (actual)")
    ax.plot(r["years"], pp(r["synthetic"]), color=ORANGE, linewidth=2, linestyle="--", label="Synthetic Louisiana")
    ax.axvline(2016.5, color=INK_2, linestyle="--", linewidth=0.9)
    ax.text(2016.6, ax.get_ylim()[0] + 1, "Expansion\n(July 2016)", fontsize=9, color=INK_2)
    ax.set_ylabel("% insured")
    ax.set_title("Louisiana vs. its synthetic control")
    ax.legend(frameon=False, loc="upper left")
    ax = axes[1]
    for _, row in placebo.iterrows():
        if not row.is_treated:
            ax.plot(r["years"], pp(row.gap), color=MUTED, linewidth=1)
    ax.plot(r["years"], pp(r["gap"]), color=BLUE, linewidth=2.5, label="Louisiana")
    ax.plot([], [], color=MUTED, label="Placebo (never-expanding states)")
    ax.axhline(0, color=INK_2, linewidth=0.9)
    ax.axvline(2016.5, color=INK_2, linestyle="--", linewidth=0.9)
    ax.set_ylabel("Actual − synthetic (pp)")
    ax.set_title("Gap vs. 16 placebo states")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(IMAGES / "synthetic_control_louisiana.png", dpi=150)
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)
    d = A.load()

    two = A.two_by_two(d)
    tw = A.twfe(d)
    _, s_never, dyn_full = A.callaway_santanna(d, "nevertreated")
    _, s_notyet, _ = A.callaway_santanna(d, "notyettreated")
    cs_never, cs_notyet = A.agg_summary(s_never), A.agg_summary(s_notyet)
    es = A.event_study_table(A.event_study(d, base_period="universal"))
    sens = A.honest_sensitivity(d)
    breakdown = A.breakdown_values(sens)

    donors = sorted(d[d.g == 0].stfips.unique())
    LA = 22
    r = S.synth(d, LA, donors, 2017)
    placebo = S.placebo_test(d, LA, donors, 2017)
    la_rank = int(placebo.loc[placebo.is_treated, "rank"].iloc[0])

    rows = pd.DataFrame([
        {"method": "2×2 DiD (2014 vs never)", "estimate": two["did"], "ci_low": np.nan, "ci_high": np.nan},
        {"method": "Two-way fixed effects", **{k: tw[k] for k in ("estimate", "ci_low", "ci_high")}},
        {"method": "Callaway-Sant'Anna (never-treated)", **{k: cs_never[k] for k in ("estimate", "ci_low", "ci_high")}},
        {"method": "Callaway-Sant'Anna (not-yet-treated)", **{k: cs_notyet[k] for k in ("estimate", "ci_low", "ci_high")}},
    ])
    rows.loc[0, ["ci_low", "ci_high"]] = rows.loc[0, "estimate"]  # 2x2 shown as a point
    rows.to_csv(RESULTS / "estimator_comparison.csv", index=False)
    es.to_csv(RESULTS / "event_study.csv", index=False)
    sens.to_csv(RESULTS / "honestdid_sensitivity.csv", index=False)
    placebo.drop(columns="gap").assign(state=placebo.stfips.map(lambda f: STATES[f][0])).to_csv(
        RESULTS / "synthetic_control_placebos.csv", index=False)

    chart_trends(d)
    chart_event_study(es)
    chart_estimators(rows)
    chart_sensitivity(sens)
    chart_synth(r, placebo)

    summary = {
        "two_by_two": {k: round(pp(v), 2) for k, v in two.items()},
        "twfe_pp": round(pp(tw["estimate"]), 2),
        "cs_never_pp": [round(pp(cs_never[k]), 2) for k in ("estimate", "ci_low", "ci_high")],
        "cs_notyet_pp": [round(pp(cs_notyet[k]), 2) for k in ("estimate", "ci_low", "ci_high")],
        "event_study_pp": {int(e): round(pp(v), 2) for e, v in zip(es.event_time, es.estimate)},
        "honestdid_breakdown_Mbar": breakdown,
        "louisiana": {"avg_post_gap_pp": round(pp(r["avg_post_gap"]), 2), "rmspe_ratio": round(r["ratio"], 1),
                      "placebo_rank": f"{la_rank} of {len(placebo)}",
                      "permutation_p_value": round(la_rank / len(placebo), 3),
                      "donor_weights": {STATES[k][0]: round(float(v), 3) for k, v in r["weights"].items() if v > 0.01}},
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
