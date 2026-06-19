"""Headline bar chart for the follow-up paper: v3 vs v2 dominance.

Renders a two-panel figure showing that NoT-v3 (cross-family trained)
beats NoT-v2 (in-family trained) on every generator on both axes the
paper claims dominance on:

  Left  - improvement in Cliff's delta on stakeholder count (raw points,
          v2_delta minus v3_delta; positive = v3 is more effective).

  Right - reduction in output length (percent shorter, v3 vs v2;
          positive = v3 is more concise).

All eight bars are positive: the chart is a single-glance demonstration
of v3 dominating v2 on every generator on both axes.

Numbers come from the paper's Table 1 (followup_paper.tex), themselves
derived from divergence_study_outputs/followup_per_generator.csv and
followup_v3_per_generator.csv. We hand-pin them here so the headline
figure is reproducible without re-reading the CSVs.

Usage:
    python3 papers/followup/figures/_make_headline_v3_vs_v2.py
Outputs:
    papers/followup/figures/v3_vs_v2_headline.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

from figure_style import (  # noqa: E402
    GENERATOR_ORDER,
    TEXT_WIDTH,
    apply_paper_style,
    short_generator_label,
)
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

# ---- Pinned data (paper Table 1 numbers) ----------------------------------
# v2 vs v1 and v3 vs v1 Cliff's delta on stakeholder count.
DELTA_SC = {
    "gpt-5.4-nano":            (+0.43, +0.12),  # (v2, v3); positive = bad
    "claude-haiku-4-5":        (-0.57, -0.73),  # negative = good
    "grok-4-1-fast-reasoning": (-0.68, -0.93),
    "claude-sonnet-4-6":       (-0.60, -0.68),
}
# v2/v1 and v3/v1 mean-output-character ratios.
LEN_RATIO = {
    "gpt-5.4-nano":            (0.82, 0.59),
    "claude-haiku-4-5":        (1.37, 0.78),
    "grok-4-1-fast-reasoning": (1.09, 0.95),
    "claude-sonnet-4-6":       (1.15, 0.64),
}

# ---- Derived: how much v3 beats v2 on each metric -------------------------
# Effect-size gain: v2_delta - v3_delta. Positive means v3 made the metric
# more negative (more improvement over v1) or pulled a v2 regression toward
# zero. All four generators come out positive.
EFFECT_GAIN = {g: round(v2 - v3, 3) for g, (v2, v3) in DELTA_SC.items()}

# Length reduction (proportion): (v2_len - v3_len) / v2_len. Positive means
# v3 produces fewer characters than v2 on the same scenarios.
LEN_REDUCTION = {
    g: round((v2 - v3) / v2, 4) for g, (v2, v3) in LEN_RATIO.items()
}

# Sanity print
print("Effect-size gain (v3 over v2):")
for g in GENERATOR_ORDER:
    print(f"  {g:<30} +{EFFECT_GAIN[g]:.2f} deliberation points")
print("Length reduction (v3 vs v2):")
for g in GENERATOR_ORDER:
    print(f"  {g:<30} {LEN_REDUCTION[g]*100:.1f}% shorter")

# ---- Render ----------------------------------------------------------------
TEAL = "#0E7C7B"        # primary intervention colour from the paper palette
TEAL_LIGHT = "#5BB0A8"  # secondary teal for the length panel
GREY = "#7E9AAB"        # axis / annotation grey
DARKGREY = "#333333"


def main() -> None:
    apply_paper_style()

    fig, (ax_eff, ax_len) = plt.subplots(
        1, 2,
        figsize=(TEXT_WIDTH, 2.6),
        gridspec_kw={"wspace": 0.22},
    )

    xs = list(range(len(GENERATOR_ORDER)))
    eff_vals = [EFFECT_GAIN[g] for g in GENERATOR_ORDER]
    len_vals = [LEN_REDUCTION[g] for g in GENERATOR_ORDER]

    # --- Left panel: effect-size gain --------------------------------------
    bars_eff = ax_eff.bar(
        xs, eff_vals,
        color=TEAL, edgecolor="white", linewidth=0.6, width=0.66, zorder=3,
    )
    ax_eff.axhline(0, color=GREY, linewidth=0.6, zorder=2)
    ax_eff.set_ylabel("v3 effect-size gain over v2\n"
                       r"(Cliff's $\delta_{\mathrm{sc}}$ points)",
                       fontsize=8.5)
    ax_eff.set_title("Stronger deliberation gain",
                      fontsize=10, pad=6, color=DARKGREY, weight="bold")
    ax_eff.set_ylim(0, max(eff_vals) * 1.28)
    ax_eff.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    for bar, v in zip(bars_eff, eff_vals):
        ax_eff.text(bar.get_x() + bar.get_width() / 2,
                    v + max(eff_vals) * 0.03,
                    f"+{v:.2f}",
                    ha="center", va="bottom", fontsize=8.0,
                    color=DARKGREY, weight="bold")

    # --- Right panel: length reduction -------------------------------------
    bars_len = ax_len.bar(
        xs, len_vals,
        color=TEAL_LIGHT, edgecolor="white", linewidth=0.6, width=0.66, zorder=3,
    )
    ax_len.axhline(0, color=GREY, linewidth=0.6, zorder=2)
    ax_len.set_ylabel("v3 output-length reduction\n(fraction of v2 length saved)",
                      fontsize=8.5)
    ax_len.set_title("Shorter outputs",
                     fontsize=10, pad=6, color=DARKGREY, weight="bold")
    ax_len.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax_len.set_ylim(0, max(len_vals) * 1.28)
    for bar, v in zip(bars_len, len_vals):
        ax_len.text(bar.get_x() + bar.get_width() / 2,
                    v + max(len_vals) * 0.03,
                    f"{v*100:.0f}%",
                    ha="center", va="bottom", fontsize=8.0,
                    color=DARKGREY, weight="bold")

    # --- Shared x-axis formatting ------------------------------------------
    for ax in (ax_eff, ax_len):
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [short_generator_label(g) for g in GENERATOR_ORDER],
            fontsize=7.5,
        )
        ax.tick_params(axis="x", pad=2, length=0)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.5, zorder=1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(GREY)
        ax.spines["bottom"].set_color(GREY)

    fig.suptitle(
        "NoT-v3 beats NoT-v2 on every generator on every axis",
        fontsize=11, color="#0A5C5B", weight="bold", y=1.04,
    )

    out = Path(__file__).resolve().parent / "v3_vs_v2_headline.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"  Saved figure: {out}")


if __name__ == "__main__":
    main()
