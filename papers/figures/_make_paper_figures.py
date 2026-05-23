"""Regenerates the four-stage debate convergence arc figure.

    python papers/figures/_make_paper_figures.py

Reads divergence_study_outputs/debate_full_arc.csv and writes both PNG
(300 DPI) and PDF (vector) versions of papers/figures/debate_full_arc.{png,pdf}.

The figure uses the project-wide palette in scripts/figure_style.py so that
its colors agree with every other figure in the paper. Specifically, the two
OpenAI generators that contributed to this experiment (gpt-5.4-nano and the
legacy gpt-4o cache) take their assigned OpenAI shades from GENERATOR_COLORS:
soft steel blue and deeper steel blue respectively. They are also
differentiated by marker shape and a faint within-color separation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "papers" / "figures"
DATA_DIR = REPO_ROOT / "divergence_study_outputs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from figure_style import (  # noqa: E402
    apply_paper_style,
    GENERATOR_COLORS,
    TEXT_WIDTH,
)

INK = "#1F2937"
MUTED = "#64748B"
LIGHT = "#CBD5E1"
GRID = "#E6E6E6"


def save(fig, name: str) -> None:
    """Write both PNG (300 DPI) and PDF copies of `fig` under FIG_DIR/name."""
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    print(f"  wrote {png.relative_to(REPO_ROOT)}  ({png.stat().st_size/1024:.0f} KB)")
    print(f"  wrote {pdf.relative_to(REPO_ROOT)}  ({pdf.stat().st_size/1024:.0f} KB)")


def make_debate_arc_figure() -> None:
    """The four-stage convergence arc, one summary line per generator with
    full-consensus as the headline trace and partial-convergence as a faint
    overlay. Per-scenario points are scattered to expose between-scenario
    spread.
    """
    apply_paper_style()
    df = pd.read_csv(DATA_DIR / "debate_full_arc.csv")

    stages = [
        "Closed\n(Rounds 0\u20132)",
        "Open\n(Round 2+)",
        "Synthesis\n(Round 3)",
        "Integrated\n(Round 4)",
    ]
    full_cols = ["sec13_consensus", "sec14_consensus", "sec15_full", "sec16_full"]
    partial_cols = [
        "sec13_consensus",
        "sec14_consensus",
        "sec15_partial",
        "sec16_majority",
    ]

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 3.1))

    # Two overlapping line series need maximum hue contrast within the
    # aquatic palette to read cleanly, so override the per-generator colours
    # used elsewhere: the current pilot model takes the jewel teal of the
    # active intervention; the legacy cache takes the abyssal navy.
    gen_styles = {
        "gpt-5.4-nano": dict(
            color="#0E7C7B",  # jewel teal (current pilot)
            marker="o",
            label="gpt-5.4-nano",
        ),
        "gpt-4o": dict(
            color="#0A3B5C",  # deep navy (legacy)
            marker="s",
            label="gpt-4o  (legacy)",
        ),
    }

    x = np.arange(len(stages))
    jitter = {"gpt-5.4-nano": -0.06, "gpt-4o": +0.06}

    for gen, style in gen_styles.items():
        sub = df[df["gen_model"] == gen]
        full = sub[full_cols].values
        partial = sub[partial_cols].values

        full_mean = full.mean(axis=0)
        partial_mean = partial.mean(axis=0)

        ax.fill_between(
            x,
            full_mean,
            partial_mean,
            color=style["color"],
            alpha=0.07,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x,
            partial_mean,
            color=style["color"],
            linestyle=(0, (4, 2)),
            linewidth=1.0,
            alpha=0.55,
            zorder=2,
        )
        for s in range(full.shape[0]):
            ax.scatter(
                x + jitter[gen],
                full[s],
                color=style["color"],
                alpha=0.22,
                s=14,
                linewidths=0,
                zorder=3,
            )
        ax.plot(
            x,
            full_mean,
            color=style["color"],
            marker=style["marker"],
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.9,
            linewidth=1.9,
            label=style["label"],
            zorder=4,
        )

    ax.axhline(1.0, color=LIGHT, linewidth=0.6, linestyle=":", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(v*100)}%" for v in np.linspace(0, 1, 6)])
    ax.set_ylim(-0.04, 1.05)
    ax.set_ylabel("Across-scenario rate")
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    pivot_bbox = dict(
        boxstyle="round,pad=0.32",
        facecolor="white",
        edgecolor=LIGHT,
        linewidth=0.5,
        alpha=0.95,
    )
    callouts = [
        (0, 0.06, "6%", 0, 12, "center", "bottom", INK, None),
        (1, 0.09, "9%", 0, 12, "center", "bottom", INK, None),
        (2, 0.0, "0%", 0, 12, "center", "bottom", INK, None),
        (2, 1.0, "100%", -8, 6, "right", "bottom", INK, None),
        (3, 0.951, "95% full\n(90% nano, 100% gpt-4o)", -8, -6, "right", "top", INK, None),
        (3, 1.0, "100% majority", -8, 6, "right", "bottom", MUTED, None),
        (
            2.55, 0.22,
            "synthesis pivot\nfull \u2192 0%   partial \u2192 100%",
            0, 0, "center", "center", MUTED, pivot_bbox,
        ),
    ]
    for xi, yi, text, dx, dy, ha, va, color, bbox in callouts:
        ax.annotate(
            text,
            xy=(xi, yi),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=7.8 if color is INK else 7.4,
            color=color,
            weight="bold" if color is INK else "normal",
            bbox=bbox,
            zorder=5,
        )

    handles, labels = ax.get_legend_handles_labels()
    series_full = plt.Line2D(
        [], [], color=INK, linewidth=1.9, label="full consensus (mean)",
    )
    series_partial = plt.Line2D(
        [], [], color=INK, linewidth=1.0, linestyle=(0, (4, 2)),
        alpha=0.55, label="partial convergence (mean)",
    )
    series_scatter = plt.Line2D(
        [], [], color=INK, alpha=0.4, marker="o",
        linewidth=0, markersize=4, label="per-scenario (full)",
    )
    legend = ax.legend(
        handles=handles + [series_full, series_partial, series_scatter],
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        handlelength=2.2,
        handletextpad=0.6,
        labelspacing=0.55,
        borderaxespad=0.0,
    )
    legend.get_frame().set_alpha(0.0)

    fig.tight_layout()
    save(fig, "debate_full_arc")
    plt.close(fig)


def main() -> None:
    print("Generating debate_full_arc...")
    make_debate_arc_figure()


if __name__ == "__main__":
    main()
