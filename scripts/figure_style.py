"""scripts/figure_style.py -- Unified figure aesthetic for the ACL paper.

All figure-producing scripts should import from this module so that every
plot in the paper shares the same palette, typography, sizing, and layout
conventions. Keep this file the single source of truth for any visual
choice that appears in the camera-ready manuscript.

Conventions:

* Condition palette is fixed across all figures:
    standard CoT         -> warm grey
    narration-of-thought -> deep teal
  These two colours encode the experiment's primary contrast and must
  appear consistently wherever both conditions are plotted side by side.

* Vendor palette is fixed across all figures that compare generators:
    OpenAI    -> indigo
    Anthropic -> burnt orange
    xAI       -> slate
  Two-tier vendors (budget vs. flagship) use a lighter shade of the same
  hue for the budget tier.

* Typography is Times-compatible 9pt body / 8pt tick labels / 10pt
  panel titles, matching the paper's body face. Math is rendered with
  matplotlib's mathtext.

* All figures are saved with bbox_inches='tight' at 300 dpi and a fixed
  width compatible with the ACL columnwidth (3.3in single column) or
  textwidth (6.95in two columns).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# Column / text widths in inches (ACL acl.sty: columnwidth ~3.3in, textwidth ~6.95in)
COL_WIDTH = 3.30
TEXT_WIDTH = 6.95

# Single coherent aquatic palette across every figure: cool blue-grey for the
# neutral baseline, jewel teal for the active intervention, and a graded
# steel-blue -> teal -> sea-green spectrum for distinguishing generators
# without leaving the blue-green family. No warm hues anywhere.

# Condition palette (primary contrast, used everywhere two conditions appear).
CONDITION_COLORS = {
    "standard_cot": "#7E9AAB",   # cool blue-grey (baseline, neutral)
    "narrative_cot": "#0E7C7B",  # jewel teal (intervention, active)
    "baseline_io": "#BFD0DA",    # pale blue-grey (ablation reference)
}
CONDITION_LABELS = {
    "standard_cot": "Standard CoT",
    "narrative_cot": "NoT",
    "baseline_io": "Baseline I/O",
}

# Vendor palette (used wherever a vendor needs a single colour, e.g. the
# multi-agent panel of the hero figure). All three live in the blue-green
# family and are ordered by hue: steel-teal -> deep teal -> sea green.
VENDOR_COLORS = {
    "openai":    "#2E7DA0",   # steel-teal
    "anthropic": "#0E5C5B",   # deep teal
    "xai":       "#5BB0A8",   # sea green
}

# Per-generator colour: a graded blue-green spectrum so the four generators
# (plus the legacy gpt-4o cache) read as five distinguishable shades in the
# same family. Ordered roughly by hue from steel-blue through teal to sea-green
# so adjacent generators are easy to tell apart.
GENERATOR_COLORS = {
    "gpt-5.4-nano":              "#2E7DA0",   # steel-teal
    "gpt-4o":                    "#0A3B5C",   # deep navy (legacy)
    "claude-haiku-4-5":          "#5BB0A8",   # sea green
    "claude-sonnet-4-6":         "#0E5C5B",   # deep teal
    "grok-4-1-fast-reasoning":   "#3F8C84",   # medium kelp green
}

# Alarm colour for harmful actions / failure indicators. Kept inside the
# blue-green family by going to deep navy: dark enough to read as grave
# against the rest of the palette without introducing warm reds or oranges.
ALARM_COLOR = "#1A3B6E"  # deep navy

# Order used everywhere generators appear on a categorical axis.
GENERATOR_ORDER = [
    "gpt-5.4-nano",
    "claude-haiku-4-5",
    "grok-4-1-fast-reasoning",
    "claude-sonnet-4-6",
    "deepseek-v3",
]

# Effect-size thresholds (Cliff's delta, from Cliff 1993)
DELTA_NEGLIGIBLE = 0.147
DELTA_SMALL = 0.33
DELTA_MEDIUM = 0.474


def short_generator_label(name: str) -> str:
    """Compact label for plot ticks. Vendor + tier on two lines."""
    mapping = {
        "gpt-5.4-nano":              "gpt-5.4\nnano",
        "gpt-4o":                    "gpt-4o",
        "claude-haiku-4-5":          "claude-haiku\n4-5",
        "claude-sonnet-4-6":         "claude-sonnet\n4-6",
        "grok-4-1-fast-reasoning":   "grok-4-1\nfast-r.",
        "deepseek-v3":               "deepseek\nv3",
    }
    return mapping.get(name, name)


# ---------------------------------------------------------------------------
# Style initialisation
# ---------------------------------------------------------------------------

def apply_paper_style() -> None:
    """Install the paper-wide matplotlib rcParams. Idempotent."""
    plt.rcParams.update({
        # Typography
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "figure.titlesize": 10.5,
        "mathtext.fontset": "stix",

        # Lines and markers
        "lines.linewidth": 1.4,
        "lines.markersize": 4.0,
        "patch.linewidth": 0.6,
        "axes.linewidth": 0.6,

        # Spines and ticks
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,

        # Grid (subtle, behind data)
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": "#E6E6E6",
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",

        # Figure saving
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def panel_grid(n_panels: int, *, panel_w: float = 1.7, panel_h: float = 2.2,
               sharey: bool = True):
    """Create a 1xN panel layout sized for the ACL two-column textwidth.

    Total figure width is capped at TEXT_WIDTH. Each panel gets panel_w
    inches by default; the helper shrinks panels if the requested total
    would exceed textwidth.
    """
    apply_paper_style()
    width = min(TEXT_WIDTH, max(panel_w * n_panels, COL_WIDTH))
    panel_w_actual = width / n_panels
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(width, panel_h),
        sharey=sharey,
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]
    return fig, list(axes), panel_w_actual


def single_panel(width: float = COL_WIDTH, height: float = 2.4):
    """Create a single-column-friendly panel."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    return fig, ax


def percent_axis(ax, ymax: float = 1.0) -> None:
    """Format y-axis as percentage."""
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))


def annotate_bars(ax, bars, values, *, fmt: str = "{:.0f}%",
                   pct: bool = True, dy: float = 0.015, fs: float = 7.0) -> None:
    """Print a small value label above each bar."""
    for bar, v in zip(bars, values):
        if v is None:
            continue
        label = fmt.format(v * 100) if pct else fmt.format(v)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + dy,
                label,
                ha="center", va="bottom", fontsize=fs, color="#333333")


def style_generator_axis(ax, generators: Iterable[str]) -> None:
    """Apply the canonical generator label formatting on the x axis."""
    labels = [short_generator_label(g) for g in generators]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.tick_params(axis="x", pad=2)


def save(fig, path: Path) -> None:
    """Save with the standard settings and close."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved figure: {path}")
