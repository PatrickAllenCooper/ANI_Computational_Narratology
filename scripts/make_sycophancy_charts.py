"""
scripts/make_sycophancy_charts.py -- Three headline bar charts for the
sycophancy paper (Phase 13).

Chart 1: Sharma SycophancyEval saturation (propositional floor).
Chart 2: ELEPHANT OEQ validation by arm and generator vs human baseline.
Chart 3: Moral both-NTA collapse on FLIP pairs.

All values are read from divergence_study_outputs artefacts; nothing is
hard-coded.

Usage:
  python -m scripts.make_sycophancy_charts
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts import figure_style as fs

OUT = Path("./divergence_study_outputs")
SUMMARY = OUT / "elephant_summary.json"
SHARMA_CSV = OUT / "sycophancyeval_results.csv"

GENS = fs.GENERATOR_ORDER  # nano, haiku, grok, sonnet
ARM_COLORS = {
    "raw": "#BFD0DA",
    "standard_cot": fs.CONDITION_COLORS["standard_cot"],
    "narrative_cot": fs.CONDITION_COLORS["narrative_cot"],
    "debate": "#0A5C5B",
}
ARM_LABELS = {
    "raw": "Raw",
    "standard_cot": "Std CoT",
    "narrative_cot": "NoT",
    "debate": "Debate NoT",
}


def _sharma_rates() -> dict[tuple[str, str], float]:
    rates: dict[tuple[str, str], list[int]] = defaultdict(list)
    with SHARMA_CSV.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["generator"], r["condition"])
            rates[key].append(1 if r["sycophantic"].strip().lower() == "true" else 0)
    return {k: sum(v) / len(v) for k, v in rates.items() if v}


def chart_sharma() -> None:
    rates = _sharma_rates()
    fig, ax = fs.single_panel(width=fs.COL_WIDTH, height=2.4)
    x = np.arange(len(GENS))
    w = 0.38
    for i, cond in enumerate(["standard_cot", "narrative_cot"]):
        vals = [rates.get((g, cond), 0.0) for g in GENS]
        bars = ax.bar(
            x + (i - 0.5) * w, vals, w,
            label=fs.CONDITION_LABELS[cond],
            color=fs.CONDITION_COLORS[cond],
        )
        fs.annotate_bars(ax, bars, vals, fmt="{:.1f}%", dy=0.001)
    fs.style_generator_axis(ax, GENS)
    fs.percent_axis(ax, 0.10)
    ax.set_ylabel("Propositional sycophancy rate")
    ax.set_title("Sharma SycophancyEval: saturated at the floor")
    ax.legend(frameon=False, loc="upper right")
    fs.save(fig, OUT / "syco_chart_sharma_floor.pdf")


def chart_oeq_validation(summary: dict) -> None:
    stats = summary["singleagent"]["cell_stats"]
    debate = summary.get("debate", {}).get("debate_stats", {})
    human = stats["oeq|human_baseline|human"]["validation_rate"]

    arms = ["raw", "standard_cot", "narrative_cot", "debate"]
    fig, ax = fs.single_panel(width=fs.TEXT_WIDTH * 0.62, height=2.6)
    x = np.arange(len(GENS))
    w = 0.19
    for i, arm in enumerate(arms):
        vals = []
        for g in GENS:
            if arm == "debate":
                vals.append(debate.get(f"oeq|{g}", {}).get("validation_rate"))
            else:
                vals.append(stats.get(f"oeq|{g}|{arm}", {}).get("validation_rate"))
        offset = (i - (len(arms) - 1) / 2) * w
        plot_vals = [v if v is not None else 0.0 for v in vals]
        bars = ax.bar(x + offset, plot_vals, w,
                      label=ARM_LABELS[arm], color=ARM_COLORS[arm])
        fs.annotate_bars(ax, bars, vals, fmt="{:.0f}", dy=0.012, fs=6.5)
    ax.axhline(human, color="#555555", linestyle="--", linewidth=1.0)
    ax.text(len(GENS) - 0.52, human + 0.015, f"human {human:.0%}",
            fontsize=7.5, color="#555555", ha="right")
    fs.style_generator_axis(ax, GENS)
    fs.percent_axis(ax, 1.0)
    ax.set_ylabel("OEQ validation rate")
    ax.set_title("ELEPHANT open-ended advice: NoT closes the human gap")
    ax.legend(frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.0))
    fs.save(fig, OUT / "syco_chart_oeq_validation.pdf")


def chart_moral_collapse(summary: dict) -> None:
    stats = summary["singleagent"]["cell_stats"]
    arms = ["raw", "standard_cot", "narrative_cot"]
    fig, ax = fs.single_panel(width=fs.COL_WIDTH, height=2.4)
    x = np.arange(len(GENS))
    w = 0.26
    for i, arm in enumerate(arms):
        vals = [stats.get(f"flip_pairs|{g}|{arm}", {}).get("moral_rate", 0.0)
                for g in GENS]
        offset = (i - 1) * w
        bars = ax.bar(x + offset, vals, w,
                      label=ARM_LABELS[arm], color=ARM_COLORS[arm])
        fs.annotate_bars(ax, bars, vals, fmt="{:.0f}", dy=0.012, fs=6.5)
    fs.style_generator_axis(ax, GENS)
    fs.percent_axis(ax, 1.12)
    ax.set_ylabel("Both-NTA rate (150 FLIP pairs)")
    ax.set_title("Moral sycophancy collapses under NoT")
    ax.legend(frameon=False, loc="upper right")
    fs.save(fig, OUT / "syco_chart_moral_collapse.pdf")


def main() -> int:
    summary = json.loads(SUMMARY.read_text())
    chart_sharma()
    chart_oeq_validation(summary)
    chart_moral_collapse(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
