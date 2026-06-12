"""
scripts/aggregate_phase14.py -- Phase 14 aggregation and optimizer comparison chart.

Merges optimizer summaries into phase14_summary.json and produces
syco_chart_optimizer_comparison.pdf.

Usage:
  python -m scripts.aggregate_phase14
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import figure_style as fs
from scripts.syco_loss import summary_path

OUT = Path("./divergence_study_outputs")
HOLDOUT_JSON = OUT / "phase14_holdout.json"

OPTIMIZERS = [
    ("narrative_grad", "sg", "Narrative grad"),
    ("textgrad_cot", "sg_tg", "TextGrad CoT"),
    ("opro", "sg_opro", "OPRO"),
    ("ape", "sg_ape", "APE"),
]
REF_LABELS = {
    "narrative_cot": "Hand NoT",
    "standard_cot": "Std CoT",
}


def load_all() -> dict:
    summary = {"optimizers": {}, "holdout": {}}
    for key, ns, _ in OPTIMIZERS:
        p = summary_path(ns)
        if p.exists():
            summary["optimizers"][key] = json.loads(p.read_text())
    if HOLDOUT_JSON.exists():
        summary["holdout"] = json.loads(HOLDOUT_JSON.read_text())
    (OUT / "phase14_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    return summary


def chart_optimizer_comparison(summary: dict) -> None:
    labels = []
    losses = []
    colors = []

    ref = summary.get("holdout", {}).get("reference_arms", {})
    if ref.get("narrative_cot"):
        labels.append("Hand NoT")
        losses.append(ref["narrative_cot"].get("loss", 0))
        colors.append(fs.CONDITION_COLORS["narrative_cot"])
    if ref.get("standard_cot"):
        labels.append("Std CoT")
        losses.append(ref["standard_cot"].get("loss", 0))
        colors.append(fs.CONDITION_COLORS["standard_cot"])

    palette = ["#0A5C5B", "#7E9AAB", "#4A6FA5", "#BFD0DA"]
    for i, (key, _, label) in enumerate(OPTIMIZERS):
        opt = summary.get("optimizers", {}).get(key, {})
        hold = opt.get("holdout_oeq", {}).get("optimised", {})
        if hold:
            labels.append(label)
            losses.append(hold.get("loss", 0))
            colors.append(palette[i % len(palette)])

    if not labels:
        print("  [chart skipped: no data]")
        return

    fig, ax = fs.single_panel(width=fs.COL_WIDTH, height=2.6)
    x = np.arange(len(labels))
    bars = ax.bar(x, losses, color=colors)
    fs.annotate_bars(ax, bars, losses, fmt="{:.2f}", dy=0.02)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("Holdout sycophancy loss (lower better)")
    ax.set_title("Phase 14: optimizer comparison on OEQ holdout")
    fs.save(fig, OUT / "syco_chart_optimizer_comparison.pdf")


def main() -> int:
    summary = load_all()
    print("=== Phase 14 optimizer holdout losses ===")
    for key, ns, label in OPTIMIZERS:
        opt = summary["optimizers"].get(key)
        if opt:
            loss = opt.get("holdout_oeq", {}).get("optimised", {}).get("loss")
            print(f"  {label}: {loss}")
    if summary.get("holdout", {}).get("winner"):
        print(f"  Winner: {summary['holdout']['winner']}")
    try:
        chart_optimizer_comparison(summary)
    except Exception as e:
        print(f"  [chart error: {e}]")
    print(f"Wrote {OUT / 'phase14_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
