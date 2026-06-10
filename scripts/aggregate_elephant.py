"""
scripts/aggregate_elephant.py -- Phase 12/13 ELEPHANT aggregation and figures.

Reads elephant_singleagent_raw.csv and elephant_debate_raw.csv; computes
per-metric rates with Wilson CIs, deltas vs standard CoT, raw, and human.

Emits elephant_summary.json, elephant_sycophancy_rates.pdf,
elephant_human_gap.pdf, elephant_per_model_panel.pdf

Usage:
  python -m scripts.aggregate_elephant
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from scripts.aggregate_phase11 import fisher_two_proportion, wilson_ci
from scripts import figure_style as fs

OUT = Path("./divergence_study_outputs")
SA_CSV = OUT / "elephant_singleagent_raw.csv"
DEBATE_CSV = OUT / "elephant_debate_raw.csv"

METRICS = ["validation", "indirectness", "framing", "moral"]
ARM_LABELS = {
    "raw": "Raw",
    "baseline_io": "IO",
    "standard_cot": "Std CoT",
    "narrative_cot": "NoT",
    "human_baseline": "Human",
    "multi_stakeholder_not": "Debate NoT",
}
ARM_ORDER = ["human_baseline", "raw", "baseline_io", "standard_cot", "narrative_cot"]
QUARTET = [
    "gpt-5.4-nano", "claude-haiku-4-5", "claude-sonnet-4-6", "grok-4-1-fast-reasoning",
]


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _rate(rows: list[dict], col: str) -> tuple[int, int, float, list, list]:
    vals = []
    for r in rows:
        if r.get("empty_response") in ("1", 1, True):
            continue
        v = r.get(col, "")
        if v == "" or v is None:
            continue
        try:
            iv = int(float(v))
            if iv < 0:
                continue
            vals.append(iv)
        except (ValueError, TypeError):
            continue
    if not vals:
        return 0, 0, 0.0, [0.0, 0.0]
    s = sum(vals)
    n = len(vals)
    p, lo, hi = wilson_ci(s, n)
    return s, n, p, [lo, hi]


def _cell_key(r: dict) -> tuple:
    if r.get("arm") == "human_baseline":
        return (r["dataset"], "human_baseline", "human", "human_baseline")
    return (
        r["dataset"],
        r.get("generator", ""),
        r.get("arm", ""),
        r.get("side", "") or "single",
    )


def aggregate_singleagent(rows: list[dict]) -> dict:
    by_group: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_group[_cell_key(r)].append(r)

    stats = {}
    for key, grp in by_group.items():
        ds, gen, arm, side = key
        if ds == "flip_pairs" and side in ("og", "flip"):
            continue
        entry = {"dataset": ds, "generator": gen, "arm": arm, "n_rows": len(grp)}
        for m in METRICS:
            col = f"sycophantic_{m}"
            if col not in grp[0] and f"score_{m}" not in grp[0]:
                continue
            use_col = col if col in grp[0] else f"score_{m}"
            s, n, p, ci = _rate(grp, use_col)
            if n:
                entry[f"{m}_rate"] = p
                entry[f"{m}_wilson_ci"] = ci
                entry[f"{m}_n"] = n
                entry[f"{m}_positives"] = s
        stats[f"{ds}|{gen}|{arm}"] = entry

    def _pair_deltas(arm_a: str, arm_b: str, label: str) -> list[dict]:
        out = []
        for ds in sorted({r["dataset"] for r in rows}):
            for gen in sorted({r["generator"] for r in rows if r.get("generator") not in ("human", "")}):
                a_key = f"{ds}|{gen}|{arm_a}"
                b_key = f"{ds}|{gen}|{arm_b}"
                if a_key not in stats or b_key not in stats:
                    continue
                for m in METRICS:
                    rk = f"{m}_rate"
                    if rk not in stats[a_key] or rk not in stats[b_key]:
                        continue
                    a_n = int(stats[a_key].get(f"{m}_n", 0))
                    b_n = int(stats[b_key].get(f"{m}_n", 0))
                    a_pos = int(stats[a_key].get(f"{m}_positives", round(stats[a_key][rk] * a_n)))
                    b_pos = int(stats[b_key].get(f"{m}_positives", round(stats[b_key][rk] * b_n)))
                    test = fisher_two_proportion(a_pos, a_n, b_pos, b_n)
                    out.append({
                        "comparison": label,
                        "dataset": ds,
                        "generator": gen,
                        "metric": m,
                        "arm_a": arm_a,
                        "arm_b": arm_b,
                        "rate_a": stats[a_key][rk],
                        "rate_b": stats[b_key][rk],
                        "diff_pp": (stats[a_key][rk] - stats[b_key][rk]) * 100,
                        "fisher_p": test["fisher_p"],
                    })
        return out

    not_vs_cot = _pair_deltas("narrative_cot", "standard_cot", "not_vs_cot")
    not_vs_raw = _pair_deltas("narrative_cot", "raw", "not_vs_raw")
    cot_vs_raw = _pair_deltas("standard_cot", "raw", "cot_vs_raw")
    io_vs_raw = _pair_deltas("baseline_io", "raw", "io_vs_raw")

    human_gaps = []
    for ds in ("oeq", "aita_yta"):
        hkey = f"{ds}|human_baseline|human"
        if hkey not in stats:
            continue
        for arm in ("raw", "baseline_io", "standard_cot", "narrative_cot"):
            for gen in sorted({
                r["generator"] for r in rows
                if r.get("dataset") == ds and r.get("arm") == arm
            }):
                akey = f"{ds}|{gen}|{arm}"
                if akey not in stats:
                    continue
                for m in ("validation", "indirectness", "framing"):
                    rk = f"{m}_rate"
                    if rk not in stats[hkey] or rk not in stats[akey]:
                        continue
                    human_gaps.append({
                        "dataset": ds,
                        "generator": gen,
                        "arm": arm,
                        "metric": m,
                        "human_rate": stats[hkey][rk],
                        "model_rate": stats[akey][rk],
                        "gap_pp": (stats[akey][rk] - stats[hkey][rk]) * 100,
                    })

    return {
        "cell_stats": stats,
        "not_vs_cot": not_vs_cot,
        "not_vs_raw": not_vs_raw,
        "cot_vs_raw": cot_vs_raw,
        "io_vs_raw": io_vs_raw,
        "human_gaps": human_gaps,
    }


def aggregate_debate(rows: list[dict], sa_rows: list[dict]) -> dict:
    if not rows:
        return {}
    stats = {}
    for r in rows:
        key = (r["dataset"], r["generator"])
        stats.setdefault(key, []).append(r)

    debate_stats = {}
    for (ds, gen), grp in stats.items():
        entry = {"dataset": ds, "generator": gen, "arm": "multi_stakeholder_not", "n": len(grp)}
        for m in ("validation", "indirectness", "framing"):
            s, n, p, ci = _rate(grp, f"sycophantic_{m}")
            if n:
                entry[f"{m}_rate"] = p
                entry[f"{m}_wilson_ci"] = ci
                entry[f"{m}_n"] = n
        debate_stats[f"{ds}|{gen}"] = entry

    sa_agg = aggregate_singleagent(sa_rows)
    comparisons = []
    for key, dentry in debate_stats.items():
        ds, gen = key.split("|", 1)
        sa_key = f"{ds}|{gen}|narrative_cot"
        sa_cell = sa_agg["cell_stats"].get(sa_key, {})
        for m in ("validation", "indirectness", "framing"):
            rk = f"{m}_rate"
            if rk not in dentry or rk not in sa_cell:
                continue
            comparisons.append({
                "dataset": ds,
                "generator": gen,
                "metric": m,
                "debate_rate": dentry[rk],
                "single_not_rate": sa_cell[rk],
                "diff_pp": (dentry[rk] - sa_cell[rk]) * 100,
            })
    return {"debate_stats": debate_stats, "debate_vs_single": comparisons}


def _arm_color(arm: str) -> str:
    if arm == "raw":
        return "#BBBBBB"
    if arm == "human_baseline":
        return "#888888"
    return fs.CONDITION_COLORS.get(arm, "#4C72B0")


def figure_rates(sa: dict) -> None:
    import numpy as np
    stats = sa["cell_stats"]
    ds = "oeq"
    metric = "validation"
    gens = [g for g in QUARTET if any(k.startswith(f"{ds}|{g}|") for k in stats)]
    if not gens:
        gens = sorted({k.split("|")[1] for k in stats if k.startswith(f"{ds}|") and "human" not in k})

    arms = [a for a in ARM_ORDER if a != "human_baseline" or f"{ds}|human_baseline|human" in stats]
    n_panels = min(4, len(gens))
    fig, axes, _ = fs.panel_grid(n_panels, panel_w=2.5, panel_h=2.8)
    if n_panels == 1:
        axes = [axes]

    for ax, gen in zip(axes, gens[:n_panels]):
        rates = []
        labels = []
        colors = []
        for arm in arms:
            key = f"{ds}|human_baseline|human" if arm == "human_baseline" else f"{ds}|{gen}|{arm}"
            if key not in stats or f"{metric}_rate" not in stats[key]:
                continue
            rates.append(stats[key][f"{metric}_rate"])
            labels.append(ARM_LABELS.get(arm, arm))
            colors.append(_arm_color(arm))
        if rates:
            x = np.arange(len(rates))
            ax.bar(x, rates, color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=6)
            fs.percent_axis(ax, 1.0)
            ax.set_title(fs.short_generator_label(gen))
            ax.set_ylabel(f"{metric} rate")
    fig.suptitle("ELEPHANT OEQ validation sycophancy by arm (quartet)", fontsize=9)
    fs.save(fig, OUT / "elephant_sycophancy_rates.pdf")


def figure_per_model_panel(sa: dict) -> None:
    import numpy as np
    stats = sa["cell_stats"]
    datasets = ["oeq", "aita_yta"]
    metric = "validation"
    arms = ["raw", "baseline_io", "standard_cot", "narrative_cot"]

    fig, axes, _ = fs.panel_grid(len(datasets), panel_w=3.2, panel_h=2.8)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        gens = [g for g in QUARTET if any(k.startswith(f"{ds}|{g}|") for k in stats)]
        if not gens:
            continue
        x_base = np.arange(len(gens))
        w = 0.18
        for i, arm in enumerate(arms):
            vals = []
            for gen in gens:
                key = f"{ds}|{gen}|{arm}"
                vals.append(stats.get(key, {}).get(f"{metric}_rate", 0))
            offset = (i - (len(arms) - 1) / 2) * w
            ax.bar(x_base + offset, vals, w, label=ARM_LABELS.get(arm, arm),
                   color=_arm_color(arm))
        ax.set_xticks(x_base)
        ax.set_xticklabels([fs.short_generator_label(g) for g in gens], fontsize=7)
        fs.percent_axis(ax, 1.0)
        ax.set_title(f"{ds.upper()} {metric}")
        ax.legend(frameon=False, fontsize=6, loc="upper right")
    fig.suptitle("Per-model ELEPHANT validation rates", fontsize=9)
    fs.save(fig, OUT / "elephant_per_model_panel.pdf")


def figure_human_gap(sa: dict) -> None:
    import numpy as np
    gaps = [g for g in sa.get("human_gaps", []) if g["arm"] == "narrative_cot" and g["metric"] == "validation"]
    if not gaps:
        return
    oeq = [g for g in gaps if g["dataset"] == "oeq"]
    if not oeq:
        return
    fig, ax = fs.single_panel(width=fs.COL_WIDTH, height=2.6)
    gens = [g["generator"] for g in oeq]
    x = np.arange(len(gens))
    not_rates = [g["model_rate"] for g in oeq]
    human_rates = [g["human_rate"] for g in oeq]
    w = 0.35
    ax.bar(x - w/2, human_rates, w, label="Human", color="#888888")
    ax.bar(x + w/2, not_rates, w, label="NoT", color=fs.CONDITION_COLORS["narrative_cot"])
    ax.set_xticks(x)
    ax.set_xticklabels([fs.short_generator_label(g) for g in gens], fontsize=7)
    fs.percent_axis(ax, 1.0)
    ax.set_ylabel("Validation rate (OEQ)")
    ax.set_title("NoT vs human baseline")
    ax.legend(frameon=False, fontsize=7)
    fs.save(fig, OUT / "elephant_human_gap.pdf")


def summarize_falsification(sa: dict) -> dict:
    """Phase 13 pre-registered falsification checks."""
    not_vs_cot = sa.get("not_vs_cot", [])
    oeq_vf = [
        d for d in not_vs_cot
        if d["dataset"] == "oeq" and d["metric"] in ("validation", "framing")
        and d["diff_pp"] < 0 and d["fisher_p"] < 0.05
    ]
    gens_with_sig = {d["generator"] for d in oeq_vf}
    quartet_hits = sum(1 for g in QUARTET if g in gens_with_sig)

    not_vs_raw = sa.get("not_vs_raw", [])
    not_beats_raw = sum(1 for d in not_vs_raw if d["diff_pp"] < -5)
    not_loses_raw = sum(1 for d in not_vs_raw if d["diff_pp"] > 5)

    return {
        "quartet_models_with_sig_oeq_vf_reduction": quartet_hits,
        "not_vs_cot_sig_cells": len(oeq_vf),
        "not_beats_raw_cells_5pp": not_beats_raw,
        "not_loses_raw_cells_5pp": not_loses_raw,
        "social_sycophancy_claim_supported": quartet_hits >= 2,
    }


def main() -> int:
    sa_rows = _read_csv(SA_CSV)
    deb_rows = _read_csv(DEBATE_CSV)

    summary: dict = {}
    if sa_rows:
        sa = aggregate_singleagent(sa_rows)
        summary["singleagent"] = sa
        summary["falsification"] = summarize_falsification(sa)
        print("=== ELEPHANT single-agent ===")
        for d in sa["not_vs_cot"][:8]:
            print(f"  {d['dataset']:12s} {d['generator'][:20]:20s} {d['metric']:12s} "
                  f"NoT={d['rate_a']:.0%} CoT={d['rate_b']:.0%} diff={d['diff_pp']:+.1f}pp "
                  f"p={d['fisher_p']:.3f}")
        print("\n  --- NoT vs raw ---")
        for d in sa["not_vs_raw"][:6]:
            print(f"  {d['dataset']:12s} {d['generator'][:20]:20s} {d['metric']:12s} "
                  f"NoT={d['rate_a']:.0%} raw={d['rate_b']:.0%} diff={d['diff_pp']:+.1f}pp")
        print(f"\n  Falsification: {summary['falsification']}")
        try:
            figure_rates(sa)
            figure_per_model_panel(sa)
            figure_human_gap(sa)
        except Exception as e:
            print(f"  [figures skipped: {e}]")
    else:
        print(f"Missing {SA_CSV}")

    if deb_rows:
        deb = aggregate_debate(deb_rows, sa_rows)
        summary["debate"] = deb
        print("\n=== ELEPHANT debate ===")
        for k, v in deb.get("debate_stats", {}).items():
            vr = v.get("validation_rate", 0)
            print(f"  {k}: validation={vr:.0%} n={v.get('n', 0)}")
    else:
        print(f"\nMissing {DEBATE_CSV} (debate arm not run yet)")

    (OUT / "elephant_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {OUT / 'elephant_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
