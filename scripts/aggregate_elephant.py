"""
scripts/aggregate_elephant.py -- Phase 12 ELEPHANT aggregation and figures.

Reads elephant_singleagent_raw.csv and elephant_debate_raw.csv; computes
per-metric rates with Wilson CIs, deltas vs standard CoT and human baseline.

Emits elephant_summary.json, elephant_sycophancy_rates.pdf,
elephant_human_gap.pdf, elephant_debate_vs_single.pdf

Usage:
  python -m scripts.aggregate_elephant
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from scripts.aggregate_phase11 import fisher_two_proportion, wilson_ci
from scripts import figure_style as fs

OUT = Path("./divergence_study_outputs")
SA_CSV = OUT / "elephant_singleagent_raw.csv"
DEBATE_CSV = OUT / "elephant_debate_raw.csv"

METRICS = ["validation", "indirectness", "framing", "moral"]
ARM_LABELS = {
    "baseline_io": "IO",
    "standard_cot": "Std CoT",
    "narrative_cot": "NoT",
    "human_baseline": "Human",
    "multi_stakeholder_not": "Debate NoT",
}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _rate(rows: list[dict], col: str) -> tuple[int, int, float, list, list]:
    vals = []
    for r in rows:
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


def aggregate_singleagent(rows: list[dict]) -> dict:
    by_group: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("arm") == "human_baseline":
            key = (r["dataset"], "human_baseline", "human", "human_baseline")
        else:
            key = (r["dataset"], r.get("generator", ""), r.get("arm", ""), r.get("side", "") or "single")
        by_group[key].append(r)

    stats = {}
    for key, grp in by_group.items():
        ds, gen, arm, side = key
        if side not in ("single", "pair", ""):
            if side not in ("pair",):
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
        stats[f"{ds}|{gen}|{arm}"] = entry

    # Deltas: NoT vs CoT per dataset x generator x metric
    deltas = []
    for ds in sorted({r["dataset"] for r in rows}):
        for gen in sorted({r["generator"] for r in rows if r.get("generator") != "human"}):
            not_key = f"{ds}|{gen}|narrative_cot"
            cot_key = f"{ds}|{gen}|standard_cot"
            if not_key not in stats or cot_key not in stats:
                continue
            for m in METRICS:
                rk = f"{m}_rate"
                if rk not in stats[not_key] or rk not in stats[cot_key]:
                    continue
                n_s = int(stats[not_key].get(f"{m}_n", 0))
                c_s = int(stats[cot_key].get(f"{m}_n", 0))
                n_pos = int(round(stats[not_key][rk] * n_s))
                c_pos = int(round(stats[cot_key][rk] * c_s))
                test = fisher_two_proportion(n_pos, n_s, c_pos, c_s)
                deltas.append({
                    "dataset": ds,
                    "generator": gen,
                    "metric": m,
                    "not_rate": stats[not_key][rk],
                    "cot_rate": stats[cot_key][rk],
                    "diff_pp": (stats[not_key][rk] - stats[cot_key][rk]) * 100,
                    "fisher_p": test["fisher_p"],
                })

    # Human gap on OEQ / AITA
    human_gaps = []
    for ds in ("oeq", "aita_yta"):
        hkey = f"{ds}|human_baseline|human"
        if hkey not in stats:
            continue
        for gen in sorted({r["generator"] for r in rows if r.get("dataset") == ds and r.get("arm") == "narrative_cot"}):
            nkey = f"{ds}|{gen}|narrative_cot"
            if nkey not in stats:
                continue
            for m in ("validation", "indirectness", "framing"):
                rk = f"{m}_rate"
                if rk not in stats[hkey] or rk not in stats[nkey]:
                    continue
                human_gaps.append({
                    "dataset": ds,
                    "generator": gen,
                    "metric": m,
                    "human_rate": stats[hkey][rk],
                    "not_rate": stats[nkey][rk],
                    "gap_pp": (stats[nkey][rk] - stats[hkey][rk]) * 100,
                })

    return {"cell_stats": stats, "not_vs_cot": deltas, "human_gaps": human_gaps}


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
        debate_stats[f"{ds}|{gen}"] = entry

    # Compare debate vs single-agent NoT
    comparisons = []
    for key, dentry in debate_stats.items():
        ds, gen = key.split("|", 1)
        sa_key = f"{ds}|{gen}|narrative_cot"
        sa_agg = aggregate_singleagent(sa_rows)
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


def figure_rates(sa: dict) -> None:
    import numpy as np
    stats = sa["cell_stats"]
    # OEQ validation by arm (pooled generators for narrative_cot vs standard_cot vs baseline_io vs human)
    ds = "oeq"
    arms = ["human_baseline", "baseline_io", "standard_cot", "narrative_cot"]
    metric = "validation"
    gens = sorted({k.split("|")[1] for k in stats if k.startswith(f"{ds}|") and "human" not in k})
    if not gens:
        return

    fig, axes, _ = fs.panel_grid(min(3, len(gens)), panel_w=2.5, panel_h=2.8)
    for ax, gen in zip(axes, gens[:3]):
        rates = []
        labels = []
        for arm in arms:
            key = f"{ds}|human|human_baseline" if arm == "human_baseline" else f"{ds}|{gen}|{arm}"
            if key not in stats or f"{metric}_rate" not in stats[key]:
                continue
            rates.append(stats[key][f"{metric}_rate"])
            labels.append(ARM_LABELS.get(arm, arm))
        if rates:
            x = np.arange(len(rates))
            ax.bar(x, rates, color=[fs.CONDITION_COLORS.get("narrative_cot", "#4C72B0")] * len(rates))
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
            fs.percent_axis(ax, 1.0)
            ax.set_title(fs.short_generator_label(gen))
            ax.set_ylabel(f"{metric} rate")
    fig.suptitle("ELEPHANT OEQ validation sycophancy by arm", fontsize=9)
    fs.save(fig, OUT / "elephant_sycophancy_rates.pdf")


def figure_human_gap(sa: dict) -> None:
    import numpy as np
    gaps = sa.get("human_gaps", [])
    if not gaps:
        return
    oeq = [g for g in gaps if g["dataset"] == "oeq" and g["metric"] == "validation"]
    if not oeq:
        return
    fig, ax = fs.single_panel(width=fs.COL_WIDTH, height=2.6)
    gens = [g["generator"] for g in oeq]
    x = np.arange(len(gens))
    not_rates = [g["not_rate"] for g in oeq]
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


def main() -> int:
    sa_rows = _read_csv(SA_CSV)
    deb_rows = _read_csv(DEBATE_CSV)

    summary: dict = {}
    if sa_rows:
        sa = aggregate_singleagent(sa_rows)
        summary["singleagent"] = sa
        print("=== ELEPHANT single-agent ===")
        for d in sa["not_vs_cot"][:12]:
            print(f"  {d['dataset']:12s} {d['generator'][:20]:20s} {d['metric']:12s} "
                  f"NoT={d['not_rate']:.0%} CoT={d['cot_rate']:.0%} diff={d['diff_pp']:+.1f}pp "
                  f"p={d['fisher_p']:.3f}")
        try:
            figure_rates(sa)
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
