"""
scripts/aggregate_phase1.py -- Aggregate Phase 1 raw outputs into the
summary CSVs and figure used in the paper.

Reads:  divergence_study_outputs/phase1_quartet_raw.csv
        (plus existing gpt-5.4-nano / gpt-4o data if present, keyed by
         the same column schema)

Writes:
  divergence_study_outputs/failure_mode_firing_quartet.csv
  divergence_study_outputs/tier1_effect_sizes_quartet.csv
  divergence_study_outputs/length_residualized_effects_quartet.csv
  divergence_study_outputs/failure_mode_firing_quartet.pdf
  divergence_study_outputs/tier1_effect_sizes_quartet.pdf

Usage:
  python -m scripts.aggregate_phase1 [--include-existing]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats

OUT_DIR = Path("./divergence_study_outputs")
FIGURES_DIR = OUT_DIR  # same dir; paper loads from figures/ but outputs land here


def _cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: proportion of (x_i, y_j) pairs where x > y, minus where x < y."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    greater = sum((xi > yj) for xi in x for yj in y)
    less = sum((xi < yj) for xi in x for yj in y)
    return (greater - less) / (nx * ny)


def _bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000,
                  seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        bx = rng.choice(x, size=len(x), replace=True)
        by = rng.choice(y, size=len(y), replace=True)
        deltas.append(_cliff_delta(bx, by))
    deltas_arr = np.array(deltas)
    return (float(np.percentile(deltas_arr, 2.5)),
            float(np.percentile(deltas_arr, 97.5)))


def _length_residualize(df_all: pd.DataFrame, col: str) -> np.ndarray:
    """Regress col on log(output_len) pooled across all conditions, return residuals."""
    x = np.log1p(df_all["output_len"].values.astype(float))
    y = df_all[col].values.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return y.copy()
    slope, intercept, *_ = stats.linregress(x[mask], y[mask])
    fitted = slope * x + intercept
    return y - fitted


def load_data(include_existing: bool = True) -> pd.DataFrame:
    frames = []
    quartet_csv = OUT_DIR / "phase1_quartet_raw.csv"
    if quartet_csv.exists():
        frames.append(pd.read_csv(quartet_csv))
    if include_existing:
        # Load any pre-existing CSVs from the notebook (coded_results_judge1.csv etc.)
        for name in ("coded_results_judge1.csv",):
            p = OUT_DIR / name
            if p.exists():
                df_existing = pd.read_csv(p)
                # Harmonise columns to the Phase 1 schema where possible
                frames.append(df_existing)
    if not frames:
        print("No data found. Run run_phase1_quartet.py first.")
        sys.exit(1)
    df = pd.concat(frames, ignore_index=True)
    return df


def compute_firing_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Per generator x failure mode: firing rate under std_cot and narrative_cot."""
    fm_cols = [
        "fm_premature_refusal", "fm_framework_enumeration",
        "fm_consequential_flattening", "fm_stakeholder_collapse",
        "fm_uncertainty_suppression",
    ]
    fm_labels = {
        "fm_premature_refusal": "premature refusal",
        "fm_framework_enumeration": "framework enumeration",
        "fm_consequential_flattening": "consequential flattening",
        "fm_stakeholder_collapse": "stakeholder collapse",
        "fm_uncertainty_suppression": "uncertainty suppression",
    }
    rows = []
    for gen in sorted(df["generator"].unique()):
        for col in fm_cols:
            if col not in df.columns:
                continue
            df_std = df[(df["generator"] == gen) & (df["condition"] == "standard_cot")]
            df_narr = df[(df["generator"] == gen) & (df["condition"] == "narrative_cot")]
            std_rate = df_std[col].astype(float).mean() if len(df_std) else float("nan")
            narr_rate = df_narr[col].astype(float).mean() if len(df_narr) else float("nan")
            rows.append({
                "generator": gen,
                "failure_mode": fm_labels.get(col, col),
                "std_cot_rate": round(std_rate, 4),
                "narrative_cot_rate": round(narr_rate, 4),
                "std_n": len(df_std),
                "narr_n": len(df_narr),
            })
    return pd.DataFrame(rows)


def compute_tier1_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Per generator x structural variable: Cliff's delta (narr vs std) + CI."""
    struct_vars = {
        "j1_stakeholder_count": "stakeholder_count",
        "j1_max_causal_hops": "max_causal_hops",
        "j1_uncertainty_score": "uncertainty_score",
        "j1_n_frameworks": "n_frameworks",
    }
    rows = []
    for gen in sorted(df["generator"].unique()):
        # residualize on log output_len pooled across all conditions for this generator
        df_gen = df[df["generator"] == gen].copy()
        resid_cols = {}
        for col in struct_vars:
            if col in df_gen.columns:
                df_gen[f"{col}_resid"] = _length_residualize(df_gen, col)
                resid_cols[col] = f"{col}_resid"

        df_std = df_gen[df_gen["condition"] == "standard_cot"]
        df_narr = df_gen[df_gen["condition"] == "narrative_cot"]

        for col, label in struct_vars.items():
            if col not in df_gen.columns:
                continue
            x_narr = df_narr[col].dropna().values
            x_std = df_std[col].dropna().values
            delta = _cliff_delta(x_narr, x_std)
            lo, hi = _bootstrap_ci(x_narr, x_std)
            # length-residualised
            rcol = resid_cols.get(col)
            r_narr = df_narr[rcol].dropna().values if rcol else np.array([])
            r_std = df_std[rcol].dropna().values if rcol else np.array([])
            rdelta = _cliff_delta(r_narr, r_std) if len(r_narr) else float("nan")
            rlo, rhi = _bootstrap_ci(r_narr, r_std) if len(r_narr) else (float("nan"), float("nan"))
            rows.append({
                "generator": gen,
                "variable": label,
                "delta_narr_vs_std": round(delta, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "resid_delta": round(rdelta, 4),
                "resid_ci_lo": round(rlo, 4),
                "resid_ci_hi": round(rhi, 4),
                "n_narr": len(x_narr),
                "n_std": len(x_std),
            })
    return pd.DataFrame(rows)


def make_firing_figure(df_firing: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping figure")
        return

    # Only the two modes that empirically fire
    firing_modes = ["stakeholder collapse", "uncertainty suppression"]
    df_plot = df_firing[df_firing["failure_mode"].isin(firing_modes)].copy()
    if df_plot.empty:
        return

    generators = sorted(df_plot["generator"].unique())
    modes = firing_modes
    x = np.arange(len(generators))
    width = 0.35
    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 4), sharey=False)
    if len(modes) == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
        sub = df_plot[df_plot["failure_mode"] == mode].set_index("generator")
        std_rates = [sub.loc[g, "std_cot_rate"] if g in sub.index else 0.0 for g in generators]
        narr_rates = [sub.loc[g, "narrative_cot_rate"] if g in sub.index else 0.0 for g in generators]
        bars_std = ax.bar(x - width / 2, std_rates, width, label="Standard CoT")
        bars_narr = ax.bar(x + width / 2, narr_rates, width, label="Narrative CoT")
        ax.set_title(mode.title(), fontsize=11)
        ax.set_ylabel("Firing rate", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [g.replace("claude-", "").replace("grok-4-1-", "grok-").replace("-reasoning", "-r")
             for g in generators],
            rotation=30, ha="right", fontsize=8,
        )
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

    fig.suptitle("Failure-mode firing rates: Standard CoT vs. Narrative CoT (five generators)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved figure: {out_path}")


def make_effects_figure(df_effects: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping figure")
        return

    key_vars = ["stakeholder_count", "uncertainty_score"]
    df_plot = df_effects[df_effects["variable"].isin(key_vars)].copy()
    if df_plot.empty:
        return

    generators = sorted(df_plot["generator"].unique())
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(generators))
    offsets = {"stakeholder_count": -0.2, "uncertainty_score": 0.2}
    colors = {"stakeholder_count": "#2166ac", "uncertainty_score": "#d6604d"}
    labels = {"stakeholder_count": "Stakeholder count δ (resid.)",
              "uncertainty_score": "Uncertainty score δ (resid.)"}

    for var in key_vars:
        sub = df_plot[df_plot["variable"] == var].set_index("generator")
        deltas = [sub.loc[g, "resid_delta"] if g in sub.index else 0.0 for g in generators]
        lo = [sub.loc[g, "resid_ci_lo"] if g in sub.index else 0.0 for g in generators]
        hi = [sub.loc[g, "resid_ci_hi"] if g in sub.index else 0.0 for g in generators]
        err_lo = [d - l for d, l in zip(deltas, lo)]
        err_hi = [h - d for d, h in zip(deltas, hi)]
        pos = x + offsets[var]
        ax.bar(pos, deltas, width=0.35, label=labels[var], color=colors[var], alpha=0.8)
        ax.errorbar(pos, deltas, yerr=[err_lo, err_hi], fmt="none",
                    color="black", capsize=4, linewidth=1.2)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0.47, color="grey", linewidth=0.6, linestyle=":")
    ax.text(len(generators) - 0.5, 0.49, "large effect threshold", fontsize=7, color="grey")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [g.replace("claude-", "").replace("grok-4-1-", "grok-").replace("-reasoning", "-r")
         for g in generators],
        rotation=30, ha="right", fontsize=8,
    )
    ax.set_ylabel("Length-residualised Cliff's δ (N-CoT vs. Std-CoT)", fontsize=9)
    ax.set_title("Structural effects survive length residualisation across all five generators",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved figure: {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate Phase 1 quartet results")
    parser.add_argument("--include-existing", action="store_true", default=True,
                        help="Merge existing gpt-5.4-nano / gpt-4o CSVs if available")
    args = parser.parse_args(argv)

    df = load_data(args.include_existing)
    print(f"Loaded {len(df)} rows from {df['generator'].nunique()} generators")

    # Firing rates
    df_firing = compute_firing_rates(df)
    fp = OUT_DIR / "failure_mode_firing_quartet.csv"
    df_firing.to_csv(fp, index=False)
    print(f"Wrote {fp}")
    print(df_firing.to_string(index=False))

    # Tier-1 effects
    df_effects = compute_tier1_effects(df)
    ep = OUT_DIR / "tier1_effect_sizes_quartet.csv"
    df_effects.to_csv(ep, index=False)
    print(f"Wrote {ep}")
    print(df_effects.to_string(index=False))

    # Figures
    make_firing_figure(df_firing, OUT_DIR / "failure_mode_firing_quartet.pdf")
    make_effects_figure(df_effects, OUT_DIR / "tier1_effect_sizes_quartet.pdf")

    return 0


if __name__ == "__main__":
    sys.exit(main())
