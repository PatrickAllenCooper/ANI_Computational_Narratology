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
    """Cliff's delta via the U-statistic identity (vectorised, O(n log n))."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    # Mann-Whitney U: U = #{x_i > y_j}.  Cliff's delta = 2*U/(nx*ny) - 1
    # Use scipy's mannwhitneyu for the U; method='asymptotic' avoids tie-corrections we don't need.
    try:
        u, _ = stats.mannwhitneyu(x, y, alternative="greater", method="asymptotic")
        # adjust for ties: convert to delta via ranks
        # Cliff's delta = (#x>y - #x<y) / (nx*ny)
        # = (2U - nx*ny) / (nx*ny) where U counts ties as 0.5
        return (2 * u - nx * ny) / (nx * ny)
    except Exception:
        # fallback: vectorised broadcast (still O(nx*ny) memory but fast)
        diff = np.subtract.outer(x, y)
        return (np.sum(diff > 0) - np.sum(diff < 0)) / (nx * ny)


def _bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 500,
                  seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        bx = rng.choice(x, size=len(x), replace=True)
        by = rng.choice(y, size=len(y), replace=True)
        deltas[i] = _cliff_delta(bx, by)
    return (float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)))


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
    df = df.dropna(subset=["generator", "condition"])
    df["generator"] = df["generator"].astype(str)
    df["condition"] = df["condition"].astype(str)
    # Exclude gpt-4o (legacy/obsolete) and ablation conditions from main aggregates
    df = df[df["generator"] != "gpt-4o"]
    df = df[df["condition"].isin(["baseline_io", "standard_cot", "narrative_cot"])]
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


def _delegate_to_rebuild() -> None:
    """Delegate figure rendering to scripts/rebuild_paper_figures.py so that
    the paper-wide aesthetic stays consistent. Aggregation must have already
    written the corresponding CSVs to divergence_study_outputs/."""
    from rebuild_paper_figures import fig_failure_mode_firing, fig_tier1_effects
    fig_failure_mode_firing()
    fig_tier1_effects()


def make_firing_figure(df_firing: pd.DataFrame, out_path: Path) -> None:
    """Deprecated entry-point retained for backwards compatibility.
    Calls the unified figure pipeline; out_path is honoured implicitly via
    the canonical filename in figure_style."""
    _delegate_to_rebuild()


def make_effects_figure(df_effects: pd.DataFrame, out_path: Path) -> None:
    """Deprecated entry-point retained for backwards compatibility.
    See `make_firing_figure` for the unified pipeline call."""
    # both figures share a single delegation call; no-op after the first.
    pass


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
