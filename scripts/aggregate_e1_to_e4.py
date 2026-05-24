"""
scripts/aggregate_e1_to_e4.py -- Aggregate E1-E4 results into updated
CSVs and print a paper-ready summary for each experiment.

What this script does:
  E1: Add standard_cot_verbose rows to the existing phase1_quartet_raw.csv,
      recompute Cliff's delta and length-residualised delta for
      standard_cot_verbose vs narrative_cot, and write
      divergence_study_outputs/e1_matched_budget_results.csv.

  E2: Summarise debate_dd_scaled.csv into
      divergence_study_outputs/e2_consensus_summary.csv with per-generator
      full-consensus rates, structural-rejection rates, and synthesis rates.

  E3: Extract claude-opus-4-1 rows from phase1_quartet_raw.csv (or a
      separate run CSV), compute refusal rates per condition, and write
      divergence_study_outputs/e3_opus_refusal_results.csv.

  E4: Load kc_proxy_correlation.csv (already written by run_phase4_kc.py),
      print correlation summary, and write
      divergence_study_outputs/e4_kc_validation_summary.csv.

Usage:
  python -m scripts.aggregate_e1_to_e4 [--skip-e1] [--skip-e2] [--skip-e3] [--skip-e4]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

OUT_DIR = Path("./divergence_study_outputs")


# ---------------------------------------------------------------------------
# Shared statistics helpers
# ---------------------------------------------------------------------------

def _cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    try:
        u, _ = stats.mannwhitneyu(x, y, alternative="greater", method="asymptotic")
        return (2 * u - nx * ny) / (nx * ny)
    except Exception:
        diff = np.subtract.outer(x, y)
        return (np.sum(diff > 0) - np.sum(diff < 0)) / (nx * ny)


def _bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 500,
                  seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        deltas.append(_cliff_delta(xb, yb))
    arr = np.array(deltas)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def _residualise(df: pd.DataFrame, metric: str) -> pd.Series:
    """Regress metric on log(output_len) across all conditions and return residuals."""
    mask = df["output_len"] > 0
    if mask.sum() < 10:
        return pd.Series(float("nan"), index=df.index)
    log_len = np.log(df.loc[mask, "output_len"].values.astype(float))
    y = df.loc[mask, metric].values.astype(float)
    # OLS: y = a + b * log_len
    b, a = np.polyfit(log_len, y, deg=1)
    residuals = pd.Series(float("nan"), index=df.index)
    residuals.loc[mask] = y - (a + b * log_len)
    return residuals


# ---------------------------------------------------------------------------
# E1: matched-budget standard CoT vs. narrative CoT
# ---------------------------------------------------------------------------

def aggregate_e1() -> None:
    print("\n=== E1: Matched-budget standard CoT vs. narrative CoT ===")
    raw_csv = OUT_DIR / "phase1_quartet_raw.csv"
    if not raw_csv.exists():
        print(f"  ERROR: {raw_csv} not found. Run run_phase1_quartet.py first.")
        return

    df = pd.read_csv(raw_csv)
    has_verbose = "standard_cot_verbose" in df["condition"].values
    if not has_verbose:
        print("  standard_cot_verbose not yet in data -- E1 has not run yet.")
        return

    rows: list[dict] = []
    for gen in sorted(df["generator"].unique()):
        for metric in ["j1_stakeholder_count", "j1_uncertainty_score"]:
            if metric not in df.columns:
                continue
            verbose = df[(df["generator"] == gen) & (df["condition"] == "standard_cot_verbose")][metric].dropna().values
            ncot = df[(df["generator"] == gen) & (df["condition"] == "narrative_cot")][metric].dropna().values
            if len(verbose) < 5 or len(ncot) < 5:
                continue
            delta = _cliff_delta(ncot, verbose)
            lo, hi = _bootstrap_ci(ncot, verbose)
            # Length-residualised
            sub = df[df["generator"] == gen][["condition", "output_len", metric]].dropna()
            resid = _residualise(sub, metric)
            sub = sub.copy()
            sub["resid"] = resid.values
            v_resid = sub[sub["condition"] == "standard_cot_verbose"]["resid"].dropna().values
            n_resid = sub[sub["condition"] == "narrative_cot"]["resid"].dropna().values
            delta_resid = _cliff_delta(n_resid, v_resid) if len(v_resid) > 2 and len(n_resid) > 2 else float("nan")
            rows.append({
                "generator": gen,
                "metric": metric,
                "n_verbose": len(verbose),
                "n_ncot": len(ncot),
                "cliff_delta_ncot_vs_verbose": round(delta, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "length_residualised_delta": round(delta_resid, 4),
                "length_null_ruled_out": (
                    "YES" if (not np.isnan(delta_resid) and delta_resid > 0.2)
                    else "NO"
                ),
            })

    out = OUT_DIR / "e1_matched_budget_results.csv"
    if rows:
        df_out = pd.DataFrame(rows)
        df_out.to_csv(out, index=False)
        print(df_out.to_string(index=False))
        print(f"\n  Wrote {out}")
        # Interpretation
        null_ruled_out = sum(1 for r in rows if r["length_null_ruled_out"] == "YES")
        print(f"\n  Length-null ruled out in {null_ruled_out}/{len(rows)} (generator, metric) pairs.")
        if null_ruled_out > len(rows) // 2:
            print("  CONCLUSION: Structural gains are not explained by output length alone.")
            print("  Paper claim 'Length is a cost, not a substitute for the prompt' is CONFIRMED.")
        else:
            print("  CONCLUSION: Length partially explains gains -- claim should be DOWNGRADED.")
    else:
        print("  No results computed.")


# ---------------------------------------------------------------------------
# E2: scaled multi-agent consensus on DailyDilemmas
# ---------------------------------------------------------------------------

def aggregate_e2() -> None:
    print("\n=== E2: Scaled multi-agent debate consensus ===")
    csv_path = OUT_DIR / "debate_dd_scaled.csv"
    if not csv_path.exists():
        print(f"  ERROR: {csv_path} not found. Run run_phase5_e2_scaled.py first.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("  Empty CSV.")
        return

    completed = df[~df.get("skipped_budget", pd.Series(False, index=df.index)).fillna(False)]
    rows = []
    for gen in sorted(completed["generator"].unique()):
        sub = completed[completed["generator"] == gen]
        n = len(sub)
        if n == 0:
            continue
        n_synth = sub["has_synthesis"].sum() if "has_synthesis" in sub else 0
        n_r4_full = sub["r4_full_consensus"].sum() if "r4_full_consensus" in sub else 0
        n_struct_rej = sub["r4_structural_rejection"].sum() if "r4_structural_rejection" in sub else 0
        rows.append({
            "generator": gen,
            "n_cells": n,
            "synthesis_rate_pct": round(100 * n_synth / n, 1) if n > 0 else 0,
            "r4_full_consensus_pct": round(100 * n_r4_full / n, 1) if n > 0 else 0,
            "structural_rejection_rate_pct": round(100 * n_struct_rej / n, 1) if n > 0 else 0,
        })

    out = OUT_DIR / "e2_consensus_summary.csv"
    if rows:
        df_out = pd.DataFrame(rows)
        df_out.to_csv(out, index=False)
        print(df_out.to_string(index=False))
        print(f"\n  Wrote {out}")

        # Falsification check
        overall_r4 = 100 * completed["r4_full_consensus"].sum() / len(completed) \
            if "r4_full_consensus" in completed.columns and len(completed) > 0 else float("nan")
        print(f"\n  Overall R4 consensus rate: {overall_r4:.1f}%")
        print(f"  Pre-registered band: 85-105% (95% +/- 10pp)")
        if 85.0 <= overall_r4 <= 105.0:
            print("  FALSIFICATION: REPLICATES -- headline claim holds.")
        else:
            print("  FALSIFICATION: OUTSIDE BAND -- headline claim downgrades.")
    else:
        print("  No completed rows.")


# ---------------------------------------------------------------------------
# E3: safety-tuned fifth generator (claude-opus-4-1)
# ---------------------------------------------------------------------------

def aggregate_e3() -> None:
    print("\n=== E3: Safety-tuned fifth generator refusal rates ===")
    raw_csv = OUT_DIR / "phase1_quartet_raw.csv"
    if not raw_csv.exists():
        print(f"  ERROR: {raw_csv} not found.")
        return

    df = pd.read_csv(raw_csv)
    opus_rows = df[df["generator"].str.contains("opus", case=False, na=False)]
    if opus_rows.empty:
        print("  No claude-opus rows found in phase1_quartet_raw.csv -- E3 has not run yet.")
        return

    rows = []
    for gen in sorted(opus_rows["generator"].unique()):
        for cond in ["baseline_io", "standard_cot", "narrative_cot"]:
            sub = opus_rows[(opus_rows["generator"] == gen) & (opus_rows["condition"] == cond)]
            if sub.empty:
                continue
            refused_col = "j1_refused" if "j1_refused" in sub else None
            if refused_col:
                n_refused = sub[refused_col].sum()
                n_total = len(sub)
                rows.append({
                    "generator": gen,
                    "condition": cond,
                    "n": n_total,
                    "n_refused": int(n_refused),
                    "refusal_rate_pct": round(100 * n_refused / n_total, 1),
                })

    out = OUT_DIR / "e3_opus_refusal_results.csv"
    if rows:
        df_out = pd.DataFrame(rows)
        df_out.to_csv(out, index=False)
        print(df_out.to_string(index=False))
        print(f"\n  Wrote {out}")

        # Falsification check
        std_cot_row = next(
            (r for r in rows if r["condition"] == "standard_cot"), None
        )
        ncot_row = next(
            (r for r in rows if r["condition"] == "narrative_cot"), None
        )
        if std_cot_row and ncot_row:
            std_pct = std_cot_row["refusal_rate_pct"]
            ncot_pct = ncot_row["refusal_rate_pct"]
            print(f"\n  standard_cot refusal: {std_pct}%")
            print(f"  narrative_cot refusal: {ncot_pct}%")
            if std_pct >= 15.0:
                reduction = (std_pct - ncot_pct) / std_pct if std_pct > 0 else 0
                print(f"  Refusal reduction: {100*reduction:.1f}%")
                if reduction >= 0.5:
                    print("  RESULT: N-CoT reduces premature refusal by >=50%. Limitation CLOSED.")
                else:
                    print("  RESULT: Refusal reduction <50%. Limitation remains.")
            else:
                print(f"  standard_cot refusal rate {std_pct}% < 15% threshold.")
                print("  RESULT: Model is not sufficiently refusal-prone. Fallback test needed.")
    else:
        print("  No opus results computed.")


# ---------------------------------------------------------------------------
# E4: K_C proxy validation
# ---------------------------------------------------------------------------

def aggregate_e4() -> None:
    print("\n=== E4: K_C proxy validation ===")
    corr_csv = OUT_DIR / "kc_proxy_correlation.csv"
    if not corr_csv.exists():
        print(f"  ERROR: {corr_csv} not found. Run run_phase4_kc.py first.")
        return

    df = pd.read_csv(corr_csv)
    if df.empty:
        print("  Empty correlation CSV.")
        return

    print(df.to_string(index=False))

    out = OUT_DIR / "e4_kc_validation_summary.csv"
    df.to_csv(out, index=False)
    print(f"\n  Wrote {out}")

    # Falsification check
    graph_lm = df[df["proxy"].isin(["kc_graph_score", "kc_lm_score"])]
    n_validated = 0
    for _, row in graph_lm.iterrows():
        rho = float(row.get("spearman_rho", 0))
        pval = float(row.get("p_value", 1.0))
        if abs(rho) >= 0.4 and pval < 0.05:
            n_validated += 1

    print(f"\n  (generator, proxy) pairs with |rho| >= 0.4, p < 0.05: {n_validated}")
    if n_validated > 0:
        print("  RESULT: K_C has partial inference-time empirical content.")
        print("  Section 6 K_C paragraph upgrades from 'theoretical scaffolding' to 'partial validation'.")
    else:
        print("  RESULT: Null holds. K_C inference-time readout remains theoretical.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate E1-E4 results")
    parser.add_argument("--skip-e1", action="store_true")
    parser.add_argument("--skip-e2", action="store_true")
    parser.add_argument("--skip-e3", action="store_true")
    parser.add_argument("--skip-e4", action="store_true")
    args = parser.parse_args(argv)

    if not args.skip_e1:
        aggregate_e1()
    if not args.skip_e2:
        aggregate_e2()
    if not args.skip_e3:
        aggregate_e3()
    if not args.skip_e4:
        aggregate_e4()

    print("\nAggregation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
