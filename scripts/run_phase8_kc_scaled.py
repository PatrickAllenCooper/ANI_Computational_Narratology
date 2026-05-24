"""
scripts/run_phase8_kc_scaled.py -- E6: Scale registered K_graph proxy to full Phase-1 cache.

Scales $\hat{K}_{\text{graph}}$ from the 30-scenario subsample (E4) to all 100
Phase-1 scenarios across the full four-generator quartet. For each cached generation
file we extract the causal graph (one API call to claude-haiku-4-5) and cache the
result so reruns skip already-processed items.

Also computes the E7 SCM-level convergent proxies (kc_graph_mdl, kc_struct_entropy)
from the same extracted graphs at zero additional API cost.

Output:
  divergence_study_outputs/kc_graph_scaled.csv  -- per-row scores
  divergence_study_outputs/kc_graph_scaled_corr.csv  -- Spearman rho per generator + pooled

Usage:
  python -m scripts.run_phase8_kc_scaled [--scenarios N] [--samples-per-cell N] [--workers N]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy.stats as stats

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.kc_proxy import kc_graph, kc_gzip, kc_graph_mdl, kc_struct_entropy

OUT_DIR = Path("./divergence_study_outputs")

CONDITION_DIRECTION = {
    "narrative_cot": 1,
    "standard_cot": -1,
    "baseline_io": 0,
    "standard_cot_verbose": -1,
    "standard_cot_refusal_tuned": -1,
}

EXTRACTOR_MODEL = "claude-haiku-4-5"


def _load_phase1_gen_files(scenarios: int, samples_per_cell: int) -> list[dict]:
    """Load a representative sample of cached gen_*.json files.

    For each (scenario_id, generator, condition) cell, pick `samples_per_cell`
    cached files deterministically.
    """
    files = sorted(OUT_DIR.glob("gen_*.json"))
    # Filter: only Phase-1 standard condition files (not refusal bench)
    files = [f for f in files if not f.name.startswith("gen_refusal_")]

    # Group by (scenario_id, generator, condition)
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for f in files:
        try:
            rec = json.loads(f.read_text())
            sid = rec.get("scenario_id", "")
            gen = rec.get("model", rec.get("generator", ""))
            cond = rec.get("condition", "")
            if sid and gen and cond:
                groups[(sid, gen, cond)].append((f, rec))
        except Exception:
            pass

    # Limit to first `scenarios` unique scenario IDs (sorted for determinism)
    all_sids = sorted({k[0] for k in groups})
    keep_sids = set(all_sids[:scenarios]) if scenarios > 0 else set(all_sids)

    rows = []
    for (sid, gen, cond), entries in groups.items():
        if sid not in keep_sids:
            continue
        # Take deterministic subsample by sorting filenames
        entries_sorted = sorted(entries, key=lambda x: x[0].name)
        for f, rec in entries_sorted[:samples_per_cell]:
            rows.append({
                "file": f,
                "scenario_id": sid,
                "generator": gen,
                "condition": cond,
                "output": rec.get("output", ""),
            })
    return rows


def _kc_cache_path(gen_file: Path) -> Path:
    return gen_file.parent / f"kc_graph_{gen_file.stem}.json"


def _atomic_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.replace(path)


def score_row(row: dict) -> dict | None:
    """Extract kc_graph for one row; return enriched row or None on error."""
    cache = _kc_cache_path(row["file"])
    output = row["output"]
    if not output:
        return None

    if cache.exists():
        try:
            cached = json.loads(cache.read_text())
            graph_data = cached
        except Exception:
            graph_data = None
    else:
        graph_data = None

    if graph_data is None:
        try:
            graph_data = kc_graph(output, extractor_model=EXTRACTOR_MODEL)
            _atomic_write(cache, graph_data)
        except Exception as e:
            print(f"  GRAPH ERROR {row['generator']}/{row['scenario_id']}/{row['condition']}: {e}", flush=True)
            return None

    # gzip (no API)
    gzip_data = kc_gzip(output)

    # E7 proxies from the extracted graph (no API)
    n_nodes = graph_data.get("n_nodes", 0)
    n_edges = graph_data.get("n_edges", 0)
    triples = graph_data.get("triples", [])
    mdl_data = kc_graph_mdl(n_nodes, n_edges, len(output))
    ent_data = kc_struct_entropy(triples)

    direction = CONDITION_DIRECTION.get(row["condition"], 0)

    return {
        "scenario_id": row["scenario_id"],
        "generator": row["generator"],
        "condition": row["condition"],
        "direction": direction,
        "output_len": len(output),
        "kc_graph_score": graph_data.get("kc_graph_score", float("nan")),
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "mean_degree": graph_data.get("mean_degree", 0.0),
        "kc_gzip_score": gzip_data["kc_gzip_score"],
        "kc_graph_mdl": mdl_data["kc_graph_mdl"],
        "kc_struct_entropy": ent_data["kc_struct_entropy"],
    }


def spearman_bootstrap(x: np.ndarray, y: np.ndarray, n_boot: int = 2000) -> dict:
    """Spearman rho with percentile bootstrap 95% CI."""
    rho, p = stats.spearmanr(x, y)
    n = len(x)
    rng = np.random.default_rng(42)
    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xi, yi = x[idx], y[idx]
        if len(set(xi)) < 2 or len(set(yi)) < 2:
            continue
        r, _ = stats.spearmanr(xi, yi)
        boot_rhos.append(r)
    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5]) if boot_rhos else (float("nan"), float("nan"))
    return {
        "rho": round(float(rho), 4),
        "p": round(float(p), 6),
        "ci_lo": round(float(ci_lo), 4),
        "ci_hi": round(float(ci_hi), 4),
        "n": n,
    }


def compute_correlations(rows: list[dict]) -> list[dict]:
    """Compute Spearman rho (per generator + pooled) for each proxy."""
    import pandas as pd
    df = pd.DataFrame(rows)
    # Keep only standard_cot and narrative_cot for directionality
    df = df[df["condition"].isin(["standard_cot", "narrative_cot"])].copy()
    if df.empty:
        return []

    proxies = ["kc_graph_score", "kc_graph_mdl", "kc_struct_entropy", "kc_gzip_score"]
    corr_rows = []
    for proxy in proxies:
        if proxy not in df.columns:
            continue
        sub = df[[proxy, "direction", "generator"]].dropna()
        if sub.empty:
            continue
        # Pooled
        pooled = spearman_bootstrap(sub[proxy].values, sub["direction"].values)
        corr_rows.append({"proxy": proxy, "generator": "POOLED", **pooled})
        # Per generator
        for gen, gdf in sub.groupby("generator"):
            if len(gdf) < 5:
                continue
            g_corr = spearman_bootstrap(gdf[proxy].values, gdf["direction"].values)
            corr_rows.append({"proxy": proxy, "generator": gen, **g_corr})
    return corr_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E6: Scale K_graph to full Phase-1 cache")
    parser.add_argument("--scenarios", type=int, default=100,
                        help="Number of unique scenario IDs to include (0=all)")
    parser.add_argument("--samples-per-cell", type=int, default=1,
                        help="Samples per (scenario, generator, condition) cell")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)

    print(f"Loading Phase-1 gen files (scenarios={args.scenarios}, samples/cell={args.samples_per_cell})...", flush=True)
    gen_rows = _load_phase1_gen_files(args.scenarios, args.samples_per_cell)
    print(f"  {len(gen_rows)} rows to score", flush=True)

    results = []
    done = 0
    total = len(gen_rows)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(score_row, row): row for row in gen_rows}
        for fut in as_completed(futs):
            done += 1
            result = fut.result()
            if result is not None:
                results.append(result)
            if done % 100 == 0 or done == total:
                print(f"  {done}/{total} rows processed ({len(results)} scored)", flush=True)

    print(f"\n{len(results)} rows successfully scored", flush=True)

    if not results:
        print("ERROR: no results", flush=True)
        return 1

    # Write per-row CSV
    out_csv = OUT_DIR / "kc_graph_scaled.csv"
    fieldnames = list(results[0].keys())
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {out_csv}", flush=True)

    # Compute correlations (E6 + E7)
    corr_rows = compute_correlations(results)
    corr_csv = OUT_DIR / "kc_graph_scaled_corr.csv"
    if corr_rows:
        with corr_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(corr_rows[0].keys()))
            writer.writeheader()
            writer.writerows(corr_rows)
        print(f"Wrote {corr_csv}", flush=True)

        print("\n=== K_C proxy correlations (Spearman rho vs N-CoT direction) ===", flush=True)
        for row in corr_rows:
            sig = "***" if row["p"] < 0.001 else ("**" if row["p"] < 0.01 else ("*" if row["p"] < 0.05 else ""))
            ci = f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]"
            print(f"  {row['proxy']:22s} {row['generator']:35s}  rho={row['rho']:+.3f}  p={row['p']:.4f}{sig:3s}  95%CI={ci}  n={row['n']}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
