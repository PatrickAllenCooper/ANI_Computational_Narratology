"""
scripts/aggregate_followup.py -- Follow-up paper registered analysis.

Consumes Phase 1 cached judge cells for conditions narrative_cot_v2
(v2) and narrative_cot (v1, already cached from Experiment 1) across all
four generators, plus third-judge cells written by run_third_judge_v2.py.

Pre-registered analyses (from Guidance_Documents/followup_study_design.md,
executed in declared order):
  1. Per-generator Cliff's delta (v1 vs v2) with bootstrap 95% CI on
     stakeholder_count, uncertainty_score, max_causal_hops, and
     length-residualised versions of sc and us.
  2. Per-generator binary firing-rate change (v2 - v1) for collapse and
     suppression with binomial CIs.
  3. Per-generator mean-token ratio (v2 / v1).
  4. Per-generator inter-judge Cohen's kappa (primary vs third) on v2 binary
     labels.
  5. Cross-generator H1 verdict (count of generators with sc delta < -0.1
     and CI hi < 0).

Outputs:
  divergence_study_outputs/followup_aggregate.json
  divergence_study_outputs/followup_per_generator.csv

Run discipline: execute before drafting any paper prose; the last line of
stdout is the pre-declared outcome verdict.
"""
from __future__ import annotations

import csv
import json
import math
import random
import re
from pathlib import Path

OUT  = Path("./divergence_study_outputs")
COND_V1 = "narrative_cot"
COND_V2 = "narrative_cot_v2"

PRIMARY_JUDGE  = "claude-haiku-4-5"
THIRD_JUDGE    = "grok-4-1-fast-reasoning"
GENERATORS     = [
    "gpt-5.4-nano",
    "claude-haiku-4-5",
    "grok-4-1-fast-reasoning",
    "claude-sonnet-4-6",
]

from scripts.run_phase1_quartet import (
    N_PER_GENERATOR, DEFAULT_N_FALLBACK, _safe,
    judge_cache_path, load_daily_dilemmas,
)
from scripts.run_third_judge_v2 import judge3_cache_path, CONDITION as V2_CONDITION


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def cliffs_delta(
    a: list[float], b: list[float], n_boot: int = 1000
) -> tuple[float, tuple[float, float]]:
    """Cliff's delta of A vs B; positive = A stochastically greater."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("nan"), (float("nan"), float("nan"))
    gt = lt = 0
    for x in a:
        for y in b:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    delta = (gt - lt) / (n * m)
    rng = random.Random(42)
    boot: list[float] = []
    for _ in range(n_boot):
        ai = [a[rng.randrange(n)] for _ in range(n)]
        bi = [b[rng.randrange(m)] for _ in range(m)]
        g = l = 0
        for x in ai:
            for y in bi:
                if x > y:
                    g += 1
                elif x < y:
                    l += 1
        boot.append((g - l) / (n * m))
    boot.sort()
    lo = boot[int(0.025 * n_boot)]
    hi = boot[int(0.975 * n_boot) - 1]
    return delta, (lo, hi)


def cohens_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    obs  = sum(1 for x, y in zip(a, b) if x == y) / n
    p_a  = sum(a) / n
    p_b  = sum(b) / n
    exp  = p_a * p_b + (1 - p_a) * (1 - p_b)
    return (obs - exp) / (1 - exp) if abs(1 - exp) > 1e-9 else 1.0


def binomial_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval."""
    if n == 0:
        return (0.0, 1.0)
    p    = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half   = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def ols_residualise(y: list[float], log_tokens: list[float]) -> list[float]:
    """OLS-residualise y on log_tokens; return residuals."""
    n = len(y)
    if n < 2:
        return y[:]
    mx = sum(log_tokens) / n
    my = sum(y) / n
    ss_x = sum((x - mx) ** 2 for x in log_tokens)
    if ss_x < 1e-12:
        return [yi - my for yi in y]
    b1 = sum((x - mx) * (yi - my) for x, yi in zip(log_tokens, y)) / ss_x
    b0 = my - b1 * mx
    return [yi - (b0 + b1 * x) for x, yi in zip(log_tokens, y)]


# ---------------------------------------------------------------------------
# Cache loaders
# ---------------------------------------------------------------------------

def load_judge_cells(
    gen_model: str, condition: str, scenarios, n_samples: int, judge_model: str
) -> list[dict]:
    """Load all primary-judge cells for (gen_model, condition)."""
    cells: list[dict] = []
    for scenario in scenarios:
        for idx in range(n_samples):
            path = judge_cache_path(judge_model, gen_model,
                                    scenario.id, condition, idx)
            if path.exists():
                d = json.loads(path.read_text())
                sc  = int(d.get("stakeholder_count", 0) or 0)
                us  = int(d.get("uncertainty_score",  0) or 0)
                mh  = int(d.get("max_causal_hops",    0) or 0)
                ct_raw = None
                # completion_tokens lives in the gen cache; load it if available
                gen_path = OUT / (
                    f"gen_{_safe(gen_model)}_{scenario.id}_{condition}"
                    f"_{idx:03d}_"
                ) 
                # find the matching gen file by prefix glob
                gen_files = list(OUT.glob(
                    f"gen_{_safe(gen_model)}_{scenario.id}_{condition}_{idx:03d}_*.json"
                ))
                if gen_files:
                    gr = json.loads(gen_files[0].read_text())
                    ct_raw = int(gr.get("completion_tokens", 0) or 0)
                cells.append({
                    "scenario_id":       scenario.id,
                    "sample_idx":        idx,
                    "sc":                sc,
                    "us":                us,
                    "mh":                mh,
                    "completion_tokens": ct_raw or 0,
                    "collapse_fired":    int(sc <= 1),
                    "suppression_fired": int(us == 0),
                })
    return cells


def load_third_judge_cells(
    gen_model: str, scenarios, n_samples: int
) -> list[dict]:
    """Load third-judge (grok) cells for v2 condition."""
    cells: list[dict] = []
    for scenario in scenarios:
        for idx in range(n_samples):
            path = judge3_cache_path(gen_model, scenario.id, idx)
            if path.exists():
                d = json.loads(path.read_text())
                sc = int(d.get("stakeholder_count", 0) or 0)
                us = int(d.get("uncertainty_score",  0) or 0)
                cells.append({
                    "scenario_id":       scenario.id,
                    "sample_idx":        idx,
                    "sc":                sc,
                    "us":                us,
                    "collapse_fired":    int(sc <= 1),
                    "suppression_fired": int(us == 0),
                })
    return cells


# ---------------------------------------------------------------------------
# Per-generator analysis
# ---------------------------------------------------------------------------

def analyse_generator(
    gen_model: str, scenarios, n: int
) -> dict:
    v1_cells = load_judge_cells(gen_model, COND_V1, scenarios, n, PRIMARY_JUDGE)
    v2_cells = load_judge_cells(gen_model, COND_V2, scenarios, n, PRIMARY_JUDGE)
    v2_third = load_third_judge_cells(gen_model, scenarios, n)

    if not v1_cells or not v2_cells:
        return {
            "gen_model": gen_model, "n_v1": len(v1_cells),
            "n_v2": len(v2_cells), "status": "MISSING_CELLS",
        }

    v1_sc  = [c["sc"] for c in v1_cells]
    v1_us  = [c["us"] for c in v1_cells]
    v1_mh  = [c["mh"] for c in v1_cells]
    v1_ct  = [c["completion_tokens"] for c in v1_cells]

    v2_sc  = [c["sc"] for c in v2_cells]
    v2_us  = [c["us"] for c in v2_cells]
    v2_mh  = [c["mh"] for c in v2_cells]
    v2_ct  = [c["completion_tokens"] for c in v2_cells]

    # 1. Cliff's delta (v1 vs v2); positive = v1 wins, negative = v2 wins
    d_sc,  ci_sc  = cliffs_delta(v1_sc, v2_sc)
    d_us,  ci_us  = cliffs_delta(v1_us, v2_us)
    d_mh,  ci_mh  = cliffs_delta(v1_mh, v2_mh)

    # 1d/1e. Length-residualised deltas (OLS on log completion_tokens)
    combined_ct  = v1_ct + v2_ct
    log_ct       = [math.log(max(t, 1)) for t in combined_ct]
    combined_sc  = v1_sc + v2_sc
    combined_us  = v1_us + v2_us
    resid_sc     = ols_residualise(combined_sc, log_ct)
    resid_us     = ols_residualise(combined_us, log_ct)
    n1 = len(v1_sc)
    v1_rsc = resid_sc[:n1];  v2_rsc = resid_sc[n1:]
    v1_rus = resid_us[:n1];  v2_rus = resid_us[n1:]
    d_rsc, ci_rsc = cliffs_delta(v1_rsc, v2_rsc)
    d_rus, ci_rus = cliffs_delta(v1_rus, v2_rus)

    # 2. Binary firing-rate deltas
    def rate(lst, key): return sum(c[key] for c in lst) / max(len(lst), 1)
    v1_col  = rate(v1_cells, "collapse_fired")
    v2_col  = rate(v2_cells, "collapse_fired")
    v1_sup  = rate(v1_cells, "suppression_fired")
    v2_sup  = rate(v2_cells, "suppression_fired")
    n_v2    = len(v2_cells)
    k_col   = sum(c["collapse_fired"]     for c in v2_cells)
    k_sup   = sum(c["suppression_fired"]  for c in v2_cells)
    ci_col2 = binomial_ci(k_col, n_v2)
    ci_sup2 = binomial_ci(k_sup, n_v2)

    # 3. Mean-token ratio
    mean_v1_ct = sum(v1_ct) / max(len(v1_ct), 1)
    mean_v2_ct = sum(v2_ct) / max(len(v2_ct), 1)
    token_ratio = mean_v2_ct / max(mean_v1_ct, 1)

    # 4. Inter-judge kappa on v2 (primary vs third)
    if v2_third:
        # align by (scenario_id, sample_idx)
        v2_idx  = {(c["scenario_id"], c["sample_idx"]): c for c in v2_cells}
        t_idx   = {(c["scenario_id"], c["sample_idx"]): c for c in v2_third}
        common  = sorted(set(v2_idx) & set(t_idx))
        prim_col = [v2_idx[k]["collapse_fired"]    for k in common]
        prim_sup = [v2_idx[k]["suppression_fired"]  for k in common]
        third_col= [t_idx[k]["collapse_fired"]     for k in common]
        third_sup= [t_idx[k]["suppression_fired"]   for k in common]
        kappa_col = cohens_kappa(prim_col, third_col)
        kappa_sup = cohens_kappa(prim_sup, third_sup)
        n_kappa   = len(common)
        # v2 sc delta under third judge
        t_sc = [t_idx[k]["sc"] for k in common]
        p_sc = [v2_idx[k]["sc"] for k in common]
        # third-judge advantage: v1 mean vs v2 third mean
        v1_mean_sc = sum(v1_sc) / max(len(v1_sc), 1)
        t_mean_sc  = sum(t_sc)  / max(len(t_sc),  1)
        third_delta_sc = v1_mean_sc - t_mean_sc   # positive = v1 wins
    else:
        kappa_col = kappa_sup = float("nan")
        n_kappa   = 0
        t_mean_sc = float("nan")
        third_delta_sc = float("nan")

    # H1 threshold: delta < -0.1 AND ci_hi < 0
    h1_met = (d_sc < -0.1) and (ci_sc[1] < 0)

    # Means
    v1_means = {
        "sc":  sum(v1_sc)  / len(v1_sc),
        "us":  sum(v1_us)  / len(v1_us),
        "mh":  sum(v1_mh)  / len(v1_mh),
        "ct":  mean_v1_ct,
    }
    v2_means = {
        "sc":  sum(v2_sc)  / len(v2_sc),
        "us":  sum(v2_us)  / len(v2_us),
        "mh":  sum(v2_mh)  / len(v2_mh),
        "ct":  mean_v2_ct,
    }

    return {
        "gen_model":      gen_model,
        "n_v1":           len(v1_cells),
        "n_v2":           len(v2_cells),
        "n_third":        len(v2_third),
        "n_kappa":        n_kappa,
        "status":         "OK",
        "v1_means":       v1_means,
        "v2_means":       v2_means,
        "cliffs_delta": {
            "sc":     {"delta": d_sc,  "ci_lo": ci_sc[0],  "ci_hi": ci_sc[1]},
            "us":     {"delta": d_us,  "ci_lo": ci_us[0],  "ci_hi": ci_us[1]},
            "mh":     {"delta": d_mh,  "ci_lo": ci_mh[0],  "ci_hi": ci_mh[1]},
            "sc_resid": {"delta": d_rsc, "ci_lo": ci_rsc[0], "ci_hi": ci_rsc[1]},
            "us_resid": {"delta": d_rus, "ci_lo": ci_rus[0], "ci_hi": ci_rus[1]},
        },
        "firing_rates": {
            "v1_collapse":    v1_col, "v2_collapse":    v2_col,
            "delta_collapse": v2_col - v1_col,
            "v2_collapse_ci": ci_col2,
            "v1_suppression": v1_sup, "v2_suppression": v2_sup,
            "delta_suppression": v2_sup - v1_sup,
            "v2_suppression_ci": ci_sup2,
        },
        "token_ratio":    token_ratio,
        "inter_judge": {
            "kappa_collapse":    kappa_col,
            "kappa_suppression": kappa_sup,
            "n_common":          n_kappa,
            "third_mean_sc":     t_mean_sc,
            "primary_delta_sc":  d_sc,      # v1 vs v2 under primary
            "third_delta_sc_raw": third_delta_sc,  # v1_mean - third_v2_mean
        },
        "h1_met": h1_met,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading 100 DailyDilemmas scenarios...", flush=True)
    scenarios = load_daily_dilemmas(n=100)
    print(f"  {len(scenarios)} scenarios.", flush=True)

    results: list[dict] = []
    for gen_model in GENERATORS:
        n = N_PER_GENERATOR.get(gen_model, DEFAULT_N_FALLBACK)
        print(f"\n--- {gen_model}  (N={n} per cell) ---", flush=True)
        r = analyse_generator(gen_model, scenarios, n)
        results.append(r)

        if r["status"] != "OK":
            print(f"  STATUS: {r['status']}  (v1={r['n_v1']}, v2={r['n_v2']})")
            continue

        cd = r["cliffs_delta"]
        fr = r["firing_rates"]
        ij = r["inter_judge"]
        print(f"  cells: v1={r['n_v1']}  v2={r['n_v2']}  third={r['n_third']}")
        print(f"  v1 means: sc={r['v1_means']['sc']:.2f}  "
              f"us={r['v1_means']['us']:.2f}  mh={r['v1_means']['mh']:.2f}  "
              f"ct={r['v1_means']['ct']:.0f}")
        print(f"  v2 means: sc={r['v2_means']['sc']:.2f}  "
              f"us={r['v2_means']['us']:.2f}  mh={r['v2_means']['mh']:.2f}  "
              f"ct={r['v2_means']['ct']:.0f}")
        print(f"  Cliff delta (v1 vs v2, negative=v2 wins):")
        print(f"    sc       {cd['sc']['delta']:+.3f}  "
              f"CI [{cd['sc']['ci_lo']:+.3f}, {cd['sc']['ci_hi']:+.3f}]"
              f"  H1={'MET' if r['h1_met'] else 'not met'}")
        print(f"    us       {cd['us']['delta']:+.3f}  "
              f"CI [{cd['us']['ci_lo']:+.3f}, {cd['us']['ci_hi']:+.3f}]")
        print(f"    mh       {cd['mh']['delta']:+.3f}  "
              f"CI [{cd['mh']['ci_lo']:+.3f}, {cd['mh']['ci_hi']:+.3f}]")
        print(f"    sc_resid {cd['sc_resid']['delta']:+.3f}  "
              f"CI [{cd['sc_resid']['ci_lo']:+.3f}, {cd['sc_resid']['ci_hi']:+.3f}]")
        print(f"    us_resid {cd['us_resid']['delta']:+.3f}  "
              f"CI [{cd['us_resid']['ci_lo']:+.3f}, {cd['us_resid']['ci_hi']:+.3f}]")
        print(f"  firing rates:  collapse v1={fr['v1_collapse']:.1%}  "
              f"v2={fr['v2_collapse']:.1%}  delta={fr['delta_collapse']:+.1%}")
        print(f"                suppression v1={fr['v1_suppression']:.1%}  "
              f"v2={fr['v2_suppression']:.1%}  delta={fr['delta_suppression']:+.1%}")
        print(f"  token ratio (v2/v1): {r['token_ratio']:.2f}x  "
              f"H4={'MET' if r['token_ratio'] < 2.0 else 'FAIL'}")
        print(f"  inter-judge kappa: collapse={ij['kappa_collapse']:.3f}  "
              f"suppression={ij['kappa_suppression']:.3f}  "
              f"(n={ij['n_common']})")

    # Cross-generator verdict
    ok = [r for r in results if r["status"] == "OK"]
    h1_count = sum(1 for r in ok if r["h1_met"])
    h2_ok    = all(
        r["firing_rates"]["delta_collapse"] <= 0
        and r["firing_rates"]["delta_suppression"] <= 0
        for r in ok
    )
    h3_ok    = all(
        r["inter_judge"]["kappa_collapse"]    >= 0.70
        and r["inter_judge"]["kappa_suppression"] >= 0.70
        for r in ok if r["n_third"] > 0
    )
    h4_ok    = all(r["token_ratio"] < 2.0 for r in ok)

    if h1_count >= 3 and h3_ok:
        verdict = "REPLICATES"
    elif h1_count == 2:
        verdict = "PARTIAL"
    else:
        verdict = "DOES_NOT_REPLICATE"

    print(f"\n{'='*60}")
    print(f"H1 generators meeting threshold: {h1_count} / {len(ok)}")
    print(f"H2 (no regression):   {'MET' if h2_ok else 'FAIL'}")
    print(f"H3 (kappa >= 0.70):   {'MET' if h3_ok else 'FAIL (or no third-judge data)'}")
    print(f"H4 (token ratio <2x): {'MET' if h4_ok else 'FAIL'}")
    print(f"\nPRE-DECLARED OUTCOME: {verdict}")
    print(f"{'='*60}\n")

    # Write aggregate JSON
    aggregate = {
        "verdict":     verdict,
        "h1_count":    h1_count,
        "h2_met":      h2_ok,
        "h3_met":      h3_ok,
        "h4_met":      h4_ok,
        "n_generators": len(ok),
        "per_generator": results,
    }
    agg_path = OUT / "followup_aggregate.json"
    agg_path.write_text(json.dumps(aggregate, indent=2, default=str))
    print(f"Aggregate JSON: {agg_path}")

    # Write per-generator CSV
    csv_path = OUT / "followup_per_generator.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "generator", "n_v1", "n_v2",
            "v1_mean_sc", "v2_mean_sc", "delta_sc", "ci_lo_sc", "ci_hi_sc",
            "h1_met",
            "v1_mean_us", "v2_mean_us", "delta_us",
            "v1_mean_mh", "v2_mean_mh", "delta_mh",
            "delta_sc_resid", "delta_us_resid",
            "v1_collapse", "v2_collapse",
            "v1_suppression", "v2_suppression",
            "token_ratio",
            "kappa_collapse", "kappa_suppression",
        ])
        for r in results:
            if r["status"] != "OK":
                w.writerow([r["gen_model"], r["n_v1"], r["n_v2"]] + ["N/A"] * 21)
                continue
            cd = r["cliffs_delta"]
            fr = r["firing_rates"]
            ij = r["inter_judge"]
            w.writerow([
                r["gen_model"], r["n_v1"], r["n_v2"],
                round(r["v1_means"]["sc"], 3), round(r["v2_means"]["sc"], 3),
                round(cd["sc"]["delta"],  3),
                round(cd["sc"]["ci_lo"],  3), round(cd["sc"]["ci_hi"], 3),
                r["h1_met"],
                round(r["v1_means"]["us"], 3), round(r["v2_means"]["us"], 3),
                round(cd["us"]["delta"],  3),
                round(r["v1_means"]["mh"], 3), round(r["v2_means"]["mh"], 3),
                round(cd["mh"]["delta"],  3),
                round(cd["sc_resid"]["delta"], 3),
                round(cd["us_resid"]["delta"], 3),
                round(fr["v1_collapse"],    4), round(fr["v2_collapse"],    4),
                round(fr["v1_suppression"], 4), round(fr["v2_suppression"], 4),
                round(r["token_ratio"],     3),
                round(ij["kappa_collapse"],    3),
                round(ij["kappa_suppression"], 3),
            ])
    print(f"Per-generator CSV: {csv_path}")


if __name__ == "__main__":
    main()
