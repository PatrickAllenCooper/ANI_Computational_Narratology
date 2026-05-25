"""Aggregate Phase 10 (Narrative-Gradients) results.

Produces a cross-phase comparison table (Phase 9 optimised standard CoT vs.
Phase 10 optimised N-CoT vs. hand-written N-CoT) on the shared 30-scenario
held-out eval set (seed-42 indices 30-59).

Pre-registration discipline: run this script BEFORE inspecting the optimised
prompt verbatim.  The outputs are the registered results; post-hoc inspection
of the prompt diff must happen after these numbers are committed.

Outputs:
  divergence_study_outputs/ng_aggregate.json
  divergence_study_outputs/ng_holdout_compare.csv
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import random
import re
from pathlib import Path

OUT = Path("./divergence_study_outputs")

# Fixed for all phases (pilot tier)
PRIMARY_JUDGE = "claude-haiku-4-5"
GEN_MODEL_P9  = "gpt-5.4-nano"       # Phase 9 generator
GEN_MODEL_P10 = "claude-haiku-4-5"   # Phase 10 generator


def _hash(p: str) -> str:
    return hashlib.sha1(p.encode("utf-8")).hexdigest()[:10]


def cliffs_delta(
    a: list[float], b: list[float], n_boot: int = 1000
) -> tuple[float, tuple[float, float]]:
    """Cliff's delta of A vs B (positive = A stochastically greater than B).

    Returns (delta, (ci_lo, ci_hi)) with bootstrap 95% CI.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0, (0.0, 0.0)
    gt = lt = 0
    for x in a:
        for y in b:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    delta = (gt - lt) / (n * m)

    rng = random.Random(42)
    boot_deltas: list[float] = []
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
        boot_deltas.append((g - l) / (n * m))
    boot_deltas.sort()
    lo = boot_deltas[int(0.025 * n_boot)]
    hi = boot_deltas[int(0.975 * n_boot) - 1]
    return delta, (lo, hi)


def cohens_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    p_a = sum(a) / n
    p_b = sum(b) / n
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if abs(1 - expected) < 1e-9:
        return 1.0
    return (observed - expected) / (1 - expected)


def _load_ng_judge_codes(
    prompt_hash: str, gen_model: str, judge_prefix: str
) -> list[dict]:
    """Load ng_judge_{prefix}_* files matching a given prompt hash and gen model.

    judge_prefix is 'p' (primary) or 't' (third).
    """
    pat = re.compile(
        rf"ng_judge_{re.escape(judge_prefix)}_{re.escape(PRIMARY_JUDGE)}"
        rf"_{re.escape(_safe(gen_model))}_(.+?)_{re.escape(prompt_hash)}\.json$"
    )
    if judge_prefix == "t":
        # third judge may be grok; match any judge name
        pat = re.compile(
            rf"ng_judge_t_[^_]+_{re.escape(_safe(gen_model))}"
            rf"_(.+?)_{re.escape(prompt_hash)}\.json$"
        )
    codes: list[dict] = []
    for path in sorted(OUT.glob(f"ng_judge_{judge_prefix}_*_{_safe(gen_model)}_*_{prompt_hash}.json")):
        m = pat.match(path.name)
        if not m:
            continue
        d = json.loads(path.read_text())
        codes.append({
            "scenario_id":       m.group(1),
            "sc":  int(d.get("stakeholder_count", 0) or 0),
            "us":  int(d.get("uncertainty_score",  0) or 0),
            "mh":  int(d.get("max_causal_hops",    0) or 0),
            "collapse_fired":    int(int(d.get("stakeholder_count", 0) or 0) <= 1),
            "suppression_fired": int(int(d.get("uncertainty_score", 0) or 0) == 0),
        })
    return codes


def _load_tg_judge_codes(prompt_hash: str, gen_model: str) -> list[dict]:
    """Load Phase-9 tg_judge_* files for a given prompt hash (gpt-5.4-nano)."""
    pat = re.compile(
        rf"tg_judge_{re.escape(PRIMARY_JUDGE)}_{re.escape(_safe(gen_model))}"
        rf"_(.+?)_([0-9a-f]{{10}})\.json$"
    )
    codes: list[dict] = []
    for path in sorted(OUT.glob(f"tg_judge_{PRIMARY_JUDGE}_{_safe(gen_model)}_*_{prompt_hash}.json")):
        m = pat.match(path.name)
        if not m:
            continue
        d = json.loads(path.read_text())
        codes.append({
            "scenario_id":       m.group(1),
            "sc":  int(d.get("stakeholder_count", 0) or 0),
            "us":  int(d.get("uncertainty_score",  0) or 0),
            "mh":  int(d.get("max_causal_hops",    0) or 0),
            "collapse_fired":    int(int(d.get("stakeholder_count", 0) or 0) <= 1),
            "suppression_fired": int(int(d.get("uncertainty_score", 0) or 0) == 0),
        })
    return codes


def _safe(s: str) -> str:
    import re as _re
    return _re.sub(r"[^A-Za-z0-9._-]", "_", s)


def _filter_to_eval_ids(codes: list[dict], eval_ids: set[str]) -> list[dict]:
    return [c for c in codes if c["scenario_id"] in eval_ids]


def _means(codes: list[dict]) -> dict:
    n = max(1, len(codes))
    return {
        "stakeholder_count": sum(c["sc"] for c in codes) / n,
        "uncertainty_score": sum(c["us"] for c in codes) / n,
        "max_causal_hops":   sum(c["mh"] for c in codes) / n,
        "collapse_rate":     sum(c["collapse_fired"]     for c in codes) / n,
        "suppression_rate":  sum(c["suppression_fired"]  for c in codes) / n,
        "n":                 n,
    }


def main() -> None:
    from scripts.run_phase1_quartet import PROMPTS, load_daily_dilemmas

    # Load scenario IDs for the shared held-out eval set
    all_scenarios  = load_daily_dilemmas(n=100)
    ng_summary     = json.loads((OUT / "ng_summary.json").read_text())
    eval_start, eval_end = ng_summary["eval_indices"]
    eval_ids = {s.id for s in all_scenarios[eval_start:eval_end]}
    print(f"Held-out eval set: {len(eval_ids)} scenarios "
          f"(indices {eval_start}-{eval_end - 1})")

    # ----- Phase 10 data -----
    ng_opt_hash  = _hash(ng_summary["final_prompt"])
    ng_init_hash = _hash(PROMPTS["narrative_cot"])   # hand N-CoT (control in Phase 10)

    p10_opt_codes   = _filter_to_eval_ids(
        _load_ng_judge_codes(ng_opt_hash,  GEN_MODEL_P10, "p"), eval_ids
    )
    p10_ncot_codes  = _filter_to_eval_ids(
        _load_ng_judge_codes(ng_init_hash, GEN_MODEL_P10, "p"), eval_ids
    )
    p10_third_codes = _filter_to_eval_ids(
        _load_ng_judge_codes(ng_opt_hash,  GEN_MODEL_P10, "t"), eval_ids
    )

    print(f"Phase 10: optimised N-CoT codes on eval: {len(p10_opt_codes)}")
    print(f"Phase 10: hand N-CoT codes on eval:      {len(p10_ncot_codes)}")
    print(f"Phase 10: third-judge codes on eval:     {len(p10_third_codes)}")

    # ----- Phase 9 data (Phase 9 gen_model = gpt-5.4-nano) -----
    tg9_summary = None
    tg9_path    = OUT / "tg_summary.json"
    p9_codes: list[dict] = []
    if tg9_path.exists():
        tg9_summary = json.loads(tg9_path.read_text())
        p9_hash     = _hash(tg9_summary["final_prompt"])
        p9_codes    = _filter_to_eval_ids(
            _load_tg_judge_codes(p9_hash, GEN_MODEL_P9), eval_ids
        )
        print(f"Phase 9:  TG-optimised std-CoT codes on eval: {len(p9_codes)}")
    else:
        print("Phase 9 summary not found; cross-phase comparison will be partial.")

    # ----- Summary statistics -----
    m_p10_opt   = _means(p10_opt_codes)
    m_p10_ncot  = _means(p10_ncot_codes)
    m_p10_third = _means(p10_third_codes)
    m_p9        = _means(p9_codes) if p9_codes else None

    print("\n=== Held-out means ===")
    for label, m in [
        ("Phase 10 optimised N-CoT (primary)",  m_p10_opt),
        ("Phase 10 hand N-CoT    (primary)",    m_p10_ncot),
        ("Phase 10 optimised N-CoT (3rd judge)", m_p10_third),
    ] + ([("Phase 9 TG-opt std-CoT (primary)", m_p9)] if m_p9 else []):
        print(f"  {label:50s}  "
              f"sc={m['stakeholder_count']:.2f}  "
              f"us={m['uncertainty_score']:.2f}  "
              f"mh={m['max_causal_hops']:.2f}  "
              f"n={m['n']}")

    # ----- Cliff's delta -----
    print("\n=== Cliff's delta (positive = first wins) ===")
    delta_results: dict = {}

    # Primary contrast: hand N-CoT vs optimised N-CoT (positive favours hand)
    for metric, key in [
        ("stakeholder_count", "sc"),
        ("uncertainty_score", "us"),
        ("max_causal_hops",   "mh"),
    ]:
        a = [c[key] for c in p10_ncot_codes]
        b = [c[key] for c in p10_opt_codes]
        d, (lo, hi) = cliffs_delta(a, b)
        delta_results[f"hand_ncot_vs_opt_ncot_{metric}"] = {
            "delta": d, "ci_lo": lo, "ci_hi": hi,
        }
        print(f"  hand N-CoT vs opt N-CoT  {metric:25s}: "
              f"delta={d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

    # Secondary contrast: Phase 9 vs Phase 10 (positive favours Phase 10)
    if p9_codes:
        for metric, key in [
            ("stakeholder_count", "sc"),
            ("uncertainty_score", "us"),
            ("max_causal_hops",   "mh"),
        ]:
            a = [c[key] for c in p10_opt_codes]
            b = [c[key] for c in p9_codes]
            d, (lo, hi) = cliffs_delta(a, b)
            delta_results[f"p10_opt_ncot_vs_p9_tg_stdcot_{metric}"] = {
                "delta": d, "ci_lo": lo, "ci_hi": hi,
            }
            print(f"  P10 opt N-CoT vs P9 TG   {metric:25s}: "
                  f"delta={d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

    # ----- Inter-judge kappa -----
    kappa_collapse = cohens_kappa(
        [c["collapse_fired"]     for c in p10_opt_codes],
        [c["collapse_fired"]     for c in p10_third_codes],
    )
    kappa_suppression = cohens_kappa(
        [c["suppression_fired"]  for c in p10_opt_codes],
        [c["suppression_fired"]  for c in p10_third_codes],
    )
    print(f"\n=== Inter-judge kappa (primary={PRIMARY_JUDGE} vs third) ===")
    print(f"  collapse_fired:     kappa={kappa_collapse:.3f}")
    print(f"  suppression_fired:  kappa={kappa_suppression:.3f}")

    # ----- Drift control -----
    drift = ng_summary.get("drift_control", {})
    print(f"\n=== Drift control ===")
    print(f"  Scenario: {drift.get('scenario_id','?')}  "
          f"start_sc={drift.get('start_sc','?')}  "
          f"end_sc={drift.get('end_sc','?')}  "
          f"delta_sc={drift.get('delta_sc','?')}  "
          f"confounded={drift.get('confounded','?')}")

    # ----- Outcome interpretation -----
    d_sc = delta_results.get("hand_ncot_vs_opt_ncot_stakeholder_count", {}).get("delta", float("nan"))
    d_us = delta_results.get("hand_ncot_vs_opt_ncot_uncertainty_score",  {}).get("delta", float("nan"))
    if abs(d_sc) < 0.1 and abs(d_us) < 0.1:
        outcome = "CONVERGENCE"
    elif d_sc < -0.2 or d_us < -0.2:
        outcome = "IMPROVEMENT"
    elif d_sc > 0.2 or d_us > 0.2:
        outcome = "DEGRADATION"
    else:
        outcome = "INCONCLUSIVE"
    print(f"\n=== Pre-declared outcome: {outcome} ===")
    print(f"  (hand-vs-opt delta_sc={d_sc:+.3f}, delta_us={d_us:+.3f}; "
          f"thresholds: |delta|<0.1 convergence, <-0.2 improvement, >+0.2 degradation)")

    # ----- Write aggregate JSON -----
    aggregate = {
        "n_eval_scenarios": len(eval_ids),
        "outcome":          outcome,
        "eval_indices":     [eval_start, eval_end],
        "p10": {
            "gen_model":     ng_summary["gen_model"],
            "judge_model":   ng_summary["judge_model"],
            "third_judge":   ng_summary.get("third_judge", DEFAULT_THIRD_JUDGE),
            "n_iters_run":   ng_summary["n_iters_run"],
            "opt_ncot_means":  m_p10_opt,
            "hand_ncot_means": m_p10_ncot,
            "third_judge_means": m_p10_third,
            "inter_judge_kappa": {
                "collapse":    kappa_collapse,
                "suppression": kappa_suppression,
            },
            "drift_control": drift,
        },
        "p9": {
            "gen_model":     GEN_MODEL_P9,
            "tg_opt_means":  m_p9,
        } if m_p9 else None,
        "cliffs_delta": delta_results,
        "training_loss_curve": ng_summary["training_loss_curve"],
    }
    agg_path = OUT / "ng_aggregate.json"
    agg_path.write_text(json.dumps(aggregate, indent=2, default=str))
    print(f"\nAggregate JSON: {agg_path}")

    # ----- Write holdout comparison CSV -----
    csv_path = OUT / "ng_holdout_compare.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "condition", "judge", "n",
            "mean_sc", "mean_us", "mean_mh",
            "collapse_rate", "suppression_rate",
        ])
        rows = [
            ("P10_optimised_N-CoT", "primary",  m_p10_opt),
            ("P10_hand_N-CoT",      "primary",  m_p10_ncot),
            ("P10_optimised_N-CoT", "3rd",       m_p10_third),
        ]
        if m_p9:
            rows.append(("P9_TG-opt_stdCoT", "primary", m_p9))
        for label, judge, m in rows:
            w.writerow([
                label, judge, m["n"],
                round(m["stakeholder_count"], 3),
                round(m["uncertainty_score"],  3),
                round(m["max_causal_hops"],    3),
                round(m["collapse_rate"],      4),
                round(m["suppression_rate"],   4),
            ])
    print(f"Holdout CSV:   {csv_path}")


DEFAULT_THIRD_JUDGE = "grok-4-1-fast-reasoning"


if __name__ == "__main__":
    main()
