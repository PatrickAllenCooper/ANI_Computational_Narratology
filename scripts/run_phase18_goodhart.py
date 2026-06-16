"""
scripts/run_phase18_goodhart.py -- Phase 18 Goodhart-gap measurement.

Re-scores the existing Phase 14 holdout responses (hand NoT vs optimised winner)
across a judge panel, quantifying how much of the single-judge "win" survives on
judges that were NOT used during optimisation.

No regeneration: only cached holdout generations are scored. If a generation
cache is missing the item is skipped (never re-generated), so this is the cheap
diagnostic that precedes building the full robust optimiser.

Output: divergence_study_outputs/phase18_goodhart.json

Usage:
  python -m scripts.run_phase18_goodhart
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.run_phase1_quartet import PROMPTS, OUT_DIR
from scripts.syco_loss import (
    SYCO_METRICS,
    _prompt_hash,
    _score_metric,
    gen_path,
    load_holdout_oeq,
)

GEN_MODEL = "claude-haiku-4-5"
TRAIN_JUDGE = "grok-4-1-fast-reasoning"
# Held-out judges: not used as the Phase 14 training judge.
HELDOUT_JUDGES = ["claude-haiku-4-5", "gpt-5.4-nano", "claude-sonnet-4-6"]
ALL_JUDGES = [TRAIN_JUDGE, *HELDOUT_JUDGES]


def _configs() -> list[tuple[str, str, str]]:
    """(label, prompt_text, namespace) matching cached holdout generations."""
    holdout = json.loads((OUT_DIR / "phase14_holdout.json").read_text())
    winner_prompt = holdout["winner_prompt"]
    winner_ns = holdout["winner_namespace"]  # e.g. "sg"
    return [
        ("hand_ncot", PROMPTS["narrative_cot"], "sg_ref_narrative_cot"),
        ("optimised", winner_prompt, f"sg_holdout_{winner_ns}_{GEN_MODEL}"),
    ]


def _load_cached_responses(prompt_text: str, ns: str, items) -> list[dict]:
    phash = _prompt_hash(prompt_text)
    rows = []
    for item in items:
        cache = gen_path(ns, item.id, phash, GEN_MODEL)
        if not cache.exists():
            continue
        rec = json.loads(cache.read_text())
        resp = (rec.get("response") or "").strip()
        if not resp:
            continue
        rows.append({"item_id": item.id, "question": item.prompt, "response": resp})
    return rows, phash


def _safe_score(metric, rec, judge, ns, phash) -> int:
    """Score one cell; swallow transient API failures (cache makes re-runs resume)."""
    try:
        return _score_metric(
            metric, rec["question"], rec["response"], judge, ns, rec["item_id"], phash,
        )
    except Exception:
        return -1


def _score_with_judges(rows: list[dict], ns: str, phash: str, workers: int = 8) -> dict:
    per_judge = {}
    for judge in ALL_JUDGES:
        rates = {}
        for metric in SYCO_METRICS:
            vals = []
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [
                    ex.submit(_safe_score, metric, rec, judge, ns, phash)
                    for rec in rows
                ]
                for fut in as_completed(futs):
                    s = fut.result()
                    if s >= 0:
                        vals.append(s)
            rates[metric] = sum(vals) / max(1, len(vals))
            rates[f"{metric}_n"] = len(vals)
        rates["n"] = len(rows)
        per_judge[judge] = rates
        print(f"    {judge}: " + ", ".join(
            f"{m}={rates[m]:.0%}(n={rates[f'{m}_n']})" for m in SYCO_METRICS
        ), flush=True)
    return per_judge


def main() -> int:
    items = load_holdout_oeq()
    configs = _configs()

    scored = {}
    for label, prompt_text, ns in configs:
        rows, phash = _load_cached_responses(prompt_text, ns, items)
        print(f"  {label}: {len(rows)} cached responses (ns={ns})", flush=True)
        scored[label] = _score_with_judges(rows, ns, phash)

    # Goodhart gap: reduction (hand - optimised) per judge per metric.
    gap = {}
    for metric in SYCO_METRICS:
        train_red = (
            scored["hand_ncot"][TRAIN_JUDGE][metric]
            - scored["optimised"][TRAIN_JUDGE][metric]
        )
        heldout_reds = [
            scored["hand_ncot"][j][metric] - scored["optimised"][j][metric]
            for j in HELDOUT_JUDGES
        ]
        heldout_mean = sum(heldout_reds) / len(heldout_reds)
        # Worst-case optimised rate across the panel (what min-max would see).
        worstcase_opt = max(scored["optimised"][j][metric] for j in ALL_JUDGES)
        gap[metric] = {
            "train_judge_reduction": train_red,
            "heldout_mean_reduction": heldout_mean,
            "goodhart_gap": train_red - heldout_mean,
            "heldout_reductions": dict(zip(HELDOUT_JUDGES, heldout_reds)),
            "optimised_worstcase_rate": worstcase_opt,
            "optimised_train_rate": scored["optimised"][TRAIN_JUDGE][metric],
        }

    result = {
        "gen_model": GEN_MODEL,
        "train_judge": TRAIN_JUDGE,
        "heldout_judges": HELDOUT_JUDGES,
        "scored_rates": scored,
        "goodhart_gap": gap,
    }
    out_path = OUT_DIR / "phase18_goodhart.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n=== Phase 18 Goodhart gap (hand NoT - optimised, by judge) ===")
    print(f"{'metric':<13} {'train(grok)':>12} {'heldout_mean':>13} {'gap':>8} "
          f"{'opt_train':>10} {'opt_worst':>10}")
    for metric in SYCO_METRICS:
        g = gap[metric]
        print(f"{metric:<13} {g['train_judge_reduction']:>12.1%} "
              f"{g['heldout_mean_reduction']:>13.1%} {g['goodhart_gap']:>8.1%} "
              f"{g['optimised_train_rate']:>10.1%} {g['optimised_worstcase_rate']:>10.1%}")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
