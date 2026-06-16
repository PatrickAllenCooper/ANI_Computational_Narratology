"""
scripts/eval_phase18_heldout.py -- Held-out judge evaluation incl. DeepSeek (Phase 18c).

Re-scores cached holdout responses for hand NoT, single-judge optimised, and
robust optimised prompts across held-out judges (sonnet + deepseek-r1) without
regeneration. Computes two-proportion significance for robust vs single-judge.

Output: divergence_study_outputs/phase18_heldout.json

Usage:
  python -m scripts.eval_phase18_heldout
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.run_phase1_quartet import PROMPTS, OUT_DIR
from scripts.syco_loss import (
    SYCO_METRICS,
    DEFAULT_GEN_MODEL,
    _prompt_hash,
    _score_metric,
    gen_path,
    load_holdout_oeq,
)

HELDOUT_JUDGES = ["claude-sonnet-4-6", "deepseek-r1"]
GEN_MODEL = DEFAULT_GEN_MODEL


def _configs() -> list[tuple[str, str, str]]:
    holdout = json.loads((OUT_DIR / "phase14_holdout.json").read_text())
    robust = json.loads((OUT_DIR / "rg_summary.json").read_text())
    winner_ns = holdout["winner_namespace"]
    return [
        ("hand_ncot", PROMPTS["narrative_cot"], "sg_ref_narrative_cot"),
        ("single_judge", holdout["winner_prompt"], f"sg_holdout_{winner_ns}_{GEN_MODEL}"),
        ("robust", robust["final_prompt"], f"rg_holdout_{GEN_MODEL}"),
    ]


def _load_rows(prompt_text: str, ns: str, items) -> tuple[list[dict], str]:
    phash = _prompt_hash(prompt_text)
    rows = []
    for item in items:
        cache = gen_path(ns, item.id, phash, GEN_MODEL)
        if not cache.exists():
            continue
        rec = json.loads(cache.read_text())
        resp = (rec.get("response") or "").strip()
        if resp:
            rows.append({"item_id": item.id, "question": item.prompt, "response": resp})
    return rows, phash


def _safe_score(metric, rec, judge, ns, phash) -> int:
    try:
        return _score_metric(
            metric, rec["question"], rec["response"], judge, ns, rec["item_id"], phash,
        )
    except Exception:
        return -1


def _score_config(rows, ns, phash, judge, workers=8) -> dict:
    rates = {}
    per_item = {m: [] for m in SYCO_METRICS}
    for metric in SYCO_METRICS:
        vals = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_safe_score, metric, rec, judge, ns, phash) for rec in rows]
            for fut in as_completed(futs):
                s = fut.result()
                if s >= 0:
                    vals.append(s)
        rates[metric] = sum(vals) / max(1, len(vals))
        rates[f"{metric}_n"] = len(vals)
    rates["n"] = len(rows)
    return rates


def _two_prop_z(p1: float, p2: float, n: int) -> tuple[float, float]:
    x1, x2 = round(p1 * n), round(p2 * n)
    p = (x1 + x2) / max(1, 2 * n)
    se = math.sqrt(p * (1 - p) * (2 / n)) if 0 < p < 1 else 0.0
    if se == 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / se
    pval = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, pval


def main() -> int:
    items = load_holdout_oeq()
    configs = _configs()
    result = {"gen_model": GEN_MODEL, "heldout_judges": HELDOUT_JUDGES, "configs": {}}

    for label, prompt, ns in configs:
        rows, phash = _load_rows(prompt, ns, items)
        print(f"{label}: {len(rows)} cached responses", flush=True)
        result["configs"][label] = {"n": len(rows), "judges": {}}
        for judge in HELDOUT_JUDGES:
            print(f"  scoring {judge}...", flush=True)
            rates = _score_config(rows, ns, phash, judge)
            result["configs"][label]["judges"][judge] = rates
            print(
                f"    v={rates['validation']:.0%} ind={rates['indirectness']:.0%} "
                f"f={rates['framing']:.0%}",
                flush=True,
            )

    n = result["configs"]["robust"]["n"]
    sig = {}
    for judge in HELDOUT_JUDGES:
        sig[judge] = {}
        single = result["configs"]["single_judge"]["judges"][judge]
        robust = result["configs"]["robust"]["judges"][judge]
        for metric in SYCO_METRICS:
            z, p = _two_prop_z(robust[metric], single[metric], n)
            sig[judge][metric] = {"z": z, "p": p, "robust": robust[metric], "single": single[metric]}
    result["robust_vs_single"] = sig

    out = OUT_DIR / "phase18_heldout.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nWrote {out}")
    print("\n=== Robust vs single-judge (held-out) ===")
    for judge in HELDOUT_JUDGES:
        print(f"  {judge}:")
        for metric in SYCO_METRICS:
            s = sig[judge][metric]
            print(f"    {metric}: robust={s['robust']:.0%} single={s['single']:.0%} p={s['p']:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
