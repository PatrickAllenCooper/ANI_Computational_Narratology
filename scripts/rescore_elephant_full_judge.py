"""
scripts/rescore_elephant_full_judge.py -- Addendum 17.8: a judge panel on the
ELEPHANT OEQ validation metric in which EVERY response is scored by EVERY judge
through one identical path.

WHY (found 2026-09-22). The alternate-judge tables of 17.2 and 17.6 were built
with `analyze_length_matched_elephant.apply_corrected_scores(rows, judge=J)`,
which overwrites the production score only for responses over 4,000 characters
(the truncation fix) and leaves every shorter response at the PRODUCTION
(haiku) judge's score. NoT responses are mostly long and CoT responses mostly
short (haiku CoT 0 of 150 long, NoT 121 of 150), so each "judge J" drop was a
haiku-scored CoT rate against a J-scored NoT rate. That is not a judge
comparison. Here every response, short or long, is scored in full by every
judge with `rescore_elephant_untruncated.score_at(metric, q, a, judge, None)`,
the production judge included, so the five judges differ only in the judge.

Cache: the `elephant_rescore_validation_<judge>_full_*` namespace already holds
the full-text scores of the long responses for haiku, nano, Llama and grok;
only short responses (and every response for a new judge) cost a call.

Readouts per judge: validation rate per generator x arm; drop (arm minus the
baseline arm, default standard_cot) per generator with an item-clustered paired
bootstrap (8,000 draws, seed 20260822); agreement with the production judge;
the unparsed share (a judge over 5% is flagged). The ceiling statistics (share
of CoT/NoT cells above 0.85, their range) are DESCRIPTIVE ONLY: the 17.8
amendment removed them as an exclusion rule, because a screen computed on the
tested cells excludes exactly the judges that see no drop (circular).

Run:
  python -m scripts.rescore_elephant_full_judge --dry-run
  python -m scripts.rescore_elephant_full_judge --judges gpt-4o --workers 16
  python -m scripts.rescore_elephant_full_judge --report --json divergence_study_outputs/judge_panel_full_oeq.json
  python -m scripts.rescore_elephant_full_judge --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from scripts.rescore_elephant_untruncated import BUILDERS, RAW, _key, load_questions, score_at
from scripts.elephant_scorers import SCORER_SYSTEM

METRIC = "validation"
PRODUCTION_JUDGE = "claude-haiku-4-5"
DEFAULT_JUDGES = ("claude-haiku-4-5", "gpt-5.4-nano", "Llama-3.3-70B-Instruct",
                  "grok-4-1-fast-reasoning", "gpt-4o")
DEFAULT_GENERATORS = ("claude-haiku-4-5", "claude-sonnet-4-6", "gpt-5.4-nano", "grok-4-1-fast-reasoning",
                      "Llama-3.3-70B-Instruct", "Mistral-Large-3-2", "DeepSeek-V4-Pro")
DEFAULT_ARMS = ("standard_cot", "narrative_cot")
SCREEN_ARMS = ("standard_cot", "narrative_cot")
ACQ_HIGH, ACQ_FRAC, ACQ_RANGE = 0.85, 0.80, 0.20
DRAWS, SEED = 8000, 20260822
REASONING_OUT_TOKENS = 600    # rough reasoning-token allowance for the cost estimate only


def load_rows(generators: Sequence[str], arms: Sequence[str], dataset: str = "oeq") -> list[dict]:
    """The shared results CSV plus every per-batch CSV written with run_elephant --out-csv
    (elephant_singleagent_raw_batch_*.csv); one row per (generator, arm, item), later files win."""
    paths = [RAW] + sorted(RAW.parent.glob("elephant_singleagent_raw_batch_*.csv"))
    keep: dict = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open() as fh:
            for r in csv.DictReader(fh):
                if (r["dataset"] == dataset and r["arm"] in arms and r["generator"] in generators
                        and (r.get("regime") or "default") == "default"):
                    keep[(r["generator"], r["arm"], r["item_id"])] = r
    return [r for r in keep.values() if (r.get("response") or "").strip()]


def cached_score(q: str, a: str, judge: str) -> Optional[int]:
    k = _key(METRIC, q, a, judge, None)
    if not k.exists():
        return None
    try:
        return int(json.loads(k.read_text()).get("score", -1))
    except Exception:
        return None


def plan(rows: list[dict], judges: Sequence[str], questions: dict) -> dict:
    from scripts.run_crowdgold_deliberation import PRICES
    out = {}
    for j in judges:
        todo = [r for r in rows if cached_score(questions[r["item_id"]], r["response"], j) is None]
        chars = sum(len(SCORER_SYSTEM) + len(BUILDERS[METRIC](questions[r["item_id"]], r["response"]))
                    for r in todo)
        tin = chars / 4.0
        reasoning = j in ("gpt-5.4-nano", "grok-4-1-fast-reasoning")
        tout = len(todo) * (REASONING_OUT_TOKENS if reasoning else 8)
        pin, pout = PRICES.get(j, (None, None))
        usd = None if pin is None else tin / 1e6 * pin + tout / 1e6 * pout
        out[j] = {"responses": len(rows), "new_calls": len(todo), "est_in_tokens": int(tin),
                  "est_usd": None if usd is None else round(usd, 2), "priced": pin is not None}
    return out


def score_all(rows: list[dict], judge: str, questions: dict, workers: int) -> int:
    """One failed call (after the backend's own retries) is counted and skipped, never
    allowed to stop the loop; the affected response stays unscored (reported as missing)."""
    todo = [r for r in rows if cached_score(questions[r["item_id"]], r["response"], judge) is None]
    done = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(score_at, METRIC, questions[r["item_id"]], r["response"], judge, None) for r in todo]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                failed += 1
                if failed <= 5:
                    print(f"    {judge}: call failed ({type(e).__name__}: {str(e)[:120]})", flush=True)
            done += 1
            if done % 100 == 0:
                print(f"    {judge}: {done}/{len(todo)} ({failed} failed)", flush=True)
    print(f"  {judge}: {len(todo)} attempted, {failed} failed", flush=True)
    return len(todo)


def table(rows: list[dict], judge: str, questions: dict) -> dict:
    """(gen, arm) -> {item: score}; unparsed (-1) and missing kept out, counted."""
    cells: dict = defaultdict(dict)
    unparsed = missing = 0
    for r in rows:
        s = cached_score(questions[r["item_id"]], r["response"], judge)
        if s is None:
            missing += 1
            continue
        if s not in (0, 1):
            unparsed += 1
            continue
        cells[(r["generator"], r["arm"])][r["item_id"]] = s
    return {"cells": cells, "unparsed": unparsed, "missing": missing}


def paired_drop(a: dict, b: dict, *, draws: int = DRAWS, seed: int = SEED) -> dict:
    """rate(b) - rate(a) on items scored in both, item-clustered bootstrap."""
    items = sorted(set(a) & set(b))
    if len(items) < 20:
        return {"n_items": len(items), "drop": None, "lo": None, "hi": None}
    d = np.array([b[i] - a[i] for i in items], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(items), size=(draws, len(items)))
    boots = d[idx].mean(1)
    return {"n_items": len(items), "rate_a": float(np.mean([a[i] for i in items])),
            "rate_b": float(np.mean([b[i] for i in items])), "drop": float(d.mean()),
            "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5))}


def acquiescence(cells: dict, generators: Sequence[str]) -> dict:
    """DESCRIPTIVE ONLY since the 17.8 amendment: never used to exclude a judge."""
    rates = [np.mean(list(cells[(g, a)].values())) for g in generators for a in SCREEN_ARMS
             if cells.get((g, a))]
    if not rates:
        return {"n_cells": 0, "would_have_been_excluded": None}
    frac_high = float(np.mean([r > ACQ_HIGH for r in rates]))
    rng_ = float(max(rates) - min(rates))
    return {"n_cells": len(rates), "frac_cells_above_0.85": frac_high, "range": rng_,
            "would_have_been_excluded": bool(frac_high >= ACQ_FRAC or rng_ < ACQ_RANGE)}


def agreement(cells_j: dict, cells_p: dict) -> dict:
    same = tot = 0
    for k, m in cells_j.items():
        p = cells_p.get(k, {})
        for i, s in m.items():
            if i in p:
                tot += 1
                same += int(s == p[i])
    return {"n": tot, "agreement": (same / tot) if tot else None}


def report(rows: list[dict], judges: Sequence[str], generators: Sequence[str], arms: Sequence[str],
           questions: dict, baseline: str = "standard_cot") -> dict:
    prod = table(rows, PRODUCTION_JUDGE, questions)
    res = {"judges": {}, "generators": list(generators), "arms": list(arms), "baseline": baseline}
    for j in judges:
        t = table(rows, j, questions)
        cells = t["cells"]
        n_scored = sum(len(m) for m in cells.values())
        per_gen = {}
        for g in generators:
            base = cells.get((g, baseline), {})
            per_gen[g] = {arm: paired_drop(base, cells.get((g, arm), {}))
                          for arm in arms if arm != baseline}
            per_gen[g]["rates"] = {arm: (float(np.mean(list(cells[(g, arm)].values())))
                                         if cells.get((g, arm)) else None) for arm in arms}
        unp_share = t["unparsed"] / max(1, n_scored + t["unparsed"])
        res["judges"][j] = {"unparsed": t["unparsed"], "missing": t["missing"],
                            "unparsed_share": unp_share, "unparsed_flag": bool(unp_share > 0.05),
                            "ceiling_descriptive": acquiescence(cells, generators),
                            "agreement_with_production": agreement(cells, prod["cells"]) if j != PRODUCTION_JUDGE else None,
                            "per_generator": per_gen}
    return res


def print_report(res: dict) -> None:
    for j, jr in res["judges"].items():
        scr = jr["ceiling_descriptive"]
        agr = jr["agreement_with_production"]
        print(f"\n=== judge {j}   unparsed {jr['unparsed']} ({100 * jr['unparsed_share']:.1f}%"
              f"{', FLAGGED' if jr['unparsed_flag'] else ''})  missing {jr['missing']}   "
              f"ceiling (descriptive): {scr.get('frac_cells_above_0.85', 0):.2f} of cells above 0.85, "
              f"range {scr.get('range', 0):.2f}"
              + (f"   agreement with production {agr['agreement']:.3f} (n={agr['n']})" if agr and agr['agreement'] is not None else ""))
        for g, pg in jr["per_generator"].items():
            rates = " ".join(f"{a}={v:.3f}" for a, v in pg["rates"].items() if v is not None)
            drops = "  ".join(
                f"{a} {100 * d['drop']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}] (n={d['n_items']})"
                for a, d in pg.items() if a != "rates" and d.get("drop") is not None)
            print(f"  {g:26s} {rates}\n      {drops}")


def _selftest() -> int:
    ok = True

    def check(msg, cond):
        nonlocal ok
        print(("[ok] " if cond else "[FAIL] ") + msg)
        ok = ok and cond

    a = {str(i): 1 for i in range(40)}
    b = {str(i): (0 if i < 20 else 1) for i in range(40)}
    d = paired_drop(a, b)
    check("paired drop of 1.0 -> 0.5 is -0.5 with a CI excluding 0", abs(d["drop"] + 0.5) < 1e-9 and d["hi"] < 0)
    check("paired drop needs >= 20 shared items", paired_drop({"1": 1}, {"1": 0})["drop"] is None)
    cells = {(g, arm): {str(i): 1 for i in range(10)} for g in ("x", "y") for arm in SCREEN_ARMS}
    check("a judge that says 1 everywhere would have been screened (descriptive only)",
          acquiescence(cells, ("x", "y"))["would_have_been_excluded"])
    cells2 = {("x", "standard_cot"): {str(i): 1 for i in range(10)},
              ("x", "narrative_cot"): {str(i): int(i < 3) for i in range(10)}}
    check("a judge with range 0.7 and 1 of 2 cells high is not flagged",
          not acquiescence(cells2, ("x",))["would_have_been_excluded"])
    rows = load_rows(DEFAULT_GENERATORS, DEFAULT_ARMS)
    check("loads the seven-generator CoT/NoT OEQ rows (> 1,900 responses)", len(rows) > 1900)
    q = load_questions("oeq", n=150)
    check("every loaded row has its question", all(r["item_id"] in q for r in rows))
    print("\nselftest: " + ("ALL OK" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default=",".join(DEFAULT_JUDGES))
    ap.add_argument("--generators", default=",".join(DEFAULT_GENERATORS))
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--baseline", default="standard_cot",
                    help="arm every drop is taken against (17.7/17.4 use standard_cot_rep)")
    ap.add_argument("--metric", default="validation", choices=sorted(BUILDERS),
                    help="ELEPHANT rubric to score (17.11 scores 'framing' on the existing "
                         "responses; the cache namespace carries the metric name)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true", help="read the caches only; no calls")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    global METRIC
    METRIC = a.metric
    judges = [x for x in a.judges.split(",") if x]
    gens = [x for x in a.generators.split(",") if x]
    arms = [x for x in a.arms.split(",") if x]
    if a.baseline not in arms:
        ap.error(f"--baseline {a.baseline} must be among --arms (every drop is against it)")
    rows = load_rows(gens, arms)
    questions = load_questions("oeq", n=150)
    missing_q = {r["item_id"] for r in rows} - set(questions)
    if missing_q:
        raise SystemExit(f"{len(missing_q)} item(s) without a question; refusing")
    if a.dry_run:
        p = plan(rows, judges, questions)
        tot = 0.0
        for j, v in p.items():
            print(f"  {j:26s} responses {v['responses']}  new calls {v['new_calls']}  est ${v['est_usd']}")
            tot += v["est_usd"] or 0.0
        print(f"  TOTAL est ${tot:.2f}")
        return 0
    if not a.report:
        for j in judges:
            n = score_all(rows, j, questions, a.workers)
            print(f"  {j}: scored {n} new")
    res = report(rows, judges, gens, arms, questions, baseline=a.baseline)
    res["metric"] = METRIC
    print_report(res)
    out = a.json or Path("divergence_study_outputs/judge_panel_full_oeq"
                         + ("" if METRIC == "validation" else f"_{METRIC}") + ".json")
    out.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
