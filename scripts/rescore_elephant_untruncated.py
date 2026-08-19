"""
scripts/rescore_elephant_untruncated.py -- does the flagship ELEPHANT result
survive removing the judge-side truncation?

THE DEFECT
----------
`scripts/elephant_scorers.py` passes `advice[:4000]` to every judge call. On the
OEQ headline cell that truncates 75-93% of NoT responses against 0-59% of CoT;
on aita_yta it is 92-96% vs 0-8.7%. The two arms are therefore not scored on the
same object, and because NoT places its commitment/answer last by construction,
the discarded tail is exactly where a closing affirmation would sit. See
`Guidance_Documents/sycophancy_scenario_redesign.md` Section 1b(i).

WHY A PAIRED DESIGN, NOT JUST "RE-SCORE AT A HIGHER LIMIT"
-----------------------------------------------------------
Re-scoring only the full text and comparing against the stored scores would
confound truncation with judge nondeterminism and with any drift since the
original run. Instead, every affected response is scored TWICE in this run:

    control : advice[:4000]   (reproduces the published condition, fresh)
    full    : advice[:limit]  (limit defaults to the whole response)

The estimand is the within-response paired difference. Judge drift cancels.

CACHE SAFETY (two traps, both real)
------------------------------------
1. `elephant_scorers._cache_key` hashes `advice[:2000]`, so changing the
   truncation limit does NOT change the key -- calling `score_response` would
   silently return the stored 4,000-char score. This module therefore keeps its
   own cache namespace and never calls `score_response`.
2. That same `[:2000]` means the production key does not uniquely identify a
   response. Keys here hash the FULL advice plus the effective limit.

The published cache is never written to; this module only adds
`elephant_rescore_*` files.

USAGE
    python -m scripts.rescore_elephant_untruncated --smoke          # 8 responses
    python -m scripts.rescore_elephant_untruncated --dataset oeq --metric validation
    python -m scripts.rescore_elephant_untruncated --dataset oeq --all-metrics
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.elephant_scorers import (
    DEFAULT_JUDGE,
    SCORER_SYSTEM,
    _framing_prompt,
    _indirectness_prompt,
    _judge_text,
    _parse_binary,
    _validation_prompt,
)
from scripts.run_phase1_quartet import OUT_DIR, _safe

RAW = Path("divergence_study_outputs/elephant_singleagent_raw.csv")
PRODUCTION_LIMIT = 4000          # the defect, reproduced as the control arm
BUILDERS = {
    "validation": _validation_prompt,
    "indirectness": _indirectness_prompt,
    "framing": _framing_prompt,
}


def _key(metric: str, question: str, advice: str, judge: str, limit: Optional[int]) -> Path:
    """Own namespace; hashes the FULL advice and the effective limit."""
    tag = "full" if limit is None else str(limit)
    h = hashlib.sha1(
        f"{metric}|{question}|{advice}|{judge}|lim={tag}".encode()
    ).hexdigest()[:16]
    return OUT_DIR / f"elephant_rescore_{metric}_{_safe(judge)}_{tag}_{h}.json"


def score_at(metric: str, question: str, advice: str, judge: str,
             limit: Optional[int]) -> int:
    cache = _key(metric, question, advice, judge, limit)
    if cache.exists():
        try:
            return int(json.loads(cache.read_text()).get("score", -1))
        except Exception:
            pass
    text = advice if limit is None else advice[:limit]
    raw = _judge_text(judge, SCORER_SYSTEM, BUILDERS[metric](question, text))
    score = _parse_binary(raw)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(
        {"score": score, "raw": raw, "limit": limit, "chars_sent": len(text),
         "chars_total": len(advice)}, ensure_ascii=False), encoding="utf-8")
    return score


def load_questions(dataset: str, n: int = 150) -> dict[str, str]:
    """item_id -> the item's prompt.

    The raw CSV stores `prompt_len` but not the prompt text, and
    `response_preview` is the *response*. Feeding the judge the wrong question
    would silently produce an incomparable score, so the questions come from the
    same loader the original run used, joined on item_id.
    """
    from scripts.load_elephant import load_elephant
    return {it.id: it.prompt for it in load_elephant(dataset, n=n)}


DEFAULT_ARMS = ("standard_cot", "narrative_cot")


def _load(dataset: str, metric: str, only_truncated: bool,
          arms: Sequence[str] = DEFAULT_ARMS) -> list[dict[str, str]]:
    with RAW.open() as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["dataset"] == dataset and r["arm"] in tuple(arms)]
    out = []
    for r in rows:
        resp = r.get("response") or ""
        if not resp.strip():
            continue                      # non-response; see Section 1b(ii)
        if r.get(f"sycophantic_{metric}") not in ("0", "1"):
            continue                      # not scored in the published run
        if only_truncated and len(resp) <= PRODUCTION_LIMIT:
            continue                      # score cannot change; skip to save spend
        out.append(r)
    return out


def run(dataset: str, metric: str, *, judge: str, limit: Optional[int],
        workers: int, max_items: Optional[int],
        questions: dict[str, str],
        arms: Sequence[str] = DEFAULT_ARMS) -> dict[str, Any]:
    rows = _load(dataset, metric, only_truncated=True, arms=arms)
    missing = sorted({r["item_id"] for r in rows} - set(questions))
    if missing:
        raise SystemExit(
            f"{len(missing)} item_id(s) have no prompt from load_elephant "
            f"(e.g. {missing[:3]}). Refusing to score against a wrong question."
        )
    if max_items:
        rows = rows[:max_items]
    if not rows:
        return {"dataset": dataset, "metric": metric, "n": 0,
                "note": "no truncated scored responses"}

    def work(r):
        q = questions[r["item_id"]]
        a = r["response"]
        return {
            "generator": r["generator"], "arm": r["arm"], "item_id": r["item_id"],
            "chars": len(a),
            "published": int(r[f"sycophantic_{metric}"]),
            "control": score_at(metric, q, a, judge, PRODUCTION_LIMIT),
            "full": score_at(metric, q, a, judge, limit),
        }

    recs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, r) for r in rows]
        for i, f in enumerate(as_completed(futs), 1):
            recs.append(f.result())
            if i % 50 == 0:
                print(f"    ... {i}/{len(futs)}", flush=True)

    good = [r for r in recs if r["control"] in (0, 1) and r["full"] in (0, 1)]
    per: dict[str, Any] = {}
    for g in sorted({r["generator"] for r in good}):
        cell = {}
        for arm in sorted({r["arm"] for r in good}):
            s = [r for r in good if r["generator"] == g and r["arm"] == arm]
            if not s:
                continue
            cell[arm] = {
                "n": len(s),
                "control_rate": st.mean(r["control"] for r in s),
                "full_rate": st.mean(r["full"] for r in s),
                "flip_rate": st.mean(1.0 if r["control"] != r["full"] else 0.0 for r in s),
                "published_rate": st.mean(r["published"] for r in s),
            }
        per[g] = cell
    return {"dataset": dataset, "metric": metric, "judge": judge,
            "limit": limit, "control_limit": PRODUCTION_LIMIT,
            "n_rescored": len(recs), "n_usable": len(good),
            "per_generator": per, "records": recs}


def corrected_cell(dataset: str, metric: str, judge: str, limit: Optional[int],
                   questions: dict[str, str]) -> dict[str, Any]:
    """Full-cell estimate with the truncation defect removed.

    A response at or under PRODUCTION_LIMIT was already scored on its complete
    text, so its published score stands. A truncated response is replaced by its
    `full` re-score. Nothing new is generated -- this reads the warm cache.

    Judge drift is handled by NOT mixing scales: the correction applied is the
    within-run paired difference (full - control), both fresh, multiplied by the
    truncated fraction. The published level is then adjusted by that amount, so
    the published and fresh scales are never averaged together.
    """
    with RAW.open() as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["dataset"] == dataset and r["arm"] in ("standard_cot", "narrative_cot")
                and (r.get("response") or "").strip()
                and r.get(f"sycophantic_{metric}") in ("0", "1")]

    out: dict[str, Any] = {}
    for g in sorted({r["generator"] for r in rows}):
        cell = {}
        for arm in ("standard_cot", "narrative_cot"):
            sub = [r for r in rows if r["generator"] == g and r["arm"] == arm]
            if not sub:
                continue
            trunc = [r for r in sub if len(r["response"]) > PRODUCTION_LIMIT]
            pairs = []
            for r in trunc:
                q = questions.get(r["item_id"])
                if q is None:
                    continue
                kc = _key(metric, q, r["response"], judge, PRODUCTION_LIMIT)
                kf = _key(metric, q, r["response"], judge, limit)
                if not (kc.exists() and kf.exists()):
                    continue
                c = int(json.loads(kc.read_text()).get("score", -1))
                f = int(json.loads(kf.read_text()).get("score", -1))
                if c in (0, 1) and f in (0, 1):
                    pairs.append((c, f))
            frac = len(trunc) / len(sub)
            shift = (st.mean(f for _, f in pairs) - st.mean(c for c, _ in pairs)) if pairs else 0.0
            published = st.mean(int(r[f"sycophantic_{metric}"]) for r in sub)
            cell[arm] = {
                "n": len(sub), "n_truncated": len(trunc), "truncated_frac": frac,
                "n_paired": len(pairs), "paired_shift": shift,
                "published_rate": published,
                "corrected_rate": published + frac * shift,
            }
        if len(cell) == 2:
            c, n = cell["standard_cot"], cell["narrative_cot"]
            cell["published_delta_pp"] = 100 * (n["published_rate"] - c["published_rate"])
            cell["corrected_delta_pp"] = 100 * (n["corrected_rate"] - c["corrected_rate"])
            cell["correction_pp"] = cell["corrected_delta_pp"] - cell["published_delta_pp"]
        out[g] = cell
    return out


def _report_corrected(corr: dict[str, Any], dataset: str, metric: str) -> None:
    print(f"\nCORRECTED FULL-CELL ESTIMATE  ({dataset} / {metric})")
    print("  " + "-" * 84)
    print(f"  {'generator':<26}{'published Δ':>13}{'corrected Δ':>14}"
          f"{'correction':>12}{'NoT trunc':>11}")
    for g, cell in corr.items():
        if "corrected_delta_pp" not in cell:
            print(f"  {g:<26}{'(one arm only)':>13}")
            continue
        nf = 100 * cell["narrative_cot"]["truncated_frac"]
        print(f"  {g:<26}{cell['published_delta_pp']:>+13.1f}"
              f"{cell['corrected_delta_pp']:>+14.1f}"
              f"{cell['correction_pp']:>+12.1f}{nf:>10.0f}%")


def _report(res: dict[str, Any]) -> None:
    if not res.get("n_usable"):
        print(f"  {res['dataset']}/{res['metric']}: {res.get('note','no usable rows')}")
        return
    print(f"\n{res['dataset']} / {res['metric']}  "
          f"(n={res['n_usable']} truncated responses, each scored twice)")
    print("  " + "-" * 84)
    print(f"  {'generator':<26}{'arm':<16}{'n':>5}{'@4000':>9}{'full':>9}{'flipped':>10}")
    for g, cell in res["per_generator"].items():
        for arm, v in cell.items():
            print(f"  {g:<26}{arm:<16}{v['n']:>5}{v['control_rate']:>9.3f}"
                  f"{v['full_rate']:>9.3f}{100 * v['flip_rate']:>9.0f}%")
    print()
    for g, cell in res["per_generator"].items():
        if len(cell) < 2:
            continue
        c, n = cell["standard_cot"], cell["narrative_cot"]
        d_ctl = 100 * (n["control_rate"] - c["control_rate"])
        d_full = 100 * (n["full_rate"] - c["full_rate"])
        print(f"  {g:<26} truncated-subsample delta:  @4000 {d_ctl:+6.1f} pp   "
              f"full {d_full:+6.1f} pp   shift {d_full - d_ctl:+6.1f} pp")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Re-score ELEPHANT without the 4k truncation")
    ap.add_argument("--dataset", default="oeq")
    ap.add_argument("--metric", default="validation", choices=sorted(BUILDERS))
    ap.add_argument("--all-metrics", action="store_true")
    ap.add_argument("--judge", default=DEFAULT_JUDGE)
    ap.add_argument("--limit", type=int, default=None,
                    help="chars sent in the 'full' arm (default: entire response)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-items", type=int)
    ap.add_argument("--n-items", type=int, default=150,
                    help="n passed to load_elephant when fetching prompts")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                    help="comma-separated arms to re-score")
    ap.add_argument("--smoke", action="store_true", help="8 responses, one metric")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)

    if a.smoke:
        a.max_items, a.all_metrics = 8, False

    metrics = sorted(BUILDERS) if a.all_metrics else [a.metric]
    questions = load_questions(a.dataset, n=a.n_items)
    print(f"[rescore] loaded {len(questions)} item prompts for {a.dataset}")
    out: dict[str, Any] = {}
    for m in metrics:
        print(f"[rescore] {a.dataset}/{m} judge={a.judge} "
              f"limit={'full' if a.limit is None else a.limit}", flush=True)
        res = run(a.dataset, m, judge=a.judge, limit=a.limit, workers=a.workers,
                  max_items=a.max_items, questions=questions,
                  arms=[x.strip() for x in a.arms.split(",") if x.strip()])
        _report(res)
        corr = corrected_cell(a.dataset, m, a.judge, a.limit, questions)
        _report_corrected(corr, a.dataset, m)
        res["corrected"] = corr
        out[m] = res

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        slim = {m: {k: v for k, v in r.items() if k != "records"} for m, r in out.items()}
        a.json.write_text(json.dumps(slim, indent=2), encoding="utf-8")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
