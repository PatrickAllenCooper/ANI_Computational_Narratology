#!/usr/bin/env python3
"""
scripts/analyze_judge_test_retest.py -- production-judge test-retest on
identical inputs. Zero spend, cache-only, no network.

What it measures
----------------
Every non-empty OEQ CoT/NoT response of the seven pillar-1 generators was
scored for validation by the production judge (claude-haiku-4-5) twice, by two
separate calls with the same system prompt (SCORER_SYSTEM), the same rubric
builder (_validation_prompt) and the same max_tokens, written to two cache
namespaces under divergence_study_outputs/:

  draw 1  elephant_score_validation_claude-haiku-4-5_<h>.json
          the production scorer (scripts/elephant_scorers.score_response), input
          advice[:4000]; written in June 2026 for the four original generators
          and on 2026-09-21 for the three open-weight deployments
  draw 2  elephant_rescore_validation_claude-haiku-4-5_full_<h>.json
          the 17.8 full-text pass (scripts/rescore_elephant_untruncated.score_at
          with limit None), input the full advice; written 2026-09-22

For a response of at most 4,000 characters the judge input is byte-identical in
the two calls, so the pair is a test-retest pair of the production judge (the
Anthropic call never sets a temperature, so both draws are samples at the API
default). Responses over 4,000 characters are NOT test-retest pairs (truncated
versus full input); they are reported under a block whose name says so.

Cache safety
------------
Reads files by exact path only: the shared results CSV, the per-batch CSVs
(elephant_singleagent_raw_batch_*.csv, the same merge rule as
rescore_elephant_full_judge.load_rows), the two judge namespaces above, and
data/elephant/OEQ.csv (the ELEPHANT CSV already on disk; the seed-44 sample is
re-drawn offline with the same RNG call as scripts/load_elephant.load_elephant).
It does not import scripts.generators, scripts.elephant_scorers or
scripts.load_elephant, and makes no network call. The two cache-key formulas
are copied here and checked in --selftest against golden file names produced by
the original functions.

Usage
-----
  python -m scripts.analyze_judge_test_retest --selftest
  python -m scripts.analyze_judge_test_retest            # writes the artefact
  python -m scripts.analyze_judge_test_retest --out divergence_study_outputs/judge_test_retest.json
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "divergence_study_outputs"
RAW = OUT_DIR / "elephant_singleagent_raw.csv"
OEQ_CSV = ROOT / "data" / "elephant" / "OEQ.csv"
DEFAULT_OUT = OUT_DIR / "judge_test_retest.json"

METRIC = "validation"
JUDGE = "claude-haiku-4-5"
PRODUCTION_LIMIT = 4000          # scripts/elephant_scorers.score_response: advice[:4000]
INLINE_ADVICE_HASHED = 2000      # scripts/elephant_scorers._cache_key hashes advice[:2000]
ELEPHANT_SEED = 44               # scripts/load_elephant.ELEPHANT_SEED
N_ITEMS = 150
GENERATORS = ("claude-haiku-4-5", "claude-sonnet-4-6", "gpt-5.4-nano", "grok-4-1-fast-reasoning",
              "Llama-3.3-70B-Instruct", "Mistral-Large-3-2", "DeepSeek-V4-Pro")
ARMS = ("standard_cot", "narrative_cot")
DRAWS, SEED = 8000, 20260822     # programme convention (item-clustered percentile bootstrap)
THIN = 20                        # 17.8 floor: nothing read on fewer than 20 paired items

FORBIDDEN_MODULES = ("scripts.generators", "scripts.elephant_scorers", "scripts.load_elephant",
                     "requests", "httpx", "openai", "anthropic", "urllib3")


# ---------------------------------------------------------------------------
# Cache keys, copied from the originals (no import of the modules that hold them)
# ---------------------------------------------------------------------------
_REASONING_HINTS = ("gpt-5", "o1", "o3", "o4", "reasoning")   # scripts/generators.py


def _is_reasoning(model: str) -> bool:                          # scripts/generators._is_reasoning
    m = model.lower()
    return any(h in m for h in _REASONING_HINTS) or "deepseek-r" in m


def _safe(s: str) -> str:                                       # scripts/run_phase1_quartet._safe
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def inline_key(metric: str, question: str, advice: str, judge: str) -> Path:
    """scripts/elephant_scorers._cache_key: hashes question[:500] and advice[:2000]."""
    ver = "|r2" if _is_reasoning(judge) else ""
    h = hashlib.sha1(
        f"{metric}|{question[:500]}|{advice[:INLINE_ADVICE_HASHED]}|{judge}{ver}".encode()
    ).hexdigest()[:16]
    return OUT_DIR / f"elephant_score_{metric}_{_safe(judge)}_{h}.json"


def full_key(metric: str, question: str, advice: str, judge: str, limit: Optional[int] = None) -> Path:
    """scripts/rescore_elephant_untruncated._key: hashes the full advice and the limit tag."""
    tag = "full" if limit is None else str(limit)
    h = hashlib.sha1(f"{metric}|{question}|{advice}|{judge}|lim={tag}".encode()).hexdigest()[:16]
    return OUT_DIR / f"elephant_rescore_{metric}_{_safe(judge)}_{tag}_{h}.json"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_questions_offline(csv_path: Path = OEQ_CSV, n: int = N_ITEMS, seed: int = ELEPHANT_SEED) -> dict[str, str]:
    """item_id -> prompt, re-drawn exactly as scripts/load_elephant.load_elephant('oeq', n=150):
    pandas read, random.Random(seed).shuffle of the row indices, first n, id from 'Unnamed: 0'."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    rng = random.Random(seed)
    idxs = list(range(len(df)))
    rng.shuffle(idxs)
    df = df.iloc[idxs[:n]].reset_index(drop=True)
    out: dict[str, str] = {}
    for i, row in df.iterrows():
        rid = None
        for col in ("id", "Unnamed: 0"):
            if col in row.index and pd.notna(row[col]):
                rid = str(row[col]).strip()
                break
        if rid is None:
            rid = str(i)
        out[rid] = str(row.get("prompt", "") or "").strip()
    return out


def batch_csvs(out_dir: Path = OUT_DIR) -> list[Path]:
    """The per-batch CSVs beside the shared file (one directory scan, prefix filter)."""
    found = []
    with os.scandir(out_dir) as it:
        for e in it:
            if e.name.startswith("elephant_singleagent_raw_batch_") and e.name.endswith(".csv"):
                found.append(Path(e.path))
    return sorted(found)


def load_rows(generators: Sequence[str], arms: Sequence[str], dataset: str = "oeq",
              batches: Optional[Sequence[Path]] = None) -> tuple[list[dict], dict]:
    """Same merge rule as rescore_elephant_full_judge.load_rows: shared CSV plus every batch CSV,
    one row per (generator, arm, item), later files win, regime default, non-empty response.
    Also returns how many default-cell rows each batch file overrode (expected 0)."""
    paths = [RAW] + list(batches if batches is not None else batch_csvs())
    keep: dict = {}
    overrides: dict[str, int] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open() as fh:
            for r in csv.DictReader(fh):
                if (r["dataset"] == dataset and r["arm"] in arms and r["generator"] in generators
                        and (r.get("regime") or "default") == "default"):
                    k = (r["generator"], r["arm"], r["item_id"])
                    if k in keep and path != RAW:
                        overrides[path.name] = overrides.get(path.name, 0) + 1
                    keep[k] = r
    rows = [r for r in keep.values() if (r.get("response") or "").strip()]
    return rows, {"batch_files": [p.name for p in paths[1:] if p.exists()], "default_cell_overrides": overrides}


def read_score(p: Path) -> tuple[Optional[int], Optional[float]]:
    """(score or None, mtime) -- None when the file is absent or unreadable; -1 stays -1 (unparsed)."""
    if not p.exists():
        return None, None
    try:
        return int(json.loads(p.read_text()).get("score", -1)), p.stat().st_mtime
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def counts_2x2(pairs: Sequence[dict]) -> np.ndarray:
    """[n11, n10, n01, n00] over (draw1, draw2)."""
    a = sum(1 for p in pairs if p["s1"] == 1 and p["s2"] == 1)
    b = sum(1 for p in pairs if p["s1"] == 1 and p["s2"] == 0)
    c = sum(1 for p in pairs if p["s1"] == 0 and p["s2"] == 1)
    d = sum(1 for p in pairs if p["s1"] == 0 and p["s2"] == 0)
    return np.array([a, b, c, d], dtype=float)


def kappa_from_counts(C: np.ndarray):
    """Agreement, Cohen's kappa, rate draw 1, rate draw 2 from [n11, n10, n01, n00] (vectorised)."""
    C = np.atleast_2d(C)
    n = C.sum(1)
    a, b, c, d = C.T
    po = (a + d) / n
    p1 = (a + b) / n
    p2 = (a + c) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(pe < 1, (po - pe) / (1 - pe), np.nan)
    return po, k, p1, p2


def _f(x, nd=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def summarise(pairs: Sequence[dict], draws: int = DRAWS, seed: int = SEED, cluster: str = "item_id") -> dict:
    """Agreement, kappa and the draw-2 minus draw-1 rate shift with item-clustered percentile CIs."""
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    C = counts_2x2(pairs)
    po, k, p1, p2 = kappa_from_counts(C)
    out = {
        "n": n,
        "n_items": len({p[cluster] for p in pairs}),
        "agreement": _f(po[0]),
        "kappa": _f(k[0]),
        "rate_draw1": _f(p1[0]),
        "rate_draw2": _f(p2[0]),
        "shift_draw2_minus_draw1_points": _f(100 * (p2[0] - p1[0]), 1),
        "flips_1to0": int(C[1]),
        "flips_0to1": int(C[2]),
        "flip_rate": _f((C[1] + C[2]) / n),
        "thin": n < THIN,
    }
    # item-clustered bootstrap: resample clusters (items) with replacement
    by = defaultdict(list)
    for p in pairs:
        by[p[cluster]].append(p)
    keys = sorted(by)
    M = np.array([counts_2x2(by[key]) for key in keys])          # (m, 4)
    rng = np.random.default_rng(seed)
    m = len(keys)
    W = rng.multinomial(m, np.full(m, 1.0 / m), size=draws).astype(float)   # (draws, m) resample weights
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):     # macOS Accelerate emits spurious
        Cb = W @ M                                               # (draws, 4)   matmul warnings on this product
    assert np.isfinite(Cb).all() and Cb.shape == (draws, 4), "bootstrap counts not finite"
    pob, kb, p1b, p2b = kappa_from_counts(Cb)
    kb = kb[~np.isnan(kb)]
    if len(kb):
        out["kappa_ci95"] = [_f(np.percentile(kb, 2.5)), _f(np.percentile(kb, 97.5))]
        out["kappa_ci_draws_defined"] = int(len(kb))
    else:
        out["kappa_ci95"] = None
    out["agreement_ci95"] = [_f(np.percentile(pob, 2.5)), _f(np.percentile(pob, 97.5))]
    sh = 100 * (p2b - p1b)
    out["shift_ci95_points"] = [_f(np.percentile(sh, 2.5), 1), _f(np.percentile(sh, 97.5), 1)]
    return out


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def build_pairs(rows: list[dict], questions: dict[str, str]) -> tuple[list[dict], dict]:
    """One record per response with both draws parsed; 'identical' marks len <= 4000."""
    pairs, counts = [], defaultdict(int)
    for r in rows:
        counts["rows_nonempty"] += 1
        q = questions.get(r["item_id"])
        if q is None:
            counts["item_not_in_sample"] += 1
            continue
        a = r["response"]
        s1, t1 = read_score(inline_key(METRIC, q, a, JUDGE))
        s2, t2 = read_score(full_key(METRIC, q, a, JUDGE, None))
        counts["draw1_present"] += s1 is not None
        counts["draw2_present"] += s2 is not None
        counts["draw1_unparsed"] += s1 == -1
        counts["draw2_unparsed"] += s2 == -1
        if s1 in (0, 1) and s2 in (0, 1):
            counts["both_parsed"] += 1
            pairs.append({"generator": r["generator"], "arm": r["arm"], "item_id": r["item_id"],
                          "length": len(a), "s1": s1, "s2": s2, "t1": t1, "t2": t2,
                          "identical": len(a) <= PRODUCTION_LIMIT})
    counts["identical_input_pairs"] = sum(1 for p in pairs if p["identical"])
    counts["truncated_input_pairs_excluded"] = sum(1 for p in pairs if not p["identical"])
    return pairs, dict(counts)


def interval_days(pairs: Sequence[dict]) -> Optional[dict]:
    d = [(p["t2"] - p["t1"]) / 86400.0 for p in pairs if p["t1"] and p["t2"]]
    if not d:
        return None
    return {"n": len(d), "median": round(float(np.median(d)), 1), "min": round(min(d), 1), "max": round(max(d), 1)}


def drop_by_draw(pairs: Sequence[dict]) -> dict:
    """Per generator, on items whose BOTH responses are identical-input pairs: NoT minus CoT
    validation (points) under draw 1 and under draw 2. Descriptive (a selected subsample)."""
    out = {}
    for g in GENERATORS:
        d: dict = defaultdict(dict)
        for p in pairs:
            if p["generator"] == g and p["identical"]:
                d[p["item_id"]][p["arm"]] = p
        both = [i for i in d if len(d[i]) == 2]
        if not both:
            out[g] = {"n_items": 0}
            continue
        d1 = 100 * (sum(d[i]["narrative_cot"]["s1"] for i in both) - sum(d[i]["standard_cot"]["s1"] for i in both)) / len(both)
        d2 = 100 * (sum(d[i]["narrative_cot"]["s2"] for i in both) - sum(d[i]["standard_cot"]["s2"] for i in both)) / len(both)
        out[g] = {"n_items": len(both), "drop_draw1_points": round(d1, 1), "drop_draw2_points": round(d2, 1),
                  "difference_points": round(d2 - d1, 1), "thin": len(both) < THIN}
    return out


def analyse(draws: int = DRAWS, seed: int = SEED, batches: Optional[Sequence[Path]] = None) -> dict:
    t0 = time.time()
    questions = load_questions_offline()
    rows, merge = load_rows(GENERATORS, ARMS, batches=batches)
    pairs, counts = build_pairs(rows, questions)
    ident = [p for p in pairs if p["identical"]]
    strict = [p for p in ident if p["length"] <= INLINE_ADVICE_HASHED]
    trunc = [p for p in pairs if not p["identical"]]

    per_cell: dict = {}
    for g in GENERATORS:
        per_cell[g] = {}
        for arm in ARMS:
            cell_all = [p for p in pairs if p["generator"] == g and p["arm"] == arm]
            cell = [p for p in cell_all if p["identical"]]
            s = summarise(cell, draws, seed) if cell else {"n": 0}
            s["n_scored_both_draws"] = len(cell_all)
            s["n_truncated_input_excluded"] = len(cell_all) - len(cell)
            s["interval_days"] = interval_days(cell)
            per_cell[g][arm] = s

    by_gen = {g: summarise([p for p in ident if p["generator"] == g], draws, seed) for g in GENERATORS}
    by_arm = {a: summarise([p for p in ident if p["arm"] == a], draws, seed) for a in ARMS}

    # not test-retest: truncated (draw 1, advice[:4000]) vs full (draw 2) input, kept for contrast only
    trunc_block = {"label": "NOT test-retest: draw 1 saw advice[:4000], draw 2 saw the full response; "
                            "disagreement here mixes truncation with judge noise",
                   "pooled": summarise(trunc, draws, seed) if trunc else {"n": 0},
                   "per_cell": {g: {a: summarise([p for p in trunc if p["generator"] == g and p["arm"] == a], draws, seed)
                                    for a in ARMS} for g in GENERATORS}}

    out = {
        "artefact": "judge_test_retest.json",
        "script": "scripts/analyze_judge_test_retest.py",
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "judge": JUDGE,
        "metric": METRIC,
        "design": ("Each non-empty OEQ CoT/NoT response scored twice by the production judge through the same "
                   "system prompt, rubric builder and max_tokens: draw 1 = production scorer cache "
                   "(elephant_score_validation_<judge>_*, input advice[:4000]); draw 2 = 17.8 full-text pass "
                   "(elephant_rescore_validation_<judge>_full_*, input full advice). Responses of at most 4,000 "
                   "characters have byte-identical judge input in both draws and form the test-retest set."),
        "sources": {
            "raw_csv": str(RAW.relative_to(ROOT)),
            "batch_csvs_merged": merge["batch_files"],
            "batch_rows_overriding_default_cells": merge["default_cell_overrides"],
            "questions": f"{OEQ_CSV.relative_to(ROOT)}, seed {ELEPHANT_SEED}, n {N_ITEMS} (offline re-draw of load_elephant)",
            "draw1_namespace": f"elephant_score_{METRIC}_{_safe(JUDGE)}_<sha1[:16] of metric|question[:500]|advice[:2000]|judge>.json",
            "draw2_namespace": f"elephant_rescore_{METRIC}_{_safe(JUDGE)}_full_<sha1[:16] of metric|question|advice|judge|lim=full>.json",
        },
        "bootstrap": {"draws": draws, "seed": seed, "cluster": "item_id (pooled, by_generator, by_arm); "
                                                                "each item appears once per cell"},
        "counts": counts,
        "pooled": summarise(ident, draws, seed),
        "pooled_strict_le2000": {"note": "responses of at most 2,000 characters, where the draw-1 cache key hashes "
                                         "the whole response rather than a 2,000-character prefix",
                                 **summarise(strict, draws, seed)},
        "by_arm": by_arm,
        "by_generator": by_gen,
        "per_cell": per_cell,
        "not_test_retest_truncated_vs_full": trunc_block,
        "drop_by_draw_on_identical_input_items": {
            "note": "NoT minus CoT (points) on items whose both responses are identical-input pairs, under "
                    "draw 1 and draw 2; a selected subsample of short responses, descriptive only",
            **drop_by_draw(pairs)},
        "interval_days": {"note": "days between the two cache files' mtimes (descriptive)",
                          "pooled": interval_days(ident)},
        "caveats": [
            "Draw 1 and draw 2 are the production run and the 17.8 pass, not two same-day calls: months apart for "
            "the four original generators (June 2026 vs 2026-09-22), a day apart for the three open-weight "
            "deployments (2026-09-21 vs 2026-09-22); the judge string was the same, the provider's serving of it "
            "is not verifiable from the caches.",
            "The draw-1 cache key hashes only the first 2,000 characters of the response; for 2,000 < length <= "
            "4,000 the pairing is on that prefix. Distinct generations sharing a 2,000-character prefix are not "
            "expected; pooled_strict_le2000 is the check.",
            "Test-retest pairs are the responses of at most 4,000 characters, a length-selected subset; per-cell "
            "coverage varies from a few responses to the whole cell (see per_cell.n and n_truncated_input_excluded).",
            "Responses over 4,000 characters were scored on different inputs in the two draws and are not "
            "test-retest pairs; their block is kept only to show that truncation adds disagreement beyond noise.",
        ],
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    return out


# ---------------------------------------------------------------------------
# Self-test (synthetic checks plus golden cache-key names; no cache read needed)
# ---------------------------------------------------------------------------
def selftest() -> None:
    # 1. kappa/agreement/flips on a hand-computed 2x2 table
    pairs = ([{"item_id": str(i), "s1": 1, "s2": 1} for i in range(40)]
             + [{"item_id": str(40 + i), "s1": 1, "s2": 0} for i in range(10)]
             + [{"item_id": str(50 + i), "s1": 0, "s2": 1} for i in range(5)]
             + [{"item_id": str(55 + i), "s1": 0, "s2": 0} for i in range(45)])
    s = summarise(pairs, draws=2000, seed=1)
    assert s["n"] == 100 and s["agreement"] == 0.85 and s["flips_1to0"] == 10 and s["flips_0to1"] == 5, s
    assert abs(s["kappa"] - 0.70) < 1e-9, s["kappa"]          # po .85, pe .5 -> (.85-.5)/.5
    assert s["rate_draw1"] == 0.5 and s["rate_draw2"] == 0.45 and s["shift_draw2_minus_draw1_points"] == -5.0
    assert s["kappa_ci95"][0] <= s["kappa"] <= s["kappa_ci95"][1], s["kappa_ci95"]
    assert s["shift_ci95_points"][0] <= -5.0 <= s["shift_ci95_points"][1]
    # 2. perfect agreement -> kappa 1 with a degenerate CI; one-label data -> kappa undefined
    perf = [{"item_id": str(i), "s1": i % 2, "s2": i % 2} for i in range(60)]
    s = summarise(perf, draws=500, seed=2)
    assert s["kappa"] == 1.0 and s["kappa_ci95"] == [1.0, 1.0] and s["flip_rate"] == 0.0
    same = [{"item_id": str(i), "s1": 1, "s2": 1} for i in range(30)]
    assert summarise(same, draws=200, seed=3)["kappa"] is None
    # 3. clustering: two pairs per item resampled together (n_items counted)
    two = [{"item_id": str(i // 2), "s1": 1, "s2": 1 - (i % 2)} for i in range(40)]
    assert summarise(two, draws=200, seed=4)["n_items"] == 20
    # 4. golden cache-key names produced by the original functions on fixed strings
    q = "Q" * 600 + "tail-of-question"
    a = "A" * 2500 + "tail-of-advice"
    assert inline_key("validation", q, a, "claude-haiku-4-5").name == \
        "elephant_score_validation_claude-haiku-4-5_d5bcf6f5086622ab.json"
    assert full_key("validation", q, a, "claude-haiku-4-5", None).name == \
        "elephant_rescore_validation_claude-haiku-4-5_full_01ca141e71ecc98a.json"
    assert inline_key("validation", q, a, "gpt-5.4-nano").name == \
        "elephant_score_validation_gpt-5.4-nano_fbebe7efcd2e50e3.json"          # reasoning judge, |r2
    assert full_key("validation", q, a, "claude-haiku-4-5", 4000).name == \
        "elephant_rescore_validation_claude-haiku-4-5_4000_8f08e61d198fb2a5.json"
    # 5. offline re-draw of the seed-44 sample matches the original loader (skipped if the CSV is absent)
    if OEQ_CSV.exists():
        qs = load_questions_offline()
        ids = list(qs)
        assert len(qs) == 150 and ids[0] == "1422" and ids[1] == "2830" and ids[-1] == "588", ids[:2]
        digest = hashlib.sha1("|".join(f"{k}:{v}" for k, v in qs.items()).encode()).hexdigest()
        assert digest == "8a8c79cc1bdff9d8cb8201a19f22a723b78ccf58", digest
        print("  sample re-draw: 150 items, ids and prompts match the original loader")
    else:
        print("  sample re-draw check skipped (data/elephant/OEQ.csv absent)")
    # 6. no generator / network module was imported
    bad = [m for m in FORBIDDEN_MODULES if m in sys.modules]
    assert not bad, f"forbidden modules imported: {bad}"
    print("selftest OK")


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--batch-csv", nargs="*", type=Path, default=None,
                    help="per-batch CSVs to merge (default: scan divergence_study_outputs/ for "
                         "elephant_singleagent_raw_batch_*.csv)")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return
    res = analyse(a.draws, a.seed, a.batch_csv)
    bad = [m for m in FORBIDDEN_MODULES if m in sys.modules]
    assert not bad, f"forbidden modules imported: {bad}"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    c, p = res["counts"], res["pooled"]
    print(f"rows {c['rows_nonempty']}  draw1 present {c['draw1_present']}  draw2 present {c['draw2_present']}  "
          f"both parsed {c['both_parsed']}  identical-input pairs {c['identical_input_pairs']}  "
          f"truncated excluded {c['truncated_input_pairs_excluded']}")
    print(f"pooled test-retest: n {p['n']} items {p['n_items']}  agreement {p['agreement']} {p['agreement_ci95']}  "
          f"kappa {p['kappa']} {p['kappa_ci95']}  flips 1->0 {p['flips_1to0']}  0->1 {p['flips_0to1']}  "
          f"rates {p['rate_draw1']} -> {p['rate_draw2']}  shift {p['shift_draw2_minus_draw1_points']} "
          f"{p['shift_ci95_points']}")
    s = res["pooled_strict_le2000"]
    print(f"strict <=2000: n {s['n']}  agreement {s['agreement']}  kappa {s['kappa']} {s['kappa_ci95']}")
    print("per cell (identical-input pairs): n/scored, agreement, kappa [CI], flips 1->0 / 0->1, rates d1 -> d2")
    for g in GENERATORS:
        for arm in ARMS:
            x = res["per_cell"][g][arm]
            if x["n"]:
                print(f"  {g:26s} {arm:14s} {x['n']:3d}/{x['n_scored_both_draws']:3d}  {x['agreement']:.3f}  "
                      f"{x['kappa']!s:6s} {x['kappa_ci95']}  {x['flips_1to0']}/{x['flips_0to1']}  "
                      f"{x['rate_draw1']} -> {x['rate_draw2']}{'  THIN' if x['thin'] else ''}")
            else:
                print(f"  {g:26s} {arm:14s}   0/{x['n_scored_both_draws']:3d}  (no identical-input pair)")
    t = res["not_test_retest_truncated_vs_full"]["pooled"]
    print(f"NOT test-retest (truncated vs full input): n {t.get('n')}  agreement {t.get('agreement')}  "
          f"kappa {t.get('kappa')} {t.get('kappa_ci95')}")
    print(f"wrote {a.out}  ({res['elapsed_seconds']}s)")


if __name__ == "__main__":
    main()
