"""
scripts/analyze_interjudge_kappa.py -- inter-judge agreement on the 17.8 model responses:
Cohen's kappa of every judge against the production judge (and of every pair) on the 2,004
CoT/NoT OEQ responses, and the base-rate region each judge operates in.

WHY (review item W33, 2026-09-23). The 17.8 artefacts report raw agreement with the
production judge (0.780 gpt-4o, 0.718 nano, 0.669 grok, 0.655 Llama). Raw agreement is
inflated by chance where two judges both call most responses validating, which is exactly
the situation of the Llama and grok judges (most CoT/NoT cells above 0.85). Kappa removes
the chance agreement. The same base rates bound how high kappa can go given two judges'
marginal rates (kappa_max), so the JSON also records, per judge, the region of validation
rates it sits in on these responses (overall, per arm, per generator x arm cell) and sets
it beside the reference rate on ELEPHANT's human-written answers (29.5 percent, from
judge_reference_agreement.json), which lies far below the 54 to 97 percent span where the
judges split on CoT responses.

POST HOC. Added after the 17.8 tables were seen, from the stored scores only: this script
makes no model calls and touches nothing outside divergence_study_outputs/ except the
ELEPHANT OEQ.csv it needs to join each cached score to its question. It does not import
any module of scripts/ (it re-implements the row loader, the seed-44 item sample and the
cache-key hash of rescore_elephant_untruncated in place). Spend: zero.

Design, matching the 17.8 readouts: item-clustered percentile bootstrap, 8,000 draws,
seed 20260822, each draw resampling items with every response of an item kept together.
Kappa on a draw is computed from the 2x2 agreement counts, which equals the flat-list
formula of scripts/judge_reference_agreement.kappa.

Run:
  python -m scripts.analyze_interjudge_kappa            # writes divergence_study_outputs/interjudge_kappa.json
  python -m scripts.analyze_interjudge_kappa --selftest
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "divergence_study_outputs"
RAW = OUT_DIR / "elephant_singleagent_raw.csv"
OEQ_CSV = ROOT / "data" / "elephant" / "OEQ.csv"
REFERENCE_JSON = OUT_DIR / "judge_reference_agreement.json"
OUT = OUT_DIR / "interjudge_kappa.json"

METRIC = "validation"
DATASET = "oeq"
N_ITEMS = 150
ELEPHANT_SEED = 44               # scripts/load_elephant.ELEPHANT_SEED (the sample seed)
PRODUCTION_JUDGE = "claude-haiku-4-5"
JUDGES = ("claude-haiku-4-5", "gpt-5.4-nano", "Llama-3.3-70B-Instruct",
          "grok-4-1-fast-reasoning", "gpt-4o")
MODELS = ("claude-haiku-4-5", "claude-sonnet-4-6", "gpt-5.4-nano", "grok-4-1-fast-reasoning",
          "Llama-3.3-70B-Instruct", "Mistral-Large-3-2", "DeepSeek-V4-Pro")
ARMS = ("standard_cot", "narrative_cot")
CEILING = 0.85                   # the descriptive ceiling line of 17.8 (share of cells above it)
DRAWS, SEED = 8000, 20260822     # the 17.8 bootstrap


# ----------------------------------------------------------------------------- data access

def _safe(s: str) -> str:
    """scripts/run_phase1_quartet._safe, verbatim."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def cache_path(question: str, advice: str, judge: str) -> Path:
    """scripts/rescore_elephant_untruncated._key with limit=None (full text), verbatim."""
    h = hashlib.sha1(f"{METRIC}|{question}|{advice}|{judge}|lim=full".encode()).hexdigest()[:16]
    return OUT_DIR / f"elephant_rescore_{METRIC}_{_safe(judge)}_full_{h}.json"


def cached_score(question: str, advice: str, judge: str) -> Optional[int]:
    p = cache_path(question, advice, judge)
    if not p.exists():
        return None
    try:
        return int(json.loads(p.read_text()).get("score", -1))
    except Exception:  # noqa: BLE001
        return None


def load_rows(models: Sequence[str] = MODELS, arms: Sequence[str] = ARMS) -> list[dict]:
    """scripts/rescore_elephant_full_judge.load_rows: the shared results CSV plus every
    per-batch CSV, one row per (model, arm, item), later files win, empty responses dropped."""
    paths = [RAW] + sorted(OUT_DIR.glob("elephant_singleagent_raw_batch_*.csv"))
    keep: dict = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open() as fh:
            for r in csv.DictReader(fh):
                if (r["dataset"] == DATASET and r["arm"] in arms and r["generator"] in models
                        and (r.get("regime") or "default") == "default"):
                    keep[(r["generator"], r["arm"], r["item_id"])] = r
    return [r for r in keep.values() if (r.get("response") or "").strip()]


def load_questions(n: int = N_ITEMS, seed: int = ELEPHANT_SEED) -> dict[str, str]:
    """item_id -> prompt, the seed-44 sample of scripts/load_elephant.load_elephant('oeq', n=150)
    re-implemented in place (same shuffle, same id rule, same prompt handling)."""
    df = pd.read_csv(OEQ_CSV)
    rng = random.Random(seed)
    idxs = list(range(len(df)))
    rng.shuffle(idxs)
    df = df.iloc[idxs[:n]].reset_index(drop=True)
    out = {}
    for i, row in df.iterrows():
        rid = str(i)
        for col in ("id", "Unnamed: 0"):
            if col in row.index and pd.notna(row[col]):
                rid = str(row[col]).strip()
                break
        out[rid] = str(row.get("prompt", "") or "").strip()
    return out


def scores_by_judge(rows: list[dict], questions: dict[str, str], judges: Sequence[str] = JUDGES) -> dict:
    """judge -> {(model, arm, item): 0/1}; unparsed (-1) and missing counted separately."""
    out = {}
    for j in judges:
        scores, unparsed, missing = {}, 0, 0
        for r in rows:
            s = cached_score(questions[r["item_id"]], r["response"], j)
            if s is None:
                missing += 1
            elif s not in (0, 1):
                unparsed += 1
            else:
                scores[(r["generator"], r["arm"], r["item_id"])] = s
        out[j] = {"scores": scores, "unparsed": unparsed, "missing": missing}
    return out


# ----------------------------------------------------------------------------- statistics

def kappa_from_counts(n11: float, n10: float, n01: float, n00: float) -> float:
    """Cohen's kappa for two binary raters from the 2x2 counts (rows rater A, columns rater B).
    Equals scripts/judge_reference_agreement.kappa on the flat lists."""
    n = n11 + n10 + n01 + n00
    if n == 0:
        return float("nan")
    po = (n11 + n00) / n
    pa, pb = (n11 + n10) / n, (n11 + n01) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def kappa_max(pa: float, pb: float) -> float:
    """The largest kappa two raters with marginal rates pa and pb can reach (observed agreement
    at its ceiling 1 - |pa - pb|). A base-rate bound, not an estimate."""
    pe = pa * pb + (1 - pa) * (1 - pb)
    po_max = 1 - abs(pa - pb)
    return (po_max - pe) / (1 - pe) if pe < 1 else float("nan")


def counts_by_item(keys: Sequence[tuple], a: dict, b: dict):
    """Per item, the 2x2 counts (n11, n10, n01, n00) over that item's responses."""
    items = sorted({k[2] for k in keys})
    pos = {it: i for i, it in enumerate(items)}
    c = np.zeros((len(items), 4), dtype=np.int64)
    for k in keys:
        x, y = a[k], b[k]
        c[pos[k[2]], 0 if (x, y) == (1, 1) else 1 if (x, y) == (1, 0) else 2 if (x, y) == (0, 1) else 3] += 1
    return items, c


def pair_stats(keys: Sequence[tuple], a: dict, b: dict, *, draws: int = DRAWS, seed: int = SEED,
               ci: bool = True) -> dict:
    """Agreement, kappa and an item-clustered percentile bootstrap interval for two judges on
    the given keys. Draw t resamples the items with replacement (one rng.integers call per
    draw, so the stream matches the 17.8-era scratch computation) and keeps every response of
    a drawn item."""
    if not keys:
        return {"n": 0, "n_items": 0, "agreement": None, "kappa": None, "kappa_lo": None, "kappa_hi": None,
                "rate_1": None, "rate_2": None, "kappa_max_given_marginals": None, "prevalence_index": None}
    items, c = counts_by_item(keys, a, b)
    tot = c.sum(0)
    n = int(tot.sum())
    k = kappa_from_counts(*tot)
    rate_1 = float((tot[0] + tot[1]) / n)
    rate_2 = float((tot[0] + tot[2]) / n)
    res = {"n": n, "n_items": len(items), "agreement": float((tot[0] + tot[3]) / n), "kappa": float(k),
           "kappa_lo": None, "kappa_hi": None, "rate_1": rate_1, "rate_2": rate_2,
           "kappa_max_given_marginals": float(kappa_max(rate_1, rate_2)),
           "prevalence_index": float((tot[0] - tot[3]) / n),
           "counts": {"both_1": int(tot[0]), "1_then_0": int(tot[1]), "0_then_1": int(tot[2]), "both_0": int(tot[3])}}
    if ci:
        rng = np.random.default_rng(seed)
        ks = np.empty(draws)
        for t in range(draws):
            idx = rng.integers(0, len(items), len(items))
            ks[t] = kappa_from_counts(*c[idx].sum(0))
        res["kappa_lo"] = float(np.nanpercentile(ks, 2.5))
        res["kappa_hi"] = float(np.nanpercentile(ks, 97.5))
    return res


def base_rate_region(scores: dict, models: Sequence[str] = MODELS, arms: Sequence[str] = ARMS) -> dict:
    """Where a judge sits on these responses: rate overall, per arm, per model x arm cell, the
    span of the cell rates per arm, and the share of cells above the descriptive ceiling."""
    by_cell = defaultdict(list)
    by_arm = defaultdict(list)
    for (g, arm, _i), s in scores.items():
        by_cell[(g, arm)].append(s)
        by_arm[arm].append(s)
    cells = {f"{g}|{arm}": float(np.mean(by_cell[(g, arm)])) for g in models for arm in arms if by_cell.get((g, arm))}
    rates_all = [v for v in cells.values()]
    out = {"n_responses": len(scores), "rate_all": float(np.mean(list(scores.values()))) if scores else None,
           "rate_by_arm": {arm: (float(np.mean(by_arm[arm])) if by_arm.get(arm) else None) for arm in arms},
           "cell_rates": cells, "n_cells": len(cells),
           "cell_rate_span": {"min": float(min(rates_all)), "max": float(max(rates_all))} if rates_all else None,
           "cell_rate_span_by_arm": {}, "frac_cells_above_0.85": (float(np.mean([r > CEILING for r in rates_all]))
                                                                  if rates_all else None)}
    for arm in arms:
        rs = [v for k, v in cells.items() if k.endswith(f"|{arm}")]
        out["cell_rate_span_by_arm"][arm] = {"min": float(min(rs)), "max": float(max(rs))} if rs else None
    return out


# ----------------------------------------------------------------------------- report

def run(rows: list[dict], questions: dict[str, str], judges: Sequence[str] = JUDGES,
        draws: int = DRAWS, seed: int = SEED) -> dict:
    tabs = scores_by_judge(rows, questions, judges)
    S = {j: tabs[j]["scores"] for j in judges}

    def block(j1: str, j2: str) -> dict:
        shared = sorted(set(S[j1]) & set(S[j2]))
        b = {"all": pair_stats(shared, S[j1], S[j2], draws=draws, seed=seed)}
        for arm in ARMS:
            b[arm] = pair_stats([k for k in shared if k[1] == arm], S[j1], S[j2], draws=draws, seed=seed)
        return b

    res = {
        "generated": date.today().isoformat(),
        "script": "scripts/analyze_interjudge_kappa.py",
        "status": "post hoc (review item W33, added 2026-09-23 after the 17.8 tables were seen); "
                  "computed from the stored full-text scores only; no model calls; spend 0",
        "metric": METRIC, "dataset": DATASET, "production_judge": PRODUCTION_JUDGE,
        "judges": list(judges), "generators": list(MODELS), "arms": list(ARMS),
        "n_responses": len(rows),
        "bootstrap": {"draws": draws, "seed": seed, "cluster": "item",
                      "interval": "percentile 2.5 / 97.5"},
        "per_judge_coverage": {j: {"n_scored": len(S[j]), "unparsed": tabs[j]["unparsed"], "missing": tabs[j]["missing"]}
                               for j in judges},
        "kappa_vs_production": {},
        "kappa_all_pairs": {},
        "base_rate_region": {j: base_rate_region(S[j]) for j in judges},
        "region_where_judges_split": {},
        "reference_human_answers": None,
        "notes": [
            "kappa_vs_production[judge][all|standard_cot|narrative_cot]: rate_1 is the judge named, "
            "rate_2 the production judge; kappa_lo/kappa_hi are the item-clustered bootstrap bounds.",
            "kappa_max_given_marginals is the ceiling on kappa set by the two judges' marginal rates "
            "(observed agreement at 1 - |rate_1 - rate_2|); prevalence_index is (both_1 - both_0)/n.",
            "base_rate_region[judge]: rates on these responses; frac_cells_above_0.85 is the "
            "descriptive ceiling statistic of the 17.8 amendment and excludes no judge.",
            "The reference rate comes from judge_reference_agreement.json (149 human-written answers "
            "with a reference label); it is the ELEPHANT release's label, not a judge's.",
        ],
    }
    for j in judges:
        if j != PRODUCTION_JUDGE:
            res["kappa_vs_production"][j] = block(j, PRODUCTION_JUDGE)
    for j1, j2 in combinations(judges, 2):
        res["kappa_all_pairs"][f"{j1} | {j2}"] = block(j1, j2)

    # The span, over every judge, of the per-cell rates: where the panel's judges disagree.
    for arm in ARMS:
        spans = [res["base_rate_region"][j]["cell_rate_span_by_arm"][arm] for j in judges]
        res["region_where_judges_split"][arm] = {"min_over_judges": min(s["min"] for s in spans),
                                                 "max_over_judges": max(s["max"] for s in spans)}
    if REFERENCE_JSON.exists():
        ref = json.loads(REFERENCE_JSON.read_text())
        cot = res["region_where_judges_split"]["standard_cot"]
        res["reference_human_answers"] = {
            "source": REFERENCE_JSON.name, "n_items": ref["n_items"], "reference_rate": ref["reference_rate"],
            "judge_rate_on_human_answers": {j: ref["judges"][j]["judge_rate"] for j in judges if j in ref["judges"]},
            "kappa_with_reference": {j: ref["judges"][j]["kappa"] for j in judges if j in ref["judges"]},
            "lies_below_cot_region": bool(ref["reference_rate"] < cot["min_over_judges"]),
            "gap_to_cot_region_min": float(cot["min_over_judges"] - ref["reference_rate"]),
        }
    return res


def print_report(res: dict) -> None:
    P = PRODUCTION_JUDGE
    print(f"inter-judge kappa on {res['n_responses']} CoT/NoT OEQ responses; production judge {P}; "
          f"bootstrap {res['bootstrap']['draws']} draws seed {res['bootstrap']['seed']}, item-clustered")
    print("\nkappa vs production judge")
    for j, b in res["kappa_vs_production"].items():
        a, c, n = b["all"], b["standard_cot"], b["narrative_cot"]
        print(f"  {j:24s} all n={a['n']} agree={a['agreement']:.3f} kappa={a['kappa']:.3f} "
              f"[{a['kappa_lo']:.3f},{a['kappa_hi']:.3f}] rates {a['rate_1']:.3f}/{a['rate_2']:.3f} "
              f"kappa_max={a['kappa_max_given_marginals']:.3f} | CoT kappa {c['kappa']:.3f} "
              f"[{c['kappa_lo']:.3f},{c['kappa_hi']:.3f}] | NoT kappa {n['kappa']:.3f} "
              f"[{n['kappa_lo']:.3f},{n['kappa_hi']:.3f}]")
    print("\nall pairs (all responses)")
    for k, b in res["kappa_all_pairs"].items():
        a = b["all"]
        print(f"  {k:55s} n={a['n']} agree={a['agreement']:.3f} kappa={a['kappa']:.3f} "
              f"[{a['kappa_lo']:.3f},{a['kappa_hi']:.3f}] CoT {b['standard_cot']['kappa']:.3f} "
              f"NoT {b['narrative_cot']['kappa']:.3f}")
    print("\nbase-rate region per judge (rate on these responses)")
    for j, r in res["base_rate_region"].items():
        c, n = r["cell_rate_span_by_arm"]["standard_cot"], r["cell_rate_span_by_arm"]["narrative_cot"]
        print(f"  {j:24s} all {r['rate_all']:.3f}  CoT {r['rate_by_arm']['standard_cot']:.3f} "
              f"(cells {c['min']:.3f}-{c['max']:.3f})  NoT {r['rate_by_arm']['narrative_cot']:.3f} "
              f"(cells {n['min']:.3f}-{n['max']:.3f})  cells>0.85 {r['frac_cells_above_0.85']:.2f}")
    s = res["region_where_judges_split"]
    print(f"\nregion where the judges split: CoT cells {s['standard_cot']['min_over_judges']:.3f}-"
          f"{s['standard_cot']['max_over_judges']:.3f}, NoT cells {s['narrative_cot']['min_over_judges']:.3f}-"
          f"{s['narrative_cot']['max_over_judges']:.3f}")
    ref = res.get("reference_human_answers")
    if ref:
        print(f"reference rate on {ref['n_items']} human-written answers {ref['reference_rate']:.3f}; "
              f"below the CoT region: {ref['lies_below_cot_region']} (gap {ref['gap_to_cot_region_min']:.3f})")


# ----------------------------------------------------------------------------- selftest

def _selftest() -> int:
    ok = True

    def check(msg, cond):
        nonlocal ok
        print(("[ok] " if cond else "[FAIL] ") + msg)
        ok = ok and cond

    def kappa_flat(a, b):  # scripts/judge_reference_agreement.kappa, copied for the check
        a, b = np.array(a), np.array(b)
        po = float(np.mean(a == b))
        pa, pb = a.mean(), b.mean()
        pe = pa * pb + (1 - pa) * (1 - pb)
        return (po - pe) / (1 - pe) if pe < 1 else float("nan")

    check("kappa 1 on identical raters", abs(kappa_from_counts(2, 0, 0, 2) - 1) < 1e-12)
    check("kappa 0 on independent raters", abs(kappa_from_counts(1, 1, 1, 1)) < 1e-12)
    check("kappa nan when both raters are constant", np.isnan(kappa_from_counts(4, 0, 0, 0)))
    rng = np.random.default_rng(7)
    a = list(rng.integers(0, 2, 500)); b = [x if rng.random() < 0.8 else 1 - x for x in a]
    n11 = sum(1 for x, y in zip(a, b) if (x, y) == (1, 1)); n10 = sum(1 for x, y in zip(a, b) if (x, y) == (1, 0))
    n01 = sum(1 for x, y in zip(a, b) if (x, y) == (0, 1)); n00 = sum(1 for x, y in zip(a, b) if (x, y) == (0, 0))
    check("counts kappa equals the flat-list kappa", abs(kappa_from_counts(n11, n10, n01, n00) - kappa_flat(a, b)) < 1e-12)
    check("kappa_max is 1 at equal marginals and below 1 otherwise",
          abs(kappa_max(0.9, 0.9) - 1) < 1e-12 and kappa_max(0.9, 0.6) < 1)
    check("kappa never exceeds kappa_max", kappa_from_counts(n11, n10, n01, n00) <= kappa_max(np.mean(a), np.mean(b)) + 1e-12)
    # pair_stats on a synthetic two-item panel: a judge that agrees perfectly on item A and never on B.
    A = {("m", "standard_cot", "A"): 1, ("m", "narrative_cot", "A"): 0, ("m", "standard_cot", "B"): 1, ("m", "narrative_cot", "B"): 0}
    B = {("m", "standard_cot", "A"): 1, ("m", "narrative_cot", "A"): 0, ("m", "standard_cot", "B"): 0, ("m", "narrative_cot", "B"): 1}
    ps = pair_stats(sorted(A), A, B, draws=200, seed=1)
    check("pair_stats: n 4, agreement 0.5, kappa 0, rates 0.5/0.5",
          ps["n"] == 4 and abs(ps["agreement"] - 0.5) < 1e-12 and abs(ps["kappa"]) < 1e-12 and ps["rate_1"] == ps["rate_2"] == 0.5)
    check("pair_stats: item-clustered draws reach kappa 1 (only A) and -1 (only B)",
          ps["kappa_lo"] <= -1 + 1e-9 and ps["kappa_hi"] >= 1 - 1e-9)
    br = base_rate_region({("x", "standard_cot", str(i)): 1 for i in range(10)} | {("x", "narrative_cot", str(i)): int(i < 3) for i in range(10)}, ("x",))
    check("base_rate_region: rates 1.0 CoT, 0.3 NoT, one of two cells above 0.85",
          abs(br["rate_by_arm"]["standard_cot"] - 1) < 1e-12 and abs(br["rate_by_arm"]["narrative_cot"] - 0.3) < 1e-12
          and abs(br["frac_cells_above_0.85"] - 0.5) < 1e-12 and br["cell_rate_span"] == {"min": 0.3, "max": 1.0})
    check("cache path has the rescore namespace and a 16-hex hash",
          re.fullmatch(r"elephant_rescore_validation_gpt-4o_full_[0-9a-f]{16}\.json", cache_path("q", "a", "gpt-4o").name) is not None)
    if RAW.exists() and OEQ_CSV.exists():
        rows = load_rows()
        q = load_questions()
        check("loads the seven-model CoT/NoT OEQ rows (> 1,900 responses)", len(rows) > 1900)
        check("the seed-44 sample has 150 items and every row has its question",
              len(q) == 150 and all(r["item_id"] in q for r in rows))
        t = scores_by_judge(rows, q, (PRODUCTION_JUDGE,))[PRODUCTION_JUDGE]
        check("every row has a cached full-text production-judge score (join and hash are right)",
              t["missing"] == 0 and len(t["scores"]) + t["unparsed"] == len(rows))
    else:
        print("[skip] data files absent; loader checks not run")
    print("\nselftest: " + ("ALL OK" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default=",".join(JUDGES))
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--json", type=Path, default=OUT)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    judges = [x for x in a.judges.split(",") if x]
    if PRODUCTION_JUDGE not in judges:
        ap.error(f"--judges must include the production judge {PRODUCTION_JUDGE}")
    rows = load_rows()
    questions = load_questions()
    missing_q = {r["item_id"] for r in rows} - set(questions)
    if missing_q:
        raise SystemExit(f"{len(missing_q)} item(s) without a question; refusing")
    res = run(rows, questions, judges, draws=a.draws, seed=a.seed)
    print_report(res)
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
