"""
scripts/analyze_run_stability.py -- run-to-run stability of the NoT minus CoT validation drop
(ELEPHANT OEQ, 150 seed-44 items), every generation run scored in full by every 17.8 judge.

NOT PRE-REGISTERED. Written 2026-09-23 after every run it reads had been generated and scored, in
answer to a reviewer question about run-to-run stability. CACHE-ONLY: it makes no model call and
no network call, imports nothing from the scripts package (so no generator or API module), and reads
only (a) the results CSVs and the `elephant_rescore_validation_<judge>_full_*` judge caches under
divergence_study_outputs/, (b) data/elephant/OEQ.csv for the item prompts (the cache key hashes the
prompt), and (c) three readout artefacts for reconciliation. A response without a cached full-text
score is dropped from that judge's pairing and counted in `missing_scores`; a cached score that is
not 0/1 is counted in `unparsed`.

The CoT/NoT contrast was generated two or three times per generator, one sample per prompt and
item in each run, and every run was scored by the five judges of Addendum 17.8:

  run 1  standard_cot / narrative_cot            the original cells: June 2026 for haiku, sonnet,
                                                 nano and grok; 2026-09-21 (Addendum 17.1) for Llama,
                                                 Mistral and DeepSeek; CoT capped at 1,024 tokens and
                                                 NoT at 2,048 (nano: 2,048 total in June), with the
                                                 truncation shares recorded in the 17.10 registration
  run 2  standard_cot_rep / narrative_cot_rep    2026-09-22 replicate, both prompts at 2,048 tokens,
                                                 the 17.7 comparator (haiku, grok, nano, Llama only)
  run 3  standard_cot_rep4k / narrative_cot_rep4k  2026-09-23 replicate, both prompts at 4,096 tokens
                                                 (Addendum 17.10, all seven generators)

So a run-to-run difference bundles sampling noise with a change of caps (and, for three
generators, of month); the design cannot separate them. The same 150 items are used in every run.

Per judge and generator the artefact carries
  per_run[r]      n_items, rate_cot, rate_not, drop, lo, hi: the paired NoT minus CoT difference on
                  the items scored in both arms, item-clustered bootstrap (8,000 draws, seed
                  20260822, the panel's settings; run 1 and run 3 reproduce
                  judge_panel_full_oeq_<judge>.json and headline_rep4k_readout.json, run 2
                  judge_panel_newarms_<judge>.json, checked in `reconciliation`); n_down / n_same /
                  n_up count items whose score fell, stayed or rose
  cross_run[a_vs_b]  on the items paired in both runs: Pearson r between the item-level drops of
                  the two runs (item bootstrap CI), the share of items with the same drop, the
                  Pearson r of the CoT scores and of the NoT scores across the runs, and how many
                  response texts are byte-identical across the runs (a cache hit would show here)
  pooled          run as a FIXED crossed factor: the mean of the per-run drops (equal run weights),
                  interval from an item-clustered bootstrap that resamples items and keeps every
                  run of a resampled item (so the crossing item x run is preserved)
  pooled_random_run  run as a RANDOM crossed factor on the balanced subset (items paired in every
                  run): two-way variance components by the method of moments (item, run,
                  residual = item x run + noise), the run-to-run SD of the drop, the single-run
                  reliability of an item's drop (ICC), and a normal interval whose SE carries
                  sigma_run^2 / R + sigma_item^2 / I + sigma_e^2 / (I R). With R = 2 or 3 the run
                  component has 1 or 2 degrees of freedom and is clipped at 0; read it as a
                  description, not a test
  stability       whether every run's drop is negative with a CI excluding 0, whether the sign
                  flips across runs, and the range of the run drops in points

Run:  python -m scripts.analyze_run_stability            (writes divergence_study_outputs/run_stability.json)
      python -m scripts.analyze_run_stability --selftest
      python -m scripts.analyze_run_stability --scan    (discover extra elephant_singleagent_raw_batch_*.csv;
                                                        ~2 min in this 800k-entry directory)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "divergence_study_outputs"
OEQ_CSV = ROOT / "data" / "elephant" / "OEQ.csv"
OUT = OUT_DIR / "run_stability.json"
RAW = OUT_DIR / "elephant_singleagent_raw.csv"
# The per-batch CSVs written by run_elephant --out-csv on 2026-09-22/23 (17.7, 17.4, 17.10). Named
# rather than globbed: a glob of this 800k-entry directory takes about two minutes. --scan discovers
# any further elephant_singleagent_raw_batch_*.csv; --batch-csv adds one by name.
DEFAULT_BATCH_CSVS = (
    "elephant_singleagent_raw_batch_1710_haiku.csv",
    "elephant_singleagent_raw_batch_1710_mistral_deepseek.csv",
    "elephant_singleagent_raw_batch_1710_nano.csv",
    "elephant_singleagent_raw_batch_1710_sonnet.csv",
    "elephant_singleagent_raw_batch_177_1710_grok.csv",
    "elephant_singleagent_raw_batch_177_1710_llama.csv",
)

METRIC = "validation"
ELEPHANT_SEED, N_ITEMS = 44, 150
PRODUCTION_JUDGE = "claude-haiku-4-5"
JUDGES = ("claude-haiku-4-5", "gpt-4o", "gpt-5.4-nano", "Llama-3.3-70B-Instruct", "grok-4-1-fast-reasoning")
GENERATORS = ("claude-haiku-4-5", "claude-sonnet-4-6", "gpt-5.4-nano", "grok-4-1-fast-reasoning",
              "Llama-3.3-70B-Instruct", "Mistral-Large-3-2", "DeepSeek-V4-Pro")
RUNS = {"1": ("standard_cot", "narrative_cot"),
        "2": ("standard_cot_rep", "narrative_cot_rep"),
        "3": ("standard_cot_rep4k", "narrative_cot_rep4k")}
RUN_LABELS = {"1": "original cells (CoT 1,024 / NoT 2,048 tokens; June 2026 for haiku, sonnet, nano, grok; "
                   "2026-09-21 for Llama, Mistral, DeepSeek)",
              "2": "2026-09-22 replicate, both prompts at 2,048 tokens (17.7 comparator; haiku, grok, nano, Llama)",
              "3": "2026-09-23 replicate, both prompts at 4,096 tokens (17.10; all seven generators)"}
RUN1_COHORT = {"claude-haiku-4-5": "June 2026", "claude-sonnet-4-6": "June 2026", "gpt-5.4-nano": "June 2026",
               "grok-4-1-fast-reasoning": "June 2026", "Llama-3.3-70B-Instruct": "2026-09-21 (17.1)",
               "Mistral-Large-3-2": "2026-09-21 (17.1)", "DeepSeek-V4-Pro": "2026-09-21 (17.1)"}
EXTRA_RUN_RE = re.compile(r"^standard_cot_rep4k_(?P<sfx>[A-Za-z0-9]+)$")
DRAWS, SEED, MIN_ITEMS = 8000, 20260822, 20


# ----------------------------------------------------------------------------- data (cache only)
def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def cache_key(question: str, advice: str, judge: str, metric: str = METRIC) -> Path:
    """The judge cache path of rescore_elephant_untruncated._key with limit=None (full text)."""
    h = hashlib.sha1(f"{metric}|{question}|{advice}|{judge}|lim=full".encode()).hexdigest()[:16]
    return OUT_DIR / f"elephant_rescore_{metric}_{_safe(judge)}_full_{h}.json"


def cached_score(question: str, advice: str, judge: str) -> Optional[int]:
    k = cache_key(question, advice, judge)
    if not k.exists():
        return None
    try:
        return int(json.loads(k.read_text()).get("score", -1))
    except Exception:  # noqa: BLE001
        return -1


def load_questions(csv_path: Path = OEQ_CSV, n: int = N_ITEMS, seed: int = ELEPHANT_SEED) -> dict[str, str]:
    """item_id -> prompt, the seed-44 sample exactly as scripts.load_elephant.load_elephant draws it
    (random.Random(seed).shuffle over the row indices, first n rows, id from `id` or `Unnamed: 0`)."""
    df = pd.read_csv(csv_path)
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


def batch_csvs(scan: bool, extra: Sequence[str]) -> list[Path]:
    names = set(DEFAULT_BATCH_CSVS) | set(extra)
    if scan:
        with os.scandir(OUT_DIR) as it:
            names |= {e.name for e in it
                      if e.name.startswith("elephant_singleagent_raw_batch_") and e.name.endswith(".csv")}
    return [OUT_DIR / n for n in sorted(names)]


def load_rows(generators: Sequence[str], arms: Sequence[str], paths: Sequence[Path],
              dataset: str = "oeq") -> list[dict]:
    """One row per (generator, arm, item); later files win; empty responses dropped (the panel's rule)."""
    keep: dict = {}
    armset = set(arms)
    for path in paths:
        if not path.exists():
            continue
        with path.open() as fh:
            for r in csv.DictReader(fh):
                if (r["dataset"] == dataset and r["generator"] in generators
                        and (r.get("regime") or "default") == "default"
                        and (r["arm"] in armset or EXTRA_RUN_RE.match(r["arm"])
                             or (r["arm"].startswith("narrative_cot_rep4k_")))):
                    keep[(r["generator"], r["arm"], r["item_id"])] = {"generator": r["generator"], "arm": r["arm"],
                                                                     "item_id": r["item_id"], "response": r["response"],
                                                                     "source": path.name}
    return [r for r in keep.values() if (r.get("response") or "").strip()]


def discover_extra_runs(rows: list[dict]) -> dict[str, tuple[str, str]]:
    found = {}
    arms = {r["arm"] for r in rows}
    for a in sorted(arms):
        m = EXTRA_RUN_RE.match(a)
        if m and f"narrative_cot_rep4k_{m['sfx']}" in arms:
            found[f"3{m['sfx']}"] = (a, f"narrative_cot_rep4k_{m['sfx']}")
    return found


# ----------------------------------------------------------------------------- statistics
def _boot_idx(n: int, draws: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(draws, n))


def paired_drop(a: dict, b: dict, *, draws: int = DRAWS, seed: int = SEED) -> dict:
    """rate(b) - rate(a) on items scored in both; item bootstrap; the panel's paired_drop, ported."""
    items = sorted(set(a) & set(b))
    if len(items) < MIN_ITEMS:
        return {"n_items": len(items), "drop": None, "lo": None, "hi": None}
    d = np.array([b[i] - a[i] for i in items], dtype=float)
    boots = d[_boot_idx(len(items), draws, seed)].mean(1)
    return {"n_items": len(items), "rate_cot": float(np.mean([a[i] for i in items])),
            "rate_not": float(np.mean([b[i] for i in items])), "drop": float(d.mean()),
            "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
            "n_down": int((d < 0).sum()), "n_same": int((d == 0).sum()), "n_up": int((d > 0).sum())}


def _pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def pearson_boot(x: np.ndarray, y: np.ndarray, *, draws: int = DRAWS, seed: int = SEED) -> dict:
    """Pearson r with an item-bootstrap percentile interval; draws with a constant series are skipped."""
    r = _pearson(x, y)
    if r is None:
        return {"n": int(len(x)), "r": None, "lo": None, "hi": None}
    idx = _boot_idx(len(x), draws, seed)
    xs, ys = x[idx], y[idx]
    xc, yc = xs - xs.mean(1, keepdims=True), ys - ys.mean(1, keepdims=True)
    den = np.sqrt((xc ** 2).sum(1) * (yc ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = (xc * yc).sum(1) / den
    rs = rs[np.isfinite(rs)]
    return {"n": int(len(x)), "r": r, "lo": float(np.percentile(rs, 2.5)), "hi": float(np.percentile(rs, 97.5)),
            "boot_draws_used": int(len(rs))}


def pooled_fixed_run(diffs: dict[str, dict[str, float]], *, draws: int = DRAWS, seed: int = SEED) -> dict:
    """Run as a fixed crossed factor: mean of the per-run drops; the bootstrap resamples items and keeps
    every run of a resampled item (missing cells stay missing), then averages the per-run means."""
    runs = sorted(diffs)
    items = sorted(set().union(*(set(diffs[r]) for r in runs)))
    m = np.full((len(items), len(runs)), np.nan)
    pos = {it: k for k, it in enumerate(items)}
    for j, r in enumerate(runs):
        for it, d in diffs[r].items():
            m[pos[it], j] = d
    point = float(np.nanmean(m, axis=0).mean())
    idx = _boot_idx(len(items), draws, seed)
    with np.errstate(invalid="ignore"):
        boots = np.nanmean(m[idx], axis=1).mean(1)          # (draws, runs) -> (draws,)
    boots = boots[np.isfinite(boots)]
    run_drops = {r: float(np.mean(list(diffs[r].values()))) for r in runs}
    return {"n_items": len(items), "n_runs": len(runs), "runs": runs, "drop": point,
            "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
            "run_drop_min": min(run_drops.values()), "run_drop_max": max(run_drops.values()),
            "range_points": 100 * (max(run_drops.values()) - min(run_drops.values()))}


def variance_components(diffs: dict[str, dict[str, float]]) -> Optional[dict]:
    """Two-way random-effects decomposition of d_ir (item i, run r) on the balanced subset, method of
    moments: sigma_e^2 = MS_e, sigma_run^2 = (MS_run - MS_e) / I, sigma_item^2 = (MS_item - MS_e) / R,
    both clipped at 0. Grand-mean SE with run random: sqrt(s_run^2/R + s_item^2/I + s_e^2/(I R))."""
    runs = sorted(diffs)
    if len(runs) < 2:
        return None
    items = sorted(set.intersection(*(set(diffs[r]) for r in runs)))
    I, R = len(items), len(runs)
    if I < MIN_ITEMS:
        return {"n_items": I, "n_runs": R, "drop": None}
    m = np.array([[diffs[r][it] for r in runs] for it in items], dtype=float)
    grand = m.mean()
    mi, mr = m.mean(1), m.mean(0)
    ms_item = R * ((mi - grand) ** 2).sum() / (I - 1)
    ms_run = I * ((mr - grand) ** 2).sum() / (R - 1)
    resid = m - mi[:, None] - mr[None, :] + grand
    ms_e = (resid ** 2).sum() / ((I - 1) * (R - 1))
    s2_run_raw, s2_item_raw = (ms_run - ms_e) / I, (ms_item - ms_e) / R
    s2_run, s2_item, s2_e = max(0.0, s2_run_raw), max(0.0, s2_item_raw), ms_e
    se = float(np.sqrt(s2_run / R + s2_item / I + s2_e / (I * R)))
    icc = (s2_item / (s2_item + s2_e)) if (s2_item + s2_e) > 0 else None
    return {"n_items": I, "n_runs": R, "runs": runs, "drop": float(grand), "se": se,
            "lo": float(grand - 1.96 * se), "hi": float(grand + 1.96 * se),
            "sigma_run_points": 100 * float(np.sqrt(s2_run)), "sigma_item": float(np.sqrt(s2_item)),
            "sigma_resid": float(np.sqrt(s2_e)), "sigma2_run_unclipped": float(s2_run_raw),
            "sigma2_item_unclipped": float(s2_item_raw), "icc_item_single_run": icc,
            "ms_run": float(ms_run), "ms_item": float(ms_item), "ms_error": float(ms_e),
            "df_run": R - 1, "run_means": {r: float(mr[j]) for j, r in enumerate(runs)}}


# ----------------------------------------------------------------------------- one judge x generator
def analyse(rows: list[dict], questions: dict, judge: str, generator: str, runs: dict,
            *, draws: int = DRAWS) -> dict:
    scores: dict[str, dict[str, int]] = {}
    texts: dict[str, dict[str, str]] = {}
    missing = unparsed = 0
    for r in rows:
        if r["generator"] != generator:
            continue
        s = cached_score(questions[r["item_id"]], r["response"], judge)
        if s is None:
            missing += 1
            continue
        if s not in (0, 1):
            unparsed += 1
            continue
        scores.setdefault(r["arm"], {})[r["item_id"]] = s
        texts.setdefault(r["arm"], {})[r["item_id"]] = r["response"]
    per_run, diffs, cot, nott = {}, {}, {}, {}
    for run, (cot_arm, not_arm) in runs.items():
        a, b = scores.get(cot_arm, {}), scores.get(not_arm, {})
        pr = paired_drop(a, b, draws=draws)
        pr.update({"arms": [cot_arm, not_arm], "n_cot_scored": len(a), "n_not_scored": len(b)})
        if pr["drop"] is None:
            continue
        per_run[run] = pr
        items = sorted(set(a) & set(b))
        diffs[run] = {i: float(b[i] - a[i]) for i in items}
        cot[run], nott[run] = {i: a[i] for i in items}, {i: b[i] for i in items}
    cross = {}
    for r1, r2 in combinations(sorted(diffs), 2):
        common = sorted(set(diffs[r1]) & set(diffs[r2]))
        if len(common) < MIN_ITEMS:
            continue
        d1, d2 = (np.array([diffs[r][i] for i in common]) for r in (r1, r2))
        c1, c2 = (np.array([cot[r][i] for i in common], float) for r in (r1, r2))
        n1, n2 = (np.array([nott[r][i] for i in common], float) for r in (r1, r2))
        same_cot = sum(texts[runs[r1][0]].get(i) == texts[runs[r2][0]].get(i) for i in common)
        same_not = sum(texts[runs[r1][1]].get(i) == texts[runs[r2][1]].get(i) for i in common)
        cross[f"{r1}_vs_{r2}"] = {
            "n_common": len(common), "drop_r": pearson_boot(d1, d2, draws=draws),
            "same_drop_share": float((d1 == d2).mean()),
            "cot_score_r": _pearson(c1, c2), "not_score_r": _pearson(n1, n2),
            "identical_cot_texts": int(same_cot), "identical_not_texts": int(same_not)}
    pooled = pooled_fixed_run(diffs, draws=draws) if diffs else None
    vc = variance_components(diffs) if len(diffs) >= 2 else None
    stab = None
    if per_run:
        drops = [v["drop"] for v in per_run.values()]
        stab = {"n_runs": len(per_run),
                "all_runs_negative_ci_excl_0": bool(all(v["hi"] < 0 for v in per_run.values())),
                "all_runs_positive_ci_excl_0": bool(all(v["lo"] > 0 for v in per_run.values())),
                "same_sign_all_runs": bool(all(x < 0 for x in drops) or all(x > 0 for x in drops)),
                "sign_flip_across_runs": bool(min(drops) < 0 < max(drops)),
                "range_points": 100 * (max(drops) - min(drops)),
                "min_pairwise_drop_r": (min((c["drop_r"]["r"] for c in cross.values() if c["drop_r"]["r"] is not None),
                                            default=None))}
    return {"per_run": per_run, "cross_run": cross, "pooled": pooled, "pooled_random_run": vc,
            "stability": stab, "missing_scores": missing, "unparsed": unparsed}


# ----------------------------------------------------------------------------- reconciliation
def _load(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def reconcile(out: dict, judges: Sequence[str], generators: Sequence[str]) -> dict:
    """Do the per-run drops reproduce the registered artefacts? run 1: judge_panel_full_oeq_<judge>.json,
    run 2: judge_panel_newarms_<judge>.json (baseline standard_cot_rep), run 3: headline_rep4k_readout.json."""
    rep4k = _load(OUT_DIR / "headline_rep4k_readout.json")
    res = {}
    for run in ("1", "2", "3"):
        n_cells = n_exact = 0
        worst = 0.0
        mism = []
        for j in judges:
            src = None
            if run == "1":
                src = _load(OUT_DIR / f"judge_panel_full_oeq_{j}.json")
            elif run == "2":
                src = _load(OUT_DIR / f"judge_panel_newarms_{j}.json")
            for g in generators:
                mine = out["judges"][j]["per_generator"][g]["per_run"].get(run)
                if mine is None:
                    continue
                ref = None
                if run in ("1", "2") and src:
                    ref = src.get("judges", {}).get(j, {}).get("per_generator", {}).get(g, {}).get(RUNS[run][1])
                elif run == "3" and rep4k:
                    ref = rep4k.get("per_generator", {}).get(g, {}).get("per_judge", {}).get(j)
                if not ref or ref.get("drop") is None:
                    mism.append({"judge": j, "generator": g, "reason": "no reference cell"})
                    continue
                n_cells += 1
                dev = max(abs(mine["drop"] - ref["drop"]), abs(mine["lo"] - ref["lo"]), abs(mine["hi"] - ref["hi"]))
                worst = max(worst, dev)
                if dev < 1e-9 and mine["n_items"] == ref["n_items"]:
                    n_exact += 1
                else:
                    mism.append({"judge": j, "generator": g, "mine": [mine["drop"], mine["lo"], mine["hi"], mine["n_items"]],
                                 "reference": [ref["drop"], ref["lo"], ref["hi"], ref["n_items"]]})
        res[run] = {"reference": {"1": "judge_panel_full_oeq_<judge>.json judges.<judge>.per_generator.<gen>.narrative_cot",
                                  "2": "judge_panel_newarms_<judge>.json judges.<judge>.per_generator.<gen>.narrative_cot_rep",
                                  "3": "headline_rep4k_readout.json per_generator.<gen>.per_judge.<judge>"}[run],
                    "n_cells_compared": n_cells, "n_exact": n_exact, "max_abs_deviation": worst, "mismatches": mism}
    return res


# ----------------------------------------------------------------------------- summaries
def summarise(out: dict, judges: Sequence[str], generators: Sequence[str]) -> dict:
    per_judge = {}
    all_r, all_sigma_run, all_icc = [], [], []
    for j in judges:
        pg = out["judges"][j]["per_generator"]
        rs = [c["drop_r"]["r"] for g in generators for c in pg[g]["cross_run"].values() if c["drop_r"]["r"] is not None]
        sig = [pg[g]["pooled_random_run"]["sigma_run_points"] for g in generators
               if pg[g]["pooled_random_run"] and pg[g]["pooled_random_run"].get("drop") is not None]
        icc = [pg[g]["pooled_random_run"]["icc_item_single_run"] for g in generators
               if pg[g]["pooled_random_run"] and pg[g]["pooled_random_run"].get("icc_item_single_run") is not None]
        stabs = [pg[g]["stability"] for g in generators if pg[g]["stability"]]
        per_judge[j] = {
            "n_generators": len(stabs),
            "n_all_runs_negative_ci_excl_0": sum(s["all_runs_negative_ci_excl_0"] for s in stabs),
            "n_all_runs_positive_ci_excl_0": sum(s["all_runs_positive_ci_excl_0"] for s in stabs),
            "n_same_sign_all_runs": sum(s["same_sign_all_runs"] for s in stabs),
            "n_sign_flip_across_runs": sum(s["sign_flip_across_runs"] for s in stabs),
            "generators_sign_flip": [g for g in generators if pg[g]["stability"] and pg[g]["stability"]["sign_flip_across_runs"]],
            "max_range_points": max((s["range_points"] for s in stabs), default=None),
            "generator_max_range": max(((s["range_points"], g) for g, s in ((g, pg[g]["stability"]) for g in generators) if s),
                                       default=(None, None))[1],
            "median_range_points": float(np.median([s["range_points"] for s in stabs])) if stabs else None,
            "mean_item_drop_r": float(np.mean(rs)) if rs else None,
            "min_item_drop_r": float(min(rs)) if rs else None, "max_item_drop_r": float(max(rs)) if rs else None,
            "mean_sigma_run_points": float(np.mean(sig)) if sig else None,
            "mean_icc_item_single_run": float(np.mean(icc)) if icc else None,
            "pooled_drop_points": {g: (100 * pg[g]["pooled"]["drop"]) if pg[g]["pooled"] else None for g in generators},
            "pooled_ci_points": {g: ([100 * pg[g]["pooled"]["lo"], 100 * pg[g]["pooled"]["hi"]] if pg[g]["pooled"] else None)
                                 for g in generators},
            "n_pooled_negative_ci_excl_0": sum(1 for g in generators if pg[g]["pooled"] and pg[g]["pooled"]["hi"] < 0)}
        all_r += rs
        all_sigma_run += sig
        all_icc += icc
    return {"per_judge": per_judge,
            "overall": {"n_judge_generator_run_pairs": len(all_r),
                        "mean_item_drop_r": float(np.mean(all_r)) if all_r else None,
                        "median_item_drop_r": float(np.median(all_r)) if all_r else None,
                        "min_item_drop_r": float(min(all_r)) if all_r else None,
                        "max_item_drop_r": float(max(all_r)) if all_r else None,
                        "mean_sigma_run_points": float(np.mean(all_sigma_run)) if all_sigma_run else None,
                        "max_sigma_run_points": float(max(all_sigma_run)) if all_sigma_run else None,
                        "mean_icc_item_single_run": float(np.mean(all_icc)) if all_icc else None}}


def print_report(out: dict) -> None:
    for j, jd in out["judges"].items():
        print(f"\n== judge {j}")
        for g, v in jd["per_generator"].items():
            if not v["per_run"]:
                print(f"{g:26s} no run with >= {MIN_ITEMS} paired items")
                continue
            runs = " | ".join(f"r{k}: {100 * x['drop']:+6.1f} [{100 * x['lo']:+6.1f},{100 * x['hi']:+6.1f}] n={x['n_items']}"
                              for k, x in sorted(v["per_run"].items()))
            p, vc, s = v["pooled"], v["pooled_random_run"], v["stability"]
            ps = f"pooled(fixed) {100 * p['drop']:+6.1f} [{100 * p['lo']:+6.1f},{100 * p['hi']:+6.1f}]"
            vs = (f"  random-run {100 * vc['drop']:+6.1f} [{100 * vc['lo']:+6.1f},{100 * vc['hi']:+6.1f}] "
                  f"sigma_run {vc['sigma_run_points']:.1f}pt ICC {vc['icc_item_single_run']:.2f}"
                  if vc and vc.get("drop") is not None else "")
            cs = " ".join(f"r[{k}]={c['drop_r']['r']:+.2f}" if c["drop_r"]["r"] is not None else f"r[{k}]=NA"
                          for k, c in v["cross_run"].items())
            print(f"{g:26s} {runs}\n{'':26s} {ps}{vs}  {cs}  range {s['range_points']:.1f}pt"
                  f"{'  SIGN FLIP' if s['sign_flip_across_runs'] else ''}  missing={v['missing_scores']} unparsed={v['unparsed']}")
    print("\n== reconciliation with the registered artefacts")
    for run, r in out["reconciliation"].items():
        print(f"  run {run}: {r['n_exact']}/{r['n_cells_compared']} cells exact (max |dev| {r['max_abs_deviation']:.2e}); "
              f"{len(r['mismatches'])} mismatch(es)" + (f": {r['mismatches'][:3]}" if r["mismatches"] else ""))
    print("\n== summary")
    for j, s in out["summary"]["per_judge"].items():
        print(f"  {j:26s} all-runs-negative {s['n_all_runs_negative_ci_excl_0']}/{s['n_generators']}  "
              f"same-sign {s['n_same_sign_all_runs']}/{s['n_generators']}  sign-flips {s['n_sign_flip_across_runs']} "
              f"{s['generators_sign_flip']}  max range {s['max_range_points']:.1f}pt ({s['generator_max_range']})  "
              f"mean item-drop r {s['mean_item_drop_r']:.2f} [{s['min_item_drop_r']:.2f}, {s['max_item_drop_r']:.2f}]  "
              f"mean sigma_run {s['mean_sigma_run_points']:.1f}pt  pooled negative {s['n_pooled_negative_ci_excl_0']}/{s['n_generators']}")
    o = out["summary"]["overall"]
    print(f"  overall: item-level drop r mean {o['mean_item_drop_r']:.3f} median {o['median_item_drop_r']:.3f} "
          f"[{o['min_item_drop_r']:.2f}, {o['max_item_drop_r']:.2f}] over {o['n_judge_generator_run_pairs']} judge x generator x run-pair cells; "
          f"sigma_run mean {o['mean_sigma_run_points']:.1f}pt max {o['max_sigma_run_points']:.1f}pt; mean ICC {o['mean_icc_item_single_run']:.2f}")


# ----------------------------------------------------------------------------- selftest
def _selftest() -> int:
    ok = True

    def check(msg, cond):
        nonlocal ok
        print(("[ok] " if cond else "[FAIL] ") + msg)
        ok = ok and cond

    # 1. source hygiene: no scripts.* import, no network module (strings assembled so this file does not contain them)
    src = Path(__file__).read_text()
    banned = ["scripts" + ".generators", "from scripts" + " import", "import scripts" + ".", "import " + "requests",
              "urllib" + ".request", "http" + ".client", "import " + "socket", "import " + "openai", "import " + "anthropic",
              "import " + "httpx", "import " + "aiohttp"]
    check("source imports nothing from scripts/ and no network module", not any(b in src for b in banned))
    # 2. paired drop
    a = {str(i): 1 for i in range(40)}
    b = {str(i): (0 if i < 20 else 1) for i in range(40)}
    d = paired_drop(a, b, draws=2000)
    check("paired drop 1.0 -> 0.5 is -0.5 with CI excluding 0", abs(d["drop"] + 0.5) < 1e-12 and d["hi"] < 0)
    check("paired drop needs >= 20 shared items", paired_drop({"1": 1}, {"1": 0})["drop"] is None)
    # 3. correlations
    rng = np.random.default_rng(0)
    x = rng.integers(-1, 2, 300).astype(float)
    check("Pearson r of a series with itself is 1", abs(pearson_boot(x, x, draws=500)["r"] - 1) < 1e-12)
    check("Pearson r with its negative is -1", abs(pearson_boot(x, -x, draws=500)["r"] + 1) < 1e-12)
    y = rng.integers(-1, 2, 300).astype(float)
    pb = pearson_boot(x, y, draws=2000)
    check("independent series: |r| < 0.15 and the bootstrap interval covers 0", abs(pb["r"]) < 0.15 and pb["lo"] < 0 < pb["hi"])
    check("constant series gives r None", pearson_boot(np.zeros(30), x[:30], draws=100)["r"] is None)
    # 4. pooled with run fixed: equals the mean of the run drops, balanced or not
    diffs = {"1": {str(i): -1.0 if i < 30 else 0.0 for i in range(100)},
             "2": {str(i): -1.0 if i < 50 else 0.0 for i in range(80)}}
    p = pooled_fixed_run(diffs, draws=1000)
    check("pooled (fixed run) = mean of per-run drops (-0.30, -0.625 -> -0.4625)", abs(p["drop"] + 0.4625) < 1e-12)
    check("pooled interval brackets the point", p["lo"] <= p["drop"] <= p["hi"])
    # 5. variance components: d_ir = a_i + b_r with no noise recovers var(b) and var(a) exactly
    I, R = 60, 3
    a_i = rng.normal(0, 0.4, I)
    b_r = np.array([-0.30, -0.10, -0.50])
    diffs = {str(r): {str(i): float(a_i[i] + b_r[r]) for i in range(I)} for r in range(R)}
    vc = variance_components(diffs)
    check("variance components: sigma_run^2 = sample var of run effects (0.04) when noise is 0",
          abs(vc["sigma2_run_unclipped"] - np.var(b_r, ddof=1)) < 1e-9 and vc["sigma_resid"] < 1e-9)
    check("variance components: sigma_item^2 = sample var of item effects when noise is 0",
          abs(vc["sigma2_item_unclipped"] - np.var(a_i, ddof=1)) < 1e-9 and abs(vc["icc_item_single_run"] - 1) < 1e-9)
    check("variance components: grand mean is the mean of run means",
          abs(vc["drop"] - (a_i.mean() + b_r.mean())) < 1e-12 and abs(vc["sigma_run_points"] - 20.0) < 1e-6)
    # 6. a run effect must widen the random-run interval relative to the fixed-run bootstrap
    noise = rng.normal(0, 0.2, (I, R))
    diffs2 = {str(r): {str(i): float(a_i[i] + b_r[r] + noise[i, r]) for i in range(I)} for r in range(R)}
    vc2, p2 = variance_components(diffs2), pooled_fixed_run(diffs2, draws=1000)
    check("random-run interval is wider than the fixed-run bootstrap when runs differ",
          (vc2["hi"] - vc2["lo"]) > (p2["hi"] - p2["lo"]))
    # 7. on-disk checks (cache only), skipped if the files are not present
    if OEQ_CSV.exists() and RAW.exists():
        q = load_questions()
        check("the seed-44 OEQ sample has 150 items", len(q) == N_ITEMS)
        rows = load_rows(GENERATORS, [a for pair in RUNS.values() for a in pair], [RAW])
        check("every original-cell row's item is in the sample", all(r["item_id"] in q for r in rows) and len(rows) > 1900)
        r0 = next(r for r in rows if r["generator"] == PRODUCTION_JUDGE and r["arm"] == "standard_cot")
        k = cache_key(q[r0["item_id"]], r0["response"], PRODUCTION_JUDGE)
        check(f"the production judge's full-text cache exists for a haiku CoT row ({k.name})",
              k.exists() and cached_score(q[r0["item_id"]], r0["response"], PRODUCTION_JUDGE) in (0, 1))
        check("run 1 haiku/haiku-judge drop reproduces the 17.8 panel (-0.2667, n 150)",
              abs(analyse(rows, q, PRODUCTION_JUDGE, PRODUCTION_JUDGE, {"1": RUNS["1"]}, draws=200)["per_run"]["1"]["drop"]
                  + 0.26666666666666666) < 1e-12)
    else:
        print("[skip] on-disk checks (OEQ.csv or the results CSV not present)")
    print("\nselftest: " + ("ALL OK" if ok else "FAILED"))
    return 0 if ok else 1


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--judges", default=",".join(JUDGES))
    ap.add_argument("--generators", default=",".join(GENERATORS))
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--scan", action="store_true", help="discover further elephant_singleagent_raw_batch_*.csv (slow)")
    ap.add_argument("--batch-csv", action="append", default=[], help="an extra batch CSV name to read")
    ap.add_argument("--include-extra", action="store_true",
                    help="analyse any standard_cot_rep4k_<sfx>/narrative_cot_rep4k_<sfx> pair found as a further run")
    ap.add_argument("--json", type=Path, default=OUT)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    judges = [x for x in a.judges.split(",") if x]
    gens = [x for x in a.generators.split(",") if x]
    paths = [RAW] + batch_csvs(a.scan, a.batch_csv)
    rows = load_rows(gens, [x for pair in RUNS.values() for x in pair], paths)
    questions = load_questions()
    missing_q = {r["item_id"] for r in rows} - set(questions)
    if missing_q:
        raise SystemExit(f"{len(missing_q)} item(s) outside the seed-44 sample; refusing")
    runs = dict(RUNS)
    extra = discover_extra_runs(rows)
    if a.include_extra:
        runs.update(extra)
    n_rows = {}
    for r in rows:
        n_rows.setdefault(r["generator"], {}).setdefault(r["arm"], 0)
        n_rows[r["generator"]][r["arm"]] += 1
    out = {"note": "NOT PRE-REGISTERED: added 2026-09-23 after every run it reads was generated and scored, in "
                   "answer to a reviewer question on run-to-run stability. Zero model calls, zero network calls: "
                   "every score is read from the elephant_rescore_validation_<judge>_full_* judge cache. A run-to-run "
                   "difference bundles sampling noise with a change of generation caps (and, on Llama, Mistral and "
                   "DeepSeek, of month); the design cannot separate them.",
           "written": str(date.today()), "metric": METRIC, "dataset": "oeq", "sample": {"seed": ELEPHANT_SEED, "n": N_ITEMS},
           "sources": {"results_csvs": [p.name for p in paths if p.exists()],
                       "judge_cache": "elephant_rescore_validation_<judge>_full_<sha1[:16]>.json",
                       "questions": str(OEQ_CSV.relative_to(ROOT))},
           "runs": {k: {"label": RUN_LABELS.get(k, "extra replicate found on disk"), "arms": list(v)} for k, v in runs.items()},
           "run1_cohort": {g: RUN1_COHORT.get(g) for g in gens},
           "extra_replicate_arms_found": {k: list(v) for k, v in extra.items()},
           "extra_runs_included": bool(a.include_extra and extra),
           "n_rows": n_rows,
           "bootstrap": {"draws": a.draws, "seed": SEED, "min_items_per_run": MIN_ITEMS,
                         "per_run_ci": "item-clustered percentile bootstrap of the paired difference",
                         "pooled_ci": "items resampled with every run of a resampled item kept (run fixed, crossed)",
                         "pooled_random_run_ci": "normal, SE from two-way method-of-moments variance components "
                                                 "(item, run, residual) on the balanced subset; run component clipped at 0"},
           "judges": {}}
    for j in judges:
        print(f"scoring tables for judge {j} ...", flush=True)
        out["judges"][j] = {"per_generator": {g: analyse(rows, questions, j, g, runs, draws=a.draws) for g in gens}}
    out["reconciliation"] = reconcile(out, judges, gens)
    out["summary"] = summarise(out, judges, gens)
    a.json.write_text(json.dumps(out, indent=1, default=float))
    print_report(out)
    print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
