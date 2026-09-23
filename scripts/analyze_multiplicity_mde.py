"""
scripts/analyze_multiplicity_mde.py -- multiplicity and minimum detectable effects for the
ICLR 2027 draft. Zero spend; artefact-only; no network; does not import scripts.generators
(or any scripts.* module).

WHAT IT DOES

1. Holm step-down within judge over the seven generators, applied to the per-cell
   NoT minus CoT contrasts behind
     Table 2  judge_panel_full_oeq_<judge>.json     (17.8, cached cells, five judges)
     Table 5  headline_rep4k_readout.json           (17.10, same-day replicate, 4,096 cap)
     Table 6  narrative_form_readout.json           (17.7 + amendment 2 + 17.4; four generators,
                                                     so that family has four members, not seven;
                                                     the other conditions' drops and gaps are
                                                     reported as separate families, labelled)
   and lists, per judge, which cells survive at family alpha 0.05 and which negative or
   positive cells fail.

2. Two-sided p-values, two ways.
   implied  -- from the artefact's 95% percentile interval alone, by normal inversion:
               se = (hi - lo) / (2 * 1.959964), z = point / se, p = erfc(|z| / sqrt 2).
               Available for every cell; approximate where the bootstrap distribution is
               skewed (rates near 0 or 1).
   exact    -- recomputed from the cached per-item judge scores when they are on disk:
               the same item-clustered paired bootstrap the producing scripts ran
               (rescore_elephant_full_judge.paired_drop: 8,000 draws, seed 20260822;
               analyze_narrative_form.contrast: 8,000 draws, seed 20260922), so the
               recomputed interval reproduces the artefact's to floating precision, which is
               the proof that the same data were read; p = 2 * min(P(B <= 0), P(B >= 0))
               over the bootstrap draws, floored at 2 / draws. A cell whose recomputed
               interval does not match its artefact falls back to the implied p and is
               flagged. Holm is run on both p-value sets; the artefact reports both.

3. Interval half-widths and 80%-power minimum detectable effects at every registered null
   reading: pushback net of instability (17.9, pooled and per model), the haiku within-one-
   loser counter lift (16.23), the four knockout shares against the registered 0.25 threshold
   (17.3), the persona-only drop under each judge (17.7), the Llama routing gain (16.28), and,
   labelled as companions, the grok routed and both models' shielding NO-DIFFERENCE readings.
   half_width = (hi - lo) / 2; se_implied = half_width / 1.959964;
   mde_80 = (1.959964 + 0.841621) * se_implied = 2.801585 * se_implied, the true effect that a
   two-sided 0.05 test at this precision would detect with 80% power.

INPUTS (read only; every path under divergence_study_outputs/ except the one ELEPHANT CSV)
   divergence_study_outputs/judge_panel_full_oeq_<judge>.json  (five judges)
   divergence_study_outputs/headline_rep4k_readout.json
   divergence_study_outputs/narrative_form_readout.json
   divergence_study_outputs/pushback_readout_pooled.json
   divergence_study_outputs/router_decomposition_haiku.json
   divergence_study_outputs/knockout_social_analysis.json
   divergence_study_outputs/seat_scaffold_comparison.json
   for the exact recompute only:
   divergence_study_outputs/elephant_singleagent_raw.csv and the six
     elephant_singleagent_raw_batch_*.csv (fixed list; later names win, as load_rows does)
   data/elephant/OEQ.csv (the ELEPHANT CSV already on disk; never downloaded here)
   divergence_study_outputs/elephant_rescore_validation_<judge>_full_<sha1[:16]>.json
     (one small file per response x judge, opened by exact path; no directory scan)

OUTPUT
   divergence_study_outputs/multiplicity_mde.json

Run:
  python scripts/analyze_multiplicity_mde.py --selftest
  python scripts/analyze_multiplicity_mde.py            # exact recompute when caches are present
  python scripts/analyze_multiplicity_mde.py --no-exact  # implied p-values only
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "divergence_study_outputs"
OEQ_CSV = ROOT / "data" / "elephant" / "OEQ.csv"
OUT = OUT_DIR / "multiplicity_mde.json"

JUDGES = ("claude-haiku-4-5", "gpt-4o", "gpt-5.4-nano", "Llama-3.3-70B-Instruct",
          "grok-4-1-fast-reasoning")
PRODUCTION_JUDGE = "claude-haiku-4-5"
GENERATORS = ("claude-haiku-4-5", "claude-sonnet-4-6", "gpt-5.4-nano", "grok-4-1-fast-reasoning",
              "Llama-3.3-70B-Instruct", "Mistral-Large-3-2", "DeepSeek-V4-Pro")
FORM_GENS = ("claude-haiku-4-5", "grok-4-1-fast-reasoning", "gpt-5.4-nano", "Llama-3.3-70B-Instruct")
LEN_GENS = ("claude-haiku-4-5", "grok-4-1-fast-reasoning")
ALPHA = 0.05
Z975 = 1.959963984540054
Z80 = 0.8416212335729143
MDE_MULT = Z975 + Z80          # 2.801585...

# the producing scripts' bootstrap settings (reproduced exactly for the exact p-values)
PANEL_DRAWS, PANEL_SEED = 8000, 20260822     # rescore_elephant_full_judge.paired_drop (Tables 2, 5)
FORM_DRAWS, FORM_SEED = 8000, 20260922       # analyze_narrative_form.contrast (Table 6)
ELEPHANT_SEED, ELEPHANT_N = 44, 150          # scripts/load_elephant.py: seed-44 sample of 150

# the raw ELEPHANT rows: the shared CSV plus the per-batch CSVs written with run_elephant --out-csv.
# load_rows globs "elephant_singleagent_raw_batch_*.csv"; the directory holds ~800k entries and a scan
# takes about a minute, so the six batch files present on 2026-09-23 are listed by name (sorted, later
# wins, exactly as the glob would order them). --scan-batches globs instead.
RAW_CSV = OUT_DIR / "elephant_singleagent_raw.csv"
BATCH_CSVS = ("elephant_singleagent_raw_batch_1710_haiku.csv",
              "elephant_singleagent_raw_batch_1710_mistral_deepseek.csv",
              "elephant_singleagent_raw_batch_1710_nano.csv",
              "elephant_singleagent_raw_batch_1710_sonnet.csv",
              "elephant_singleagent_raw_batch_177_1710_grok.csv",
              "elephant_singleagent_raw_batch_177_1710_llama.csv")

COT, NOT = "standard_cot", "narrative_cot"
COT4K, NOT4K = "standard_cot_rep4k", "narrative_cot_rep4k"
COTR, NOTR = "standard_cot_rep", "narrative_cot_rep"
CHECK, NARR, LEN, PERSONA = "not_checklist", "not_narrative_only", "standard_cot_lengthmatched", "advisor_cot"

INPUTS_READ: list[str] = []


# ----------------------------------------------------------------------------- statistics

def p_implied(point: float, lo: float, hi: float) -> Optional[float]:
    """Two-sided p implied by a 95% percentile interval under a normal bootstrap distribution."""
    if point is None or lo is None or hi is None:
        return None
    se = (hi - lo) / (2.0 * Z975)
    if se <= 0:
        return 0.0 if point != 0 else 1.0
    z = abs(point) / se
    return float(math.erfc(z / math.sqrt(2.0)))


def p_exact(boots: np.ndarray) -> float:
    """Two-sided percentile-bootstrap p: 2 * min(P(B <= 0), P(B >= 0)), floored at 2 / draws."""
    b = np.asarray(boots, dtype=float)
    b = b[~np.isnan(b)]
    if b.size == 0:
        return float("nan")
    p = 2.0 * min(float(np.mean(b <= 0)), float(np.mean(b >= 0)))
    return float(max(min(p, 1.0), 2.0 / b.size))


def holm(pvals: dict, alpha: float = ALPHA) -> dict:
    """Holm step-down. Returns name -> {p, rank, adj_p, reject}. None p-values are excluded."""
    items = [(k, v) for k, v in pvals.items() if v is not None and not (isinstance(v, float) and math.isnan(v))]
    m = len(items)
    out = {}
    running = 0.0
    for rank, (k, p) in enumerate(sorted(items, key=lambda kv: kv[1]), start=1):
        adj = min(1.0, (m - rank + 1) * p)
        running = max(running, adj)
        out[k] = {"p": float(p), "rank": rank, "adj_p": float(running), "reject": bool(running < alpha)}
    for k, v in pvals.items():
        if k not in out:
            out[k] = {"p": None, "rank": None, "adj_p": None, "reject": None}
    return out


def mde_block(point, lo, hi, *, null_value: float = 0.0, scale: float = 1.0, label: str = "") -> dict:
    """Half-width, implied se, 80%-power MDE, and where the interval sits against the null."""
    if point is None or lo is None or hi is None:
        return {"label": label, "point": None, "lo": None, "hi": None, "available": False}
    hw = (hi - lo) / 2.0
    se = hw / Z975
    return {
        "label": label,
        "point": point * scale, "lo": lo * scale, "hi": hi * scale,
        "null_value": null_value * scale,
        "half_width": hw * scale,
        "se_implied": se * scale,
        "mde_80": MDE_MULT * se * scale,
        "p_implied_vs_null": p_implied(point - null_value, lo - null_value, hi - null_value),
        "ci_excludes_null": bool(lo > null_value or hi < null_value),
        "distance_to_null_in_half_widths": (abs(point - null_value) / hw) if hw > 0 else None,
        "scale_note": "points (x100)" if scale == 100 else "proportion",
    }


# ----------------------------------------------------------------------------- artefact readers

def _load(path: Path):
    INPUTS_READ.append(str(path.relative_to(ROOT)))
    with path.open() as fh:
        return json.load(fh)


def cells_table2(judge: str) -> dict:
    d = _load(OUT_DIR / f"judge_panel_full_oeq_{judge}.json")
    per = d["judges"][judge]["per_generator"]
    out = {}
    for g in GENERATORS:
        c = per[g][NOT]
        out[g] = {"point": c["drop"], "lo": c["lo"], "hi": c["hi"], "n_items": c["n_items"],
                  "rate_cot": c["rate_a"], "rate_not": c["rate_b"]}
    return out


def cells_table5(judge: str, d: dict) -> dict:
    out = {}
    for g in GENERATORS:
        c = d["per_generator"][g]["per_judge"][judge]
        out[g] = {"point": c["drop"], "lo": c["lo"], "hi": c["hi"], "n_items": c["n_items"],
                  "rate_cot": c["rate_a"], "rate_not": c["rate_b"], "relative": c.get("relative")}
    return out


def families_table6(judge: str, d: dict) -> dict:
    """Per-generator contrasts behind Table 6, one family per (condition, quantity)."""
    j = d["judges"][judge]
    fams = {}

    def fam(name, gens, getter, note):
        cells = {}
        for g in gens:
            c = getter(g)
            if c is None or c.get("point") is None:
                continue
            cells[g] = {"point": c["point"], "lo": c["lo"], "hi": c["hi"]}
        fams[name] = {"cells": cells, "note": note}

    fpg, ppg, lpg = j["form_per_generator"], j["persona_per_generator"], j["length_per_generator"]
    fam("not_rep_drop", FORM_GENS, lambda g: fpg[g]["checklist"]["drop_not"],
        "NoT minus CoT, same-day replicates at 2,048 (items present in the CoT, checklist and NoT cells); "
        "the NoT minus CoT contrast behind Table 6")
    fam("checklist_drop", FORM_GENS, lambda g: fpg[g]["checklist"]["drop_arm"], "checklist minus CoT")
    fam("checklist_gap", FORM_GENS, lambda g: fpg[g]["checklist"]["gap"], "NoT minus checklist (positive: checklist validates less)")
    fam("narrative_only_drop", FORM_GENS, lambda g: fpg[g]["narrative_only"]["drop_arm"], "narrative-only minus CoT")
    fam("narrative_only_gap", FORM_GENS, lambda g: fpg[g]["narrative_only"]["gap"], "NoT minus narrative-only")
    fam("persona_drop", FORM_GENS, lambda g: ppg[g]["drop_arm"], "persona-only (advisor CoT) minus CoT")
    fam("lengthmatched_drop", LEN_GENS, lambda g: lpg[g]["drop_arm"], "length-matched CoT minus CoT (haiku, grok only)")
    fam("lengthmatched_gap", LEN_GENS, lambda g: lpg[g]["gap"], "NoT minus length-matched CoT")
    return fams


# ----------------------------------------------------------------------------- exact recompute

def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def cache_path(question: str, advice: str, judge: str) -> Path:
    """rescore_elephant_untruncated._key for metric 'validation' at the full (uncut) limit."""
    h = hashlib.sha1(f"validation|{question}|{advice}|{judge}|lim=full".encode()).hexdigest()[:16]
    return OUT_DIR / f"elephant_rescore_validation_{_safe(judge)}_full_{h}.json"


def load_questions() -> dict:
    """item_id -> prompt for the seed-44 OEQ sample of 150, reproducing scripts/load_elephant.py
    (pandas read, shuffle of the row indices with random.Random(44), first 150, id from the first
    'Unnamed: 0' column, prompt stripped) with the csv module and no pandas."""
    if not OEQ_CSV.exists():
        return {}
    INPUTS_READ.append(str(OEQ_CSV.relative_to(ROOT)))
    with OEQ_CSV.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)
    # pandas names an empty header "Unnamed: 0"; a literal "Unnamed: 0" header that follows becomes
    # "Unnamed: 0.1", so the id column is the first one named or renamed "Unnamed: 0".
    cols = []
    seen = defaultdict(int)
    for i, h in enumerate(header):
        name = h if h != "" else f"Unnamed: {i}"
        if seen[name]:
            name = f"{name}.{seen[name]}"
        seen[h if h != "" else f"Unnamed: {i}"] += 1
        cols.append(name)
    try:
        id_col = cols.index("id")
    except ValueError:
        id_col = cols.index("Unnamed: 0")
    prompt_col = cols.index("prompt")
    idxs = list(range(len(rows)))
    random.Random(ELEPHANT_SEED).shuffle(idxs)
    out = {}
    for i, k in enumerate(idxs[:ELEPHANT_N]):
        r = rows[k]
        rid = r[id_col].strip() if r[id_col].strip() != "" else str(i)
        out[rid] = r[prompt_col].strip()
    return out


def load_rows(arms: Sequence[str], generators: Sequence[str], *, scan: bool = False) -> list[dict]:
    """rescore_elephant_full_judge.load_rows: one row per (generator, arm, item), later files win,
    dataset oeq, regime default, non-empty response."""
    if scan:
        paths = [RAW_CSV] + sorted(OUT_DIR.glob("elephant_singleagent_raw_batch_*.csv"))
    else:
        paths = [RAW_CSV] + [OUT_DIR / n for n in sorted(BATCH_CSVS)]
    keep = {}
    csv.field_size_limit(1 << 30)
    for path in paths:
        if not path.exists():
            continue
        INPUTS_READ.append(str(path.relative_to(ROOT)))
        with path.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if (r.get("dataset") == "oeq" and r.get("arm") in arms and r.get("generator") in generators
                        and (r.get("regime") or "default") == "default"):
                    keep[(r["generator"], r["arm"], r["item_id"])] = r
    return [r for r in keep.values() if (r.get("response") or "").strip()]


def score_cells(rows: list[dict], judge: str, questions: dict) -> tuple[dict, dict]:
    """(gen, arm) -> {item: 0/1} from the cached full-text judge scores; plus hit statistics."""
    cells = defaultdict(dict)
    n, hit, unparsed, noq = 0, 0, 0, 0
    for r in rows:
        q = questions.get(r["item_id"])
        if q is None:
            noq += 1
            continue
        n += 1
        p = cache_path(q, r["response"], judge)
        if not p.exists():
            continue
        try:
            s = int(json.loads(p.read_text()).get("score", -1))
        except Exception:
            continue
        hit += 1
        if s in (0, 1):
            cells[(r["generator"], r["arm"])][r["item_id"]] = s
        else:
            unparsed += 1
    return cells, {"rows": n, "cache_hits": hit, "hit_rate": (hit / n if n else 0.0), "unparsed": unparsed,
                   "rows_without_question": noq}


def paired_boot(a: dict, b: dict, *, draws: int, seed: int) -> Optional[dict]:
    """rate(b) - rate(a) on items scored in both; item bootstrap; returns point, lo, hi, draws."""
    items = sorted(set(a) & set(b))
    if len(items) < 20:
        return None
    d = np.array([b[i] - a[i] for i in items], dtype=float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(draws, len(d)))].mean(1)
    return {"n_items": len(items), "point": float(d.mean()),
            "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
            "boots": boots}


def form_contrast_boot(cells: dict, g: str, base: str, arm: str, not_arm: str, *, draws: int, seed: int) -> Optional[dict]:
    """analyze_narrative_form.contrast for one generator: returns per-quantity point/lo/hi and draws."""
    cb, ca, cn = cells.get((g, base), {}), cells.get((g, arm), {}), cells.get((g, not_arm), {})
    items = sorted(set(cb) & set(ca) & set(cn))
    if len(items) < 20:
        return None
    arr = [np.array([(cb[i], ca[i], cn[i])], dtype=float) for i in items]
    sums = np.array([a.sum(0) for a in arr])
    cnts = np.array([len(a) for a in arr], dtype=float)

    def stats(s, c):
        m = s / c
        drop_arm, drop_not = m[1] - m[0], m[2] - m[0]
        return drop_arm, drop_not, (drop_arm / drop_not if drop_not != 0 else np.nan), m[2] - m[1]
    pt = stats(sums.sum(0), cnts.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(items), size=(draws, len(items)))
    bs = np.array([stats(sums[k].sum(0), cnts[k].sum()) for k in idx])
    names = ("drop_arm", "drop_not", "share", "gap")
    out = {"n_items": len(items)}
    for k, n in enumerate(names):
        out[n] = {"point": float(pt[k]), "lo": float(np.nanpercentile(bs[:, k], 2.5)),
                  "hi": float(np.nanpercentile(bs[:, k], 97.5)), "boots": bs[:, k]}
    return out


CI_TOL = 1e-9


def _match(rec: Optional[dict], art: dict) -> bool:
    return (rec is not None and art.get("point") is not None
            and abs(rec["point"] - art["point"]) < CI_TOL
            and abs(rec["lo"] - art["lo"]) < CI_TOL and abs(rec["hi"] - art["hi"]) < CI_TOL)


# ----------------------------------------------------------------------------- family analysis

def analyse_family(cells: dict, exact: Optional[dict] = None) -> dict:
    """cells: gen -> {point, lo, hi, ...}; exact: gen -> {p_exact, ci_match} or None."""
    pi = {g: p_implied(c["point"], c["lo"], c["hi"]) for g, c in cells.items()}
    hi_ = holm(pi)
    have_exact = exact is not None and all(g in exact and exact[g].get("ci_match") for g in cells)
    pe = {g: (exact[g]["p_exact"] if have_exact else None) for g in cells}
    he = holm(pe) if have_exact else None
    out_cells = {}
    for g, c in cells.items():
        e = (exact or {}).get(g, {})
        rec = {**{k: v for k, v in c.items()},
               "sign": ("negative" if c["point"] < 0 else "positive" if c["point"] > 0 else "zero"),
               "ci_excludes_0": bool(c["lo"] > 0 or c["hi"] < 0),
               "p_implied": pi[g], "holm_adj_p_implied": hi_[g]["adj_p"], "survives_holm_implied": hi_[g]["reject"],
               "p_exact": e.get("p_exact"), "exact_ci_match": e.get("ci_match"),
               "holm_adj_p_exact": (he[g]["adj_p"] if he else None),
               "survives_holm_exact": (he[g]["reject"] if he else None)}
        if e.get("recomputed") is not None:
            rec["recomputed"] = e["recomputed"]
        out_cells[g] = rec
    primary = "exact" if have_exact else "implied"
    surv_key = f"survives_holm_{primary}"
    survivors = [g for g, r in out_cells.items() if r[surv_key]]
    fail_neg = [g for g, r in out_cells.items() if not r[surv_key] and r["sign"] == "negative"]
    fail_pos = [g for g, r in out_cells.items() if not r[surv_key] and r["sign"] != "negative"]
    return {
        "m": len(cells), "alpha": ALPHA, "primary_p": primary,
        "cells": out_cells,
        "n_raw_ci_excludes_0": sum(1 for r in out_cells.values() if r["ci_excludes_0"]),
        "n_raw_negative_ci_excludes_0": sum(1 for r in out_cells.values() if r["ci_excludes_0"] and r["sign"] == "negative"),
        "n_survive_holm": len(survivors),
        "n_survive_holm_negative": sum(1 for g in survivors if out_cells[g]["sign"] == "negative"),
        "survivors": survivors, "fail_negative": fail_neg, "fail_positive": fail_pos,
        "survivors_implied": [g for g, r in out_cells.items() if r["survives_holm_implied"]],
        "survivors_exact": ([g for g, r in out_cells.items() if r["survives_holm_exact"]] if have_exact else None),
        "holm_disagreement_implied_vs_exact": (
            sorted(g for g, r in out_cells.items() if r["survives_holm_implied"] != r["survives_holm_exact"])
            if have_exact else None),
    }


# ----------------------------------------------------------------------------- main run

def run(*, exact: bool = True, scan: bool = False) -> dict:
    res = {"method": {
        "holm": "step-down within judge over the generators in the family; adj_p(i) = max_{j<=i} (m-j+1) p_(j), capped at 1; "
                "survives iff adj_p < 0.05",
        "p_implied": "normal inversion of the 95% percentile interval: se = (hi-lo)/(2*1.959964), p = erfc(|point/se|/sqrt 2)",
        "p_exact": "item-clustered paired bootstrap recomputed from the cached per-item judge scores with the producing "
                   "script's draws and seed (Tables 2, 5: 8,000 / 20260822; Table 6: 8,000 / 20260922); "
                   "p = 2*min(P(B<=0), P(B>=0)), floored at 2/draws; used only where the recomputed interval "
                   "reproduces the artefact's to 1e-9",
        "mde": "half_width = (hi-lo)/2; se_implied = half_width/1.959964; mde_80 = 2.801585*se_implied "
               "(the true effect a two-sided 0.05 test at this precision detects with 80% power)",
        "units": "Tables 2, 5, 6 and the persona / pushback / seat-gain entries are in points (x100); "
                 "the one-loser lift and knockout shares are in their artefacts' units (proportion of debates; share of the drop)",
    }, "tables": {}, "null_readings": {}, "exact_recompute": {"requested": exact}}

    rep4k = _load(OUT_DIR / "headline_rep4k_readout.json")
    form = _load(OUT_DIR / "narrative_form_readout.json")

    # ---- exact recompute set-up
    questions, cache_stats = {}, {}
    ex2, ex5, ex6 = {}, {}, {}
    if exact:
        questions = load_questions()
        if not questions:
            res["exact_recompute"]["available"] = False
            res["exact_recompute"]["reason"] = f"{OEQ_CSV} not on disk; nothing downloaded"
        else:
            arms = (COT, NOT, COT4K, NOT4K, COTR, NOTR, CHECK, NARR, LEN, PERSONA)
            rows = load_rows(arms, GENERATORS, scan=scan)
            for judge in JUDGES:
                cells, stats = score_cells(rows, judge, questions)
                cache_stats[judge] = stats
                # Table 2
                ex2[judge] = {}
                for g in GENERATORS:
                    rec = paired_boot(cells.get((g, COT), {}), cells.get((g, NOT), {}), draws=PANEL_DRAWS, seed=PANEL_SEED)
                    ex2[judge][g] = rec
                # Table 5
                ex5[judge] = {}
                for g in GENERATORS:
                    rec = paired_boot(cells.get((g, COT4K), {}), cells.get((g, NOT4K), {}), draws=PANEL_DRAWS, seed=PANEL_SEED)
                    ex5[judge][g] = rec
                # Table 6: per-generator contrasts (checklist, narrative_only, persona, lengthmatched)
                ex6[judge] = {}
                for name, arm, gens in (("checklist", CHECK, FORM_GENS), ("narrative_only", NARR, FORM_GENS),
                                        ("persona", PERSONA, FORM_GENS), ("lengthmatched", LEN, LEN_GENS)):
                    ex6[judge][name] = {g: form_contrast_boot(cells, g, COTR, arm, NOTR, draws=FORM_DRAWS, seed=FORM_SEED)
                                        for g in gens}
            res["exact_recompute"]["available"] = True
            res["exact_recompute"]["cache_stats_per_judge"] = cache_stats
            res["exact_recompute"]["n_rows_loaded"] = len(rows)
            res["exact_recompute"]["batch_csvs"] = list(sorted(BATCH_CSVS)) if not scan else "globbed"
    else:
        res["exact_recompute"]["available"] = False

    def exact_for(recs: dict, arts: dict) -> Optional[dict]:
        if not recs:
            return None
        out = {}
        for g, art in arts.items():
            rec = recs.get(g)
            ok = _match(rec, art)
            out[g] = {"p_exact": (p_exact(rec["boots"]) if ok else None), "ci_match": ok,
                      "recomputed": ({"point": rec["point"], "lo": rec["lo"], "hi": rec["hi"], "n_items": rec["n_items"]}
                                     if rec else None)}
        return out

    # ---- Table 2
    t2 = {"source": "judge_panel_full_oeq_<judge>.json (17.8)", "family": "seven generators within judge", "per_judge": {}}
    for judge in JUDGES:
        cells = cells_table2(judge)
        t2["per_judge"][judge] = analyse_family(cells, exact_for(ex2.get(judge), cells))
    res["tables"]["table2_judge_panel"] = t2

    # ---- Table 5
    t5 = {"source": "headline_rep4k_readout.json (17.10)", "family": "seven generators within judge", "per_judge": {}}
    for judge in JUDGES:
        cells = cells_table5(judge, rep4k)
        t5["per_judge"][judge] = analyse_family(cells, exact_for(ex5.get(judge), cells))
    res["tables"]["table5_rep4k"] = t5

    # ---- Table 6
    t6 = {"source": "narrative_form_readout.json (17.7, amendment 2, 17.4)",
          "family": "the four 17.7 generators (two for the length control) within judge, one family per condition and quantity; "
                    "not_rep_drop is the NoT minus CoT contrast", "per_judge": {}}
    qmap = {"not_rep_drop": ("checklist", "drop_not"), "checklist_drop": ("checklist", "drop_arm"),
            "checklist_gap": ("checklist", "gap"), "narrative_only_drop": ("narrative_only", "drop_arm"),
            "narrative_only_gap": ("narrative_only", "gap"), "persona_drop": ("persona", "drop_arm"),
            "lengthmatched_drop": ("lengthmatched", "drop_arm"), "lengthmatched_gap": ("lengthmatched", "gap")}
    for judge in JUDGES:
        fams = families_table6(judge, form)
        per = {}
        for fname, fam in fams.items():
            cond, qty = qmap[fname]
            recs = None
            if ex6.get(judge):
                recs = {}
                for g in fam["cells"]:
                    r = ex6[judge][cond].get(g)
                    recs[g] = ({"point": r[qty]["point"], "lo": r[qty]["lo"], "hi": r[qty]["hi"],
                                "n_items": r["n_items"], "boots": r[qty]["boots"]} if r else None)
            per[fname] = {"note": fam["note"], **analyse_family(fam["cells"], exact_for(recs, fam["cells"]))}
        t6["per_judge"][judge] = per
    res["tables"]["table6_form"] = t6

    # ---- summary counts
    summary = {}
    for tname in ("table2_judge_panel", "table5_rep4k"):
        summary[tname] = {j: {"raw_negative_ci_excl_0": v["n_raw_negative_ci_excludes_0"],
                              "holm_survivors": v["n_survive_holm"],
                              "holm_survivors_negative": v["n_survive_holm_negative"],
                              "primary_p": v["primary_p"],
                              "fail_negative": v["fail_negative"], "fail_positive": v["fail_positive"]}
                          for j, v in res["tables"][tname]["per_judge"].items()}
    summary["table6_form"] = {j: {f: {"m": v["m"], "raw_ci_excl_0": v["n_raw_ci_excludes_0"], "holm_survivors": v["n_survive_holm"],
                                      "survivors": v["survivors"], "fail_negative": v["fail_negative"],
                                      "fail_positive": v["fail_positive"], "primary_p": v["primary_p"]}
                                  for f, v in fams.items()}
                              for j, fams in res["tables"]["table6_form"]["per_judge"].items()}
    res["summary"] = summary

    # ---- null readings: half-widths and MDEs
    nr = {}
    pb = _load(OUT_DIR / "pushback_readout_pooled.json")
    c = pb["pooled"]["net"]["not_minus_cot"]
    nr["pushback_net_pooled"] = {**mde_block(c["diff"], c["lo"], c["hi"], scale=100,
                                             label="17.9 capitulation NET of neutral re-ask instability, NoT minus CoT, pooled "
                                                   "(nano, Llama, grok); registered primary, reading NULL"),
                                 "registered_reading": pb["reading"]["primary_net_of_instability"],
                                 "n_clusters": c["n_clusters"], "n_pairs": c["n_pairs"],
                                 "source": "pushback_readout_pooled.json pooled.net.not_minus_cot"}
    nr["pushback_net_per_model"] = {}
    for m, v in pb["per_model"].items():
        cc = v["net"]["not_minus_cot"]
        nr["pushback_net_per_model"][m] = {**mde_block(cc.get("diff"), cc.get("lo"), cc.get("hi"), scale=100,
                                                       label=f"17.9 net of instability, {m}"),
                                           "n_common_support": v["n_common_support"],
                                           "source": f"pushback_readout_pooled.json per_model.{m}.net.not_minus_cot"}
    nr["pushback_capitulation_raw_pooled"] = {**mde_block(pb["pooled"]["cap"]["not_minus_cot"]["diff"],
                                                          pb["pooled"]["cap"]["not_minus_cot"]["lo"],
                                                          pb["pooled"]["cap"]["not_minus_cot"]["hi"], scale=100,
                                                          label="17.9 raw capitulation, NoT minus CoT, pooled (companion; reading NOT-WORSE)"),
                                              "registered_reading": pb["reading"]["capitulation_raw"],
                                              "source": "pushback_readout_pooled.json pooled.cap.not_minus_cot"}

    rd = _load(OUT_DIR / "router_decomposition_haiku.json")
    ol = rd["counter_lift_by_stratum"]["one_loser"]
    nr["haiku_one_loser_lift"] = {**mde_block(ol["lift"], ol["lo"], ol["hi"], scale=1.0,
                                              label="16.23 haiku-249 counter lift within one-loser verdicts, P(wrong|fired) - P(wrong|unfired)"),
                                  "n_one_loser": ol["n"], "n_fired": ol["n_fired"], "n_unfired": ol["n_unfired"],
                                  "p_wrong_fired": ol["p_wrong_fired"], "p_wrong_unfired": ol["p_wrong_unfired"], "ratio": ol["ratio"],
                                  "registered_reading": "includes 0 (16.23 RESULTS; the mechanism condition fails on haiku)",
                                  "source": "router_decomposition_haiku.json counter_lift_by_stratum.one_loser"}

    ko = _load(OUT_DIR / "knockout_social_analysis.json")
    nr["knockout_shares"] = {"registered_rule": "17.3: NEITHER iff no knockout's pooled share CI clears 0.25 from above; "
                                                "half-widths are on the share of intact NoT's drop the knockout retakes",
                             "registered_reading": ko["reading"], "pooled": {}, "per_generator": {}}
    for k, v in ko["pooled"].items():
        s = v["share"]
        nr["knockout_shares"]["pooled"][k] = {**mde_block(s["point"], s["lo"], s["hi"], null_value=0.25, scale=1.0,
                                                          label=f"17.3 pooled share, {k}, against the registered 0.25 threshold"),
                                              "vs_zero": mde_block(s["point"], s["lo"], s["hi"], null_value=0.0, scale=1.0,
                                                                   label=f"17.3 pooled share, {k}, against 0"),
                                              "items": s["items"], "se_reported": s["se"],
                                              "source": f"knockout_social_analysis.json pooled.{k}.share"}
    for g, v in ko["per_generator"].items():
        nr["knockout_shares"]["per_generator"][g] = {}
        for k in ("not_drop_stakeholders", "not_drop_consequences", "not_drop_uncertainty", "not_commit_first"):
            s = v[k]["share"]
            nr["knockout_shares"]["per_generator"][g][k] = mde_block(s["point"], s["lo"], s["hi"], null_value=0.25, scale=1.0,
                                                                     label=f"17.3 {g} share, {k}, against 0.25")

    nr["persona_pooled"] = {}
    for judge in JUDGES:
        pp = form["judges"][judge]["persona_pooled"]["drop_arm"]
        nr["persona_pooled"][judge] = {**mde_block(pp["point"], pp["lo"], pp["hi"], scale=100,
                                                   label=f"17.7 persona-only (advisor CoT) minus CoT, pooled over four generators, judge {judge}"),
                                       "registered_reading": form["judges"][judge]["persona_reading"],
                                       "n_items": form["judges"][judge]["persona_pooled"]["n_items"],
                                       "n_pairs": form["judges"][judge]["persona_pooled"]["n_pairs"],
                                       "source": f"narrative_form_readout.json judges.{judge}.persona_pooled.drop_arm"}

    seat = _load(OUT_DIR / "seat_scaffold_comparison.json")
    lg = seat["per_model"]["llama"]["gain"]
    nr["llama_seat_gain"] = {**mde_block(lg["diff"], lg["lo"], lg["hi"], scale=100,
                                         label="16.28 routing gain, narrated minus plain seats, Llama k=2 (points)"),
                             "narrated": lg["narrated"], "plain": lg["plain"], "n_items": lg["n_items"],
                             "registered_reading": seat["reading"]["primary"],
                             "source": "seat_scaffold_comparison.json per_model.llama.gain"}
    companions = {}
    for m, key, lab in (("grok", "routed", "16.28 routed accuracy, narrated minus plain, grok k=4 (companion, CI includes 0)"),
                        ("llama", "fire", "16.28 counter fire rate, narrated minus plain, Llama (companion, CI includes 0)"),
                        ("grok", "shield", "16.28 asker-shielding, narrated minus plain, grok (registered NO-DIFFERENCE)"),
                        ("llama", "shield", "16.28 asker-shielding, narrated minus plain, Llama (registered NO-DIFFERENCE)")):
        cc = seat["per_model"][m][key]
        companions[f"{m}_{key}"] = {**mde_block(cc["diff"], cc["lo"], cc["hi"], scale=100, label=lab),
                                    "n_items": cc["n_items"], "source": f"seat_scaffold_comparison.json per_model.{m}.{key}"}
    nr["seat_companions"] = companions
    res["null_readings"] = nr
    res["inputs_read"] = sorted(set(INPUTS_READ))
    return res


# ----------------------------------------------------------------------------- report

def print_report(res: dict) -> None:
    ex = res["exact_recompute"]
    print(f"exact recompute: {ex.get('available')}" + (f"  ({ex.get('reason')})" if ex.get("reason") else ""))
    if ex.get("cache_stats_per_judge"):
        for j, s in ex["cache_stats_per_judge"].items():
            print(f"  cache {j}: rows {s['rows']}  hits {s['cache_hits']}  ({100 * s['hit_rate']:.1f}%)  unparsed {s['unparsed']}")
    for tname in ("table2_judge_panel", "table5_rep4k"):
        t = res["tables"][tname]
        print(f"\n== {tname}  ({t['source']}); Holm over 7 generators within judge, alpha 0.05")
        for j, v in t["per_judge"].items():
            print(f"  {j:26s} raw neg excl 0: {v['n_raw_negative_ci_excludes_0']}/7  Holm survivors: {v['n_survive_holm']}/7 "
                  f"(neg {v['n_survive_holm_negative']}; p={v['primary_p']})"
                  + (f"  disagreement implied/exact: {v['holm_disagreement_implied_vs_exact']}"
                     if v["holm_disagreement_implied_vs_exact"] else ""))
            for g, c in v["cells"].items():
                pe = f"{c['p_exact']:.2e}" if c["p_exact"] is not None else "n/a"
                adj = c["holm_adj_p_exact"] if c["holm_adj_p_exact"] is not None else c["holm_adj_p_implied"]
                print(f"      {g:26s} {100 * c['point']:+6.1f} [{100 * c['lo']:+6.1f},{100 * c['hi']:+6.1f}]  "
                      f"p_impl {c['p_implied']:.2e}  p_exact {pe}  match {c['exact_ci_match']}  "
                      f"adj {adj:.3f}  {'SURVIVES' if (c['survives_holm_exact'] if c['survives_holm_exact'] is not None else c['survives_holm_implied']) else 'fails'}")
            if v["fail_negative"] or v["fail_positive"]:
                print(f"      fail negative: {v['fail_negative']}   fail positive: {v['fail_positive']}")
    t = res["tables"]["table6_form"]
    print(f"\n== table6_form  ({t['source']}); Holm over the four 17.7 generators within judge")
    for j, fams in t["per_judge"].items():
        print(f"  {j}")
        for f, v in fams.items():
            print(f"      {f:22s} m={v['m']} raw excl 0 {v['n_raw_ci_excludes_0']}  Holm {v['n_survive_holm']} (p={v['primary_p']})"
                  f"  fail neg {v['fail_negative']}  fail pos {v['fail_positive']}")
    print("\n== null readings: half-width and 80%-power MDE")
    nr = res["null_readings"]

    def line(name, b):
        if not b.get("available", True):
            print(f"  {name}: unavailable"); return
        print(f"  {name:40s} {b['point']:+7.2f} [{b['lo']:+7.2f},{b['hi']:+7.2f}]  half-width {b['half_width']:.2f}  "
              f"mde80 {b['mde_80']:.2f}  {'(' + b['scale_note'] + ')'}  null {b['null_value']}  excl null {b['ci_excludes_null']}")
    line("pushback_net_pooled", nr["pushback_net_pooled"])
    for m, b in nr["pushback_net_per_model"].items():
        line(f"pushback_net {m}", b)
    line("pushback_capitulation_raw_pooled", nr["pushback_capitulation_raw_pooled"])
    line("haiku_one_loser_lift", nr["haiku_one_loser_lift"])
    for k, b in nr["knockout_shares"]["pooled"].items():
        line(f"knockout share {k} (vs 0.25)", b)
    for j, b in nr["persona_pooled"].items():
        line(f"persona {j}", b)
    line("llama_seat_gain", nr["llama_seat_gain"])
    for k, b in nr["seat_companions"].items():
        line(f"seat {k}", b)


# ----------------------------------------------------------------------------- selftest

def _selftest() -> int:
    ok = True

    def check(msg, cond):
        nonlocal ok
        print(("ok   " if cond else "FAIL ") + msg)
        ok = ok and bool(cond)

    # implied p: an interval whose upper bound sits exactly at 0 gives p = 0.05
    check("p_implied at CI touching 0 is 0.05", abs(p_implied(-0.10, -0.20, 0.0) - 0.05) < 1e-9)
    check("p_implied symmetric in sign", abs(p_implied(0.1, 0.02, 0.18) - p_implied(-0.1, -0.18, -0.02)) < 1e-12)
    check("p_implied near 1 at point 0", abs(p_implied(0.0, -0.1, 0.1) - 1.0) < 1e-12)
    # exact p from draws
    b = np.array([-1.0] * 9990 + [1.0] * 10)
    check("p_exact two-sided from draws", abs(p_exact(b) - 0.002) < 1e-12)
    check("p_exact floor at 2/draws", p_exact(np.full(8000, -1.0)) == 2 / 8000)
    # Holm against a hand computation
    h = holm({"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.5})
    check("Holm adj p rank 1 = 4*0.001", abs(h["a"]["adj_p"] - 0.004) < 1e-12 and h["a"]["reject"])
    check("Holm adj p rank 2 = max(0.004, 3*0.02)", abs(h["b"]["adj_p"] - 0.06) < 1e-12 and not h["b"]["reject"])
    check("Holm adj p rank 3 = max(0.06, 2*0.03)", abs(h["c"]["adj_p"] - 0.06) < 1e-12 and not h["c"]["reject"])
    check("Holm adj p rank 4 = max(0.06, 0.5)", abs(h["d"]["adj_p"] - 0.5) < 1e-12)
    h2 = holm({"a": 0.001, "b": None})
    check("Holm excludes None, m=1", h2["a"]["adj_p"] == 0.001 and h2["b"]["adj_p"] is None)
    # MDE arithmetic
    m = mde_block(0.05, -0.02, 0.12, scale=100)
    check("half-width 7 points", abs(m["half_width"] - 7.0) < 1e-9)
    check("mde_80 = 2.8016 * se", abs(m["mde_80"] - MDE_MULT * 7.0 / Z975) < 1e-9)
    check("ci_excludes_null false when interval spans 0", m["ci_excludes_null"] is False)
    m2 = mde_block(0.18, 0.0036, 0.333, null_value=0.25)
    check("knockout share against 0.25: interval includes 0.25", m2["ci_excludes_null"] is False)
    # paired bootstrap: a clear effect has a small exact p and its interval excludes 0
    rng = np.random.default_rng(1)
    a = {str(i): int(v) for i, v in enumerate(rng.random(150) < 0.7)}
    bb = {str(i): int(v) for i, v in enumerate(rng.random(150) < 0.3)}
    r = paired_boot(a, bb, draws=2000, seed=3)
    check("paired_boot point = mean difference", abs(r["point"] - (np.mean(list(bb.values())) - np.mean(list(a.values())))) < 1e-12)
    check("paired_boot interval excludes 0 for a 40-point effect", r["hi"] < 0)
    check("p_exact small for that effect", p_exact(r["boots"]) <= 2 / 2000 + 1e-12)
    check("paired_boot deterministic in seed", paired_boot(a, bb, draws=500, seed=3)["lo"] == paired_boot(a, bb, draws=500, seed=3)["lo"])
    check("paired_boot None under 20 items", paired_boot({"1": 1}, {"1": 0}, draws=10, seed=1) is None)
    # form contrast: drop_not equals the paired point on the same items
    cells = {("g", "base"): a, ("g", "arm"): {k: 1 - v for k, v in bb.items()}, ("g", "not"): bb}
    fc = form_contrast_boot(cells, "g", "base", "arm", "not", draws=500, seed=5)
    check("form_contrast drop_not = paired point", abs(fc["drop_not"]["point"] - r["point"]) < 1e-12)
    check("form_contrast gap = rate(not) - rate(arm)",
          abs(fc["gap"]["point"] - (np.mean(list(bb.values())) - np.mean([1 - v for v in bb.values()]))) < 1e-12)
    # family analysis on synthetic cells: implied-only path, then exact path with a mismatch fallback
    fam = {"x": {"point": -0.3, "lo": -0.4, "hi": -0.2}, "y": {"point": -0.05, "lo": -0.15, "hi": 0.05},
           "z": {"point": 0.1, "lo": 0.02, "hi": 0.18}}
    an = analyse_family(fam)
    check("implied-only family: primary is implied", an["primary_p"] == "implied")
    check("implied-only family: x survives, y fails negative, z fails or survives by its own p",
          "x" in an["survivors"] and "y" in an["fail_negative"])
    exact = {g: {"p_exact": 0.0005, "ci_match": True} for g in fam}
    an2 = analyse_family(fam, exact)
    check("exact family: primary is exact when every cell matches", an2["primary_p"] == "exact")
    exact["y"]["ci_match"] = False
    an3 = analyse_family(fam, exact)
    check("exact family: one mismatch drops the whole family to implied", an3["primary_p"] == "implied")
    # cache path shape and the id/prompt reader are pure functions of their inputs
    p = cache_path("q", "a", "gpt-4o")
    check("cache_path in OUT_DIR with the validation/full namespace",
          p.parent == OUT_DIR and p.name.startswith("elephant_rescore_validation_gpt-4o_full_") and p.name.endswith(".json"))
    check("_safe replaces slashes", _safe("a/b c") == "a_b_c")
    print("ALL OK" if ok else "SELFTEST FAILED")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-exact", action="store_true", help="skip the per-item recompute; implied p-values only")
    ap.add_argument("--scan-batches", action="store_true",
                    help="glob the batch CSVs instead of the fixed list (about a minute on this directory)")
    ap.add_argument("--json", default=str(OUT), help=f"artefact path (default {OUT})")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    res = run(exact=not a.no_exact, scan=a.scan_batches)
    out = Path(a.json)
    out.write_text(json.dumps(res, indent=1, default=lambda o: None))
    print_report(res)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
