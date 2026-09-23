"""
analyze_readouts_17_3_17_5.py -- the registered readouts that the Addendum 17.3 and 17.5
RESULTS blocks and their artefacts (knockout_social_analysis.json,
counterparty_causal_analysis.json) do not report.

Zero spend, cache-only, stdlib only. Reads exactly these files:
  divergence_study_outputs/elephant_singleagent_raw.csv                the OEQ cells
  divergence_study_outputs/elephant_rescore_validation_*_full_*.json   the untruncated
      production-judge re-scores (the same cache apply_corrected_scores reads)
  divergence_study_outputs/elephant_cp_<item>.json                     Z2's per-item
      counterparty coding (one YES/NO per item, already on disk)
  divergence_study_outputs/knockout_social_analysis.json               cross-check only
  divergence_study_outputs/counterparty_causal_analysis.json           cross-check only
  data/elephant/OEQ.csv                                                the item prompts,
      needed only to rebuild the re-score cache key, which hashes the prompt text

It deliberately does NOT import scripts.generators, scripts.elephant_scorers,
scripts.rescore_elephant_untruncated, scripts.run_phase1_quartet, scripts.load_elephant
or scripts.scaffold_permutations (each of those imports the API client layer or can
download), and it installs a socket guard at import so that any network call raises.
The bootstrap, Manski and section-label primitives are imported from
scripts.analyze_length_matched_elephant, whose import chain is stdlib only, so every
interval here is computed by the same code as 16.15.7, 17.3, 17.4 and 17.5.

17.3 (section knockouts; registration ~L7156-7160 and the Addendum 17 shared guards
~L7080-7081 of Guidance_Documents/prereg_embodiment_community.md), per generator and arm,
and pooled over the three generators:
  * non-response: items absent from the CSV or with an empty response, out of the 150
    registered items (the one grok commit-first item lost to a transient generation
    failure counts as a non-response here);
  * compliance: the share of non-empty responses carrying EVERY section label the arm's
    prompt mandates, under the crude substring rule of sections_present (intact NoT and
    commit-first mandate five labels, each drop arm the four it keeps). A descriptive,
    not a mechanism: the compliance retraction of HANDOFF 2026-08-26 R1 stands;
  * mean and median response length in characters over non-empty responses;
  * the Manski bracket on each arm's validation rate (every unscored item imputed as
    not validating, then as validating) and on the knockout-minus-intact-NoT and
    knockout-minus-CoT contrasts, the widest defensible bracket, exactly the _bounds
    convention 16.15.7 uses for NoT minus CoT.

17.5 (one-line counterparty instruction; registration ~L7992-7994): the share of intact
NoT's reduction the one-line arm recovers,
      share_recovered = (CoT rate - arm rate) / (CoT rate - intact NoT rate),
on the items scored in all three arms (1 = the one-line arm reproduces the whole
reduction, 0 = none of it, negative = it moves validation the other way, above CoT),
per generator on all items, on counterparty-present items and on counterparty-absent
items (Z2's coding), item-clustered percentile bootstrap (8,000 draws, seed 20260922,
the 17.5 seed), and pooled as the unweighted mean of the per-generator shares with
items clustered within generator (the 17.3 pooled estimator). The complement,
share_taken_back = 1 - share_recovered, is the 17.3 "share taken back" convention
(0 = behaves like intact NoT, 1 = like CoT) and is reported beside it.

Usage:
  .venv/bin/python -m scripts.analyze_readouts_17_3_17_5             # writes the artefact
  .venv/bin/python -m scripts.analyze_readouts_17_3_17_5 --selftest  # reads no files
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import socket
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "divergence_study_outputs"
RAW_CSV = OUT / "elephant_singleagent_raw.csv"
OEQ_CSV = ROOT / "data" / "elephant" / "OEQ.csv"
ARTEFACT = OUT / "readouts_17_3_17_5.json"
KNOCKOUT_ARTEFACT = OUT / "knockout_social_analysis.json"
COUNTERPARTY_ARTEFACT = OUT / "counterparty_causal_analysis.json"


def _no_network(*_a, **_k):
    raise RuntimeError("network disabled: analyze_readouts_17_3_17_5 is cache-only")


socket.socket = _no_network          # type: ignore[assignment]
socket.create_connection = _no_network
socket.getaddrinfo = _no_network

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.analyze_length_matched_elephant import (  # noqa: E402  stdlib-only module
    SECTION_LABELS, _cluster_bootstrap, _manski_mean, _scored, sections_present,
)

FORBIDDEN_MODULES = (
    "scripts.generators", "scripts.elephant_scorers", "scripts.rescore_elephant_untruncated",
    "scripts.run_phase1_quartet", "scripts.load_elephant", "scripts.scaffold_permutations",
    "requests", "httpx", "urllib3", "openai", "anthropic",
)

COL = "sycophantic_validation"
METRIC = "validation"
JUDGE = "claude-haiku-4-5"
PRODUCTION_LIMIT = 4000           # scripts/elephant_scorers.py: advice[:4000]
N_ITEMS = 150
GENS = ("claude-haiku-4-5", "grok-4-1-fast-reasoning", "gpt-5.4-nano")
COT, INTACT = "standard_cot", "narrative_cot"
# Section labels each knockout prompt mandates (verified against
# scripts.scaffold_permutations.PERMUTATIONS on 2026-09-23; not imported, see above).
KNOCKOUT_LABELS = {
    "not_drop_stakeholders": ("protagonist", "consequence", "uncertain", "decision"),
    "not_drop_consequences": ("protagonist", "stakeholder", "uncertain", "decision"),
    "not_drop_uncertainty": ("protagonist", "stakeholder", "consequence", "decision"),
    "not_commit_first": tuple(SECTION_LABELS),
}
ARM_LABELS = {COT: (), INTACT: tuple(SECTION_LABELS), **KNOCKOUT_LABELS}
ONELINE_ARMS = ("not_counterparty_only", "not_style_control")
ARMS = (COT, INTACT, *KNOCKOUT_LABELS, *ONELINE_ARMS)
B, SEED = 8000, 20260922
MIN_ITEMS_PER_GEN, MIN_ITEMS_POOLED = 10, 5   # the 17.3 estimator's gates


# --------------------------------------------------------------------------
# loaders (cache only)
# --------------------------------------------------------------------------
def load_rows() -> list[dict[str, str]]:
    """The OEQ rows of the eight arms. Opened exactly as scripts.analyze_length_matched_elephant._rows
    and scripts.rescore_elephant_untruncated._load open it (default universal newlines), because the
    re-score cache key hashes the response text as those loaders read it."""
    csv.field_size_limit(sys.maxsize)
    if not RAW_CSV.exists():
        raise SystemExit(f"missing {RAW_CSV}")
    with RAW_CSV.open() as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["dataset"] == "oeq" and r["generator"] in GENS and r["arm"] in ARMS]
    if not rows:
        raise SystemExit("no OEQ rows for the registered generators and arms")
    return rows


def _safe(s: str) -> str:
    """scripts.run_phase1_quartet._safe, copied so that module is not imported."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def rescore_key(question: str, advice: str) -> Path:
    """Byte-identical to scripts.rescore_elephant_untruncated._key(METRIC, question, advice, JUDGE, None)."""
    h = hashlib.sha1(f"{METRIC}|{question}|{advice}|{JUDGE}|lim=full".encode()).hexdigest()[:16]
    return OUT / f"elephant_rescore_{METRIC}_{_safe(JUDGE)}_full_{h}.json"


def load_questions(item_ids) -> dict[str, str]:
    """item_id -> prompt, built as scripts.load_elephant builds them: the id is the OEQ.csv
    'Unnamed: 0' column and the prompt is the 'prompt' column stripped. OEQ.csv carries no
    carriage returns, so csv and the loader's pandas read give the same text."""
    wanted = set(item_ids)
    out: dict[str, str] = {}
    with OEQ_CSV.open(newline="") as fh:
        for r in csv.DictReader(fh):
            rid = (r.get("Unnamed: 0") or "").strip()
            if rid in wanted:
                out[rid] = (r.get("prompt") or "").strip()
    return out


def apply_corrected_scores(rows: list[dict[str, str]]) -> dict:
    """Overwrite COL in place with the untruncated re-score wherever the response exceeds the
    production judge's 4,000-character cut, exactly as
    scripts.analyze_length_matched_elephant.apply_corrected_scores does, reading only the warm
    cache. Rows at or under the limit already had complete text and are left alone."""
    questions = load_questions({r["item_id"] for r in rows})
    replaced = missing = eligible = 0
    for r in rows:
        resp = r.get("response") or ""
        if len(resp) <= PRODUCTION_LIMIT:
            continue
        eligible += 1
        q = questions.get(r["item_id"])
        k = rescore_key(q, resp) if q is not None else None
        if k is None or not k.exists():
            missing += 1
            continue
        try:
            s = int(json.loads(k.read_text()).get("score", -1))
        except Exception:
            missing += 1
            continue
        if s in (0, 1):
            r[COL] = str(s)
            replaced += 1
        else:
            missing += 1
    return {"judge": JUDGE, "limit_chars": PRODUCTION_LIMIT, "eligible_over_limit": eligible,
            "replaced": replaced, "missing_from_cache": missing}


def load_coding(item_ids) -> dict[str, str]:
    """Z2's counterparty coding, one file per item; read by name, no directory glob."""
    coding: dict[str, str] = {}
    for i in sorted(set(item_ids)):
        p = OUT / f"elephant_cp_{i}.json"
        if p.exists():
            coding[i] = str(json.loads(p.read_text()).get("answer", "")).strip().upper()
    return coding


# --------------------------------------------------------------------------
# 17.3 descriptives and Manski brackets
# --------------------------------------------------------------------------
def cell_rows(rows, gen, arm):
    return [r for r in rows if r["generator"] == gen and r["arm"] == arm]


def compliant(text, labels) -> bool:
    """Every mandated label present, the crude substring rule of sections_present."""
    t = (text or "").lower()
    return all(lab in t for lab in labels)


def descriptives(sub, labels, n_items=N_ITEMS) -> dict:
    nonempty = [r for r in sub if (r.get("response") or "").strip()]
    lens = [len(r["response"]) for r in nonempty]
    d = {
        "n_rows": len(sub),
        "n_absent": n_items - len(sub),
        "n_empty": len(sub) - len(nonempty),
        "n_nonempty": len(nonempty),
        "n_scored": sum(1 for r in sub if _scored(r, COL)),
        "nonresponse_pct": 100.0 * (n_items - len(nonempty)) / n_items,
        "mean_len_chars": st.mean(lens) if lens else None,
        "median_len_chars": st.median(lens) if lens else None,
        "mean_sections_present": st.mean(sections_present(r["response"]) for r in nonempty) if nonempty else None,
        "mandated_labels": list(labels),
        "n_compliant": None,
        "compliance_pct": None,
    }
    if labels and nonempty:
        n_c = sum(1 for r in nonempty if compliant(r["response"], labels))
        d["n_compliant"] = n_c
        d["compliance_pct"] = 100.0 * n_c / len(nonempty)
    return d


def arm_bounds(sub, n_items=N_ITEMS) -> dict:
    """Manski bracket on the arm's validation rate: every unscored item (absent, empty or
    unjudged) imputed as not validating (lo), then as validating (hi)."""
    obs = [int(r[COL]) for r in sub if _scored(r, COL)]
    miss = n_items - len(obs)
    lo, hi = _manski_mean(obs, [(0.0, 1.0)] * miss)
    return {"n_scored": len(obs), "n_missing": miss,
            "rate_pct_complete_case": 100.0 * st.mean(obs) if obs else None,
            "rate_pct_manski_lo": 100.0 * lo if lo is not None else None,
            "rate_pct_manski_hi": 100.0 * hi if hi is not None else None}


def contrast_bracket(a: dict, b: dict) -> dict:
    """a minus b in points: complete case, and the widest defensible Manski bracket
    [a_lo - b_hi, a_hi - b_lo] (the 16.15.7 _bounds convention)."""
    cc = (a["rate_pct_complete_case"] - b["rate_pct_complete_case"]
          if a["rate_pct_complete_case"] is not None and b["rate_pct_complete_case"] is not None else None)
    return {"complete_case": cc,
            "manski_lo": a["rate_pct_manski_lo"] - b["rate_pct_manski_hi"],
            "manski_hi": a["rate_pct_manski_hi"] - b["rate_pct_manski_lo"]}


def compute_knockouts(rows) -> dict:
    out: dict = {"per_generator": {}, "pooled_rows": {}}
    for gen in GENS:
        g: dict = {}
        cot, intact = cell_rows(rows, gen, COT), cell_rows(rows, gen, INTACT)
        cb, ib = arm_bounds(cot), arm_bounds(intact)
        g[COT] = {**descriptives(cot, ARM_LABELS[COT]), **cb}
        g[INTACT] = {**descriptives(intact, ARM_LABELS[INTACT]), **ib,
                     "not_minus_cot_pp": contrast_bracket(ib, cb)}
        for arm, labels in KNOCKOUT_LABELS.items():
            sub = cell_rows(rows, gen, arm)
            kb = arm_bounds(sub)
            g[arm] = {**descriptives(sub, labels), **kb,
                      "knock_minus_intact_pp": contrast_bracket(kb, ib),
                      "knock_minus_cot_pp": contrast_bracket(kb, cb)}
        out["per_generator"][gen] = g
    # rows pooled over the three generators (descriptives only; 450 registered items)
    for arm in (COT, INTACT, *KNOCKOUT_LABELS):
        sub = [r for r in rows if r["arm"] == arm]
        d = descriptives(sub, ARM_LABELS[arm], n_items=N_ITEMS * len(GENS))
        out["pooled_rows"][arm] = {k: d[k] for k in ("n_rows", "n_absent", "n_empty", "n_nonempty", "n_scored",
                                                    "nonresponse_pct", "mean_len_chars", "median_len_chars",
                                                    "mean_sections_present", "compliance_pct", "n_compliant")}
    return out


# --------------------------------------------------------------------------
# 17.5 share recovered
# --------------------------------------------------------------------------
def share_recovered(cot_vals, intact_vals, arm_vals):
    """(CoT - arm)/(CoT - intact) on matched lists; None when the denominator is zero."""
    c, n, a = st.mean(cot_vals), st.mean(intact_vals), st.mean(arm_vals)
    denom = c - n
    if abs(denom) < 1e-9:
        return None
    return (c - a) / denom


def share_stat_factory(cot, intact, arm, min_items=MIN_ITEMS_PER_GEN):
    """Per-resample share on the drawn items (duplicates kept, as a bootstrap requires)."""
    def stat(recs):
        ids = [r["item_id"] for r in recs if r["item_id"] in cot and r["item_id"] in intact and r["item_id"] in arm]
        if len(ids) < min_items:
            return None
        return share_recovered([cot[i] for i in ids], [intact[i] for i in ids], [arm[i] for i in ids])
    return stat


def pooled_share_stat_factory(by, arm, min_items=MIN_ITEMS_POOLED):
    """Unweighted mean of per-generator shares; records carry 'gen:item' ids (the 17.3 pooled estimator)."""
    def stat(recs):
        groups: dict[str, list[str]] = {}
        for r in recs:
            gen, item = r["item_id"].split(":", 1)
            groups.setdefault(gen, []).append(item)
        shares = []
        for gen, items in groups.items():
            cot, intact, a = by[(gen, COT)], by[(gen, INTACT)], by[(gen, arm)]
            ids = [i for i in items if i in cot and i in intact and i in a]
            if len(ids) < min_items:
                continue
            s = share_recovered([cot[i] for i in ids], [intact[i] for i in ids], [a[i] for i in ids])
            if s is not None:
                shares.append(s)
        return st.mean(shares) if shares else None
    return stat


def _ci_block(ci: dict) -> dict:
    """Keep the interval fields and add the exact complement (share taken back = 1 - share)."""
    pt, lo, hi = ci["point"], ci["lo"], ci["hi"]
    return {"share_recovered": {"point": pt, "lo": lo, "hi": hi, "se": ci.get("se")},
            "share_taken_back": {"point": None if pt is None else 1.0 - pt,
                                 "lo": None if hi is None else 1.0 - hi,
                                 "hi": None if lo is None else 1.0 - lo},
            "items": ci["items"], "b": ci["b"], "degenerate": ci["degenerate"]}


def compute_counterparty(rows, coding, *, b=B, seed=SEED) -> dict:
    present = {i for i, v in coding.items() if v == "YES"}
    absent = {i for i, v in coding.items() if v == "NO"}
    by = {}
    for gen in GENS:
        for arm in (COT, INTACT, *ONELINE_ARMS):
            by[(gen, arm)] = {r["item_id"]: int(r[COL]) for r in cell_rows(rows, gen, arm) if _scored(r, COL)}
    strata = (("all", None), ("counterparty_present", present), ("counterparty_absent", absent))
    out: dict = {"coding": {"present": len(present), "absent": len(absent), "coded": len(coding)},
                 "bootstrap": {"b": b, "seed": seed, "min_items_per_generator": MIN_ITEMS_PER_GEN,
                               "min_items_pooled_per_generator": MIN_ITEMS_POOLED},
                 "per_generator": {}, "pooled": {}}
    for arm in ONELINE_ARMS:
        out["per_generator"][arm] = {}
        for gen in GENS:
            cot, intact, a = by[(gen, COT)], by[(gen, INTACT)], by[(gen, arm)]
            common = set(cot) & set(intact) & set(a)
            cell: dict = {}
            for name, restrict in strata:
                ids = sorted(common if restrict is None else common & restrict)
                recs = [{"item_id": i} for i in ids]
                ci = _cluster_bootstrap(recs, share_stat_factory(cot, intact, a), b=b, seed=seed)
                blk = _ci_block(ci)
                blk["rates_pct_on_these_items"] = {
                    "cot": 100.0 * st.mean(cot[i] for i in ids) if ids else None,
                    "intact_not": 100.0 * st.mean(intact[i] for i in ids) if ids else None,
                    "arm": 100.0 * st.mean(a[i] for i in ids) if ids else None}
                cell[name] = blk
            cell["rate_pct_own_items"] = 100.0 * st.mean(a.values()) if a else None
            cell["n_own_items"] = len(a)
            out["per_generator"][arm][gen] = cell
        out["pooled"][arm] = {}
        for name, restrict in strata:
            recs = []
            for gen in GENS:
                ids = set(by[(gen, COT)]) & set(by[(gen, INTACT)]) & set(by[(gen, arm)])
                if restrict is not None:
                    ids &= restrict
                recs.extend({"item_id": f"{gen}:{i}"} for i in sorted(ids))
            ci = _cluster_bootstrap(recs, pooled_share_stat_factory(by, arm), b=b, seed=seed)
            out["pooled"][arm][name] = _ci_block(ci)
    return out


# --------------------------------------------------------------------------
# cross-check against the registered artefacts (same cells, same corrected scores)
# --------------------------------------------------------------------------
def crosscheck(knock: dict, cp: dict) -> dict:
    res: dict = {}
    if KNOCKOUT_ARTEFACT.exists():
        ref = json.loads(KNOCKOUT_ARTEFACT.read_text())["per_generator"]
        diffs = []
        for gen in GENS:
            mine = knock["per_generator"][gen]
            diffs.append(abs(mine[COT]["rate_pct_complete_case"] - ref[gen]["cot_rate"]))
            diffs.append(abs(mine[INTACT]["rate_pct_complete_case"] - ref[gen]["intact_rate"]))
            for arm in KNOCKOUT_LABELS:
                diffs.append(abs(mine[arm]["rate_pct_complete_case"] - ref[gen][arm]["rate"]))
                diffs.append(abs(mine[arm]["n_scored"] - ref[gen][arm]["n"]))
        res["knockout_social_analysis"] = {"cells_compared": len(diffs), "max_abs_diff": max(diffs),
                                           "match": max(diffs) < 1e-6}
    else:
        res["knockout_social_analysis"] = {"match": None, "note": "artefact not on disk"}
    if COUNTERPARTY_ARTEFACT.exists():
        ref = json.loads(COUNTERPARTY_ARTEFACT.read_text())["per_generator"]
        diffs = []
        for arm in ONELINE_ARMS:
            for gen in GENS:
                mine = cp["per_generator"][arm][gen]
                diffs.append(abs(mine["rate_pct_own_items"] - ref[gen][arm]["rate"]))
                diffs.append(abs(mine["n_own_items"] - ref[gen][arm]["n"]))
        res["counterparty_causal_analysis"] = {"cells_compared": len(diffs), "max_abs_diff": max(diffs),
                                               "match": max(diffs) < 1e-6}
    else:
        res["counterparty_causal_analysis"] = {"match": None, "note": "artefact not on disk"}
    return res


# --------------------------------------------------------------------------
# main / print
# --------------------------------------------------------------------------
def _fmt(x, w=6, p=1):
    return f"{x:{w}.{p}f}" if isinstance(x, (int, float)) and x is not None else " " * (w - 1) + "-"


def print_report(out: dict) -> None:
    c = out["corrected_scores"]
    print(f"[corrected] {c['replaced']} of {c['eligible_over_limit']} over-limit responses replaced from the "
          f"untruncated cache; {c['missing_from_cache']} missing")
    print("\n17.3 KNOCKOUTS  rows nonempty nonresp%  compl%  meanchars  rate(cc)  Manski[lo,hi]   "
          "knock-intact cc [Manski]        knock-CoT cc [Manski]")
    for gen in GENS:
        print(gen)
        g = out["knockouts_17_3"]["per_generator"][gen]
        for arm in (COT, INTACT, *KNOCKOUT_LABELS):
            d = g[arm]
            line = (f"  {arm:22s} {d['n_rows']:4d} {d['n_nonempty']:8d} {_fmt(d['nonresponse_pct'], 8)} "
                    f"{_fmt(d['compliance_pct'], 7)} {_fmt(d['mean_len_chars'], 10, 0)} "
                    f"{_fmt(d['rate_pct_complete_case'], 9)}  [{_fmt(d['rate_pct_manski_lo'], 5)},{_fmt(d['rate_pct_manski_hi'], 5)}]")
            if "knock_minus_intact_pp" in d:
                ki, kc = d["knock_minus_intact_pp"], d["knock_minus_cot_pp"]
                line += (f"  {ki['complete_case']:+6.1f} [{ki['manski_lo']:+6.1f},{ki['manski_hi']:+6.1f}]"
                         f"   {kc['complete_case']:+6.1f} [{kc['manski_lo']:+6.1f},{kc['manski_hi']:+6.1f}]")
            elif "not_minus_cot_pp" in d:
                nc = d["not_minus_cot_pp"]
                line += f"  NoT-CoT {nc['complete_case']:+6.1f} [{nc['manski_lo']:+6.1f},{nc['manski_hi']:+6.1f}]"
            print(line)
    print("  pooled rows (450 items per arm)")
    for arm, d in out["knockouts_17_3"]["pooled_rows"].items():
        print(f"  {arm:22s} {d['n_rows']:4d} {d['n_nonempty']:8d} {_fmt(d['nonresponse_pct'], 8)} "
              f"{_fmt(d['compliance_pct'], 7)} {_fmt(d['mean_len_chars'], 10, 0)}")
    cp = out["counterparty_17_5"]
    print(f"\n17.5 SHARE RECOVERED = (CoT - arm)/(CoT - intact NoT); coding {cp['coding']}; "
          f"b={cp['bootstrap']['b']} seed={cp['bootstrap']['seed']}")
    for arm in ONELINE_ARMS:
        print(arm)
        for gen in (*GENS, "pooled"):
            cell = cp["pooled"][arm] if gen == "pooled" else cp["per_generator"][arm][gen]
            parts = []
            for name in ("all", "counterparty_present", "counterparty_absent"):
                s = cell[name]["share_recovered"]
                parts.append(f"{name.replace('counterparty_', '')}: {s['point']:+.2f} [{s['lo']:+.2f},{s['hi']:+.2f}] "
                             f"(n={cell[name]['items']}, degen {cell[name]['degenerate']})")
            print(f"  {gen:24s} " + "  ".join(parts))
    xc = out["crosscheck"]
    print(f"\n[crosscheck] knockout_social_analysis match={xc['knockout_social_analysis'].get('match')}  "
          f"counterparty_causal_analysis match={xc['counterparty_causal_analysis'].get('match')}")


def build(b=B, seed=SEED) -> dict:
    rows = load_rows()
    corrected = apply_corrected_scores(rows)
    knock = compute_knockouts(rows)
    coding = load_coding({r["item_id"] for r in rows})
    cp = compute_counterparty(rows, coding, b=b, seed=seed)
    forbidden_loaded = [m for m in FORBIDDEN_MODULES if m in sys.modules]
    out = {
        "script": "scripts/analyze_readouts_17_3_17_5.py",
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "registered": {"17.3": "prereg 17.3 registration ~L7156-7160; Addendum 17 shared guards ~L7080-7081",
                       "17.5": "prereg 17.5 registration ~L7992-7994"},
        "inputs": {"raw_csv": str(RAW_CSV.relative_to(ROOT)), "oeq_csv": str(OEQ_CSV.relative_to(ROOT)),
                   "rescore_cache": "divergence_study_outputs/elephant_rescore_validation_claude-haiku-4-5_full_<sha1[:16]>.json",
                   "counterparty_coding": "divergence_study_outputs/elephant_cp_<item>.json"},
        "hygiene": {"forbidden_modules_loaded": forbidden_loaded, "socket_guard": True},
        "corrected_scores": corrected,
        "definitions": {
            "nonresponse_pct": "100 * (150 - non-empty responses) / 150 per (generator, arm); absent items count",
            "compliance_pct": "share of non-empty responses containing every mandated section label as a lowercase "
                              "substring (sections_present rule); descriptive only, not a mechanism",
            "mean_len_chars": "mean response length in characters over non-empty responses",
            "rate_pct_manski_lo/hi": "validation rate with every unscored item (of 150) imputed as not validating / validating",
            "contrast manski_lo/hi": "[a_lo - b_hi, a_hi - b_lo] in points, the 16.15.7 _bounds convention",
            "share_recovered": "(CoT rate - arm rate) / (CoT rate - intact NoT rate) on items scored in all three arms; "
                               "1 = the arm reproduces NoT's whole reduction, 0 = none, negative = the arm raises "
                               "validation above CoT",
            "share_taken_back": "1 - share_recovered (the 17.3 convention: 0 = like intact NoT, 1 = like CoT)",
            "pooled share": "unweighted mean of per-generator shares, items clustered within generator, "
                            "item-clustered percentile bootstrap",
        },
        "caveats": [
            "nano's CoT and intact NoT are the June cells generated under a 2,048-token total budget before the "
            "8,192 reasoning floor; the 17.3 and 17.5 nano contrasts carry that budget mismatch (pre-spend amendment).",
            "grok's cached CoT was cut at 1,024 tokens on 42% of items; it is the CoT baseline in every grok share "
            "and contrast here.",
            "Compliance is the crude substring rule and is a descriptive; the 17.3 registration says it is not a mechanism.",
            "A share_recovered outside [0, 1] is expected for not_counterparty_only, whose rate lies above CoT (17.5 REVERSED).",
            "Scores are the production judge's (claude-haiku-4-5) untruncated re-scores; no other judge covers these arms.",
        ],
        "knockouts_17_3": knock,
        "counterparty_17_5": cp,
    }
    out["crosscheck"] = crosscheck(knock, cp)
    return out


# --------------------------------------------------------------------------
# selftest: pure, reads no files
# --------------------------------------------------------------------------
def _selftest() -> int:
    fails = 0

    def check(name, ok):
        nonlocal fails
        print(f"  {'ok ' if ok else 'FAIL'} {name}")
        if not ok:
            fails += 1

    try:
        socket.socket()
        check("socket guard raises", False)
    except RuntimeError:
        check("socket guard raises", True)
    check("no forbidden module loaded", not [m for m in FORBIDDEN_MODULES if m in sys.modules])
    check("_safe leaves the judge name untouched", _safe(JUDGE) == JUDGE)
    exp = hashlib.sha1("validation|Q?|advice text|claude-haiku-4-5|lim=full".encode()).hexdigest()[:16]
    check("rescore key reproduces the _key recipe",
          rescore_key("Q?", "advice text").name == f"elephant_rescore_validation_claude-haiku-4-5_full_{exp}.json")
    full = "Section 1 - Protagonist ... Stakeholders ... Consequences ... Uncertainty ... Decision: go."
    no_unc = "Protagonist ... Stakeholders ... Consequences ... Decision: go."
    check("compliance: intact needs all five", compliant(full, SECTION_LABELS) and not compliant(no_unc, SECTION_LABELS))
    check("compliance: drop_uncertainty needs four", compliant(no_unc, KNOCKOUT_LABELS["not_drop_uncertainty"]))
    check("compliance: drop_stakeholders keeps 'stakeholder' out of its rule",
          "stakeholder" not in KNOCKOUT_LABELS["not_drop_stakeholders"] and len(KNOCKOUT_LABELS["not_drop_stakeholders"]) == 4)
    lo, hi = _manski_mean([1, 1, 0, 0], [(0.0, 1.0)] * 2)
    check("Manski primitive brackets 2 missing of 6", abs(lo - 2 / 6) < 1e-12 and abs(hi - 4 / 6) < 1e-12)
    rows = [{"generator": "g", "arm": "a", "item_id": str(i), "response": "x" * 10, COL: "1"} for i in range(4)]
    rows += [{"generator": "g", "arm": "a", "item_id": "4", "response": "", COL: ""}]
    d = descriptives(rows, (), n_items=6)
    check("descriptives counts empty and absent as non-response",
          d["n_rows"] == 5 and d["n_empty"] == 1 and d["n_absent"] == 1 and abs(d["nonresponse_pct"] - 100 * 2 / 6) < 1e-9
          and d["mean_len_chars"] == 10 and d["compliance_pct"] is None)
    ab = arm_bounds(rows, n_items=6)
    check("arm bounds impute unscored items both ways",
          ab["n_scored"] == 4 and ab["n_missing"] == 2 and abs(ab["rate_pct_manski_lo"] - 100 * 4 / 6) < 1e-9
          and abs(ab["rate_pct_manski_hi"] - 100.0) < 1e-9 and ab["rate_pct_complete_case"] == 100.0)
    a = {"rate_pct_complete_case": 50.0, "rate_pct_manski_lo": 40.0, "rate_pct_manski_hi": 60.0}
    bb = {"rate_pct_complete_case": 30.0, "rate_pct_manski_lo": 25.0, "rate_pct_manski_hi": 35.0}
    cb = contrast_bracket(a, bb)
    check("contrast bracket is [a_lo - b_hi, a_hi - b_lo]",
          cb["complete_case"] == 20.0 and cb["manski_lo"] == 5.0 and cb["manski_hi"] == 35.0)
    ids = [str(i) for i in range(20)]
    cot = {i: 1 for i in ids}
    intact = {i: 0 for i in ids}
    half = {i: (1 if int(i) < 10 else 0) for i in ids}
    recs = [{"item_id": i} for i in ids]
    check("share: arm halfway between CoT and NoT recovers 0.5", abs(share_stat_factory(cot, intact, half)(recs) - 0.5) < 1e-12)
    check("share: arm identical to NoT recovers 1", share_stat_factory(cot, intact, dict(intact))(recs) == 1.0)
    check("share: arm identical to CoT recovers 0", share_stat_factory(cot, intact, dict(cot))(recs) == 0.0)
    cot_half = {i: (1 if int(i) < 10 else 0) for i in ids}
    check("share: arm above CoT is negative", share_stat_factory(cot_half, intact, cot)(recs) == -1.0)
    check("share: zero denominator is None", share_stat_factory(cot, dict(cot), half)(recs) is None)
    check("share: below the item gate is None", share_stat_factory(cot, intact, half)(recs[:5]) is None)
    check("share: duplicates in a draw count with multiplicity",
          abs(share_stat_factory(cot, intact, half)(recs[:10] + recs[:10] + recs[10:]) - (1 - 20 / 30)) < 1e-12)
    by = {("g1", COT): cot, ("g1", INTACT): intact, ("g1", "arm"): half,
          ("g2", COT): cot, ("g2", INTACT): intact, ("g2", "arm"): {i: (1 if int(i) < 4 else 0) for i in ids}}
    precs = [{"item_id": f"{g}:{i}"} for g in ("g1", "g2") for i in ids]
    check("pooled share is the unweighted mean of per-generator shares",
          abs(pooled_share_stat_factory(by, "arm")(precs) - (0.5 + 0.8) / 2) < 1e-12)
    ci1 = _cluster_bootstrap(recs, share_stat_factory(cot, intact, half), b=200, seed=1)
    ci2 = _cluster_bootstrap(recs, share_stat_factory(cot, intact, half), b=200, seed=1)
    check("bootstrap is seed-reproducible and brackets the point",
          ci1 == ci2 and ci1["lo"] <= ci1["point"] <= ci1["hi"] and ci1["degenerate"] == 0)
    blk = _ci_block({"point": 0.25, "lo": -0.1, "hi": 0.6, "se": 0.2, "items": 20, "b": 200, "degenerate": 0})
    check("share_taken_back is the exact complement with swapped ends",
          blk["share_taken_back"] == {"point": 0.75, "lo": 0.4, "hi": 1.1})
    # tiny end-to-end on synthetic rows: one-line arm identical to NoT on present items, to CoT on absent
    syn = []
    coding = {}
    for gen in GENS:
        for i in ids:
            present = int(i) < 10
            coding[i] = "YES" if present else "NO"
            y_cot, y_not = 1, 0
            y_arm = y_not if present else y_cot
            for arm, y in ((COT, y_cot), (INTACT, y_not), ("not_counterparty_only", y_arm), ("not_style_control", y_cot)):
                syn.append({"generator": gen, "arm": arm, "item_id": i, "response": f"Protagonist stakeholders consequences uncertain decision {arm}", COL: str(y)})
    cp = compute_counterparty(syn, coding, b=50, seed=3)
    pg = cp["per_generator"]["not_counterparty_only"][GENS[0]]
    check("end-to-end: present items recover 1, absent 0, all 10/20",
          pg["counterparty_present"]["share_recovered"]["point"] == 1.0
          and pg["counterparty_absent"]["share_recovered"]["point"] == 0.0
          and abs(pg["all"]["share_recovered"]["point"] - 0.5) < 1e-12
          and cp["pooled"]["not_style_control"]["all"]["share_recovered"]["point"] == 0.0)
    for gen in GENS:
        for arm in KNOCKOUT_LABELS:
            for i in ids:
                syn.append({"generator": gen, "arm": arm, "item_id": i, "response": "Protagonist consequences uncertain decision", COL: "1"})
    kn = compute_knockouts(syn)
    g0 = kn["per_generator"][GENS[0]]
    check("end-to-end: compliance follows the arm's mandated labels",
          g0["not_drop_stakeholders"]["compliance_pct"] == 100.0 and g0["not_drop_consequences"]["compliance_pct"] == 0.0
          and g0[INTACT]["compliance_pct"] == 100.0 and g0[COT]["compliance_pct"] is None)
    check("end-to-end: knockout minus intact bracket widens by the 130 unscored items",
          g0["not_drop_stakeholders"]["knock_minus_intact_pp"]["complete_case"] == 100.0
          and g0["not_drop_stakeholders"]["n_missing"] == 130)
    print(f"selftest: {fails} failure(s)")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selftest", action="store_true", help="run the pure self-test (reads no files) and exit")
    ap.add_argument("--b", type=int, default=B, help="bootstrap draws (default 8000)")
    ap.add_argument("--seed", type=int, default=SEED, help="bootstrap seed (default 20260922, the 17.5 seed)")
    ap.add_argument("--out", type=Path, default=ARTEFACT, help="artefact path")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    out = build(b=args.b, seed=args.seed)
    args.out.write_text(json.dumps(out, indent=1))
    print_report(out)
    if out["hygiene"]["forbidden_modules_loaded"]:
        print("GUARD FAILED: forbidden module loaded:", out["hygiene"]["forbidden_modules_loaded"])
        return 2
    if out["corrected_scores"]["missing_from_cache"]:
        print(f"WARNING: {out['corrected_scores']['missing_from_cache']} over-limit responses have no cached re-score; "
              "their production (truncated) score was kept")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
