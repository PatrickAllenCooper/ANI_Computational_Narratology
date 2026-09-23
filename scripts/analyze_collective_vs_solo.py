"""
analyze_collective_vs_solo.py -- the collective against the same model's solo
majority-of-three, paired on the same items and framings, with item-clustered
bootstrap intervals. Zero spend: reads only CSV and JSON artefacts already on
disk under divergence_study_outputs/; no model calls, no network, no import of
the generator layer. Promoted from the W12 scratch check (2026-09-23).

Why. The pillar-3 text says the collective's verdict sits at or below the same
model answering alone, and that the routed collective beats the solo on some
models and not others (sections/pillar3.tex, "The collective is a sensor, not a
decider"). Those comparisons were descriptive: collective accuracy over debates
against solo accuracy over cells, with no paired interval. This script pairs
the two on the same (framing, item) cells and gives the interval.

Conventions, each mirrored from a registered analyzer:
  collective  S2 `verdict` against `gold_verdict`, published ESH coding (YTA and
              ESH at fault, NTA and NAH not); debates with an uncodable S1 or S2
              dropped, as scripts/verify_pillar3_headline.py.
  routed      S2 unless the counter fired (n_objectors >= 2), then the sonnet
              standard majority-of-3 verdict, ties falling back to S2, as
              verify_pillar3_headline.routed_ok. This is the per-debate rule of
              the routed-minus-collective headline, NOT the 16.22 cell-level
              rule (a cell fires when more than half its samples fire), so the
              routed-minus-solo figures here are not the 16.22 R1 certification
              in routing_certification*.json, which stands as registered.
  solo        the same model's standard-CoT majority over its three cached
              samples per (framing, item) cell; tie and uncodable cells dropped,
              as scripts/analyze_router_decomposition.solo_ok.
  pairing     a debate pairs with the solo cell of its own framing and item;
              framings third_person and as_asker only (the two the deliberation
              runs used). Debates whose cell has no codable solo are dropped
              from the paired readout and counted.
  estimator   paired item-clustered percentile bootstrap, 4,000 draws, seed 101
              (verify_pillar3_headline.boot_delta): per-item cluster sums of the
              paired difference, ratio of sums across resampled items. Two
              weightings:
                debate_weighted  every codable debate is one paired record, its
                                 cell's solo correctness subtracted (the
                                 weighting of the routed-minus-collective
                                 headline).
                cell_weighted    one record per cell, the cell's mean debate
                                 correctness minus its solo correctness (the
                                 16.22 population unit).
              A replicate at seed 13 (analyze_router_decomposition's seed) is
              recorded so the intervals can be seen not to hinge on the seed.

Cross-checks (read-only, reported and not gated): the debate-level collective
and routed accuracies against verify_pillar3_headline_<cell>.json (narrated
cells) or seat_scaffold_comparison.json (plain-seat cells), and the solo
accuracy on the paired cells against routing_certification*.json acc_base.

Run:
  python -m scripts.analyze_collective_vs_solo             # all seven cells
  python -m scripts.analyze_collective_vs_solo --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "divergence_study_outputs"
DEFAULT_OUT = OUT / "collective_vs_solo_paired.json"
DRAWS = 4000
SEED = 101              # verify_pillar3_headline.boot_delta
SEED_REPLICATE = 13     # analyze_router_decomposition DEFAULT_SEED
JUDGE = ("cg_sonnet_actuator_rows.csv", "claude-sonnet-4-6")
ARMS = ("third_person", "as_asker")

# cell -> (deliberation rows CSV, solo k=3 rows CSV, solo model, seats, cross-check sources)
CONDITIONS = {
    "grok":        {"rows": "cg_deliberation_rows.csv", "solo": "cg_scaffold_combined_rows_k3.csv",
                    "model": "grok-4-1-fast-reasoning", "seats": "narrated", "k": 4,
                    "verify": "verify_pillar3_headline_grok.json", "cert": "routing_certification.json",
                    "label": "grok-4-1-fast-reasoning, narrated seats, k = 4 (16.15.1 / 16.22 panel)"},
    "haiku":       {"rows": "cg_deliberation_haiku_249_rows.csv", "solo": "cg_scaffold_combined_rows_k3.csv",
                    "model": "claude-haiku-4-5", "seats": "narrated", "k": 1,
                    "verify": "verify_pillar3_headline_haiku_249.json", "cert": "routing_certification_haiku.json",
                    "label": "claude-haiku-4-5, narrated seats, k = 1, 249 items (16.23)"},
    "llama":       {"rows": "cg_deliberation_llama3370binstruct_rows.csv", "solo": "cg_llama_standard_k3_rows.csv",
                    "model": "Llama-3.3-70B-Instruct", "seats": "narrated", "k": 2,
                    "verify": "verify_pillar3_headline_llama.json", "cert": "routing_certification_llama.json",
                    "label": "Llama-3.3-70B-Instruct, narrated seats, k = 2 (16.24)"},
    "mistral":     {"rows": "cg_deliberation_mistrallarge32_rows.csv", "solo": "cg_mistral_standard_k3_rows.csv",
                    "model": "Mistral-Large-3-2", "seats": "narrated", "k": 2,
                    "verify": "verify_pillar3_headline_mistral.json", "cert": "routing_certification_mistral.json",
                    "label": "Mistral-Large-3-2, narrated seats, k = 2 (16.26)"},
    "deepseek":    {"rows": "cg_deliberation_deepseekv4pro_rows.csv", "solo": "cg_deepseek_standard_k3_rows.csv",
                    "model": "DeepSeek-V4-Pro", "seats": "narrated", "k": 2,
                    "verify": "verify_pillar3_headline_deepseek.json", "cert": "routing_certification_deepseek.json",
                    "label": "DeepSeek-V4-Pro, narrated seats, k = 2 (16.27)"},
    "grok_plain":  {"rows": "cg_deliberation_stdcot_rows.csv", "solo": "cg_scaffold_combined_rows_k3.csv",
                    "model": "grok-4-1-fast-reasoning", "seats": "plain", "k": 4,
                    "seat_cmp": ("seat_scaffold_comparison.json", "grok"), "cert": "routing_certification_grok_stdcot_k4.json",
                    "label": "grok-4-1-fast-reasoning, plain chain-of-thought seats, k = 4 (16.28)"},
    "llama_plain": {"rows": "cg_deliberation_llama3370binstruct_stdcot_rows.csv", "solo": "cg_llama_standard_k3_rows.csv",
                    "model": "Llama-3.3-70B-Instruct", "seats": "plain", "k": 2,
                    "seat_cmp": ("seat_scaffold_comparison.json", "llama"), "cert": "routing_certification_llama_stdcot.json",
                    "label": "Llama-3.3-70B-Instruct, plain chain-of-thought seats, k = 2 (16.28)"},
}


# ---------------------------------------------------------------------------
# coding and majorities (published ESH coding, as the registered analyzers)
# ---------------------------------------------------------------------------

def code(v):
    """YTA and ESH are at fault (1), NTA and NAH are not (0), anything else None."""
    return {"YTA": 1, "ESH": 1, "NTA": 0, "NAH": 0}.get(v)


def maj(codes):
    """Majority of the codable votes; None on no codable vote or a tie."""
    cs = [c for c in codes if c is not None]
    if not cs:
        return None
    n1 = sum(cs)
    n0 = len(cs) - n1
    return None if n1 == n0 else int(n1 > n0)


def gold_of(row) -> int:
    return 1 if row["gold_verdict"] == "YTA" else 0


# ---------------------------------------------------------------------------
# loaders (CSV only, exact paths, no globbing: the outputs dir holds ~800k files)
# ---------------------------------------------------------------------------

def load_judge(judge_csv: Path, judge_model: str) -> dict:
    """(arm, item) -> majority-of-3 code of the judge's cached standard samples."""
    votes = defaultdict(list)
    with open(judge_csv, newline="") as fh:
        for j in csv.DictReader(fh):
            if j["model"] == judge_model and j["scaffold"] == "standard":
                votes[(j["arm"], j["item_id"])].append(code(j["verdict"]))
    return {k: maj(v) for k, v in votes.items()}


def load_solo(solo_csv: Path, model: str):
    """Returns (solo_ok, gold): solo_ok[(arm, item)] = 1/0 for cells whose k=3
    standard majority is codable and not a tie; gold[item] = the item's gold."""
    votes = defaultdict(list)
    gold = {}
    with open(solo_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["model"] != model or r["scaffold"] != "standard" or r["arm"] not in ARMS:
                continue
            votes[(r["arm"], r["item_id"])].append(code(r["verdict"]))
            g = gold_of(r)
            if gold.setdefault(r["item_id"], g) != g:
                raise ValueError(f"gold_verdict disagrees within item {r['item_id']} in {solo_csv.name}")
    solo_ok = {}
    for k, v in votes.items():
        m = maj(v)
        if m is not None:
            solo_ok[k] = int(m == gold[k[1]])
    return solo_ok, gold, len(votes)


def debate_from_row(r: dict, judge_maj: dict):
    """One codable debate, or None (uncodable S1 or S2, as verify_pillar3_headline)."""
    s1, s2 = code(r.get("synthesis_verdict", "")), code(r["verdict"])
    if s1 is None or s2 is None:
        return None
    g = gold_of(r)
    ok = int(s2 == g)
    fired = int(int(r.get("n_objectors") or 0) >= 2)
    jv = judge_maj.get((r["arm"], r["item_id"]))
    routed = int(jv == g) if (fired and jv is not None) else ok
    return {"arm": r["arm"], "item": r["item_id"], "gold": g, "ok": ok, "fired": fired, "routed": routed}


def load_debates(rows_csv: Path, judge_maj: dict, gold: dict):
    debates, n_rows, n_dropped = [], 0, 0
    with open(rows_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            n_rows += 1
            if r["arm"] not in ARMS:
                raise ValueError(f"unexpected framing {r['arm']!r} in {rows_csv.name}")
            d = debate_from_row(r, judge_maj)
            if d is None:
                n_dropped += 1
                continue
            if r["item_id"] in gold and gold[r["item_id"]] != d["gold"]:
                raise ValueError(f"gold_verdict disagrees between solo and debate rows on item {r['item_id']}")
            debates.append(d)
    return debates, n_rows, n_dropped


# ---------------------------------------------------------------------------
# estimator: paired item-clustered percentile bootstrap, ratio of cluster sums
# ---------------------------------------------------------------------------

def boot_ratio(clusters, draws: int, seed: int):
    """clusters: list of (sum, count) per item. Point = sum/count over all;
    interval = 2.5/97.5 percentiles of the same ratio over resampled items."""
    sums = np.array([s for s, _ in clusters], dtype=float)
    cnts = np.array([c for _, c in clusters], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(draws, len(sums)))
    stats = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(sums.sum() / cnts.sum()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def _fmt(point, lo, hi, n_records, n_items):
    return {"delta": round(point, 4), "lo": round(lo, 4), "hi": round(hi, 4),
            "points": round(100 * point, 1), "points_lo": round(100 * lo, 1), "points_hi": round(100 * hi, 1),
            "ci_excludes_zero": bool(lo > 0 or hi < 0), "n_records": n_records, "n_items": n_items}


def paired_deltas(debates, solo_ok: dict, draws: int, seed: int) -> dict:
    """The four paired deltas on the debates whose cell has a codable solo."""
    paired = [d for d in debates if (d["arm"], d["item"]) in solo_ok]
    cells = sorted({(d["arm"], d["item"]) for d in paired})
    items = sorted({d["item"] for d in paired})
    out = {"n_paired_debates": len(paired), "n_paired_cells": len(cells), "n_items": len(items),
           "n_debates_without_solo": len(debates) - len(paired)}
    if not paired:
        return out
    # debate-weighted: one record per debate
    for lab, key in (("collective", "ok"), ("routed", "routed")):
        per = defaultdict(lambda: [0.0, 0])
        for d in paired:
            per[d["item"]][0] += d[key] - solo_ok[(d["arm"], d["item"])]
            per[d["item"]][1] += 1
        p, lo, hi = boot_ratio([tuple(v) for v in per.values()], draws, seed)
        out[f"{lab}_minus_solo_debate_weighted"] = _fmt(p, lo, hi, len(paired), len(items))
    # cell-weighted: one record per cell, the cell's mean debate correctness
    cell_acc = defaultdict(lambda: {"ok": [], "routed": []})
    for d in paired:
        cell_acc[(d["arm"], d["item"])]["ok"].append(d["ok"])
        cell_acc[(d["arm"], d["item"])]["routed"].append(d["routed"])
    for lab, key in (("collective", "ok"), ("routed", "routed")):
        per = defaultdict(lambda: [0.0, 0])
        for c in cells:
            per[c[1]][0] += float(np.mean(cell_acc[c][key])) - solo_ok[c]
            per[c[1]][1] += 1
        p, lo, hi = boot_ratio([tuple(v) for v in per.values()], draws, seed)
        out[f"{lab}_minus_solo_cell_weighted"] = _fmt(p, lo, hi, len(cells), len(items))
    out["solo_acc_paired_cells"] = round(float(np.mean([solo_ok[c] for c in cells])), 4)
    out["collective_acc_paired_debates"] = round(float(np.mean([d["ok"] for d in paired])), 4)
    out["routed_acc_paired_debates"] = round(float(np.mean([d["routed"] for d in paired])), 4)
    out["collective_acc_paired_cells_cell_weighted"] = round(float(np.mean([np.mean(cell_acc[c]["ok"]) for c in cells])), 4)
    out["routed_acc_paired_cells_cell_weighted"] = round(float(np.mean([np.mean(cell_acc[c]["routed"]) for c in cells])), 4)
    out["n_fired_paired"] = int(sum(d["fired"] for d in paired))
    # per-framing breakdown, descriptive (point estimates only)
    by_arm = {}
    for arm in ARMS:
        sub = [d for d in paired if d["arm"] == arm]
        if not sub:
            continue
        sc = [solo_ok[(d["arm"], d["item"])] for d in sub]
        by_arm[arm] = {"n_debates": len(sub),
                       "collective": round(float(np.mean([d["ok"] for d in sub])), 4),
                       "routed": round(float(np.mean([d["routed"] for d in sub])), 4),
                       "solo_debate_weighted": round(float(np.mean(sc)), 4),
                       "collective_minus_solo_points": round(100 * float(np.mean([d["ok"] - s for d, s in zip(sub, sc)])), 1),
                       "routed_minus_solo_points": round(100 * float(np.mean([d["routed"] - s for d, s in zip(sub, sc)])), 1)}
    out["by_framing_descriptive"] = by_arm
    return out


# ---------------------------------------------------------------------------
# read-only cross-checks against the artefacts already on disk
# ---------------------------------------------------------------------------

def _agree(a, b, tol):
    return None if (a is None or b is None) else bool(abs(a - b) <= tol)


def cross_checks(name: str, spec: dict, table: dict, res: dict) -> dict:
    chk = {}
    if "verify" in spec and (OUT / spec["verify"]).exists():
        v = json.load(open(OUT / spec["verify"]))["recomputed"]
        chk["verify_pillar3_headline"] = {
            "file": spec["verify"], "n": v["n"], "collective": round(v["collective"], 4), "routed": round(v["routed"], 4),
            "n_agree": v["n"] == table["n_debates"],
            "collective_agree": _agree(v["collective"], table["collective"], 0.0015),
            "routed_agree": _agree(v["routed"], table["routed"], 0.0015)}
    if "seat_cmp" in spec and (OUT / spec["seat_cmp"][0]).exists():
        s = json.load(open(OUT / spec["seat_cmp"][0]))["per_model"][spec["seat_cmp"][1]]
        chk["seat_scaffold_comparison"] = {
            "file": spec["seat_cmp"][0], "slot": spec["seat_cmp"][1],
            "collective_plain": round(s["acc"]["plain"], 4), "routed_plain": round(s["routed"]["plain"], 4),
            "collective_agree": _agree(s["acc"]["plain"], table["collective"], 0.005),
            "routed_agree": _agree(s["routed"]["plain"], table["routed"], 0.005),
            "note": "16.28 artefact; item-level means, so a few thousandths of slack against the debate mean"}
    if "cert" in spec and (OUT / spec["cert"]).exists():
        c = json.load(open(OUT / spec["cert"]))["routed_vs_solo"]
        slot = c.get("sonnet") or next(iter(c.values()))
        chk["routing_certification"] = {
            "file": spec["cert"], "n_cells": slot["n_cells"], "acc_base": round(slot["acc_base"], 4),
            "solo_acc_agree": _agree(slot["acc_base"], res.get("solo_acc_paired_cells"), 0.005),
            "n_cells_agree": slot["n_cells"] == res.get("n_paired_cells"),
            "note": "acc_base is the same solo majority-of-3 on that artefact's cell population; the 16.22 R1 "
                    "routed-minus-base figure there uses the cell-level firing rule and is not recomputed here"}
    return chk


# ---------------------------------------------------------------------------
# one condition end to end
# ---------------------------------------------------------------------------

def analyze(name: str, draws: int = DRAWS, seed: int = SEED) -> dict:
    spec = CONDITIONS[name]
    judge_maj = load_judge(OUT / JUDGE[0], JUDGE[1])
    solo_ok, gold, n_solo_cells_total = load_solo(OUT / spec["solo"], spec["model"])
    debates, n_rows, n_dropped = load_debates(OUT / spec["rows"], judge_maj, gold)
    table = {"n_rows": n_rows, "n_uncodable_dropped": n_dropped, "n_debates": len(debates),
             "collective": round(float(np.mean([d["ok"] for d in debates])), 4),
             "routed": round(float(np.mean([d["routed"] for d in debates])), 4),
             "fire": round(float(np.mean([d["fired"] for d in debates])), 4),
             "n_solo_cells_total": n_solo_cells_total, "n_solo_cells_codable": len(solo_ok),
             "solo_acc_all_codable_cells": round(float(np.mean(list(solo_ok.values()))), 4)}
    res = paired_deltas(debates, solo_ok, draws, seed)
    rep = paired_deltas(debates, solo_ok, draws, SEED_REPLICATE)
    res["replicate_seed_13"] = {k: {kk: rep[k][kk] for kk in ("lo", "hi", "points_lo", "points_hi", "ci_excludes_zero")}
                               for k in rep if "_minus_solo_" in k}
    return {"label": spec["label"], "model": spec["model"], "seats": spec["seats"], "k": spec["k"],
            "sources": {"rows": spec["rows"], "solo": spec["solo"], "judge": JUDGE[0]},
            "table_all_debates": table, "paired": res, "cross_checks": cross_checks(name, spec, table, res)}


# ---------------------------------------------------------------------------
# selftest: synthetic inputs with hand-computed answers, no file reads
# ---------------------------------------------------------------------------

def selftest() -> bool:
    ok = True
    ok &= maj([1, 1, 0]) == 1 and maj([1, 0]) is None and maj([None, 0]) == 0 and maj([]) is None
    ok &= code("ESH") == 1 and code("NAH") == 0 and code("NOVERDICT") is None
    # routing rule
    jm = {("a", "i1"): 1, ("a", "i2"): None}
    row = {"arm": "a", "item_id": "i1", "gold_verdict": "YTA", "verdict": "NTA", "synthesis_verdict": "NTA", "n_objectors": "2"}
    d = debate_from_row(row, jm)
    ok &= d["ok"] == 0 and d["fired"] == 1 and d["routed"] == 1            # fired, judge right -> routed right
    d = debate_from_row({**row, "item_id": "i2"}, jm)
    ok &= d["routed"] == 0                                                   # fired, judge tie -> S2
    d = debate_from_row({**row, "n_objectors": "1"}, jm)
    ok &= d["fired"] == 0 and d["routed"] == 0                              # not fired -> S2
    ok &= debate_from_row({**row, "synthesis_verdict": "NOVERDICT"}, jm) is None
    # bootstrap: degenerate clusters give a degenerate interval
    p, lo, hi = boot_ratio([(1, 1)] * 5, 200, 1)
    ok &= p == 1.0 and lo == 1.0 and hi == 1.0
    # paired deltas, hand computed
    solo = {("a", "i1"): 1, ("a", "i2"): 0, ("a", "i3"): 1}
    deb = [{"arm": "a", "item": "i1", "gold": 1, "ok": 1, "fired": 0, "routed": 1},
           {"arm": "a", "item": "i1", "gold": 1, "ok": 0, "fired": 1, "routed": 1},
           {"arm": "a", "item": "i2", "gold": 0, "ok": 1, "fired": 0, "routed": 1},
           {"arm": "a", "item": "i3", "gold": 1, "ok": 0, "fired": 0, "routed": 0},
           {"arm": "a", "item": "i4", "gold": 1, "ok": 1, "fired": 0, "routed": 1}]   # no solo: dropped
    r = paired_deltas(deb, solo, 500, 3)
    ok &= r["n_paired_debates"] == 4 and r["n_paired_cells"] == 3 and r["n_debates_without_solo"] == 1
    ok &= abs(r["collective_minus_solo_debate_weighted"]["delta"] - (-0.25)) < 1e-9     # (0 -1 +1 -1)/4
    ok &= abs(r["routed_minus_solo_debate_weighted"]["delta"] - 0.0) < 1e-9             # (0  0 +1 -1)/4
    ok &= abs(r["collective_minus_solo_cell_weighted"]["delta"] - (-0.5 / 3)) < 6e-5    # (-0.5 +1 -1)/3, 4-decimal rounding
    ok &= abs(r["routed_minus_solo_cell_weighted"]["delta"] - 0.0) < 1e-9               # (0 +1 -1)/3
    for k in ("collective_minus_solo_debate_weighted", "routed_minus_solo_cell_weighted"):
        ok &= r[k]["lo"] <= r[k]["delta"] <= r[k]["hi"]
    ok &= r["solo_acc_paired_cells"] == round(2 / 3, 4) and r["n_fired_paired"] == 1
    return bool(ok)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    if a.selftest:
        ok = selftest()
        print("selftest", "OK" if ok else "FAILED")
        return 0 if ok else 1
    out = {"note": "Collective and routed collective against the same model's solo standard majority-of-3, paired per "
                   "(framing, item) cell, item-clustered percentile bootstrap. Zero model calls; every number replayed "
                   "from cached CSV rows. See the module docstring of scripts/analyze_collective_vs_solo.py.",
           "estimator": {"kind": "paired item-clustered percentile bootstrap, ratio of per-item sums (verify_pillar3_headline.boot_delta)",
                         "draws": a.draws, "seed": a.seed, "replicate_seed": SEED_REPLICATE,
                         "judge": f"{JUDGE[1]} standard majority-of-3 from {JUDGE[0]}", "framings": list(ARMS),
                         "routed_rule": "per debate: S2 unless n_objectors >= 2, then the judge majority (tie -> S2)",
                         "solo_rule": "standard-CoT majority of three cached samples per (framing, item); tie -> cell dropped"},
           "conditions": {}}
    for name in a.conditions:
        res = analyze(name, a.draws, a.seed)
        out["conditions"][name] = res
        t, p = res["table_all_debates"], res["paired"]
        print(f"== {name}: {res['label']}")
        print(f"   all debates n {t['n_debates']} (dropped {t['n_uncodable_dropped']}): collective {t['collective']:.4f} "
              f"routed {t['routed']:.4f} fire {t['fire']:.3f}; solo codable cells {t['n_solo_cells_codable']} "
              f"acc {t['solo_acc_all_codable_cells']:.4f}")
        print(f"   paired: {p['n_paired_debates']} debates, {p['n_paired_cells']} cells, {p['n_items']} items, "
              f"{p['n_debates_without_solo']} debates without a codable solo; solo on paired cells {p['solo_acc_paired_cells']:.4f}")
        for k in ("collective_minus_solo_debate_weighted", "routed_minus_solo_debate_weighted",
                  "collective_minus_solo_cell_weighted", "routed_minus_solo_cell_weighted"):
            e = p[k]; r13 = p["replicate_seed_13"][k]
            print(f"   {k:40s} {e['points']:+6.1f} [{e['points_lo']:+6.1f}, {e['points_hi']:+6.1f}]"
                  f"  {'excl 0' if e['ci_excludes_zero'] else 'incl 0'}   (seed 13: [{r13['points_lo']:+6.1f}, {r13['points_hi']:+6.1f}])")
        for ck, cv in res["cross_checks"].items():
            flags = {kk: vv for kk, vv in cv.items() if kk.endswith("_agree")}
            print(f"   cross-check {ck}: {flags}")
    a.out.write_text(json.dumps(out, indent=1))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
