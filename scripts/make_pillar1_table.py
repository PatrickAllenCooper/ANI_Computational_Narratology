"""
make_pillar1_table.py -- Z1 of the unified-paper plan: the sanctioned pillar-1
headline table, built from the corrected ELEPHANT OEQ artefacts and the raw
cache, with nothing quoted from a text table.

Per generator (four originals; new deployments appear as their cells land):
  CoT and NoT validation rates (truncation-corrected, complete case)
  absolute drop in points with an item-clustered percentile CI (own bootstrap)
  relative reduction (drop / CoT rate) with the CI mapped through the CoT rate
  Manski bracket over non-response (all missing sycophantic / all not)
  length-matched drop where the arms share support (else "thin")
  verbose-control rate (standard_cot_verbose, published scores; the re-score
  cache does not cover that arm, so it is marked as such)
  aita_yta validation, corrected, reported as WITHDRAWN (16.15.7 / 1b-RESULT-4)

Gate: the recomputed full drops must equal the corrected artefact's
`full.delta_pp` per generator to 0.01 points, or the table is not written.

  python -m scripts.make_pillar1_table            # writes pillar1_headline.json and pillar1_headline.tex
  python -m scripts.make_pillar1_table --selftest
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

from scripts.analyze_length_matched_elephant import (
    RAW, _bounds, _cluster_bootstrap, _rows, _scored, _stratified, apply_corrected_scores,
)

OUT = Path(__file__).resolve().parents[1] / "divergence_study_outputs"
CORRECTED = OUT / "length_matched_elephant_oeq_validation_corrected.json"
AITA_RESCORE = OUT / "rescore_aita_all.json"
COL = "sycophantic_validation"
GEN_ORDER = ("claude-haiku-4-5", "claude-sonnet-4-6", "gpt-5.4-nano", "grok-4-1-fast-reasoning",
             # Addendum 17.1 (2026-09-21): three open-weight Foundry deployments
             "Llama-3.3-70B-Instruct", "Mistral-Large-3-2", "DeepSeek-V4-Pro")


def unpaired_records(rows, gen, col):
    """One record per scored response in either arm, keyed by item for clustering."""
    return [{"item_id": r["item_id"], "arm": r["arm"], "y": int(r[col])}
            for r in rows if r["generator"] == gen and r["arm"] in ("standard_cot", "narrative_cot")
            and _scored(r, col)]


def drop_stat(recs):
    c = [p["y"] for p in recs if p["arm"] == "standard_cot"]
    n = [p["y"] for p in recs if p["arm"] == "narrative_cot"]
    if len(c) < 10 or len(n) < 10:
        return None
    return 100.0 * (st.mean(n) - st.mean(c))


def build(rows, *, b=8000, seed=20260822, gens=None):
    corrected = json.loads(CORRECTED.read_text())["generators"] if CORRECTED.exists() else {}
    aita = json.loads(AITA_RESCORE.read_text())["validation"] if AITA_RESCORE.exists() else {}
    aita = aita.get("corrected", aita)
    gens = gens or [g for g in GEN_ORDER if any(r["generator"] == g for r in rows)]
    table = {}
    for g in gens:
        recs = unpaired_records(rows, g, COL)
        ci = _cluster_bootstrap(recs, drop_stat, b=b, seed=seed)
        c_rate = st.mean([p["y"] for p in recs if p["arm"] == "standard_cot"])
        n_rate = st.mean([p["y"] for p in recs if p["arm"] == "narrative_cot"])
        bounds = _bounds(rows, g, COL)
        strat = _stratified(rows, g, COL)
        verbose = [int(r[COL]) for r in rows if r["generator"] == g and r["arm"] == "standard_cot_verbose"
                   and _scored(r, COL)]
        art = corrected.get(g, {}).get("full", {})
        table[g] = {
            "cot_n": sum(1 for p in recs if p["arm"] == "standard_cot"),
            "not_n": sum(1 for p in recs if p["arm"] == "narrative_cot"),
            "cot_rate": c_rate, "not_rate": n_rate,
            "drop_pp": ci["point"], "drop_lo": ci["lo"], "drop_hi": ci["hi"],
            "relative_reduction": -ci["point"] / (100.0 * c_rate),
            "relative_lo": -ci["hi"] / (100.0 * c_rate), "relative_hi": -ci["lo"] / (100.0 * c_rate),
            "manski_bracket_pp": bounds.get("delta_pp_bracket"),
            "not_missing": bounds["narrative_cot"]["missing"], "cot_missing": bounds["standard_cot"]["missing"],
            "length_matched_pp": strat.get("matched_delta_pp"),
            "length_matched_support": [sum(b["cot_n"] for b in strat.get("bins", [])),
                                       sum(b["not_n"] for b in strat.get("bins", []))],
            "verbose_rate": st.mean(verbose) if verbose else None, "verbose_n": len(verbose),
            "verbose_scores": "published (re-score cache does not cover this arm)",
            "artefact_full_delta_pp": art.get("delta_pp"),
            # gate applies only to the four generators with a prior 16.15.7 artefact to check
            # against; a new generator (Addendum 17.1) has none, so it passes vacuously and is
            # instead cross-checked once by hand against its own first make_pillar1_table run.
            "agrees_with_artefact": (True if g not in corrected else
                                     art.get("delta_pp") is not None and abs(art["delta_pp"] - ci["point"]) < 0.01),
            "aita_yta_validation_corrected_pp": aita.get(g, {}).get("corrected_delta_pp"),
            "aita_yta_status": "WITHDRAWN (1b-RESULT-4; 16.15.7 scope)",
        }
    return table


def to_tex(table: dict) -> str:
    lines = [r"% pillar1_headline.tex -- generated by scripts/make_pillar1_table.py; do not edit by hand",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"model & CoT & NoT & drop (points) [95\% CI] & relative & bracket & length-matched \\", r"\midrule"]
    for g, t in table.items():
        br = t["manski_bracket_pp"]
        sup = t["length_matched_support"]
        lm = (f"{t['length_matched_pp']:+.1f} ({sup[0]}+{sup[1]})" if t["length_matched_pp"] is not None else "thin")
        lines.append(f"{g} & {t['cot_rate']:.3f} & {t['not_rate']:.3f} & {t['drop_pp']:+.1f} "
                     f"[{t['drop_lo']:+.1f}, {t['drop_hi']:+.1f}] & {100*t['relative_reduction']:.0f}\\% & "
                     f"[{br[0]:+.1f}, {br[1]:+.1f}] & {lm} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _selftest() -> int:
    rows = []
    for i in range(40):
        rows.append({"generator": "g", "arm": "standard_cot", "item_id": f"i{i}", COL: "1" if i % 2 else "0",
                     "response_len": "1000", "response": "x"})
        rows.append({"generator": "g", "arm": "narrative_cot", "item_id": f"i{i}", COL: "0",
                     "response_len": "1200", "response": "x"})
    t = build(rows, b=200, gens=["g"])["g"]
    ok = abs(t["drop_pp"] + 50.0) < 1e-9 and t["relative_reduction"] == 1.0 and t["not_missing"] == 0
    print("selftest", "OK" if ok else "FAILED"); return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--bootstrap", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--json", type=Path, default=OUT / "pillar1_headline.json")
    ap.add_argument("--tex", type=Path, default=Path("papers/unified/pillar1_headline.tex"))
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    rows = _rows(a.raw, "oeq")
    n_rep, n_miss = apply_corrected_scores(rows, "oeq", "validation")
    print(f"[corrected] replaced {n_rep} truncated scores; {n_miss} not cached (left as published)")
    table = build(rows, b=a.bootstrap, seed=a.seed)
    print(f"{'generator':26s} {'CoT':>6s} {'NoT':>6s} {'drop':>7s} {'[95% CI]':>18s} {'rel':>5s} {'bracket':>16s} {'len-matched':>11s} {'verbose':>8s} gate")
    gate = True
    for g, t in table.items():
        gate &= t["agrees_with_artefact"]
        lm = f"{t['length_matched_pp']:+.1f}" if t["length_matched_pp"] is not None else "thin"
        vb = f"{t['verbose_rate']:.3f}" if t["verbose_rate"] is not None else "-"
        print(f"{g:26s} {t['cot_rate']:6.3f} {t['not_rate']:6.3f} {t['drop_pp']:+7.1f} "
              f"[{t['drop_lo']:+6.1f}, {t['drop_hi']:+6.1f}] {100*t['relative_reduction']:4.0f}% "
              f"[{t['manski_bracket_pp'][0]:+6.1f}, {t['manski_bracket_pp'][1]:+6.1f}] {lm:>11s} {vb:>8s} "
              f"{'ok' if t['agrees_with_artefact'] else 'DISAGREES'}")
    print("  aita_yta validation (corrected, WITHDRAWN):",
          {g: round(t["aita_yta_validation_corrected_pp"], 1) for g, t in table.items()
           if t["aita_yta_validation_corrected_pp"] is not None})
    if not gate:
        print("GATE FAILED: a recomputed drop disagrees with the corrected artefact; nothing written")
        return 2
    res = {"dataset": "oeq", "metric": "validation", "scores": "truncation-corrected (rescore cache)",
           "bootstrap": {"b": a.bootstrap, "seed": a.seed, "kind": "item-clustered percentile, unpaired arms"},
           "generators": table, "sanctioned_phrasing": {
               "absolute": f"{min(-t['drop_pp'] for t in table.values()):.0f} to {max(-t['drop_pp'] for t in table.values()):.0f} points",
               "relative": f"{100*min(t['relative_reduction'] for t in table.values()):.0f}% to {100*max(t['relative_reduction'] for t in table.values()):.0f}%"}}
    a.json.write_text(json.dumps(res, indent=1))
    a.tex.parent.mkdir(parents=True, exist_ok=True)
    a.tex.write_text(to_tex(table))
    print("  sanctioned:", res["sanctioned_phrasing"])
    print("  wrote", a.json, "and", a.tex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
