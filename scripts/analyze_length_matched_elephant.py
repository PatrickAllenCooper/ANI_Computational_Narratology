"""
scripts/analyze_length_matched_elephant.py -- the length-matched control the
sycophancy paper never ran, computed offline from the existing cache.

WHY THIS EXISTS
---------------
The flagship claim (NoT cuts ELEPHANT validation 26-53 pp on 4/4 models) is
measured across a 2-4x response-length gap, against a documented LLM-judge
length bias, with NO length-matched comparator. `PROMPTS["standard_cot_verbose"]`
exists and was calibrated per generator -- but it appears in no ELEPHANT cell;
it was spent on the ACL paper's depth metrics instead. Meanwhile
`Guidance_Documents/sycophancy_scenario_redesign.md` Section 6 lists
"length-matched sycophancy scoring as standard practice" as a NOVELTY CLAIM of
this program.

This module runs the zero-API-cost half of that control: a length-stratified
reanalysis of the 27,600 already-scored rows in
`divergence_study_outputs/elephant_singleagent_raw.csv`. It does not replace
running the verbose arm -- it tells you whether that run is even necessary, and
it is free.

THE TRUNCATION INTERACTION (why the <4000 cut is the important one)
-------------------------------------------------------------------
`scripts/elephant_scorers.py` passes `advice[:4000]` to every judge call. On the
OEQ cell that truncates 75-93% of NoT responses against 0-59% of CoT, so the two
arms are not scored on the same object (see redesign doc Section 1b). Restricting to
`response_len < 4000` therefore does two jobs at once: it removes the truncation
confound (both arms complete) AND shrinks the length gap. Where that subsample
is large enough, it is the cleanest estimate available without spending a cent.

MISSING DATA
------------
`sycophantic_*` uses -1 as a sentinel for a non-response, and it is NOT
missing-at-random: on gpt-5.4-nano the NoT cell is 34/150 = 22.7% empty vs 5/150
for CoT, and the dropped items are disproportionately the high-validation ones.
Complete-case rates are reported alongside per-arm non-response counts, and
`--bounds` reports the Manski-style worst/best-case bracket. Never read the
point estimate for a cell without its non-response rate.

USAGE
    python -m scripts.analyze_length_matched_elephant
    python -m scripts.analyze_length_matched_elephant --dataset aita_yta --metric framing
    python -m scripts.analyze_length_matched_elephant --bounds
    python -m scripts.analyze_length_matched_elephant --json out.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path
from typing import Any, Optional, Sequence

RAW = Path("divergence_study_outputs/elephant_singleagent_raw.csv")
TRUNCATION_LIMIT = 4000  # scripts/elephant_scorers.py: advice[:4000]
METRICS = ("validation", "indirectness", "framing", "moral")


def _rows(path: Path, dataset: str) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"missing {path}; run the ELEPHANT pipeline first")
    with path.open() as fh:
        rows = [r for r in csv.DictReader(fh) if r["dataset"] == dataset]
    if not rows:
        raise SystemExit(f"no rows for dataset={dataset!r} in {path}")
    return rows


def _scored(r: dict[str, str], col: str) -> bool:
    return r.get(col) in ("0", "1")


def _mean(xs: Sequence[float]) -> Optional[float]:
    return st.mean(xs) if xs else None


def _cell(rows, gen, arm, col, *, max_len=None):
    out = []
    for r in rows:
        if r["generator"] != gen or r["arm"] != arm:
            continue
        if max_len is not None and float(r["response_len"]) >= max_len:
            continue
        if _scored(r, col):
            out.append(r)
    return out


def _delta(rows, gen, col, *, max_len=None, min_n=10):
    c = _cell(rows, gen, "standard_cot", col, max_len=max_len)
    n = _cell(rows, gen, "narrative_cot", col, max_len=max_len)
    if len(c) < min_n or len(n) < min_n:
        return {"cot_n": len(c), "not_n": len(n), "delta_pp": None, "thin": True}
    cv = _mean([int(r[col]) for r in c])
    nv = _mean([int(r[col]) for r in n])
    return {
        "cot_n": len(c), "cot_rate": cv,
        "not_n": len(n), "not_rate": nv,
        "delta_pp": 100.0 * (nv - cv), "thin": False,
    }


def _stratified(rows, gen, col, *, bin_width=1000, min_per_bin=5):
    """Length-matched delta: bin the common support, weight bins by cell count."""
    c = _cell(rows, gen, "standard_cot", col)
    n = _cell(rows, gen, "narrative_cot", col)
    if not c or not n:
        return {"bins": [], "matched_delta_pp": None, "reason": "empty arm"}
    lens_c = [float(r["response_len"]) for r in c]
    lens_n = [float(r["response_len"]) for r in n]
    lo, hi = max(min(lens_c), min(lens_n)), min(max(lens_c), max(lens_n))
    if hi <= lo:
        return {"bins": [], "matched_delta_pp": None, "reason": "no common support"}

    bins, wsum, wn, used_c, used_n = [], 0.0, 0, 0, 0
    edge = lo
    while edge < hi:
        cc = [int(r[col]) for r in c if edge <= float(r["response_len"]) < edge + bin_width]
        nn = [int(r[col]) for r in n if edge <= float(r["response_len"]) < edge + bin_width]
        if len(cc) >= min_per_bin and len(nn) >= min_per_bin:
            d = _mean(nn) - _mean(cc)
            w = len(cc) + len(nn)
            wsum += w * d
            wn += w
            used_c += len(cc)
            used_n += len(nn)
            bins.append({
                "lo": edge, "hi": edge + bin_width,
                "cot_n": len(cc), "cot_rate": _mean(cc),
                "not_n": len(nn), "not_rate": _mean(nn),
                "delta_pp": 100.0 * d,
            })
        edge += bin_width
    return {
        "support": [lo, hi], "bins": bins,
        "matched_delta_pp": (100.0 * wsum / wn) if wn else None,
        "coverage": {"cot_used": used_c, "cot_total": len(c),
                     "not_used": used_n, "not_total": len(n)},
        "reason": None if wn else f"no bin reached {min_per_bin} in both arms",
    }


def _nonresponse(rows, gen, arm, col):
    sub = [r for r in rows if r["generator"] == gen and r["arm"] == arm]
    missing = sum(1 for r in sub if not _scored(r, col))
    return {"n": len(sub), "missing": missing,
            "rate": (missing / len(sub)) if sub else None}


def _bounds(rows, gen, col):
    """Manski bracket: impute all-missing as sycophantic, then as non-sycophantic."""
    out = {}
    for arm in ("standard_cot", "narrative_cot"):
        sub = [r for r in rows if r["generator"] == gen and r["arm"] == arm]
        if not sub:
            continue
        obs = [int(r[col]) for r in sub if _scored(r, col)]
        miss = len(sub) - len(obs)
        out[arm] = {
            "complete_case": _mean(obs),
            "worst": (sum(obs) + miss) / len(sub),   # all missing were sycophantic
            "best": sum(obs) / len(sub),             # all missing were not
            "missing": miss, "n": len(sub),
        }
    if len(out) == 2:
        c, n = out["standard_cot"], out["narrative_cot"]
        # widest defensible bracket on the delta
        out["delta_pp_bracket"] = [100.0 * (n["best"] - c["worst"]),
                                   100.0 * (n["worst"] - c["best"])]
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--dataset", default="oeq")
    ap.add_argument("--metric", default="validation", choices=METRICS)
    ap.add_argument("--bin-width", type=int, default=1000)
    ap.add_argument("--min-per-bin", type=int, default=5)
    ap.add_argument("--bounds", action="store_true", help="Manski bracket for non-response")
    ap.add_argument("--json", type=Path, help="write the full result object here")
    a = ap.parse_args(argv)

    col = f"sycophantic_{a.metric}"
    rows = _rows(a.raw, a.dataset)
    gens = sorted({r["generator"] for r in rows
                   if any(x["generator"] == r["generator"] and x["arm"] == "narrative_cot"
                          for x in rows)})

    result: dict[str, Any] = {
        "dataset": a.dataset, "metric": a.metric,
        "truncation_limit": TRUNCATION_LIMIT, "generators": {},
    }

    print(f"dataset={a.dataset}  metric={a.metric}  "
          f"(judge truncation limit = {TRUNCATION_LIMIT} chars)")
    print("=" * 88)
    print(f"{'generator':<26}{'full delta':>12}{'<4000 delta':>14}{'len-matched':>14}"
          f"{'NoT missing':>13}")
    for g in gens:
        full = _delta(rows, g, col)
        clean = _delta(rows, g, col, max_len=TRUNCATION_LIMIT)
        strat = _stratified(rows, g, col, bin_width=a.bin_width,
                            min_per_bin=a.min_per_bin)
        nr = _nonresponse(rows, g, "narrative_cot", col)
        cell = {"full": full, "untruncated": clean, "stratified": strat,
                "nonresponse_not": nr,
                "nonresponse_cot": _nonresponse(rows, g, "standard_cot", col)}
        if a.bounds:
            cell["bounds"] = _bounds(rows, g, col)
        result["generators"][g] = cell

        def fmt(x):
            return f"{x:+.1f}" if isinstance(x, (int, float)) else "thin"
        print(f"{g:<26}{fmt(full.get('delta_pp')):>12}"
              f"{fmt(clean.get('delta_pp')):>14}"
              f"{fmt(strat.get('matched_delta_pp')):>14}"
              f"{100 * (nr['rate'] or 0):>12.1f}%")

    print()
    print("per-generator length strata")
    print("=" * 88)
    for g in gens:
        s = result["generators"][g]["stratified"]
        print(f"\n{g}")
        if not s["bins"]:
            print(f"   {s['reason']}")
            continue
        for b in s["bins"]:
            print(f"   {b['lo']:>6.0f}-{b['hi']:<6.0f} "
                  f"CoT n={b['cot_n']:<4} {b['cot_rate']:.3f}   "
                  f"NoT n={b['not_n']:<4} {b['not_rate']:.3f}   "
                  f"d={b['delta_pp']:+6.1f} pp")
        cov = s["coverage"]
        print(f"   => matched delta {s['matched_delta_pp']:+.1f} pp "
              f"(uses {cov['cot_used']}+{cov['not_used']} of "
              f"{cov['cot_total']}+{cov['not_total']})")

    if a.bounds:
        print()
        print("non-response bounds (Manski bracket on the delta)")
        print("=" * 88)
        for g in gens:
            b = result["generators"][g].get("bounds", {})
            if "delta_pp_bracket" not in b:
                continue
            lo, hi = b["delta_pp_bracket"]
            print(f"   {g:<26} complete-case "
                  f"{result['generators'][g]['full'].get('delta_pp', float('nan')):+.1f} pp"
                  f"   bracket [{lo:+.1f}, {hi:+.1f}] pp"
                  f"   (NoT missing {b['narrative_cot']['missing']}/"
                  f"{b['narrative_cot']['n']})")

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
