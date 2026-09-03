"""
scripts/analyze_sharpening_gate.py -- Addendum 11 Gate G1: did a Phase 1
sharpening arm improve the Phase-0-frozen sensor (`r4_nonunanimous`) over
plain few-shot, on the SAME 47-item nano panel?

Population: nano's plain-fewshot cache (`cg_deliberation_nano_fewshot_
{rows,votes}.csv`, Addendum 10) vs each Phase 1 sharpened cache
(`cg_deliberation_nano_fewshot_truthgated_*` for 1a,
`cg_deliberation_nano_fewshot_stakecot_*` for 1b), all built on the
identical item/arm panel. Sensor and actuator are FROZEN from Phase 0:
`r4_nonunanimous`, cross-vendor cached `grok-4-1-fast-reasoning:standard`
majority-of-3.

Primary outcome: a PAIRED, item-blocked bootstrap of the difference in the
actuator's flagged-set delta (sharpened minus plain), computed by resampling
item ids with replacement and, for each draw, taking BOTH conditions'
flagged-population mean actuator delta over the debates belonging to the
SAME resampled items -- the same "paired G3 delta" pattern this project
already uses to compare two conditions sharing an item panel (Addendum 8).

Gate G1: CI excludes zero on (sharpened flagged-delta - plain flagged-delta).

Usage
-----
  python -m scripts.analyze_sharpening_gate --selftest
  python -m scripts.analyze_sharpening_gate
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

from scripts.analyze_actuator_ladder import correct, load_comparator, load_debates, make_rule_vendor
from scripts.analyze_sensor_search import (
    CANDIDATES, COMPARATOR_PATH, attach_r3_reject_count,
)
from scripts.run_phase1_quartet import OUT_DIR

ANALYSIS_PATH = OUT_DIR / "sharpening_gate_analysis.json"
FROZEN_SENSOR = "r4_nonunanimous"
ACTUATOR = ("grok-4-1-fast-reasoning", "standard")
ALPHA = 0.05 / 2      # Bonferroni over the two Phase 1 arms tested
DEFAULT_DRAWS = 4000

ARMS = (
    ("1a_truthgated", "cg_deliberation_nano_fewshot_truthgated"),
    ("1b_fewshot_stakecot", "cg_deliberation_nano_fewshot_stakecot"),
)
PLAIN_TAG = "cg_deliberation_nano_fewshot"
MODEL = "gpt-5.4-nano"


def load(tag: str) -> list[dict]:
    rows_path = OUT_DIR / f"{tag}_rows.csv"
    votes_path = OUT_DIR / f"{tag}_votes.csv"
    debates = [d for d in load_debates(rows_path, votes_path)
              if d["model"] == MODEL]
    attach_r3_reject_count(debates, rows_path)
    return debates


def actuated_ok(d: dict, rule: Callable) -> bool:
    v = rule(d)
    ok = correct(v, d["gold"]) if v is not None else None
    return d["s2_ok"] if ok is None else ok


def paired_gate(plain: Sequence[dict], sharp: Sequence[dict], *,
               sig: Callable[[dict], bool], rule: Callable, alpha: float,
               draws: int, seed: int, label: str) -> dict:
    """Item-blocked bootstrap of (sharp flagged-delta - plain flagged-delta),
    resampling item ids shared by both conditions."""
    def per_debate(d):
        return int(actuated_ok(d, rule)) - int(d["s2_ok"])

    by_item_plain: dict[str, list[dict]] = defaultdict(list)
    for d in plain:
        by_item_plain[d["item"]].append(d)
    by_item_sharp: dict[str, list[dict]] = defaultdict(list)
    for d in sharp:
        by_item_sharp[d["item"]].append(d)
    ids = sorted(set(by_item_plain) & set(by_item_sharp))

    def flagged_delta_mean(items: Sequence[str], by_item: dict) -> float | None:
        vals = [per_debate(d) for i in items for d in by_item.get(i, [])
                if sig(d)]
        return sum(vals) / len(vals) if vals else None

    def diff(items: Sequence[str]) -> float | None:
        dp, ds = (flagged_delta_mean(items, by_item_plain),
                  flagged_delta_mean(items, by_item_sharp))
        return None if dp is None or ds is None else ds - dp

    point = diff(ids)
    rng = random.Random(seed)
    vals = sorted(v for v in (
        diff([ids[rng.randrange(len(ids))] for _ in ids])
        for _ in range(draws)) if v is not None)
    lo = vals[int((alpha / 2) * len(vals))] if vals else None
    hi = vals[min(int((1 - alpha / 2) * len(vals)), len(vals) - 1)] if vals else None
    n_plain_flagged = sum(1 for i in ids for d in by_item_plain[i] if sig(d))
    n_sharp_flagged = sum(1 for i in ids for d in by_item_sharp[i] if sig(d))
    return {
        "label": label, "n_items": len(ids),
        "n_flagged_plain": n_plain_flagged, "n_flagged_sharp": n_sharp_flagged,
        "point": point, "lo": lo, "hi": hi, "ci_level": 1 - alpha,
        "positive": bool(point is not None and lo is not None and lo > 0),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", type=Path, default=ANALYSIS_PATH)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()

    comp = load_comparator(COMPARATOR_PATH)
    rule = make_rule_vendor(comp, *ACTUATOR)
    sig = CANDIDATES[FROZEN_SENSOR]
    plain = load(PLAIN_TAG)
    out = []
    for label, tag in ARMS:
        sharp = load(tag)
        res = paired_gate(plain, sharp, sig=sig, rule=rule, alpha=ALPHA,
                          draws=a.draws, seed=a.seed, label=label)
        out.append(res)
        print(f"=== Gate G1: {label} vs plain few-shot "
              f"(sensor={FROZEN_SENSOR}, actuator={ACTUATOR}) ===")
        print(f"  n_items={res['n_items']}  n_flagged plain={res['n_flagged_plain']} "
              f"sharp={res['n_flagged_sharp']}")
        print(f"  paired delta (sharp - plain): {res['point']:+.3f} "
              f"[{res['lo']:+.3f}, {res['hi']:+.3f}] (CI {res['ci_level']:.3f})")
        print(f"  GATE G1: {'PASS' if res['positive'] else 'fail'}\n")

    verdict = "PASS -> Phase 2" if any(r["positive"] for r in out) else \
             "fail -> stop, report sensor ceiling"
    print(f"GATE G1 OVERALL VERDICT: {verdict}")
    a.out.write_text(json.dumps({"arms": out, "verdict": verdict}, indent=1))
    print(f"wrote {a.out}")
    return 0


def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    plain_ready = (OUT_DIR / f"{PLAIN_TAG}_rows.csv").exists()
    check("plain few-shot nano cache present", plain_ready)
    arm_ready = {label: (OUT_DIR / f"{tag}_rows.csv").exists()
                for label, tag in ARMS}
    for label, ok in arm_ready.items():
        check(f"{label} cache present", ok)
    check("comparator cache present", COMPARATOR_PATH.exists())

    if plain_ready and all(arm_ready.values()) and COMPARATOR_PATH.exists():
        comp = load_comparator(COMPARATOR_PATH)
        rule = make_rule_vendor(comp, *ACTUATOR)
        sig = CANDIDATES[FROZEN_SENSOR]
        plain = load(PLAIN_TAG)
        check("plain population nonempty", len(plain) > 0)
        for label, tag in ARMS:
            sharp = load(tag)
            check(f"{label} population nonempty", len(sharp) > 0)
            check(f"{label} shares item ids with plain (same panel)",
                  bool({d["item"] for d in plain} & {d["item"] for d in sharp}))
            r1 = paired_gate(plain, sharp, sig=sig, rule=rule, alpha=ALPHA,
                             draws=200, seed=1, label=label)
            r2 = paired_gate(plain, sharp, sig=sig, rule=rule, alpha=ALPHA,
                             draws=200, seed=1, label=label)
            check(f"{label} point estimate is deterministic given fixed inputs",
                  r1["point"] == r2["point"])
        # identity check: comparing plain against itself must give point 0
        r0 = paired_gate(plain, plain, sig=sig, rule=rule, alpha=ALPHA,
                         draws=200, seed=1, label="self")
        check("comparing a condition against itself gives a point delta of "
              "exactly zero", r0["point"] == 0.0)
    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
