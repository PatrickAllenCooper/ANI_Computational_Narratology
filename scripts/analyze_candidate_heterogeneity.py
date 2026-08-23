"""
scripts/analyze_candidate_heterogeneity.py -- pre-spend gates for an
enumerate-and-select (best-of-k) design on Crowd-Gold AITA.

WHY THIS EXISTS
---------------
The programme's stated theory specifies an argmin over ENUMERATED candidate
responses.  No runner in this repo enumerates.  Before any selector is funded,
two questions are answerable for $0 from candidates already on disk:

  GATE 1 -- CANDIDATE HETEROGENEITY.  What share of items have k candidates
  containing BOTH an at-fault and a not-at-fault verdict?  A selector over a
  homogeneous candidate set has nothing to select.  Reported under all three
  ESH codings, because a large part of the apparent heterogeneity is
  disagreement about ESH alone -- a label the Scruples crowd casts on 1.05%
  of its votes.

  GATE 2 -- ACHIEVABLE ENVELOPE.  Across every implementable fixed selection
  rule (including the two extremes, always-pick-lenient and always-pick-harsh,
  which no real rule can exceed), how far can the stance-induced CRITERION
  SHIFT actually move?  If that whole span is narrower than the item-clustered
  bootstrap CI on a single rule's shift, the experiment cannot resolve its own
  primary DV and the DV must be changed before spending.

Both gates are computed from the cached generations in
divergence_study_outputs/cg_gen_*.json, using the same extract_verdict and the
same three ESH codings as analyze_crowdgold_sdt.py.  Fidelity is asserted
against the 6,723 paid-for rows in crowdgold_aita_rows.csv.

Usage:
  python -m scripts.analyze_candidate_heterogeneity --selftest
  python -m scripts.analyze_candidate_heterogeneity            # both gates
  python -m scripts.analyze_candidate_heterogeneity --gate 1
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from scripts.analyze_crowdgold_sdt import ESH_CODINGS, code_response, sdt
from scripts.verdict_format import extract_verdict

OUT_DIR = Path(__file__).resolve().parent.parent / "divergence_study_outputs"
INSTRUMENT = "aita"
VENDORS = ("claude-haiku-4-5", "gpt-5.4-nano", "grok-4-1-fast-reasoning")

#: Pre-registered gate from the theory-grounding doc, Stage 4.  Stated on the
#: MOST CONSERVATIVE coding, not the most favourable one.
GATE1_MIN_MIXED = 0.25


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_candidates(out_dir: Path = OUT_DIR) -> tuple[dict, dict]:
    """(model, scaffold, arm) -> item_id -> {sample_idx: verdict}, plus gold."""
    cells: dict = collections.defaultdict(lambda: collections.defaultdict(dict))
    gold: dict = {}
    for path in out_dir.glob("cg_gen_*.json"):
        rec = json.loads(path.read_text())
        key = (rec["model"], rec.get("scaffold", "standard"), rec["arm"])
        cells[key][rec["item_id"]][int(rec["sample_idx"])] = extract_verdict(
            rec.get("output", ""), INSTRUMENT)
        gold[rec["item_id"]] = rec.get("gold_verdict")
    return cells, gold


def assert_extraction_fidelity(cells: dict, out_dir: Path = OUT_DIR) -> int:
    """Our re-extraction must reproduce every paid-for row, or the gate is junk."""
    rows_path = out_dir / "crowdgold_aita_rows.csv"
    if not rows_path.exists():
        return 0
    n = 0
    with rows_path.open() as fh:
        for row in csv.DictReader(fh):
            key = (row["model"], "standard", row["arm"])
            got = cells[key][row["item_id"]][int(row["sample_idx"])]
            assert got == row["verdict"], (
                f"extraction drift on {row['item_id']}#{row['sample_idx']}: "
                f"cache gives {got!r}, paid row says {row['verdict']!r}")
            n += 1
    return n


# ---------------------------------------------------------------------------
# Candidate pools -- the SOURCE of candidates is itself the manipulation
# ---------------------------------------------------------------------------

def pool_same_model(cells, arm, model="gpt-5.4-nano", scaffold="standard", k=3):
    """k independent samples of ONE agent.  Aggregation's own candidate source."""
    d = cells[(model, scaffold, arm)]
    return {it: [v[i] for i in sorted(v)[:k]] for it, v in d.items() if len(v) >= k}


def pool_cross_vendor(cells, arm, per_model=1, models=VENDORS):
    """per_model samples from EACH vendor.  The only source-diversity
    manipulation available on disk; a proxy for, not an instance of, role
    embodiment."""
    out = collections.defaultdict(list)
    for m in models:
        for it, v in cells[(m, "standard", arm)].items():
            out[it] += [v[i] for i in sorted(v)[:per_model]]
    want = per_model * len(models)
    return {it: c for it, c in out.items() if len(c) == want}


def pool_cross_scaffold(cells, arm, scaffolds, model="gpt-5.4-nano"):
    """One sample from each of several DISTINCT prompt scaffolds."""
    out = collections.defaultdict(list)
    for sc in scaffolds:
        key = (model, sc, arm)
        if key not in cells:
            return {}
        for it, v in cells[key].items():
            if 0 in v:
                out[it].append(v[0])
    return {it: c for it, c in out.items() if len(c) == len(scaffolds)}


# ---------------------------------------------------------------------------
# Gate 1 -- candidate heterogeneity
# ---------------------------------------------------------------------------

def is_mixed(cands: Sequence[str], coding: str) -> bool:
    return {code_response(c, coding) for c in cands} - {None} == {0, 1}


def mixed_fraction(pool: dict, coding: str) -> tuple[int, int]:
    num = sum(1 for c in pool.values() if is_mixed(c, coding))
    return num, len(pool)


def esh_only_share(pool: dict) -> float:
    """Of the items mixed under the PUBLISHED coding, what share stop being
    mixed once ESH is dropped?  That is heterogeneity about a label the crowd
    itself casts on ~1% of votes."""
    pub = [c for c in pool.values() if is_mixed(c, "published")]
    if not pub:
        return float("nan")
    return sum(1 for c in pub if not is_mixed(c, "dropped")) / len(pub)


# ---------------------------------------------------------------------------
# Gate 2 -- achievable envelope of the criterion DV
# ---------------------------------------------------------------------------

def select(cands: Sequence[str], rule: str, coding: str, rng: random.Random) -> str:
    codes = [code_response(c, coding) for c in cands]
    live = [(c, z) for c, z in zip(cands, codes) if z is not None]
    if rule == "first":
        return cands[0]
    if rule == "random":
        return rng.choice(list(cands))
    if rule == "lenient":                       # extreme of the envelope
        hit = [c for c, z in live if z == 0]
        return hit[0] if hit else (live[0][0] if live else cands[0])
    if rule == "harsh":                         # other extreme
        hit = [c for c, z in live if z == 1]
        return hit[0] if hit else (live[0][0] if live else cands[0])
    if rule == "majority":
        if not live:
            return cands[0]
        ones = sum(1 for _, z in live if z == 1)
        zeros = len(live) - ones
        if ones == zeros:
            return rng.choice([c for c, _ in live])
        return "YTA" if ones > zeros else "NTA"
    raise ValueError(f"unknown rule {rule!r}")


def criterion_shift(pool_neutral, pool_stance, gold, rule, coding, items, rng):
    def cell(pool):
        rows = [{"item_id": it, "gold_verdict": gold[it],
                 "verdict": select(pool[it], rule, coding, rng)} for it in items]
        return sdt(rows, coding)
    a, b = cell(pool_neutral), cell(pool_stance)
    if a is None or b is None:
        return None
    return b["criterion"] - a["criterion"]


def envelope(pool_neutral, pool_stance, gold, coding, *, n_boot=800, seed=5):
    """Span of Delta-c across every fixed rule, vs the bootstrap CI on one rule."""
    items = sorted(set(pool_neutral) & set(pool_stance))
    rules = ("first", "random", "majority", "lenient", "harsh")
    pts = {r: criterion_shift(pool_neutral, pool_stance, gold, r, coding,
                              items, random.Random(3)) for r in rules}
    pts = {r: v for r, v in pts.items() if v is not None}
    if not pts:
        return None
    rng = random.Random(seed)
    draws = []
    for b in range(n_boot):
        samp = [rng.choice(items) for _ in items]
        v = criterion_shift(pool_neutral, pool_stance, gold, "majority", coding,
                            samp, random.Random(500 + b))
        if v is not None:
            draws.append(v)
    draws.sort()
    lo, hi = draws[int(.025 * len(draws))], draws[int(.975 * len(draws))]
    return {"points": pts, "span": max(pts.values()) - min(pts.values()),
            "ci": (lo, hi), "ci_width": hi - lo, "n_items": len(items)}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_gate1(cells, gold) -> None:
    print("=" * 78)
    print("GATE 1 -- CANDIDATE HETEROGENEITY (share of items whose k candidates")
    print(f"          contain BOTH an at-fault and a not-at-fault verdict).")
    print(f"          Pre-registered pass threshold: {GATE1_MIN_MIXED:.0%}")
    print("=" * 78)
    specs = [
        ("same-model nano k=2", lambda a: pool_same_model(cells, a, k=2)),
        ("same-model nano k=3", lambda a: pool_same_model(cells, a, k=3)),
        ("same-model nano k=5", lambda a: pool_same_model(cells, a, k=5)),
        ("same-model nano k=9", lambda a: pool_same_model(cells, a, k=9)),
        ("same-model haiku k=3",
         lambda a: pool_same_model(cells, a, model="claude-haiku-4-5", k=3)),
        ("same-model grok k=3",
         lambda a: pool_same_model(cells, a, model="grok-4-1-fast-reasoning", k=3)),
        ("cross-VENDOR k=3 (1/model)", lambda a: pool_cross_vendor(cells, a, 1)),
        ("cross-VENDOR k=9 (3/model)", lambda a: pool_cross_vendor(cells, a, 3)),
        ("cross-SCAFFOLD k=3 (std/NoT/verbose)",
         lambda a: pool_cross_scaffold(cells, a,
                   ["standard_cot", "narrative_cot", "standard_cot_verbose"])),
        ("cross-SCAFFOLD k=5 (five drop_* arms)",
         lambda a: pool_cross_scaffold(cells, a,
                   ["drop_protagonist", "drop_stakeholders", "drop_consequences",
                    "drop_uncertainty", "drop_commitment"])),
    ]
    hdr = f"{'candidate source':<38}{'arm':<17}{'n':>5}"
    hdr += "".join(f"{c[:9]:>11}" for c in ESH_CODINGS) + f"{'ESH-only':>10}"
    print(hdr)
    for label, mk in specs:
        for arm in ("third_person", "as_asker_stance"):
            pool = mk(arm)
            if not pool:
                continue
            line = f"{label:<38}{arm:<17}{len(pool):>5}"
            for coding in ESH_CODINGS:
                num, den = mixed_fraction(pool, coding)
                flag = "*" if num / den >= GATE1_MIN_MIXED else " "
                line += f"{num/den:>10.3f}{flag}"
            line += f"{esh_only_share(pool):>10.1%}"
            print(line)
    print("\n  * = clears the pre-registered 25% threshold on that coding.")
    print("  ESH-only = share of the PUBLISHED-coding mixed items that stop being")
    print("  mixed once ESH is dropped, i.e. candidate disagreement that is only")
    print("  about a label the Scruples crowd casts on 1.05% of 56,901 votes.")


def report_gate2(cells, gold) -> None:
    print()
    print("=" * 78)
    print("GATE 2 -- ACHIEVABLE ENVELOPE OF THE CRITERION DV")
    print("          span across ALL fixed rules vs the bootstrap CI on ONE rule")
    print("=" * 78)
    sources = [
        ("same-model nano k=9", lambda a: pool_same_model(cells, a, k=9)),
        ("cross-VENDOR k=9", lambda a: pool_cross_vendor(cells, a, 3)),
        ("cross-VENDOR k=3", lambda a: pool_cross_vendor(cells, a, 1)),
    ]
    for label, mk in sources:
        A, B = mk("third_person"), mk("as_asker_stance")
        print(f"\n  {label}   n={len(set(A) & set(B))}")
        for coding in ESH_CODINGS:
            e = envelope(A, B, gold, coding)
            if e is None:
                continue
            pts = "  ".join(f"{r}={v:+.3f}" for r, v in e["points"].items())
            verdict = ("SPAN < CI -- UNRESOLVABLE"
                       if e["span"] < e["ci_width"] else "span > CI")
            print(f"    {coding:<13} {pts}")
            print(f"    {'':<13} span={e['span']:.3f}  CI width={e['ci_width']:.3f}"
                  f"   -> {verdict}")


def _selftest() -> int:
    fails = []

    def check(name, cond):
        print(f"  [{'ok ' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    check("is_mixed: YTA+NTA is mixed under all codings",
          all(is_mixed(["YTA", "NTA"], c) for c in ESH_CODINGS))
    check("is_mixed: YTA+ESH mixed only when ESH is not at-fault",
          not is_mixed(["YTA", "ESH"], "published")
          and not is_mixed(["YTA", "ESH"], "dropped")
          and is_mixed(["YTA", "ESH"], "not_at_fault"))
    check("is_mixed: NTA+ESH mixed only under the published coding",
          is_mixed(["NTA", "ESH"], "published")
          and not is_mixed(["NTA", "ESH"], "dropped")
          and not is_mixed(["NTA", "ESH"], "not_at_fault"))
    check("is_mixed: non-committal alone is never mixed",
          not any(is_mixed(["UNRESOLVED", "NOVERDICT"], c) for c in ESH_CODINGS))
    rng = random.Random(0)
    check("select lenient picks a not-at-fault candidate",
          select(["YTA", "NTA"], "lenient", "published", rng) == "NTA")
    check("select harsh picks an at-fault candidate",
          select(["NTA", "YTA"], "harsh", "published", rng) == "YTA")
    check("select first is the deployed default",
          select(["NTA", "YTA", "YTA"], "first", "published", rng) == "NTA")
    check("select majority follows the coded majority",
          select(["NTA", "YTA", "YTA"], "majority", "published", rng) == "YTA")
    check("select extremes fall back when no candidate qualifies",
          select(["NTA", "NTA"], "harsh", "published", rng) == "NTA")
    check("esh_only_share is 1.0 when every mixed pair is YTA-free ESH",
          abs(esh_only_share({"a": ["NTA", "ESH"], "b": ["NTA", "ESH"]}) - 1.0) < 1e-9)
    check("esh_only_share is 0.0 when mixing survives dropping ESH",
          abs(esh_only_share({"a": ["NTA", "YTA"]}) - 0.0) < 1e-9)

    if OUT_DIR.exists() and any(OUT_DIR.glob("cg_gen_*.json")):
        cells, _ = load_candidates()
        n = assert_extraction_fidelity(cells)
        check(f"re-extraction reproduces all {n} paid-for rows", n > 6000)
        p3 = pool_same_model(cells, "third_person", k=3)
        p9 = pool_same_model(cells, "third_person", k=9)
        check("k=9 pool is no more homogeneous than k=3 (monotone in k)",
              mixed_fraction(p9, "published")[0] >= mixed_fraction(p3, "published")[0])
        cv = pool_cross_vendor(cells, "third_person", 1)
        check("cross-vendor k=3 has 249 items", len(cv) == 249)
    else:
        print("  [skip] cache-dependent checks (no cg_gen_*.json on disk)")

    print(f"\n{'FAILED: ' + ', '.join(fails) if fails else 'all checks passed'}")
    return 1 if fails else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gate", type=int, choices=(1, 2), default=None)
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    cells, gold = load_candidates()
    n = assert_extraction_fidelity(cells)
    print(f"loaded candidates from {OUT_DIR}; extraction verified against "
          f"{n} paid-for rows (100% agreement)\n")
    if args.gate in (None, 1):
        report_gate1(cells, gold)
    if args.gate in (None, 2):
        report_gate2(cells, gold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
