"""
scripts/probe_neutral_baseline.py -- cheap neutral-arm-only baseline check
across candidate models for the redesigned BrokenMath ablation.

WHY THIS EXISTS
---------------
Before committing a full pilot to any one model for the section-ablation
redesign (Guidance_Documents/sycophancy_scenario_redesign.md Sec 6.4), we need
to know which of the candidate models actually retain real neutral-arm error
under THIS instrument (forced VERDICT: line, decidable-claim reframing,
standard_cot scaffold). The original per-model BrokenMath "sycophant rate"
(haiku 83%, sonnet 69%, nano 55%, grok 51%, from divergence_study_outputs/
brokenmath_summary.json) is NOT a reliable predictor here: it comes from a
different task framing (free-form proof-writing, judge-scored) and a different
scoring path. Confirmed the hard way: grok's own neutral error collapsed from
~51% under that original scoring to 3.9% under this instrument, on the SAME
unreframed task text -- see the redesign doc's 2026-08-20 revision. A model's
old rate does not tell us how it behaves under the instrument we would
actually run the ablation with.

HOW -- reuses tested machinery, spends nothing on stance arms
----------------------------------------------------------------
This does NOT reimplement prompt construction. It calls
run_stance_factorial.build_units() unchanged (so prompts, caching, and item
selection are byte-for-byte what the real ablation would use), then keeps only
the neutral-arm units (channel="neutral_filler", direction="none") before
execution -- the four stance cells are built (build_units requires
k_estimate>=1) but their units are simply never passed to execute_units, so
nothing is generated or spent on them. cell_estimate() -- the same estimator
the ablation itself uses -- computes p(TRUE) per model.

USAGE
    python -m scripts.probe_neutral_baseline --n 40 --k-select 5              # plan/cache only
    python -m scripts.probe_neutral_baseline --n 40 --k-select 5 --run --workers 8
    python -m scripts.probe_neutral_baseline --selftest                        # offline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from scripts.run_stance_factorial import (
    build_units,
    cell_estimate,
    execute_units,
    load_cached_records,
    load_items,
)

DEFAULT_MODELS = "gpt-5.4-nano,claude-haiku-4-5,claude-sonnet-4-6"


def plan(models: Sequence[str], *, n: int, k_select: int, seed: int,
         scaffold: str, max_tokens: int, temperature: float,
         allow_unresolved: bool):
    items = load_items("brokenmath", n, seed)
    all_units = build_units(
        items, models=list(models), scaffolds=[scaffold], instrument="brokenmath",
        k_select=k_select, k_estimate=1, paraphrases=1, dose=1,
        max_tokens=max_tokens, temperature=temperature, seed=seed,
        allow_unresolved=allow_unresolved, reframe=True,
    )
    neutral = [u for u in all_units
               if u.channel == "neutral_filler" and u.direction == "none"]
    return items, neutral, len(all_units) - len(neutral)


def report(models: Sequence[str], records: list[dict]) -> dict:
    print(f"\n{'model':<26}{'n':>5}{'p(TRUE)':>10}{'noverdict':>11}")
    out: dict[str, dict] = {}
    for m in models:
        recs = [r for r in records if r.get("model") == m]
        est = cell_estimate(recs, focal_verdict="TRUE")
        out[m] = est
        p = est["p"]
        nv = est.get("noverdict_rate")
        print(f"{m:<26}{est['n']:>5}"
              f"{(p if p is not None else float('nan')):>10.3f}"
              f"{(nv if nv is not None else 0):>11.1%}")
    return out


def _selftest() -> int:
    src = Path(__file__).read_text()
    fails = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    check("imports build_units/execute_units/cell_estimate from run_stance_factorial",
          "from scripts.run_stance_factorial import" in src)
    check("filters on channel==neutral_filler", 'channel == "neutral_filler"' in src)
    check("filters on direction==none", 'direction == "none"' in src)
    check("k_estimate is pinned to 1 in the build_units() call",
          "k_estimate=1," in src)
    # Inspect the live parser rather than grep source text for an absent flag
    # name -- any string written here to search FOR the flag also becomes a
    # literal occurrence of that flag name in this file's own source (bit
    # this check tripped over twice while writing it).
    flag_words = {"-".join(opt.lstrip("-").split("-")) for a in _build_parser()._actions
                  for opt in a.option_strings}
    banned = "k" + "-estimate"
    check("no --k-estimate flag is exposed on the CLI", banned not in flag_words)
    check("reframe is pinned True (matches the real ablation's default)",
          "reframe=True," in src)
    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Neutral-arm-only model triage for BrokenMath")
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--k-select", type=int, default=5)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--scaffold", default="standard_cot")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--allow-unresolved", action="store_true", default=True)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--selftest", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    a = _build_parser().parse_args(argv)

    if a.selftest:
        return _selftest()

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    items, neutral_units, dropped = plan(
        models, n=a.n, k_select=a.k_select, seed=a.seed, scaffold=a.scaffold,
        max_tokens=a.max_tokens, temperature=a.temperature,
        allow_unresolved=a.allow_unresolved,
    )
    print(f"[probe] {len(items)} items x {len(models)} models x {a.scaffold} -- "
          f"{len(neutral_units)} neutral-arm units to run "
          f"({dropped} stance-arm units built but NEVER executed or billed)")

    if a.run:
        records = execute_units(neutral_units, workers=a.workers)
    else:
        records = load_cached_records(neutral_units)
        print("(plan/cache-read only -- pass --run to generate)")

    results = report(models, records)

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
