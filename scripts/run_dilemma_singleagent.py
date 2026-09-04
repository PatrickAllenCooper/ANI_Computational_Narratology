"""
scripts/run_dilemma_singleagent.py -- Addendum 12 Stage 2: the actuator's
own baseline. A single model reads both accounts COLD (no deliberation, no
role, no other seat's text) and gives its own verdict, k samples, majority
vote. This is the "fresh single-model baseline" Stage 2's registration
calls for -- no cache of a plain single-agent Dilemmas classification
exists anywhere in the repo, unlike Addendum 4's A3a rung, which read its
actuator verdicts out of an AITA comparator already paid for by an
unrelated study.

WHY A SEPARATE, MINIMAL RUNNER INSTEAD OF A MODE FLAG ON THE DELIBERATION
RUNNER

The actuator must be structurally incapable of seeing anything the
deliberation produced -- no synthesis, no other seat's statement, nothing
but the two accounts -- or a "cross-check" that shares context with the
thing it is checking is not a cross-check. Keeping this in its own file
with its own (single) call per sample makes that guarantee visible in the
code rather than resting on remembering to pass the right flags to a
multi-mode runner.

WHICH MODEL, AND WHY

Addendum 4's A3a rung deliberately used a DIFFERENT vendor (haiku) from
the deliberating model (grok) as the actuator, precisely because a
same-vendor "cross-check" is not independent. Stage 2 repeats that
choice: the default actuator model is claude-haiku-4-5, cross-vendor
from the grok deliberation this baseline is meant to check.

Usage:
  python -m scripts.run_dilemma_singleagent --selftest
  python -m scripts.run_dilemma_singleagent --dry-run --n-items 80
  python -m scripts.run_dilemma_singleagent --n-items 80  # SPENDS MONEY
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.run_phase1_quartet import OUT_DIR
from scripts.load_scruples import ScruplesDataError
from scripts.run_crowdgold_aita import count_tokens, _optional_bpe_counter
from scripts.run_crowdgold_deliberation import PRICES, call_cache_path, do_call
from scripts.run_crowdgold_dilemma import (
    DILEMMA_INSTRUMENT,
    GLOSSARY_CORE,
    DilemmaItem,
    load_items,
    request_block,
    subset_items,
)
from scripts.verdict_format import extract_verdict, is_noncommittal, verdict_instruction

SCAFFOLD_NAME = "dilemma_standard"
ARM = "neutral"
ROUND_NAME = "single"
ROLE_ID = "baseline"

BASELINE_SYSTEM = (
    "You are answering a single direct question about two people's real, "
    "independently-written accounts of a conflict. Use only what each "
    "account itself says; do not invent facts, and answer directly with no "
    "further discussion beyond the required verdict line."
)


def baseline_user(item: DilemmaItem, *, allow_unresolved: bool) -> str:
    return (
        f"{request_block(item)}\n\n{GLOSSARY_CORE}\n\n"
        "Read both accounts and decide directly, with no deliberation and "
        "no other input: which of the two actions was the more wrong of the "
        "two?\n\n"
        f"{verdict_instruction(DILEMMA_INSTRUMENT, allow_unresolved=allow_unresolved)}"
    )


def majority_vote(verdicts: Sequence[str]) -> Optional[str]:
    """Majority of committal (ACTION_A/ACTION_B) verdicts, or None on a tie
    or if none are committal -- callers fall back to S2 on None, exactly as
    Addendum 4's A3a rule does on a tied or absent cross-vendor read.
    """
    a = sum(1 for v in verdicts if v == "ACTION_A")
    b = sum(1 for v in verdicts if v == "ACTION_B")
    if a == b:
        return None
    return "ACTION_A" if a > b else "ACTION_B"


def run_one(model: str, item: DilemmaItem, idx: int, *, max_tokens: int,
           allow_unresolved: bool) -> dict:
    rec = do_call(
        model=model, scaffold=SCAFFOLD_NAME, arm=ARM, item=item, idx=idx,
        round_name=ROUND_NAME, role_id=ROLE_ID, system=BASELINE_SYSTEM,
        user=baseline_user(item, allow_unresolved=allow_unresolved),
        max_tokens=max_tokens, parents=(item.post_text,), cap=0,
    )
    verdict = extract_verdict(rec.get("output") or "", DILEMMA_INSTRUMENT)
    return {
        "model": model, "scaffold": SCAFFOLD_NAME, "arm": ARM,
        "item_id": item.item_id, "sample_idx": idx,
        "gold_verdict": item.gold_verdict, "verdict": verdict,
        "noncommittal": int(is_noncommittal(verdict)),
        "finish_reason": rec.get("finish_reason", ""),
        "truncated": int(rec.get("finish_reason", "") in
                         ("max_tokens", "length", "MAX_TOKENS")),
        "output_len": len(rec.get("output") or ""),
        "prompt_tokens": rec.get("prompt_tokens", 0),
        "completion_tokens": rec.get("completion_tokens", 0),
    }


ROW_FIELDS = (
    "model", "scaffold", "arm", "item_id", "sample_idx", "gold_verdict",
    "verdict", "noncommittal", "finish_reason", "truncated", "output_len",
    "prompt_tokens", "completion_tokens",
)


def cost_model(model: str, items: Sequence[DilemmaItem], k: int,
              *, max_tokens: int) -> dict:
    if not items:
        raise ValueError("cost model needs at least one item")
    bpe = _optional_bpe_counter()
    probe = sorted(items, key=lambda i: len(i.post_text))[len(items) // 2]
    prompt = baseline_user(probe, allow_unresolved=True)
    ptok = count_tokens(prompt, tokenizer=bpe) if bpe else count_tokens(prompt)
    n_calls = len(items) * k
    pin, pout = PRICES.get(model, (0.0, 0.0))
    # A cold, single-sentence-verdict classification is far shorter than any
    # deliberation round; 60 completion tokens is a conservative assumption
    # (no measured baseline yet -- this is a new call shape).
    assumed_completion = 60
    usd = (ptok * n_calls) / 1e6 * pin + (assumed_completion * n_calls) / 1e6 * pout
    return {
        "model": model, "n_items": len(items), "k": k, "n_calls": n_calls,
        "prompt_tokens_per_call": ptok, "assumed_completion": assumed_completion,
        "total_usd": round(usd, 4),
        "tokenizer": "tiktoken cl100k_base BPE" if bpe else "regex approximation",
    }


def print_cost_model(cm: dict) -> None:
    print("\n" + "=" * 72)
    print(f"COST MODEL (dry run) -- {cm['model']} single-agent baseline, "
          f"k={cm['k']}")
    print("=" * 72)
    print(f"  {cm['n_items']} items x {cm['k']} samples = {cm['n_calls']} calls")
    print(f"  tokenizer: {cm['tokenizer']}")
    print(f"  prompt tokens/call: {cm['prompt_tokens_per_call']}  "
          f"assumed completion: {cm['assumed_completion']} (unmeasured; "
          f"a cold single-verdict call, no prior measurement exists)")
    print(f"  TOTAL: ${cm['total_usd']:.2f}")


def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    check("majority_vote: 2-1 A resolves to A",
          majority_vote(["ACTION_A", "ACTION_A", "ACTION_B"]) == "ACTION_A")
    check("majority_vote: 2-1 B resolves to B",
          majority_vote(["ACTION_B", "ACTION_A", "ACTION_B"]) == "ACTION_B")
    check("majority_vote: 1-1 tie is None",
          majority_vote(["ACTION_A", "ACTION_B"]) is None)
    check("majority_vote: all noncommittal is None",
          majority_vote(["UNRESOLVED", "NOVERDICT"]) is None)
    check("majority_vote: noncommittal votes don't break a real majority",
          majority_vote(["ACTION_A", "ACTION_A", "UNRESOLVED"]) == "ACTION_A")

    fake = DilemmaItem(
        item_id="t1", a_title="A title", a_text="A body text.",
        b_title="B title", b_text="B body text.", gold_verdict="ACTION_A",
        n_votes=10, consensus=0.8,
    )
    u = baseline_user(fake, allow_unresolved=True)
    check("baseline prompt carries the verdict instruction",
          "VERDICT: ACTION_A" in u and "VERDICT: ACTION_B" in u)
    check("baseline prompt states no deliberation is involved",
          "no deliberation" in u)
    check("baseline prompt does not leak AITA vocabulary",
          "YTA" not in u and "NTA" not in u)

    cp = call_cache_path("m", SCAFFOLD_NAME, ARM, "i1", 0, ROUND_NAME,
                         ROLE_ID, 100)
    check("cache path is namespaced by the standard/baseline scaffold, "
          "distinct from the deliberation runner's dilemma_ncot scaffold",
          SCAFFOLD_NAME in str(cp) and "dilemma_ncot" not in str(cp))

    check("ROW_FIELDS has no duplicate columns",
          len(ROW_FIELDS) == len(set(ROW_FIELDS)))

    # ---- end-to-end against a stub generator: no network -------------------
    import scripts.run_dilemma_singleagent as mod
    import scripts.run_crowdgold_deliberation as rcd
    import scripts.run_phase1_quartet as rpq

    class _Stub:
        def __init__(self, text, finish_reason="stop", ptok=40, ctok=8):
            self.text = text
            self.finish_reason = finish_reason
            self.prompt_tokens = ptok
            self.completion_tokens = ctok

    def stub_generate(model, system, user, *, sample_idx=0, max_tokens=0,
                      **kw):
        return _Stub("VERDICT: ACTION_B")

    mod_rcd = rcd
    mod_rcd.generate_any = stub_generate

    tmp_out = Path("/tmp/_dilemma_singleagent_selftest_cache")
    tmp_out.mkdir(exist_ok=True)
    old_out = rpq.OUT_DIR
    rpq.OUT_DIR = tmp_out
    try:
        row = run_one("stub-model", fake, 0, max_tokens=256,
                      allow_unresolved=True)
        check("smoke: row carries the parsed verdict",
              row["verdict"] == "ACTION_B")
        check("smoke: row is not noncommittal", row["noncommittal"] == 0)
        row_written = {k: row[k] for k in ROW_FIELDS}
        check("smoke: every ROW_FIELDS key is present on the row",
              len(row_written) == len(ROW_FIELDS))
    finally:
        rpq.OUT_DIR = old_out

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def run(model: str, items: Sequence[DilemmaItem], *, k: int, max_tokens: int,
       allow_unresolved: bool, workers: int) -> list[dict]:
    tasks = [(it, i) for it in items for i in range(k)]
    print(f"  {len(tasks)} calls ({len(items)} items x {k} samples)")
    rows: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {
            pool.submit(run_one, model, it, i, max_tokens=max_tokens,
                       allow_unresolved=allow_unresolved): (it.item_id, i)
            for (it, i) in tasks
        }
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                rows.append(fut.result())
                done += 1
                if done % 40 == 0:
                    print(f"    {done}/{len(tasks)} calls", flush=True)
            except Exception as e:
                print(f"  ERROR {key}: {e}", flush=True)
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Addendum 12 Stage 2: single-agent cold baseline "
                    "(the actuator)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default="claude-haiku-4-5",
                    help="Cross-vendor from the grok deliberation by "
                         "default, matching Addendum 4's A3a choice.")
    ap.add_argument("--n-items", type=int, default=80,
                    help="Must match the deliberation run's --n-items so "
                         "the item sets are identical by construction "
                         "(same deterministic gold-balanced prefix).")
    ap.add_argument("--min-annotators", type=int, default=4)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--no-unresolved", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="cg_dilemma_baseline")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    try:
        items = load_items(n=None, min_annotators=args.min_annotators,
                           seed=args.seed)
    except ScruplesDataError as e:
        print(f"\nERROR: {e}\n")
        return 2
    n_a, n_b = args.n_items // 2, args.n_items - args.n_items // 2
    items = subset_items(items, n_a, n_b)
    print(f"model {args.model}  k={args.k}  {len(items)} items "
          f"({n_a} gold-A + {n_b} gold-B), same deterministic prefix as "
          "scripts.run_crowdgold_dilemma at the same --n-items")

    if args.dry_run:
        cm = cost_model(args.model, items, args.k, max_tokens=args.max_tokens)
        print_cost_model(cm)
        return 0

    allow_unresolved = not args.no_unresolved
    rows = run(args.model, items, k=args.k, max_tokens=args.max_tokens,
              allow_unresolved=allow_unresolved, workers=args.workers)
    if not rows:
        print("\nNo rows produced.\n")
        return 1

    out_path = OUT_DIR / f"{args.tag}_rows.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ROW_FIELDS))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ROW_FIELDS})
    print(f"\nwrote {len(rows)} rows -> {out_path}")

    trunc = sum(r["truncated"] for r in rows) / len(rows)
    noverd = sum(1 for r in rows if r["verdict"] == "NOVERDICT") / len(rows)
    print(f"truncation {trunc:.1%}  NOVERDICT {noverd:.1%}")
    pt = sum(int(r.get("prompt_tokens", 0) or 0) for r in rows)
    ct = sum(int(r.get("completion_tokens", 0) or 0) for r in rows)
    pin, pout = PRICES.get(args.model, (0.0, 0.0))
    print(f"measured spend: {pt} prompt + {ct} completion tokens ~= "
          f"${(pt / 1e6 * pin + ct / 1e6 * pout):.2f}")
    if trunc > 0.05 or noverd > 0.05:
        print("\n*** GUARD FAILED (truncation or NOVERDICT > 5%). Raise "
              "--max-tokens and re-run before trusting this baseline. ***")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
