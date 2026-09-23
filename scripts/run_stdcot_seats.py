"""
scripts/run_stdcot_seats.py -- Addendum 16.28: the screened deliberation for a
content-filtered deployment with the seats on plain chain of thought instead of
the narration scaffold.

`run_crowdgold_filter_screen.py` fixes `--agent-scaffold` to its SCREEN_SCAFFOLD
(narrative_cot) and threads that name through its footprint, cost model and
readout, so it cannot run this cell without a wide change. This driver reuses
its pieces unchanged: the same panel (`load_panel`), the same registered screen
(`print_screen_report`, survivors = panel minus `rep["registered"]`), the same
runner argv (`deliberate_argv`) with exactly two tokens changed (the seat
scaffold and the tag), and the same completeness guard (`completeness_check` /
`expected_debate_keys`) after the run. The base runner applies its own
content-filter screen, which is the registered one, so the panel matches 16.24.

Run:
  python -m scripts.run_stdcot_seats --model Llama-3.3-70B-Instruct --samples 2 --dry-run
  python -m scripts.run_stdcot_seats --model Llama-3.3-70B-Instruct --samples 2 --run --workers 20
"""
from __future__ import annotations

import argparse
import json
import sys

from scripts import run_crowdgold_deliberation as rcd
from scripts import run_crowdgold_filter_screen as fs

SEAT_SCAFFOLD = "standard_cot"


def stdcot_tag(model: str) -> str:
    return f"{fs.default_tags(model)[1]}_stdcot"


def build_argv(model: str, tag: str, *, samples: int, workers: int, dry_run: bool) -> list[str]:
    _, narrated_tag = fs.default_tags(model)
    screen_tag, _ = fs.default_tags(model)
    argv = fs.deliberate_argv(model, tag, n_yta=fs.N_YTA, n_nta=fs.N_NTA, workers=workers,
                              n_boot=1000, compare_rows=rcd.OUT_DIR / f"{screen_tag}_rows.csv",
                              moderator_model=None, dry_run=dry_run, samples=samples)
    i = argv.index("--agent-scaffold")
    assert argv[i + 1] == fs.SCREEN_SCAFFOLD, argv
    argv[i + 1] = SEAT_SCAFFOLD
    assert tag != narrated_tag
    return argv


def survivors(model: str):
    items = fs.load_panel(fs.N_YTA, fs.N_NTA)
    status = fs.screen_status(model, items, fs.SCREEN_SAMPLES)
    chain = fs.chain_filtered_items(model)
    rep = fs.print_screen_report(model, items, status, chain)
    drop = set(rep["registered"])
    return [it for it in items if it.item_id not in drop], status


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--run", action="store_true", help="REQUIRED to generate; spends money")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        base = fs.deliberate_argv("m", "t", n_yta=1, n_nta=1, workers=1, n_boot=1,
                                  compare_rows=rcd.OUT_DIR / "x.csv", moderator_model=None,
                                  dry_run=True, samples=2)
        ours = build_argv("Llama-3.3-70B-Instruct", stdcot_tag("Llama-3.3-70B-Instruct"),
                          samples=2, workers=1, dry_run=True)
        diff = [(x, y) for x, y in zip(base, ours) if x != y]
        changed = {x for x, _ in diff} | {y for _, y in diff}
        ok = (SEAT_SCAFFOLD in ours and fs.SCREEN_SCAFFOLD not in ours
              and stdcot_tag("Llama-3.3-70B-Instruct") == "cg_deliberation_llama3370binstruct_stdcot"
              and len(base) == len(ours))
        print(f"argv tokens differing from the wrapper's (model/tag/panel tokens included): {len(diff)}")
        print(f"seat scaffold swapped to {SEAT_SCAFFOLD}, narrated scaffold absent, tag suffixed: "
              f"{'OK' if ok else 'FAIL'}")
        return 0 if ok else 1

    if not a.model:
        ap.error("--model is required unless --selftest")
    tag = stdcot_tag(a.model)
    surv, status = survivors(a.model)
    if not status["complete"]:
        print("ERROR: the step-1 screen is not complete for this model; run it first.")
        return 2
    print(f"\n[stdcot-seats] {a.model}  tag {tag}  survivors {len(surv)}  samples {a.samples}")
    run_argv = build_argv(a.model, tag, samples=a.samples, workers=a.workers,
                          dry_run=(a.dry_run or not a.run))
    print("  delegates to: python -m scripts.run_crowdgold_deliberation " + " ".join(run_argv))
    if a.dry_run or not a.run:
        rc = rcd.main(run_argv)
        if not a.run and not a.dry_run:
            print("\n*** NOT RUNNING: pass --run to generate (this spends money). ***")
            return 2
        return rc
    rc = rcd.main(run_argv)
    if rc in (0, 1, 4):
        summ_p = rcd.OUT_DIR / f"{tag}_summary.json"
        sn = None
        if summ_p.exists():
            try:
                sn = json.loads(summ_p.read_text()).get("n_items")
            except json.JSONDecodeError:
                sn = None
        comp = fs.completeness_check(tag, a.model, fs.expected_debate_keys(surv, a.samples),
                                     summary_n_items=sn)
        fs.print_completeness(comp)
        (rcd.OUT_DIR / f"{tag}_completeness.json").write_text(json.dumps(comp, indent=2))
        if not comp["pass"]:
            rc = 4
    if rc == 4:
        print("\n  *** GUARD FAILED (round-level, outcome, or completeness). Read nothing off this run. ***")
    return rc


if __name__ == "__main__":
    sys.exit(main())
