"""
scripts/analyze_rolelock_by_model.py -- role-lock and seat accuracy at the
opening statement (r0) for every community model, replayed from the r0 caches.

WHY THIS EXISTS
---------------
The headline PDF (papers/embodied_sensor/headline_theory.html, R1 and Fig. 1)
reports role-lock and per-seat verdict accuracy for four community models. The
grok row is in the prereg (Addendum 16.8, L3276-3279) but the sonnet, nano and
haiku rows were computed in-session and lived only in a scratchpad JSON
(papers/embodied_sensor/figures/fig1_rolelock.json) with no generating script.
The house rule for the ICLR submission is that every number traces to a prereg
RESULTS block, the PDF source, or a divergence_study_outputs artefact, so this
module regenerates the table from the cached r0 calls and writes
`divergence_study_outputs/rolelock_by_model.json`.

Zero spend: it reads only `cgd_<model>_narrative_cot_t2560_<arm>_<item>_<idx>_r0_<seat>.json`
records that the deliberation runs already wrote. No generator is constructed.

DEFINITIONS (identical to scripts/analyze_topology_2x2.py, analyse_cell)
-------------------------------------------------------------------------
For each debate row in `<tag>_rows.csv` and each of the three seats, the r0
opening-statement verdict is coded to an at-fault bit by the registered
five-token collapse (`code_response(extract_verdict(...), "published")`).
P(at-fault) and seat accuracy are computed per seat over the rows whose r0
verdict for that seat is codable, so the three denominators differ slightly.
Role-lock is |P(at-fault | writer_advocate) - P(at-fault | counterparty)|.
`n_r0` is the writer's advocate's codable count, matching fig1_rolelock.json.

USAGE
    python -m scripts.analyze_rolelock_by_model
    python -m scripts.analyze_rolelock_by_model --json divergence_study_outputs/rolelock_by_model.json
    python -m scripts.analyze_rolelock_by_model --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Sequence

import scripts.run_crowdgold_deliberation as rcd
from scripts.analyze_crowdgold_sdt import code_response
from scripts.run_phase1_quartet import OUT_DIR
from scripts.verdict_format import extract_verdict

ANALYSIS_PATH = OUT_DIR / "rolelock_by_model.json"
R0_CAP = 2560
SEATS = ("writer_advocate", "counterparty", "neutral_adjudicator")

#: short name -> (model id, rows tag). The haiku and sonnet tags are the
#: corrected re-runs of the Addendum 6 CORRECTION block (prereg L1255-1276);
#: the retracted screen tags are deliberately not listed.
PANELS = {
    "grok": ("grok-4-1-fast-reasoning", "cg_deliberation"),
    "sonnet": ("claude-sonnet-4-6", "cg_deliberation_sonnet_fixed"),
    "nano": ("gpt-5.4-nano", "cg_deliberation_nano"),
    "haiku": ("claude-haiku-4-5", "cg_deliberation_haiku_fixed"),
}


def r0_verdict_code(model: str, arm: str, item: str, idx: int, role_id: str) -> Optional[int]:
    p = rcd.call_cache_path(model, "narrative_cot", arm, item, idx, "r0", role_id, R0_CAP)
    if not p.exists():
        return None
    txt = json.loads(p.read_text()).get("output") or ""
    return code_response(extract_verdict(txt, rcd.INSTRUMENT), "published")


def analyse_panel(model: str, tag: str) -> Optional[dict]:
    rows_path = OUT_DIR / f"{tag}_rows.csv"
    if not rows_path.exists():
        return None
    at = {s: [0, 0] for s in SEATS}
    acc = {s: [0, 0] for s in SEATS}
    n_rows = 0
    with open(rows_path) as f:
        for r in csv.DictReader(f):
            if r.get("model") != model:
                continue
            n_rows += 1
            g = 1 if r["gold_verdict"] == "YTA" else 0
            for s in SEATS:
                c = r0_verdict_code(model, r["arm"], r["item_id"], int(r["sample_idx"]), s)
                if c is None:
                    continue
                at[s][0] += c; at[s][1] += 1
                acc[s][0] += int(c == g); acc[s][1] += 1
    if not n_rows:
        return None
    p_at = {s: (at[s][0] / at[s][1] if at[s][1] else None) for s in SEATS}
    seat_acc = {s: (acc[s][0] / acc[s][1] if acc[s][1] else None) for s in SEATS}
    lock = (abs(p_at[SEATS[0]] - p_at[SEATS[1]])
            if None not in (p_at[SEATS[0]], p_at[SEATS[1]]) else None)
    return {
        "model": model, "tag": tag, "n_debates": n_rows,
        "n_r0": at[SEATS[0]][1],
        "n_codable_by_seat": {s: at[s][1] for s in SEATS},
        "p_at_fault": p_at, "seat_acc": seat_acc, "role_lock": lock,
    }


def run() -> dict:
    return {k: analyse_panel(*v) for k, v in PANELS.items()}


def _print(res: dict) -> None:
    print(f"{'model':8s} {'n_r0':>6s}  {'P(at-fault) w / c / n':>26s}  {'lock':>6s}  {'acc w / c / n':>22s}")
    for k, r in res.items():
        if r is None:
            print(f"{k:8s}  (rows file absent)")
            continue
        pa = " / ".join(f"{r['p_at_fault'][s]:.3f}" for s in SEATS)
        sa = " / ".join(f"{r['seat_acc'][s]:.3f}" for s in SEATS)
        print(f"{k:8s} {r['n_r0']:6d}  {pa:>26s}  {r['role_lock']:6.3f}  {sa:>22s}")


def _selftest() -> int:
    """The grok row must reproduce the 16.8 RESULTS per-seat P(at-fault)
    (prereg L3405: 0.003 / 0.974 / 0.430) and a lock that rounds to 0.971."""
    r = analyse_panel(*PANELS["grok"])
    assert r is not None, "grok rows file missing"
    got = tuple(round(r["p_at_fault"][s], 3) for s in SEATS)
    assert got == (0.003, 0.974, 0.430), got
    assert round(r["role_lock"], 3) == 0.971, r["role_lock"]
    assert all(0.0 <= r["seat_acc"][s] <= 1.0 for s in SEATS)
    print("selftest OK: grok r0 row reproduces prereg L3405 (0.003 / 0.974 / 0.430, lock 0.971)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--json", type=Path, default=ANALYSIS_PATH)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    res = run()
    _print(res)
    a.json.write_text(json.dumps(res, indent=1))
    print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
