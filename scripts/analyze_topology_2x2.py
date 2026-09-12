"""
scripts/analyze_topology_2x2.py -- Addendum 16.10 readout: the role x
connectivity 2x2, every cell through identical code.

Per cell, on the cell's own surviving debates AND on the intersection of items
surviving in all cells (the matched comparison):
  role-lock            |P(at_fault) seat1 - seat2| at r0 (manipulation check for
                       identical cells, where it is ~0 by construction)
  per-seat P(at_fault) and per-seat r0 verdict accuracy
  objection marginals, localisation excess vs the Poisson-binomial null,
                       fire rate P(>=2)
  G3 stake concentration  where stakes exist (embodied cells only)
  error lift of >=2    codable population, read only at >= 60 fired
  seat-majority (r0) vs best single seat vs group verdict  -- with three
                       exchangeable seats the majority is a genuine
                       aggregation for the first time, so this is where
                       Lambda can first exceed 1
  transfer             P(grok-solo wrong | cell fired) vs not, grok solo =
                       cached standard majority-of-3 the cell never touched
  guards               mean n_r3_unparsed, P(non-codable | fired)

Gates and conventions inherited from 16.8: five-token verdicts scored by the
registered code_response collapse (never string equality); raw and codable
populations side by side; item-clustered bootstrap CIs; partial panels
reported, never retried.

Usage
-----
  python -m scripts.analyze_topology_2x2 --selftest
  python -m scripts.analyze_topology_2x2            # all cells present on disk
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import scripts.run_crowdgold_deliberation as rcd
from scripts.analyze_actuator_ladder import COMPARATOR_PATH, correct, load_comparator, make_rule_vendor
from scripts.analyze_crowdgold_sdt import code_response
from scripts.analyze_unembodied_ablation import (
    load_rows, localisation, per_seat_objection_rates, verdict_correct,
)
from scripts.run_crowdgold_topology import CELLS, build_roles
from scripts.run_phase1_quartet import OUT_DIR
from scripts.verdict_format import extract_verdict

ANALYSIS_PATH = OUT_DIR / "topology_2x2_analysis.json"
MODEL = "grok-4-1-fast-reasoning"
R0_CAP = 2560
MIN_FIRED = 60

#: (roles, edges) -> (tag, seat role ids)
CELL_SPECS = {
    ("embodied", "on"): ("cg_deliberation",
                         ("writer_advocate", "counterparty", "neutral_adjudicator")),
}
for _cell, _spec in CELLS.items():
    CELL_SPECS[_cell] = (_spec["tag"], tuple(r.role_id for r in build_roles(*_cell)))


def _boot(by_item: dict, stat, *, draws: int, seed: int, alpha: float = 0.05):
    ids = sorted(by_item)
    rng = random.Random(seed)
    point = stat([x for i in ids for x in by_item[i]])
    vals = sorted(v for v in (
        stat([x for i in (ids[rng.randrange(len(ids))] for _ in ids) for x in by_item[i]])
        for _ in range(draws)) if v is not None)
    if not vals:
        return point, None, None
    return point, vals[int(alpha / 2 * len(vals))], vals[min(int((1 - alpha / 2) * len(vals)), len(vals) - 1)]


def r0_verdict_code(arm: str, item: str, idx: int, role_id: str) -> Optional[int]:
    p = rcd.call_cache_path(MODEL, "narrative_cot", arm, item, idx, "r0", role_id, R0_CAP)
    if not p.exists():
        return None
    txt = json.loads(p.read_text()).get("output") or ""
    return code_response(extract_verdict(txt, rcd.INSTRUMENT), "published")


def g3_stake_concentration(votes_path: Path, keep_items: Optional[set], *, draws: int, seed: int) -> dict:
    """R4 reject rate | own stake undermined minus | not, stake seats only.
    n/a when no seat carries a stake (identical cells)."""
    sv = []
    with open(votes_path) as f:
        for v in csv.DictReader(f):
            if v.get("model") != MODEL:
                continue
            if keep_items is not None and v["item_id"] not in keep_items:
                continue
            if v.get("role_stake") in ("at_fault", "not_at_fault") and v.get("stake_undermined") not in ("", None):
                sv.append({"item": v["item_id"], "reject": int(v.get("reject") or 0),
                           "und": int(v["stake_undermined"])})
    if not sv:
        return {"applicable": False}
    by = defaultdict(list)
    for v in sv:
        by[v["item"]].append(v)

    def stat(vs):
        u = [x["reject"] for x in vs if x["und"]]
        n = [x["reject"] for x in vs if not x["und"]]
        return None if not u or not n else sum(u) / len(u) - sum(n) / len(n)
    p, lo, hi = _boot(by, stat, draws=draws, seed=seed)
    return {"applicable": True, "n_stake_votes": len(sv), "delta": p, "lo": lo, "hi": hi}


def analyse_cell(cell: tuple, *, keep_items: Optional[set], draws: int, seed: int) -> Optional[dict]:
    tag, seats = CELL_SPECS[cell]
    rows_path, votes_path = OUT_DIR / f"{tag}_rows.csv", OUT_DIR / f"{tag}_votes.csv"
    if not rows_path.exists() or not votes_path.exists():
        return None
    rows = load_rows(rows_path, MODEL, keep_items)
    if not rows:
        return None
    rates = per_seat_objection_rates(votes_path, MODEL, keep_items)
    loc = localisation(rows, rates)
    n = len(rows)
    codable = [r for r in rows if r["codable"]]

    # ---- r0 verdicts per seat: role-lock, seat accuracy, seat-majority ----
    at = {s: [0, 0] for s in seats}
    acc = {s: [0, 0] for s in seats}
    maj_recs = []
    for r in rows:
        codes = {}
        for s in seats:
            c = r0_verdict_code(r["arm"], r["item"], r["sample_idx"], s)
            if c is None:
                continue
            codes[s] = c
            g = 1 if r["gold"] == "YTA" else 0
            at[s][0] += c; at[s][1] += 1
            acc[s][0] += int(c == g); acc[s][1] += 1
        if len(codes) == 3:
            g = 1 if r["gold"] == "YTA" else 0
            maj = 1 if sum(codes.values()) >= 2 else 0
            maj_recs.append({"item": r["item"], "maj": int(maj == g),
                             "seats": {s: int(codes[s] == g) for s in seats},
                             "group": int(bool(r["s2_ok"])) if r["s2_ok"] is not None else None})
    p_at = {s: (at[s][0] / at[s][1] if at[s][1] else None) for s in seats}
    seat_acc = {s: (acc[s][0] / acc[s][1] if acc[s][1] else None) for s in seats}
    lock = (abs(p_at[seats[0]] - p_at[seats[1]])
            if None not in (p_at[seats[0]], p_at[seats[1]]) else None)
    best_seat = max(seats, key=lambda s: seat_acc[s] if seat_acc[s] is not None else -1)
    m = len(maj_recs)
    maj_acc = sum(x["maj"] for x in maj_recs) / m if m else None
    best_acc_on_maj = sum(x["seats"][best_seat] for x in maj_recs) / m if m else None
    grp = [x for x in maj_recs if x["group"] is not None]
    grp_acc = sum(x["group"] for x in grp) / len(grp) if grp else None
    maj_eq_best = sum(1 for x in maj_recs if x["maj"] == x["seats"][best_seat]) / m if m else None
    by = defaultdict(list)
    for x in maj_recs:
        by[x["item"]].append(x["maj"] - x["seats"][best_seat])
    mb_p, mb_lo, mb_hi = _boot(by, lambda xs: sum(xs) / len(xs) if xs else None, draws=draws, seed=seed + 3)

    # ---- fire, lift ----
    fired = lambda r: r["n_objectors"] >= 2
    n_fired = sum(1 for r in rows if fired(r))

    def lift_on(pop):
        fl = [r for r in pop if fired(r)]; un = [r for r in pop if not fired(r)]
        if not fl or not un:
            return None
        return {"n_fired": len(fl),
                "err_fired": 1 - sum(bool(r["s2_ok"]) for r in fl) / len(fl),
                "err_unfired": 1 - sum(bool(r["s2_ok"]) for r in un) / len(un)}
    lc = lift_on(codable)
    byl = defaultdict(list)
    for r in codable:
        byl[r["item"]].append(r)

    def lift_stat(rs):
        fl = [r for r in rs if fired(r)]; un = [r for r in rs if not fired(r)]
        return None if not fl or not un else ((1 - sum(bool(r["s2_ok"]) for r in fl) / len(fl))
                                              - (1 - sum(bool(r["s2_ok"]) for r in un) / len(un)))
    l_p, l_lo, l_hi = _boot(byl, lift_stat, draws=draws, seed=seed + 1)

    # ---- transfer to grok solo (standard maj-3), which no cell touched ----
    comp = load_comparator(COMPARATOR_PATH)
    solo = make_rule_vendor(comp, MODEL, "standard")
    fl_c = defaultdict(int); k_c = defaultdict(int)
    for r in rows:
        c = (r["arm"], r["item"]); fl_c[c] += int(fired(r)); k_c[c] += 1
    gold = {(r["arm"], r["item"]): r["gold"] for r in rows}
    trec = []
    for c, g in gold.items():
        v = solo({"arm": c[0], "item": c[1]}); ok = correct(v, g) if v is not None else None
        if ok is None:
            continue
        trec.append({"item": c[1], "w": int(not ok), "f": int(fl_c[c] * 2 > k_c[c])})
    byt = defaultdict(list)
    for x in trec:
        byt[x["item"]].append(x)

    def tstat(rs):
        f = [x for x in rs if x["f"]]; u = [x for x in rs if not x["f"]]
        return None if not f or not u else sum(x["w"] for x in f) / len(f) - sum(x["w"] for x in u) / len(u)
    t_p, t_lo, t_hi = _boot(byt, tstat, draws=draws, seed=seed + 2)
    tf = [x for x in trec if x["f"]]; tu = [x for x in trec if not x["f"]]

    n_fired_raw = n_fired
    noncod_f = sum(1 for r in rows if fired(r) and not r["codable"])
    return {
        "cell": {"roles": cell[0], "edges": cell[1]}, "tag": tag, "seats": list(seats),
        "n_debates": n, "n_items": len({r["item"] for r in rows}), "n_codable": len(codable),
        "role_lock": lock, "p_at_fault": p_at, "seat_r0_accuracy": seat_acc,
        "best_seat": best_seat,
        "aggregation": {"n": m, "seat_majority_r0": maj_acc, "best_seat_r0": best_acc_on_maj,
                        "group_verdict_s2": grp_acc, "majority_equals_best_seat_rate": maj_eq_best,
                        "majority_minus_best_seat": {"delta": mb_p, "lo": mb_lo, "hi": mb_hi}},
        "objection": {"per_seat": rates["per_seat"], "pooled": rates["pooled"],
                      "localisation_excess": loc["localisation_excess"],
                      "fire_rate": loc["fire_observed"], "fire_independence": loc["fire_independence"]},
        "g3": g3_stake_concentration(votes_path, keep_items, draws=draws, seed=seed + 4),
        "lift": {"n_fired": n_fired_raw, "powered": n_fired_raw >= MIN_FIRED, "codable": lc,
                 "delta": l_p, "lo": l_lo, "hi": l_hi},
        "transfer_grok_solo": {"n_cells": len(trec), "n_flagged_cells": len(tf),
                               "p_wrong_flagged": (sum(x["w"] for x in tf) / len(tf) if tf else None),
                               "p_wrong_unflagged": (sum(x["w"] for x in tu) / len(tu) if tu else None),
                               "lift": t_p, "lo": t_lo, "hi": t_hi},
        "guards": {"mean_n_r3_unparsed": sum(r["n_r3_unparsed"] for r in rows) / n,
                   "p_noncodable_given_fired": noncod_f / n_fired_raw if n_fired_raw else None},
    }


def _f(x, fmt="+.3f"):
    return "n/a" if x is None else format(x, fmt)


def print_grid(res: dict) -> None:
    order = [("embodied", "on"), ("embodied", "off"), ("identical", "on"), ("identical", "off")]
    cells = [res["cells"].get(f"{r}/{e}") for r, e in order]
    hdr = [f"{r}/{e}" for r, e in order]
    print("\n" + "=" * 96)
    print(f"ADDENDUM 16.10 -- ROLE x CONNECTIVITY 2x2   (matched items: {res['n_items_matched']})")
    print("=" * 96)
    print(f"  {'':<34}" + "".join(f"{h:>15}" for h in hdr))

    def row(label, get, fmt="+.3f"):
        print(f"  {label:<34}" + "".join(
            f"{(_f(get(c), fmt) if c else 'absent'):>15}" for c in cells))
    row("n debates", lambda c: c["n_debates"], "d")
    row("ROLE-LOCK", lambda c: c["role_lock"], ".3f")
    row("localisation excess (P-binom)", lambda c: c["objection"]["localisation_excess"])
    row("fire rate P(>=2)", lambda c: c["objection"]["fire_rate"], ".3f")
    row("G3 grip (stake seats)", lambda c: c["g3"]["delta"] if c["g3"]["applicable"] else None)
    row("error lift (>=2), codable", lambda c: c["lift"]["delta"] if c["lift"]["powered"] else None)
    row("  n fired", lambda c: c["lift"]["n_fired"], "d")
    row("seat-majority (r0) acc", lambda c: c["aggregation"]["seat_majority_r0"], ".3f")
    row("best single seat acc", lambda c: c["aggregation"]["best_seat_r0"], ".3f")
    row("majority - best seat", lambda c: c["aggregation"]["majority_minus_best_seat"]["delta"])
    row("group verdict (S2) acc", lambda c: c["aggregation"]["group_verdict_s2"], ".3f")
    row("TRANSFER lift on grok solo", lambda c: c["transfer_grok_solo"]["lift"])
    row("  P(solo wrong | flagged)", lambda c: c["transfer_grok_solo"]["p_wrong_flagged"], ".3f")
    row("  P(solo wrong | not)", lambda c: c["transfer_grok_solo"]["p_wrong_unflagged"], ".3f")
    row("guard: mean n_r3_unparsed", lambda c: c["guards"]["mean_n_r3_unparsed"], ".4f")
    print()
    for c in cells:
        if not c:
            continue
        print(f"  {c['cell']['roles']}/{c['cell']['edges']}: seats {c['seats']}")
        print(f"     P(at_fault) " + "  ".join(f"{s.split('_')[0][:8]}={_f(v, '.3f')}" for s, v in c["p_at_fault"].items()))
        print(f"     lift CI [{_f(c['lift']['lo'])}, {_f(c['lift']['hi'])}]   transfer CI "
              f"[{_f(c['transfer_grok_solo']['lo'])}, {_f(c['transfer_grok_solo']['hi'])}]   "
              f"majority-best CI [{_f(c['aggregation']['majority_minus_best_seat']['lo'])}, "
              f"{_f(c['aggregation']['majority_minus_best_seat']['hi'])}]")


def _selftest() -> int:
    fails = []
    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond: fails.append(name)
    check("four cells specified", len(CELL_SPECS) == 4)
    check("embodied/on maps to the registered run", CELL_SPECS[("embodied", "on")][0] == "cg_deliberation")
    ids = [s for _, seats in CELL_SPECS.values() for s in seats]
    check("12 distinct seat ids across the four cells", len(set(ids)) == 12)
    check("identical cells carry no stake seats -> G3 must report not applicable",
          all(r.stake == "none" for r in build_roles("identical", "on")))
    # G3 on a synthetic votes file with and without stakes
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        vp = Path(td) / "v.csv"
        with open(vp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "item_id", "role_id", "role_stake", "stake_undermined", "reject"])
            w.writeheader()
            for i in range(30):
                w.writerow({"model": MODEL, "item_id": f"i{i}", "role_id": "a", "role_stake": "at_fault",
                            "stake_undermined": "1", "reject": "1"})
                w.writerow({"model": MODEL, "item_id": f"i{i}", "role_id": "b", "role_stake": "not_at_fault",
                            "stake_undermined": "0", "reject": "0"})
                w.writerow({"model": MODEL, "item_id": f"i{i}", "role_id": "n", "role_stake": "none",
                            "stake_undermined": "", "reject": "0"})
        g = g3_stake_concentration(vp, None, draws=300, seed=1)
        check("G3 is exactly +1.0 on a perfectly stake-tracking synthetic panel",
              g["applicable"] and abs(g["delta"] - 1.0) < 1e-9)
        with open(vp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "item_id", "role_id", "role_stake", "stake_undermined", "reject"])
            w.writeheader()
            w.writerow({"model": MODEL, "item_id": "i0", "role_id": "n", "role_stake": "none",
                        "stake_undermined": "", "reject": "0"})
        check("G3 reports not-applicable when no seat has a stake",
              not g3_stake_concentration(vp, None, draws=10, seed=1)["applicable"])
    check("verdict scoring goes through the registered five-token collapse (ESH vs gold YTA is CORRECT)",
          verdict_correct("ESH", "YTA") is True)
    # the embodied/on cell must reproduce the registered numbers
    emb = analyse_cell(("embodied", "on"), keep_items=None, draws=200, seed=1)
    if emb:
        check("embodied/on role-lock reproduces 0.971 (+/-0.005)", abs(emb["role_lock"] - 0.971) < 0.005)
        check("embodied/on majority == best seat on ~100% of debates",
              emb["aggregation"]["majority_equals_best_seat_rate"] > 0.99)
        check("embodied/on fire rate reproduces 0.196 (+/-0.01)",
              abs(emb["objection"]["fire_rate"] - 0.196) < 0.01)
        check("embodied/on G3 reproduces +0.63 (+/-0.02)",
              emb["g3"]["applicable"] and abs(emb["g3"]["delta"] - 0.63) < 0.02)
        # The transfer readout uses the stake-blind `>=2 objected` counter, not
        # the composite stake flag, because the identical cells have no stakes
        # to read. On embodied/on the two differ by 9 debates (328 vs 337) and
        # 2 cell-level firings (51 vs 53); the composite gave 0.264 / 0.063, the
        # counter gives 0.275 / 0.062. Same finding, slightly sharper.
        check("embodied/on transfer to grok solo reproduces ~4x (0.275 vs 0.062, stake-blind counter)",
              abs(emb["transfer_grok_solo"]["p_wrong_flagged"] - 0.275) < 0.01
              and abs(emb["transfer_grok_solo"]["p_wrong_unflagged"] - 0.062) < 0.01)
    else:
        check("embodied/on cell present on disk", False)
    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--out", type=Path, default=ANALYSIS_PATH)
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    present = {}
    for cell, (tag, _) in CELL_SPECS.items():
        if (OUT_DIR / f"{tag}_rows.csv").exists():
            present[cell] = {r["item"] for r in load_rows(OUT_DIR / f"{tag}_rows.csv", MODEL)}
    if not present:
        print("no cells on disk"); return 2
    matched = set.intersection(*present.values())
    res = {"n_items_matched": len(matched), "cells_present": [f"{r}/{e}" for r, e in present],
           "cells": {}, "cells_unmatched_full": {}}
    for i, cell in enumerate(present):
        res["cells"][f"{cell[0]}/{cell[1]}"] = analyse_cell(cell, keep_items=matched, draws=a.n_boot, seed=a.seed + 10 * i)
        res["cells_unmatched_full"][f"{cell[0]}/{cell[1]}"] = analyse_cell(cell, keep_items=None, draws=a.n_boot, seed=a.seed + 10 * i + 5)
    print_grid(res)
    a.out.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
