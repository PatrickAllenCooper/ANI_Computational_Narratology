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

SINGLE-CELL MODE (per-model readout; zero spend, reads caches and CSVs only).
The same analyse_cell code path on ONE deliberation run for ONE panel model,
e.g. the nano community (cg_deliberation_nano_*, gpt-5.4-nano):

  --tag TAG --model MODEL [--scaffold narrative_cot] [--cell embodied/on]
  [--solo-models model:scaffold,...] [--extra-comparator CSV ...] [--out PATH]

The cell is run twice: on its own surviving items, and on the intersection
with the grok cg_deliberation items (the matched readout, comparable with the
2x2 grid). The transfer readout is one block per --solo-models entry (default
MODEL:standard): the solo model's cached standard majority-of-3 read from
cg_scaffold_combined_rows_k3.csv merged with any --extra-comparator files
(analyze_actuator_ladder.load_comparator format, samples concatenated per
(model, scaffold, arm, item) key). Nothing is added to CELL_SPECS; the grid
path (no --tag) is unchanged and byte-identical to topology_2x2_analysis.json
(the selftest reruns it and diffs). Seeds follow the grid's convention for
cell 0 -- matched readout at --seed, own-items readout at --seed + 5 -- so the
grok cell through this mode reproduces cells.embodied/on and
cells_unmatched_full.embodied/on exactly (selftest). Output
topology_cell_<tag>.json; that default name encodes only the tag, so main()
refuses to overwrite an existing one whose recorded model / scaffold / cell /
solo_models / comparator differ from this run's (pass --out for a second
readout of the same tag; a refresh of the same readout overwrites as before).
Per-model readouts other than the registered grok cell are exploratory, not
registered.

Usage
-----
  python -m scripts.analyze_topology_2x2 --selftest
  python -m scripts.analyze_topology_2x2            # all cells present on disk
  python -m scripts.analyze_topology_2x2 --tag cg_deliberation_nano --model gpt-5.4-nano
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
DEFAULT_DRAWS = 3000
DEFAULT_SEED = 29
#: single-cell mode: the matched readout intersects with this run's items
REFERENCE_TAG = "cg_deliberation"
SINGLE_CELL_DEFAULT = ("embodied", "on")

#: (roles, edges) -> (tag, seat role ids)
CELL_SPECS = {
    ("embodied", "on"): ("cg_deliberation",
                         ("writer_advocate", "counterparty", "neutral_adjudicator")),
}
for _cell, _spec in CELLS.items():
    CELL_SPECS[_cell] = (_spec["tag"], tuple(r.role_id for r in build_roles(*_cell)))
#: Addendum 16.16: the embodied cell with plain chain-of-thought seats. Same
#: roles, same edges, same panel; only the seat system prompt differs, so its
#: r0 caches live under scaffold "standard_cot" (cgd_<model>_standard_cot_*).
CELL_SPECS[("embodied_stdcot", "on")] = ("cg_deliberation_stdcot",
                                         ("writer_advocate", "counterparty", "neutral_adjudicator"))
CELL_SCAFFOLD = {c: "narrative_cot" for c in CELL_SPECS}
CELL_SCAFFOLD[("embodied_stdcot", "on")] = "standard_cot"


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


def parse_model_scaffold(s: str) -> tuple[str, str]:
    """'model:scaffold' -> (model, scaffold); the scaffold defaults to standard."""
    m, _, sc = s.strip().partition(":")
    if not m:
        raise ValueError(f"bad model:scaffold spec {s!r}")
    return m, (sc or "standard")


#: The fields of a single-cell artefact that define WHICH readout it is. The
#: default output name encodes only the tag, so two exploratory runs on one
#: tag with different --model / --cell / --scaffold / --solo-models /
#: --extra-comparator would silently overwrite each other; main() refuses
#: that unless --out names the destination explicitly.
SINGLE_CELL_IDENTITY = ("model", "scaffold", "cell", "solo_models", "comparator")


def single_cell_overwrite_conflict(path: Path, res: dict) -> Optional[str]:
    """None when `path` may be (over)written with `res`: it does not exist,
    is not a readable single-cell artefact, or records the same identity
    fields. Otherwise the fields that differ, for the refusal message."""
    if not path.exists():
        return None
    try:
        old = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(old, dict) or old.get("mode") != "single_cell":
        return None
    diff = [f"{k}: on disk {old.get(k)!r} vs this run {res.get(k)!r}"
            for k in SINGLE_CELL_IDENTITY if old.get(k) != res.get(k)]
    return "; ".join(diff) if diff else None


def load_merged_comparator(paths: Sequence[Path]) -> dict[tuple, list[str]]:
    """load_comparator over several CSVs, samples concatenated per key."""
    out: dict[tuple, list[str]] = defaultdict(list)
    for p in paths:
        for k, vs in load_comparator(p).items():
            out[k].extend(vs)
    return out


def r0_verdict_code(arm: str, item: str, idx: int, role_id: str,
                    scaffold: str = "narrative_cot", model: str = MODEL) -> Optional[int]:
    p = rcd.call_cache_path(model, scaffold, arm, item, idx, "r0", role_id, R0_CAP)
    if not p.exists():
        return None
    txt = json.loads(p.read_text()).get("output") or ""
    return code_response(extract_verdict(txt, rcd.INSTRUMENT), "published")


def g3_stake_concentration(votes_path: Path, keep_items: Optional[set], *, draws: int, seed: int,
                           model: str = MODEL) -> dict:
    """R4 reject rate | own stake undermined minus | not, stake seats only.
    n/a when no seat carries a stake (identical cells)."""
    sv = []
    with open(votes_path) as f:
        for v in csv.DictReader(f):
            if v.get("model") != model:
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


def _fired(r: dict) -> bool:
    return r["n_objectors"] >= 2


def _transfer_block(rows: Sequence[dict], solo, *, draws: int, seed: int) -> dict:
    """P(solo wrong | cell fired) vs not, at the (arm, item) cell level; a cell
    fires when strictly more than half of its debates fire (stake-blind
    counter). `solo` is a make_rule_vendor rule the cell never touched."""
    fl_c = defaultdict(int); k_c = defaultdict(int)
    for r in rows:
        c = (r["arm"], r["item"]); fl_c[c] += int(_fired(r)); k_c[c] += 1
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
    t_p, t_lo, t_hi = _boot(byt, tstat, draws=draws, seed=seed)
    tf = [x for x in trec if x["f"]]; tu = [x for x in trec if not x["f"]]
    return {"n_cells": len(trec), "n_flagged_cells": len(tf),
            "p_wrong_flagged": (sum(x["w"] for x in tf) / len(tf) if tf else None),
            "p_wrong_unflagged": (sum(x["w"] for x in tu) / len(tu) if tu else None),
            "lift": t_p, "lo": t_lo, "hi": t_hi}


def analyse_cell(cell: tuple, *, keep_items: Optional[set], draws: int, seed: int,
                 model: str = MODEL, tag: Optional[str] = None,
                 scaffold: Optional[str] = None,
                 solo_models: Optional[Sequence[tuple[str, str]]] = None,
                 comp: Optional[dict] = None) -> Optional[dict]:
    """One cell through the 16.10 code path. Grid path (tag, scaffold,
    solo_models, comp all None): run and r0 scaffold from CELL_SPECS /
    CELL_SCAFFOLD, the grok standard solo from the k3 comparator under
    `transfer_grok_solo`. Single-cell mode overrides the run (`tag`), the
    panel model and the r0 cache scaffold, and writes one transfer block per
    solo model under `transfer_solo` ("model:scaffold" -> block)."""
    spec_tag, seats = CELL_SPECS[cell]
    tag = spec_tag if tag is None else tag
    scaffold = CELL_SCAFFOLD.get(cell, "narrative_cot") if scaffold is None else scaffold
    rows_path, votes_path = OUT_DIR / f"{tag}_rows.csv", OUT_DIR / f"{tag}_votes.csv"
    if not rows_path.exists() or not votes_path.exists():
        return None
    rows = load_rows(rows_path, model, keep_items)
    if not rows:
        return None
    rates = per_seat_objection_rates(votes_path, model, keep_items)
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
            c = r0_verdict_code(r["arm"], r["item"], r["sample_idx"], s, scaffold, model)
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
    fired = _fired
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

    # ---- transfer to the solo model(s) (cached standard maj-3), which no cell touched ----
    if comp is None:
        comp = load_comparator(COMPARATOR_PATH)
    if solo_models is None:
        transfer = {"transfer_grok_solo": _transfer_block(
            rows, make_rule_vendor(comp, model, "standard"), draws=draws, seed=seed + 2)}
    else:
        transfer = {"transfer_solo": {
            f"{sm}:{ss}": _transfer_block(rows, make_rule_vendor(comp, sm, ss), draws=draws, seed=seed + 2)
            for sm, ss in solo_models}}

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
        "g3": g3_stake_concentration(votes_path, keep_items, draws=draws, seed=seed + 4, model=model),
        "lift": {"n_fired": n_fired_raw, "powered": n_fired_raw >= MIN_FIRED, "codable": lc,
                 "delta": l_p, "lo": l_lo, "hi": l_hi},
        **transfer,
        "guards": {"mean_n_r3_unparsed": sum(r["n_r3_unparsed"] for r in rows) / n,
                   "p_noncodable_given_fired": noncod_f / n_fired_raw if n_fired_raw else None},
    }


def reference_items(tag: str = REFERENCE_TAG, model: str = MODEL) -> set:
    p = OUT_DIR / f"{tag}_rows.csv"
    return {r["item"] for r in load_rows(p, model)} if p.exists() else set()


def analyse_single_cell(tag: str, model: str, *, scaffold: str = "narrative_cot",
                        cell: tuple = SINGLE_CELL_DEFAULT,
                        solo_models: Optional[Sequence[tuple[str, str]]] = None,
                        comp: Optional[dict] = None, draws: int, seed: int) -> Optional[dict]:
    """Single-cell mode: the cell on its own items (seed + 5, the grid's
    cells_unmatched_full convention for cell 0) and on the intersection with
    the grok cg_deliberation items (seed, the grid's matched convention)."""
    rows_path = OUT_DIR / f"{tag}_rows.csv"
    if not rows_path.exists() or not (OUT_DIR / f"{tag}_votes.csv").exists():
        return None
    own_items = {r["item"] for r in load_rows(rows_path, model)}
    ref = reference_items()
    matched = own_items & ref
    solo_models = [(model, "standard")] if solo_models is None else list(solo_models)
    kw = dict(model=model, tag=tag, scaffold=scaffold, solo_models=solo_models, comp=comp, draws=draws)
    return {
        "mode": "single_cell", "tag": tag, "model": model, "scaffold": scaffold,
        "cell": f"{cell[0]}/{cell[1]}", "seats": list(CELL_SPECS[cell][1]),
        "solo_models": [f"{sm}:{ss}" for sm, ss in solo_models],
        "reference": {"tag": REFERENCE_TAG, "model": MODEL, "n_items": len(ref)},
        "registered": (tag == REFERENCE_TAG and model == MODEL),
        "n_items_own": len(own_items), "n_items_matched": len(matched),
        "draws": draws, "seed": seed,
        "matched": analyse_cell(cell, keep_items=matched, seed=seed, **kw),
        "own": analyse_cell(cell, keep_items=None, seed=seed + 5, **kw),
    }


def _f(x, fmt="+.3f"):
    return "n/a" if x is None else format(x, fmt)


def _table(cells: Sequence[Optional[dict]], hdr: Sequence[str], names: Sequence[str],
           transfers: Sequence[tuple]) -> None:
    """The per-cell rows shared by the grid and the single-cell report.
    transfers: (label, getter(cell) -> transfer block) per solo readout."""
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
    for label, tget in transfers:
        row(f"TRANSFER lift on {label}", lambda c: tget(c)["lift"])
        row("  P(solo wrong | flagged)", lambda c: tget(c)["p_wrong_flagged"], ".3f")
        row("  P(solo wrong | not)", lambda c: tget(c)["p_wrong_unflagged"], ".3f")
    row("guard: mean n_r3_unparsed", lambda c: c["guards"]["mean_n_r3_unparsed"], ".4f")
    print()
    for c, name in zip(cells, names):
        if not c:
            continue
        print(f"  {name}: seats {c['seats']}")
        print(f"     P(at_fault) " + "  ".join(f"{s.split('_')[0][:8]}={_f(v, '.3f')}" for s, v in c["p_at_fault"].items()))
        print(f"     lift CI [{_f(c['lift']['lo'])}, {_f(c['lift']['hi'])}]   "
              + "".join(f"transfer CI [{_f(tget(c)['lo'])}, {_f(tget(c)['hi'])}]   " for _, tget in transfers)
              + f"majority-best CI [{_f(c['aggregation']['majority_minus_best_seat']['lo'])}, "
              f"{_f(c['aggregation']['majority_minus_best_seat']['hi'])}]")


def print_grid(res: dict) -> None:
    order = [("embodied", "on"), ("embodied", "off"), ("identical", "on"), ("identical", "off"),
             ("embodied_stdcot", "on")]
    cells = [res["cells"].get(f"{r}/{e}") for r, e in order]
    hdr = [("stdcot/on" if r == "embodied_stdcot" else f"{r}/{e}") for r, e in order]
    print("\n" + "=" * 96)
    print(f"ADDENDUM 16.10 -- ROLE x CONNECTIVITY 2x2   (matched items: {res['n_items_matched']})")
    print("=" * 96)
    _table(cells, hdr, [f"{r}/{e}" for r, e in order],
           [("grok solo", lambda c: c["transfer_grok_solo"])])


def print_cell(res: dict) -> None:
    print("\n" + "=" * 96)
    print(f"SINGLE CELL {res['cell']} -- run {res['tag']}, panel model {res['model']}, "
          f"r0 scaffold {res['scaffold']}")
    print(f"  {'registered (the 16.10 grok cell)' if res['registered'] else 'EXPLORATORY, not registered'}; "
          f"items own {res['n_items_own']}, matched with {res['reference']['tag']} "
          f"({res['reference']['model']}, {res['reference']['n_items']} items): {res['n_items_matched']}")
    print("=" * 96)
    transfers = [(k, (lambda k_: (lambda c: c["transfer_solo"][k_]))(k)) for k in res["solo_models"]]
    _table([res["own"], res["matched"]], ["own items", "matched"], ["own items", "matched"], transfers)


def _selftest() -> int:
    import contextlib
    import inspect
    import io
    import tempfile
    fails = []
    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond: fails.append(name)
    check("four 2x2 cells plus the 16.16 stdcot cell specified", len(CELL_SPECS) == 5)
    check("embodied/on maps to the registered run", CELL_SPECS[("embodied", "on")][0] == "cg_deliberation")
    ids = [s for _, seats in CELL_SPECS.values() for s in seats]
    check("12 distinct seat ids across the four cells", len(set(ids)) == 12)
    check("identical cells carry no stake seats -> G3 must report not applicable",
          all(r.stake == "none" for r in build_roles("identical", "on")))
    # G3 on a synthetic votes file with and without stakes
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
        check("G3 filters on the panel model (another model sees no stake votes)",
              not g3_stake_concentration(vp, None, draws=10, seed=1, model="other-model")["applicable"])
        with open(vp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "item_id", "role_id", "role_stake", "stake_undermined", "reject"])
            w.writeheader()
            w.writerow({"model": MODEL, "item_id": "i0", "role_id": "n", "role_stake": "none",
                        "stake_undermined": "", "reject": "0"})
        check("G3 reports not-applicable when no seat has a stake",
              not g3_stake_concentration(vp, None, draws=10, seed=1)["applicable"])
        # comparator merge: samples concatenate per (model, scaffold, arm, item)
        c1, c2 = Path(td) / "c1.csv", Path(td) / "c2.csv"
        for p, vs in ((c1, ["YTA", "YTA"]), (c2, ["NTA", "NTA", "NTA"])):
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["model", "scaffold", "arm", "item_id", "verdict"])
                w.writeheader()
                for v in vs:
                    w.writerow({"model": "m", "scaffold": "standard", "arm": "as_asker", "item_id": "x", "verdict": v})
        merged = load_merged_comparator([c1, c2])
        check("merged comparator concatenates samples across files (2 + 3 -> NTA majority)",
              len(merged[("m", "standard", "as_asker", "x")]) == 5
              and make_rule_vendor(merged, "m", "standard")({"arm": "as_asker", "item": "x"}) == "NTA"
              and make_rule_vendor(load_merged_comparator([c1]), "m", "standard")({"arm": "as_asker", "item": "x"}) == "YTA")
    check("model:scaffold parsing (scaffold defaults to standard)",
          parse_model_scaffold("gpt-5.4-nano:standard") == ("gpt-5.4-nano", "standard")
          and parse_model_scaffold("claude-haiku-4-5") == ("claude-haiku-4-5", "standard"))
    # single-cell default output name encodes only the tag: a different
    # readout of the same tag must NOT silently overwrite it without --out
    with tempfile.TemporaryDirectory() as td:
        art_p = Path(td) / "topology_cell_t.json"
        base = {"mode": "single_cell", "tag": "t", "model": "gpt-5.4-nano", "scaffold": "narrative_cot",
                "cell": "embodied/on", "solo_models": ["gpt-5.4-nano:standard"],
                "comparator": ["c.csv"], "draws": 3000, "seed": 29}
        check("overwrite guard: absent file, unreadable file and a non-single-cell JSON are writable",
              single_cell_overwrite_conflict(art_p, base) is None
              and (art_p.write_text("not json") or single_cell_overwrite_conflict(art_p, base) is None)
              and (art_p.write_text(json.dumps({"cells": {}})) or single_cell_overwrite_conflict(art_p, base) is None))
        art_p.write_text(json.dumps(base))
        check("overwrite guard: the SAME identity (model, scaffold, cell, solo_models, comparator) may "
              "overwrite (a refresh with other draws/seed is not a different readout)",
              single_cell_overwrite_conflict(art_p, dict(base, draws=10, seed=1)) is None)
        for k, v in (("cell", "identical/on"), ("model", "claude-haiku-4-5"), ("scaffold", "standard_cot"),
                     ("solo_models", ["gpt-5.4-nano:standard", "claude-haiku-4-5:standard"]),
                     ("comparator", ["c.csv", "extra.csv"])):
            c = single_cell_overwrite_conflict(art_p, dict(base, **{k: v}))
            check(f"overwrite guard: a different --{k.replace('_', '-')} on the same tag is refused and named",
                  c is not None and c.startswith(f"{k}:"))
        check("main() refuses the conflicting overwrite before writing (source: conflict check precedes write_text)",
              "single_cell_overwrite_conflict(out, res)" in inspect.getsource(main)
              and inspect.getsource(main).index("single_cell_overwrite_conflict(out, res)")
              < inspect.getsource(main).index("out.write_text(json.dumps(res"))
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
        # single-cell mode with an explicit solo list writes transfer_solo, same numbers
        emb2 = analyse_cell(("embodied", "on"), keep_items=None, draws=200, seed=1,
                            tag="cg_deliberation", model=MODEL, scaffold="narrative_cot",
                            solo_models=[(MODEL, "standard"), ("claude-haiku-4-5", "standard")])
        check("single-cell path: transfer_solo[grok:standard] == the grid's transfer_grok_solo, "
              "plus a second solo block",
              emb2["transfer_solo"][f"{MODEL}:standard"] == emb["transfer_grok_solo"]
              and "transfer_grok_solo" not in emb2
              and emb2["transfer_solo"]["claude-haiku-4-5:standard"]["n_cells"] > 0
              and {k: v for k, v in emb2.items() if k != "transfer_solo"}
              == {k: v for k, v in emb.items() if k != "transfer_grok_solo"})
        check("a model with no rows in the run yields no cell (not a silent zero)",
              analyse_cell(("embodied", "on"), keep_items=None, draws=10, seed=1,
                           tag="cg_deliberation", model="no-such-model") is None)
    else:
        check("embodied/on cell present on disk", False)
    # --- single-cell mode on the registered grok cell reproduces the grid's
    #     embodied/on numbers EXACTLY (matched at seed 29, own at seed 34; 3000 draws)
    if ANALYSIS_PATH.exists() and emb:
        art = json.loads(ANALYSIS_PATH.read_text())
        single = analyse_single_cell("cg_deliberation", MODEL, draws=DEFAULT_DRAWS, seed=DEFAULT_SEED)
        for key, art_key in (("matched", "cells"), ("own", "cells_unmatched_full")):
            got = json.loads(json.dumps(single[key], default=str))
            exp = art[art_key]["embodied/on"]
            got_t = got.pop("transfer_solo")[f"{MODEL}:standard"]
            exp_t = exp["transfer_grok_solo"]
            exp_rest = {k: v for k, v in exp.items() if k != "transfer_grok_solo"}
            check(f"single-cell {key} readout == artefact {art_key}[embodied/on], every field", got == exp_rest)
            check(f"single-cell {key} transfer block == artefact transfer_grok_solo", got_t == exp_t)
        check("single-cell metadata: registered grok cell, 210 own == 210 matched items",
              single["registered"] and single["n_items_own"] == 210 == single["n_items_matched"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_cell(single)
        check("print_cell renders both columns and the transfer row", "matched" in buf.getvalue()
              and f"TRANSFER lift on {MODEL}:standard" in buf.getvalue())
        # --- grid path regression: the default invocation is byte-identical
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "grid.json"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = main(["--out", str(out)])
            check("2x2 grid path (no --tag) reruns BYTE-IDENTICAL to topology_2x2_analysis.json",
                  rc == 0 and out.read_bytes() == ANALYSIS_PATH.read_bytes())
    else:
        check("topology_2x2_analysis.json present for the single-cell regression", False)
    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Addendum 16.10 2x2 grid, or one cell per model (--tag)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-boot", type=int, default=DEFAULT_DRAWS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", type=Path, default=None,
                    help="grid: topology_2x2_analysis.json; single cell: topology_cell_<tag>.json")
    sc = ap.add_argument_group("single-cell mode (per-model readout)")
    sc.add_argument("--tag", default=None, help="run tag, e.g. cg_deliberation_nano (<tag>_rows/_votes.csv)")
    sc.add_argument("--model", default=None, help="panel model in the rows/votes and r0 caches")
    sc.add_argument("--scaffold", default=None, help="r0 cache scaffold (default narrative_cot)")
    sc.add_argument("--cell", default=None, help="roles/edges topology for the seats (default embodied/on)")
    sc.add_argument("--solo-models", default=None,
                    help="comma list of model:scaffold solo comparators (default MODEL:standard)")
    sc.add_argument("--extra-comparator", type=Path, nargs="+", action="extend", default=None,
                    help="CSVs merged into cg_scaffold_combined_rows_k3.csv (load_comparator format)")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    single_opts = (a.model, a.scaffold, a.cell, a.solo_models, a.extra_comparator)
    if a.tag is None:
        if any(o is not None for o in single_opts):
            ap.error("--model/--scaffold/--cell/--solo-models/--extra-comparator need --tag (single-cell mode)")
        return _main_grid(a)
    if a.model is None:
        ap.error("--tag needs --model")
    cell = SINGLE_CELL_DEFAULT if a.cell is None else tuple(a.cell.split("/", 1))
    if cell not in CELL_SPECS:
        ap.error(f"--cell must be one of {sorted(f'{r}/{e}' for r, e in CELL_SPECS)}")
    solo = None if a.solo_models is None else [parse_model_scaffold(s) for s in a.solo_models.split(",") if s.strip()]
    comp_paths = [COMPARATOR_PATH] + list(a.extra_comparator or [])
    comp = load_merged_comparator(comp_paths)
    res = analyse_single_cell(a.tag, a.model, scaffold=a.scaffold or "narrative_cot", cell=cell,
                              solo_models=solo, comp=comp, draws=a.n_boot, seed=a.seed)
    if res is None:
        print(f"no rows/votes on disk for tag {a.tag}"); return 2
    if res["matched"] is None and res["own"] is None:
        print(f"no rows for model {a.model} in {a.tag}_rows.csv"); return 2
    res["comparator"] = [str(p) for p in comp_paths]
    print_cell(res)
    out = a.out if a.out is not None else OUT_DIR / f"topology_cell_{a.tag}.json"
    if a.out is None:
        conflict = single_cell_overwrite_conflict(out, res)
        if conflict:
            print(f"\nERROR: {out} already holds a different single-cell readout of tag {a.tag} "
                  f"({conflict}); the default output name encodes only the tag, so pass --out "
                  "to write this readout elsewhere (nothing was written)")
            return 2
    out.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


def _main_grid(a) -> int:
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
    out = a.out if a.out is not None else ANALYSIS_PATH
    out.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
