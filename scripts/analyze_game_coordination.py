"""
scripts/analyze_game_coordination.py -- Addendum 16.13d: the mediator where
Aumann says it matters. Multiple-equilibrium games (CONFLICT_OF_EQUILIBRIA,
nine TMGBench games with two pure Nash cells, each favouring a different
player), NON-BINDING regime, both third-seat arms plus the solo comparator.

Reads the nb runner's rows / votes CSVs (scripts/run_crowdgold_game_nb.py)
and the solo non-binding rows (scripts/run_game_singleagent.py --regime
nonbinding). Recommendations are mapped through the row's own
`nash_verdicts` column (display-indexed, "|"-separated) and the canonical
payoffs from scripts.tmg_games via the row's row_perm / col_perm.

Pre-declared readouts (16.13d), item-clustered bootstrap over the 9 games:
  COORD    P(group recommendation is a pure Nash cell), per arm and solo
  WELFARE  P(group recommendation is the welfare-maximising Nash cell)
  ACCEPT   advocate r4 ACCEPT share on equilibrium recommendations, split by
           whether the recommended equilibrium favours that advocate's player
  LOCK     P(advocate at r0 names the action of the equilibrium favouring its
           own player)
  FLAG     stake-blind (n_objectors >= 2) fire rate; lift if >= 20 fired
Readings D1 (planner - neutral on COORD), D2 (arm - solo on COORD), D3
(disadvantaged advocate's ACCEPT rate on the other player's equilibrium).

Usage
-----
  python -m scripts.analyze_game_coordination --selftest
  python -m scripts.analyze_game_coordination --neutral-tag cgnb_game_neutral_d \
      --planner-tag cgnb_game_planner_d --solo-rows divergence_study_outputs/game_solo_nb_rows.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

from scripts.run_phase1_quartet import OUT_DIR
from scripts.tmg_games import cell_for_plan, load_games

CONFLICT_IDS = ("122", "123", "124", "132", "133", "134", "142", "143", "144")
ANALYSIS_PATH = OUT_DIR / "game_coordination_analysis.json"
GUARD_MAX_NOVERDICT = 0.05


def _perm(s: str) -> tuple[int, int]:
    return (int(s[0]), int(s[1]))


def _welfare_best_nash(pA, pB, nash_cells) -> set:
    w = {c: pA[c[0]][c[1]] + pB[c[0]][c[1]] for c in nash_cells}
    m = max(w.values())
    return {c for c, v in w.items() if v == m}


def load_rows(path: Path) -> list[dict]:
    games = {g.game_id: g for g in load_games()}
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            gid = str(r["game_id"])
            if gid not in CONFLICT_IDS:
                continue
            g = games[gid]
            rp, cp = _perm(r["row_perm"]), _perm(r["col_perm"])
            nash_disp = {v for v in (r.get("nash_verdicts") or "").split("|") if v}
            nash_canon = set(g.nash_cells)
            wbest = _welfare_best_nash(g.pA, g.pB, g.nash_cells)
            # which equilibrium favours which player (canonical cells)
            row_pref = max(nash_canon, key=lambda c: g.pA[c[0]][c[1]])
            col_pref = max(nash_canon, key=lambda c: g.pB[c[0]][c[1]])
            v = r["verdict"]
            cell = cell_for_plan(v, rp, cp) if v.startswith("PLAN_") else None
            rec = {
                "game_id": gid, "sample_idx": int(r["sample_idx"]), "verdict": v,
                "cell": cell, "is_nash": cell in nash_canon if cell else False,
                "is_welfare_best": cell in wbest if cell else False,
                "favours": ("row" if cell == row_pref else "col" if cell == col_pref else None) if cell in nash_canon else None,
                "noverdict": v == "NOVERDICT",
                "n_objectors": int(r.get("n_objectors") or 0),
                "r0_row": r.get("r0_verdict_row_advocate", ""), "r0_col": r.get("r0_verdict_col_advocate", ""),
                "row_pref_action_disp": None, "col_pref_action_disp": None,
            }
            # the DISPLAYED action index of each player's preferred equilibrium
            inv_r = {rp[i]: i for i in range(2)}; inv_c = {cp[i]: i for i in range(2)}
            rec["row_pref_action_disp"] = f"ACT_{inv_r[row_pref[0]] + 1}"
            rec["col_pref_action_disp"] = f"ACT_{inv_c[col_pref[1]] + 1}"
            rec["nash_disp_check"] = (nash_disp == {f"PLAN_{inv_r[c[0]]+1}{inv_c[c[1]]+1}" for c in nash_canon}) if nash_disp else None
            out.append(rec)
    return out


def load_votes(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if str(r["item_id"]) not in CONFLICT_IDS:
                continue
            # nb CSV: `vote` holds the token (ACCEPT / REJECT / NOVERDICT), `vote_parsed` is a 0/1 flag
            tok = (r.get("vote") or "").strip().upper()
            out.append({"game_id": str(r["item_id"]), "sample_idx": int(r["sample_idx"]), "role_id": r["role_id"],
                        "group_verdict": r["group_verdict"], "vote": tok,
                        "vote_parsed": tok if r.get("vote_parsed", "1") in ("1", "True", "true", tok) else "NOVERDICT",
                        "objected_r3": r.get("objected_r3", "")})
    return out


def load_solo(path: Path) -> list[dict]:
    games = {g.game_id: g for g in load_games()}
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            gid = str(r["item_id"])
            if gid not in CONFLICT_IDS or r.get("masked", "") in ("1", "True", "true"):
                continue
            g = games[gid]; rp, cp = _perm(r["row_perm"]), _perm(r["col_perm"])
            v = r["verdict"]; cell = cell_for_plan(v, rp, cp) if v.startswith("PLAN_") else None
            out.append({"game_id": gid, "sample_idx": int(r["sample_idx"]), "verdict": v,
                        "is_nash": cell in set(g.nash_cells) if cell else False,
                        "is_welfare_best": cell in _welfare_best_nash(g.pA, g.pB, g.nash_cells) if cell else False,
                        "noverdict": v == "NOVERDICT"})
    return out


def boot(stat, items: Sequence[str], *, draws: int, seed: int):
    rng = random.Random(seed)
    pt = stat(list(items))
    vals = sorted(v for v in (stat([items[rng.randrange(len(items))] for _ in items]) for _ in range(draws)) if v is not None)
    if not vals:
        return {"point": pt, "lo": None, "hi": None}
    return {"point": pt, "lo": vals[int(.025 * len(vals))], "hi": vals[min(int(.975 * len(vals)), len(vals) - 1)]}


def _share(recs, key):
    by = defaultdict(list)
    for r in recs:
        by[r["game_id"]].append(1.0 if r[key] else 0.0)
    def stat(items):
        xs = [x for g in items for x in by.get(g, [])]
        return sum(xs) / len(xs) if xs else None
    return stat


def _diff(recs_a, recs_b, key):
    a, b = _share(recs_a, key), _share(recs_b, key)
    return lambda items: (None if a(items) is None or b(items) is None else a(items) - b(items))


def analyze(neutral_tag: str, planner_tag: str, solo_rows: Optional[Path], *, draws: int = 2000, seed: int = 7) -> dict:
    arms = {}
    for name, tag in (("neutral", neutral_tag), ("planner", planner_tag)):
        rows = load_rows(OUT_DIR / f"{tag}_rows.csv"); votes = load_votes(OUT_DIR / f"{tag}_votes.csv")
        arms[name] = {"rows": rows, "votes": votes}
    solo = load_solo(solo_rows) if solo_rows else []
    items = sorted({r["game_id"] for a in arms.values() for r in a["rows"]})
    res = {"games": items, "n_items": len(items), "guard": {}, "arms": {}, "solo": {}, "contrasts": {}, "readings": {}}

    # guard: NOVERDICT share on the group verdict and on r4 votes
    for name, a in arms.items():
        nov_group = sum(r["noverdict"] for r in a["rows"]) / max(1, len(a["rows"]))
        nov_vote = sum(1 for v in a["votes"] if v["vote_parsed"] in ("", "NOVERDICT")) / max(1, len(a["votes"]))
        ok = nov_group <= GUARD_MAX_NOVERDICT and nov_vote <= GUARD_MAX_NOVERDICT
        res["guard"][name] = {"n_cells": len(a["rows"]), "noverdict_group": nov_group, "noverdict_vote": nov_vote, "pass": ok,
                              "nash_column_consistent": all(r["nash_disp_check"] in (True, None) for r in a["rows"])}
    res["guard"]["all_pass"] = all(g["pass"] for g in res["guard"].values() if isinstance(g, dict))
    if not res["guard"]["all_pass"]:
        return res

    for name, a in arms.items():
        rows, votes = a["rows"], a["votes"]
        # ACCEPT split by whether the recommended equilibrium favours the voter's player
        acc = {"favours_own": [], "favours_other": []}
        rowmap = {(r["game_id"], r["sample_idx"]): r for r in rows}
        for v in votes:
            r = rowmap.get((v["game_id"], v["sample_idx"]))
            if r is None or not r["is_nash"] or v["vote_parsed"] not in ("ACCEPT", "REJECT"):
                continue
            if v["role_id"].startswith("player_r"):
                own = r["favours"] == "row"
            elif v["role_id"].startswith("player_c"):
                own = r["favours"] == "col"
            else:
                continue
            acc["favours_own" if own else "favours_other"].append({"game_id": v["game_id"], "ok": v["vote_parsed"] == "ACCEPT"})
        lock = [{"game_id": r["game_id"], "ok": r["r0_row"] == r["row_pref_action_disp"]} for r in rows] + \
               [{"game_id": r["game_id"], "ok": r["r0_col"] == r["col_pref_action_disp"]} for r in rows]
        fired = [r for r in rows if r["n_objectors"] >= 2]
        res["arms"][name] = {
            "n_cells": len(rows),
            "COORD": boot(_share(rows, "is_nash"), items, draws=draws, seed=seed),
            "WELFARE": boot(_share(rows, "is_welfare_best"), items, draws=draws, seed=seed + 1),
            "NO_RECOMMENDATION_share": sum(r["verdict"] == "NO_RECOMMENDATION" for r in rows) / len(rows),
            "ACCEPT_favours_own": boot(_share(acc["favours_own"], "ok"), items, draws=draws, seed=seed + 2) if acc["favours_own"] else None,
            "ACCEPT_favours_other": boot(_share(acc["favours_other"], "ok"), items, draws=draws, seed=seed + 3) if acc["favours_other"] else None,
            "n_votes_favours_own": len(acc["favours_own"]), "n_votes_favours_other": len(acc["favours_other"]),
            "LOCK_r0_prefers_own_equilibrium": boot(_share(lock, "ok"), items, draws=draws, seed=seed + 4),
            "FLAG_fire_rate": len(fired) / len(rows), "FLAG_n_fired": len(fired),
            "FLAG_lift": (boot(lambda its: (sum(1 for r in fired if r["game_id"] in its and not r["is_nash"]) / max(1, sum(1 for r in fired if r["game_id"] in its))
                                          - sum(1 for r in rows if r["game_id"] in its and r["n_objectors"] < 2 and not r["is_nash"]) / max(1, sum(1 for r in rows if r["game_id"] in its and r["n_objectors"] < 2))),
                               items, draws=draws, seed=seed + 5) if len(fired) >= 20 else "under-gated (<20 fired)"),
        }
    if solo:
        res["solo"] = {"n_rows": len(solo), "COORD": boot(_share(solo, "is_nash"), items, draws=draws, seed=seed + 6),
                       "WELFARE": boot(_share(solo, "is_welfare_best"), items, draws=draws, seed=seed + 7),
                       "NO_RECOMMENDATION_share": sum(r["verdict"] == "NO_RECOMMENDATION" for r in solo) / len(solo)}
    N, P = arms["neutral"]["rows"], arms["planner"]["rows"]
    res["contrasts"]["D1 planner - neutral, COORD"] = boot(_diff(P, N, "is_nash"), items, draws=draws, seed=seed + 8)
    res["contrasts"]["planner - neutral, WELFARE"] = boot(_diff(P, N, "is_welfare_best"), items, draws=draws, seed=seed + 9)
    if solo:
        res["contrasts"]["D2 neutral - solo, COORD"] = boot(_diff(N, solo, "is_nash"), items, draws=draws, seed=seed + 10)
        res["contrasts"]["D2 planner - solo, COORD"] = boot(_diff(P, solo, "is_nash"), items, draws=draws, seed=seed + 11)

    def excl0(c): return c["lo"] is not None and (c["lo"] > 0 or c["hi"] < 0)
    d1 = res["contrasts"]["D1 planner - neutral, COORD"]
    res["readings"]["D1 mediator as coordination device"] = "HOLDS" if (d1["point"] > 0 and excl0(d1)) else "fails"
    if solo:
        d2n, d2p = res["contrasts"]["D2 neutral - solo, COORD"], res["contrasts"]["D2 planner - solo, COORD"]
        res["readings"]["D2 collective vs solo"] = ("HOLDS (neutral)" if d2n["point"] > 0 and excl0(d2n) else "") + \
                                                   (" HOLDS (planner)" if d2p["point"] > 0 and excl0(d2p) else "") or "fails"
    for name in arms:
        ao = res["arms"][name]["ACCEPT_favours_other"]
        res["readings"][f"D3 disadvantaged advocate accepts other's equilibrium [{name}]"] = (
            "n/a" if ao is None else ("role-locked at the vote (<0.5)" if ao["point"] < 0.5 else "yields (>=0.9)" if ao["point"] >= 0.9 else "between"))
    return res


def _fmt(c):
    if c is None: return "n/a"
    if isinstance(c, str): return c
    return f"{c['point']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]" if c["lo"] is not None else f"{c['point']:.3f} [n/a]"


def print_report(res: dict) -> None:
    print("=" * 72); print(f"ADDENDUM 16.13d -- COORDINATION IN MULTIPLE-EQUILIBRIUM GAMES  (games {res['n_items']}: {', '.join(res['games'])})"); print("=" * 72)
    for name, g in res["guard"].items():
        if isinstance(g, dict):
            print(f"  {'GUARD OK' if g['pass'] else 'GUARD FAILED'} [{name}] cells {g['n_cells']} NOVERDICT group {g['noverdict_group']:.3f} votes {g['noverdict_vote']:.3f} nash column consistent {g['nash_column_consistent']}")
    if not res["guard"]["all_pass"]:
        print("  GUARD FAILED: numbers not printed"); return
    for name, a in res["arms"].items():
        print(f"\n  arm {name}: cells {a['n_cells']}")
        print(f"    COORD   P(recommendation is an equilibrium)      {_fmt(a['COORD'])}   NO_RECOMMENDATION share {a['NO_RECOMMENDATION_share']:.3f}")
        print(f"    WELFARE P(recommendation is the welfare-best eq) {_fmt(a['WELFARE'])}")
        print(f"    LOCK    P(advocate r0 names its own equilibrium)  {_fmt(a['LOCK_r0_prefers_own_equilibrium'])}")
        print(f"    ACCEPT  on eq favouring own player {_fmt(a['ACCEPT_favours_own'])} (n={a['n_votes_favours_own']})   favouring the other {_fmt(a['ACCEPT_favours_other'])} (n={a['n_votes_favours_other']})")
        print(f"    FLAG    fire rate {a['FLAG_fire_rate']:.3f} ({a['FLAG_n_fired']} fired)  lift {_fmt(a['FLAG_lift'])}")
    if res["solo"]:
        s = res["solo"]; print(f"\n  solo non-binding: rows {s['n_rows']}  COORD {_fmt(s['COORD'])}  WELFARE {_fmt(s['WELFARE'])}  NO_RECOMMENDATION share {s['NO_RECOMMENDATION_share']:.3f}")
    print()
    for k, c in res["contrasts"].items():
        print(f"  {k:<36} {_fmt(c)}")
    print()
    for k, v in res["readings"].items():
        print(f"  {k}: {v}")


def _selftest() -> int:
    fails = []
    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond: fails.append(name)
    games = {g.game_id: g for g in load_games()}
    for gid in CONFLICT_IDS:
        g = games[gid]
        check(f"{gid} has exactly two pure Nash cells favouring different players",
              len(g.nash_cells) == 2 and max(g.nash_cells, key=lambda c: g.pA[c[0]][c[1]]) != max(g.nash_cells, key=lambda c: g.pB[c[0]][c[1]]))
    g = games["122"]
    wb = _welfare_best_nash(g.pA, g.pB, g.nash_cells)
    check("welfare-best equilibrium of 122 is a subset of its Nash cells", wb <= set(g.nash_cells) and len(wb) >= 1)
    # synthetic rows: identity permutation, group picks (0,0) -> PLAN_11
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x_rows.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["game_id", "sample_idx", "row_perm", "col_perm", "nash_verdicts", "verdict", "n_objectors", "r0_verdict_row_advocate", "r0_verdict_col_advocate"])
            w.writeheader()
            w.writerow({"game_id": "122", "sample_idx": 0, "row_perm": "01", "col_perm": "01", "nash_verdicts": "PLAN_11|PLAN_22", "verdict": "PLAN_11", "n_objectors": 2, "r0_verdict_row_advocate": "ACT_1", "r0_verdict_col_advocate": "ACT_2"})
            w.writerow({"game_id": "122", "sample_idx": 1, "row_perm": "10", "col_perm": "10", "nash_verdicts": "PLAN_22|PLAN_11", "verdict": "PLAN_12", "n_objectors": 0, "r0_verdict_row_advocate": "ACT_2", "r0_verdict_col_advocate": "ACT_1"})
            w.writerow({"game_id": "122", "sample_idx": 2, "row_perm": "01", "col_perm": "01", "nash_verdicts": "PLAN_11|PLAN_22", "verdict": "NO_RECOMMENDATION", "n_objectors": 0, "r0_verdict_row_advocate": "ACT_1", "r0_verdict_col_advocate": "ACT_1"})
        rows = load_rows(p)
        check("identity permutation: PLAN_11 -> canonical (0,0), a Nash cell", rows[0]["cell"] == (0, 0) and rows[0]["is_nash"])
        check("swapped permutation: PLAN_12 -> canonical (1,0), NOT a Nash cell", rows[1]["cell"] == (1, 0) and not rows[1]["is_nash"])
        check("NO_RECOMMENDATION counts as failure to coordinate", not rows[2]["is_nash"] and rows[2]["cell"] is None)
        check("nash_verdicts column reproduced through the permutation on every synthetic row", all(r["nash_disp_check"] for r in rows))
        pref_row = max(g.nash_cells, key=lambda c: g.pA[c[0]][c[1]])
        check("row advocate's preferred-equilibrium action is display-indexed through the permutation",
              rows[0]["row_pref_action_disp"] == f"ACT_{pref_row[0] + 1}" and rows[1]["row_pref_action_disp"] == f"ACT_{2 - pref_row[0]}")
        st = _share(rows, "is_nash"); check("COORD share on the synthetic rows is 1/3", abs(st(["122"]) - 1 / 3) < 1e-9)
    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--neutral-tag", default="cgnb_game_neutral_d")
    ap.add_argument("--planner-tag", default="cgnb_game_planner_d")
    ap.add_argument("--solo-rows", type=Path, default=OUT_DIR / "game_solo_nb_rows.csv")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=ANALYSIS_PATH)
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    res = analyze(a.neutral_tag, a.planner_tag, a.solo_rows if a.solo_rows.exists() else None, draws=a.n_boot, seed=a.seed)
    print_report(res)
    a.out.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {a.out}")
    return 0 if res["guard"].get("all_pass") else 4


if __name__ == "__main__":
    raise SystemExit(main())
