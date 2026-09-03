"""
scripts/analyze_stake_grip.py -- Addendum 6: the stake-grip screen.

Grip (all three):
  G1  composite-flag fire rate <= 0.50
  G2  REJECT share of R4 votes >= 0.05
  G3  stake concentration on the stake seats (R4 reject rate | undermined
      minus | not undermined) >= +0.20, 95% item-clustered CI excludes 0

Also reports measured token spend for the run from its cache files, and
(--stage0) the exploratory nano sensor recalibration.

Usage
-----
  python -m scripts.analyze_stake_grip --selftest
  python -m scripts.analyze_stake_grip --tag cg_deliberation_haiku_screen --model claude-haiku-4-5
  python -m scripts.analyze_stake_grip --tag cg_deliberation --model grok-4-1-fast-reasoning
  python -m scripts.analyze_stake_grip --stage0
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, Sequence

import scripts.run_crowdgold_deliberation as rcd
from scripts.analyze_actuator_ladder import (
    evaluate, load_comparator, load_debates, make_rule_vendor,
)
from scripts.analyze_crowdgold_sdt import code_response
from scripts.analyze_loop_step import flagged, mislocalised, neutral_objected
from scripts.run_phase1_quartet import OUT_DIR, _safe

STAKE_SEATS = ("writer_advocate", "counterparty")
G1_MAX_FIRE, G2_MIN_REJECT, G3_MIN_CONC = 0.50, 0.05, 0.20


def load_votes(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            out.append({
                "item": r["item_id"], "arm": r["arm"], "role": r["role_id"],
                "sample_idx": int(r["sample_idx"]),
                "reject": int(r.get("reject") or 0),
                "undermined": int(r.get("stake_undermined") or 0),
                "objected": int(r.get("objected_r3") or 0),
                "r3_label": r.get("r3_label", ""),
            })
    return out


def _boot_ci(by_item: dict[str, list], stat: Callable[[list], Optional[float]],
             *, draws: int, seed: int, alpha: float = 0.05):
    ids = sorted(by_item)
    rng = random.Random(seed)
    point = stat([x for i in ids for x in by_item[i]])
    vals = sorted(v for v in (
        stat([x for i in (ids[rng.randrange(len(ids))] for _ in ids)
              for x in by_item[i]]) for _ in range(draws)) if v is not None)
    if not vals:
        return point, None, None
    return (point, vals[int(alpha / 2 * len(vals))],
            vals[min(int((1 - alpha / 2) * len(vals)), len(vals) - 1)])


def stake_concentration(votes: Sequence[dict], *, draws: int, seed: int) -> dict:
    sv = [v for v in votes if v["role"] in STAKE_SEATS]
    by_item: dict[str, list] = defaultdict(list)
    for v in sv:
        by_item[v["item"]].append(v)

    def stat(vs):
        u = [v["reject"] for v in vs if v["undermined"]]
        n = [v["reject"] for v in vs if not v["undermined"]]
        if not u or not n:
            return None
        return sum(u) / len(u) - sum(n) / len(n)

    p, lo, hi = _boot_ci(by_item, stat, draws=draws, seed=seed)
    return {
        "n_votes": len(sv),
        "n_undermined": sum(v["undermined"] for v in sv),
        "reject_rate_undermined": (sum(v["reject"] for v in sv if v["undermined"])
                                   / max(1, sum(v["undermined"] for v in sv))),
        "reject_rate_not_undermined": (sum(v["reject"] for v in sv if not v["undermined"])
                                       / max(1, sum(1 for v in sv if not v["undermined"]))),
        "delta": p, "lo": lo, "hi": hi,
    }


def paired_g3_delta(votes_a: Sequence[dict], votes_b: Sequence[dict], *,
                    draws: int, seed: int) -> dict:
    """Blocked-bootstrap delta of G3 (stake concentration) between two
    conditions -- e.g. nudged vs. this model's own Addendum-6 baseline --
    RESTRICTED to the item set they share, so the comparison is paired on
    items rather than on independent samples of each condition.

    Each bootstrap draw resamples item ids ONCE (with replacement) and
    evaluates the G3 statistic on BOTH conditions using that SAME resampled
    item list, so item-level noise that pushes both conditions the same way
    cancels in the difference. This is the Addendum 7 registered test.
    """
    sa = [v for v in votes_a if v["role"] in STAKE_SEATS]
    sb = [v for v in votes_b if v["role"] in STAKE_SEATS]
    ids = sorted(set(v["item"] for v in sa) & set(v["item"] for v in sb))
    by_a: dict[str, list] = defaultdict(list)
    by_b: dict[str, list] = defaultdict(list)
    for v in sa:
        if v["item"] in ids:
            by_a[v["item"]].append(v)
    for v in sb:
        if v["item"] in ids:
            by_b[v["item"]].append(v)

    def stat(vs):
        u = [v["reject"] for v in vs if v["undermined"]]
        n = [v["reject"] for v in vs if not v["undermined"]]
        if not u or not n:
            return None
        return sum(u) / len(u) - sum(n) / len(n)

    def g3_of(by_item, sel):
        return stat([x for i in sel for x in by_item[i]])

    rng = random.Random(seed)
    point_a, point_b = g3_of(by_a, ids), g3_of(by_b, ids)
    point = None if point_a is None or point_b is None else point_a - point_b
    diffs = []
    for _ in range(draws):
        sel = [ids[rng.randrange(len(ids))] for _ in ids]
        ga, gb = g3_of(by_a, sel), g3_of(by_b, sel)
        if ga is not None and gb is not None:
            diffs.append(ga - gb)
    diffs.sort()
    lo = hi = None
    if diffs:
        lo = diffs[int(0.025 * len(diffs))]
        hi = diffs[min(int(0.975 * len(diffs)), len(diffs) - 1)]
    return {
        "n_shared_items": len(ids), "n_boot_valid": len(diffs),
        "g3_a": point_a, "g3_b": point_b, "delta": point, "lo": lo, "hi": hi,
        "positive": bool(point is not None and lo is not None and lo > 0),
    }


def grip(debates: Sequence[dict], votes: Sequence[dict], *, draws: int = 2000,
         seed: int = 7) -> dict:
    n = len(debates)
    fire = sum(flagged(d) for d in debates) / n if n else None
    rej = sum(v["reject"] for v in votes) / len(votes) if votes else None
    conc = stake_concentration(votes, draws=draws, seed=seed)
    seat = {}
    for s in STAKE_SEATS + ("neutral_adjudicator",):
        sv = [v for v in votes if v["role"] == s]
        seat[s] = {
            "objected_r3": sum(v["objected"] for v in sv) / len(sv) if sv else None,
            "reject_r4": sum(v["reject"] for v in sv) / len(sv) if sv else None,
            "undermined": (sum(v["undermined"] for v in sv) / len(sv)
                           if sv and s in STAKE_SEATS else None),
        }
    g1 = fire is not None and fire <= G1_MAX_FIRE
    g2 = rej is not None and rej >= G2_MIN_REJECT
    g3 = (conc["delta"] is not None and conc["delta"] >= G3_MIN_CONC
          and conc["lo"] is not None and conc["lo"] > 0)
    return {
        "n_debates": n, "n_items": len({d["item"] for d in debates}),
        "n_votes": len(votes),
        "G1_fire_rate": fire, "G2_reject_share": rej, "G3_stake_concentration": conc,
        "per_seat": seat,
        "neutral_objected_rate": sum(neutral_objected(d) for d in debates) / n if n else None,
        "mislocalised_rate": sum(mislocalised(d) for d in debates) / n if n else None,
        "s2_accuracy": sum(d["s2_ok"] for d in debates) / n if n else None,
        "criteria": {"G1": g1, "G2": g2, "G3": g3},
        "grip": bool(g1 and g2 and g3),
    }


def measured_spend(model: str, items: set[str], *, price_in: float,
                   price_out: float) -> dict:
    pt = ct = k = 0
    for p in OUT_DIR.glob(f"cgd_{_safe(model)}_*"):
        if not any(i in p.name for i in items):
            continue
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        pt += r.get("prompt_tokens") or 0
        ct += r.get("completion_tokens") or 0
        k += 1
    return {"calls": k, "prompt_tokens": pt, "completion_tokens": ct,
            "usd": pt / 1e6 * price_in + ct / 1e6 * price_out,
            "prices_per_mtok": [price_in, price_out]}


# ---------------------------------------------------------------------------
# Stage 0: exploratory nano recalibration
# ---------------------------------------------------------------------------

def _revised(d):
    a, b = code_response(d["s1"], "published"), code_response(d["s2"], "published")
    return a is not None and b is not None and a != b


STRICT_SENSORS: dict[str, Callable[[dict], bool]] = {
    "composite (registered)": flagged,
    "all_three_object": lambda d: d["n_objectors"] >= 3,
    "neutral_AND_mislocalised": lambda d: neutral_objected(d) and mislocalised(d),
    "any_r4_reject": lambda d: d["n_reject"] >= 1,
    "objectors>=2_AND_revised": lambda d: d["n_objectors"] >= 2 and _revised(d),
    "revised_only": _revised,
}


def stage0(rows: Path, votes: Path, *, draws: int) -> dict:
    debates = load_debates(rows, votes)
    comp = load_comparator()
    haiku = make_rule_vendor(comp, "claude-haiku-4-5", "standard")
    grok = make_rule_vendor(comp, "grok-4-1-fast-reasoning", "standard")
    wrong = [d for d in debates if not d["syn_ok"]]
    out = {"n_debates": len(debates), "sensors": {}}
    for name, sig in STRICT_SENSORS.items():
        fl = [d for d in debates if sig(d)]
        un = [d for d in debates if not sig(d)]
        if not fl:
            out["sensors"][name] = {"fire_rate": 0.0}
            continue
        prec = sum(1 for d in fl if not d["syn_ok"]) / len(fl)
        rec = sum(1 for d in wrong if sig(d)) / len(wrong) if wrong else None
        p_un = (sum(1 for d in un if not d["syn_ok"]) / len(un)) if un else None
        eh = evaluate(fl, haiku, alpha=0.05, draws=draws, seed=11, label=name)
        eg = evaluate(fl, grok, alpha=0.05, draws=draws, seed=12, label=name)
        ehu = evaluate(un, haiku, alpha=0.05, draws=draws, seed=13, label=name) if un else None
        out["sensors"][name] = {
            "fire_rate": len(fl) / len(debates), "n": len(fl),
            "precision": prec, "recall": rec,
            "lift": (prec - p_un) if p_un is not None else None,
            "haiku_route_delta": eh["delta"], "haiku_lo": eh["lo"], "haiku_hi": eh["hi"],
            "haiku_unflagged_delta": ehu["delta"] if ehu else None,
            "grok_route_delta": eg["delta"], "grok_lo": eg["lo"], "grok_hi": eg["hi"],
        }
    return out


# ---------------------------------------------------------------------------
# Addendum 6a: confirm a discovery-set flag on fresh items
# ---------------------------------------------------------------------------

def confirm_flag(debates: Sequence[dict], discovery_items: set[str], *,
                 actuator_model: str = "grok-4-1-fast-reasoning",
                 draws: int = 2000, seed: int = 17) -> dict:
    comp = load_comparator()
    rule = make_rule_vendor(comp, actuator_model, "standard")
    new = [d for d in debates if d["item"] not in discovery_items]
    fl = [d for d in new if flagged(d)]
    un = [d for d in new if not flagged(d)]

    # C1: flag lift on synthesis error, item-clustered
    by_item: dict[str, list] = defaultdict(list)
    for d in new:
        by_item[d["item"]].append(d)

    def lift(ds):
        f = [int(not d["syn_ok"]) for d in ds if flagged(d)]
        u = [int(not d["syn_ok"]) for d in ds if not flagged(d)]
        if not f or not u:
            return None
        return sum(f) / len(f) - sum(u) / len(u)

    lp, llo, lhi = _boot_ci(by_item, lift, draws=draws, seed=seed)
    c1 = lp is not None and llo is not None and llo > 0
    ef = evaluate(fl, rule, alpha=0.05, draws=draws, seed=seed + 1,
                  label=f"{actuator_model} on new flagged") if fl else None
    eu = evaluate(un, rule, alpha=0.05, draws=draws, seed=seed + 2,
                  label=f"{actuator_model} on new unflagged") if un else None
    c2 = bool(ef and ef["lo"] is not None and ef["lo"] > 0
              and all(v is not None and v > 0 for v in ef["delta_by_arm"].values()))
    c3 = bool(ef and eu and ef["delta"] is not None and eu["delta"] is not None
              and ef["delta"] - eu["delta"] > 0 and eu["delta"] <= 0)
    return {
        "n_new_debates": len(new), "n_new_items": len({d["item"] for d in new}),
        "n_flagged": len(fl), "fire_rate": len(fl) / len(new) if new else None,
        "n_synthesis_wrong": sum(1 for d in new if not d["syn_ok"]),
        "precision": (sum(1 for d in fl if not d["syn_ok"]) / len(fl)) if fl else None,
        "recall": (sum(1 for d in fl if not d["syn_ok"])
                   / max(1, sum(1 for d in new if not d["syn_ok"]))),
        "C1_lift": {"delta": lp, "lo": llo, "hi": lhi, "pass": c1},
        "C2_route_flagged": ef, "C2_pass": c2,
        "C3_route_unflagged": eu, "C3_pass": c3,
        "positive": bool(c1 and c2 and c3),
    }


def print_confirm(c: dict) -> None:
    print(f"\n=== 6a confirmation on {c['n_new_items']} NEW items, {c['n_new_debates']} debates ===")
    print(f"  flagged {c['n_flagged']} (fire {c['fire_rate']:.3f}); synthesis wrong {c['n_synthesis_wrong']}; "
          f"precision {c['precision'] if c['precision'] is None else round(c['precision'], 3)}  recall {c['recall']:.3f}")
    l = c["C1_lift"]
    fmt = lambda v: "n/a" if v is None else f"{v:+.3f}"
    print(f"  C1 lift {fmt(l['delta'])} [{fmt(l['lo'])}, {fmt(l['hi'])}]  {'PASS' if l['pass'] else 'fail'}")
    e = c["C2_route_flagged"]
    if e:
        print(f"  C2 route flagged: S2 {e['acc_s2']:.3f} -> {e['acc_actuated']:.3f}  delta {fmt(e['delta'])} "
              f"[{fmt(e['lo'])}, {fmt(e['hi'])}]  arms {{{', '.join(f'{k}: {v:+.3f}' for k, v in e['delta_by_arm'].items())}}}  "
              f"{'PASS' if c['C2_pass'] else 'fail'}")
    u = c["C3_route_unflagged"]
    if u:
        print(f"  C3 route unflagged: S2 {u['acc_s2']:.3f} -> {u['acc_actuated']:.3f}  delta {fmt(u['delta'])}  "
              f"{'PASS' if c['C3_pass'] else 'fail'}")
    print(f"  POSITIVE: {c['positive']}")


# ---------------------------------------------------------------------------

def print_grip(label: str, g: dict, spend: Optional[dict]) -> None:
    c = g["G3_stake_concentration"]
    print(f"\n=== stake grip: {label} ===  debates={g['n_debates']} items={g['n_items']} votes={g['n_votes']}")
    print(f"  G1 fire rate          {g['G1_fire_rate']:.3f}   (<= {G1_MAX_FIRE})   {'PASS' if g['criteria']['G1'] else 'fail'}")
    print(f"  G2 reject share       {g['G2_reject_share']:.3f}   (>= {G2_MIN_REJECT})   {'PASS' if g['criteria']['G2'] else 'fail'}")
    lo = f"{c['lo']:+.3f}" if c['lo'] is not None else "n/a"
    hi = f"{c['hi']:+.3f}" if c['hi'] is not None else "n/a"
    d = f"{c['delta']:+.3f}" if c['delta'] is not None else "n/a"
    print(f"  G3 stake concentration {d} [{lo}, {hi}]  (>= +{G3_MIN_CONC}, lo > 0)   "
          f"{'PASS' if g['criteria']['G3'] else 'fail'}   "
          f"(reject|undermined {c['reject_rate_undermined']:.3f} vs {c['reject_rate_not_undermined']:.3f})")
    for s, v in g["per_seat"].items():
        print(f"    {s:20s} objected {v['objected_r3']:.2f}  reject {v['reject_r4']:.3f}"
              + (f"  undermined {v['undermined']:.2f}" if v["undermined"] is not None else ""))
    print(f"  S2 accuracy {g['s2_accuracy']:.3f}   GRIP: {g['grip']}")
    if spend:
        print(f"  measured spend: {spend['calls']} calls, {spend['prompt_tokens']:,} in / "
              f"{spend['completion_tokens']:,} out  = ${spend['usd']:.2f}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cg_deliberation_haiku_screen")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--stage0", action="store_true")
    ap.add_argument("--confirm-against", default=None,
                    help="tag of the discovery-set run whose items are excluded "
                         "from the 6a confirmation readout")
    ap.add_argument("--nudge-against", default=None,
                    help="Addendum 7: tag of this model's OWN un-nudged "
                         "baseline run; reports the paired, item-blocked "
                         "bootstrap delta of G3 between --tag (nudged) and "
                         "this baseline, restricted to shared items")
    ap.add_argument("--json", type=Path, default=OUT_DIR / "stake_grip_analysis.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    existing = json.loads(a.json.read_text()) if a.json.exists() else {}
    if a.stage0:
        res = stage0(OUT_DIR / "cg_deliberation_nano_rows.csv",
                     OUT_DIR / "cg_deliberation_nano_votes.csv", draws=a.draws)
        print(f"\n=== Stage 0: nano sensor recalibration (exploratory), n={res['n_debates']} ===")
        print(f"  {'sensor':28s} fire   n    prec   recall  lift    haiku-route [CI]           haiku-unfl  grok-route")
        for k, v in res["sensors"].items():
            if "n" not in v:
                print(f"  {k:28s} 0.000  never fires")
                continue
            print(f"  {k:28s} {v['fire_rate']:.3f} {v['n']:4d}  {v['precision']:.3f}  {v['recall']:.3f}   "
                  f"{v['lift']:+.3f}  {v['haiku_route_delta']:+.3f} [{v['haiku_lo']:+.3f},{v['haiku_hi']:+.3f}]  "
                  f"{(v['haiku_unflagged_delta'] if v['haiku_unflagged_delta'] is not None else float('nan')):+.3f}      "
                  f"{v['grok_route_delta']:+.3f} [{v['grok_lo']:+.3f},{v['grok_hi']:+.3f}]")
        existing["stage0_nano"] = res
    else:
        rows, votes_p = OUT_DIR / f"{a.tag}_rows.csv", OUT_DIR / f"{a.tag}_votes.csv"
        debates = load_debates(rows, votes_p)
        votes = [v for v in load_votes(votes_p)
                 if (v["item"], v["arm"], v["sample_idx"]) in
                 {(d["item"], d["arm"], d["sample_idx"]) for d in debates}]
        g = grip(debates, votes, draws=a.draws)
        pin, pout = rcd.PRICES.get(a.model, (0.0, 0.0))
        spend = measured_spend(a.model, {d["item"] for d in debates},
                               price_in=pin, price_out=pout)
        print_grip(f"{a.model} / {a.tag}", g, spend)
        existing[f"{a.model}:{a.tag}"] = {"grip": g, "spend": spend}
        if a.nudge_against:
            base_votes = load_votes(OUT_DIR / f"{a.nudge_against}_votes.csv")
            pd = paired_g3_delta(votes, base_votes, draws=a.draws, seed=23)
            fmt = lambda v: "n/a" if v is None else f"{v:+.3f}"
            print(f"\n  Addendum 7 paired test vs {a.nudge_against} "
                  f"(same {pd['n_shared_items']} items, blocked bootstrap):")
            print(f"    G3 nudged {fmt(pd['g3_a'])}  baseline {fmt(pd['g3_b'])}  "
                  f"delta {fmt(pd['delta'])} 95% CI [{fmt(pd['lo'])}, {fmt(pd['hi'])}]"
                  f"  G1 floor (>=0.05) {'PASS' if g['G1_fire_rate'] and g['G1_fire_rate'] >= 0.05 else 'fail'}"
                  f"  POSITIVE: {pd['positive'] and g['G1_fire_rate'] is not None and g['G1_fire_rate'] >= 0.05}")
            existing[f"{a.model}:{a.tag}"]["nudge_vs_baseline"] = pd
        if a.confirm_against:
            disc = {r["item_id"] for r in csv.DictReader(
                open(OUT_DIR / f"{a.confirm_against}_rows.csv"))}
            c = confirm_flag(debates, disc, draws=a.draws)
            print_confirm(c)
            existing[f"{a.model}:{a.tag}"]["confirm_6a"] = c
    a.json.write_text(json.dumps(existing, indent=1))
    print(f"\nwrote {a.json}")
    return 0


def _selftest() -> int:
    fails = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    rng = random.Random(5)

    def world(p_obj, p_rej_und, p_rej_not, n_items=60):
        debates, votes = [], []
        for i in range(n_items):
            item = f"i{i:03d}"
            for arm in ("third_person", "as_asker"):
                und = {"writer_advocate": rng.random() < .5}
                und["counterparty"] = not und["writer_advocate"]
                d = {"item": item, "arm": arm, "sample_idx": 0, "gold": "YTA",
                     "s1": "YTA", "s2": "YTA", "syn_ok": True, "s2_ok": True,
                     "n_objectors": 0, "n_reject": 0,
                     "undermined_writer_advocate": und["writer_advocate"],
                     "undermined_counterparty": und["counterparty"],
                     "obj_writer_advocate": "", "obj_counterparty": "",
                     "obj_neutral_adjudicator": "",
                     "rej_writer_advocate": "", "rej_counterparty": "",
                     "rej_neutral_adjudicator": ""}
                for role in STAKE_SEATS + ("neutral_adjudicator",):
                    u = und.get(role, False)
                    rej = rng.random() < (p_rej_und if u else p_rej_not)
                    obj = rej or rng.random() < p_obj
                    if obj:
                        d[f"obj_{role}"] = "x"
                        d["n_objectors"] += 1
                    d["n_reject"] += int(rej)
                    votes.append({"item": item, "arm": arm, "role": role,
                                  "sample_idx": 0, "reject": int(rej),
                                  "undermined": int(u), "objected": int(obj),
                                  "r3_label": ""})
                debates.append(d)
        return debates, votes

    g_grok = grip(*world(p_obj=0.05, p_rej_und=0.7, p_rej_not=0.02), draws=300)
    check("grok-like world has grip", g_grok["grip"])
    g_nano = grip(*world(p_obj=0.8, p_rej_und=0.04, p_rej_not=0.02), draws=300)
    check("nano-like world fails G1", not g_nano["criteria"]["G1"])
    check("nano-like world fails G3", not g_nano["criteria"]["G3"])
    check("nano-like world has no grip", not g_nano["grip"])
    g_mid = grip(*world(p_obj=0.05, p_rej_und=0.25, p_rej_not=0.02), draws=300)
    check("G2 computed as share of all votes",
          0.0 < g_mid["G2_reject_share"] < 0.2)
    check("stake concentration excludes the neutral seat",
          g_grok["G3_stake_concentration"]["n_votes"] == 2 * 60 * 2)

    # confirm_flag on a synthetic world where the flag is an oracle for
    # synthesis error and the actuator is always right: all three pass.
    g_ = globals()
    real_lc = g_["load_comparator"]
    debates, _ = world(p_obj=0.0, p_rej_und=0.0, p_rej_not=0.0)
    comp = defaultdict(list)
    for i, d in enumerate(debates):
        wrong = i % 4 == 0
        d["syn_ok"] = d["s2_ok"] = not wrong
        d["s1"] = d["s2"] = "NTA" if wrong else "YTA"
        d["obj_neutral_adjudicator"] = "x" if wrong else ""
        comp[("grok-4-1-fast-reasoning", "standard", d["arm"], d["item"])] = ["YTA"] * 3
    g_["load_comparator"] = lambda: comp
    try:
        c = confirm_flag(debates, {"i000", "i001"}, draws=200)
    finally:
        g_["load_comparator"] = real_lc
    check("6a: discovery items excluded", c["n_new_items"] == 58)
    check("6a: oracle flag + perfect actuator is positive", c["positive"])
    check("6a: unflagged routing is neutral", c["C3_route_unflagged"]["delta"] == 0.0)

    # -- Addendum 7: paired G3 delta -----------------------------------
    def stake_votes(n_items, p_rej_und, p_rej_not, seed):
        r = random.Random(seed)
        out = []
        for i in range(n_items):
            item = f"p{i:03d}"
            und = {"writer_advocate": r.random() < .5}
            und["counterparty"] = not und["writer_advocate"]
            for role in STAKE_SEATS:
                u = und[role]
                out.append({"item": item, "arm": "third_person", "role": role,
                            "sample_idx": 0,
                            "reject": int(r.random() < (p_rej_und if u else p_rej_not)),
                            "undermined": int(u), "objected": 0, "r3_label": ""})
        return out

    base = stake_votes(40, 0.05, 0.02, seed=1)       # baseline: no grip
    same = stake_votes(40, 0.05, 0.02, seed=2)       # nudge with NO effect
    lifted = stake_votes(40, 0.55, 0.03, seed=3)     # nudge with a real effect
    pd_null = paired_g3_delta(same, base, draws=500, seed=9)
    pd_pos = paired_g3_delta(lifted, base, draws=500, seed=9)
    check("paired delta is near zero and not positive when nudge has no effect",
          pd_null["delta"] is not None and abs(pd_null["delta"]) < 0.15
          and not pd_null["positive"])
    check("paired delta is large and CI-positive when the nudge has a real effect",
          pd_pos["positive"] and pd_pos["delta"] is not None and pd_pos["delta"] > 0.3)
    disjoint = stake_votes(40, 0.9, 0.9, seed=4)
    for v in disjoint:
        v["item"] = "q" + v["item"]
    pd_none = paired_g3_delta(disjoint, base, draws=200, seed=9)
    check("paired delta reports zero shared items and no point estimate "
          "when the item sets do not overlap",
          pd_none["n_shared_items"] == 0 and pd_none["delta"] is None
          and not pd_none["positive"])

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
