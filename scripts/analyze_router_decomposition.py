"""
scripts/analyze_router_decomposition.py -- Addendum 16.15.1, 16.15.2, 16.15.3:
is the dissent sensor's error concentration the dissent, or the verdict type?
(zero spend; every number is a recomputation on cached calls and CSVs)

THE QUESTION

The stake-blind counter `n_objectors >= 2` fires on 328 of the 1,677 codable
grok AITA debates and concentrates the deliberation's own errors about 4.4x.
Addendum 16.12 showed that on both-party verdicts (ESH "everyone at fault",
NAH "no one at fault") the advocate edge collapses and the flag fires on ~60%
of debates, and that those verdicts are wrong 44-52% of the time whether or
not any seat objects. So the pooled concentration could be two signals mixed:
dissent on one-loser verdicts (YTA/NTA, where the theory says the edge
exists) and verdict type on both-party verdicts. This module separates them.

16.15.1  Counter lift INSIDE one-loser and INSIDE both-party verdicts, with
         the registered estimator. Composed accuracy at the deployed rule (S2
         unless routed; cached haiku standard majority-of-3 if routed; fall
         back to S2 when the judge is non-codable) for four routers: the
         counter; verdict type (S2 is ESH or NAH); their union; and random
         routing at the union's coverage (200 seeds, mean and best). The same
         with the cached sonnet judge of 16.11a, restricted to the debates its
         cache covers. Transfer within strata: P(grok-solo wrong | fired) vs
         unfired inside each stratum.
16.15.2  Grok-solo's OWN free signals on the same 420 (arm, item) cells: (a)
         any ESH/NAH among its three standard samples, (b) disagreement among
         them, against (c) the collective's counter (majority of the cell's
         debate samples), each as an error-concentration ratio and as a router
         at matched coverage composed with the cached haiku judge.
16.15.3  The four transfer readouts of headline_theory.html R5 (grok standard
         solo, grok narrative solo, nano, haiku; all cached majority-of-3)
         recomputed with the registered estimator and written to
         transfer_readouts.json, stating for each whether the interval
         excludes zero. R5 quoted the composite (stake-aware) flag; the
         registered sensor of 16.15 is the counter; both are reported.

Verdict type is read from the deliberation's FINAL verdict (the `verdict`
column, S2, the verdict the deployed system emits when unrouted), which is
the "group verdict" of the 16.12 table (1,288 / 272 / 118). All verdicts are
scored by the registered code_response collapse (ESH -> at fault, NAH -> not
at fault), never string equality. Estimator everywhere: item-clustered
percentile bootstrap, 4,000 draws, seed 13.

RESAMPLING FRAME (convention, fixed after the 16.15 audit). Every
error-concentration interval resamples the FULL registered population's items
-- the 210 items behind the 1,677 codable debates, or the 420 (arm, item)
cells -- and applies the readout's restriction (verdict stratum, codable
target verdict) INSIDE each resample, treating stratum membership and target
codability as outcomes of the debate. Items with no debate in a stratum are
therefore still part of the frame. Restricting the frame to the items that
happen to have a debate in the stratum (199 one-loser, 101 both-party items)
moved the one-loser lower bound from +0.201 to +0.195 and the both-party
upper bound from +0.043 to +0.038; point estimates and every n are identical
under either convention. Each concentration block records `n_items_frame`
(resampled) and `n_items_kept` (with at least one record in the readout).

Machinery reused, not reimplemented: analyze_actuator_ladder (load_debates,
load_comparator, make_rule_vendor, actuated_ok, correct), analyze_claim_audit
(n_objectors_ge2, composed_ok, paired_sensor_diff, trim_sensor),
analyze_unembodied_ablation (_boot, load_rows), analyze_loop_step (flagged).

Usage
-----
  python -m scripts.analyze_router_decomposition --selftest
  python -m scripts.analyze_router_decomposition
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, Sequence

from scripts.analyze_actuator_ladder import (
    COMPARATOR_PATH, ROWS_PATH, VOTES_PATH, actuated_ok, correct,
    load_comparator, load_debates, make_rule_vendor,
)
from scripts.analyze_claim_audit import (
    composed_ok, n_objectors_ge2, paired_sensor_diff, trim_sensor,
)
from scripts.analyze_crowdgold_sdt import code_response
from scripts.analyze_loop_step import flagged as composite_flagged
from scripts.analyze_unembodied_ablation import _boot as boot_ci, load_rows
from scripts.run_phase1_quartet import OUT_DIR

ANALYSIS_PATH = OUT_DIR / "router_decomposition.json"
TRANSFER_PATH = OUT_DIR / "transfer_readouts.json"
SONNET_ROWS_PATH = OUT_DIR / "cg_sonnet_actuator_rows.csv"

GROK = "grok-4-1-fast-reasoning"
HAIKU_JUDGE = ("claude-haiku-4-5", "standard")
SONNET_JUDGE = ("claude-sonnet-4-6", "standard")
GROK_SOLO = (GROK, "standard")
DEFAULT_DRAWS = 4000
DEFAULT_SEED = 13
TRIM_SEEDS = 200
ONE_LOSER = ("YTA", "NTA")
BOTH_PARTY = ("ESH", "NAH")

# Guards (checked on the literal string GUARD FAILED before any number is read)
MIN_CODABLE_SHARE = 0.95        # codable debates / raw debates
MAX_MEAN_R3_UNPARSED = 0.05     # seats per debate (16.8 slack over a ~0 base)
MIN_JUDGE_EMISSION = 0.95       # judge codable on routed debates / cells

#: R5 transfer targets: label -> (model, scaffold) in the k3 comparator CSV
TRANSFER_TARGETS = {
    "grok_solo_standard": (GROK, "standard"),
    "grok_solo_narrative": (GROK, "narrative_cot"),
    "nano_standard": ("gpt-5.4-nano", "standard"),
    "haiku_standard": ("claude-haiku-4-5", "standard"),
}

#: Auditor's replay values (registration brackets) and R5 quoted values.
REPLAY = {
    "one_loser": {"p_wrong_fired": 0.387, "p_wrong_unfired": 0.039,
                  "lift": 0.348, "n_fired": 93},
    "both_party": {"p_wrong_fired": 0.426, "p_wrong_unfired": 0.523,
                   "lift": -0.097},
    "composed_haiku": {"counter": 0.880, "verdict_type": 0.894,
                       "union": 0.905, "collective_alone": 0.843},
    "solo_signal_ratio": {"any_eshnah": 8.2, "disagree": 6.4, "counter": 4.4},
}
R5_QUOTED = {   # composite flag, cell level, headline_theory.html R5
    "grok_solo_standard": {"p_wrong_fired": 0.264, "p_wrong_unfired": 0.063,
                           "ratio": 4.2, "lift": 0.201, "lo": 0.075, "hi": 0.346},
    "grok_solo_narrative": {"ratio": 4.8, "lo": 0.132, "hi": 0.413},
    "nano_standard": {"ratio": 2.3, "lift": 0.200, "lo": 0.060, "hi": 0.360},
    "haiku_standard": {"ratio": 1.8, "lift": 0.109, "lo": -0.018, "hi": 0.256},
}


# ---------------------------------------------------------------------------
# Routers on debates (16.15.1)
# ---------------------------------------------------------------------------

def verdict_type(d: dict) -> str:
    v = d.get("s2")
    if v in ONE_LOSER:
        return "one_loser"
    if v in BOTH_PARTY:
        return "both_party"
    return "other"


def counter_router(d: dict) -> bool:
    return n_objectors_ge2(d)


def verdict_type_router(d: dict) -> bool:
    return d.get("s2") in BOTH_PARTY


def union_router(d: dict) -> bool:
    return counter_router(d) or verdict_type_router(d)


def never(d: dict) -> bool:
    return False


ROUTERS: dict[str, Callable[[dict], bool]] = {
    "collective_alone": never,
    "counter": counter_router,
    "verdict_type": verdict_type_router,
    "union": union_router,
}


# ---------------------------------------------------------------------------
# Estimator helpers (item-clustered percentile bootstrap via _boot)
# ---------------------------------------------------------------------------

def _by_item(recs: Sequence[dict]) -> dict:
    by: dict = defaultdict(list)
    for r in recs:
        by[r["item"]].append(r)
    return by


def _lift_stat(recs: Sequence[dict]) -> Optional[float]:
    """Lift on the records that belong to the readout (w is not None); the
    restriction is applied here, INSIDE each bootstrap resample."""
    f = [r["w"] for r in recs if r["w"] is not None and r["f"]]
    u = [r["w"] for r in recs if r["w"] is not None and not r["f"]]
    if not f or not u:
        return None
    return sum(f) / len(f) - sum(u) / len(u)


def concentration(recs: Sequence[dict], *, draws: int, seed: int) -> dict:
    """recs: {item, f (fired 0/1), w (wrong 0/1, or None)}. A record with
    w None is in the resampling FRAME (its item is resampled) but not in the
    readout (out of stratum, or the target's verdict is non-codable); it is
    dropped inside each resample by `_lift_stat`. P(wrong | fired) vs unfired,
    lift with item-clustered CI, ratio, and whether the CI excludes zero."""
    kept = [r for r in recs if r["w"] is not None]
    f = [r["w"] for r in kept if r["f"]]
    u = [r["w"] for r in kept if not r["f"]]
    pf = sum(f) / len(f) if f else None
    pu = sum(u) / len(u) if u else None
    pt, lo, hi = boot_ci(_by_item(recs), _lift_stat, draws=draws, seed=seed)
    return {
        "n": len(kept), "n_fired": len(f), "n_unfired": len(u),
        "n_frame": len(recs),
        "n_items_frame": len({r["item"] for r in recs}),
        "n_items_kept": len({r["item"] for r in kept}),
        "coverage": len(f) / len(kept) if kept else None,
        "p_wrong_fired": pf, "p_wrong_unfired": pu,
        "ratio": (pf / pu if pf is not None and pu else None),
        "lift": pt, "lo": lo, "hi": hi,
        "ci_excludes_zero": (lo is not None and hi is not None
                             and (lo > 0 or hi < 0)),
    }


def acc_with_ci(recs: Sequence[dict], key: str, *, draws: int, seed: int) -> dict:
    pt, lo, hi = boot_ci(
        _by_item(recs), lambda rs: (sum(r[key] for r in rs) / len(rs)) if rs else None,
        draws=draws, seed=seed)
    return {"acc": pt, "lo": lo, "hi": hi, "n": len(recs)}


# ---------------------------------------------------------------------------
# 16.15.1 -- within-stratum lift of the counter on the deliberation's errors
# ---------------------------------------------------------------------------

def stratum_lifts(debates: Sequence[dict], sensor, *, draws: int, seed: int,
                  wrong: Callable[[dict], Optional[bool]] = lambda d: not d["s2_ok"],
                  ) -> dict:
    """Counter lift within each verdict stratum (and pooled). `wrong` may
    return None to drop a debate from the readout (used for the grok-solo
    transfer, where the solo majority can be a tie). Every debate stays in
    the resampling frame; the stratum restriction and the drop are applied
    inside each resample (see RESAMPLING FRAME in the module docstring)."""
    out = {}
    for name, sel in (("pooled", lambda d: True),
                      ("one_loser", lambda d: verdict_type(d) == "one_loser"),
                      ("both_party", lambda d: verdict_type(d) == "both_party")):
        recs = []
        for d in debates:
            w = wrong(d) if sel(d) else None
            recs.append({"item": d["item"], "f": int(sensor(d)),
                         "w": None if w is None else int(w)})
        out[name] = concentration(recs, draws=draws, seed=seed)
    return out


def _random_at_coverage(debates: Sequence[dict], rule, target_n: int, *,
                        seed: int, trim_seeds: int) -> dict:
    accs = []
    n = len(debates)
    for s in range(trim_seeds):
        t = trim_sensor(lambda d: True, debates, target_n, seed + 5000 + s)
        accs.append(sum(composed_ok(d, t, rule) for d in debates) / n)
    accs.sort()
    return {"target_n": target_n, "coverage": target_n / n if n else None,
            "mean": sum(accs) / len(accs), "best": accs[-1], "worst": accs[0],
            "p05": accs[int(0.05 * len(accs))],
            "p95": accs[min(int(0.95 * len(accs)), len(accs) - 1)],
            "n_seeds": trim_seeds}


def composed_block(debates: Sequence[dict], rule, *, draws: int, seed: int,
                   trim_seeds: int, judge_label: str) -> dict:
    """Composed accuracy of the four routers at the deployed rule, item-
    clustered CIs, paired deltas, random routing at the union's coverage."""
    n = len(debates)
    out: dict = {"judge": judge_label, "n_debates": n,
                 "n_items": len({d["item"] for d in debates}), "routers": {}}
    for name, sensor in ROUTERS.items():
        recs = [{"item": d["item"], "ok": int(composed_ok(d, sensor, rule))}
                for d in debates]
        fired = [d for d in debates if sensor(d)]
        emitted = sum(1 for d in fired if rule(d) is not None
                      and correct(rule(d), d["gold"]) is not None)
        by_stratum = {}
        for st in ("one_loser", "both_party"):
            sub = [d for d in debates if verdict_type(d) == st]
            by_stratum[st] = {
                "n": len(sub),
                "n_fired": sum(1 for d in sub if sensor(d)),
                "acc_s2": (sum(d["s2_ok"] for d in sub) / len(sub)) if sub else None,
                "acc_composed": (sum(composed_ok(d, sensor, rule) for d in sub)
                                 / len(sub)) if sub else None,
            }
        out["routers"][name] = {
            "n_fire": len(fired), "coverage": len(fired) / n if n else None,
            "composed": acc_with_ci(recs, "ok", draws=draws, seed=seed),
            "judge_emitted_on_fired": (emitted / len(fired)) if fired else None,
            "n_fallback_to_s2": len(fired) - emitted,
            "by_stratum": by_stratum,
        }
    pairs = [("counter", "collective_alone"), ("verdict_type", "collective_alone"),
             ("union", "collective_alone"), ("verdict_type", "counter"),
             ("union", "counter"), ("union", "verdict_type")]
    out["paired_deltas"] = {}
    for a, b in pairs:
        r = paired_sensor_diff(debates, ROUTERS[a], ROUTERS[b], rule,
                               draws=draws, seed=seed)
        r["ci_excludes_zero"] = (r["lo"] is not None and r["hi"] is not None
                                 and (r["lo"] > 0 or r["hi"] < 0))
        out["paired_deltas"][f"{a}_minus_{b}"] = r
    n_union = out["routers"]["union"]["n_fire"]
    out["random_at_union_coverage"] = _random_at_coverage(
        debates, rule, n_union, seed=seed, trim_seeds=trim_seeds)
    return out


def sonnet_covered(debates: Sequence[dict], sonnet_comp: dict) -> list[dict]:
    key = (SONNET_JUDGE[0], SONNET_JUDGE[1])
    return [d for d in debates if sonnet_comp.get((*key, d["arm"], d["item"]))]


# ---------------------------------------------------------------------------
# Cells (16.15.2, 16.15.3): one record per (arm, item), flags by majority of
# the cell's debate samples; grok-solo signals from its three cached samples
# ---------------------------------------------------------------------------

def build_cells(debates: Sequence[dict], comp: dict) -> list[dict]:
    """Cell = (arm, item). A cell FIRES on a sensor when strictly more than
    half of its debate samples fire (the 16.11a / topology_2x2 convention).
    Grok-solo fields come from its three cached standard samples."""
    agg: dict[tuple, dict] = {}
    for d in debates:
        c = agg.setdefault((d["arm"], d["item"]), {
            "item": d["item"], "arm": d["arm"], "sample_idx": 0,
            "gold": d["gold"], "k": 0, "k_counter": 0, "k_composite": 0})
        c["k"] += 1
        c["k_counter"] += int(counter_router(d))
        c["k_composite"] += int(composite_flagged(d))
    cells = []
    for c in agg.values():
        c["counter_fired"] = c["k_counter"] * 2 > c["k"]
        c["composite_fired"] = c["k_composite"] * 2 > c["k"]
        vs = comp.get((*GROK_SOLO, c["arm"], c["item"]), [])
        codes = [code_response(v, "published") for v in vs]
        codes = [x for x in codes if x is not None]
        c["solo_n_samples"] = len(vs)
        c["solo_any_eshnah"] = any(v in BOTH_PARTY for v in vs)
        c["solo_disagree"] = len(set(codes)) > 1          # code_response collapse
        c["solo_disagree_raw"] = len(set(vs)) > 1         # raw five-token strings
        cells.append(c)
    cells.sort(key=lambda c: (c["arm"], c["item"]))
    return cells


def solo_ok(c: dict, rule) -> Optional[bool]:
    v = rule(c)
    return correct(v, c["gold"]) if v is not None else None


def cell_composed_ok(c: dict, base_ok: bool, sensor, judge_rule) -> bool:
    """Deployed rule at the cell level: the base generator unless the sensor
    fires, the judge if it fires, fall back to the base when the judge is
    non-codable."""
    if not sensor(c):
        return base_ok
    ok = solo_ok(c, judge_rule)
    return base_ok if ok is None else ok


CELL_SIGNALS: dict[str, Callable[[dict], bool]] = {
    "counter": lambda c: c["counter_fired"],
    "any_eshnah": lambda c: c["solo_any_eshnah"],
    "disagree": lambda c: c["solo_disagree"],
    "disagree_raw": lambda c: c["solo_disagree_raw"],
    "composite": lambda c: c["composite_fired"],
}


def solo_signal_block(cells: Sequence[dict], solo_rule, judge_rule, *,
                      draws: int, seed: int, trim_seeds: int) -> dict:
    """16.15.2: each signal as an error-concentration ratio on grok-solo and
    as a router composed with the haiku judge, natural and matched coverage."""
    pop = [c for c in cells if solo_ok(c, solo_rule) is not None]
    base = {id(c): bool(solo_ok(c, solo_rule)) for c in pop}
    n = len(pop)
    out: dict = {"n_cells_total": len(cells), "n_cells_codable": n,
                 "n_items": len({c["item"] for c in pop}),
                 "solo_accuracy": sum(base.values()) / n if n else None,
                 "signals": {}}
    for name in ("counter", "any_eshnah", "disagree", "disagree_raw", "composite"):
        sig = CELL_SIGNALS[name]
        # frame = all cells; a non-codable cell has w None (dropped inside)
        recs = [{"item": c["item"], "f": int(sig(c)),
                 "w": (int(not base[id(c)]) if id(c) in base else None)}
                for c in cells]
        conc = concentration(recs, draws=draws, seed=seed)
        comp_recs = [{"item": c["item"],
                      "ok": int(cell_composed_ok(c, base[id(c)], sig, judge_rule))}
                     for c in pop]
        fired = [c for c in pop if sig(c)]
        emitted = sum(1 for c in fired if solo_ok(c, judge_rule) is not None)
        out["signals"][name] = {
            "concentration": conc,
            "composed_natural": acc_with_ci(comp_recs, "ok", draws=draws, seed=seed),
            "judge_emitted_on_fired": (emitted / len(fired)) if fired else None,
        }
    # paired deltas vs the collective's counter, natural coverage
    out["paired_vs_counter_natural"] = {}
    for name in ("any_eshnah", "disagree", "disagree_raw", "composite"):
        by = defaultdict(list)
        a_tot = b_tot = 0
        for c in pop:
            a = int(cell_composed_ok(c, base[id(c)], CELL_SIGNALS[name], judge_rule))
            b = int(cell_composed_ok(c, base[id(c)], CELL_SIGNALS["counter"], judge_rule))
            a_tot += a
            b_tot += b
            by[c["item"]].append({"d": a - b})
        pt, lo, hi = boot_ci(by, lambda rs: sum(r["d"] for r in rs) / len(rs) if rs else None,
                             draws=draws, seed=seed)
        out["paired_vs_counter_natural"][f"{name}_minus_counter"] = {
            "acc_a": a_tot / n, "acc_b": b_tot / n, "delta": pt, "lo": lo,
            "hi": hi, "n": n,
            "ci_excludes_zero": (lo is not None and hi is not None
                                 and (lo > 0 or hi < 0))}
    # matched coverage: trim every signal to a common n (the smallest natural
    # n among the three registered signals) and to the counter's n
    n_by = {k: out["signals"][k]["concentration"]["n_fired"]
            for k in ("counter", "any_eshnah", "disagree")}
    targets = {"common_min": min(n_by.values()), "counter_n": n_by["counter"]}
    out["matched_coverage"] = {}
    for tname, target in targets.items():
        block = {"target_n": target, "coverage": target / n if n else None}
        for name in ("counter", "any_eshnah", "disagree", "disagree_raw", "composite"):
            accs = []
            for s in range(trim_seeds):
                t = trim_sensor(CELL_SIGNALS[name], pop, target, seed + 1000 + s)
                accs.append(sum(cell_composed_ok(c, base[id(c)], t, judge_rule)
                                for c in pop) / n)
            accs.sort()
            block[name] = {"mean": sum(accs) / len(accs), "best": accs[-1],
                           "p05": accs[int(0.05 * len(accs))],
                           "p95": accs[min(int(0.95 * len(accs)), len(accs) - 1)],
                           "n_seeds": trim_seeds}
        accs = []
        for s in range(trim_seeds):
            t = trim_sensor(lambda c: True, pop, target, seed + 5000 + s)
            accs.append(sum(cell_composed_ok(c, base[id(c)], t, judge_rule)
                            for c in pop) / n)
        accs.sort()
        block["random"] = {"mean": sum(accs) / len(accs), "best": accs[-1],
                           "p05": accs[int(0.05 * len(accs))],
                           "p95": accs[min(int(0.95 * len(accs)), len(accs) - 1)],
                           "n_seeds": trim_seeds}
        out["matched_coverage"][tname] = block
    return out


# ---------------------------------------------------------------------------
# 16.15.3 -- the four R5 transfer readouts, both sensors
# ---------------------------------------------------------------------------

def transfer_readouts(cells: Sequence[dict], comp: dict, *, draws: int,
                      seed: int) -> dict:
    out: dict = {"n_cells": len(cells), "n_items": len({c["item"] for c in cells}),
                 "targets": {}}
    for label, (model, scaffold) in TRANSFER_TARGETS.items():
        rule = make_rule_vendor(comp, model, scaffold)
        pop = [c for c in cells if solo_ok(c, rule) is not None]
        n = len(pop)
        entry = {"model": model, "scaffold": scaffold, "n_cells_codable": n,
                 "n_cells_noncodable": len(cells) - n,
                 "solo_accuracy": (sum(bool(solo_ok(c, rule)) for c in pop) / n
                                   if n else None),
                 "n_samples_per_cell": sorted({len(comp.get(
                     (model, scaffold, c["arm"], c["item"]), [])) for c in pop}),
                 "by_sensor": {}}
        for sname in ("counter", "composite"):
            sig = CELL_SIGNALS[sname]
            # frame = all 420 cells; the target's non-codable cells (a tied
            # majority-of-3) carry w None and are dropped inside each resample
            recs = []
            for c in cells:
                ok = solo_ok(c, rule)
                recs.append({"item": c["item"], "f": int(sig(c)),
                             "w": None if ok is None else int(not ok)})
            entry["by_sensor"][sname] = concentration(recs, draws=draws, seed=seed)
        out["targets"][label] = entry
    for sname in ("counter", "composite"):
        out[f"n_targets_ci_excludes_zero_{sname}"] = sum(
            1 for t in out["targets"].values()
            if t["by_sensor"][sname]["ci_excludes_zero"])
    return out


R5_TOLERANCE = {"ratio": 0.1, "lo": 0.015, "hi": 0.015, "default": 0.005}


def r5_agreement(tr: dict) -> dict:
    """Compare the composite-flag recomputation with the R5 quoted values.
    Agreement is a tolerance check on the in-session quotes, not a claim
    that they are the numbers to cite: the paper quotes THIS artefact."""
    out = {"note": ("agreement = |got - quoted| within R5_TOLERANCE (ratio 0.1, "
                    "CI bounds 0.015, else 0.005); the paper cites the artefact "
                    "values verbatim, never the R5 quotes (e.g. the nano lower "
                    "bound is the artefact's, not R5's +0.060)"),
           "tolerance": R5_TOLERANCE}
    for label, q in R5_QUOTED.items():
        got = tr["targets"][label]["by_sensor"]["composite"]
        cmp = {}
        for k, v in q.items():
            g = got.get(k)
            if g is None:
                cmp[k] = {"quoted": v, "got": None, "agrees": False}
                continue
            tol = R5_TOLERANCE.get(k, R5_TOLERANCE["default"])
            cmp[k] = {"quoted": v, "got": g, "agrees": abs(g - v) <= tol,
                      "abs_diff": abs(g - v)}
        out[label] = {"fields": cmp, "agrees": all(x["agrees"] for x in cmp.values())}
    return out


# ---------------------------------------------------------------------------
# Pre-declared readings of 16.15.1 and 16.15.2, applied mechanically
# ---------------------------------------------------------------------------

def _excl(e: dict) -> bool:
    return bool(e.get("ci_excludes_zero"))


def pre_declared_readings(res: dict) -> dict:
    """The registration's pre-declared readings, evaluated on the artefact's
    own numbers. Complementarity is judge-specific: the union-minus-verdict-
    type interval is reported per judge and the reading names the judge."""
    ol = res["counter_lift_by_stratum"]["one_loser"]
    out: dict = {
        "16.15.1_sensor_claim": {
            "rule": "within-one-loser lift positive with CI excluding 0 -> sensor "
                    "claim stands, SHARPER than the pooled figure, which mixes two signals",
            "one_loser_lift": ol["lift"], "lo": ol["lo"], "hi": ol["hi"],
            "holds": ol["lift"] is not None and ol["lift"] > 0 and _excl(ol) and ol["lo"] > 0,
        },
        "16.15.1_verdict_type_second_signal": {},
        "16.15.1_complementarity": {},
    }
    blocks = {"haiku": res["composed_haiku"]}
    if res.get("composed_sonnet"):
        blocks["sonnet"] = res["composed_sonnet"]["block"]
    for judge, b in blocks.items():
        r = b["routers"]
        acc = {k: r[k]["composed"]["acc"] for k in ROUTERS}
        vt_ge_counter = acc["verdict_type"] >= acc["counter"]
        best_single = max(acc["counter"], acc["verdict_type"])
        union_gt = acc["union"] > best_single
        d = b["paired_deltas"]
        out["16.15.1_verdict_type_second_signal"][judge] = {
            "rule": "verdict-type router composes at least as high as the counter -> "
                    "present verdict type as a second, cheaper error signal; dissent is "
                    "the signal where verdict type is silent (one-loser verdicts)",
            "composed": acc, "verdict_type_ge_counter": vt_ge_counter,
            "verdict_type_minus_counter": {k: d["verdict_type_minus_counter"][k]
                                           for k in ("delta", "lo", "hi", "ci_excludes_zero")},
        }
        out["16.15.1_complementarity"][judge] = {
            "rule": "union does not exceed the better single router -> no complementarity claim",
            "union": acc["union"], "best_single_router": best_single,
            "union_exceeds_best_single_on_point": union_gt,
            "union_minus_verdict_type": {k: d["union_minus_verdict_type"][k]
                                         for k in ("delta", "lo", "hi", "ci_excludes_zero")},
            "reading": (f"union exceeds verdict-type alone by "
                        f"{d['union_minus_verdict_type']['delta']:+.3f} "
                        f"[{d['union_minus_verdict_type']['lo']:+.3f}, "
                        f"{d['union_minus_verdict_type']['hi']:+.3f}] with the {judge} judge"
                        + (" (CI excludes 0)" if d["union_minus_verdict_type"]["ci_excludes_zero"]
                           else " (CI includes 0)")) if union_gt else
                       f"no complementarity claim with the {judge} judge",
        }
    # 16.15.2: (a) or (b) composes at least as high as (c) -> demotion
    sb = res["solo_signals"]
    nat = {k: sb["signals"][k]["composed_natural"]["acc"] for k in ("counter", "any_eshnah", "disagree")}
    matched = sb["matched_coverage"]["common_min"]
    mm = {k: matched[k]["mean"] for k in ("counter", "any_eshnah", "disagree")}
    pv = sb["paired_vs_counter_natural"]
    out["16.15.2_transfer_demotion"] = {
        "rule": "(a) any ESH/NAH or (b) disagreement composes at least as high as (c) "
                "the collective's counter -> demote to 'the collective's flag concentrates "
                "a solo model's errors, and so do the solo model's own signals'",
        "composed_natural": nat, "matched_common_min_mean": mm,
        "triggers_on_natural": (nat["any_eshnah"] >= nat["counter"] or nat["disagree"] >= nat["counter"]),
        "triggers_on_matched": (mm["any_eshnah"] >= mm["counter"] or mm["disagree"] >= mm["counter"]),
        "no_pairwise_delta_excludes_zero": not any(
            pv[k]["ci_excludes_zero"] for k in ("any_eshnah_minus_counter", "disagree_minus_counter")),
    }
    out["16.15.2_transfer_demotion"]["triggers"] = (
        out["16.15.2_transfer_demotion"]["triggers_on_natural"]
        or out["16.15.2_transfer_demotion"]["triggers_on_matched"])
    return out


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def run_guards(raw_rows: Sequence[dict], debates: Sequence[dict],
               haiku_block: dict, sonnet_block: Optional[dict],
               solo_block: dict) -> dict:
    n_raw, n_cod = len(raw_rows), len(debates)
    share = n_cod / n_raw if n_raw else 0.0
    mean_unparsed = (sum(r["n_r3_unparsed"] for r in raw_rows) / n_raw
                     if n_raw else None)
    haiku_em = haiku_block["routers"]["union"]["judge_emitted_on_fired"]
    sonnet_em = (sonnet_block["routers"]["union"]["judge_emitted_on_fired"]
                 if sonnet_block else None)
    cell_ems = [v["judge_emitted_on_fired"] for v in solo_block["signals"].values()
                if v["judge_emitted_on_fired"] is not None]
    cell_em = min(cell_ems) if cell_ems else 1.0
    checks = {
        "codable_share": {"value": share, "limit": MIN_CODABLE_SHARE,
                          "ok": share >= MIN_CODABLE_SHARE},
        "mean_n_r3_unparsed": {"value": mean_unparsed, "limit": MAX_MEAN_R3_UNPARSED,
                               "ok": mean_unparsed is not None
                               and mean_unparsed <= MAX_MEAN_R3_UNPARSED},
        "haiku_judge_emission_on_union_fired": {
            "value": haiku_em, "limit": MIN_JUDGE_EMISSION,
            "ok": haiku_em is not None and haiku_em >= MIN_JUDGE_EMISSION},
        "haiku_judge_emission_on_cells_min_over_signals": {
            "value": cell_em, "limit": MIN_JUDGE_EMISSION,
            "ok": cell_em >= MIN_JUDGE_EMISSION},
    }
    if sonnet_block is not None:
        checks["sonnet_judge_emission_on_union_fired"] = {
            "value": sonnet_em, "limit": MIN_JUDGE_EMISSION,
            "ok": sonnet_em is not None and sonnet_em >= MIN_JUDGE_EMISSION}
    return {"n_raw": n_raw, "n_codable": n_cod, "checks": checks,
            "passed": all(c["ok"] for c in checks.values())}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _f(x, fmt="+.3f"):
    return "n/a" if x is None else format(x, fmt)


def _ci(e: dict, lo="lo", hi="hi"):
    return f"[{_f(e.get(lo))}, {_f(e.get(hi))}]"


def _conc_line(e: dict) -> str:
    return (f"P(wrong|fired) {_f(e['p_wrong_fired'], '.3f')} vs "
            f"{_f(e['p_wrong_unfired'], '.3f')}  ratio {_f(e['ratio'], '.2f')}x  "
            f"lift {_f(e['lift'])} {_ci(e)}  n_fired={e['n_fired']} "
            f"n_unfired={e['n_unfired']}  items {e['n_items_kept']}/{e['n_items_frame']} frame  "
            f"{'CI excludes 0' if e['ci_excludes_zero'] else 'CI includes 0'}")


def _readings_lines(pr: dict) -> None:
    print("\n  PRE-DECLARED READINGS (applied mechanically):")
    s = pr["16.15.1_sensor_claim"]
    print(f"    16.15.1 sensor claim within one-loser: lift {_f(s['one_loser_lift'])} "
          f"[{_f(s['lo'])}, {_f(s['hi'])}] -> {'STANDS (sharper than pooled)' if s['holds'] else 'does NOT stand'}")
    for judge, v in pr["16.15.1_verdict_type_second_signal"].items():
        d = v["verdict_type_minus_counter"]
        print(f"    16.15.1 verdict type >= counter ({judge}): {v['verdict_type_ge_counter']}  "
              f"delta {_f(d['delta'])} [{_f(d['lo'])}, {_f(d['hi'])}]")
    for judge, v in pr["16.15.1_complementarity"].items():
        print(f"    16.15.1 complementarity ({judge}): {v['reading']}")
    t = pr["16.15.2_transfer_demotion"]
    print(f"    16.15.2 demotion triggers: {t['triggers']} (natural {t['triggers_on_natural']}, "
          f"matched {t['triggers_on_matched']}; no pairwise delta excludes 0: "
          f"{t['no_pairwise_delta_excludes_zero']})")


def _composed_lines(block: dict, title: str) -> None:
    print(f"\n  {title}: n={block['n_debates']} debates, {block['n_items']} items")
    for name, r in block["routers"].items():
        c = r["composed"]
        print(f"    {name:<17} fires {r['n_fire']:>4} (cov {_f(r['coverage'], '.3f')})  "
              f"composed {_f(c['acc'], '.3f')} {_ci(c)}  fallback {r['n_fallback_to_s2']}")
        for st, s in r["by_stratum"].items():
            print(f"        {st:<11} n={s['n']:<5} fired {s['n_fired']:<4} "
                  f"S2 {_f(s['acc_s2'], '.3f')} -> composed {_f(s['acc_composed'], '.3f')}")
    rr = block["random_at_union_coverage"]
    print(f"    {'random@union':<17} fires {rr['target_n']:>4} (cov {_f(rr['coverage'], '.3f')})  "
          f"mean {_f(rr['mean'], '.3f')}  best {_f(rr['best'], '.3f')}  "
          f"p05 {_f(rr['p05'], '.3f')}  p95 {_f(rr['p95'], '.3f')}  ({rr['n_seeds']} seeds)")
    print("    paired deltas (item-clustered):")
    for k, v in block["paired_deltas"].items():
        print(f"      {k:<34} {_f(v['delta'])} {_ci(v)}  "
              f"{'excludes 0' if v['ci_excludes_zero'] else 'includes 0'}")


def print_report(res: dict, tr: dict) -> None:
    g = res["guards"]
    print("\n" + "=" * 78)
    print("ADDENDUM 16.15.1 -- COUNTER LIFT WITHIN VERDICT STRATA (deliberation's own errors)")
    print("=" * 78)
    print(f"  population: {g['n_codable']} codable of {g['n_raw']} debates; "
          f"estimator item-clustered bootstrap, {res['draws']} draws, seed {res['seed']}")
    print(f"  resampling frame: {res['resampling_frame']}")
    for st, e in res["counter_lift_by_stratum"].items():
        print(f"  {st:<11} {_conc_line(e)}")
    print("  agreement with replay:")
    for st, a in res["agreement"]["stratum_lift"].items():
        print(f"    {st:<11} " + "  ".join(
            f"{k} {_f(v['got'], '.3f')} vs {v['quoted']} {'OK' if v['agrees'] else 'DIFFERS'}"
            for k, v in a["fields"].items()))
    print("\n  verdict-type strata (S2 verdict):", res["stratum_counts"])
    print("  S1-synthesis-verdict-type sensitivity (router on S1 instead of S2):",
          {k: round(v, 3) if isinstance(v, float) else v
           for k, v in res["s1_verdict_type_sensitivity"].items()})

    _composed_lines(res["composed_haiku"], "COMPOSED ACCURACY, cached haiku standard majority-3 judge")
    print("  agreement with replay (composed, haiku):")
    for k, v in res["agreement"]["composed_haiku"].items():
        print(f"    {k:<17} {_f(v['got'], '.3f')} vs {v['quoted']} "
              f"{'OK' if v['agrees'] else 'DIFFERS'}")
    if res.get("composed_sonnet"):
        sb = res["composed_sonnet"]
        print(f"\n  sonnet cache coverage: {sb['coverage']['n_debates_covered']} of "
              f"{sb['coverage']['n_debates']} debates "
              f"({_f(sb['coverage']['share'], '.3f')}), "
              f"{sb['coverage']['n_cells_covered']} of {sb['coverage']['n_cells']} cells; "
              f"sonnet majority codable on {sb['coverage']['n_debates_sonnet_codable']} covered debates")
        _composed_lines(sb["block"], "COMPOSED ACCURACY, cached sonnet standard majority-3 judge (16.11a), covered debates only")
    else:
        print("\n  sonnet judge: NOT COMPUTABLE (cache absent)")

    print("\n  TRANSFER WITHIN STRATA (grok-solo standard majority-3 wrong, debate level):")
    for st, e in res["transfer_within_strata"].items():
        print(f"    {st:<11} {_conc_line(e)}")

    print("\n" + "=" * 78)
    print("ADDENDUM 16.15.2 -- GROK-SOLO'S OWN SIGNALS vs THE COLLECTIVE'S COUNTER (cells)")
    print("=" * 78)
    sb = res["solo_signals"]
    print(f"  {sb['n_cells_codable']} codable of {sb['n_cells_total']} cells, "
          f"{sb['n_items']} items; grok-solo accuracy {_f(sb['solo_accuracy'], '.3f')}")
    for name, s in sb["signals"].items():
        c = s["concentration"]
        cn = s["composed_natural"]
        print(f"  {name:<13} {_conc_line(c)}")
        print(f"  {'':<13} composed(natural) {_f(cn['acc'], '.3f')} {_ci(cn)}  "
              f"judge emitted {_f(s['judge_emitted_on_fired'], '.3f')}")
    print("  paired vs counter (natural coverage):")
    for k, v in sb["paired_vs_counter_natural"].items():
        print(f"    {k:<28} {_f(v['delta'])} {_ci(v)}  "
              f"{'excludes 0' if v['ci_excludes_zero'] else 'includes 0'}")
    for tname, blk in sb["matched_coverage"].items():
        print(f"  matched coverage [{tname}]: n_fired={blk['target_n']} "
              f"(cov {_f(blk['coverage'], '.3f')}), {TRIM_SEEDS} trim seeds, mean / best")
        for name in ("counter", "any_eshnah", "disagree", "disagree_raw", "composite", "random"):
            v = blk[name]
            print(f"    {name:<13} {_f(v['mean'], '.3f')} / {_f(v['best'], '.3f')}  "
                  f"(p05 {_f(v['p05'], '.3f')}, p95 {_f(v['p95'], '.3f')})")
    print("  agreement with replay (ratios):")
    for k, v in res["agreement"]["solo_signal_ratio"].items():
        print(f"    {k:<13} {_f(v['got'], '.2f')}x vs {v['quoted']}x "
              f"{'OK' if v['agrees'] else 'DIFFERS'}")

    print("\n" + "=" * 78)
    print("ADDENDUM 16.15.3 -- THE FOUR R5 TRANSFER READOUTS, registered estimator (cells)")
    print("=" * 78)
    print(f"  {tr['n_cells']} cells, {tr['n_items']} items")
    for label, t in tr["targets"].items():
        print(f"  {label:<20} {t['model']} {t['scaffold']}  codable cells {t['n_cells_codable']}  "
              f"solo acc {_f(t['solo_accuracy'], '.3f')}  samples/cell {t['n_samples_per_cell']}")
        for sname, e in t["by_sensor"].items():
            print(f"      {sname:<10} {_conc_line(e)}")
    print(f"  readouts with CI excluding zero: counter {tr['n_targets_ci_excludes_zero_counter']}/4, "
          f"composite {tr['n_targets_ci_excludes_zero_composite']}/4")
    print("  agreement with R5 quoted (composite flag; tolerance check only, cite the artefact):")
    for label, a in tr["r5_agreement"].items():
        if label in ("note", "tolerance"):
            continue
        print(f"    {label:<20} {'OK' if a['agrees'] else 'DIFFERS'}  " + "  ".join(
            f"{k} {_f(v['got'], '.3f')}/{v['quoted']}" for k, v in a["fields"].items()))

    _readings_lines(res["pre_declared_readings"])

    print("\n" + "-" * 78)
    for k, c in g["checks"].items():
        print(f"  guard {k:<48} {_f(c['value'], '.4f')}  limit {c['limit']}  "
              f"{'ok' if c['ok'] else 'FAIL'}")
    print("  GUARDS PASSED" if g["passed"] else "  GUARD FAILED -- numbers above are NOT to be read")


# ---------------------------------------------------------------------------
# Agreement with the auditor's replay values
# ---------------------------------------------------------------------------

def _agree(got, quoted, tol):
    return {"got": got, "quoted": quoted,
            "agrees": got is not None and abs(got - quoted) <= tol}


def replay_agreement(res: dict) -> dict:
    out: dict = {"stratum_lift": {}, "composed_haiku": {}, "solo_signal_ratio": {}}
    for st in ("one_loser", "both_party"):
        e = res["counter_lift_by_stratum"][st]
        q = REPLAY[st]
        fields = {
            "p_wrong_fired": _agree(e["p_wrong_fired"], q["p_wrong_fired"], 0.005),
            "p_wrong_unfired": _agree(e["p_wrong_unfired"], q["p_wrong_unfired"], 0.005),
            "lift": _agree(e["lift"], q["lift"], 0.005),
        }
        if "n_fired" in q:
            fields["n_fired"] = _agree(e["n_fired"], q["n_fired"], 0)
        out["stratum_lift"][st] = {"fields": fields,
                                   "agrees": all(v["agrees"] for v in fields.values())}
    for k, q in REPLAY["composed_haiku"].items():
        out["composed_haiku"][k] = _agree(
            res["composed_haiku"]["routers"][k]["composed"]["acc"], q, 0.005)
    for k, q in REPLAY["solo_signal_ratio"].items():
        out["solo_signal_ratio"][k] = _agree(
            res["solo_signals"]["signals"][k]["concentration"]["ratio"], q, 0.1)
    return out


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse(debates: Sequence[dict], raw_rows: Sequence[dict], comp: dict,
            sonnet_comp: Optional[dict], *, draws: int, seed: int,
            trim_seeds: int) -> tuple[dict, dict]:
    haiku_rule = make_rule_vendor(comp, *HAIKU_JUDGE)
    solo_rule = make_rule_vendor(comp, *GROK_SOLO)

    res: dict = {"registration": "Addendum 16.15.1 / 16.15.2", "draws": draws,
                 "seed": seed, "trim_seeds": trim_seeds,
                 "population": "grok panel, codable debates, code_response collapse",
                 "verdict_type_source": "S2 final verdict (`verdict` column)",
                 "judge_haiku": list(HAIKU_JUDGE), "judge_sonnet": list(SONNET_JUDGE),
                 "grok_solo": list(GROK_SOLO)}
    res["stratum_counts"] = {
        st: sum(1 for d in debates if verdict_type(d) == st)
        for st in ("one_loser", "both_party", "other")}
    res["counter_fire_total"] = sum(1 for d in debates if counter_router(d))

    # 16.15.1 -- lift within strata on the deliberation's own errors
    res["counter_lift_by_stratum"] = stratum_lifts(debates, counter_router,
                                                   draws=draws, seed=seed)
    # sensitivity: verdict type read from S1 (moderator synthesis) instead of S2
    s1_bp = lambda d: d.get("s1") in BOTH_PARTY
    s1_union = lambda d: counter_router(d) or s1_bp(d)
    n = len(debates)
    res["s1_verdict_type_sensitivity"] = {
        "note": ("point estimates only, no CI, not registered: the registered "
                 "verdict type is the deployed S2 verdict (`verdict` column), "
                 "which alone reproduces the 93/235 firing split and 16.12's "
                 "272/118; cite this variant only with its own intervals"),
        "n_s1_both_party": sum(1 for d in debates if s1_bp(d)),
        "composed_s1_verdict_type": sum(composed_ok(d, s1_bp, haiku_rule) for d in debates) / n,
        "composed_s1_union": sum(composed_ok(d, s1_union, haiku_rule) for d in debates) / n,
    }

    # composed accuracy, haiku judge
    res["composed_haiku"] = composed_block(debates, haiku_rule, draws=draws,
                                           seed=seed, trim_seeds=trim_seeds,
                                           judge_label="claude-haiku-4-5 standard majority-3 (cached)")
    # composed accuracy, sonnet judge on covered debates
    if sonnet_comp is not None:
        sonnet_rule = make_rule_vendor(sonnet_comp, *SONNET_JUDGE)
        cov = sonnet_covered(debates, sonnet_comp)
        cells_all = {(d["arm"], d["item"]) for d in debates}
        cells_cov = {(d["arm"], d["item"]) for d in cov}
        res["composed_sonnet"] = {
            "coverage": {"n_debates": n, "n_debates_covered": len(cov),
                         "share": len(cov) / n if n else None,
                         "n_cells": len(cells_all), "n_cells_covered": len(cells_cov),
                         "n_debates_sonnet_codable": sum(
                             1 for d in cov if sonnet_rule(d) is not None
                             and correct(sonnet_rule(d), d["gold"]) is not None)},
            "block": composed_block(cov, sonnet_rule, draws=draws, seed=seed,
                                    trim_seeds=trim_seeds,
                                    judge_label="claude-sonnet-4-6 standard majority-3 (16.11a cache)"),
        }
    else:
        res["composed_sonnet"] = None

    # transfer within strata (debate level; the solo majority can tie -> dropped)
    def solo_wrong(d):
        ok = solo_ok(d, solo_rule)
        return None if ok is None else (not ok)
    res["transfer_within_strata"] = stratum_lifts(
        debates, counter_router, draws=draws, seed=seed, wrong=solo_wrong)

    # 16.15.2 -- cells
    cells = build_cells(debates, comp)
    res["solo_signals"] = solo_signal_block(cells, solo_rule, haiku_rule,
                                            draws=draws, seed=seed,
                                            trim_seeds=trim_seeds)

    # 16.15.3 -- transfer readouts
    tr = transfer_readouts(cells, comp, draws=draws, seed=seed)
    tr["registration"] = "Addendum 16.15.3"
    tr["draws"], tr["seed"] = draws, seed
    tr["cell_flag_rule"] = "fires when strictly more than half of the cell's codable debate samples fire"
    tr["r5_quoted"] = R5_QUOTED
    tr["r5_agreement"] = r5_agreement(tr)

    res["agreement"] = replay_agreement(res)
    res["pre_declared_readings"] = pre_declared_readings(res)
    res["resampling_frame"] = (
        "every concentration interval resamples the items of the FULL registered "
        "population (all codable debates / all cells) and applies the stratum or "
        "codability restriction inside each resample; see n_items_frame / "
        "n_items_kept in each block")
    res["guards"] = run_guards(raw_rows, debates, res["composed_haiku"],
                               res["composed_sonnet"]["block"] if res["composed_sonnet"] else None,
                               res["solo_signals"])
    tr["guards_passed"] = res["guards"]["passed"]
    tr["resampling_frame"] = res["resampling_frame"]
    return res, tr


# ---------------------------------------------------------------------------
# Selftest on synthetic debates
# ---------------------------------------------------------------------------

def _synthetic() -> tuple[list[dict], dict]:
    """Hand-built panel. One-loser debates: the counter fires on the wrong
    ones far more than on the right ones (positive lift). Both-party debates:
    the counter fires MORE on the right ones (negative lift). The haiku judge
    is always right; grok-solo is wrong exactly where the counter fires at the
    cell level, and its own three samples carry ESH on exactly those cells."""
    debates = []
    comp: dict = defaultdict(list)
    base = {"model": "m", "arm": "as_asker", "syn_ok": True, "n_reject": 0,
            "unanimous_accept": True, "verdict_revised": False,
            "obj_writer_advocate": False, "obj_counterparty": False,
            "obj_neutral_adjudicator": False,
            "undermined_writer_advocate": True, "undermined_counterparty": True}

    def add(item, idx, s2, gold, n_obj):
        s2_ok = code_response(s2, "published") == (1 if gold == "YTA" else 0)
        d = dict(base, item=item, sample_idx=idx, s1=s2, s2=s2, gold=gold,
                 s2_ok=s2_ok, syn_code=code_response(s2, "published"),
                 n_objectors=n_obj)
        debates.append(d)

    # 20 one-loser items x 2 samples: items 0-9 wrong, counter fires on 8 of
    # them; items 10-19 right, counter fires on 1 of them.
    for i in range(20):
        wrong = i < 10
        fire = (wrong and i < 8) or (not wrong and i == 10)
        gold = "YTA"
        s2 = "NTA" if wrong else "YTA"
        for idx in range(2):
            add(f"ol{i}", idx, s2, gold, 2 if fire else 1)
        cell_fires = fire
        # grok solo: wrong iff the cell fires; samples carry ESH iff it fires
        comp[(GROK, "standard", "as_asker", f"ol{i}")] = (
            ["NTA", "NAH", "NTA"] if cell_fires else ["YTA", "YTA", "YTA"])
        comp[("claude-haiku-4-5", "standard", "as_asker", f"ol{i}")] = ["YTA"] * 3
    # 10 both-party items: items 0-4 wrong (gold NTA, verdict ESH), counter
    # fires on 1; items 5-9 right (gold NTA, verdict NAH), counter fires on 4.
    for i in range(10):
        wrong = i < 5
        fire = (wrong and i == 0) or (not wrong and i >= 6)
        s2 = "ESH" if wrong else "NAH"
        for idx in range(2):
            add(f"bp{i}", idx, s2, "NTA", 3 if fire else 0)
        comp[(GROK, "standard", "as_asker", f"bp{i}")] = (
            ["YTA", "NAH", "YTA"] if fire else ["NTA", "NTA", "NTA"])
        comp[("claude-haiku-4-5", "standard", "as_asker", f"bp{i}")] = ["NTA"] * 3
    return debates, comp


def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    debates, comp = _synthetic()
    haiku_rule = make_rule_vendor(comp, *HAIKU_JUDGE)
    solo_rule = make_rule_vendor(comp, *GROK_SOLO)

    check("verdict_type reads YTA/NTA as one-loser and ESH/NAH as both-party",
          verdict_type({"s2": "YTA"}) == "one_loser"
          and verdict_type({"s2": "NTA"}) == "one_loser"
          and verdict_type({"s2": "ESH"}) == "both_party"
          and verdict_type({"s2": "NAH"}) == "both_party"
          and verdict_type({"s2": "UNRESOLVED"}) == "other")

    L = stratum_lifts(debates, counter_router, draws=200, seed=13)
    ol, bp = L["one_loser"], L["both_party"]
    check("one-loser: P(wrong|fired) = 16/18 and P(wrong|unfired) = 4/22",
          abs(ol["p_wrong_fired"] - 16 / 18) < 1e-9
          and abs(ol["p_wrong_unfired"] - 4 / 22) < 1e-9)
    check("one-loser lift is POSITIVE with CI excluding zero",
          ol["lift"] > 0 and ol["ci_excludes_zero"] and ol["lo"] > 0)
    check("both-party: P(wrong|fired) = 2/10 and P(wrong|unfired) = 8/10",
          abs(bp["p_wrong_fired"] - 0.2) < 1e-9
          and abs(bp["p_wrong_unfired"] - 0.8) < 1e-9)
    check("both-party lift is NEGATIVE", bp["lift"] < 0)
    check("pooled lift mixes the two strata (n_fired = 18 + 10)",
          L["pooled"]["n_fired"] == 28)
    # resampling frame: every item of the population is resampled for every
    # stratum; only the records inside the stratum are scored
    check("stratum CI resamples the FULL frame (30 items) and keeps only the stratum's items (20 / 10)",
          ol["n_items_frame"] == 30 and ol["n_items_kept"] == 20
          and bp["n_items_frame"] == 30 and bp["n_items_kept"] == 10
          and ol["n_frame"] == 60 and ol["n"] == 40 and bp["n"] == 20)
    ol_reduced = concentration([{"item": d["item"], "f": int(counter_router(d)),
                                 "w": int(not d["s2_ok"])}
                                for d in debates if verdict_type(d) == "one_loser"],
                               draws=200, seed=13)
    check("point estimate and n are identical under the reduced (stratum-only) frame",
          ol_reduced["lift"] == ol["lift"] and ol_reduced["n"] == ol["n"]
          and ol_reduced["n_fired"] == ol["n_fired"] and ol_reduced["n_items_frame"] == 20)
    check("a w=None record is in the frame but not the readout",
          concentration([{"item": "a", "f": 1, "w": 1}, {"item": "b", "f": 0, "w": 0},
                         {"item": "c", "f": 1, "w": None}], draws=20, seed=1)["n"] == 2)

    n_counter = sum(1 for d in debates if counter_router(d))
    n_vt = sum(1 for d in debates if verdict_type_router(d))
    n_union = sum(1 for d in debates if union_router(d))
    n_or = sum(1 for d in debates if counter_router(d) or verdict_type_router(d))
    check("union coverage == counter OR verdict-type (and < counter + verdict-type)",
          n_union == n_or and n_union < n_counter + n_vt)
    check("verdict-type router fires on every both-party debate and no other",
          n_vt == 20 and all(verdict_type_router(d) == (verdict_type(d) == "both_party")
                             for d in debates))

    blk = composed_block(debates, haiku_rule, draws=200, seed=13, trim_seeds=20,
                         judge_label="synthetic")
    r = blk["routers"]
    check("collective alone == mean S2 accuracy",
          abs(r["collective_alone"]["composed"]["acc"]
              - sum(d["s2_ok"] for d in debates) / len(debates)) < 1e-12)
    check("a perfect judge: verdict-type router repairs every both-party error "
          "(composed = 1 - 20/60)",
          abs(r["verdict_type"]["composed"]["acc"] - (1 - 20 / 60)) < 1e-12)
    check("union >= max(counter, verdict-type) under a perfect judge",
          r["union"]["composed"]["acc"] >= max(r["counter"]["composed"]["acc"],
                                               r["verdict_type"]["composed"]["acc"]))
    check("random routing at the union's coverage fires exactly n_union",
          blk["random_at_union_coverage"]["target_n"] == n_union)
    check("random routing best <= union (perfect judge, same coverage)",
          blk["random_at_union_coverage"]["best"] <= r["union"]["composed"]["acc"] + 1e-12)
    pd_ = blk["paired_deltas"]["union_minus_collective_alone"]
    check("paired delta union - collective is positive with CI excluding zero",
          pd_["delta"] > 0 and pd_["ci_excludes_zero"])
    check("fallback: a silent judge leaves the debate on S2",
          composed_ok({"item": "x", "arm": "as_asker", "sample_idx": 0,
                       "gold": "YTA", "s2_ok": False, "n_objectors": 2},
                      counter_router, lambda d: None) is False)

    cells = build_cells(debates, comp)
    check("one cell per (arm, item): 30 cells from 60 debates", len(cells) == 30)
    c_by = {c["item"]: c for c in cells}
    check("cell fires when strictly more than half its samples fire",
          c_by["ol0"]["counter_fired"] and not c_by["ol9"]["counter_fired"]
          and c_by["bp0"]["counter_fired"] and not c_by["bp5"]["counter_fired"])
    check("solo any-ESH/NAH and disagreement read the three cached samples",
          c_by["ol0"]["solo_any_eshnah"] and not c_by["ol0"]["solo_disagree"]
          and c_by["bp0"]["solo_any_eshnah"] and c_by["bp0"]["solo_disagree"]
          and not c_by["ol9"]["solo_any_eshnah"] and not c_by["ol9"]["solo_disagree"])
    check("raw-string disagreement is a superset of code-collapse disagreement",
          all(c["solo_disagree_raw"] or not c["solo_disagree"] for c in cells))
    # half-and-half cell must NOT fire
    half = build_cells([dict(debates[0], n_objectors=2, sample_idx=0),
                        dict(debates[0], n_objectors=1, sample_idx=1)], comp)
    check("a 1-of-2 cell does not fire (strict majority)", not half[0]["counter_fired"])

    sb = solo_signal_block(cells, solo_rule, haiku_rule, draws=200, seed=13,
                           trim_seeds=20)
    cc = sb["signals"]["counter"]["concentration"]
    check("grok-solo is wrong exactly on counter-fired cells: P(wrong|fired)=1, "
          "P(wrong|unfired)=0", cc["p_wrong_fired"] == 1.0 and cc["p_wrong_unfired"] == 0.0)
    check("any-ESH/NAH signal coincides with the counter here (same concentration)",
          sb["signals"]["any_eshnah"]["concentration"]["lift"] == cc["lift"])
    check("counter composed with a perfect judge on the codable cells is 1.0",
          abs(sb["signals"]["counter"]["composed_natural"]["acc"] - 1.0) < 1e-12)
    check("matched-coverage block trims to the common minimum n_fired",
          sb["matched_coverage"]["common_min"]["target_n"]
          == min(sb["signals"][k]["concentration"]["n_fired"]
                 for k in ("counter", "any_eshnah", "disagree")))
    check("random at matched coverage does not beat the counter (perfect judge)",
          sb["matched_coverage"]["counter_n"]["random"]["best"]
          <= sb["matched_coverage"]["counter_n"]["counter"]["mean"] + 1e-12)

    # transfer readouts on the synthetic comparator: only grok standard present
    comp2 = {k: v for k, v in comp.items() if k[:2] == GROK_SOLO}
    tr = transfer_readouts(cells, comp2, draws=200, seed=13)
    g = tr["targets"]["grok_solo_standard"]["by_sensor"]["counter"]
    check("transfer: grok standard lift = +1.0 with CI excluding zero",
          g["lift"] == 1.0 and g["ci_excludes_zero"])
    check("transfer: a target with no cached rows has zero codable cells and no lift",
          tr["targets"]["haiku_standard"]["n_cells_codable"] == 0
          and tr["targets"]["haiku_standard"]["by_sensor"]["counter"]["lift"] is None)
    check("count of readouts with CI excluding zero is 1 of 4 on the synthetic cache",
          tr["n_targets_ci_excludes_zero_counter"] == 1)

    # guards
    raw = [{"n_r3_unparsed": 0} for _ in debates]
    gd = run_guards(raw, debates, blk, None, sb)
    check("guards pass on the clean synthetic panel", gd["passed"])
    raw_bad = [{"n_r3_unparsed": 1} for _ in debates]
    gd2 = run_guards(raw_bad, debates, blk, None, sb)
    check("guards fail when R3 parsing is broken", not gd2["passed"])
    gd3 = run_guards(raw + [{"n_r3_unparsed": 0}] * 10, debates, blk, None, sb)
    check("guards fail when too many debates are non-codable", not gd3["passed"])

    # end-to-end on the synthetic panel; the literal GUARD FAILED must reach
    # stdout through print_report when a guard fails
    import contextlib
    import io
    res_syn, tr_syn = analyse(debates, raw, comp, None, draws=50, seed=13, trim_seeds=5)
    check("analyse() runs end-to-end on the synthetic panel with guards passing",
          res_syn["guards"]["passed"] and tr_syn["guards_passed"])
    pr = res_syn["pre_declared_readings"]
    check("pre-declared readings: sensor claim holds on the synthetic panel",
          pr["16.15.1_sensor_claim"]["holds"])
    check("pre-declared readings: complementarity block names the judge",
          "haiku" in pr["16.15.1_complementarity"]
          and "haiku judge" in pr["16.15.1_complementarity"]["haiku"]["reading"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_report(res_syn, tr_syn)
    check("print_report does NOT print GUARD FAILED when guards pass",
          "GUARD FAILED" not in buf.getvalue() and "GUARDS PASSED" in buf.getvalue())
    res_bad = dict(res_syn, guards=dict(res_syn["guards"], passed=False))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_report(res_bad, tr_syn)
    check("print_report prints the literal GUARD FAILED on a failing guard dict",
          "GUARD FAILED" in buf.getvalue())

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Addendum 16.15.1-16.15.3 router decomposition")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-boot", type=int, default=DEFAULT_DRAWS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--trim-seeds", type=int, default=TRIM_SEEDS)
    ap.add_argument("--rows", type=Path, default=ROWS_PATH)
    ap.add_argument("--votes", type=Path, default=VOTES_PATH)
    ap.add_argument("--comparator", type=Path, default=COMPARATOR_PATH)
    ap.add_argument("--sonnet-rows", type=Path, default=SONNET_ROWS_PATH)
    ap.add_argument("--out", type=Path, default=ANALYSIS_PATH)
    ap.add_argument("--transfer-out", type=Path, default=TRANSFER_PATH)
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()

    debates = load_debates(a.rows, a.votes)
    raw_rows = load_rows(a.rows, GROK)
    comp = load_comparator(a.comparator)
    sonnet_comp = load_comparator(a.sonnet_rows) if a.sonnet_rows.exists() else None
    if sonnet_comp is None:
        print(f"sonnet cache {a.sonnet_rows} absent: sonnet block NOT COMPUTABLE")
    res, tr = analyse(debates, raw_rows, comp, sonnet_comp, draws=a.n_boot,
                      seed=a.seed, trim_seeds=a.trim_seeds)
    res["inputs"] = {"rows": str(a.rows), "votes": str(a.votes),
                     "comparator": str(a.comparator),
                     "sonnet_rows": str(a.sonnet_rows) if sonnet_comp else None}
    tr["inputs"] = dict(res["inputs"])
    print_report(res, tr)
    a.out.write_text(json.dumps(res, indent=2, default=str))
    a.transfer_out.write_text(json.dumps(tr, indent=2, default=str))
    print(f"\nwrote {a.out}\nwrote {a.transfer_out}")
    if not res["guards"]["passed"]:
        print("GUARD FAILED")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
