"""
analyze_routing_judge_deepseek.py -- UNREGISTERED post-review check (W44, 2026-09-23).

Question. The pillar-3 routing gain (routed minus collective) was registered with one
judge: claude-sonnet-4-6, standard scaffold, majority-of-three. Does the gain depend on
that judge? Here the counter-flagged debates of each of the six verified conditions are
sent instead to the cached DeepSeek-V4-Pro standard-scaffold majority-of-three verdicts
(divergence_study_outputs/cg_deepseek_standard_k3_rows.csv, a pre-existing cache from
Addendum 17.1's screening work, file dated 2026-09-14; three verdicts for every (arm, item)
pair). The sonnet column is recomputed
alongside and checked against the registered values, so the pipeline is verified in the
same run.

Everything else is the repository's own independent recomputation code
(scripts.verify_pillar3_headline): the same debate loader, the same routing rule (the
collective's S2 verdict unless the counter fired, then the judge's majority-of-three;
when the judge has no majority for the pair -- a tie, NOVERDICT/UNRESOLVED rows, or no
cached verdict -- the S2 verdict is kept) and the same item-clustered percentile
bootstrap (4,000 draws, seed 101). Only the judge CSV differs.

Cache-only: no model calls, no network access, no import of scripts.generators. Reads
only files under divergence_study_outputs/.

Run:
  python -m scripts.analyze_routing_judge_deepseek            # writes divergence_study_outputs/routing_judge_deepseek.json
  python -m scripts.analyze_routing_judge_deepseek --selftest
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import datetime as _dt
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import scripts.verify_pillar3_headline as v

OUT = v.OUT
CONDITIONS = ("grok", "noedge", "haiku_249", "llama", "deepseek", "mistral")
JUDGES = {
    "sonnet": {"file": "cg_sonnet_actuator_rows.csv", "model": "claude-sonnet-4-6",
               "role": "registered comparator (pillar-3 headline judge)"},
    "deepseek": {"file": "cg_deepseek_standard_k3_rows.csv", "model": "DeepSeek-V4-Pro",
                 "role": "unregistered alternative judge; pre-existing cache (file dated 2026-09-14; the pre-registration "
                         "log's 16.15.3 extension note attributes it to Addendum 17.1's screening work)"},
}
DRAWS, SEED = 4000, 101
STATUS = "UNREGISTERED"
FORBIDDEN_MODULES = ("scripts.generators", "requests", "httpx", "urllib3", "openai", "anthropic", "aiohttp")


# ---------------------------------------------------------------------------- loading
@contextlib.contextmanager
def judge_swapped(name: str, judge_file: str, judge_model: str):
    """Temporarily point one community's judge at another cached CSV; always restore."""
    saved = copy.deepcopy(v.COMMUNITIES)
    try:
        v.COMMUNITIES[name]["judge"] = judge_file
        v.COMMUNITIES[name]["judge_model"] = judge_model
        yield
    finally:
        v.COMMUNITIES.clear()
        v.COMMUNITIES.update(saved)


def load_with_judge(name: str, judge_key: str):
    j = JUDGES[judge_key]
    with judge_swapped(name, j["file"], j["model"]):
        return v.load(name)


def majority(codes):
    """Majority of the coded verdicts; None when there is none (same rule as v.load)."""
    codes = [c for c in codes if c is not None]
    n1 = sum(codes); n0 = len(codes) - n1
    return None if (not codes or n1 == n0) else int(n1 > n0)


def judge_map(judge_key: str):
    j = JUDGES[judge_key]
    jm = defaultdict(list)
    with open(OUT / j["file"]) as fh:
        for r in csv.DictReader(fh):
            if r["model"] == j["model"] and r["scaffold"] == "standard":
                jm[(r["arm"], r["item_id"])].append(v.code(r["verdict"]))
    return {k: majority(c) for k, c in jm.items()}


# ---------------------------------------------------------------------------- statistics
def judge_stats(deb):
    """Judge-everywhere accuracy, and coverage/accuracy of the judge on the fired debates."""
    je = [d["judge_ok"] for d in deb if d["judge_ok"] is not None]
    fired = [d for d in deb if d["fired"]]
    cov = [d for d in fired if d["judge_ok"] is not None]
    return {
        "judge_everywhere_acc": round(sum(je) / len(je), 4) if je else None,
        "judge_everywhere_n": len(je),
        "fired_with_judge_verdict": len(cov),
        "fired_without_judge_verdict_kept_s2": len(fired) - len(cov),
        "judge_acc_on_fired": round(sum(d["judge_ok"] for d in cov) / len(cov), 4) if cov else None,
        "collective_acc_on_fired": round(sum(d["ok"] for d in fired) / len(fired), 4) if fired else None,
    }


def boot_paired_items(per_item_diffs: dict, draws: int = DRAWS, seed: int = SEED):
    """Item-clustered percentile bootstrap of a mean per-debate difference (mirrors v.boot_delta)."""
    items = sorted(per_item_diffs)
    arr = [np.array(per_item_diffs[i], dtype=float) for i in items]
    sums = np.array([a.sum() for a in arr]); cnts = np.array([len(a) for a in arr])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(items), size=(draws, len(items)))
    stats = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(sums.sum() / cnts.sum()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def routed_difference(deb_a, deb_b, draws: int = DRAWS, seed: int = SEED):
    """Routed accuracy under judge B minus under judge A, paired by debate, item-clustered CI."""
    key = lambda d: (d["arm"], d["item"], d["sample"])
    assert [key(d) for d in deb_a] == [key(d) for d in deb_b], "debate lists must align"
    per = defaultdict(list)
    for a, b in zip(deb_a, deb_b):
        per[a["item"]].append(v.routed_ok(b) - v.routed_ok(a))
    return boot_paired_items(per, draws, seed)


def judge_agreement_on_fired(deb, map_a: dict, map_b: dict):
    """Share of counter-fired (arm, item) pairs on which both judges have a majority and agree."""
    fired = {(d["arm"], d["item"]) for d in deb if d["fired"]}
    both = [k for k in fired if map_a.get(k) is not None and map_b.get(k) is not None]
    agree = sum(1 for k in both if map_a[k] == map_b[k])
    return {"fired_pairs": len(fired), "pairs_with_both_verdicts": len(both), "agree": agree,
            "share": round(agree / len(both), 4) if both else None}


def check_registered(name: str, r: dict) -> bool:
    reg = v.COMMUNITIES[name]["registered"]
    return all(abs(r[k] - reg[k]) <= v.TOL[k]
               for k in ("n", "collective", "fire", "n_fired", "routed", "delta", "lo", "hi"))


# ---------------------------------------------------------------------------- driver
def judge_block(r: dict, deb, draws: int, seed: int) -> dict:
    blk = {"routed": round(r["routed"], 4), "gain": round(r["delta"], 4),
           "gain_lo": round(r["lo"], 4), "gain_hi": round(r["hi"], 4),
           "gain_excludes_zero": bool(r["lo"] > 0 or r["hi"] < 0)}
    blk.update(judge_stats(deb))
    blk["breakdown"] = r["breakdown"]
    return blk


def run(conditions=CONDITIONS, draws: int = DRAWS, seed: int = SEED, verbose: bool = True) -> dict:
    out = {
        "registration_status": STATUS,
        "note": ("Post-review check W44, 2026-09-23. Not registered in prereg_embodiment_community.md. "
                 "Zero model calls: both judge columns are cached CSVs. Routing rule and bootstrap are "
                 "scripts.verify_pillar3_headline's; only the judge CSV differs."),
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "method": {
            "routing_rule": "S2 (collective verdict) unless the counter fired (n_objectors >= 2), then the judge's "
                            "standard-scaffold majority-of-three; S2 kept when the judge has no majority for the pair",
            "gain": "routed accuracy minus collective accuracy, gold = published ESH coding",
            "bootstrap": {"kind": "item-clustered percentile, numpy", "draws": draws, "seed": seed},
            "code": "scripts/verify_pillar3_headline.py (load, compute, boot_delta) via scripts/analyze_routing_judge_deepseek.py",
        },
        "judges": JUDGES,
        "conditions": {},
    }
    maps = {k: judge_map(k) for k in JUDGES}
    for name in conditions:
        debs, res = {}, {}
        for jk in JUDGES:
            debs[jk] = load_with_judge(name, jk)
            res[jk] = v.compute(debs[jk])
        rs, rd = res["sonnet"], res["deepseek"]
        assert rs["n"] == rd["n"] and rs["n_fired"] == rd["n_fired"] and abs(rs["collective"] - rd["collective"]) < 1e-12
        diff = routed_difference(debs["sonnet"], debs["deepseek"], draws, seed)
        cond = {
            "rows": v.COMMUNITIES[name]["rows"], "n": rs["n"],
            "collective": round(rs["collective"], 4), "fire": round(rs["fire"], 4), "n_fired": rs["n_fired"],
            "sonnet": judge_block(rs, debs["sonnet"], draws, seed),
            "deepseek": judge_block(rd, debs["deepseek"], draws, seed),
            "deepseek_minus_sonnet_routed": {"point": round(diff[0], 4), "lo": round(diff[1], 4), "hi": round(diff[2], 4),
                                             "excludes_zero": bool(diff[1] > 0 or diff[2] < 0)},
            "judge_agreement_on_fired": judge_agreement_on_fired(debs["sonnet"], maps["sonnet"], maps["deepseek"]),
            "sonnet_registered_reproduced": check_registered(name, rs),
            "reading": "GAIN-HOLDS" if rd["lo"] > 0 else ("REVERSES" if rd["hi"] < 0 else "NULL"),
        }
        cond["sonnet"]["registered"] = {k: v.COMMUNITIES[name]["registered"][k]
                                        for k in ("routed", "delta", "lo", "hi", "source")}
        out["conditions"][name] = cond
        if verbose:
            print(f"{name:10s} n {rs['n']:5d} coll {rs['collective']:.4f} fired {rs['n_fired']:4d} ({rs['fire']:.3f})  "
                  f"sonnet routed {rs['routed']:.4f} gain {rs['delta']:+.4f} [{rs['lo']:+.4f},{rs['hi']:+.4f}]  "
                  f"deepseek routed {rd['routed']:.4f} gain {rd['delta']:+.4f} [{rd['lo']:+.4f},{rd['hi']:+.4f}]  "
                  f"ds-minus-sonnet {diff[0]:+.4f} [{diff[1]:+.4f},{diff[2]:+.4f}]  "
                  f"ds judge on fired {cond['deepseek']['fired_with_judge_verdict']}/{rs['n_fired']}  "
                  f"agree {cond['judge_agreement_on_fired']['share']}  reg-reproduced {cond['sonnet_registered_reproduced']}")
    c = out["conditions"]
    out["summary"] = {
        "n_conditions": len(c),
        "deepseek_gain_excludes_zero_on": [k for k in c if c[k]["deepseek"]["gain_excludes_zero"]],
        "deepseek_gain_range": [min(c[k]["deepseek"]["gain"] for k in c), max(c[k]["deepseek"]["gain"] for k in c)],
        "sonnet_gain_range": [min(c[k]["sonnet"]["gain"] for k in c), max(c[k]["sonnet"]["gain"] for k in c)],
        "deepseek_minus_sonnet_excludes_zero_on": [k for k in c if c[k]["deepseek_minus_sonnet_routed"]["excludes_zero"]],
        "all_sonnet_registered_reproduced": all(c[k]["sonnet_registered_reproduced"] for k in c),
        "reading": ("GAIN-HOLDS on all conditions under the DeepSeek judge"
                    if all(c[k]["reading"] == "GAIN-HOLDS" for k in c) else "MIXED, see per-condition readings"),
    }
    return out


# ---------------------------------------------------------------------------- selftest
def selftest() -> bool:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(f"  {'ok ' if cond else 'FAIL'} {msg}")
        ok &= bool(cond)

    # 1. no forbidden module has been imported by loading this script and its dependency
    loaded = [m for m in sys.modules if any(m == f or m.startswith(f + ".") for f in FORBIDDEN_MODULES)]
    check(not loaded, f"no generator/network module imported (found {loaded})")
    # 2. neither this script nor its one dependency imports a generator or network module
    bad_names = ("generators", "requests", "httpx", "urllib", "openai", "anthropic", "aiohttp", "socket")
    for f in (Path(__file__), Path(v.__file__)):
        imports = [l.strip() for l in f.read_text().splitlines() if l.startswith(("import ", "from "))]
        offenders = [l for l in imports if any(b in l for b in bad_names)]
        check(not offenders, f"{f.name}: import lines name no generator or network module (found {offenders})")
    # 3. majority rule
    check(majority([1, 1, 0]) == 1 and majority([0, 0, 1]) == 0, "majority of three")
    check(majority([1, 0]) is None and majority([None, None]) is None and majority([1, None, None]) == 1,
          "tie / empty / single verdict handling")
    # 4. the judge swap restores the community table
    before = copy.deepcopy(v.COMMUNITIES)
    with judge_swapped("grok", "x.csv", "x-model"):
        inside = (v.COMMUNITIES["grok"]["judge"], v.COMMUNITIES["grok"]["judge_model"])
    check(inside == ("x.csv", "x-model") and v.COMMUNITIES == before, "judge swap applies and restores")
    # 5. routing arithmetic through v.compute on a synthetic community
    deb_s = [{"arm": "x", "item": "i1", "sample": 0, "gold": 1, "s2": 0, "ok": 0, "n_obj": 2, "n_obj_votes": 2,
              "fired": 1, "one_loser": True, "judge_ok": 1},
             {"arm": "x", "item": "i2", "sample": 0, "gold": 0, "s2": 0, "ok": 1, "n_obj": 0, "n_obj_votes": 0,
              "fired": 0, "one_loser": True, "judge_ok": 0},
             {"arm": "x", "item": "i3", "sample": 0, "gold": 1, "s2": 0, "ok": 0, "n_obj": 3, "n_obj_votes": 3,
              "fired": 1, "one_loser": True, "judge_ok": None}]
    r = v.compute(deb_s)
    check(abs(r["collective"] - 1 / 3) < 1e-12 and abs(r["routed"] - 2 / 3) < 1e-12 and abs(r["delta"] - 1 / 3) < 1e-12,
          "routing keeps S2 when the judge has no verdict; gain = 1/3")
    js = judge_stats(deb_s)
    check(js["fired_with_judge_verdict"] == 1 and js["fired_without_judge_verdict_kept_s2"] == 1
          and js["judge_acc_on_fired"] == 1.0 and js["collective_acc_on_fired"] == 0.0 and js["judge_everywhere_n"] == 2,
          "judge coverage statistics")
    # 6. paired difference between two judges
    deb_d = copy.deepcopy(deb_s)
    deb_d[0]["judge_ok"] = 0          # the second judge gets the first fired debate wrong
    deb_d[2]["judge_ok"] = 1          # and has a (correct) verdict where the first had none
    point, lo, hi = routed_difference(deb_s, deb_d, draws=200, seed=1)
    check(abs(point) < 1e-12 and lo <= 0 <= hi, "paired routed difference: -1/3 + 1/3 = 0 with CI covering 0")
    deb_d[2]["judge_ok"] = 0
    point, lo, hi = routed_difference(deb_s, deb_d, draws=200, seed=1)
    check(abs(point + 1 / 3) < 1e-12 and -1 <= lo <= point <= hi <= 0, "paired routed difference: -1/3")
    # 7. agreement on fired pairs
    agr = judge_agreement_on_fired(deb_s, {("x", "i1"): 1, ("x", "i3"): 0}, {("x", "i1"): 1, ("x", "i3"): None})
    check(agr == {"fired_pairs": 2, "pairs_with_both_verdicts": 1, "agree": 1, "share": 1.0}, "agreement on fired pairs")
    # 8. every input this script can read lives under divergence_study_outputs/
    files = [OUT / j["file"] for j in JUDGES.values()] + [OUT / v.COMMUNITIES[c][k] for c in CONDITIONS for k in ("rows", "votes")]
    check(all(f.resolve().is_relative_to(OUT.resolve()) for f in files), "all inputs are under divergence_study_outputs/")
    check(all(f.exists() for f in files), "all input CSVs are present on disk")
    print("selftest", "OK" if ok else "FAILED")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=OUT / "routing_judge_deepseek.json")
    a = ap.parse_args(argv)
    if a.selftest:
        return 0 if selftest() else 1
    print(f"ROUTING JUDGE CHECK ({STATUS}): flagged debates to DeepSeek-V4-Pro majority-of-3 instead of sonnet")
    out = run(a.conditions, a.draws, a.seed)
    a.out.write_text(json.dumps(out, indent=1))
    s = out["summary"]
    print(f"summary: deepseek gain excludes zero on {len(s['deepseek_gain_excludes_zero_on'])}/{s['n_conditions']} "
          f"({', '.join(s['deepseek_gain_excludes_zero_on'])}); gain range {s['deepseek_gain_range']}; "
          f"deepseek-minus-sonnet excludes zero on {s['deepseek_minus_sonnet_excludes_zero_on'] or 'none'}; "
          f"sonnet registered values reproduced: {s['all_sonnet_registered_reproduced']}")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
