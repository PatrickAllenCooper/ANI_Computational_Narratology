"""
scripts/analyze_seat_scaffold.py -- Addendum 16.28: narrated-seat minus
plain-CoT-seat, paired on items, for the routed collective.

Per model, two deliberation cells on the same screened panel (narrated seats,
the registered community; plain chain-of-thought seats, the 16.28 cell). The
S2 coding, the counter (>= 2 objectors) and the routing rule (S2 unless the
counter fires, then the sonnet standard majority-of-3 judge, ties keep S2) are
imported from `verify_pillar3_headline`, the independent recomputation, so this
comparison shares no code with the registered analyzers either.

Per item (averaged over its arms and samples), narrated minus plain:
  (a) S2 accuracy  (b) routed accuracy  (c) routing gain  (d) fire rate
  (e) asker-shielding of S2 on gold-YTA items:
      P(not at fault | as_asker) - P(not at fault | third_person)
Item-clustered bootstrap over items present in both cells, 8,000 draws.

Amended before spend (pre-spend review, 2026-09-22): (f) coverage-matched routed
accuracy, because routed accuracy can favour whichever seats escalate more (a
flooding cell routes nearly everything to the stronger judge). The higher-firing
cell's flags are randomly thinned to the lower cell's fire rate (200 seeds), each
item's expected routed accuracy is averaged over seeds, then paired as above.
The analyzer refuses to read a cell whose row count or sample set is incomplete.

Run:
  python -m scripts.analyze_seat_scaffold
  python -m scripts.analyze_seat_scaffold --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.verify_pillar3_headline import OUT, SEATS, code, routed_ok

JUDGE = ("cg_sonnet_actuator_rows.csv", "claude-sonnet-4-6")
PAIRS = {
    "grok": {"narrated": "cg_deliberation", "plain": "cg_deliberation_stdcot", "rows": 1680, "samples": 4},
    "llama": {"narrated": "cg_deliberation_llama3370binstruct",
              "plain": "cg_deliberation_llama3370binstruct_stdcot", "rows": 900, "samples": 2},
    # 16.28 amendment E-D (2026-09-23): haiku on the 249-item panel, k = 1; the narrated cell has
    # 491 of 498 rows (16.23 partial-panel rule), so the pair is read on the common debates.
    "haiku": {"narrated": "cg_deliberation_haiku_249", "plain": "cg_deliberation_haiku_249_stdcot",
              "rows": 498, "samples": 1},
}
THIN_SEEDS = 200
DRAWS, SEED = 8000, 20260922


def completeness(tag: str, expected_rows: int, samples: int) -> dict:
    rows = list(csv.DictReader(open(OUT / f"{tag}_rows.csv")))
    idx = sorted({int(r["sample_idx"]) for r in rows})
    ok = len(rows) == expected_rows and idx == list(range(samples))
    comp_p = OUT / f"{tag}_completeness.json"
    if comp_p.exists():
        ok = ok and bool(json.loads(comp_p.read_text()).get("pass"))
    return {"tag": tag, "rows": len(rows), "expected": expected_rows, "sample_idx": idx, "pass": ok}


def load_tag(tag: str) -> list[dict]:
    rows = list(csv.DictReader(open(OUT / f"{tag}_rows.csv")))
    judge = list(csv.DictReader(open(OUT / JUDGE[0])))
    jm = defaultdict(list)
    for j in judge:
        if j["model"] == JUDGE[1] and j["scaffold"] == "standard":
            jm[(j["arm"], j["item_id"])].append(code(j["verdict"]))
    deb = []
    for r in rows:
        s1, s2 = code(r.get("synthesis_verdict", "")), code(r["verdict"])
        if s1 is None or s2 is None:
            continue
        gold = 1 if r["gold_verdict"] == "YTA" else 0
        codes = [c for c in jm.get((r["arm"], r["item_id"]), []) if c is not None]
        n1 = sum(codes); n0 = len(codes) - n1
        jv = None if (not codes or n1 == n0) else int(n1 > n0)
        n_obj = int(r.get("n_objectors") or 0)
        deb.append({"arm": r["arm"], "item": r["item_id"], "sample": int(r["sample_idx"]), "gold": gold,
                    "s2": s2, "ok": int(s2 == gold), "fired": int(n_obj >= 2),
                    "judge_ok": None if jv is None else int(jv == gold)})
    return deb


def per_item(deb: list[dict]) -> dict:
    by = defaultdict(list)
    for d in deb:
        by[d["item"]].append(d)
    out = {}
    for i, ds in by.items():
        acc = np.mean([d["ok"] for d in ds])
        rt = np.mean([routed_ok(d) for d in ds])
        rec = {"acc": acc, "routed": rt, "gain": rt - acc, "fire": np.mean([d["fired"] for d in ds]),
               "gold": ds[0]["gold"], "n": len(ds)}
        if ds[0]["gold"] == 1:
            ask = [1 - d["s2"] for d in ds if d["arm"] == "as_asker"]
            tp = [1 - d["s2"] for d in ds if d["arm"] == "third_person"]
            rec["shield"] = (np.mean(ask) - np.mean(tp)) if ask and tp else None
        out[i] = rec
    return out


def thinned_routed(deb: list[dict], target_fire: float, seeds: int = THIN_SEEDS) -> dict:
    """Per item, expected routed accuracy when this cell's flags are randomly thinned so its
    overall fire rate equals target_fire (no thinning if it already fires at or below it)."""
    fired_idx = [k for k, d in enumerate(deb) if d["fired"]]
    keep_n = min(len(fired_idx), int(round(target_fire * len(deb))))
    acc = defaultdict(list)
    rng = np.random.default_rng(SEED)
    for _ in range(seeds if keep_n < len(fired_idx) else 1):
        keep = set(rng.choice(fired_idx, size=keep_n, replace=False).tolist()) if keep_n < len(fired_idx) \
            else set(fired_idx)
        per = defaultdict(list)
        for k, d in enumerate(deb):
            dd = dict(d, fired=int(k in keep))
            per[d["item"]].append(routed_ok(dd))
        for i, v in per.items():
            acc[i].append(np.mean(v))
    return {i: {"routed_matched": float(np.mean(v))} for i, v in acc.items()}


def paired(a: dict, b: dict, key: str, *, draws=DRAWS, seed=SEED) -> dict:
    items = sorted(i for i in set(a) & set(b) if a[i].get(key) is not None and b[i].get(key) is not None)
    if len(items) < 20:
        return {"n_items": len(items), "diff": None, "lo": None, "hi": None}
    d = np.array([a[i][key] - b[i][key] for i in items], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(items), size=(draws, len(items)))
    bs = d[idx].mean(1)
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    return {"n_items": len(items), "narrated": float(np.mean([a[i][key] for i in items])),
            "plain": float(np.mean([b[i][key] for i in items])), "diff": float(d.mean()),
            "lo": lo, "hi": hi, "ci_excludes_0": bool(lo > 0 or hi < 0)}


def reading(per_model: dict) -> dict:
    # routed accuracy counts toward NARRATION-LOAD-BEARING only at matched coverage
    keys = ("acc", "routed_matched")
    fav_n = {m: any(r[k]["diff"] is not None and r[k]["lo"] > 0 for k in keys if k in r)
             for m, r in per_model.items()}
    fav_p = {m: any(r[k]["diff"] is not None and r[k]["hi"] < 0 for k in keys if k in r)
             for m, r in per_model.items()}
    other = {m: any(r[k]["ci_excludes_0"] for k in ("gain", "fire") if r[k]["diff"] is not None)
             for m, r in per_model.items()}
    if any(fav_p.values()):
        main = "PLAIN-BETTER"
    elif any(fav_n.values()):
        main = "NARRATION-LOAD-BEARING"
    elif any(other.values()):
        main = "AMPLIFIER-ONLY"
    else:
        main = "INERT"
    shield = {}
    for m, r in per_model.items():
        s = r["shield"]
        shield[m] = (None if s["diff"] is None else
                     "NARRATION-SHIELDS-LESS" if s["hi"] < 0 else
                     "NARRATION-SHIELDS-MORE" if s["lo"] > 0 else "NO-DIFFERENCE")
    return {"primary": main, "shielding": shield}


def run() -> dict:
    per_model, comp = {}, {}
    for m, p in PAIRS.items():
        comp[m] = [completeness(p["narrated"], p["rows"], p["samples"]),
                   completeness(p["plain"], p["rows"], p["samples"])]
        if not all(c["pass"] for c in comp[m]):
            per_model[m] = {"UNREAD": "completeness failed", "completeness": comp[m]}
            continue
        da, db = load_tag(p["narrated"]), load_tag(p["plain"])
        a, b = per_item(da), per_item(db)
        fa, fb = np.mean([d["fired"] for d in da]), np.mean([d["fired"] for d in db])
        target = min(fa, fb)
        ta, tb = thinned_routed(da, target), thinned_routed(db, target)
        for i in a:
            a[i]["routed_matched"] = ta.get(i, {}).get("routed_matched")
        for i in b:
            b[i]["routed_matched"] = tb.get(i, {}).get("routed_matched")
        per_model[m] = {k: paired(a, b, k) for k in ("acc", "routed", "routed_matched", "gain", "fire", "shield")}
        per_model[m]["n_items_narrated"], per_model[m]["n_items_plain"] = len(a), len(b)
        per_model[m]["matched_fire_rate"] = float(target)
        per_model[m]["completeness"] = comp[m]
    readable = {m: r for m, r in per_model.items() if "UNREAD" not in r}
    return {"per_model": per_model, "reading": reading(readable) if readable else {"primary": "UNREAD"}}


def _selftest() -> int:
    ok = True

    def check(msg, cond):
        nonlocal ok
        print(("[ok] " if cond else "[FAIL] ") + msg)
        ok = ok and cond

    a = {str(i): {"acc": 1.0, "routed": 1.0, "routed_matched": 1.0, "gain": 0.0, "fire": 0.2, "shield": 0.0}
         for i in range(40)}
    b = {str(i): {"acc": 0.5, "routed": 0.5, "routed_matched": 0.5, "gain": 0.0, "fire": 0.2, "shield": 0.0}
         for i in range(40)}
    r = paired(a, b, "acc")
    check("narrated 1.0 vs plain 0.5 -> +0.5, CI excludes 0", abs(r["diff"] - 0.5) < 1e-9 and r["lo"] > 0)
    K = ("acc", "routed", "routed_matched", "gain", "fire", "shield")
    pm = {"x": {k: paired(a, b, k) for k in K}}
    check("a narrated-favouring accuracy difference reads NARRATION-LOAD-BEARING",
          reading(pm)["primary"] == "NARRATION-LOAD-BEARING")
    pm2 = {"x": {k: paired(b, a, k) for k in K}}
    check("a plain-favouring difference reads PLAIN-BETTER", reading(pm2)["primary"] == "PLAIN-BETTER")
    same = {"x": {k: paired(a, a, k) for k in K}}
    check("identical cells read INERT", reading(same)["primary"] == "INERT")
    deb = load_tag("cg_deliberation")
    pi = per_item(deb)
    acc = np.mean([d["ok"] for d in deb])
    check(f"grok narrated collective reproduces the registered 0.8426 (got {acc:.4f})", abs(acc - 0.8426) < 5e-4)
    rt = np.mean([routed_ok(d) for d in deb])
    check(f"grok narrated routed reproduces the registered 0.9129 (got {rt:.4f})", abs(rt - 0.9129) < 5e-4)
    check("per-item table covers the 210 grok items", len(pi) == 210)
    th = thinned_routed(deb, 0.0)
    check("thinning to zero coverage returns the collective's own accuracy",
          abs(np.mean([v["routed_matched"] for v in th.values()]) - np.mean([pi[i]["acc"] for i in th])) < 1e-9)
    check("the narrated grok cell passes its completeness check",
          completeness("cg_deliberation", 1680, 4)["pass"])
    print("\nselftest: " + ("ALL OK" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", type=Path, default=OUT / "seat_scaffold_comparison.json")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    res = run()
    for m, r in res["per_model"].items():
        if "UNREAD" in r:
            print(f"\n== {m}  UNREAD: {r['completeness']}")
            continue
        print(f"\n== {m}  items narrated {r['n_items_narrated']}  plain {r['n_items_plain']}")
        for k in ("acc", "routed", "routed_matched", "gain", "fire", "shield"):
            v = r[k]
            if v["diff"] is None:
                print(f"  {k:7s} under-gated (n_items {v['n_items']})")
                continue
            print(f"  {k:7s} narrated {v['narrated']:.4f}  plain {v['plain']:.4f}  narrated-plain "
                  f"{v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]  n_items {v['n_items']}")
    print(f"\nREADING: {res['reading']}")
    a.json.write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
