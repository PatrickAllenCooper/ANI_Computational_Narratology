"""
scripts/analyze_narrative_form.py -- readouts for Addendum 17.7 (narrative form
versus content) and 17.4 (length-matched CoT), as amended before spend.

Every contrast is against the SAME-DAY replicate comparators `standard_cot_rep`
and `narrative_cot_rep` (same 2,048 cap, nano's same 8,192 reasoning floor),
never the June cells. Scores come from the 17.8 full judge panel (every response
scored in full by every judge, `rescore_elephant_full_judge.cached_score`);
each judge is read separately and the readings are applied per judge.

17.7 (pooled over generators; the bootstrap clusters on the ITEM because the
same 150 items appear under every generator):
  drop(arm)     rate(arm) - rate(cot_rep)
  share(arm)    drop(arm) / drop(not_rep)
  gap(arm)      rate(not_rep) - rate(arm)      (negative: NoT validates less)
Readings, applied in order to the checklist:
  CHECKLIST-BETTER    gap(checklist) CI excludes 0 above      (reversal)
  NARRATIVE-ADDS      gap(checklist) CI excludes 0 below
  CONTENT-SUFFICIENT  gap(checklist) CI includes 0 and share(checklist) >= 0.75
  UNRESOLVED          otherwise
and to narrative_only in the same order (NARRATIVE-ONLY-BETTER, CONTENT-ADDS,
NARRATIVE-SUFFICIENT, UNRESOLVED); INTEGRATION is flagged when both partial arms'
shares are below 0.5 with NoT's drop CI excluding 0.

17.4 (per generator, haiku and grok), manipulation check first: the achieved
mean length of the length-matched arm over NoT's must lie in [0.75, 1.33], else
LENGTH-MISS (not read). Then, exclusive and ordered:
  LENGTH-INERT     gap(lengthmatched) CI excludes 0 below on both generators
  LENGTH-CARRIES   share(lengthmatched) >= 0.5 and its drop CI excludes 0 below on either
  UNRESOLVED       otherwise

Run:
  python -m scripts.analyze_narrative_form
  python -m scripts.analyze_narrative_form --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.rescore_elephant_full_judge import DEFAULT_JUDGES, cached_score, load_rows
from scripts.rescore_elephant_untruncated import load_questions

COT, NOT = "standard_cot_rep", "narrative_cot_rep"
CHECK, NARR, LEN = "not_checklist", "not_narrative_only", "standard_cot_lengthmatched"
FORM_GENS = ("claude-haiku-4-5", "grok-4-1-fast-reasoning", "gpt-5.4-nano", "Llama-3.3-70B-Instruct")
LEN_GENS = ("claude-haiku-4-5", "grok-4-1-fast-reasoning")
DRAWS, SEED = 8000, 20260922
OUT = Path("divergence_study_outputs")


def scores(rows, judge, questions) -> dict:
    """(gen, arm) -> {item: score}"""
    cells = defaultdict(dict)
    for r in rows:
        s = cached_score(questions[r["item_id"]], r["response"], judge)
        if s in (0, 1):
            cells[(r["generator"], r["arm"])][r["item_id"]] = s
    return cells


def lengths(rows) -> dict:
    out = defaultdict(list)
    for r in rows:
        out[(r["generator"], r["arm"])].append(len(r["response"]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def _stack(cells, gens, arm_a, arm_b):
    """Paired (item, gen) values, clustered by item: item -> list of (a, b)."""
    by = defaultdict(list)
    for g in gens:
        a, b = cells.get((g, arm_a), {}), cells.get((g, arm_b), {})
        for i in set(a) & set(b):
            by[i].append((a[i], b[i]))
    return by


def contrast(cells, gens, base, arm, not_arm=NOT, *, draws=DRAWS, seed=SEED) -> dict:
    """drop(arm), drop(not), share, gap = rate(not) - rate(arm), all on items present in all
    three cells for a generator; cluster bootstrap over items."""
    rows = defaultdict(list)          # item -> list of (base, arm, not)
    for g in gens:
        cb, ca, cn = cells.get((g, base), {}), cells.get((g, arm), {}), cells.get((g, not_arm), {})
        for i in set(cb) & set(ca) & set(cn):
            rows[i].append((cb[i], ca[i], cn[i]))
    items = sorted(rows)
    if len(items) < 20:
        return {"n_items": len(items), "drop": None}
    arr = [np.array(rows[i], dtype=float) for i in items]
    sums = np.array([a.sum(0) for a in arr]); cnts = np.array([len(a) for a in arr], dtype=float)

    def stats(s, c):
        m = s / c
        drop_arm, drop_not = m[1] - m[0], m[2] - m[0]
        return drop_arm, drop_not, (drop_arm / drop_not if drop_not != 0 else np.nan), m[2] - m[1]
    pt = stats(sums.sum(0), cnts.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(items), size=(draws, len(items)))
    bs = np.array([stats(sums[k].sum(0), cnts[k].sum()) for k in idx])
    ci = lambda col: (float(np.nanpercentile(bs[:, col], 2.5)), float(np.nanpercentile(bs[:, col], 97.5)))
    names = ("drop_arm", "drop_not", "share", "gap")
    return {"n_items": len(items), "n_pairs": int(cnts.sum()),
            **{n: {"point": float(pt[k]), "lo": ci(k)[0], "hi": ci(k)[1]} for k, n in enumerate(names)}}


def read_partial(c: dict, *, better: str, adds: str, sufficient: str) -> str:
    if c.get("drop_arm") is None:
        return "UNDER-GATED"
    gap, share = c["gap"], c["share"]["point"]
    if gap["lo"] > 0:
        return better
    if gap["hi"] < 0:
        return adds
    if share >= 0.75:
        return sufficient
    return "UNRESOLVED"


def form_reading(ck: dict, no: dict) -> dict:
    out = {"checklist": read_partial(ck, better="CHECKLIST-BETTER", adds="NARRATIVE-ADDS",
                                     sufficient="CONTENT-SUFFICIENT"),
           "narrative_only": read_partial(no, better="NARRATIVE-ONLY-BETTER", adds="CONTENT-ADDS",
                                          sufficient="NARRATIVE-SUFFICIENT")}
    integ = (ck.get("drop_arm") is not None and no.get("drop_arm") is not None
             and ck["drop_not"]["hi"] < 0 and ck["share"]["point"] < 0.5 and no["share"]["point"] < 0.5)
    out["integration_flag"] = bool(integ)
    return out


def length_reading(per_gen: dict, len_ratio: dict) -> str:
    if any(not (0.75 <= len_ratio.get(g, 0) <= 1.33) for g in per_gen):
        return "LENGTH-MISS"
    if any(c.get("drop_arm") is None for c in per_gen.values()):
        return "UNDER-GATED"
    if all(c["gap"]["hi"] < 0 for c in per_gen.values()):
        return "LENGTH-INERT"
    if any(c["share"]["point"] >= 0.5 and c["drop_arm"]["hi"] < 0 for c in per_gen.values()):
        return "LENGTH-CARRIES"
    return "UNRESOLVED"


def run(judges) -> dict:
    arms = (COT, NOT, CHECK, NARR, LEN)
    rows = load_rows(FORM_GENS, arms)
    q = load_questions("oeq", n=150)
    L = lengths(rows)
    res = {"lengths_chars": {f"{g}|{a}": v for (g, a), v in L.items()}, "judges": {}}
    len_ratio = {g: (L.get((g, LEN), 0) / L[(g, NOT)]) if L.get((g, NOT)) else 0 for g in LEN_GENS}
    res["length_ratio_lengthmatched_over_not"] = len_ratio
    for j in judges:
        cells = scores(rows, j, q)
        jr = {"rates": {f"{g}|{a}": float(np.mean(list(cells[(g, a)].values())))
                        for g in FORM_GENS for a in arms if cells.get((g, a))}}
        ck = contrast(cells, FORM_GENS, COT, CHECK)
        no = contrast(cells, FORM_GENS, COT, NARR)
        jr["form_pooled"] = {"checklist": ck, "narrative_only": no}
        jr["form_per_generator"] = {g: {"checklist": contrast(cells, (g,), COT, CHECK),
                                        "narrative_only": contrast(cells, (g,), COT, NARR)} for g in FORM_GENS}
        jr["form_reading"] = form_reading(ck, no)
        lp = {g: contrast(cells, (g,), COT, LEN) for g in LEN_GENS}
        jr["length_per_generator"] = lp
        jr["length_reading"] = length_reading(lp, len_ratio)
        res["judges"][j] = jr
    res["agreement"] = {
        "form_checklist": sorted({v["form_reading"]["checklist"] for v in res["judges"].values()}),
        "form_narrative_only": sorted({v["form_reading"]["narrative_only"] for v in res["judges"].values()}),
        "length": sorted({v["length_reading"] for v in res["judges"].values()}),
    }
    return res


def _selftest() -> int:
    ok = True

    def check(msg, cond):
        nonlocal ok
        print(("[ok] " if cond else "[FAIL] ") + msg)
        ok = ok and cond

    items = [str(i) for i in range(60)]
    cells = {("g", COT): {i: 1 for i in items}, ("g", NOT): {i: int(k < 20) for k, i in enumerate(items)},
             ("g", CHECK): {i: int(k < 50) for k, i in enumerate(items)},
             ("g", NARR): {i: int(k < 22) for k, i in enumerate(items)}}
    ck = contrast(cells, ("g",), COT, CHECK)
    no = contrast(cells, ("g",), COT, NARR)
    check("checklist far above NoT reads NARRATIVE-ADDS", form_reading(ck, no)["checklist"] == "NARRATIVE-ADDS")
    check("narrative_only near NoT reads NARRATIVE-SUFFICIENT",
          form_reading(ck, no)["narrative_only"] == "NARRATIVE-SUFFICIENT")
    cells[("g", CHECK)] = {i: int(k < 5) for k, i in enumerate(items)}
    check("checklist far below NoT reads CHECKLIST-BETTER",
          form_reading(contrast(cells, ("g",), COT, CHECK), no)["checklist"] == "CHECKLIST-BETTER")
    check("length outside [0.75, 1.33] of NoT reads LENGTH-MISS", length_reading({"g": ck}, {"g": 0.5}) == "LENGTH-MISS")
    lm = {("g", COT): {i: 1 for i in items}, ("g", NOT): {i: int(k < 20) for k, i in enumerate(items)},
          ("g", LEN): {i: 1 for i in items}}
    check("a length arm that does nothing reads LENGTH-INERT",
          length_reading({"g": contrast(lm, ("g",), COT, LEN)}, {"g": 1.0}) == "LENGTH-INERT")
    print("\nselftest: " + ("ALL OK" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default=",".join(DEFAULT_JUDGES))
    ap.add_argument("--json", type=Path, default=OUT / "narrative_form_readout.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    res = run([j for j in a.judges.split(",") if j])
    for j, jr in res["judges"].items():
        f = jr["form_pooled"]
        print(f"\n== judge {j}  form: {jr['form_reading']}  length: {jr['length_reading']}")
        for k, c in f.items():
            if c.get("drop_arm") is None:
                print(f"  {k}: under-gated ({c['n_items']} items)")
                continue
            print(f"  {k:15s} drop {100 * c['drop_arm']['point']:+.1f} [{100 * c['drop_arm']['lo']:+.1f}, "
                  f"{100 * c['drop_arm']['hi']:+.1f}]  NoT drop {100 * c['drop_not']['point']:+.1f}  share "
                  f"{c['share']['point']:.2f} [{c['share']['lo']:.2f}, {c['share']['hi']:.2f}]  NoT-arm "
                  f"{100 * c['gap']['point']:+.1f} [{100 * c['gap']['lo']:+.1f}, {100 * c['gap']['hi']:+.1f}]")
    print(f"\nlength ratio (lengthmatched / NoT): {res['length_ratio_lengthmatched_over_not']}")
    print(f"agreement across judges: {res['agreement']}")
    a.json.write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
