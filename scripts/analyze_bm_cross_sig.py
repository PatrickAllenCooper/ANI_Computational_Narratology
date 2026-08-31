"""
scripts/analyze_bm_cross_sig.py -- estimation for the E2/E3 addendum in
Guidance_Documents/prereg_embodiment_community.md.

E2 (embodiment x narrative-commitment cross):
  H-AMP           DiD = [p(author_nar) - p(rival_nar)] - [p(author) - p(rival)],
                  item-paired: one bootstrap resample of items drives all four
                  arm terms jointly.  Manski bounds compose the worst cases of
                  both differences.
  H-BACKFIRE-UNIF per-role narrative gap p(role_nar) - p(role), one CI per
                  role, read against the Section-0 narrative-standard gap.
  H-PANEL-NAR     majority-of-3 panels under commitment: mixed_nar vs
                  author_nar x3 (does opposed-alignment de-biasing survive?),
                  and mixed_nar vs mixed_plain (how much backfire the panel
                  absorbs).

E3 (powered signal):
  H-SIG-POW       Spearman rho between s_i = p_i(author) - p_i(rival) and
                  e_i = p_i(TRUE | standard_cot) on the 50-item / k=30 grid.
  Reliability     split-half (even vs odd sample indices) Spearman-Brown
                  reliability of s_i and of e_i.  Registered gate: the null
                  is only licensed if rel(s) >= 0.2; below that the verdict
                  is "still underpowered".

All interval estimates are item-clustered bootstrap CIs (10k draws), sharing
the estimator code in scripts/analyze_bm_embodiment.py.

Usage
-----
  python -m scripts.analyze_bm_cross_sig --selftest
  python -m scripts.analyze_bm_cross_sig
  python -m scripts.analyze_bm_cross_sig --json divergence_study_outputs/bm_cross_sig_analysis.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

from scripts.analyze_bm_embodiment import (
    FALSE,
    MISSING,
    TRUE,
    arm_estimate,
    bootstrap_ci,
    item_p_true,
    panel_stat,
    pooled_p,
    spearman,
)
from scripts.run_bm_cross_sig import (
    CROSS_ARMS,
    SIG_ARMS,
    plan_cross,
    plan_sig,
    register_cross_prompts,
)
from scripts.run_bm_embodiment import plan as plan_e1
from scripts.run_stance_factorial import OUT_DIR, load_cached_records

DEFAULT_DRAWS = 10_000
ANALYSIS_PATH = OUT_DIR / "bm_cross_sig_analysis.json"

PLAIN_ARMS = ("embody_author", "embody_neutral", "embody_rival")
S0_NARRATIVE_GAP = 0.223 - 0.157  # Section-0 narrative_cot - standard_cot


def _index_units(units, *, assume_complete: bool,
                 ) -> dict[str, dict[str, dict[int, str]]]:
    """{arm: {item: {sample_idx: verdict}}} with MISSING for uncached units."""
    recs = load_cached_records(units)
    by_key = {(r["scaffold"], str(r["problem_id"]), r["sample_idx"]): r
              for r in recs}
    out: dict[str, dict[str, dict[int, str]]] = defaultdict(
        lambda: defaultdict(dict))
    for u in units:
        r = by_key.get((u.scaffold, str(u.problem_id), u.sample_idx))
        if r is None:
            if assume_complete:
                out[u.scaffold][str(u.problem_id)][u.sample_idx] = MISSING
            continue
        v = r.get("verdict") or MISSING
        out[u.scaffold][str(u.problem_id)][u.sample_idx] = (
            v if v in (TRUE, FALSE) else MISSING)
    return {a: {i: dict(s) for i, s in d.items()} for a, d in out.items()}


def _flatten(indexed: dict[str, dict[int, str]]) -> dict[str, list[str]]:
    return {i: list(s.values()) for i, s in indexed.items()}


def collect_cross(*, assume_complete: bool):
    register_cross_prompts()
    _, cross_units = plan_cross(pilot=False)
    _, plain_units = plan_e1(PLAIN_ARMS, pilot=False)
    data = _index_units(list(cross_units) + list(plain_units),
                        assume_complete=assume_complete)
    return {a: _flatten(data.get(a, {})) for a in (*CROSS_ARMS, *PLAIN_ARMS)}


def collect_sig(*, assume_complete: bool):
    register_cross_prompts()
    _, units = plan_sig()
    return _index_units(units, assume_complete=assume_complete)


# ---------------------------------------------------------------------------
# E2 statistics
# ---------------------------------------------------------------------------

def manski_did(arms: dict[str, dict[str, list[str]]]) -> dict:
    """Compose Manski worst cases across the four DiD terms."""
    def rates(items):
        vs = [v for s in items.values() for v in s]
        n = len(vs)
        if not n:
            return None
        hit = sum(v == TRUE for v in vs)
        mis = sum(v == MISSING for v in vs)
        return hit / n, (hit + mis) / n

    r = {a: rates(arms[a]) for a in ("embody_author_nar", "embody_rival_nar",
                                     "embody_author", "embody_rival")}
    if any(v is None for v in r.values()):
        return {"lo": None, "hi": None}
    lo = ((r["embody_author_nar"][0] - r["embody_rival_nar"][1])
          - (r["embody_author"][1] - r["embody_rival"][0]))
    hi = ((r["embody_author_nar"][1] - r["embody_rival_nar"][0])
          - (r["embody_author"][0] - r["embody_rival"][1]))
    return {"lo": lo, "hi": hi}


def analyze_cross(arms: dict[str, dict[str, list[str]]], *, draws: int,
                  seed: int = 17) -> dict:
    ids = sorted({i for a in CROSS_ARMS for i in arms.get(a, {})})
    res: dict = {
        "n_items": len(ids),
        "arms": {a: arm_estimate(arms.get(a, {}))
                 for a in (*CROSS_ARMS, *PLAIN_ARMS)},
    }
    if not ids:
        return res

    def did_stat(sample_ids):
        terms = [pooled_p(arms[a], sample_ids)
                 for a in ("embody_author_nar", "embody_rival_nar",
                           "embody_author", "embody_rival")]
        if any(t is None for t in terms):
            return None
        return (terms[0] - terms[1]) - (terms[2] - terms[3])

    res["H_AMP_did"] = bootstrap_ci(did_stat, ids, draws=draws, seed=seed)
    res["H_AMP_manski"] = manski_did(arms)

    # Per-role narrative gap, one CI each, against the Section-0 gap.
    res["H_BACKFIRE_UNIF"] = {"s0_narrative_gap": S0_NARRATIVE_GAP}
    for k, role in enumerate(("embody_author", "embody_neutral",
                              "embody_rival")):
        nar = f"{role}_nar"

        def gap_stat(sample_ids, a=nar, b=role):
            pa, pb = pooled_p(arms[a], sample_ids), pooled_p(arms[b], sample_ids)
            return None if pa is None or pb is None else pa - pb

        res["H_BACKFIRE_UNIF"][f"{role}_gap"] = bootstrap_ci(
            gap_stat, ids, draws=draws, seed=seed + 1 + k)

    # Panels under commitment.
    author_n, neutral_n, rival_n = (arms["embody_author_nar"],
                                    arms["embody_neutral_nar"],
                                    arms["embody_rival_nar"])
    author_p, neutral_p, rival_p = (arms["embody_author"],
                                    arms["embody_neutral"],
                                    arms["embody_rival"])
    panels = {
        "mixed_nar": [author_n, neutral_n, rival_n],
        "author_nar_x3": [author_n] * 3,
        "mixed_plain": [author_p, neutral_p, rival_p],
    }
    res["H_PANEL_NAR"] = {
        name: {"p_true": bootstrap_ci(
            lambda s, m=members: panel_stat(m, s, TRUE),
            ids, draws=draws, seed=seed + 5),
            "abstain": panel_stat(members, ids, "NOPANEL")}
        for name, members in panels.items()
    }

    def panel_contrast(sample_ids, a, b):
        pa = panel_stat(panels[a], sample_ids, TRUE)
        pb = panel_stat(panels[b], sample_ids, TRUE)
        return None if pa is None or pb is None else pa - pb

    res["H_PANEL_NAR_mixed_minus_authorx3"] = bootstrap_ci(
        lambda s: panel_contrast(s, "mixed_nar", "author_nar_x3"),
        ids, draws=draws, seed=seed + 6)
    res["H_PANEL_NAR_mixednar_minus_mixedplain"] = bootstrap_ci(
        lambda s: panel_contrast(s, "mixed_nar", "mixed_plain"),
        ids, draws=draws, seed=seed + 7)
    return res


# ---------------------------------------------------------------------------
# E3 statistics
# ---------------------------------------------------------------------------

def split_half_reliability(indexed_a: dict[str, dict[int, str]],
                           indexed_b: Optional[dict[str, dict[int, str]]],
                           ids: Sequence[str]) -> Optional[dict]:
    """Spearman-Brown corrected even/odd split-half reliability.

    With indexed_b given, the quantity is the difference of the two arms'
    per-item rates (the sensitivity s_i); alone, it is the single arm's rate.
    """
    def half(indexed, item, parity):
        vs = [v for idx, v in indexed.get(item, {}).items() if idx % 2 == parity]
        return item_p_true(vs)

    evens, odds = [], []
    for i in ids:
        ae, ao = half(indexed_a, i, 0), half(indexed_a, i, 1)
        if ae is None or ao is None:
            continue
        if indexed_b is not None:
            be, bo = half(indexed_b, i, 0), half(indexed_b, i, 1)
            if be is None or bo is None:
                continue
            evens.append(ae - be)
            odds.append(ao - bo)
        else:
            evens.append(ae)
            odds.append(ao)
    r = spearman(evens, odds)
    if r is None:
        return None
    return {"half_r": r, "spearman_brown": 2 * r / (1 + r) if r > -1 else None,
            "n_items": len(evens)}


def analyze_sig(indexed: dict[str, dict[str, dict[int, str]]], *, draws: int,
                rel_gate: float = 0.2, seed: int = 29) -> dict:
    author = indexed.get("embody_author", {})
    rival = indexed.get("embody_rival", {})
    std = indexed.get("standard_cot", {})
    flat = {a: _flatten(indexed.get(a, {})) for a in SIG_ARMS}
    ids = sorted(set(author) & set(rival) & set(std))
    res: dict = {
        "n_items": len(ids),
        "arms": {a: arm_estimate(flat[a]) for a in SIG_ARMS},
    }
    if not ids:
        return res

    fa, fr, fs = flat["embody_author"], flat["embody_rival"], flat["standard_cot"]

    def rho_stat(sample_ids):
        xs, ys = [], []
        for i in sample_ids:
            sa, sr = item_p_true(fa.get(i, [])), item_p_true(fr.get(i, []))
            e = item_p_true(fs.get(i, []))
            if sa is None or sr is None or e is None:
                continue
            xs.append(sa - sr)
            ys.append(e)
        return spearman(xs, ys)

    res["H_SIG_POW_rho"] = bootstrap_ci(rho_stat, ids, draws=draws, seed=seed)
    res["reliability_sensitivity"] = split_half_reliability(author, rival, ids)
    res["reliability_error"] = split_half_reliability(std, None, ids)

    rel = (res["reliability_sensitivity"] or {}).get("spearman_brown")
    ci = res["H_SIG_POW_rho"]
    if rel is None or rel < rel_gate:
        verdict = "STILL UNDERPOWERED (reliability gate not met); H-SIG stays open"
    elif (ci["lo95"] is not None and ci["lo95"] <= 0 <= ci["hi95"]
          and abs(ci["point"]) < 0.2):
        verdict = ("NULL LICENSED: reliability gate met and rho is flat; "
                   "the signal design is dead on this instrument")
    elif ci["lo95"] is not None and (ci["lo95"] > 0 or ci["hi95"] < 0):
        verdict = "SIGNAL DETECTED: rho CI excludes zero at adequate reliability"
    else:
        verdict = "INCONCLUSIVE: reliability adequate but CI wide"
    res["registered_verdict"] = verdict
    return res


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_ci(d: Optional[dict]) -> str:
    if not d or d.get("point") is None:
        return "n/a"
    lo = f"{d['lo95']:+.3f}" if d.get("lo95") is not None else "?"
    hi = f"{d['hi95']:+.3f}" if d.get("hi95") is not None else "?"
    return f"{d['point']:+.3f} [{lo}, {hi}]"


def print_report(res: dict) -> None:
    c = res.get("cross", {})
    if c.get("n_items"):
        print(f"\n=== E2 cross ({c['n_items']} items) ===")
        print(f"{'arm':<22}{'n':>6}{'p(TRUE)':>10}{'missing':>10}")
        for a, e in c["arms"].items():
            if e["n"]:
                print(f"{a:<22}{e['n']:>6}{e['p']:>10.3f}"
                      f"{(e['missing_rate'] or 0):>10.1%}")
        print(f"\nH-AMP  DiD (commitment amplifies interest): "
              f"{_fmt_ci(c['H_AMP_did'])}")
        mk = c["H_AMP_manski"]
        if mk.get("lo") is not None:
            print(f"       Manski bounds: [{mk['lo']:+.3f}, {mk['hi']:+.3f}]")
        b = c["H_BACKFIRE_UNIF"]
        print(f"H-BACKFIRE-UNIF (S0 narrative gap {b['s0_narrative_gap']:+.3f}):")
        for role in ("embody_author", "embody_neutral", "embody_rival"):
            print(f"  {role:<18} nar-plain gap: {_fmt_ci(b[f'{role}_gap'])}")
        print("H-PANEL-NAR p(panel affirms TRUE), majority-of-3:")
        for name, d in c["H_PANEL_NAR"].items():
            ab = d["abstain"]
            print(f"  {name:<18}{_fmt_ci(d['p_true'])}   abstain="
                  f"{(ab if ab is not None else float('nan')):.3f}")
        print(f"  mixed_nar - author_nar_x3:  "
              f"{_fmt_ci(c['H_PANEL_NAR_mixed_minus_authorx3'])}")
        print(f"  mixed_nar - mixed_plain:    "
              f"{_fmt_ci(c['H_PANEL_NAR_mixednar_minus_mixedplain'])}")

    s = res.get("sig", {})
    if s.get("n_items"):
        print(f"\n=== E3 powered signal ({s['n_items']} items, k=30) ===")
        print(f"{'arm':<22}{'n':>6}{'p(TRUE)':>10}{'missing':>10}")
        for a, e in s["arms"].items():
            if e["n"]:
                print(f"{a:<22}{e['n']:>6}{e['p']:>10.3f}"
                      f"{(e['missing_rate'] or 0):>10.1%}")
        print(f"\nH-SIG-POW rho: {_fmt_ci(s['H_SIG_POW_rho'])}")
        for k, label in (("reliability_sensitivity", "rel(s_i) split-half SB"),
                         ("reliability_error", "rel(e_i) split-half SB")):
            r = s.get(k)
            print(f"{label}: "
                  + (f"{r['spearman_brown']:+.3f} (half r {r['half_r']:+.3f}, "
                     f"n={r['n_items']})" if r else "n/a"))
        print(f"verdict: {s['registered_verdict']}")


# ---------------------------------------------------------------------------
# Selftest on synthetic data
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import random
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    rng = random.Random(3)

    def synth_arm(p, items=60, k=6):
        return {str(i): [TRUE if rng.random() < p else FALSE for _ in range(k)]
                for i in range(items)}

    # Planted interaction: author-rival gap of 0.02 plain, 0.12 under nar.
    arms = {
        "embody_author": synth_arm(0.18), "embody_rival": synth_arm(0.16),
        "embody_neutral": synth_arm(0.17),
        "embody_author_nar": synth_arm(0.30),
        "embody_rival_nar": synth_arm(0.18),
        "embody_neutral_nar": synth_arm(0.24),
    }
    res = analyze_cross(arms, draws=400)
    did = res["H_AMP_did"]
    check("DiD recovers the planted interaction (~+0.10)",
          did["point"] is not None and 0.03 < did["point"] < 0.18)
    check("DiD CI excludes zero on a planted interaction", did["lo95"] > 0)
    mk = res["H_AMP_manski"]
    check("Manski brackets the DiD point", mk["lo"] <= did["point"] <= mk["hi"])
    check("per-role gaps computed for all three roles",
          all(f"{r}_gap" in res["H_BACKFIRE_UNIF"]
              for r in ("embody_author", "embody_neutral", "embody_rival")))
    check("panel contrasts computed",
          res["H_PANEL_NAR_mixed_minus_authorx3"]["point"] is not None)

    # E3: an item-level signal that exists must be found, and reliability must
    # gate a design too noisy to see it.
    def synth_indexed(item_ps, k):
        return {str(i): {idx: (TRUE if rng.random() < p else FALSE)
                         for idx in range(k)}
                for i, p in enumerate(item_ps)}

    n_items = 40
    base = [rng.uniform(0.05, 0.5) for _ in range(n_items)]
    sens = [rng.uniform(0.0, 0.45) for _ in range(n_items)]
    indexed = {
        "embody_author": synth_indexed(
            [min(b + s, 0.95) for b, s in zip(base, sens)], 40),
        "embody_rival": synth_indexed(base, 40),
        "standard_cot": synth_indexed(
            [min(0.05 + s, 0.9) for s in sens], 40),  # error tracks sensitivity
    }
    res_sig = analyze_sig(indexed, draws=400)
    check("powered synthetic signal detected",
          res_sig["H_SIG_POW_rho"]["lo95"] is not None
          and res_sig["H_SIG_POW_rho"]["lo95"] > 0)
    check("verdict says SIGNAL DETECTED",
          res_sig["registered_verdict"].startswith("SIGNAL"))
    rel = res_sig["reliability_sensitivity"]
    check("sensitivity reliability clears the gate on a strong synthetic signal",
          rel is not None and rel["spearman_brown"] > 0.3)

    # k=4 with tiny true spread: the gate must refuse to declare a null.
    tiny = [rng.uniform(0.0, 0.03) for _ in range(n_items)]
    indexed_noisy = {
        "embody_author": synth_indexed(
            [min(b + s, 0.95) for b, s in zip(base, tiny)], 4),
        "embody_rival": synth_indexed(base, 4),
        "standard_cot": synth_indexed(base, 4),
    }
    res_noisy = analyze_sig(indexed_noisy, draws=200)
    check("underpowered design is reported as underpowered, not null",
          "UNDERPOWERED" in res_noisy["registered_verdict"]
          or "INCONCLUSIVE" in res_noisy["registered_verdict"])

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E2/E3 addendum analysis")
    ap.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    ap.add_argument("--complete-case", action="store_true")
    ap.add_argument("--json", type=Path, default=ANALYSIS_PATH)
    ap.add_argument("--skip-cross", action="store_true")
    ap.add_argument("--skip-sig", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return _selftest()

    assume = not a.complete_case
    res: dict = {"assume_complete": assume}
    if not a.skip_cross:
        res["cross"] = analyze_cross(collect_cross(assume_complete=assume),
                                     draws=a.draws)
    if not a.skip_sig:
        res["sig"] = analyze_sig(collect_sig(assume_complete=assume),
                                 draws=a.draws)
    print_report(res)
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
