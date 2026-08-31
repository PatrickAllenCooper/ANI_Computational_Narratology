"""
scripts/analyze_bm_embodiment.py -- estimation for the embodiment-on-BrokenMath
pre-registration (Guidance_Documents/prereg_embodiment_community.md).

Computes, all with item-clustered bootstrap CIs (resample items, 10k draws):

  H-ORD   Delta = p(TRUE | embody_author) - p(TRUE | embody_rival), plus the
          monotone-trend Spearman rho across the four registered arms.
  H-CTRL  Delta against the replicate-noise band from the accidental
          standard_cot duplicate.
  H-PANEL exact majority-of-3 panel affirmation per item from per-arm verdict
          distributions: mixed {author, neutral, rival} vs homogeneous
          {author x3}, {neutral x3}, {rival x3}.
  H-SIG   item-level interest sensitivity s_i = p_i(author) - p_i(rival) as a
          predictor of untreated (standard_cot) item-level false affirmation.
  Manski  bounds on Delta treating every missing/NOVERDICT unit as TRUE or
          FALSE in the least favourable combination.

Also settles the handoff's standing debt: the item-clustered bootstrap CI for
the Section-0 three-arm contrast (narrative_cot - standard_cot) on the same
caches, with the mislabelled one_line_baseline arm reported as what it is,
a second standard_cot replicate.

ESTIMAND NOTE (why --assume-complete exists and defaults ON for full runs)
--------------------------------------------------------------------------
execute_unit never caches empty responses, so a cache-only read silently
becomes a complete-case analysis: the Section-0 standard_cot arm has 410
cache files for 600 executed units, and hits/410 = 0.229 while the reported
0.157 is hits/600 with empties counted as NOVERDICT in the denominator.
Analysis here therefore treats planned-but-uncached units as NOVERDICT when
--assume-complete is set (the run is known to have executed every unit) and
reports both denominators either way.

Usage
-----
  python -m scripts.analyze_bm_embodiment --selftest
  python -m scripts.analyze_bm_embodiment --pilot            # indices 0-1
  python -m scripts.analyze_bm_embodiment --json divergence_study_outputs/bm_embody_analysis.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

from scripts.run_bm_embodiment import (
    EMBODY_BM_ORDER,
    PILOT_SAMPLE_IDX,
    plan,
    register_prompts,
)
from scripts.run_stance_factorial import OUT_DIR, load_cached_records

TRUE, FALSE, MISSING = "TRUE", "FALSE", "MISSING"
REFERENCE_ARMS = ("standard_cot", "narrative_cot", "one_line_baseline")
DEFAULT_DRAWS = 10_000
ANALYSIS_PATH = OUT_DIR / "bm_embody_analysis.json"


# ---------------------------------------------------------------------------
# Data assembly: {arm: {item: [verdict per executed unit]}}
# ---------------------------------------------------------------------------

def collect(arms: Sequence[str], *, pilot: bool, assume_complete: bool,
            ) -> dict[str, dict[str, list[str]]]:
    register_prompts()
    _, units = plan(arms, pilot=pilot)
    recs = load_cached_records(units)
    by_key = {(r["scaffold"], str(r["problem_id"]), r["sample_idx"]): r
              for r in recs}
    out: dict[str, dict[str, list[str]]] = {a: defaultdict(list) for a in arms}
    for u in units:
        r = by_key.get((u.scaffold, str(u.problem_id), u.sample_idx))
        if r is None:
            if assume_complete:
                out[u.scaffold][str(u.problem_id)].append(MISSING)
            continue
        v = r.get("verdict") or MISSING
        out[u.scaffold][str(u.problem_id)].append(v if v in (TRUE, FALSE) else MISSING)
    return {a: dict(d) for a, d in out.items()}


def item_p_true(verdicts: Sequence[str]) -> Optional[float]:
    """P(TRUE) with non-committed verdicts in the denominator (S0 semantics)."""
    if not verdicts:
        return None
    return sum(v == TRUE for v in verdicts) / len(verdicts)


def arm_estimate(items: dict[str, list[str]]) -> dict:
    all_v = [v for vs in items.values() for v in vs]
    n = len(all_v)
    if not n:
        return {"p": None, "n": 0, "missing_rate": None}
    return {
        "p": sum(v == TRUE for v in all_v) / n,
        "n": n,
        "missing_rate": sum(v == MISSING for v in all_v) / n,
    }


# ---------------------------------------------------------------------------
# Item-clustered bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(stat_fn, item_ids: Sequence[str], *, draws: int, seed: int,
                 ) -> dict:
    """Percentile CI of stat_fn(list-of-item-ids) under item resampling."""
    rng = random.Random(seed)
    point = stat_fn(list(item_ids))
    vals = []
    for _ in range(draws):
        sample = [item_ids[rng.randrange(len(item_ids))] for _ in item_ids]
        v = stat_fn(sample)
        if v is not None:
            vals.append(v)
    vals.sort()
    if not vals:
        return {"point": point, "lo95": None, "hi95": None, "draws": 0}
    lo = vals[int(0.025 * len(vals))]
    hi = vals[min(int(0.975 * len(vals)), len(vals) - 1)]
    return {"point": point, "lo95": lo, "hi95": hi, "draws": len(vals)}


def pooled_p(arm_items: dict[str, list[str]], ids: Sequence[str]) -> Optional[float]:
    num = den = 0
    for i in ids:
        vs = arm_items.get(i, [])
        num += sum(v == TRUE for v in vs)
        den += len(vs)
    return num / den if den else None


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman rho with average ranks; no scipy dependency."""
    if len(xs) < 3 or len(xs) != len(ys):
        return None

    def ranks(v: Sequence[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy) ** 0.5


# ---------------------------------------------------------------------------
# Panels: exact majority-of-3 from per-item verdict distributions
# ---------------------------------------------------------------------------

def _dist(verdicts: Sequence[str]) -> Optional[dict[str, float]]:
    if not verdicts:
        return None
    n = len(verdicts)
    return {c: sum(v == c for v in verdicts) / n for c in (TRUE, FALSE, MISSING)}


def panel_probs(dists: Sequence[dict[str, float]]) -> dict[str, float]:
    """Exact P(panel verdict) for three independent members.

    Majority = a verdict held by >=2 committed members; otherwise the panel
    abstains (counted as NOPANEL, the panel-level analogue of NOVERDICT).
    """
    out = {TRUE: 0.0, FALSE: 0.0, "NOPANEL": 0.0}
    for combo in itertools.product((TRUE, FALSE, MISSING), repeat=3):
        p = 1.0
        for d, c in zip(dists, combo):
            p *= d[c]
        if p == 0.0:
            continue
        t, f = combo.count(TRUE), combo.count(FALSE)
        if t >= 2:
            out[TRUE] += p
        elif f >= 2:
            out[FALSE] += p
        else:
            out["NOPANEL"] += p
    return out


def panel_stat(arm_items_by_member: Sequence[dict[str, list[str]]],
               ids: Sequence[str], key: str) -> Optional[float]:
    """Mean panel P(key) across items (uniform item weights)."""
    vals = []
    for i in ids:
        dists = [_dist(m.get(i, [])) for m in arm_items_by_member]
        if any(d is None for d in dists):
            continue
        vals.append(panel_probs(dists)[key])
    return statistics.fmean(vals) if vals else None


# ---------------------------------------------------------------------------
# Manski bounds on Delta(author - rival)
# ---------------------------------------------------------------------------

def manski_delta(a_items: dict[str, list[str]], b_items: dict[str, list[str]],
                 ) -> dict:
    def rates(items):
        all_v = [v for vs in items.values() for v in vs]
        n = len(all_v)
        hit = sum(v == TRUE for v in all_v)
        mis = sum(v != TRUE and v != FALSE for v in all_v)
        return hit, mis, n

    ha, ma, na = rates(a_items)
    hb, mb, nb = rates(b_items)
    if not na or not nb:
        return {"lo": None, "hi": None}
    return {
        "lo": ha / na - (hb + mb) / nb,
        "hi": (ha + ma) / na - hb / nb,
        "missing_a": ma / na, "missing_b": mb / nb,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(*, pilot: bool, assume_complete: bool, draws: int, seed: int = 7,
            ) -> dict:
    embody = collect(list(EMBODY_BM_ORDER), pilot=pilot,
                     assume_complete=assume_complete)
    # Reference arms executed to completion in the Section-0 run, so
    # assume-complete semantics is what reproduces its published numbers.
    ref = collect(list(REFERENCE_ARMS), pilot=pilot, assume_complete=True)

    ids = sorted({i for d in embody.values() for i in d})
    res: dict = {
        "pilot": pilot, "assume_complete": assume_complete,
        "n_items": len(ids), "order": list(EMBODY_BM_ORDER),
        "arms": {a: arm_estimate(embody[a]) for a in EMBODY_BM_ORDER},
        "reference": {a: arm_estimate(ref[a]) for a in REFERENCE_ARMS},
    }
    if not ids:
        return res

    author, rival = embody["embody_author"], embody["embody_rival"]

    # H-ORD primary: Delta(author - rival)
    def delta_stat(sample_ids):
        pa, pr = pooled_p(author, sample_ids), pooled_p(rival, sample_ids)
        return None if pa is None or pr is None else pa - pr

    res["H_ORD_delta"] = bootstrap_ci(delta_stat, ids, draws=draws, seed=seed)

    # H-ORD secondary: monotone trend across registered arm ranks.
    def trend_stat(sample_ids):
        ps = [pooled_p(embody[a], sample_ids) for a in EMBODY_BM_ORDER]
        if any(p is None for p in ps):
            return None
        # Registered order predicts DECREASING p; rho of rank vs p should be
        # negative, reported so that +1 = perfectly as predicted.
        rho = spearman(list(range(len(ps))), ps)
        return None if rho is None else -rho

    res["H_ORD_trend"] = bootstrap_ci(trend_stat, ids, draws=draws, seed=seed + 1)

    # Manski bounds on the primary contrast.
    res["H_ORD_manski"] = manski_delta(author, rival)

    # H-PANEL: mixed vs homogeneous majority-of-3.
    mixed = [author, embody["embody_neutral"], rival]
    panels = {
        "mixed_author_neutral_rival": mixed,
        "author_x3": [author] * 3,
        "neutral_x3": [embody["embody_neutral"]] * 3,
        "rival_x3": [rival] * 3,
    }
    res["H_PANEL"] = {}
    for name, members in panels.items():
        res["H_PANEL"][name] = {
            "p_true": bootstrap_ci(
                lambda s, m=members: panel_stat(m, s, TRUE),
                ids, draws=draws, seed=seed + 2),
            "abstain": panel_stat(members, ids, "NOPANEL"),
        }

    def panel_contrast(sample_ids, a, b):
        pa = panel_stat(panels[a], sample_ids, TRUE)
        pb = panel_stat(panels[b], sample_ids, TRUE)
        return None if pa is None or pb is None else pa - pb

    res["H_PANEL_mixed_minus_author"] = bootstrap_ci(
        lambda s: panel_contrast(s, "mixed_author_neutral_rival", "author_x3"),
        ids, draws=draws, seed=seed + 3)
    res["H_PANEL_mixed_minus_neutral"] = bootstrap_ci(
        lambda s: panel_contrast(s, "mixed_author_neutral_rival", "neutral_x3"),
        ids, draws=draws, seed=seed + 4)

    # H-SIG: interest sensitivity predicts untreated item error.
    std = ref["standard_cot"]

    def sig_stat(sample_ids):
        xs, ys = [], []
        for i in sample_ids:
            sa, sr = item_p_true(author.get(i, [])), item_p_true(rival.get(i, []))
            e = item_p_true(std.get(i, []))
            if sa is None or sr is None or e is None:
                continue
            xs.append(sa - sr)
            ys.append(e)
        return spearman(xs, ys)

    res["H_SIG_rho"] = bootstrap_ci(sig_stat, ids, draws=draws, seed=seed + 5)

    # Handoff debt: Section-0 three-arm contrast with a real clustered CI.
    ref_ids = sorted({i for d in ref.values() for i in d})

    def s0_stat(arm_a, arm_b):
        def f(sample_ids):
            pa, pb = pooled_p(ref[arm_a], sample_ids), pooled_p(ref[arm_b], sample_ids)
            return None if pa is None or pb is None else pa - pb
        return f

    res["S0_narrative_minus_standard"] = bootstrap_ci(
        s0_stat("narrative_cot", "standard_cot"), ref_ids, draws=draws, seed=seed + 6)
    res["S0_replicate_gap_standard_minus_duplicate"] = bootstrap_ci(
        s0_stat("standard_cot", "one_line_baseline"), ref_ids, draws=draws, seed=seed + 7)
    return res


def _fmt_ci(d: dict) -> str:
    if d.get("point") is None:
        return "n/a"
    lo = f"{d['lo95']:+.3f}" if d.get("lo95") is not None else "?"
    hi = f"{d['hi95']:+.3f}" if d.get("hi95") is not None else "?"
    return f"{d['point']:+.3f} [{lo}, {hi}]"


def print_report(res: dict) -> None:
    print(f"\n=== embodiment on BrokenMath "
          f"({'PILOT' if res['pilot'] else 'FULL'}; "
          f"{'missing=NOVERDICT' if res['assume_complete'] else 'complete-case'}; "
          f"{res['n_items']} items) ===")
    print(f"\n{'arm':<22}{'n':>6}{'p(TRUE)':>10}{'missing':>10}")
    for a in res["order"]:
        e = res["arms"][a]
        p = e["p"] if e["p"] is not None else float("nan")
        m = e["missing_rate"] if e["missing_rate"] is not None else float("nan")
        print(f"{a:<22}{e['n']:>6}{p:>10.3f}{m:>10.1%}")
    for a, e in res.get("reference", {}).items():
        if e["n"]:
            label = a + (" (=std replicate)" if a == "one_line_baseline" else "")
            print(f"{label:<22}{e['n']:>6}{e['p']:>10.3f}"
                  f"{(e['missing_rate'] or 0):>10.1%}")
    if "H_ORD_delta" in res:
        print(f"\nH-ORD  Delta(author-rival):   {_fmt_ci(res['H_ORD_delta'])}")
        print(f"H-ORD  trend (+1=as predicted): {_fmt_ci(res['H_ORD_trend'])}")
        mk = res["H_ORD_manski"]
        if mk.get("lo") is not None:
            print(f"       Manski bounds on Delta: [{mk['lo']:+.3f}, {mk['hi']:+.3f}] "
                  f"(missing a={mk['missing_a']:.1%} r={mk['missing_b']:.1%})")
    if "H_PANEL" in res:
        print("\nH-PANEL p(panel affirms TRUE), majority-of-3:")
        for name, d in res["H_PANEL"].items():
            ab = d["abstain"]
            print(f"  {name:<28}{_fmt_ci(d['p_true'])}   abstain="
                  f"{(ab if ab is not None else float('nan')):.3f}")
        print(f"  mixed - author_x3:   {_fmt_ci(res['H_PANEL_mixed_minus_author'])}")
        print(f"  mixed - neutral_x3:  {_fmt_ci(res['H_PANEL_mixed_minus_neutral'])}")
    if "H_SIG_rho" in res:
        print(f"\nH-SIG  rho(sensitivity, untreated error): {_fmt_ci(res['H_SIG_rho'])}")
    for k, label in (("S0_narrative_minus_standard", "S0 narrative-standard"),
                     ("S0_replicate_gap_standard_minus_duplicate",
                      "S0 replicate gap (std-dup)")):
        if k in res:
            print(f"{label:<28}{_fmt_ci(res[k])}")


# ---------------------------------------------------------------------------
# Selftest on synthetic data (no caches touched)
# ---------------------------------------------------------------------------

def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    # panel_probs: degenerate members give the majority verdict surely.
    sure_t = {TRUE: 1.0, FALSE: 0.0, MISSING: 0.0}
    sure_f = {TRUE: 0.0, FALSE: 1.0, MISSING: 0.0}
    pp = panel_probs([sure_t, sure_t, sure_f])
    check("panel 2T+1F -> TRUE surely", abs(pp[TRUE] - 1.0) < 1e-12)
    pp = panel_probs([sure_t, sure_f, {TRUE: 0, FALSE: 0, MISSING: 1.0}])
    check("panel T+F+missing -> abstains surely", abs(pp["NOPANEL"] - 1.0) < 1e-12)
    mix = {TRUE: 0.5, FALSE: 0.5, MISSING: 0.0}
    pp = panel_probs([mix, mix, mix])
    check("symmetric panel splits evenly", abs(pp[TRUE] - pp[FALSE]) < 1e-12)

    check("spearman of identical order is 1",
          abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-12)
    check("spearman of reversed order is -1",
          abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12)

    # Bootstrap recovers a known gap on synthetic two-arm data.
    rng = random.Random(0)
    a_items = {str(i): [TRUE if rng.random() < 0.30 else FALSE for _ in range(6)]
               for i in range(80)}
    b_items = {str(i): [TRUE if rng.random() < 0.10 else FALSE for _ in range(6)]
               for i in range(80)}
    ids = sorted(a_items)

    def stat(sample):
        return pooled_p(a_items, sample) - pooled_p(b_items, sample)

    ci = bootstrap_ci(stat, ids, draws=500, seed=1)
    check("bootstrap point estimate near 0.20",
          ci["point"] is not None and 0.12 < ci["point"] < 0.28)
    check("bootstrap CI excludes zero on a real gap", ci["lo95"] > 0)

    mk = manski_delta(a_items, b_items)
    check("manski bounds bracket the point estimate",
          mk["lo"] <= ci["point"] <= mk["hi"])

    # Manski widens with missingness.
    c_items = {k: [MISSING if i < 2 else v for i, v in enumerate(vs)]
               for k, vs in a_items.items()}
    mk2 = manski_delta(c_items, b_items)
    check("manski widens under missingness",
          (mk2["hi"] - mk2["lo"]) > (mk["hi"] - mk["lo"]))

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Embodiment-on-BrokenMath analysis")
    ap.add_argument("--pilot", action="store_true",
                    help=f"restrict to sample indices {PILOT_SAMPLE_IDX}")
    ap.add_argument("--complete-case", action="store_true",
                    help="cache-only denominators (default treats planned-but-"
                         "uncached units as NOVERDICT)")
    ap.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    ap.add_argument("--json", type=Path, default=ANALYSIS_PATH)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return _selftest()

    res = analyze(pilot=a.pilot, assume_complete=not a.complete_case,
                  draws=a.draws)
    print_report(res)
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
