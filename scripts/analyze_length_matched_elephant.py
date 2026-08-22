"""
scripts/analyze_length_matched_elephant.py -- the length-matched control the
sycophancy paper never ran, computed offline from the existing cache.

WHY THIS EXISTS
---------------
The flagship claim (NoT cuts ELEPHANT validation 26-53 pp on 4/4 models) is
measured across a 2-4x response-length gap, against a documented LLM-judge
length bias, with NO length-matched comparator. `PROMPTS["standard_cot_verbose"]`
exists and was calibrated per generator -- but it appears in no ELEPHANT cell;
it was spent on the ACL paper's depth metrics instead. Meanwhile
`Guidance_Documents/sycophancy_scenario_redesign.md` Section 6 lists
"length-matched sycophancy scoring as standard practice" as a NOVELTY CLAIM of
this program.

This module runs the zero-API-cost half of that control: a length-stratified
reanalysis of the 27,600 already-scored rows in
`divergence_study_outputs/elephant_singleagent_raw.csv`. It does not replace
running the verbose arm -- it tells you whether that run is even necessary, and
it is free.

THE TRUNCATION INTERACTION (why the <4000 cut is the important one)
-------------------------------------------------------------------
`scripts/elephant_scorers.py` passes `advice[:4000]` to every judge call. On the
OEQ cell that truncates 75-93% of NoT responses against 0-59% of CoT, so the two
arms are not scored on the same object (see redesign doc Section 1b). Restricting to
`response_len < 4000` therefore does two jobs at once: it removes the truncation
confound (both arms complete) AND shrinks the length gap. Where that subsample
is large enough, it is the cleanest estimate available without spending a cent.

MISSING DATA
------------
`sycophantic_*` uses -1 as a sentinel for a non-response, and it is NOT
missing-at-random: on gpt-5.4-nano the NoT cell is 34/150 = 22.7% empty vs 5/150
for CoT, and the dropped items are disproportionately the high-validation ones.
Complete-case rates are reported alongside per-arm non-response counts, and
`--bounds` reports the Manski-style worst/best-case bracket. Never read the
point estimate for a cell without its non-response rate.

INFERENCE ON THE COMPLIANCE CONTRAST
------------------------------------
The compliance-stratified contrast below is the programme's most-quoted causal
claim, and until 2026-08-22 it carried no interval anywhere: this file contained
no bootstrap, no McNemar, no binomtest and no scipy call. It now reports an
item-clustered percentile bootstrap on each stratum, a formal interaction test
between the compliant and non-compliant strata, and per-stratum n and median
NoT/CoT length ratios. The ratios are printed because the claim that compliance
stratification is "a better length control than length-matching" is FALSE --
the item-paired median non-compliant ratios run 0.91-1.21, and on gpt-5.4-nano
the non-compliant NoT response is SHORTER than its own CoT. Stdlib only (the
normal tail comes from `math.erfc`); no dependency is added.

WHAT THE COMPLETE-CASE CONTRAST CANNOT SURVIVE
----------------------------------------------
Two defects found by adversarial verification on 2026-08-22, both fixed here and
both printed next to the numbers they qualify, never separately:

(1) NON-RESPONSE IS THE STRATUM. `sections_present` counts section labels in the
    response text and an empty response contains none, so every unscored NoT
    response is filed NON-COMPLIANT by construction: non-response and stratum
    membership are the same event. On oeq/validation 40 narrative_cot rows are
    unscored (34 nano, 5 grok, 1 sonnet, 0 haiku; 39 of them literally empty),
    against a complete-case non-compliant stratum of only 74 pairs out of 113
    potential ones. `compliance_bounds` carries `_bounds`' Manski logic onto the
    paired strata AND onto the interaction. On the TRUNCATION-CORRECTED scores
    the audit actually quotes (compliant -34.3, non-compliant -4.1) the
    worst-case edge drags the non-compliant delta to -34.5 -- onto the compliant
    one -- and the interaction bracket [-42.4, +0.3] contains zero: the
    moderation disappears. On the uncorrected scores the ordering survives, but
    the non-compliant bracket spans 42.5 pp and contains zero either way, so
    "NoT does nothing where the scaffold left no trace" is not identified in
    this cell under any version of the scores.

(2) THE POOLED INTERACTION CONFOUNDS STRATUM WITH GENERATOR. The non-compliant
    stratum is 70% grok and the compliant stratum 19%, and the two do not share
    a CoT length distribution, so they are not exchangeable subsets of the same
    items. Computing the interaction WITHIN each generator and pooling
    inverse-variance gives a half-again larger effect, and Cochran's Q rejects a
    common effect (haiku's interaction is positive while nano's and grok's are
    around -50 pp). No single number is "the" moderation; the output says so.

The bootstrap p is a Monte Carlo estimate. In the far tail it supports an order
of magnitude and not a mantissa -- the pooled interaction returned 6.9e-8,
7.8e-8, 9.5e-8 and 1.2e-7 across draw counts and seeds -- so the default draw
count is 8000 and `_fmt_p` prints bootstrap tails as `~1e-07`. Exact McNemar
p-values are not coarsened; they have no MC error.

USAGE
    python -m scripts.analyze_length_matched_elephant
    python -m scripts.analyze_length_matched_elephant --dataset aita_yta --metric framing
    python -m scripts.analyze_length_matched_elephant --bounds
    python -m scripts.analyze_length_matched_elephant --corrected --bootstrap 5000
    python -m scripts.analyze_length_matched_elephant --selftest
    python -m scripts.analyze_length_matched_elephant --json out.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics as st
from pathlib import Path
from typing import Any, Optional, Sequence

RAW = Path("divergence_study_outputs/elephant_singleagent_raw.csv")
TRUNCATION_LIMIT = 4000  # scripts/elephant_scorers.py: advice[:4000]
METRICS = ("validation", "indirectness", "framing", "moral")


def _rows(path: Path, dataset: str) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"missing {path}; run the ELEPHANT pipeline first")
    with path.open() as fh:
        rows = [r for r in csv.DictReader(fh) if r["dataset"] == dataset]
    if not rows:
        raise SystemExit(f"no rows for dataset={dataset!r} in {path}")
    return rows


def apply_corrected_scores(rows: list[dict[str, str]], dataset: str, metric: str,
                           judge: str = "claude-haiku-4-5") -> tuple[int, int]:
    """Overwrite `sycophantic_{metric}` in place with untruncated re-scores.

    WHY THIS IS NECESSARY FOR THE COMPLIANCE ANALYSIS SPECIFICALLY
    ---------------------------------------------------------------
    Scaffold-compliant NoT responses are much longer than non-compliant ones
    (oeq: 5,824 vs 2,156 chars on haiku), so the compliant stratum is very
    nearly the *truncated* stratum. Compliance and the Section 1b(i) truncation
    defect are therefore confounded by construction, and the raw compliance
    contrast inherits whatever bias truncation introduced. Re-running it on
    corrected scores is not a refinement; it is the difference between a
    confounded and an unconfounded estimate.

    Reads only the warm `elephant_rescore_*` cache -- no API calls. Rows whose
    response is at or under the limit already had complete text and are left
    alone. Returns (n_replaced, n_missing_from_cache).
    """
    from scripts.rescore_elephant_untruncated import (
        PRODUCTION_LIMIT, _key, load_questions,
    )
    questions = load_questions(dataset, n=150)
    replaced = missing = 0
    for r in rows:
        resp = r.get("response") or ""
        if len(resp) <= PRODUCTION_LIMIT:
            continue
        q = questions.get(r["item_id"])
        if q is None:
            missing += 1
            continue
        k = _key(metric, q, resp, judge, None)
        if not k.exists():
            missing += 1
            continue
        try:
            s = int(json.loads(k.read_text()).get("score", -1))
        except Exception:
            missing += 1
            continue
        if s in (0, 1):
            r[f"sycophantic_{metric}"] = str(s)
            replaced += 1
        else:
            missing += 1
    return replaced, missing


def _scored(r: dict[str, str], col: str) -> bool:
    return r.get(col) in ("0", "1")


def _mean(xs: Sequence[float]) -> Optional[float]:
    return st.mean(xs) if xs else None


def _cell(rows, gen, arm, col, *, max_len=None):
    out = []
    for r in rows:
        if r["generator"] != gen or r["arm"] != arm:
            continue
        if max_len is not None and float(r["response_len"]) >= max_len:
            continue
        if _scored(r, col):
            out.append(r)
    return out


def _delta(rows, gen, col, *, max_len=None, min_n=10):
    c = _cell(rows, gen, "standard_cot", col, max_len=max_len)
    n = _cell(rows, gen, "narrative_cot", col, max_len=max_len)
    if len(c) < min_n or len(n) < min_n:
        return {"cot_n": len(c), "not_n": len(n), "delta_pp": None, "thin": True}
    cv = _mean([int(r[col]) for r in c])
    nv = _mean([int(r[col]) for r in n])
    return {
        "cot_n": len(c), "cot_rate": cv,
        "not_n": len(n), "not_rate": nv,
        "delta_pp": 100.0 * (nv - cv), "thin": False,
    }


def _stratified(rows, gen, col, *, bin_width=1000, min_per_bin=5):
    """Length-matched delta: bin the common support, weight bins by cell count."""
    c = _cell(rows, gen, "standard_cot", col)
    n = _cell(rows, gen, "narrative_cot", col)
    if not c or not n:
        return {"bins": [], "matched_delta_pp": None, "reason": "empty arm"}
    lens_c = [float(r["response_len"]) for r in c]
    lens_n = [float(r["response_len"]) for r in n]
    lo, hi = max(min(lens_c), min(lens_n)), min(max(lens_c), max(lens_n))
    if hi <= lo:
        return {"bins": [], "matched_delta_pp": None, "reason": "no common support"}

    bins, wsum, wn, used_c, used_n = [], 0.0, 0, 0, 0
    edge = lo
    while edge < hi:
        cc = [int(r[col]) for r in c if edge <= float(r["response_len"]) < edge + bin_width]
        nn = [int(r[col]) for r in n if edge <= float(r["response_len"]) < edge + bin_width]
        if len(cc) >= min_per_bin and len(nn) >= min_per_bin:
            d = _mean(nn) - _mean(cc)
            w = len(cc) + len(nn)
            wsum += w * d
            wn += w
            used_c += len(cc)
            used_n += len(nn)
            bins.append({
                "lo": edge, "hi": edge + bin_width,
                "cot_n": len(cc), "cot_rate": _mean(cc),
                "not_n": len(nn), "not_rate": _mean(nn),
                "delta_pp": 100.0 * d,
            })
        edge += bin_width
    return {
        "support": [lo, hi], "bins": bins,
        "matched_delta_pp": (100.0 * wsum / wn) if wn else None,
        "coverage": {"cot_used": used_c, "cot_total": len(c),
                     "not_used": used_n, "not_total": len(n)},
        "reason": None if wn else f"no bin reached {min_per_bin} in both arms",
    }


# --------------------------------------------------------------------------
# Scaffold-compliance stratification
# --------------------------------------------------------------------------
# The NoT prompt mandates five labelled sections. Models do not always produce
# them -- grok omits them outright on ~36% of OEQ items. A response that ignored
# the scaffold is NOT the treatment, so pooling it with compliant responses
# dilutes the estimate toward null. Stratifying by compliance and pairing each
# NoT response against its OWN CoT response on the SAME item turns the flagship
# comparison into an as-treated analysis with item identity held fixed.
SECTION_LABELS = ("protagonist", "stakeholder", "consequence", "uncertain", "decision")


def sections_present(text: Optional[str]) -> int:
    """Count how many of the five mandated section labels appear. Crude but
    conservative: it under-counts models that paraphrase the headers, which
    biases the compliant stratum toward genuine compliance."""
    t = (text or "").lower()
    return sum(1 for lab in SECTION_LABELS if lab in t)


def compliance_paired(rows, col, *, full=5, broken=2, min_pairs=5):
    """Item-paired NoT-vs-own-CoT delta, split by whether NoT obeyed the scaffold.

    WHAT THIS ACTUALLY COMPUTES: `cs` and `ns` are appended inside one loop over
    the same matched item, so they are index-aligned lists of equal length and
    `mean(ns) - mean(cs)` is algebraically the mean of the per-item differences.
    The estimator IS item-paired and its point estimates are correct as such --
    but the pairing is discarded at the end of the loop, so no paired standard
    error can be formed from it. `compliance_inference` below keeps the pairs and
    supplies the intervals; it reproduces every number here exactly (selftest).
    """
    cot = {(r["generator"], r["item_id"]): r
           for r in rows if r["arm"] == "standard_cot"}
    out: dict[str, Any] = {"per_generator": {}, "pooled": {}}
    pool: dict[str, tuple[list, list]] = {"compliant": ([], []),
                                          "non_compliant": ([], [])}

    for gen in sorted({r["generator"] for r in rows if r["arm"] == "narrative_cot"}):
        cell = {}
        for name, keep in (("compliant", lambda k: k >= full),
                           ("non_compliant", lambda k: k <= broken)):
            cs, ns = [], []
            for r in rows:
                if r["generator"] != gen or r["arm"] != "narrative_cot":
                    continue
                if not _scored(r, col) or not keep(sections_present(r.get("response"))):
                    continue
                c = cot.get((gen, r["item_id"]))
                if c is None or not _scored(c, col):
                    continue
                cs.append(int(c[col]))
                ns.append(int(r[col]))
            pool[name][0].extend(cs)
            pool[name][1].extend(ns)
            cell[name] = (
                {"pairs": len(ns), "cot_rate": _mean(cs), "not_rate": _mean(ns),
                 "delta_pp": 100.0 * (_mean(ns) - _mean(cs))}
                if len(ns) >= min_pairs else {"pairs": len(ns), "delta_pp": None}
            )
        cell["compliance_rate"] = _mean([
            1.0 if sections_present(r.get("response")) >= full else 0.0
            for r in rows
            if r["generator"] == gen and r["arm"] == "narrative_cot"
            and (r.get("response") or "").strip()
        ])
        out["per_generator"][gen] = cell

    for name, (cs, ns) in pool.items():
        out["pooled"][name] = (
            {"pairs": len(ns), "cot_rate": _mean(cs), "not_rate": _mean(ns),
             "delta_pp": 100.0 * (_mean(ns) - _mean(cs))} if ns else {"pairs": 0}
        )
    return out


# --------------------------------------------------------------------------
# Item-clustered inference on the compliance strata
# --------------------------------------------------------------------------
# WHAT THE ESTIMATOR IS, AND WHAT IT WAS MISSING
# ----------------------------------------------
# `compliance_paired` accumulated two parallel score lists and returned
# mean(NoT) - mean(CoT). Both lists were appended inside ONE loop over the same
# matched item, so they are index-aligned and equal-length, and that quantity is
# algebraically identical to the mean of the per-item differences: the estimator
# was already paired and NO published point estimate changes below (the selftest
# asserts the identity, and asserts the refactor reproduces the old numbers to
# 1e-9). What the parallel lists threw away was the pairing itself, and with it
# any possibility of a paired standard error -- which is why the flagship
# compliance contrast has never carried an interval. `compliance_pairs` now
# keeps the records, and every interval below resamples ITEMS, not responses.
#
# WHY THE CLUSTER IS THE ITEM
# ---------------------------
# The same OEQ item is answered by all four generators, so its (up to four)
# pairs are not independent draws. Resampling items moves them together. It also
# ties the two strata: an item can be compliant on one generator and
# non-compliant on another, which is exactly the covariance an unpaired
# two-proportion approximation to the interaction gets wrong.


def compliance_pairs(rows, col, *, full=5, broken=2, unscored=False):
    """Item-paired (NoT, its OWN CoT) records, split by scaffold compliance.

    One record per (generator, item) scored in BOTH arms, carrying both scores
    and both response lengths. Returns {generator: {stratum: [record]}}. Same
    selection rule as `compliance_paired`; keeping the records rather than two
    score lists is what makes clustered resampling and the length audit possible.

    `unscored=True` ALSO emits the pairs an arm never produced or the judge never
    scored, with `cot`/`not` set to None on the missing side. Only
    `compliance_bounds` asks for those, and it must: they are exactly the pairs
    the complete-case analysis drops, they are overwhelmingly NoT-side
    non-responses, and `sections_present` files every one of them as
    non-compliant by construction. Every other caller gets the complete-case set
    byte-for-byte unchanged (the selftest asserts the two agree on it).
    """
    cot = {(r["generator"], r["item_id"]): r
           for r in rows if r["arm"] == "standard_cot"}
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for gen in sorted({r["generator"] for r in rows if r["arm"] == "narrative_cot"}):
        cell: dict[str, list[dict[str, Any]]] = {"compliant": [], "non_compliant": []}
        for r in rows:
            if r["generator"] != gen or r["arm"] != "narrative_cot":
                continue
            n_ok = _scored(r, col)
            if not n_ok and not unscored:
                continue
            k = sections_present(r.get("response"))
            stratum = ("compliant" if k >= full
                       else "non_compliant" if k <= broken else None)
            if stratum is None:
                continue
            c = cot.get((gen, r["item_id"]))
            if c is None:
                continue
            c_ok = _scored(c, col)
            if not c_ok and not unscored:
                continue
            cell[stratum].append({
                "generator": gen, "item_id": r["item_id"], "stratum": stratum,
                "cot": int(c[col]) if c_ok else None,
                "not": int(r[col]) if n_ok else None,
                "cot_len": float(c["response_len"]),
                "not_len": float(r["response_len"]),
            })
        out[gen] = cell
    return out


def _paired_delta(recs) -> Optional[float]:
    """Mean of the per-item (NoT - CoT) differences, in points."""
    return 100.0 * st.mean([p["not"] - p["cot"] for p in recs]) if recs else None


def _len_audit(recs) -> dict[str, Optional[float]]:
    """Median NoT and CoT lengths for a stratum, and their ratio both ways round.

    `ratio_of_medians` reproduces the presentation of the audit's Section 2b
    table; `median_of_ratios` is the genuinely item-paired version. Both are
    reported because "compliance stratification is a better length control than
    length-matching" is a claim about exactly this number, and it fails wherever
    the non-compliant ratio sits near (or below) 1.0.
    """
    if not recs:
        return {"median_not_len": None, "median_cot_len": None,
                "ratio_of_medians": None, "median_of_ratios": None}
    mn = st.median([p["not_len"] for p in recs])
    mc = st.median([p["cot_len"] for p in recs])
    per = [p["not_len"] / p["cot_len"] for p in recs if p["cot_len"] > 0]
    return {"median_not_len": mn, "median_cot_len": mc,
            "ratio_of_medians": (mn / mc) if mc else None,
            "median_of_ratios": st.median(per) if per else None}


def _mcnemar_exact(recs) -> dict[str, Any]:
    """Exact McNemar on the discordant pairs -- the honest n for a paired binary DV.

    `not_lower` counts items where NoT was scored non-sycophantic and its own CoT
    sycophantic; `cot_lower` the reverse. Concordant pairs carry no information
    about the contrast, so a stratum with 74 pairs but 11 discordant ones has 11.
    Two-sided exact binomial at p=0.5; `math.comb` only, no scipy.
    """
    b = sum(1 for p in recs if p["not"] < p["cot"])
    c = sum(1 for p in recs if p["not"] > p["cot"])
    n = b + c
    if n == 0:
        return {"not_lower": b, "cot_lower": c, "discordant": 0, "p": None}
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2.0 ** n)
    return {"not_lower": b, "cot_lower": c, "discordant": n,
            "p": min(1.0, 2.0 * tail)}


def _two_sided_p(z: float) -> float:
    """Two-sided normal tail. `math.erfc` only -- no scipy, no new dependency."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def _fmt_p(p: Optional[float]) -> str:
    """Print a BOOTSTRAP p at the precision the estimator actually supports.

    The pooled interaction p came back 6.9e-8, 7.8e-8, 9.5e-8 and 1.2e-7 across
    draw counts and seeds: the same number to an order of magnitude and to
    nothing finer, because it is a normal tail on a resampled SD. Below 1e-4 this
    prints the order of magnitude only, so nobody can quote a mantissa that is a
    lucky draw. Exact McNemar p-values are NOT routed through here.
    """
    if p is None:
        return "-"
    if p >= 1e-4:
        return f"{p:.1e}"
    if p <= 0.0:
        return "~0"
    return f"~1e{round(math.log10(p)):+03d}"


def _chi2_sf(x: float, df: int) -> float:
    """Upper tail of chi-square with integer `df`. Stdlib only, exact.

    Two-term recursion Q(x; k+2) = Q(x; k) + (x/2)^(k/2) e^(-x/2) / Gamma(k/2+1),
    seeded with Q(x; 1) = erfc(sqrt(x/2)) and Q(x; 2) = e^(-x/2). `df` here is at
    most three (one fewer than the number of generators), so nothing asymptotic
    is involved and no scipy import is needed for the heterogeneity test.
    """
    if df < 1:
        raise ValueError("df must be >= 1")
    if x <= 0:
        return 1.0
    if df % 2:
        q, term, k = math.erfc(math.sqrt(x / 2.0)), \
            math.sqrt(2.0 * x / math.pi) * math.exp(-x / 2.0), 1
    else:
        q, term, k = math.exp(-x / 2.0), (x / 2.0) * math.exp(-x / 2.0), 2
    while k < df:
        q += term
        term *= (x / 2.0) / (k / 2.0 + 1.0)
        k += 2
    return min(1.0, max(0.0, q))


def _cluster_bootstrap(records, stat, *, b=8000, seed=20260822, alpha=0.05):
    """Percentile CI for `stat`, resampling ITEMS with replacement.

    Records sharing an `item_id` move together. Draws on which `stat` is
    undefined (a stratum fell below `min_pairs`) are counted in `degenerate`
    rather than quietly shrinking the interval.
    """
    by_item: dict[str, list] = {}
    for p in records:
        by_item.setdefault(p["item_id"], []).append(p)
    items = sorted(by_item)
    point = stat(records) if records else None
    rng = random.Random(seed)
    reps: list[float] = []
    degenerate = 0
    for _ in range(b if items else 0):
        draw: list = []
        for _ in range(len(items)):
            draw.extend(by_item[items[rng.randrange(len(items))]])
        v = stat(draw)
        if v is None:
            degenerate += 1
        else:
            reps.append(v)
    out = {"point": point, "lo": None, "hi": None, "se": None,
           "b": b, "n_resamples_ok": len(reps), "degenerate": degenerate,
           "items": len(items)}
    if point is not None and len(reps) >= 2:
        reps.sort()
        last = len(reps) - 1
        out["lo"] = reps[max(0, int(round((alpha / 2.0) * last)))]
        out["hi"] = reps[min(last, int(round((1.0 - alpha / 2.0) * last)))]
        out["se"] = st.stdev(reps)
    return out


def _with_z(ci: dict[str, Any]) -> dict[str, Any]:
    """Attach a bootstrap-SE z and its two-sided p to a CI dict.

    `z_mc_se` is the Monte Carlo error of z itself. The replicate SD has relative
    error ~1/sqrt(2(B-1)) and z is inversely proportional to it, so at B=8000 a
    z of -5.3 carries about +/-0.04 of pure resampling noise -- enough to move a
    far-tail p by a factor of two and not enough to move its exponent. It is
    reported so the printed tail is read as an order of magnitude (see `_fmt_p`).
    """
    pt, se = ci.get("point"), ci.get("se")
    if pt is not None and se:
        ci["z"] = pt / se
        ci["p"] = _two_sided_p(ci["z"])
        n = ci.get("n_resamples_ok") or 0
        ci["z_mc_se"] = abs(ci["z"]) / math.sqrt(2.0 * (n - 1)) if n > 1 else None
    else:
        ci["z"] = ci["p"] = ci["z_mc_se"] = None
    return ci


def _composition(recs) -> dict[str, float]:
    """Share of a stratum's pairs contributed by each generator.

    This is the number that makes the pooled interaction a DIFFERENT estimand
    from the within-generator one: on oeq/validation the non-compliant stratum is
    70% grok and the compliant stratum 19%, so differencing the two pooled deltas
    differences two differently-composed populations as well as two strata.
    """
    n = len(recs)
    if not n:
        return {}
    return {g: sum(1 for p in recs if p["generator"] == g) / n
            for g in sorted({p["generator"] for p in recs})}


def _pool_within(per_generator) -> dict[str, Any]:
    """Inverse-variance pool the WITHIN-generator interactions, then ask Cochran's
    Q whether there is a single moderation to quote at all.

    Holding generator fixed removes the stratum/generator composition confound
    that attenuates the pooled figure. Q then tests the prior question: whether
    the per-generator interactions are estimates of one effect. On
    oeq/validation they are not -- haiku's is positive and the other two are
    around -50 pp -- so the pooled number is a weighted average of effects that
    disagree in sign, and reporting it alone is reporting an artefact of mixing.
    Weights are 1/SE^2 on the item-clustered bootstrap SEs already computed per
    generator; a generator whose stratum fell below `min_pairs` has no
    interaction and is listed under `dropped` rather than silently omitted.
    """
    parts = [(g, c["interaction"]["point"], c["interaction"]["se"])
             for g, c in per_generator.items()
             if c["interaction"].get("point") is not None and c["interaction"].get("se")]
    out: dict[str, Any] = {
        "generators": [{"generator": g, "point": p, "se": s} for g, p, s in parts],
        "dropped": [g for g, c in per_generator.items()
                    if c["interaction"].get("point") is None
                    or not c["interaction"].get("se")],
        "point": None, "se": None, "z": None, "p": None,
        "q": None, "df": None, "q_p": None, "i2": None,
    }
    if not parts:
        return out
    w = [1.0 / (s * s) for _, _, s in parts]
    tot = sum(w)
    pooled = sum(wi * p for wi, (_, p, _) in zip(w, parts)) / tot
    se = math.sqrt(1.0 / tot)
    out.update(point=pooled, se=se, z=pooled / se, p=_two_sided_p(pooled / se))
    if len(parts) > 1:
        q = sum(wi * (p - pooled) ** 2 for wi, (_, p, _) in zip(w, parts))
        df = len(parts) - 1
        out.update(q=q, df=df, q_p=_chi2_sf(q, df),
                   i2=(max(0.0, (q - df) / q) * 100.0 if q > 0 else 0.0))
    return out


def compliance_inference(rows, col, *, full=5, broken=2, min_pairs=5,
                         b=8000, seed=20260822):
    """Item-clustered CIs on each compliance stratum, plus the interaction test.

    The interaction statistic is the difference of the two paired deltas
    (compliant minus non-compliant), bootstrapped on the SAME item resamples as
    the strata themselves, so the shared-item covariance is carried rather than
    assumed away. Stratum membership is an OUTCOME, not a design variable, so
    the per-stratum intervals resample from the union of both strata: the number
    of compliant pairs is itself random and the interval should say so.
    """
    pairs = compliance_pairs(rows, col, full=full, broken=broken)

    def stat_for(stratum):
        def f(recs):
            sub = [p for p in recs if p["stratum"] == stratum]
            return _paired_delta(sub) if len(sub) >= min_pairs else None
        return f

    def interaction(recs):
        a, n = stat_for("compliant")(recs), stat_for("non_compliant")(recs)
        return None if a is None or n is None else a - n

    out: dict[str, Any] = {"b": b, "seed": seed, "min_pairs": min_pairs,
                           "per_generator": {}, "pooled": {}}
    for gen, cell in pairs.items():
        both = cell["compliant"] + cell["non_compliant"]
        g: dict[str, Any] = {}
        for stratum in ("compliant", "non_compliant"):
            g[stratum] = {**_cluster_bootstrap(both, stat_for(stratum), b=b, seed=seed),
                          "pairs": len(cell[stratum]), **_len_audit(cell[stratum]),
                          "mcnemar": _mcnemar_exact(cell[stratum])}
        g["interaction"] = _with_z(_cluster_bootstrap(both, interaction, b=b, seed=seed))
        out["per_generator"][gen] = g

    allrec = [p for cell in pairs.values()
              for stratum in ("compliant", "non_compliant") for p in cell[stratum]]
    for stratum in ("compliant", "non_compliant"):
        sub = [p for p in allrec if p["stratum"] == stratum]
        out["pooled"][stratum] = {
            **_cluster_bootstrap(allrec, stat_for(stratum), b=b, seed=seed),
            "pairs": len(sub), **_len_audit(sub), "mcnemar": _mcnemar_exact(sub),
            "composition": _composition(sub)}
    out["interaction"] = _with_z(_cluster_bootstrap(allrec, interaction, b=b, seed=seed))
    out["interaction_within"] = _pool_within(out["per_generator"])
    return out


def _nonresponse(rows, gen, arm, col):
    sub = [r for r in rows if r["generator"] == gen and r["arm"] == arm]
    missing = sum(1 for r in sub if not _scored(r, col))
    return {"n": len(sub), "missing": missing,
            "rate": (missing / len(sub)) if sub else None}


def _manski_mean(observed: Sequence[float],
                 intervals: Sequence[tuple[float, float]],
                 ) -> tuple[Optional[float], Optional[float]]:
    """Worst/best-case bracket on the mean of a partly unobserved sample.

    `observed` are the values in hand; `intervals` gives, for each unobserved
    value, the (lo, hi) it is known to lie in. The bracket is attained by pushing
    every unobserved value to its own edge, so it is the widest defensible
    statement and assumes nothing about WHY a value went missing. This is the one
    Manski primitive in the module: the arm-level `_bounds` calls it with a
    [0, 1] interval per missing score, and the paired `compliance_bounds` calls
    it with an interval that depends on which side of the pair went missing.
    """
    n = len(observed) + len(intervals)
    if not n:
        return (None, None)
    s = float(sum(observed))
    return ((s + sum(a for a, _ in intervals)) / n,
            (s + sum(b for _, b in intervals)) / n)


def _bounds(rows, gen, col):
    """Manski bracket: impute all-missing as sycophantic, then as non-sycophantic."""
    out = {}
    for arm in ("standard_cot", "narrative_cot"):
        sub = [r for r in rows if r["generator"] == gen and r["arm"] == arm]
        if not sub:
            continue
        obs = [int(r[col]) for r in sub if _scored(r, col)]
        miss = len(sub) - len(obs)
        best, worst = _manski_mean(obs, [(0.0, 1.0)] * miss)
        out[arm] = {
            "complete_case": _mean(obs),
            "worst": worst,   # all missing were sycophantic
            "best": best,     # all missing were not
            "missing": miss, "n": len(sub),
        }
    if len(out) == 2:
        c, n = out["standard_cot"], out["narrative_cot"]
        # widest defensible bracket on the delta
        out["delta_pp_bracket"] = [100.0 * (n["best"] - c["worst"]),
                                   100.0 * (n["worst"] - c["best"])]
    return out


# --------------------------------------------------------------------------
# Manski bracketing of the compliance contrast
# --------------------------------------------------------------------------
# WHY THE COMPLETE-CASE CI ON THE NON-COMPLIANT STRATUM IS NOT AN INTERVAL ON
# THE QUANTITY ANYONE MEANS
# --------------------------------------------------------------------------
# `sections_present` counts section labels in the response text. An empty
# response contains none, so it scores zero sections and lands in the
# non-compliant stratum by construction: non-response and stratum membership are
# the SAME EVENT, not two variables one could adjust for. The stratum carrying
# the entire null is therefore the stratum that lost the most rows, and its
# complete-case interval conditions on the outcome of the very process under
# test. Bracketing is the only honest response short of an explicit missingness
# model, and the bracket has to reach the interaction, because that is the
# statistic the programme quotes.
#
# WHAT IS AND IS NOT BRACKETED HERE
# ---------------------------------
# Bracketed: the unobserved SCORE of every pair whose NoT arm, CoT arm or both
# went unscored, over {0, 1}. Not bracketed: which stratum an empty response
# "really" belongs to. That is unknowable -- there is no text to read a scaffold
# off -- and bracketing it would be wider still. The printed bracket is thus a
# LOWER bound on the true uncertainty, and it already erases the moderation.


def _pair_interval(p) -> tuple[float, float]:
    """(lo, hi) that one item's NoT-minus-CoT difference is known to lie in.

    A scored side pins its own term; an unscored one is only known to lie in
    {0, 1}. Both sides unscored gives the full [-1, +1] -- and it is precisely
    the items whose CoT was ALSO unscored that carry the oeq non-compliant
    bracket across zero, which is why they are not quietly dropped.
    """
    n, c = p["not"], p["cot"]
    n_lo, n_hi = (n, n) if n is not None else (0, 1)
    c_lo, c_hi = (c, c) if c is not None else (0, 1)
    return (float(n_lo - c_hi), float(n_hi - c_lo))


def _stratum_bracket(recs, *, min_pairs=5) -> dict[str, Any]:
    """Manski bracket, in points, on one stratum's paired delta.

    `recs` is the POTENTIAL pair set from `compliance_pairs(..., unscored=True)`.
    Complete pairs contribute their observed difference, incomplete ones their
    interval, and the complete-case point estimate rides along so the gap between
    the two is visible on one line. `*_cot_scored` repeats the exercise on the
    subset whose CoT WAS scored -- the narrower, NoT-side-only bracket -- so the
    contribution of the both-missing pairs can be read off directly.
    """
    obs = [p["not"] - p["cot"] for p in recs
           if p["not"] is not None and p["cot"] is not None]
    inc = [_pair_interval(p) for p in recs
           if p["not"] is None or p["cot"] is None]
    lo, hi = _manski_mean(obs, inc)
    have = len(recs) >= min_pairs
    with_cot = [p for p in recs if p["cot"] is not None]
    lo2, hi2 = _manski_mean(
        [p["not"] - p["cot"] for p in with_cot if p["not"] is not None],
        [_pair_interval(p) for p in with_cot if p["not"] is None])
    return {
        "complete_case_pp": (100.0 * st.mean(obs)) if len(obs) >= min_pairs else None,
        "lo_pp": (100.0 * lo) if (have and lo is not None) else None,
        "hi_pp": (100.0 * hi) if (have and hi is not None) else None,
        "lo_pp_cot_scored": (100.0 * lo2) if (have and lo2 is not None) else None,
        "hi_pp_cot_scored": (100.0 * hi2) if (have and hi2 is not None) else None,
        "n_complete": len(obs), "n_potential": len(recs),
        "missing_not": sum(1 for p in recs
                           if p["not"] is None and p["cot"] is not None),
        "missing_cot": sum(1 for p in recs
                           if p["cot"] is None and p["not"] is not None),
        "missing_both": sum(1 for p in recs
                            if p["cot"] is None and p["not"] is None),
        "empty_not": sum(1 for p in recs
                         if p["not"] is None and p["not_len"] == 0.0),
    }


def _bracket_difference(a, b) -> tuple[Optional[float], Optional[float]]:
    """Bracket on (a - b) from brackets on a and b -- plain interval arithmetic.

    Deliberately the JOINT worst case: nothing ties the two strata's missing
    scores together, so the interaction can be as small as (a_lo - b_hi) even
    though no single imputation is being asserted.
    """
    if None in (a.get("lo_pp"), a.get("hi_pp"), b.get("lo_pp"), b.get("hi_pp")):
        return (None, None)
    return (a["lo_pp"] - b["hi_pp"], a["hi_pp"] - b["lo_pp"])


def compliance_bounds(rows, col, *, full=5, broken=2, min_pairs=5):
    """Manski brackets on each compliance stratum AND on the interaction.

    Mirrors `compliance_inference`'s shape (per_generator / pooled / interaction)
    so the two can be printed side by side, which is the point: the complete-case
    CI must never appear without the bracket beside it. The non-response tallies
    that drive the bracket ride along in each entry.
    """
    pot = compliance_pairs(rows, col, full=full, broken=broken, unscored=True)
    out: dict[str, Any] = {"per_generator": {}, "pooled": {}}
    for gen, cell in pot.items():
        g: dict[str, Any] = {s: _stratum_bracket(cell[s], min_pairs=min_pairs)
                             for s in ("compliant", "non_compliant")}
        lo, hi = _bracket_difference(g["compliant"], g["non_compliant"])
        g["interaction"] = {"lo_pp": lo, "hi_pp": hi}
        out["per_generator"][gen] = g
    for s in ("compliant", "non_compliant"):
        out["pooled"][s] = _stratum_bracket(
            [p for cell in pot.values() for p in cell[s]], min_pairs=min_pairs)
    lo, hi = _bracket_difference(out["pooled"]["compliant"],
                                 out["pooled"]["non_compliant"])
    out["interaction"] = {"lo_pp": lo, "hi_pp": hi}
    return out


def _selftest(b: int = 500) -> int:
    """Offline arithmetic checks on the estimator, the bootstrap and the
    interaction test. No API calls; touches the raw CSV only if it is present."""
    fails = []

    def check(name, cond, detail=""):
        print(f"[{'ok' if cond else 'FAIL'}] {name}"
              + (f"  {detail}" if detail and not cond else ""))
        if not cond:
            fails.append(name)

    def rec(item, stratum, c, n, cl=1000.0, nl=2000.0, gen="g"):
        return {"item_id": item, "generator": gen, "stratum": stratum,
                "cot": c, "not": n, "cot_len": cl, "not_len": nl}

    def row(gen, arm, item, v, ln=1000):
        return {"generator": gen, "arm": arm, "item_id": item,
                "sycophantic_x": str(v), "response_len": str(ln),
                "response": "protagonist stakeholder consequence uncertain decision"}

    pairs = [rec(f"i{i}", "compliant", c, n)
             for i, (c, n) in enumerate([(1, 0), (1, 1), (0, 0), (1, 0), (0, 1)])]

    # The claim this module was accused of: mean(ns) - mean(cs) over separately
    # accumulated lists is NOT item-paired. It is, whenever the lists are
    # appended in lockstep -- which they were. Assert the identity so nobody has
    # to re-derive it, and so a future refactor that breaks the lockstep fails here.
    lockstep = 100.0 * (st.mean([p["not"] for p in pairs])
                        - st.mean([p["cot"] for p in pairs]))
    check("mean(NoT)-mean(CoT) on lockstep lists == mean(NoT-CoT): point estimates unchanged",
          abs(lockstep - _paired_delta(pairs)) < 1e-9,
          f"{lockstep} vs {_paired_delta(pairs)}")

    la = _len_audit(pairs)
    check("length audit medians and ratios",
          la["median_not_len"] == 2000.0 and la["median_cot_len"] == 1000.0
          and abs(la["ratio_of_medians"] - 2.0) < 1e-9
          and abs(la["median_of_ratios"] - 2.0) < 1e-9, str(la))

    ci1 = _cluster_bootstrap(pairs, _paired_delta, b=b, seed=11)
    ci2 = _cluster_bootstrap(pairs, _paired_delta, b=b, seed=11)
    check("bootstrap is seeded and reproducible", ci1 == ci2)
    check("percentile interval brackets the point estimate",
          ci1["lo"] <= ci1["point"] <= ci1["hi"], str(ci1))
    check("degenerate draws are counted, not dropped silently",
          ci1["n_resamples_ok"] + ci1["degenerate"] == b)

    # Clustering is not cosmetic: four generators answering the SAME items carry
    # a quarter of the independent information.
    dup = [dict(p, generator=f"g{k}") for k in range(4) for p in pairs]
    flat = [dict(p, item_id=f"{p['item_id']}_{p['generator']}") for p in dup]
    c_ci = _cluster_bootstrap(dup, _paired_delta, b=b, seed=7)
    f_ci = _cluster_bootstrap(flat, _paired_delta, b=b, seed=7)
    check("item clustering widens the SE vs treating responses as independent",
          c_ci["se"] > 1.5 * f_ci["se"], f"clustered {c_ci['se']:.2f} vs flat {f_ci['se']:.2f}")

    def synth(stratum, n, n_flip):
        return [rec(f"{stratum}{i}", stratum, 1, 0 if i < n_flip else 1)
                for i in range(n)]

    def inter(recs):
        a = _paired_delta([p for p in recs if p["stratum"] == "compliant"])
        n = _paired_delta([p for p in recs if p["stratum"] == "non_compliant"])
        return None if a is None or n is None else a - n

    strong = _with_z(_cluster_bootstrap(synth("compliant", 100, 70)
                                        + synth("non_compliant", 100, 5),
                                        inter, b=b, seed=3))
    check("interaction test detects a planted -65 pp interaction",
          strong["p"] is not None and strong["p"] < 1e-3 and strong["hi"] < 0,
          f"point={strong['point']:.1f} p={strong['p']}")
    null = _with_z(_cluster_bootstrap(synth("compliant", 100, 50)
                                      + synth("non_compliant", 100, 50),
                                      inter, b=b, seed=3))
    check("interaction test does not invent one when the strata match",
          null["p"] is not None and null["p"] > 0.5,
          f"point={null['point']:.1f} p={null['p']}")
    check("two-sided normal tail matches the textbook value",
          abs(_two_sided_p(1.959964) - 0.05) < 1e-6, str(_two_sided_p(1.959964)))
    check("chi-square upper tail matches the textbook 5% points on 1, 2 and 3 df",
          all(abs(_chi2_sf(x, k) - 0.05) < 5e-5
              for x, k in ((3.841459, 1), (5.991465, 2), (7.814728, 3))),
          str([_chi2_sf(x, k) for x, k in ((3.841459, 1), (5.991465, 2), (7.814728, 3))]))

    # ---------------------------------------------------------------- DEFECT 1
    # Manski bracketing of the compliance contrast.
    check("_manski_mean brackets the complete-case mean and collapses to it "
          "when nothing is missing",
          _manski_mean([0.0, 1.0], []) == (0.5, 0.5)
          and _manski_mean([1.0, 1.0], [(0.0, 1.0)]) == (2 / 3, 1.0),
          f"{_manski_mean([0.0, 1.0], [])} {_manski_mean([1.0, 1.0], [(0.0, 1.0)])}")

    # The refactor must not move `_bounds`. Hand-computed: CoT has one scored
    # sycophantic row and one unscored, NoT two scored non-sycophantic rows, so
    # the CoT rate is bracketed [0.5, 1.0] and the delta [-100, -50] pp.
    brows = [row("g", "standard_cot", "1", 1), row("g", "standard_cot", "2", -1),
             row("g", "narrative_cot", "1", 0), row("g", "narrative_cot", "2", 0)]
    bb = _bounds(brows, "g", "sycophantic_x")
    check("_bounds is unchanged by the _manski_mean refactor",
          bb["standard_cot"] == {"complete_case": 1.0, "worst": 1.0, "best": 0.5,
                                 "missing": 1, "n": 2}
          and bb["delta_pp_bracket"] == [-100.0, -50.0], str(bb))

    check("_pair_interval pins a scored side and opens only the unscored one",
          _pair_interval(rec("i", "s", 1, 0)) == (-1.0, -1.0)
          and _pair_interval(rec("i", "s", 1, None)) == (-1.0, 0.0)
          and _pair_interval(rec("i", "s", None, 0)) == (-1.0, 0.0)
          and _pair_interval(rec("i", "s", None, None)) == (-1.0, 1.0))

    # Five complete pairs at -40 pp, plus five pairs whose NoT never came back.
    # Worst case they were all non-sycophantic against sycophantic CoTs (-70 pp);
    # best case sycophantic against the same CoTs (-20 pp).
    base = [rec(f"c{i}", "compliant", 1, 0 if i < 2 else 1) for i in range(5)]
    gone = [rec(f"m{i}", "compliant", 1, None) for i in range(5)]
    sb = _stratum_bracket(base + gone)
    check("stratum bracket contains the complete-case delta and prices the gap",
          abs(sb["complete_case_pp"] + 40.0) < 1e-9
          and abs(sb["lo_pp"] + 70.0) < 1e-9 and abs(sb["hi_pp"] + 20.0) < 1e-9
          and sb["n_complete"] == 5 and sb["n_potential"] == 10
          and sb["missing_not"] == 5 and sb["missing_both"] == 0, str(sb))
    check("with nothing missing the bracket collapses onto the point estimate",
          all(abs(_stratum_bracket(base)[k] + 40.0) < 1e-9
              for k in ("complete_case_pp", "lo_pp", "hi_pp")))
    check("a pair unscored in BOTH arms widens the bracket further than a "
          "NoT-side non-response does",
          _stratum_bracket(base + [rec("b", "compliant", None, None)])["hi_pp"]
          > _stratum_bracket(base + [rec("b", "compliant", 1, None)])["hi_pp"])
    lo, hi = _bracket_difference({"lo_pp": -40.0, "hi_pp": -30.0},
                                 {"lo_pp": -25.0, "hi_pp": 5.0})
    check("interaction bracket is the joint worst case, not the paired one",
          (lo, hi) == (-45.0, -5.0), f"{lo} {hi}")

    # An empty response contains no section labels, so it MUST land in the
    # non-compliant stratum -- that is the confound, asserted rather than assumed.
    check("an empty response scores zero sections and is filed non-compliant",
          sections_present("") == 0 and sections_present(None) == 0)
    urows = [row("g", "standard_cot", "1", 1), row("g", "narrative_cot", "1", 0),
             row("g", "standard_cot", "2", 1), row("g", "narrative_cot", "2", -1)]
    urows[3]["response"] = ""
    scored_only = compliance_pairs(urows, "sycophantic_x")
    withmiss = compliance_pairs(urows, "sycophantic_x", unscored=True)
    check("unscored=True is a strict superset that reproduces the complete-case set",
          scored_only["g"] == {"compliant": [withmiss["g"]["compliant"][0]],
                               "non_compliant": []}
          and len(withmiss["g"]["non_compliant"]) == 1
          and withmiss["g"]["non_compliant"][0]["not"] is None,
          str(withmiss["g"]))

    # ---------------------------------------------------------------- DEFECT 2
    # Within-generator pooling and the heterogeneity test.
    same = {f"g{i}": {"interaction": {"point": -40.0, "se": s}}
            for i, s in enumerate((5.0, 10.0, 15.0))}
    pw = _pool_within(same)
    check("inverse-variance pooling recovers a common effect with Q ~ 0",
          abs(pw["point"] + 40.0) < 1e-9 and pw["df"] == 2 and pw["q"] < 1e-9
          and pw["q_p"] > 0.99 and pw["i2"] == 0.0, str(pw))
    diff = {"a": {"interaction": {"point": 3.6, "se": 16.0}},
            "b": {"interaction": {"point": -51.9, "se": 10.6}},
            "c": {"interaction": {"point": -54.6, "se": 7.4}},
            "d": {"interaction": {"point": None, "se": None}}}
    pd_ = _pool_within(diff)
    check("Cochran Q rejects a common effect when the generators disagree in sign",
          pd_["df"] == 2 and pd_["q"] > 6.0 and pd_["q_p"] < 0.05
          and pd_["dropped"] == ["d"] and pd_["point"] < -40.0,
          f"Q={pd_['q']:.2f} p={pd_['q_p']:.4f} pooled={pd_['point']:.1f}")
    check("composition shares sum to one and name the generators",
          _composition([rec("i1", "s", 1, 0, gen="a"), rec("i2", "s", 1, 0, gen="b"),
                        rec("i3", "s", 1, 0, gen="b")]) == {"a": 1 / 3, "b": 2 / 3})

    check("far-tail bootstrap p is printed as an order of magnitude, so the four "
          "values one seed choice can produce all print identically",
          len({_fmt_p(x) for x in (6.9e-8, 7.8e-8, 9.5e-8, 1.2e-7)}) == 1
          and _fmt_p(6.9e-8) == "~1e-07" and _fmt_p(0.0037) == "3.7e-03",
          str([_fmt_p(x) for x in (6.9e-8, 7.8e-8, 9.5e-8, 1.2e-7)]))

    if RAW.exists():
        rows = _rows(RAW, "oeq")
        old = compliance_paired(rows, "sycophantic_validation")
        new = compliance_inference(rows, "sycophantic_validation", b=20)
        for stratum in ("compliant", "non_compliant"):
            o, n = old["pooled"][stratum], new["pooled"][stratum]
            check(f"refactor reproduces the published pooled {stratum} delta and n",
                  o["pairs"] == n["pairs"]
                  and abs(o["delta_pp"] - n["point"]) < 1e-9,
                  f"{o} vs point={n['point']} pairs={n['pairs']}")
        bd = compliance_bounds(rows, "sycophantic_validation")
        for stratum in ("compliant", "non_compliant"):
            s, o = bd["pooled"][stratum], old["pooled"][stratum]
            check(f"on-disk {stratum} bracket contains its own complete-case delta",
                  s["lo_pp"] - 1e-9 <= o["delta_pp"] <= s["hi_pp"] + 1e-9
                  and s["n_complete"] == o["pairs"]
                  and s["n_potential"] >= s["n_complete"], str(s))
        nc = bd["pooled"]["non_compliant"]
        check("every unscored NoT row on this cell lands in the non-compliant "
              "stratum, i.e. the defect is real and not hypothetical",
              bd["pooled"]["compliant"]["missing_not"] == 1
              and nc["missing_not"] + nc["missing_both"] >= 35
              and nc["empty_not"] == nc["missing_not"] + nc["missing_both"],
              f"compliant {bd['pooled']['compliant']['missing_not']}, "
              f"non-compliant {nc['missing_not']}+{nc['missing_both']} "
              f"empty={nc['empty_not']}")
        check("the on-disk non-compliant bracket is wide enough to matter -- it "
              "spans more than the whole complete-case effect it qualifies",
              nc["hi_pp"] - nc["lo_pp"] > abs(old["pooled"]["non_compliant"]["delta_pp"]),
              f"nc bracket [{nc['lo_pp']:.1f}, {nc['hi_pp']:.1f}] around "
              f"{old['pooled']['non_compliant']['delta_pp']:.1f}")
        inf = compliance_inference(rows, "sycophantic_validation", b=800)
        w = inf["interaction_within"]
        check("the on-disk within-generator interactions are heterogeneous, so no "
              "single pooled number is 'the' moderation",
              w["df"] == 2 and w["q_p"] < 0.05
              and abs(w["point"] - inf["interaction"]["point"]) > 5.0,
              f"Q={w['q']:.1f} p={w['q_p']:.4f} within={w['point']:.1f} "
              f"across={inf['interaction']['point']:.1f} "
              + str([(e["generator"], round(e["point"], 1)) for e in w["generators"]]))

        # The published compliance contrast is quoted on the TRUNCATION-CORRECTED
        # scores (audit S2b: compliant -34.3, non-compliant -4.1), and both defects
        # bite hardest there, so they have to be asserted there. Warm re-score
        # cache only -- no API calls -- and a missing cache skips rather than fails.
        crows = _rows(RAW, "oeq")
        try:
            n_rep, _ = apply_corrected_scores(crows, "oeq", "validation")
        except Exception as exc:               # cache or loader unavailable
            n_rep = 0
            print(f"[skip] re-score cache unusable ({type(exc).__name__}); "
                  "corrected-score checks not run")
        if n_rep:
            cb = compliance_bounds(crows, "sycophantic_validation")
            cp = compliance_paired(crows, "sycophantic_validation")
            cnc, ccp = cb["pooled"]["non_compliant"], cp["pooled"]["compliant"]
            ib = cb["interaction"]
            check("corrected scores: the non-compliant null does NOT survive the "
                  "worst-case imputation -- its lower edge reaches the compliant "
                  "point estimate and the interaction bracket contains zero",
                  cnc["lo_pp"] <= ccp["delta_pp"]
                  and ib["lo_pp"] <= 0.0 <= ib["hi_pp"],
                  f"nc bracket [{cnc['lo_pp']:.1f}, {cnc['hi_pp']:.1f}] vs compliant "
                  f"{ccp['delta_pp']:.1f}; interaction "
                  f"[{ib['lo_pp']:.1f}, {ib['hi_pp']:.1f}]")
            cw = compliance_inference(crows, "sycophantic_validation",
                                      b=800)["interaction_within"]
            check("corrected scores: the within-generator interactions differ in "
                  "SIGN, so the pooled figure averages effects that disagree",
                  cw["df"] == 2 and cw["q_p"] < 0.05
                  and max(e["point"] for e in cw["generators"]) > 0
                  and min(e["point"] for e in cw["generators"]) < -40.0,
                  f"Q={cw['q']:.1f} p={cw['q_p']:.4f} "
                  + str([(e["generator"], round(e["point"], 1))
                         for e in cw["generators"]]))
    else:
        print(f"[skip] {RAW} absent; on-disk consistency checks not run")

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--dataset", default="oeq")
    ap.add_argument("--metric", default="validation", choices=METRICS)
    ap.add_argument("--bin-width", type=int, default=1000)
    ap.add_argument("--min-per-bin", type=int, default=5)
    ap.add_argument("--bounds", action="store_true", help="Manski bracket for non-response")
    ap.add_argument("--corrected", action="store_true",
                    help="use untruncated re-scores where available (warm cache, no API calls)")
    ap.add_argument("--bootstrap", type=int, default=8000,
                    help="item-clustered resamples for the compliance CIs (0 to skip); "
                         "the default is deliberately large -- at 2000 the far-tail "
                         "interaction p moves by a factor of two between seeds")
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--selftest", action="store_true",
                    help="offline checks on the estimator, the bootstrap and the "
                         "interaction test; no API calls")
    ap.add_argument("--json", type=Path, help="write the full result object here")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()

    col = f"sycophantic_{a.metric}"
    rows = _rows(a.raw, a.dataset)
    if a.corrected:
        n_rep, n_miss = apply_corrected_scores(rows, a.dataset, a.metric)
        print(f"[corrected] replaced {n_rep} truncated scores from the re-score cache"
              f"{f'; {n_miss} not cached (left as published)' if n_miss else ''}")
    gens = sorted({r["generator"] for r in rows
                   if any(x["generator"] == r["generator"] and x["arm"] == "narrative_cot"
                          for x in rows)})

    result: dict[str, Any] = {
        "dataset": a.dataset, "metric": a.metric,
        "truncation_limit": TRUNCATION_LIMIT, "generators": {},
    }

    print(f"dataset={a.dataset}  metric={a.metric}  "
          f"(judge truncation limit = {TRUNCATION_LIMIT} chars)")
    print("=" * 88)
    print(f"{'generator':<26}{'full delta':>12}{'<4000 delta':>14}{'len-matched':>14}"
          f"{'NoT missing':>13}")
    for g in gens:
        full = _delta(rows, g, col)
        clean = _delta(rows, g, col, max_len=TRUNCATION_LIMIT)
        strat = _stratified(rows, g, col, bin_width=a.bin_width,
                            min_per_bin=a.min_per_bin)
        nr = _nonresponse(rows, g, "narrative_cot", col)
        cell = {"full": full, "untruncated": clean, "stratified": strat,
                "nonresponse_not": nr,
                "nonresponse_cot": _nonresponse(rows, g, "standard_cot", col)}
        if a.bounds:
            cell["bounds"] = _bounds(rows, g, col)
        result["generators"][g] = cell

        def fmt(x):
            return f"{x:+.1f}" if isinstance(x, (int, float)) else "thin"
        print(f"{g:<26}{fmt(full.get('delta_pp')):>12}"
              f"{fmt(clean.get('delta_pp')):>14}"
              f"{fmt(strat.get('matched_delta_pp')):>14}"
              f"{100 * (nr['rate'] or 0):>12.1f}%")

    print()
    print("per-generator length strata")
    print("=" * 88)
    for g in gens:
        s = result["generators"][g]["stratified"]
        print(f"\n{g}")
        if not s["bins"]:
            print(f"   {s['reason']}")
            continue
        for b in s["bins"]:
            print(f"   {b['lo']:>6.0f}-{b['hi']:<6.0f} "
                  f"CoT n={b['cot_n']:<4} {b['cot_rate']:.3f}   "
                  f"NoT n={b['not_n']:<4} {b['not_rate']:.3f}   "
                  f"d={b['delta_pp']:+6.1f} pp")
        cov = s["coverage"]
        print(f"   => matched delta {s['matched_delta_pp']:+.1f} pp "
              f"(uses {cov['cot_used']}+{cov['not_used']} of "
              f"{cov['cot_total']}+{cov['not_total']})")

    comp = compliance_paired(rows, col)
    result["compliance"] = comp
    print()
    print("scaffold-compliance stratification (item-paired: NoT vs its OWN CoT)")
    print("=" * 88)
    print(f"{'generator':<26}{'complied':>10}{'compliant Δ':>15}{'non-compliant Δ':>18}")
    for g, cell in comp["per_generator"].items():
        cr = cell.get("compliance_rate")
        def _fmt(stratum: str) -> str:
            s = cell[stratum]
            d = s.get("delta_pp")
            return "too few" if d is None else f"{d:+.1f} (n={s['pairs']})"
        pct = 100 * cr if cr is not None else 0.0
        print(f"{g:<26}{pct:>9.0f}%{_fmt('compliant'):>15}{_fmt('non_compliant'):>18}")
    print()
    for label, key in (("compliant", "compliant"), ("non-compliant", "non_compliant")):
        p = comp["pooled"][key]
        if not p.get("pairs"):
            # e.g. aita_yta, where every model emits all five sections: there is
            # no untreated stratum, so the dissociation cannot be tested there.
            print(f"   POOLED {label:<14} n=0     (no such stratum in this cell)")
            continue
        print(f"   POOLED {label:<14} n={p['pairs']:<5} "
              f"CoT={p['cot_rate']:.3f} NoT={p['not_rate']:.3f}  "
              f"Δ={p['delta_pp']:+.1f} pp")

    if a.bootstrap:
        inf = compliance_inference(rows, col, b=a.bootstrap, seed=a.seed)
        bnd = compliance_bounds(rows, col)
        result["compliance_inference"] = inf
        result["compliance_bounds"] = bnd

        def ci(s):
            if s.get("point") is None:
                return f"too few (n={s.get('pairs', 0)})"
            if s.get("lo") is None:
                return f"{s['point']:+.1f} [no CI]"
            return f"{s['point']:+.1f} [{s['lo']:+.1f}, {s['hi']:+.1f}]"

        def lens(s):
            r = s.get("ratio_of_medians")
            return ("-" if r is None
                    else f"{s['median_not_len']:.0f}/{s['median_cot_len']:.0f}={r:.2f}")

        def zp(s):
            return "-" if s.get("p") is None else f"z={s['z']:+.1f} p={_fmt_p(s['p'])}"

        def brk(s):
            lo, hi = s.get("lo_pp"), s.get("hi_pp")
            return "-" if lo is None else f"[{lo:+.1f}, {hi:+.1f}]"

        print()
        print(f"item-clustered bootstrap ({inf['b']} draws, resampling ITEMS not responses)")
        print("   'Manski' is the worst/best-case bracket over the pairs an arm or the")
        print("   judge never produced. It is printed IN this table on purpose: the CI to")
        print("   its left is complete-case and must not be quoted without it.")
        print("=" * 110)
        print(f"{'generator':<21}{'stratum':<17}{'n':>5}{'Δ pp [95% CI]':>26}"
              f"{'Manski [lo, hi]':>22}{'med NoT/CoT len':>19}")
        for g, cell in inf["per_generator"].items():
            gb = bnd["per_generator"][g]
            for stratum in ("compliant", "non_compliant"):
                s = cell[stratum]
                print(f"{g[:20]:<21}{stratum:<17}{s['pairs']:>5}{ci(s):>26}"
                      f"{brk(gb[stratum]):>22}{lens(s):>19}")
            it = cell["interaction"]
            print(f"{'':<21}{'  interaction':<17}{'':>5}{ci(it):>26}"
                  f"{brk(gb['interaction']):>22}{zp(it):>19}")
        print("-" * 110)
        for stratum in ("compliant", "non_compliant"):
            s = inf["pooled"][stratum]
            print(f"{'POOLED':<21}{stratum:<17}{s['pairs']:>5}{ci(s):>26}"
                  f"{brk(bnd['pooled'][stratum]):>22}{lens(s):>19}")
        it = inf["interaction"]
        print(f"{'POOLED':<21}{'  interaction':<17}{'':>5}{ci(it):>26}"
              f"{brk(bnd['interaction']):>22}{zp(it):>19}")
        print("   interaction = compliant Δ minus non-compliant Δ, bootstrapped on the")
        print("   same item resamples (an item can be compliant on one generator and")
        print("   non-compliant on another, so the strata are not independent).")
        if it.get("z_mc_se"):
            print(f"   the bootstrap p is a Monte Carlo estimate: z = {it['z']:+.2f} "
                  f"± {it['z_mc_se']:.2f} from resampling alone, so the tail is good to")
            print("   an order of magnitude and no further. Do not quote its mantissa.")

        # ------------------------------------------------------------------
        # DEFECT 1 -- non-response and stratum membership are the same event
        # ------------------------------------------------------------------
        print()
        print("non-response, and the stratum it is filed into BY CONSTRUCTION")
        print("=" * 110)
        print("   sections_present() counts section labels in the response text, and an")
        print("   empty response contains none. Every non-response therefore scores zero")
        print("   sections and joins the non-compliant stratum -- the stratum that carries")
        print("   the null. This is the quantity the bracket above is pricing.")
        print(f"   {'generator / stratum':<40}{'complete':>10}{'potential':>11}"
              f"{'NoT unscored':>14}{'CoT unscored':>14}{'neither':>9}")
        cells = [(f"{g} {s}", bnd["per_generator"][g][s])
                 for g in bnd["per_generator"] for s in ("compliant", "non_compliant")]
        cells += [(f"POOLED {s}", bnd["pooled"][s])
                  for s in ("compliant", "non_compliant")]
        for lab, s in cells:
            print(f"   {lab:<40}{s['n_complete']:>10}{s['n_potential']:>11}"
                  f"{s['missing_not']:>14}{s['missing_cot']:>14}{s['missing_both']:>9}")

        pc, pn, pi = (bnd["pooled"]["compliant"], bnd["pooled"]["non_compliant"],
                      bnd["interaction"])
        # aita_yta has no non-compliant stratum at all (every model emits all five
        # sections), so the block below is only reachable where there is a null to
        # be robust or not robust about.
        if any(x is None for x in (pn["lo_pp"], pn["complete_case_pp"],
                                   pc["lo_pp"], pc["complete_case_pp"])):
            print()
            print("   no bracket on the contrast: one stratum has fewer than the "
                  "minimum pairs in this cell.")
        else:
            # The verdict is read off the bracket, not asserted: on oeq/validation
            # it fails, on the ss/framing replication it holds. Saying so either way
            # is the whole point of printing the bracket next to the CI.
            crosses = (pi["lo_pp"] is not None and pi["lo_pp"] <= 0 <= pi["hi_pp"])
            reaches = pn["lo_pp"] <= pc["complete_case_pp"]
            print()
            print("   THE NON-COMPLIANT NULL IS NOT ROBUST TO THE WORST-CASE IMPUTATION."
                  if (crosses or reaches) else
                  "   the MODERATION survives the worst-case imputation: the two strata "
                  "stay ordered\n   at every edge. The non-compliant null is a separate "
                  "question, below.")
            print(f"   complete case {pn['complete_case_pp']:+.1f} pp on {pn['n_complete']} "
                  f"pairs; bracket [{pn['lo_pp']:+.1f}, {pn['hi_pp']:+.1f}] pp on "
                  f"{pn['n_potential']} potential pairs.")
            print(f"   At its lower edge the non-compliant Δ is {pn['lo_pp']:+.1f} pp, "
                  f"against a compliant Δ of")
            print(f"   {pc['complete_case_pp']:+.1f} pp (bracket [{pc['lo_pp']:+.1f}, "
                  f"{pc['hi_pp']:+.1f}])"
                  + (" -- the two are no longer distinguishable." if reaches else "."))
            if pi["lo_pp"] is not None:
                print(f"   The interaction bracket is [{pi['lo_pp']:+.1f}, "
                      f"{pi['hi_pp']:+.1f}] pp, which "
                      + ("CONTAINS zero: the moderation vanishes at that edge."
                         if crosses else
                         "excludes zero: the moderation survives it."))
            # Whether the MODERATION survives and whether the non-compliant stratum
            # is really null are two different claims; the second is the one the
            # programme keeps making, and it needs its own bracket read out.
            width, spans0 = pn["hi_pp"] - pn["lo_pp"], pn["lo_pp"] <= 0 <= pn["hi_pp"]
            print(f"   The non-compliant stratum's OWN bracket spans {width:.1f} pp and "
                  + ("contains zero:" if spans0 else "excludes zero:"))
            print("   \"NoT does nothing where the scaffold left no trace\" is "
                  + ("NOT identified in this cell." if spans0
                     else "identified only to that width."))
            gone = pn["missing_not"] + pn["missing_cot"] + pn["missing_both"]
            if gone:
                print(f"   {gone} of those pairs are unscored: {pn['missing_not']} on the NoT "
                      f"side only, {pn['missing_cot']} on the CoT side only,")
                print(f"   {pn['missing_both']} on neither. Imputing only the ones whose CoT "
                      f"WAS scored gives [{pn['lo_pp_cot_scored']:+.1f}, "
                      f"{pn['hi_pp_cot_scored']:+.1f}] pp,")
                print("   so the both-unscored pairs are what carry the rest of the width.")
            if crosses or reaches:
                print("   The moderation is a complete-case result. It is not identified")
                print("   without an assumption about why those responses are missing, and")
                print("   the bracket is itself a LOWER bound: it does not bracket the")
                print("   stratum label, only the score.")

        # ------------------------------------------------------------------
        # DEFECT 2 -- the pooled interaction confounds stratum with generator
        # ------------------------------------------------------------------
        w = inf["interaction_within"]
        print()
        print("within-generator interaction (composition held fixed) and its heterogeneity")
        print("=" * 110)
        cshare = inf["pooled"]["compliant"].get("composition") or {}
        nshare = inf["pooled"]["non_compliant"].get("composition") or {}
        print("   stratum composition -- the pooled contrast differences these two")
        print("   differently-composed populations, not two strata of one population")
        print(f"   {'generator':<32}{'compliant':>12}{'non-compliant':>15}")
        for g in sorted(set(cshare) | set(nshare)):
            print(f"   {g:<32}{100 * cshare.get(g, 0.0):>11.0f}%"
                  f"{100 * nshare.get(g, 0.0):>14.0f}%")
        print(f"   {'(pairs)':<32}{inf['pooled']['compliant']['pairs']:>12}"
              f"{inf['pooled']['non_compliant']['pairs']:>15}")
        mc = inf["pooled"]["compliant"].get("median_cot_len")
        mn = inf["pooled"]["non_compliant"].get("median_cot_len")
        if mc and mn:
            print(f"   median CoT length: compliant {mc:.0f} vs non-compliant {mn:.0f} chars"
                  " -- the strata are")
            print("   not exchangeable subsets of the same items, so pooling across")
            print("   generators differences composition as well as compliance.")
        print()
        print(f"   {'generator':<32}{'interaction pp':>16}{'bootstrap SE':>14}")
        for e in w["generators"]:
            print(f"   {e['generator']:<32}{e['point']:>+16.1f}{e['se']:>14.1f}")
        for g in w["dropped"]:
            print(f"   {g:<32}{'-':>16}{'-':>14}   (a stratum is below min_pairs)")
        if w["point"] is not None:
            print(f"   {'IV-POOLED within generator':<32}{w['point']:>+16.1f}{w['se']:>14.1f}"
                  f"   z={w['z']:+.1f} p={_fmt_p(w['p'])}")
            print(f"   {'pooled ACROSS generators':<32}{inf['interaction']['point']:>+16.1f}"
                  f"{inf['interaction']['se']:>14.1f}   z={inf['interaction']['z']:+.1f}"
                  f" p={_fmt_p(inf['interaction']['p'])}")
        if w["q"] is not None:
            print()
            print(f"   Cochran Q = {w['q']:.1f} on {w['df']} df, p = {w['q_p']:.4f}, "
                  f"I2 = {w['i2']:.0f}%")
            if w["q_p"] < 0.05:
                pts = [e["point"] for e in w["generators"]]
                gap = abs(inf["interaction"]["point"] - w["point"])
                print("   NO SINGLE NUMBER IS 'THE' MODERATION. The per-generator")
                print("   interactions are significantly heterogeneous, so they are not")
                print("   estimates of one effect"
                      + ("." if min(pts) * max(pts) >= 0 else " -- they differ in SIGN."))
                print(f"   Pooling across generators gives "
                      f"{inf['interaction']['point']:+.1f} pp against {w['point']:+.1f} pp")
                print(f"   with composition held fixed, a gap of {gap:.1f} pp"
                      + (" -- that gap is the" if gap >= 5.0 else " -- small, so"))
                print("   stratum/generator confounding." if gap >= 5.0 else
                      "   composition is not the problem here; the disagreement between"
                      " generators is.")
                print("   Report the per-generator contrasts; a single pooled moderation")
                print("   is not a quantity this design estimates.")
            else:
                print("   the per-generator interactions are consistent with one effect.")

        print()
        print("   exact McNemar -- concordant pairs carry no information, so these")
        print("   discordant counts are the real n behind each contrast")
        print(f"   {'generator / stratum':<40}{'NoT lower':>11}{'CoT lower':>11}{'p':>12}")
        cells = [(f"{g} {s}", inf["per_generator"][g][s]["mcnemar"])
                 for g in inf["per_generator"] for s in ("compliant", "non_compliant")]
        cells += [(f"POOLED {s}", inf["pooled"][s]["mcnemar"])
                  for s in ("compliant", "non_compliant")]
        for lab, m in cells:
            pt = "-" if m["p"] is None else f"{m['p']:.1e}"
            print(f"   {lab:<40}{m['not_lower']:>11}{m['cot_lower']:>11}{pt:>12}")

        flat = sorted((g, c["non_compliant"]["ratio_of_medians"])
                      for g, c in inf["per_generator"].items()
                      if c["non_compliant"]["ratio_of_medians"] is not None
                      and c["non_compliant"]["ratio_of_medians"] <= 1.25)
        if flat:
            print()
            print(f"   NOT A LENGTH CONTROL: the non-compliant median NoT/CoT ratio is "
                  f"<= 1.25 on {len(flat)}/{len(inf['per_generator'])} generators")
            for g, r in flat:
                print(f"      {g:<26} {r:.2f}"
                      + ("   (NoT SHORTER than its own CoT)" if r < 1.0 else ""))
            print("   Stratifying on compliance does not equalise length; do not describe")
            print("   it as a better length control than length-matching.")

    if a.bounds:
        print()
        print("non-response bounds (Manski bracket on the delta)")
        print("=" * 88)
        for g in gens:
            b = result["generators"][g].get("bounds", {})
            if "delta_pp_bracket" not in b:
                continue
            lo, hi = b["delta_pp_bracket"]
            print(f"   {g:<26} complete-case "
                  f"{result['generators'][g]['full'].get('delta_pp', float('nan')):+.1f} pp"
                  f"   bracket [{lo:+.1f}, {hi:+.1f}] pp"
                  f"   (NoT missing {b['narrative_cot']['missing']}/"
                  f"{b['narrative_cot']['n']})")

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
