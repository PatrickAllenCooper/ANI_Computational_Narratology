"""
scripts/analyze_bm_notes.py -- three zero-spend RESULTS notes on the BrokenMath
embodiment record and the ELEPHANT manner deltas, serving Addendum 16.15 of
Guidance_Documents/prereg_embodiment_community.md, items 16.15.5, 16.15.6 and
16.15.7.  No model call is made anywhere in this file: every number is read
from a committed JSON artefact, a CSV, or a cache file already on disk, and
the one subprocess it launches (16.15.7) runs with every provider key blanked
in a mode that has no API path.

THE THREE QUESTIONS
-------------------
16.15.5  Narration under differential non-response (finding C1).  The
         Section-0 contrast narrative_cot - standard_cot (+0.067 on the S0
         denominator, where a NOVERDICT unit counts as "not affirmed") is
         measured across a 6.7-point gap in non-response (34.5% vs 27.8%).
         Report the affirmation rate among COMMITTED units for both arms,
         p / (1 - missing), the committed-only contrast, and the Manski bounds
         on the S0 contrast under the registered `manski_delta` (every missing
         unit pushed to TRUE in one arm and FALSE in the other).  Registered
         replay: 0.239 / 0.309; bounds cross zero.

16.15.6  BrokenMath mixed panel: affirmation vs rejection vs abstention
         (finding C4).  H_PANEL in bm_embody_analysis.json records P(TRUE) and
         P(abstain) for the majority-of-3 panels; P(FALSE) is the remaining
         mass, because `panel_probs` is a probability distribution over
         {TRUE, FALSE, NOPANEL} (selftest) and `panel_stat` averages it over
         the same items for every key.  Report all three for the mixed
         (author, neutral, rival) and author-homogeneous panels, the same
         under narration from bm_cross_sig_analysis.json H_PANEL_NAR, and the
         three deltas.  Registered replay: P(FALSE) 0.552 in both; the -0.026
         in affirmation is matched by +0.027 in abstention.

16.15.7  The manner deltas as a RESULTS note (finding C1, house rule).  Run
         scripts/analyze_length_matched_elephant.py --corrected --bounds (the
         untruncated re-score cache; that mode reads only elephant_rescore_*
         files) as a subprocess and record the four per-model oeq/validation
         drops with the artefact path, so that "22 to 41 points" has a prereg
         line.  No new claim.

HOW INTERVALS ARE OBTAINED WITHOUT A CALL
-----------------------------------------
bm_embody_analysis.json carries no interval for the committed-only rates,
P(FALSE) or P(abstain), and no item-level data.  The cache it was computed
from has since been back-filled by the cross run (bm_cross_sig_analysis.json,
five hours later), so a naive re-read of the cache gives different arm
figures (standard_cot 0.172/0.318 today against the JSON's 0.157/0.345).  The
registered state is recovered by treating every cache file whose mtime is
later than the JSON's as absent, which assume-complete semantics turn into
NOVERDICT.  The guard requires that this snapshot reproduce every arm's p, n
and missing rate in the JSON to 1e-9 AND reproduce the registered intervals
it is asked to (S0 contrast, the four H_PANEL p_true intervals, the
mixed - author_x3 contrast) bit-for-bit under the JSON's own seeds, before
any new interval is reported.  New intervals use the same estimator, draws
and seeds as the JSON field they complement (item-clustered percentile
bootstrap, 10,000 draws, seed 7+k from analyze_bm_embodiment and 17+k from
analyze_bm_cross_sig), so they sit on the very same resamples.

The four ELEPHANT drops carry, in addition to the script's own output, an
item-clustered interval from that script's `_cluster_bootstrap` with its
own defaults (8,000 draws, seed 20260822) and the Manski bracket that
--bounds prints, plus a coverage audit of the re-score cache restricted to
the two arms the drops use.

USAGE
    python -m scripts.analyze_bm_notes --selftest
    python -m scripts.analyze_bm_notes
    python -m scripts.analyze_bm_notes --json divergence_study_outputs/bm_notes.json
    python -m scripts.analyze_bm_notes --skip-elephant     # 16.15.5 and 16.15.6 only
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
# Every module below addresses divergence_study_outputs by a RELATIVE path and
# run_stance_factorial mkdirs it at import, so the cwd is pinned first.
os.chdir(REPO_ROOT)

from scripts.analyze_bm_embodiment import (  # noqa: E402
    FALSE,
    MISSING,
    TRUE,
    _dist,
    arm_estimate,
    bootstrap_ci,
    manski_delta,
    panel_probs,
    panel_stat,
    pooled_p,
)

OUT_DIR = REPO_ROOT / "divergence_study_outputs"
EMBODY_JSON = OUT_DIR / "bm_embody_analysis.json"
CROSS_JSON = OUT_DIR / "bm_cross_sig_analysis.json"
NOTES_JSON = OUT_DIR / "bm_notes.json"
ELEPHANT_JSON = OUT_DIR / "length_matched_elephant_oeq_validation_corrected_16_15_7.json"
ELEPHANT_LOG = OUT_DIR / "length_matched_elephant_oeq_validation_corrected_16_15_7.log"
ELEPHANT_PRIOR = OUT_DIR / "length_matched_elephant_oeq_validation_corrected.json"

EMBODY_DRAWS = 10_000     # analyze_bm_embodiment.DEFAULT_DRAWS
EMBODY_SEED = 7           # analyze_bm_embodiment.analyze(seed=7): S0 uses +6, panels +2, contrast +3
CROSS_SEED = 17           # analyze_bm_cross_sig.analyze_cross(seed=17): panels +5, contrast +6
ELEPHANT_BOOTSTRAP = 8000  # analyze_length_matched_elephant default
ELEPHANT_SEED = 20260822   # analyze_length_matched_elephant default
NOPANEL = "NOPANEL"
API_ENV_BLANKED = (
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY", "DEEPSEEK_API_KEY",
    "AZURE_AI_API_KEY", "AZURE_AI_PROJECT_ENDPOINT", "ANTHROPIC_ENDPOINT",
)

# The auditor's motivating estimates, verbatim from the registration.
REPLAY = {
    "16.15.5": {"committed_standard": 0.239, "committed_narrative": 0.309,
                "manski_crosses_zero": True},
    "16.15.6": {"p_false_both": 0.552, "delta_p_true": -0.026, "delta_abstain": 0.027},
    "16.15.7": {"range_phrase": "22 to 41 points", "lo": 22, "hi": 41},
}

GUARD_FAILURES: list[str] = []


def guard(cond: bool, msg: str) -> bool:
    if not cond:
        print(f"GUARD FAILED: {msg}")
        GUARD_FAILURES.append(msg)
    return cond


# ---------------------------------------------------------------------------
# 16.15.5 primitives: counts behind an arm, committed-only rate, Manski
# ---------------------------------------------------------------------------

def arm_counts(arm: dict) -> dict:
    """Integer (hits, missing, n) behind an arm's {p, n, missing_rate}."""
    n = int(arm["n"])
    hits_f, miss_f = arm["p"] * n, arm["missing_rate"] * n
    hits, miss = int(round(hits_f)), int(round(miss_f))
    integral = abs(hits_f - hits) < 1e-6 and abs(miss_f - miss) < 1e-6
    return {"hits": hits, "missing": miss, "n": n, "integral": integral}


def pseudo_items(c: dict) -> dict[str, list[str]]:
    """One pseudo-item carrying an arm's pooled counts.

    `manski_delta` reads only the pooled (hit, missing, n) counts of each arm,
    so calling it on this collapsed structure reproduces its output on the
    real item structure exactly (selftest: collapse invariance; and the
    snapshot check below confirms it on the real caches).
    """
    rest = c["n"] - c["hits"] - c["missing"]
    return {"_all": [TRUE] * c["hits"] + [MISSING] * c["missing"] + [FALSE] * rest}


def committed_only(arm: dict) -> dict:
    """Affirmation among committed units, by the registered formula and by counts."""
    c = arm_counts(arm)
    by_formula = arm["p"] / (1.0 - arm["missing_rate"])
    n_committed = c["n"] - c["missing"]
    by_counts = c["hits"] / n_committed if n_committed else None
    return {
        "p_S0": arm["p"], "n": c["n"], "missing_rate": arm["missing_rate"],
        "hits": c["hits"], "missing": c["missing"], "n_committed": n_committed,
        "committed_only": by_formula, "committed_only_from_counts": by_counts,
        "formulas_agree": by_counts is not None and abs(by_formula - by_counts) < 1e-9,
        "counts_integral": c["integral"],
    }


def committed_p(arm_items: dict[str, list[str]], ids: Sequence[str]) -> Optional[float]:
    """P(TRUE | committed) pooled over the sampled items (bootstrap statistic)."""
    num = den = 0
    for i in ids:
        vs = arm_items.get(i, [])
        num += sum(v == TRUE for v in vs)
        den += sum(v in (TRUE, FALSE) for v in vs)
    return num / den if den else None


# ---------------------------------------------------------------------------
# 16.15.6 primitives: P(FALSE) from the recorded pair, fast panel bootstrap
# ---------------------------------------------------------------------------

def p_false_from_record(rec: dict) -> float:
    """P(FALSE) = 1 - P(TRUE) - P(abstain), the identity `panel_probs` guarantees."""
    return 1.0 - rec["p_true"]["point"] - rec["abstain"]


def per_item_panel(members: Sequence[dict[str, list[str]]], ids: Sequence[str],
                   ) -> dict[str, dict[str, float]]:
    """{item: panel_probs(...)} for the items `panel_stat` would keep."""
    out = {}
    for i in ids:
        dists = [_dist(m.get(i, [])) for m in members]
        if any(d is None for d in dists):
            continue
        out[i] = panel_probs(dists)
    return out


def panel_key_stat(per_item: dict[str, dict[str, float]], key: str,
                   ) -> Callable[[Sequence[str]], Optional[float]]:
    """A bootstrap statistic numerically identical to panel_stat(members, s, key)
    (same values, same order, same fmean) but without recomputing the 27-term
    product on every draw."""
    def f(sample: Sequence[str]) -> Optional[float]:
        vals = [per_item[i][key] for i in sample if i in per_item]
        return statistics.fmean(vals) if vals else None
    return f


def panel_contrast_stat(pa: dict, pb: dict, key: str,
                        ) -> Callable[[Sequence[str]], Optional[float]]:
    fa, fb = panel_key_stat(pa, key), panel_key_stat(pb, key)

    def f(sample):
        a, b = fa(sample), fb(sample)
        return None if a is None or b is None else a - b
    return f


# ---------------------------------------------------------------------------
# Cache snapshot at a JSON artefact's mtime (no call; reads existing files)
# ---------------------------------------------------------------------------

def snapshot_collect(units, cutoff: Optional[float]) -> dict[str, dict[str, list[str]]]:
    """{scaffold: {item: [verdict per planned unit]}} exactly as
    analyze_bm_embodiment.collect assembles it under assume-complete
    semantics, except that a cache file written after `cutoff` (POSIX mtime)
    counts as absent, i.e. NOVERDICT.  `cutoff=None` is today's cache."""
    from scripts.run_stance_factorial import load_cached_records
    recs = load_cached_records(list(units))
    by_key = {(r["scaffold"], str(r["problem_id"]), r["sample_idx"]): r for r in recs}
    out: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    n_after = 0
    for u in units:
        r = by_key.get((u.scaffold, str(u.problem_id), u.sample_idx))
        v = MISSING
        if r is not None:
            if cutoff is not None and os.path.getmtime(u.cache_path) > cutoff:
                n_after += 1
            else:
                v = r.get("verdict") or MISSING
                v = v if v in (TRUE, FALSE) else MISSING
        out[u.scaffold][str(u.problem_id)].append(v)
    res = {a: dict(d) for a, d in out.items()}
    res["_n_after_cutoff"] = n_after  # type: ignore[assignment]
    return res


def embody_units():
    from scripts.analyze_bm_embodiment import REFERENCE_ARMS
    from scripts.run_bm_embodiment import EMBODY_BM_ORDER, plan, register_prompts
    register_prompts()
    _, e_units = plan(list(EMBODY_BM_ORDER), pilot=False)
    _, r_units = plan(list(REFERENCE_ARMS), pilot=False)
    return list(EMBODY_BM_ORDER), list(e_units), list(REFERENCE_ARMS), list(r_units)


def cross_units():
    from scripts.analyze_bm_cross_sig import PLAIN_ARMS
    from scripts.run_bm_cross_sig import CROSS_ARMS, plan_cross, register_cross_prompts
    from scripts.run_bm_embodiment import plan as plan_e1
    register_cross_prompts()
    _, cu = plan_cross(pilot=False)
    _, pu = plan_e1(list(PLAIN_ARMS), pilot=False)
    return list(CROSS_ARMS) + list(PLAIN_ARMS), list(cu) + list(pu)


def arms_reproduced(snapshot: dict, recorded: dict[str, dict], tol: float = 1e-9) -> dict:
    """Does the snapshot give every recorded arm's p, n, missing_rate?"""
    out = {}
    for a, rec in recorded.items():
        est = arm_estimate(snapshot.get(a, {}))
        out[a] = {
            "snapshot": est, "recorded": {k: rec[k] for k in ("p", "n", "missing_rate")},
            "agrees": (est["n"] == rec["n"] and est["p"] is not None
                       and abs(est["p"] - rec["p"]) < tol
                       and abs(est["missing_rate"] - rec["missing_rate"]) < tol),
        }
    out["all_agree"] = all(v["agrees"] for k, v in out.items() if k != "all_agree")
    return out


def ci_equal(a: dict, b: dict, tol: float = 1e-12) -> bool:
    return all(a.get(k) is not None and b.get(k) is not None
               and abs(a[k] - b[k]) < tol for k in ("point", "lo95", "hi95"))


def _ci(d: Optional[dict]) -> dict:
    """Trim a bootstrap_ci result to the fields the report prints."""
    if d is None:
        return {"point": None, "lo95": None, "hi95": None, "draws": 0}
    return {k: d.get(k) for k in ("point", "lo95", "hi95", "draws")}


# ---------------------------------------------------------------------------
# 16.15.5
# ---------------------------------------------------------------------------

def note_16_15_5(embody: dict, *, draws: int) -> dict:
    ref = embody["reference"]
    std, nar = committed_only(ref["standard_cot"]), committed_only(ref["narrative_cot"])
    guard(std["counts_integral"] and nar["counts_integral"],
          "16.15.5: p*n or missing_rate*n is not an integer count")
    guard(std["formulas_agree"] and nar["formulas_agree"],
          "16.15.5: p/(1-missing) disagrees with hits/(n-missing)")

    # Manski bounds on the S0 contrast (narrative - standard), registered
    # manski_delta, called on the counts the JSON records.
    mk = manski_delta(pseudo_items(arm_counts(ref["narrative_cot"])),
                      pseudo_items(arm_counts(ref["standard_cot"])))
    res: dict[str, Any] = {
        "orientation": "narrative_cot - standard_cot (the S0 contrast)",
        "registered_contrast_S0": _ci(embody["S0_narrative_minus_standard"]),
        "arms": {"standard_cot": std, "narrative_cot": nar},
        "committed_only_contrast": {
            "point": nar["committed_only"] - std["committed_only"],
            "n_committed": {"standard_cot": std["n_committed"],
                            "narrative_cot": nar["n_committed"]},
        },
        "manski_S0_contrast": {
            **mk, "crosses_zero": mk["lo"] < 0.0 < mk["hi"],
            "width": mk["hi"] - mk["lo"],
            "note": "every missing unit pushed to TRUE in one arm and FALSE in the other; "
                    "lo = p_nar - (p_std + miss_std), hi = (p_nar + miss_nar) - p_std",
        },
        "replay": {
            "committed_standard": {"replay": REPLAY["16.15.5"]["committed_standard"],
                                   "agrees": round(std["committed_only"], 3)
                                   == REPLAY["16.15.5"]["committed_standard"]},
            "committed_narrative": {"replay": REPLAY["16.15.5"]["committed_narrative"],
                                    "agrees": round(nar["committed_only"], 3)
                                    == REPLAY["16.15.5"]["committed_narrative"]},
            "manski_crosses_zero": {"replay": True,
                                    "agrees": mk["lo"] < 0.0 < mk["hi"]},
        },
    }

    # Snapshot of the registered cache state for item-clustered intervals.
    cutoff = os.path.getmtime(EMBODY_JSON)
    _, _, ref_arms, r_units = embody_units()
    snap = snapshot_collect(r_units, cutoff)
    rep = arms_reproduced(snap, {a: ref[a] for a in ref_arms})
    snap_ok = guard(rep["all_agree"],
                    "16.15.5: cache snapshot at the JSON mtime does not reproduce the "
                    "reference arms; snapshot intervals withheld")
    snap_res: dict[str, Any] = {
        "cutoff_mtime": cutoff, "n_cache_files_after_cutoff": snap["_n_after_cutoff"],
        "arms_reproduced": rep, "usable": snap_ok,
        "estimator": {"kind": "item-clustered percentile bootstrap (analyze_bm_embodiment."
                              "bootstrap_ci)", "draws": draws,
                      "seed": EMBODY_SEED + 6, "seed_note": "the S0 field's own seed, so "
                      "every interval below sits on the registered resamples"},
    }
    if snap_ok:
        ref_ids = sorted({i for a in ref_arms for i in snap[a]})
        s_std, s_nar = snap["standard_cot"], snap["narrative_cot"]
        seed = EMBODY_SEED + 6
        s0 = bootstrap_ci(lambda s: (None if pooled_p(s_nar, s) is None
                                     or pooled_p(s_std, s) is None
                                     else pooled_p(s_nar, s) - pooled_p(s_std, s)),
                          ref_ids, draws=draws, seed=seed)
        snap_res["S0_contrast_reproduced"] = {
            "snapshot": _ci(s0), "recorded": _ci(embody["S0_narrative_minus_standard"]),
            "bit_identical": (ci_equal(s0, embody["S0_narrative_minus_standard"])
                              if draws == EMBODY_DRAWS else None),
        }
        guard(draws != EMBODY_DRAWS or snap_res["S0_contrast_reproduced"]["bit_identical"],
              "16.15.5: snapshot does not reproduce the registered S0 interval")
        snap_res["committed_only"] = {
            "standard_cot": _ci(bootstrap_ci(lambda s: committed_p(s_std, s),
                                             ref_ids, draws=draws, seed=seed)),
            "narrative_cot": _ci(bootstrap_ci(lambda s: committed_p(s_nar, s),
                                              ref_ids, draws=draws, seed=seed)),
            "contrast": _ci(bootstrap_ci(
                lambda s: (None if committed_p(s_nar, s) is None
                           or committed_p(s_std, s) is None
                           else committed_p(s_nar, s) - committed_p(s_std, s)),
                ref_ids, draws=draws, seed=seed)),
        }
        mk_items = manski_delta(s_nar, s_std)
        snap_res["manski_on_item_structure"] = {
            **mk_items,
            "agrees_with_counts_call": all(abs(mk_items[k] - mk[k]) < 1e-12
                                           for k in ("lo", "hi", "missing_a", "missing_b")),
        }
        guard(snap_res["manski_on_item_structure"]["agrees_with_counts_call"],
              "16.15.5: manski_delta on the item structure differs from the counts call")
        for arm in ("standard_cot", "narrative_cot"):
            pt = snap_res["committed_only"][arm]["point"]
            guard(abs(pt - res["arms"][arm]["committed_only"]) < 1e-9,
                  f"16.15.5: snapshot committed-only point for {arm} differs from the JSON")
    res["snapshot"] = snap_res
    return res


# ---------------------------------------------------------------------------
# 16.15.6
# ---------------------------------------------------------------------------

def panel_triplet(rec: dict) -> dict:
    return {
        "p_true": _ci(rec["p_true"]),
        "p_false": {"point": p_false_from_record(rec)},
        "p_abstain": {"point": rec["abstain"]},
        "sums_to_one": abs(rec["p_true"]["point"] + p_false_from_record(rec)
                           + rec["abstain"] - 1.0) < 1e-12,
    }


def panel_block(recorded_panels: dict[str, dict], contrasts: dict[str, tuple[str, str, dict]],
                ) -> dict:
    """P(TRUE)/P(FALSE)/P(abstain) per panel and the three deltas per contrast."""
    panels = {name: panel_triplet(rec) for name, rec in recorded_panels.items()}
    deltas = {}
    for label, (a, b, rec_ci) in contrasts.items():
        pa, pb = panels[a], panels[b]
        deltas[label] = {
            "p_true": _ci(rec_ci),
            "p_false": {"point": pa["p_false"]["point"] - pb["p_false"]["point"]},
            "p_abstain": {"point": pa["p_abstain"]["point"] - pb["p_abstain"]["point"]},
            "delta_point_check": abs(rec_ci["point"] - (pa["p_true"]["point"]
                                                        - pb["p_true"]["point"])) < 1e-9,
        }
    return {"panels": panels, "deltas": deltas}


def panel_intervals(snapshot: dict, members_by_panel: dict[str, list[str]],
                    recorded_panels: dict[str, dict],
                    contrasts: dict[str, tuple[str, str, dict, int]],
                    *, ids: Sequence[str], draws: int, seed_levels: int,
                    label: str) -> dict:
    """Item-clustered intervals for P(FALSE) and P(abstain), with the recorded
    P(TRUE) intervals reproduced bit-for-bit on the same seeds as the check.
    Each contrast carries the seed its recorded field was drawn with."""
    per = {name: per_item_panel([snapshot[m] for m in mem], ids)
           for name, mem in members_by_panel.items()}
    out: dict[str, Any] = {"panels": {}, "deltas": {}, "n_items": len(ids)}
    all_bit = True
    for name, rec in recorded_panels.items():
        pt = bootstrap_ci(panel_key_stat(per[name], TRUE), ids, draws=draws, seed=seed_levels)
        bit = ci_equal(pt, rec["p_true"]) if draws == EMBODY_DRAWS else None
        all_bit &= bool(bit) if bit is not None else True
        # panel_stat itself (the registered function) must agree with the fast path.
        direct = {k: panel_stat([snapshot[m] for m in members_by_panel[name]], ids, k)
                  for k in (TRUE, FALSE, NOPANEL)}
        out["panels"][name] = {
            "p_true_reproduced": {"snapshot": _ci(pt), "bit_identical": bit},
            "p_false": _ci(bootstrap_ci(panel_key_stat(per[name], FALSE), ids,
                                        draws=draws, seed=seed_levels)),
            "p_abstain": _ci(bootstrap_ci(panel_key_stat(per[name], NOPANEL), ids,
                                          draws=draws, seed=seed_levels)),
            "panel_stat_direct": direct,
        }
        guard(abs(direct[FALSE] - out["panels"][name]["p_false"]["point"]) < 1e-12
              and abs(direct[NOPANEL] - out["panels"][name]["p_abstain"]["point"]) < 1e-12,
              f"16.15.6 {label}: fast panel statistic disagrees with panel_stat for {name}")
    for lab, (a, b, rec_ci, seed_contrast) in contrasts.items():
        dt = bootstrap_ci(panel_contrast_stat(per[a], per[b], TRUE), ids,
                          draws=draws, seed=seed_contrast)
        bit = ci_equal(dt, rec_ci) if draws == EMBODY_DRAWS else None
        all_bit &= bool(bit) if bit is not None else True
        out["deltas"][lab] = {
            "seed": seed_contrast,
            "p_true_reproduced": {"snapshot": _ci(dt), "bit_identical": bit},
            "p_false": _ci(bootstrap_ci(panel_contrast_stat(per[a], per[b], FALSE), ids,
                                        draws=draws, seed=seed_contrast)),
            "p_abstain": _ci(bootstrap_ci(panel_contrast_stat(per[a], per[b], NOPANEL), ids,
                                          draws=draws, seed=seed_contrast)),
        }
    out["all_recorded_intervals_bit_identical"] = all_bit if draws == EMBODY_DRAWS else None
    guard(draws != EMBODY_DRAWS or all_bit,
          f"16.15.6 {label}: snapshot does not reproduce the recorded P(TRUE) intervals")
    return out


def note_16_15_6(embody: dict, cross: Optional[dict], *, draws: int) -> dict:
    hp = embody["H_PANEL"]
    plain = panel_block(hp, {
        "mixed_minus_author_x3": ("mixed_author_neutral_rival", "author_x3",
                                  embody["H_PANEL_mixed_minus_author"]),
        "mixed_minus_neutral_x3": ("mixed_author_neutral_rival", "neutral_x3",
                                   embody["H_PANEL_mixed_minus_neutral"]),
    })
    mixed, author = plain["panels"]["mixed_author_neutral_rival"], plain["panels"]["author_x3"]
    d = plain["deltas"]["mixed_minus_author_x3"]
    res: dict[str, Any] = {
        "plain": plain,
        "replay": {
            "p_false_mixed": {"replay": REPLAY["16.15.6"]["p_false_both"],
                              "agrees": round(mixed["p_false"]["point"], 3)
                              == REPLAY["16.15.6"]["p_false_both"]},
            "p_false_author_x3": {"replay": REPLAY["16.15.6"]["p_false_both"],
                                  "agrees": round(author["p_false"]["point"], 3)
                                  == REPLAY["16.15.6"]["p_false_both"]},
            "delta_p_true": {"replay": REPLAY["16.15.6"]["delta_p_true"],
                             "agrees": round(d["p_true"]["point"], 3)
                             == REPLAY["16.15.6"]["delta_p_true"]},
            "delta_abstain": {"replay": REPLAY["16.15.6"]["delta_abstain"],
                              "agrees": round(d["p_abstain"]["point"], 3)
                              == REPLAY["16.15.6"]["delta_abstain"]},
        },
    }

    # Narration, if the cross artefact carries the fields.
    hn = (cross or {}).get("cross", {}).get("H_PANEL_NAR")
    if hn and "mixed_nar" in hn and "author_nar_x3" in hn:
        contrasts = {"mixed_nar_minus_author_nar_x3": (
            "mixed_nar", "author_nar_x3", cross["cross"]["H_PANEL_NAR_mixed_minus_authorx3"])}
        if "mixed_plain" in hn and "H_PANEL_NAR_mixednar_minus_mixedplain" in cross["cross"]:
            contrasts["mixed_nar_minus_mixed_plain"] = (
                "mixed_nar", "mixed_plain", cross["cross"]["H_PANEL_NAR_mixednar_minus_mixedplain"])
        res["narration"] = panel_block(hn, contrasts)
        res["narration"]["source"] = str(CROSS_JSON.relative_to(REPO_ROOT))
    else:
        res["narration"] = {"not_computable": "bm_cross_sig_analysis.json lacks H_PANEL_NAR"}

    # Snapshot intervals, plain panels (registered cache state at the JSON mtime).
    cutoff = os.path.getmtime(EMBODY_JSON)
    e_arms, e_units, _, _ = embody_units()
    snap = snapshot_collect(e_units, cutoff)
    rep = arms_reproduced(snap, {a: embody["arms"][a] for a in e_arms})
    ok = guard(rep["all_agree"], "16.15.6: cache snapshot at the JSON mtime does not "
                                 "reproduce the embody arms; snapshot intervals withheld")
    ids = sorted({i for a in e_arms for i in snap[a]})
    res["snapshot_plain"] = {
        "cutoff_mtime": cutoff, "n_cache_files_after_cutoff": snap["_n_after_cutoff"],
        "arms_reproduced": rep, "usable": ok, "n_items": len(ids),
        "estimator": {"kind": "item-clustered percentile bootstrap (analyze_bm_embodiment."
                              "bootstrap_ci)", "draws": draws,
                      "seed_levels": EMBODY_SEED + 2,
                      "seed_contrast": {"mixed_minus_author_x3": EMBODY_SEED + 3,
                                        "mixed_minus_neutral_x3": EMBODY_SEED + 4},
                      "seed_note": "the H_PANEL fields' own seeds"},
    }
    if ok:
        members = {
            "mixed_author_neutral_rival": ["embody_author", "embody_neutral", "embody_rival"],
            "author_x3": ["embody_author"] * 3,
            "neutral_x3": ["embody_neutral"] * 3,
            "rival_x3": ["embody_rival"] * 3,
        }
        res["snapshot_plain"].update(panel_intervals(
            snap, members, hp, {
                "mixed_minus_author_x3": ("mixed_author_neutral_rival", "author_x3",
                                          embody["H_PANEL_mixed_minus_author"], EMBODY_SEED + 3),
                "mixed_minus_neutral_x3": ("mixed_author_neutral_rival", "neutral_x3",
                                           embody["H_PANEL_mixed_minus_neutral"], EMBODY_SEED + 4),
            }, ids=ids, draws=draws, seed_levels=EMBODY_SEED + 2, label="plain"))

    # Snapshot intervals, narration panels (cache state at the cross JSON mtime).
    if hn and "mixed_nar" in hn:
        cutoff_c = os.path.getmtime(CROSS_JSON)
        c_arms, c_units = cross_units()
        snap_c = snapshot_collect(c_units, cutoff_c)
        rep_c = arms_reproduced(snap_c, {a: cross["cross"]["arms"][a] for a in c_arms
                                         if a in cross["cross"]["arms"]})
        ok_c = guard(rep_c["all_agree"], "16.15.6: cache snapshot at the cross JSON mtime "
                                         "does not reproduce the cross arms; withheld")
        ids_c = sorted({i for a in c_arms for i in snap_c[a]})
        res["snapshot_narration"] = {
            "cutoff_mtime": cutoff_c, "n_cache_files_after_cutoff": snap_c["_n_after_cutoff"],
            "arms_reproduced": rep_c, "usable": ok_c, "n_items": len(ids_c),
            "estimator": {"kind": "item-clustered percentile bootstrap (analyze_bm_embodiment."
                                  "bootstrap_ci, as analyze_bm_cross_sig calls it)",
                          "draws": draws, "seed_levels": CROSS_SEED + 5,
                          "seed_contrast": {"mixed_nar_minus_author_nar_x3": CROSS_SEED + 6,
                                            "mixed_nar_minus_mixed_plain": CROSS_SEED + 7},
                          "seed_note": "the H_PANEL_NAR fields' own seeds"},
        }
        if ok_c:
            members_c = {
                "mixed_nar": ["embody_author_nar", "embody_neutral_nar", "embody_rival_nar"],
                "author_nar_x3": ["embody_author_nar"] * 3,
                "mixed_plain": ["embody_author", "embody_neutral", "embody_rival"],
            }
            rec_panels = {k: hn[k] for k in members_c if k in hn}
            res["snapshot_narration"].update(panel_intervals(
                snap_c, members_c, rec_panels, {
                    "mixed_nar_minus_author_nar_x3": (
                        "mixed_nar", "author_nar_x3",
                        cross["cross"]["H_PANEL_NAR_mixed_minus_authorx3"], CROSS_SEED + 6),
                }, ids=ids_c, draws=draws, seed_levels=CROSS_SEED + 5, label="narration"))
            if "mixed_plain" in rec_panels:
                per_a = per_item_panel([snap_c[m] for m in members_c["mixed_nar"]], ids_c)
                per_b = per_item_panel([snap_c[m] for m in members_c["mixed_plain"]], ids_c)
                sub = {}
                for key, slot in ((TRUE, "p_true"), (FALSE, "p_false"), (NOPANEL, "p_abstain")):
                    sub[slot] = _ci(bootstrap_ci(panel_contrast_stat(per_a, per_b, key), ids_c,
                                                 draws=draws, seed=CROSS_SEED + 7))
                rec_ci = cross["cross"]["H_PANEL_NAR_mixednar_minus_mixedplain"]
                sub["seed"] = CROSS_SEED + 7
                sub["p_true_reproduced"] = {
                    "snapshot": sub.pop("p_true"),
                    "bit_identical": None}
                sub["p_true_reproduced"]["bit_identical"] = (
                    ci_equal(sub["p_true_reproduced"]["snapshot"], rec_ci)
                    if draws == EMBODY_DRAWS else None)
                guard(draws != EMBODY_DRAWS or bool(sub["p_true_reproduced"]["bit_identical"]),
                      "16.15.6 narration: snapshot does not reproduce the recorded "
                      "mixed_nar - mixed_plain interval")
                res["snapshot_narration"]["deltas"]["mixed_nar_minus_mixed_plain"] = sub
    return res


# ---------------------------------------------------------------------------
# 16.15.7
# ---------------------------------------------------------------------------

def parse_corrected_line(text: str) -> Optional[dict]:
    """'[corrected] replaced N truncated scores from the re-score cache; M not cached ...'"""
    for line in text.splitlines():
        if line.startswith("[corrected] replaced "):
            toks = line.split()
            try:
                replaced = int(toks[2])
            except (IndexError, ValueError):
                return None
            not_cached = 0
            if "not cached" in line:
                try:
                    not_cached = int(line.split(";")[1].split()[0])
                except (IndexError, ValueError):
                    return None
            return {"replaced": replaced, "not_cached": not_cached, "line": line}
    return None


def rescore_coverage(rows, dataset: str, metric: str, judge: str = "claude-haiku-4-5") -> dict:
    """Coverage of the untruncated re-score cache for the two arms the drops use.

    A truncated row that is uncached but was never scored in the published
    data (-1) cannot move a complete-case rate; a truncated row that IS
    published-scored and uncached would leave the drop partly uncorrected,
    and the guard fires on it."""
    from scripts.analyze_length_matched_elephant import _scored
    from scripts.rescore_elephant_untruncated import PRODUCTION_LIMIT, _key, load_questions
    questions = load_questions(dataset, n=150)
    col = f"sycophantic_{metric}"
    out: dict[str, dict] = {}
    for r in rows:
        if r["arm"] not in ("standard_cot", "narrative_cot"):
            continue
        g = out.setdefault(r["generator"], defaultdict(int))
        g["rows"] += 1
        resp = r.get("response") or ""
        if len(resp) <= PRODUCTION_LIMIT:
            g["untruncated"] += 1
            continue
        g["truncated"] += 1
        q = questions.get(r["item_id"])
        k = _key(metric, q, resp, judge, None) if q is not None else None
        cached = False
        if k is not None and k.exists():
            try:
                cached = int(json.loads(k.read_text()).get("score", -1)) in (0, 1)
            except Exception:
                cached = False
        if cached:
            g["truncated_rescored"] += 1
        else:
            g["truncated_uncached"] += 1
            if _scored(r, col):
                g["truncated_uncached_but_published_scored"] += 1
    return {g: dict(d) for g, d in out.items()}


def note_16_15_7(*, artefact: Path, log: Path, bootstrap: int, seed: int) -> dict:
    from scripts.analyze_length_matched_elephant import (
        RAW, _cell, _cluster_bootstrap, _delta, _rows, apply_corrected_scores,
    )
    res: dict[str, Any] = {"dataset": "oeq", "metric": "validation"}
    if not (REPO_ROOT / RAW).exists():
        res["not_computable"] = f"raw file {RAW} is absent"
        guard(False, f"16.15.7: raw file {RAW} is absent")
        return res

    cmd = [sys.executable, "-m", "scripts.analyze_length_matched_elephant",
           "--corrected", "--bounds", "--bootstrap", str(bootstrap), "--seed", str(seed),
           "--json", str(artefact.relative_to(REPO_ROOT))]
    env = dict(os.environ)
    for k in API_ENV_BLANKED:
        env[k] = ""
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    log.write_text(proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr.strip() else ""))
    res.update({
        "command": cmd, "returncode": proc.returncode,
        "artefact": str(artefact.relative_to(REPO_ROOT)),
        "console_log": str(log.relative_to(REPO_ROOT)),
        "env_blanked": list(API_ENV_BLANKED),
        "mode_note": "--corrected reads only elephant_rescore_* cache files "
                     "(apply_corrected_scores); the subprocess has no API path in this mode",
    })
    if not guard(proc.returncode == 0, f"16.15.7: elephant subprocess exited {proc.returncode}"):
        res["stderr_tail"] = proc.stderr[-2000:]
        return res
    corrected = parse_corrected_line(proc.stdout)
    guard(corrected is not None, "16.15.7: could not parse the [corrected] line")
    res["corrected_line"] = corrected
    if not guard(artefact.exists(), f"16.15.7: artefact {artefact} was not written"):
        return res
    art = json.loads(artefact.read_text())
    gens = sorted(art["generators"])
    guard(len(gens) == 4, f"16.15.7: expected four generators, found {gens}")

    # Coverage audit and item-clustered intervals on the corrected rows,
    # using the script's own loaders and bootstrap.
    rows = _rows(RAW, "oeq")
    col = "sycophantic_validation"
    n_rep, n_miss = apply_corrected_scores(rows, "oeq", "validation")
    guard(corrected is None or (n_rep == corrected["replaced"] and n_miss == corrected["not_cached"]),
          "16.15.7: in-process corrected counts differ from the subprocess")
    cov = rescore_coverage(rows, "oeq", "validation")
    res["rescore_coverage_two_arms"] = cov
    guard(all(d.get("truncated_uncached_but_published_scored", 0) == 0 for d in cov.values()),
          "16.15.7: a published-scored truncated row in standard_cot/narrative_cot has no "
          "untruncated re-score; the drop would be partly uncorrected")

    drops: dict[str, Any] = {}
    for g in gens:
        cell = art["generators"][g]
        full = cell["full"]
        recs = [{"item_id": r["item_id"], "arm": r["arm"], "score": int(r[col])}
                for arm in ("standard_cot", "narrative_cot") for r in _cell(rows, g, arm, col)]

        def stat(rs):
            c = [x["score"] for x in rs if x["arm"] == "standard_cot"]
            n = [x["score"] for x in rs if x["arm"] == "narrative_cot"]
            if len(c) < 10 or len(n) < 10:
                return None
            return 100.0 * (statistics.mean(n) - statistics.mean(c))

        ci = _cluster_bootstrap(recs, stat, b=bootstrap, seed=seed)
        again = _delta(rows, g, col)
        guard(again["delta_pp"] is not None and full["delta_pp"] is not None
              and abs(again["delta_pp"] - full["delta_pp"]) < 1e-9
              and abs(ci["point"] - full["delta_pp"]) < 1e-9,
              f"16.15.7: in-process drop for {g} differs from the artefact")
        bounds = cell.get("bounds", {})
        drops[g] = {
            "cot_n": full["cot_n"], "cot_rate": full["cot_rate"],
            "not_n": full["not_n"], "not_rate": full["not_rate"],
            "delta_pp": full["delta_pp"],
            "ci_pp": {"lo": ci["lo"], "hi": ci["hi"], "se": ci["se"], "b": ci["b"],
                      "items": ci["items"], "degenerate": ci["degenerate"], "seed": seed},
            "nonresponse_not": cell["nonresponse_not"],
            "nonresponse_cot": cell["nonresponse_cot"],
            "manski_bracket_pp": bounds.get("delta_pp_bracket"),
            "untruncated_lt4000_delta_pp": cell["untruncated"].get("delta_pp"),
            "length_matched_delta_pp": cell["stratified"].get("matched_delta_pp"),
        }
    res["drops"] = drops
    mags = [abs(d["delta_pp"]) for d in drops.values()]
    res["range_pp"] = {
        "min_abs": min(mags), "max_abs": max(mags),
        "phrase": REPLAY["16.15.7"]["range_phrase"],
        "phrase_consistent": (int(min(mags)) == REPLAY["16.15.7"]["lo"]
                              and int(max(mags)) == REPLAY["16.15.7"]["hi"]),
    }
    prior: dict[str, Any] = {"path": str(ELEPHANT_PRIOR.relative_to(REPO_ROOT)),
                             "exists": ELEPHANT_PRIOR.exists()}
    if ELEPHANT_PRIOR.exists():
        try:
            pr = json.loads(ELEPHANT_PRIOR.read_text())["generators"]
            prior["per_generator"] = {
                g: {"prior_delta_pp": pr.get(g, {}).get("full", {}).get("delta_pp"),
                    "agrees": (pr.get(g, {}).get("full", {}).get("delta_pp") is not None
                               and abs(pr[g]["full"]["delta_pp"] - drops[g]["delta_pp"]) < 1e-9)}
                for g in gens}
            prior["all_agree"] = all(v["agrees"] for v in prior["per_generator"].values())
        except Exception as e:  # pragma: no cover
            prior["error"] = repr(e)
    res["prior_artefact"] = prior
    return res


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _f(x: Optional[float], nd: int = 3, sign: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _ci_s(d: Optional[dict], sign: bool = False) -> str:
    if not d or d.get("point") is None:
        return "n/a"
    return f"{_f(d['point'], sign=sign)} [{_f(d.get('lo95'), sign=sign)}, {_f(d.get('hi95'), sign=sign)}]"


def print_report(res: dict) -> None:
    a = res["16.15.5"]
    print("\n=== 16.15.5  narration under differential non-response (bm_embody_analysis.json) ===")
    print(f"registered S0 contrast narrative - standard (NOVERDICT in denominator): "
          f"{_ci_s(a['registered_contrast_S0'], sign=True)}  n=600 per arm")
    for arm in ("standard_cot", "narrative_cot"):
        d = a["arms"][arm]
        s = a.get("snapshot", {}).get("committed_only", {}).get(arm)
        print(f"  {arm:<14} p_S0={_f(d['p_S0'])}  missing={_f(d['missing_rate'])}  "
              f"committed-only={_f(d['committed_only'])} "
              f"(= {d['hits']}/{d['n_committed']})"
              + (f"  CI [{_f(s['lo95'])}, {_f(s['hi95'])}]" if s else "  CI n/a"))
    c = a["committed_only_contrast"]
    s = a.get("snapshot", {}).get("committed_only", {}).get("contrast")
    print(f"  committed-only contrast: {_f(c['point'], sign=True)}"
          + (f"  CI [{_f(s['lo95'], sign=True)}, {_f(s['hi95'], sign=True)}]" if s else "  CI n/a"))
    mk = a["manski_S0_contrast"]
    print(f"  Manski bounds on the S0 contrast (manski_delta): [{_f(mk['lo'], sign=True)}, "
          f"{_f(mk['hi'], sign=True)}]  width {_f(mk['width'])}  crosses zero: {mk['crosses_zero']}")
    print(f"  replay agreement: {json.dumps({k: v['agrees'] for k, v in a['replay'].items()})}")
    sn = a.get("snapshot", {})
    print(f"  snapshot: usable={sn.get('usable')}  files after cutoff={sn.get('n_cache_files_after_cutoff')}"
          f"  S0 interval bit-identical={sn.get('S0_contrast_reproduced', {}).get('bit_identical')}")

    b = res["16.15.6"]
    print("\n=== 16.15.6  mixed panel: P(TRUE) / P(FALSE) / P(abstain) (H_PANEL, majority-of-3) ===")

    def show_block(block: dict, snap: Optional[dict], n_items) -> None:
        for name, p in block["panels"].items():
            sp = (snap or {}).get("panels", {}).get(name, {})
            pf, pa = sp.get("p_false"), sp.get("p_abstain")
            print(f"  {name:<28} P(TRUE)={_ci_s(p['p_true'])}  "
                  f"P(FALSE)={_f(p['p_false']['point'])}"
                  + (f" [{_f(pf['lo95'])}, {_f(pf['hi95'])}]" if pf else "")
                  + f"  P(abstain)={_f(p['p_abstain']['point'])}"
                  + (f" [{_f(pa['lo95'])}, {_f(pa['hi95'])}]" if pa else "")
                  + f"  n_items={n_items}")
        for lab, d in block["deltas"].items():
            sd = (snap or {}).get("deltas", {}).get(lab, {})
            pf, pa = sd.get("p_false"), sd.get("p_abstain")
            print(f"  delta {lab:<28} P(TRUE) {_ci_s(d['p_true'], sign=True)}  "
                  f"P(FALSE) {_f(d['p_false']['point'], sign=True)}"
                  + (f" [{_f(pf['lo95'], sign=True)}, {_f(pf['hi95'], sign=True)}]" if pf else "")
                  + f"  P(abstain) {_f(d['p_abstain']['point'], sign=True)}"
                  + (f" [{_f(pa['lo95'], sign=True)}, {_f(pa['hi95'], sign=True)}]" if pa else ""))

    sp = b.get("snapshot_plain", {})
    print("plain (bm_embody_analysis.json):")
    show_block(b["plain"], sp if sp.get("usable") else None, sp.get("n_items"))
    print(f"  replay agreement: {json.dumps({k: v['agrees'] for k, v in b['replay'].items()})}")
    print(f"  snapshot: usable={sp.get('usable')}  recorded P(TRUE) intervals bit-identical="
          f"{sp.get('all_recorded_intervals_bit_identical')}")
    if "panels" in b.get("narration", {}):
        sn = b.get("snapshot_narration", {})
        print("narration (bm_cross_sig_analysis.json H_PANEL_NAR):")
        show_block(b["narration"], sn if sn.get("usable") else None, sn.get("n_items"))
        print(f"  snapshot: usable={sn.get('usable')}  recorded P(TRUE) intervals bit-identical="
              f"{sn.get('all_recorded_intervals_bit_identical')}")
    else:
        print(f"narration: {b['narration']}")

    c = res.get("16.15.7")
    print("\n=== 16.15.7  manner deltas, corrected scorer (oeq / validation, NoT - CoT, pp) ===")
    if not c or "drops" not in c:
        print(f"  {c}")
    else:
        print(f"  artefact: {c['artefact']}   console: {c['console_log']}   rc={c['returncode']}")
        print(f"  {c['corrected_line']['line']}")
        print(f"  {'generator':<26}{'CoT n':>6}{'CoT':>7}{'NoT n':>7}{'NoT':>7}{'drop pp':>9}"
              f"{'95% CI (item-clustered)':>26}{'Manski pp':>20}{'NoT miss':>9}")
        for g, d in c["drops"].items():
            ci = d["ci_pp"]
            mb = d["manski_bracket_pp"]
            print(f"  {g:<26}{d['cot_n']:>6}{d['cot_rate']:>7.3f}{d['not_n']:>7}"
                  f"{d['not_rate']:>7.3f}{d['delta_pp']:>+9.3f}"
                  f"{'[' + _f(ci['lo'], sign=True) + ', ' + _f(ci['hi'], sign=True) + ']':>26}"
                  f"{('[' + _f(mb[0], sign=True) + ', ' + _f(mb[1], sign=True) + ']') if mb else 'n/a':>20}"
                  f"{d['nonresponse_not']['missing']:>9}")
        r = c["range_pp"]
        print(f"  |drop| range {_f(r['min_abs'])} to {_f(r['max_abs'])} pp; "
              f"'{r['phrase']}' consistent: {r['phrase_consistent']}")
        print(f"  coverage (two arms): " + json.dumps(c["rescore_coverage_two_arms"]))
        print(f"  prior artefact {c['prior_artefact']['path']}: exists={c['prior_artefact']['exists']}"
              f" all_agree={c['prior_artefact'].get('all_agree')}")
    print(f"\nGUARD: {'ok' if not GUARD_FAILURES else str(len(GUARD_FAILURES)) + ' failure(s)'}")


# ---------------------------------------------------------------------------
# Selftest (synthetic; touches no cache, no CSV, no artefact)
# ---------------------------------------------------------------------------

def _selftest() -> int:
    fails: list[str] = []

    def check(name: str, cond: bool) -> None:
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    # The guard: a failing condition prints the literal GUARD FAILED and is
    # recorded; a passing one prints nothing. GUARD_FAILURES is reset after.
    import contextlib
    import io
    saved = list(GUARD_FAILURES)
    GUARD_FAILURES.clear()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ret_bad = guard(False, "synthetic failure")
    check("guard(False, ...) prints the literal GUARD FAILED and returns False",
          not ret_bad and "GUARD FAILED" in buf.getvalue() and "synthetic failure" in buf.getvalue())
    check("guard(False, ...) appends the message to GUARD_FAILURES",
          GUARD_FAILURES == ["synthetic failure"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ret_ok = guard(True, "synthetic pass")
    check("guard(True, ...) prints nothing, returns True and records nothing",
          ret_ok and buf.getvalue() == "" and GUARD_FAILURES == ["synthetic failure"])
    GUARD_FAILURES.clear()
    GUARD_FAILURES.extend(saved)
    check("GUARD_FAILURES is restored after the guard checks", GUARD_FAILURES == saved)

    # manski_delta hand case: a = 3 TRUE, 1 MISSING of 10; b = 2 TRUE, 2 MISSING of 10.
    a = {"i1": [TRUE, TRUE, FALSE, FALSE, MISSING], "i2": [TRUE, FALSE, FALSE, FALSE, FALSE]}
    b = {"i1": [TRUE, FALSE, FALSE, MISSING, MISSING], "i2": [TRUE, FALSE, FALSE, FALSE, FALSE]}
    mk = manski_delta(a, b)
    check("manski_delta hand case: lo = 0.3 - 0.4 = -0.100", abs(mk["lo"] + 0.1) < 1e-12)
    check("manski_delta hand case: hi = 0.4 - 0.2 = +0.200", abs(mk["hi"] - 0.2) < 1e-12)
    check("manski_delta hand case: missing rates 0.1 / 0.2",
          abs(mk["missing_a"] - 0.1) < 1e-12 and abs(mk["missing_b"] - 0.2) < 1e-12)
    mk2 = manski_delta(pseudo_items(arm_counts(arm_estimate(a))),
                       pseudo_items(arm_counts(arm_estimate(b))))
    check("manski_delta is invariant to collapsing items into one pseudo-item",
          all(abs(mk[k] - mk2[k]) < 1e-12 for k in ("lo", "hi", "missing_a", "missing_b")))

    co = committed_only(arm_estimate(a))
    check("committed-only p/(1-missing) equals hits/(n-missing) = 3/9",
          co["formulas_agree"] and abs(co["committed_only"] - 3 / 9) < 1e-12)
    check("committed_p pooled over items equals 3/9",
          abs(committed_p(a, ["i1", "i2"]) - 3 / 9) < 1e-12)
    check("non-integral counts are flagged",
          not arm_counts({"p": 0.3333, "n": 600, "missing_rate": 0.1})["integral"]
          and arm_counts({"p": 0.15666666666666668, "n": 600, "missing_rate": 0.345})["integral"])

    # panel_probs is a distribution over {TRUE, FALSE, NOPANEL}.
    rng = random.Random(0)
    ok = True
    for _ in range(200):
        dists = []
        for _ in range(3):
            x = [rng.random() for _ in range(3)]
            s = sum(x)
            dists.append({TRUE: x[0] / s, FALSE: x[1] / s, MISSING: x[2] / s})
        pp = panel_probs(dists)
        ok &= abs(pp[TRUE] + pp[FALSE] + pp[NOPANEL] - 1.0) < 1e-12
    check("panel_probs sums to one over 200 random member distributions", ok)
    rec = {"p_true": {"point": 0.162}, "abstain": 0.286}
    check("P(FALSE) from a recorded pair is the remaining mass",
          abs(p_false_from_record(rec) - 0.552) < 1e-12)

    # panel_stat means sum to one across items, and the fast path matches it.
    arms = {}
    for name in ("x", "y", "z"):
        arms[name] = {str(i): [rng.choice((TRUE, FALSE, MISSING)) for _ in range(6)]
                      for i in range(20)}
    ids = sorted(arms["x"])
    members = [arms["x"], arms["y"], arms["z"]]
    tot = sum(panel_stat(members, ids, k) for k in (TRUE, FALSE, NOPANEL))
    check("panel_stat P(TRUE)+P(FALSE)+P(abstain) = 1 across items", abs(tot - 1.0) < 1e-12)
    per = per_item_panel(members, ids)
    check("fast per-item statistic reproduces panel_stat for every key",
          all(abs(panel_key_stat(per, k)(ids) - panel_stat(members, ids, k)) < 1e-15
              for k in (TRUE, FALSE, NOPANEL)))
    ci_a = bootstrap_ci(lambda s: panel_stat(members, s, FALSE), ids, draws=200, seed=3)
    ci_b = bootstrap_ci(panel_key_stat(per, FALSE), ids, draws=200, seed=3)
    check("fast path gives the identical bootstrap interval on the same seed", ci_equal(ci_a, ci_b))

    # Snapshot assembly on synthetic units with a cutoff.
    class U:
        """The WorkUnit fields load_cached_records reads, nothing else."""
        model, role, nuisance_idx, channel, direction, arm_id = (
            "m", "r", 0, "neutral_filler", "none", "arm")

        def __init__(self, scaffold, pid, idx, path):
            self.scaffold, self.problem_id, self.sample_idx, self.cache_path = scaffold, pid, idx, path
    import tempfile
    import time
    with tempfile.TemporaryDirectory() as td:
        units = []
        for k in range(4):
            p = Path(td) / f"u{k}.json"
            p.write_text(json.dumps({"scaffold": "s", "problem_id": "1", "sample_idx": k,
                                     "verdict": TRUE if k % 2 == 0 else FALSE}))
            units.append(U("s", "1", k, str(p)))
        units.append(U("s", "1", 4, str(Path(td) / "absent.json")))
        cut = time.time() + 5
        snap_all = snapshot_collect(units, cut)
        os.utime(units[3].cache_path, (cut + 100, cut + 100))
        snap_cut = snapshot_collect(units, cut)
        check("snapshot: absent unit is MISSING, all cached read before the cutoff",
              snap_all["s"]["1"] == [TRUE, FALSE, TRUE, FALSE, MISSING]
              and snap_all["_n_after_cutoff"] == 0)
        check("snapshot: a file written after the cutoff is treated as MISSING",
              snap_cut["s"]["1"] == [TRUE, FALSE, TRUE, MISSING, MISSING]
              and snap_cut["_n_after_cutoff"] == 1)

    # 16.15.7 pieces: the corrected-line parser and the drop statistic.
    pc = parse_corrected_line("x\n[corrected] replaced 936 truncated scores from the re-score "
                              "cache; 1129 not cached (left as published)\ny")
    check("parse_corrected_line reads replaced/not-cached counts",
          pc == {"replaced": 936, "not_cached": 1129, "line": pc["line"]} if pc else False)
    pc2 = parse_corrected_line("[corrected] replaced 5 truncated scores from the re-score cache")
    check("parse_corrected_line: no 'not cached' clause means zero",
          pc2 is not None and pc2["not_cached"] == 0)
    from scripts.analyze_length_matched_elephant import _cluster_bootstrap
    recs = ([{"item_id": str(i), "arm": "standard_cot", "score": 1 if i < 12 else 0} for i in range(20)]
            + [{"item_id": str(i), "arm": "narrative_cot", "score": 1 if i < 4 else 0} for i in range(20)])

    def stat(rs):
        c = [x["score"] for x in rs if x["arm"] == "standard_cot"]
        n = [x["score"] for x in rs if x["arm"] == "narrative_cot"]
        return None if len(c) < 10 or len(n) < 10 else 100.0 * (statistics.mean(n) - statistics.mean(c))

    ci = _cluster_bootstrap(recs, stat, b=300, seed=1)
    check("drop statistic on synthetic paired rows: 20% - 60% = -40.000 pp",
          abs(ci["point"] + 40.0) < 1e-12 and ci["items"] == 20 and ci["lo"] < -40.0 < ci["hi"])

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Addendum 16.15 items 5-7: BrokenMath notes")
    ap.add_argument("--json", type=Path, default=NOTES_JSON)
    ap.add_argument("--draws", type=int, default=EMBODY_DRAWS,
                    help="bootstrap draws for the snapshot intervals (10,000 reproduces "
                         "the recorded intervals bit-for-bit; anything else skips that check)")
    ap.add_argument("--elephant-json", type=Path, default=ELEPHANT_JSON)
    ap.add_argument("--elephant-log", type=Path, default=ELEPHANT_LOG)
    ap.add_argument("--elephant-bootstrap", type=int, default=ELEPHANT_BOOTSTRAP)
    ap.add_argument("--elephant-seed", type=int, default=ELEPHANT_SEED)
    ap.add_argument("--skip-elephant", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()

    embody = json.loads(EMBODY_JSON.read_text())
    cross = json.loads(CROSS_JSON.read_text()) if CROSS_JSON.exists() else None
    res: dict[str, Any] = {
        "registration": "Guidance_Documents/prereg_embodiment_community.md, Addendum 16.15, "
                        "items 16.15.5, 16.15.6, 16.15.7",
        "spend_usd": 0.0,
        "interval_estimator_note": (
            "the registered readouts of 16.15.5-16.15.7 are point values read from the named "
            "JSON fields and the elephant artefact; the intervals ADDED here are NOT on the "
            "16.15.1 settings (4,000 draws, seed 13) but on each parent artefact's own estimator, "
            "draws and seeds (bm_embody_analysis: 10,000 draws, seeds 7+k; bm_cross_sig: 10,000 "
            "draws, seeds 17+k; elephant: 8,000 draws, seed 20260822), so that they sit on the "
            "very resamples the parent intervals were drawn from; when quoted, say 'same "
            "estimator, draws and seeds as the parent artefact'"),
        "sources": {
            "bm_embody_analysis": {"path": str(EMBODY_JSON.relative_to(REPO_ROOT)),
                                   "mtime": os.path.getmtime(EMBODY_JSON)},
            "bm_cross_sig_analysis": {"path": str(CROSS_JSON.relative_to(REPO_ROOT)),
                                      "mtime": os.path.getmtime(CROSS_JSON) if cross else None},
        },
        "16.15.5": note_16_15_5(embody, draws=a.draws),
        "16.15.6": note_16_15_6(embody, cross, draws=a.draws),
    }
    if a.skip_elephant:
        res["16.15.7"] = {"skipped": True}
    else:
        res["16.15.7"] = note_16_15_7(artefact=a.elephant_json.resolve(),
                                      log=a.elephant_log.resolve(),
                                      bootstrap=a.elephant_bootstrap, seed=a.elephant_seed)
    res["guard"] = {"ok": not GUARD_FAILURES, "failures": list(GUARD_FAILURES)}
    print_report(res)
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(res, indent=2))
    print(f"wrote {a.json}")
    return 0 if not GUARD_FAILURES else 2


if __name__ == "__main__":
    raise SystemExit(main())
