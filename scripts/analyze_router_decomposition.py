"""
scripts/analyze_router_decomposition.py -- Addendum 16.15.1, 16.15.2, 16.15.3
and 16.22: is the dissent sensor's error concentration the dissent, or the
verdict type; and what does routing certify, at what price?
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

16.22    ROUTING CERTIFICATION on the same 420 cells. Base = grok standard
         majority-of-3 (the solo generator), judges = the cached haiku and
         sonnet standard majority-of-3. For six signals (counter, composite,
         any ESH/NAH, disagreement, raw-string disagreement, and the union
         counter OR disagreement): routed accuracy (base unless the signal
         fires, judge if it fires, base when the judge is non-codable), the
         paired delta over the base and over the judge used everywhere
         (item-clustered CIs), coverage, judge emission, random routing at
         the same coverage, and the pairwise routed-vs-routed deltas. A
         REPLAY GATE precedes every new number: the composite -> sonnet
         routed accuracy and its delta over the base must reproduce 16.11a
         (0.9354, +0.0239 on 418 cells) within 0.005 on the point, on the
         420-cell population and on the 418-cell population (the two sonnet
         majority-tie cells dropped, as 16.11a did), and the existing judge-
         emission guard must hold on every signal; otherwise GUARD FAILED
         and the block is UNREAD. The registration's pre-declared readings
         R1-R5 (counter -> sonnet vs base; vs sonnet everywhere; disagreement
         vs counter; union vs disagreement; the accuracy-cost frontier) are
         applied mechanically, with the haiku replicate reported only.
         TWO UNIONS, not to be conflated: the CERTIFIED union of this block
         (signal "union", R4) is counter OR disagreement, as the 16.22
         registration text defines it; 16.11a's union (0.9498 / +0.0383 /
         cov 0.1746 on 418 cells) was the COMPOSITE (embodied) flag OR
         disagreement, and appears here only as the non-gating secondary
         replay line union_composite_or_disagree_*. The prereg's parenthesis
         "(16.11a's union)" on the counter-based definition is a mislabel.
         COST PER DECISION from the cached records'
         own prompt/completion token counts priced with
         run_crowdgold_deliberation.PRICES (means only): the base alone, the
         judge everywhere, each solo-signal router, the collective k=1
         sensor, and the k=4 sensor as actually run. Written to
         router_decomposition.json["routed_vs_solo" / "cost_per_decision"]
         and to the standalone routing_certification.json. Zero spend: cost
         records are READ from the cache by their runner-issued names and
         never generated; no code path here can call a model.

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

PER-MODEL READOUT MODE (exploratory, not registered; zero spend). The same
decomposition on another panel's rows/votes: --model M --rows/--votes the
panel's CSVs, --solo model:scaffold (the solo comparator in the 16.15.2
signals and the within-strata transfer; default grok standard), --judges
model:scaffold,... (first = primary judge, the "haiku" slot, read from the
merged comparator; optional second = the coverage-restricted judge, the
"sonnet" slot, read from --sonnet-rows merged with the comparator; default
haiku,sonnet), --extra-comparator CSVs merged into cg_scaffold_combined_rows_k3
(load_comparator format, samples concatenated per key). Transfer targets are
the registered four for grok, otherwise every (model, scaffold) pair in the
merged comparator, restricted by --transfer-targets. The grok-only blocks --
agreement with the auditor's replay, R5 agreement, and the pre-declared
readings of 16.15.1/2 -- are marked not_applicable when --model is not grok.
Artefacts default to router_decomposition_<short>.json /
transfer_readouts_<short>.json (short: nano, haiku, sonnet, else the model id)
so the registered grok artefacts are never overwritten. The artefact keys
judge_haiku / judge_sonnet / grok_solo name the SLOTS and carry the actual
(model, scaffold) used. The default invocation is byte-identical to
router_decomposition.json (the selftest reruns it and diffs).

REGISTERED = THE FULL DEFAULT INVOCATION, not merely --model grok
(registered_deviations()): grok panel rows/votes, the k3 comparator, the
16.11a sonnet cache, --solo grok standard, --judges haiku,sonnet standard,
no --extra-comparator / --transfer-targets, --records-dir the outputs dir,
--n-boot 4000 / --seed 13 / --trim-seeds 200, k derived from the panel (4).
Any deviation on the grok panel is exploratory: main() refuses to run it
without explicit --out / --transfer-out / --cert-out (and refuses those
naming a registered artefact), every artefact it writes carries the
`readout` marker with the deviations, the grok-only agreement / R5 /
pre-declared-reading blocks are not_applicable, and a passed 16.22 gate
reads "exploratory (gate passed; not the registered invocation)", never
READ. The k-sensor bundle of the cost table prices the k the panel actually
ran (max sample_idx + 1: 4 grok, 2 nano; --k-sensor overrides), and
routing_certification_<short>.json names the parent artefact it was
actually written beside.

Usage
-----
  python -m scripts.analyze_router_decomposition --selftest
  python -m scripts.analyze_router_decomposition
  python -m scripts.analyze_router_decomposition --model gpt-5.4-nano \
      --rows divergence_study_outputs/cg_deliberation_nano_rows.csv \
      --votes divergence_study_outputs/cg_deliberation_nano_votes.csv \
      --solo gpt-5.4-nano:standard
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, Sequence

import scripts.run_crowdgold_deliberation as rcd
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
from scripts.run_crowdgold_aita import gen_cache_path
from scripts.run_phase1_quartet import OUT_DIR

ANALYSIS_PATH = OUT_DIR / "router_decomposition.json"
TRANSFER_PATH = OUT_DIR / "transfer_readouts.json"
SONNET_ROWS_PATH = OUT_DIR / "cg_sonnet_actuator_rows.csv"
CERT_PATH = OUT_DIR / "routing_certification.json"          # Addendum 16.22

GROK = "grok-4-1-fast-reasoning"
HAIKU_JUDGE = ("claude-haiku-4-5", "standard")
SONNET_JUDGE = ("claude-sonnet-4-6", "standard")
GROK_SOLO = (GROK, "standard")
DEFAULT_DRAWS = 4000
DEFAULT_SEED = 13
TRIM_SEEDS = 200
REGISTERED_K = 4            # samples per (arm, item) in the registered grok panel
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

#: Per-model mode: artefact suffix router_decomposition_<short>.json
SHORT_NAMES = {GROK: "grok", "gpt-5.4-nano": "nano", "claude-haiku-4-5": "haiku",
               "claude-sonnet-4-6": "sonnet"}


def short_name(model: str) -> str:
    return SHORT_NAMES.get(model, model.replace("/", "_"))


def parse_model_scaffold(s: str) -> tuple[str, str]:
    """'model:scaffold' -> (model, scaffold); the scaffold defaults to standard."""
    m, _, sc = s.strip().partition(":")
    if not m:
        raise ValueError(f"bad model:scaffold spec {s!r}")
    return m, (sc or "standard")


def panel_k(debates: Sequence[dict]) -> int:
    """Samples per (arm, item) actually run on a panel: max sample_idx + 1
    (4 on the registered grok panel, 2 on the nano community). The cost
    table's k-sensor bundle prices exactly this many debates per cell, so it
    is derived from the data rather than assumed to be the grok value."""
    idx = [int(d["sample_idx"]) for d in debates if d.get("sample_idx") is not None]
    return (max(idx) + 1) if idx else REGISTERED_K


def registered_deviations(*, model: str, solo: tuple[str, str],
                          judges: Sequence[tuple[str, str]],
                          rows: Path, votes: Path, comparator: Path, sonnet_rows: Path,
                          extras: Sequence[Path] = (), transfer_targets: Optional[str] = None,
                          records_dir: Path = OUT_DIR, draws: int = DEFAULT_DRAWS,
                          seed: int = DEFAULT_SEED, trim_seeds: int = TRIM_SEEDS,
                          k_sensor: Optional[int] = None) -> list[str]:
    """Every way an invocation differs from the REGISTERED one (Addendum
    16.15 / 16.22): the grok panel's rows and votes, the k3 comparator, the
    16.11a sonnet cache, grok standard as the solo, haiku then sonnet
    standard as the judges, no extra comparator, the registered transfer
    targets, the cached records under OUT_DIR, the default estimator, and k
    derived from the panel. Empty list = registered. The registered artefact
    paths (router_decomposition.json / transfer_readouts.json /
    routing_certification.json) are written ONLY by a registered invocation;
    any deviation needs explicit output paths and is marked exploratory."""
    dev: list[str] = []
    if model != GROK:
        dev.append(f"--model {model} (registered {GROK})")
    if tuple(solo) != GROK_SOLO:
        dev.append(f"--solo {solo[0]}:{solo[1]} (registered {GROK_SOLO[0]}:{GROK_SOLO[1]})")
    if tuple(tuple(j) for j in judges) != (HAIKU_JUDGE, SONNET_JUDGE):
        dev.append(f"--judges {','.join(':'.join(j) for j in judges)} "
                   f"(registered {':'.join(HAIKU_JUDGE)},{':'.join(SONNET_JUDGE)})")
    for flag, got, reg in (("--rows", rows, ROWS_PATH), ("--votes", votes, VOTES_PATH),
                           ("--comparator", comparator, COMPARATOR_PATH),
                           ("--sonnet-rows", sonnet_rows, SONNET_ROWS_PATH),
                           ("--records-dir", records_dir, OUT_DIR)):
        if Path(got).resolve() != Path(reg).resolve():
            dev.append(f"{flag} {got} (registered {reg})")
    if extras:
        dev.append(f"--extra-comparator {[str(p) for p in extras]}")
    if transfer_targets is not None:
        dev.append(f"--transfer-targets {transfer_targets} (registered: the R5 four)")
    for flag, got, reg in (("--n-boot", draws, DEFAULT_DRAWS), ("--seed", seed, DEFAULT_SEED),
                           ("--trim-seeds", trim_seeds, TRIM_SEEDS)):
        if got != reg:
            dev.append(f"{flag} {got} (registered {reg})")
    if k_sensor is not None and k_sensor != REGISTERED_K:
        dev.append(f"--k-sensor {k_sensor} (registered {REGISTERED_K}, derived from the panel)")
    return dev


def merge_comparators(comps: Sequence[dict]) -> dict[tuple, list[str]]:
    """Union of load_comparator dicts, samples concatenated per key."""
    out: dict[tuple, list[str]] = defaultdict(list)
    for c in comps:
        for k, vs in c.items():
            out[k].extend(vs)
    return out


def comparator_targets(comp: dict, restrict: Optional[Sequence[tuple[str, str]]] = None,
                       ) -> dict[str, tuple[str, str]]:
    """Transfer targets from the (model, scaffold) pairs of a comparator,
    labelled model:scaffold. `restrict` keeps only the listed pairs, in the
    listed order (a pair absent from the comparator stays, and reads as zero
    codable cells rather than silently vanishing)."""
    pairs = sorted({(k[0], k[1]) for k in comp})
    if restrict is not None:
        pairs = [tuple(p) for p in restrict]
    return {f"{m}:{s}": (m, s) for m, s in pairs}


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

#: 16.11a RESULTS (run 2026-09-12; prereg_embodiment_community.md), the replay
#: target of the 16.22 gate. 418 (arm, item) cells: the two sonnet majority-
#: tie cells were dropped. "embodied flag" there = the composite flag by
#: majority of the k=4 samples; "union" there = composite OR own-samples-
#: disagree. Only the two GATE fields gate (point values, tolerance
#: GATE_TOL_16_11A); the rest are reported as secondary agreement.
REPLAY_16_11A = {
    "n_cells": 418, "base": 0.9115,
    "routed_composite": 0.9354,                # GATE
    "delta_composite_vs_base": 0.0239,         # GATE
    "lo": 0.0024, "hi": 0.0452,                # 16.11a's own interval (unrecorded resampling seed)
    "coverage_composite": 0.127,
    "routed_union_composite_or_disagree": 0.9498, "coverage_union": 0.175,
    "delta_union_vs_base": 0.0383,
    "judge_everywhere": 0.9761,
}
GATE_TOL_16_11A = 0.005

#: Regression snapshot (Addendum 16.22 engineering rule): sha256 of the
#: canonical JSON (sort_keys, default=str) of every top-level key that
#: existed in the artefacts BEFORE 16.22 was added (all keys except `inputs`).
#: The selftest reruns the default invocation and requires equality, so any
#: change to a pre-existing number fails the selftest.
PREEXISTING_KEYS_RD = (
    "registration", "draws", "seed", "trim_seeds", "population",
    "verdict_type_source", "judge_haiku", "judge_sonnet", "grok_solo",
    "stratum_counts", "counter_fire_total", "counter_lift_by_stratum",
    "s1_verdict_type_sensitivity", "composed_haiku", "composed_sonnet",
    "transfer_within_strata", "solo_signals", "agreement",
    "pre_declared_readings", "resampling_frame", "guards")
PREEXISTING_KEYS_TR = (
    "n_cells", "n_items", "targets", "n_targets_ci_excludes_zero_counter",
    "n_targets_ci_excludes_zero_composite", "registration", "draws", "seed",
    "cell_flag_rule", "r5_quoted", "r5_agreement", "guards_passed",
    "resampling_frame")
PREEXISTING_DIGEST_RD = "1ae6e164a34e8593e0d27117da84e59e553c6be8cb0c3e843fc03aa0e94625e6"
PREEXISTING_DIGEST_TR = "73a5ed528d98407f4610b5f02877206402a4779b4c7eb5f89e5c3cc6b45a1182"


def preexisting_digest(art: dict, keys: Sequence[str]) -> str:
    canon = json.dumps({k: art[k] for k in keys if k in art}, sort_keys=True,
                       default=str)
    return hashlib.sha256(canon.encode()).hexdigest()


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


def sonnet_covered(debates: Sequence[dict], sonnet_comp: dict,
                   judge: tuple[str, str] = SONNET_JUDGE) -> list[dict]:
    key = (judge[0], judge[1])
    return [d for d in debates if sonnet_comp.get((*key, d["arm"], d["item"]))]


# ---------------------------------------------------------------------------
# Cells (16.15.2, 16.15.3): one record per (arm, item), flags by majority of
# the cell's debate samples; grok-solo signals from its three cached samples
# ---------------------------------------------------------------------------

def build_cells(debates: Sequence[dict], comp: dict,
                solo: tuple[str, str] = GROK_SOLO) -> list[dict]:
    """Cell = (arm, item). A cell FIRES on a sensor when strictly more than
    half of its debate samples fire (the 16.11a / topology_2x2 convention).
    Grok-solo fields come from its three cached standard samples (`solo`
    names the (model, scaffold) in per-model mode)."""
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
        vs = comp.get((*solo, c["arm"], c["item"]), [])
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
# 16.22 -- routing certification on cells: routed vs the solo base, per judge
# ---------------------------------------------------------------------------

def union_cell(c: dict) -> bool:
    """The 16.22 union: the collective's counter OR the solo's own-samples
    disagreement (code_response collapse)."""
    return bool(c["counter_fired"] or c["solo_disagree"])


#: Signals of the 16.22 block, in report order. CELL_SIGNALS is left as is
#: (16.15.2 iterates its own explicit list); `union` is new here.
ROUTED_SIGNALS: dict[str, Callable[[dict], bool]] = {
    "counter": CELL_SIGNALS["counter"],
    "composite": CELL_SIGNALS["composite"],
    "any_eshnah": CELL_SIGNALS["any_eshnah"],
    "disagree": CELL_SIGNALS["disagree"],
    "disagree_raw": CELL_SIGNALS["disagree_raw"],
    "union": union_cell,
}
ROUTED_PAIRS = (("disagree", "counter"), ("union", "counter"),
                ("union", "disagree"), ("composite", "counter"))
SOLO_SIGNALS = ("any_eshnah", "disagree", "disagree_raw")     # need only the base's 3 samples
COLLECTIVE_SIGNALS = ("counter", "composite")                 # need the deliberation


def _excludes_zero(lo, hi) -> bool:
    return lo is not None and hi is not None and (lo > 0 or hi < 0)


def _paired_cells(pop: Sequence[dict], ok_a: Callable[[dict], bool],
                  ok_b: Callable[[dict], bool], *, draws: int, seed: int) -> dict:
    """Accuracy of system a minus system b, paired per cell, item-clustered
    percentile bootstrap (boot_ci)."""
    by: dict = defaultdict(list)
    a_tot = b_tot = 0
    for c in pop:
        a, b = int(ok_a(c)), int(ok_b(c))
        a_tot += a
        b_tot += b
        by[c["item"]].append(a - b)
    n = len(pop)
    pt, lo, hi = boot_ci(by, lambda rs: (sum(rs) / len(rs)) if rs else None,
                         draws=draws, seed=seed)
    return {"acc_a": a_tot / n if n else None, "acc_b": b_tot / n if n else None,
            "delta": pt, "lo": lo, "hi": hi, "n": n,
            "ci_excludes_zero": _excludes_zero(lo, hi)}


def _random_cells_at_coverage(pop: Sequence[dict], base: dict, judge_rule,
                              target_n: int, *, seed: int, trim_seeds: int) -> dict:
    """Composed accuracy when a random `target_n` of the cells are routed to
    the judge (same seeds as the 16.15.2 random rows: seed + 5000 + s)."""
    n = len(pop)
    accs = []
    for s in range(trim_seeds):
        t = trim_sensor(lambda c: True, pop, target_n, seed + 5000 + s)
        accs.append(sum(cell_composed_ok(c, base[id(c)], t, judge_rule)
                        for c in pop) / n if n else None)
    accs.sort()
    return {"target_n": target_n, "coverage": target_n / n if n else None,
            "mean": sum(accs) / len(accs), "best": accs[-1], "worst": accs[0],
            "p05": accs[int(0.05 * len(accs))],
            "p95": accs[min(int(0.95 * len(accs)), len(accs) - 1)],
            "n_seeds": trim_seeds}


def routed_vs_base_block(cells: Sequence[dict], base_rule, judge_rule,
                         judge_label: str, *, draws: int, seed: int,
                         trim_seeds: int, signals: Optional[dict] = None,
                         pairs: Sequence[tuple[str, str]] = ROUTED_PAIRS) -> dict:
    """16.22. Population = the cells on which the base majority is codable.
    For each signal: routed accuracy (cell_composed_ok: base unless the
    signal fires, judge if it fires, base if the judge is non-codable), the
    base, the paired routed - base delta, the judge used everywhere (with
    the same base fallback) and the paired routed - judge-everywhere delta,
    coverage, judge emission on fired cells, random routing at the signal's
    coverage; then the pairwise routed-vs-routed deltas of `pairs`."""
    signals = ROUTED_SIGNALS if signals is None else signals
    pop = [c for c in cells if solo_ok(c, base_rule) is not None]
    base = {id(c): bool(solo_ok(c, base_rule)) for c in pop}
    n = len(pop)
    always = lambda c: True

    def routed_fn(sig):
        return lambda c: cell_composed_ok(c, base[id(c)], sig, judge_rule)

    base_ok = lambda c: base[id(c)]
    ev_ok = routed_fn(always)
    n_judge_codable = sum(1 for c in pop if solo_ok(c, judge_rule) is not None)
    out: dict = {
        "judge": judge_label, "n_cells_total": len(cells), "n_cells": n,
        "n_items": len({c["item"] for c in pop}),
        "n_judge_codable": n_judge_codable,
        "acc_base": (sum(base.values()) / n) if n else None,
        "acc_judge_everywhere": (sum(ev_ok(c) for c in pop) / n) if n else None,
        "judge_everywhere_minus_base": _paired_cells(pop, ev_ok, base_ok,
                                                     draws=draws, seed=seed),
        "signals": {}, "pairwise": {},
    }
    for name, sig in signals.items():
        r_ok = routed_fn(sig)
        fired = [c for c in pop if sig(c)]
        emitted = sum(1 for c in fired if solo_ok(c, judge_rule) is not None)
        vs_base = _paired_cells(pop, r_ok, base_ok, draws=draws, seed=seed)
        vs_ev = _paired_cells(pop, r_ok, ev_ok, draws=draws, seed=seed)
        rnd = _random_cells_at_coverage(pop, base, judge_rule, len(fired),
                                        seed=seed, trim_seeds=trim_seeds)
        out["signals"][name] = {
            "n_fired": len(fired), "coverage": len(fired) / n if n else None,
            "acc_routed": vs_base["acc_a"], "acc_base": vs_base["acc_b"],
            "routed_minus_base": vs_base,
            "ci_excludes_zero": vs_base["ci_excludes_zero"],
            "acc_judge_everywhere": vs_ev["acc_b"],
            "routed_minus_judge_everywhere": vs_ev,
            "judge_emitted_on_fired": (emitted / len(fired)) if fired else None,
            "n_fallback_to_base": len(fired) - emitted,
            "random_at_coverage": rnd,
            "routed_minus_random_mean": (vs_base["acc_a"] - rnd["mean"]
                                         if vs_base["acc_a"] is not None and rnd["mean"] is not None
                                         else None),
        }
    for a, b in pairs:
        out["pairwise"][f"{a}_minus_{b}"] = _paired_cells(
            pop, routed_fn(signals[a]), routed_fn(signals[b]), draws=draws, seed=seed)
    return out


def _point_routed(cells: Sequence[dict], base_rule, judge_rule, sig) -> dict:
    """Point-only routed accuracy of a signal (no interval), for the
    secondary agreement lines of the gate."""
    pop = [c for c in cells if solo_ok(c, base_rule) is not None]
    n = len(pop)
    base = {id(c): bool(solo_ok(c, base_rule)) for c in pop}
    r = sum(cell_composed_ok(c, base[id(c)], sig, judge_rule) for c in pop)
    b = sum(base.values())
    f = sum(1 for c in pop if sig(c))
    return {"acc_routed": r / n if n else None, "acc_base": b / n if n else None,
            "delta": (r - b) / n if n else None, "coverage": f / n if n else None,
            "n": n}


def replay_gate_16_11a(block_full: Optional[dict], block_judge_codable: Optional[dict],
                       *, replay: dict = REPLAY_16_11A, tol: float = GATE_TOL_16_11A,
                       secondary: Optional[dict] = None,
                       emission_blocks: Optional[dict] = None) -> dict:
    """The 16.22 replay gate. GATE fields: the composite -> sonnet routed
    accuracy and its paired delta over the base, each within `tol` of
    16.11a's point values, on the full (420) population and on the judge-
    codable (418) population. The gate PASSES when at least one population
    reproduces both gate fields AND the existing judge-emission guard holds
    on every signal of every block in `emission_blocks` (judge codable on
    >= MIN_JUDGE_EMISSION of the fired cells); the artefact states which
    population reproduces. 16.11a's own interval was computed in-session
    with an unrecorded resampling seed and is reported, not gated.
    `secondary`: point-only agreement lines (the 16.11a union, the judge
    everywhere) that are reported, never gated."""
    out: dict = {"replay": dict(replay), "tolerance": tol, "gate_fields":
                 ["acc_routed_composite", "delta_composite_vs_base"],
                 "populations": {}, "secondary": secondary or {},
                 "emission": {}, "emission_limit": MIN_JUDGE_EMISSION, "emission_ok": True}
    for slot, blk in (emission_blocks or {}).items():
        if blk is None:
            continue
        ems = {k: v["judge_emitted_on_fired"] for k, v in blk["signals"].items()}
        worst = min((e for e in ems.values() if e is not None), default=None)
        ok = worst is None or worst >= MIN_JUDGE_EMISSION
        out["emission"][slot] = {"by_signal": ems, "min": worst, "ok": ok}
        out["emission_ok"] = out["emission_ok"] and ok
    if block_full is None or block_judge_codable is None:
        out.update({"status": "not_computable", "passed": False, "reproduces": [],
                    "reason": "sonnet judge block absent (cache missing)"})
        return out
    for label, blk in (("full", block_full), ("judge_codable", block_judge_codable)):
        s = blk["signals"]["composite"]
        fields = {
            "acc_routed_composite": _agree(s["acc_routed"], replay["routed_composite"], tol),
            "delta_composite_vs_base": _agree(s["routed_minus_base"]["delta"],
                                              replay["delta_composite_vs_base"], tol),
            # reported, not gated
            "acc_base": _agree(blk["acc_base"], replay["base"], tol),
            "coverage_composite": _agree(s["coverage"], replay["coverage_composite"], tol),
            "acc_judge_everywhere": _agree(blk["acc_judge_everywhere"],
                                           replay["judge_everywhere"], tol),
        }
        out["populations"][label] = {
            "n_cells": blk["n_cells"], "fields": fields,
            "reproduces": all(fields[k]["agrees"] for k in out["gate_fields"]),
            "artefact_interval": [s["routed_minus_base"]["lo"], s["routed_minus_base"]["hi"]],
            "replay_interval": [replay["lo"], replay["hi"]],
        }
    rep = [k for k, v in out["populations"].items() if v["reproduces"]]
    out["reproduces"] = rep
    out["passed"] = bool(rep) and out["emission_ok"]
    out["status"] = "passed" if out["passed"] else "failed"
    out["reason"] = ("" if out["passed"] else
                     ("16.11a not reproduced on either population" if not rep else "")
                     + ("" if out["emission_ok"] else
                        ("; " if not rep else "") + "judge emission on fired cells below the guard"))
    out["note"] = ("gate on the two point values within tolerance, plus the existing "
                   "judge-emission guard on every signal; intervals reported, not gated "
                   "(16.11a's resampling seed is unrecorded)")
    return out


# ---- pre-declared readings of Addendum 16.22, applied mechanically

def _sign_reading(e: dict, above: str, below: str, zero: str) -> str:
    if e is None or e.get("delta") is None:
        return "not computable"
    if e["ci_excludes_zero"]:
        return above if e["delta"] > 0 else below
    return zero


def _frontier(points: list[dict]) -> list[dict]:
    """Pareto frontier on (cost low, accuracy high): a point survives when no
    other point has accuracy >= and cost <= with one strict."""
    keep = []
    for p in points:
        if p["usd"] is None or p["acc"] is None:
            continue
        dominated = any(q is not p and q["usd"] is not None and q["acc"] is not None
                        and q["acc"] >= p["acc"] and q["usd"] <= p["usd"]
                        and (q["acc"] > p["acc"] or q["usd"] < p["usd"])
                        for q in points)
        if not dominated:
            keep.append(p)
    return sorted(keep, key=lambda p: (p["usd"], -p["acc"]))


def readings_16_22(rv: dict, cost: Optional[dict]) -> dict:
    """The registration's pre-declared readings R1-R5 on the artefact's own
    numbers (sonnet judge primary; the haiku replicate reported alongside)."""
    out: dict = {"note": "applied mechanically on the sonnet slot (the judge that beats the "
                         "base); the haiku replicate is reported, expected non-positive"}
    for slot in ("sonnet", "haiku"):
        blk = rv.get(slot)
        if blk is None:
            out[slot] = None
            continue
        s = blk["signals"]
        r1 = s["counter"]["routed_minus_base"]
        r2 = s["counter"]["routed_minus_judge_everywhere"]
        r3 = blk["pairwise"]["disagree_minus_counter"]
        r4 = blk["pairwise"]["union_minus_disagree"]
        out[slot] = {
            "R1_counter_routed_minus_base": {
                "delta": r1["delta"], "lo": r1["lo"], "hi": r1["hi"],
                "ci_excludes_zero": r1["ci_excludes_zero"],
                "reading": _sign_reading(
                    r1, "CERTIFIED: routing gain over the solo model on the registered counter, "
                        "with this interval",
                    "routing on the counter LOSES to the base with CI excluding 0; certification "
                    "stands on the composite flag only (16.11a)",
                    "CI includes 0: certification stands on the composite flag only (16.11a); the "
                    "counter's routing claim stays scoped to the collective's own verdicts (16.15.1)")},
            "R2_counter_routed_minus_judge_everywhere": {
                "delta": r2["delta"], "lo": r2["lo"], "hi": r2["hi"],
                "ci_excludes_zero": r2["ci_excludes_zero"],
                "reading": _sign_reading(
                    r2, "routing at or above the judge everywhere with CI excluding 0: strengthens "
                        "the abstract's routing sentence",
                    "below the judge everywhere with CI excluding 0 (expected, 16.11a): the "
                    "cost-efficiency framing stands",
                    "CI includes 0: routing not distinguishable from the judge everywhere")},
            "R3_disagree_routed_minus_counter_routed": {
                "delta": r3["delta"], "lo": r3["lo"], "hi": r3["hi"],
                "ci_excludes_zero": r3["ci_excludes_zero"],
                "reading": _sign_reading(
                    r3, "in favour of disagreement: the solo's own disagreement routes the solo's "
                        "errors at least as well as the counter at 3 calls against 17; the collective "
                        "sensor is scoped to the collective's own errors and dominated on cost for solo errors",
                    "in favour of the counter: the counter carries information the samples do not",
                    "CI includes 0: disagreement routes the solo's errors at least as well as the "
                    "counter at 3 calls against 17; the collective sensor is scoped to the collective's "
                    "own errors and dominated on cost for solo errors")},
            "R4_union_minus_disagree": {
                "delta": r4["delta"], "lo": r4["lo"], "hi": r4["hi"],
                "ci_excludes_zero": r4["ci_excludes_zero"],
                "reading": _sign_reading(
                    r4, "the counter adds to disagreement (union above disagreement, CI excluding 0)",
                    "the counter subtracts from disagreement (CI excluding 0)",
                    "CI includes 0: no claim that the counter adds to disagreement")},
        }
        # R5: accuracy-cost frontier from measured means, this judge, systems
        # whose accuracy the block measures, priced as run (k=4 collective)
        pts: list[dict] = []
        if cost is not None:
            sy = cost["systems"]
            pts.append({"system": "base_alone", "acc": blk["acc_base"], "usd": sy["base_alone"]["usd"]})
            pts.append({"system": f"judge_everywhere/{slot}", "acc": blk["acc_judge_everywhere"],
                        "usd": sy["judge_everywhere"].get(slot, {}).get("usd")})
            for sig in SOLO_SIGNALS:
                pts.append({"system": f"solo_signal_router/{sig}/{slot}", "acc": s[sig]["acc_routed"],
                            "usd": sy["solo_signal_router"].get(sig, {}).get(slot, {}).get("usd")})
            k4 = [k for k in sy if k.endswith("_sensor_over_base") and k.startswith("k")]
            for k in k4:
                for sig in COLLECTIVE_SIGNALS:
                    pts.append({"system": f"{k}/{sig}/{slot}", "acc": s[sig]["acc_routed"],
                                "usd": sy[k].get(sig, {}).get(slot, {}).get("usd")})
            for k in [k for k in sy if k.startswith("union_router_k") and k.endswith("_over_base")]:
                pts.append({"system": f"{k}/{slot}", "acc": s["union"]["acc_routed"],
                            "usd": sy[k].get(slot, {}).get("usd")})
        k_run = cost["k_sensor"] if cost is not None else REGISTERED_K
        out[slot]["R5_accuracy_cost_frontier"] = {
            "points": pts, "frontier": _frontier(pts),
            "note": f"descriptive; accuracy from this block (k={k_run} majority flags), cost as run "
                    f"(the k={k_run} collective); k=1 costs are projections in cost_per_decision"}
        if slot == "haiku":       # the replicate is reported, never read as a verdict
            out[slot]["replicate"] = True
            for key in ("R1_counter_routed_minus_base", "R2_counter_routed_minus_judge_everywhere",
                        "R3_disagree_routed_minus_counter_routed", "R4_union_minus_disagree"):
                out[slot][key]["reading"] = ("haiku replicate, reported only (the verdicts are the "
                                             "sonnet slot's): " + out[slot][key]["reading"])
    return out


# ---- cost per decision (cached records, runner-issued names, never generated)

#: The seven stages of one deliberation (run_crowdgold_deliberation.run_deliberation):
#: (round name, who, token cap at the registered defaults) -> 17 call records.
DELIB_ROUNDS = (("r0", "seat", 2560), ("r1", "seat", 2560), ("r2", "seat", 2560),
                ("synthesis", "moderator", 1024), ("r3_label", "seat", 1024),
                ("integration", "moderator", 1024), ("r4_vote", "seat", 512))


def deliberation_record_paths(model: str, arm: str, item: str, idx: int, *,
                              scaffold: str = "narrative_cot",
                              records_dir: Path = OUT_DIR) -> list[Path]:
    """The 17 cached call records of one debate, named by the runner's own
    call_cache_path (re-rooted to `records_dir`). Read-only: a missing record
    is reported as missing, never generated."""
    mod = rcd.moderator_role_id(model)
    seats = [r.role_id for r in rcd.ROLES]
    out = []
    for rd, who, cap in DELIB_ROUNDS:
        for role in (seats if who == "seat" else [mod]):
            out.append(records_dir / rcd.call_cache_path(
                model, scaffold, arm, item, idx, rd, role, cap).name)
    return out


def single_agent_record_paths(model: str, arm: str, item: str, *, n: int = 3,
                              records_dir: Path = OUT_DIR) -> list[Path]:
    """The n cached standard-scaffold single-agent records of one cell, named
    by run_crowdgold_aita.gen_cache_path (re-rooted). Read-only."""
    return [records_dir / gen_cache_path(model, arm, item, i).name for i in range(n)]


def _record_tokens(path: Path) -> Optional[tuple[int, int]]:
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return (int(rec.get("prompt_tokens") or 0), int(rec.get("completion_tokens") or 0))


def usd(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    """List price of one bundle of tokens; None when the model is unpriced
    (never a silent $0 -- the landmine PRICES itself warns about)."""
    if model not in rcd.PRICES:
        return None
    pin, pout = rcd.PRICES[model]
    return prompt_tokens * pin / 1e6 + completion_tokens * pout / 1e6


def mean_bundle_cost(cells: Sequence[dict], paths_fn: Callable[[dict], list[Path]],
                     model: str, label: str) -> dict:
    """Mean $ per cell of a bundle of records (all records of the bundle
    must be present for a cell to be priced; missing cells are counted)."""
    costs: list[float] = []
    pts: list[int] = []
    cts: list[int] = []
    missing = 0
    n_calls = None
    for c in cells:
        paths = paths_fn(c)
        n_calls = len(paths) if n_calls is None else n_calls
        ts = [_record_tokens(p) for p in paths]
        if any(t is None for t in ts):
            missing += 1
            continue
        pt = sum(t[0] for t in ts)
        ct = sum(t[1] for t in ts)
        pts.append(pt)
        cts.append(ct)
        u = usd(model, pt, ct)
        if u is not None:
            costs.append(u)
    n = len(pts)
    priced = model in rcd.PRICES
    return {"label": label, "model": model, "n_calls_per_cell": n_calls,
            "priced": priced, "n_cells_priced": n if priced else 0,
            "n_cells_missing": missing,
            "mean_usd": (sum(costs) / len(costs)) if costs else None,
            "mean_prompt_tokens": (sum(pts) / n) if n else None,
            "mean_completion_tokens": (sum(cts) / n) if n else None}


def cost_per_decision(cells: Sequence[dict], coverage: dict, *,
                      collective_model: str = GROK,
                      collective_scaffold: str = "narrative_cot",
                      base: tuple[str, str] = GROK_SOLO,
                      judges: Optional[dict] = None,
                      k_sensor: int = 4,
                      coverage_k1_sample0: Optional[dict] = None,
                      records_dir: Path = OUT_DIR) -> dict:
    """16.22 cost table, $ per decision, MEANS ONLY, from the cached records'
    own prompt/completion token counts priced with rcd.PRICES.

    coverage: signal -> share of cells routed (the k=`k_sensor` majority
    flags of the accuracy block, identical for every judge).
    coverage_k1_sample0: collective signal -> fire rate of the sample-0
    debate alone (the coverage a k=1 deployment would actually see).
    judges: slot -> (model, scaffold); default haiku / sonnet standard.

    Systems (formulas in the table):
      base_alone                       3 base calls
      judge_everywhere                 3 judge calls
      solo_signal_router               3 base + cov x 3 judge   (any_eshnah, disagree, disagree_raw)
      collective_k1_sensor             17 + cov x 3 judge       (counter, composite; the collective's
                                       own verdict when unrouted, as in 16.15.1)
      collective_k1_sensor_over_base   3 base + 17 + cov x 3 judge (the routed-over-solo system
                                       whose accuracy the 16.22 block certifies)
      union_router                     3 base + 17 + cov x 3 judge (counter OR disagree)
      k4_sensor                        68 (the four samples as run) + cov x 3 judge
      k4_sensor_over_base              3 base + 68 + cov x 3 judge
    """
    judges = {"haiku": HAIKU_JUDGE, "sonnet": SONNET_JUDGE} if judges is None else judges
    base_model = base[0]
    unit: dict = {}
    unit["base_3_calls"] = mean_bundle_cost(
        cells, lambda c: single_agent_record_paths(base_model, c["arm"], c["item"],
                                                    records_dir=records_dir),
        base_model, f"{base_model} {base[1]} x3")
    for slot, (jm, js) in judges.items():
        unit[f"judge_{slot}_3_calls"] = mean_bundle_cost(
            cells, lambda c, jm=jm: single_agent_record_paths(jm, c["arm"], c["item"],
                                                              records_dir=records_dir),
            jm, f"{jm} {js} x3")
    unit["collective_k1_17_calls"] = mean_bundle_cost(
        cells, lambda c: deliberation_record_paths(collective_model, c["arm"], c["item"], 0,
                                                   scaffold=collective_scaffold,
                                                   records_dir=records_dir),
        collective_model, f"{collective_model} {collective_scaffold} deliberation, sample 0 (17 calls)")
    unit[f"collective_k{k_sensor}_{17 * k_sensor}_calls"] = mean_bundle_cost(
        cells, lambda c: [p for i in range(k_sensor)
                          for p in deliberation_record_paths(collective_model, c["arm"], c["item"], i,
                                                             scaffold=collective_scaffold,
                                                             records_dir=records_dir)],
        collective_model, f"{collective_model} {collective_scaffold} deliberation, samples 0-{k_sensor - 1} ({17 * k_sensor} calls)")
    u_base = unit["base_3_calls"]["mean_usd"]
    u_k1 = unit["collective_k1_17_calls"]["mean_usd"]
    u_k4 = unit[f"collective_k{k_sensor}_{17 * k_sensor}_calls"]["mean_usd"]

    def add(*parts):
        return None if any(p is None for p in parts) else sum(parts)

    def mul(cov, u):
        return None if cov is None or u is None else cov * u

    systems: dict = {
        "base_alone": {"formula": "3 base calls", "coverage": None, "usd": u_base},
        "judge_everywhere": {}, "solo_signal_router": {}, "collective_k1_sensor": {},
        "collective_k1_sensor_at_sample0_fire_rate": {},
        "collective_k1_sensor_over_base": {}, "union_router": {},
        f"k{k_sensor}_sensor": {}, f"k{k_sensor}_sensor_over_base": {},
    }
    for slot in judges:
        u_j = unit[f"judge_{slot}_3_calls"]["mean_usd"]
        systems["judge_everywhere"][slot] = {"formula": "3 judge calls", "coverage": 1.0, "usd": u_j}
        for s in SOLO_SIGNALS:
            cov = coverage.get(s)
            systems["solo_signal_router"].setdefault(s, {})[slot] = {
                "formula": "3 base + cov x 3 judge", "coverage": cov,
                "usd": add(u_base, mul(cov, u_j))}
        for s in COLLECTIVE_SIGNALS:
            cov = coverage.get(s)
            systems["collective_k1_sensor"].setdefault(s, {})[slot] = {
                "formula": "17 + cov x 3 judge (cov = k=4 majority flag, as certified)",
                "coverage": cov, "usd": add(u_k1, mul(cov, u_j))}
            cov1 = (coverage_k1_sample0 or {}).get(s)
            systems["collective_k1_sensor_at_sample0_fire_rate"].setdefault(s, {})[slot] = {
                "formula": "17 + cov x 3 judge (cov = sample-0 debate fire rate)",
                "coverage": cov1, "usd": add(u_k1, mul(cov1, u_j))}
            systems["collective_k1_sensor_over_base"].setdefault(s, {})[slot] = {
                "formula": "3 base + 17 + cov x 3 judge", "coverage": cov,
                "usd": add(u_base, u_k1, mul(cov, u_j))}
            systems[f"k{k_sensor}_sensor"].setdefault(s, {})[slot] = {
                "formula": f"{17 * k_sensor} + cov x 3 judge", "coverage": cov,
                "usd": add(u_k4, mul(cov, u_j))}
            systems[f"k{k_sensor}_sensor_over_base"].setdefault(s, {})[slot] = {
                "formula": f"3 base + {17 * k_sensor} + cov x 3 judge", "coverage": cov,
                "usd": add(u_base, u_k4, mul(cov, u_j))}
        cov = coverage.get("union")
        systems["union_router"][slot] = {
            "formula": "3 base + 17 + cov x 3 judge (counter OR disagree)", "coverage": cov,
            "usd": add(u_base, u_k1, mul(cov, u_j))}
        systems.setdefault(f"union_router_k{k_sensor}_over_base", {})[slot] = {
            "formula": f"3 base + {17 * k_sensor} + cov x 3 judge (counter OR disagree, as run)",
            "coverage": cov, "usd": add(u_base, u_k4, mul(cov, u_j))}
    return {
        "unit": "USD per decision, mean over priced cells; list prices",
        "prices_usd_per_mtok": {m: list(rcd.PRICES[m]) for m in
                                sorted({base_model, collective_model, *[j[0] for j in judges.values()]}
                                       & set(rcd.PRICES))},
        "records_dir": str(records_dir), "k_sensor": k_sensor,
        "unit_costs": unit, "coverage": dict(coverage),
        "coverage_k1_sample0": dict(coverage_k1_sample0 or {}),
        "systems": systems,
        "note": ("means only, no intervals; token counts are the providers' own "
                 "prompt_tokens / completion_tokens as cached by the runners; "
                 f"{short_name(collective_model)}'s cached completion count excludes any hidden reasoning tokens "
                 "not reported by the provider"),
    }


def certification_summary(res: dict, parent: Path = ANALYSIS_PATH) -> dict:
    """The standalone routing_certification.json: the 16.22 blocks of the
    parent artefact plus provenance, nothing recomputed. `parent` is the
    path the parent artefact was actually written to (the registered
    router_decomposition.json by default; per-model / exploratory runs pass
    their own --out so the provenance names the real parent)."""
    rv = res.get("routed_vs_solo") or {}
    return {
        "registration": "Addendum 16.22 (routing certification; zero spend)",
        "status": rv.get("status"),
        **({"readout": res["readout"]} if res.get("readout") else {}),
        "population": rv.get("population"),
        "base": res.get("grok_solo"),
        "judges": {"haiku": res.get("judge_haiku"), "sonnet": res.get("judge_sonnet")},
        "signals": list(ROUTED_SIGNALS),
        "pairwise": [f"{a}_minus_{b}" for a, b in ROUTED_PAIRS],
        "estimator": {"draws": res.get("draws"), "seed": res.get("seed"),
                      "trim_seeds": res.get("trim_seeds"),
                      "kind": "item-clustered percentile bootstrap, paired per cell"},
        "replay_gate_16_11a": rv.get("replay_gate_16_11a"),
        "routed_vs_solo": {k: rv.get(k) for k in ("haiku", "sonnet", "sonnet_judge_codable")},
        "pre_declared_readings": rv.get("pre_declared_readings"),
        "cost_per_decision": res.get("cost_per_decision"),
        "parent_artefact": str(parent),
        "inputs": res.get("inputs"),
    }


# ---------------------------------------------------------------------------
# 16.15.3 -- the four R5 transfer readouts, both sensors
# ---------------------------------------------------------------------------

def transfer_readouts(cells: Sequence[dict], comp: dict, *, draws: int,
                      seed: int, targets: Optional[dict] = None) -> dict:
    targets = TRANSFER_TARGETS if targets is None else targets
    out: dict = {"n_cells": len(cells), "n_items": len({c["item"] for c in cells}),
                 "targets": {}}
    for label, (model, scaffold) in targets.items():
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
    na_agree = "not_applicable" in res["agreement"]
    print("\n" + "=" * 78)
    print("ADDENDUM 16.15.1 -- COUNTER LIFT WITHIN VERDICT STRATA (deliberation's own errors)")
    print("=" * 78)
    if res.get("readout"):
        r = res["readout"]
        print(f"  PER-MODEL READOUT: panel {r['model']}, solo {':'.join(r['solo'])}, judges "
              f"{', '.join(':'.join(j) for j in r['judges'])} -- EXPLORATORY, not registered")
        if r.get("deviations"):
            print(f"  deviations from the registered invocation: {'; '.join(r['deviations'])}")
    print(f"  population: {g['n_codable']} codable of {g['n_raw']} debates; "
          f"estimator item-clustered bootstrap, {res['draws']} draws, seed {res['seed']}")
    print(f"  resampling frame: {res['resampling_frame']}")
    for st, e in res["counter_lift_by_stratum"].items():
        print(f"  {st:<11} {_conc_line(e)}")
    print("  agreement with replay:" + ("  not applicable (grok-only block)" if na_agree else ""))
    for st, a in ({} if na_agree else res["agreement"]["stratum_lift"]).items():
        print(f"    {st:<11} " + "  ".join(
            f"{k} {_f(v['got'], '.3f')} vs {v['quoted']} {'OK' if v['agrees'] else 'DIFFERS'}"
            for k, v in a["fields"].items()))
    print("\n  verdict-type strata (S2 verdict):", res["stratum_counts"])
    print("  S1-synthesis-verdict-type sensitivity (router on S1 instead of S2):",
          {k: round(v, 3) if isinstance(v, float) else v
           for k, v in res["s1_verdict_type_sensitivity"].items()})

    _composed_lines(res["composed_haiku"], "COMPOSED ACCURACY, cached haiku standard majority-3 judge")
    print("  agreement with replay (composed, haiku):" + ("  not applicable" if na_agree else ""))
    for k, v in ({} if na_agree else res["agreement"]["composed_haiku"]).items():
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

    solo_lbl = " ".join(res["grok_solo"]) if res.get("readout") else "grok-solo standard"
    print(f"\n  TRANSFER WITHIN STRATA ({solo_lbl} majority-3 wrong, debate level):")
    for st, e in res["transfer_within_strata"].items():
        print(f"    {st:<11} {_conc_line(e)}")

    print("\n" + "=" * 78)
    solo_name = res["grok_solo"][0] + "-solo" if res.get("readout") else "grok-solo"
    print(f"ADDENDUM 16.15.2 -- {solo_name.upper()}'S OWN SIGNALS vs THE COLLECTIVE'S COUNTER (cells)")
    print("=" * 78)
    sb = res["solo_signals"]
    print(f"  {sb['n_cells_codable']} codable of {sb['n_cells_total']} cells, "
          f"{sb['n_items']} items; {solo_name} accuracy {_f(sb['solo_accuracy'], '.3f')}")
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
    print("  agreement with replay (ratios):" + ("  not applicable" if na_agree else ""))
    for k, v in ({} if na_agree else res["agreement"]["solo_signal_ratio"]).items():
        print(f"    {k:<13} {_f(v['got'], '.2f')}x vs {v['quoted']}x "
              f"{'OK' if v['agrees'] else 'DIFFERS'}")

    print("\n" + "=" * 78)
    print("ADDENDUM 16.15.3 -- THE FOUR R5 TRANSFER READOUTS, registered estimator (cells)"
          if len(tr["targets"]) == 4 and "grok_solo_standard" in tr["targets"] else
          f"ADDENDUM 16.15.3 -- {len(tr['targets'])} TRANSFER READOUTS (merged-comparator targets), "
          "registered estimator (cells)")
    print("=" * 78)
    print(f"  {tr['n_cells']} cells, {tr['n_items']} items")
    for label, t in tr["targets"].items():
        print(f"  {label:<20} {t['model']} {t['scaffold']}  codable cells {t['n_cells_codable']}  "
              f"solo acc {_f(t['solo_accuracy'], '.3f')}  samples/cell {t['n_samples_per_cell']}")
        for sname, e in t["by_sensor"].items():
            print(f"      {sname:<10} {_conc_line(e)}")
    n_t = len(tr["targets"])
    print(f"  readouts with CI excluding zero: counter {tr['n_targets_ci_excludes_zero_counter']}/{n_t}, "
          f"composite {tr['n_targets_ci_excludes_zero_composite']}/{n_t}")
    na_r5 = "not_applicable" in tr["r5_agreement"]
    print("  agreement with R5 quoted (composite flag; tolerance check only, cite the artefact):"
          + ("  not applicable" if na_r5 else ""))
    for label, a in ({} if na_r5 else tr["r5_agreement"]).items():
        if label in ("note", "tolerance"):
            continue
        print(f"    {label:<20} {'OK' if a['agrees'] else 'DIFFERS'}  " + "  ".join(
            f"{k} {_f(v['got'], '.3f')}/{v['quoted']}" for k, v in a["fields"].items()))

    if "not_applicable" in res["pre_declared_readings"]:
        print("\n  PRE-DECLARED READINGS: not applicable (grok-only; "
              f"{res['pre_declared_readings']['not_applicable']})")
    else:
        _readings_lines(res["pre_declared_readings"])

    print("\n" + "-" * 78)
    for k, c in g["checks"].items():
        print(f"  guard {k:<48} {_f(c['value'], '.4f')}  limit {c['limit']}  "
              f"{'ok' if c['ok'] else 'FAIL'}")
    print("  GUARDS PASSED" if g["passed"] else "  GUARD FAILED -- numbers above are NOT to be read")

    if res.get("routed_vs_solo") is not None:
        _routed_lines(res)


def _gate_lines(gate: dict) -> None:
    print("  REPLAY GATE (16.11a, composite -> sonnet; point values within "
          f"{gate['tolerance']}; intervals reported, not gated):")
    for label, p in gate.get("populations", {}).items():
        fz = p["fields"]
        print(f"    {label:<14} n={p['n_cells']:<4} "
              + "  ".join(f"{k} {_f(v['got'], '.4f')}/{v['quoted']:.4f} "
                          f"{'OK' if v['agrees'] else 'DIFFERS'}"
                          for k, v in fz.items() if k in gate["gate_fields"])
              + f"  -> {'REPRODUCES' if p['reproduces'] else 'does NOT reproduce'}")
        print(f"    {'':<14} secondary: "
              + "  ".join(f"{k} {_f(v['got'], '.4f')}/{v['quoted']:.4f} "
                          f"{'OK' if v['agrees'] else 'DIFFERS'}"
                          for k, v in fz.items() if k not in gate["gate_fields"]))
        ai, ri = p["artefact_interval"], p["replay_interval"]
        print(f"    {'':<14} interval: artefact [{_f(ai[0], '+.4f')}, {_f(ai[1], '+.4f')}] "
              f"vs 16.11a [{ri[0]:+.4f}, {ri[1]:+.4f}] (not gated)")
    for k, v in gate.get("secondary", {}).items():
        print(f"    secondary {k}: " + "  ".join(
            f"{kk} {_f(vv['got'], '.4f')}/{vv['quoted']:.4f} {'OK' if vv['agrees'] else 'DIFFERS'}"
            for kk, vv in v.items()))
    for slot, e in gate.get("emission", {}).items():
        print(f"    judge emission on fired cells [{slot}]: min over signals "
              f"{_f(e['min'], '.3f')}  limit {gate['emission_limit']}  {'ok' if e['ok'] else 'FAIL'}")


def _readings_16_22_lines(pr: dict) -> None:
    print("\n  PRE-DECLARED READINGS 16.22 (applied mechanically):")
    for slot in ("sonnet", "haiku"):
        r = pr.get(slot)
        if r is None:
            print(f"    [{slot}] not computable")
            continue
        for key in ("R1_counter_routed_minus_base", "R2_counter_routed_minus_judge_everywhere",
                    "R3_disagree_routed_minus_counter_routed", "R4_union_minus_disagree"):
            e = r[key]
            print(f"    [{slot}] {key:<42} {_f(e['delta'], '+.4f')} [{_f(e['lo'], '+.4f')}, "
                  f"{_f(e['hi'], '+.4f')}] -> {e['reading']}")
        fr = r["R5_accuracy_cost_frontier"]["frontier"]
        # the k as run is in the block's own note ("... (the k=N collective) ...")
        k_note = r["R5_accuracy_cost_frontier"].get("note", "")
        k_run = k_note.split("(the k=", 1)[1].split(" ", 1)[0] if "(the k=" in k_note else str(REGISTERED_K)
        print(f"    [{slot}] R5 accuracy-cost frontier (cost as run, k={k_run} collective): "
              + ("; ".join(f"{p['system']} acc {_f(p['acc'], '.4f')} ${_f(p['usd'], '.6f')}" for p in fr)
                 if fr else "not computable (no priced records)"))


def _routed_block_lines(blk: dict, title: str) -> None:
    print(f"\n  {title}: {blk['n_cells']} cells ({blk['n_cells_total']} total), "
          f"{blk['n_items']} items; judge codable on {blk['n_judge_codable']}")
    ev = blk["judge_everywhere_minus_base"]
    print(f"    base {_f(blk['acc_base'], '.4f')}   judge everywhere {_f(blk['acc_judge_everywhere'], '.4f')}   "
          f"everywhere - base {_f(ev['delta'], '+.4f')} [{_f(ev['lo'], '+.4f')}, {_f(ev['hi'], '+.4f')}] "
          f"{'excludes 0' if ev['ci_excludes_zero'] else 'includes 0'}")
    print(f"    {'signal':<13}{'fires':>6}{'cov':>7}{'routed':>8}{'base':>8}  "
          f"{'routed-base [95% CI]':<30}{'':<11}{'routed-everywhere [95% CI]':<30}"
          f"{'emit':>6}  {'random@cov mean/best':<22}")
    for name, s in blk["signals"].items():
        vb, ve, rnd = s["routed_minus_base"], s["routed_minus_judge_everywhere"], s["random_at_coverage"]
        print(f"    {name:<13}{s['n_fired']:>6}{_f(s['coverage'], '.3f'):>7}"
              f"{_f(s['acc_routed'], '.4f'):>8}{_f(s['acc_base'], '.4f'):>8}  "
              f"{_f(vb['delta'], '+.4f') + ' [' + _f(vb['lo'], '+.4f') + ', ' + _f(vb['hi'], '+.4f') + ']':<30}"
              f"{('excl 0' if vb['ci_excludes_zero'] else 'incl 0'):<11}"
              f"{_f(ve['delta'], '+.4f') + ' [' + _f(ve['lo'], '+.4f') + ', ' + _f(ve['hi'], '+.4f') + ']':<30}"
              f"{_f(s['judge_emitted_on_fired'], '.2f'):>6}  "
              f"{_f(rnd['mean'], '.4f')} / {_f(rnd['best'], '.4f')}")
    print("    pairwise routed - routed (item-clustered):")
    for k, v in blk["pairwise"].items():
        print(f"      {k:<26} {_f(v['acc_a'], '.4f')} - {_f(v['acc_b'], '.4f')} = "
              f"{_f(v['delta'], '+.4f')} [{_f(v['lo'], '+.4f')}, {_f(v['hi'], '+.4f')}]  "
              f"{'excludes 0' if v['ci_excludes_zero'] else 'includes 0'}")


def _cost_lines(cost: dict) -> None:
    print("\n  COST PER DECISION (USD, means only; cached token counts x list prices "
          f"{cost['prices_usd_per_mtok']})")
    for k, u in cost["unit_costs"].items():
        print(f"    unit {k:<28} {_f(u['mean_usd'], '.6f')}  "
              f"({u['label']}; priced {u['n_cells_priced']}, missing {u['n_cells_missing']}; "
              f"mean tokens {_f(u['mean_prompt_tokens'], '.0f')} in / "
              f"{_f(u['mean_completion_tokens'], '.0f')} out)")
    sy = cost["systems"]
    slots = list(sy["judge_everywhere"])
    print(f"    {'system':<44}{'signal':<14}{'cov':>7}" + "".join(f"{s:>12}" for s in slots)
          + "   formula")
    print(f"    {'base_alone':<44}{'--':<14}{'--':>7}"
          + "".join(f"{_f(sy['base_alone']['usd'], '.6f'):>12}" for _ in slots)
          + f"   {sy['base_alone']['formula']}")
    for name, block in sy.items():
        if name == "base_alone":
            continue
        if slots and slots[0] in block:          # judge-keyed directly
            print(f"    {name:<44}{'--':<14}{_f(block[slots[0]]['coverage'], '.3f'):>7}"
                  + "".join(f"{_f(block[s]['usd'], '.6f'):>12}" for s in slots)
                  + f"   {block[slots[0]]['formula']}")
            continue
        for sig, per in block.items():
            print(f"    {name:<44}{sig:<14}{_f(per[slots[0]]['coverage'], '.3f'):>7}"
                  + "".join(f"{_f(per[s]['usd'], '.6f'):>12}" for s in slots)
                  + f"   {per[slots[0]]['formula']}")


def _routed_lines(res: dict) -> None:
    """ADDENDUM 16.22. The replay gate is printed and evaluated BEFORE any
    new number; on failure the literal GUARD FAILED is printed, the block is
    marked UNREAD and its numbers are withheld (they remain in the artefact
    under status UNREAD)."""
    rv = res["routed_vs_solo"]
    gate = rv["replay_gate_16_11a"]
    print("\n" + "=" * 78)
    print("ADDENDUM 16.22 -- ROUTING CERTIFICATION: routed vs the solo base, per judge (cells)")
    print("=" * 78)
    print(f"  population: {rv['population']}")
    st = gate.get("status")
    if st == "not_applicable":
        print(f"  replay gate: not applicable ({gate.get('reason')}) -- EXPLORATORY block, "
              "not registered; numbers below are not certified")
    elif st == "not_computable":
        print(f"  replay gate: NOT COMPUTABLE ({gate.get('reason')}) -- ADDENDUM 16.22 UNREAD; "
              "numbers withheld")
        return
    else:
        _gate_lines(gate)
        if not gate["passed"]:
            print(f"  GUARD FAILED -- replay gate: {gate.get('reason')}; "
                  "ADDENDUM 16.22 UNREAD, numbers withheld (in the artefact under status UNREAD)")
            return
        print(f"  gate PASSED (reproduces on: {', '.join(gate['reproduces'])}; "
              "judge emission ok) -> block READ")
    for slot in ("haiku", "sonnet", "sonnet_judge_codable"):
        blk = rv.get(slot)
        if blk is None:
            print(f"\n  JUDGE [{slot}]: NOT COMPUTABLE")
            continue
        _routed_block_lines(blk, f"JUDGE [{slot}] {blk['judge']}")
    if res.get("cost_per_decision"):
        _cost_lines(res["cost_per_decision"])
    if rv.get("pre_declared_readings"):
        _readings_16_22_lines(rv["pre_declared_readings"])


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
            trim_seeds: int, model: str = GROK, solo: tuple[str, str] = GROK_SOLO,
            judges: Sequence[tuple[str, str]] = (HAIKU_JUDGE, SONNET_JUDGE),
            targets: Optional[dict] = None, records_dir: Path = OUT_DIR,
            replay: dict = REPLAY_16_11A, k_sensor: Optional[int] = None,
            registered: Optional[bool] = None,
            deviations: Sequence[str] = ()) -> tuple[dict, dict]:
    """Defaults are the registered grok readout. `judges[0]` is the primary
    judge (the "haiku" slot, read from `comp`); `judges[1]`, if given, the
    coverage-restricted judge (the "sonnet" slot, read from `sonnet_comp`).
    `targets` None -> the registered R5 four. The grok-only agreement, R5 and
    pre-declared-reading blocks are not_applicable when the readout is not
    the registered one. `records_dir` / `replay` (16.22): where the cached
    call records are read for the cost table, and the replay target of the
    16.22 gate (the gate is evaluated only on the registered grok / sonnet
    configuration). `k_sensor`: samples per cell the cost table prices for
    the k-sensor bundle; None -> derived from the panel (panel_k: 4 on the
    registered grok panel, 2 on the nano community). `registered`: None ->
    derived from (model, solo, judges) alone -- the readout-defining
    configuration; main() passes the stricter verdict of
    registered_deviations() (paths, estimator, targets, records dir), with
    the `deviations` it found, so that ANY non-default invocation carries the
    `readout` marker and never takes the registered branch."""
    primary = tuple(judges[0])
    secondary = tuple(judges[1]) if len(judges) > 1 else None
    config_registered = (model == GROK and tuple(solo) == GROK_SOLO
                         and tuple(tuple(j) for j in judges) == (HAIKU_JUDGE, SONNET_JUDGE))
    registered = config_registered if registered is None else bool(registered)
    if k_sensor is None:
        k_sensor = panel_k(debates)
    haiku_rule = make_rule_vendor(comp, *primary)
    solo_rule = make_rule_vendor(comp, *solo)

    res: dict = {"registration": "Addendum 16.15.1 / 16.15.2", "draws": draws,
                 "seed": seed, "trim_seeds": trim_seeds,
                 "population": f"{short_name(model)} panel, codable debates, code_response collapse",
                 "verdict_type_source": "S2 final verdict (`verdict` column)",
                 "judge_haiku": list(primary),
                 "judge_sonnet": list(secondary) if secondary else None,
                 "grok_solo": list(solo)}
    if not registered:
        dev = list(deviations) or ([] if config_registered else [
            f"model {model}" if model != GROK else "",
            f"solo {solo[0]}:{solo[1]}" if tuple(solo) != GROK_SOLO else "",
            f"judges {','.join(':'.join(j) for j in judges)}"
            if tuple(tuple(j) for j in judges) != (HAIKU_JUDGE, SONNET_JUDGE) else ""])
        res["readout"] = {"model": model, "registered": False, "solo": list(solo),
                          "judges": [list(j) for j in judges],
                          "deviations": [d for d in dev if d],
                          "note": "exploratory per-model readout; the registered "
                                  "readout of Addendum 16.15 is the grok panel"}
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
                                           judge_label=f"{primary[0]} {primary[1]} majority-3 (cached)")
    # composed accuracy, sonnet judge on covered debates
    cov = (sonnet_covered(debates, sonnet_comp, secondary)
           if sonnet_comp is not None and secondary is not None else [])
    if cov:
        sonnet_rule = make_rule_vendor(sonnet_comp, *secondary)
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
                                    judge_label=(f"{secondary[0]} {secondary[1]} majority-3 (16.11a cache)"
                                                 if secondary == SONNET_JUDGE else
                                                 f"{secondary[0]} {secondary[1]} majority-3 (cached, covered debates only)")),
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
    cells = build_cells(debates, comp, solo)
    res["solo_signals"] = solo_signal_block(cells, solo_rule, haiku_rule,
                                            draws=draws, seed=seed,
                                            trim_seeds=trim_seeds)

    # 16.15.3 -- transfer readouts
    tr = transfer_readouts(cells, comp, draws=draws, seed=seed, targets=targets)
    tr["registration"] = "Addendum 16.15.3"
    tr["draws"], tr["seed"] = draws, seed
    tr["cell_flag_rule"] = "fires when strictly more than half of the cell's codable debate samples fire"
    if not registered:
        tr["readout"] = dict(res["readout"])
    na = {"not_applicable": (f"registered for the grok panel (Addendum 16.15); "
                             f"this is a {model} readout, exploratory, not registered"
                             + (f" (deviations: {'; '.join(res['readout']['deviations'])})"
                                if res.get("readout", {}).get("deviations") else ""))}
    if registered and targets is None:
        tr["r5_quoted"] = R5_QUOTED
        tr["r5_agreement"] = r5_agreement(tr)
    else:
        tr["r5_quoted"] = tr["r5_agreement"] = (
            na if not registered else {"not_applicable": "custom --transfer-targets; "
                                       "R5 quoted the registered four targets"})

    if registered:
        res["agreement"] = replay_agreement(res)
        res["pre_declared_readings"] = pre_declared_readings(res)
    else:
        res["agreement"] = res["pre_declared_readings"] = na
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

    # 16.22 -- routing certification on the cells (new keys only; every key
    # above is untouched). Sonnet slot: the coverage-restricted judge; its
    # judge-codable population (418 on the registered data) is the 16.11a one.
    res["routed_vs_solo"] = routed_vs_solo_block(
        cells, debates, solo_rule, haiku_rule,
        sonnet_rule if cov else None, primary, secondary, model=model, solo=solo,
        draws=draws, seed=seed, trim_seeds=trim_seeds, replay=replay,
        registered=registered, deviations=res.get("readout", {}).get("deviations", []))
    res["cost_per_decision"] = cost_per_decision(
        cells, res["routed_vs_solo"]["coverage"],
        collective_model=model, base=solo,
        judges={"haiku": primary, **({"sonnet": secondary} if secondary else {})},
        k_sensor=k_sensor,
        coverage_k1_sample0=res["routed_vs_solo"]["coverage_k1_sample0"],
        records_dir=records_dir)
    res["routed_vs_solo"]["pre_declared_readings"] = readings_16_22(
        res["routed_vs_solo"], res["cost_per_decision"])
    return res, tr


def routed_vs_solo_block(cells: Sequence[dict], debates: Sequence[dict],
                         base_rule, primary_rule, secondary_rule,
                         primary: tuple[str, str], secondary: Optional[tuple[str, str]],
                         *, model: str, solo: tuple[str, str], draws: int, seed: int,
                         trim_seeds: int, replay: dict = REPLAY_16_11A,
                         registered: bool = True, deviations: Sequence[str] = ()) -> dict:
    """The 16.22 parent block: both judge slots on the full cell population,
    the sonnet slot again on its judge-codable population, the replay gate,
    and the coverages the cost table prices. Status READ only when the gate
    passed AND the invocation is the registered one; a passed gate on a
    non-registered invocation (e.g. a grok run with --n-boot 200) reads
    "exploratory (gate passed; not the registered invocation)" and is never
    certified; UNREAD when the gate failed or could not be evaluated;
    not_applicable (exploratory) off the registered grok / sonnet
    configuration."""
    label = lambda j, extra="": f"{j[0]} {j[1]} majority-3 (cached{extra})"
    pop = [c for c in cells if solo_ok(c, base_rule) is not None]
    out: dict = {
        "registration": "Addendum 16.22",
        "population": (f"{len(pop)} (arm, item) cells on which the base majority-of-3 is "
                       f"codable, of {len(cells)}; base = {solo[0]} {solo[1]} majority-3; "
                       "a cell fires when strictly more than half of its codable debate "
                       "samples fire"),
        "base": list(solo), "n_cells": len(pop), "n_cells_total": len(cells),
        "n_items": len({c["item"] for c in pop}),
        "signals": list(ROUTED_SIGNALS), "pairs": [f"{a}_minus_{b}" for a, b in ROUTED_PAIRS],
        "haiku": routed_vs_base_block(cells, base_rule, primary_rule, label(primary),
                                      draws=draws, seed=seed, trim_seeds=trim_seeds),
        "sonnet": None, "sonnet_judge_codable": None,
    }
    if secondary_rule is not None and secondary is not None:
        out["sonnet"] = routed_vs_base_block(
            cells, base_rule, secondary_rule, label(secondary, ", 16.11a cache" if secondary == SONNET_JUDGE else ""),
            draws=draws, seed=seed, trim_seeds=trim_seeds)
        jc = [c for c in cells if solo_ok(c, secondary_rule) is not None]
        out["sonnet_judge_codable"] = routed_vs_base_block(
            jc, base_rule, secondary_rule, label(secondary, ", judge-codable cells only"),
            draws=draws, seed=seed, trim_seeds=trim_seeds)
        out["sonnet_judge_codable"]["n_cells_total"] = len(cells)
        out["sonnet_judge_codable"]["n_dropped_judge_noncodable"] = len(cells) - len(jc)
    # coverages for the cost table (identical for every judge)
    out["coverage"] = {k: v["coverage"] for k, v in out["haiku"]["signals"].items()}
    s0 = [d for d in debates if d["sample_idx"] == 0]
    out["coverage_k1_sample0"] = {
        "counter": (sum(1 for d in s0 if counter_router(d)) / len(s0)) if s0 else None,
        "composite": (sum(1 for d in s0 if composite_flagged(d)) / len(s0)) if s0 else None,
        "n_sample0_debates": len(s0)}
    # the gate: registered grok panel, grok standard base, sonnet standard judge
    registered_cfg = (model == GROK and tuple(solo) == GROK_SOLO and secondary == SONNET_JUDGE)
    if not registered_cfg:
        out["replay_gate_16_11a"] = {
            "status": "not_applicable", "passed": False, "reproduces": [],
            "reason": (f"16.11a replay is defined for the grok panel with the grok standard "
                       f"base and the sonnet standard judge; this is {model} / "
                       f"{solo[0]}:{solo[1]} / {secondary[0] + ':' + secondary[1] if secondary else 'no secondary judge'}")}
    else:
        secondary_lines = None
        if out["sonnet"] is not None:
            u16 = lambda c: bool(c["composite_fired"] or c["solo_disagree"])
            secondary_lines = {}
            for lab, cs in (("full", cells), ("judge_codable",
                                              [c for c in cells if solo_ok(c, secondary_rule) is not None])):
                p = _point_routed(cs, base_rule, secondary_rule, u16)
                secondary_lines[f"union_composite_or_disagree_{lab}"] = {
                    "acc_routed": _agree(p["acc_routed"], replay["routed_union_composite_or_disagree"], GATE_TOL_16_11A),
                    "delta_vs_base": _agree(p["delta"], replay["delta_union_vs_base"], GATE_TOL_16_11A),
                    "coverage": _agree(p["coverage"], replay["coverage_union"], GATE_TOL_16_11A),
                }
        out["replay_gate_16_11a"] = replay_gate_16_11a(
            out["sonnet"], out["sonnet_judge_codable"], replay=replay, secondary=secondary_lines,
            emission_blocks={k: out[k] for k in ("haiku", "sonnet", "sonnet_judge_codable")})
    g = out["replay_gate_16_11a"]
    out["status"] = ("READ" if g["status"] == "passed" else
                     "not_applicable (exploratory)" if g["status"] == "not_applicable" else "UNREAD")
    if not registered:
        if g["status"] == "passed":
            out["status"] = "exploratory (gate passed; not the registered invocation)"
        out["readout"] = {"registered": False, "deviations": list(deviations),
                          "note": "not the registered invocation; nothing here is certified"}
    return out


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


def _synthetic_cells() -> tuple[list[dict], dict]:
    """16.22 hand-built cells with KNOWN deltas. 20 cells (10 items x 2
    arms, gold YTA), cell j = (as_asker if j < 10 else third_person, i{j%10}).
    Base wrong on {0..5} (14/20). Judge (model "j") right everywhere except
    wrong on {4, 10, 11} and a majority TIE (non-codable) on {5, 12}.
      counter      fires {0,1,2,10,12}     -> routed 16/20 (+3 repaired, -1 on 10, 12 falls back)
      disagree     fires {0,1,2,3,4,11}    -> 17/20 (+4, 0 on 4, -1 on 11)
      union        = counter OR disagree   -> 16/20
      composite    fires {0,5}             -> 15/20 (+1; 5 falls back to the wrong base)
      any_eshnah   never fires             -> 14/20
      disagree_raw fires {0..5, 11}        -> 17/20
      judge everywhere                     -> 16/20 (3 wrong; 5 falls back wrong; 12 right)
    """
    cells, comp = [], {}
    wrong_base = {0, 1, 2, 3, 4, 5}
    wrong_judge = {4, 10, 11}
    tie_judge = {5, 12}
    fires = {"counter": {0, 1, 2, 10, 12}, "disagree": {0, 1, 2, 3, 4, 11},
             "composite": {0, 5}, "any_eshnah": set(), "disagree_raw": {0, 1, 2, 3, 4, 5, 11}}
    for j in range(20):
        arm = "as_asker" if j < 10 else "third_person"
        item = f"i{j % 10}"
        cells.append({"item": item, "arm": arm, "sample_idx": 0, "gold": "YTA", "k": 4,
                      "solo_n_samples": 3,
                      "counter_fired": j in fires["counter"],
                      "composite_fired": j in fires["composite"],
                      "solo_any_eshnah": j in fires["any_eshnah"],
                      "solo_disagree": j in fires["disagree"],
                      "solo_disagree_raw": j in fires["disagree_raw"]})
        comp[("b", "standard", arm, item)] = ["NTA"] * 3 if j in wrong_base else ["YTA"] * 3
        comp[("j", "standard", arm, item)] = (["YTA", "NTA"] if j in tie_judge else
                                              ["NTA"] * 3 if j in wrong_judge else ["YTA"] * 3)
    return cells, comp


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

    # ---- per-model wiring ---------------------------------------------------
    check("model:scaffold parsing (scaffold defaults to standard)",
          parse_model_scaffold("gpt-5.4-nano:standard") == ("gpt-5.4-nano", "standard")
          and parse_model_scaffold(" claude-haiku-4-5 ") == ("claude-haiku-4-5", "standard"))
    check("short names: grok / nano / haiku / sonnet, else the model id",
          short_name(GROK) == "grok" and short_name("gpt-5.4-nano") == "nano"
          and short_name("claude-haiku-4-5") == "haiku" and short_name("x-y") == "x-y")
    m1 = {("m", "standard", "as_asker", "i"): ["YTA", "YTA"]}
    m2 = {("m", "standard", "as_asker", "i"): ["NTA", "NTA", "NTA"], ("n", "standard", "as_asker", "i"): ["YTA"]}
    mm = merge_comparators([m1, m2])
    check("merge_comparators concatenates samples per key (2 + 3 -> NTA majority) and keeps other keys",
          len(mm[("m", "standard", "as_asker", "i")]) == 5
          and make_rule_vendor(mm, "m", "standard")({"arm": "as_asker", "item": "i"}) == "NTA"
          and mm[("n", "standard", "as_asker", "i")] == ["YTA"])
    check("comparator_targets lists every (model, scaffold) pair, labelled model:scaffold; restrict keeps order",
          comparator_targets(comp) == {f"{GROK}:standard": (GROK, "standard"),
                                       "claude-haiku-4-5:standard": ("claude-haiku-4-5", "standard")}
          and list(comparator_targets(comp, [("z", "s"), (GROK, "standard")])) == ["z:s", f"{GROK}:standard"])
    cells_other = build_cells(debates, comp, ("no-such-model", "standard"))
    check("build_cells reads the solo signals from the requested (model, scaffold)",
          all(c["solo_n_samples"] == 0 and not c["solo_any_eshnah"] for c in cells_other)
          and any(c["solo_n_samples"] == 3 for c in cells))
    tr_c = transfer_readouts(cells, comp, draws=50, seed=13, targets=comparator_targets(comp))
    check("transfer_readouts on comparator-derived targets: grok block identical to the registered label's",
          tr_c["targets"][f"{GROK}:standard"] == tr["targets"]["grok_solo_standard"]
          and set(tr_c["targets"]) == {f"{GROK}:standard", "claude-haiku-4-5:standard"})
    # per-model analyse on the synthetic panel (model "m"): grok-only blocks not_applicable
    res_m, tr_m = analyse(debates, raw, comp, None, draws=50, seed=13, trim_seeds=5,
                          model="m", solo=GROK_SOLO, judges=(HAIKU_JUDGE,),
                          targets=comparator_targets(comp))
    check("per-model analyse: agreement / pre-declared readings / R5 marked not_applicable",
          "not_applicable" in res_m["agreement"] and "not_applicable" in res_m["pre_declared_readings"]
          and "not_applicable" in tr_m["r5_agreement"] and "not_applicable" in tr_m["r5_quoted"])
    check("per-model analyse: readout block names the model, one judge -> no sonnet block, "
          "population string carries the model, numbers identical to the grok-labelled run",
          res_m["readout"]["model"] == "m" and res_m["composed_sonnet"] is None
          and res_m["judge_sonnet"] is None and res_m["population"].startswith("m panel")
          and res_m["counter_lift_by_stratum"] == res_syn["counter_lift_by_stratum"]
          and res_m["solo_signals"] == res_syn["solo_signals"]
          and res_m["composed_haiku"]["routers"] == res_syn["composed_haiku"]["routers"])
    check("registered analyse with custom targets keeps agreement but marks R5 not_applicable",
          "not_applicable" in analyse(debates, raw, comp, None, draws=20, seed=13, trim_seeds=3,
                                      targets=comparator_targets(comp))[1]["r5_agreement"]
          and "stratum_lift" in res_syn["agreement"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_report(res_m, tr_m)
    check("print_report handles the not_applicable blocks and names the per-model readout",
          "not applicable" in buf.getvalue() and "PER-MODEL READOUT: panel m" in buf.getvalue()
          and "GUARDS PASSED" in buf.getvalue())
    # a secondary judge with no cached rows anywhere -> no sonnet block, no crash
    res_nj, _ = analyse(debates, raw, comp, comp, draws=20, seed=13, trim_seeds=3,
                        judges=(HAIKU_JUDGE, ("no-such-judge", "standard")))
    check("secondary judge without cached rows -> composed_sonnet None (no empty-panel crash)",
          res_nj["composed_sonnet"] is None)

    # ---- 16.22: routed vs base on hand-built cells with known deltas ---------
    import tempfile
    sc, sc_comp = _synthetic_cells()
    b_rule = make_rule_vendor(sc_comp, "b", "standard")
    j_rule = make_rule_vendor(sc_comp, "j", "standard")
    rb = routed_vs_base_block(sc, b_rule, j_rule, "synthetic judge", draws=200,
                              seed=13, trim_seeds=20)
    S = rb["signals"]
    check("16.22 population: 20 cells, 10 items, base 14/20, judge codable on 18",
          rb["n_cells"] == 20 and rb["n_items"] == 10 and abs(rb["acc_base"] - 0.70) < 1e-12
          and rb["n_judge_codable"] == 18)
    check("16.22 judge everywhere = 16/20 (3 wrong, one tie falls back to a wrong base) and +0.10 over base",
          abs(rb["acc_judge_everywhere"] - 0.80) < 1e-12
          and abs(rb["judge_everywhere_minus_base"]["delta"] - 0.10) < 1e-12)
    check("16.22 counter: fires 5, routed 16/20, delta +0.10, judge emitted 4/5, one fallback",
          S["counter"]["n_fired"] == 5 and abs(S["counter"]["acc_routed"] - 0.80) < 1e-12
          and abs(S["counter"]["routed_minus_base"]["delta"] - 0.10) < 1e-12
          and abs(S["counter"]["judge_emitted_on_fired"] - 0.8) < 1e-12
          and S["counter"]["n_fallback_to_base"] == 1)
    check("16.22 disagree: fires 6, routed 17/20, delta +0.15; composite: fires 2, routed 15/20, +0.05",
          S["disagree"]["n_fired"] == 6 and abs(S["disagree"]["acc_routed"] - 0.85) < 1e-12
          and abs(S["disagree"]["routed_minus_base"]["delta"] - 0.15) < 1e-12
          and S["composite"]["n_fired"] == 2 and abs(S["composite"]["acc_routed"] - 0.75) < 1e-12
          and abs(S["composite"]["routed_minus_base"]["delta"] - 0.05) < 1e-12)
    check("16.22 union = counter OR disagree: fires 8, routed 16/20 (two judge-wrong cells cost one each)",
          S["union"]["n_fired"] == 8 and abs(S["union"]["acc_routed"] - 0.80) < 1e-12
          and all(union_cell(c) == (c["counter_fired"] or c["solo_disagree"]) for c in sc))
    check("16.22 a never-firing signal routes nothing: any_eshnah = base, delta 0, emission n/a",
          S["any_eshnah"]["n_fired"] == 0 and S["any_eshnah"]["acc_routed"] == rb["acc_base"]
          and S["any_eshnah"]["routed_minus_base"]["delta"] == 0
          and S["any_eshnah"]["judge_emitted_on_fired"] is None)
    check("16.22 routed - judge-everywhere: counter 0, disagree +0.05 (paired per cell)",
          abs(S["counter"]["routed_minus_judge_everywhere"]["delta"]) < 1e-12
          and abs(S["disagree"]["routed_minus_judge_everywhere"]["delta"] - 0.05) < 1e-12)
    P = rb["pairwise"]
    check("16.22 pairwise: disagree-counter +0.05, union-counter 0, union-disagree -0.05, composite-counter -0.05",
          abs(P["disagree_minus_counter"]["delta"] - 0.05) < 1e-12
          and abs(P["union_minus_counter"]["delta"]) < 1e-12
          and abs(P["union_minus_disagree"]["delta"] + 0.05) < 1e-12
          and abs(P["composite_minus_counter"]["delta"] + 0.05) < 1e-12)
    check("16.22 random routing fires exactly n_fired and never beats the perfect-where-codable ceiling",
          all(s["random_at_coverage"]["target_n"] == s["n_fired"] for s in S.values())
          and all(s["random_at_coverage"]["best"] <= 1.0 for s in S.values())
          and abs(S["any_eshnah"]["random_at_coverage"]["mean"] - rb["acc_base"]) < 1e-12)
    check("16.22 ci_excludes_zero mirrors the routed-minus-base interval",
          all(s["ci_excludes_zero"] == s["routed_minus_base"]["ci_excludes_zero"] for s in S.values()))
    # the gate: a replay target equal to the composite numbers passes on the
    # full population; the real 16.11a target does not match synthetic cells
    jc = [c for c in sc if solo_ok(c, j_rule) is not None]
    rb18 = routed_vs_base_block(jc, b_rule, j_rule, "synthetic judge", draws=200, seed=13, trim_seeds=20)
    syn_replay = dict(REPLAY_16_11A, routed_composite=0.75, delta_composite_vs_base=0.05,
                      base=0.70, coverage_composite=0.10, judge_everywhere=0.80)
    g_ok = replay_gate_16_11a(rb, rb18, replay=syn_replay)
    check("16.22 gate passes when the full population reproduces both gate fields and names it",
          g_ok["passed"] and g_ok["status"] == "passed" and "full" in g_ok["reproduces"])
    check("16.22 gate reports the judge-codable population separately (18 cells; composite 14/18 does not reproduce 0.75)",
          g_ok["populations"]["judge_codable"]["n_cells"] == 18
          and abs(rb18["signals"]["composite"]["acc_routed"] - 14 / 18) < 1e-12
          and not g_ok["populations"]["judge_codable"]["reproduces"])
    g_bad = replay_gate_16_11a(rb, rb18)
    check("16.22 gate FAILS against the real 16.11a target on synthetic cells",
          not g_bad["passed"] and g_bad["status"] == "failed" and g_bad["reproduces"] == [])
    check("16.22 gate is not computable without the sonnet block",
          replay_gate_16_11a(None, rb18)["status"] == "not_computable")
    check("16.22 gate tolerance is a strict window (0.0051 off on the point fails)",
          not replay_gate_16_11a(rb, rb18, replay=dict(syn_replay, routed_composite=0.75 + 0.0051))["passed"]
          and replay_gate_16_11a(rb, rb18, replay=dict(syn_replay, routed_composite=0.75 + 0.0049))["passed"])
    g_em = replay_gate_16_11a(rb, rb18, replay=syn_replay, emission_blocks={"x": rb, "y": rb18})
    check("16.22 gate applies the existing judge-emission guard: composite emits on 1 of 2 fired cells -> FAIL",
          not g_em["passed"] and g_em["status"] == "failed" and "emission" in g_em["reason"]
          and abs(g_em["emission"]["x"]["min"] - 0.5) < 1e-12 and g_em["emission"]["y"]["ok"]
          and g_em["reproduces"] == ["full"])
    check("16.22 gate passes with the emission guard when every fired cell is judged",
          replay_gate_16_11a(rb, rb18, replay=syn_replay, emission_blocks={"y": rb18})["passed"])
    # pre-declared readings, mechanically
    check("16.22 readings: sign logic (above / below / includes 0 / not computable)",
          _sign_reading({"delta": 0.1, "ci_excludes_zero": True}, "A", "B", "Z") == "A"
          and _sign_reading({"delta": -0.1, "ci_excludes_zero": True}, "A", "B", "Z") == "B"
          and _sign_reading({"delta": 0.1, "ci_excludes_zero": False}, "A", "B", "Z") == "Z"
          and _sign_reading({"delta": None, "ci_excludes_zero": False}, "A", "B", "Z") == "not computable")
    fr = _frontier([{"system": "a", "acc": 0.90, "usd": 0.001}, {"system": "b", "acc": 0.95, "usd": 0.010},
                    {"system": "c", "acc": 0.92, "usd": 0.020}, {"system": "d", "acc": 0.90, "usd": 0.002},
                    {"system": "e", "acc": None, "usd": 0.0}])
    check("16.22 readings: Pareto frontier keeps a and b, drops c (dominated by b) and d (dominated by a)",
          [p["system"] for p in fr] == ["a", "b"])
    pr_syn = readings_16_22({"sonnet": rb, "haiku": None}, None)
    check("16.22 readings: R1-R4 carry the block's own deltas (counter +0.10, R3 +0.05, R4 -0.05) and haiku None",
          abs(pr_syn["sonnet"]["R1_counter_routed_minus_base"]["delta"] - 0.10) < 1e-12
          and abs(pr_syn["sonnet"]["R3_disagree_routed_minus_counter_routed"]["delta"] - 0.05) < 1e-12
          and abs(pr_syn["sonnet"]["R4_union_minus_disagree"]["delta"] + 0.05) < 1e-12
          and pr_syn["sonnet"]["R5_accuracy_cost_frontier"]["frontier"] == []
          and pr_syn["haiku"] is None)

    # ---- 16.22: cost per decision on synthetic records with known token counts
    with tempfile.TemporaryDirectory() as td:
        rd_ = Path(td)
        two = sc[:2]
        for c in two:
            for i in range(4):
                for p in deliberation_record_paths(GROK, c["arm"], c["item"], i, records_dir=rd_):
                    p.write_text(json.dumps({"prompt_tokens": 1000, "completion_tokens": 100}))
            for m in (GROK, "claude-haiku-4-5", "claude-sonnet-4-6"):
                for p in single_agent_record_paths(m, c["arm"], c["item"], records_dir=rd_):
                    p.write_text(json.dumps({"prompt_tokens": 1000, "completion_tokens": 100}))
        paths17 = deliberation_record_paths(GROK, "as_asker", "i0", 0, records_dir=rd_)
        check("16.22 cost: 17 runner-named records per debate (9 seat rounds, 2 moderator, 3 labels, 3 votes)",
              len(paths17) == 17 and len(set(paths17)) == 17
              and sum(1 for p in paths17 if "_r0_" in p.name) == 3
              and sum(1 for p in paths17 if "_synthesis_" in p.name) == 1
              and sum(1 for p in paths17 if "_r4_vote_" in p.name) == 3
              and all(p.name == rcd.call_cache_path(GROK, "narrative_cot", "as_asker", "i0", 0, "r0",
                                                    "writer_advocate", 2560).name
                      for p in paths17[:1]))
        check("16.22 cost: single-agent record names come from run_crowdgold_aita.gen_cache_path",
              [p.name for p in single_agent_record_paths(GROK, "as_asker", "i0", records_dir=rd_)]
              == [gen_cache_path(GROK, "as_asker", "i0", i).name for i in range(3)])
        cov = {k: S[k]["coverage"] for k in ROUTED_SIGNALS}
        cost = cost_per_decision(sc, cov, coverage_k1_sample0={"counter": 0.5, "composite": 0.25},
                                 records_dir=rd_)
        u = cost["unit_costs"]
        grok_call = 1000 * 0.20 / 1e6 + 100 * 0.50 / 1e6
        son_call = 1000 * 3.00 / 1e6 + 100 * 15.00 / 1e6
        hai_call = 1000 * 1.00 / 1e6 + 100 * 5.00 / 1e6
        check("16.22 cost: unit costs = tokens x PRICES (base 3 grok calls, 17-call k=1, 68-call k=4, 3 judge calls)",
              abs(u["base_3_calls"]["mean_usd"] - 3 * grok_call) < 1e-12
              and abs(u["collective_k1_17_calls"]["mean_usd"] - 17 * grok_call) < 1e-12
              and abs(u["collective_k4_68_calls"]["mean_usd"] - 68 * grok_call) < 1e-12
              and abs(u["judge_sonnet_3_calls"]["mean_usd"] - 3 * son_call) < 1e-12
              and abs(u["judge_haiku_3_calls"]["mean_usd"] - 3 * hai_call) < 1e-12)
        check("16.22 cost: cells with any record missing are counted, not priced (2 priced, 18 missing)",
              u["base_3_calls"]["n_cells_priced"] == 2 and u["base_3_calls"]["n_cells_missing"] == 18
              and u["collective_k4_68_calls"]["n_calls_per_cell"] == 68)
        sy = cost["systems"]
        check("16.22 cost: system formulas (base alone; judge everywhere; 3 base + cov x 3 judge; 17 + cov x 3; 68 + cov x 3)",
              abs(sy["base_alone"]["usd"] - 3 * grok_call) < 1e-12
              and abs(sy["judge_everywhere"]["sonnet"]["usd"] - 3 * son_call) < 1e-12
              and abs(sy["solo_signal_router"]["disagree"]["sonnet"]["usd"]
                      - (3 * grok_call + 0.30 * 3 * son_call)) < 1e-12
              and abs(sy["collective_k1_sensor"]["counter"]["haiku"]["usd"]
                      - (17 * grok_call + 0.25 * 3 * hai_call)) < 1e-12
              and abs(sy["collective_k1_sensor_at_sample0_fire_rate"]["counter"]["haiku"]["usd"]
                      - (17 * grok_call + 0.5 * 3 * hai_call)) < 1e-12
              and abs(sy["k4_sensor"]["composite"]["sonnet"]["usd"]
                      - (68 * grok_call + 0.10 * 3 * son_call)) < 1e-12
              and abs(sy["union_router"]["sonnet"]["usd"]
                      - (3 * grok_call + 17 * grok_call + 0.40 * 3 * son_call)) < 1e-12
              and abs(sy["union_router_k4_over_base"]["sonnet"]["usd"]
                      - (3 * grok_call + 68 * grok_call + 0.40 * 3 * son_call)) < 1e-12)
        check("16.22 cost: an unpriced model is None, never $0",
              usd("no-such-model", 1000, 100) is None
              and mean_bundle_cost(two, lambda c: single_agent_record_paths(GROK, c["arm"], c["item"], records_dir=rd_),
                                   "no-such-model", "x")["mean_usd"] is None)
        empty = cost_per_decision(sc, cov, records_dir=rd_ / "empty")
        check("16.22 cost: no records -> every mean None, nothing generated",
              empty["unit_costs"]["base_3_calls"]["mean_usd"] is None
              and empty["systems"]["base_alone"]["usd"] is None
              and not (rd_ / "empty").exists())

        # end-to-end on the synthetic PANEL with a perfect synthetic sonnet judge:
        # the gate passes against a replay target equal to its own numbers,
        # fails against the real target, and print_report withholds the numbers
        sonnet_syn = {(SONNET_JUDGE[0], SONNET_JUDGE[1], k[2], k[3]): v
                      for k, v in comp.items() if (k[0], k[1]) == HAIKU_JUDGE}
        res_e, _ = analyse(debates, raw, comp, sonnet_syn, draws=50, seed=13, trim_seeds=5,
                           records_dir=rd_ / "empty")
        rv = res_e["routed_vs_solo"]
        check("16.22 end-to-end: both judge slots and the judge-codable slot computed on the synthetic panel",
              rv["haiku"] is not None and rv["sonnet"] is not None
              and rv["sonnet_judge_codable"]["n_cells"] == 30
              and set(rv["haiku"]["signals"]) == set(ROUTED_SIGNALS)
              and rv["replay_gate_16_11a"]["status"] == "failed" and rv["status"] == "UNREAD")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_report(res_e, tr_syn)
        txt = buf.getvalue()
        check("16.22 end-to-end: a failed gate prints GUARD FAILED, marks the block UNREAD and withholds the numbers",
              "ADDENDUM 16.22" in txt and "GUARD FAILED -- replay gate" in txt and "UNREAD" in txt
              and "JUDGE [haiku]" not in txt and "COST PER DECISION" not in txt)
        comp_s = rv["sonnet"]["signals"]["composite"]
        rep_ok = dict(REPLAY_16_11A, routed_composite=comp_s["acc_routed"],
                      delta_composite_vs_base=comp_s["routed_minus_base"]["delta"])
        res_p, _ = analyse(debates, raw, comp, sonnet_syn, draws=50, seed=13, trim_seeds=5,
                           records_dir=rd_ / "empty", replay=rep_ok)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_report(res_p, tr_syn)
        txt = buf.getvalue()
        check("16.22 end-to-end: a passed gate prints the block (both judges, cost table) and no GUARD FAILED",
              res_p["routed_vs_solo"]["status"] == "READ" and "gate PASSED" in txt
              and "JUDGE [haiku]" in txt and "JUDGE [sonnet]" in txt and "COST PER DECISION" in txt
              and "GUARD FAILED" not in txt)
        check("16.22 end-to-end: sonnet cache absent -> gate not computable, block UNREAD, no GUARD FAILED",
              res_syn["routed_vs_solo"]["replay_gate_16_11a"]["status"] == "not_computable"
              and res_syn["routed_vs_solo"]["status"] == "UNREAD")
        cert = certification_summary(res_p)
        check("16.22 certification summary carries the gate, both judge blocks and the cost table",
              cert["status"] == "READ" and cert["replay_gate_16_11a"]["passed"]
              and cert["routed_vs_solo"]["sonnet"]["signals"]["union"]["n_fired"]
              == rv["sonnet"]["signals"]["union"]["n_fired"]
              and cert["cost_per_decision"]["systems"]["base_alone"]["usd"] is None)
        check("16.22 pre-existing keys of the synthetic artefact are untouched by the new block",
              all(res_p[k] == res_syn[k] for k in PREEXISTING_KEYS_RD if k in res_syn
                  and k not in ("composed_sonnet", "guards", "pre_declared_readings")))
        check("16.22 per-model mode marks the gate not_applicable (exploratory)",
              analyse(debates, raw, comp, None, draws=20, seed=13, trim_seeds=3, model="m",
                      judges=(HAIKU_JUDGE,), targets=comparator_targets(comp),
                      records_dir=rd_ / "empty")[0]["routed_vs_solo"]["replay_gate_16_11a"]["status"]
              == "not_applicable")
        # k-sensor derived from the panel (not assumed to be grok's 4): the
        # synthetic panel has sample_idx 0..1, so the bundle is 34 calls
        check("panel_k derives k from the data (synthetic 2, registered 4 by default, override honoured)",
              panel_k(debates) == 2 and panel_k([]) == REGISTERED_K
              and panel_k([{"sample_idx": 3}, {"sample_idx": 0}]) == 4)
        check("16.22 cost table in analyse() prices the k the panel actually ran (k=2 -> 34-call bundle; "
              "union row keyed on the same k; frontier note names it)",
              res_p["cost_per_decision"]["k_sensor"] == 2
              and "collective_k2_34_calls" in res_p["cost_per_decision"]["unit_costs"]
              and "collective_k4_68_calls" not in res_p["cost_per_decision"]["unit_costs"]
              and "union_router_k2_over_base" in res_p["cost_per_decision"]["systems"]
              and "k=2" in res_p["routed_vs_solo"]["pre_declared_readings"]["sonnet"]["R5_accuracy_cost_frontier"]["note"]
              and analyse(debates, raw, comp, sonnet_syn, draws=20, seed=13, trim_seeds=3,
                          records_dir=rd_ / "empty", replay=rep_ok, k_sensor=4)[0]["cost_per_decision"]["k_sensor"] == 4)
        check("16.22 cost note names the collective model, not grok, in per-model mode",
              res_p["cost_per_decision"]["note"].count("grok's cached completion") == 1
              and "m's cached completion" in analyse(
                  debates, raw, comp, None, draws=20, seed=13, trim_seeds=3, model="m",
                  judges=(HAIKU_JUDGE,), targets=comparator_targets(comp),
                  records_dir=rd_ / "empty")[0]["cost_per_decision"]["note"])
        # a NON-registered invocation on the grok configuration (here: the
        # main()-level verdict, e.g. --n-boot 200) never reads as READ/certified
        res_x, _ = analyse(debates, raw, comp, sonnet_syn, draws=20, seed=13, trim_seeds=3,
                           records_dir=rd_ / "empty", replay=rep_ok, registered=False,
                           deviations=["--n-boot 20 (registered 4000)"])
        check("analyse(registered=False) on the grok configuration: readout marker with the deviations, "
              "agreement / pre-declared readings not_applicable, a passed gate reads exploratory (never READ), "
              "and the cert carries the marker",
              res_x["readout"]["deviations"] == ["--n-boot 20 (registered 4000)"]
              and "not_applicable" in res_x["agreement"] and "not_applicable" in res_x["pre_declared_readings"]
              and res_x["routed_vs_solo"]["replay_gate_16_11a"]["status"] == "passed"
              and res_x["routed_vs_solo"]["status"] != "READ"
              and res_x["routed_vs_solo"]["status"].startswith("exploratory")
              and res_x["routed_vs_solo"]["readout"]["registered"] is False
              and certification_summary(res_x, parent=Path("x.json"))["readout"]["registered"] is False
              and certification_summary(res_x, parent=Path("x.json"))["status"] != "READ"
              and certification_summary(res_x, parent=Path("x.json"))["parent_artefact"] == "x.json")
        check("certification_summary names the parent it is given (registered default: router_decomposition.json)",
              certification_summary(res_p)["parent_artefact"] == str(ANALYSIS_PATH)
              and "readout" not in certification_summary(res_p))

    # ---- registration = the FULL default invocation (not merely --model grok)
    reg_kw = dict(model=GROK, solo=GROK_SOLO, judges=[HAIKU_JUDGE, SONNET_JUDGE], rows=ROWS_PATH,
                  votes=VOTES_PATH, comparator=COMPARATOR_PATH, sonnet_rows=SONNET_ROWS_PATH)
    check("registered_deviations: the default invocation has none",
          registered_deviations(**reg_kw) == [])
    devs = {
        "solo": registered_deviations(**dict(reg_kw, solo=("gpt-5.4-nano", "standard"))),
        "judges": registered_deviations(**dict(reg_kw, judges=[HAIKU_JUDGE])),
        "rows": registered_deviations(**dict(reg_kw, rows=Path("x_rows.csv"))),
        "extras": registered_deviations(**reg_kw, extras=[Path("e.csv")]),
        "targets": registered_deviations(**reg_kw, transfer_targets=f"{GROK}:standard"),
        "records": registered_deviations(**reg_kw, records_dir=Path("elsewhere")),
        "draws": registered_deviations(**reg_kw, draws=200),
        "seed": registered_deviations(**reg_kw, seed=1),
        "trim": registered_deviations(**reg_kw, trim_seeds=5),
        "k": registered_deviations(**reg_kw, k_sensor=2),
        "k4": registered_deviations(**reg_kw, k_sensor=4),
        "model": registered_deviations(**dict(reg_kw, model="gpt-5.4-nano")),
    }
    check("registered_deviations names each single deviation (solo, judges, rows, extra comparator, "
          "targets, records dir, draws, seed, trim seeds, k, model) and accepts --k-sensor 4",
          all(len(v) == 1 for k, v in devs.items() if k != "k4") and devs["k4"] == []
          and "--solo" in devs["solo"][0] and "--judges" in devs["judges"][0]
          and "--rows" in devs["rows"][0] and "--extra-comparator" in devs["extras"][0]
          and "--transfer-targets" in devs["targets"][0] and "--records-dir" in devs["records"][0]
          and "--n-boot" in devs["draws"][0] and "--seed" in devs["seed"][0]
          and "--trim-seeds" in devs["trim"][0] and "--k-sensor" in devs["k"][0]
          and "--model" in devs["model"][0])
    # main() with a non-default option on the grok panel must NOT resolve to
    # the registered artefact paths: without explicit outputs it refuses
    # (argparse error, exit 2) before reading any data or writing anything
    import contextlib
    import io
    snap = {p: (p.read_bytes() if p.exists() else None) for p in (ANALYSIS_PATH, TRANSFER_PATH, CERT_PATH)}
    for argv in (["--solo", "gpt-5.4-nano:standard"], ["--n-boot", "200"], ["--judges", "claude-haiku-4-5:standard"],
                 ["--trim-seeds", "5"], ["--transfer-targets", f"{GROK}:standard"], ["--k-sensor", "2"],
                 ["--records-dir", "elsewhere"], ["--extra-comparator", "e.csv"],
                 ["--n-boot", "200", "--out", "x.json"],                       # partial outputs: still refused
                 ["--n-boot", "200", "--out", str(ANALYSIS_PATH), "--transfer-out", "t.json",
                  "--cert-out", "c.json"]):                                     # explicit registered path: refused
        rc = None
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            try:
                rc = main(argv)
            except SystemExit as e:
                rc = e.code
        check(f"main({' '.join(argv)}) on the grok panel refuses without explicit exploratory outputs (exit 2, "
              "registered artefacts untouched)",
              rc == 2 and ("not the registered invocation" in err.getvalue()
                           or "is a registered artefact" in err.getvalue())
              and all((p.read_bytes() if p.exists() else None) == b for p, b in snap.items())
              and not Path("x.json").exists() and not Path("t.json").exists() and not Path("c.json").exists())

    # ---- regression: the default invocation must be BYTE-IDENTICAL to the
    #      registered artefacts (a few minutes; 4000 draws, 200 trim seeds),
    #      and every key that existed BEFORE 16.22 must hash to the digest
    #      taken from the pre-16.22 artefacts (PREEXISTING_DIGEST_*)
    if ANALYSIS_PATH.exists() and TRANSFER_PATH.exists() and ROWS_PATH.exists():
        with tempfile.TemporaryDirectory() as td:
            o1, o2, o3 = Path(td) / "rd.json", Path(td) / "tr.json", Path(td) / "cert.json"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = main(["--out", str(o1), "--transfer-out", str(o2), "--cert-out", str(o3)])
            check("default invocation reruns BYTE-IDENTICAL to router_decomposition.json",
                  rc == 0 and o1.read_bytes() == ANALYSIS_PATH.read_bytes())
            check("default invocation reruns BYTE-IDENTICAL to transfer_readouts.json",
                  o2.read_bytes() == TRANSFER_PATH.read_bytes())
            rd_now = json.loads(o1.read_text())
            tr_now = json.loads(o2.read_text())
            check("REGRESSION: every pre-16.22 key of router_decomposition.json is unchanged (digest)",
                  preexisting_digest(rd_now, PREEXISTING_KEYS_RD) == PREEXISTING_DIGEST_RD)
            check("REGRESSION: every pre-16.22 key of transfer_readouts.json is unchanged (digest)",
                  preexisting_digest(tr_now, PREEXISTING_KEYS_TR) == PREEXISTING_DIGEST_TR)
            check("16.22 replay gate passes on the registered data (block READ) and the cert artefact agrees",
                  rd_now["routed_vs_solo"]["status"] == "READ"
                  and json.loads(o3.read_text())["status"] == "READ"
                  and "GUARD FAILED" not in buf.getvalue())
            cert_now = json.loads(o3.read_text())
            check("default invocation reruns routing_certification.json IDENTICAL up to the parent path "
                  "(which names the --out actually written; the default writes router_decomposition.json)",
                  CERT_PATH.exists() and cert_now["parent_artefact"] == str(o1)
                  and dict(cert_now, parent_artefact=str(ANALYSIS_PATH)) == json.loads(CERT_PATH.read_text())
                  and json.loads(CERT_PATH.read_text())["parent_artefact"] == str(ANALYSIS_PATH))
            check("default invocation: no readout marker, k derived as 4, union row keyed k4, grok cost note",
                  "readout" not in rd_now and "readout" not in rd_now["routed_vs_solo"]
                  and rd_now["cost_per_decision"]["k_sensor"] == 4
                  and "union_router_k4_over_base" in rd_now["cost_per_decision"]["systems"]
                  and rd_now["cost_per_decision"]["note"].startswith("means only")
                  and "grok's cached completion" in rd_now["cost_per_decision"]["note"])
    else:
        check("registered artefacts present for the default-invocation regression", False)

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Addendum 16.15.1-16.15.3 router decomposition")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-boot", type=int, default=DEFAULT_DRAWS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--trim-seeds", type=int, default=TRIM_SEEDS)
    ap.add_argument("--rows", type=Path, default=None,
                    help="deliberation rows (default: the registered grok run; required for another --model)")
    ap.add_argument("--votes", type=Path, default=None)
    ap.add_argument("--comparator", type=Path, default=COMPARATOR_PATH)
    ap.add_argument("--sonnet-rows", type=Path, default=SONNET_ROWS_PATH)
    ap.add_argument("--out", type=Path, default=None,
                    help="default router_decomposition.json (grok) / router_decomposition_<short>.json")
    ap.add_argument("--transfer-out", type=Path, default=None,
                    help="default transfer_readouts.json (grok) / transfer_readouts_<short>.json")
    ap.add_argument("--cert-out", type=Path, default=None,
                    help="16.22 standalone artefact; default routing_certification.json (grok) / "
                         "routing_certification_<short>.json")
    ap.add_argument("--records-dir", type=Path, default=OUT_DIR,
                    help="where the cached call records are READ for the 16.22 cost table")
    ap.add_argument("--k-sensor", type=int, default=None,
                    help="samples per cell the cost table prices for the k-sensor bundle "
                         "(default: derived from the panel as max sample_idx + 1; 4 on the "
                         "registered grok panel, 2 on the nano community)")
    pm = ap.add_argument_group("per-model readout (exploratory, not registered)")
    pm.add_argument("--model", default=GROK, help="panel model of the rows/votes (default grok)")
    pm.add_argument("--solo", default=f"{GROK_SOLO[0]}:{GROK_SOLO[1]}",
                    help="model:scaffold solo comparator (16.15.2 signals, within-strata transfer)")
    pm.add_argument("--judges", default=",".join(f"{m}:{s}" for m, s in (HAIKU_JUDGE, SONNET_JUDGE)),
                    help="comma list of model:scaffold: primary judge, then the optional "
                         "coverage-restricted judge (read from --sonnet-rows merged with the comparator)")
    pm.add_argument("--extra-comparator", type=Path, nargs="+", action="extend", default=None,
                    help="CSVs merged into --comparator (load_comparator format)")
    pm.add_argument("--transfer-targets", default=None,
                    help="comma list of model:scaffold to restrict the transfer readouts "
                         "(default: the registered four for grok, else every pair in the merged comparator)")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()

    model = a.model
    grok_panel = model == GROK
    if a.rows is None or a.votes is None:
        if not grok_panel:
            ap.error(f"--model {model} needs --rows and --votes for that panel")
        a.rows = ROWS_PATH if a.rows is None else a.rows
        a.votes = VOTES_PATH if a.votes is None else a.votes
    solo = parse_model_scaffold(a.solo)
    judges = [parse_model_scaffold(s) for s in a.judges.split(",") if s.strip()]
    if not 1 <= len(judges) <= 2:
        ap.error("--judges takes one or two model:scaffold entries (primary, coverage-restricted)")
    extras = list(a.extra_comparator or [])
    if a.k_sensor is not None and a.k_sensor < 1:
        ap.error("--k-sensor must be >= 1")
    # REGISTERED = the full default invocation, not merely --model grok. Any
    # deviation (solo, judges, rows/votes/comparator/sonnet cache, extra
    # comparator, transfer targets, records dir, estimator, k) is exploratory:
    # it never resolves to the registered artefact paths and carries the
    # `readout` marker in every artefact it writes.
    deviations = registered_deviations(
        model=model, solo=solo, judges=judges, rows=a.rows, votes=a.votes,
        comparator=a.comparator, sonnet_rows=a.sonnet_rows, extras=extras,
        transfer_targets=a.transfer_targets, records_dir=a.records_dir,
        draws=a.n_boot, seed=a.seed, trim_seeds=a.trim_seeds, k_sensor=a.k_sensor)
    registered = not deviations
    short = short_name(model)
    if grok_panel and not registered:
        missing = [f for f, v in (("--out", a.out), ("--transfer-out", a.transfer_out),
                                  ("--cert-out", a.cert_out)) if v is None]
        if missing:
            ap.error(f"not the registered invocation ({'; '.join(deviations)}): the registered "
                     f"grok artefacts are written only by the default invocation, so pass "
                     f"{', '.join(missing)} explicitly for this exploratory run")
    if a.out is None:
        a.out = ANALYSIS_PATH if registered else OUT_DIR / f"router_decomposition_{short}.json"
    if a.transfer_out is None:
        a.transfer_out = TRANSFER_PATH if registered else OUT_DIR / f"transfer_readouts_{short}.json"
    if a.cert_out is None:
        a.cert_out = CERT_PATH if registered else OUT_DIR / f"routing_certification_{short}.json"
    for flag, path in (("--out", a.out), ("--transfer-out", a.transfer_out), ("--cert-out", a.cert_out)):
        if not registered and Path(path).resolve() in {ANALYSIS_PATH.resolve(), TRANSFER_PATH.resolve(),
                                                       CERT_PATH.resolve()}:
            ap.error(f"{flag} {path} is a registered artefact; this invocation is not the "
                     f"registered one ({'; '.join(deviations)})")

    debates = [d for d in load_debates(a.rows, a.votes) if d["model"] == model]
    raw_rows = load_rows(a.rows, model)
    comp = merge_comparators([load_comparator(p) for p in [a.comparator, *extras]])
    sonnet_cache = load_comparator(a.sonnet_rows) if a.sonnet_rows.exists() else None
    if sonnet_cache is None:
        print(f"sonnet cache {a.sonnet_rows} absent: sonnet block NOT COMPUTABLE")
    # the coverage-restricted judge reads the sonnet cache merged with the comparator
    sonnet_comp = (merge_comparators([comp, sonnet_cache]) if sonnet_cache is not None
                   else (comp if len(judges) > 1 else None))
    if a.transfer_targets is not None:
        targets = comparator_targets(comp, [parse_model_scaffold(s) for s in a.transfer_targets.split(",") if s.strip()])
    else:
        targets = None if grok_panel else comparator_targets(comp)
    res, tr = analyse(debates, raw_rows, comp, sonnet_comp, draws=a.n_boot,
                      seed=a.seed, trim_seeds=a.trim_seeds, model=model, solo=solo,
                      judges=judges, targets=targets, records_dir=a.records_dir,
                      k_sensor=a.k_sensor, registered=registered, deviations=deviations)
    res["inputs"] = {"rows": str(a.rows), "votes": str(a.votes),
                     "comparator": str(a.comparator),
                     "sonnet_rows": str(a.sonnet_rows) if sonnet_cache else None}
    if extras:
        res["inputs"]["extra_comparator"] = [str(p) for p in extras]
    tr["inputs"] = dict(res["inputs"])
    print_report(res, tr)
    a.out.write_text(json.dumps(res, indent=2, default=str))
    a.transfer_out.write_text(json.dumps(tr, indent=2, default=str))
    a.cert_out.write_text(json.dumps(certification_summary(res, parent=a.out), indent=2, default=str))
    print(f"\nwrote {a.out}\nwrote {a.transfer_out}\nwrote {a.cert_out}")
    if not res["guards"]["passed"]:
        print("GUARD FAILED")
        return 3
    if res["routed_vs_solo"]["replay_gate_16_11a"]["status"] == "failed":
        print("GUARD FAILED (Addendum 16.22 replay gate; block UNREAD)")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
