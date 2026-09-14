"""
scripts/run_crowdgold_multivendor.py -- Task 16.18: the cross-vendor decider
layer. The (identical, off) cell of Addendum 16.10 with each of the three
seats served by a DIFFERENT model.

THE CELL THIS EXTENDS

Addendum 16.10's (identical, off) cell is three byte-identical uninterested
reader briefs (the embodied neutral_adjudicator brief, verbatim), no seat
ever reading any other, the stake-free R0 preamble and moderator prompts of
16.8, one external moderator, the registered 7-round / 17-call protocol. Its
finding: three independent draws of one grok brief do not aggregate to more
than one reader (seat-majority minus best seat -0.002 [-0.020, +0.015]);
the collective's S2 accuracy was 0.875, still below grok answering alone
(0.911). That cell's three voters shared one model, so their errors were
never independent in the Condorcet sense.

THIS MODULE holds everything in that cell fixed -- briefs, labels, no-edge
r1/r2 wording, stake-free preamble and moderator prompts, caps, arms, panel
rule, protocol, external moderator -- and changes exactly one thing: the
three seats are three DIFFERENT vendors' models (default claude-haiku-4-5 /
gpt-5.4-nano / grok-4-1-fast-reasoning; --seat-models), moderated by
--moderator-model (default grok-4-1-fast-reasoning, as in every 16.10 cell).
The question it puts to scripts/analyze_multivendor.py is Ladha / Condorcet:
does a majority of three independent-error voters beat its best voter, and
does a cross-vendor collective beat the single-vendor collective (0.875) and
the solo model (0.911)? It also gives the embodied sensor of cg_deliberation
an actuator that is not one model's judgement: route the debates the
embodied cell flags (n_objectors >= 2) to this cross-vendor majority, at
zero extra spend, and compare against the haiku-judge and sonnet-judge
routers of router_decomposition.json.

WHAT IS AND IS NOT CHANGED, stated precisely

  UNCHANGED (imported, never retyped): the neutral brief (rcd.ROLES), the
    seat labels Reader Alpha / Beta / Gamma, R0_PREAMBLE_CONTROL,
    SYNTHESIS_SYSTEM_CONTROL, INTEGRATION_SYSTEM_CONTROL
    (run_crowdgold_unembodied), r1_user_noedge / r2_user_noedge
    (run_crowdgold_topology), do_call and its parent_sha check, the row
    builder, the guards, the SDT / role / defeasibility readouts and the CSV
    writer (all rcd.main's, reached by delegation exactly as the sibling
    runners do). A given seat's R0 / r1 / r2 / r3 / r4 prompt is therefore
    BYTE-IDENTICAL to the (identical, off) cell's prompt for the same seat
    position; only the model answering it differs.
  CHANGED: the model behind each seat (do_call is wrapped so the seat's
    role id selects its model), and the R3/R4 token caps of Anthropic seats
    (below).
  DIFFERS BY THE ROLE-ID TOKENS ONLY: the two MODERATOR-facing prompts.
    rcd.synthesis_user writes "--- Reader X (<role id>) : final position ---"
    per seat and rcd.integration_user lists the seat ids after "from:", so
    the moderator here sees "(reader_1_noedge)" where the (identical, off)
    moderator saw "(reader_alpha_noedge)", and an ADDRESSED list of the new
    ids. The ids are position-only (no vendor name leaks into any prompt);
    the selftest constructs both cells' synthesis and integration prompts
    and asserts they are equal after the three id substitutions and differ
    nowhere else, so the diff cannot grow unnoticed.

PER-MODEL TOKEN CAPS -- exactly what the base runner did for haiku / sonnet

Addendum 6 retracted haiku's and sonnet's first grip readouts because the
registered R3/R4 caps (1024 / 512) truncated 81-97% of haiku's R4 votes
(mean 487 completion tokens against a 512 cap; measured again here from the
cache: haiku R4 at cap 512 truncated 552 of 600). The corrected runs
(cg_deliberation_haiku_fixed, cg_deliberation_sonnet_fixed) re-ran R3 and R4
at --max-tokens-label 3072 --max-tokens-vote 3072 with r0/r1/r2 at 2560 and
the moderator at 1024, and passed the guard (0 truncations in 598 + 580
haiku calls). grok and nano ran and passed at the registered 1024 / 512 and
every 16.10 cell used those. This module reproduces that per model: a seat
whose model is Anthropic (generators._is_anthropic, prefix "claude") gets
max(cap, 3072) on r3_label and r4_vote; every other seat and the moderator
keep the caps passed on the command line (defaults 2560 / 1024 / 1024 /
512, the registered values). --seat-r3r4-caps model=cap[,model=cap] sets or
overrides the per-model R3/R4 cap so a new deployment can be given one
without a code change. The --r3r4-reasoning-effort / --r3r4-thinking-budget
knobs (Addendum 8 rungs 3-4, test-time compute on R3/R4 only) and the
--stake-* interventions are REFUSED here, as in every 16.10 cell: the
readout is the plain protocol.

CONTENT-FILTER SCREEN -- why the panel is smaller than 210

The base runner drops items whose cached single-agent NoT call was refused
by the content filter, as the union over the models in play; a 17-call
chain cannot absorb a per-call refusal rate. The 16.10 cells were grok-only
and that screen left 210 of 249 items. Measured on the cache today: grok
refuses on 39 items, nano on 40, haiku on 0; the union is 52 (13 nano-only,
12 grok-only), so the default screen here (union over the seat models AND
the moderator, --screen-models to override) leaves 197 items, 394 debates.
Screening on grok alone would keep 210 items but put the nano seat's
refusal share at 13/210 = 6.2% per arm, over the guard's 5% NOVERDICT
limit, so the run would be unreadable by construction. The analyzer
compares cells on matched items, so the 197 are compared to the 16.10 cells
on the same 197.

CACHE NAMESPACES -- the landmine, and which calls replay

call_cache_path keys on (model, scaffold, cap, arm, item, sample, round,
role id) and NOT on brief text, topology or the other seats' models; r1's
parent_sha covers all three r0 texts whether or not the seat sees them. If
the grok seat here reused the (identical, off) cell's role id
reader_gamma_noedge, its r0 would legitimately replay (byte-identical
prompt), but its r1's parent_sha would differ (the other two r0s are now
haiku's and nano's), the record would be regenerated AND WRITTEN OVER the
16.10 cell's r1 at the same path. So every seat here has its own role id
(reader_1_noedge, reader_2_noedge, reader_3_noedge; the seat's model is in
every vote row as seat_model and in the rows as seat_models) and the
moderator has its own prefix (modmultivendor-<model>). Consequently NO CALL
REPLAYS from any existing run: all 17 calls per debate are new, and the
dry-run counts all of them (replayed_calls = 0). The same holds for the
moderator: its prompts quote the three new seats' r2 texts, so even under a
shared prefix nothing could be served.

A future panel that shares a (model, seat position) with this one -- e.g.
haiku, nano, <fourth vendor> -- would replay this run's r0 for the shared
seats and regenerate r1..r4 over this run's records. cache_is_clean()
refuses that on the --run path unless --resume is passed; use --namespace
<tag> to give such a panel its own ids (reader_1_noedge_<tag>) and prefix
(modmultivendor<tag>) instead.

The selftest verifies all namespaces EMPTY (no file in divergence_study_
outputs carries any of them -- or, once the registered run has landed,
only that run's own footprint, which the --run path refuses without
--resume -- and no stub / selftest file carries this runner's names),
stubs generate_any, and redirects run_crowdgold_deliberation.OUT_DIR to a
temporary directory. In that directory it drives the WHOLE delegated path
offline (this main -> rcd.main -> rcd.run -> the wrapped run_deliberation
and do_call -> guards -> readouts -> CSVs / summary / sidecar) with --run,
then shows a second --run is refused as dirty and --run --resume replays
the intact cache with zero new generations. cache_is_clean() resolves the
directory at call time (rcd.OUT_DIR), so the clean check always looks at
the directory the run writes to.

OUTPUTS AND READOUT

<tag>_rows.csv / _votes.csv / _summary.json are the base runner's, under
the widened schemas: rows carry model=multivendor (the panel label, since
no single vendor answers for the collective), seat_models, seat_role_ids,
r0_verdicts and r2_verdicts; votes carry seat_index, seat_model,
r0_verdict and r2_verdict per seat. <tag>_panel.json is this module's
provenance sidecar (seat -> model map, per-seat caps, moderator id,
screen, the delegated argv, the base runner's exit code and the
COMPLETENESS record below). scripts/analyze_multivendor.py reads them:
per-seat accuracy, majority vs best seat (Ladha / Condorcet), S2 vs the
(identical, off) cell and vs grok solo, the 16.10-style structure, and the
composed bipartite graph (embodied counter -> this majority) against the
haiku / sonnet judges of router_decomposition.json.

GUARDS AFTER THE DELEGATED RUN (this wrapper's, on top of the base runner's)

  * COMPLETENESS. rcd.run swallows an exception raised inside
    run_deliberation with one console ERROR line and omits that debate from
    rows / votes / calls; the base round and outcome guards and every
    analyzer denominator are then computed on the survivors, so a vendor
    outage concentrated in one arm (or one seat's vendor) is arm-correlated
    attrition that nothing reports. After rcd.main returns, this wrapper
    compares the rows CSV with the (arm, item, sample) keys it launched (it
    built the panel itself), writes the record into <tag>_panel.json
    ("completeness": n_expected, n_rows, n_missing, the missing keys,
    per-arm counts) and as <tag>_completeness.json, and prints the literal
    GUARD FAILED and exits 4 on any shortfall.
  * QUARANTINE. The base runner writes its CSVs BEFORE its guard verdict.
    On exit 4 (its round / outcome guard, or the completeness guard) the
    rows and votes CSVs are renamed <tag>_GUARD_FAILED_rows.csv /
    _votes.csv so nothing under the readable tag can be tabulated; re-run
    with --run --resume once the cause is fixed (cached calls replay).
    scripts/analyze_multivendor.py refuses a cell whose panel.json exit
    code, summary guards or completeness record is failed or missing.

SPEND SAFETY

Nothing here generates unless the literal flag --run is given. --dry-run
(or simply omitting --run) prints the mixed-price cost model -- every seat
billed at its own PRICES rate, the moderator at its own -- and exits.
--run refuses a seat or moderator model absent from PRICES ($0.00 is a
placeholder, not a ceiling). Completion lengths in the cost model are
MEASURED from the cache where a measurement exists (r0-r2: rcd.MEASURED_
NCOT_COMPLETION; r3/r4 and the moderator rounds: the per-model means
recorded in MEASURED_R3R4_COMPLETION / MEASURED_MODERATOR_COMPLETION below,
read from the cgd_* records on 2026-09-13) and fall back to the base
runner's ASSUMED values, flagged, for a model with none.

Usage
-----
  python -m scripts.run_crowdgold_multivendor --selftest
  python -m scripts.run_crowdgold_multivendor --verify-cache-clean
  python -m scripts.run_crowdgold_multivendor --dry-run
  python -m scripts.run_crowdgold_multivendor --run                 # SPENDS
  python -m scripts.run_crowdgold_multivendor --run --seat-models claude-haiku-4-5,gpt-5.4-nano,<new> \
        --namespace b --tag cg_deliberation_multivendor_b       # a second panel: own ids, own CSVs
  python -m scripts.analyze_multivendor                          # the readout (zero spend)

Exit codes: 0 ok, 1 no rows, 2 bad config / --run not given / unpriced
model / banned flag, 3 cache not clean (pass --resume), 4 guard failed
(rcd.main prints the literal GUARD FAILED).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import scripts.run_crowdgold_deliberation as rcd
from scripts.run_crowdgold_deliberation import (
    AGENT_ROUNDS, ASSUMED_COMPLETION, MEASURED_NCOT_COMPLETION, MODERATOR_ROUNDS,
    POST_DELIM, PRICES, Role,
)
from scripts.run_crowdgold_topology import (
    CELLS as TOPOLOGY_CELLS, build_roles as topology_roles, r1_user_noedge,
    r2_user_noedge,
)
from scripts.run_crowdgold_unembodied import (
    INTEGRATION_SYSTEM_CONTROL, MODERATOR_PREFIX as UNEMBODIED_PREFIX,
    R0_PREAMBLE_CONTROL, SYNTHESIS_SYSTEM_CONTROL, UNEMBODIED_ROLES,
)
from scripts.run_phase1_quartet import OUT_DIR, _safe
from scripts.verdict_format import extract_verdict

try:
    from scripts.generators import _is_anthropic as is_anthropic
except ImportError:  # pragma: no cover
    def is_anthropic(model: str) -> bool:
        return model.lower().startswith("claude")

DEFAULT_SEAT_MODELS: tuple[str, ...] = (
    "claude-haiku-4-5", "gpt-5.4-nano", "grok-4-1-fast-reasoning")
DEFAULT_MODERATOR = "grok-4-1-fast-reasoning"
DEFAULT_ARMS = (rcd.THIRD_PERSON, rcd.AS_ASKER)
#: the rows / votes `model` column: the panel, not any one seat's model
PANEL_LABEL = "multivendor"
DEFAULT_TAG = "cg_deliberation_multivendor"
MODERATOR_PREFIX_BASE = "modmultivendor"
SEAT_ID_BASE = ("reader_1_noedge", "reader_2_noedge", "reader_3_noedge")
SEAT_LABELS = ("Reader Alpha", "Reader Beta", "Reader Gamma")
N_SEATS = 3
ANTHROPIC_R3R4_CAP = 3072
R3R4_ROUNDS = ("r3_label", "r4_vote")
#: registered caps (rcd.main defaults), restated so the delegated command is explicit
DEFAULT_CAPS = {"agent": 2560, "moderator": 1024, "label": 1024, "vote": 512}

#: Captured ONCE at import, before any install() mutates rcd. The neutral
#: brief is read from the embodied role set, never from a mutated rcd.ROLES.
_ORIG_ROLES: tuple[Role, ...] = rcd.ROLES
_NEUTRAL_BRIEF: str = next(r for r in _ORIG_ROLES if r.role_id == "neutral_adjudicator").brief
_ORIG_DO_CALL = rcd.do_call
_ORIG_RUN_DELIBERATION = rcd.run_deliberation
_ORIG_FILTER_PRONE = rcd.filter_prone_items
_ORIG_ROW_FIELDS = rcd.ROW_FIELDS
_ORIG_VOTE_FIELDS = rcd.VOTE_FIELDS

#: MEASURED mean completion tokens at the caps the registered runs used,
#: read from every cgd_*_narrative_cot_* cache record on 2026-09-13 (n in
#: brackets; 0 truncations in every line quoted): r3_label / r4_vote. A model
#: absent here (a new deployment) falls back to rcd.ASSUMED_COMPLETION and
#: the dry-run flags the line "(assumed)"; add a row once its first run has
#: landed. Data, not code: nothing else keys on this table.
MEASURED_R3R4_COMPLETION: dict[str, tuple[int, int]] = {
    "claude-haiku-4-5": (565, 1026),          # t3072 / t3072  (598 / 580)
    "claude-sonnet-4-6": (232, 308),          # t3072 / t3072  (96 / 95)
    "gpt-5.4-nano": (256, 456),               # t1024 / t512   (2508 / 2508)
    "grok-4-1-fast-reasoning": (30, 88),      # t1024 / t512   (10190 / 10179)
    "gpt-4o": (36, 64),                       # t3072 / t3072  (240 / 240)
}
#: synthesis / integration, moderator at t1024 (same source and date)
MEASURED_MODERATOR_COMPLETION: dict[str, tuple[int, int]] = {
    "claude-haiku-4-5": (264, 244),           # (200 / 198)
    "claude-sonnet-4-6": (221, 221),          # (32 / 32)
    "gpt-5.4-nano": (219, 297),               # (836 / 836)
    "grok-4-1-fast-reasoning": (156, 172),    # (3398 / 3394)
    "gpt-4o": (111, 179),                     # (80 / 80)
}

ROW_FIELDS_MV = tuple(_ORIG_ROW_FIELDS) + (
    "seat_models", "seat_role_ids", "r0_verdicts", "r2_verdicts")
VOTE_FIELDS_MV = tuple(_ORIG_VOTE_FIELDS) + (
    "seat_index", "seat_model", "r0_verdict", "r2_verdict", "r0_output_len",
    "r0_finish_reason")

# ---------------------------------------------------------------------------
# Panel: seats, models, caps
# ---------------------------------------------------------------------------

#: set by install(); read by the do_call / run_deliberation wrappers
SEAT_MODEL_BY_ROLE: dict[str, str] = {}
R3R4_CAP_BY_MODEL: dict[str, int] = {}
SCREEN_MODELS: tuple[str, ...] = ()
NAMESPACE = ""


def seat_ids(namespace: str = "") -> tuple[str, ...]:
    ns = _safe(namespace) if namespace else ""
    return tuple(f"{b}_{ns}" if ns else b for b in SEAT_ID_BASE)


def moderator_prefix(namespace: str = "") -> str:
    return MODERATOR_PREFIX_BASE + (_safe(namespace) if namespace else "")


def moderator_role_id(mod_model: str, namespace: str = "") -> str:
    return f"{moderator_prefix(namespace)}-{_safe(mod_model)}"


def build_roles(namespace: str = "") -> tuple[Role, ...]:
    """Three seats, the neutral brief verbatim, labels Alpha/Beta/Gamma as in
    the (identical, off) cell, ids of this module's own."""
    return tuple(Role(role_id=rid, label=lab, paper_role="multivendor_reader",
                      stake="none", brief=_NEUTRAL_BRIEF)
                 for rid, lab in zip(seat_ids(namespace), SEAT_LABELS))


def parse_models(spec: str) -> tuple[str, ...]:
    return tuple(m.strip() for m in spec.split(",") if m.strip())


def validate_seat_models(models: Sequence[str]) -> Optional[str]:
    """None when valid, else the reason."""
    if len(models) != N_SEATS:
        return f"--seat-models needs exactly {N_SEATS} models, got {len(models)}: {list(models)}"
    if len(set(models)) != N_SEATS:
        return f"--seat-models must be three DIFFERENT models, got {list(models)}"
    return None


def parse_cap_overrides(spec: Optional[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--seat-r3r4-caps entry {part!r} is not model=cap")
        m, c = part.split("=", 1)
        out[m.strip()] = int(c)
    return out


def resolve_r3r4_caps(models: Sequence[str],
                      overrides: Optional[dict[str, int]] = None) -> dict[str, int]:
    """Per-model R3/R4 cap FLOOR: Anthropic seats 3072 (Addendum 6's
    correction), explicit overrides win, every other model absent (keeps the
    command-line cap)."""
    out: dict[str, int] = {}
    for m in models:
        if overrides and m in overrides:
            out[m] = overrides[m]
        elif is_anthropic(m):
            out[m] = ANTHROPIC_R3R4_CAP
    return out


def seat_cap(model: str, round_name: str, default_cap: int,
             caps: Optional[dict[str, int]] = None) -> int:
    caps = R3R4_CAP_BY_MODEL if caps is None else caps
    if round_name in R3R4_ROUNDS and model in caps:
        return max(default_cap, caps[model])
    return default_cap


# ---------------------------------------------------------------------------
# The wrappers install() puts in place of rcd's own
# ---------------------------------------------------------------------------

def _do_call_multivendor(*, model: str, round_name: str, role_id: str,
                         max_tokens: int, **kw) -> dict:
    """rcd.run_deliberation calls do_call with model = the panel label for
    the agent rounds and model = the moderator for the moderator rounds. A
    seat's role id selects its model and its per-model cap; the moderator
    passes through untouched."""
    seat_model = SEAT_MODEL_BY_ROLE.get(role_id)
    if seat_model is not None:
        model = seat_model
        max_tokens = seat_cap(seat_model, round_name, max_tokens)
    return _ORIG_DO_CALL(model=model, round_name=round_name, role_id=role_id,
                         max_tokens=max_tokens, **kw)


def _run_deliberation_multivendor(model: str, arm: str, item, idx: int, **kw):
    """The base row builder, then the per-seat model and the per-seat r0/r2
    verdicts written into the records (the analyzer's Condorcet readouts
    need each seat's own verdict, and its model, without touching the cache)."""
    row, votes, calls = _ORIG_RUN_DELIBERATION(model, arm, item, idx, **kw)
    r0 = {c["role_id"]: c for c in calls if c.get("round") == "r0"}
    r2 = {c["role_id"]: c for c in calls if c.get("round") == "r2"}

    def verdict(recs, rid):
        rec = recs.get(rid)
        return extract_verdict(rec.get("output") or "", rcd.INSTRUMENT) if rec else rcd.NOVERDICT

    order = tuple(rcd.ROLE_ORDER)
    row["seat_models"] = "|".join(SEAT_MODEL_BY_ROLE.get(rid, "") for rid in order)
    row["seat_role_ids"] = "|".join(order)
    row["r0_verdicts"] = "|".join(verdict(r0, rid) for rid in order)
    row["r2_verdicts"] = "|".join(verdict(r2, rid) for rid in order)
    for v in votes:
        rid = v["role_id"]
        v["seat_index"] = order.index(rid) + 1 if rid in order else ""
        v["seat_model"] = SEAT_MODEL_BY_ROLE.get(rid, "")
        v["r0_verdict"] = verdict(r0, rid)
        v["r2_verdict"] = verdict(r2, rid)
        rec0 = r0.get(rid) or {}
        v["r0_output_len"] = len(rec0.get("output") or "")
        v["r0_finish_reason"] = rec0.get("finish_reason", "")
    return row, votes, calls


def filter_prone_union(models: Sequence[str]) -> tuple[set[str], dict[str, int]]:
    """Union of the base runner's per-model screen over ``models``."""
    drop: set[str] = set()
    per: dict[str, int] = {}
    for m in models:
        f = _ORIG_FILTER_PRONE(m)
        per[m] = len(f)
        drop |= f
    return drop, per


def _filter_prone_multivendor(model: str) -> set[str]:
    """What rcd.main calls once per --models entry. The panel label is not a
    model; the screen is the union over SCREEN_MODELS."""
    if model == PANEL_LABEL:
        return filter_prone_union(SCREEN_MODELS)[0]
    return _ORIG_FILTER_PRONE(model)


def install(seat_models: Sequence[str], *, screen_models: Sequence[str] = (),
            r3r4_caps: Optional[dict[str, int]] = None, namespace: str = "") -> None:
    """Swap in the cell: roles, stake-free framing, no-edge rounds, the
    moderator prefix, the seat -> model map, the per-model caps, the screen,
    the widened CSV schemas. Everything is referenced by name at call time
    inside rcd, so patching the module attributes is total."""
    global SEAT_MODEL_BY_ROLE, R3R4_CAP_BY_MODEL, SCREEN_MODELS, NAMESPACE
    bad = validate_seat_models(seat_models)
    if bad:
        raise SystemExit(f"ERROR: {bad}")
    R = build_roles(namespace)
    rcd.ROLES = R
    rcd.ROLE_BY_ID = {r.role_id: r for r in R}
    rcd.ROLE_ORDER = tuple(r.role_id for r in R)
    rcd.N_AGENTS = len(R)
    rcd.CALLS_PER_CELL = rcd.N_AGENTS * len(AGENT_ROUNDS) + len(MODERATOR_ROUNDS)
    rcd.R0_PREAMBLE = R0_PREAMBLE_CONTROL
    rcd.SYNTHESIS_SYSTEM = SYNTHESIS_SYSTEM_CONTROL
    rcd.INTEGRATION_SYSTEM = INTEGRATION_SYSTEM_CONTROL
    rcd.r1_user = r1_user_noedge
    rcd.r2_user = r2_user_noedge
    rcd.moderator_role_id = lambda m, _ns=namespace: moderator_role_id(m, _ns)
    NAMESPACE = namespace
    SEAT_MODEL_BY_ROLE = dict(zip(rcd.ROLE_ORDER, seat_models))
    R3R4_CAP_BY_MODEL = resolve_r3r4_caps(seat_models, r3r4_caps)
    SCREEN_MODELS = tuple(screen_models) or tuple(seat_models)
    rcd.do_call = _do_call_multivendor
    rcd.run_deliberation = _run_deliberation_multivendor
    rcd.filter_prone_items = _filter_prone_multivendor
    rcd.ROW_FIELDS = ROW_FIELDS_MV
    rcd.VOTE_FIELDS = VOTE_FIELDS_MV


# ---------------------------------------------------------------------------
# Cache namespaces (keyed by NAME): one pass over the directory
# ---------------------------------------------------------------------------

def _out_dir(out_dir: Optional[Path]) -> Path:
    """The directory call_cache_path actually writes to, resolved AT CALL
    TIME (rcd.OUT_DIR), never the import-time constant: the selftest and the
    base runner's --smoke redirect rcd.OUT_DIR, and a clean-cache check that
    looked at a different directory from the one the run writes to would
    pass on a dirty footprint (the exact failure the first draft of this
    module had)."""
    return rcd.OUT_DIR if out_dir is None else out_dir


def cache_namespace_counts(mod_model: str, *, namespace: str = "", tag: str = DEFAULT_TAG,
                           out_dir: Optional[Path] = None) -> dict[str, int]:
    """Files in every namespace this run writes to: the three seat ids (exact
    suffix), the moderator id (exact suffix -- the prefix alone would also
    match a --namespace sibling), and the tag's CSV / summary files (the
    dry-run's <tag>_dryrun.json is expected before a run and not counted)."""
    out_dir = _out_dir(out_dir)
    ids = seat_ids(namespace)
    mid = moderator_role_id(mod_model, namespace)
    counts = {**{rid: 0 for rid in ids}, mid: 0, f"tag:{tag}": 0}
    if not out_dir.is_dir():
        return counts
    suffixes = {rid: f"_{rid}.json" for rid in ids}
    suffixes[mid] = f"_{mid}.json"
    with os.scandir(out_dir) as it:
        for e in it:
            n = e.name
            if not n.startswith("cgd_") and not n.startswith(tag):
                continue
            for k, suf in suffixes.items():
                if n.endswith(suf):
                    counts[k] += 1
            if n.startswith(f"{tag}_") and not n.endswith("_dryrun.json"):
                counts[f"tag:{tag}"] += 1
    return counts


def cache_is_clean(mod_model: str, *, namespace: str = "", tag: str = DEFAULT_TAG,
                   out_dir: Optional[Path] = None) -> tuple[bool, dict[str, int]]:
    counts = cache_namespace_counts(mod_model, namespace=namespace, tag=tag, out_dir=out_dir)
    return sum(counts.values()) == 0, counts


def foreign_files(namespace: str = "", out_dir: Optional[Path] = None) -> list[str]:
    """Stub / selftest files carrying any of this runner's names -- must be []."""
    out_dir = _out_dir(out_dir)
    ids = seat_ids(namespace)
    pre = moderator_prefix(namespace)
    out = []
    if not out_dir.is_dir():
        return out
    with os.scandir(out_dir) as it:
        for e in it:
            n = e.name
            if not (any(f"_{rid}." in n for rid in ids) or f"_{pre}-" in n
                    or n.startswith(DEFAULT_TAG)):
                continue
            if "stub-model" in n or "selftest" in n or "_m_" in n:
                out.append(n)
    return out


# ---------------------------------------------------------------------------
# Cost model -- mixed prices, every seat at its own rate
# ---------------------------------------------------------------------------

def _completion(model: str, round_name: str, mod: bool) -> tuple[int, bool]:
    """(assumed completion tokens, measured?) for one round of one model."""
    if round_name in ("r0", "r1", "r2"):
        c = MEASURED_NCOT_COMPLETION.get(model)
        return (c, True) if c else (max(MEASURED_NCOT_COMPLETION.values()), False)
    if round_name in R3R4_ROUNDS:
        c = MEASURED_R3R4_COMPLETION.get(model)
        if c:
            return c[0 if round_name == "r3_label" else 1], True
        return ASSUMED_COMPLETION[round_name], False
    c = MEASURED_MODERATOR_COMPLETION.get(model)
    if c:
        return c[0 if round_name == "synthesis" else 1], True
    return ASSUMED_COMPLETION[round_name], False


def cost_model(seat_models: Sequence[str], moderator_model: str, arms: Sequence[str],
               items: Sequence, samples: int, *, caps: dict[str, int] = DEFAULT_CAPS,
               r3r4_caps: Optional[dict[str, int]] = None, transcript_cap: int = 0,
               use_bpe: bool = True) -> dict:
    """Exact call counts on real constructed prompts (the installed cell's
    own builders, so the no-edge r1/r2 and the stake-free preamble are what
    is counted), each seat's five rounds billed at that seat's PRICES rate
    and the two moderator rounds at the moderator's. Requires install()."""
    if not items:
        raise ValueError("cost model needs at least one item")
    r3r4 = resolve_r3r4_caps(seat_models, r3r4_caps) if r3r4_caps is not None or not R3R4_CAP_BY_MODEL else R3R4_CAP_BY_MODEL
    bpe = rcd._optional_bpe_counter() if use_bpe else None
    tok = (lambda s: rcd.count_tokens(s, tokenizer=bpe)) if bpe else rcd.count_tokens
    probe = sorted(items, key=lambda i: len(i.post_text))[len(items) // 2]
    arm0 = arms[0]
    roles = tuple(rcd.ROLES)
    order = tuple(rcd.ROLE_ORDER)
    cells = len(arms) * len(items) * samples
    calls_per_cell = len(roles) * len(AGENT_ROUNDS) + len(MODERATOR_ROUNDS)
    stand_in = {rid: "word " * _completion(m, "r0", False)[0]
                for rid, m in zip(order, seat_models)}
    synth_si = "word " * _completion(moderator_model, "synthesis", True)[0]
    prop_si = "word " * _completion(moderator_model, "integration", True)[0]
    obj_si = "word " * 40
    labels0 = {rid: "ACCEPT_WITH_MODIFICATION" for rid in order}
    objs0 = {rid: obj_si for rid in order}

    out: dict = {
        "panel": PANEL_LABEL, "seat_models": list(seat_models),
        "moderator_model": moderator_model, "arms": list(arms),
        "n_items": len(items), "samples": samples, "cells": cells,
        "calls_per_cell": calls_per_cell, "total_calls": cells * calls_per_cell,
        "replayed_calls": 0, "new_calls": cells * calls_per_cell,
        "replay_note": ("every seat id and the moderator prefix are this module's own, "
                        "so no call can be served from another run's cache: all "
                        "calls are new and all are counted"),
        "caps": dict(caps), "r3r4_cap_floor_by_model": dict(r3r4),
        "per_seat": {}, "moderator": {}, "priced": True,
        "assumptions": {
            "r0_r1_r2_completion": "MEASURED per model (rcd.MEASURED_NCOT_COMPLETION); "
                                   "fallback = max over measured models, flagged",
            "r3_r4_completion": "MEASURED per model at the cap that run used "
                                "(MEASURED_R3R4_COMPLETION, 2026-09-13); fallback "
                                "rcd.ASSUMED_COMPLETION, flagged",
            "moderator_completion": "MEASURED per model (MEASURED_MODERATOR_COMPLETION); "
                                    "fallback rcd.ASSUMED_COMPLETION, flagged",
            "prompt_tokens": "real constructed prompts of the installed cell with "
                             "stand-in prior-round text at the measured length",
            "tokenizer": ("tiktoken cl100k_base BPE" if bpe else
                          "regex approximation ONLY -- prompt tokens UNDERCOUNTED "
                          "by roughly 20-35% and so is the cost"),
            "prices": "run_crowdgold_deliberation.PRICES list rates, UNVERIFIED "
                      "against vendor pages",
        },
    }
    total = 0.0
    for rid, role, m in zip(order, roles, seat_models):
        pin, pout = PRICES.get(m, (0.0, 0.0))
        priced = m in PRICES
        prompts = {
            "r0": rcd.r0_user(arm0, probe, role, allow_unresolved=True),
            "r1": rcd.r1_user(arm0, probe, role, stand_in[rid], stand_in,
                              allow_unresolved=True, cap=transcript_cap),
            "r2": rcd.r2_user(arm0, probe, role, stand_in[rid], stand_in[rid], stand_in,
                              allow_unresolved=True, cap=transcript_cap),
            "r3_label": rcd.r3_label_user(arm0, probe, role, stand_in[rid], synth_si,
                                          cap=transcript_cap),
            "r4_vote": rcd.r4_vote_user(arm0, probe, role, synth_si,
                                        "ACCEPT_WITH_MODIFICATION", obj_si, prop_si,
                                        cap=transcript_cap),
        }
        rounds: dict = {}
        tin = tout = 0
        usd = 0.0
        measured_all = True
        for rd in AGENT_ROUNDS:
            comp, measured = _completion(m, rd, False)
            measured_all &= measured
            ptok = tok(prompts[rd])
            cap = seat_cap(m, rd, caps["agent"] if rd in ("r0", "r1", "r2")
                           else caps["label"] if rd == "r3_label" else caps["vote"], r3r4)
            i_, o_ = ptok * cells, comp * cells
            u = i_ / 1e6 * pin + o_ / 1e6 * pout
            rounds[rd] = {"calls": cells, "billed_to": m, "max_tokens": cap,
                          "prompt_tokens_each": ptok, "completion_tokens_each": comp,
                          "completion_measured": measured, "input_tokens": i_,
                          "output_tokens": o_, "usd": round(u, 2)}
            tin += i_
            tout += o_
            usd += u
        out["per_seat"][rid] = {
            "model": m, "priced": priced, "price_in_per_mtok": pin,
            "price_out_per_mtok": pout, "completion_measured": measured_all,
            "anthropic": bool(is_anthropic(m)),
            "r3r4_cap_floor": r3r4.get(m), "calls": cells * len(AGENT_ROUNDS),
            "input_tokens": tin, "output_tokens": tout, "usd": round(usd, 2),
            "rounds": rounds,
        }
        out["priced"] &= priced
        total += usd
    # moderator
    mm = moderator_model
    pin, pout = PRICES.get(mm, (0.0, 0.0))
    mprompts = {
        "synthesis": rcd.synthesis_user(arm0, probe, stand_in, allow_unresolved=True,
                                        cap=transcript_cap),
        "integration": rcd.integration_user(arm0, probe, synth_si, labels0, objs0,
                                            allow_unresolved=True, cap=transcript_cap),
    }
    rounds = {}
    tin = tout = 0
    usd = 0.0
    measured_all = True
    for rd in MODERATOR_ROUNDS:
        comp, measured = _completion(mm, rd, True)
        measured_all &= measured
        ptok = tok(mprompts[rd])
        i_, o_ = ptok * cells, comp * cells
        u = i_ / 1e6 * pin + o_ / 1e6 * pout
        rounds[rd] = {"calls": cells, "billed_to": mm, "max_tokens": caps["moderator"],
                      "prompt_tokens_each": ptok, "completion_tokens_each": comp,
                      "completion_measured": measured, "input_tokens": i_,
                      "output_tokens": o_, "usd": round(u, 2)}
        tin += i_
        tout += o_
        usd += u
    out["moderator"] = {
        "model": mm, "priced": mm in PRICES, "price_in_per_mtok": pin,
        "price_out_per_mtok": pout, "completion_measured": measured_all,
        "calls": cells * len(MODERATOR_ROUNDS), "input_tokens": tin,
        "output_tokens": tout, "usd": round(usd, 2), "rounds": rounds,
    }
    out["priced"] &= mm in PRICES
    total += usd
    out["total_usd"] = round(total, 2)
    return out


def print_cost_model(cm: dict) -> None:
    print("\n" + "=" * 78)
    print(f"COST MODEL (dry run) -- cross-vendor decider layer, panel {cm['panel']}")
    print("=" * 78)
    print(f"seats                  {cm['seat_models']}")
    print(f"moderator              {cm['moderator_model']}")
    print(f"arms                   {cm['arms']}   items {cm['n_items']}   samples {cm['samples']}")
    print(f"cells                  {cm['cells']:,}")
    print(f"calls per cell         {cm['calls_per_cell']}  (3 seats x r0,r1,r2,r3,r4 = 15, "
          f"synthesis + integration = 2)")
    print(f"TOTAL GENERATION CALLS {cm['total_calls']:,}   new {cm['new_calls']:,}   "
          f"replayed from cache {cm['replayed_calls']:,}")
    print(f"caps                   agent {cm['caps']['agent']} moderator {cm['caps']['moderator']} "
          f"label {cm['caps']['label']} vote {cm['caps']['vote']}; R3/R4 floor by model "
          f"{cm['r3r4_cap_floor_by_model'] or '(none)'}")
    blocks = list(cm["per_seat"].items()) + [("moderator", cm["moderator"])]
    for rid, v in blocks:
        flags = ("" if v["completion_measured"] else "   [completion length ASSUMED]") + \
                ("" if v["priced"] else "   [NO PRICE ON FILE]")
        print(f"\n  {rid}: {v['model']}  ({v['price_in_per_mtok']}/{v['price_out_per_mtok']} "
              f"per Mtok){flags}")
        print(f"    {'round':<12}{'calls':>8}{'cap':>7}{'prompt tok':>12}{'compl tok':>11}"
              f"{'in tok':>12}{'out tok':>12}{'usd':>9}")
        for rd, r in v["rounds"].items():
            print(f"    {rd:<12}{r['calls']:>8,}{r['max_tokens']:>7,}{r['prompt_tokens_each']:>12,}"
                  f"{r['completion_tokens_each']:>11,}{r['input_tokens']:>12,}"
                  f"{r['output_tokens']:>12,}{r['usd']:>9.2f}"
                  f"{'' if r['completion_measured'] else '  (assumed)'}")
        print(f"    {'TOTAL':<12}{v['calls']:>8,}{'':>7}{'':>12}{'':>11}{v['input_tokens']:>12,}"
              f"{v['output_tokens']:>12,}{v['usd']:>9.2f}")
    print(f"\n  TOTAL ESTIMATED COST  ${cm['total_usd']:.2f}")
    if not cm["priced"]:
        print("  *** WARNING: a model here has no PRICES entry; the $ figure above is a "
              "placeholder, not a ceiling. Add it to run_crowdgold_deliberation.PRICES "
              "before spending. ***")
    for k, val in cm["assumptions"].items():
        print(f"    assumption[{k}]: {val}")


# ---------------------------------------------------------------------------
# Selftest (offline; generate_any stubbed; cache redirected to a tempdir)
# ---------------------------------------------------------------------------

class _Stub:
    def __init__(self, text, finish_reason="stop", ptok=50, ctok=50):
        self.text = text
        self.finish_reason = finish_reason
        self.prompt_tokens = ptok
        self.completion_tokens = ctok


#: generations issued by _stub_generate since the last reset (the --resume
#: replay check reads it: a second pass over an intact cache must add 0)
_STUB_CALLS = [0]


def _stub_generate(model, system, user, *, sample_idx=0, max_tokens=0, **kw):
    """Deterministic offline stand-in keyed on the MODEL (the point of the
    cell: the model identifies the seat). Planted: the Anthropic seat says
    YTA, the nano seat NTA, the grok seat alternates by prompt hash; at R3
    the Anthropic seat asks for a modification and the nano seat rejects,
    grok accepts; the integration addresses the first requester; R4 votes
    follow the planted defeasibility rule."""
    _STUB_CALLS[0] += 1
    h = int(hashlib.sha1(user.encode()).hexdigest()[:8], 16)
    anth, nano = is_anthropic(model), "nano" in model
    if "This is the final round and your vote is binary" in user:
        if "You labelled it: ACCEPT" in user and "ACCEPT_WITH" not in user.split("You labelled it:", 1)[1][:30]:
            return _Stub("Fine as it stands.\n\nVERDICT: ACCEPT", ctok=12)
        addressed = rcd.marker_line(user.split("--- INTEGRATED PROPOSAL ---", 1)[1],
                                    rcd.ADDRESSED_MARKER)
        mine = [rid for rid in rcd.ROLE_ORDER if f"speaking as {rcd.ROLE_BY_ID[rid].label}" in user]
        ok = bool(mine) and mine[0] in addressed
        return _Stub(("Addressed.\n\nVERDICT: ACCEPT" if ok else
                      "Still unaddressed.\n\nVERDICT: REJECT"), ctok=12)
    if "Construct one integrated proposal" in user:
        asked = [rid for rid in rcd.ROLE_ORDER
                 if f"({rid})" in user and "label: ACCEPT_WITH_MODIFICATION" in
                 user.split(f"({rid})", 1)[1][:200]]
        return _Stub(f"PROPOSAL: the shared reading.\n\n{rcd.ADDRESSED_MARKER} "
                     f"{asked[0] if asked else 'none'}\n\nVERDICT: YTA", ctok=40)
    if "Write one synthesis position" in user:
        return _Stub("SYNTHESIS: the shared reading.\n\nVERDICT: YTA", ctok=30)
    if "Respond to this synthesis" in user:
        if anth:
            return _Stub(f"One change.\n{rcd.MOD_MARKER} name the constraint.\n\n"
                         "VERDICT: ACCEPT_WITH_MODIFICATION", ctok=25)
        if nano:
            return _Stub(f"No.\n{rcd.UNRESOLVABLE_MARKER} the account is misread.\n\n"
                         "VERDICT: REJECT", ctok=25)
        return _Stub("Reads correctly.\n\nVERDICT: ACCEPT", ctok=10)
    v = "YTA" if anth else "NTA" if nano else ("YTA" if h % 2 else "NTA")
    body = ("Section 1 - Protagonist: the writer.\n"
            "Section 2 - Stakeholders: writer, other party, onlookers.\n"
            "Section 3 - Consequences: two steps forward for each.\n"
            "Section 4 - Uncertainty: what the account does not say.\n"
            "Section 5 - Decision: stated plainly.\n"
            f"VERDICT: {v}")
    return _Stub(body, ctok=120)


def _synthetic_items(n: int = 4) -> list:
    out = []
    for i in range(n):
        gold = "YTA" if i % 2 == 0 else "NTA"
        text = (f"I told my sibling {i} times that the plan would not work and "
                f"then went ahead without them. They say I should have waited. "
                f"Item {i} of the synthetic panel, sentence {'long ' * (i + 1)}end.")
        out.append(rcd.CrowdGoldItem(item_id=f"syn{i:02d}", post_text=text,
                                     gold_verdict=gold, n_votes=100, consensus=0.95))
    return out


def _selftest() -> int:
    import contextlib
    import io
    import tempfile

    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    # ---- seats, briefs, ids ---------------------------------------------------
    R = build_roles()
    check("three seats", len(R) == 3)
    check("every seat carries the embodied neutral brief byte-for-byte, stake none",
          all(r.brief == _NEUTRAL_BRIEF and r.stake == "none" for r in R))
    check("labels are the (identical, off) cell's Reader Alpha / Beta / Gamma",
          tuple(r.label for r in R) == SEAT_LABELS
          == tuple(r.label for r in topology_roles("identical", "off")))
    check("no brief contains a blank line", all(POST_DELIM not in r.brief for r in R))
    taken = {r.role_id for r in _ORIG_ROLES} | {r.role_id for r in UNEMBODIED_ROLES}
    for cell in TOPOLOGY_CELLS:
        taken |= {r.role_id for r in topology_roles(*cell)}
    mine = {r.role_id for r in R}
    check(f"seat ids {sorted(mine)} collide with no embodied / unembodied / topology id",
          not (mine & taken))
    check("--namespace gives distinct ids and a distinct moderator prefix",
          not (set(seat_ids("b")) & mine)
          and moderator_prefix("b") != moderator_prefix("")
          and moderator_role_id("grok-4-1-fast-reasoning", "b")
          != moderator_role_id("grok-4-1-fast-reasoning", ""))
    prefixes = {c["mod"] for c in TOPOLOGY_CELLS.values()} | {UNEMBODIED_PREFIX, "mod"}
    check(f"moderator prefix {MODERATOR_PREFIX_BASE!r} is distinct from {sorted(prefixes)}",
          MODERATOR_PREFIX_BASE not in prefixes
          and not any(p.startswith(MODERATOR_PREFIX_BASE) for p in prefixes))
    # r0_user reads role.label and role.brief, never role.role_id, so with the
    # same preamble installed a seat's R0 prompt equals the (identical, off)
    # cell's for the same position byte for byte; likewise the no-edge r1/r2
    # over the same own text. Checked on the installed builders below (after
    # install()), and here independently of any installation.
    _it = rcd.CrowdGoldItem(item_id="X", post_text="[POST]", gold_verdict="YTA")
    _top = topology_roles("identical", "off")
    check("a seat's R0 / r1 / r2 prompt is byte-identical to the (identical, off) cell's "
          "for the same position (only the model differs)",
          all(_r0_with_preamble(arm, _it, a) == _r0_with_preamble(arm, _it, b)
              and r1_user_noedge(arm, _it, a, "[R0]", {}, allow_unresolved=True, cap=0)
              == r1_user_noedge(arm, _it, b, "[R0]", {}, allow_unresolved=True, cap=0)
              and r2_user_noedge(arm, _it, a, "[R0]", "[R1]", {}, allow_unresolved=True, cap=0)
              == r2_user_noedge(arm, _it, b, "[R0]", "[R1]", {}, allow_unresolved=True, cap=0)
              for a, b in zip(R, _top) for arm in DEFAULT_ARMS)
          and all(a.role_id != b.role_id for a, b in zip(R, _top)))

    # the two moderator-facing prompts embed the seat ids: assert the diff is
    # exactly the three id tokens (rcd.synthesis_user's "(<rid>)" labels and
    # rcd.integration_user's "from:" list) and nothing else
    _texts_top = {r.role_id: f"[FINAL {i}]" for i, r in enumerate(_top)}
    _texts_mine = {r.role_id: f"[FINAL {i}]" for i, r in enumerate(R)}
    _labels_top = {r.role_id: "ACCEPT_WITH_MODIFICATION" for r in _top}
    _labels_mine = {r.role_id: "ACCEPT_WITH_MODIFICATION" for r in R}
    _objs_top = {r.role_id: f"[OBJ {i}]" for i, r in enumerate(_top)}
    _objs_mine = {r.role_id: f"[OBJ {i}]" for i, r in enumerate(R)}
    _id_map = {b.role_id: a.role_id for a, b in zip(R, _top)}

    def _subst(s: str) -> str:
        for old, new in _id_map.items():
            s = s.replace(old, new)
        return s
    _saved = (rcd.ROLES, rcd.ROLE_ORDER, rcd.ROLE_BY_ID)
    try:
        rcd.ROLES, rcd.ROLE_ORDER = _top, tuple(r.role_id for r in _top)
        rcd.ROLE_BY_ID = {r.role_id: r for r in _top}
        syn_top = rcd.synthesis_user(DEFAULT_ARMS[0], _it, _texts_top, allow_unresolved=True, cap=0)
        int_top = rcd.integration_user(DEFAULT_ARMS[0], _it, "[SYN]", _labels_top, _objs_top,
                                       allow_unresolved=True, cap=0)
        rcd.ROLES, rcd.ROLE_ORDER = R, tuple(r.role_id for r in R)
        rcd.ROLE_BY_ID = {r.role_id: r for r in R}
        syn_mine = rcd.synthesis_user(DEFAULT_ARMS[0], _it, _texts_mine, allow_unresolved=True, cap=0)
        int_mine = rcd.integration_user(DEFAULT_ARMS[0], _it, "[SYN]", _labels_mine, _objs_mine,
                                        allow_unresolved=True, cap=0)
    finally:
        rcd.ROLES, rcd.ROLE_ORDER, rcd.ROLE_BY_ID = _saved
    check("the moderator-facing synthesis / integration prompts differ from the (identical, off) "
          "cell's by the three seat-id tokens ONLY (equal after substituting the ids; no vendor "
          "name in either), and they do differ before it",
          syn_mine != syn_top and int_mine != int_top
          and _subst(syn_top) == syn_mine and _subst(int_top) == int_mine
          and not any(m in syn_mine or m in int_mine for m in DEFAULT_SEAT_MODELS)
          and all(f"({rid})" in syn_mine for rid in _texts_mine))

    # ---- seat models, caps ------------------------------------------------------
    check("default seat models are three different priced models",
          validate_seat_models(DEFAULT_SEAT_MODELS) is None
          and all(m in PRICES for m in DEFAULT_SEAT_MODELS) and DEFAULT_MODERATOR in PRICES)
    check("duplicate or short seat lists are refused",
          validate_seat_models(("a", "a", "b")) is not None
          and validate_seat_models(("a", "b")) is not None)
    caps = resolve_r3r4_caps(DEFAULT_SEAT_MODELS)
    check("Anthropic seat gets the 3072 R3/R4 floor; nano and grok keep the registered caps",
          caps == {"claude-haiku-4-5": ANTHROPIC_R3R4_CAP}
          and seat_cap("claude-haiku-4-5", "r3_label", 1024, caps) == 3072
          and seat_cap("claude-haiku-4-5", "r4_vote", 512, caps) == 3072
          and seat_cap("claude-haiku-4-5", "r0", 2560, caps) == 2560
          and seat_cap("gpt-5.4-nano", "r3_label", 1024, caps) == 1024
          and seat_cap("grok-4-1-fast-reasoning", "r4_vote", 512, caps) == 512)
    check("a larger command-line cap is kept for the Anthropic seat (floor, not override)",
          seat_cap("claude-haiku-4-5", "r4_vote", 4096, caps) == 4096)
    ov = parse_cap_overrides("new-vendor-x=2048,claude-haiku-4-5=1536")
    caps2 = resolve_r3r4_caps(("claude-haiku-4-5", "new-vendor-x", "grok-4-1-fast-reasoning"), ov)
    check("--seat-r3r4-caps adds a cap for an unknown deployment and overrides the Anthropic floor",
          caps2 == {"claude-haiku-4-5": 1536, "new-vendor-x": 2048})
    check("is_anthropic agrees with the generators' routing prefix",
          is_anthropic("claude-haiku-4-5") and is_anthropic("claude-sonnet-4-6")
          and not is_anthropic("gpt-5.4-nano") and not is_anthropic("grok-4-1-fast-reasoning"))

    # ---- install(): totality, cache-key separation ------------------------------
    install(DEFAULT_SEAT_MODELS)
    check("install() swaps ROLE_ORDER to the three seat ids",
          rcd.ROLE_ORDER == seat_ids())
    check("install() keeps CALLS_PER_CELL at 17 and N_AGENTS at 3",
          rcd.CALLS_PER_CELL == 17 and rcd.N_AGENTS == 3)
    check("install() puts the 16.8 stake-free preamble and moderator prompts in place",
          rcd.R0_PREAMBLE == R0_PREAMBLE_CONTROL and "stakes" not in rcd.R0_PREAMBLE
          and rcd.SYNTHESIS_SYSTEM == SYNTHESIS_SYSTEM_CONTROL
          and rcd.INTEGRATION_SYSTEM == INTEGRATION_SYSTEM_CONTROL)
    check("install() puts the no-edge r1/r2 in place (imported, not retyped)",
          rcd.r1_user is r1_user_noedge and rcd.r2_user is r2_user_noedge)
    check("after install(), the INSTALLED r0 builder gives each seat the (identical, off) "
          "cell's prompt for its position, byte for byte",
          all(rcd.r0_user(arm, _it, a, allow_unresolved=True)
              == rcd.r0_user(arm, _it, b, allow_unresolved=True)
              == _r0_with_preamble(arm, _it, b)
              for a, b in zip(rcd.ROLES, _top) for arm in DEFAULT_ARMS))
    check("install() maps seat -> model in order",
          SEAT_MODEL_BY_ROLE == dict(zip(seat_ids(), DEFAULT_SEAT_MODELS)))
    check("install() widens the CSV schemas with the seat model and per-seat verdicts",
          rcd.ROW_FIELDS is ROW_FIELDS_MV and rcd.VOTE_FIELDS is VOTE_FIELDS_MV
          and "seat_model" in VOTE_FIELDS_MV and "seat_models" in ROW_FIELDS_MV)
    check("install() is idempotent (no double suffix on a second call)",
          (install(DEFAULT_SEAT_MODELS), rcd.ROLE_ORDER == seat_ids())[1])
    k_new = rcd.call_cache_path("grok-4-1-fast-reasoning", "narrative_cot", "as_asker",
                                "itemX", 0, "r1", seat_ids()[2], 2560)
    k_top = rcd.call_cache_path("grok-4-1-fast-reasoning", "narrative_cot", "as_asker",
                                "itemX", 0, "r1", "reader_gamma_noedge", 2560)
    check("the grok seat's cache path differs from the (identical, off) cell's "
          "(so a regenerated r1 can never overwrite a 16.10 record)",
          k_new != k_top and seat_ids()[2] in k_new.name)
    km = rcd.call_cache_path("grok-4-1-fast-reasoning", "narrative_cot", "as_asker",
                             "itemX", 0, "synthesis",
                             rcd.moderator_role_id("grok-4-1-fast-reasoning"), 1024)
    check("moderator cache path carries modmultivendor and not mod- / modidentnoedge-",
          f"_{MODERATOR_PREFIX_BASE}-grok" in km.name and "_mod-grok" not in km.name
          and "modidentnoedge" not in km.name)

    # ---- namespaces empty, no foreign file ----------------------------------------
    clean, counts = cache_is_clean(DEFAULT_MODERATOR)
    check(f"every namespace EMPTY in {OUT_DIR.name} {counts} (or holds only a registered "
          f"run's footprint, in which case --run refuses without --resume)",
          clean or all(v > 0 for v in counts.values()))
    ff = foreign_files()
    check(f"no foreign (stub / selftest) file carries this runner's names ({ff[:3]})", ff == [])

    # ---- smoke: full protocol offline, stub generator, cache in a tempdir ----------
    real_gen, real_out = rcd.generate_any, rcd.OUT_DIR
    items = _synthetic_items(2)
    rows: list = []
    votes: list = []
    calls: list = []
    names: list[str] = []
    try:
        rcd.generate_any = _stub_generate  # type: ignore[assignment]
        with tempfile.TemporaryDirectory() as td:
            rcd.OUT_DIR = Path(td)
            try:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rows, votes, calls = rcd.run(
                        [PANEL_LABEL], list(DEFAULT_ARMS), items, samples=1,
                        scaffold="narrative_cot", moderator_model=DEFAULT_MODERATOR,
                        max_tokens_agent=DEFAULT_CAPS["agent"],
                        max_tokens_moderator=DEFAULT_CAPS["moderator"],
                        max_tokens_label=DEFAULT_CAPS["label"],
                        max_tokens_vote=DEFAULT_CAPS["vote"],
                        allow_unresolved=True, transcript_cap=0, workers=1)
                names = sorted(p.name for p in Path(td).iterdir())
            finally:
                rcd.OUT_DIR = real_out
    finally:
        rcd.generate_any = real_gen
    check("smoke: 4 debates (2 items x 2 arms), 17 calls each, one cache file per call "
          "in the TEMP dir", len(rows) == 4 and len(calls) == 68 and len(names) == 68)
    check("smoke: nothing landed in the real cache directory",
          foreign_files() == [] and cache_is_clean(DEFAULT_MODERATOR)[1] == counts)
    s1, s2, s3 = seat_ids()
    check("smoke: each seat's calls are cached under ITS model and ITS role id",
          any(n.startswith("cgd_claude-haiku-4-5_narrative_cot_t2560_") and n.endswith(f"_r0_{s1}.json") for n in names)
          and any(n.startswith("cgd_gpt-5.4-nano_narrative_cot_t2560_") and n.endswith(f"_r0_{s2}.json") for n in names)
          and any(n.startswith("cgd_grok-4-1-fast-reasoning_narrative_cot_t2560_") and n.endswith(f"_r0_{s3}.json") for n in names)
          and not any(f"_{PANEL_LABEL}_" in n for n in names))
    check("smoke: the Anthropic seat's R3/R4 records are at t3072, nano/grok at t1024/t512, r0-r2 at t2560",
          any(n.startswith("cgd_claude-haiku-4-5_narrative_cot_t3072_") and n.endswith(f"_r3_label_{s1}.json") for n in names)
          and any(n.startswith("cgd_claude-haiku-4-5_narrative_cot_t3072_") and n.endswith(f"_r4_vote_{s1}.json") for n in names)
          and any(n.startswith("cgd_gpt-5.4-nano_narrative_cot_t1024_") and n.endswith(f"_r3_label_{s2}.json") for n in names)
          and any(n.startswith("cgd_grok-4-1-fast-reasoning_narrative_cot_t512_") and n.endswith(f"_r4_vote_{s3}.json") for n in names)
          and not any("_t3072_" in n and s1 not in n for n in names))
    mid = moderator_role_id(DEFAULT_MODERATOR)
    check("smoke: moderator rounds cached under the moderator model, t1024, modmultivendor id",
          sum(1 for n in names if n.startswith("cgd_grok-4-1-fast-reasoning_narrative_cot_t1024_")
              and n.endswith(f"_{mid}.json")) == 8)
    check("smoke: rows carry the panel label as model, the seat models, and per-seat verdicts",
          all(r["model"] == PANEL_LABEL and r["seat_models"] == "|".join(DEFAULT_SEAT_MODELS)
              and r["r0_verdicts"].count("|") == 2 and r["r0_verdicts"].split("|")[0] == "YTA"
              and r["r0_verdicts"].split("|")[1] == "NTA" for r in rows))
    check("smoke: every vote row carries its seat's model, index and r0 verdict",
          len(votes) == 12 and all(v["seat_model"] == SEAT_MODEL_BY_ROLE[v["role_id"]]
                                    and v["seat_index"] in (1, 2, 3)
                                    and v["r0_verdict"] in ("YTA", "NTA") for v in votes))
    check("smoke: planted R3 pattern (haiku AWM, nano REJECT, grok ACCEPT) -> n_objectors 2, "
          "addressed = the haiku seat",
          all(r["r3_labels"] == "ACCEPT_WITH_MODIFICATION|REJECT|ACCEPT" and r["n_objectors"] == 2
              and r["addressed_roles"] == s1 for r in rows))
    check("smoke: planted defeasibility -> haiku revises (ACCEPT), nano still rejects",
          all(v["vote"] == ("ACCEPT" if v["role_id"] != s2 else "REJECT") for v in votes))
    check("smoke: stakes none everywhere, stake_undermined blank-or-0, group verdict codable",
          all(v["role_stake"] == "none" and v["stake_undermined"] in ("", 0) for v in votes)
          and all(r["verdict"] == "YTA" for r in rows))
    guard = rcd.truncation_report([rcd.call_guard_row(c) for c in calls])
    check("smoke: the imported round-level guard passes and reports per SEAT MODEL",
          guard["pass"] and any(k.startswith("claude-haiku-4-5|") for k in guard["cells"])
          and any(k.startswith("gpt-5.4-nano|") for k in guard["cells"]))
    cm_calls = sum(int(c.get("prompt_tokens", 0) or 0) > 0 for c in calls)
    check("smoke: 17 records per debate carry token counters (billable)", cm_calls == 68)
    check("smoke: rows / votes serialise under the widened schemas",
          (lambda td: (rcd.write_csv(rows, Path(td) / "r.csv", ROW_FIELDS_MV),
                       rcd.write_csv(votes, Path(td) / "v.csv", VOTE_FIELDS_MV),
                       "seat_model" in (Path(td) / "v.csv").read_text().splitlines()[0]
                       and "seat_models" in (Path(td) / "r.csv").read_text().splitlines()[0])[2])
          (tempfile.mkdtemp()))

    # ---- GUARD FAILED literal on the imported guard -------------------------------
    bad = [{"model": "m", "scaffold": "s", "arm": "a", "verdict": rcd.NOVERDICT if i < 2 else "YTA",
            "output_len": 10, "finish_reason": "stop", "truncated": 0} for i in range(20)]
    rep = rcd.truncation_report(bad)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rcd.print_truncation_report(rep)
    check("imported guard fails at 10% NOVERDICT and prints GUARD FAILED",
          not rep["pass"] and "GUARD FAILED" in buf.getvalue())

    # ---- dry-run cost model: mixed prices, all calls new -------------------------
    cm = cost_model(DEFAULT_SEAT_MODELS, DEFAULT_MODERATOR, list(DEFAULT_ARMS),
                    _synthetic_items(5), 1, use_bpe=False)
    check("dry-run: 5 items x 2 arms = 10 cells x 17 = 170 calls, all new, 0 replayed",
          cm["cells"] == 10 and cm["total_calls"] == 170 and cm["new_calls"] == 170
          and cm["replayed_calls"] == 0)
    ps = cm["per_seat"]
    check("dry-run: each seat billed at ITS OWN PRICES rate, the moderator at its own",
          all(ps[rid]["price_out_per_mtok"] == PRICES[m][1] and
              all(r["billed_to"] == m for r in ps[rid]["rounds"].values())
              for rid, m in zip(seat_ids(), DEFAULT_SEAT_MODELS))
          and cm["moderator"]["price_out_per_mtok"] == PRICES[DEFAULT_MODERATOR][1])
    check("dry-run: total = sum of seat totals + moderator (mixed prices)",
          abs(cm["total_usd"] - round(sum(v["usd"] for v in ps.values()) + cm["moderator"]["usd"], 2)) < 0.02
          and cm["priced"])
    check("dry-run: the haiku seat's r3/r4 lines carry cap 3072 and its measured 565/1026 completions",
          ps[s1]["rounds"]["r3_label"]["max_tokens"] == 3072
          and ps[s1]["rounds"]["r4_vote"]["max_tokens"] == 3072
          and ps[s1]["rounds"]["r3_label"]["completion_tokens_each"] == 565
          and ps[s1]["rounds"]["r4_vote"]["completion_tokens_each"] == 1026
          and ps[s3]["rounds"]["r4_vote"]["max_tokens"] == 512)
    check("dry-run: haiku seat costs more than the grok seat at equal call count",
          ps[s1]["usd"] > ps[s3]["usd"] and ps[s1]["calls"] == ps[s3]["calls"] == 50)
    install(("claude-haiku-4-5", "gpt-5.4-nano", "no-such-model"))
    cm2 = cost_model(("claude-haiku-4-5", "gpt-5.4-nano", "no-such-model"), DEFAULT_MODERATOR,
                     list(DEFAULT_ARMS), _synthetic_items(2), 1, use_bpe=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_cost_model(cm2)
    check("dry-run flags an unpriced seat model instead of reporting $0 silently",
          not cm2["priced"] and "NO PRICE ON FILE" in buf.getvalue()
          and "WARNING" in buf.getvalue())
    install(DEFAULT_SEAT_MODELS)

    # ---- CLI: nothing generates without --run ---------------------------------------
    real_load = rcd.load_items
    real_gen, real_out = rcd.generate_any, rcd.OUT_DIR
    try:
        rcd.load_items = lambda **kw: _synthetic_items(6)  # type: ignore[assignment]
        rcd.generate_any = _stub_generate  # type: ignore[assignment]
        with tempfile.TemporaryDirectory() as td:
            rcd.OUT_DIR = Path(td)
            try:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = main(["--no-bpe"])
                check("main() without --run prints the cost model and REFUSES (exit 2)",
                      rc == 2 and "COST MODEL" in buf.getvalue() and "--run" in buf.getvalue())
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = main(["--dry-run", "--no-bpe", "--samples", "2"])
                check("main(--dry-run) exits 0, 6 items x 2 arms x 2 samples = 24 cells, writes "
                      "<tag>_dryrun.json in the (redirected) out dir",
                      rc == 0 and "cells                  24" in buf.getvalue()
                      and (Path(td) / f"{DEFAULT_TAG}_dryrun.json").exists())
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = main(["--run", "--no-bpe", "--seat-models",
                               "claude-haiku-4-5,gpt-5.4-nano,no-such-model"])
                check("main(--run) refuses an unpriced seat model (exit 2) before any call",
                      rc == 2 and "PRICES" in buf.getvalue()
                      and not any(p.name.startswith("cgd_") for p in Path(td).iterdir()))
                for flag in ("--stake-nudge", "--r3r4-thinking-budget=4096", "--models=x",
                             "--pilot", "--r3r4-reasoning-effort", "--smoke"):
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        rc = main(["--run", "--no-bpe", flag])
                    check(f"main(--run {flag}) is refused (exit 2)", rc == 2 and "ERROR" in buf.getvalue())
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = main(["--run", "--no-bpe", "--seat-models", "a,a,b"])
                check("main(--run) refuses duplicate seat models (exit 2)", rc == 2)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = main(["--run", "--no-bpe", "--tag", "cg_deliberation_identical"])
                check("main(--run) refuses a --tag that is another cell's (exit 2)", rc == 2)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = main(["--run", "--no-bpe", "--namespace", "b"])
                check("main(--run --namespace b) without its own --tag is refused (exit 2)",
                      rc == 2 and "--tag" in buf.getvalue())
                # plant one record in a seat namespace -> the run path must refuse
                planted = Path(td) / f"cgd_claude-haiku-4-5_narrative_cot_t2560_as_asker_syn00_00_r0_{s1}.json"
                planted.write_text("{}")
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = main(["--run", "--no-bpe"])
                check("main(--run) refuses a non-empty seat namespace without --resume (exit 3)",
                      rc == 3 and "not clean" in buf.getvalue())
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = main(["--verify-cache-clean"])
                check("--verify-cache-clean reports DIRTY (exit 3) on the planted record", rc == 3)
                planted.unlink()
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = main(["--verify-cache-clean"])
                check("--verify-cache-clean reports CLEAN (exit 0) once it is gone", rc == 0)
                check("no CLI path above generated anything (no cgd_ file in the out dir)",
                      not any(p.name.startswith("cgd_") for p in Path(td).iterdir()))

                # ---- end to end: the DELEGATED path, stub generator, tempdir cache --
                # This is the only place the whole chain (wrapper main -> rcd.main
                # -> rcd.run -> patched run_deliberation -> patched do_call ->
                # guards -> readouts -> CSV / summary) is exercised offline.
                _STUB_CALLS[0] = 0
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = main(["--run", "--no-bpe", "--workers", "1", "--n-boot", "20"])
                out = buf.getvalue()
                rows_p = Path(td) / f"{DEFAULT_TAG}_rows.csv"
                votes_p = Path(td) / f"{DEFAULT_TAG}_votes.csv"
                summ_p = Path(td) / f"{DEFAULT_TAG}_summary.json"
                n_cgd = sum(1 for p in Path(td).iterdir() if p.name.startswith("cgd_"))
                check("end-to-end --run (stubbed) exits 0, delegates with --models multivendor, "
                      "passes the guard, writes rows / votes / summary under the tag",
                      rc == 0 and "--models multivendor" in out and "GUARD FAILED" not in out
                      and rows_p.exists() and votes_p.exists() and summ_p.exists())
                check("end-to-end: 6 items x 2 arms = 12 debates x 17 = 204 generations, "
                      "204 cache files, every one under a seat model or the moderator",
                      _STUB_CALLS[0] == 204 and n_cgd == 204
                      and not any(f"_{PANEL_LABEL}_" in p.name for p in Path(td).iterdir()
                                  if p.name.startswith("cgd_")))
                import csv as _csv
                with open(rows_p) as fh:
                    rrows = list(_csv.DictReader(fh))
                with open(votes_p) as fh:
                    vrows = list(_csv.DictReader(fh))
                check("end-to-end: rows CSV carries model=multivendor, seat_models, r0_verdicts; "
                      "votes CSV carries seat_model per seat",
                      len(rrows) == 12 and all(r["model"] == PANEL_LABEL
                                               and r["seat_models"] == "|".join(DEFAULT_SEAT_MODELS)
                                               and r["moderator_model"] == DEFAULT_MODERATOR
                                               and r["r0_verdicts"].count("|") == 2 for r in rrows)
                      and len(vrows) == 36
                      and {v["seat_model"] for v in vrows} == set(DEFAULT_SEAT_MODELS))
                summ = json.loads(summ_p.read_text())
                check("end-to-end: summary names the three seat ids, the panel label and the moderator",
                      [r["role_id"] for r in summ["roles"]] == list(seat_ids())
                      and summ["models"] == [PANEL_LABEL]
                      and summ["moderator_model"] == DEFAULT_MODERATOR)
                panel_p = Path(td) / f"{DEFAULT_TAG}_panel.json"
                pj = json.loads(panel_p.read_text()) if panel_p.exists() else {}
                check("end-to-end: the provenance sidecar <tag>_panel.json maps every seat to its "
                      "model and cap, names the moderator id and the screen",
                      [(x["role_id"], x["model"]) for x in pj.get("seats", [])]
                      == list(zip(seat_ids(), DEFAULT_SEAT_MODELS))
                      and pj["seats"][0]["r3r4_cap"] == 3072 and pj["seats"][2]["r4_cap"] == 512
                      and pj["moderator_role_id"] == moderator_role_id(DEFAULT_MODERATOR)
                      and pj["exit_code"] == 0)
                comp_p = Path(td) / f"{DEFAULT_TAG}_completeness.json"
                check("end-to-end: the completeness guard PASSES (12 launched, 12 rows, summary n_items "
                      "agrees) and is recorded in panel.json and in <tag>_completeness.json",
                      pj.get("completeness", {}).get("pass") is True
                      and pj["completeness"]["n_expected"] == 12 and pj["completeness"]["n_rows"] == 12
                      and pj["completeness"]["summary_n_items"] == 6
                      and comp_p.exists() and json.loads(comp_p.read_text())["pass"] is True
                      and "COMPLETENESS GUARD" in out)
                # the footprint is now dirty: --run must refuse, --resume must replay
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = main(["--run", "--no-bpe"])
                check("a second --run over the footprint just written is refused (exit 3)",
                      rc == 3 and "not clean" in buf.getvalue())
                _STUB_CALLS[0] = 0
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = main(["--run", "--resume", "--no-bpe", "--workers", "1", "--n-boot", "20"])
                check("--run --resume replays the intact cache: exit 0 and ZERO new generations",
                      rc == 0 and _STUB_CALLS[0] == 0
                      and sum(1 for p in Path(td).iterdir() if p.name.startswith("cgd_")) == 204)
                # ---- an errored debate in a second namespace: dropped by rcd.run, caught here ----
                syn_items = _synthetic_items(6)
                victim = syn_items[1]
                as_asker_lead = rcd.build_user_turn(rcd.AS_ASKER, "")[:40]

                def _erroring_stub(model, system, user, *, sample_idx=0, max_tokens=0, **kw):
                    if ("Respond to this synthesis" in user and as_asker_lead in user
                            and victim.post_text[:120] in user):
                        raise RuntimeError("simulated vendor outage on one as_asker r3 call")
                    return _stub_generate(model, system, user, sample_idx=sample_idx,
                                          max_tokens=max_tokens, **kw)
                rcd.generate_any = _erroring_stub  # type: ignore[assignment]
                tag_b = f"{DEFAULT_TAG}_b"
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc_e = main(["--run", "--no-bpe", "--workers", "1", "--n-boot", "20",
                                 "--namespace", "b", "--tag", tag_b])
                rcd.generate_any = _stub_generate  # type: ignore[assignment]
                out_e = buf.getvalue()
                names_e = sorted(p.name for p in Path(td).iterdir())
                pj_b = json.loads((Path(td) / f"{tag_b}_panel.json").read_text())
                check("an errored as_asker debate is DROPPED by rcd.run (11 rows for 12 launched): the "
                      "completeness guard fails, GUARD FAILED is printed, exit 4, panel.json records "
                      "exit_code 4 and the missing key",
                      rc_e == 4 and "GUARD FAILED" in out_e and pj_b["exit_code"] == 4
                      and pj_b["completeness"]["pass"] is False and pj_b["completeness"]["n_rows"] == 11
                      and pj_b["completeness"]["missing"] == [[rcd.AS_ASKER, victim.item_id, 0]]
                      and pj_b["completeness"]["per_arm"][rcd.AS_ASKER] == {"expected": 6, "rows": 5})
                check("on exit 4 the CSVs are quarantined as <tag>_GUARD_FAILED_rows/_votes.csv and "
                      "nothing is left under the readable tag",
                      f"{tag_b}_GUARD_FAILED_rows.csv" in names_e and f"{tag_b}_GUARD_FAILED_votes.csv" in names_e
                      and f"{tag_b}_rows.csv" not in names_e and f"{tag_b}_votes.csv" not in names_e)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc_f = main(["--run", "--resume", "--no-bpe", "--workers", "1", "--n-boot", "20",
                                 "--namespace", "b", "--tag", tag_b])
                pj_b2 = json.loads((Path(td) / f"{tag_b}_panel.json").read_text())
                check("--run --resume completes the missing debate from the intact cache: exit 0, 12 rows, "
                      "completeness PASS, CSVs back under the readable tag",
                      rc_f == 0 and pj_b2["completeness"]["pass"] is True and pj_b2["completeness"]["n_rows"] == 12
                      and (Path(td) / f"{tag_b}_rows.csv").exists())
                # the delegated argv is explicit and carries the panel label, moderator and tag
                argv = delegated_argv(argparse.Namespace(
                    n_yta=99, n_nta=150, samples=1, seed=44, min_votes=50, min_consensus=0.90,
                    keep_filtered=False, no_unresolved=False, transcript_cap=0,
                    max_tokens_agent=2560, max_tokens_moderator=1024, max_tokens_label=1024,
                    max_tokens_vote=512, workers=4, n_boot=1000, arms="third_person,as_asker",
                    agent_scaffold="narrative_cot", moderator_model=DEFAULT_MODERATOR,
                    tag=DEFAULT_TAG), [])
                check("delegated argv names --models <panel>, --moderator-model, --tag and the "
                      "registered panel / caps",
                      argv[argv.index("--models") + 1] == PANEL_LABEL
                      and argv[argv.index("--moderator-model") + 1] == DEFAULT_MODERATOR
                      and argv[argv.index("--tag") + 1] == DEFAULT_TAG
                      and argv[argv.index("--n-yta") + 1] == "99"
                      and argv[argv.index("--n-nta") + 1] == "150"
                      and argv[argv.index("--max-tokens-vote") + 1] == "512")
            finally:
                rcd.OUT_DIR = real_out
    finally:
        rcd.load_items = real_load  # type: ignore[assignment]
        rcd.generate_any = real_gen
    check("selftest left the real cache directory untouched",
          cache_is_clean(DEFAULT_MODERATOR)[1] == counts and foreign_files() == [])

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def _r0_with_preamble(arm, item, role) -> str:
    """R0 prompt under the stake-free preamble, independent of rcd's current
    R0_PREAMBLE (selftest helper for the byte-identity check)."""
    return (f"{R0_PREAMBLE_CONTROL}{rcd.request_block(arm, item.post_text)}\n\n"
            f"{rcd.GLOSSARY_CORE}\n\n{rcd._role_header(role)}\n\n"
            "Write your opening statement. State the position you hold on this "
            "request and why.\n\n"
            f"{rcd.verdict_instruction(rcd.INSTRUMENT, allow_unresolved=True)}")


# ---------------------------------------------------------------------------
# Completeness guard: every launched debate must be a row
# ---------------------------------------------------------------------------

def completeness_check(tag: str, expected: set, *, out_dir: Optional[Path] = None,
                       model: str = PANEL_LABEL, summary_n_items: Optional[int] = None) -> dict:
    """Compare <tag>_rows.csv (rows with model == the panel label) with the
    launched (arm, item, sample) keys; rcd.run drops an errored debate with
    one console line, so any shortfall is a guard failure."""
    import csv
    out_dir = rcd.OUT_DIR if out_dir is None else out_dir
    rows_path = out_dir / f"{tag}_rows.csv"
    present: set = set()
    if rows_path.exists():
        with rows_path.open() as f:
            for r in csv.DictReader(f):
                if r.get("model") != model:
                    continue
                present.add((r.get("arm", ""), r.get("item_id", ""), int(r.get("sample_idx") or 0)))
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    per: dict = {}
    for arm, _, _ in expected:
        per.setdefault(arm, {"expected": 0, "rows": 0})["expected"] += 1
    for arm, _, _ in present & expected:
        per[arm]["rows"] += 1
    exp_items = {k[1] for k in expected}
    agree = summary_n_items is None or summary_n_items == len(exp_items)
    return {"tag": tag, "model": model, "n_expected": len(expected), "n_rows": len(present),
            "n_missing": len(missing), "n_unexpected": len(extra),
            "missing": [list(k) for k in missing], "unexpected": [list(k) for k in extra],
            "per_arm": per, "n_items_expected": len(exp_items),
            "summary_n_items": summary_n_items, "summary_n_items_agrees": agree,
            "pass": bool(expected) and not missing and not extra and agree}


def print_completeness(c: dict) -> None:
    print("\nCOMPLETENESS GUARD (every launched debate must be a row; rcd.run drops an errored "
          "debate with one console line)")
    print(f"  expected {c['n_expected']} debates over {c['n_items_expected']} items, rows {c['n_rows']}, "
          f"missing {c['n_missing']}, unexpected {c['n_unexpected']}; "
          + "  ".join(f"{a}={v['rows']}/{v['expected']}" for a, v in c["per_arm"].items())
          + f"  {'PASS' if c['pass'] else 'FAIL'}")
    if c["summary_n_items"] is not None and not c["summary_n_items_agrees"]:
        print(f"  the runner's summary reports n_items {c['summary_n_items']} but this wrapper launched "
              f"{c['n_items_expected']}")
    if c["missing"]:
        print("  missing: " + ", ".join(f"{a}/{i[:8]}/{s}" for a, i, s in c["missing"][:12])
              + (" ..." if len(c["missing"]) > 12 else ""))
    if not c["pass"]:
        print("\n  *** GUARD FAILED (completeness): the surviving rows are arm-correlated attrition "
              "until the missing debates are re-run (--run --resume replays the cached calls). ***")


def quarantine_csvs(tag: str, out_dir: Optional[Path] = None) -> list[str]:
    """The base runner writes its CSVs before its guard verdict; move them out
    of the readable tag. Returns the new names."""
    out_dir = rcd.OUT_DIR if out_dir is None else out_dir
    moved = []
    for kind in ("rows", "votes"):
        p = out_dir / f"{tag}_{kind}.csv"
        if p.exists():
            q = out_dir / f"{tag}_GUARD_FAILED_{kind}.csv"
            p.rename(q)
            moved.append(q.name)
    return moved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

BANNED = ("--stake-nudge", "--stake-cot", "--stake-fewshot", "--stake-fewshot-set",
          "--r3r4-reasoning-effort", "--r3r4-thinking-budget", "--models", "--pilot",
          "--full", "--smoke", "--selftest", "--compare-rows")
OTHER_CELL_TAGS = {"cg_deliberation", "cg_deliberation_unembodied"} | {
    c["tag"] for c in TOPOLOGY_CELLS.values()}


def delegated_argv(a: argparse.Namespace, rest: Sequence[str]) -> list[str]:
    """The exact rcd.main command: every registered parameter explicit."""
    argv = ["--models", PANEL_LABEL, "--moderator-model", a.moderator_model,
            "--tag", a.tag, "--arms", a.arms, "--agent-scaffold", a.agent_scaffold,
            "--n-yta", str(a.n_yta), "--n-nta", str(a.n_nta), "--samples", str(a.samples),
            "--seed", str(a.seed), "--min-votes", str(a.min_votes),
            "--min-consensus", str(a.min_consensus),
            "--max-tokens-agent", str(a.max_tokens_agent),
            "--max-tokens-moderator", str(a.max_tokens_moderator),
            "--max-tokens-label", str(a.max_tokens_label),
            "--max-tokens-vote", str(a.max_tokens_vote),
            "--transcript-cap", str(a.transcript_cap), "--workers", str(a.workers),
            "--n-boot", str(a.n_boot)]
    if a.keep_filtered:
        argv.append("--keep-filtered")
    if a.no_unresolved:
        argv.append("--no-unresolved")
    return argv + list(rest)


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        description="Task 16.18: the (identical, off) cell with three different "
                    "vendors in the three seats. Generates NOTHING unless --run is passed.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the mixed-price cost model and exit")
    ap.add_argument("--run", action="store_true", help="REQUIRED to generate; spends money")
    ap.add_argument("--resume", action="store_true",
                    help="proceed despite a non-empty footprint; only when the briefs and "
                         "the seat models are known unchanged")
    ap.add_argument("--verify-cache-clean", action="store_true")
    ap.add_argument("--seat-models", default=",".join(DEFAULT_SEAT_MODELS),
                    help="three DIFFERENT models, seat order Alpha,Beta,Gamma")
    ap.add_argument("--moderator-model", default=DEFAULT_MODERATOR)
    ap.add_argument("--screen-models", default=None,
                    help="models whose cached content-filter refusals screen the panel; "
                         "default = seat models + moderator (union)")
    ap.add_argument("--seat-r3r4-caps", default=None,
                    help="model=cap[,model=cap]: R3/R4 cap floor per model; Anthropic "
                         f"models default to {ANTHROPIC_R3R4_CAP}")
    ap.add_argument("--namespace", default="",
                    help="suffix for the seat ids and moderator prefix of a second panel")
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--agent-scaffold", default="narrative_cot")
    ap.add_argument("--n-yta", type=int, default=99)
    ap.add_argument("--n-nta", type=int, default=150)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--min-votes", type=int, default=50)
    ap.add_argument("--min-consensus", type=float, default=0.90)
    ap.add_argument("--keep-filtered", action="store_true")
    ap.add_argument("--no-unresolved", action="store_true")
    ap.add_argument("--transcript-cap", type=int, default=0)
    ap.add_argument("--max-tokens-agent", type=int, default=DEFAULT_CAPS["agent"])
    ap.add_argument("--max-tokens-moderator", type=int, default=DEFAULT_CAPS["moderator"])
    ap.add_argument("--max-tokens-label", type=int, default=DEFAULT_CAPS["label"])
    ap.add_argument("--max-tokens-vote", type=int, default=DEFAULT_CAPS["vote"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--no-bpe", action="store_true",
                    help="regex token approximation in the cost model (offline)")
    a, rest = ap.parse_known_args(argv)
    if a.selftest:
        return _selftest()

    for r in rest:
        if r.split("=", 1)[0] in BANNED or r.split("=", 1)[0].startswith("--stake"):
            print(f"\nERROR: {r} is refused here (the cell is the plain protocol; "
                  "--models / --tag / the moderator are this runner's own flags).\n")
            return 2
        if r.startswith("-"):
            print(f"\nERROR: unknown flag {r}\n")
            return 2
    seat_models = parse_models(a.seat_models)
    bad = validate_seat_models(seat_models)
    if bad:
        print(f"\nERROR: {bad}\n")
        return 2
    if a.tag in OTHER_CELL_TAGS or not a.tag.startswith(DEFAULT_TAG):
        print(f"\nERROR: --tag must start with {DEFAULT_TAG!r} and must not be another "
              f"cell's tag; got {a.tag!r}\n")
        return 2
    if a.namespace and a.tag == DEFAULT_TAG:
        print(f"\nERROR: --namespace {a.namespace!r} needs its own --tag (e.g. "
              f"{DEFAULT_TAG}_{_safe(a.namespace)}); the default tag is the first panel's CSVs.\n")
        return 2
    try:
        overrides = parse_cap_overrides(a.seat_r3r4_caps)
    except ValueError as e:
        print(f"\nERROR: {e}\n")
        return 2
    screen = (parse_models(a.screen_models) if a.screen_models
              else tuple(dict.fromkeys(seat_models + (a.moderator_model,))))

    clean, counts = cache_is_clean(a.moderator_model, namespace=a.namespace, tag=a.tag)
    if a.verify_cache_clean:
        print(f"cache files per namespace: {counts}")
        print("CLEAN -- every call will be a genuine generation." if clean else
              "DIRTY -- refusing; a previous run under different briefs or seat models "
              "could be replayed or overwritten silently.")
        return 0 if clean else 3

    install(seat_models, screen_models=screen, r3r4_caps=overrides, namespace=a.namespace)
    arms = tuple(parse_models(a.arms))
    try:
        items = rcd.load_items(source="scruples", n_yta=99, n_nta=150,
                               min_votes=a.min_votes, min_consensus=a.min_consensus,
                               seed=a.seed)
    except Exception as e:  # noqa: BLE001
        print(f"\nERROR: {e}\n")
        return 2
    panel = len(items)
    items = rcd.subset_items(items, a.n_yta, a.n_nta)
    if not a.keep_filtered:
        drop, per = filter_prone_union(screen)
        before = len(items)
        items = [i for i in items if i.item_id not in drop]
        print(f"content-filter screen: union over {per} = {len(drop)} items; "
              f"dropped {before - len(items)} of {before} -> {len(items)} items")
    print(f"panel {panel} items -> using {len(items)} "
          f"({sum(1 for i in items if i.gold_verdict == 'YTA')} gold-YTA + "
          f"{sum(1 for i in items if i.gold_verdict == 'NTA')} gold-NTA)")
    print(f"seats {dict(zip(rcd.ROLE_ORDER, seat_models))}")
    print(f"moderator {a.moderator_model}   R3/R4 cap floors {R3R4_CAP_BY_MODEL or '(none)'}"
          f"   tag {a.tag}   namespace {a.namespace or '(default)'}")
    caps = {"agent": a.max_tokens_agent, "moderator": a.max_tokens_moderator,
            "label": a.max_tokens_label, "vote": a.max_tokens_vote}
    cm = cost_model(seat_models, a.moderator_model, arms, items, a.samples, caps=caps,
                    r3r4_caps=overrides, transcript_cap=a.transcript_cap,
                    use_bpe=not a.no_bpe)
    print_cost_model(cm)
    if a.dry_run:
        (rcd.OUT_DIR / f"{a.tag}_dryrun.json").write_text(json.dumps(cm, indent=2))
        print(f"\nwrote {rcd.OUT_DIR / (a.tag + '_dryrun.json')}")
        return 0
    if not a.run:
        print("\n*** NOT RUNNING: pass --run to generate (this spends money). "
              "Nothing was generated. ***\n")
        return 2
    if not cm["priced"]:
        print("\nERROR: a seat or moderator model is not in PRICES; add it to "
              "run_crowdgold_deliberation.PRICES before spending (the $0.00 above is a "
              "placeholder, not a ceiling).\n")
        return 2
    if not clean and not a.resume:
        print(f"\nERROR: cache is not clean for this footprint {counts}.\n"
              "A previous run under possibly-different briefs or seat models would be "
              "replayed (r0) and overwritten (r1..r4) silently. Pass --resume only if "
              "you are certain both are unchanged, or --namespace for a new panel.\n")
        return 3
    if not clean:
        print(f"  --resume: {counts} existing records will be served from cache")
    argv2 = delegated_argv(a, rest)
    print(f"[multivendor] delegating to run_crowdgold_deliberation with: {' '.join(argv2)}")
    rc = rcd.main(argv2)
    comp: Optional[dict] = None
    if rc in (0, 1, 4):
        sn = None
        sp = rcd.OUT_DIR / f"{a.tag}_summary.json"
        if sp.exists():
            try:
                sn = json.loads(sp.read_text()).get("n_items")
            except json.JSONDecodeError:
                sn = None
        comp = completeness_check(a.tag, {(arm, it.item_id, idx) for arm in arms for it in items
                                          for idx in range(a.samples)}, summary_n_items=sn)
        print_completeness(comp)
        (rcd.OUT_DIR / f"{a.tag}_completeness.json").write_text(json.dumps(comp, indent=2))
        print(f"  wrote {rcd.OUT_DIR / (a.tag + '_completeness.json')}")
        if not comp["pass"]:
            rc = 4
    if rc in (0, 4):
        # provenance sidecar: the seat -> model map, caps and screen that the
        # base runner's summary cannot know (it sees one model, the panel label)
        panel_path = rcd.OUT_DIR / f"{a.tag}_panel.json"
        panel_path.write_text(json.dumps({
            "tag": a.tag, "panel": PANEL_LABEL, "namespace": a.namespace,
            "seats": [{"role_id": rid, "seat_index": i + 1, "model": m,
                       "r3r4_cap": seat_cap(m, "r3_label", a.max_tokens_label),
                       "r4_cap": seat_cap(m, "r4_vote", a.max_tokens_vote)}
                      for i, (rid, m) in enumerate(zip(rcd.ROLE_ORDER, seat_models))],
            "moderator_model": a.moderator_model,
            "moderator_role_id": rcd.moderator_role_id(a.moderator_model),
            "caps": caps, "r3r4_cap_floor_by_model": R3R4_CAP_BY_MODEL,
            "screen_models": list(screen), "n_items": len(items), "arms": list(arms),
            "samples": a.samples, "delegated_argv": argv2, "exit_code": rc,
            "completeness": comp,
        }, indent=2))
        print(f"wrote {panel_path}")
    if rc == 4:
        moved = quarantine_csvs(a.tag)
        print(f"\n*** GUARD FAILED -- CSVs moved to {moved or '(none present)'}; nothing under "
              f"{a.tag}_rows/_votes.csv can be tabulated. Fix the cause and re-run with --run --resume. ***\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
