"""
scripts/run_crowdgold_filter_screen.py -- Addendum 16.20: a model that lives
behind a platform content filter (gpt-4o on Azure first) enters the grip table
THROUGH the deliberation runner's own content-filter screen, in two separately
authorised steps, on the registered 249-item AITA panel.

WHAT 16.11b SHOWED AND WHY IT FAILED
------------------------------------
16.11b ran the 40-item gpt-4o grip screen (16 YTA / 24 NTA, both arms, k=1,
80 debates, 1,360 calls, 3072/3072 r3/r4 caps, tag cg_deliberation_gpt4o_screen)
and its round-level guard FAILED: worst NOVERDICT 15.0% at the integration
round and 7.5% at every r0 seat, every failing call `finish_reason:
content_filter` with empty output. Azure's content filter refuses a subset of
the AITA posts, and the deliberation runner's defence against exactly that --
`filter_prone_items()`, which drops an item ahead of the 17-call chain when
ANY cached single-agent NoT call on it was refused -- found NOTHING for
gpt-4o, because the screen globs

    divergence_study_outputs/cg_gen_<model>_narrative_cot_t2560_*.json

and no single-agent gpt-4o call had ever been made on this instrument (the
run's summary records `filter_prone_items_per_model: {"gpt-4o": 0}`). The
screen was applied and was empty. Measured from the 16.11b cache itself: 11
of the 40 items carry a refusal (24/240 r0 calls, cascading to r1-r4 and both
moderator calls once an empty upstream text is in the prompt); 29 are clean;
all 11 refused items are ALSO in gpt-5.4-nano's screen and 9 of 11 in grok's,
i.e. this is the platform's prompt-side filter on the post text, shared by
every deployment behind the same Azure resource, not model behaviour.

THE TWO STEPS (each its own authorisation, per the 16.11b RESULTS block)
------------------------------------------------------------------------
STEP 1  --step screen     the single-agent calls the screen reads. Exactly

    cg_gen_<model>_narrative_cot_t2560_<arm>_<item>_<idx>.json
        for arm in (third_person, as_asker) x the 249 panel items
        x idx in 00..0(k-1),  k = --screen-samples, DEFAULT 3
        = 1,494 calls for one model at k=3 (498 at k=1),

produced by delegating to scripts/run_crowdgold_aita.main with

    --models <model> --scaffolds narrative_cot --max-tokens-scaffolded 2560
    --n 99 --n-nta 150 --samples <k> --arms third_person,as_asker
    --seed 44 --min-votes 50 --min-consensus 0.90
    --no-memorization --tag cg_screen_<model>

ALWAYS with --n 99 --n-nta 150: the single-agent runner SAMPLES n items per
gold class with the seed, so only the full (99, 150) draw is the panel the
deliberation runner's subset_items() prefixes; a smaller --n there is a
different set of items. The prompts are byte-identical to every other
model's cached narrative_cot single-agent cells (system =
PROMPTS["narrative_cot"] + the AITA glossary, user = the registered arm
wrapper around the byte-identical post) and the cache paths are the ones
`rcd.SCREEN_GLOB` globs, every sample included -- the selftest fnmatches
every path against the runner's own glob. `--no-memorization` because the
memorization probe (`cg_mem_*`) is not read by the screen and buys nothing
here.

WHY k=3 AND NOT THE MINIMAL 498 (measured, the dry-run prints it). The
Azure filter is STOCHASTIC PER CALL. On the nano and grok single-agent
caches -- 3 arms x k=3 identical-prompt samples per item, behind the same
Azure resource -- 36% of the refused (arm, item) cells are refused in some
samples and not others, and the per-call refusal probability on a prone
item is ~0.6. A two-arm, sample-00 screen (498 calls) would have caught 72%
(nano) / 64% (grok) of the prone items their full 9-call screens found; k=2
87%; k=3 90% / 97%, pooled 0.94. Projected onto gpt-4o with 16.11b's
measured prone rate (11 of 40 chain items) and its measured chain factor
(every post-r0 round refuses 0.50 of a prone item's debates, empty output):
k=1 leaves ~22 prone items among ~202 survivors and a worst-round NOVERDICT
of ~5.4% -- the 5% guard fails AGAIN and the step-2 spend is lost a second
time; k=3 leaves ~4 among ~185 and projects ~1.2%. The screen is cheap
(~$10 at k=3) next to the chain it protects (~$37); the extra samples are
the insurance. The selftest plants a refusal on sample 01 only and checks
that a k=1 screen misses it and the k=3 screen and the runner's own glob
both see it.

The single-agent runner's own truncation/NOVERDICT guard MAY report a
failure on this step (a prompt-side refusal is an empty output, i.e.
NOVERDICT), and that is the point, not a defect: the screen is the readout.
This module's own guard on step 1 is the TRUNCATION share only (finish_reason
max_tokens/length), because a truncated single-agent NoT call at the 2560 cap
means the chain's agent cap is wrong too; it prints the literal GUARD FAILED
and exits 4 if that exceeds 5%. The rows CSV this step writes doubles as the
model's own single-agent narrative_cot comparator for step 2's criterion and
resistance read-outs (`--compare-rows`) and for the readout's transfer onto
the model's own solo verdict (majority of the k samples), which no other
file on disk provides for a new model.

STEP 2  --step deliberate  the registered deliberation, delegated to
scripts/run_crowdgold_deliberation.main with the PANEL DEFAULTS and the
corrected r3/r4 caps:

    --models <model> --arms third_person,as_asker --n-yta 99 --n-nta 150
    --samples 1 --seed 44 --agent-scaffold narrative_cot
    --max-tokens-agent 2560 --max-tokens-moderator 1024
    --max-tokens-label 3072 --max-tokens-vote 3072 --transcript-cap 0
    --tag cg_deliberation_<model>     (gpt-4o: cg_deliberation_gpt4o)
    --compare-rows <step 1 rows CSV>

The runner applies its screen itself at launch ("content-filter screen:
dropped N of 249 items (union over {...})"); this module refuses to launch
step 2 until every one of the step-1 records (1,494 at k=3) exists, so the
screen the runner applies is the complete one and not a partial run's.
--keep-filtered is never passed and cannot be passed through. The 16.11b
run is NOT re-done: see CACHE below. Once step 1 is on disk the step-2
dry-run also reports the screen's MEASURED catch rate on the 11 items the
16.11b chain refused, and projects the residual NOVERDICT from it; a
projected GUARD FAIL there means raise --screen-samples and re-run step 1
with --resume (only the missing sample indices are new calls).

--screen registered (default) is the runner's screen verbatim: an item is
dropped iff one of its cached single-agent NoT calls was refused. --screen
chain-aware additionally drops any item whose cached DELIBERATION call for
this model (cgd_<model>_*) was refused -- measured, zero-cost, a strict
superset of the registered set, and the only defence against the refusals
that begin at r1 (2 of the 31 r0-clean items in 16.11b were refused from r1
onward in every seat, because a seat's own r0 text tripped the prompt filter;
a single r0-like call cannot see that). It is a deviation from the protocol
named in the 16.11b RESULTS block and is therefore OFF by default; the
dry-run prints what each screen would drop so the choice can be registered
before any call. It is installed by rebinding rcd.filter_prone_items
(the install() pattern of run_crowdgold_unembodied / run_crowdgold_topology).

STEP 3  --step readout     offline. The grip-table row for the model:
analyze_stake_grip (G1 fire / G2 reject share / G3 stake concentration with
item-clustered CI, per-seat objection and reject rates, S2 accuracy, measured
spend from the cache) plus the 16.10-table columns computed by the same
definitions as analyze_topology_2x2.analyse_cell but parametrised by model:
role-lock |P(at_fault | writer_advocate) - P(at_fault | counterparty)| at r0,
per-seat r0 accuracy, localisation excess vs the Poisson-binomial null, fire
rate P(>=2 objectors), error lift of >=2 on the codable population read only
at >= 60 fired, transfer P(grok-solo wrong | fired) - P(| not) on grok's
cached standard majority-of-3, and the same transfer onto the model's OWN
single-agent narrative_cot verdict (step 1 rows, majority of the k screen
samples; a tie falls back to none). Reported on the
model's own surviving items and, when the four 16.10 cells are on disk, on
the intersection with the 16.10 matched item set, so the row is comparable
to the 16.10 table on the same items.

CACHE -- WHICH CALLS REPLAY AND WHICH ARE NEW (the landmine, stated)
------------------------------------------------------------------
call_cache_path keys on (model, scaffold, arm, item, sample, role id) and
the token cap, NOT on brief text; r1..r4 records carry a parent_sha over the
texts they were shown. This module introduces NO new role set, NO new brief
and NO new moderator prefix: the seats are the three registered embodied
seats verbatim and the moderator is `mod-<model>`, exactly as in 16.11b and
in every other model's registered run. Therefore:

  * step 1 writes into a namespace that is EMPTY today for gpt-4o
    (cg_gen_gpt-4o_narrative_cot_t2560_*; the selftest reports the count and
    the run path refuses a non-empty namespace without --resume). All 1,494
    step-1 calls are NEW; none can replay, because no single-agent gpt-4o
    call has ever been made on this instrument;
  * step 2's cache paths for the 40 items of 16.11b are BYTE-IDENTICAL to the
    16.11b paths (same model, scaffold, role ids, arms, sample 0, caps
    2560/1024/3072/3072, self-moderation) and the prompts are byte-identical
    because they are rebuilt from the same post text and the same cached
    upstream texts, which parent_sha verifies at replay. So for every
    surviving item among those 40, all 17 calls x 2 arms REPLAY from cache and
    are not new spend (29 clean items = 986 replays if the screen catches
    exactly the 16.11b refusals). Every other surviving item is NEW: 17 x 2
    calls each. The dry-run counts ONLY the new ones and prices them from the
    per-round prompt/completion tokens MEASURED on the model's own cached
    chain records (the runner's own dry-run, printed alongside, assumes
    1,311 completion tokens per agent round and is an upper bound: gpt-4o's
    measured r0/r1/r2 completions are 484/584/150). Before step 1 lands the
    step-2 dry-run necessarily counts the UNSCREENED panel (7,106 new calls,
    ~$49); it then prints the projection for the screened panel at k
    (~185 survivors, ~5,300 new calls, ~$37 at k=3).
  * the 16.11b records of the 11 refused items are `content_filter` records
    with empty output and WOULD REPLAY THE REFUSAL if those items reached
    step 2 (do_call replays any cached record whose parent_sha matches,
    refusal or not). The screen is what keeps them out; the dry-run prints,
    per screen, how many of the chain-refused items it catches.
  * a file in cgd_<model>_narrative_cot_* that is NOT one of the expected
    paths (another cap, sample idx > 0, a third arm, a foreign role id or
    moderator prefix) is FOREIGN, and the run path refuses with it present
    unless --resume; --resume is the only override.
  * --moderator-model is REFUSED (exit 2) whenever it names a model other
    than the agent model, on every step and before anything is read or
    written. The base runner keys r3_label / r4_vote on the AGENT model and
    the bare seat id, and synthesis / integration on mod-<moderator>, so a
    cross-vendor moderator here would (a) regenerate gpt-4o's r3/r4 records
    at the 16.11b paths with a parent_sha over the other vendor's synthesis
    and WRITE OVER them (no cgd_ record is git-tracked), and (b) land the
    moderator calls on that vendor's own self-moderated mod-<vendor> paths
    (claude-sonnet-4-6: the sonnet self-pilot's 64 records). The selftest
    constructs both collisions in a tempdir through the same delegation and
    proves main() refuses before a byte changes. A cross-vendor moderator
    over these seats is scripts/run_crowdgold_modvendor.py, which namespaces
    the moderator (modvendor-<agent>) and the r3/r4 seats (<seat>_xmod-
    <moderator>) and verifies both empty. `--moderator-model <the agent
    model>` is the self-moderated default and is normalised to it.
  * --tag must start with cg_deliberation_<short_name(model)> and may not be
    PRIOR_SCREEN_TAG (cg_deliberation_gpt4o_screen, the 16.11b CSVs, which
    share that prefix); --screen-tag must start with cg_screen_<short>.
    rcd.main / rca.main write <tag>_rows.csv, _votes.csv, _summary.json (and
    <tag>_dryrun.json on --dry-run) unconditionally, so a free-text tag
    could overwrite a sibling cell's CSVs (cg_deliberation, _stdcot, ...).
    On --run a <tag>_rows.csv already holding rows for another model is
    refused as well.

COMPLETENESS GUARD (every launched debate must be a row)
--------------------------------------------------------
rcd.run swallows an exception raised inside run_deliberation, prints one
console ERROR line and omits that debate from rows / votes / calls; both
base guards and every readout denominator are then computed on the
survivors, so a vendor outage concentrated in one arm is arm-correlated
attrition that nothing reports. After the delegated run returns, step 2
compares the rows CSV with the (arm, item, sample) keys it launched
(survivors x 2 arms x 1 sample; the summary's n_items is cross-checked
against the survivor count), writes <tag>_completeness.json (n_expected,
n_rows, n_missing, the missing keys, per-arm counts) and prints the literal
GUARD FAILED and exits 4 on any shortfall. Step 3 reads that sidecar and the
summary's round / outcome guards and REFUSES (GUARD FAILED, exit 4, the grip
row written only as <tag>_grip_row.json flagged UNREAD_GUARD_FAILED and the
G1-G3 readout to <tag>_stake_grip_UNREAD.json, never into the shared
registry) when either is failed or missing.

S2 CONVENTION: the grip-table row's S2 accuracy is on the CODABLE population
(rows whose group verdict codes under the published collapse), with
n_codable printed beside it -- the convention of analyze_topology_2x2's
group_verdict_s2 and of the lift. Dividing by all rows would score every
NOVERDICT / UNRESOLVED as wrong, and on the one cell where the guard tolerates
up to 5% non-codable that understates S2 by up to 0.05 against the 16.10 row
printed beside it.

AZURE-SPECIFIC SETTINGS (read from scripts/generators.py, nothing new)
--------------------------------------------------------------------
gpt-4o routes through generators._call_openai on the AzureOpenAI client
(AZURE_AI_PROJECT_ENDPOINT / AZURE_AI_API_KEY, AZURE_AI_API_VERSION default
2025-04-01-preview), non-reasoning branch: `max_tokens` is honoured as given
(no 8192 floor), `temperature=0.7`, `seed=sample_idx`, 180 s timeout, five
attempts with backoff. A prompt-side refusal (HTTP 400 `content_filter` /
ResponsibleAIPolicyViolation, caught by generators._is_content_filter) comes
back as an EMPTY GenerationResult with finish_reason "content_filter"; a
completion-side refusal comes back with finish_reason "content_filter" and
whatever partial text Azure returned (18 of the 24 refused 16.11b r0 records
are empty, 6 carry partial output; the runner's guard counts the empty ones
as NOVERDICT, 7.5% per r0 seat). Both kinds are CACHED like any other record
with that finish_reason, so both are read by the screen and by the chain-
evidence screen, and a refusal is permanent for that (model, arm, item,
round, seat, sample) until the record is removed. Note `seed=sample_idx` is
sent, yet the refusals are not reproducible across samples: the filter is
upstream of the sampler. No per-request knob
controls the filter; the deployment-level content-filter policy in the Azure
resource is the only lever and is outside this code -- changing it between
16.11b and this run would change the platform under the model and must be
recorded if done. Nothing else is model-specific; a fourth vendor deployment
enters with `--models <deployment>` plus a PRICES entry in
run_crowdgold_deliberation.PRICES (the run path refuses an unpriced model).

Usage
-----
  python -m scripts.run_crowdgold_filter_screen --selftest
  python -m scripts.run_crowdgold_filter_screen --step screen --models gpt-4o --dry-run
  python -m scripts.run_crowdgold_filter_screen --step screen --models gpt-4o --run        # SPENDS (~$10, k=3)
  python -m scripts.run_crowdgold_filter_screen --step deliberate --models gpt-4o --dry-run
  python -m scripts.run_crowdgold_filter_screen --step deliberate --models gpt-4o --run    # SPENDS (~$37 projected)
  python -m scripts.run_crowdgold_filter_screen --step readout --models gpt-4o
  (--screen-samples k changes the step-1 draw; step 2 requires the screen complete at that k)

Nothing here generates unless the literal flag --run is given; --dry-run (or
omitting --run) prints the plan and the cost model and exits. --selftest stubs
both generators and redirects BOTH cache directories (run_crowdgold_aita.OUT_DIR
and run_crowdgold_deliberation.OUT_DIR) to a temporary directory.

Exit codes: 0 ok, 2 bad config / --run not given / unpriced or registered
model / screen incomplete / cross-vendor --moderator-model / bad --tag,
3 cache not clean (foreign files, no --resume), 4 guard failed (round,
outcome or completeness guard of step 2; step 3 on a failed or missing
guard record).
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import inspect
import io
import json
import random
import re
import sys
import tempfile
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable, Optional, Sequence

import scripts.run_crowdgold_aita as rca
import scripts.run_crowdgold_deliberation as rcd
from scripts.run_crowdgold_aita import AS_ASKER, THIRD_PERSON, CrowdGoldItem
from scripts.run_phase1_quartet import OUT_DIR, _safe

# ---------------------------------------------------------------------------
# The registered configuration (panel defaults + the corrected r3/r4 caps)
# ---------------------------------------------------------------------------

SCREEN_SCAFFOLD = "narrative_cot"
SCREEN_CAP = 2560                 # the cap rcd.SCREEN_GLOB reads (t2560)
#: Samples per (arm, item) in the screen. 3, not 1, because the Azure filter
#: is STOCHASTIC per call (measured on the nano/grok k=3 caches: a third of the
#: refused cells are refused in some samples and not others; per-call refusal
#: probability on a prone item ~0.6). A sample-00 two-arm screen catches only
#: 64-72% of the prone items the full 9-call screens found; k=3 catches
#: 92-97%. The runner's screen globs every sample, so k=3 needs no code change
#: there. Overridable with --screen-samples; the dry-run prints the projection
#: for 1..3 so the choice can be registered.
SCREEN_SAMPLES = 3
SCREEN_SAMPLE = 0                 # the first sample; sample idx runs 0..k-1
ARMS = (THIRD_PERSON, AS_ASKER)
N_YTA, N_NTA = 99, 150
SEED, MIN_VOTES, MIN_CONSENSUS = 44, 50, 0.90
CAPS = {"agent": 2560, "moderator": 1024, "label": 3072, "vote": 3072}
MAX_TRUNCATION_SHARE = rca.MAX_TRUNCATION_SHARE
MIN_FIRED = 60
DEFAULT_MODELS = "gpt-4o"

#: Models with a registered deliberation run on this panel. They are not
#: "behind a screen": their screens were produced by their own single-agent
#: runs long ago and their grip rows stand. Refused here so this two-step can
#: never be pointed at a namespace that already holds a registered run.
REGISTERED_MODELS = frozenset(rcd.FULL_MODELS) | {"claude-sonnet-4-6"}

#: The 16.11b run this module must NOT re-do (its rows/votes stay on disk
#: under this tag; the 40-item prefix replays from its cache).
PRIOR_SCREEN_TAG = "cg_deliberation_gpt4o_screen"
TRUNC_REASONS = ("max_tokens", "length", "MAX_TOKENS")


def short_name(model: str) -> str:
    return re.sub(r"[^a-z0-9]", "", model.lower())


def default_tags(model: str) -> tuple[str, str]:
    s = short_name(model)
    return f"cg_screen_{s}", f"cg_deliberation_{s}"


def tag_guard(model: str, tag: str, screen_tag: str) -> Optional[str]:
    """Why a (--tag, --screen-tag) pair is refused, or None. The delegated
    runners write <tag>_rows.csv / _votes.csv / _summary.json (and
    <tag>_dryrun.json on a dry-run) unconditionally, so both tags are held
    to this model's own prefix and the 16.11b tag is refused by name."""
    stag_d, dtag_d = default_tags(model)
    if tag == PRIOR_SCREEN_TAG:
        return (f"--tag {tag!r} is the 16.11b run's tag; its rows/votes/summary stay on "
                "disk and are not re-done")
    if not tag.startswith(dtag_d):
        return (f"--tag {tag!r} must start with {dtag_d!r} (cg_deliberation_<short_name(model)>); "
                "any other tag could overwrite a sibling cell's CSVs")
    if not screen_tag.startswith(stag_d):
        return (f"--screen-tag {screen_tag!r} must start with {stag_d!r} (cg_screen_<short_name(model)>)")
    return None


def moderator_guard(model: str, moderator_model: Optional[str]) -> Optional[str]:
    """Why --moderator-model is refused, or None. Anything but the agent model
    itself is a cross-vendor moderator, which the base runner does not
    namespace (see CACHE in the module docstring) -- that cell is
    run_crowdgold_modvendor's."""
    if moderator_model and moderator_model != model:
        return (f"--moderator-model {moderator_model!r} over {model!r} seats is a cross-vendor "
                "moderator: run_crowdgold_deliberation keys r3_label/r4_vote on the agent model "
                "and the bare seat id and synthesis/integration on mod-<moderator>, so this "
                f"would OVERWRITE {model}'s cached r3/r4 records (the 16.11b footprint) and the "
                f"moderator's own self-moderated mod-{_safe(moderator_model)} records. Use "
                "scripts/run_crowdgold_modvendor.py, which namespaces both. Refusing.")
    return None


def rows_models_on_disk(rows_path: Path) -> set[str]:
    """The `model` values a rows CSV holds (empty set when absent)."""
    if not rows_path.exists():
        return set()
    with rows_path.open() as f:
        return {r.get("model", "") for r in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

def load_panel(n_yta: int = N_YTA, n_nta: int = N_NTA) -> list[CrowdGoldItem]:
    """The registered 249-item panel, then the same deterministic prefix the
    deliberation runner takes (subset_items sorts by item_id per gold class),
    so a smaller --n-yta/--n-nta is always a SUBSET of the panel."""
    items = rca.load_items(source="scruples", n_yta=N_YTA, n_nta=N_NTA,
                           min_votes=MIN_VOTES, min_consensus=MIN_CONSENSUS,
                           seed=SEED)
    return rcd.subset_items(items, n_yta, n_nta)


# ---------------------------------------------------------------------------
# Step 1: the screen's own cache namespace
# ---------------------------------------------------------------------------

def screen_path(model: str, arm: str, item_id: str, idx: int = SCREEN_SAMPLE) -> Path:
    """The exact file run_crowdgold_aita writes for one screen call, resolved
    against the CURRENT rca.OUT_DIR (so a redirected selftest stays in its
    tempdir)."""
    p = rca.gen_cache_path(model, arm, item_id, idx, SCREEN_SCAFFOLD,
                           SCREEN_CAP, rca.INSTRUMENT)
    return rca.OUT_DIR / p.name


def screen_paths(model: str, items: Sequence[CrowdGoldItem],
                 samples: int = SCREEN_SAMPLES) -> dict[tuple[str, str, int], Path]:
    """(arm, item, sample_idx) -> path, for the k samples the screen is run at."""
    return {(arm, it.item_id, idx): screen_path(model, arm, it.item_id, idx)
            for arm in ARMS for it in items for idx in range(samples)}


def screen_glob_matches(model: str, path: Path) -> bool:
    return fnmatch.fnmatch(path.name, rcd.SCREEN_GLOB.format(model=_safe(model)))


def screen_status(model: str, items: Sequence[CrowdGoldItem],
                  samples: int = SCREEN_SAMPLES) -> dict:
    """What the screen namespace holds for THIS panel at k samples: present /
    missing records, the refused items (union over arms AND samples, as the
    runner's glob unions), the truncated records (this module's step-1
    guard), token totals."""
    paths = screen_paths(model, items, samples)
    present, missing, refused_by_arm, truncated = {}, [], defaultdict(set), []
    pt = ct = 0
    for key, p in paths.items():
        if not p.exists():
            missing.append(key)
            continue
        try:
            rec = json.loads(p.read_text())
        except json.JSONDecodeError:
            missing.append(key)
            continue
        present[key] = rec
        pt += int(rec.get("prompt_tokens") or 0)
        ct += int(rec.get("completion_tokens") or 0)
        if rec.get("finish_reason") == "content_filter":
            refused_by_arm[key[0]].add(key[1])
        if rec.get("finish_reason") in TRUNC_REASONS:
            truncated.append(key)
    refused = set().union(*refused_by_arm.values()) if refused_by_arm else set()
    return {
        "samples": samples,
        "n_expected": len(paths), "n_present": len(present), "n_missing": len(missing),
        "missing": sorted(missing), "refused_items": refused,
        "refused_by_arm": {a: sorted(s) for a, s in refused_by_arm.items()},
        "n_truncated": len(truncated), "truncated": sorted(truncated),
        "prompt_tokens": pt, "completion_tokens": ct,
        "complete": len(missing) == 0,
        "mean_completion": (ct / len(present)) if present else None,
    }


def namespace_files(model: str, out_dir: Path) -> list[str]:
    """Every file in the screen namespace for this model (any arm, any item,
    any sample) -- the run path refuses step 1 when this is non-empty."""
    pat = rcd.SCREEN_GLOB.format(model=_safe(model))
    if not out_dir.is_dir():
        return []
    return sorted(p.name for p in out_dir.iterdir() if fnmatch.fnmatch(p.name, pat))


def step1_guard(status: dict) -> dict:
    """Truncation share of the screen calls; prints GUARD FAILED on breach."""
    n = max(1, status["n_present"])
    share = status["n_truncated"] / n
    ok = share <= MAX_TRUNCATION_SHARE
    print(f"\nSTEP-1 GUARD (truncation of the single-agent NoT call at the "
          f"{SCREEN_CAP} cap; refusals are NOT truncation and are the screen's "
          f"readout, not a guard breach)")
    print(f"  truncated {status['n_truncated']} of {status['n_present']} = "
          f"{share:.3f}  (limit {MAX_TRUNCATION_SHARE})  {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("\n  *** GUARD FAILED: the single-agent cap truncates; the chain's "
              "agent cap is the same cap and would truncate too. Do not launch "
              "step 2 on this cap. ***")
    return {"truncation_share": share, "pass": ok}


# ---------------------------------------------------------------------------
# The chain-evidence screen (zero cost; measured from cgd_<model>_* records)
# ---------------------------------------------------------------------------

def chain_filtered_items(model: str, out_dir: Optional[Path] = None) -> dict:
    """Items with ANY cached deliberation call for this model refused by the
    content filter, with per-round counts. Reads cgd_<model>_* only."""
    out_dir = rcd.OUT_DIR if out_dir is None else out_dir
    items: dict[str, set] = defaultdict(set)
    by_round: dict[str, int] = defaultdict(int)
    n = 0
    if not out_dir.is_dir():
        return {"items": set(), "by_round": {}, "n_records": 0}
    pre = f"cgd_{_safe(model)}_"
    for p in out_dir.iterdir():
        if not p.name.startswith(pre):
            continue
        try:
            rec = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        n += 1
        if rec.get("finish_reason") == "content_filter":
            items[str(rec.get("item_id", ""))].add(rec.get("round", ""))
            by_round[rec.get("round", "")] += 1
    items.pop("", None)
    return {"items": set(items), "rounds_per_item": {k: sorted(v) for k, v in items.items()},
            "by_round": dict(by_round), "n_records": n}


# ---------------------------------------------------------------------------
# How much of the filter a k-sample screen sees (measured, zero cost)
# ---------------------------------------------------------------------------

#: Models whose full single-agent NoT screens (3 arms x k=3) are on disk and
#: sit behind the SAME Azure resource as gpt-4o. Their caches are the only
#: place the per-call stochasticity of the filter can be measured without
#: spending: every (arm, item) there has three identical-prompt samples.
CATCH_RATE_MODELS = ("gpt-5.4-nano", "grok-4-1-fast-reasoning")


def screen_catch_rates(max_k: int = 3, models: Sequence[str] = CATCH_RATE_MODELS,
                       out_dir: Optional[Path] = None) -> dict:
    """For each k in 1..max_k: the share of PRONE items (any refusal anywhere
    in the model's full cached screen) that a two-arm, k-sample screen --
    arms (third_person, as_asker), samples 0..k-1 -- would have caught. Pooled
    over `models`, per model too. Empty when no such cache is present (the
    selftest's tempdir), never asserted as a count."""
    out_dir = rca.OUT_DIR if out_dir is None else out_dir
    pat = re.compile(r"cg_gen_(?P<model>.+?)_narrative_cot_t2560_(?P<arm>third_person|as_asker|as_asker_stance)"
                     r"_(?P<item>[A-Za-z0-9]+)_(?P<idx>\d\d)\.json$")
    safe = {_safe(m): m for m in models}
    cells: dict[str, dict[tuple, dict[int, bool]]] = {m: defaultdict(dict) for m in models}
    if not out_dir.is_dir():
        return {}
    for m_safe in safe:
        for path in out_dir.glob(f"cg_gen_{m_safe}_narrative_cot_t2560_*.json"):
            mm = pat.match(path.name)
            if not mm or mm.group("model") != m_safe:
                continue
            try:
                rec = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            cells[safe[m_safe]][(mm.group("arm"), mm.group("item"))][int(mm.group("idx"))] = (
                rec.get("finish_reason") == "content_filter")
    res: dict = {"per_model": {}, "pooled": {}, "models": []}
    tot = {k: [0, 0] for k in range(1, max_k + 1)}
    mixed = tot_ref = 0
    for m, cc in cells.items():
        if not cc:
            continue
        prone = {item for (arm, item), v in cc.items() if any(v.values())}
        if not prone:
            continue
        res["models"].append(m)
        rates = {}
        for k in range(1, max_k + 1):
            caught = sum(1 for i in prone
                         if any(cc.get((a, i), {}).get(idx, False) for a in ARMS for idx in range(k)))
            rates[k] = caught / len(prone)
            tot[k][0] += caught; tot[k][1] += len(prone)
        calls = sum(len(v) for (a, i), v in cc.items() if i in prone)
        ref = sum(sum(v.values()) for (a, i), v in cc.items() if i in prone)
        mx = sum(1 for v in cc.values() if len(v) >= 2 and 0 < sum(v.values()) < len(v))
        anyref = sum(1 for v in cc.values() if any(v.values()))
        mixed += mx; tot_ref += anyref
        res["per_model"][m] = {"prone_items": len(prone), "catch_by_k": rates,
                               "per_call_refusal_on_prone": (ref / calls if calls else None),
                               "refused_cells": anyref, "mixed_cells": mx}
    if res["models"]:
        res["pooled"] = {k: (c / n if n else None) for k, (c, n) in tot.items()}
        res["mixed_share_of_refused_cells"] = (mixed / tot_ref if tot_ref else None)
    return res


def refusal_projection(*, k: int, catch: Optional[float], panel_n: int,
                       prone_rate: float, worst_round_factor: float,
                       limit: float = rca.MAX_NOVERDICT_SHARE) -> dict:
    """PROJECTION, stated as one: with a prone-item rate `prone_rate` on the
    panel and a screen that catches `catch` of prone items, how many prone
    items survive the screen and what NOVERDICT share the worst chain round
    would show. `worst_round_factor` is the 16.11b-measured share of a prone
    item's debates that the worst round (integration) refused: 11 of 22 =
    0.50. A refused prone item that survives the screen is assumed to refuse
    in the chain at the 16.11b rate."""
    if catch is None:
        return {"k": k, "catch": None}
    prone = prone_rate * panel_n
    caught = prone * catch
    uncaught = prone - caught
    survivors = panel_n - caught
    share = worst_round_factor * uncaught / survivors if survivors else None
    return {"k": k, "catch": catch, "prone_items_expected": prone, "caught": caught,
            "uncaught": uncaught, "survivors": survivors,
            "worst_round_noverdict_projected": share, "limit": limit,
            "projected_pass": (share is not None and share <= limit)}


def chain_refusal_factors(model: str) -> dict:
    """From the model's own cached chain (16.11b): per round, refused-and-EMPTY
    calls per prone-item debate, i.e. the NOVERDICT the guard would count.
    integration = 11/22 = 0.50 on 16.11b; r0 seats 3/22 each."""
    out_dir = rcd.OUT_DIR
    if not out_dir.is_dir():
        return {}
    per_round: dict[str, int] = defaultdict(int)
    prone_debates: set = set()
    pre = f"cgd_{_safe(model)}_{SCREEN_SCAFFOLD}_"
    for p in out_dir.iterdir():
        if not p.name.startswith(pre):
            continue
        try:
            rec = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if rec.get("finish_reason") == "content_filter":
            prone_debates.add((rec.get("arm"), rec.get("item_id")))
            if not (rec.get("output") or "").strip():
                per_round[rec.get("round", "")] += 1
    prone_items = {i for _, i in prone_debates}
    n_deb = 2 * len(prone_items)          # both arms of every prone item
    if not n_deb:
        return {}
    seats = {rd: (len(rcd.ROLE_ORDER) if rd in rcd.AGENT_ROUNDS else 1) for rd in rcd.ROUNDS}
    return {rd: per_round.get(rd, 0) / (n_deb * seats[rd]) for rd in rcd.ROUNDS} | {
        "_prone_items": len(prone_items), "_prone_debates": n_deb}


def worst_round(factors: dict) -> tuple[Optional[str], float]:
    """(label of the round(s) attaining the max factor, the factor). On 16.11b
    every post-r0 round ties at 0.50 (a refused prone debate stays refused
    downstream), so the label lists the tie; 0.5 ASSUMED when unmeasured."""
    vals = {rd: factors[rd] for rd in rcd.ROUNDS if rd in factors}
    if not vals:
        return None, 0.5
    wf = max(vals.values())
    return "/".join(rd for rd in rcd.ROUNDS if vals.get(rd) == wf), wf


def print_projection(model: str, items_n: int, prone_rate: float, prone_src: str,
                     rates: dict, factors: dict, ks: Sequence[int]) -> dict:
    print("\n" + "=" * 78)
    print(f"WHAT A k-SAMPLE SCREEN WOULD CATCH -- measured on the caches, projected to {model}")
    print("=" * 78)
    if not rates.get("models"):
        print("  no full single-agent screen cache on disk to measure the filter's per-call "
              "stochasticity (expected in the selftest tempdir); no projection.")
        return {}
    for m, r in rates["per_model"].items():
        print(f"  {m:<26} prone items {r['prone_items']:3d}  per-call refusal on prone "
              f"{r['per_call_refusal_on_prone']:.2f}  mixed cells {r['mixed_cells']}/{r['refused_cells']}  "
              "catch by k: " + "  ".join(f"k={k}:{v:.2f}" for k, v in r["catch_by_k"].items()))
    print(f"  pooled catch rate of a two-arm screen: "
          + "  ".join(f"k={k}: {v:.2f}" for k, v in rates["pooled"].items())
          + f"   (a refused cell is refused in SOME samples only {rates['mixed_share_of_refused_cells']:.0%} "
            "of the time: the filter is stochastic per call, so extra samples are extra sight)")
    worst_rd, wf = worst_round(factors)
    print(f"  prone-item rate on the panel: {prone_rate:.3f} [{prone_src}]; worst chain round(s) "
          f"{worst_rd or 'integration'} refuse {wf:.2f} of a prone item's debates "
          f"[{'measured on ' + model + ' chain records' if worst_rd else 'ASSUMED'}]")
    print(f"  {'k':>3} {'calls':>6} {'catch':>6} {'prone':>6} {'caught':>7} {'uncaught':>9} "
          f"{'survivors':>10} {'worst NOVERDICT':>16}  vs limit {rca.MAX_NOVERDICT_SHARE:.0%}")
    out = {}
    for k in ks:
        pr = refusal_projection(k=k, catch=rates["pooled"].get(k), panel_n=items_n,
                                prone_rate=prone_rate, worst_round_factor=wf)
        out[k] = pr
        if pr.get("catch") is None:
            print(f"  {k:3d} {2 * items_n * k:6d}   n/a")
            continue
        print(f"  {k:3d} {2 * items_n * k:6d} {pr['catch']:6.2f} {pr['prone_items_expected']:6.1f} "
              f"{pr['caught']:7.1f} {pr['uncaught']:9.1f} {pr['survivors']:10.1f} "
              f"{pr['worst_round_noverdict_projected']:16.3f}  "
              f"{'projected PASS' if pr['projected_pass'] else 'projected GUARD FAIL'}")
    print("  This is a projection from measured rates, not a guarantee; the guard is read on the run.")
    return out


def chain_prefix_items(model: str, items: Sequence[CrowdGoldItem]) -> list[str]:
    """Panel items with ANY cached chain record for this model (the 16.11b
    40-item prefix for gpt-4o): the population the chain's prone rate is
    measured on."""
    fp = delib_footprint(model, items)
    return fp["items_with_records"]


def install_chain_aware_screen(model: str) -> None:
    """Rebind the runner's screen to the union (registered | chain-evidence).
    Called by name inside rcd.main, so the rebinding takes effect."""
    registered = rcd.filter_prone_items

    def union(m: str) -> set[str]:
        s = set(registered(m))
        if m == model:
            s |= chain_filtered_items(m)["items"]
        return s
    rcd.filter_prone_items = union  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Step 2: expected deliberation footprint, replays, foreign files
# ---------------------------------------------------------------------------

def expected_delib_paths(model: str, items: Sequence[CrowdGoldItem], *,
                         moderator_model: Optional[str] = None) -> dict[tuple, Path]:
    """The 17 cache paths per (arm, item) this run would read or write, with
    the registered caps and self-moderation, keyed by (arm, item, round, role)."""
    mod = moderator_model or model
    mod_id = rcd.moderator_role_id(mod)
    out: dict[tuple, Path] = {}
    for arm in ARMS:
        for it in items:
            for rd in ("r0", "r1", "r2"):
                for rid in rcd.ROLE_ORDER:
                    out[(arm, it.item_id, rd, rid)] = rcd.call_cache_path(
                        model, SCREEN_SCAFFOLD, arm, it.item_id, 0, rd, rid, CAPS["agent"])
            out[(arm, it.item_id, "synthesis", mod_id)] = rcd.call_cache_path(
                mod, SCREEN_SCAFFOLD, arm, it.item_id, 0, "synthesis", mod_id, CAPS["moderator"])
            for rid in rcd.ROLE_ORDER:
                out[(arm, it.item_id, "r3_label", rid)] = rcd.call_cache_path(
                    model, SCREEN_SCAFFOLD, arm, it.item_id, 0, "r3_label", rid, CAPS["label"])
            out[(arm, it.item_id, "integration", mod_id)] = rcd.call_cache_path(
                mod, SCREEN_SCAFFOLD, arm, it.item_id, 0, "integration", mod_id, CAPS["moderator"])
            for rid in rcd.ROLE_ORDER:
                out[(arm, it.item_id, "r4_vote", rid)] = rcd.call_cache_path(
                    model, SCREEN_SCAFFOLD, arm, it.item_id, 0, "r4_vote", rid, CAPS["vote"])
    return out


def delib_footprint(model: str, items: Sequence[CrowdGoldItem], *,
                    moderator_model: Optional[str] = None) -> dict:
    """Classify every file in cgd_<model>_<scaffold>_* (and the moderator's
    namespace if it differs) as an expected replay path or FOREIGN."""
    expected = expected_delib_paths(model, items, moderator_model=moderator_model)
    names = {p.name: k for k, p in expected.items()}
    prefixes = {f"cgd_{_safe(model)}_{SCREEN_SCAFFOLD}_"}
    if moderator_model and moderator_model != model:
        prefixes.add(f"cgd_{_safe(moderator_model)}_{SCREEN_SCAFFOLD}_")
    present: dict[tuple, str] = {}
    foreign: list[str] = []
    refused: dict[tuple, int] = defaultdict(int)     # (arm, item) -> refused calls
    if rcd.OUT_DIR.is_dir():
        for p in rcd.OUT_DIR.iterdir():
            if not any(p.name.startswith(pre) for pre in prefixes):
                continue
            k = names.get(p.name)
            if k is None:
                foreign.append(p.name)
                continue
            present[k] = p.name
            try:
                if json.loads(p.read_text()).get("finish_reason") == "content_filter":
                    refused[(k[0], k[1])] += 1
            except (json.JSONDecodeError, OSError):
                pass
    by_cell: dict[tuple, int] = defaultdict(int)
    for k in present:
        by_cell[(k[0], k[1])] += 1
    return {"n_expected": len(expected), "n_present": len(present),
            "present_by_cell": dict(by_cell), "refused_by_cell": dict(refused),
            "foreign": sorted(foreign),
            "items_with_records": sorted({k[1] for k in present}),
            "items_with_refusal": sorted({k[1] for k in refused})}


def new_call_plan(footprint: dict, survivors: Sequence[CrowdGoldItem]) -> dict:
    """Calls on the surviving panel split into replays (cached at an expected
    path) and NEW calls, per round."""
    ids = {it.item_id for it in survivors}
    cells = len(ARMS) * len(ids)
    total = cells * rcd.CALLS_PER_CELL
    replay = sum(n for (arm, iid), n in footprint["present_by_cell"].items() if iid in ids)
    replay_cells = sum(1 for (arm, iid), n in footprint["present_by_cell"].items()
                       if iid in ids and n == rcd.CALLS_PER_CELL)
    refused_cells = sum(1 for (arm, iid) in footprint["refused_by_cell"] if iid in ids)
    return {"survivor_items": len(ids), "cells": cells, "total_calls": total,
            "replay_calls": replay, "replay_full_cells": replay_cells,
            "new_calls": total - replay,
            "cells_that_would_replay_a_refusal": refused_cells}


# ---------------------------------------------------------------------------
# Cost models (only NEW calls are counted)
# ---------------------------------------------------------------------------

def _measured_rounds(model: str) -> dict[str, dict]:
    """Per-round mean prompt/completion tokens over the model's own cached,
    un-refused chain records. Empty when the model has no chain records."""
    agg: dict[str, list] = defaultdict(lambda: [0, 0, 0])
    if not rcd.OUT_DIR.is_dir():
        return {}
    pre = f"cgd_{_safe(model)}_{SCREEN_SCAFFOLD}_"
    for p in rcd.OUT_DIR.iterdir():
        if not p.name.startswith(pre):
            continue
        try:
            rec = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if rec.get("finish_reason") != "stop":
            continue
        a = agg[rec.get("round", "")]
        a[0] += 1
        a[1] += int(rec.get("prompt_tokens") or 0)
        a[2] += int(rec.get("completion_tokens") or 0)
    return {rd: {"n": n, "prompt": pt / n, "completion": ct / n}
            for rd, (n, pt, ct) in agg.items() if n}


def screen_cost(model: str, items: Sequence[CrowdGoldItem], status: dict) -> dict:
    """Price the MISSING screen calls. Prompt tokens are counted on the real
    constructed prompts; completion is measured from the model's own cached
    single-agent records if any, else from its cached r0 chain records, else
    the runner's MEASURED_NCOT_COMPLETION / its max."""
    bpe = rca._optional_bpe_counter()
    system = rca.build_system(allow_unresolved=True, scaffold=SCREEN_SCAFFOLD,
                              instrument=rca.INSTRUMENT)
    by_id = {it.item_id: it for it in items}
    ptoks = 0
    for arm, iid, _idx in status["missing"]:
        ptoks += rca.count_tokens(system + "\n" + rca.build_user_turn(arm, by_id[iid].post_text),
                                  tokenizer=bpe)
    if status["mean_completion"] is not None:
        comp, src = status["mean_completion"], "measured on this model's cached screen records"
    else:
        r0 = _measured_rounds(model).get("r0")
        if r0:
            comp, src = r0["completion"], "measured on this model's cached r0 chain records"
        elif model in rcd.MEASURED_NCOT_COMPLETION:
            comp, src = rcd.MEASURED_NCOT_COMPLETION[model], "runner's MEASURED_NCOT_COMPLETION"
        else:
            comp, src = max(rcd.MEASURED_NCOT_COMPLETION.values()), "ASSUMED: max of the measured models"
    n_new = status["n_missing"]
    pin, pout = rcd.PRICES.get(model, (0.0, 0.0))
    priced = model in rcd.PRICES
    usd = ptoks / 1e6 * pin + n_new * comp / 1e6 * pout
    return {"new_calls": n_new, "replayed_calls": status["n_present"],
            "prompt_tokens": ptoks, "completion_tokens_each": comp,
            "completion_source": src, "priced": priced,
            "price_in_per_mtok": pin, "price_out_per_mtok": pout, "usd": usd,
            "tokenizer": "tiktoken cl100k_base" if bpe else "regex approximation"}


def delib_cost(model: str, survivors: Sequence[CrowdGoldItem], footprint: dict, *,
               moderator_model: Optional[str] = None) -> dict:
    """Price the NEW chain calls per round: prompt/completion from the model's
    own measured chain records where present, else the runner's cost model
    (constructed prompts + assumed completions) for that round."""
    plan = new_call_plan(footprint, survivors)
    measured = _measured_rounds(model)
    fallback = None
    if survivors:
        fallback = rcd.cost_model([model], list(ARMS), list(survivors), 1,
                                  scaffold=SCREEN_SCAFFOLD, transcript_cap=0,
                                  moderator_model=moderator_model)["per_model"][model]["rounds"]
    ids = {it.item_id for it in survivors}
    # replays per round: count present expected paths for survivor cells
    # (footprint stores per-cell totals; re-derive per round from disk names)
    per_round_present: dict[str, int] = defaultdict(int)
    exp = expected_delib_paths(model, survivors, moderator_model=moderator_model)
    for (arm, iid, rd, rid), p in exp.items():
        if iid in ids and p.exists():
            per_round_present[rd] += 1
    mod = moderator_model or model
    pin, pout = rcd.PRICES.get(model, (0.0, 0.0))
    mpin, mpout = rcd.PRICES.get(mod, (0.0, 0.0))
    rounds: dict[str, dict] = {}
    total = 0.0
    cells = plan["cells"]
    for rd in rcd.ROUNDS:
        calls_all = cells * (len(rcd.ROLE_ORDER) if rd in rcd.AGENT_ROUNDS else 1)
        n_new = calls_all - per_round_present.get(rd, 0)
        if rd in measured:
            pt, ctk, src = measured[rd]["prompt"], measured[rd]["completion"], "measured"
        elif fallback:
            pt, ctk, src = (fallback[rd]["prompt_tokens_each"],
                            fallback[rd]["completion_tokens_each"], "runner cost model")
        else:
            pt, ctk, src = 0, 0, "n/a"
        p_in, p_out = (mpin, mpout) if rd in rcd.MODERATOR_ROUNDS else (pin, pout)
        usd = n_new * (pt / 1e6 * p_in + ctk / 1e6 * p_out)
        total += usd
        rounds[rd] = {"calls_on_panel": calls_all, "replayed": per_round_present.get(rd, 0),
                      "new_calls": n_new, "prompt_tokens_each": round(pt, 1),
                      "completion_tokens_each": round(ctk, 1), "source": src,
                      "usd": round(usd, 2)}
    return {"plan": plan, "rounds": rounds, "usd_new_calls": round(total, 2),
            "priced": model in rcd.PRICES and mod in rcd.PRICES,
            "measured_rounds": sorted(measured)}


def print_delib_cost(model: str, dc: dict) -> None:
    plan = dc["plan"]
    print("\n" + "=" * 78)
    print(f"NEW-CALL COST MODEL -- {model} -- counts only calls not already cached at an "
          "expected path")
    print("=" * 78)
    print(f"  survivors {plan['survivor_items']} items x {len(ARMS)} arms = {plan['cells']} cells; "
          f"{plan['total_calls']} calls on the panel, {plan['replay_calls']} replay "
          f"({plan['replay_full_cells']} complete cells), {plan['new_calls']} NEW")
    if plan["cells_that_would_replay_a_refusal"]:
        print(f"  WARNING: {plan['cells_that_would_replay_a_refusal']} surviving cell(s) hold a cached "
              "content_filter record that would REPLAY the refusal; the screen did not "
              "catch that item (see --screen chain-aware)")
    print(f"  {'round':12s} {'on panel':>9} {'replay':>7} {'new':>6} {'prompt':>8} {'compl':>7}  "
          f"{'usd':>7}  source")
    for rd, r in dc["rounds"].items():
        print(f"  {rd:12s} {r['calls_on_panel']:9d} {r['replayed']:7d} {r['new_calls']:6d} "
              f"{r['prompt_tokens_each']:8.0f} {r['completion_tokens_each']:7.0f}  "
              f"{r['usd']:7.2f}  {r['source']}")
    print(f"  TOTAL NEW-CALL COST  ${dc['usd_new_calls']:.2f}"
          + ("" if dc["priced"] else "   *** WARNING: model not in PRICES; $0 is a placeholder ***"))
    print(f"  measured rounds: {dc['measured_rounds'] or 'none (all rounds from the runner cost model)'}")


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

def preflight_model(model: str) -> Optional[str]:
    if model in REGISTERED_MODELS:
        return (f"{model} already has a registered deliberation run on this panel; "
                "this two-step is for a model entering the panel behind a content-filter screen")
    return None


def preflight_screen(model: str, items: Sequence[CrowdGoldItem], *, resume: bool) -> tuple[bool, str]:
    existing = namespace_files(model, rca.OUT_DIR)
    if existing and not resume:
        return False, (f"screen namespace for {model} is not empty ({len(existing)} files, e.g. "
                       f"{existing[0]}); pass --resume to fill in the missing records only")
    return True, f"screen namespace holds {len(existing)} files"


def preflight_deliberate(model: str, items: Sequence[CrowdGoldItem], status: dict,
                         footprint: dict, *, resume: bool,
                         moderator_model: Optional[str] = None,
                         tag: Optional[str] = None) -> tuple[bool, str, int]:
    """(ok, reason, exit code). Order: cross-vendor moderator 2, unpriced
    (agent AND moderator) 2, tag holding another model's rows 2, screen
    incomplete 2, foreign 3."""
    why = moderator_guard(model, moderator_model)
    if why:
        return False, why, 2
    if model not in rcd.PRICES:
        return False, "model not in run_crowdgold_deliberation.PRICES; add a price before spending", 2
    if moderator_model and moderator_model not in rcd.PRICES:
        return False, (f"moderator {moderator_model!r} not in run_crowdgold_deliberation.PRICES; "
                       "add a price before spending"), 2
    if tag:
        others = rows_models_on_disk(rcd.OUT_DIR / f"{tag}_rows.csv") - {model}
        if others:
            return False, (f"{tag}_rows.csv already holds rows for {sorted(others)}; refusing to "
                           f"overwrite them with a {model} run. Pick another --tag."), 2
    if not status["complete"]:
        return False, (f"screen incomplete: {status['n_present']} of {status['n_expected']} single-agent "
                       "records present; run --step screen (--run) first"), 2
    if footprint["foreign"] and not resume:
        return False, (f"{len(footprint['foreign'])} FOREIGN file(s) in the deliberation namespace "
                       f"(not an expected replay path), e.g. {footprint['foreign'][0]}; "
                       "pass --resume only if they are known and inert"), 3
    return True, "ok", 0


# ---------------------------------------------------------------------------
# Completeness guard: every launched debate must be a row
# ---------------------------------------------------------------------------

def completeness_check(tag: str, model: str, expected: set, *,
                       out_dir: Optional[Path] = None,
                       summary_n_items: Optional[int] = None) -> dict:
    """rcd.run swallows an exception inside run_deliberation (one console
    ERROR line, the debate omitted from rows/votes/calls) and both base
    guards are computed on the survivors. Compare the rows CSV with the
    (arm, item, sample) keys the launch expected; any shortfall is a guard
    failure. `summary_n_items` (the base summary's n_items) is cross-checked
    against the expected item count so a divergence between this module's
    survivor set and the runner's is itself reported."""
    out_dir = rcd.OUT_DIR if out_dir is None else out_dir
    rows_path = out_dir / f"{tag}_rows.csv"
    present: set = set()
    n_rows = 0
    if rows_path.exists():
        with rows_path.open() as f:
            for r in csv.DictReader(f):
                if r.get("model") != model:
                    continue
                n_rows += 1
                present.add((r["arm"], r["item_id"], int(r.get("sample_idx") or 0)))
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    exp_items = {k[1] for k in expected}
    per_arm = {arm: {"expected": sum(1 for k in expected if k[0] == arm),
                     "rows": sum(1 for k in present if k[0] == arm)} for arm in ARMS}
    n_items_agree = (summary_n_items is None or summary_n_items == len(exp_items))
    ok = not missing and not extra and n_items_agree and bool(expected)
    return {"tag": tag, "model": model, "n_expected": len(expected), "n_rows": n_rows,
            "n_missing": len(missing), "n_unexpected": len(extra),
            "missing": [list(k) for k in missing], "unexpected": [list(k) for k in extra],
            "per_arm": per_arm, "n_items_expected": len(exp_items),
            "summary_n_items": summary_n_items, "summary_n_items_agrees": n_items_agree,
            "pass": ok}


def print_completeness(c: dict) -> None:
    print(f"\nCOMPLETENESS GUARD (every launched debate must be a row; rcd.run drops an errored "
          "debate with one console line)")
    print(f"  expected {c['n_expected']} debates ({c['n_items_expected']} items x {len(ARMS)} arms), "
          f"rows {c['n_rows']}, missing {c['n_missing']}, unexpected {c['n_unexpected']}; per arm "
          + "  ".join(f"{a}={v['rows']}/{v['expected']}" for a, v in c["per_arm"].items())
          + f"  {'PASS' if c['pass'] else 'FAIL'}")
    if c["summary_n_items"] is not None and not c["summary_n_items_agrees"]:
        print(f"  the runner's summary reports n_items {c['summary_n_items']} but this module "
              f"expected {c['n_items_expected']}: the survivor sets diverge")
    if c["missing"]:
        print("  missing: " + ", ".join(f"{a}/{i[:8]}/{s}" for a, i, s in c["missing"][:12])
              + (" ..." if len(c["missing"]) > 12 else ""))
    if not c["pass"]:
        print("\n  *** GUARD FAILED (completeness): errored debates were dropped by the runner; "
              "the surviving rows are arm-correlated attrition until they are re-run "
              "(--run --resume replays the cached calls). ***")


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def screen_argv(model: str, tag: str, *, workers: int, dry_run: bool,
                samples: int = SCREEN_SAMPLES) -> list[str]:
    """The single-agent runner's argv for the screen. ALWAYS the registered
    249-item panel (--n 99 --n-nta 150), never a smaller test panel: the
    single-agent runner SAMPLES n items per gold class with the seed
    (load_scruples_consensus(n=want, seed=44)), so `--n 6 --n-nta 6` there is
    a different set of items from the deliberation runner's sorted-prefix
    subset_items(panel, 6, 6). Only the full (99, 150) draw is the panel both
    runners agree on. The screen is in any case defined over the whole panel
    (rcd.filter_prone_items globs every item for the model), so a partial
    screen is never the right object; --n-yta/--n-nta on this module restrict
    step 2 and the status report only."""
    return (["--models", model, "--scaffolds", SCREEN_SCAFFOLD,
             "--max-tokens-scaffolded", str(SCREEN_CAP), "--n", str(N_YTA),
             "--n-nta", str(N_NTA), "--samples", str(samples), "--arms", ",".join(ARMS),
             "--seed", str(SEED), "--min-votes", str(MIN_VOTES),
             "--min-consensus", str(MIN_CONSENSUS), "--workers", str(workers),
             "--no-memorization", "--tag", tag]
            + (["--dry-run"] if dry_run else []))


def deliberate_argv(model: str, tag: str, *, n_yta: int, n_nta: int, workers: int,
                    n_boot: int, compare_rows: Path, moderator_model: Optional[str],
                    dry_run: bool) -> list[str]:
    return (["--models", model, "--arms", ",".join(ARMS), "--n-yta", str(n_yta),
             "--n-nta", str(n_nta), "--samples", "1", "--seed", str(SEED),
             "--min-votes", str(MIN_VOTES), "--min-consensus", str(MIN_CONSENSUS),
             "--agent-scaffold", SCREEN_SCAFFOLD,
             "--max-tokens-agent", str(CAPS["agent"]),
             "--max-tokens-moderator", str(CAPS["moderator"]),
             "--max-tokens-label", str(CAPS["label"]),
             "--max-tokens-vote", str(CAPS["vote"]),
             "--transcript-cap", "0", "--workers", str(workers),
             "--n-boot", str(n_boot), "--tag", tag,
             "--compare-rows", str(compare_rows)]
            + (["--moderator-model", moderator_model] if moderator_model else [])
            + (["--dry-run"] if dry_run else []))


def print_screen_report(model: str, items: Sequence[CrowdGoldItem], status: dict,
                        chain: dict) -> dict:
    ids = {it.item_id for it in items}
    reg = status["refused_items"]
    ch = chain["items"] & ids
    union = reg | ch
    print("\n" + "=" * 78)
    print(f"CONTENT-FILTER SCREEN -- {model} -- {len(items)}-item panel")
    print("=" * 78)
    print(f"  single-agent records: {status['n_present']} of {status['n_expected']} present"
          + ("" if status["complete"] else f"  ({status['n_missing']} MISSING -- screen not producible yet)"))
    print(f"  registered screen (runner's filter_prone_items): {len(reg)} item(s) refused "
          f"{ {a: len(v) for a, v in status['refused_by_arm'].items()} }")
    print(f"  chain evidence (cgd_{_safe(model)}_* refusals, {chain['n_records']} records): "
          f"{len(ch)} item(s) on this panel, by round {chain['by_round'] or {}}")
    print(f"    caught by the registered screen: {len(reg & ch)} of {len(ch)}; "
          f"chain-aware union would drop {len(union)}")
    print(f"  survivors: registered {len(ids) - len(reg)}   chain-aware {len(ids) - len(union)}")
    return {"registered": sorted(reg), "chain": sorted(ch), "union": sorted(union),
            "chain_caught_by_registered": len(reg & ch)}


def run_screen(model: str, tag: str, items: Sequence[CrowdGoldItem], *, workers: int,
               samples: int, run: bool, dry_run: bool, resume: bool) -> int:
    status = screen_status(model, items, samples)
    chain = chain_filtered_items(model)
    print(f"\n[filter-screen] STEP 1 -- single-agent screen calls for {model}")
    print(f"  namespace: {rcd.SCREEN_GLOB.format(model=_safe(model))}   tag {tag}   "
          f"samples per (arm, item): {samples}")
    print(f"  the screen is ALWAYS produced on the registered {N_YTA}+{N_NTA} panel; "
          f"the report below covers the {len(items)} item(s) this invocation inspects")
    print(f"  delegates to: python -m scripts.run_crowdgold_aita "
          + " ".join(screen_argv(model, tag, workers=workers, dry_run=False, samples=samples)))
    print_screen_report(model, items, status, chain)
    # what k buys, measured on the sibling caches behind the same filter
    fac = chain_refusal_factors(model)
    prone_rate, prone_src = ((fac["_prone_items"] / max(1, len(chain_prefix_items(model, items))),
                              f"{fac['_prone_items']} of the {len(chain_prefix_items(model, items))} "
                              f"items of {model}'s cached chain refused")
                             if fac else (0.0, "no chain records for this model"))
    print_projection(model, len(items), prone_rate, prone_src, screen_catch_rates(),
                     fac, ks=sorted({1, 2, 3, samples}))
    sc = screen_cost(model, items, status)
    print("\n" + "=" * 78)
    print(f"STEP-1 COST MODEL -- {model} -- NEW calls only (k={samples})")
    print("=" * 78)
    print(f"  new calls {sc['new_calls']}  (replayed from cache {sc['replayed_calls']})")
    print(f"  prompt tokens {sc['prompt_tokens']:,} ({sc['tokenizer']}); completion "
          f"{sc['completion_tokens_each']:.0f} each [{sc['completion_source']}]")
    print(f"  price {sc['price_in_per_mtok']}/{sc['price_out_per_mtok']} per Mtok -> "
          f"${sc['usd']:.2f}" + ("" if sc["priced"] else "   *** WARNING: model not in PRICES ***"))
    if dry_run or not run:
        # the single-agent runner's own dry-run, for the record
        rc = rca.main(screen_argv(model, tag, workers=workers, dry_run=True, samples=samples))
        if not run and not dry_run:
            print("\n*** NOT RUNNING: pass --run to generate (this spends money). "
                  "Nothing was generated. ***\n")
            return 2
        return rc
    if model not in rcd.PRICES:
        print("\nERROR: model not in run_crowdgold_deliberation.PRICES; add a price before spending.\n")
        return 2
    ok, why = preflight_screen(model, items, resume=resume)
    if not ok:
        print(f"\nERROR: {why}\n")
        return 3
    print(f"  {why}")
    rc = rca.main(screen_argv(model, tag, workers=workers, dry_run=False, samples=samples))
    # rc 4 from the single-agent runner is its NOVERDICT/truncation guard, which
    # a refusal trips by construction; the screen readout below is the result.
    print(f"\n[filter-screen] run_crowdgold_aita exited {rc} "
          f"({'its guard tripped, expected when calls are refused' if rc == 4 else 'ok' if rc == 0 else 'ERROR'})")
    if rc not in (0, 4):
        return rc
    status = screen_status(model, items, samples)
    rep = print_screen_report(model, items, status, chain_filtered_items(model))
    g = step1_guard(status)
    (rca.OUT_DIR / f"{tag}_screen.json").write_text(json.dumps({
        "model": model, "panel_items": len(items), "samples": samples, "status": {
            k: v for k, v in status.items() if k not in ("refused_items",)},
        "refused_items": sorted(status["refused_items"]), "screens": rep, "guard": g,
        "spend_measured": {"prompt_tokens": status["prompt_tokens"],
                           "completion_tokens": status["completion_tokens"],
                           "usd": (status["prompt_tokens"] / 1e6 * rcd.PRICES[model][0]
                                   + status["completion_tokens"] / 1e6 * rcd.PRICES[model][1])},
    }, indent=2, default=str))
    print(f"\nwrote {rca.OUT_DIR / (tag + '_screen.json')}")
    if not status["complete"]:
        print(f"\n  NOTE: {status['n_missing']} record(s) still missing (errored calls); "
              "re-run with --resume to fill them before step 2.")
    return 4 if not g["pass"] else 0


def run_deliberate(model: str, tag: str, screen_tag: str, items: Sequence[CrowdGoldItem], *,
                   n_yta: int, n_nta: int, workers: int, n_boot: int, screen: str,
                   samples: int, moderator_model: Optional[str], run: bool, dry_run: bool,
                   resume: bool) -> int:
    status = screen_status(model, items, samples)
    chain = chain_filtered_items(model)
    fp = delib_footprint(model, items, moderator_model=moderator_model)
    print(f"\n[filter-screen] STEP 2 -- screened deliberation for {model}   tag {tag}   "
          f"(screen at k={samples})")
    rep = print_screen_report(model, items, status, chain)
    if status["complete"] and rep["chain"]:
        # the screen is on disk: its catch rate on the items the 16.11b chain
        # actually refused is MEASURED, and the projection for the rest of the
        # panel uses it instead of the sibling-cache rate.
        c = rep["chain_caught_by_registered"] / len(rep["chain"])
        prefix = chain_prefix_items(model, items)
        n_new = len(items) - len(prefix)
        fac = chain_refusal_factors(model)
        worst_rd, wf = worst_round(fac)
        prone_rate = len(rep["chain"]) / max(1, len(prefix))
        unc_new = n_new * prone_rate * (1 - c)
        surv = len(items) - len(rep["registered"])
        print(f"  MEASURED catch rate of this screen on the {len(rep['chain'])} chain-refused items: "
              f"{c:.2f}; at the chain's prone rate {prone_rate:.3f} that leaves ~{unc_new:.1f} "
              f"uncaught prone items among the {n_new} items with no chain record; projected "
              f"worst-round ({worst_rd or 'integration'}) NOVERDICT ~{wf * unc_new / max(1, surv):.3f} "
              f"vs limit {rca.MAX_NOVERDICT_SHARE:.0%} "
              f"({'projected PASS' if wf * unc_new / max(1, surv) <= rca.MAX_NOVERDICT_SHARE else 'projected GUARD FAIL -- raise --screen-samples and re-run step 1 with --resume'})")
    drop = set(rep["union"] if screen == "chain-aware" else rep["registered"])
    survivors = [it for it in items if it.item_id not in drop]
    print(f"  --screen {screen}: dropping {len(drop)} -> {len(survivors)} survivors "
          f"({sum(1 for i in survivors if i.gold_verdict == 'YTA')} YTA + "
          f"{sum(1 for i in survivors if i.gold_verdict == 'NTA')} NTA)")
    print(f"\n  deliberation namespace cgd_{_safe(model)}_{SCREEN_SCAFFOLD}_*: "
          f"{fp['n_present']} expected-path records over {len(fp['items_with_records'])} item(s)"
          f" (replay candidates), {len(fp['foreign'])} foreign")
    if fp["items_with_refusal"]:
        caught = [i for i in fp["items_with_refusal"] if i in drop]
        print(f"  items whose cached chain records hold a refusal: {len(fp['items_with_refusal'])}; "
              f"{len(caught)} dropped by --screen {screen}, "
              f"{len(fp['items_with_refusal']) - len(caught)} would REPLAY the refusal")
    compare_rows = rca.OUT_DIR / f"{screen_tag}_rows.csv"
    argv = deliberate_argv(model, tag, n_yta=n_yta, n_nta=n_nta, workers=workers, n_boot=n_boot,
                           compare_rows=compare_rows, moderator_model=moderator_model,
                           dry_run=(dry_run or not run))
    print(f"  delegates to: python -m scripts.run_crowdgold_deliberation " + " ".join(argv))
    if not compare_rows.exists():
        print(f"  NOTE: {compare_rows.name} not on disk yet (step 1 writes it); the "
              "criterion/resistance read-out will have no same-model comparator")
    dc = delib_cost(model, survivors, fp, moderator_model=moderator_model)
    print_delib_cost(model, dc)
    if not status["complete"]:
        # The screen is not on disk yet, so the count above is the UNSCREENED
        # panel. Project what step 2 will cost once step 1 has landed at k.
        rates, fac = screen_catch_rates(), chain_refusal_factors(model)
        prefix = chain_prefix_items(model, items)
        if rates.get("pooled") and fac:
            _, wf = worst_round(fac)
            prone_rate = fac["_prone_items"] / max(1, len(prefix))
            pr = refusal_projection(k=samples, catch=rates["pooled"].get(samples), panel_n=len(items),
                                    prone_rate=prone_rate, worst_round_factor=wf)
            clean_prefix = len(prefix) - fac["_prone_items"]
            exp_cells = max(0.0, pr["survivors"] - clean_prefix) * len(ARMS)
            per_call = dc["usd_new_calls"] / max(1, dc["plan"]["new_calls"])
            print(f"\n  PROJECTED for step 2 once step 1 lands at k={samples} (catch {pr['catch']:.2f}): "
                  f"~{pr['survivors']:.0f} survivors; the {clean_prefix} chain-clean items of the cached "
                  f"prefix replay in full ({clean_prefix * len(ARMS) * rcd.CALLS_PER_CELL} calls, $0); "
                  f"~{exp_cells:.0f} new cells x {rcd.CALLS_PER_CELL} = ~{exp_cells * rcd.CALLS_PER_CELL:.0f} "
                  f"new calls ~= ${exp_cells * rcd.CALLS_PER_CELL * per_call:.2f} at the measured per-call "
                  f"cost above; ~{pr['uncaught']:.1f} uncaught prone items -> projected worst-round NOVERDICT "
                  f"{pr['worst_round_noverdict_projected']:.3f} vs {rca.MAX_NOVERDICT_SHARE:.0%} "
                  f"({'PASS' if pr['projected_pass'] else 'GUARD FAIL'})")
    if screen == "chain-aware":
        install_chain_aware_screen(model)
    if dry_run or not run:
        rc = rcd.main(argv)      # the runner's own dry-run (upper bound; assumed completions)
        (rcd.OUT_DIR / f"{tag}_step2_plan.json").write_text(json.dumps({
            "model": model, "screen": screen, "screen_samples": samples, "screens": rep,
            "survivors": len(survivors),
            "footprint": {k: v for k, v in fp.items() if k not in ("present_by_cell", "refused_by_cell")},
            "refused_calls_by_cell": {f"{arm}|{iid}": n for (arm, iid), n in fp["refused_by_cell"].items()},
            "new_call_cost": dc}, indent=2, default=str))
        if not run and not dry_run:
            print("\n*** NOT RUNNING: pass --run to generate (this spends money). "
                  "Nothing was generated. ***\n")
            return 2
        return rc
    ok, why, code = preflight_deliberate(model, items, status, fp, resume=resume,
                                         moderator_model=moderator_model, tag=tag)
    if not ok:
        print(f"\nERROR: {why}\n")
        return code
    if fp["foreign"]:
        print(f"  --resume: {len(fp['foreign'])} foreign file(s) present and ignored")
    rc = rcd.main(argv)
    if rc == 4:
        print("\n  *** GUARD FAILED (round-level or outcome guard of the deliberation runner). "
              "Do not read G1-G3, role-lock, lift or transfer off this run. ***\n")
    if rc in (0, 1, 4):
        # rc 1 = "No rows produced": every debate errored, which is a total shortfall
        summ_p = rcd.OUT_DIR / f"{tag}_summary.json"
        sn = None
        if summ_p.exists():
            try:
                sn = json.loads(summ_p.read_text()).get("n_items")
            except json.JSONDecodeError:
                sn = None
        comp = completeness_check(tag, model, {(arm, it.item_id, 0) for arm in ARMS for it in survivors},
                                  summary_n_items=sn)
        print_completeness(comp)
        (rcd.OUT_DIR / f"{tag}_completeness.json").write_text(json.dumps(comp, indent=2))
        print(f"  wrote {rcd.OUT_DIR / (tag + '_completeness.json')}")
        if not comp["pass"]:
            rc = 4
    return rc


# ---------------------------------------------------------------------------
# Step 3: the grip-table row (offline)
# ---------------------------------------------------------------------------

def _boot(by_item: dict, stat: Callable[[list], Optional[float]], *, draws: int,
          seed: int, alpha: float = 0.05):
    ids = sorted(by_item)
    rng = random.Random(seed)
    point = stat([x for i in ids for x in by_item[i]])
    vals = sorted(v for v in (
        stat([x for i in (ids[rng.randrange(len(ids))] for _ in ids) for x in by_item[i]])
        for _ in range(draws)) if v is not None)
    if not vals:
        return point, None, None
    return (point, vals[int(alpha / 2 * len(vals))],
            vals[min(int((1 - alpha / 2) * len(vals)), len(vals) - 1)])


def r0_verdict_code(model: str, arm: str, item: str, idx: int, role_id: str) -> Optional[int]:
    from scripts.analyze_crowdgold_sdt import code_response
    from scripts.verdict_format import extract_verdict
    p = rcd.call_cache_path(model, SCREEN_SCAFFOLD, arm, item, idx, "r0", role_id, CAPS["agent"])
    if not p.exists():
        return None
    txt = json.loads(p.read_text()).get("output") or ""
    return code_response(extract_verdict(txt, rcd.INSTRUMENT), "published")


def matched_16_10_items() -> Optional[set]:
    """Intersection of the item sets of the four 16.10 cells, when all are on
    disk (the 210-item matched panel); None otherwise."""
    try:
        from scripts.analyze_topology_2x2 import CELL_SPECS, MODEL as GROK
        from scripts.analyze_unembodied_ablation import load_rows
    except Exception:  # noqa: BLE001
        return None
    sets = []
    for tag, _ in CELL_SPECS.values():
        p = OUT_DIR / f"{tag}_rows.csv"
        if not p.exists():
            return None
        sets.append({r["item"] for r in load_rows(p, GROK)})
    return set.intersection(*sets) if sets else None


def grip_table_row(model: str, tag: str, *, screen_tag: Optional[str] = None,
                   keep_items: Optional[set] = None, draws: int = 2000,
                   seed: int = 29, comparator_path: Optional[Path] = None) -> Optional[dict]:
    """The 16.10-table columns for one model, by the definitions of
    analyze_topology_2x2.analyse_cell, plus analyze_stake_grip's G1-G3."""
    import scripts.analyze_stake_grip as asg
    from scripts.analyze_actuator_ladder import (COMPARATOR_PATH, correct,
                                                 load_comparator, make_rule_vendor)
    from scripts.analyze_unembodied_ablation import (load_rows, localisation,
                                                     per_seat_objection_rates)
    rows_path, votes_path = rcd.OUT_DIR / f"{tag}_rows.csv", rcd.OUT_DIR / f"{tag}_votes.csv"
    if not rows_path.exists() or not votes_path.exists():
        return None
    rows = load_rows(rows_path, model, keep_items)
    if not rows:
        return None
    seats = tuple(rcd.ROLE_ORDER)
    rates = per_seat_objection_rates(votes_path, model, keep_items)
    loc = localisation(rows, rates)
    codable = [r for r in rows if r["codable"]]

    # --- G1-G3 via analyze_stake_grip on the same debates -------------------
    debates = asg.load_debates(rows_path, votes_path)
    if keep_items is not None:
        debates = [d for d in debates if d["item"] in keep_items]
    keys = {(d["item"], d["arm"], d["sample_idx"]) for d in debates}
    votes = [v for v in asg.load_votes(votes_path) if (v["item"], v["arm"], v["sample_idx"]) in keys]
    g = asg.grip(debates, votes, draws=draws, seed=seed) if debates else None

    # --- r0: role-lock and seat accuracy --------------------------------------
    at = {s: [0, 0] for s in seats}
    acc = {s: [0, 0] for s in seats}
    for r in rows:
        gold = 1 if r["gold"] == "YTA" else 0
        for s in seats:
            c = r0_verdict_code(model, r["arm"], r["item"], r["sample_idx"], s)
            if c is None:
                continue
            at[s][0] += c; at[s][1] += 1
            acc[s][0] += int(c == gold); acc[s][1] += 1
    p_at = {s: (at[s][0] / at[s][1] if at[s][1] else None) for s in seats}
    seat_acc = {s: (acc[s][0] / acc[s][1] if acc[s][1] else None) for s in seats}
    lock = (abs(p_at[seats[0]] - p_at[seats[1]])
            if None not in (p_at[seats[0]], p_at[seats[1]]) else None)

    # --- fire and lift (stake-blind counter, codable, >= MIN_FIRED) ----------
    fired = lambda r: r["n_objectors"] >= 2  # noqa: E731
    n_fired = sum(1 for r in rows if fired(r))
    byl: dict = defaultdict(list)
    for r in codable:
        byl[r["item"]].append(r)

    def lift_stat(rs):
        fl = [r for r in rs if fired(r)]; un = [r for r in rs if not fired(r)]
        return None if not fl or not un else (
            (1 - sum(bool(r["s2_ok"]) for r in fl) / len(fl))
            - (1 - sum(bool(r["s2_ok"]) for r in un) / len(un)))
    l_p, l_lo, l_hi = _boot(byl, lift_stat, draws=draws, seed=seed + 1)

    # --- transfer: onto grok solo (standard maj-3) and onto own solo (k=1) ---
    def transfer(vendor) -> dict:
        fl_c: dict = defaultdict(int); k_c: dict = defaultdict(int)
        for r in rows:
            c = (r["arm"], r["item"]); fl_c[c] += int(fired(r)); k_c[c] += 1
        gold = {(r["arm"], r["item"]): r["gold"] for r in rows}
        trec = []
        for c, gd in gold.items():
            v = vendor({"arm": c[0], "item": c[1]})
            ok = correct(v, gd) if v is not None else None
            if ok is None:
                continue
            trec.append({"item": c[1], "w": int(not ok), "f": int(fl_c[c] * 2 > k_c[c])})
        byt: dict = defaultdict(list)
        for x in trec:
            byt[x["item"]].append(x)

        def tstat(rs):
            f = [x for x in rs if x["f"]]; u = [x for x in rs if not x["f"]]
            return None if not f or not u else (sum(x["w"] for x in f) / len(f)
                                               - sum(x["w"] for x in u) / len(u))
        t_p, t_lo, t_hi = _boot(byt, tstat, draws=draws, seed=seed + 2)
        tf = [x for x in trec if x["f"]]; tu = [x for x in trec if not x["f"]]
        return {"n_cells": len(trec), "n_flagged_cells": len(tf),
                "p_wrong_flagged": (sum(x["w"] for x in tf) / len(tf) if tf else None),
                "p_wrong_unflagged": (sum(x["w"] for x in tu) / len(tu) if tu else None),
                "lift": t_p, "lo": t_lo, "hi": t_hi}

    cpath = comparator_path or COMPARATOR_PATH
    t_grok = (transfer(make_rule_vendor(load_comparator(cpath), "grok-4-1-fast-reasoning", "standard"))
              if cpath.exists() else {"n_cells": 0, "lift": None, "lo": None, "hi": None,
                                      "p_wrong_flagged": None, "p_wrong_unflagged": None,
                                      "n_flagged_cells": 0})
    own_path = rca.OUT_DIR / f"{screen_tag or default_tags(model)[0]}_rows.csv"
    own_k = None
    if own_path.exists():
        comp_own = load_comparator(own_path)
        ks = {len(v) for (m, sc, a, i), v in comp_own.items() if m == model and sc == SCREEN_SCAFFOLD}
        own_k = max(ks) if ks else 0
    t_own = (transfer(make_rule_vendor(comp_own, model, SCREEN_SCAFFOLD))
             if own_path.exists() else None)
    if t_own is not None:
        t_own["k"] = own_k

    return {
        "model": model, "tag": tag, "n_debates": len(rows),
        "n_items": len({r["item"] for r in rows}), "n_codable": len(codable),
        "role_lock": lock, "p_at_fault": p_at, "seat_r0_accuracy": seat_acc,
        "objection": {"per_seat": rates["per_seat"], "pooled": rates["pooled"],
                      "localisation_excess": loc["localisation_excess"],
                      "fire_rate": loc["fire_observed"],
                      "fire_independence": loc["fire_independence"]},
        "grip": (None if g is None else {
            "G1_fire_rate": g["G1_fire_rate"], "G2_reject_share": g["G2_reject_share"],
            "G3": g["G3_stake_concentration"], "criteria": g["criteria"], "grip": g["grip"],
            "per_seat": g["per_seat"], "s2_accuracy": g["s2_accuracy"]}),
        "lift": {"n_fired": n_fired, "powered": n_fired >= MIN_FIRED,
                 "delta": l_p, "lo": l_lo, "hi": l_hi},
        "transfer_grok_solo": t_grok, "transfer_own_solo": t_own,
        # codable population (group verdict codes under the published collapse),
        # the convention of analyze_topology_2x2.group_verdict_s2 and of the lift
        "s2_accuracy": ((sum(bool(r["s2_ok"]) for r in rows if r["s2_ok"] is not None)
                         / sum(1 for r in rows if r["s2_ok"] is not None))
                        if any(r["s2_ok"] is not None for r in rows) else None),
        "s2_n_codable": sum(1 for r in rows if r["s2_ok"] is not None),
        "guards": {"mean_n_r3_unparsed": sum(r["n_r3_unparsed"] for r in rows) / len(rows),
                   "p_noncodable": 1 - len(codable) / len(rows)},
    }


def _f(x, fmt="+.3f"):
    return "n/a" if x is None else format(x, fmt)


def print_grip_table_row(row: dict, *, label: str, reference: Optional[dict] = None) -> None:
    g = row["grip"] or {}
    g3 = g.get("G3") or {}
    print("\n" + "=" * 78)
    print(f"GRIP-TABLE ROW -- {row['model']} -- {label}   debates {row['n_debates']}  "
          f"items {row['n_items']}  codable {row['n_codable']}")
    print("=" * 78)
    ref = (lambda k: f"   [16.10 embodied/on: {reference[k]}]" if reference and k in reference else "")
    print(f"  ROLE-LOCK                     {_f(row['role_lock'], '.3f')}{ref('role_lock')}")
    print(f"    P(at_fault) " + "  ".join(f"{s.split('_')[0][:8]}={_f(v, '.3f')}" for s, v in row["p_at_fault"].items()))
    print(f"    seat r0 acc " + "  ".join(f"{s.split('_')[0][:8]}={_f(v, '.3f')}" for s, v in row["seat_r0_accuracy"].items()))
    print(f"  G1 fire rate (composite)      {_f(g.get('G1_fire_rate'), '.3f')}   "
          f"{'PASS' if g.get('criteria', {}).get('G1') else 'fail'}")
    print(f"  G2 reject share               {_f(g.get('G2_reject_share'), '.3f')}   "
          f"{'PASS' if g.get('criteria', {}).get('G2') else 'fail'}")
    print(f"  G3 stake concentration        {_f(g3.get('delta'))} [{_f(g3.get('lo'))}, {_f(g3.get('hi'))}]   "
          f"{'PASS' if g.get('criteria', {}).get('G3') else 'fail'}{ref('g3')}")
    print(f"  GRIP                          {g.get('grip')}")
    print(f"  localisation excess           {_f(row['objection']['localisation_excess'])}")
    print(f"  fire rate P(>=2 objectors)    {_f(row['objection']['fire_rate'], '.3f')}{ref('fire_rate')}")
    lf = row["lift"]
    print(f"  error lift (>=2), codable     "
          f"{_f(lf['delta']) if lf['powered'] else 'n/a (' + str(lf['n_fired']) + ' fired < ' + str(MIN_FIRED) + ')'}"
          f" [{_f(lf['lo'])}, {_f(lf['hi'])}]  n fired {lf['n_fired']}{ref('lift')}")
    t = row["transfer_grok_solo"]
    print(f"  TRANSFER lift on grok solo    {_f(t['lift'])} [{_f(t['lo'])}, {_f(t['hi'])}]  "
          f"P(wrong|flagged) {_f(t['p_wrong_flagged'], '.3f')} vs {_f(t['p_wrong_unflagged'], '.3f')}  "
          f"cells {t['n_cells']}{ref('transfer')}")
    t2 = row["transfer_own_solo"]
    if t2:
        print(f"  transfer on own solo (maj k={t2.get('k')}) {_f(t2['lift'])} [{_f(t2['lo'])}, {_f(t2['hi'])}]  "
              f"P(wrong|flagged) {_f(t2['p_wrong_flagged'], '.3f')} vs {_f(t2['p_wrong_unflagged'], '.3f')}")
    else:
        print("  transfer on own solo          n/a (step-1 rows not on disk)")
    print(f"  S2 accuracy (codable)         {_f(row['s2_accuracy'], '.3f')}   n codable "
          f"{row['s2_n_codable']} of {row['n_debates']}")
    print(f"  guard: mean n_r3_unparsed     {_f(row['guards']['mean_n_r3_unparsed'], '.4f')}   "
          f"P(non-codable) {_f(row['guards']['p_noncodable'], '.3f')}")


def reference_16_10() -> Optional[dict]:
    p = OUT_DIR / "topology_2x2_analysis.json"
    if not p.exists():
        return None
    try:
        c = json.loads(p.read_text())["cells"]["embodied/on"]
    except (KeyError, json.JSONDecodeError):
        return None
    return {"role_lock": _f(c.get("role_lock"), ".3f"),
            "fire_rate": _f(c["objection"].get("fire_rate"), ".3f"),
            "g3": _f(c["g3"].get("delta")) if c["g3"].get("applicable") else "n/a",
            "lift": _f(c["lift"].get("delta")) if c["lift"].get("powered") else "n/a",
            "transfer": _f(c["transfer_grok_solo"].get("lift"))}


def run_readout(model: str, tag: str, screen_tag: str, *, draws: int) -> int:
    import scripts.analyze_stake_grip as asg
    rows_path = rcd.OUT_DIR / f"{tag}_rows.csv"
    if not rows_path.exists():
        print(f"\nERROR: {rows_path} not on disk; run --step deliberate first.\n")
        return 2
    gf = rcd.OUT_DIR / f"{tag}_summary.json"
    cf = rcd.OUT_DIR / f"{tag}_completeness.json"
    guard_failed = False
    reasons: list[str] = []
    if gf.exists():
        s = json.loads(gf.read_text())
        rg, og = s.get("round_guard", {}), s.get("outcome_guard", {})
        if not (rg.get("pass", False) and og.get("pass", False)):
            reasons.append(f"round guard {rg.get('pass')}, outcome guard {og.get('pass')}")
    else:
        reasons.append(f"{gf.name} missing: the run's guard verdict is unknown")
    if cf.exists():
        c = json.loads(cf.read_text())
        if not c.get("pass", False):
            reasons.append(f"completeness guard: {c.get('n_missing')} of {c.get('n_expected')} "
                           "launched debates missing from the rows")
    else:
        reasons.append(f"{cf.name} missing: it is not known whether every launched debate is a row")
    if reasons:
        guard_failed = True
        print(f"\n  *** GUARD FAILED on {tag} ({'; '.join(reasons)}): the numbers below are NOT "
              "to be tabulated; exit 4. ***\n")
    print(f"\n[filter-screen] STEP 3 -- readout for {model} / {tag}")
    # analyze_stake_grip MERGES into the shared registry stake_grip_analysis.json
    # keyed model:tag. A run whose guard failed must not touch that registry
    # (its numbers are UNREAD), so it goes to a per-tag file instead.
    grip_json = (rcd.OUT_DIR / (f"{tag}_stake_grip_UNREAD.json" if guard_failed
                                else "stake_grip_analysis.json"))
    print(f"  (a) analyze_stake_grip, the registered G1-G3 readout (-> {grip_json.name}):")
    # analyze_stake_grip resolves <tag>_rows/_votes.csv against its own module
    # OUT_DIR; point it at the deliberation cache dir in force here (the real
    # one, or the selftest's tempdir) for the duration of the call.
    asg_out = asg.OUT_DIR
    asg.OUT_DIR = rcd.OUT_DIR
    try:
        asg.main(["--tag", tag, "--model", model, "--draws", str(draws), "--json", str(grip_json)])
    finally:
        asg.OUT_DIR = asg_out
    print("\n  (b) the 16.10-table columns, same definitions as analyze_topology_2x2:")
    ref = reference_16_10()
    own = grip_table_row(model, tag, screen_tag=screen_tag, draws=draws)
    if own is None:
        print("  no rows for this model in the tag's CSV")
        return 1
    print_grip_table_row(own, label="own surviving items", reference=ref)
    matched = matched_16_10_items()
    out = {"own": own}
    if matched is not None:
        both = grip_table_row(model, tag, screen_tag=screen_tag, keep_items=matched, draws=draws)
        if both:
            print_grip_table_row(both, label=f"intersection with the 16.10 matched set ({len(matched)} items)",
                                 reference=ref)
            out["matched_16_10"] = both
    out = {"UNREAD_GUARD_FAILED": True, "guard_reasons": reasons, **out} if guard_failed else out
    p = rcd.OUT_DIR / f"{tag}_grip_row.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {p}" + ("   (flagged UNREAD_GUARD_FAILED: not to be tabulated)" if guard_failed else ""))
    if guard_failed:
        print(f"\n  *** GUARD FAILED -- {tag} is not readable ({'; '.join(reasons)}). ***\n")
        return 4
    return 0


# ---------------------------------------------------------------------------
# Selftest (offline; both generators stubbed; both cache dirs redirected)
# ---------------------------------------------------------------------------

def _make_single_stub(filtered_user: str, refuse_on_sample: int = 1):
    """Refuses the filtered user turn on ONE sample index only, the way the
    Azure filter refuses a prone prompt in some samples and not others; a
    k=1 screen therefore misses it and a k>=2 screen catches it."""
    def stub(model, system, user, *, sample_idx=0, max_tokens=512, **kw):
        if user == filtered_user and sample_idx == refuse_on_sample:
            return rcd._StubResult("", finish_reason="content_filter", prompt_tokens=700)
        return rcd._StubResult(
            "Section 1 - Protagonist: the writer.\nSection 2 - Stakeholders: all.\n"
            "Section 3 - Consequences: two steps.\nSection 4 - Uncertainty: some.\n"
            "Section 5 - Decision: stated.\nVERDICT: NTA",
            finish_reason="stop", prompt_tokens=700, completion_tokens=480)
    return stub


def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    model = "gpt-4o"
    # --- 1. configuration coherent with the runner --------------------------
    check("SCREEN_CAP is the cap rcd.SCREEN_GLOB reads",
          f"_t{SCREEN_CAP}_" in rcd.SCREEN_GLOB and SCREEN_SCAFFOLD in rcd.SCREEN_GLOB)
    check("registered panel defaults match the deliberation runner's",
          (N_YTA, N_NTA) == (rcd.PILOT_N_YTA, rcd.PILOT_N_NTA) and ARMS == rcd.PILOT_ARMS)
    check("default tags: gpt-4o -> cg_screen_gpt4o / cg_deliberation_gpt4o",
          default_tags("gpt-4o") == ("cg_screen_gpt4o", "cg_deliberation_gpt4o"))
    check("the four registered models are refused; gpt-4o is not",
          all(preflight_model(m) for m in REGISTERED_MODELS) and preflight_model("gpt-4o") is None)
    check("gpt-4o is priced (PRICES lookup, no code change for a new deployment beyond a price)",
          "gpt-4o" in rcd.PRICES and rcd.PRICES["gpt-4o"] == (2.50, 10.00))
    check("--keep-filtered can never reach the runner (not in deliberate_argv)",
          "--keep-filtered" not in deliberate_argv("m", "t", n_yta=1, n_nta=1, workers=1, n_boot=1,
                                                   compare_rows=Path("x"), moderator_model=None,
                                                   dry_run=True))
    check("GUARD FAILED literal is printed by the step-1 guard and on a step-2 rc 4",
          "GUARD FAILED" in inspect.getsource(step1_guard) and "GUARD FAILED" in inspect.getsource(run_deliberate))
    check("GUARD FAILED literal is printed by the completeness guard and by the step-3 readout",
          "GUARD FAILED" in inspect.getsource(print_completeness)
          and "GUARD FAILED" in inspect.getsource(run_readout))
    check("tag guard: the default tags pass; cg_deliberation / cg_deliberation_stdcot / the 16.11b tag "
          "cg_deliberation_gpt4o_screen / a foreign --screen-tag are refused for gpt-4o",
          tag_guard("gpt-4o", "cg_deliberation_gpt4o", "cg_screen_gpt4o") is None
          and tag_guard("gpt-4o", "cg_deliberation_gpt4o_v2", "cg_screen_gpt4o_k3") is None
          and tag_guard("gpt-4o", "cg_deliberation", "cg_screen_gpt4o") is not None
          and tag_guard("gpt-4o", "cg_deliberation_stdcot", "cg_screen_gpt4o") is not None
          and tag_guard("gpt-4o", PRIOR_SCREEN_TAG, "cg_screen_gpt4o") is not None
          and tag_guard("gpt-4o", "cg_deliberation_gpt4o", "cg_gen_x") is not None)
    check("moderator guard: None and the agent model itself pass; any other model is refused with a "
          "pointer to run_crowdgold_modvendor",
          moderator_guard("gpt-4o", None) is None and moderator_guard("gpt-4o", "gpt-4o") is None
          and "run_crowdgold_modvendor" in (moderator_guard("gpt-4o", "claude-sonnet-4-6") or "")
          and moderator_guard("gpt-4o", "vendor4-deploy") is not None)

    # --- 2. panel ------------------------------------------------------------
    try:
        panel = load_panel()
    except Exception as e:  # noqa: BLE001
        check(f"panel loads ({type(e).__name__}: {e})", False)
        print(f"\n{len(fails)} FAILED")
        return 1
    check("registered panel is 249 = 99 YTA + 150 NTA", len(panel) == 249
          and sum(1 for i in panel if i.gold_verdict == "YTA") == 99)
    sub = load_panel(6, 6)
    check("a smaller panel is the deterministic prefix the runner takes",
          [i.item_id for i in sub] == [i.item_id for i in rcd.subset_items(panel, 6, 6)])

    # --- 3. screen paths are exactly what the runner globs -------------------
    sp = screen_paths(model, panel)
    check(f"{2 * 249 * SCREEN_SAMPLES} screen paths for the panel (2 arms x 249 items x k={SCREEN_SAMPLES}); "
          "498 at k=1", len(sp) == 2 * 249 * SCREEN_SAMPLES and len(set(sp.values())) == len(sp)
          and len(screen_paths(model, panel, 1)) == 498)
    check("every screen path fnmatches rcd.SCREEN_GLOB for the model",
          all(screen_glob_matches(model, p) for p in sp.values()))
    check("screen path is run_crowdgold_aita's own gen_cache_path (scaffold, t2560, arm, item, idx)",
          sp[(THIRD_PERSON, panel[0].item_id, 0)].name
          == f"cg_gen_gpt-4o_narrative_cot_t2560_third_person_{panel[0].item_id}_00.json"
          and sp[(AS_ASKER, panel[0].item_id, 2)].name
          == f"cg_gen_gpt-4o_narrative_cot_t2560_as_asker_{panel[0].item_id}_02.json")
    check("a cg_gen file at another cap or scaffold does NOT match the screen glob",
          not screen_glob_matches(model, Path("cg_gen_gpt-4o_narrative_cot_t1024_as_asker_x_00.json"))
          and not screen_glob_matches(model, Path("cg_gen_gpt-4o_as_asker_x_00.json")))

    # --- 4. expected deliberation footprint -----------------------------------
    exp = expected_delib_paths(model, panel[:2])
    names = [p.name for p in exp.values()]
    check("expected footprint: 17 paths per (arm, item), all distinct",
          len(exp) == 2 * 2 * rcd.CALLS_PER_CELL and len(set(names)) == len(names))
    check("footprint caps: 9 x t2560, 2 x t1024 (mod-<model>), 6 x t3072 per cell",
          sum("_t2560_" in n for n in names) == 4 * 9 and sum("_t1024_" in n for n in names) == 4 * 2
          and sum("_t3072_" in n for n in names) == 4 * 6
          and sum("_mod-gpt-4o.json" in n for n in names) == 4 * 2)
    check("footprint uses the registered seat ids and the runner's moderator prefix verbatim",
          all(any(n.endswith(f"_{rid}.json") for rid in rcd.ROLE_ORDER) or n.endswith("_mod-gpt-4o.json")
              for n in names))

    # --- 5. the real cache, reported (never asserted as a count) --------------
    real_ns = namespace_files(model, OUT_DIR)
    print(f"      real screen namespace for {model}: {len(real_ns)} file(s)")
    check("no stub / selftest file carries this module's names in the real cache dir",
          not any(("stub-model" in n or "selftest" in n) for n in real_ns)
          and not any(p.name.startswith("cgd_stub-model_narrative_cot") or p.name.startswith("cg_gen_stub-model_")
                      for p in OUT_DIR.iterdir()))
    fp_real = delib_footprint(model, panel)
    print(f"      real deliberation footprint for {model}: {fp_real['n_present']} expected-path records over "
          f"{len(fp_real['items_with_records'])} item(s), {len(fp_real['foreign'])} foreign, "
          f"{len(fp_real['items_with_refusal'])} item(s) with a cached refusal")
    check("every cgd_gpt-4o record on disk is an expected replay path (16.11b footprint; "
          "a foreign file would make the run path refuse without --resume)",
          fp_real["foreign"] == [])
    prior_rows = OUT_DIR / f"{PRIOR_SCREEN_TAG}_rows.csv"
    if prior_rows.exists():
        ids40 = {r["item_id"] for r in csv.DictReader(prior_rows.open())}
        check("16.11b's 40 items are the deterministic (16, 24) prefix of the panel, so its "
              "records replay for its surviving items",
              ids40 == {i.item_id for i in rcd.subset_items(panel, 16, 24)})
        ch = chain_filtered_items(model)
        check("chain-evidence refusals for gpt-4o are a subset of the panel and lie within the 16.11b items",
              ch["items"] <= {i.item_id for i in panel} and ch["items"] <= ids40)
        print(f"      chain-evidence refusals (measured): {len(ch['items'])} of 40 items, by round {ch['by_round']}")
    else:
        print("      (16.11b rows not on disk; prefix and chain-evidence checks skipped)")
    check("screen namespace is EMPTY today (step 1 not yet run) -> registered screen finds nothing, "
          "which is the 16.11b failure mode",
          (len(real_ns) == 0) == (len(rcd.filter_prone_items(model)) == 0))
    cr_real = screen_catch_rates()
    if cr_real.get("models"):
        print("      measured catch rate of a two-arm screen on the sibling caches (pooled): "
              + "  ".join(f"k={k}: {v:.2f}" for k, v in cr_real["pooled"].items())
              + f"; mixed share of refused cells {cr_real['mixed_share_of_refused_cells']:.2f}")
        check("filter is stochastic per call on the sibling caches (some refused cells are mixed across "
              "identical-prompt samples) and catch is non-decreasing in k",
              cr_real["mixed_share_of_refused_cells"] > 0
              and cr_real["pooled"][1] <= cr_real["pooled"][2] <= cr_real["pooled"][3])
    else:
        print("      (no sibling single-agent caches on disk; catch-rate measurement skipped)")
    fac_real = chain_refusal_factors(model)
    if fac_real:
        print(f"      chain refusal factors (empty-output refusals per prone-item debate, {model}): "
              + "  ".join(f"{rd}={fac_real[rd]:.2f}" for rd in rcd.ROUNDS))
        w_rd, w_f = worst_round(fac_real)
        check("the worst chain round refuses about half of a prone item's debates and integration "
              "attains it (the factor the projection uses)",
              "integration" in (w_rd or "") and 0.3 <= w_f <= 0.7)

    # --- 6. offline end-to-end in a tempdir -------------------------------------
    real_gen_a, real_out_a = rca.generate_any, rca.OUT_DIR
    real_gen_d, real_out_d, real_fpi = rcd.generate_any, rcd.OUT_DIR, rcd.filter_prone_items
    stub = "stub-model"
    small = load_panel(6, 6)
    filt_item = small[0]
    filtered_user = rca.build_user_turn(AS_ASKER, filt_item.post_text)
    stub_before = sorted(p.name for p in OUT_DIR.iterdir()
                         if p.name.startswith(("cg_gen_stub-model_", "cgd_stub-model_narrative_cot", "cg_screen_stubmodel", "cg_deliberation_stubmodel")))
    try:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            rca.OUT_DIR = tdp; rcd.OUT_DIR = tdp
            rca.generate_any = _make_single_stub(filtered_user)  # type: ignore[assignment]
            rcd.generate_any = rcd._stub_generate               # type: ignore[assignment]
            stag, dtag = default_tags(stub)
            K = SCREEN_SAMPLES
            n_small = 2 * len(small) * K          # 72 at k=3
            n_full = 2 * 249 * K                  # 1,494 at k=3
            check(f"default screen is k={K} samples per (arm, item): {n_full} calls on the panel",
                  K == 3 and n_full == 1494)
            st0 = screen_status(stub, small)
            check(f"tempdir: screen not producible before step 1 ({n_small} missing, screen empty)",
                  st0["n_missing"] == n_small and not st0["complete"] and not st0["refused_items"])
            sc0 = screen_cost(stub, small, st0)
            check(f"step-1 cost model counts only MISSING calls ({n_small} new, 0 replayed) and flags an unpriced model",
                  sc0["new_calls"] == n_small and sc0["replayed_calls"] == 0 and not sc0["priced"])
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(["--step", "screen", "--models", stub, "--n-yta", "6", "--n-nta", "6"])
            check("main(--step screen) without --run prints the plan and REFUSES (exit 2)",
                  rc == 2 and "NOT RUNNING" in buf.getvalue() and "STEP-1 COST MODEL" in buf.getvalue())
            check("main(--step screen) without --run wrote nothing into the tempdir cache",
                  not any(p.name.startswith("cg_gen_") for p in tdp.iterdir()))
            check("main(--step screen) prints the k-sample projection block (no cache to measure in the tempdir)",
                  "WHAT A k-SAMPLE SCREEN WOULD CATCH" in buf.getvalue())
            # step 1 with the stub, via the same delegation the run path uses
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc1 = rca.main(screen_argv(stub, stag, workers=2, dry_run=False))
            st1 = screen_status(stub, small)
            n_ns = sum(1 for p in tdp.iterdir() if p.name.startswith("cg_gen_stub-model_narrative_cot_t2560_"))
            check(f"stub step 1: run_crowdgold_aita wrote the FULL {n_full}-record screen (2 arms x 249 panel "
                  f"items x k={K}) into the TEMP dir, never a sampled sub-panel",
                  n_ns == n_full and all(screen_glob_matches(stub, p) for p in tdp.iterdir()
                                         if p.name.startswith("cg_gen_stub-model_")))
            check(f"stub step 1: the {n_small} records this small panel needs are all present (the screen "
                  "covers the deliberation runner's prefix subset)",
                  st1["complete"] and st1["n_present"] == n_small)
            check(f"stub step 1: the aita runner's own guard does NOT trip on 1 refusal in {n_full} "
                  "(<5% NOVERDICT), rc 0; step 1 tolerates rc 4 too when the refusal share is higher",
                  rc1 == 0)
            check("stub step 1: the registered screen unions over arms AND samples (refused on as_asker "
                  "sample 01 only -> 1 item), and the runner's own glob sees it",
                  st1["refused_items"] == {filt_item.item_id} and list(st1["refused_by_arm"]) == [AS_ASKER]
                  and rcd.filter_prone_items(stub) == {filt_item.item_id})
            st1_k1 = screen_status(stub, small, 1)
            check("a k=1 screen on the same cache MISSES the item refused on sample 01 only (why k=3)",
                  st1_k1["complete"] and st1_k1["n_present"] == 2 * len(small) and not st1_k1["refused_items"])
            cr = screen_catch_rates(models=(stub,))
            check("screen_catch_rates measures catch by k on the stub cache: k=1 0.0, k=2 1.0, k=3 1.0",
                  cr.get("models") == [stub] and cr["pooled"] == {1: 0.0, 2: 1.0, 3: 1.0}
                  and cr["per_model"][stub]["prone_items"] == 1)
            pr1 = refusal_projection(k=1, catch=0.68, panel_n=249, prone_rate=11 / 40, worst_round_factor=0.5)
            pr3 = refusal_projection(k=3, catch=0.94, panel_n=249, prone_rate=11 / 40, worst_round_factor=0.5)
            check("refusal_projection: at 16.11b's prone rate a k=1 screen projects a GUARD FAIL "
                  "(~5.4% at integration) and k=3 a PASS (~1.1%)",
                  not pr1["projected_pass"] and abs(pr1["worst_round_noverdict_projected"] - 0.0541) < 0.002
                  and pr3["projected_pass"] and abs(pr3["worst_round_noverdict_projected"] - 0.0111) < 0.002)
            check("stub step 1: rows CSV (the same-model comparator for step 2) is in the tempdir",
                  (tdp / f"{stag}_rows.csv").exists())
            buf = io.StringIO()
            with redirect_stdout(buf):
                g_ok = step1_guard(st1)
            check("step-1 guard passes at 0% truncation", g_ok["pass"] and "GUARD FAILED" not in buf.getvalue())
            buf = io.StringIO()
            with redirect_stdout(buf):
                g_bad = step1_guard({"n_present": 20, "n_truncated": 2})
            check("step-1 guard fails at 10% truncation and prints GUARD FAILED",
                  not g_bad["pass"] and "GUARD FAILED" in buf.getvalue())
            # pre-flight for step 2 before any chain record
            fp0 = delib_footprint(stub, small)
            ok, why, code = preflight_deliberate(stub, small, st1, fp0, resume=False)
            check("pre-flight refuses an UNPRICED model first (exit 2)", not ok and code == 2)
            rcd.PRICES[stub] = (1.0, 1.0)
            try:
                ok, why, code = preflight_deliberate(stub, small, st0, fp0, resume=False)
                check("pre-flight refuses an INCOMPLETE screen (exit 2)", not ok and code == 2 and "incomplete" in why)
                ok, why, code = preflight_deliberate(stub, small, st1, fp0, resume=False)
                check("pre-flight passes with a complete screen and an empty deliberation namespace", ok)
                ok_m, why_m, code_m = preflight_deliberate(stub, small, st1, fp0, resume=False,
                                                           moderator_model="stub-mod")
                ok_mu, why_mu, code_mu = preflight_deliberate(stub, small, st1, fp0, resume=False,
                                                              moderator_model=stub)
                check("pre-flight refuses a cross-vendor --moderator-model (exit 2) and accepts the agent "
                      "model as its own moderator",
                      not ok_m and code_m == 2 and "modvendor" in why_m and ok_mu)
                (tdp / f"{dtag}_other_rows.csv").write_text("model,arm,item_id,sample_idx\nother-model,as_asker,x,0\n")
                ok_t, why_t, code_t = preflight_deliberate(stub, small, st1, fp0, resume=False,
                                                           tag=f"{dtag}_other")
                (tdp / f"{dtag}_other_rows.csv").unlink()
                check("pre-flight refuses a --tag whose rows CSV already holds another model's rows (exit 2)",
                      not ok_t and code_t == 2 and "other-model" in why_t)
                plan0 = new_call_plan(fp0, [i for i in small if i.item_id != filt_item.item_id])
                check("new-call plan before step 2: 11 survivors x 2 arms x 17 = 374 new, 0 replay",
                      plan0["new_calls"] == 374 and plan0["replay_calls"] == 0)
                # step 2 dry-run through main (the runner applies the screen itself). A
                # refused chain record is planted first so the plan JSON's per-cell
                # refusal map (tuple-keyed on disk) is exercised, as it is for gpt-4o.
                planted = rcd.call_cache_path(stub, SCREEN_SCAFFOLD, THIRD_PERSON, filt_item.item_id, 0,
                                              "r0", "writer_advocate", CAPS["agent"])
                planted.write_text(json.dumps({"item_id": filt_item.item_id, "arm": THIRD_PERSON,
                                               "round": "r0", "finish_reason": "content_filter",
                                               "output": ""}))
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = main(["--step", "deliberate", "--models", stub, "--n-yta", "6", "--n-nta", "6",
                               "--dry-run"])
                out = buf.getvalue()
                plan_json = json.loads((tdp / f"{dtag}_step2_plan.json").read_text())
                check("dry-run plan JSON serialises the per-cell refusal map of a planted chain refusal "
                      "and records the screen's k",
                      plan_json["refused_calls_by_cell"] == {f"{THIRD_PERSON}|{filt_item.item_id}": 1}
                      and plan_json["screen_samples"] == K and plan_json["footprint"]["n_present"] == 1)
                planted.unlink()
                check("main(--step deliberate --dry-run) exits 0, the runner drops the refused item "
                      "('content-filter screen: dropped 1 of 12 items'), and the new-call model prints",
                      rc == 0 and "content-filter screen: dropped 1 of 12 items" in out
                      and "NEW-CALL COST MODEL" in out and "374 NEW" in out)
                check("dry-run wrote the plan JSON and the runner's dryrun JSON into the TEMP dir only",
                      (tdp / f"{dtag}_step2_plan.json").exists() and (tdp / f"{dtag}_dryrun.json").exists()
                      and not any(p.name.startswith("cgd_stub-model") for p in tdp.iterdir()))
                # step 2 for real (stubbed, tempdir) through THIS module's --run path:
                # pre-flight, the delegated runner, then the completeness guard
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc2 = main(["--step", "deliberate", "--models", stub, "--n-yta", "6", "--n-nta", "6",
                                "--workers", "2", "--n-boot", "50", "--moderator-model", stub, "--run"])
                out = buf.getvalue()
                rows = list(csv.DictReader((tdp / f"{dtag}_rows.csv").open())) if (tdp / f"{dtag}_rows.csv").exists() else []
                check("stub step 2 (main --run, stubbed): exits 0 (both guards passed) with 22 rows = "
                      "11 survivors x 2 arms",
                      rc2 == 0 and len(rows) == 22)
                check("--moderator-model <the agent model> is normalised to the self-moderated default "
                      "(the delegated argv carries no --moderator-model)",
                      "--moderator-model" not in out.split("delegates to:", 1)[1].split("\n", 1)[0])
                comp_p = tdp / f"{dtag}_completeness.json"
                comp_j = json.loads(comp_p.read_text()) if comp_p.exists() else {}
                check("completeness guard: 22 expected, 22 rows, 0 missing, summary n_items agrees, PASS; "
                      "sidecar written",
                      comp_j.get("pass") is True and comp_j.get("n_expected") == 22 and comp_j.get("n_rows") == 22
                      and comp_j.get("n_missing") == 0 and comp_j.get("summary_n_items") == 11
                      and "COMPLETENESS GUARD" in out)
                exp22 = {(arm, i.item_id, 0) for arm in ARMS for i in small if i.item_id != filt_item.item_id}
                rows_bak = (tdp / f"{dtag}_rows.csv").read_text()
                lines = rows_bak.splitlines()
                (tdp / f"{dtag}_rows.csv").write_text("\n".join(lines[:-1]) + "\n")   # drop the last debate
                buf = io.StringIO()
                with redirect_stdout(buf):
                    comp_bad = completeness_check(dtag, stub, exp22, summary_n_items=11)
                    print_completeness(comp_bad)
                (tdp / f"{dtag}_rows.csv").write_text(rows_bak)
                check("completeness guard FAILS on a rows CSV missing one launched debate (1 missing, the "
                      "key named, per-arm counts 11/11 vs 10/11) and prints GUARD FAILED",
                      not comp_bad["pass"] and comp_bad["n_missing"] == 1 and comp_bad["n_rows"] == 21
                      and len(comp_bad["missing"]) == 1
                      and sorted(v["rows"] for v in comp_bad["per_arm"].values()) == [10, 11]
                      and "GUARD FAILED" in buf.getvalue())
                check("completeness guard FAILS when the runner's n_items disagrees with the survivor count",
                      not completeness_check(dtag, stub, exp22, summary_n_items=10)["pass"])
                check("stub step 2: the refused item is absent from every row",
                      rows and all(r["item_id"] != filt_item.item_id for r in rows))
                n_cgd = sum(1 for p in tdp.iterdir() if p.name.startswith("cgd_stub-model_narrative_cot_"))
                check("stub step 2: 22 x 17 = 374 chain records, one per call, in the TEMP dir", n_cgd == 374)
                fp1 = delib_footprint(stub, small)
                check("footprint after step 2: every chain record is an expected path, none foreign",
                      fp1["n_present"] == 374 and fp1["foreign"] == [])
                plan1 = new_call_plan(fp1, [i for i in small if i.item_id != filt_item.item_id])
                check("new-call plan after step 2: 0 new, 374 replay, 22 complete cells (a re-run is free)",
                      plan1["new_calls"] == 0 and plan1["replay_calls"] == 374 and plan1["replay_full_cells"] == 22)
                dc = delib_cost(stub, [i for i in small if i.item_id != filt_item.item_id], fp1)
                check("delib cost after step 2 is $0 and every round is priced from MEASURED chain records",
                      dc["usd_new_calls"] == 0.0 and set(dc["measured_rounds"]) == set(rcd.ROUNDS))
                # foreign files -> refusal without --resume, accepted with it
                (tdp / f"cgd_stub-model_narrative_cot_t4096_third_person_{small[1].item_id}_00_r0_writer_advocate.json").write_text("{}")
                (tdp / f"cgd_stub-model_narrative_cot_t2560_third_person_{small[1].item_id}_01_r0_writer_advocate.json").write_text("{}")
                fp2 = delib_footprint(stub, small)
                ok, why, code = preflight_deliberate(stub, small, st1, fp2, resume=False)
                ok_r, _, _ = preflight_deliberate(stub, small, st1, fp2, resume=True)
                check("another cap and a sample idx 01 are FOREIGN: refused (exit 3) without --resume, "
                      "accepted with it",
                      len(fp2["foreign"]) == 2 and not ok and code == 3 and ok_r)
                # chain-aware screen: plant a refused chain record for a clean item
                victim = small[2].item_id
                vp = rcd.call_cache_path(stub, SCREEN_SCAFFOLD, THIRD_PERSON, victim, 0, "r1", "counterparty", CAPS["agent"])
                vrec = json.loads(vp.read_text()); vrec["finish_reason"] = "content_filter"; vrec["output"] = ""
                vp.write_text(json.dumps(vrec))
                ch = chain_filtered_items(stub)
                check("chain-evidence screen finds the planted r1 refusal on an r0-clean item",
                      ch["items"] == {victim} and ch["by_round"] == {"r1": 1})
                install_chain_aware_screen(stub)
                check("chain-aware screen = registered | chain (2 items); registered alone still 1",
                      rcd.filter_prone_items(stub) == {filt_item.item_id, victim}
                      and real_fpi(stub) == {filt_item.item_id})
                rcd.filter_prone_items = real_fpi
                fp3 = delib_footprint(stub, small)
                plan3 = new_call_plan(fp3, [i for i in small if i.item_id != filt_item.item_id])
                check("new-call plan flags the cell that would replay a refusal under the registered screen",
                      plan3["cells_that_would_replay_a_refusal"] == 1)
                # readout on the stub run
                buf = io.StringIO()
                with redirect_stdout(buf):
                    row = grip_table_row(stub, dtag, screen_tag=stag, draws=50)
                check("readout: role-lock is exactly 1.0 on the stub (writer NTA / counterparty YTA planted)",
                      row is not None and row["role_lock"] is not None and abs(row["role_lock"] - 1.0) < 1e-9)
                check("readout: 22 debates, G1-G3 computed through analyze_stake_grip, fire rate present",
                      row["n_debates"] == 22 and row["grip"] is not None
                      and row["objection"]["fire_rate"] is not None)
                check(f"readout: transfer onto the model's OWN single-agent verdict (majority of the k={K} "
                      "screen samples) is computed from step-1 rows",
                      row["transfer_own_solo"] is not None and row["transfer_own_solo"]["n_cells"] == 22
                      and row["transfer_own_solo"]["k"] == K)
                check("readout: lift is unpowered below 60 fired and reported as such",
                      not row["lift"]["powered"])
                check("readout: S2 accuracy is on the codable population with n_codable reported (22 of 22 here)",
                      row["s2_n_codable"] == 22 and row["s2_accuracy"] is not None
                      and row["guards"]["p_noncodable"] == 0)
                # plant one non-codable group verdict under a sibling tag: S2 must be read on the
                # other 21, not scored as wrong out of 22
                rws = list(csv.DictReader((tdp / f"{dtag}_rows.csv").open()))
                rws[0]["verdict"] = rcd.NOVERDICT
                with (tdp / f"{dtag}_nc_rows.csv").open("w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rws[0])); w.writeheader(); w.writerows(rws)
                (tdp / f"{dtag}_nc_votes.csv").write_text((tdp / f"{dtag}_votes.csv").read_text())
                from scripts.analyze_unembodied_ablation import verdict_correct as _vc
                exp_acc = [int(_vc(r["verdict"], r["gold_verdict"])) for r in rws[1:]]
                buf = io.StringIO()
                with redirect_stdout(buf):
                    row_nc = grip_table_row(stub, f"{dtag}_nc", screen_tag=stag, draws=20)
                check("readout: with one NOVERDICT row S2 is the accuracy on the 21 codable rows (n_codable 21), "
                      "not (correct / 22)",
                      row_nc is not None and row_nc["s2_n_codable"] == 21
                      and abs(row_nc["s2_accuracy"] - sum(exp_acc) / 21) < 1e-12
                      and abs(row_nc["guards"]["p_noncodable"] - 1 / 22) < 1e-12)
                buf = io.StringIO()
                with redirect_stdout(buf):
                    print_grip_table_row(row, label="selftest")
                check("readout prints the ROLE-LOCK / G1 / G2 / G3 / TRANSFER lines",
                      all(k in buf.getvalue() for k in ("ROLE-LOCK", "G1 fire", "G2 reject", "G3 stake", "TRANSFER")))
                # step 3 through main on the stub run (guard passed -> merges into the
                # tempdir's registry), then on a summary whose guard failed (-> UNREAD file)
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc3 = main(["--step", "readout", "--models", stub, "--draws", "50"])
                check("main(--step readout) on a run whose guard PASSED exits 0, merges into "
                      "stake_grip_analysis.json under model:tag, writes the grip row without an UNREAD flag",
                      rc3 == 0 and f"{stub}:{dtag}" in json.loads((tdp / "stake_grip_analysis.json").read_text())
                      and "UNREAD_GUARD_FAILED" not in json.loads((tdp / f"{dtag}_grip_row.json").read_text()))
                summ = tdp / f"{dtag}_summary.json"
                sj = json.loads(summ.read_text()); sj["round_guard"]["pass"] = False
                summ.write_text(json.dumps(sj))
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc3b = main(["--step", "readout", "--models", stub, "--draws", "50"])
                check("main(--step readout) on a run whose guard FAILED prints GUARD FAILED, EXITS 4, writes "
                      f"{dtag}_stake_grip_UNREAD.json instead of touching the registry, and flags the row",
                      rc3b == 4 and "GUARD FAILED" in buf.getvalue()
                      and (tdp / f"{dtag}_stake_grip_UNREAD.json").exists()
                      and json.loads((tdp / f"{dtag}_grip_row.json").read_text()).get("UNREAD_GUARD_FAILED") is True)
                sj["round_guard"]["pass"] = True
                summ.write_text(json.dumps(sj))
                comp_bak = comp_p.read_text()
                cj = json.loads(comp_bak); cj["pass"] = False; cj["n_missing"] = 3
                comp_p.write_text(json.dumps(cj))
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc3c = main(["--step", "readout", "--models", stub, "--draws", "50"])
                comp_p.write_text(comp_bak)
                check("main(--step readout) exits 4 and names the completeness guard when the sidecar "
                      "reports missing debates",
                      rc3c == 4 and "completeness" in buf.getvalue() and "GUARD FAILED" in buf.getvalue())
                summ_bak = summ.read_text()
                summ.unlink()
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc3d = main(["--step", "readout", "--models", stub, "--draws", "50"])
                summ.write_text(summ_bak)
                check("main(--step readout) treats a MISSING summary (no guard verdict) as a failed guard: exit 4",
                      rc3d == 4 and "GUARD FAILED" in buf.getvalue())
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc3e = main(["--step", "readout", "--models", stub, "--draws", "50"])
                check("main(--step readout) exits 0 again once the guard records are intact", rc3e == 0)
                # ---- tags: refused before anything is written ----
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc_t1 = main(["--step", "deliberate", "--models", stub, "--n-yta", "6", "--n-nta", "6",
                                  "--tag", "cg_deliberation", "--dry-run"])
                    rc_t2 = main(["--step", "screen", "--models", stub, "--n-yta", "6", "--n-nta", "6",
                                  "--screen-tag", "cg_gen_x", "--dry-run"])
                    rc_t3 = main(["--step", "deliberate", "--models", stub, "--n-yta", "6", "--n-nta", "6",
                                  "--tag", f"{dtag}_v2", "--dry-run"])
                check("main() refuses --tag cg_deliberation and a foreign --screen-tag (exit 2, nothing "
                      "written) and accepts a tag under this model's prefix",
                      rc_t1 == 2 and rc_t2 == 2 and rc_t3 == 0
                      and not (tdp / "cg_deliberation_dryrun.json").exists()
                      and not (tdp / "cg_deliberation_step2_plan.json").exists()
                      and (tdp / f"{dtag}_v2_step2_plan.json").exists())
                # ---- the --moderator-model landmine: constructed in the tempdir, refused by main ----
                snap = {p.name: p.read_bytes() for p in tdp.iterdir() if p.name.startswith("cgd_stub-model_")}
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc_m1 = main(["--step", "deliberate", "--models", stub, "--n-yta", "6", "--n-nta", "6",
                                  "--moderator-model", "stub-mod", "--dry-run"])
                    rc_m2 = main(["--step", "deliberate", "--models", stub, "--n-yta", "6", "--n-nta", "6",
                                  "--moderator-model", "stub-mod", "--run"])
                    rc_m3 = main(["--step", "readout", "--models", stub, "--moderator-model", "stub-mod"])
                snap_after = {p.name: p.read_bytes() for p in tdp.iterdir() if p.name.startswith("cgd_stub-model_")}
                check("main() refuses a cross-vendor --moderator-model on every step (exit 2), before any "
                      "record is read or written, and names run_crowdgold_modvendor",
                      rc_m1 == 2 and rc_m2 == 2 and rc_m3 == 2 and snap_after == snap
                      and "run_crowdgold_modvendor" in buf.getvalue()
                      and "NEW-CALL COST MODEL" not in buf.getvalue())
                # the hazard itself, through the bare delegation main() no longer allows: a
                # model-sensitive stub so the other moderator's synthesis differs in TEXT
                def _xmod_stub(model, system, user, *, sample_idx=0, max_tokens=512, **kw):
                    res = rcd._stub_generate(model, system, user, sample_idx=sample_idx,
                                             max_tokens=max_tokens, **kw)
                    if "Write one synthesis position" in user:
                        return rcd._StubResult(res.text.replace("shared_account_of_the_conflict",
                                                                f"shared_account [{model}]"),
                                               completion_tokens=res.completion_tokens)
                    return res
                rcd.generate_any = _xmod_stub  # type: ignore[assignment]
                rcd.PRICES["stub-mod"] = (1.0, 1.0)
                try:
                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        rcd.main(deliberate_argv(stub, f"{dtag}_xmod", n_yta=6, n_nta=6, workers=2, n_boot=20,
                                                 compare_rows=tdp / f"{stag}_rows.csv",
                                                 moderator_model="stub-mod", dry_run=False))
                finally:
                    rcd.PRICES.pop("stub-mod", None)
                    rcd.generate_any = rcd._stub_generate  # type: ignore[assignment]
                snap_x = {p.name: p.read_bytes() for p in tdp.iterdir() if p.name.startswith("cgd_stub-model_")}
                clobbered = sorted(n for n, b in snap.items() if snap_x.get(n) != b)
                clob_r34 = [n for n in clobbered if "_r3_label_" in n or "_r4_vote_" in n]
                # the r1 record planted as a refusal above (victim item) makes that debate's
                # r2 stale too; those regenerations are the plant's, not the collision's
                clob_other = [n for n in clobbered if n not in clob_r34]
                check("the base runner WITH a cross-vendor moderator overwrites the agent model's cached "
                      "r3_label / r4_vote records at their registered paths (the collision main() refuses): "
                      f"{len(clob_r34)} of 132 clobbered with a new parent_sha",
                      len(clob_r34) == 22 * 6
                      and all(json.loads(snap_x[n])["parent_sha"] != json.loads(snap[n])["parent_sha"]
                              for n in clob_r34)
                      and all("_r2_" in n and victim in n for n in clob_other))
                check("the same bare delegation lands the cross-vendor synthesis on the moderator's OWN "
                      "self-moderated path (cgd_stub-mod_..._mod-stub-mod.json)",
                      any(p.name.startswith("cgd_stub-mod_") and p.name.endswith("_synthesis_mod-stub-mod.json")
                          for p in tdp.iterdir()))
                for n, b in snap.items():          # restore the tempdir for the checks below
                    (tdp / n).write_bytes(b)
            finally:
                rcd.PRICES.pop(stub, None)
    finally:
        rca.generate_any, rca.OUT_DIR = real_gen_a, real_out_a
        rcd.generate_any, rcd.OUT_DIR, rcd.filter_prone_items = real_gen_d, real_out_d, real_fpi
    stub_after = sorted(p.name for p in OUT_DIR.iterdir()
                        if p.name.startswith(("cg_gen_stub-model_", "cgd_stub-model_narrative_cot", "cg_screen_stubmodel", "cg_deliberation_stubmodel")))
    check("nothing landed in the real cache directory (no stub-model file in this module's namespaces)",
          stub_after == stub_before == [])
    check("generators and cache dirs restored", rca.generate_any is real_gen_a and rcd.generate_any is real_gen_d
          and rca.OUT_DIR == real_out_a and rcd.OUT_DIR == real_out_d)

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Addendum 16.20: content-filter screen (step 1) and screened "
                    "deliberation (step 2) for a model behind a platform filter; "
                    "readout (step 3). Generates NOTHING unless --run is passed.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--step", choices=("screen", "deliberate", "readout"), default=None)
    ap.add_argument("--models", default=DEFAULT_MODELS,
                    help="comma-separated deployment names; each needs a PRICES entry")
    ap.add_argument("--dry-run", action="store_true", help="plan + cost model, exit 0")
    ap.add_argument("--run", action="store_true", help="REQUIRED to generate; spends money")
    ap.add_argument("--resume", action="store_true",
                    help="step 1: fill a non-empty screen namespace; step 2: proceed despite "
                         "foreign files in the deliberation namespace")
    ap.add_argument("--screen-samples", type=int, default=SCREEN_SAMPLES,
                    help=f"samples per (arm, item) in the step-1 screen (default {SCREEN_SAMPLES}); "
                         "the filter is stochastic per call, so k=1 sees ~2/3 of the prone items and "
                         "k=3 ~19/20 (measured; the dry-run prints the projection). Step 2 requires "
                         "the screen complete at this k.")
    ap.add_argument("--screen", choices=("registered", "chain-aware"), default="registered",
                    help="registered = the runner's own screen (single-agent refusals); "
                         "chain-aware = union with refusals in this model's cached chain records")
    ap.add_argument("--tag", default=None,
                    help="deliberation tag (single model only); must start with "
                         "cg_deliberation_<short_name(model)> and may not be the 16.11b tag")
    ap.add_argument("--screen-tag", default=None,
                    help="step-1 tag (single model only); must start with cg_screen_<short_name(model)>")
    ap.add_argument("--moderator-model", default=None,
                    help="accepted ONLY as the agent model itself (the self-moderated default); any "
                         "other model is refused -- see run_crowdgold_modvendor for a namespaced "
                         "cross-vendor moderator")
    ap.add_argument("--n-yta", type=int, default=N_YTA, help="testing only; the panel is the default")
    ap.add_argument("--n-nta", type=int, default=N_NTA, help="testing only; the panel is the default")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=2000, help="readout bootstrap draws")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if not args.step:
        print("ERROR: --step screen|deliberate|readout is required (or --selftest)")
        return 2
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("ERROR: --models is empty"); return 2
    if args.screen_samples < 1:
        print("ERROR: --screen-samples must be >= 1"); return 2
    if (args.tag or args.screen_tag) and len(models) != 1:
        print("ERROR: --tag / --screen-tag apply to a single model"); return 2
    for m in models:
        why = preflight_model(m)
        if why:
            print(f"\nERROR: {why}\n"); return 2
        why = moderator_guard(m, args.moderator_model)
        if why:
            print(f"\nERROR: {why}\n"); return 2
        stag_d, dtag_d = default_tags(m)
        why = tag_guard(m, args.tag or dtag_d, args.screen_tag or stag_d)
        if why:
            print(f"\nERROR: {why}\n"); return 2
    if args.moderator_model in models:
        args.moderator_model = None      # the self-moderated default, stated explicitly
    try:
        items = load_panel(args.n_yta, args.n_nta)
    except Exception as e:  # noqa: BLE001
        print(f"\nERROR: panel unavailable ({e})\n"); return 2
    print(f"[filter-screen] panel {len(items)} items "
          f"({sum(1 for i in items if i.gold_verdict == 'YTA')} YTA + "
          f"{sum(1 for i in items if i.gold_verdict == 'NTA')} NTA)   models {models}   step {args.step}")

    worst = 0
    for m in models:
        stag_d, dtag_d = default_tags(m)
        stag, dtag = args.screen_tag or stag_d, args.tag or dtag_d
        if args.step == "screen":
            rc = run_screen(m, stag, items, workers=args.workers, samples=args.screen_samples,
                            run=args.run, dry_run=args.dry_run, resume=args.resume)
        elif args.step == "deliberate":
            rc = run_deliberate(m, dtag, stag, items, n_yta=args.n_yta, n_nta=args.n_nta,
                                workers=args.workers, n_boot=args.n_boot, screen=args.screen,
                                samples=args.screen_samples,
                                moderator_model=args.moderator_model, run=args.run,
                                dry_run=args.dry_run, resume=args.resume)
        else:
            rc = run_readout(m, dtag, stag, draws=args.draws)
        worst = max(worst, rc)
        if rc not in (0,) and len(models) > 1:
            print(f"  [{m}] exited {rc}; continuing with the next model")
    return worst


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
