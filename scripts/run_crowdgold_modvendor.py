"""
scripts/run_crowdgold_modvendor.py -- Addendum 16.17: the CROSS-VENDOR
MODERATOR cell on the registered crowd-gold AITA deliberation.

    advocates + adjudicator   grok-4-1-fast-reasoning   (the registered seats,
                                                        briefs and stakes VERBATIM)
    moderator                 claude-sonnet-4-6         cell 1, tag cg_deliberation_modsonnet
                              claude-haiku-4-5          cell 2, tag cg_deliberation_modhaiku
                              <fourth vendor>           config slot, see MODERATOR_CELLS

Same 210-item screened panel (--n-yta 99 --n-nta 150, seed 44), same two
arms, --samples 1, same seven-round protocol, 17 calls per debate. Rows and
votes CSVs carry the base schema and the embodied seat ids, so the cell is
read by the same analyze_topology_2x2-style code as the 16.10 table.

WHICH CALLS REPLAY AND WHICH ARE NEW (per debate, 17 calls)
-----------------------------------------------------------
  r0, r1, r2   3 seats x 3 rounds = 9   REPLAY from the registered cg_deliberation
                                        cache. Nothing upstream of the moderator
                                        is touched: same model, same scaffold,
                                        same seat ids, same arm, same item, same
                                        token cap, byte-identical prompts, and
                                        the same parent_sha chain (r0 <- post
                                        text, r1 <- the three r0s, r2 <- r0s+r1s).
                                        The selftest verifies this against the
                                        REAL cache records, read-only. The cell
                                        is DEFINED at the registered transcript
                                        cap 0: --transcript-cap is accepted only
                                        as 0 (see BANNED FLAGS below), because a
                                        non-zero cap changes the r1/r2 prompt
                                        text without changing call_cache_path or
                                        the r0 parent_sha, so the registered
                                        uncapped records would be replayed for a
                                        prompt they were not generated under
                                        (measured: r1 prompt 12,442 chars at cap
                                        0 vs 5,218 at cap 400, same parent_sha).
  synthesis    1   NEW   moderator model, cache id modvendor-<agent model>
  r3_label     3   NEW   grok, cache id <seat>_xmod-<moderator model>
  integration  1   NEW   moderator model, cache id modvendor-<agent model>
  r4_vote      3   NEW   grok, cache id <seat>_xmod-<moderator model>
                                                             ------
                          8 NEW generation calls per debate  (2 moderator + 6 grok)

The dry-run probes the real cache for every debate and counts ONLY the new
calls; a debate whose r0-r2 chain is missing or parent_sha-broken is reported
separately and the run path refuses it unless --allow-fresh-r0r2 is passed.
Prompt tokens for the four new rounds are the API-billed prompt_tokens of the
registered run's own records for the same debates (the synthesis prompt is
byte-identical; r3/integration/r4 differ only by the other vendor's text), so
the estimate does not depend on a local tokenizer. Moderator completion
lengths are rcd.ASSUMED_COMPLETION, flagged: no other-vendor measurement yet.

DRY-RUN RESULT ON THIS MACHINE, 2026-09-13 (both cells, 420/420 chains full)
  claude-sonnet-4-6  3,780 replay / 3,360 NEW   est. $10.69 (moderator $9.89)
  claude-haiku-4-5   3,780 replay / 3,360 NEW   est.  $4.10 (moderator $3.30)
  Counted on this panel: the bare base runner with --moderator-model would
  have OVERWRITTEN the registered cell's 2,520 grok r3/r4 records (both
  cells), plus 56 mod-claude-sonnet-4-6 records (the sonnet self-pilot) for
  cell 1 and 318 mod-claude-haiku-4-5 records (the haiku self-moderated
  runs) for cell 2. Zero of this wrapper's own paths existed.

WHY THE BASE RUNNER ALONE IS NOT SAFE (verified by reading the code and by
the selftest, which constructs both collisions in a tempdir)
--------------------------------------------------------------------------
run_crowdgold_deliberation.call_cache_path keys on (model, scaffold, cap, arm,
item, idx, round, role id) and on NOTHING that names the moderator for the
agent rounds. Passing --moderator-model claude-sonnet-4-6 to the base runner
therefore:

  (b) lands synthesis/integration at
      cgd_claude-sonnet-4-6_..._synthesis_mod-claude-sonnet-4-6.json.
      That is disjoint from the grok moderator's mod-grok-... records, as the
      task assumed -- but it is the SAME PATH the sonnet SELF-moderated pilot
      (cg_deliberation_sonnet_screen, 16 items x 2 arms, 14 of the 16 items
      in the grok panel) already occupies. parent_sha differs (grok r2 texts
      vs sonnet r2 texts), so do_call would not replay it -- it would
      regenerate and OVERWRITE the pilot's record.
  (c) lands r3_label / r4_vote at the registered grok run's OWN paths
      (cgd_grok-..._r3_label_writer_advocate.json). parent_sha differs
      because the synthesis text differs, so the record is not replayed --
      but do_call writes the fresh record to the same path and CLOBBERS the
      registered cell's r3/r4 records. A later --resume of cg_deliberation
      would then find its own r3 stale and silently re-spend and re-sample
      the registered cell. The parent_sha check is a replay guard, not a
      write guard.

WHAT "MODERATOR VENDOR" MEANS HERE (disclosed; read the cell accordingly)
--------------------------------------------------------------------------
The swap changes the moderator MODEL and, with it, the request-side sampling
parameters, which are each vendor's DEFAULTS rather than a held constant:
scripts/generators.py's Anthropic path sends {model, max_tokens, messages,
system} with no temperature and no seed; the xAI reasoning path sends
max_completion_tokens with no temperature and no seed; the registered grok
moderator ran under the same xAI defaults. So the cell reads "moderator
vendor at its own default decoding", not "moderator model at matched
sampling". That is inherent to a cross-vendor cell (the vendors do not
expose a common sampler) and is stated here and in the addendum rather than
controlled.

So this wrapper installs, without touching the seats or any prompt text:

  moderator cache id   mod-<mod>            ->  modvendor-<agent model>
                       (path model segment is already the moderator model,
                        so this names the (moderator, agents) pair uniquely
                        and cannot meet a self-moderated run of either model)
  r3/r4 cache role id  <seat>               ->  <seat>_xmod-<moderator model>
                       (r0/r1/r2 keep the bare seat id, which is what makes
                        them replay)

Both namespaces are verified EMPTY of foreign files by the selftest and
cache_is_clean() is enforced on the --run path, --resume being the only
override. Record contents are unchanged: the record's role_id field is still
the seat id and the moderator is named by the path and by the rows CSV's
moderator_model column.

GUARDS AFTER THE DELEGATED RUN (this wrapper's, on top of the base runner's)
---------------------------------------------------------------------------
  * COMPLETENESS. rcd.run swallows an exception raised inside
    run_deliberation with one console ERROR line and omits that debate from
    rows / votes / calls; the base round and outcome guards and every
    readout denominator are then computed on the survivors (arm-correlated
    attrition that nothing reports). After rcd.main returns, the wrapper
    compares the rows CSV with the (arm, item, sample) keys it launched (it
    built the panel itself), writes <tag>_completeness.json and prints the
    literal GUARD FAILED and exits 4 on any shortfall.
  * QUARANTINE. On exit 4 (round / outcome / completeness guard) the CSVs
    are renamed <tag>_GUARD_FAILED_rows/_votes.csv.
  * --readout REFUSES (GUARD FAILED, exit 4, no analysis JSON) a cell whose
    <tag>_summary.json or <tag>_completeness.json is missing or failed, or
    whose codable share of group verdicts is below MIN_CODABLE_SHARE (0.95):
    call_guard_row passes a synthesis / integration of "VERDICT: UNRESOLVED"
    as parsed, so a moderator that emits UNRESOLVED / NOVERDICT is invisible
    to the base guards and visible only here.

THE READOUT IS PAIRED
---------------------
Because r0-r2 replay the registered cell's sample-0 records, on sample 0
the moderator is the ONLY difference between this cell and embodied/on.
--readout therefore prints, first, the PAIRED comparison against embodied
sample_idx 0 on the matched (arm, item) debates: modvendor S2 minus
embodied-sample-0 S2, fire-rate delta and error-lift delta, each with an
item-clustered bootstrap CI (analyze_coalition.s2_paired's convention).
The pooled 16.10-style table (embodied/on over all four samples, as
analyze_topology_2x2 prints it) follows as the secondary view, now with
CIs on lift, G3 and transfer and with n_codable printed.

CONFIG, NOT CODE
----------------
--models (exactly one agent model per invocation) and --moderator-model are
free strings priced through run_crowdgold_deliberation.PRICES; a deployment
not yet on file can be priced for the dry-run with --price NAME=IN,OUT (USD
per Mtok) and --run refuses an unpriced model. MODERATOR_CELLS holds the
default tag per moderator; an unlisted moderator gets cg_deliberation_mod<name>.
Add the fourth vendor as a PRICES entry (one config line) or via --price.

SPEND SAFETY
------------
Nothing generates unless the literal flag --run is given; --dry-run (or
omitting --run) prints the new-calls cost model and exits. --selftest stubs
generation and redirects the cache directory the base runner reads
(run_crowdgold_deliberation.OUT_DIR) to a tempdir, then re-checks the real
directory. The selftest never passes --run.

Usage
-----
  python -m scripts.run_crowdgold_modvendor --selftest
  python -m scripts.run_crowdgold_modvendor --moderator-model claude-sonnet-4-6 --verify-cache-clean
  python -m scripts.run_crowdgold_modvendor --moderator-model claude-sonnet-4-6 --dry-run
  python -m scripts.run_crowdgold_modvendor --run --models grok-4-1-fast-reasoning \\
      --moderator-model claude-sonnet-4-6 --n-yta 99 --n-nta 150 --samples 1 --seed 44 \\
      --tag cg_deliberation_modsonnet                                   # SPENDS MONEY
  python -m scripts.run_crowdgold_modvendor --run --moderator-model claude-haiku-4-5 \\
      --tag cg_deliberation_modhaiku                                    # SPENDS MONEY
  python -m scripts.run_crowdgold_modvendor --moderator-model claude-sonnet-4-6 --readout

Exit codes: 0 ok, 1 no rows, 2 bad config / --run not given / unpriced model
/ tag collision / r0-r2 chain missing, 3 cache not clean (pass --resume),
4 guard failed (round / outcome / completeness on --run; summary, completeness
or codable-share guard on --readout), a banned flag, or --transcript-cap != 0.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import io
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional, Sequence

import scripts.run_crowdgold_deliberation as rcd
from scripts.run_phase1_quartet import OUT_DIR, _safe

DEFAULT_AGENT_MODEL = "grok-4-1-fast-reasoning"
DEFAULT_SCAFFOLD = "narrative_cot"
MODERATOR_PREFIX = "modvendor"     # moderator cache role id: modvendor-<agent model>
SEAT_SUFFIX = "xmod"               # r3/r4 cache role id: <seat>_xmod-<moderator model>
REPLAY_ROUNDS = ("r0", "r1", "r2")
POST_SYNTHESIS_ROUNDS = ("r3_label", "r4_vote")
NEW_ROUNDS = ("synthesis", "r3_label", "integration", "r4_vote")
NEW_CALLS_PER_DEBATE = 1 + rcd.N_AGENTS + 1 + rcd.N_AGENTS   # 8
TAG_PREFIX = "cg_deliberation_mod"
#: The registered run's --max-tokens-moderator; the cost model reads its
#: mod-<agent> records for measured prompt tokens at this cap.
REGISTERED_MODERATOR_CAP = 1024

#: Config slots, keyed by moderator deployment name. Anything not listed still
#: runs (tag TAG_PREFIX + _safe(model)); the price must be in rcd.PRICES or
#: supplied with --price. Add the fourth vendor here when its name is known:
#:   "<deployment-name>": {"tag": "cg_deliberation_mod<short>", "note": "..."}
MODERATOR_CELLS: dict[str, dict] = {
    "claude-sonnet-4-6": {"tag": "cg_deliberation_modsonnet",
                          "note": "Addendum 16.17 cell 1 (registered)"},
    "claude-haiku-4-5": {"tag": "cg_deliberation_modhaiku",
                         "note": "Addendum 16.17 cell 2"},
}

#: Tags this wrapper must never write, whoever asks.
FOREIGN_TAGS = {"cg_deliberation", "cg_deliberation_unembodied",
                "cg_deliberation_noedge", "cg_deliberation_identical",
                "cg_deliberation_identical_noedge"}

#: Base-runner flags that would change the seats, the scaffold or the r3/r4
#: prompt text. The cell is defined on the registered transcripts; refuse.
#: --no-unresolved is banned for a sharper reason: it changes the r0/r1/r2
#: prompt text but NOT call_cache_path or the r0 parent_sha (post text only),
#: so the registered allow-unresolved records would be silently served back
#: for a prompt they were not generated under. --arms is banned because the
#: replay probe and cost model are built on PILOT_ARMS; a forwarded --arms
#: would price one panel and run another.
BANNED_FLAGS = ("--stake-nudge", "--stake-cot", "--stake-fewshot",
                "--stake-fewshot-set", "--r3r4-reasoning-effort",
                "--r3r4-thinking-budget", "--agent-scaffold", "--pilot",
                "--full", "--smoke", "--selftest", "--no-unresolved", "--arms")
#: The registered transcript cap; the only value this wrapper forwards. A
#: non-zero cap is refused for the --no-unresolved reason (prompt text the
#: cache key and the r0 parent_sha cannot see).
REGISTERED_TRANSCRIPT_CAP = 0
#: --readout refuses a cell whose codable share of group verdicts is below this.
MIN_CODABLE_SHARE = 0.95

#: Every sibling runner's moderator prefix, so the selftest can prove
#: MODERATOR_PREFIX is not one of them and not a prefix/suffix of one
#: (namespace_counts matches "_<prefix>-" as a substring).
SIBLING_MODERATOR_PREFIXES = ("mod", "modunembodied", "modidentical", "modnoedge",
                              "modidentnoedge", "modgame", "modgamenb",
                              "modcoalition", "modmultivendor")

#: Captured ONCE at import, before install() patches anything.
_PRISTINE = {
    "ROLES": rcd.ROLES, "ROLE_ORDER": rcd.ROLE_ORDER, "ROLE_BY_ID": rcd.ROLE_BY_ID,
    "R0_PREAMBLE": rcd.R0_PREAMBLE, "SYNTHESIS_SYSTEM": rcd.SYNTHESIS_SYSTEM,
    "INTEGRATION_SYSTEM": rcd.INTEGRATION_SYSTEM,
    "r1_user": rcd.r1_user, "r2_user": rcd.r2_user,
    "call_cache_path": rcd.call_cache_path,
    "moderator_role_id": rcd.moderator_role_id,
}
_INSTALLED: dict = {}


# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------

def moderator_cache_id(agent_model: str) -> str:
    """Role id of the two moderator rounds. The path's model segment is the
    moderator model, so (moderator, agents) is named uniquely."""
    return f"{MODERATOR_PREFIX}-{_safe(agent_model)}"


def seat_cache_id(role_id: str, mod_model: str) -> str:
    """Role id of the r3/r4 records ONLY. r0-r2 keep the bare seat id."""
    return f"{role_id}_{SEAT_SUFFIX}-{_safe(mod_model)}"


def default_tag(mod_model: str) -> str:
    return MODERATOR_CELLS.get(mod_model, {}).get("tag", f"{TAG_PREFIX}{_safe(mod_model)}")


def pristine_path(model, scaffold, arm, item_id, idx, round_name, role_id,
                  max_tokens, *, out_dir: Path) -> Path:
    """The base runner's cache path, rooted at an explicit directory."""
    return out_dir / _PRISTINE["call_cache_path"](
        model, scaffold, arm, item_id, idx, round_name, role_id, max_tokens).name


def install(agent_model: str, mod_model: str) -> None:
    """Patch the two cache-key functions do_call resolves by name. The seats,
    the preamble, both moderator system prompts and every prompt builder are
    left untouched (asserted by the selftest), which is what makes r0-r2
    replay. Idempotent: closes over the pristine functions, never over the
    currently installed ones."""
    if mod_model == agent_model:
        raise SystemExit(f"moderator {mod_model!r} == agent model: that is the "
                         "self-moderated registered run (cg_deliberation); refusing")
    base = _PRISTINE["call_cache_path"]

    def _path(model, scaffold, arm, item_id, idx, round_name, role_id, max_tokens):
        if round_name in POST_SYNTHESIS_ROUNDS:
            role_id = seat_cache_id(role_id, mod_model)
        return base(model, scaffold, arm, item_id, idx, round_name, role_id, max_tokens)

    def _mod_id(m: str) -> str:
        if m != mod_model:
            raise ValueError(f"moderator_role_id({m!r}) but {mod_model!r} is installed")
        return moderator_cache_id(agent_model)

    rcd.call_cache_path = _path
    rcd.moderator_role_id = _mod_id
    _INSTALLED.clear()
    _INSTALLED.update(agent=agent_model, moderator=mod_model)


def uninstall() -> None:
    rcd.call_cache_path = _PRISTINE["call_cache_path"]
    rcd.moderator_role_id = _PRISTINE["moderator_role_id"]
    _INSTALLED.clear()


def namespace_counts(agent_model: str, mod_model: str, *,
                     out_dir: Path = OUT_DIR) -> tuple[dict[str, int], list[str]]:
    """One scandir pass. Counts every file in the namespaces this cell writes
    to, and lists FOREIGN files (stub / selftest names) found in them."""
    mid = moderator_cache_id(agent_model)
    mod_pre = f"cgd_{_safe(mod_model)}_"
    mod_suf = f"_{mid}.json"
    ag_pre = f"cgd_{_safe(agent_model)}_"
    seat_sufs = {f"{rd}:{r.role_id}": f"_{rd}_{seat_cache_id(r.role_id, mod_model)}.json"
                 for rd in POST_SYNTHESIS_ROUNDS for r in _PRISTINE["ROLES"]}
    counts = {f"moderator:{mod_pre}*{mod_suf}": 0}
    counts.update({f"{k}:{ag_pre}*{v}": 0 for k, v in seat_sufs.items()})
    foreign: list[str] = []
    if not out_dir.is_dir():
        return counts, foreign
    with os.scandir(out_dir) as it:
        for e in it:
            n = e.name
            if not n.startswith("cgd_"):
                continue
            hit = False
            if n.startswith(mod_pre) and n.endswith(mod_suf):
                counts[f"moderator:{mod_pre}*{mod_suf}"] += 1
                hit = True
            elif n.startswith(ag_pre):
                for k, suf in seat_sufs.items():
                    if n.endswith(suf):
                        counts[f"{k}:{ag_pre}*{suf}"] += 1
                        hit = True
                        break
            if hit and ("stub" in n or "selftest" in n):
                foreign.append(n)
    return counts, foreign


def cache_is_clean(agent_model: str, mod_model: str, *,
                   out_dir: Path = OUT_DIR) -> tuple[bool, dict[str, int]]:
    counts, _ = namespace_counts(agent_model, mod_model, out_dir=out_dir)
    return sum(counts.values()) == 0, counts


def any_modvendor_foreign(out_dir: Path = OUT_DIR) -> list[str]:
    """Foreign files anywhere in this wrapper's TWO namespace families,
    whatever the model names: the selftest's no-foreign-file check."""
    out: list[str] = []
    if not out_dir.is_dir():
        return out
    with os.scandir(out_dir) as it:
        for e in it:
            n = e.name
            if not n.startswith("cgd_"):
                continue
            if (f"_{MODERATOR_PREFIX}-" in n or f"_{SEAT_SUFFIX}-" in n) and (
                    "stub" in n or "selftest" in n):
                out.append(n)
    return out


# ---------------------------------------------------------------------------
# Replay probe: which of r0/r1/r2 would be served from the real cache
# ---------------------------------------------------------------------------

def _read_rec(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def probe_chain(agent_model: str, scaffold: str, arm: str, item, idx: int, *,
                max_tokens_agent: int, out_dir: Path) -> dict:
    """Walk r0 -> r1 -> r2 for one debate exactly as do_call would: a round
    replays iff all three seat records exist AND carry the parent_sha of the
    texts do_call will hand them. Returns the per-round status, the r2 texts
    (for exact synthesis-prompt measurement) and the replay call count."""
    status: dict[str, str] = {}
    texts: dict[str, dict[str, str]] = {}
    parents: tuple = (item.post_text,)
    for i, rd in enumerate(REPLAY_ROUNDS):
        outs: dict[str, str] = {}
        verdict = "replay"
        for rid in _PRISTINE["ROLE_ORDER"]:
            rec = _read_rec(pristine_path(agent_model, scaffold, arm, item.item_id,
                                          idx, rd, rid, max_tokens_agent,
                                          out_dir=out_dir))
            if rec is None:
                verdict = "missing"
                break
            if rec.get("parent_sha") != rcd._parent_sha(*parents):
                verdict = "stale"
                break
            outs[rid] = rec.get("output") or ""
        status[rd] = verdict
        if verdict != "replay":
            for rd2 in REPLAY_ROUNDS[i + 1:]:
                status[rd2] = "downstream"
            break
        texts[rd] = outs
        seq = tuple(outs[rid] for rid in _PRISTINE["ROLE_ORDER"])
        parents = seq if rd == "r0" else parents + seq
    n_replay = sum(1 for v in status.values() if v == "replay")
    return {"status": status, "texts": texts,
            "replay_calls": rcd.N_AGENTS * n_replay,
            "new_r0r2_calls": rcd.N_AGENTS * (len(REPLAY_ROUNDS) - n_replay),
            "full_chain": n_replay == len(REPLAY_ROUNDS)}


def replay_plan(agent_model: str, arms: Sequence[str], items: Sequence, samples: int, *,
                scaffold: str, max_tokens_agent: int, out_dir: Path) -> dict:
    debates = []
    for arm in arms:
        for it in items:
            for idx in range(samples):
                pr = probe_chain(agent_model, scaffold, arm, it, idx,
                                 max_tokens_agent=max_tokens_agent, out_dir=out_dir)
                debates.append({"arm": arm, "item": it, "idx": idx, **pr})
    n = len(debates)
    n_full = sum(1 for d in debates if d["full_chain"])
    stale = [d for d in debates if "stale" in d["status"].values()]
    return {
        "n_debates": n,
        "n_full_chain": n_full,
        "n_incomplete": n - n_full,
        "n_stale": len(stale),
        "replay_calls": sum(d["replay_calls"] for d in debates),
        "new_r0r2_calls": sum(d["new_r0r2_calls"] for d in debates),
        "new_moderator_calls": 2 * n,
        "new_r3r4_calls": 2 * rcd.N_AGENTS * n,
        "new_calls": sum(d["new_r0r2_calls"] for d in debates) + NEW_CALLS_PER_DEBATE * n,
        "debates": debates,
    }


# ---------------------------------------------------------------------------
# Cost model: NEW calls only
# ---------------------------------------------------------------------------

def parse_prices(extra: Sequence[str]) -> dict[str, tuple[float, float]]:
    prices = dict(rcd.PRICES)
    for spec in extra:
        try:
            name, val = spec.split("=", 1)
            pin, pout = (float(x) for x in val.split(","))
        except ValueError:
            raise SystemExit(f"--price expects NAME=IN,OUT (USD per Mtok), got {spec!r}")
        prices[name.strip()] = (pin, pout)
    return prices


def _measured(agent_model: str, scaffold: str, plan: dict, round_name: str,
              max_tokens: int, field: str, *, out_dir: Path,
              cap: int = 400) -> Optional[float]:
    """Mean API-billed ``field`` (prompt_tokens / completion_tokens) of the
    registered self-moderated run's OWN records for the same debates and
    round. Read-only. For the agent rounds the seats' records are read; for
    the two moderator rounds the registered run's mod-<agent> records. None
    if nothing is on disk.

    Why these are the right priors:
      synthesis prompt     byte-identical to the registered run's (same r2
                           texts) -- exact up to vendor tokenizer drift
      r3 / r4 prompt       same seat, same instruction, a different vendor's
                           synthesis in place of grok's -- close
      r3 / r4 completion   same model, same instruction -- the best prior
      integration prompt   same structure, other-vendor synthesis/labels -- close
      moderator completion NOT used: another vendor's moderator has no
                           measurement, so rcd.ASSUMED_COMPLETION stays."""
    vals: list[int] = []
    if round_name in rcd.MODERATOR_ROUNDS:
        rids: tuple = (_PRISTINE["moderator_role_id"](agent_model),)
    else:
        rids = tuple(r.role_id for r in _PRISTINE["ROLES"])
    for d in plan["debates"]:
        if not d["full_chain"]:
            continue
        for rid in rids:
            rec = _read_rec(pristine_path(agent_model, scaffold, d["arm"], d["item"].item_id,
                                          d["idx"], round_name, rid, max_tokens,
                                          out_dir=out_dir))
            if rec and rec.get(field):
                vals.append(int(rec[field]))
            if len(vals) >= cap:
                break
        if len(vals) >= cap:
            break
    return (sum(vals) / len(vals)) if vals else None


def _measured_completion(agent_model: str, scaffold: str, plan: dict, round_name: str,
                         max_tokens: int, *, out_dir: Path, cap: int = 400) -> Optional[float]:
    return _measured(agent_model, scaffold, plan, round_name, max_tokens,
                     "completion_tokens", out_dir=out_dir, cap=cap)


def cost_model_new(agent_model: str, mod_model: str, arms: Sequence[str], items: Sequence,
                   samples: int, plan: dict, *, prices: dict, scaffold: str,
                   transcript_cap: int, max_tokens_agent: int, max_tokens_label: int,
                   max_tokens_vote: int, out_dir: Path, use_bpe: bool = True) -> dict:
    """Prompt tokens are counted on REAL constructed prompts: the synthesis
    prompt on the actual cached r2 texts of every replaying debate, the r3
    prompt on the seat's actual cached r2 plus a synthesis stand-in; the
    integration and vote prompts on stand-ins (their inputs do not exist yet).
    Completion tokens: r3/r4 from the registered run's measured records for the
    same debates; the two moderator rounds at the base runner's ASSUMED lengths
    (a different vendor's moderator has no measurement yet) -- FLAGGED."""
    if not items:
        raise ValueError("cost model needs at least one item")
    bpe = rcd._optional_bpe_counter() if use_bpe else None

    def tok(s: str) -> int:
        return rcd.count_tokens(s, tokenizer=bpe) if bpe else rcd.count_tokens(s)

    comp_agent = rcd.MEASURED_NCOT_COMPLETION.get(agent_model) or max(
        rcd.MEASURED_NCOT_COMPLETION.values())
    stand_in = "word " * comp_agent
    synth_stand_in = "word " * rcd.ASSUMED_COMPLETION["synthesis"]
    prop_stand_in = "word " * rcd.ASSUMED_COMPLETION["integration"]
    obj_stand_in = "word " * 40
    roles = _PRISTINE["ROLES"]
    order = _PRISTINE["ROLE_ORDER"]
    probe = sorted(items, key=lambda i: len(i.post_text))[len(items) // 2]

    # synthesis + r3 measured on real cached r2 texts where the chain replays
    syn_tok: list[int] = []
    r3_tok: list[int] = []
    for d in plan["debates"]:
        if not d["full_chain"]:
            continue
        r2 = d["texts"]["r2"]
        syn_tok.append(tok(rcd.synthesis_user(d["arm"], d["item"], r2, allow_unresolved=True,
                                              cap=transcript_cap)))
        for r in roles:
            r3_tok.append(tok(rcd.r3_label_user(d["arm"], d["item"], r, r2[r.role_id],
                                                synth_stand_in, cap=transcript_cap)))
    prior0 = {rid: stand_in for rid in order}
    labels0 = {rid: "ACCEPT_WITH_MODIFICATION" for rid in order}
    objs0 = {rid: obj_stand_in for rid in order}
    r = roles[0]
    stand_prompts = {
        "r0": rcd.r0_user(arms[0], probe, r, allow_unresolved=True),
        "r1": _PRISTINE["r1_user"](arms[0], probe, r, stand_in, prior0, allow_unresolved=True,
                                   cap=transcript_cap),
        "r2": _PRISTINE["r2_user"](arms[0], probe, r, stand_in, stand_in, prior0,
                                   allow_unresolved=True, cap=transcript_cap),
        "synthesis": rcd.synthesis_user(arms[0], probe, prior0, allow_unresolved=True,
                                        cap=transcript_cap),
        "r3_label": rcd.r3_label_user(arms[0], probe, r, stand_in, synth_stand_in,
                                      cap=transcript_cap),
        "integration": rcd.integration_user(arms[0], probe, synth_stand_in, labels0, objs0,
                                            allow_unresolved=True, cap=transcript_cap),
        "r4_vote": rcd.r4_vote_user(arms[0], probe, r, synth_stand_in,
                                    "ACCEPT_WITH_MODIFICATION", obj_stand_in, prop_stand_in,
                                    cap=transcript_cap),
    }
    ptok = {k: tok(v) for k, v in stand_prompts.items()}
    measured_prompt: dict[str, object] = {}
    if syn_tok:
        ptok["synthesis"] = round(sum(syn_tok) / len(syn_tok))
        ptok["r3_label"] = round(sum(r3_tok) / len(r3_tok))
        measured_prompt = {"synthesis": f"{len(syn_tok)} constructed prompts",
                           "r3_label": f"{len(r3_tok)} constructed prompts"}
    # API-billed prompt_tokens of the registered run's own records for the
    # same debates beat any local tokenizer, and are the only honest count
    # when tiktoken is absent (the regex fallback undercounts 20-35%). The
    # synthesis prompt is byte-identical to the registered run's; the other
    # three differ only by the other vendor's synthesis/label text.
    max_of = {"synthesis": REGISTERED_MODERATOR_CAP, "r3_label": max_tokens_label,
              "integration": REGISTERED_MODERATOR_CAP, "r4_vote": max_tokens_vote}
    for rd in NEW_ROUNDS:
        mp = _measured(agent_model, scaffold, plan, rd, max_of[rd], "prompt_tokens",
                       out_dir=out_dir)
        if mp is not None and (bpe is None or rd in ("synthesis",)):
            ptok[rd] = round(mp)
            measured_prompt[rd] = "API-billed prompt_tokens of the registered run's records"

    m_r3 = _measured_completion(agent_model, scaffold, plan, "r3_label", max_tokens_label,
                                out_dir=out_dir)
    m_r4 = _measured_completion(agent_model, scaffold, plan, "r4_vote", max_tokens_vote,
                                out_dir=out_dir)
    comps = {"r0": comp_agent, "r1": comp_agent, "r2": comp_agent,
             "synthesis": rcd.ASSUMED_COMPLETION["synthesis"],
             "r3_label": m_r3 if m_r3 is not None else rcd.ASSUMED_COMPLETION["r3_label"],
             "integration": rcd.ASSUMED_COMPLETION["integration"],
             "r4_vote": m_r4 if m_r4 is not None else rcd.ASSUMED_COMPLETION["r4_vote"]}
    n_calls = {"r0": plan["new_r0r2_calls"] // 3, "r1": plan["new_r0r2_calls"] // 3,
               "r2": plan["new_r0r2_calls"] // 3,
               "synthesis": plan["n_debates"], "r3_label": rcd.N_AGENTS * plan["n_debates"],
               "integration": plan["n_debates"], "r4_vote": rcd.N_AGENTS * plan["n_debates"]}
    # a partial chain replays r0 but not r1/r2: split the fresh r0-r2 calls by round
    fresh = Counter()
    for d in plan["debates"]:
        for rd, st in d["status"].items():
            if st != "replay":
                fresh[rd] += rcd.N_AGENTS
    for rd in REPLAY_ROUNDS:
        n_calls[rd] = fresh[rd]

    pin, pout = prices.get(agent_model, (0.0, 0.0))
    min_, mout_ = prices.get(mod_model, (0.0, 0.0))
    rounds: dict = {}
    tot_in = tot_out = 0
    tot_usd = 0.0
    for rd in rcd.ROUNDS:
        calls = n_calls[rd]
        is_mod = rd in rcd.MODERATOR_ROUNDS
        p_in, p_out = (min_, mout_) if is_mod else (pin, pout)
        tin = ptok[rd] * calls
        tout = round(comps[rd] * calls)
        usd = tin / 1e6 * p_in + tout / 1e6 * p_out
        tot_in += tin
        tot_out += tout
        tot_usd += usd
        rounds[rd] = {"calls": calls, "billed_to": mod_model if is_mod else agent_model,
                      "prompt_tokens_each": ptok[rd],
                      "prompt_measured_on": measured_prompt.get(rd, "stand-in prompt, local tokenizer"),
                      "completion_tokens_each": round(comps[rd], 1),
                      "completion_source": ("assumed (other-vendor moderator, no measurement)"
                                            if is_mod else
                                            "measured from the registered grok run's records"
                                            if rd in POST_SYNTHESIS_ROUNDS and
                                            (m_r3 if rd == "r3_label" else m_r4) is not None
                                            else "measured single-agent narrative_cot"),
                      "input_tokens": tin, "output_tokens": tout,
                      "price_in_per_mtok": p_in, "price_out_per_mtok": p_out,
                      "usd": round(usd, 2)}
    return {
        "agent_model": agent_model, "moderator_model": mod_model,
        "priced": agent_model in prices and mod_model in prices,
        "arms": list(arms), "n_items": len(items), "samples": samples,
        "n_debates": plan["n_debates"], "n_full_chain": plan["n_full_chain"],
        "n_incomplete_chain": plan["n_incomplete"], "n_stale_chain": plan["n_stale"],
        "replay_calls": plan["replay_calls"], "new_calls": plan["new_calls"],
        "new_calls_per_full_chain_debate": NEW_CALLS_PER_DEBATE,
        "total_calls_incl_replay": 17 * plan["n_debates"],
        "input_tokens": tot_in, "output_tokens": tot_out, "usd": round(tot_usd, 2),
        "moderator_usd": round(sum(v["usd"] for k, v in rounds.items()
                                   if k in rcd.MODERATOR_ROUNDS), 2),
        "rounds": rounds,
        "tokenizer": (
            "the 4 NEW rounds are priced on API-billed prompt_tokens of the registered "
            "run's records; " if all(str(measured_prompt.get(rd, "")).startswith("API")
                                     for rd in NEW_ROUNDS) else "")
            + ("local counts: tiktoken cl100k_base BPE" if bpe else
               "local counts: regex approximation, undercounts 20-35% (fresh r0-r2 only)"),
        "prices": "run_crowdgold_deliberation.PRICES (+ --price overrides); unverified list rates",
    }


def print_cost_model(cm: dict) -> None:
    print("\n" + "=" * 78)
    print(f"DRY RUN -- cross-vendor moderator: {cm['moderator_model']} over "
          f"{cm['agent_model']} seats   (NEW calls only)")
    print("=" * 78)
    print(f"debates                 {cm['n_debates']:,}   "
          f"({cm['n_items']} items x {len(cm['arms'])} arms x {cm['samples']} samples)")
    print(f"r0-r2 chain on disk     {cm['n_full_chain']:,} full  "
          f"{cm['n_incomplete_chain']:,} incomplete  {cm['n_stale_chain']:,} stale")
    print(f"REPLAY from cache       {cm['replay_calls']:,} calls  (no spend)")
    print(f"NEW generation calls    {cm['new_calls']:,}   "
          f"({cm['new_calls_per_full_chain_debate']} per full-chain debate: "
          f"2 moderator + {2 * rcd.N_AGENTS} agent)")
    if not cm["priced"]:
        print("  WARNING: a model is NOT in PRICES -- the $ below is a placeholder; "
              "--run refuses until it is priced (rcd.PRICES entry or --price NAME=IN,OUT)")
    print(f"\n    {'round':<12}{'calls':>8}{'prompt tok':>12}{'compl tok':>11}"
          f"{'in tok':>13}{'out tok':>13}{'usd':>9}  billed to")
    for rd, r in cm["rounds"].items():
        print(f"    {rd:<12}{r['calls']:>8,}{r['prompt_tokens_each']:>12,}"
              f"{r['completion_tokens_each']:>11,}{r['input_tokens']:>13,}"
              f"{r['output_tokens']:>13,}{r['usd']:>9.2f}  {r['billed_to']}")
    print(f"    {'TOTAL':<12}{cm['new_calls']:>8,}{'':>12}{'':>11}"
          f"{cm['input_tokens']:>13,}{cm['output_tokens']:>13,}{cm['usd']:>9.2f}")
    print(f"\n  TOTAL ESTIMATED COST  ${cm['usd']:.2f}   "
          f"(moderator ${cm['moderator_usd']:.2f}, agent ${cm['usd'] - cm['moderator_usd']:.2f})")
    print(f"    tokenizer: {cm['tokenizer']}")
    print(f"    prices:    {cm['prices']}")
    for rd, r in cm["rounds"].items():
        if r["calls"]:
            print(f"    {rd:<12} prompt {r['prompt_measured_on']}  |  completion "
                  f"{r['completion_source']}")


# ---------------------------------------------------------------------------
# Panel (mirrors run_crowdgold_deliberation.main so the probe sees the same
# items the delegated run will)
# ---------------------------------------------------------------------------

def build_panel(agent_model: str, *, n_yta: int, n_nta: int, min_votes: int,
                min_consensus: float, seed: int, keep_filtered: bool) -> tuple[list, int, int]:
    items = rcd.load_items(source="scruples", n_yta=99, n_nta=150, min_votes=min_votes,
                           min_consensus=min_consensus, seed=seed)
    panel = len(items)
    items = rcd.subset_items(items, n_yta, n_nta)
    dropped = 0
    if not keep_filtered:
        drop = rcd.filter_prone_items(agent_model)
        before = len(items)
        items = [i for i in items if i.item_id not in drop]
        dropped = before - len(items)
    return items, panel, dropped


def launch_command(agent_model: str, mod_model: str, tag: str, *, n_yta: int, n_nta: int,
                   samples: int, seed: int) -> str:
    return (f".venv/bin/python -m scripts.run_crowdgold_modvendor --run "
            f"--models {agent_model} --moderator-model {mod_model} "
            f"--n-yta {n_yta} --n-nta {n_nta} --samples {samples} --seed {seed} "
            f"--tag {tag}")


def existing_rows_moderators(rows_path: Path) -> set[str]:
    if not rows_path.exists():
        return set()
    with rows_path.open() as f:
        return {r.get("moderator_model", "") for r in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# Readout (offline): this cell next to the registered embodied/on cell,
# through analyze_topology_2x2.analyse_cell unchanged
# ---------------------------------------------------------------------------

def paired_vs_embodied_sample0(tag: str, embodied_tag: str, model: str, *, draws: int,
                               seed: int, out_dir: Path = OUT_DIR) -> dict:
    """The paired readout the replay makes free: on sample 0 the moderator is
    the ONLY difference between this cell and embodied/on, so compare the two
    on the matched (arm, item) debates -- S2 correctness (codable on both
    sides), the >= 2-objector flag and the codable error lift -- with
    item-clustered bootstrap CIs on the paired deltas. Both sides are read
    through analyze_unembodied_ablation.load_rows (the 16.10 conventions)."""
    from scripts.analyze_unembodied_ablation import _boot, load_rows
    mv = {(r["arm"], r["item"]): r for r in load_rows(out_dir / f"{tag}_rows.csv", model)
          if r["sample_idx"] == 0}
    emb_p = out_dir / f"{embodied_tag}_rows.csv"
    if not emb_p.exists():
        return {"available": False, "why": f"{emb_p.name} absent"}
    emb = {(r["arm"], r["item"]): r for r in load_rows(emb_p, model) if r["sample_idx"] == 0}
    keys = sorted(set(mv) & set(emb))
    fired = lambda r: r["n_objectors"] >= 2  # noqa: E731
    s2_by: dict = defaultdict(list)
    fire_by: dict = defaultdict(list)
    lift_by: dict = defaultdict(list)
    n_s2 = 0
    for k in keys:
        a, b = mv[k], emb[k]
        fire_by[k[1]].append((int(fired(a)), int(fired(b))))
        if a["s2_ok"] is not None and b["s2_ok"] is not None:
            n_s2 += 1
            s2_by[k[1]].append(int(a["s2_ok"]) - int(b["s2_ok"]))
            lift_by[k[1]].append((int(fired(a)), int(not a["s2_ok"]), int(fired(b)), int(not b["s2_ok"])))
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None  # noqa: E731

    def fire_delta(xs):
        return None if not xs else sum(a for a, _ in xs) / len(xs) - sum(b for _, b in xs) / len(xs)

    def lift_of(pairs, fi, wi):
        f = [p[wi] for p in pairs if p[fi]]
        u = [p[wi] for p in pairs if not p[fi]]
        return None if not f or not u else sum(f) / len(f) - sum(u) / len(u)

    def lift_delta(xs):
        a, b = lift_of(xs, 0, 1), lift_of(xs, 2, 3)
        return None if a is None or b is None else a - b
    s2_p, s2_lo, s2_hi = _boot(s2_by, mean, draws=draws, seed=seed) if s2_by else (None, None, None)
    f_p, f_lo, f_hi = _boot(fire_by, fire_delta, draws=draws, seed=seed + 1) if fire_by else (None, None, None)
    l_p, l_lo, l_hi = _boot(lift_by, lift_delta, draws=draws, seed=seed + 2) if lift_by else (None, None, None)
    mv_ok = [int(mv[k]["s2_ok"]) for k in keys if mv[k]["s2_ok"] is not None]
    em_ok = [int(emb[k]["s2_ok"]) for k in keys if emb[k]["s2_ok"] is not None]
    return {"available": True, "n_matched_debates": len(keys),
            "n_items": len({k[1] for k in keys}), "n_s2_pairs": n_s2,
            "modvendor_s2_sample0": mean(mv_ok), "embodied_s2_sample0": mean(em_ok),
            "s2_delta": {"delta": s2_p, "lo": s2_lo, "hi": s2_hi},
            "fire_rate_delta": {"delta": f_p, "lo": f_lo, "hi": f_hi,
                                "modvendor": mean([int(fired(mv[k])) for k in keys]),
                                "embodied": mean([int(fired(emb[k])) for k in keys])},
            "error_lift_delta": {"delta": l_p, "lo": l_lo, "hi": l_hi,
                                 "modvendor_lift": lift_of([p for v in lift_by.values() for p in v], 0, 1),
                                 "embodied_lift": lift_of([p for v in lift_by.values() for p in v], 2, 3),
                                 "n_fired_modvendor": sum(p[0] for v in lift_by.values() for p in v),
                                 "n_fired_embodied": sum(p[2] for v in lift_by.values() for p in v)}}


def readout_guards(tag: str, cell: Optional[dict], *, out_dir: Path = OUT_DIR) -> tuple[bool, list[str]]:
    """Why the cell is NOT readable: a missing / failed summary guard, a missing /
    failed completeness sidecar, or a codable share below MIN_CODABLE_SHARE."""
    reasons: list[str] = []
    sp = out_dir / f"{tag}_summary.json"
    if sp.exists():
        try:
            s = json.loads(sp.read_text())
        except json.JSONDecodeError:
            s = {}
        rg, og = s.get("round_guard", {}), s.get("outcome_guard", {})
        if not (rg.get("pass", False) and og.get("pass", False)):
            reasons.append(f"round guard {rg.get('pass')}, outcome guard {og.get('pass')}")
    else:
        reasons.append(f"{sp.name} missing: the run's guard verdict is unknown")
    cp = out_dir / f"{tag}_completeness.json"
    if cp.exists():
        c = json.loads(cp.read_text())
        if not c.get("pass", False):
            reasons.append(f"completeness: {c.get('n_missing')} of {c.get('n_expected')} launched "
                           "debates missing from the rows")
    else:
        reasons.append(f"{cp.name} missing: it is not known whether every launched debate is a row")
    if cell is not None and cell["n_debates"]:
        share = cell["n_codable"] / cell["n_debates"]
        if share < MIN_CODABLE_SHARE:
            reasons.append(f"codable share {share:.3f} < {MIN_CODABLE_SHARE} "
                           f"({cell['n_debates'] - cell['n_codable']} non-codable group verdicts)")
    return not reasons, reasons


def readout(tag: str, mod_model: str, agent_model: str, *, draws: int, seed: int) -> int:
    """Offline. Paths resolve against rcd.OUT_DIR (the directory the run wrote
    to; the real one, or the selftest's tempdir), as main()'s other paths do."""
    import scripts.analyze_topology_2x2 as a2
    from scripts.analyze_unembodied_ablation import load_rows
    out_dir = rcd.OUT_DIR
    rows_path = out_dir / f"{tag}_rows.csv"
    if not rows_path.exists() or not (out_dir / f"{tag}_votes.csv").exists():
        print(f"\nERROR: {rows_path} / {tag}_votes.csv not found.\n")
        return 2
    if a2.MODEL != agent_model:
        print(f"  NOTE: analyze_topology_2x2 is pinned to {a2.MODEL}; rows for "
              f"{agent_model} will be filtered out of its readout")
    cell = ("modvendor", mod_model)
    a2.CELL_SPECS[cell] = (tag, tuple(_PRISTINE["ROLE_ORDER"]))
    ref = ("embodied", "on")
    present = {}
    for c in (ref, cell):
        p = OUT_DIR / f"{a2.CELL_SPECS[c][0]}_rows.csv"
        if p.exists():
            present[c] = {r["item"] for r in load_rows(p, a2.MODEL)}
    matched = set.intersection(*present.values()) if present else set()
    res = {"n_items_matched": len(matched), "cells": {}}
    for i, c in enumerate(present):
        res["cells"][f"{c[0]}/{c[1]}"] = a2.analyse_cell(c, keep_items=matched, draws=draws,
                                                          seed=seed + 10 * i)
    cells = [res["cells"].get(f"{c[0]}/{c[1]}") for c in (ref, cell)]
    own = cells[1]
    ok, reasons = readout_guards(tag, own, out_dir=out_dir)
    res["guards"] = {"pass": ok, "reasons": reasons, "min_codable_share": MIN_CODABLE_SHARE,
                     "n_debates": own["n_debates"] if own else 0,
                     "n_codable": own["n_codable"] if own else 0}
    print("\n" + "=" * 84)
    print(f"ADDENDUM 16.17 -- CROSS-VENDOR MODERATOR   {tag}   (matched items: {len(matched)})")
    print("=" * 84)
    print(f"  guards: " + ("PASS" if ok else "GUARD FAILED -- " + "; ".join(reasons)))
    if own:
        print(f"  codable group verdicts {own['n_codable']} of {own['n_debates']} "
              f"({own['n_codable'] / own['n_debates']:.3f}; limit {MIN_CODABLE_SHARE})")
    if not ok:
        print(f"\n  *** GUARD FAILED -- {tag} is not readable ({'; '.join(reasons)}); no analysis "
              "JSON written. ***\n")
        return 4

    # ---- 1. PAIRED against embodied sample 0 (the moderator is the only difference)
    pr = paired_vs_embodied_sample0(tag, a2.CELL_SPECS[ref][0], a2.MODEL, draws=draws, seed=seed + 50,
                                    out_dir=out_dir)
    res["paired_vs_embodied_sample0"] = pr
    print("\n  1. PAIRED vs embodied/on sample 0 on the matched (arm, item) debates -- r0-r2 replay, "
          "so the moderator is the only difference; item-clustered CIs")
    if pr.get("available"):
        ci = lambda d: f"{a2._f(d['delta'])} [{a2._f(d.get('lo'))}, {a2._f(d.get('hi'))}]"  # noqa: E731
        print(f"     matched debates {pr['n_matched_debates']} over {pr['n_items']} items; "
              f"S2 pairs (codable both sides) {pr['n_s2_pairs']}")
        print(f"     S2 acc      modvendor {a2._f(pr['modvendor_s2_sample0'], '.3f')}   embodied s0 "
              f"{a2._f(pr['embodied_s2_sample0'], '.3f')}   paired delta {ci(pr['s2_delta'])}")
        fd = pr["fire_rate_delta"]
        print(f"     fire P(>=2) modvendor {a2._f(fd['modvendor'], '.3f')}   embodied s0 "
              f"{a2._f(fd['embodied'], '.3f')}   paired delta {ci(fd)}")
        ld = pr["error_lift_delta"]
        print(f"     error lift  modvendor {a2._f(ld['modvendor_lift'])} (n fired {ld['n_fired_modvendor']})   "
              f"embodied s0 {a2._f(ld['embodied_lift'])} (n fired {ld['n_fired_embodied']})   "
              f"paired delta {ci(ld)}")
    else:
        print(f"     not computable: {pr.get('why')}")

    # ---- 2. the pooled 16.10-style table (secondary view) --------------------
    hdr = ["embodied/on (16.10)", f"mod {mod_model}"]
    print("\n  2. POOLED 16.10 table (embodied/on over all its samples; secondary view)")
    print(f"  {'':<34}" + "".join(f"{h:>24}" for h in hdr))

    def row(label, get, fmt="+.3f"):
        print(f"  {label:<34}" + "".join(
            f"{(a2._f(get(c), fmt) if c else 'absent'):>24}" for c in cells))

    def ci_row(label, get):
        def fmt_ci(c):
            if not c:
                return "absent"
            d = get(c)
            if d is None or d.get("delta") is None:
                return "n/a"
            return f"{a2._f(d['delta'])} [{a2._f(d.get('lo'))}, {a2._f(d.get('hi'))}]"
        print(f"  {label:<34}" + "".join(f"{fmt_ci(c):>24}" for c in cells))
    row("n debates", lambda c: c["n_debates"], "d")
    row("n codable", lambda c: c["n_codable"], "d")
    row("ROLE-LOCK", lambda c: c["role_lock"], ".3f")
    row("localisation excess (P-binom)", lambda c: c["objection"]["localisation_excess"])
    row("fire rate P(>=2)", lambda c: c["objection"]["fire_rate"], ".3f")
    ci_row("G3 grip (stake seats) [CI]", lambda c: c["g3"] if c["g3"]["applicable"] else None)
    ci_row("error lift (>=2), codable [CI]", lambda c: c["lift"] if c["lift"]["powered"] else None)
    row("  n fired", lambda c: c["lift"]["n_fired"], "d")
    row("group verdict (S2) acc, codable", lambda c: c["aggregation"]["group_verdict_s2"], ".3f")
    ci_row("TRANSFER lift on grok solo [CI]", lambda c: c["transfer_grok_solo"])
    row("guard: mean n_r3_unparsed", lambda c: c["guards"]["mean_n_r3_unparsed"], ".4f")
    row("guard: P(non-codable | fired)", lambda c: c["guards"]["p_noncodable_given_fired"], ".3f")
    out = out_dir / f"{tag}_analysis.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


def completeness_check(tag: str, model: str, expected: set, *, out_dir: Optional[Path] = None,
                       summary_n_items: Optional[int] = None) -> dict:
    """Every launched (arm, item, sample) must be a row of <tag>_rows.csv;
    rcd.run drops an errored debate with one console line."""
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


# ---------------------------------------------------------------------------
# Selftest (offline; generation stubbed; cache redirected to a tempdir)
# ---------------------------------------------------------------------------

STUB_AGENT = "stub-agent"
STUB_MOD = "stub-mod"


def _round_of(user: str) -> str:
    if "This is the final round" in user:
        return "r4_vote"
    if "Construct one integrated proposal" in user:
        return "integration"
    if "Write one synthesis position" in user:
        return "synthesis"
    if "Respond to this synthesis" in user:
        return "r3_label"
    if "Write your opening statement" in user:
        return "r0"
    if "Write your rebuttal" in user:
        return "r1"
    return "r2"


def _make_stub(log: list):
    """The base runner's own stub, made model-sensitive: agent statements and
    the synthesis carry the model name, so a second model's transcripts and a
    second moderator's synthesis differ in TEXT -- which is what makes the
    downstream parent_sha differ, exactly as in the real cross-vendor case."""
    def stub(model, system, user, *, sample_idx=0, max_tokens=512, **kw):
        rd = _round_of(user)
        log.append((model, rd))
        res = rcd._stub_generate(model, system, user, sample_idx=sample_idx,
                                 max_tokens=max_tokens, **kw)
        if rd in REPLAY_ROUNDS:
            return rcd._StubResult(res.text.replace("VERDICT:", f"[{model}]\nVERDICT:", 1),
                                   completion_tokens=res.completion_tokens)
        if rd == "synthesis":
            return rcd._StubResult(res.text.replace("SYNTHESIS: shared_account_of_the_conflict",
                                                    f"SYNTHESIS: shared_account [{model}]"),
                                   completion_tokens=res.completion_tokens)
        return res
    return stub


def _snapshot(d: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in d.iterdir() if p.is_file()}


def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond, detail=""):
        print(f"[{'ok' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
        if not cond:
            fails.append(name)

    R = _PRISTINE["ROLE_ORDER"]
    check("pristine seats are the three registered embodied ids",
          R == ("writer_advocate", "counterparty", "neutral_adjudicator"))
    check("8 new calls per debate = 2 moderator + 6 agent", NEW_CALLS_PER_DEBATE == 8)
    check("MODERATOR_CELLS: sonnet -> cg_deliberation_modsonnet, haiku -> cg_deliberation_modhaiku, "
          "an unlisted fourth vendor -> cg_deliberation_mod<name>",
          default_tag("claude-sonnet-4-6") == "cg_deliberation_modsonnet"
          and default_tag("claude-haiku-4-5") == "cg_deliberation_modhaiku"
          and default_tag("vendor4-deploy") == "cg_deliberation_modvendor4-deploy")
    check("every default tag starts with TAG_PREFIX and none is a foreign tag",
          all(c["tag"].startswith(TAG_PREFIX) and c["tag"] not in FOREIGN_TAGS
              for c in MODERATOR_CELLS.values()))
    check("--price parses NAME=IN,OUT on top of rcd.PRICES without mutating it",
          parse_prices(["vendor4-deploy=1.5,6"])["vendor4-deploy"] == (1.5, 6.0)
          and "vendor4-deploy" not in rcd.PRICES
          and parse_prices([])["grok-4-1-fast-reasoning"] == rcd.PRICES["grok-4-1-fast-reasoning"])
    check("base runner's guard prints the GUARD FAILED literal (delegated run path)",
          "GUARD FAILED" in inspect.getsource(rcd.main))
    check(f"moderator prefix {MODERATOR_PREFIX!r} is not any sibling's and no sibling "
          f"prefix contains it (namespace_counts matches '_{MODERATOR_PREFIX}-' as a substring)",
          MODERATOR_PREFIX not in SIBLING_MODERATOR_PREFIXES
          and not any(f"_{MODERATOR_PREFIX}-" in f"_{p}-" or f"_{p}-" in f"_{MODERATOR_PREFIX}-"
                      for p in SIBLING_MODERATOR_PREFIXES))
    check(f"seat suffix '_{SEAT_SUFFIX}-' appears in no sibling runner's source "
          "(the r3/r4 namespace is this wrapper's alone)",
          not any(f"_{SEAT_SUFFIX}-" in (Path(__file__).parent / f).read_text()
                  for f in ("run_crowdgold_deliberation.py", "run_crowdgold_unembodied.py",
                            "run_crowdgold_topology.py", "run_crowdgold_game.py",
                            "run_crowdgold_coalition.py", "run_crowdgold_multivendor.py")
                  if (Path(__file__).parent / f).exists()))
    check("both registered moderator cells are priced in rcd.PRICES (--run would refuse otherwise)",
          all(m in rcd.PRICES for m in MODERATOR_CELLS)
          and DEFAULT_AGENT_MODEL in rcd.PRICES)
    check("--no-unresolved and --arms are banned (they change the r0-r2 prompt or the panel "
          "without changing the cache key, so replay would serve the wrong record)",
          "--no-unresolved" in BANNED_FLAGS and "--arms" in BANNED_FLAGS)

    # ---- (a) static: install() leaves everything upstream of the moderator alone
    G, A, M = "grok-4-1-fast-reasoning", "claude-sonnet-4-6", "claude-haiku-4-5"
    ex = ("as_asker", "itemX", 0)
    base = _PRISTINE["call_cache_path"]
    p_before = {rd: base(G, "narrative_cot", ex[0], ex[1], ex[2], rd, "writer_advocate", 2560)
                for rd in REPLAY_ROUNDS}
    install(G, A)
    try:
        check("install() keeps ROLES / ROLE_ORDER / ROLE_BY_ID / preamble / both moderator "
              "system prompts / r1_user / r2_user byte-identical (r0-r2 prompts unchanged)",
              rcd.ROLES is _PRISTINE["ROLES"] and rcd.ROLE_ORDER == R
              and rcd.ROLE_BY_ID is _PRISTINE["ROLE_BY_ID"]
              and rcd.R0_PREAMBLE == _PRISTINE["R0_PREAMBLE"]
              and rcd.SYNTHESIS_SYSTEM == _PRISTINE["SYNTHESIS_SYSTEM"]
              and rcd.INTEGRATION_SYSTEM == _PRISTINE["INTEGRATION_SYSTEM"]
              and rcd.r1_user is _PRISTINE["r1_user"] and rcd.r2_user is _PRISTINE["r2_user"]
              and rcd.CALLS_PER_CELL == 17)
        p_after = {rd: rcd.call_cache_path(G, "narrative_cot", ex[0], ex[1], ex[2], rd,
                                           "writer_advocate", 2560) for rd in REPLAY_ROUNDS}
        check("(a) r0/r1/r2 cache paths under install() == the registered run's paths",
              p_after == p_before)
        # (b) moderator namespace
        ps = rcd.call_cache_path(A, "narrative_cot", ex[0], ex[1], ex[2], "synthesis",
                                 rcd.moderator_role_id(A), 1024)
        p_grokmod = base(G, "narrative_cot", ex[0], ex[1], ex[2], "synthesis",
                         _PRISTINE["moderator_role_id"](G), 1024)
        p_sonnetself = base(A, "narrative_cot", ex[0], ex[1], ex[2], "synthesis",
                            _PRISTINE["moderator_role_id"](A), 1024)
        check("(b) synthesis path is under the moderator model with id modvendor-<agent>",
              ps.name.startswith("cgd_claude-sonnet-4-6_") and ps.name.endswith(
                  "_synthesis_modvendor-grok-4-1-fast-reasoning.json"))
        check("(b) it cannot collide with the grok moderator's mod-grok record",
              ps != p_grokmod and "_mod-grok" not in ps.name)
        check("(b) it cannot collide with the sonnet SELF-moderated pilot's mod-claude-sonnet "
              "record either (the base runner's --moderator-model WOULD: same path)",
              ps != p_sonnetself and p_sonnetself.name.endswith("_mod-claude-sonnet-4-6.json"))
        check("moderator_role_id refuses a moderator other than the installed one",
              (lambda: (rcd.moderator_role_id(G), False))() if False else
              _raises(lambda: rcd.moderator_role_id(G), ValueError))
        # (c) r3/r4 namespace
        for rd, cap in (("r3_label", 1024), ("r4_vote", 512)):
            pr = rcd.call_cache_path(G, "narrative_cot", ex[0], ex[1], ex[2], rd,
                                     "writer_advocate", cap)
            pb = base(G, "narrative_cot", ex[0], ex[1], ex[2], rd, "writer_advocate", cap)
            check(f"(c) {rd} path carries the seat id + _xmod-<moderator> and differs from "
                  f"the registered run's",
                  pr != pb and pr.name.endswith(f"_{rd}_writer_advocate_xmod-claude-sonnet-4-6.json"))
        pr_h = rcd.call_cache_path
    finally:
        uninstall()
    check("uninstall() restores the pristine functions",
          rcd.call_cache_path is _PRISTINE["call_cache_path"]
          and rcd.moderator_role_id is _PRISTINE["moderator_role_id"])
    install(G, M)
    try:
        ph = rcd.call_cache_path(G, "narrative_cot", ex[0], ex[1], ex[2], "r3_label",
                                 "writer_advocate", 1024)
        pm = rcd.call_cache_path(M, "narrative_cot", ex[0], ex[1], ex[2], "synthesis",
                                 rcd.moderator_role_id(M), 1024)
        check("haiku cell: its own r3/r4 and moderator namespaces, disjoint from the sonnet cell's",
              ph.name.endswith("_r3_label_writer_advocate_xmod-claude-haiku-4-5.json")
              and pm.name.startswith("cgd_claude-haiku-4-5_")
              and pm.name.endswith("_synthesis_modvendor-grok-4-1-fast-reasoning.json")
              and ph != pr_h(G, "narrative_cot", ex[0], ex[1], ex[2], "r3_label",
                             "writer_advocate", 1024))
    finally:
        uninstall()
    check("install() refuses moderator == agent (that is the registered self-moderated run)",
          _raises(lambda: install(G, G), SystemExit))
    check("install() is idempotent (a second install does not double-suffix)",
          (install(G, A), install(G, A),
           rcd.call_cache_path(G, "narrative_cot", "as_asker", "i", 0, "r3_label", "counterparty",
                               1024).name.endswith("_r3_label_counterparty_xmod-claude-sonnet-4-6.json"),
           uninstall())[2])

    # ---- real cache, read-only: the registered chain replays under install()
    items_all = rcd.load_items(source="scruples", n_yta=99, n_nta=150, min_votes=50,
                               min_consensus=0.90, seed=44)
    panel = rcd.subset_items(items_all, 99, 150)
    real = [it for it in panel
            if pristine_path(G, "narrative_cot", "as_asker", it.item_id, 0, "r0",
                             "writer_advocate", 2560, out_dir=OUT_DIR).exists()][:3]
    if real:
        install(G, A)
        try:
            ok_all = True
            detail = []
            for it in real:
                for arm in ("third_person", "as_asker"):
                    pr = probe_chain(G, "narrative_cot", arm, it, 0, max_tokens_agent=2560,
                                     out_dir=OUT_DIR)
                    if not pr["full_chain"]:
                        ok_all = False
                        detail.append(f"{arm}/{it.item_id[:8]}: {pr['status']}")
                        continue
                    r0, r1 = pr["texts"]["r0"], pr["texts"]["r1"]
                    for r in rcd.ROLES:
                        prompts = {
                            "r0": rcd.r0_user(arm, it, r, allow_unresolved=True),
                            "r1": rcd.r1_user(arm, it, r, r0[r.role_id], r0, allow_unresolved=True, cap=0),
                            "r2": rcd.r2_user(arm, it, r, r0[r.role_id], r1[r.role_id], r1,
                                              allow_unresolved=True, cap=0),
                        }
                        for rd, user in prompts.items():
                            rec = _read_rec(rcd.call_cache_path(G, "narrative_cot", arm, it.item_id,
                                                                0, rd, r.role_id, 2560))
                            if rec is None or rec.get("prompt_chars") != len(user) or \
                                    rec.get("post_sha256") != rcd.sha256_text(it.post_text):
                                ok_all = False
                                detail.append(f"{arm}/{it.item_id[:8]}/{rd}/{r.role_id}: "
                                              f"prompt_chars {rec and rec.get('prompt_chars')} "
                                              f"vs {len(user)}")
            check(f"(a) REAL cache: {len(real)} panel items x 2 arms replay r0/r1/r2 under "
                  "install() -- paths resolve to the registered records, parent_sha chain "
                  "matches, prompt_chars == len(prompt built now), post sha matches",
                  ok_all, "; ".join(detail[:4]))
        finally:
            uninstall()
    else:
        check("(a) REAL cache check SKIPPED -- no registered grok r0 record on this machine "
              "(vacuous; run on the machine that holds divergence_study_outputs)", True)

    # ---- no foreign file in either namespace family; registered cells' footprint
    foreign = any_modvendor_foreign()
    check(f"no foreign (stub / selftest) file carries this runner's namespaces "
          f"(_{MODERATOR_PREFIX}- / _{SEAT_SUFFIX}-) in {OUT_DIR.name}", foreign == [],
          str(foreign[:3]))
    for mm in MODERATOR_CELLS:
        clean, counts = cache_is_clean(G, mm)
        state = "EMPTY" if clean else f"footprint {counts}"
        check(f"{mm} cell namespaces: {state} (a non-empty footprint may only be a "
              f"registered run's own files; the run path refuses it without --resume)",
              clean or all(v > 0 for v in counts.values()), str(counts))

    # ---- tempdir, stubbed: construct the collisions, then prove the wrapper
    real_gen, real_out = rcd.generate_any, rcd.OUT_DIR
    it2 = rcd.subset_items(panel, 1, 1)
    it0 = it2[0]
    arm = "as_asker"
    log: list = []
    try:
        rcd.generate_any = _make_stub(log)
        # ---------------- base runner as-is: (b) and (c) are NOT safe ----------
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            rcd.OUT_DIR = tdp
            try:
                rowA, _, callsA = rcd.run_deliberation(STUB_AGENT, arm, it0, 0)   # "grok cell"
                snapA = _snapshot(tdp)
                r3A = base(STUB_AGENT, "narrative_cot", arm, it0.item_id, 0, "r3_label",
                           "writer_advocate", 1024)
                r3A_sha = json.loads(snapA[r3A.name])["parent_sha"]
                rowB, _, _ = rcd.run_deliberation(STUB_MOD, arm, it0, 0)         # "sonnet self pilot"
                synB = base(STUB_MOD, "narrative_cot", arm, it0.item_id, 0, "synthesis",
                            _PRISTINE["moderator_role_id"](STUB_MOD), 1024)
                synB_bytes = synB.read_bytes()
                check("collision setup: 34 records (two self-moderated stub cells) in the tempdir",
                      len(snapA) == 17 and len(_snapshot(tdp)) == 34 and rowA["n_calls"] == 17)
                log.clear()
                rowC, _, callsC = rcd.run_deliberation(STUB_AGENT, arm, it0, 0,
                                                       moderator_model=STUB_MOD)
                rounds_c = Counter(rd for _, rd in log)
                check("base runner + --moderator-model: r0/r1/r2 replay (0 calls) and 8 calls "
                      "are made (parent_sha rejects the stale r3/r4, so they are NOT replayed)",
                      rounds_c == Counter({"synthesis": 1, "integration": 1, "r3_label": 3,
                                           "r4_vote": 3}), str(dict(rounds_c)))
                r3_now = json.loads(r3A.read_bytes())
                check("(c) UNSAFE in the base runner: the cross-vendor r3 call used the "
                      "registered run's r3 PATH and OVERWROTE its record (parent_sha changed)",
                      r3A.read_bytes() != snapA[r3A.name] and r3_now["parent_sha"] != r3A_sha)
                check("(b) UNSAFE in the base runner: the cross-vendor synthesis landed on the "
                      "moderator model's SELF-moderated synthesis path and overwrote it",
                      synB.read_bytes() != synB_bytes)
                log.clear()
                rcd.run_deliberation(STUB_AGENT, arm, it0, 0)   # the registered cell again
                # Its own integration record was NOT clobbered (the cross-vendor
                # integration landed on the moderator's self-pilot path), and the
                # deterministic stub regenerates byte-identical r3 text, so the
                # integration parent_sha still matches and it replays. Under a
                # real model the regenerated r3 would differ and integration
                # would be re-spent too; the 6 here is the stub's floor.
                check("consequence: re-running the registered self-moderated cell now finds its "
                      "own r3/r4 stale and RE-GENERATES them (6 calls; integration replays only "
                      "because the stub is deterministic) -- the registered cell would be "
                      "silently re-spent and re-sampled",
                      Counter(rd for _, rd in log) == Counter({"r3_label": 3, "r4_vote": 3}),
                      str(dict(Counter(rd for _, rd in log))))
            finally:
                rcd.OUT_DIR = real_out
        # ---------------- the wrapper: disjoint, replaying, non-clobbering ------
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            rcd.OUT_DIR = tdp
            try:
                for a in ("third_person", "as_asker"):
                    for it in it2:
                        rcd.run_deliberation(STUB_AGENT, a, it, 0)       # registered cell
                rcd.run_deliberation(STUB_MOD, arm, it0, 0)              # moderator's self pilot
                snap0 = _snapshot(tdp)
                clean0, _ = cache_is_clean(STUB_AGENT, STUB_MOD, out_dir=tdp)
                check("cache_is_clean(agent, moderator) is True with only the two self-moderated "
                      "cells on disk (their namespaces are not ours)", clean0)
                plan0 = replay_plan(STUB_AGENT, ("third_person", "as_asker"), it2, 1,
                                    scaffold="narrative_cot", max_tokens_agent=2560, out_dir=tdp)
                check("replay_plan: 4 debates, all full-chain, 36 replay calls, 32 NEW calls "
                      "(8 per debate), 0 fresh r0-r2",
                      (plan0["n_debates"], plan0["n_full_chain"], plan0["replay_calls"],
                       plan0["new_calls"], plan0["new_r0r2_calls"]) == (4, 4, 36, 32, 0))
                install(STUB_AGENT, STUB_MOD)
                log.clear()
                rowW, votesW, callsW = rcd.run_deliberation(STUB_AGENT, arm, it0, 0,
                                                            moderator_model=STUB_MOD)
                rounds_w = Counter(rd for _, rd in log)
                models_w = Counter(m for m, _ in log)
                check("wrapper: exactly 8 generation calls per debate -- synthesis + integration "
                      "by the moderator, 3 r3 + 3 r4 by the agent; r0/r1/r2 replayed",
                      rounds_w == Counter({"synthesis": 1, "integration": 1, "r3_label": 3,
                                           "r4_vote": 3})
                      and models_w == Counter({STUB_MOD: 2, STUB_AGENT: 6}),
                      f"{dict(rounds_w)} {dict(models_w)}")
                snap1 = _snapshot(tdp)
                check("wrapper: EVERY pre-existing record is byte-identical afterwards (the "
                      "registered cell's r3/r4 and the moderator's self-pilot synthesis are untouched)",
                      all(snap1.get(k) == v for k, v in snap0.items()))
                new = sorted(set(snap1) - set(snap0))
                mod_new = [n for n in new if n.startswith(f"cgd_{STUB_MOD}_")
                           and n.endswith(f"_{moderator_cache_id(STUB_AGENT)}.json")]
                seat_new = [n for n in new if n.startswith(f"cgd_{STUB_AGENT}_")
                            and f"_{SEAT_SUFFIX}-{STUB_MOD}.json" in n]
                check("wrapper: 8 new files -- 2 under cgd_<moderator>_*_modvendor-<agent>, "
                      "6 under cgd_<agent>_*_{r3_label,r4_vote}_<seat>_xmod-<moderator>",
                      len(new) == 8 and len(mod_new) == 2 and len(seat_new) == 6, str(new))
                r3W = [json.loads(snap1[n]) for n in seat_new if "_r3_label_" in n]
                check("wrapper: r3 records carry the seat id inside, the moderator's synthesis "
                      "in parent_sha (differs from the registered cell's r3 parent_sha)",
                      all(r["role_id"] in R for r in r3W)
                      and all(r["parent_sha"] != json.loads(snap0[base(
                          STUB_AGENT, "narrative_cot", arm, it0.item_id, 0, "r3_label", r["role_id"],
                          1024).name])["parent_sha"] for r in r3W))
                check("wrapper: row/votes name the moderator; n_calls 17; seats are the embodied ids",
                      rowW["moderator_model"] == STUB_MOD and rowW["n_calls"] == 17
                      and [v["role_id"] for v in votesW] == list(R))
                check("wrapper: the R0/R1/R2 texts the moderator read are the REGISTERED cell's "
                      "(same output bytes as the pre-existing r2 records)",
                      all(c["output"] == json.loads(snap0[base(
                          STUB_AGENT, "narrative_cot", arm, it0.item_id, 0, c["round"], c["role_id"],
                          2560).name])["output"] for c in callsW if c["round"] in REPLAY_ROUNDS))
                clean1, counts1 = cache_is_clean(STUB_AGENT, STUB_MOD, out_dir=tdp)
                check("cache_is_clean is now False and counts exactly the 8 new records (the "
                      "--run path refuses without --resume)",
                      not clean1 and sum(counts1.values()) == 8, str(counts1))
                log.clear()
                rcd.run_deliberation(STUB_AGENT, arm, it0, 0, moderator_model=STUB_MOD)
                check("--resume semantics: a second wrapper pass makes 0 generation calls",
                      log == [])
                # a DIFFERENT moderator is a disjoint namespace
                uninstall()
                install(STUB_AGENT, "stub-mod2")
                log.clear()
                rcd.run_deliberation(STUB_AGENT, arm, it0, 0, moderator_model="stub-mod2")
                snap2 = _snapshot(tdp)
                check("second moderator cell: 8 new calls, 8 new files, nothing pre-existing touched",
                      len(log) == 8 and len(set(snap2) - set(snap1)) == 8
                      and all(snap2.get(k) == v for k, v in snap1.items()))
                clean_s, _ = cache_is_clean(STUB_AGENT, STUB_MOD, out_dir=tdp)
                clean_2, _ = cache_is_clean(STUB_AGENT, "stub-mod2", out_dir=tdp)
                check("cache_is_clean is per moderator cell", not clean_s and not clean_2
                      and cache_is_clean(STUB_AGENT, "stub-mod3", out_dir=tdp)[0])
                uninstall()
                # ---- (d) through rcd.main: the tag writes its own CSVs ----------
                install(STUB_AGENT, STUB_MOD)
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = rcd.main(["--models", STUB_AGENT, "--moderator-model", STUB_MOD,
                                   "--n-yta", "1", "--n-nta", "1", "--samples", "1",
                                   "--n-boot", "20", "--workers", "1", "--keep-filtered",
                                   "--tag", "cg_deliberation_modstub"])
                uninstall()
                names = {p.name for p in tdp.iterdir()}
                mods = existing_rows_moderators(tdp / "cg_deliberation_modstub_rows.csv")
                check("(d) --tag cg_deliberation_modstub writes its OWN rows/votes/summary "
                      "(rc 0, moderator_model column = the moderator) and no cg_deliberation_* "
                      "base file",
                      rc == 0 and {"cg_deliberation_modstub_rows.csv",
                                   "cg_deliberation_modstub_votes.csv",
                                   "cg_deliberation_modstub_summary.json"} <= names
                      and mods == {STUB_MOD}
                      and not any(n.startswith("cg_deliberation_rows") or
                                  n.startswith("cg_deliberation_votes") for n in names),
                      f"rc={rc} mods={mods}")
                # ---- CLI: dry-run counts NEW calls only; no --run refuses --------
                pr_args = ["--models", STUB_AGENT, "--moderator-model", STUB_MOD,
                           "--n-yta", "1", "--n-nta", "1", "--keep-filtered", "--no-bpe",
                           "--price", f"{STUB_AGENT}=0.2,0.5", "--price", f"{STUB_MOD}=3,15"]
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = main(["--dry-run", "--tag", "cg_deliberation_modstub_dry"] + pr_args)
                out = buf.getvalue()
                check("main(--dry-run): exit 0; counts 4 debates, 36 replay calls, 32 NEW calls, "
                      "prices the moderator rounds to the moderator; prints the launch command",
                      rc == 0 and "REPLAY from cache       36 calls" in out
                      and "NEW generation calls    32" in out
                      and f"synthesis" in out and STUB_MOD in out
                      and "--run --models" in out, out[-600:] if rc else "")
                cm_path = tdp / "cg_deliberation_modstub_dry_dryrun.json"
                cmj = json.loads(cm_path.read_text()) if cm_path.exists() else {}
                check("dry-run JSON: new_calls 32, replay_calls 36, moderator rounds billed to "
                      "the moderator at its price, agent rounds to the agent",
                      cmj.get("new_calls") == 32 and cmj.get("replay_calls") == 36
                      and cmj["rounds"]["synthesis"]["billed_to"] == STUB_MOD
                      and cmj["rounds"]["synthesis"]["price_in_per_mtok"] == 3.0
                      and cmj["rounds"]["r3_label"]["billed_to"] == STUB_AGENT
                      and cmj["rounds"]["r0"]["calls"] == 0)
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = main(["--tag", "cg_deliberation_modstub_dry"] + pr_args)
                check("main() without --run prints the cost model and REFUSES (exit 2)",
                      rc == 2 and "--run" in buf.getvalue() and "NEW generation calls" in buf.getvalue())
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = main(["--dry-run", "--tag", "cg_deliberation"] + pr_args)
                check("main() refuses --tag cg_deliberation (the registered cell's tag) before "
                      "doing anything (exit 2)", rc == 2 and "DRY RUN" not in buf.getvalue())
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = main(["--dry-run", "--stake-nudge"] + pr_args)
                check("main() refuses --stake-nudge / --agent-scaffold / --pilot etc. (exit 4)",
                      rc == 4 and "refus" in buf.getvalue().lower())
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc_nu = main(["--dry-run", "--no-unresolved"] + pr_args)
                    rc_ar = main(["--dry-run", "--arms", "as_asker"] + pr_args)
                check("main() refuses --no-unresolved and --arms (exit 4) before probing anything",
                      rc_nu == 4 and rc_ar == 4 and "DRY RUN" not in buf.getvalue())
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc_tc = main(["--dry-run", "--transcript-cap", "400"] + pr_args)
                    rc_t0 = main(["--dry-run", "--transcript-cap", "0", "--tag",
                                  "cg_deliberation_modstub_dry"] + pr_args)
                check("main() refuses --transcript-cap != 0 (exit 4, same class as --no-unresolved: prompt "
                      "text the cache key cannot see) and accepts the registered 0",
                      rc_tc == 4 and "DRY RUN" not in buf.getvalue().split("DRY RUN")[0]
                      and rc_t0 == 0 and "parent_sha" in buf.getvalue())
                # ---- completeness guard on the (d) cell: 2 items x 2 arms x 1 sample ----
                exp4 = {(a, it.item_id, 0) for a in ("third_person", "as_asker") for it in it2}
                comp_ok = completeness_check("cg_deliberation_modstub", STUB_AGENT, exp4, out_dir=tdp,
                                             summary_n_items=2)
                comp_bad = completeness_check("cg_deliberation_modstub", STUB_AGENT,
                                              exp4 | {("as_asker", "never-ran", 0)}, out_dir=tdp,
                                              summary_n_items=3)
                buf = io.StringIO()
                with redirect_stdout(buf):
                    print_completeness(comp_bad)
                check("completeness guard: the (d) cell has every launched debate as a row (4/4, PASS); "
                      "a launched debate with no row fails it (missing named, per-arm 2/2 vs 2/3) and "
                      "prints GUARD FAILED",
                      comp_ok["pass"] and comp_ok["n_rows"] == 4 and not comp_bad["pass"]
                      and comp_bad["missing"] == [["as_asker", "never-ran", 0]]
                      and comp_bad["per_arm"]["as_asker"] == {"expected": 3, "rows": 2}
                      and "GUARD FAILED" in buf.getvalue())
                check("the run path applies the completeness guard after rcd.main and quarantines the "
                      "CSVs on exit 4 (verified in source)",
                      "completeness_check(" in inspect.getsource(main)
                      and "_GUARD_FAILED_" in inspect.getsource(main))
                # ---- --readout refuses a cell without its guard records ----
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc_ro = main(["--readout", "--models", STUB_AGENT, "--moderator-model", STUB_MOD,
                                  "--tag", "cg_deliberation_modstub"])
                check("--readout REFUSES (GUARD FAILED, exit 4, no analysis JSON) a cell whose "
                      "completeness sidecar is missing",
                      rc_ro == 4 and "GUARD FAILED" in buf.getvalue() and "completeness" in buf.getvalue()
                      and not (tdp / "cg_deliberation_modstub_analysis.json").exists())
                (tdp / "cg_deliberation_modstub_completeness.json").write_text(json.dumps(comp_ok))
                sj_p = tdp / "cg_deliberation_modstub_summary.json"
                sj = json.loads(sj_p.read_text()); sj["outcome_guard"]["pass"] = False
                sj_bak = sj_p.read_text(); sj_p.write_text(json.dumps(sj))
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc_ro2 = main(["--readout", "--models", STUB_AGENT, "--moderator-model", STUB_MOD,
                                   "--tag", "cg_deliberation_modstub"])
                sj_p.write_text(sj_bak)
                check("--readout REFUSES (exit 4) a cell whose summary records a failed outcome guard",
                      rc_ro2 == 4 and "outcome guard False" in buf.getvalue())
                # ---- the codable-share gate and the paired block, on synthetic grok-labelled CSVs ----
                G = "grok-4-1-fast-reasoning"
                fields = ["model", "arm", "item_id", "sample_idx", "gold_verdict", "verdict",
                          "synthesis_verdict", "n_objectors", "n_r3_unparsed"]

                def _write_rows(path, recs):
                    with path.open("w", newline="") as fh:
                        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(recs)
                mv_rows, emb_rows = [], []
                for i in range(10):
                    item, gold = f"i{i:02d}", "YTA" if i % 2 == 0 else "NTA"
                    wrong = "NTA" if gold == "YTA" else "YTA"
                    for arm in ("third_person", "as_asker"):
                        mv_rows.append({"model": G, "arm": arm, "item_id": item, "sample_idx": 0,
                                        "gold_verdict": gold, "verdict": gold, "synthesis_verdict": gold,
                                        "n_objectors": 2 if i < 3 else 0, "n_r3_unparsed": 0})
                        for s in (0, 1):
                            emb_rows.append({"model": G, "arm": arm, "item_id": item, "sample_idx": s,
                                             "gold_verdict": gold,
                                             "verdict": (wrong if (s == 0 and i < 2) or (s == 1) else gold),
                                             "synthesis_verdict": gold,
                                             "n_objectors": 2 if (i == 0 and s == 0) else 0,
                                             "n_r3_unparsed": 0})
                _write_rows(tdp / "mv_synth_rows.csv", mv_rows)
                _write_rows(tdp / "emb_synth_rows.csv", emb_rows)
                pr = paired_vs_embodied_sample0("mv_synth", "emb_synth", G, draws=200, seed=3, out_dir=tdp)
                check("paired readout vs embodied SAMPLE 0 only: 20 matched debates, S2 delta +0.200 "
                      "(modvendor 1.000 vs embodied s0 0.800; the all-wrong sample 1 ignored), fire delta "
                      "+0.200 (6/20 vs 2/20), CIs present",
                      pr["available"] and pr["n_matched_debates"] == 20 and pr["n_s2_pairs"] == 20
                      and abs(pr["modvendor_s2_sample0"] - 1.0) < 1e-12
                      and abs(pr["embodied_s2_sample0"] - 0.8) < 1e-12
                      and abs(pr["s2_delta"]["delta"] - 0.2) < 1e-12 and pr["s2_delta"]["lo"] is not None
                      and abs(pr["fire_rate_delta"]["delta"] - 0.2) < 1e-12
                      and pr["fire_rate_delta"]["lo"] is not None)
                check("paired readout: the embodied side is unavailable when its rows file is absent "
                      "(never fabricated)",
                      paired_vs_embodied_sample0("mv_synth", "no_such", G, draws=5, seed=1, out_dir=tdp)
                      == {"available": False, "why": "no_such_rows.csv absent"})
                bad_cell = {"n_debates": 20, "n_codable": 18}
                (tdp / "mv_synth_summary.json").write_text(json.dumps(
                    {"round_guard": {"pass": True}, "outcome_guard": {"pass": True}}))
                (tdp / "mv_synth_completeness.json").write_text(json.dumps({"pass": True}))
                ok_g, why_g = readout_guards("mv_synth", bad_cell, out_dir=tdp)
                ok_g2, _ = readout_guards("mv_synth", {"n_debates": 20, "n_codable": 19}, out_dir=tdp)
                check("readout guard: a codable share of 0.90 (UNRESOLVED / NOVERDICT group verdicts the "
                      "base guard passes as parsed) is refused; 0.95 passes with intact guard records",
                      not ok_g and any("codable share 0.900" in w for w in why_g) and ok_g2)
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = main(["--dry-run", "--models", "a,b", "--moderator-model", STUB_MOD])
                check("main() refuses more than one agent model per invocation (exit 2)", rc == 2)
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = main(["--verify-cache-clean", "--models", STUB_AGENT,
                               "--moderator-model", STUB_MOD, "--n-yta", "1", "--n-nta", "1"])
                check("--verify-cache-clean exits 3 (DIRTY) once the cell has records",
                      rc == 3 and "DIRTY" in buf.getvalue())
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = main(["--verify-cache-clean", "--models", STUB_AGENT,
                               "--moderator-model", "stub-mod3"])
                check("--verify-cache-clean exits 0 (CLEAN) for a moderator never run", rc == 0)
                check("main(--run) on an unpriced model is refused (exit 2) -- asserted on "
                      "the source, the selftest never passes --run",
                      "not in PRICES" in inspect.getsource(main))
                check("the run path enforces cache_is_clean with --resume as the only override",
                      "if not clean and not known.resume" in inspect.getsource(main))
            finally:
                rcd.OUT_DIR = real_out
                uninstall()
    finally:
        rcd.generate_any = real_gen
        rcd.OUT_DIR = real_out
        uninstall()
    check("nothing landed in the real cache directory (no foreign file in either namespace)",
          any_modvendor_foreign() == [] and rcd.OUT_DIR == real_out)
    check("selftest left the base runner pristine",
          rcd.call_cache_path is _PRISTINE["call_cache_path"]
          and rcd.moderator_role_id is _PRISTINE["moderator_role_id"]
          and rcd.generate_any is real_gen)

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def _raises(fn, exc) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        description="Addendum 16.17: cross-vendor moderator over the registered grok "
                    "deliberation. r0-r2 replay from cache; 8 NEW calls per debate. "
                    "Generates NOTHING unless --run is passed. Unrecognised flags "
                    "(--workers, --n-boot, --no-unresolved, ...) are forwarded to "
                    "run_crowdgold_deliberation.main.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="new-calls cost model; exit")
    ap.add_argument("--run", action="store_true", help="REQUIRED to generate; spends money")
    ap.add_argument("--resume", action="store_true",
                    help="proceed despite a non-empty cache footprint for this cell")
    ap.add_argument("--verify-cache-clean", action="store_true")
    ap.add_argument("--readout", action="store_true",
                    help="offline: this cell next to the 16.10 embodied/on cell")
    ap.add_argument("--models", default=DEFAULT_AGENT_MODEL,
                    help="agent model (exactly ONE per invocation)")
    ap.add_argument("--moderator-model", default=None, help="REQUIRED (except --selftest)")
    ap.add_argument("--price", action="append", default=[],
                    help="NAME=IN,OUT USD per Mtok for a deployment not yet in PRICES")
    ap.add_argument("--tag", default=None, help="default: MODERATOR_CELLS tag or "
                                                f"{TAG_PREFIX}<moderator>")
    ap.add_argument("--allow-fresh-r0r2", action="store_true",
                    help="generate r0-r2 for debates with no cached chain (NOT the registered cell)")
    ap.add_argument("--n-yta", type=int, default=rcd.PILOT_N_YTA)
    ap.add_argument("--n-nta", type=int, default=rcd.PILOT_N_NTA)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--min-votes", type=int, default=50)
    ap.add_argument("--min-consensus", type=float, default=0.90)
    ap.add_argument("--max-tokens-agent", type=int, default=2560)
    ap.add_argument("--max-tokens-moderator", type=int, default=1024)
    ap.add_argument("--max-tokens-label", type=int, default=1024)
    ap.add_argument("--max-tokens-vote", type=int, default=512)
    ap.add_argument("--transcript-cap", type=int, default=REGISTERED_TRANSCRIPT_CAP,
                    help=f"accepted only as {REGISTERED_TRANSCRIPT_CAP} (the registered cap); any other "
                         "value is refused, see BANNED FLAGS in the module docstring")
    ap.add_argument("--keep-filtered", action="store_true")
    ap.add_argument("--no-bpe", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    known, rest = ap.parse_known_args(argv)

    if known.selftest:
        return _selftest()

    for a in rest:
        if a.split("=", 1)[0] in BANNED_FLAGS:
            print(f"\nERROR: {a} would change the seats, the scaffold or the r3/r4 prompt "
                  "text; the cell is defined on the registered transcripts. Refusing.\n")
            return 4
    if known.transcript_cap != REGISTERED_TRANSCRIPT_CAP:
        print(f"\nERROR: --transcript-cap {known.transcript_cap} would change the r1/r2 prompt text "
              "without changing the cache key or the r0 parent_sha, so the registered uncapped "
              f"r0-r2 records would be replayed for a prompt they were not generated under; the "
              f"cell is defined at the registered cap {REGISTERED_TRANSCRIPT_CAP}. Refusing.\n")
        return 4
    models = [m.strip() for m in known.models.split(",") if m.strip()]
    if len(models) != 1:
        print(f"\nERROR: exactly one agent model per invocation (got {models}); the "
              "moderator cache id is bound to it.\n")
        return 2
    agent = models[0]
    mod = known.moderator_model
    if not mod:
        print("\nERROR: --moderator-model is required.\n")
        return 2
    if mod == agent:
        print(f"\nERROR: moderator == agent model ({mod}); that is the self-moderated "
              "registered run (cg_deliberation). Refusing.\n")
        return 2
    tag = known.tag or default_tag(mod)
    if tag in FOREIGN_TAGS or not tag.startswith(TAG_PREFIX):
        print(f"\nERROR: --tag {tag!r} must start with {TAG_PREFIX!r} and must not be "
              f"another cell's tag {sorted(FOREIGN_TAGS)}.\n")
        return 2
    prices = parse_prices(known.price)
    out_dir = rcd.OUT_DIR

    if known.readout:
        return readout(tag, mod, agent, draws=known.n_boot, seed=29)

    clean, counts = cache_is_clean(agent, mod, out_dir=out_dir)
    if known.verify_cache_clean:
        print(f"cache files per namespace for moderator={mod} over {agent}: {counts}")
        print("CLEAN -- every moderator / r3 / r4 call will be a genuine generation."
              if clean else "DIRTY -- refusing without --resume")
        return 0 if clean else 3

    items, panel, dropped = build_panel(agent, n_yta=known.n_yta, n_nta=known.n_nta,
                                        min_votes=known.min_votes,
                                        min_consensus=known.min_consensus, seed=known.seed,
                                        keep_filtered=known.keep_filtered)
    arms = list(rcd.PILOT_ARMS)
    print(f"panel {panel} items -> {len(items)} after the {agent} content-filter screen "
          f"(dropped {dropped}); arms {arms}; samples {known.samples}")
    print(f"agents {agent}   moderator {mod}   tag {tag}")
    print(f"cache ids: moderator {moderator_cache_id(agent)}   r3/r4 seats "
          f"{[seat_cache_id(r.role_id, mod) for r in _PRISTINE['ROLES']]}")
    plan = replay_plan(agent, arms, items, known.samples, scaffold=DEFAULT_SCAFFOLD,
                       max_tokens_agent=known.max_tokens_agent, out_dir=out_dir)
    cm = cost_model_new(agent, mod, arms, items, known.samples, plan, prices=prices,
                        scaffold=DEFAULT_SCAFFOLD, transcript_cap=known.transcript_cap,
                        max_tokens_agent=known.max_tokens_agent,
                        max_tokens_label=known.max_tokens_label,
                        max_tokens_vote=known.max_tokens_vote, out_dir=out_dir,
                        use_bpe=not known.no_bpe)
    print_cost_model(cm)
    cmd = launch_command(agent, mod, tag, n_yta=known.n_yta, n_nta=known.n_nta,
                         samples=known.samples, seed=known.seed)
    print(f"\nlaunch command:\n  {cmd}")
    if plan["n_incomplete"]:
        print(f"\n  NOTE: {plan['n_incomplete']} debate(s) have no complete r0-r2 chain on disk "
              f"({plan['n_stale']} stale). --run refuses them unless --allow-fresh-r0r2.")
    dry_path = out_dir / f"{tag}_dryrun.json"
    cm_out = dict(cm)
    cm_out["incomplete_debates"] = [
        {"arm": d["arm"], "item_id": d["item"].item_id, "idx": d["idx"], "status": d["status"]}
        for d in plan["debates"] if not d["full_chain"]]
    cm_out["launch_command"] = cmd
    dry_path.write_text(json.dumps(cm_out, indent=2))
    print(f"wrote {dry_path}")
    if known.dry_run:
        return 0
    if not known.run:
        print("\n*** NOT RUNNING: pass --run to generate (this spends money). "
              "Nothing was generated. ***\n")
        return 2

    # ---- --run: the checks, then delegate -----------------------------------
    if not cm["priced"]:
        print(f"\nERROR: model not in PRICES ({agent!r} / {mod!r}); add a price to "
              "run_crowdgold_deliberation.PRICES or pass --price NAME=IN,OUT before "
              "spending (the $ above is a placeholder, not a ceiling).\n")
        return 2
    if plan["n_incomplete"] and not known.allow_fresh_r0r2:
        print(f"\nERROR: {plan['n_incomplete']} debate(s) lack a cached r0-r2 chain; generating "
              "them fresh would make this cell's transcripts differ from the registered "
              "cell's on those items. Pass --allow-fresh-r0r2 to accept that.\n")
        return 2
    on_disk = existing_rows_moderators(out_dir / f"{tag}_rows.csv")
    if on_disk and on_disk != {mod}:
        print(f"\nERROR: {tag}_rows.csv already holds rows moderated by {sorted(on_disk)}; "
              f"refusing to overwrite them with a {mod} run. Pick another --tag.\n")
        return 2
    if not clean and not known.resume:
        print(f"\nERROR: cache is not clean for this cell {counts}.\n"
              "A previous partial run would be replayed. Pass --resume only if you are "
              "certain nothing about the cell has changed.\n")
        return 3
    if not clean:
        print(f"  --resume: {counts} existing records will be served from cache")

    install(agent, mod)
    fwd = ["--models", agent, "--moderator-model", mod,
           "--n-yta", str(known.n_yta), "--n-nta", str(known.n_nta),
           "--samples", str(known.samples), "--seed", str(known.seed),
           "--min-votes", str(known.min_votes), "--min-consensus", str(known.min_consensus),
           "--max-tokens-agent", str(known.max_tokens_agent),
           "--max-tokens-moderator", str(known.max_tokens_moderator),
           "--max-tokens-label", str(known.max_tokens_label),
           "--max-tokens-vote", str(known.max_tokens_vote),
           "--transcript-cap", str(known.transcript_cap),
           "--n-boot", str(known.n_boot), "--tag", tag]
    if known.keep_filtered:
        fwd.append("--keep-filtered")
    fwd += rest
    print(f"[modvendor] {plan['replay_calls']} calls replay from the registered cache; "
          f"{plan['new_calls']} NEW generation calls will be made")
    print(f"[modvendor] delegating to run_crowdgold_deliberation with: {' '.join(fwd)}")
    rc = rcd.main(fwd)
    if rc in (0, 1, 4):
        sn = None
        sp = out_dir / f"{tag}_summary.json"
        if sp.exists():
            try:
                sn = json.loads(sp.read_text()).get("n_items")
            except json.JSONDecodeError:
                sn = None
        comp = completeness_check(tag, agent, {(arm, it.item_id, idx) for arm in arms for it in items
                                               for idx in range(known.samples)},
                                  out_dir=out_dir, summary_n_items=sn)
        print_completeness(comp)
        (out_dir / f"{tag}_completeness.json").write_text(json.dumps(comp, indent=2))
        print(f"  wrote {out_dir / (tag + '_completeness.json')}")
        if not comp["pass"]:
            rc = 4
    if rc == 4:
        # the base runner writes its CSVs before its guard verdict; keep failed
        # numbers out of the tag a --readout would read
        for kind in ("rows", "votes"):
            p = out_dir / f"{tag}_{kind}.csv"
            if p.exists():
                p.rename(out_dir / f"{tag}_GUARD_FAILED_{kind}.csv")
        print(f"\n*** GUARD FAILED -- CSVs moved to {tag}_GUARD_FAILED_*.csv; fix the cause "
              "(a failing round's cap, or the errored debates) and re-run with --resume. ***\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
