"""
scripts/run_cross_vendor_moderator.py -- Re-run Experiment 2 moderator steps
(open-synthesis, R3 synthesis acceptance, R4 binary vote) with claude-sonnet-4-6
as the moderator over the existing gpt-5.4-nano agent traces.

This directly addresses reviewer W5 (within-vendor moderator confound): the
original 95% full-consensus result used gpt-4o-mini as moderator with
gpt-5.4-nano agents. Swapping to a cross-vendor moderator (Anthropic's
claude-sonnet-4-6) while reusing the cached agent R2 final-position texts
tests whether the consensus rate is an artefact of within-vendor familiarity.

Cache scheme:
  debate_xmod_sonnet_openmod_{sid}_{idx:03d}.json       -- open moderator pass
  debate_xmod_sonnet_r3_{sid}_{pid}_{idx:03d}.json      -- R3 agent synthesis response
  debate_xmod_sonnet_r3dec_{sid}_{pid}_{idx:03d}.json   -- R3 decision extraction
  debate_xmod_sonnet_r3final_{sid}_{idx:03d}.json       -- R3 final moderator pass
  debate_xmod_sonnet_r4integ_{sid}_{idx:03d}.json       -- R4 integration proposal
  debate_xmod_sonnet_r4vote_{sid}_{pid}_{idx:03d}.json  -- R4 agent binary vote
  debate_xmod_sonnet_r4final_{sid}_{idx:03d}.json       -- R4 final vote tally

Output:
  divergence_study_outputs/debate_cross_vendor_moderator_sonnet.csv

Usage (from repo root):
  python -m scripts.run_cross_vendor_moderator [--workers N]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generators import generate as _generate

# Import shared debate data (scenarios, perspectives, prompts)
from run_debate_std_cot_ablation import (
    SCENARIOS, SCENARIO_BY_ID, PERSPECTIVES_BY_SCENARIO,
    MP_STD_COT_SYSTEM,
    OPEN_MODERATOR_SYSTEM, OPEN_MODERATOR_USER_TEMPLATE,
    SYNTH_R3_EXTRACT_SYSTEM, SYNTH_R3_EXTRACT_USER_TEMPLATE,
    SYNTH_FINAL_MOD_SYSTEM, SYNTH_FINAL_MOD_USER_TEMPLATE,
    INTEG_MOD_SYSTEM, INTEG_MOD_USER_TEMPLATE,
    R4_FINAL_MOD_SYSTEM, R4_FINAL_MOD_USER_TEMPLATE,
    Scenario, PerspectiveGeneration,
    _safe,
)

# The original synthesis-acceptance and R4-vote agent prompts
SYNTH_ACCEPT_SYSTEM = MP_STD_COT_SYSTEM

SYNTH_ACCEPT_USER_TEMPLATE = (
    "{scenario}\n\n"
    "You are {perspective_description}.\n\n"
    "The moderator has reviewed the full exchange and identified the following synthesis "
    "position that attempts to address concerns raised across all perspectives:\n\n"
    "**SYNTHESIS: {synthesis_label}**\n"
    "{synthesis_description}\n\n"
    "As {perspective_description}, respond to this synthesis. Your response must:\n"
    "1. State whether you ACCEPT, ACCEPT_WITH_MODIFICATION, or REJECT this synthesis.\n"
    "2. If ACCEPT: briefly explain why this synthesis addresses your core concerns.\n"
    "3. If ACCEPT_WITH_MODIFICATION: state exactly what modification you require and why, "
    "then describe the modified synthesis in one sentence.\n"
    "4. If REJECT: explain which of your core concerns the synthesis fails to address "
    "and what you believe is unresolvable.\n\n"
    "Begin your response with exactly one of: ACCEPT / ACCEPT_WITH_MODIFICATION / REJECT"
)

R4_VOTE_USER_TEMPLATE = (
    "{scenario}\n\n"
    "You are {perspective_description}.\n\n"
    "The debate has gone through four rounds:\n"
    "  R0: opening positions\n"
    "  R1: rebuttals\n"
    "  R2: final positions (open action space)\n"
    "  R3: you and the other stakeholders each responded to the moderator's synthesis\n\n"
    "The mediator has now read all responses and constructed a single integrated "
    "proposal that attempts to satisfy as many concerns as possible:\n\n"
    "**INTEGRATED PROPOSAL: {integrated_label}**\n"
    "{integrated_description}\n\n"
    "The mediator notes: {integration_rationale}\n\n"
    "This is the final round. You must respond with exactly one word on the first line: "
    "ACCEPT or REJECT.\n"
    "Then in 2-3 sentences explain your decision. If you REJECT, state the single most "
    "important unresolvable concern. If you ACCEPT, confirm which of your core concerns "
    "the integrated proposal addresses."
)

GEN_MODEL = "gpt-5.4-nano"
MOD_MODEL = "claude-sonnet-4-6"
N_SAMPLES = 10
CACHE_PFX = "debate_xmod_sonnet"

OUT_DIR = Path("./divergence_study_outputs")
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Extract the first JSON object from text (handles markdown fences)."""
    if not text:
        return {}
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return {}


def _call_mod(system: str, user: str, *, seed: int, max_tokens: int = 2048) -> dict:
    """Call the cross-vendor moderator (claude-sonnet-4-6) and parse JSON."""
    result = _generate(
        MOD_MODEL, system, user,
        sample_idx=seed,
        max_tokens=max_tokens,
    )
    if not result.text:
        return {"content_filtered": True}
    return _extract_json(result.text) or {}


def _call_agent(system: str, user: str, *, seed: int, max_tokens: int = 2000) -> str:
    """Call the generator agent (gpt-5.4-nano) for R3/R4 responses."""
    result = _generate(
        GEN_MODEL, system, user,
        sample_idx=seed,
        max_tokens=max_tokens,
    )
    return result.text


# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------

def _cp(name: str, sid: str, idx: int, pid: str = "") -> Path:
    if pid:
        return OUT_DIR / f"{CACHE_PFX}_{name}_{sid}_{pid}_{idx:03d}.json"
    return OUT_DIR / f"{CACHE_PFX}_{name}_{sid}_{idx:03d}.json"


def _seed(tag: str, sid: str, idx: int) -> int:
    return int(hashlib.sha1(f"{tag}|{sid}|{idx}".encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Load existing R2 texts (from the original v2 debate cache)
# ---------------------------------------------------------------------------

def _load_r1_text(sid: str, pid: str, idx: int) -> str:
    """Load gpt-5.4-nano R1 rebuttal text from the original debate cache."""
    p = OUT_DIR / f"debate_round1_{_safe(GEN_MODEL)}_{sid}_{pid}_{idx:03d}.json"
    if p.exists():
        try:
            return json.loads(p.read_text()).get("output", "[R1 not cached]")
        except Exception:
            pass
    return "[R1 not cached]"


def _load_r2_text(sid: str, pid: str, idx: int) -> str:
    """Load gpt-5.4-nano R2 final-position text from the v2 debate cache."""
    p = OUT_DIR / f"debate_v2_round2_{_safe(GEN_MODEL)}_{sid}_{pid}_{idx:03d}.json"
    if p.exists():
        try:
            return json.loads(p.read_text()).get("output", "[R2 not cached]")
        except Exception:
            pass
    return "[R2 not cached]"


# ---------------------------------------------------------------------------
# Step 1: Open moderator pass (sonnet reads debate transcript, proposes synthesis)
# ---------------------------------------------------------------------------

def run_open_mod(s: Scenario, idx: int) -> dict:
    cp = _cp("openmod", s.id, idx)
    if cp.exists():
        try:
            return json.loads(cp.read_text())
        except Exception:
            cp.unlink()

    ps = PERSPECTIVES_BY_SCENARIO[s.id]
    parts = []
    for p in ps:
        lbl = p.perspective_id.replace("_", " ").title()
        r1 = _load_r1_text(s.id, p.perspective_id, idx)
        r2 = _load_r2_text(s.id, p.perspective_id, idx)
        parts.append(
            f"=== {lbl} ===\n"
            f"[R1 (rebuttal)]:\n{r1[:600]}\n\n"
            f"[R2 (final position)]:\n{r2[:800]}"
        )
    transcript = "\n\n".join(parts)
    taxonomy_labels = ", ".join(f'"{k}"' for k in s.decision_taxonomy)
    user = OPEN_MODERATOR_USER_TEMPLATE.format(
        scenario=s.prompt,
        transcript=transcript,
        taxonomy_labels=taxonomy_labels,
    )
    result = _call_mod(OPEN_MODERATOR_SYSTEM, user, seed=_seed("openmod", s.id, idx))
    for k, dv in [
        ("consensus_reached", False), ("consensus_kind", "none"),
        ("consensus_decision", None), ("novel_actions_proposed", []),
        ("synthesis_label", None), ("synthesis_description", None),
        ("points_of_disagreement", []), ("summary", ""),
    ]:
        result.setdefault(k, dv)
    result["scenario_id"] = s.id
    result["sample_idx"] = idx
    result["moderator"] = MOD_MODEL
    cp.write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Step 2: R3 agent responses to synthesis (gpt-5.4-nano agents respond)
# ---------------------------------------------------------------------------

def gen_r3(s: Scenario, pid: str, idx: int, synth_label: str, synth_desc: str) -> str:
    cp = _cp("r3", s.id, idx, pid)
    if cp.exists():
        try:
            return json.loads(cp.read_text()).get("output", "")
        except Exception:
            cp.unlink()

    ps = PERSPECTIVES_BY_SCENARIO[s.id]
    p = next((pp for pp in ps if pp.perspective_id == pid), None)
    if p is None:
        return ""

    user = SYNTH_ACCEPT_USER_TEMPLATE.format(
        scenario=s.prompt,
        perspective_description=p.description,
        synthesis_label=synth_label,
        synthesis_description=synth_desc,
    )
    output = _call_agent(SYNTH_ACCEPT_SYSTEM, user, seed=idx + 4000)
    cp.write_text(json.dumps({"scenario_id": s.id, "perspective_id": pid,
                               "sample_idx": idx, "output": output}, indent=2))
    return output


# ---------------------------------------------------------------------------
# Step 3: R3 decision extraction (sonnet extracts structured decision)
# ---------------------------------------------------------------------------

def extract_r3_decision(s: Scenario, pid: str, idx: int,
                        synth_label: str, synth_desc: str, r3_text: str) -> dict:
    cp = _cp("r3dec", s.id, idx, pid)
    if cp.exists():
        try:
            return json.loads(cp.read_text())
        except Exception:
            cp.unlink()

    if not r3_text or not r3_text.strip():
        result = {"response_type": "UNKNOWN", "decision": "TRUNCATED",
                  "modification_description": None, "rejection_reason": None,
                  "decision_confidence": 0.0}
        cp.write_text(json.dumps(result, indent=2))
        return result

    taxonomy_text = "\n".join(f"- `{lbl}`: {desc}" for lbl, desc in s.decision_taxonomy.items())
    user = SYNTH_R3_EXTRACT_USER_TEMPLATE.format(
        scenario=s.prompt,
        synthesis_label=synth_label,
        synthesis_description=synth_desc,
        taxonomy=taxonomy_text,
        response=r3_text[:1200],
    )
    result = _call_mod(SYNTH_R3_EXTRACT_SYSTEM, user,
                       seed=_seed("r3dec", s.id + pid, idx), max_tokens=400)
    for k, dv in [
        ("response_type", "UNKNOWN"), ("decision", "NO_COMMITMENT"),
        ("modification_description", None), ("rejection_reason", None),
        ("decision_confidence", 0.5),
    ]:
        result.setdefault(k, dv)
    result["perspective_id"] = pid
    cp.write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Step 4: R3 final moderator pass (sonnet assesses synthesis round outcome)
# ---------------------------------------------------------------------------

def run_r3_final(s: Scenario, idx: int, synth_label: str, synth_desc: str,
                 r3_decisions: dict) -> dict:
    cp = _cp("r3final", s.id, idx)
    if cp.exists():
        try:
            return json.loads(cp.read_text())
        except Exception:
            cp.unlink()

    ps = PERSPECTIVES_BY_SCENARIO[s.id]
    parts = []
    for p in ps:
        lbl = p.perspective_id.replace("_", " ").title()
        dec = r3_decisions.get(p.perspective_id, {})
        rtype = dec.get("response_type", "UNKNOWN")
        mod_desc = dec.get("modification_description", "")
        rej_reason = dec.get("rejection_reason", "")
        if rtype == "ACCEPT":
            parts.append(f"[{lbl}]: ACCEPT")
        elif rtype == "ACCEPT_WITH_MODIFICATION":
            parts.append(f"[{lbl}]: ACCEPT_WITH_MODIFICATION -- {mod_desc}")
        else:
            parts.append(f"[{lbl}]: REJECT -- {rej_reason}")
    responses_text = "\n\n".join(parts)

    user = SYNTH_FINAL_MOD_USER_TEMPLATE.format(
        scenario=s.prompt,
        synthesis_label=synth_label,
        synthesis_description=synth_desc,
        responses=responses_text,
    )
    result = _call_mod(SYNTH_FINAL_MOD_SYSTEM, user,
                       seed=_seed("r3final", s.id, idx), max_tokens=600)
    for k, dv in [
        ("consensus_reached", False), ("partial_convergence", False),
        ("unanimous_accept", False), ("accept_count", 0), ("modify_count", 0),
        ("reject_count", 0), ("final_synthesis_label", None),
        ("final_synthesis_description", None), ("remaining_disagreements", []),
    ]:
        result.setdefault(k, dv)
    result["scenario_id"] = s.id
    result["sample_idx"] = idx
    result["synthesis_label"] = synth_label
    cp.write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Step 5: Integration pass (sonnet constructs integrated proposal from modifications)
# ---------------------------------------------------------------------------

def run_integration(s: Scenario, idx: int, synth_label: str, synth_desc: str,
                    r3_decisions: dict) -> dict:
    cp = _cp("r4integ", s.id, idx)
    if cp.exists():
        try:
            return json.loads(cp.read_text())
        except Exception:
            cp.unlink()

    ps = PERSPECTIVES_BY_SCENARIO[s.id]
    mod_parts = []
    for p in ps:
        lbl = p.perspective_id.replace("_", " ").title()
        dec = r3_decisions.get(p.perspective_id, {})
        rtype = dec.get("response_type", "UNKNOWN")
        mod_desc = dec.get("modification_description") or ""
        if rtype in ("ACCEPT", "ACCEPT_WITH_MODIFICATION"):
            if mod_desc:
                mod_parts.append(f"[{lbl}]: {mod_desc}")
            else:
                mod_parts.append(f"[{lbl}]: No modification required (pure accept)")
    if not mod_parts:
        mod_parts = [f"[All]: Accepted synthesis as presented"]

    user = INTEG_MOD_USER_TEMPLATE.format(
        scenario=s.prompt,
        synthesis_label=synth_label,
        synthesis_description=synth_desc,
        modification_requests="\n\n".join(mod_parts),
    )
    result = _call_mod(INTEG_MOD_SYSTEM, user,
                       seed=_seed("r4integ", s.id, idx), max_tokens=600)
    for k, dv in [
        ("integrated_label", synth_label), ("integrated_description", synth_desc),
        ("modifications_addressed", []), ("modifications_unaddressed", []),
        ("integration_rationale", "Integration carried forward from synthesis."),
    ]:
        result.setdefault(k, dv)
    result["scenario_id"] = s.id
    result["sample_idx"] = idx
    cp.write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Step 6: R4 agent binary votes (gpt-5.4-nano votes on integrated proposal)
# ---------------------------------------------------------------------------

def gen_r4_vote(s: Scenario, pid: str, idx: int,
                integ_label: str, integ_desc: str, integ_rationale: str) -> str:
    cp = _cp("r4vote", s.id, idx, pid)
    if cp.exists():
        try:
            return json.loads(cp.read_text()).get("output", "")
        except Exception:
            cp.unlink()

    ps = PERSPECTIVES_BY_SCENARIO[s.id]
    p = next((pp for pp in ps if pp.perspective_id == pid), None)
    if p is None:
        return ""

    user = R4_VOTE_USER_TEMPLATE.format(
        scenario=s.prompt,
        perspective_description=p.description,
        integrated_label=integ_label,
        integrated_description=integ_desc,
        integration_rationale=integ_rationale,
    )
    output = _call_agent(SYNTH_ACCEPT_SYSTEM, user, seed=idx + 5000)
    cp.write_text(json.dumps({"scenario_id": s.id, "perspective_id": pid,
                               "sample_idx": idx, "output": output}, indent=2))
    return output


# ---------------------------------------------------------------------------
# Step 7: R4 final moderator tally (sonnet counts votes)
# ---------------------------------------------------------------------------

def run_r4_final(s: Scenario, idx: int, integ_label: str, integ_desc: str,
                 r4_votes: dict) -> dict:
    cp = _cp("r4final", s.id, idx)
    if cp.exists():
        try:
            return json.loads(cp.read_text())
        except Exception:
            cp.unlink()

    ps = PERSPECTIVES_BY_SCENARIO[s.id]
    parts = []
    for p in ps:
        lbl = p.perspective_id.replace("_", " ").title()
        vote_text = r4_votes.get(p.perspective_id, "[No vote]")
        parts.append(f"[{lbl}]:\n{(vote_text or '[empty]')[:400]}")

    user = R4_FINAL_MOD_USER_TEMPLATE.format(
        scenario=s.prompt,
        integrated_label=integ_label,
        integrated_description=integ_desc,
        votes="\n\n".join(parts),
    )
    result = _call_mod(R4_FINAL_MOD_SYSTEM, user,
                       seed=_seed("r4final", s.id, idx), max_tokens=400)
    for k, dv in [
        ("consensus_reached", False), ("majority_accept", False),
        ("accept_count", 0), ("reject_count", 0), ("unanimous", False),
        ("final_position_label", None), ("primary_rejection_reason", None),
    ]:
        result.setdefault(k, dv)
    result["scenario_id"] = s.id
    result["sample_idx"] = idx
    result["integrated_label"] = integ_label
    result["moderator"] = MOD_MODEL
    cp.write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Orchestration: one debate
# ---------------------------------------------------------------------------

def run_debate(s: Scenario, idx: int) -> dict:
    """Run one full 4-round debate with cross-vendor moderator."""
    ps = PERSPECTIVES_BY_SCENARIO.get(s.id, [])
    base = {
        "scenario_id": s.id,
        "sample_idx": idx,
        "gen_model": GEN_MODEL,
        "moderator": MOD_MODEL,
        "consensus_reached": False,
        "majority_accept": False,
        "accept_count": 0,
        "reject_count": 0,
        "unanimous": False,
        "final_position_label": None,
        "primary_rejection_reason": None,
        "synthesis_label": None,
        "openmod_consensus_kind": "none",
        "pipeline_short_circuit": None,
    }

    # Step 1: Open mod
    omod = run_open_mod(s, idx)
    base["openmod_consensus_kind"] = omod.get("consensus_kind", "none")
    base["synthesis_label"] = omod.get("synthesis_label")

    # Short-circuit: if the agents already converged in R2, record full consensus
    if omod.get("consensus_reached") and omod.get("consensus_kind") in ("taxonomy", "novel_action"):
        base["consensus_reached"] = True
        base["majority_accept"] = True
        base["accept_count"] = len(ps)
        base["unanimous"] = True
        base["final_position_label"] = omod.get("consensus_decision") or omod.get("synthesis_label")
        base["pipeline_short_circuit"] = "r2_convergence"
        return base

    synth_label = omod.get("synthesis_label")
    synth_desc = omod.get("synthesis_description", "")

    if not synth_label:
        # No synthesis was proposed -- treat as structural rejection
        base["pipeline_short_circuit"] = "no_synthesis"
        return base

    # Step 2+3: Agents respond to synthesis, extract decisions
    r3_decisions: dict[str, dict] = {}
    for p in ps:
        r3_text = gen_r3(s, p.perspective_id, idx, synth_label, synth_desc)
        r3_dec = extract_r3_decision(s, p.perspective_id, idx,
                                      synth_label, synth_desc, r3_text)
        r3_decisions[p.perspective_id] = r3_dec

    # Step 4: R3 final mod
    r3_final = run_r3_final(s, idx, synth_label, synth_desc, r3_decisions)

    # If anyone outright rejected at R3, no integration needed
    if r3_final.get("reject_count", 0) == len(ps):
        base["pipeline_short_circuit"] = "r3_unanimous_reject"
        return base

    # Step 5: Integration
    integ = run_integration(s, idx, synth_label, synth_desc, r3_decisions)
    integ_label = integ.get("integrated_label", synth_label)
    integ_desc = integ.get("integrated_description", synth_desc)
    integ_rationale = integ.get("integration_rationale", "")

    # Step 6+7: R4 binary votes
    r4_votes: dict[str, str] = {}
    for p in ps:
        r4_votes[p.perspective_id] = gen_r4_vote(
            s, p.perspective_id, idx, integ_label, integ_desc, integ_rationale
        )

    r4_final = run_r4_final(s, idx, integ_label, integ_desc, r4_votes)

    base.update({
        "consensus_reached": r4_final.get("consensus_reached", False),
        "majority_accept": r4_final.get("majority_accept", False),
        "accept_count": r4_final.get("accept_count", 0),
        "reject_count": r4_final.get("reject_count", 0),
        "unanimous": r4_final.get("unanimous", False),
        "final_position_label": r4_final.get("final_position_label"),
        "primary_rejection_reason": r4_final.get("primary_rejection_reason"),
    })
    return base


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-vendor moderator cell (W5)")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--n", type=int, default=N_SAMPLES)
    args = parser.parse_args(argv)

    tasks = [(s, idx) for s in SCENARIOS for idx in range(args.n)]
    print(f"Cross-vendor moderator: {MOD_MODEL} over {GEN_MODEL} agents")
    print(f"Tasks: {len(SCENARIOS)} scenarios x {args.n} samples = {len(tasks)} debates")
    print(f"Workers: {args.workers}")

    rows: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_debate, s, idx): (s.id, idx) for s, idx in tasks}
        for fut in as_completed(futs):
            sid, idx = futs[fut]
            try:
                row = fut.result()
                rows.append(row)
            except Exception as e:
                print(f"  ERROR {sid}[{idx}]: {e}", flush=True)
                rows.append({
                    "scenario_id": sid, "sample_idx": idx,
                    "gen_model": GEN_MODEL, "moderator": MOD_MODEL,
                    "consensus_reached": False, "error": str(e),
                })
            done += 1
            if done % 5 == 0:
                print(f"  {done}/{len(tasks)}", flush=True)

    out = OUT_DIR / "debate_cross_vendor_moderator_sonnet.csv"
    cols = [
        "scenario_id", "sample_idx", "gen_model", "moderator",
        "consensus_reached", "majority_accept", "accept_count", "reject_count",
        "unanimous", "final_position_label", "primary_rejection_reason",
        "synthesis_label", "openmod_consensus_kind", "pipeline_short_circuit",
    ]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {out} ({len(rows)} rows)")

    # Headline summary
    import pandas as pd
    df = pd.DataFrame(rows)
    total = len(df)
    n_consensus = df["consensus_reached"].sum() if "consensus_reached" in df.columns else 0
    print(f"\nFull consensus: {n_consensus}/{total} ({100*n_consensus/total:.1f}%)")
    print("\nBy scenario:")
    print(df.groupby("scenario_id")["consensus_reached"].agg(["sum", "count"]).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
