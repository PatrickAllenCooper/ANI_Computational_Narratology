"""
scripts/run_phase5_e2_scaled.py -- E2: Scaled multi-agent debate on 30
DailyDilemmas scenarios with the full R0-R4 integration protocol.

Replaces the 5 hand-crafted calibration scenarios with a 30-scenario
stratified subsample of DailyDilemmas (SCALED_SAMPLE_SEED+1 = 43 to avoid
overlap with E1's 100-scenario pool -- the subsample is drawn as a shuffle
of the full 100-scenario pool and the first 30 taken; since E1 uses the same
100 scenarios, there IS overlap on the dilemmas themselves, but the task
here is the multi-agent debate, which is a different experiment).

Two generators: gpt-5.4-nano (cheap bulk) + claude-sonnet-4-6 (confirmation).
N-CoT arm only.  Hard budget cap enforced by --budget-cap-usd.

R0-R4 pipeline mirrors run_debate_std_cot_ablation.py / run_phase3_debate.py:
  R0: opening statement (N-CoT, 3 perspectives)
  R1: rebuttal (N-CoT)
  R2: final open position (N-CoT, open action space)
  R3: synthesis acceptance vote (each perspective)
  R4: binary accept/reject on integrated proposal

Perspectives: 3 generic deliberation roles (stakeholder_a, stakeholder_b,
stakeholder_c) created fresh for each DailyDilemmas scenario.

Falsification criterion: full-consensus rate (unanimous R4 ACCEPT) must be
within +/- 10 percentage points of 95%.  Outside this band the headline claim
downgrades to "calibration-set result, DailyDilemmas replication failed".

Hard budget cap: tracks estimated cost (token counts x per-model rates) and
pauses before exceeding --budget-cap-usd.  Checkpoint JSON is written every
N scenarios so the run can be resumed.

Usage:
  python -m scripts.run_phase5_e2_scaled [--scenarios 30] [--samples 1]
         [--generators gpt-5.4-nano,claude-sonnet-4-6]
         [--workers 2] [--budget-cap-usd 110]

Required env vars: same as run_phase1_quartet.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
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

from scripts.generators import generate as _generate, GenerationResult
from scripts.run_phase1_quartet import (
    load_daily_dilemmas, Scenario as DD_Scenario,
    PROMPTS, SCALED_SAMPLE_SEED, SUBSAMPLE_SEED,
)

OUT_DIR = Path("./divergence_study_outputs")
OUT_DIR.mkdir(exist_ok=True)

E2_SAMPLE_SEED = SUBSAMPLE_SEED  # = 43

# ---------------------------------------------------------------------------
# Cost estimation (approximate; adjust if actual rates differ)
# ---------------------------------------------------------------------------

# USD per 1M tokens (input / output) -- conservative estimates
_COST_TABLE: dict[str, tuple[float, float]] = {
    "gpt-5.4-nano":       (0.15,   0.60),
    "claude-sonnet-4-6":  (3.00,  15.00),
    "claude-haiku-4-5":   (0.25,   1.25),
    "grok-4-1-fast-reasoning": (2.00, 10.00),
    "default":            (1.00,   5.00),
}

_total_cost_usd: float = 0.0


def _accrue_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    global _total_cost_usd
    in_rate, out_rate = _COST_TABLE.get(model, _COST_TABLE["default"])
    cost = (prompt_tokens / 1e6) * in_rate + (completion_tokens / 1e6) * out_rate
    _total_cost_usd += cost
    return cost


def _over_budget(cap: float) -> bool:
    return _total_cost_usd >= cap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def _extract_json(text: str) -> dict:
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


def _call_gen(model: str, system: str, user: str, seed: int,
              max_tokens: int = 4096) -> str:
    result = _generate(model, system, user, sample_idx=seed, max_tokens=max_tokens)
    _accrue_cost(model, result.prompt_tokens, result.completion_tokens)
    return result.text


def _call_mod(model: str, system: str, user: str, seed: int,
              max_tokens: int = 2048) -> dict:
    result = _generate(model, system, user, sample_idx=seed,
                       max_tokens=max_tokens, json_mode=True)
    _accrue_cost(model, result.prompt_tokens, result.completion_tokens)
    if not result.text:
        return {}
    return _extract_json(result.text) or {}


# ---------------------------------------------------------------------------
# Perspective templates -- generic roles for DailyDilemmas
# ---------------------------------------------------------------------------

NCOT_SYSTEM = PROMPTS["narrative_cot"]

# Three generic deliberation perspectives; each gets scenario-specific framing
PERSPECTIVE_ROLES = [
    {
        "id": "stakeholder_a",
        "label": "Primary Decision Maker",
        "framing": (
            "You are the primary decision maker -- the person who must make the choice "
            "and will live with the consequences most directly."
        ),
    },
    {
        "id": "stakeholder_b",
        "label": "Affected Third Party",
        "framing": (
            "You are an affected third party -- someone whose life and wellbeing "
            "will be significantly impacted by whatever choice the decision maker makes."
        ),
    },
    {
        "id": "stakeholder_c",
        "label": "External Advisor",
        "framing": (
            "You are an external advisor -- someone with relevant expertise or "
            "perspective who is not directly involved but is asked to give a considered "
            "recommendation."
        ),
    },
]

PERSPECTIVE_IDS = [r["id"] for r in PERSPECTIVE_ROLES]
PERSPECTIVE_BY_ID = {r["id"]: r for r in PERSPECTIVE_ROLES}

# ---------------------------------------------------------------------------
# Moderator prompts (adapted from run_debate_std_cot_ablation.py)
# ---------------------------------------------------------------------------

OPEN_MOD_SYSTEM = (
    "You are a neutral moderator synthesizing a structured multi-stakeholder debate about an ethical "
    "dilemma. In this debate, agents were allowed to propose novel actions beyond the original options. "
    "Your tasks are: (1) identify any novel actions proposed, (2) normalise each to a canonical "
    "snake_case label of 2-5 words, (3) detect whether two or more agents converged on the same "
    "novel action, and (4) if no agent-level convergence emerged but the agents reveal complementary "
    "concerns, formulate a synthesis position that addresses those concerns -- describe it concisely "
    "and give it a canonical snake_case label. Return only valid JSON."
)

OPEN_MOD_USER = (
    "{scenario}\n\n"
    "Original decision options: {taxonomy_labels}.\n\n"
    "Debate transcript (R0=opening, R1=rebuttal, R2=final open position):\n\n"
    "{transcript}\n\n"
    "Return a JSON object with EXACTLY these keys:\n"
    "- \"consensus_reached\": true/false\n"
    "- \"consensus_kind\": \"taxonomy\" | \"novel_action\" | \"synthesis\" | \"none\"\n"
    "- \"consensus_decision\": canonical label or null\n"
    "- \"novel_actions_proposed\": list of {{\"agent\", \"label\", \"description\"}}. Empty if none.\n"
    "- \"synthesis_label\": snake_case label or null\n"
    "- \"synthesis_description\": 1-2 sentence description or null\n"
    "- \"points_of_disagreement\": list of up to 3 short strings\n"
    "- \"summary\": one paragraph synthesizing the outcome"
)

SYNTH_ACCEPT_SYSTEM = (
    "You are a deliberation participant. A neutral moderator has reviewed your debate and proposed "
    "a synthesis position. Respond from your assigned perspective: do you ACCEPT, "
    "ACCEPT_WITH_MODIFICATION, or REJECT this synthesis? Be specific."
)

SYNTH_ACCEPT_USER = (
    "Scenario:\n{scenario}\n\n"
    "Your perspective:\n{perspective_description}\n\n"
    "Debate so far:\n{transcript}\n\n"
    "Proposed synthesis:\n  {synthesis_label}: {synthesis_description}\n\n"
    "Respond from your perspective. State ACCEPT, ACCEPT_WITH_MODIFICATION (with your specific "
    "requested change), or REJECT (with your core unresolvable concern). Work through your "
    "reasoning before giving your verdict."
)

SYNTH_R3_EXTRACT_SYSTEM = (
    "You are a careful extractor. Given an agent response to a moderator synthesis proposal, "
    "extract the response type and final position. Return only valid JSON."
)

SYNTH_R3_EXTRACT_USER = (
    "Scenario:\n{scenario}\n\n"
    "Synthesis presented:\n  {synthesis_label}: {synthesis_description}\n\n"
    "Agent response:\n{response}\n\n"
    "Return JSON with EXACTLY these keys:\n"
    "- \"response_type\": \"ACCEPT\" | \"ACCEPT_WITH_MODIFICATION\" | \"REJECT\" | \"UNKNOWN\"\n"
    "- \"modification_description\": specific change if ACCEPT_WITH_MODIFICATION, else null\n"
    "- \"rejection_reason\": core concern if REJECT, else null\n"
    "- \"decision_confidence\": float 0.0-1.0"
)

SYNTH_FINAL_MOD_SYSTEM = (
    "You are a neutral moderator. Each stakeholder responded to a proposed synthesis. "
    "Assess the outcome: consensus, partial convergence, or continued disagreement. Return only valid JSON."
)

SYNTH_FINAL_MOD_USER = (
    "{scenario}\n\n"
    "Synthesis proposed:\n  {synthesis_label}: {synthesis_description}\n\n"
    "Stakeholder responses:\n{responses}\n\n"
    "Return JSON with EXACTLY these keys:\n"
    "- \"consensus_reached\": true if ALL accepted (with or without modification)\n"
    "- \"partial_convergence\": true if >=2 of 3 accepted\n"
    "- \"unanimous_accept\": true if all 3 accepted WITHOUT modification\n"
    "- \"accept_count\": int\n"
    "- \"modify_count\": int\n"
    "- \"reject_count\": int\n"
    "- \"final_synthesis_label\": updated label if consensus, else null\n"
    "- \"final_synthesis_description\": 1-2 sentence description, or null\n"
    "- \"remaining_disagreements\": list of up to 3 short strings"
)

INTEG_MOD_SYSTEM = (
    "You are a skilled mediator. Three stakeholders each accepted a synthesis but each requested "
    "a specific modification. Construct a single integrated proposal addressing as many modifications "
    "as possible without abandoning the core synthesis direction. Return only valid JSON."
)

INTEG_MOD_USER = (
    "{scenario}\n\n"
    "Original synthesis:\n  {synthesis_label}: {synthesis_description}\n\n"
    "Modification requests:\n{modification_requests}\n\n"
    "Return JSON with EXACTLY these keys:\n"
    "- \"integrated_label\": snake_case 2-5 word label\n"
    "- \"integrated_description\": 2-3 sentence concrete description\n"
    "- \"modifications_addressed\": list of perspective_ids addressed\n"
    "- \"modifications_unaddressed\": list of perspective_ids not addressed\n"
    "- \"integration_rationale\": 1-2 sentences"
)

R4_USER = (
    "Scenario:\n{scenario}\n\n"
    "Your perspective:\n{perspective_description}\n\n"
    "An integrated proposal has been constructed:\n"
    "  {integrated_label}: {integrated_description}\n\n"
    "Rationale: {integration_rationale}\n\n"
    "Cast your final vote. State ACCEPT if this integrated proposal adequately addresses "
    "your core concerns, or REJECT if it does not and explain the unresolvable issue. "
    "Work through your reasoning before giving your final verdict."
)

R4_FINAL_MOD_SYSTEM = (
    "You are a neutral moderator. Each stakeholder cast a final ACCEPT or REJECT vote on "
    "an integrated proposal. Tally and assess. Return only valid JSON."
)

R4_FINAL_MOD_USER = (
    "{scenario}\n\n"
    "Integrated proposal:\n  {integrated_label}: {integrated_description}\n\n"
    "Stakeholder votes:\n{votes}\n\n"
    "Return JSON with EXACTLY these keys:\n"
    "- \"consensus_reached\": true if ALL voted ACCEPT\n"
    "- \"majority_accept\": true if >=2 of 3 voted ACCEPT\n"
    "- \"accept_count\": int\n"
    "- \"reject_count\": int\n"
    "- \"unanimous\": true if all 3 accept\n"
    "- \"final_position_label\": integrated_label if consensus, else null\n"
    "- \"structural_rejection\": true if any REJECT vote (signals non-negotiable concern)\n"
    "- \"primary_rejection_reason\": core concern from any reject, or null"
)

R0_USER = (
    "Scenario:\n{scenario}\n\n"
    "Your perspective:\n{perspective_description}\n\n"
    "State your opening position on this ethical dilemma from your perspective, "
    "working through all five sections of your narrative before giving your final position."
)

R1_USER = (
    "Scenario:\n{scenario}\n\n"
    "Your perspective:\n{perspective_description}\n\n"
    "Your opening statement:\n{own_r0}\n\n"
    "Other perspectives' opening statements:\n{other_r0}\n\n"
    "Now write your rebuttal. Engage directly with the other perspectives' concerns, "
    "update your position if warranted, and clarify your key considerations."
)

R2_USER = (
    "Scenario:\n{scenario}\n\n"
    "Your perspective:\n{perspective_description}\n\n"
    "Your opening statement:\n{own_r0}\n\n"
    "Your rebuttal:\n{own_r1}\n\n"
    "Other perspectives' rebuttals:\n{other_r1}\n\n"
    "Give your final open position. You may propose a novel action beyond the original options "
    "if the debate has revealed a better path. Be concrete and decisive."
)

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_e2(tag: str, model: str, sid: str, pid: str, idx: int) -> Path:
    return OUT_DIR / f"e2_{tag}_{_safe(model)}_{sid}_{pid}_{idx:03d}.json"


def _cache_mod_e2(tag: str, model: str, sid: str, idx: int) -> Path:
    return OUT_DIR / f"e2_{tag}_{_safe(model)}_{sid}_{idx:03d}.json"


# ---------------------------------------------------------------------------
# Per-round generation functions
# ---------------------------------------------------------------------------

def gen_r0(scenario: DD_Scenario, pid: str, idx: int,
           gen_model: str, mod_model: str) -> dict:
    role = PERSPECTIVE_BY_ID[pid]
    cache = _cache_e2("r0", gen_model, scenario.id, pid, idx)
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            cache.unlink()
    user = R0_USER.format(
        scenario=scenario.prompt,
        perspective_description=role["framing"],
    )
    text = _call_gen(gen_model, NCOT_SYSTEM, user, seed=idx)
    rec = {"scenario_id": scenario.id, "pid": pid, "idx": idx,
           "model": gen_model, "round": "r0", "output": text}
    cache.write_text(json.dumps(rec, ensure_ascii=False))
    return rec


def gen_r1(scenario: DD_Scenario, pid: str, idx: int,
           gen_model: str, r0_map: dict) -> dict:
    role = PERSPECTIVE_BY_ID[pid]
    cache = _cache_e2("r1", gen_model, scenario.id, pid, idx)
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            cache.unlink()
    own_r0 = (r0_map.get(pid) or {}).get("output", "[No statement]")
    others = "\n\n".join(
        f"[{PERSPECTIVE_BY_ID[p]['label']}]:\n{(r0_map.get(p) or {}).get('output', '[No statement]')}"
        for p in PERSPECTIVE_IDS if p != pid
    )
    user = R1_USER.format(
        scenario=scenario.prompt,
        perspective_description=role["framing"],
        own_r0=own_r0,
        other_r0=others,
    )
    text = _call_gen(gen_model, NCOT_SYSTEM, user, seed=idx + 1000)
    rec = {"scenario_id": scenario.id, "pid": pid, "idx": idx,
           "model": gen_model, "round": "r1", "output": text}
    cache.write_text(json.dumps(rec, ensure_ascii=False))
    return rec


def gen_r2(scenario: DD_Scenario, pid: str, idx: int,
           gen_model: str, r0_map: dict, r1_map: dict) -> dict:
    role = PERSPECTIVE_BY_ID[pid]
    cache = _cache_e2("r2", gen_model, scenario.id, pid, idx)
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            cache.unlink()
    own_r0 = (r0_map.get(pid) or {}).get("output", "[No statement]")
    own_r1 = (r1_map.get(pid) or {}).get("output", "[No rebuttal]")
    others_r1 = "\n\n".join(
        f"[{PERSPECTIVE_BY_ID[p]['label']}]:\n{(r1_map.get(p) or {}).get('output', '[No rebuttal]')}"
        for p in PERSPECTIVE_IDS if p != pid
    )
    user = R2_USER.format(
        scenario=scenario.prompt,
        perspective_description=role["framing"],
        own_r0=own_r0,
        own_r1=own_r1,
        other_r1=others_r1,
    )
    text = _call_gen(gen_model, NCOT_SYSTEM, user, seed=idx + 3000)
    rec = {"scenario_id": scenario.id, "pid": pid, "idx": idx,
           "model": gen_model, "round": "r2", "output": text}
    cache.write_text(json.dumps(rec, ensure_ascii=False))
    return rec


def run_open_mod(scenario: DD_Scenario, idx: int,
                 gen_model: str, mod_model: str,
                 r0_map: dict, r1_map: dict, r2_map: dict) -> dict:
    cache = _cache_mod_e2("open_mod", gen_model, scenario.id, idx)
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            cache.unlink()
    parts = []
    for pid in PERSPECTIVE_IDS:
        lbl = PERSPECTIVE_BY_ID[pid]["label"]
        r0t = (r0_map.get(pid) or {}).get("output", "[empty]")[:700]
        r1t = (r1_map.get(pid) or {}).get("output", "[empty]")[:700]
        r2t = (r2_map.get(pid) or {}).get("output", "[empty]")[:900]
        parts.append(
            f"=== {lbl} ===\n[R0]:\n{r0t}\n\n[R1]:\n{r1t}\n\n[R2]:\n{r2t}"
        )
    transcript = "\n\n".join(parts)
    taxonomy_labels = '"ACTION_1" (to do), "ACTION_2" (not to do), "NO_COMMITMENT"'
    user = OPEN_MOD_USER.format(
        scenario=scenario.prompt,
        taxonomy_labels=taxonomy_labels,
        transcript=transcript,
    )
    seed = int(hashlib.sha1(f"e2_omod|{scenario.id}|{idx}|{gen_model}".encode()).hexdigest()[:8], 16)
    result = _call_mod(mod_model, OPEN_MOD_SYSTEM, user, seed=seed, max_tokens=1200)
    for k, dv in [("consensus_reached", False), ("consensus_kind", "none"),
                  ("consensus_decision", None), ("novel_actions_proposed", []),
                  ("synthesis_label", None), ("synthesis_description", None),
                  ("points_of_disagreement", []), ("summary", "")]:
        result.setdefault(k, dv)
    result["gen_model"] = gen_model
    result["scenario_id"] = scenario.id
    result["sample_idx"] = idx
    cache.write_text(json.dumps(result, ensure_ascii=False))
    return result


def gen_r3(scenario: DD_Scenario, pid: str, idx: int,
           gen_model: str, synth_label: str, synth_desc: str,
           r0_map: dict, r1_map: dict, r2_map: dict) -> dict:
    cache = _cache_e2("r3", gen_model, scenario.id, pid, idx)
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            cache.unlink()
    role = PERSPECTIVE_BY_ID[pid]
    parts = []
    for p in PERSPECTIVE_IDS:
        lbl = PERSPECTIVE_BY_ID[p]["label"]
        r0t = (r0_map.get(p) or {}).get("output", "[empty]")[:400]
        r1t = (r1_map.get(p) or {}).get("output", "[empty]")[:400]
        r2t = (r2_map.get(p) or {}).get("output", "[empty]")[:500]
        parts.append(f"[{lbl}] R0: {r0t}\n[{lbl}] R1: {r1t}\n[{lbl}] R2: {r2t}")
    transcript = "\n\n".join(parts)
    user = SYNTH_ACCEPT_USER.format(
        scenario=scenario.prompt,
        perspective_description=role["framing"],
        transcript=transcript,
        synthesis_label=synth_label,
        synthesis_description=synth_desc,
    )
    text = _call_gen(gen_model, SYNTH_ACCEPT_SYSTEM, user, seed=idx + 4000)
    rec = {"scenario_id": scenario.id, "pid": pid, "idx": idx,
           "model": gen_model, "round": "r3", "output": text}
    cache.write_text(json.dumps(rec, ensure_ascii=False))
    return rec


def extract_r3_decision(scenario: DD_Scenario, pid: str, idx: int,
                        gen_model: str, mod_model: str,
                        r3_output: str, synth_label: str, synth_desc: str) -> dict:
    cache = _cache_e2("r3_dec", mod_model, scenario.id, pid, idx)
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            cache.unlink()
    user = SYNTH_R3_EXTRACT_USER.format(
        scenario=scenario.prompt,
        synthesis_label=synth_label,
        synthesis_description=synth_desc,
        response=r3_output[:2000],
    )
    seed = int(hashlib.sha1(
        f"e2_r3dec|{scenario.id}|{pid}|{idx}|{gen_model}".encode()
    ).hexdigest()[:8], 16)
    result = _call_mod(mod_model, SYNTH_R3_EXTRACT_SYSTEM, user, seed=seed, max_tokens=400)
    valid = {"ACCEPT", "ACCEPT_WITH_MODIFICATION", "REJECT", "UNKNOWN"}
    if result.get("response_type") not in valid:
        result["response_type"] = "UNKNOWN"
    result.setdefault("modification_description", None)
    result.setdefault("rejection_reason", None)
    result.setdefault("decision_confidence", 0.5)
    cache.write_text(json.dumps(result, ensure_ascii=False))
    return result


def run_synth_final_mod(scenario: DD_Scenario, idx: int,
                        gen_model: str, mod_model: str,
                        synth_label: str, synth_desc: str,
                        r3_map: dict) -> dict:
    cache = _cache_mod_e2("r3_final", gen_model, scenario.id, idx)
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            cache.unlink()
    parts = []
    for pid in PERSPECTIVE_IDS:
        lbl = PERSPECTIVE_BY_ID[pid]["label"]
        r3 = (r3_map.get(pid) or {}).get("output", "[empty]")[:600]
        parts.append(f"[{lbl}]:\n{r3}")
    user = SYNTH_FINAL_MOD_USER.format(
        scenario=scenario.prompt,
        synthesis_label=synth_label,
        synthesis_description=synth_desc,
        responses="\n\n".join(parts),
    )
    seed = int(hashlib.sha1(f"e2_synth_final|{scenario.id}|{idx}|{gen_model}".encode()).hexdigest()[:8], 16)
    result = _call_mod(mod_model, SYNTH_FINAL_MOD_SYSTEM, user, seed=seed, max_tokens=600)
    for k, dv in [("consensus_reached", False), ("partial_convergence", False),
                  ("unanimous_accept", False), ("accept_count", 0),
                  ("modify_count", 0), ("reject_count", 0),
                  ("final_synthesis_label", None), ("final_synthesis_description", None),
                  ("remaining_disagreements", [])]:
        result.setdefault(k, dv)
    result["gen_model"] = gen_model
    result["scenario_id"] = scenario.id
    result["sample_idx"] = idx
    cache.write_text(json.dumps(result, ensure_ascii=False))
    return result


def run_integration(scenario: DD_Scenario, idx: int,
                    gen_model: str, mod_model: str,
                    synth_label: str, synth_desc: str,
                    r3_dec_map: dict) -> dict:
    cache = _cache_mod_e2("integ", gen_model, scenario.id, idx)
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            cache.unlink()
    parts = []
    for pid in PERSPECTIVE_IDS:
        lbl = PERSPECTIVE_BY_ID[pid]["label"]
        dec = r3_dec_map.get(pid, {})
        rt = dec.get("response_type", "UNKNOWN")
        mod = dec.get("modification_description") or "[none]"
        parts.append(f"[{lbl} ({pid})]: {rt}\nModification: {mod}")
    user = INTEG_MOD_USER.format(
        scenario=scenario.prompt,
        synthesis_label=synth_label,
        synthesis_description=synth_desc,
        modification_requests="\n\n".join(parts),
    )
    seed = int(hashlib.sha1(f"e2_integ|{scenario.id}|{idx}|{gen_model}".encode()).hexdigest()[:8], 16)
    result = _call_mod(mod_model, INTEG_MOD_SYSTEM, user, seed=seed, max_tokens=600)
    result.setdefault("integrated_label", synth_label + "_integrated")
    result.setdefault("integrated_description", synth_desc)
    result.setdefault("modifications_addressed", [])
    result.setdefault("modifications_unaddressed", [])
    result.setdefault("integration_rationale", "")
    result["gen_model"] = gen_model
    result["scenario_id"] = scenario.id
    result["sample_idx"] = idx
    cache.write_text(json.dumps(result, ensure_ascii=False))
    return result


def gen_r4(scenario: DD_Scenario, pid: str, idx: int,
           gen_model: str,
           integ_label: str, integ_desc: str, integ_rationale: str) -> dict:
    cache = _cache_e2("r4", gen_model, scenario.id, pid, idx)
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            cache.unlink()
    role = PERSPECTIVE_BY_ID[pid]
    user = R4_USER.format(
        scenario=scenario.prompt,
        perspective_description=role["framing"],
        integrated_label=integ_label,
        integrated_description=integ_desc,
        integration_rationale=integ_rationale,
    )
    text = _call_gen(gen_model, NCOT_SYSTEM, user, seed=idx + 5000, max_tokens=2000)
    rec = {"scenario_id": scenario.id, "pid": pid, "idx": idx,
           "model": gen_model, "round": "r4", "output": text}
    cache.write_text(json.dumps(rec, ensure_ascii=False))
    return rec


def run_r4_final_mod(scenario: DD_Scenario, idx: int,
                     gen_model: str, mod_model: str,
                     integ_label: str, integ_desc: str,
                     r4_map: dict) -> dict:
    cache = _cache_mod_e2("r4_final", gen_model, scenario.id, idx)
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            cache.unlink()
    parts = []
    for pid in PERSPECTIVE_IDS:
        lbl = PERSPECTIVE_BY_ID[pid]["label"]
        r4t = (r4_map.get(pid) or {}).get("output", "[empty]")[:500]
        parts.append(f"[{lbl}]:\n{r4t}")
    user = R4_FINAL_MOD_USER.format(
        scenario=scenario.prompt,
        integrated_label=integ_label,
        integrated_description=integ_desc,
        votes="\n\n".join(parts),
    )
    seed = int(hashlib.sha1(f"e2_r4final|{scenario.id}|{idx}|{gen_model}".encode()).hexdigest()[:8], 16)
    result = _call_mod(mod_model, R4_FINAL_MOD_SYSTEM, user, seed=seed, max_tokens=400)
    for k, dv in [("consensus_reached", False), ("majority_accept", False),
                  ("accept_count", 0), ("reject_count", 0), ("unanimous", False),
                  ("final_position_label", None), ("structural_rejection", False),
                  ("primary_rejection_reason", None)]:
        result.setdefault(k, dv)
    result["gen_model"] = gen_model
    result["scenario_id"] = scenario.id
    result["sample_idx"] = idx
    result["integrated_label"] = integ_label
    cache.write_text(json.dumps(result, ensure_ascii=False))
    return result


# ---------------------------------------------------------------------------
# Full per-scenario orchestration
# ---------------------------------------------------------------------------

def run_scenario_sample(
    scenario: DD_Scenario,
    idx: int,
    gen_model: str,
    mod_model: str,
    budget_cap: float,
) -> dict:
    """Run one complete R0-R4 debate for one scenario x sample x generator.

    Returns a summary dict with key outcome fields.
    """
    result_base = {
        "generator": gen_model,
        "moderator": mod_model,
        "scenario_id": scenario.id,
        "sample_idx": idx,
        "r2_consensus_reached": False,
        "has_synthesis": False,
        "r3_full_consensus": False,
        "r4_full_consensus": False,
        "r4_structural_rejection": False,
        "r4_accept_count": 0,
        "r4_reject_count": 0,
        "skipped_budget": False,
    }

    if _over_budget(budget_cap):
        result_base["skipped_budget"] = True
        return result_base

    # --- R0 ---
    r0_map: dict[str, dict] = {}
    for pid in PERSPECTIVE_IDS:
        r0_map[pid] = gen_r0(scenario, pid, idx, gen_model, mod_model)
        if _over_budget(budget_cap):
            result_base["skipped_budget"] = True
            return result_base

    # --- R1 ---
    r1_map: dict[str, dict] = {}
    for pid in PERSPECTIVE_IDS:
        r1_map[pid] = gen_r1(scenario, pid, idx, gen_model, r0_map)
        if _over_budget(budget_cap):
            result_base["skipped_budget"] = True
            return result_base

    # --- R2 ---
    r2_map: dict[str, dict] = {}
    for pid in PERSPECTIVE_IDS:
        r2_map[pid] = gen_r2(scenario, pid, idx, gen_model, r0_map, r1_map)
        if _over_budget(budget_cap):
            result_base["skipped_budget"] = True
            return result_base

    # --- Open moderator (R2 synthesis check) ---
    open_mod = run_open_mod(scenario, idx, gen_model, mod_model, r0_map, r1_map, r2_map)
    result_base["r2_consensus_reached"] = bool(open_mod.get("consensus_reached"))
    result_base["r2_consensus_kind"] = open_mod.get("consensus_kind", "none")
    result_base["r2_open_mod_summary"] = open_mod.get("summary", "")

    synth_label = open_mod.get("synthesis_label")
    synth_desc = open_mod.get("synthesis_description") or synth_label or ""
    result_base["has_synthesis"] = bool(synth_label)

    if not synth_label:
        # No synthesis emerged: debate ended without convergence after R2
        return result_base

    if _over_budget(budget_cap):
        result_base["skipped_budget"] = True
        return result_base

    # --- R3: synthesis acceptance ---
    r3_map: dict[str, dict] = {}
    for pid in PERSPECTIVE_IDS:
        r3_map[pid] = gen_r3(scenario, pid, idx, gen_model, synth_label, synth_desc,
                             r0_map, r1_map, r2_map)
        if _over_budget(budget_cap):
            result_base["skipped_budget"] = True
            return result_base

    # --- R3 decision extraction ---
    r3_dec_map: dict[str, dict] = {}
    for pid in PERSPECTIVE_IDS:
        r3_dec_map[pid] = extract_r3_decision(
            scenario, pid, idx, gen_model, mod_model,
            (r3_map.get(pid) or {}).get("output", ""),
            synth_label, synth_desc,
        )

    # --- Synthesis final-mod ---
    synth_final = run_synth_final_mod(scenario, idx, gen_model, mod_model,
                                      synth_label, synth_desc, r3_map)
    r3_consensus = bool(synth_final.get("consensus_reached"))
    result_base["r3_full_consensus"] = r3_consensus
    result_base["r3_accept_count"] = synth_final.get("accept_count", 0)
    result_base["r3_modify_count"] = synth_final.get("modify_count", 0)
    result_base["r3_reject_count"] = synth_final.get("reject_count", 0)

    if _over_budget(budget_cap):
        result_base["skipped_budget"] = True
        return result_base

    # --- Integration pass (if all modified) ---
    any_reject = synth_final.get("reject_count", 0) > 0
    if any_reject:
        # Structural rejection: at least one perspective is holding a non-negotiable concern
        result_base["r4_structural_rejection"] = True
        # Still run R4 on the best integration we can construct
    integ = run_integration(scenario, idx, gen_model, mod_model,
                            synth_label, synth_desc, r3_dec_map)
    integ_label = integ.get("integrated_label", synth_label + "_integrated")
    integ_desc = integ.get("integrated_description", synth_desc)
    integ_rationale = integ.get("integration_rationale", "")

    if _over_budget(budget_cap):
        result_base["skipped_budget"] = True
        return result_base

    # --- R4: binary vote ---
    r4_map: dict[str, dict] = {}
    for pid in PERSPECTIVE_IDS:
        r4_map[pid] = gen_r4(scenario, pid, idx, gen_model,
                             integ_label, integ_desc, integ_rationale)
        if _over_budget(budget_cap):
            result_base["skipped_budget"] = True
            return result_base

    # --- R4 final moderator ---
    r4_final = run_r4_final_mod(scenario, idx, gen_model, mod_model,
                                integ_label, integ_desc, r4_map)
    result_base["r4_full_consensus"] = bool(r4_final.get("consensus_reached"))
    result_base["r4_majority_accept"] = bool(r4_final.get("majority_accept"))
    result_base["r4_accept_count"] = r4_final.get("accept_count", 0)
    result_base["r4_reject_count"] = r4_final.get("reject_count", 0)
    result_base["r4_structural_rejection"] = bool(r4_final.get("structural_rejection"))
    result_base["r4_final_label"] = r4_final.get("final_position_label")
    result_base["integ_label"] = integ_label
    result_base["synth_label"] = synth_label

    return result_base


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    global _total_cost_usd

    parser = argparse.ArgumentParser(description="E2: Scaled multi-agent debate on DailyDilemmas")
    parser.add_argument("--scenarios", type=int, default=30,
                        help="Number of DailyDilemmas scenarios to run")
    parser.add_argument("--samples", type=int, default=1,
                        help="Samples per (scenario x generator) cell")
    parser.add_argument("--generators", default="gpt-5.4-nano,claude-sonnet-4-6",
                        help="Comma-separated generator model names")
    parser.add_argument("--moderator", default="claude-sonnet-4-6",
                        help="Moderator model (used for R3/R4 synthesis)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Concurrent threads (keep low for Anthropic rate limits)")
    parser.add_argument("--budget-cap-usd", type=float, default=110.0,
                        help="Hard budget cap in USD; run pauses if exceeded")
    args = parser.parse_args(argv)

    generators = [g.strip() for g in args.generators.split(",") if g.strip()]
    mod_model = args.moderator

    print(f"E2 scaled debate: {args.scenarios} scenarios x {len(generators)} generators x "
          f"{args.samples} samples")
    print(f"Generators: {generators}")
    print(f"Moderator: {mod_model}")
    print(f"Budget cap: ${args.budget_cap_usd}")
    print(f"Workers: {args.workers}")

    print("Loading DailyDilemmas 100-scenario pool...")
    all_scenarios = load_daily_dilemmas(100)

    # Subsample: shuffle with E2_SAMPLE_SEED, take first --scenarios
    rng = random.Random(E2_SAMPLE_SEED)
    shuffled = list(all_scenarios)
    rng.shuffle(shuffled)
    scenarios = shuffled[:args.scenarios]
    print(f"  Selected {len(scenarios)} scenarios (seed={E2_SAMPLE_SEED})")

    tasks = [
        (scenario, idx, gen_model)
        for gen_model in generators
        for scenario in scenarios
        for idx in range(args.samples)
    ]
    print(f"  Total (scenario, sample, generator) cells: {len(tasks)}")

    all_rows: list[dict] = []
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(
                run_scenario_sample,
                scenario, idx, gen_model, mod_model, args.budget_cap_usd,
            ): (scenario.id, idx, gen_model)
            for scenario, idx, gen_model in tasks
        }

        for fut in as_completed(futs):
            key_t = futs[fut]
            try:
                row = fut.result()
                all_rows.append(row)
                done += 1
                if done % 10 == 0 or done == len(tasks):
                    print(f"  {done}/{len(tasks)} cells done | "
                          f"est. cost so far: ${_total_cost_usd:.2f}", flush=True)
                if _total_cost_usd >= args.budget_cap_usd:
                    print(f"\nBUDGET CAP ${args.budget_cap_usd:.2f} reached. "
                          f"Stopping submission of new tasks.", flush=True)
                    # Cancel remaining futures
                    for f in list(futs):
                        if not f.done():
                            f.cancel()
                    break
            except Exception as e:
                print(f"  ERROR {key_t}: {e}", flush=True)

    # --- Write checkpoint CSV ---
    out_csv = OUT_DIR / "debate_dd_scaled.csv"
    if all_rows:
        fieldnames = sorted(set().union(*[r.keys() for r in all_rows]))
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nWrote {len(all_rows)} rows to {out_csv}")
    else:
        print("No rows collected.")

    # --- Print falsification check ---
    completed = [r for r in all_rows if not r.get("skipped_budget")]
    if completed:
        n_total = len(completed)
        n_r4_consensus = sum(1 for r in completed if r.get("r4_full_consensus"))
        n_has_synth = sum(1 for r in completed if r.get("has_synthesis"))
        n_struct_reject = sum(1 for r in completed if r.get("r4_structural_rejection"))

        print(f"\n--- E2 FALSIFICATION CHECK ---")
        print(f"Completed cells:      {n_total}")
        print(f"Synthesis emerged:    {n_has_synth} ({100*n_has_synth/n_total:.1f}%)")
        print(f"R4 full consensus:    {n_r4_consensus} ({100*n_r4_consensus/n_total:.1f}%)")
        print(f"Structural rejection: {n_struct_reject} ({100*n_struct_reject/n_total:.1f}%)")
        print(f"\nPre-registered band: 85-105% (95% +/- 10pp)")
        consensus_rate = 100 * n_r4_consensus / n_total
        if 85.0 <= consensus_rate <= 105.0:
            print(f"RESULT: REPLICATES ({consensus_rate:.1f}% in band)")
        else:
            print(f"RESULT: OUTSIDE BAND ({consensus_rate:.1f}% -- headline claim downgrades)")

    print(f"\nTotal estimated cost: ${_total_cost_usd:.3f}")
    print(f"Skipped (budget cap): {sum(1 for r in all_rows if r.get('skipped_budget'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
