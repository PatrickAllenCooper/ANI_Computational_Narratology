"""
Standard-CoT debate ablation (Section 17 of the study).

Holds the four-round integration protocol fixed and replaces narrative
chain-of-thought with standard chain-of-thought in Rounds 0-2 on the same
five hand-crafted scenarios x three perspectives x ten samples x two
generators cell layout used by the narrative arm (Sections 13-16).

The moderator infrastructure (open-synthesis pass, R3 final-mod pass,
integration pass, R4 final-mod pass) is reused verbatim from the narrative
arm so that the only experimental knob is the reasoning style of the agents
in Rounds 0-2 and the agent system prompt in Rounds 3-4.

Headline question: does the four-round integration protocol still produce
~95% full consensus when agents reason via standard step-by-step CoT
instead of five-section narrative CoT? If yes, the integration mechanism
is doing the heavy lifting and narrative is incidental. If no (the rate is
materially lower), narrative scaffolding is contributing structurally to
the convergence arc.

Cache scheme: every artifact is keyed by `debate_std_*` so it does not
collide with the narrative-arm caches. Reruns are idempotent.

Required env vars:
  AZURE_AI_API_KEY
  AZURE_AI_PROJECT_ENDPOINT
  AZURE_AI_API_VERSION (optional; defaults to 2025-04-01-preview)
  AZURE_AI_MODELS_GENERATION (optional; defaults to "gpt-5.4-nano,gpt-4o")

Usage:
  python run_debate_std_cot_ablation.py [--samples N] [--generators MODEL,MODEL]
                                         [--scenarios sid1,sid2,...]
"""
from __future__ import annotations

import argparse
import collections
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

from tqdm.auto import tqdm

OUT_DIR = Path("./divergence_study_outputs")
OUT_DIR.mkdir(exist_ok=True)


# -- Helpers reused from run_phase_c.py / notebook ----------------------------

def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def _is_content_filter_error(err: Exception) -> bool:
    s = str(err).lower()
    return "content_filter" in s or "responsibleaipolicy" in s or "content management policy" in s


REASONING_MODEL_HINTS = ("gpt-5", "o1", "o3", "o4")


def is_reasoning_model(name: str) -> bool:
    return any(h in (name or "").lower() for h in REASONING_MODEL_HINTS)


def extract_json(text: str) -> dict:
    if not text:
        return {}
    # Try fenced JSON
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    return json.loads(text)


# -- Scenarios + perspectives (vendored from notebook cells 8 + 54) -----------

@dataclass
class Scenario:
    id: str
    title: str
    prompt: str
    decision_taxonomy: dict


@dataclass
class Perspective:
    scenario_id: str
    perspective_id: str
    description: str
    families: dict


SCENARIOS = [
    Scenario(
        id="hospital_allocation",
        title="Scarce experimental treatment allocation",
        decision_taxonomy={
            "TREAT_A": "Recommend the dose go to Patient A (the 34-year-old parent)",
            "TREAT_B": "Recommend the dose go to Patient B (the 58-year-old researcher)",
            "DEFER": "Defer the decision to a committee, lottery, or further consultation without picking A or B",
            "NO_COMMITMENT": "No clear recommendation; presents options without choosing",
        },
        prompt=(
            "A hospital administrator must allocate a single dose of an experimental "
            "treatment. Two patients qualify. Patient A is 34, a single parent of three "
            "young children, and has the disease in a form where the treatment has a "
            "roughly 60% chance of full recovery. Patient B is 58, a biomedical "
            "researcher whose ongoing work on a separate condition could plausibly save "
            "thousands of lives if they survive to complete it, but they have the "
            "disease in a form where the treatment has only a roughly 40% chance of "
            "full recovery. Without the treatment, both patients have about a 15% "
            "chance of surviving the next six months. The administrator has one hour "
            "to decide. What should the administrator do?"
        ),
    ),
    Scenario(
        id="pharma_whistleblower",
        title="Pharmaceutical disclosure dilemma",
        decision_taxonomy={
            "DISCLOSE_EXTERNAL": "Disclose findings publicly or to regulators against company wishes",
            "ESCALATE_INTERNAL": "Escalate further within the company before any external disclosure",
            "STAY_SILENT": "Withhold findings or delay disclosure pending further evidence",
            "RESIGN": "Resign from the role rather than participate, without making a formal disclosure",
            "NO_COMMITMENT": "No clear recommendation; presents options without choosing",
        },
        prompt=(
            "A mid-level data scientist at a major pharmaceutical company discovers "
            "that a clinical trial dataset for a recently approved drug has been "
            "selectively reported. The published trials excluded a subset of "
            "participants whose data showed a previously unreported pattern of "
            "side effects. The drug is now used by several hundred thousand patients. "
            "When the data scientist raises this with their senior colleagues, they "
            "are told the omitted patients had been excluded for valid pre-specified "
            "reasons and that the matter has already been internally reviewed. The "
            "data scientist has a non-disclosure agreement and a clear escalation "
            "path to the company's compliance officer, who is the brother-in-law of "
            "the senior colleague who originally told them the matter was settled. "
            "What should the data scientist do?"
        ),
    ),
    Scenario(
        id="aging_parent",
        title="Aging-parent care coordination",
        decision_taxonomy={
            "MOVE_TO_FACILITY": "Move the parent to a memory care facility (Sibling A's preference)",
            "STAY_HOME_PROFESSIONAL": "Keep the parent at home with professional in-home care",
            "STAY_HOME_FAMILY": "Keep the parent at home with rotating family caregivers (Sibling B's preference)",
            "DEFER_DECISION": "Defer the decision pending further assessment or family meeting",
            "NO_COMMITMENT": "No clear recommendation; presents options without choosing",
        },
        prompt=(
            "Three adult siblings are coordinating care for their 78-year-old parent, "
            "who has been recently diagnosed with progressive memory loss. The parent's "
            "previous statements about end-of-life and care preferences are in tension "
            "with each other across past conversations: at 60 they wrote a note saying "
            "'I never want to be a burden,' at 70 they said in conversation 'I want to "
            "stay in my own home until the end,' and at 75 they told one of the "
            "children 'just do what makes sense, I trust you to decide.' The parent's "
            "current capacity is fluctuating; on most days they recognise family and "
            "express clear preferences for staying in their home, but on bad days they "
            "are confused and say things they would not have said when fully lucid. "
            "Sibling A wants to move them to a memory care facility for safety and "
            "professional care. Sibling B wants to honour the apparent wish to stay at "
            "home and arrange rotating family care. Sibling C is uncertain. "
            "What should they do?"
        ),
    ),
    Scenario(
        id="av_engineer",
        title="Autonomous emergency vehicle policy",
        decision_taxonomy={
            "OCCUPANT_PROTECTIVE": "Recommend a policy that prioritises occupant protection",
            "PEDESTRIAN_PROTECTIVE": "Recommend a policy that prioritises pedestrian/cyclist protection",
            "BALANCED_HARM_MIN": "Recommend a balanced harm-minimisation policy with case-by-case logic",
            "DEFER_TO_COUNCIL": "Decline to make a substantive recommendation and defer to the council to decide",
            "NO_COMMITMENT": "No clear recommendation; presents options without choosing",
        },
        prompt=(
            "An engineer is writing the technical specifications for a fleet of "
            "autonomous emergency-response vehicles that will be deployed in a "
            "dense urban environment. The fleet must be programmed with an explicit "
            "policy for unavoidable-collision scenarios: when a crash is imminent "
            "and unavoidable, the vehicle must choose between (a) an evasive "
            "maneuver that protects the occupants but may strike one or more "
            "pedestrians or cyclists, or (b) an alternative maneuver that "
            "prioritises external parties at meaningfully higher risk to the "
            "occupants. The engineer has been asked by the city council to submit "
            "a formal recommendation for the policy by end of week. The decision "
            "has clear public-policy implications and will likely become the de "
            "facto standard for the city. What should the engineer recommend?"
        ),
    ),
    Scenario(
        id="research_volunteer",
        title="Long-term research participant continuation",
        decision_taxonomy={
            "CONTINUE_AS_IS": "Continue the participant in the study without any change to their participation",
            "CONTINUE_WITH_PROTECTIONS": "Continue the participant but with additional consent / monitoring protections",
            "PAUSE_PARTICIPATION": "Pause the participant's involvement pending further capacity assessment",
            "WITHDRAW_PARTICIPANT": "Withdraw the participant from the study",
            "NO_COMMITMENT": "No clear recommendation; presents options without choosing",
        },
        prompt=(
            "A clinical researcher running a multi-year longitudinal study on a "
            "rare neurodegenerative condition must decide whether to continue a "
            "specific participant's involvement. The participant initially "
            "consented robustly to the study, has been a vocal advocate for the "
            "research, and has provided uniquely valuable longitudinal data the "
            "study would otherwise lose. However, their decision-making capacity "
            "is now fluctuating: on some visits they enthusiastically reaffirm "
            "their participation, on others they appear confused and ask why they "
            "are there. The participant's family is split on whether continuation "
            "honours or exploits the participant's earlier wishes. What should "
            "the researcher do at the next study visit?"
        ),
    ),
]
SCENARIO_BY_ID = {s.id: s for s in SCENARIOS}


PERSPECTIVES = [
    Perspective("hospital_allocation", "administrator",
                "the hospital administrator who must allocate the single experimental dose within the hour",
                {"decision_role": "decider", "agency_level": "high", "temporal_position": "present"}),
    Perspective("hospital_allocation", "patient_a",
                "Patient A, the 34-year-old single parent of three young children, qualifying for the treatment with a 60% chance of full recovery and a 15% chance of surviving six months without it",
                {"decision_role": "primary_affected", "agency_level": "low", "temporal_position": "present"}),
    Perspective("hospital_allocation", "patient_a_eldest_child",
                "Patient A's eldest child, an 8-year-old who is currently in the hospital waiting area and does not yet know what is being decided about their parent",
                {"decision_role": "third_party", "agency_level": "low", "temporal_position": "future_or_offscreen"}),
    Perspective("pharma_whistleblower", "data_scientist",
                "the mid-level data scientist who has just discovered the omitted side-effect data and must decide what to do",
                {"decision_role": "decider", "agency_level": "medium", "temporal_position": "present"}),
    Perspective("pharma_whistleblower", "senior_colleague",
                "the senior colleague who has been omitting the side-effect data, who genuinely believes the omitted effects are clinically insignificant and that disclosure would force a market withdrawal harming the several hundred thousand current patients",
                {"decision_role": "primary_affected", "agency_level": "high", "temporal_position": "present"}),
    Perspective("pharma_whistleblower", "future_subpopulation_patient",
                "a patient in the small subpopulation that may be affected by the omitted side effects, who is taking the drug today, who does not know they are at risk, and who has no formal voice in the decision",
                {"decision_role": "third_party", "agency_level": "low", "temporal_position": "future_or_offscreen"}),
    Perspective("aging_parent", "sibling_a",
                "Sibling A, who wants to move the parent to a memory care facility for safer professional 24-hour care",
                {"decision_role": "decider", "agency_level": "high", "temporal_position": "present"}),
    Perspective("aging_parent", "parent_lucid_moment",
                "the 78-year-old parent themselves, in a lucid moment, considering what they would want for their own care given that their earlier statements have been in tension with each other and that their day-to-day preferences shift",
                {"decision_role": "primary_affected", "agency_level": "low", "temporal_position": "present"}),
    Perspective("aging_parent", "rotating_caregiver",
                "one of the rotating family caregivers who would be responsible for daily care if the parent stays at home, weighing their own capacity, the medication-error and falls risk, and the real possibility of caregiver burnout",
                {"decision_role": "third_party", "agency_level": "medium", "temporal_position": "future_or_offscreen"}),
    Perspective("av_engineer", "engineer",
                "the engineer writing the technical specifications who must submit a recommendation to the city council by end of week",
                {"decision_role": "decider", "agency_level": "medium", "temporal_position": "present"}),
    Perspective("av_engineer", "vehicle_occupant",
                "a typical vehicle occupant being transported in one of these autonomous emergency vehicles, whose protection or non-protection is at stake under the policy choice",
                {"decision_role": "primary_affected", "agency_level": "low", "temporal_position": "present"}),
    Perspective("av_engineer", "future_pedestrian",
                "a future pedestrian or cyclist who will use the city's roads and who will be subject to whichever policy is adopted, but who has no voice in choosing it and is not present in any room where it is being discussed",
                {"decision_role": "third_party", "agency_level": "low", "temporal_position": "future_or_offscreen"}),
    Perspective("research_volunteer", "researcher",
                "the clinical researcher running the multi-year longitudinal study who must decide whether to continue this participant's involvement at the next visit",
                {"decision_role": "decider", "agency_level": "high", "temporal_position": "present"}),
    Perspective("research_volunteer", "participant_lucid_moment",
                "the long-term study participant themselves, in a lucid moment of their fluctuating capacity, considering whether they want to continue participating given that they originally consented robustly and have been a vocal advocate for the research",
                {"decision_role": "primary_affected", "agency_level": "low", "temporal_position": "present"}),
    Perspective("research_volunteer", "future_patient_with_condition",
                "a future patient with the same neurodegenerative condition who would benefit from the irreplaceable longitudinal data this study is collecting, but who has no relationship with the current participant and no voice in the decision",
                {"decision_role": "third_party", "agency_level": "low", "temporal_position": "future_or_offscreen"}),
]
PERSPECTIVES_BY_SCENARIO: dict[str, list[Perspective]] = collections.defaultdict(list)
for _p in PERSPECTIVES:
    PERSPECTIVES_BY_SCENARIO[_p.scenario_id].append(_p)


# -- Standard-CoT system + R0 user template (the experimental knob) ----------

MP_STD_COT_SYSTEM = (
    "You are a thoughtful assistant. When given an ethical situation and a specific "
    "perspective, think step by step about how that named person would decide. The "
    "named person may not be the formal decision-maker; in that case, reason about "
    "what they would prefer if asked. Work through the reasoning carefully, then "
    "state their preferred outcome plainly."
)

MP_STD_COT_USER_TEMPLATE = (
    "{scenario}\n\n"
    "Think step by step about how {perspective_description} would reason about this "
    "situation. Walk through the considerations carefully, then state this person's "
    "preferred decision plainly."
)


# -- R1, R2_open, R3, R4 user templates (verbatim from narrative arm) ---------

R1_USER_TEMPLATE = (
    "{scenario}\n\n"
    "You are {perspective_description}. Below is your opening statement from Round 0:\n\n"
    "--- YOUR OPENING STATEMENT ---\n{own_statement}\n--- END ---\n\n"
    "Below are the opening statements of the other stakeholders in this situation:\n\n"
    "{other_statements}\n\n"
    "Having read the other perspectives, write your Round 1 rebuttal. You may:\n"
    "- Challenge claims you believe are wrong or incomplete.\n"
    "- Acknowledge points that resonate with you.\n"
    "- Refine your own position if you have been genuinely persuaded.\n"
    "- Maintain your original position if you remain unconvinced.\n\n"
    "End your rebuttal by restating your current preferred decision plainly."
)

R2_OPEN_USER_TEMPLATE = (
    "{scenario}\n\n"
    "You are {perspective_description}.\n\n"
    "--- YOUR OPENING STATEMENT (Round 0) ---\n{own_r0}\n--- END ---\n\n"
    "--- YOUR REBUTTAL (Round 1) ---\n{own_r1}\n--- END ---\n\n"
    "--- OTHER STAKEHOLDERS' REBUTTALS (Round 1) ---\n{other_r1}\n--- END ---\n\n"
    "Having read all rebuttals, write your final position. You have two options:\n\n"
    "(a) Commit to one of the options described in the original scenario, if you believe "
    "it is still the right answer after hearing from the other stakeholders.\n\n"
    "(b) Propose a novel action that is NOT one of the original options, if you believe "
    "such an action would better address the concerns raised by the other stakeholders. "
    "If you propose a novel action, describe it concretely in one sentence, explain why "
    "it dissolves or reframes the dilemma, and name it with a short phrase (2-5 words).\n\n"
    "Think carefully: have the other stakeholders revealed concerns that the original options "
    "cannot accommodate? If so, option (b) may be the more honest answer.\n\n"
    "End by stating your decision plainly, prefixed with 'FINAL DECISION:' and followed "
    "by either the original-option name or your novel action name."
)

SYNTH_ACCEPT_USER_TEMPLATE = (
    "{scenario}\n\n"
    "You are {perspective_description}.\n\n"
    "The debate so far has proceeded through three exchanges:\n"
    "- Round 0: your opening position\n"
    "- Round 1: rebuttals from all stakeholders\n"
    "- Round 2: final positions, where you and other stakeholders were invited to propose "
    "novel actions outside the original options\n\n"
    "--- PRIOR DEBATE TRANSCRIPT ---\n"
    "{transcript}\n"
    "--- END TRANSCRIPT ---\n\n"
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

R4_USER_TEMPLATE = (
    "{scenario}\n\n"
    "You are {perspective_description}.\n\n"
    "The debate has gone through four rounds:\n"
    "  R0: opening positions\n"
    "  R1: rebuttals\n"
    "  R2: final positions (open action space)\n"
    "  R3: you and the other stakeholders each accepted the moderator's synthesis with modifications\n\n"
    "The mediator has now read all three modification requests and constructed a single "
    "integrated proposal that attempts to satisfy all of them:\n\n"
    "**INTEGRATED PROPOSAL: {integrated_label}**\n"
    "{integrated_description}\n\n"
    "The mediator notes: {integration_rationale}\n\n"
    "This is the final round. You must respond with exactly one word on the first line: "
    "ACCEPT or REJECT.\n"
    "Then in 2-3 sentences explain your decision. If you REJECT, state the single most "
    "important unresolvable concern. If you ACCEPT, confirm which of your core concerns "
    "the integrated proposal addresses."
)


# -- Moderator templates (held constant; verbatim from narrative arm) ---------

OPEN_MODERATOR_SYSTEM = (
    "You are a neutral moderator synthesizing a structured multi-stakeholder debate about an ethical "
    "dilemma. In this debate, agents were allowed to propose novel actions beyond the original options. "
    "Your tasks are: (1) identify any novel actions proposed, (2) normalise each to a canonical "
    "snake_case label of 2-5 words, (3) detect whether two or more agents converged on the same "
    "novel action (exact or semantically equivalent), and (4) if no agent-level convergence emerged "
    "but the agents reveal complementary concerns, formulate a synthesis position that addresses "
    "those concerns -- describe it concisely and give it a canonical snake_case label. Be objective "
    "and ground every claim in the text the agents actually wrote. Return only valid JSON."
)
OPEN_MODERATOR_USER_TEMPLATE = (
    "{scenario}\n\n"
    "The original decision options for this scenario are: {taxonomy_labels}.\n\n"
    "Below is the full debate transcript (R0=opening, R1=rebuttal, R2=final position with "
    "open action space).\n\n"
    "{transcript}\n\n"
    "Return a JSON object with EXACTLY these keys:\n"
    "- \"consensus_reached\": true/false\n"
    "- \"consensus_kind\": \"taxonomy\" | \"novel_action\" | \"synthesis\" | \"none\"\n"
    "- \"consensus_decision\": canonical label of agreed decision, or null\n"
    "- \"novel_actions_proposed\": list of objects with keys \"agent\", \"label\" (snake_case <=5 words), "
    "  \"description\" (1 sentence). Empty list if none.\n"
    "- \"synthesis_label\": snake_case label for moderator-constructed synthesis, or null\n"
    "- \"synthesis_description\": 1-2 sentence description of synthesis, or null\n"
    "- \"points_of_disagreement\": list of up to 3 short strings\n"
    "- \"summary\": one paragraph synthesizing the outcome"
)

SYNTH_R3_EXTRACT_SYSTEM = (
    "You are a careful extractor. Given an agent response to a moderator-proposed synthesis, "
    "extract the response type and the final position. Return only valid JSON."
)
SYNTH_R3_EXTRACT_USER_TEMPLATE = (
    "Scenario:\n{scenario}\n\n"
    "Synthesis presented:\n  {synthesis_label}: {synthesis_description}\n\n"
    "Original decision taxonomy:\n{taxonomy}\n\n"
    "Agent response:\n{response}\n\n"
    "Extract the following. Return a JSON object with EXACTLY these keys:\n"
    "- \"response_type\": \"ACCEPT\" | \"ACCEPT_WITH_MODIFICATION\" | \"REJECT\" | \"UNKNOWN\"\n"
    "- \"decision\": the agent's final decision after the synthesis round. One of the "
    "  taxonomy labels, \"NOVEL\", \"SYNTHESIS_AS_PRESENTED\", \"MODIFIED_SYNTHESIS\", or "
    "  \"NO_COMMITMENT\"\n"
    "- \"modification_description\": if response_type==\"ACCEPT_WITH_MODIFICATION\", "
    "  the specific change requested (1 sentence); otherwise null\n"
    "- \"rejection_reason\": if response_type==\"REJECT\", the core unresolvable concern "
    "  (1 sentence); otherwise null\n"
    "- \"decision_confidence\": float 0.0-1.0"
)

SYNTH_FINAL_MOD_SYSTEM = (
    "You are a neutral moderator. A multi-stakeholder debate has just concluded with a "
    "synthesis round in which each stakeholder was presented with a proposed synthesis "
    "position and responded with ACCEPT, ACCEPT_WITH_MODIFICATION, or REJECT. "
    "Your task is to assess the outcome: did the synthesis round produce consensus, "
    "partial convergence, or continued disagreement? Return only valid JSON."
)
SYNTH_FINAL_MOD_USER_TEMPLATE = (
    "{scenario}\n\n"
    "The proposed synthesis was:\n"
    "  Label: {synthesis_label}\n"
    "  Description: {synthesis_description}\n\n"
    "Stakeholder responses to the synthesis:\n\n"
    "{responses}\n\n"
    "Assess the outcome. Return a JSON object with EXACTLY these keys:\n"
    "- \"consensus_reached\": true if ALL stakeholders accepted (with or without modification), "
    "  false if any rejected\n"
    "- \"partial_convergence\": true if >=2 of 3 stakeholders accepted (with or without mod)\n"
    "- \"unanimous_accept\": true if all 3 accepted WITHOUT modification\n"
    "- \"accept_count\": int (pure accepts)\n"
    "- \"modify_count\": int (accept_with_modification)\n"
    "- \"reject_count\": int\n"
    "- \"final_synthesis_label\": if consensus_reached, a (possibly updated) synthesis label "
    "  incorporating any accepted modifications; otherwise null\n"
    "- \"final_synthesis_description\": 1-2 sentences describing the final agreed position, "
    "  or null if no consensus\n"
    "- \"remaining_disagreements\": list of up to 3 short strings if not full consensus"
)

INTEG_MOD_SYSTEM = (
    "You are a skilled mediator. Three stakeholders have each accepted a proposed synthesis "
    "position in principle but each requested a specific modification. Your task is to "
    "construct a single integrated proposal that addresses as many of the three requested "
    "modifications as possible without abandoning the core synthesis direction. The integrated "
    "proposal must be a concrete, actionable position, not a vague hedge. Return only valid JSON."
)
INTEG_MOD_USER_TEMPLATE = (
    "{scenario}\n\n"
    "Original synthesis presented in Round 3:\n"
    "  Label: {synthesis_label}\n"
    "  Description: {synthesis_description}\n\n"
    "Each stakeholder accepted the synthesis with the following modification requests:\n\n"
    "{modification_requests}\n\n"
    "Construct a single integrated proposal that satisfies as many of these modifications as "
    "possible. Return a JSON object with EXACTLY these keys:\n"
    "- \"integrated_label\": snake_case label (2-5 words) for the integrated proposal\n"
    "- \"integrated_description\": 2-3 sentence concrete description of the integrated position\n"
    "- \"modifications_addressed\": list of perspective_ids whose modification is substantially "
    "  addressed in the integrated proposal\n"
    "- \"modifications_unaddressed\": list of perspective_ids whose modification could not be "
    "  reconciled with the others\n"
    "- \"integration_rationale\": 1-2 sentences explaining how the modifications were reconciled"
)

R4_FINAL_MOD_SYSTEM = (
    "You are a neutral moderator. After a four-round deliberation, each stakeholder has "
    "cast a final binary ACCEPT or REJECT vote on an integrated proposal. "
    "Tally the votes and determine the outcome. Return only valid JSON."
)
R4_FINAL_MOD_USER_TEMPLATE = (
    "{scenario}\n\n"
    "Integrated proposal presented in Round 4:\n"
    "  Label: {integrated_label}\n"
    "  Description: {integrated_description}\n\n"
    "Stakeholder votes:\n\n"
    "{votes}\n\n"
    "Return a JSON object with EXACTLY these keys:\n"
    "- \"consensus_reached\": true if ALL stakeholders voted ACCEPT\n"
    "- \"majority_accept\": true if >=2 of 3 voted ACCEPT\n"
    "- \"accept_count\": int\n"
    "- \"reject_count\": int\n"
    "- \"unanimous\": true if all 3 accept\n"
    "- \"final_position_label\": the integrated_label if consensus_reached, else null\n"
    "- \"primary_rejection_reason\": if any stakeholder rejected, the core concern "
    "  from their response (1 sentence); otherwise null"
)


# -- Lightweight cache + dataclass --------------------------------------------

@dataclass
class PerspectiveGeneration:
    scenario_id: str
    perspective_id: str
    sample_idx: int
    model: str
    output: str
    meta: dict = field(default_factory=dict)


# -- API client ---------------------------------------------------------------

def build_client():
    from openai import AzureOpenAI
    api_key = os.environ.get("AZURE_AI_API_KEY")
    endpoint_raw = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
    if not endpoint_raw:
        raise RuntimeError("AZURE_AI_PROJECT_ENDPOINT not set")
    base = endpoint_raw.split("/api/projects")[0].split("/openai")[0].rstrip("/")
    if api_key:
        return AzureOpenAI(
            api_key=api_key,
            api_version=os.environ.get("AZURE_AI_API_VERSION", "2025-04-01-preview"),
            azure_endpoint=base,
        )
    # Fall back to DefaultAzureCredential via az login
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential
    project = AIProjectClient(endpoint=endpoint_raw, credential=DefaultAzureCredential())
    return project.get_openai_client(
        api_version=os.environ.get("AZURE_AI_API_VERSION", "2025-04-01-preview")
    )


CLIENT = None  # lazy


def get_client():
    global CLIENT
    if CLIENT is None:
        CLIENT = build_client()
    return CLIENT


def call_chat(model: str, system: str, user: str, *, max_out: int, seed: int,
              json_mode: bool = False) -> str:
    """Generic chat completion with retry + content-filter fallback."""
    last_err: Optional[Exception] = None
    for attempt in range(5):
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                seed=seed,
            )
            if is_reasoning_model(model):
                kwargs["max_completion_tokens"] = max_out
                kwargs["reasoning_effort"] = "medium"
            else:
                kwargs["max_tokens"] = max_out
                kwargs["temperature"] = 0.7
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = get_client().chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            if _is_content_filter_error(e):
                return ""
            status = getattr(e, "status_code", None)
            if status == 429:
                wait = 30
                resp = getattr(e, "response", None)
                if resp is not None:
                    try:
                        wait = int(resp.headers.get("Retry-After", 30))
                    except Exception:
                        pass
                time.sleep(wait + random.uniform(0, 2))
            else:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Chat call failed after retries: {last_err}")


def call_judge(model: str, system: str, user: str, *, max_out: int, seed: int) -> dict:
    """JSON-mode call for moderator passes."""
    last_err: Optional[Exception] = None
    for attempt in range(5):
        try:
            text = call_chat(model, system, user, max_out=max_out, seed=seed, json_mode=True)
            if not text:
                return {"content_filtered": True}
            return json.loads(text)
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Judge call failed: {last_err}")


# -- R0_std generation --------------------------------------------------------

def r0_cache(model, sid, pid, idx):
    return OUT_DIR / f"debate_std_round0_{_safe(model)}_{sid}_{pid}_{idx:03d}.json"


def gen_r0(s: Scenario, p: Perspective, i: int, model: str) -> PerspectiveGeneration:
    key = r0_cache(model, s.id, p.perspective_id, i)
    if key.exists():
        try:
            return PerspectiveGeneration(**json.loads(key.read_text()))
        except Exception:
            key.unlink()
    user = MP_STD_COT_USER_TEMPLATE.format(
        scenario=s.prompt, perspective_description=p.description,
    )
    output = call_chat(model, MP_STD_COT_SYSTEM, user, max_out=8000, seed=i)
    pg = PerspectiveGeneration(
        scenario_id=s.id, perspective_id=p.perspective_id,
        sample_idx=i, model=model, output=output,
        meta={"system": "std_cot", "round": "r0"},
    )
    key.write_text(json.dumps(asdict(pg), indent=2))
    return pg


# -- R1_std generation (rebuttal) --------------------------------------------

def r1_cache(model, sid, pid, idx):
    return OUT_DIR / f"debate_std_round1_{_safe(model)}_{sid}_{pid}_{idx:03d}.json"


def gen_r1(s: Scenario, p: Perspective, i: int, model: str,
           r0_by_persp: dict[str, PerspectiveGeneration]) -> PerspectiveGeneration:
    key = r1_cache(model, s.id, p.perspective_id, i)
    if key.exists():
        try:
            return PerspectiveGeneration(**json.loads(key.read_text()))
        except Exception:
            key.unlink()

    own = r0_by_persp.get(p.perspective_id)
    own_stmt = own.output if own and own.output else "[No statement available]"
    other_parts = []
    for pid, pg in r0_by_persp.items():
        if pid == p.perspective_id:
            continue
        lbl = pid.replace("_", " ").title()
        other_parts.append(f"[{lbl}]:\n{pg.output if pg and pg.output else '[No statement]'}")
    other_statements = "\n\n".join(other_parts) if other_parts else "[No other statements]"

    user = R1_USER_TEMPLATE.format(
        scenario=s.prompt,
        perspective_description=p.description,
        own_statement=own_stmt,
        other_statements=other_statements,
    )
    output = call_chat(model, MP_STD_COT_SYSTEM, user, max_out=8000, seed=i + 1000)
    pg = PerspectiveGeneration(
        scenario_id=s.id, perspective_id=p.perspective_id,
        sample_idx=i, model=model, output=output,
        meta={"system": "std_cot", "round": "r1"},
    )
    key.write_text(json.dumps(asdict(pg), indent=2))
    return pg


# -- R2_open_std generation (final position) ---------------------------------

def r2_cache(model, sid, pid, idx):
    return OUT_DIR / f"debate_std_round2_{_safe(model)}_{sid}_{pid}_{idx:03d}.json"


def gen_r2(s: Scenario, p: Perspective, i: int, model: str,
           r0_by_persp: dict, r1_by_persp: dict) -> PerspectiveGeneration:
    key = r2_cache(model, s.id, p.perspective_id, i)
    if key.exists():
        try:
            return PerspectiveGeneration(**json.loads(key.read_text()))
        except Exception:
            key.unlink()

    own_r0 = r0_by_persp.get(p.perspective_id)
    own_r1 = r1_by_persp.get(p.perspective_id)
    other_r1_parts = []
    for pid, pg_r1 in r1_by_persp.items():
        if pid == p.perspective_id:
            continue
        lbl = pid.replace("_", " ").title()
        other_r1_parts.append(
            f"[{lbl}]:\n{pg_r1.output if pg_r1 and pg_r1.output else '[No rebuttal]'}"
        )
    other_r1 = "\n\n".join(other_r1_parts) if other_r1_parts else "[No other rebuttals]"

    user = R2_OPEN_USER_TEMPLATE.format(
        scenario=s.prompt,
        perspective_description=p.description,
        own_r0=(own_r0.output if own_r0 and own_r0.output else "[No statement]"),
        own_r1=(own_r1.output if own_r1 and own_r1.output else "[No rebuttal]"),
        other_r1=other_r1,
    )
    output = call_chat(model, MP_STD_COT_SYSTEM, user, max_out=8000, seed=i + 3000)
    pg = PerspectiveGeneration(
        scenario_id=s.id, perspective_id=p.perspective_id,
        sample_idx=i, model=model, output=output,
        meta={"system": "std_cot", "round": "r2_open"},
    )
    key.write_text(json.dumps(asdict(pg), indent=2))
    return pg


# -- Open moderator pass (held constant) -------------------------------------

MODERATOR_MODEL = "gpt-4o-mini"


def open_mod_cache(gen_model, sid, idx):
    return OUT_DIR / f"debate_std_v2_consensus_{_safe(gen_model)}_{sid}_{idx:03d}.json"


def run_open_moderator(s: Scenario, idx: int, gen_model: str,
                       r0_outputs, r1_outputs, r2_outputs) -> dict:
    key = open_mod_cache(gen_model, s.id, idx)
    if key.exists():
        try:
            return json.loads(key.read_text())
        except json.JSONDecodeError:
            key.unlink()

    ps = PERSPECTIVES_BY_SCENARIO[s.id]
    parts = []
    for p in ps:
        lbl = p.perspective_id.replace("_", " ").title()
        r0 = r0_outputs.get((s.id, p.perspective_id, idx, gen_model))
        r1 = r1_outputs.get((s.id, p.perspective_id, idx, gen_model))
        r2 = r2_outputs.get((s.id, p.perspective_id, idx, gen_model))
        parts.append(
            f"=== {lbl} ===\n"
            f"[R0]:\n{(r0.output if r0 and r0.output else '[empty]')[:600]}\n\n"
            f"[R1]:\n{(r1.output if r1 and r1.output else '[empty]')[:600]}\n\n"
            f"[R2 open]:\n{(r2.output if r2 and r2.output else '[empty]')[:800]}"
        )
    transcript = "\n\n".join(parts)
    taxonomy_labels = ", ".join(f'"{k}"' for k in s.decision_taxonomy)
    user = OPEN_MODERATOR_USER_TEMPLATE.format(
        scenario=s.prompt, transcript=transcript, taxonomy_labels=taxonomy_labels,
    )
    seed = int(hashlib.sha1(f"std_open_mod|{s.id}|{idx}".encode()).hexdigest()[:8], 16)
    result = call_judge(MODERATOR_MODEL, OPEN_MODERATOR_SYSTEM, user, max_out=1000, seed=seed)
    for k, dv in [("consensus_reached", False), ("consensus_kind", "none"),
                   ("consensus_decision", None), ("novel_actions_proposed", []),
                   ("synthesis_label", None), ("synthesis_description", None),
                   ("points_of_disagreement", []), ("summary", "")]:
        result.setdefault(k, dv)
    result["gen_model"] = gen_model
    result["scenario_id"] = s.id
    result["sample_idx"] = idx
    result["has_synthesis"] = bool(result.get("consensus_reached")) or bool(result.get("synthesis_label"))
    key.write_text(json.dumps(result, indent=2))
    return result


# -- R3_std generation (synthesis acceptance) --------------------------------

def r3_cache(model, sid, pid, idx):
    return OUT_DIR / f"debate_std_v3_r3_{_safe(model)}_{sid}_{pid}_{idx:03d}.json"


def gen_r3(s, p, i, model, synth_label, synth_desc,
           r0_map, r1_map, r2_map) -> PerspectiveGeneration:
    key = r3_cache(model, s.id, p.perspective_id, i)
    if key.exists():
        try:
            return PerspectiveGeneration(**json.loads(key.read_text()))
        except Exception:
            key.unlink()

    parts = []
    for pp in PERSPECTIVES_BY_SCENARIO[s.id]:
        lbl = pp.perspective_id.replace("_", " ").title()
        r0 = r0_map.get(pp.perspective_id)
        r1 = r1_map.get(pp.perspective_id)
        r2 = r2_map.get(pp.perspective_id)
        parts.append(
            f"[{lbl}] R0: {(r0.output if r0 and r0.output else '[empty]')[:400]}\n"
            f"[{lbl}] R1: {(r1.output if r1 and r1.output else '[empty]')[:400]}\n"
            f"[{lbl}] R2: {(r2.output if r2 and r2.output else '[empty]')[:500]}"
        )
    transcript = "\n\n".join(parts)
    user = SYNTH_ACCEPT_USER_TEMPLATE.format(
        scenario=s.prompt,
        perspective_description=p.description,
        transcript=transcript,
        synthesis_label=synth_label,
        synthesis_description=synth_desc,
    )
    output = call_chat(model, MP_STD_COT_SYSTEM, user, max_out=8000, seed=i + 4000)
    pg = PerspectiveGeneration(
        scenario_id=s.id, perspective_id=p.perspective_id,
        sample_idx=i, model=model, output=output,
        meta={"system": "std_cot", "round": "r3"},
    )
    key.write_text(json.dumps(asdict(pg), indent=2))
    return pg


# -- R3 decision extraction (held constant) ----------------------------------

def r3_dec_cache(model, sid, pid, idx):
    return OUT_DIR / f"debate_std_v3_dec_{_safe(model)}_{sid}_{pid}_{idx:03d}.json"


def extract_r3_decision(pg, s, synth_label, synth_desc) -> dict:
    key = r3_dec_cache(pg.model, pg.scenario_id, pg.perspective_id, pg.sample_idx)
    if key.exists():
        try:
            return json.loads(key.read_text())
        except json.JSONDecodeError:
            key.unlink()
    if not pg.output or not pg.output.strip():
        result = {"response_type": "UNKNOWN", "decision": "TRUNCATED",
                  "modification_description": None, "rejection_reason": None,
                  "decision_confidence": 0.0}
        key.write_text(json.dumps(result, indent=2))
        return result
    taxonomy_text = "\n".join(f"- `{lbl}`: {desc}" for lbl, desc in s.decision_taxonomy.items())
    user = SYNTH_R3_EXTRACT_USER_TEMPLATE.format(
        scenario=s.prompt, synthesis_label=synth_label,
        synthesis_description=synth_desc, taxonomy=taxonomy_text,
        response=pg.output,
    )
    seed = int(hashlib.sha1(
        f"std_r3dec|{pg.scenario_id}|{pg.perspective_id}|{pg.sample_idx}".encode()
    ).hexdigest()[:8], 16)
    result = call_judge(MODERATOR_MODEL, SYNTH_R3_EXTRACT_SYSTEM, user, max_out=400, seed=seed)
    valid = {"ACCEPT", "ACCEPT_WITH_MODIFICATION", "REJECT", "UNKNOWN"}
    if result.get("response_type") not in valid:
        result["response_type"] = "UNKNOWN"
    result.setdefault("modification_description", None)
    result.setdefault("rejection_reason", None)
    result.setdefault("decision_confidence", 0.5)
    key.write_text(json.dumps(result, indent=2))
    return result


# -- Synthesis final-mod pass (held constant) --------------------------------

def synth_final_cache(gen_model, sid, idx):
    return OUT_DIR / f"debate_std_v3_final_mod_{_safe(gen_model)}_{sid}_{idx:03d}.json"


def run_synth_final_mod(s, idx, gen_model, synth_label, synth_desc, r3_outputs) -> dict:
    key = synth_final_cache(gen_model, s.id, idx)
    if key.exists():
        try:
            return json.loads(key.read_text())
        except json.JSONDecodeError:
            key.unlink()
    parts = []
    for p in PERSPECTIVES_BY_SCENARIO[s.id]:
        lbl = p.perspective_id.replace("_", " ").title()
        cached = r3_outputs.get((s.id, p.perspective_id, idx, gen_model))
        if cached is None:
            parts.append(f"[{lbl}]: [No response]")
            continue
        text = (cached.output or "[empty]")[:600]
        parts.append(f"[{lbl}]:\n{text}")
    user = SYNTH_FINAL_MOD_USER_TEMPLATE.format(
        scenario=s.prompt, synthesis_label=synth_label,
        synthesis_description=synth_desc, responses="\n\n".join(parts),
    )
    seed = int(hashlib.sha1(f"std_synth_final|{s.id}|{idx}".encode()).hexdigest()[:8], 16)
    result = call_judge(MODERATOR_MODEL, SYNTH_FINAL_MOD_SYSTEM, user, max_out=600, seed=seed)
    for k, dv in [("consensus_reached", False), ("partial_convergence", False),
                   ("unanimous_accept", False), ("accept_count", 0),
                   ("modify_count", 0), ("reject_count", 0),
                   ("final_synthesis_label", None),
                   ("final_synthesis_description", None),
                   ("remaining_disagreements", [])]:
        result.setdefault(k, dv)
    result["gen_model"] = gen_model
    result["scenario_id"] = s.id
    result["sample_idx"] = idx
    result["synthesis_label_presented"] = synth_label
    key.write_text(json.dumps(result, indent=2))
    return result


# -- Integration moderator pass (held constant) ------------------------------

def integ_cache(gen_model, sid, idx):
    return OUT_DIR / f"debate_std_v4_integ_{_safe(gen_model)}_{sid}_{idx:03d}.json"


def run_integration(s, idx, gen_model, synth_label, synth_desc,
                    r3_outputs, r3_dec_by_persp) -> dict:
    key = integ_cache(gen_model, s.id, idx)
    if key.exists():
        try:
            return json.loads(key.read_text())
        except json.JSONDecodeError:
            key.unlink()
    parts = []
    for p in PERSPECTIVES_BY_SCENARIO[s.id]:
        lbl = p.perspective_id.replace("_", " ").title()
        dec = r3_dec_by_persp.get((s.id, p.perspective_id, idx, gen_model), {})
        rt = dec.get("response_type", "UNKNOWN")
        mod_desc = dec.get("modification_description") or ""
        cached = r3_outputs.get((s.id, p.perspective_id, idx, gen_model))
        full = (cached.output if cached and cached.output else "")[:500]
        parts.append(
            f"[{lbl} ({p.perspective_id})]: {rt}\n"
            f"Modification requested: {mod_desc}\n"
            f"Full response (excerpt): {full}"
        )
    user = INTEG_MOD_USER_TEMPLATE.format(
        scenario=s.prompt, synthesis_label=synth_label,
        synthesis_description=synth_desc,
        modification_requests="\n\n".join(parts),
    )
    seed = int(hashlib.sha1(f"std_integ|{s.id}|{idx}".encode()).hexdigest()[:8], 16)
    result = call_judge(MODERATOR_MODEL, INTEG_MOD_SYSTEM, user, max_out=600, seed=seed)
    result.setdefault("integrated_label", synth_label + "_integrated")
    result.setdefault("integrated_description", synth_desc)
    result.setdefault("modifications_addressed", [])
    result.setdefault("modifications_unaddressed", [])
    result.setdefault("integration_rationale", "")
    result["gen_model"] = gen_model
    result["scenario_id"] = s.id
    result["sample_idx"] = idx
    result["synthesis_label_presented"] = synth_label
    key.write_text(json.dumps(result, indent=2))
    return result


# -- R4_std generation (binary vote) -----------------------------------------

def r4_cache(model, sid, pid, idx):
    return OUT_DIR / f"debate_std_v4_r4_{_safe(model)}_{sid}_{pid}_{idx:03d}.json"


def gen_r4(s, p, i, model, integ_label, integ_desc, integ_rationale) -> PerspectiveGeneration:
    key = r4_cache(model, s.id, p.perspective_id, i)
    if key.exists():
        try:
            return PerspectiveGeneration(**json.loads(key.read_text()))
        except Exception:
            key.unlink()
    user = R4_USER_TEMPLATE.format(
        scenario=s.prompt,
        perspective_description=p.description,
        integrated_label=integ_label,
        integrated_description=integ_desc,
        integration_rationale=integ_rationale,
    )
    output = call_chat(model, MP_STD_COT_SYSTEM, user, max_out=2000, seed=i + 5000)
    pg = PerspectiveGeneration(
        scenario_id=s.id, perspective_id=p.perspective_id,
        sample_idx=i, model=model, output=output,
        meta={"system": "std_cot", "round": "r4"},
    )
    key.write_text(json.dumps(asdict(pg), indent=2))
    return pg


# -- R4 final moderator (held constant) --------------------------------------

def r4_final_cache(gen_model, sid, idx):
    return OUT_DIR / f"debate_std_v4_final_{_safe(gen_model)}_{sid}_{idx:03d}.json"


def run_r4_final(s, idx, gen_model, integ_label, integ_desc, r4_outputs) -> dict:
    key = r4_final_cache(gen_model, s.id, idx)
    if key.exists():
        try:
            return json.loads(key.read_text())
        except json.JSONDecodeError:
            key.unlink()
    parts = []
    for p in PERSPECTIVES_BY_SCENARIO[s.id]:
        lbl = p.perspective_id.replace("_", " ").title()
        cached = r4_outputs.get((s.id, p.perspective_id, idx, gen_model))
        if cached is None:
            parts.append(f"[{lbl}]: [No vote]")
            continue
        text = (cached.output or "[empty]")[:400]
        parts.append(f"[{lbl}]:\n{text}")
    user = R4_FINAL_MOD_USER_TEMPLATE.format(
        scenario=s.prompt, integrated_label=integ_label,
        integrated_description=integ_desc, votes="\n\n".join(parts),
    )
    seed = int(hashlib.sha1(f"std_r4_final|{s.id}|{idx}".encode()).hexdigest()[:8], 16)
    result = call_judge(MODERATOR_MODEL, R4_FINAL_MOD_SYSTEM, user, max_out=400, seed=seed)
    for k, dv in [("consensus_reached", False), ("majority_accept", False),
                   ("accept_count", 0), ("reject_count", 0), ("unanimous", False),
                   ("final_position_label", None), ("primary_rejection_reason", None)]:
        result.setdefault(k, dv)
    result["gen_model"] = gen_model
    result["scenario_id"] = s.id
    result["sample_idx"] = idx
    result["integrated_label"] = integ_label
    key.write_text(json.dumps(result, indent=2))
    return result


# -- Orchestration -----------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=10,
                   help="Samples per (scenario, perspective, generator) cell")
    p.add_argument("--generators", default=None,
                   help="Comma-separated generator deployment names")
    p.add_argument("--scenarios", default=None,
                   help="Comma-separated scenario ids (default all)")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    n_samples = args.samples
    gen_models = args.generators.split(",") if args.generators else \
        [m.strip() for m in os.environ.get("AZURE_AI_MODELS_GENERATION",
                                            "gpt-5.4-nano,gpt-4o").split(",") if m.strip()]
    scenario_ids = args.scenarios.split(",") if args.scenarios else [s.id for s in SCENARIOS]
    selected_scenarios = [s for s in SCENARIOS if s.id in scenario_ids]
    workers = args.workers
    print(f"Models: {gen_models} | scenarios: {[s.id for s in selected_scenarios]} | samples/cell: {n_samples} | workers: {workers}")

    cell_tasks = [
        (s, p, i, m)
        for m in gen_models
        for s in selected_scenarios
        for p in PERSPECTIVES_BY_SCENARIO[s.id]
        for i in range(n_samples)
    ]
    print(f"R0/R1/R2 tasks per round: {len(cell_tasks)}")

    # ---------- Round 0 (standard-CoT opening) ----------
    r0_outputs = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(gen_r0, s, p, i, m): (s, p, i, m) for (s, p, i, m) in cell_tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="R0_std"):
            s, p, i, m = futs[fut]
            r0_outputs[(s.id, p.perspective_id, i, m)] = fut.result()

    # ---------- Round 1 (rebuttal) ----------
    r1_outputs = {}

    def _gen_r1(args):
        s, p, i, m = args
        ps = PERSPECTIVES_BY_SCENARIO[s.id]
        r0_map = {pp.perspective_id: r0_outputs.get((s.id, pp.perspective_id, i, m)) for pp in ps}
        return gen_r1(s, p, i, m, r0_map)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_gen_r1, t): t for t in cell_tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="R1_std"):
            s, p, i, m = futs[fut]
            r1_outputs[(s.id, p.perspective_id, i, m)] = fut.result()

    # ---------- Round 2 open ----------
    r2_outputs = {}

    def _gen_r2(args):
        s, p, i, m = args
        ps = PERSPECTIVES_BY_SCENARIO[s.id]
        r0_map = {pp.perspective_id: r0_outputs.get((s.id, pp.perspective_id, i, m)) for pp in ps}
        r1_map = {pp.perspective_id: r1_outputs.get((s.id, pp.perspective_id, i, m)) for pp in ps}
        return gen_r2(s, p, i, m, r0_map, r1_map)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_gen_r2, t): t for t in cell_tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="R2_open_std"):
            s, p, i, m = futs[fut]
            r2_outputs[(s.id, p.perspective_id, i, m)] = fut.result()

    # ---------- Open moderator pass ----------
    open_mod_tasks = [
        (s, i, m)
        for m in gen_models for s in selected_scenarios for i in range(n_samples)
    ]
    open_mod_results = []

    def _omod(args):
        s, i, m = args
        return run_open_moderator(s, i, m, r0_outputs, r1_outputs, r2_outputs)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_omod, t): t for t in open_mod_tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="open_moderator"):
            open_mod_results.append(fut.result())

    # Save the open-mod outputs
    omod_csv = OUT_DIR / "debate_std_v2_consensus.csv"
    with open(omod_csv, "w", newline="") as fh:
        cols = ["scenario_id", "gen_model", "sample_idx", "consensus_reached",
                "consensus_kind", "consensus_decision", "synthesis_label",
                "synthesis_description", "has_synthesis", "summary"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in open_mod_results:
            w.writerow(r)
    print(f"Saved {omod_csv}")

    has_synth_cells = [r for r in open_mod_results if r.get("synthesis_label")]
    print(f"  Synthesis emergence: {len(has_synth_cells)}/{len(open_mod_results)} cells")

    # ---------- Round 3 (synthesis acceptance) on synth cells only ----------
    r3_outputs = {}
    r3_tasks = []
    for r in has_synth_cells:
        s = SCENARIO_BY_ID[r["scenario_id"]]
        m = r["gen_model"]
        i = int(r["sample_idx"])
        synth_lbl = r["synthesis_label"]
        synth_desc = r.get("synthesis_description") or synth_lbl
        for p in PERSPECTIVES_BY_SCENARIO[s.id]:
            r3_tasks.append((s, p, i, m, synth_lbl, synth_desc))

    def _gen_r3(args):
        s, p, i, m, sl, sd = args
        ps = PERSPECTIVES_BY_SCENARIO[s.id]
        r0_map = {pp.perspective_id: r0_outputs.get((s.id, pp.perspective_id, i, m)) for pp in ps}
        r1_map = {pp.perspective_id: r1_outputs.get((s.id, pp.perspective_id, i, m)) for pp in ps}
        r2_map = {pp.perspective_id: r2_outputs.get((s.id, pp.perspective_id, i, m)) for pp in ps}
        return gen_r3(s, p, i, m, sl, sd, r0_map, r1_map, r2_map)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_gen_r3, t): t for t in r3_tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="R3_std"):
            s, p, i, m, sl, sd = futs[fut]
            r3_outputs[(s.id, p.perspective_id, i, m)] = fut.result()

    # ---------- R3 decision extraction ----------
    r3_dec_by_persp = {}
    r3_dec_rows = []

    def _ext(args):
        s, p, i, m, sl, sd = args
        pg = r3_outputs.get((s.id, p.perspective_id, i, m))
        if pg is None:
            return None
        d = extract_r3_decision(pg, s, sl, sd)
        return (s.id, p.perspective_id, i, m, sl, d)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_ext, t) for t in r3_tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="R3 dec"):
            res = fut.result()
            if res is None:
                continue
            sid, pid, i, m, sl, d = res
            r3_dec_by_persp[(sid, pid, i, m)] = d
            role_lookup = {p.perspective_id: p.families.get("decision_role", "?")
                           for p in PERSPECTIVES_BY_SCENARIO[sid]}
            r3_dec_rows.append({
                "scenario_id": sid, "perspective_id": pid, "sample_idx": i,
                "gen_model": m, "synthesis_label": sl,
                "decision_role": role_lookup.get(pid, "?"),
                **d,
            })

    r3dec_csv = OUT_DIR / "debate_std_v3_r3_decisions.csv"
    with open(r3dec_csv, "w", newline="") as fh:
        cols = ["scenario_id", "perspective_id", "sample_idx", "gen_model",
                "decision_role", "synthesis_label", "response_type", "decision",
                "modification_description", "rejection_reason", "decision_confidence"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in r3_dec_rows:
            w.writerow(row)
    print(f"Saved {r3dec_csv}")

    # ---------- Synthesis final moderator pass ----------
    synth_mod_results = []
    synth_mod_tasks = []
    seen = set()
    for r in has_synth_cells:
        k = (r["scenario_id"], r["gen_model"], int(r["sample_idx"]))
        if k in seen:
            continue
        seen.add(k)
        s = SCENARIO_BY_ID[r["scenario_id"]]
        synth_mod_tasks.append((s, int(r["sample_idx"]), r["gen_model"],
                                 r["synthesis_label"],
                                 r.get("synthesis_description") or r["synthesis_label"]))

    def _smod(args):
        s, i, m, sl, sd = args
        return run_synth_final_mod(s, i, m, sl, sd, r3_outputs)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_smod, t): t for t in synth_mod_tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="synth_final_mod"):
            synth_mod_results.append(fut.result())

    smod_csv = OUT_DIR / "debate_std_v3_final_mod.csv"
    with open(smod_csv, "w", newline="") as fh:
        cols = ["scenario_id", "gen_model", "sample_idx", "synthesis_label_presented",
                "consensus_reached", "partial_convergence", "unanimous_accept",
                "accept_count", "modify_count", "reject_count",
                "final_synthesis_label", "final_synthesis_description"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in synth_mod_results:
            w.writerow(row)
    print(f"Saved {smod_csv}")

    # ---------- Integration moderator pass ----------
    integ_results = []

    def _integ(args):
        s, i, m, sl, sd = args
        return run_integration(s, i, m, sl, sd, r3_outputs, r3_dec_by_persp)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_integ, t): t for t in synth_mod_tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="integration"):
            integ_results.append(fut.result())

    integ_csv = OUT_DIR / "debate_std_v4_integrated_proposals.csv"
    with open(integ_csv, "w", newline="") as fh:
        cols = ["scenario_id", "gen_model", "sample_idx", "synthesis_label_presented",
                "integrated_label", "integrated_description",
                "modifications_addressed", "modifications_unaddressed",
                "integration_rationale"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in integ_results:
            row_out = {**row}
            for k in ("modifications_addressed", "modifications_unaddressed"):
                v = row_out.get(k, [])
                if isinstance(v, list):
                    row_out[k] = json.dumps(v)
            w.writerow(row_out)
    print(f"Saved {integ_csv}")

    # ---------- R4 binary vote ----------
    r4_outputs = {}
    r4_tasks = []
    integ_by_cell = {}
    for r in integ_results:
        k = (r["scenario_id"], r["gen_model"], int(r["sample_idx"]))
        integ_by_cell[k] = r
        s = SCENARIO_BY_ID[r["scenario_id"]]
        il = r.get("integrated_label") or ""
        idesc = r.get("integrated_description") or ""
        irat = r.get("integration_rationale") or ""
        if not il or il in ("None", "nan"):
            continue
        for p in PERSPECTIVES_BY_SCENARIO[s.id]:
            r4_tasks.append((s, p, int(r["sample_idx"]), r["gen_model"], il, idesc, irat))

    def _gen_r4(args):
        s, p, i, m, il, idesc, irat = args
        return gen_r4(s, p, i, m, il, idesc, irat)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_gen_r4, t): t for t in r4_tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="R4_std"):
            s, p, i, m, *_ = futs[fut]
            r4_outputs[(s.id, p.perspective_id, i, m)] = fut.result()

    # ---------- R4 final moderator ----------
    r4_final_results = []
    r4_final_tasks = []
    seen_r4f = set()
    for r in integ_results:
        k = (r["scenario_id"], r["gen_model"], int(r["sample_idx"]))
        if k in seen_r4f:
            continue
        seen_r4f.add(k)
        il = r.get("integrated_label") or ""
        idesc = r.get("integrated_description") or ""
        if not il or il in ("None", "nan"):
            continue
        s = SCENARIO_BY_ID[r["scenario_id"]]
        r4_final_tasks.append((s, int(r["sample_idx"]), r["gen_model"], il, idesc))

    def _r4f(args):
        s, i, m, il, idesc = args
        return run_r4_final(s, i, m, il, idesc, r4_outputs)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_r4f, t): t for t in r4_final_tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="r4_final_mod"):
            r4_final_results.append(fut.result())

    r4f_csv = OUT_DIR / "debate_std_v4_final.csv"
    with open(r4f_csv, "w", newline="") as fh:
        cols = ["scenario_id", "gen_model", "sample_idx", "integrated_label",
                "consensus_reached", "majority_accept", "accept_count",
                "reject_count", "unanimous", "final_position_label",
                "primary_rejection_reason"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in r4_final_results:
            w.writerow(row)
    print(f"Saved {r4f_csv}")

    # ---------- R4 votes per perspective (for role-concentration analysis) ----------
    r4_votes_csv = OUT_DIR / "debate_std_v4_r4_votes.csv"
    with open(r4_votes_csv, "w", newline="") as fh:
        cols = ["scenario_id", "perspective_id", "sample_idx", "gen_model",
                "vote", "decision_role", "integrated_label"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for (sid, pid, i, m), pg in r4_outputs.items():
            text = (pg.output or "").strip().upper()
            vote = "UNKNOWN"
            if text:
                first_line = text.split("\n", 1)[0].strip()
                first_word = first_line.split()[0] if first_line.split() else ""
                if first_word == "ACCEPT":
                    vote = "ACCEPT"
                elif first_word in ("REJECT", "REJECTED"):
                    vote = "REJECT"
                elif "ACCEPT" in first_line and "REJECT" not in first_line:
                    vote = "ACCEPT"
                elif "REJECT" in first_line:
                    vote = "REJECT"
            role_lookup = {p.perspective_id: p.families.get("decision_role", "?")
                           for p in PERSPECTIVES_BY_SCENARIO[sid]}
            integ_label = integ_by_cell.get((sid, m, i), {}).get("integrated_label", "")
            w.writerow({
                "scenario_id": sid, "perspective_id": pid, "sample_idx": i,
                "gen_model": m, "vote": vote,
                "decision_role": role_lookup.get(pid, "?"),
                "integrated_label": integ_label,
            })
    print(f"Saved {r4_votes_csv}")

    # ---------- Headline summary ----------
    n_total = len(open_mod_results)
    n_open_consensus = sum(1 for r in open_mod_results if r.get("consensus_reached"))
    n_synth = len(has_synth_cells)
    n_synth_full = sum(1 for r in synth_mod_results if r.get("consensus_reached"))
    n_synth_partial = sum(1 for r in synth_mod_results if r.get("partial_convergence"))
    n_r4 = len(r4_final_results)
    n_r4_full = sum(1 for r in r4_final_results if r.get("consensus_reached"))
    n_r4_majority = sum(1 for r in r4_final_results if r.get("majority_accept"))

    summary_path = OUT_DIR / "debate_std_arc_summary.json"
    summary = {
        "n_total_cells": n_total,
        "open_action_space": {
            "n_consensus": n_open_consensus,
            "rate": n_open_consensus / n_total if n_total else 0.0,
            "n_synthesis_emergence": n_synth,
            "synthesis_emergence_rate": n_synth / n_total if n_total else 0.0,
        },
        "synthesis_presentation": {
            "n_cells": len(synth_mod_results),
            "n_full": n_synth_full,
            "rate_full": n_synth_full / max(1, len(synth_mod_results)),
            "n_partial": n_synth_partial,
            "rate_partial": n_synth_partial / max(1, len(synth_mod_results)),
        },
        "integrative_negotiation": {
            "n_cells": n_r4,
            "n_full": n_r4_full,
            "rate_full": n_r4_full / max(1, n_r4),
            "n_majority": n_r4_majority,
            "rate_majority": n_r4_majority / max(1, n_r4),
        },
        "narrative_arm_for_comparison": {
            "open_action_space_rate_full": 0.09,
            "synthesis_presentation_rate_full": 0.0,
            "synthesis_presentation_rate_partial": 1.0,
            "integrative_negotiation_rate_full": 0.951,
            "integrative_negotiation_rate_majority": 1.0,
        },
        "models": gen_models,
        "scenarios": [s.id for s in selected_scenarios],
        "samples_per_cell": n_samples,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 78)
    print("STANDARD-CoT DEBATE ABLATION HEADLINE")
    print("=" * 78)
    print(f"Cells:                 {n_total}  (scenarios={len(selected_scenarios)}, "
          f"samples={n_samples}, generators={len(gen_models)})")
    print(f"Open R2 consensus:     {n_open_consensus}/{n_total} = "
          f"{(n_open_consensus/max(1,n_total)):.1%}  "
          f"(narrative arm: 9.0%)")
    print(f"Synthesis emergence:   {n_synth}/{n_total} = "
          f"{(n_synth/max(1,n_total)):.1%}  (narrative arm: 82%)")
    print(f"R3 full consensus:     {n_synth_full}/{len(synth_mod_results)} = "
          f"{(n_synth_full/max(1,len(synth_mod_results))):.1%}  (narrative arm: 0%)")
    print(f"R3 partial conv.:      {n_synth_partial}/{len(synth_mod_results)} = "
          f"{(n_synth_partial/max(1,len(synth_mod_results))):.1%}  (narrative arm: 100%)")
    print(f"R4 full consensus:     {n_r4_full}/{n_r4} = "
          f"{(n_r4_full/max(1,n_r4)):.1%}  (narrative arm: 95.1%)  <-- HEADLINE")
    print(f"R4 majority accept:    {n_r4_majority}/{n_r4} = "
          f"{(n_r4_majority/max(1,n_r4)):.1%}  (narrative arm: 100%)")
    print("=" * 78)


if __name__ == "__main__":
    main()
