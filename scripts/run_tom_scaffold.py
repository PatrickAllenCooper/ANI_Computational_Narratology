"""
scripts/run_tom_scaffold.py -- does the narrative scaffold help higher-order
Theory-of-Mind reasoning? Evaluates baseline_io / standard_cot /
narrative_cot_v3_tom on:
  - HiToM belief_choice questions (accuracy, stratified by ToM order 0-4)
  - FANToM belief_choice questions (accuracy, first-order vs second-order)
  - FANToM list-type questions (exact-set-match accuracy, comparable to the
    paper, + F1 as supplementary; answerability + info_accessibility --
    these are first-order access questions, not nested-order beliefs, so
    they're reported separately from the order-stratified accuracy table)
  - FANToM binary Y/N accessibility questions (accuracy + weighted F1,
    comparable to the paper's AnswerabilityQ[Y/N]/InfoAccessQ[Y/N])
  - FANToM free-form belief generation, BeliefQ[Dist.]-style (accuracy +
    token F1; graded by embedding-similarity classification against the
    correct/wrong reference via Azure text-embedding-3-small rather than
    the paper's local sentence-transformers model -- not bit-for-bit
    comparable to the paper's numbers, only across our own conditions)

narrative_cot_v3_tom is an adaptation of scripts/run_phase1_quartet.py's
narrative_cot_v3 (the paper's "best generalizing" optimized variant) for QA
over nested beliefs rather than ethical-dilemma decisions. It preserves
v3's five-section structure and rhetorical pattern (bold headers, "cast a
wide net", "do not hedge into vagueness") section-for-section:
  v3 section                                -> ToM adaptation
  Who is deciding / their stakes            -> Who the question asks about, their vantage point
  Who else is affected (cast a wide net)    -> Who else's presence/absence matters
  What could go wrong or remain unknown     -> What each link in the belief chain could/couldn't know
  What the realistic options are / ripple   -> How the belief chain resolves, one mind at a time
  The call                                  -> The answer
This is a rewrite, not the verbatim ethical-dilemma text -- treat results as
a test of "a scaffold in the spirit of v3", and review NARRATIVE_COT_V3_TOM
below before citing it as v3 itself.

Within each of the 4 task formats, all three conditions share that task's
header (TOM_TASK_HEADERS[task]) so every arm knows what the question is
asking; the only thing that differs between arms is reasoning structure,
not task framing. The header and the narrative_cot_v3_tom scaffold body
ARE keyed per task format, though (see the block above CONDITIONS_BY_TASK)
-- a nested belief-chain question needs different framing than a single-hop
binary access check.

Usage:
  python -m scripts.run_tom_scaffold --smoke
  python -m scripts.run_tom_scaffold --model gpt-5.4-nano --n-per-order 15 --workers 6
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.generators import generate
from scripts.generators import _get_openai_client
from scripts.load_hitom import load_hitom
from scripts.load_fantom import load_fantom, load_fantom_list, load_fantom_binary, load_fantom_gen
from scripts.run_phase1_quartet import PROMPTS, _safe

OUT_DIR = Path("./divergence_study_outputs")
OUT_DIR.mkdir(exist_ok=True)
CHOICE_CSV = OUT_DIR / "tom_scaffold_results.csv"
LIST_CSV = OUT_DIR / "tom_scaffold_list_results.csv"
BINARY_CSV = OUT_DIR / "tom_scaffold_binary_results.csv"
GEN_CSV = OUT_DIR / "tom_scaffold_gen_results.csv"
EMBED_MODEL = "text-embedding-3-small"

# Cross-run sample tracking, so each run draws items disjoint from every
# prior run's selection for the same bucket ("select a different N next
# time"). Results across runs combine for free: CSVs are append-only and
# item_ids never repeat within a bucket, so a cumulative read of the CSV
# is automatically the pooled sample across all runs to date.
TRACKER_PATH = OUT_DIR / "tom_scaffold_sample_tracker.json"
# Live progress, so a status check mid-run doesn't have to wait for
# completion -- read this file directly at any time.
PROGRESS_PATH = OUT_DIR / "tom_scaffold_progress.json"

# ---------------------------------------------------------------------------
# Task-specific prompts. The 4 task formats ask genuinely different
# questions (nested belief chain vs. single-hop access vs. enumerate a set),
# so the shared task header is keyed per task -- edit the entries below
# directly. build_conditions() prepends TOM_TASK_HEADERS[task] to ALL THREE
# conditions (baseline_io, standard_cot, narrative_cot_v3_tom) uniformly,
# so a header edit here applies to every condition for that task, not just
# the narrative one.
#
# NARRATIVE_COT_V3_TOM_BODY (below) is intentionally left shared across all
# 4 tasks -- its "who's present/absent, trace access, commit to an answer"
# structure is judged general enough to not need per-task rewriting.
# baseline_io / standard_cot bodies are the shared repo-wide prompts from
# run_phase1_quartet.py (see PROMPTS import) and are untouched here too.
#
# Drafts below -- edit freely, nothing else in the file needs to change to
# pick up edits.
# ---------------------------------------------------------------------------

TOM_TASK_HEADERS = {
    # HiToM (nested "where does A think B thinks...is?") + FANToM belief
    # multiple-choice: pick the correct lettered option.
    "belief_choice": (
        "This is a theory-of-mind test. Answer the question about where "
        "something is, or what someone believes or knows, based on the "
        "following story or conversation."
    ),
    # Same underlying belief questions as belief_choice, answered in free
    # text instead of a lettered choice.
    "gen": (
        "This is a theory-of-mind test. Answer the question about where "
        "something is, or what someone believes or knows, based on the "
        "following story or conversation."
    ),
    # FANToM answerability/info-accessibility LIST questions: "list all
    # characters who know/can answer X" -- output is a SET of names, and
    # this is a first-order access question (no nested belief chain).
    "list": (
        "This is a theory-of-mind test. The following conversation "
        "involves several people who were not all present for the same "
        "parts of it. Answer the question by identifying every person who "
        "currently has direct access to the specific piece of information "
        "being asked about -- not everyone in the conversation, only those "
        "who personally witnessed it or were told it."
    ),
    # FANToM answerability/info-accessibility BINARY questions: "does
    # character X know/can answer Y?" -- a single named person, yes/no,
    # first-order access only (no nested belief chain).
    "binary": (
        "This is a theory-of-mind test. The following conversation "
        "involves several people who were not all present for the same "
        "parts of it. Answer the question by judging whether the ONE "
        "specific person it names currently has direct access to the "
        "specific piece of information being asked about."
    ),
}

# Shared across all 4 task formats (unlike TOM_TASK_HEADERS) -- judged
# general enough not to need per-task rewriting.
NARRATIVE_COT_V3_TOM_BODY = (
    "Before committing to an answer, think carefully and concisely through "
    "the following.\n\n"
    "**Who the question is asking about, and what vantage point they "
    "have.** Identify the specific person named in the question, or if the "
    "question nests belief, the main vantage point. State "
    "in one or two sentences their role in the story or conversation and "
    "when they were present.\n\n"
    "**Who else's presence or absence matters.** Cast a wide net. Start "
    "with everyone present when the relevant fact was established or the "
    "object was last moved, then move outward: anyone who left before it "
    "happened, anyone who arrived only afterward, and anyone who was told "
    "about it later by someone else. For each person relevant to the chain "
    "in the question, name concretely when they were present or absent -- "
    "not that they ‘might know,’ but the exact moment their "
    "access began or ended.\n\n"
    "**What each link in the chain could and could not have known.** Where "
    "the question nests one person's belief about another's belief, treat "
    "every link as its own vantage point: person A's belief about person "
    "B's belief depends only on what A themselves witnessed about B's "
    "actions or absence, never on the ground truth A had no way to "
    "observe. Name the precise gap in access at each link, and flag any "
    "link where a wrong assumption would flip the final answer.\n\n"
    "**How the chain resolves.** Trace the belief chain in the order the "
    "question asks it, one mind at a time, from the outermost person "
    "inward. At each step, update only from what that specific person had "
    "access to.\n\n"
    "**The answer.** Commit to a specific answer. Justify it by direct "
    "reference to the access chain you traced above. Do not hedge into "
    "vagueness -- give the exact answer the question is asking for.\n\n"
    "Throughout, write concisely -- one sharp sentence per person's "
    "access, not a paragraph. Work through all sections before giving "
    "your final answer."
)


def build_conditions(task: str) -> dict[str, str]:
    header = TOM_TASK_HEADERS[task]
    return {
        "baseline_io": f"{header}\n\n{PROMPTS['baseline_io']}",
        "standard_cot": f"{header}\n\n{PROMPTS['standard_cot']}",
        "narrative_cot_v3_tom": f"{header}\n\n{NARRATIVE_COT_V3_TOM_BODY}",
    }


CONDITIONS_BY_TASK = {task: build_conditions(task) for task in TOM_TASK_HEADERS}

ANSWER_INSTRUCTION = (
    "\n\nEnd your response with a new line formatted exactly as:\nAnswer: <LETTER>"
)
LIST_INSTRUCTION = (
    "\n\nEnd your response with a new line formatted exactly as:\n"
    "Aware: <comma-separated character names, or None>"
)
BINARY_INSTRUCTION = (
    "\n\nEnd your response with a new line formatted exactly as:\nAnswer: <Yes or No>"
)

_ANSWER_RE = re.compile(r"answer\s*[:=]?\s*\(?([A-O])\)?", re.IGNORECASE)
_AWARE_RE = re.compile(r"aware\s*:\s*(.+)", re.IGNORECASE)
_YESNO_RE = re.compile(r"answer\s*[:=]?\s*\(?(yes|no)\)?", re.IGNORECASE)


def extract_letter(text: str) -> str | None:
    matches = _ANSWER_RE.findall(text or "")
    if matches:
        return matches[-1].upper()
    for line in reversed((text or "").strip().splitlines()):
        line = line.strip()
        m = re.fullmatch(r"\(?([A-O])\)?[.:]?", line)
        if m:
            return m.group(1).upper()
    return None


def extract_names(text: str, universe: list[str]) -> set[str]:
    m = list(_AWARE_RE.finditer(text or ""))
    tail = m[-1].group(1) if m else (text or "")
    found = set()
    for name in universe:
        if re.search(rf"\b{re.escape(name)}\b", tail, re.IGNORECASE):
            found.add(name)
    return found


def extract_yesno(text: str) -> str | None:
    matches = _YESNO_RE.findall(text or "")
    if matches:
        return matches[-1].lower()
    for line in reversed((text or "").strip().splitlines()):
        line = line.strip().lower().rstrip(".")
        if line in ("yes", "no"):
            return line
    return None


def token_f1(gold: str, pred: str) -> float:
    """Word-overlap F1, same definition as the FANToM paper's eval_fantom.py
    FantomEvalAgent.compute_f1 (Counter intersection over whitespace tokens)."""
    from collections import Counter
    gold_toks, pred_toks = gold.split(), pred.split()
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0 or not pred_toks or not gold_toks:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def _embed(texts: list[str]) -> list[list[float]]:
    client = _get_openai_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def _build_choice_prompt(context: str, question: str, choices: list[str]) -> str:
    choices_block = "\n".join(choices)
    return f"{context}\n\n{question}\n\n{choices_block}{ANSWER_INSTRUCTION}"


def _build_list_prompt(context: str, question: str) -> str:
    return f"{context}\n\n{question}{LIST_INSTRUCTION}"


def _build_binary_prompt(context: str, question: str) -> str:
    return f"{context}\n\n{question}{BINARY_INSTRUCTION}"


def _build_gen_prompt(context: str, question: str) -> str:
    return f"{context}\n\n{question}\n\nAnswer in your own words, in 1-3 sentences."


def _correct_letter_hitom(item: dict) -> str:
    for c in item["choices"]:
        letter, _, text = c.partition(". ")
        if text.strip() == item["answer"].strip():
            return letter.strip()
    raise ValueError(f"answer not found in choices for {item['id']}")


def _gen_cache_path(kind: str, dataset: str, model: str, condition: str, item_id: str) -> Path:
    return OUT_DIR / f"tom_gen_{kind}_{dataset}_{_safe(model)}_{condition}_{_safe(item_id)}.json"


# ---------------------------------------------------------------------------
# Cross-run sample tracking ("shuffle and select N, never repeat")
# ---------------------------------------------------------------------------

def _load_tracker() -> dict:
    if TRACKER_PATH.exists():
        return json.loads(TRACKER_PATH.read_text())
    return {}


def _save_tracker(tracker: dict) -> None:
    TRACKER_PATH.write_text(json.dumps(tracker, indent=2))


def select_new_items(all_items: list[dict], bucket_key: str, n: int, tracker: dict,
                      seed: int | None = None) -> list[dict]:
    """Shuffle-select n items from all_items not already used for bucket_key
    in a previous run, and record the selection in `tracker` (caller saves
    it). Deterministic per-run given `seed`; uniqueness across runs comes
    from the used-id exclusion, not from the seed.

    Add a new bucket by giving it a distinct bucket_key -- no other change
    needed, which is the "easy to extend" part.
    """
    used = set(tracker.get(bucket_key, {}).get("used_ids", []))
    available = [it for it in all_items if it["id"] not in used]
    if len(available) < n:
        raise ValueError(
            f"bucket '{bucket_key}': only {len(available)} unused items left "
            f"(requested {n}); {len(used)} of {len(all_items)} already used "
            f"in prior runs. Lower --n-per-order or reset the tracker."
        )
    rng = random.Random(seed)
    selected = rng.sample(available, n)
    entry = tracker.setdefault(bucket_key, {"used_ids": [], "runs": []})
    entry["used_ids"].extend(it["id"] for it in selected)
    entry["runs"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n": n,
        "ids": [it["id"] for it in selected],
    })
    return selected


# ---------------------------------------------------------------------------
# Live progress file (read anytime for on-demand status; also drives the
# periodic stdout lines a Monitor can watch for proactive updates)
# ---------------------------------------------------------------------------

_progress_lock = threading.Lock()


def _init_progress(total: int, by_family: dict[str, int]) -> dict:
    progress = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "done": 0,
        "by_family": {k: {"total": v, "done": 0} for k, v in by_family.items()},
    }
    PROGRESS_PATH.write_text(json.dumps(progress, indent=2))
    return progress


def _bump_progress(progress: dict, family: str) -> None:
    with _progress_lock:
        progress["done"] += 1
        progress["by_family"][family]["done"] += 1
        progress["updated_at"] = datetime.now(timezone.utc).isoformat()
        PROGRESS_PATH.write_text(json.dumps(progress, indent=2))


def _finish_progress(progress: dict, status: str = "completed") -> None:
    progress["status"] = status
    progress["updated_at"] = datetime.now(timezone.utc).isoformat()
    PROGRESS_PATH.write_text(json.dumps(progress, indent=2))


def _call(system: str, user: str, cache_path: Path, condition: str) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    # Reasoning models (gpt-5.x) spend max_tokens on hidden reasoning before
    # visible output; the narrative scaffold's 5-section structure needs
    # more headroom than the single-shot conditions.
    max_tokens = 4096 if condition == "narrative_cot_v3_tom" else 2560
    effort = "low" if condition == "baseline_io" else "medium"
    gen = generate("gpt-5.4-nano", system, user, sample_idx=0, max_tokens=max_tokens,
                    reasoning_effort=effort)
    result = {"text": gen.text, "prompt_tokens": gen.prompt_tokens,
              "completion_tokens": gen.completion_tokens}
    cache_path.write_text(json.dumps(result))
    return result


def _run_choice(dataset: str, model: str, condition: str, item: dict) -> dict:
    system = CONDITIONS_BY_TASK["belief_choice"][condition]
    if dataset == "hitom":
        user = _build_choice_prompt(item["story"], item["question"], item["choices"])
        correct_letter = _correct_letter_hitom(item)
        order_label = str(item["order"])
    else:
        user = _build_choice_prompt(item["context"], item["question"], item["choices"])
        correct_letter = item["correct_letter"]
        order_label = item["order"]

    cache_path = _gen_cache_path("choice", dataset, model, condition, item["id"])
    result = _call(system, user, cache_path, condition)
    predicted = extract_letter(result["text"])
    return {
        "dataset": dataset, "model": model, "condition": condition,
        "item_id": item["id"], "order": order_label,
        "correct_letter": correct_letter, "predicted_letter": predicted,
        "is_correct": int(predicted == correct_letter),
        "prompt_tokens": result.get("prompt_tokens", 0),
        "completion_tokens": result.get("completion_tokens", 0),
    }


def _run_list(qtype: str, model: str, condition: str, item: dict) -> dict:
    system = CONDITIONS_BY_TASK["list"][condition]
    user = _build_list_prompt(item["context"], item["question"])
    cache_path = _gen_cache_path(f"list_{qtype}", "fantom", model, condition, item["id"])
    result = _call(system, user, cache_path, condition)

    predicted = extract_names(result["text"], item["universe"])
    gold = set(item["correct_set"])
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else (1.0 if not gold else 0.0)
    recall = tp / len(gold) if gold else (1.0 if not predicted else 0.0)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {
        "dataset": "fantom", "list_qtype": qtype, "model": model, "condition": condition,
        "item_id": item["id"], "gold": "|".join(sorted(gold)),
        "predicted": "|".join(sorted(predicted)),
        "exact_match": int(predicted == gold),
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "prompt_tokens": result.get("prompt_tokens", 0),
        "completion_tokens": result.get("completion_tokens", 0),
    }


def _run_binary(qtype: str, model: str, condition: str, item: dict) -> dict:
    system = CONDITIONS_BY_TASK["binary"][condition]
    user = _build_binary_prompt(item["context"], item["question"])
    cache_path = _gen_cache_path(f"binary_{qtype}", "fantom", model, condition, item["id"])
    result = _call(system, user, cache_path, condition)
    predicted = extract_yesno(result["text"])
    return {
        "dataset": "fantom", "binary_qtype": qtype, "model": model, "condition": condition,
        "item_id": item["id"], "correct_answer": item["correct_answer"],
        "predicted": predicted, "is_correct": int(predicted == item["correct_answer"]),
        "prompt_tokens": result.get("prompt_tokens", 0),
        "completion_tokens": result.get("completion_tokens", 0),
    }


def _run_gen(model: str, condition: str, item: dict) -> dict:
    system = CONDITIONS_BY_TASK["gen"][condition]
    user = _build_gen_prompt(item["context"], item["question"])
    cache_path = _gen_cache_path("gen", "fantom", model, condition, item["id"])
    result = _call(system, user, cache_path, condition)
    response = (result["text"] or "").strip()

    # Same classification rule as the paper's evaluate_belief_q: embed the
    # response and both references, compare cosine similarity; if the wrong
    # reference is at least as close, count it wrong and F1 against wrong,
    # else count it correct and F1 against correct. Embeds via
    # text-embedding-3-small (Azure) rather than the paper's local
    # sentence-transformers model, per the "use OpenAI/Azure embeddings"
    # design call -- absolute numbers aren't bit-for-bit paper-comparable.
    correct_ref, wrong_ref = item["correct_answer"], item["wrong_answer"]
    if not response:
        is_correct, f1 = False, 0.0
    else:
        from sklearn.metrics.pairwise import cosine_similarity
        vecs = _embed([response, correct_ref, wrong_ref])
        resp_v, correct_v, wrong_v = [v for v in vecs]
        sim_correct = cosine_similarity([resp_v], [correct_v])[0][0]
        sim_wrong = cosine_similarity([resp_v], [wrong_v])[0][0]
        if sim_wrong >= sim_correct:
            is_correct, f1 = False, token_f1(wrong_ref, response)
        else:
            is_correct, f1 = True, token_f1(correct_ref, response)

    return {
        "dataset": "fantom", "model": model, "condition": condition,
        "item_id": item["id"], "order": item["order"],
        "is_correct": int(is_correct), "token_f1": round(f1, 4),
        "prompt_tokens": result.get("prompt_tokens", 0),
        "completion_tokens": result.get("completion_tokens", 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4-nano")
    ap.add_argument("--conditions", default="baseline_io,standard_cot,narrative_cot_v3_tom")
    ap.add_argument("--n-per-order", type=int, default=None,
                     help="items per ToM order/list-qtype per dataset; default 2 (--smoke) or 15")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="tiny run (n=2) to sanity-check the pipeline")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    n = args.n_per_order or (2 if args.smoke else 15)
    conditions = args.conditions.split(",")
    tracker = _load_tracker()

    choice_tasks = []
    for order in range(5):
        items = select_new_items(load_hitom(order=order), f"hitom_choice_order{order}", n, tracker, args.seed)
        for item in items:
            for cond in conditions:
                choice_tasks.append(("hitom", args.model, cond, item))
    for order in ("first-order", "second-order"):
        items = select_new_items(load_fantom(tom_order=order), f"fantom_choice_{order}", n, tracker, args.seed)
        for item in items:
            for cond in conditions:
                choice_tasks.append(("fantom", args.model, cond, item))

    list_tasks = []
    for qtype in ("answerability", "info_accessibility"):
        items = select_new_items(load_fantom_list(qtype=qtype), f"fantom_list_{qtype}", n, tracker, args.seed)
        for item in items:
            for cond in conditions:
                list_tasks.append((qtype, args.model, cond, item))

    binary_tasks = []
    for qtype in ("answerability", "info_accessibility"):
        items = select_new_items(load_fantom_binary(qtype=qtype), f"fantom_binary_{qtype}", n, tracker, args.seed)
        for item in items:
            for cond in conditions:
                binary_tasks.append((qtype, args.model, cond, item))

    gen_tasks = []
    for order in ("first-order", "second-order"):
        items = select_new_items(load_fantom_gen(tom_order=order), f"fantom_gen_{order}", n, tracker, args.seed)
        for item in items:
            for cond in conditions:
                gen_tasks.append((args.model, cond, item))

    _save_tracker(tracker)  # persist selections up front, before any calls run

    total = len(choice_tasks) + len(list_tasks) + len(binary_tasks) + len(gen_tasks)
    by_family = {"choice": len(choice_tasks), "list": len(list_tasks),
                 "binary": len(binary_tasks), "gen": len(gen_tasks)}
    print(f"Running {len(choice_tasks)} belief_choice + {len(list_tasks)} list-type + "
          f"{len(binary_tasks)} binary + {len(gen_tasks)} gen generations "
          f"(model={args.model}, n={n} new items/bucket, conditions={conditions})", flush=True)

    progress = _init_progress(total, by_family)
    choice_rows, list_rows, binary_rows, gen_rows = [], [], [], []
    t0 = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run_choice, *t): "choice" for t in choice_tasks}
            futs.update({ex.submit(_run_list, *t): "list" for t in list_tasks})
            futs.update({ex.submit(_run_binary, *t): "binary" for t in binary_tasks})
            futs.update({ex.submit(_run_gen, *t): "gen" for t in gen_tasks})
            buckets = {"choice": choice_rows, "list": list_rows, "binary": binary_rows, "gen": gen_rows}
            done = 0
            for fut in as_completed(futs):
                family = futs[fut]
                row = fut.result()  # raises here if a task failed -- caught below
                buckets[family].append(row)
                _bump_progress(progress, family)
                done += 1
                if done % 5 == 0 or done == total:
                    elapsed = time.monotonic() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta_s = (total - done) / rate if rate > 0 else 0
                    print(f"  PROGRESS {done}/{total} done ({elapsed:.0f}s elapsed, "
                          f"~{eta_s:.0f}s remaining)", flush=True)
    except Exception:
        _finish_progress(progress, status="failed")
        print("ERROR: run_tom_scaffold crashed -- see traceback below", flush=True)
        raise

    _append_csv(CHOICE_CSV, choice_rows,
                ["dataset", "model", "condition", "item_id", "order", "correct_letter",
                 "predicted_letter", "is_correct", "prompt_tokens", "completion_tokens"])
    _append_csv(LIST_CSV, list_rows,
                ["dataset", "list_qtype", "model", "condition", "item_id", "gold", "predicted",
                 "exact_match", "precision", "recall", "f1", "prompt_tokens", "completion_tokens"])
    _append_csv(BINARY_CSV, binary_rows,
                ["dataset", "binary_qtype", "model", "condition", "item_id", "correct_answer",
                 "predicted", "is_correct", "prompt_tokens", "completion_tokens"])
    _append_csv(GEN_CSV, gen_rows,
                ["dataset", "model", "condition", "item_id", "order", "is_correct", "token_f1",
                 "prompt_tokens", "completion_tokens"])
    print(f"Wrote {len(choice_rows)} choice, {len(list_rows)} list, {len(binary_rows)} binary, "
          f"{len(gen_rows)} gen rows (this run)", flush=True)
    _finish_progress(progress, status="completed")

    # Summaries read the FULL cumulative CSVs (every run to date), not just
    # this run's rows -- results across runs combine automatically since
    # item_ids never repeat within a bucket (see select_new_items).
    print("\n=== CUMULATIVE RESULTS (all runs to date) ===", flush=True)
    _print_choice_summary(_read_cumulative(CHOICE_CSV, ["is_correct"]))
    _print_list_summary(_read_cumulative(LIST_CSV, ["exact_match", "f1"]))
    _print_binary_summary(_read_cumulative(BINARY_CSV, []))
    _print_gen_summary(_read_cumulative(GEN_CSV, ["is_correct", "token_f1"]))
    print("\nRUN COMPLETE", flush=True)


def _read_cumulative(path: Path, numeric_fields: list[str]) -> list[dict]:
    """Read a results CSV back in full (all runs to date), coercing the
    given fields from CSV strings to float so the summary printers' sum()/
    len() aggregation works the same as it does on freshly-computed rows."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for field in numeric_fields:
            r[field] = float(r[field])
    return rows


def _append_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    if not rows:
        return
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _print_choice_summary(rows: list[dict]) -> None:
    from collections import defaultdict
    acc = defaultdict(lambda: [0, 0])
    for r in rows:
        acc[(r["dataset"], r["order"], r["condition"])][0] += r["is_correct"]
        acc[(r["dataset"], r["order"], r["condition"])][1] += 1
    print("\n[belief_choice accuracy]")
    print("dataset      order          condition               accuracy")
    for key in sorted(acc):
        correct, total = acc[key]
        print(f"{key[0]:<12} {str(key[1]):<14} {key[2]:<23} {correct}/{total} = {correct/total:.2f}")


def _print_list_summary(rows: list[dict]) -> None:
    # accuracy here is exact-set-match (all aware characters named, none
    # hallucinated) -- the same all-or-nothing criterion the FANToM paper's
    # own eval_list_q uses for AnswerabilityQ[List]/InfoAccessQ[List]
    # accuracy, so it's directly comparable to the paper's published numbers.
    # F1 (precision/recall over the named set) is not in the paper for this
    # question type; it's supplementary, comparable only across our own arms.
    from collections import defaultdict
    agg = defaultdict(lambda: {"f1": [], "exact_match": []})
    for r in rows:
        agg[(r["list_qtype"], r["condition"])]["f1"].append(r["f1"])
        agg[(r["list_qtype"], r["condition"])]["exact_match"].append(r["exact_match"])
    print("\n[list-type accuracy (exact-set-match, comparable to paper) + F1 (ours only)]")
    print("qtype               condition               accuracy  mean_f1   n")
    for key in sorted(agg):
        em, f1 = agg[key]["exact_match"], agg[key]["f1"]
        print(f"{key[0]:<19} {key[1]:<23} {sum(em)/len(em):.2f}      {sum(f1)/len(f1):.2f}      {len(em)}")


def _print_binary_summary(rows: list[dict]) -> None:
    # weighted F1 over yes/no classes, matching the paper's
    # AnswerabilityQ[Y/N]/InfoAccessQ[Y/N] "weighted F1" metric.
    from collections import defaultdict
    from sklearn.metrics import f1_score
    agg = defaultdict(lambda: {"y_true": [], "y_pred": []})
    for r in rows:
        agg[(r["binary_qtype"], r["condition"])]["y_true"].append(r["correct_answer"])
        agg[(r["binary_qtype"], r["condition"])]["y_pred"].append(r["predicted"] or "no")
    print("\n[binary Y/N accessibility: accuracy + weighted F1 (comparable to paper)]")
    print("qtype               condition               accuracy  weighted_f1  n")
    for key in sorted(agg):
        yt, yp = agg[key]["y_true"], agg[key]["y_pred"]
        acc = sum(int(a == b) for a, b in zip(yt, yp)) / len(yt)
        wf1 = f1_score(yt, yp, average="weighted", zero_division=0)
        print(f"{key[0]:<19} {key[1]:<23} {acc:.2f}      {wf1:.2f}         {len(yt)}")


def _print_gen_summary(rows: list[dict]) -> None:
    # accuracy = embedding-classified correct/wrong (paper's BeliefQ[Dist.]
    # accuracy); token_f1 is F1 against whichever reference the response was
    # classified against. Grading uses Azure text-embedding-3-small, not the
    # paper's local sentence-transformers model, so treat these as
    # comparable across our own conditions, not bit-for-bit vs. the paper.
    from collections import defaultdict
    agg = defaultdict(lambda: {"is_correct": [], "token_f1": []})
    for r in rows:
        agg[(r["dataset"], r["order"], r["condition"])]["is_correct"].append(r["is_correct"])
        agg[(r["dataset"], r["order"], r["condition"])]["token_f1"].append(r["token_f1"])
    print("\n[free-form belief (BeliefQ[Dist.]-style): accuracy + token F1 (ours, embed model differs from paper)]")
    print("dataset      order          condition               accuracy  mean_token_f1  n")
    for key in sorted(agg):
        ic, f1 = agg[key]["is_correct"], agg[key]["token_f1"]
        print(f"{key[0]:<12} {str(key[1]):<14} {key[2]:<23} "
              f"{sum(ic)/len(ic):.2f}      {sum(f1)/len(f1):.2f}         {len(ic)}")


if __name__ == "__main__":
    main()
