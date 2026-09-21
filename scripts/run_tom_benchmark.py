"""
run_tom_benchmark.py -- Addendum 18 (B1): Narration-of-Thought against standard
chain-of-thought on ToMBench (Chen et al., ACL 2024; MIT licence; English fields
of the eight task files in data/tombench/, 2,470 items), mechanically scored by
the answer letter. Sign-agnostic by registration.

Design. 40 items per task x 8 tasks = 320 items, stratified sample with seed 42
(fixed by registration); two arms, `standard_cot` and `narrative_cot`, the
system prompts VERBATIM from scripts.run_phase1_quartet.PROMPTS; the user turn
is the story, the question, the four options and one fixed instruction to end
with a line `ANSWER: <letter>`; one sample per (model, arm, item); cache
`divergence_study_outputs/tom_gen_<model>_<arm>_<task>_<row>.json`, keyed by NAME,
where <row> is the item's 0-based row position within its task file (fixed 2026-09-21;
ToMBench's own INDEX field is a within-story question counter, not a unique row id,
see load_items's docstring). Token caps as in run_elephant (1,024 CoT, 2,048 NoT; the Foundry route
applies DeepSeek-V4-Pro's floor by name).

Readouts. Accuracy per (model, arm); paired NoT minus CoT per model with an
item-clustered percentile CI (4,000 draws, seed 18); the pooled delta over
models (items as clusters); per-task deltas with the False Belief Task as the
pre-declared primary stratum; non-response and unparsed-answer rates per arm
(an unparsed answer scores wrong and is reported).

  python -m scripts.run_tom_benchmark --dry-run                      # counts and priced estimate, no calls
  python -m scripts.run_tom_benchmark --run --models <a,b,...>       # generate (cached) and analyse
  python -m scripts.run_tom_benchmark --analyse                      # analysis on cache only
  python -m scripts.run_tom_benchmark --selftest
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from scripts.run_phase1_quartet import PROMPTS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "tombench"
OUT = ROOT / "divergence_study_outputs"
TASKS = ("Ambiguous Story Task", "False Belief Task", "Faux-pas Recognition Test", "Hinting Task Test",
         "Persuasion Story Task", "Scalar Implicature Test", "Strange Story Task", "Unexpected Outcome Test")
PRIMARY_TASK = "False Belief Task"
PER_TASK, SEED = 40, 42
ARMS = ("standard_cot", "narrative_cot")
DEFAULT_MODELS = ("claude-haiku-4-5", "claude-sonnet-4-6", "gpt-5.4-nano", "grok-4-1-fast-reasoning",
                  "Llama-3.3-70B-Instruct", "Mistral-Large-3-2", "DeepSeek-V4-Pro")
INSTRUCTION = ("Read the story and answer the question by choosing one of the four options. "
               "Finish your response with a final line of the form\nANSWER: <letter>\nwhere <letter> is A, B, C or D.")
ANS_RE = re.compile(r"ANSWER\s*[:：]\s*\(?([ABCD])\)?", re.IGNORECASE)


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def load_items(per_task=PER_TASK, seed=SEED):
    """Sample per_task items from each task file.

    BUG FOUND 2026-09-21, fixed here: ToMBench's `\u5e8f\u53f7 / INDEX` field is a
    WITHIN-STORY question counter (1, 2, 3... resetting for every new vignette), not a
    unique row id -- across an entire task file it takes as few as 1 and at most 6
    distinct values (checked: Ambiguous Story Task 2, False Belief Task 6, Faux-pas
    Recognition Test 4, Hinting Task Test 2, Persuasion Story Task 1, Scalar Implicature
    Test 2, Strange Story Task 4, Unexpected Outcome Test 3). Keying the cache path on
    (task, index) therefore funnelled dozens of DIFFERENT stories onto a handful of
    filenames, each write silently clobbering the last (and, under threading, some
    concurrent writes to the same path corrupted the file, the "Extra data" JSON errors
    in cell_18_tom_stage1.log). Stage-1 cells landed on 23 of the intended 320 items per
    model per arm -- see divergence_study_outputs/tom_gen_VOID_2026_09_21/ (quarantined,
    not deleted) and tom_benchmark_analysis_VOID_2026_09_21.json (the resulting pooled
    delta, -0.033 [-0.130, +0.065], is an artefact of this bug and was never read as a
    finding). The unique key is now the row's 0-based position in the task's jsonl file,
    stable because data/tombench/*.jsonl is static and committed.
    """
    items = []
    for t in TASKS:
        rows = [json.loads(l) for l in open(DATA / f"{t}.jsonl")]
        rng = random.Random(f"{seed}:{t}")
        pool = list(enumerate(rows))                    # (row_position, row) pairs
        picked = rng.sample(pool, per_task)
        for row_pos, r in sorted(picked, key=lambda p: p[0]):
            items.append({"task": t, "row": row_pos, "within_story_index": int(r["序号\nINDEX"]),
                          "ability": r["能力\nABILITY"], "story": r["STORY"], "question": r["QUESTION"],
                          "options": {k: r[f"OPTION-{k}"] for k in "ABCD"}, "answer": r["答案\nANSWER"].strip().upper()})
    return items


def user_turn(it) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in it["options"].items())
    return f"Story: {it['story']}\n\nQuestion: {it['question']}\n\n{opts}\n\n{INSTRUCTION}"


def cache_path(model, arm, it) -> Path:
    return OUT / f"tom_gen_{_safe(model)}_{arm}_{_safe(it['task'])}_{it['row']:04d}.json"


def parse_answer(text: str):
    m = ANS_RE.findall(text or "")
    return m[-1].upper() if m else None


def generate_one(model, arm, it):
    p = cache_path(model, arm, it)
    if p.exists():
        return json.loads(p.read_text())
    from scripts.generators import generate
    r = generate(model, PROMPTS[arm], user_turn(it), sample_idx=0, max_tokens=2048 if arm == "narrative_cot" else 1024)
    rec = {"model": model, "arm": arm, "task": it["task"], "row": it["row"], "text": r.text or "",
           "finish_reason": r.finish_reason, "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
           "answer": parse_answer(r.text or ""), "gold": it["answer"]}
    if rec["text"].strip():
        p.write_text(json.dumps(rec, ensure_ascii=False))
    return rec


def boot_mean(x, draws=4000, seed=18):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return {"point": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(draws, len(x)))
    m = x[idx].mean(1)
    return {"point": float(x.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5)), "n": int(len(x))}


def analyse(items, models):
    res = {"items": len(items), "per_task_n": PER_TASK, "seed": SEED, "models": {}, "pooled": {}}
    pooled_by_item = defaultdict(list)
    for model in models:
        recs = {arm: {} for arm in ARMS}
        for arm in ARMS:
            for it in items:
                p = cache_path(model, arm, it)
                if p.exists():
                    recs[arm][(it["task"], it["row"])] = json.loads(p.read_text())
        if not recs["standard_cot"] or not recs["narrative_cot"]:
            res["models"][model] = {"status": "ABSENT"}
            continue
        m = {"n_cached": {a: len(recs[a]) for a in ARMS}, "acc": {}, "unparsed": {}, "mean_completion_tokens": {}}
        for arm in ARMS:
            rs = list(recs[arm].values())
            m["acc"][arm] = sum(r["answer"] == r["gold"] for r in rs) / len(rs)
            m["unparsed"][arm] = sum(r["answer"] is None for r in rs) / len(rs)
            m["mean_completion_tokens"][arm] = sum(r["completion_tokens"] or 0 for r in rs) / len(rs)
        keys = sorted(set(recs["standard_cot"]) & set(recs["narrative_cot"]))
        d = {k: int(recs["narrative_cot"][k]["answer"] == recs["narrative_cot"][k]["gold"])
                - int(recs["standard_cot"][k]["answer"] == recs["standard_cot"][k]["gold"]) for k in keys}
        m["paired_items"] = len(keys)
        m["delta"] = boot_mean([d[k] for k in keys])
        m["delta_by_task"] = {t: boot_mean([d[k] for k in keys if k[0] == t]) for t in TASKS}
        m["delta_primary_task"] = m["delta_by_task"][PRIMARY_TASK]
        for k in keys:
            pooled_by_item[k].append(d[k])
        res["models"][model] = m
    # pooled over models, items as clusters (mean of per-item mean deltas)
    keys = sorted(pooled_by_item)
    res["pooled"]["delta"] = boot_mean([np.mean(pooled_by_item[k]) for k in keys])
    res["pooled"]["delta_primary_task"] = boot_mean([np.mean(pooled_by_item[k]) for k in keys if k[0] == PRIMARY_TASK])
    res["pooled"]["n_models"] = sum(1 for m in res["models"].values() if "delta" in m)
    return res


def price_estimate(models, n_items):
    from scripts.run_crowdgold_deliberation import PRICES
    est = {}
    for model in models:
        pin, pout = PRICES.get(model, (0.0, 0.0))
        if pin == 0.0:
            raise SystemExit(f"unpriced model {model}; add it to PRICES before any call")
        # assumed tokens: 420 in per call; 350 out for CoT, 1,100 out for NoT
        est[model] = n_items * ((420 * pin + 350 * pout) + (420 * pin + 1100 * pout)) / 1e6
    return est


def _selftest() -> int:
    ok = parse_answer("blah\nANSWER: C") == "C" and parse_answer("answer: (b)") == "B" and parse_answer("none") is None
    items = load_items(per_task=3)
    ok &= len(items) == 24 and all(it["answer"] in "ABCD" for it in items) and load_items(per_task=3) == items
    ok &= "ANSWER:" in user_turn(items[0]) and PROMPTS["narrative_cot"].startswith("You are a thoughtful advisor")
    # Regression check for the 2026-09-21 collision bug: the cache key must be unique
    # per item even though ToMBench's own within-story INDEX repeats heavily.
    cache_keys = {cache_path("m", "standard_cot", it) for it in items}
    ok &= len(cache_keys) == len(items)
    full = load_items()
    full_keys = {(it["task"], it["row"]) for it in full}
    ok &= len(full) == 320 and len(full_keys) == 320
    print("selftest", "OK" if ok else "FAILED"); return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--json", type=Path, default=OUT / "tom_benchmark_analysis.json")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    models = [m for m in a.models.split(",") if m]
    items = load_items()
    print(f"ToMBench: {len(items)} items ({PER_TASK} per task, seed {SEED}), arms {ARMS}, models {models}")
    if a.dry_run or a.run:
        todo = [(m, arm, it) for m in models for arm in ARMS for it in items if not cache_path(m, arm, it).exists()]
        est = price_estimate(models, len(items))
        print(f"  calls total {len(models) * len(ARMS) * len(items)}, cached {len(models) * len(ARMS) * len(items) - len(todo)}, new {len(todo)}")
        for m, v in est.items():
            print(f"  est ${v:6.2f}  {m}")
        print(f"  est total ${sum(est.values()):.2f} (assumed 420 in, 350/1,100 out)")
        if a.dry_run:
            return 0
        done = 0
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futs = [pool.submit(generate_one, m, arm, it) for m, arm, it in todo]
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:  # noqa: BLE001
                    print("  ERROR", str(e)[:160])
                done += 1
                if done % 50 == 0:
                    print(f"    {done}/{len(todo)}", flush=True)
    res = analyse(items, models)
    for model, m in res["models"].items():
        if "delta" not in m:
            print(f"  {model:26s} ABSENT"); continue
        d, dp = m["delta"], m["delta_primary_task"]
        print(f"  {model:26s} CoT {m['acc']['standard_cot']:.3f} NoT {m['acc']['narrative_cot']:.3f} "
              f"delta {d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] n={d['n']}  false-belief {dp['point']:+.3f} [{dp['lo']:+.3f}, {dp['hi']:+.3f}]  "
              f"unparsed CoT {m['unparsed']['standard_cot']:.3f} NoT {m['unparsed']['narrative_cot']:.3f}")
    p = res["pooled"]
    if p.get("delta", {}).get("point") is not None:
        print(f"  POOLED over {p['n_models']} models: delta {p['delta']['point']:+.3f} [{p['delta']['lo']:+.3f}, {p['delta']['hi']:+.3f}]  "
              f"false-belief {p['delta_primary_task']['point']:+.3f} [{p['delta_primary_task']['lo']:+.3f}, {p['delta_primary_task']['hi']:+.3f}]")
    a.json.write_text(json.dumps(res, indent=1))
    print("  wrote", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
