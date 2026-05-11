"""
scripts/merge_phase1_cache.py -- Build the Phase 1 master CSV from cache files
without doing any new API calls.

For each generation cache file gen_<model>_<scenario>_<condition>_<idx>_<hash>.json,
look up whatever judge1, judge2, decision, and tag cache files exist (any judge
model) and emit a row.  This is robust to judge-model swaps that happened
mid-experiment.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

OUT_DIR = Path("./divergence_study_outputs")

GEN_RE = re.compile(
    r"^gen_(?P<model>[A-Za-z0-9._\-]+?)_(?P<sid>dd_\d+)_(?P<cond>[a-z_]+)_(?P<idx>\d{3})_[0-9a-f]+\.json$"
)


def load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def find_one(pattern: str) -> Path | None:
    matches = sorted(OUT_DIR.glob(pattern))
    return matches[0] if matches else None


def main() -> int:
    rows: list[dict] = []
    gen_files = sorted(OUT_DIR.glob("gen_*_dd_*.json"))
    print(f"Found {len(gen_files)} generation cache files")

    skipped = 0
    for gp in gen_files:
        m = GEN_RE.match(gp.name)
        if not m:
            skipped += 1
            continue
        model = m.group("model")
        sid = m.group("sid")
        cond = m.group("cond")
        idx = int(m.group("idx"))

        gen = load_json(gp)
        if not gen.get("output"):
            continue
        scenario_id = gen.get("scenario_id", sid)
        condition = gen.get("condition", cond)

        # Find any judge1 (prefer claude-haiku-4-5, then sonnet, then any glob)
        j1 = {}
        j1_used = ""
        for judge in ["claude-haiku-4-5", "claude-sonnet-4-6", "gpt-4o-mini", "gpt-5.4-nano"]:
            jp = OUT_DIR / f"judge_{judge}_gen_{model}_{scenario_id}_{condition}_{idx:03d}.json"
            if jp.exists():
                j1 = load_json(jp)
                j1_used = judge
                break
        # Find any judge2 (prefer different model from j1)
        j2 = {}
        for judge in ["gpt-5.4-nano", "gpt-4o-mini", "claude-haiku-4-5", "claude-sonnet-4-6"]:
            if judge == j1_used:
                continue
            jp = OUT_DIR / f"judge_{judge}_gen_{model}_{scenario_id}_{condition}_{idx:03d}.json"
            if jp.exists():
                j2 = load_json(jp)
                break
        # Decision (any extractor)
        dec = {}
        for ext in ["gpt-5.4-nano", "gpt-4o-mini"]:
            dp = OUT_DIR / f"decision_{ext}_gen_{model}_{scenario_id}_{condition}_{idx:03d}.json"
            if dp.exists():
                dec = load_json(dp)
                break
        # Failure-mode tag — use cached LLM-coded tag if present, else derive
        # deterministically from j1 fields (same anchors the tagger uses).
        tp = OUT_DIR / f"tag_fm_{model}_{scenario_id}_{condition}_{idx:03d}.json"
        if tp.exists():
            tag = load_json(tp)
        elif j1:
            def _safe_int(v) -> int:
                try:
                    iv = int(v)
                    return iv
                except (TypeError, ValueError):
                    return 0
            sc = _safe_int(j1.get("stakeholder_count", 0))
            mh = _safe_int(j1.get("max_causal_hops", 0))
            us = _safe_int(j1.get("uncertainty_score", 0))
            fw = j1.get("framework_names_invoked", []) or []
            tag = {
                "premature_refusal": bool(j1.get("refused", False)),
                "framework_enumeration": len(fw) >= 2,
                "consequential_flattening": mh <= 1,
                "stakeholder_collapse": sc <= 1,
                "uncertainty_suppression": us == 0,
            }
        else:
            tag = {}

        rows.append({
            "generator": model,
            "scenario_id": scenario_id,
            "condition": condition,
            "sample_idx": idx,
            "output_len": len(gen.get("output", "")),
            "j1_refused": j1.get("refused", False),
            "j1_commits": j1.get("commits_to_action", False),
            "j1_stakeholder_count": j1.get("stakeholder_count", 0),
            "j1_max_causal_hops": j1.get("max_causal_hops", 0),
            "j1_uncertainty_score": j1.get("uncertainty_score", 0),
            "j1_n_frameworks": len(j1.get("framework_names_invoked", []) or []),
            "j2_refused": j2.get("refused", False),
            "j2_commits": j2.get("commits_to_action", False),
            "j2_stakeholder_count": j2.get("stakeholder_count", 0),
            "j2_max_causal_hops": j2.get("max_causal_hops", 0),
            "j2_uncertainty_score": j2.get("uncertainty_score", 0),
            "j2_n_frameworks": len(j2.get("framework_names_invoked", []) or []),
            "decision": dec.get("decision", "NO_COMMITMENT"),
            "fm_premature_refusal": tag.get("premature_refusal", False),
            "fm_framework_enumeration": tag.get("framework_enumeration", False),
            "fm_consequential_flattening": tag.get("consequential_flattening", False),
            "fm_stakeholder_collapse": tag.get("stakeholder_collapse", False),
            "fm_uncertainty_suppression": tag.get("uncertainty_suppression", False),
        })

    print(f"Built {len(rows)} rows; skipped {skipped} unparseable filenames")
    out_path = OUT_DIR / "phase1_quartet_raw.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {out_path}")

    # Quick coverage report
    from collections import Counter
    by_gen = Counter(r["generator"] for r in rows)
    by_gen_cond = Counter((r["generator"], r["condition"]) for r in rows)
    print("\nRows per generator:")
    for g, n in sorted(by_gen.items()):
        print(f"  {g}: {n}")
    print("\nRows per (generator, condition):")
    for (g, c), n in sorted(by_gen_cond.items()):
        print(f"  {g} / {c}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
