"""
scripts/run_phase2_ablation.py -- Phase 2: sub-instruction ablation on
claude-sonnet-4-6 (the flagship generator).

Evaluates which of the five N-CoT sections carries the structural shift by
dropping exactly one section at a time. Six conditions total:
  narrative_cot_full           -- all five sections (control)
  narrative_cot_drop_protagonist
  narrative_cot_drop_stakeholders
  narrative_cot_drop_consequences
  narrative_cot_drop_uncertainty
  narrative_cot_drop_commitment

Sample: 50 scenarios (SUBSAMPLE_SEED=43, stratified from Phase 1's 100).
N=10 per cell -> 6 x 50 x 10 = 3,000 generations on claude-sonnet-4-6.
Full judging stack (claude-sonnet-4-6 as judge1, gpt-4o-mini as judge2).

Outputs:
  divergence_study_outputs/subinstruction_attribution.csv
  divergence_study_outputs/subinstruction_attribution.pdf  (5x6 heatmap)

Usage:
  python -m scripts.run_phase2_ablation [--n N] [--workers W]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.generators import generate
from scripts.run_phase1_quartet import (
    load_daily_dilemmas,
    run_one_sample,
    SCALED_SAMPLE_SEED,
    SUBSAMPLE_SEED,
    Scenario,
    OUT_DIR,
)

# ---------------------------------------------------------------------------
# Sub-instruction prompts
# ---------------------------------------------------------------------------

_NCOT_FULL_SECTIONS = """Section 1 – Protagonist: Name and briefly characterise the decision-maker \
(who they are, their role, what they know).
Section 2 – Stakeholders: List every person whose life intersects this \
decision and state what is at stake for each.
Section 3 – Consequences: For each available action, narrate its \
consequences at least two steps into the future for each stakeholder.
Section 4 – Uncertainty: State what remains genuinely uncertain about \
each projected future.
Section 5 – Decision: Commit to a specific decision and explain, within \
the narrative frame, why that trajectory is preferable to the alternatives."""

_NCOT_HEADER = (
    "You are a thoughtful advisor. When given an ethical dilemma, reason "
    "through it as a five-part first-person narrative before giving your answer.\n\n"
)
_NCOT_FOOTER = "\n\nWork through all remaining sections before giving your final answer."


def _build_ncot_drop(drop_section_num: int) -> str:
    """Return an N-CoT prompt with one section removed, renumbered."""
    sections = [
        "Section 1 – Protagonist: Name and briefly characterise the decision-maker "
        "(who they are, their role, what they know).",
        "Section 2 – Stakeholders: List every person whose life intersects this "
        "decision and state what is at stake for each.",
        "Section 3 – Consequences: For each available action, narrate its "
        "consequences at least two steps into the future for each stakeholder.",
        "Section 4 – Uncertainty: State what remains genuinely uncertain about "
        "each projected future.",
        "Section 5 – Decision: Commit to a specific decision and explain, within "
        "the narrative frame, why that trajectory is preferable to the alternatives.",
    ]
    kept = [s for i, s in enumerate(sections, 1) if i != drop_section_num]
    # Renumber
    renumbered = []
    for new_num, sec in enumerate(kept, 1):
        # Replace leading "Section N" with new number
        sec = f"Section {new_num} – " + "–".join(sec.split("–")[1:]).lstrip(" –")
        renumbered.append(sec)
    body = "\n".join(renumbered)
    return (
        "You are a thoughtful advisor. When given an ethical dilemma, reason "
        "through it as a multi-part first-person narrative before giving your answer.\n\n"
        + body
        + "\n\nWork through all sections before giving your final answer."
    )


ABLATION_CONDITIONS = {
    "narrative_cot_full": (
        "You are a thoughtful advisor. When given an ethical dilemma, reason "
        "through it as a five-part first-person narrative before giving your answer.\n\n"
        + _NCOT_FULL_SECTIONS
        + "\n\nWork through all five sections before giving your final answer."
    ),
    "narrative_cot_drop_protagonist": _build_ncot_drop(1),
    "narrative_cot_drop_stakeholders": _build_ncot_drop(2),
    "narrative_cot_drop_consequences": _build_ncot_drop(3),
    "narrative_cot_drop_uncertainty": _build_ncot_drop(4),
    "narrative_cot_drop_commitment": _build_ncot_drop(5),
}


def pick_subsample(scenarios: list[Scenario], n: int = 50) -> list[Scenario]:
    """Deterministic stratified subsample from a 100-scenario list."""
    rng = random.Random(SUBSAMPLE_SEED)
    by_topic: dict[str, list[Scenario]] = {}
    for s in scenarios:
        by_topic.setdefault(s.topic, []).append(s)
    topics = sorted(by_topic.keys())
    n_topics = len(topics)
    quota = n // n_topics
    remainder = n - quota * n_topics
    selected = []
    for i, topic in enumerate(topics):
        pool = list(by_topic[topic])
        rng.shuffle(pool)
        take = quota + (1 if i < remainder else 0)
        selected.extend(pool[:take])
    return selected[:n]


# ---------------------------------------------------------------------------
# Effect-size computation
# ---------------------------------------------------------------------------

def _cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    greater = sum((xi > yj) for xi in x for yj in y)
    less = sum((xi < yj) for xi in x for yj in y)
    return (greater - less) / (nx * ny)


def compute_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """Cliff's delta for each (drop condition, structural variable) vs full N-CoT."""
    struct_vars = {
        "j1_stakeholder_count": "stakeholder_count",
        "j1_max_causal_hops": "max_causal_hops",
        "j1_uncertainty_score": "uncertainty_score",
        "j1_n_frameworks": "n_frameworks",
    }
    conditions = [c for c in ABLATION_CONDITIONS if c != "narrative_cot_full"]
    rows = []
    df_full = df[df["condition"] == "narrative_cot_full"]
    for cond in conditions:
        df_cond = df[df["condition"] == cond]
        for col, label in struct_vars.items():
            if col not in df.columns:
                continue
            x_full = df_full[col].dropna().values
            x_drop = df_cond[col].dropna().values
            delta = _cliff_delta(x_drop, x_full)
            rows.append({
                "condition": cond.replace("narrative_cot_drop_", "drop_"),
                "variable": label,
                "delta_drop_vs_full": round(delta, 4),
                "n_full": len(x_full),
                "n_drop": len(x_drop),
            })
    return pd.DataFrame(rows)


def make_heatmap(df_attr: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("  matplotlib not available; skipping heatmap")
        return

    conditions = sorted(df_attr["condition"].unique())
    variables = ["stakeholder_count", "max_causal_hops", "uncertainty_score", "n_frameworks"]
    variables = [v for v in variables if v in df_attr["variable"].values]

    matrix = np.full((len(conditions), len(variables)), float("nan"))
    for i, cond in enumerate(conditions):
        for j, var in enumerate(variables):
            sub = df_attr[(df_attr["condition"] == cond) & (df_attr["variable"] == var)]
            if not sub.empty:
                matrix[i, j] = sub["delta_drop_vs_full"].values[0]

    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = matplotlib.colormaps["RdYlGn"]
    vmax = max(0.5, np.nanmax(np.abs(matrix)))
    im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(variables)))
    ax.set_xticklabels(variables, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions, fontsize=8)
    for i in range(len(conditions)):
        for j in range(len(variables)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if abs(val) < 0.4 else "white")
    plt.colorbar(im, ax=ax, label="Cliff's δ (drop vs. full N-CoT)")
    ax.set_title("Sub-instruction attribution: effect of dropping each N-CoT section\n"
                 "(claude-sonnet-4-6, N=10, 50 scenarios)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved figure: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2: sub-instruction ablation")
    parser.add_argument("--n", type=int, default=10, help="Samples per cell")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    gen_model = "claude-sonnet-4-6"
    judge_model = os.environ.get("AZURE_AI_MODEL_JUDGE", "claude-sonnet-4-6")
    judge2_model = os.environ.get("AZURE_AI_MODEL_JUDGE_2", "gpt-4o-mini")
    extractor_model = os.environ.get("AZURE_AI_MODEL_DECISION", "gpt-4o-mini")

    print("Loading DailyDilemmas 100-scenario sample...")
    scenarios_100 = load_daily_dilemmas(100)
    scenarios = pick_subsample(scenarios_100, 50)
    print(f"  Using {len(scenarios)} scenarios")

    # Inject custom system prompts into PROMPTS for this run
    import scripts.run_phase1_quartet as p1
    original_prompts = dict(p1.PROMPTS)
    p1.PROMPTS.update(ABLATION_CONDITIONS)

    conditions = list(ABLATION_CONDITIONS.keys())
    tasks = [
        (scenario, condition, idx)
        for scenario in scenarios
        for condition in conditions
        for idx in range(args.n)
    ]
    print(f"Total tasks: {len(tasks)}")

    all_rows: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(
                run_one_sample,
                scenario, condition, idx,
                gen_model, judge_model, judge2_model, extractor_model,
            ): (scenario.id, condition, idx)
            for scenario, condition, idx in tasks
        }
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                row = fut.result()
                all_rows.append(row)
                done += 1
                if done % 100 == 0:
                    print(f"  {done}/{len(tasks)} done", flush=True)
            except Exception as e:
                print(f"  ERROR {key}: {e}", flush=True)

    # Restore original prompts
    p1.PROMPTS.clear()
    p1.PROMPTS.update(original_prompts)

    if not all_rows:
        print("No rows collected.")
        return 1

    df = pd.DataFrame(all_rows)
    raw_path = OUT_DIR / "phase2_ablation_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"Wrote raw data: {raw_path}")

    df_attr = compute_attribution(df)
    attr_path = OUT_DIR / "subinstruction_attribution.csv"
    df_attr.to_csv(attr_path, index=False)
    print(f"Wrote attribution: {attr_path}")
    print(df_attr.to_string(index=False))

    make_heatmap(df_attr, OUT_DIR / "subinstruction_attribution.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
