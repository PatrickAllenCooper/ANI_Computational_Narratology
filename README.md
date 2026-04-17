# ANI — Narrative Chain-of-Thought Divergence Pilot

A minimum-viable empirical test of whether narrative-structured prompting produces systematically different ethical reasoning outputs than standard chain-of-thought, across two generation model families and two cross-vendor judges.

The canonical study design lives in `Guidance_Documents/study_design.md`. The paper draft is `paper.tex`. Implementation is `ncot_divergence_pilot.ipynb`. Per-run artifacts (CSVs and figures) are in `divergence_study_outputs/`; per-sample cache JSON is gitignored and regenerated from the notebook.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install azure-ai-projects azure-identity openai pandas matplotlib seaborn scipy tqdm python-dotenv requests scikit-learn jupyter
```

## Running

The notebook reads its model configuration from environment variables. Set these before launching Jupyter (do not commit the values):

```bash
export AZURE_AI_PROJECT_ENDPOINT="https://<resource>.cognitiveservices.azure.com"
export AZURE_AI_API_KEY="<your-key>"
export AZURE_AI_API_VERSION="2025-04-01-preview"
export AZURE_AI_MODELS_GENERATION="gpt-5.4-nano,gpt-4o"
export AZURE_AI_MODEL_JUDGE="claude-sonnet-4-6"
export AZURE_AI_MODEL_JUDGE_2="gpt-4o-mini"
export AZURE_AI_MODEL_DECISION="gpt-4o-mini"

jupyter nbconvert --to notebook --execute ncot_divergence_pilot.ipynb \
  --output ncot_divergence_pilot.ipynb \
  --ExecutePreprocessor.timeout=7200
```

Generation, judging, and decision-extractor results are cached per `(generator, judge, scenario, condition, sample)`; re-running with a new generator or judge does not re-do prior work.

## What the notebook produces

Per-cell artifacts (gitignored): `divergence_study_outputs/gen_*.json`, `judge_*.json`, `decision_*.json`.

Analysis tables (committed): `coded_results_judge1.csv`, `coded_results_judge2.csv`, `decisions.csv`, `interjudge_agreement.csv`, `tier1_effect_sizes.csv`, `tier2_summary.csv`, `failure_mode_profile.csv`, `failure_mode_aggregate.csv`, `length_residualized_effects.csv`, `cross_judge_effect_sizes.csv`, `decision_entropy.csv`, `cross_generator_tier1.csv`, `cross_generator_tier2.csv`, `cross_generator_excess_matrix.csv`, `per_generator_failure_modes.csv`.

Figures (committed): `tier1_structural_metrics.png`, `tier1_refusal.png`, `tier2_decision_distributions.png`, `divergence_profile_heatmap.png`, `cross_generator_heatmaps.png`.

## Reading the results

Start with the auto-generated headline summary in section 7f of the notebook, then the per-generator comparisons in section 9. Detailed interpretation thresholds are in section 8 (Interpretation guide) and in `Guidance_Documents/study_design.md`.

## Repository discipline

- The author is Patrick Cooper. All commits are made by him.
- `Guidance_Documents/study_design.md` is the source of truth. The notebook implements it. When the design changes, the guidance doc is updated in the same commit.
- API keys are never committed. The `.gitignore` excludes `.env`, `*.key`, and similar.
