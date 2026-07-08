# ANI — Narrative Chain-of-Thought Divergence Study

An empirical study of whether narrative-structured prompting (Narration-of-Thought, NoT) produces systematically different ethical reasoning than standard chain-of-thought, across four frontier generators from three vendors and multiple cross-vendor judges.

The canonical study design lives in `Guidance_Documents/study_design.md`. The notebook and scripts implement it; when the design changes, the guidance doc is updated in the same commit.

## Project map

| Path | What it is |
|------|------------|
| `Guidance_Documents/` | Source of truth: study design, pre-registrations, execution logs |
| `papers/` | LaTeX projects, one folder per paper (see status table below) |
| `ncot_divergence_pilot.ipynb` | Main-study implementation (generation, judging, analysis) |
| `scripts/` | All phase runners, aggregators, and shared library code |
| `activation_guided_thinking/` | Follow-up project sketch (inference-time activation steering); see its README and `agt_bootstrap.ipynb` |
| `data/` | Probe sets and gold labels (large benchmark downloads are gitignored) |
| `divergence_study_outputs/` | Run artifacts: analysis CSVs and figures are committed; per-sample cache JSON is gitignored and regeneratable |

## Papers and their status

| Paper | Path | Status | Build |
|-------|------|--------|-------|
| Narration-of-Thought (main paper) | `papers/acl/ACL_paper.tex` | Canonical and most complete. arXiv preprint prepared 2026-06-24; incorporates the textual-gradient follow-up study as Section 4.2 and Appendix G. | `papers/build.sh acl` |
| Sycophancy under deliberative scaffolding | `papers/sycophancy/sycophancy_paper.tex` | Active. Phase 18 (judge-robust prompt optimization) is the gate for the ACL Rolling Review submission; referee-audit hardening pass completed 2026-06-19. | `papers/build.sh sycophancy` |
| TextGrad follow-up | `papers/archive/followup/followup_paper.tex` | Superseded. Fully integrated into the main paper on 2026-06-24; kept as the originating draft. | `papers/archive/followup/build.sh` |
| Early drafts (position, framing, NeurIPS) | `papers/archive/` | Historical only. | — |

Shared ACL style files and `references.bib` are in `papers/shared/`. The prepared arXiv bundle is `papers/acl/arxiv_submission.zip`.

## Current areas of development

1. Main paper (`papers/acl`): arXiv preprint is prepared; remaining work is the ARR submission cycle (camera-ready polish, reviewer response).
2. Sycophancy paper (`papers/sycophancy`): Phase 18 judge-robust optimization results are in and hardened (held-out judge, human gold, full quartet, BrokenMath replication). Submission is gated on the methods win holding up; Phase 19 (activation steering, open-weight) is the pre-registered backup direction.
3. Deferred: Tier-3 human pairwise preference study (`Guidance_Documents/tier3_preregistration.md`) awaits IRB and funding.

The phase-by-phase history, pre-registrations, and execution logs are in `Guidance_Documents/study_design.md` (main study, Phases 1-19) and `Guidance_Documents/followup_study_design.md` (optimisation follow-up).

## Onboarding quickstart

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install azure-ai-projects azure-identity openai pandas matplotlib seaborn scipy tqdm python-dotenv requests scikit-learn jupyter datasets
cp .env.template .env   # then fill in keys; never commit .env
```

Run the smoke tests before anything expensive:

```bash
python -m scripts.test_phase14
python -m scripts.smoke_generators   # requires live API keys
```

Reproduce the main study (long-running; results are cached per `(generator, judge, scenario, condition, sample)` so re-runs only do new work):

```bash
jupyter nbconvert --to notebook --execute ncot_divergence_pilot.ipynb \
  --output ncot_divergence_pilot.ipynb \
  --ExecutePreprocessor.timeout=7200
```

Phase runners under `scripts/` are standalone; run them from the repository root (they write to `divergence_study_outputs/` relative to the working directory). Model configuration comes from `.env`; see `.env.template` for every variable.

## Reading the results

Start with the auto-generated headline summary in section 7f of the notebook, then the per-generator comparisons in section 9. Interpretation thresholds are in section 8 of the notebook and in `Guidance_Documents/study_design.md`. Committed analysis tables and figures live in `divergence_study_outputs/`.

## Credential rotation

Any API key that has left the local machine (pasted into chat, email, or any web form) must be rotated immediately:

- Azure Foundry key (`AZURE_AI_API_KEY`): Azure portal -> your resource -> Keys and Endpoint -> Regenerate.
- xAI key (`XAI_API_KEY`): console.x.ai -> API Keys -> Delete and create new.

Never commit a filled-in `.env`. The `.gitignore` already excludes it.

## Repository discipline

- The author is Patrick Cooper. All commits are made by him.
- `Guidance_Documents/study_design.md` is the source of truth. When the design changes, the guidance doc is updated in the same commit.
- Markdown is consolidated to this README and `Guidance_Documents/`; no other markdown files.
- API keys are never committed. The `.gitignore` excludes `.env`, `*.key`, and similar.
- Keep commits short and frequent.
