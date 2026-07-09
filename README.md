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
| Narration-of-Thought (main paper) | `papers/acl/ACL_paper.tex` | **ARR resubmission draft** (May 2026 cycle, submission 6086): reviewer-driven revision applied 9 Jul 2026; `[review]{acl}`, anonymized, 7-page main body. arXiv v1 2026-06-24; v2 reference cure uploaded as `papers/acl/arxiv_submission_v2.zip`. | `papers/build.sh acl` |
| Sycophancy under deliberative scaffolding | `papers/sycophancy/sycophancy_paper.tex` | Active. Phase 18 (judge-robust prompt optimization) is the gate for the ACL Rolling Review submission; referee-audit hardening pass completed 2026-06-19. | `papers/build.sh sycophancy` |
| TextGrad follow-up | `papers/archive/followup/followup_paper.tex` | Superseded. Fully integrated into the main paper on 2026-06-24; kept as the originating draft. | `papers/archive/followup/build.sh` |
| Early drafts (position, framing, NeurIPS) | `papers/archive/` | Historical only. | — |

Shared ACL style files and `references.bib` are in `papers/shared/`. The prepared arXiv bundles are `papers/acl/arxiv_submission.zip` (v1) and `papers/acl/arxiv_submission_v2.zip` (reference cure only).

## ARR resubmission (main paper)

**Start here when the next ACL Rolling Review cycle opens.** This is the locator for the May 2026 submission 6086 revision work.

| What | Where |
|------|--------|
| **Paper source (submit from this)** | `papers/acl/ACL_paper.tex` — review mode, anonymous author block |
| **Build** | `cd papers/acl && bash build.sh` → `papers/acl/ACL_paper.pdf` |
| **Reviewer feedback, author responses, revision checklist** | `Guidance_Documents/arr_may2026_reviewer_feedback.md` |
| **Bibliography** | `papers/shared/references.bib` (verified entries for shaffer19, bientzle21/24, rottger23, macaskill20, newberry21, model cards, chiu24data) |
| **Staged figures** | `papers/acl/figures/` (rebuild: `python -m scripts.rebuild_paper_figures && python -m scripts.stage_figures`) |
| **Exp 1 aggregates (five-model panel incl. deepseek-v3)** | `divergence_study_outputs/failure_mode_firing_quartet.csv`, `tier1_effect_sizes_quartet.csv` |
| **DeepSeek-v3 Exp 1 caches** | `divergence_study_outputs/gen_deepseek-v3_*`, `judge_*_gen_deepseek-v3_*` |
| **Phase 11 head-to-head (TGD / debate)** | `divergence_study_outputs/p11_summary.json` |
| **Phase 15 judge-reliability (ELEPHANT caveat)** | `divergence_study_outputs/judge_reliability_summary.json` |
| **arXiv v2 (reference cure, pre-revision body)** | `papers/acl/arxiv_submission_v2.zip` — arXiv 2606.26366 |

**Pre-submit checklist (from the guidance doc):**

1. Rebuild PDF and confirm Limitations starts on page 8 (main body ≤ 8 pages).
2. Confirm `\usepackage[review]{acl}` and anonymous author block (switch to `[preprint]` only after acceptance).
3. Entry-by-entry bibliography spot-check against primary sources.
4. Re-run `python -m scripts.aggregate_phase1` if any new generator caches were added.
5. Cross-check Table 1 and `figures/failure_mode_firing_quartet.pdf` against the CSVs.

Author responses for the May 2026 cycle are archived in the guidance doc (reviewers MqVt, WvQ2, Xrff). Decision: revise and resubmit; do not withdraw; do not commit this version to EMNLP.

## Current areas of development

1. Main paper (`papers/acl`): **ARR resubmission package ready** — see section above. Remaining gate is the next ARR cycle submission window.
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
