# Activation-Guided Thinking (AGT) — Follow-up Project Sketch

Status: rough sketch (2026-07-08). This folder seeds the follow-up line that generalizes the parent project's findings toward inference-time activation control. Before any expensive execution, the chosen subset of these ideas must graduate to a pre-registration block in `Guidance_Documents/study_design.md`, per repository discipline. The Phase 19 pre-registration note there (activation steering, open-weight) is the seed this project grows from; running it as an active line rather than a Phase-18-failure backup is a logged deviation (see execution log v1.17).

## The abstraction move

The parent project established two behavioral facts using black-box prompting:

1. A narrative scaffold (NoT) reshapes the structure of moral reasoning: stakeholder collapse and uncertainty suppression drop on every tested generator, attributable to specific sub-instructions.
2. "Sycophancy" fragments into constructs (social validation, propositional agreement, moral endorsement) that respond differently to the same intervention, and LLM-judge-optimized prompts Goodhart the judge rather than the behavior.

Both facts live at the prompt level. AGT asks the general question one level down: is "thinking shape" a controllable direction in activation space, and does controlling it there escape the failure modes of controlling it with text? Prompts are surface-visible (gameable, judge-legible, brittle); activation interventions are not part of the token stream. That asymmetry is the core of the abstraction.

## Lines of inquiry (rough)

1. Prompt-to-vector distillation. Extract a "narration direction" by contrasting residual streams under NoT vs standard-CoT on matched scenarios (CAA / difference-in-means). Test whether adding the vector at inference reproduces NoT's structural effects (stakeholder count, uncertainty score) with no scaffold prompt. Success would show the scaffold is a probe of a latent reasoning-structure axis, not a trick of instruction-following.
2. Construct geometry (Phase 19 core). Do validation, propositional, and moral sycophancy occupy distinct, independently steerable directions? Extend to an atlas of the parent project's coded failure modes (stakeholder collapse, uncertainty suppression, premature foreclosure). Orthogonality and cross-steering interference are the measurements.
3. Composability. If narration and anti-sycophancy are separate directions, do they compose additively (steer both, get both effects)? Interference patterns are evidence about shared vs distinct circuitry, and composable control is the practically valuable capability.
4. Goodhart resistance. The Phase 18 result showed prompt optimization against a judge reverse-engineers the rubric. Hypothesis: steering-induced reductions are less judge-legible than prompt-induced ones (no banned-word artifacts, no format tells), so steered wins should transfer across held-out judges better than prompt-optimized wins at matched behavioral effect. This directly reuses the Phase 18 Goodhart-gap machinery as the measuring instrument.
5. Dose-response. A steering coefficient is a continuous dial where a prompt is a binary switch. Map coefficient-vs-construct curves and per-layer effects; identify whether structural gains and side effects (e.g. the nano over-refusal effect) decouple at intermediate doses.
6. Cross-model transfer. Do the directions found on Llama-3.1-8B transfer to Qwen3-8B (and across sizes) after subspace alignment? Universality evidence would elevate the finding from a model fact to a representation fact.
7. Dual-stance specificity audit (mandatory, inherited from the Phase 19 pre-registration). Every steering result must show reduced sycophantic agreement without suppressed factually-correct agreement.

## Method sketch

- Models: Llama-3.1-8B and Qwen3-8B (open weights, residual-stream access), per the Phase 19 pre-registration.
- Interventions: Contrastive Activation Addition for direction extraction; linear probes for geometry claims; activation patching for localization if warranted.
- Contrast data: matched (NoT, standard-CoT) generations over the existing scenario sets — the 100-scenario DailyDilemmas sample, ELEPHANT OEQ items, BrokenMath perturbed problems. No new benchmark construction needed to start.
- Evaluation: the parent project's instruments unchanged — `scripts/elephant_scorers.py`, `scripts/brokenmath_scorer.py`, `scripts/syco_loss.py`, `scripts/judge_panel.py`, `scripts/krippendorff.py` — so behavioral results are directly comparable to the papers' tables.

## Reuse map

| Existing asset | Role in AGT |
|---|---|
| `scripts/run_phase1_quartet.py::PROMPTS` | Canonical NoT / CoT / verbose-CoT prompt texts for contrast pairs |
| `divergence_study_outputs/phase18_robust.json` | Panel-robust optimized prompt + training curves (the prompt-level ceiling to beat) |
| `divergence_study_outputs/phase18_goodhart.json`, `phase18_heldout.json`, `phase18_quartet.json` | Goodhart-gap and held-out-judge baselines for line 4 |
| `divergence_study_outputs/tier1_effect_sizes.csv`, `cross_generator_tier1.csv` | Behavioral effect sizes the distilled vector must reproduce (line 1) |
| `divergence_study_outputs/judge_reliability_summary.json` | Judge-panel reliability context for any LLM-scored outcome |
| `scripts/load_elephant.py`, `scripts/load_brokenmath.py` | Task loaders (raw datasets are gitignored; loaders handle download/fallback) |

## Getting started

`agt_bootstrap.ipynb` in this folder is the single entry point. It loads the parent project's headline artifacts, exposes the canonical prompt assets, builds contrast-pair specifications, and stubs the steering pipeline behind a GPU guard so the notebook runs end-to-end on a laptop and picks up the heavy sections on a cluster.

## Infrastructure notes

- GPU work targets CURC. Store activations and model weights on scratch (`/scratch/alpine`), never `/projects` (near quota). Residual-stream dumps for 8B models at fp16 are large; cache per-layer means, not full streams, unless a specific analysis needs them.
- API-scored evaluations reuse the parent `.env` configuration and the per-cell caching conventions of `divergence_study_outputs/`.
