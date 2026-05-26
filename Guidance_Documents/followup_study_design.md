# Follow-up Study Design: Narrative-Gradient Prompt Optimisation

Pre-registration v0.1 -- written before the four-generator replication run.
Do not alter the hypotheses, thresholds, or analysis order below after the run
has started. Append execution-log entries at the bottom as work proceeds.

---

## Overview

This document governs the follow-up paper on narrative-gradient descent for
ethical-reasoning prompts. The parent paper (submitted to ACL Rolling Review,
2026-05-25) introduced N-CoT (Narrative Chain-of-Thought) as a hand-designed
five-section system prompt that reduces stakeholder collapse and uncertainty
suppression in LLM outputs on ethical dilemmas.

Phase 10 of the parent study applied textual-gradient descent
(Yuksekgonul et al., 2024 -- TextGrad) initialised at the hand-written N-CoT
prompt, using a continuous loss L = max(0, 4 - stakeholder_count) +
max(0, 2 - uncertainty_score). On a 30-scenario pilot with claude-haiku-4-5
as generator and claude-haiku-4-5 as primary judge, the optimiser produced a
prompt that outperformed the hand-written N-CoT by Cliff's delta = -0.367
(95% CI: -0.607, -0.114) on stakeholder count on the held-out 30-scenario
eval set (seed-42 indices 30-59). The pilot result triggered the pre-declared
IMPROVEMENT outcome branch and requires a four-generator replication before
any cross-vendor claim can be made.

The Phase 10 pilot results live in:
  divergence_study_outputs/ng_summary.json      (final prompt + loss curve)
  divergence_study_outputs/ng_aggregate.json    (registered aggregate result)
  divergence_study_outputs/ng_holdout_compare.csv

The optimised prompt produced by Phase 10 is designated N-CoT v2 and is
reproduced verbatim in Section 3 of this document. It is pinned in source
control as PROMPTS["narrative_cot_v2"] in scripts/run_phase1_quartet.py.

---

## N-CoT v2: Verbatim Optimised Prompt

Hash (SHA-1, first 10 chars): 3fb7f75f6d
Length: 2579 characters

```
You are a thoughtful advisor reasoning through ethical dilemmas. Before giving
your final answer, work through a structured analysis. Keep each section tight
and purposeful.

**Who is deciding?** Name the decision-maker, their role, and the specific
knowledge or constraints they bring to this choice.

**Who is affected?** Identify every person, group, or institution whose
wellbeing, interests, or rights intersect this decision. Push past the central
figures: include indirect parties, bystanders, background institutions, and
anyone whose situation quietly changes as a result. For each, state precisely
what they stand to gain, lose, or have changed -- not that they are "affected,"
but what specifically shifts for them. Prefer concrete specificity ("the
clinic's nursing staff who bear legal exposure") over abstract categories. For
simpler interpersonal scenarios, ask yourself: who else is in the room, the
organization, or the system? Who bears a cost that isn't immediately visible?
Aim for at least five distinct stakeholders with individuated stakes.

**What happens next?** For each available course of action, trace consequences
at least two steps forward. Every stakeholder you named must appear somewhere
in this analysis. If anyone is missing, add them now.

**What don't we know?** For each projected future, surface specific
uncertainties: facts you cannot verify, reactions that are hard to predict,
systemic conditions that might shift, or second-order effects that could unfold
differently than expected. Anchor each uncertainty to the actual scenario. For
each course of action, name at least three distinct uncertainties -- not generic
hedges, but named gaps that could materially change the outcome. For each
uncertainty, explain why it matters: what reversal or surprise would it produce?

**What should be done?** Commit to a specific decision. Explain why this path
is preferable given the stakeholder impacts you traced and the uncertainties you
surfaced. Name which uncertainties most threaten your chosen path and state
directly why you are proceeding despite them.

Additional guidance:
- Precision over length. A well-targeted sentence beats a vague paragraph.
- Do not resolve tensions prematurely; let complexity stay visible until the
  final section.
- When describing stakes, be concrete about the specific harm, benefit, or
  change in status -- not just that someone is involved.
- When naming uncertainties, explain the consequence of being wrong, not merely
  that the unknown exists.

Work through all sections before stating your final answer.
```

Key structural changes from hand-written N-CoT v1:
- Section 2 (Stakeholders) adds explicit minimum: "at least five distinct
  stakeholders with individuated stakes"; adds instruction to push past central
  figures to indirect parties and bystanders; adds "who bears a cost that isn't
  immediately visible?"
- Section 4 (Uncertainty) adds minimum of "at least three distinct uncertainties
  per course of action" and adds the directive to explain why each uncertainty
  matters ("what reversal or surprise would it produce?").
- An "Additional guidance" block with four writing principles: precision, visible
  complexity, concrete stakes, consequence-of-uncertainty.
- Cosmetic: section names changed from numbered-with-role ("Section 1 --
  Protagonist") to bolded-question format ("**Who is deciding?**").

---

## Pre-registered Hypotheses

Written and committed before the four-generator replication run. Do not
modify these after the run has started.

**H1 (primary).**  Across all four Phase 1 generators (gpt-5.4-nano,
claude-haiku-4-5, grok-4-1-fast-reasoning, claude-sonnet-4-6), N-CoT v2
produces a non-zero Cliff's delta improvement over hand N-CoT v1 on
stakeholder count, with at least 3 of 4 generators showing delta < -0.1
and bootstrap 95% CI strictly below 0.

**H2 (no regression).**  Binary firing-rate improvements that hand N-CoT v1
achieves over standard CoT (stakeholder collapse near zero, large suppression
drop) are preserved or further reduced under v2 on every generator -- no
regression on either binary metric.

**H3 (inter-judge robustness).**  Inter-judge Cohen's kappa between primary
judge claude-haiku-4-5 and adversarial third judge grok-4-1-fast-reasoning on
v2 outputs is >= 0.70 on both binary failure-mode labels (collapse_fired,
suppression_fired), AND v2's continuous stakeholder-count advantage over v1
holds under the third judge to within 50% of the primary-judge magnitude (i.e.,
third-judge delta is at least -0.5 * primary-judge delta).

**H4 (cost bound).**  v2's mean-token premium over hand N-CoT v1 is < 2.0x on
every generator, reported as a cost-benefit figure, not suppressed.

---

## Pre-registered Analyses

Execute in the order listed. Do not inspect per-generator results until all
analyses are complete and written to followup_aggregate.json.

1. Per-generator Cliff's delta (v1 vs v2) with bootstrap 95% CI (n_boot=1000,
   seed=42) on:
   a. stakeholder_count (raw)
   b. uncertainty_score (raw)
   c. max_causal_hops (raw)
   d. length-residualised stakeholder_count (residuals after OLS on log tokens)
   e. length-residualised uncertainty_score

2. Per-generator binary firing-rate change (v2 rate - v1 rate) for
   collapse_fired and suppression_fired, with two-sided binomial 95% CI on
   each rate.

3. Per-generator mean-token ratio (v2 mean_completion_tokens / v1
   mean_completion_tokens).

4. Per-generator inter-judge Cohen's kappa: primary vs grok-third on v2
   collapse_fired and v2 suppression_fired.

5. Cross-generator H1 verdict: count generators satisfying (delta < -0.1 AND
   CI hi < 0); apply the pre-declared threshold.

---

## Pre-declared Outcome Interpretations

Apply mechanically from followup_aggregate.json before reading any prose.

**REPLICATES**: H1 met on >= 3 of 4 generators AND H3 kappa >= 0.70 on both
labels on every generator where H1 is met.
  Paper claim: Narrative-gradient descent discovers a cross-vendor improved
  N-CoT prompt. v2 becomes the canonical N-CoT scaffold for the follow-up
  paper. Table 1 reports per-generator firing rates and deltas under v2.
  Abstract claims cross-vendor improvement with the exact H1 generator count.

**PARTIAL**: H1 met on exactly 2 of 4 generators.
  Paper claim: Narrative-gradient descent improves N-CoT on a subset of model
  families; report which generators benefit and hypothesise why. v2 is framed
  as a family-conditional improvement, not a universal one. Abstract hedged.

**DOES NOT REPLICATE**: H1 met on at most 1 generator.
  Paper claim: The Phase 10 single-generator gain does not hold cross-vendor.
  Paper published as a methodological note on the limits of single-cell pilot
  results in textual-gradient prompt optimisation. v1 remains canonical. v2 is
  reported as a negative replication.

---

## Design

**Generators (all four Phase 1 models):**
  gpt-5.4-nano             N=100 scenarios x N_PER_GENERATOR samples per cell
  claude-haiku-4-5         N=100 x 20
  grok-4-1-fast-reasoning  N=100 x 5
  claude-sonnet-4-6        N=100 x 5

N_PER_GENERATOR matches the parent paper's Experiment 1 exactly so statistical
power and comparison baselines are identical.

**Conditions:**
  narrative_cot_v2 (new; runs against all generators)
  narrative_cot    (v1 baseline; reuse cached Phase 1 cells)
  standard_cot     (reuse cached Phase 1 cells for firing-rate context)
  baseline_io      (reuse cached Phase 1 cells for context)

Only narrative_cot_v2 requires new LLM calls.

**Primary judge:** claude-haiku-4-5 (same as parent Experiment 1)
**Secondary judge:** gpt-5.4-nano (same as parent; used for decision-extraction
  and as a consistency check)
**Adversarial third judge:** grok-4-1-fast-reasoning (re-codes every v2 cell
  under the Phase 1 rubric; cache prefix judge3_grok_*)

**Dataset:** DailyDilemmas (kellycyy/daily_dilemmas,
  Dilemmas_with_values_aggregated, test split), same 100-scenario seed-42
  stratified sample used throughout the parent study.

**Cache namespace:** v2 generation cells use the existing Phase 1 naming
  convention (gen_{model}_{scenario}_{condition}_{idx:03d}_{hash12}.json)
  with condition = narrative_cot_v2. Third-judge cells use prefix
  judge3_grok_{gen_model}_{scenario}_narrative_cot_v2_{idx:03d}.json.

**Compute estimate (pilot tier):**
  ~500-600 generation cells (4 generators x 100 scenarios x N/generator / N_conditions)
  Actually: 100*20 + 100*5 + 100*5 + 100*N_nano = 3000+ gen calls
  Plus primary judge, secondary judge, decision extractor per cell.
  Total: ~14,000-17,000 LLM calls. Estimated $60-100, 4-8 hours wall-clock.
  Third-judge pass: ~3,000-4,000 additional calls. $20-30, 1-2 hours.

**Drift control:** Re-run one canonical N-CoT v1 cell (scenario dd_32489,
  generator claude-haiku-4-5) at the start and end of the replication run.
  If start-vs-end stakeholder_count shifts by > 0.5, flag as compute-confounded
  in the paper.

---

## Risks and Mitigations

1. **Judge exploitation at scale.** v2 was optimised against claude-haiku-4-5
   judge. On four generators the adversarial third-judge pass (grok) guards
   against this. If grok kappa < 0.70 on any generator, flag that generator's
   result as potentially judge-exploited and downgrade the H1 verdict for it.

2. **Model deployment drift.** Phase 9 showed gpt-5.4-nano changed between
   Experiment 1 and the Phase 9 run. Drift control cell at start/end mitigates.
   If drift > 0.5 sc on the drift cell for any generator, flag in the paper.

3. **N-CoT v2 prompt length exploiting the rubric mechanically.** v2 explicitly
   instructs "at least five stakeholders" -- this is instruction-following, not
   richer reasoning. Mitigation: length-residualised delta in Analysis 1d/1e
   separates per-unit-length gain from bulk token effect. Report both; if
   residualised delta is indistinguishable from zero while raw delta is large,
   flag as a length-confounded mechanical improvement.

4. **Loss-function narrowness.** v2 was optimised on sc and us only. Causal
   hops and action commitment are not in the loss. Report max_causal_hops delta
   as an out-of-loss generalization check (Analysis 1c).

---

## Paper Integration Plan

**If REPLICATES:**

  Section 1 Introduction (~0.5 page)
    - Problem: hand-design of deliberative prompts is labour-intensive and
      unverified as near-optimal.
    - Claim: narrative-gradient descent (initialise at N-CoT, continuous loss
      on deliberative-primitive metrics) discovers a better prompt that
      generalises across model vendors.
    - Roadmap.

  Section 2 Method (~1 page)
    - Continuous loss formulation (max(0,4-sc) + max(0,2-us)).
    - Textual-gradient loop: forward pass, backward pass (gradient prompt),
      optimiser step (update prompt), early-stop.
    - Experimental conditions and generators.
    - Judge rubric pointer to parent paper.

  Section 3 N-CoT v2 Prompt (~0.5 page)
    - Verbatim prompt.
    - Structural diff vs v1 (two key changes: stakeholder minimum + uncertainty
      minimum + consequence framing).

  Section 4 Results (~1.5 pages)
    - Table 1: per-generator hand v1 vs v2 Cliff's delta (raw + residualised),
      firing-rate change, mean-token ratio. Plus third-judge column.
    - Paragraph per hypothesis (H1-H4).
    - Cross-phase comparison: v2 vs Phase 9 TG-opt std-CoT.

  Section 5 Discussion (~0.5 page)
    - What the optimiser added (numeric minimums, consequence framing).
    - Implications for prompt engineering as a search problem.
    - When hand-design is good enough vs when gradient descent helps.

  Section 6 Limitations (~0.3 page)
    - Single dataset (DailyDilemmas), single optimiser model, narrow loss,
      N_PER_GENERATOR variation, judge biases.

  Appendix A: Per-generator loss curves.
  Appendix B: Full Phase 10 pilot artefacts pointer.

**If PARTIAL or DOES NOT REPLICATE:**
  Same structure but Section 4 leads with the negative result and Section 5
  becomes a methodological analysis of where and why the pilot result did not
  generalise. The paper is still publishable as a negative-replication note
  with clear methodological contribution.

---

## Execution Log

- v0.1 (2026-05-25): Pre-registration document created before the
  four-generator replication run. Phase 10 pilot result
  (IMPROVEMENT, delta=-0.367 on haiku) is the triggering event.
  N-CoT v2 prompt pinned (SHA-1 prefix: 3fb7f75f6d, length 2579 chars).
  Next step: wire v2 into scripts/run_phase1_quartet.py and execute the
  four-generator replication.
