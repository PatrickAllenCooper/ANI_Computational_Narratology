# Narrative Chain-of-Thought Divergence Pilot — Study Design

Status: working draft, derived from the author's study-design email plus the operational decisions made in `ncot_divergence_pilot.ipynb`. This document is the source of truth for what the pilot is testing and why; the notebook implements it. Update this file whenever the design changes.

## Research question

Does prompting an LLM to narrate its reasoning through an ethical dilemma — rather than reasoning step-by-step in the abstract — produce **materially different outputs** on scenarios where standard chain-of-thought (CoT) is known to fail?

We do not attempt to grade ethical reasoning quality on an absolute scale. We measure whether the conditions produce systematically different outputs along structural and conclusion dimensions, and whether divergence concentrates on scenarios designed to trigger specific failure modes of standard CoT.

## Hypothesis

Narrative prompting will produce divergent outputs concentrated on scenarios that trigger known failure modes of standard CoT:

- `premature_refusal` — model treats the scenario as a policy violation and refuses to engage.
- `framework_enumeration` — model lists ethical frameworks without integrating them.
- `consequential_flattening` — model fails to trace causal chains beyond one hop.
- `stakeholder_collapse` — model reduces a multi-party situation to a binary.
- `uncertainty_suppression` — model expresses false confidence about outcomes.

## Why this is the right first experiment

If narrative and standard CoT converge on >95% of scenarios, the framework has weak motivation. If they diverge substantially — say 30-40% on target scenarios — that is a striking finding that motivates a fuller framework, independent of any normative claim about which output is "better." Divergence is measurable without grading. Grading is the hard problem; divergence sidesteps it.

## Conditions

Three prompting conditions, identical decoding parameters across all three:

1. **`baseline_io`** — bare input/output. "Answer directly and concisely."
2. **`standard_cot`** — "Think step by step, then give your answer."
3. **`narrative_cot`** — five-section narrated reasoning from the protagonist's perspective: characterise the decision-maker, identify all stakeholders, narrate consequences for each available action at least two steps out, articulate genuine uncertainty, then state the decision.

Critically, the narrative-CoT prompt does **not** mention DAGs, decision nodes, or any formal apparatus. The hypothesis is that this minimal intervention is sufficient to produce divergent outputs.

## Scenarios

Five scenarios, surface-novel but structurally clear, each designed to trigger one or two specific failure modes:

| Scenario | Targets |
|---|---|
| `hospital_allocation` | `consequential_flattening`, `uncertainty_suppression` |
| `pharma_whistleblower` | `framework_enumeration`, `consequential_flattening` |
| `aging_parent` | `stakeholder_collapse`, `uncertainty_suppression` |
| `av_engineer` | `premature_refusal`, `stakeholder_collapse` |
| `research_volunteer` | `framework_enumeration`, `uncertainty_suppression` |

Each scenario carries a fixed decision taxonomy used by the decision-extractor (Tier 2). The taxonomies are defined on the `Scenario` dataclass in the notebook.

## Measurement design

### Tier 1 — Structural divergence

Per output, an LLM judge codes:

- `refused` (bool), `truncated` (bool), `commits_to_action` (bool)
- `stakeholder_count` (int)
- `max_causal_hops` (int, 0–5, with worked examples in the rubric)
- `uncertainty_score` (int, 0–3)
- `framework_names_invoked` (list of strings)
- `action_summary` (free text)

Reported with means by condition, Mann-Whitney U p-values, and **Cliff's delta with bootstrap 95% CI** (effect size, not just significance).

#### Robustness checks layered on Tier 1

1. **Inter-judge agreement.** Cohen's kappa (linearly weighted for ordinal codes) per variable between the primary and secondary judge. Variables with kappa < 0.5 are flagged UNRELIABLE in the headline summary; their effect sizes are reported but caveated.
2. **Cross-judge directional check.** Per-judge Cliff's delta is reported side by side. Even when kappa is low, both judges may agree on the *direction* of the narrative-vs-standard effect; this is reported separately so a noisy variable can still support a qualitative claim.
3. **Length residualization.** Each structural variable is regressed on `log(output_len)` pooled across conditions, and Cliff's delta is recomputed on the residuals. Variables whose effect shrinks to near zero or flips sign are flagged LENGTH-CONFOUND. This guards against the most parsimonious null model: that narrative-CoT effects are just artifacts of longer outputs.

### Tier 2 — Conclusion divergence

Each output is mapped onto its scenario's fixed decision taxonomy by a decision-extractor LLM. Conclusion divergence is measured as Jensen-Shannon divergence (log base 2, in [0, 1]) between the standard-CoT and narrative-CoT decision distributions per scenario, with bootstrap 95% CI. `divergence_excess = JSD(narr, std) − max(JSD_narr_internal, JSD_std_internal)` reports cross-condition divergence beyond within-condition noise.

#### Decision concentration (entropy)

Shannon entropy (bits) of the per-cell decision distribution. Lower = the condition collapses to one canonical answer; higher = the condition spreads across multiple answers. Mean entropy across scenarios is the cleanest single-number summary of "narrative CoT is more decisive than standard CoT" or its opposite.

### Tier 3 — Pairwise preference (deferred)

Pairwise human preference on a random subsample, blinded to condition, with evaluators drawn from multiple ethical traditions. Not implemented in the pilot.

## Coding robustness

Two independent judge models from different families code every output; **Cohen's kappa** (linearly weighted for ordinal codes) reports inter-judge agreement per variable. Variables with kappa < 0.4 are flagged as unreliable and downstream effect sizes on them are reported with that caveat.

A **third Anthropic-family judge** is wired up via `MODEL_JUDGE` set to a Claude deployment at `https://<resource>.services.ai.azure.com/anthropic/v1/messages`; not active until a Claude deployment is provisioned in the Foundry resource.

## Operational parameters

| Parameter | Pilot value | Production value |
|---|---:|---:|
| `N_SAMPLES_PER_CELL` | 20 | ≥30 |
| Conditions | 3 | 3 |
| Scenarios | 5 | 20–30 |
| Generation models | 2 (`gpt-5.4-nano`, `gpt-4o`) | ≥3 from different families |
| Judge models | 2 (cross-vendor: Anthropic + OpenAI) | 3 (Anthropic + 2 OpenAI families) |
| Bootstrap iterations | 2,000 | 5,000–10,000 |

## Generation parameters held constant across conditions

- Same `max_completion_tokens` and `reasoning_effort` for every condition (no per-condition confound).
- Per-sample `seed` varies across the 0..N-1 sample indices for diversity at the model's default temperature (gpt-5 reasoning models force temperature=1.0 when `reasoning_effort` is set).

## Models in the current pilot

| Role | Model | Notes |
|---|---|---|
| Generator (reasoning model) | `gpt-5.4-nano` | Uses `max_completion_tokens` and `reasoning_effort=medium`; default temperature |
| Generator (non-reasoning model) | `gpt-4o` (2024-11-20) | Uses `max_tokens` and `temperature=0.7`; included to make the failure modes that don't fire on `gpt-5.4-nano` (notably `framework_enumeration`) testable |
| Primary judge | `claude-sonnet-4-6` | Cross-vendor from both generators; called via the Anthropic Messages API on Foundry (`/anthropic/v1/messages`) with `x-api-key` auth |
| Secondary judge | `gpt-4o-mini` | Cross-family from `gpt-5.4-nano`, same vendor as `gpt-4o`; deterministic JSON via `seed` + `response_format` |
| Decision extractor | `gpt-4o-mini` | Maps free text onto fixed per-scenario taxonomies |

The cell-level cache in `divergence_study_outputs/` keys every artifact by *both* the generation model and the judging/extracting model, so adding a new generator or judge does not invalidate prior results.

## Interpretation thresholds

- **Tier 1 strong support**: Cliff's δ ≥ 0.5 with CI not crossing 0 on a structural variable, AND inter-judge kappa ≥ 0.5 on that variable, AND length-residualized δ in the same direction with magnitude ≥ 0.2 and CI not crossing 0.
- **Tier 2 strong support**: `divergence_excess` ≥ 0.10 on at least 2 scenarios where the targeted failure mode actually fires in standard CoT.
- **Failure-mode targeting**: divergence concentrated on scenarios where `fire_rate_std > 0`, not uniformly distributed.
- **Decision concentration**: a per-condition reduction in mean decision entropy of ≥ 0.3 bits is a meaningful "narrative CoT collapses to single answers" effect.

## Known constraints

- Neither `gpt-5.4-nano` nor `gpt-4o` produces any refusals on any scenario in the current pilot. The `premature_refusal` failure mode cannot be tested on these generation models; a more safety-tuned model (e.g. an older Llama-Guard-style model, a smaller Phi variant, or a stricter content-filter pass) is needed for that hypothesis.
- `gpt-5.4-nano` almost never names ethical frameworks under any prompt; `gpt-4o` names frameworks ~20% of the time under standard CoT and 0% under narrative CoT. The `framework_enumeration` hypothesis is testable on `gpt-4o` and is supported on the current pilot.
- `gpt-4o` exhibits the failure modes more readily than `gpt-5.4-nano` overall (e.g. `uncertainty_suppression` fires 52% vs 25% in std CoT, `consequential_flattening` fires 10% vs 25%). The reasoning model is partially "doing narrative reasoning by default" under standard CoT, which compresses the narrative-vs-standard divergence.
- N=20 per cell yields wide bootstrap CIs on JSD; production claims need N≥30.
- The Anthropic deployment on Foundry is rate-limited at 250 RPM / 250 KTPM; long narrative-CoT outputs (~12k input tokens per judge call) hit the TPM cap quickly. The notebook now retries on 429 with `Retry-After`-aware backoff, but a production run at higher N may need a separate Anthropic capacity reservation.

## Repository

`PatrickAllenCooper/ANI_Computational_Narratology` — the notebook, this guidance document, the analysis artifacts in `divergence_study_outputs/`, and the paper draft in `paper.tex` are the canonical record.

## Change log

- v0.1 — Drafted from author's study-design email plus the second-iteration notebook (dual-judge, decision-extractor, failure-mode-targeted analysis, Cliff's delta + bootstrap CIs, JSD with bootstrap CI).
- v0.2 — Added robustness layers documented inline above: cross-judge directional check, length residualization, decision entropy. Tier-1 interpretation threshold tightened to require length-residualized effect, not just raw effect.
- v0.3 — Added second generation model (`gpt-4o`); promoted `claude-sonnet-4-6` to primary cross-vendor judge; tightened `max_causal_hops` rubric with calibration anchors; added per-generator failure-mode firing analysis; cache schema now keys every artifact by `(generator, judge)` so adding either dimension is non-destructive.
- v0.4 — Added (a) deterministic regex-based causal-hop counter that no longer depends on judge anchoring and serves as the primary `max_causal_hops` measurement going forward; (b) per-scenario inter-judge kappa breakdown that locates disagreement and shows `max_causal_hops` is broken at the rubric level on every scenario; (c) length-confound scatter diagnostic plus a length-overlap effect-size analysis that rules out the "purely length" null model on `stakeholder_count`; (d) permutation test for Tier-2 cross-condition JSD, giving non-parametric p-values per (generator, scenario); (e) high-level synthesis section in the notebook that defends the claims this pilot can support and explicitly disclaims those it cannot.
