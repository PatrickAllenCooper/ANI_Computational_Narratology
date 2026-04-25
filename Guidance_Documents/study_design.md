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

## Scaled DailyDilemmas pilot (Section 12)

Section 12 of the notebook runs the full analysis stack on 100 scenarios sampled from the DailyDilemmas dataset (Chiu et al. 2024), providing higher-fidelity statistics across a broader, automatically-sampled scenario distribution. Dataset: `kellycyy/daily_dilemmas`, `Dilemmas_with_values_aggregated` split (1,360 unique dilemmas).

### Scaled headline results

| Finding | gpt-5.4-nano | gpt-4o |
|---|---|---|
| Tier-1: stakeholder_count length-residualized δ | +0.51 [0.48, 0.54] | +0.75 [0.72, 0.77] |
| Tier-1: uncertainty_score length-residualized δ | +0.77 [0.75, 0.80] | +0.98 [0.98, 0.99] |
| Tier-2: mean JSD(narr, std) | 0.087 | 0.178 |
| Tier-2: mean divergence_excess | 0.055 | 0.138 |
| Tier-2: significant cells (perm p < 0.05) | 26 / 100 | 26 / 100 |
| `uncertainty_suppression` fire rate (std) | 98% | 98% |
| `uncertainty_suppression` fire rate (narr) | 0% | 0% |
| `stakeholder_collapse` fire rate (std) | 80% | 74% |
| `stakeholder_collapse` fire rate (narr) | 0% | 4% |
| MP-NCoT: PRESERVE / AMPLIFY / ELIMINATE (80 informative cells) | 39 / 21 / 20 | (pooled) |

### Key scaled-pilot findings

1. **Uncertainty_suppression fires on 98% of DailyDilemmas scenarios under standard CoT and is eliminated to 0% by narrative CoT on both generators.** The effect is not concentrated in a few hand-picked scenarios; it is near-universal across everyday moral dilemmas.

2. **Stakeholder_collapse is the dominant failure mode** (firing 74–80% of the time in standard CoT). Narrative CoT reduces it to near-zero (gpt-5.4-nano: 0%, gpt-4o: 4%).

3. **Structural effect sizes survive length residualization on both generators.** Unlike the 5-scenario pilot where `uncertainty_score` was length-confounded on gpt-5.4-nano (residualized δ ≈ 0.11), the 100-scenario scaled pilot shows residualized δ = +0.77 on gpt-5.4-nano and +0.98 on gpt-4o — both with CIs far from zero. The larger sample size (N=2,000 per condition × generator cell) tightens the CIs enough to resolve what was noise at N=200.

4. **Tier-1 and Tier-2 are uncorrelated (Spearman ρ ≈ 0).** Scenarios where structural metrics diverge most between narrative and standard CoT are no more likely to show conclusion divergence (Tier-2 JSD) than scenarios where structural metrics are similar. The structural and conclusion dimensions are measuring independent aspects of narrative CoT's effect.

5. **MP-NCoT at scale: PRESERVE is the plurality verdict (39/80 informative cells) but AMPLIFY and ELIMINATE are nearly tied (21 vs 20).** The result is not dominated by one regime. A large minority (120/200 cells) show "n/a — no original divergence"; narrative CoT changed no decisions on scenarios where standard CoT was already deterministic.

6. **Topic stratification reveals real between-domain variation.** Narrative CoT shows strongest Tier-2 divergence on `workplace` scenarios (gpt-5.4-nano, excess=0.20) and `religion_custom` scenarios (gpt-4o, excess=0.41); weakest on `role_duty_responsibility` and `committed_relationship`. This suggests narrative framing is more disruptive when a scenario has contextual and role-based considerations that standard CoT flattens.

### Auto-tagger quality on scaled scenarios

The auto-tagger for failure modes has strong precision/recall on `stakeholder_collapse` (P=0.87–0.93, R=1.00) and `uncertainty_suppression` (P=0.99, R=0.87) — the two failure modes that actually fire. It has zero precision on `premature_refusal`, `framework_enumeration`, and `consequential_flattening` (all auto-tagged but none fire empirically), which means the auto-tagger over-labels those modes. For the analysis cells that condition on auto-tagged failure modes, this introduces noise only for the three non-firing modes; the two firing modes are reliably tagged.

## Section 13 — Multi-Agent Narrative Debate Experiment

Section 13 implements Alvaro's suggestion: turn the multi-protagonist NCoT result into a structured debate. If single-protagonist narrative CoT is a commitment device for the named protagonist (Section 11), what happens when those agents argue with each other? The section re-uses the original five hand-crafted scenarios and their cached MP-NCoT outputs as Round 0 opening statements.

### Design

- **Scope**: 5 original scenarios, 3 stakeholder perspectives per scenario (decider, primary_affected, third_party), N=10 samples per cell, both generators (`gpt-5.4-nano`, `gpt-4o`).
- **Round 0** (opening): Cached MP-NCoT outputs from Section 11, sample_idx 0–9. Zero new API calls.
- **Round 1** (rebuttal): Each agent reads the other two agents' Round 0 statements and writes a rebuttal, optionally revising its position. 300 new generation calls.
- **Round 2** (final position): Each agent reads all Round 1 rebuttals and writes a final position. 300 new generation calls.
- **Moderator consensus pass**: A `gpt-4o-mini` moderator reads the full 9-statement transcript and returns `consensus_reached`, `consensus_decision`, `points_of_disagreement`, and `summary`. 100 calls.
- **Decision extraction**: Per-agent, per-round decision labels extracted by `gpt-4o-mini`. Round 0 decisions reuse Section 11 cache; Rounds 1–2 add 600 new extraction calls.
- **Cache scheme**: `debate_round1_{gen}_{scenario}_{perspective}_{idx}.json`, `debate_round2_*`, `debate_consensus_*`, `debate_dec_{round}_{gen}_{scenario}_{perspective}_{idx}.json`. All artifacts in `divergence_study_outputs/`.

### Saved artifacts

| File | Contents |
|---|---|
| `debate_all_round_decisions.csv` | Per-agent, per-round canonical decision labels (900 rows: 5 x 3 x 10 x 2 generators x 3 rounds) |
| `debate_consensus.csv` | Moderator outputs (100 rows: 5 x 10 x 2 generators) |
| `debate_mind_change_rates.csv` | Per-(agent, scenario, generator, sample) change flag + role |
| `debate_mind_change_by_role.csv` | Aggregated change rate by `decision_role` and generator |
| `debate_consensus_rate.csv` | Consensus rate by scenario and generator |
| `debate_consensus_vs_single_protagonist.csv` | JSD between moderator consensus distribution and single-protagonist NCoT |
| `debate_per_round_distributions_{gen}.png` | Per-scenario x per-round decision distributions |
| `debate_decision_trajectories_{gen}_{scenario}.png` | 3x3 trajectory grids per scenario x generator |
| `debate_consensus_rate_by_scenario.png` | Consensus rate bar chart |

### Headline findings (Section 13)

**Q1: Does debate produce different conclusions than single-protagonist NCoT?**

No, and the mechanism is clear: the consensus rate is very low (6% overall, range 0–15% across scenarios and generators). When the moderator does reach consensus, the decision closely tracks what single-protagonist narrative CoT already produced (JSD ≈ 0.025 — the one scenario with sufficient consensus data). Debate does not unlock different conclusions; it mostly fails to converge, exposing persistent stakeholder disagreement.

**Q2: Do agents update their positions across rounds (genuine deliberation)?**

Yes, but unevenly by role:
- `primary_affected` agents change decision 38% of the time (R0 to R2)
- `decider` agents change decision 20% of the time
- `third_party` agents change decision 14% of the time
- Overall mind-change rate: 24%

Agents do update positions across rounds. The debate is not pure restatement. Directly-affected parties are the most malleable; formal decision-makers and third parties are more stable. This is consistent with the narrative CoT commitment-device finding from Section 11: the decider perspective generates the strongest narrative commitment and is hardest to dislodge.

**Interpretation**: The dominant dynamic is persistent divergence, not convergence. Stakeholders argue across the three rounds but rarely reach common ground, and the rare consensus that does emerge mirrors what single-protagonist NCoT already found. This suggests that narrative-format multi-agent debate, at this scale, functions as a mechanism for *articulating* disagreement rather than *resolving* it. Whether structured resolution prompts (e.g., a defeasibility round after Round 2) could shift the consensus rate is an open question for the next experiment.

## Section 14 — Open Action-Space Debate Experiment

Section 14 is a direct follow-up to Section 13. The core question: was the 6% consensus rate in Section 13 primarily downstream of the closed taxonomy, or does genuine normative divergence persist even when agents are given room to invent novel solutions?

### Design changes from Section 13

R0 and R1 are reused from Section 13 with zero new API calls. R2 is regenerated with an explicit invitation: each agent may either (a) commit to an original-taxonomy option, or (b) propose a novel action not in the taxonomy if the debate revealed concerns the original options cannot accommodate. The moderator is upgraded from a passive consensus-detector to an **active synthesizer** that extracts novel action proposals, normalises them to canonical snake_case labels, identifies cross-agent convergence on novel actions, and when no convergence emerged, attempts to construct a synthesis position from complementary concerns in the R2 texts.

### Saved artifacts (Section 14)

| File | Contents |
|---|---|
| `debate_v2_all_round_decisions.csv` | Per-agent R0/R1/R2_open decisions with `NOVEL` class and `novel_action_label` |
| `debate_v2_consensus.csv` | Active moderator outputs (100 rows) |
| `debate_v2_novel_actions_catalog.csv` | All extracted novel action labels from extractor + moderator |
| `debate_v2_open_vs_closed_comparison.csv` | Per-(scenario, generator) consensus comparison to Section 13 |
| `debate_v2_mind_change_rates.csv` | Per-agent R0->R2_open change flag |
| `debate_v2_mind_change_delta.csv` | Change rate delta (open minus closed) |
| `debate_v2_synthesis_emergence.csv` | Synthesis emergence flag per debate |
| `debate_v2_novel_action_rate_by_role.csv` | Novel-action proposal rate by decision role |
| `debate_v2_novel_action_rate_by_role.png` | Bar chart of above |
| `debate_v2_open_vs_closed_consensus.png` | Grouped bar chart comparing closed and open consensus rates |

All per-sample JSON cache files (`debate_v2_round2_*.json`, `debate_v2_consensus_*.json`, `debate_v2_dec_round2_*.json`) are gitignored.

### Headline findings (Section 14)

**Q1: Does opening the action space produce convergence the closed protocol could not?**

Direct consensus rate increased only marginally: 6.0% (closed, Section 13) to 9.0% (open, Section 14), a delta of +3.0%. Of the open consensus events, 63% were on novel-action kind (not taxonomy). However, the synthesis emergence rate -- the fraction of debates where the active moderator was able to formulate a coherent synthesis position grounded in agent statements -- was 82% overall, ranging from 40% to 100% across scenarios and generators. This is the primary positive finding: the open action space did not produce spontaneous agent-level convergence, but the underlying concerns the agents surfaced were coherent enough that an active synthesizer could construct a synthesis position in the vast majority of cases.

The interpretation: the bottleneck on consensus is not the closed taxonomy alone, nor is it mere stubbornness. Agents do reveal complementary concerns across perspectives (hence the high synthesis emergence rate), but they do not spontaneously recognise that those concerns are compatible (hence the low direct consensus rate).

**Q2: What does a new useful revelation look like empirically?**

- Novel-action proposal rate in R2: 73.2% overall. gpt-4o proposed novel actions in nearly every R2 output (decider: 100%, primary_affected: 96%, third_party: 98%). gpt-5.4-nano was more conservative (52-55%).
- 270 unique novel action labels extracted across 367 catalog entries.
- Top recurrent novel actions span all five scenarios: `gradual_transition_plan` (aging parent), `collaborative_disclosure_plan` / `transparent_conditional_disclosure` (pharma whistleblower), `split_the_dose` (hospital allocation), `shared_hope` (research volunteer).
- Mind-change rate (R0 to R2): 77.9% open vs 24.1% closed, a delta of +53.8%. Opening the action space nearly tripled the rate at which agents moved to a different position. Critically, much of this change is to `NOVEL` -- agents are not just switching between original-taxonomy options but genuinely exiting the original option space.

**The gap between 73% novel-action proposals and 9% consensus** is the central finding. Agents overwhelmingly exit the closed option space when given permission, and when they do, the moderator can usually find a synthesis. But without an explicit synthesis mechanism, those novel proposals do not self-organise into convergence. This confirms the plan hypothesis: the bottleneck is not the taxonomy, it is the absence of a mechanism that surfaces the compatibility of divergent concerns.

### Implication for next experiments

The synthesis positions the moderator formulated represent exactly the "new useful revelations in the space of normative possibility" the Section 14 plan was designed to surface. A natural next step is a Round 3 in which those moderator-generated synthesis positions are presented back to the agents and each agent is asked to accept, modify, or reject the synthesis with justification. This would test whether a structured mediator round can convert the latent synthesis potential (82% emergence rate) into actual agent-level convergence.

## Change log

- v0.1 -- Drafted from author's study-design email plus the second-iteration notebook (dual-judge, decision-extractor, failure-mode-targeted analysis, Cliff's delta + bootstrap CIs, JSD with bootstrap CI).
- v0.2 -- Added robustness layers documented inline above: cross-judge directional check, length residualization, decision entropy. Tier-1 interpretation threshold tightened to require length-residualized effect, not just raw effect.
- v0.3 -- Added second generation model (`gpt-4o`); promoted `claude-sonnet-4-6` to primary cross-vendor judge; tightened `max_causal_hops` rubric with calibration anchors; added per-generator failure-mode firing analysis; cache schema now keys every artifact by `(generator, judge)` so adding either dimension is non-destructive.
- v0.4 -- Added (a) deterministic regex-based causal-hop counter; (b) per-scenario inter-judge kappa breakdown; (c) length-confound scatter diagnostic + length-overlap effect-size analysis; (d) permutation test for Tier-2 cross-condition JSD; (e) high-level synthesis section.
- v0.5 -- Added Section 12: scaled DailyDilemmas pilot (100 scenarios, full Tier-1/Tier-2/MP-NCoT/length-residualization/family-aggregation analysis stack, plus scale-only analyses: per-scenario effect-size histogram, Tier-1 vs Tier-2 correlation, topic stratification). Key finding: the original 5-scenario pilot's core results replicate at scale with substantially tighter CIs; `uncertainty_suppression` and `stakeholder_collapse` fire in 74-98% of standard-CoT responses and are near-universally eliminated by narrative CoT across both generators.
- v0.6 -- Added Section 13: multi-agent narrative debate experiment (5 scenarios x 3 perspectives x 10 samples x 2 generators x 3 rounds + moderator). Key findings: overall consensus rate is 6%; when consensus is reached it tracks single-protagonist NCoT (JSD ~0.025); overall mind-change rate is 24% with `primary_affected` agents most malleable (38%) and `third_party` agents most stable (14%).
- v0.7 -- Added Section 14: open action-space debate experiment. Key findings: novel-action proposal rate 73.2% (gpt-4o ~98%, gpt-5.4-nano ~50%); direct consensus rate +3% vs closed; mind-change rate +53.8% vs closed; synthesis emergence rate 82%. The central finding is that agents overwhelmingly exit the closed option space when permitted, the moderator can usually find a synthesis from the revealed concerns, but agents do not self-organise convergence without an explicit synthesis presentation mechanism.
