# NoT origins review: what the original programme said, what was dropped, and what the role-embodiment pivot should inherit

Date: 2026-08-31. Branch: `sycophancy-scenario-redesign`.

Purpose. The PI has confirmed the programme's three live themes: (1) the design
question (can a prompt alone reduce sycophancy), (2) the split verdict (a real
register effect that backfires on propositional ground truth), and (5) role
embodiment as the growth axis. The PI has also restated the original intent of
Narration-of-Thought: role embodiment and identification of causes and effects
within the model's world conception, so the model develops a better theory of
mind, yielding superior alignment, less sycophancy, and moral judgments that
better align with human moral judgment. This document audits the pre-pivot
materials (`papers/archive/position_paper.tex`, `papers/archive/framing_paper.tex`,
`papers/acl/ACL_paper.tex`, `Guidance_Documents/study_design.md` and the five
sibling design documents) against that intent, and lists every justification,
experimental idea, and theme that was dropped or under-carried, so the pivot
inherits deliberately rather than by accident.

Sources were read in full by four parallel audits on 2026-08-31. Citation
veracity was spot-checked: Trabasso & van den Broek 1985 (JML 24:612-630),
Chen & Bornstein 2024 (TiCS), Shaffer et al. 2019 (PLoS ONE), and the Bientzle
perspective-taking work are all real and correctly invoked; the ARR reviewer's
bibliography corrections were applied in `papers/shared/references.bib`.

---

## 1. The original conception already contains the pivot's theory

The PI's restatement is not a new direction. It is the original direction,
which the ARR-era empirical work narrowed away from. Three passages carry it.

**Perspective simulation as conditional world-model rollout.**
`position_paper.tex` (l.174) defines perspective simulation as "the conditional
execution of M* obtained by fixing a subset of variables to a character's
subjective state (beliefs, desires, accessible knowledge) and forward-simulating."
This is a theory-of-mind operation stated as causal inference over a world
model. It is the exact formal object the pivot needs, and it appears nowhere in
the current manuscript or in the embodiment arms as built.

**Mental-state abduction (Kuleshov).** Narrative comprehension is modeled as
hypothesising latent minds to explain event sequences (l.164). This grounds why
narration, specifically, should exercise ToM machinery that abstract CoT does not.

**Sycophancy as a high-K_C trajectory.** Maintaining a delusion is causally
expensive across simulated futures; the false-counsellor archetype ends in
peripeteia (ll.65, 182). This is the cause-and-effect-in-world-conception claim.
It was operationalised once, as design 4.6 (Costly-Correction) in
`sycophancy_scenario_redesign.md`, and never run.

The important nuance: the original theory predicts sycophancy reduction through
*simulation of consequences*, not through *commitment*. The scaffold as built
bundles both. Section 2 below argues the bundle is what backfired.

## 2. The unifying mechanistic hypothesis the archive hands us

Four previously separate observations line up, and the first is as old as the
programme itself.

1. **The first pilot's own conclusion.** The original divergence notebook
   (`ncot_divergence_pilot.ipynb`) states it plainly: "narrative reasoning, on
   these models, is a commitment device, not an exploration device," on the
   observation that mean decision entropy drops from 0.65 bits (standard CoT)
   to 0.29 bits (narrative CoT).
2. **The commitment-device finding at the social layer.** `study_design.md`
   (Sections 11-13) carried it forward: single-protagonist NCoT is "a
   commitment device for the named protagonist," and in debate, deciders
   changed their minds least (20% vs 38% for primary-affected).
3. **The predicted amplification channel.** `scripts/scaffold_permutations.py`
   (the 4.8-C rationale, in code comments): "narrating from inside the user's
   perspective could AMPLIFY identification with their position. This is the
   one family that could plausibly make sycophancy worse than no scaffold at
   all." The programme predicted a backfire direction before observing one.
4. **The observed backfire.** The fresh BrokenMath forced-verdict run:
   NoT raises false-premise affirmation (0.223 vs 0.157, n=600/arm), with fewer
   abstentions, replicating the judge-scored backfire (54.7% to 69.3%).

Hypothesis worth stating once and testing: **the scaffold's register benefit
and its propositional backfire are two faces of the same mechanism, narrative
commitment.** Committing inside a story resists social pressure (the ELEPHANT
validation drop, robust) but also entrenches whatever proposition the narrative
happens to be built around, including a false premise the model has absorbed
(the BrokenMath backfire). If true, the next scaffold should separate the
perspective-simulation component (keep, strengthen) from the commitment
component (remove, or move after verification). Two already-built, never-run
arms test this directly: `commit_first` (order permutation, Tetlock
post-commitment entrenchment analogue) and `notc` (a premise-cost section).
The Commitment-knockout arm of the ablation is a third probe already partly run.

## 3. Dropped or under-carried justifications

| Justification | Where it lived | Status now |
|---|---|---|
| Perspective simulation = conditional SCM rollout (ToM as causal inference) | position_paper l.174 | Absent from manuscript and embodiment arms |
| Kuleshov mental-state abduction | position_paper l.164 | Dropped |
| Human narrative interventions raise perspective-taking (Shaffer, Bientzle) | position/ACL related work | Cited only; never tested in-model |
| Sycophancy as high-K_C delusion maintenance | position_paper ll.65,182 | Operationalised as 4.6, never run |
| Defeasibility: values revised by others' narrated concerns, not reward | position_paper demo 2 | Deflated by conformity findings, but never tested with genuine embodiment |
| Moral simulacra / archetype fidelity, Phi structural-alignment metric, NC-DAG | framing_paper | Archived entirely |
| VoT format-matching: ethics' native format is narrative, role-situated causality | framing_paper ll.63,100,223 | Dropped from current framing |
| Moral judgment alignment with humans as the endpoint | tier3_preregistration.md | Deferred indefinitely; the PI's restatement makes it central again |

Also worth recording: the original programme's own honesty items anticipated the
split verdict. Tier-1 structure was uncorrelated with Tier-2 conclusion
divergence, and the position paper flagged that increased decisiveness could be
"better calibration or premature foreclosure." The current thesis (register
effect real, propositional backfire) is the answer to that flagged question.

## 4. Dropped experimental ideas, ranked by relevance to the growth axis

Built and never run (zero design cost remaining):

1. **The four `embody_*` arms** (`run_crowdgold_aita.py`, commit f644278).
   Token-matched interest roles (asker, ally, neutral, counterparty) with a
   registered predicted criterion ordering. Killed by the E1 binary-instrument
   null under a pre-registered gate ("no measurable target"). See Section 5:
   the gate's premise has changed.
2. **`protagonist_{user,advisor,observer}`, `notc`, `commit_first`**
   (`scaffold_permutations.py`). Registered, never generated. The protagonist
   family is the direct embodiment-inside-NoT test; commit_first and notc test
   the commitment hypothesis of Section 2. External urgency: 2026 work already
   shows first-to-third-person input reframing mitigates sycophancy, so the
   protagonist-assignment question is being approached from outside.
3. **Crowd-Gold deliberation protocol / 4.8-F** (`run_crowdgold_deliberation.py`).
   17-call ACL-faithful roles (writer_advocate, counterparty, neutral_adjudicator)
   with defeasibility and vote-independence estimators. Pilot approx $5, full
   approx $63.

Designed and never built or run:

4. **4.6 Costly-Correction.** One claim carrying both a checkable falsehood and
   a relational reward. The only instrument that puts the social effect and the
   propositional effect in tension on the same item, which is exactly the seam
   the split verdict lives on. Pilot approx 2.9k generations.
5. **Sealed-ballot (Design B, audience_paraphrase_taxonomy_proposal.md).**
   Separates observability ("others will read me") from peer information
   ("I read others"). The clean mechanism split for communal deliberation.
6. **Tetlock ports (A+C).** Views-known-times-observability, and post-commitment
   accountability producing defensive bolstering. The latter is the social-layer
   version of the commitment hypothesis.
7. **Tier-3 human pairwise preference** (pre-registered, IRB pending). Required
   by the restated goal (alignment with human moral judgment); no structural
   metric substitutes for it.
8. **Std-CoT ablation of the four-round debate arc.** Partially closed by
   Phase 11 with an ambiguous headline; a clean version remains open.
9. **4.R trace-transplant, 4.S narrator-swap, paraphrase envelope I4,
   MP-NCoT PRESERVE/AMPLIFY/ELIMINATE analysis** (that last one is a
   family-of-perspectives measure, i.e. a proto-ToM instrument, run once at
   small scale and abandoned).

Standing verification debts (inherited, not new ideas): grok replication of the
E1 binary null; aggregation-passthrough recheck on the binary instrument;
item-clustered bootstrap CI on the n=600 BrokenMath result.

## 5. The E2 gate should be re-evaluated, not overridden

The embodiment arms were killed by a pre-registered criterion: no measurable
untreated sycophantic shift on the binary Crowd-Gold instrument, therefore
nothing for a directed intervention to reduce. That logic was sound and remains
sound *for that instrument*.

But the premise has changed elsewhere. The BrokenMath forced-verdict instrument
now shows two measurable targets on gpt-5.4-nano: an untreated false-affirmation
rate of 0.157, and a scaffold-induced backfire of +6.6pp. A directed
intervention now has something to move, on an instrument with hard ground truth
rather than judge-scored blame. The natural first embodiment run is therefore
not Crowd-Gold but BrokenMath: port the four interest roles (or a
mathematically sensible analogue: author of the proof, reviewer, neutral
grader, rival) onto the forced-verdict instrument and test whether interest
ordering moves p(TRUE | false premise). This respects the original kill
criterion instead of ignoring it: the gate asked for a measurable target, and
one now exists.

Decision needed from the PI before spend: whether to port embodiment to
BrokenMath as above, run the commitment-hypothesis arms (commit_first, notc,
protagonist family) on the same instrument, or both in one factorial.

## 6. Where the external field is in 2026 (checked 2026-08-31)

- **Persona steering vs sycophancy.** "Playing Devil's Advocate" (arXiv
  2605.21006) shows off-the-shelf doubt/scrutiny persona vectors reach 68-98%
  of targeted CAA steering's sycophancy reduction, preserve accuracy when the
  user is correct, and are geometrically independent of the sycophancy
  direction; sycophancy framed as a persona-level property. This is our thesis
  in activation space and strengthens the prompt-level embodiment programme
  (the persona-level claim implies role manipulations are the right lever),
  but it also means pure "role prompts reduce sycophancy" is no longer novel.
  Our differentiators: interest-role *ordering* (a graded, falsifiable
  prediction, not a binary skeptic-vs-default), ground-truth instruments with
  the ESH/hedge discipline, and the commitment-separation hypothesis.
- **Perspective reframing.** First-person to third-person input reframing is
  reported to mitigate sycophancy via psychological distance. Overlaps with
  protagonist assignment (4.8-C). Run ours soon or cite theirs.
- **ToM benchmarks.** CogToM (ACL 2026, 46 paradigms, 8.5k instances),
  Avalon-ToM-Bench (arXiv 2608.09638, asymmetric-information roles), ToM-SSI
  (EMNLP 2025). Avalon's key finding: models *represent* correct mental states
  internally (linear probes 77-82%) but fail to *express* them in generation
  (62-70%), and reasoning training helps (+11pp) where test-time CoT barely
  does (+1.1pp). A narrative scaffold is an expression-side intervention, so
  this literature gives the pivot both a measurement instrument and a live
  question: does NoT (or an embodiment successor) close the
  representation-expression gap that generic CoT does not? That would be a ToM
  claim measurable with existing public benchmarks, no judge pipeline needed.

## 7. Recommended inheritance, in one list

Carry forward as first-class: perspective simulation as conditional rollout
(theory), the commitment-separation hypothesis (Section 2), the built-but-unrun
arms (embody_*, protagonist_*, notc, commit_first), Costly-Correction (4.6),
sealed-ballot, Tier-3 human preference, and the ToM expression-gap question
(Section 6). Keep the house measurement discipline: item-clustered bootstrap
CIs, Manski bounds on missingness, hedge-category checks, forced-choice
instruments, token-matched arms.

Leave archived unless a specific need arises: NC-DAG formalism, Phi corpus
alignment, gzip/K_C surface proxies (falsified), the 5-way ESH instrument, raw
audience-presence designs (scooped), SycophancyEval (saturated), the
95%-consensus headline (vote carries no signal).

## 8. Steering-amplification roadmap (added 2026-09-01, after E1-E4)

The E1-E4 results (prereg_embodiment_community.md) decompose embodiment into
three components with different amplification profiles: a weak, item-invariant
criterion offset (+2.8pp, sensitivity reliability 0.064 -- the geometry of a
content-independent additive bias, i.e. of a steering vector); a saturable
verdict channel shared with commitment (E2 crowding-out); and a high-gain
stake-conditioned objection response (+63pp concentration, +36pp
defeasibility). Amplification targets the third, never the second.

Substrates, in cost order:

1. **Prompt dose (existing machinery).** Dose-graded stake heads via the
   stance factorial's dose axis, applied antisymmetrically to opposed seats
   in the deliberation protocol (aggregation-passthrough forbids unbalanced
   amplification). Expected: a gain curve with a saturation knee (E2).
   Also: raise R3 objection bandwidth (severity grading, forced modification
   content).
2. **Activation space (local model; contrastive corpora already cached from
   E1/E3).** Extract v_stake by CAA / difference-in-means (persona vectors,
   Chen et al. 2025, arXiv 2507.21509; role vectors rival targeted
   anti-sycophancy steering and are geometrically independent of the
   sycophancy direction, arXiv 2605.21006; internalized debate leaves
   linearly separable agent subspaces, IMAD, ACL 2026, arXiv 2604.24881).
   Four registered-in-spirit predictions: (a) alpha-dose reproduces the E1
   ordering then extends it; dose-response uniform across items per E3
   item-invariance; (b) opposed steering +/- alpha*v with byte-identical
   prompts reproduces stake concentration -- embodiment without words,
   eliminating token-matching confounds entirely; (c) substitution geometry:
   projections of v_interest and v_commitment onto the verdict direction
   overlap and saturate even if the full vectors are near-orthogonal;
   remedy is orthogonalisation or stage separation; (d) v_stake activation
   read online is a per-token stake meter, the live version of the
   derivable signal.
3. **Control loop.** Close the open-loop doctrine: moderator re-synthesises
   until residual objections concentrate only in necessarily-undermined
   seats, using the defeasibility gap (38.3% vs 2.1%) to separate principled
   objection from stubbornness. Amplification = loop gain; defeasibility is
   the stability criterion.

Guardrails from the data: stop at the saturation knee; abort if the
defeasibility gap closes (zealotry overdose); watch P(reject | not
undermined) (currently 13%) as the false-objection rate; amplify the
opposed/critical direction before the advocate direction (both our seat
asymmetry and arXiv 2605.21006's persona asymmetry point that way).
