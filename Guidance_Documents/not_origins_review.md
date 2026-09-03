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

### 8.1 Control-loop investigation: Phase 0 done, ladder registered in spirit

Phase 0 (2026-09-01, zero spend, `scripts/analyze_loop_step.py` on the E4
caches, 875 debates / 220 items, item-clustered CIs). Every deliberation
debate already contains one closed loop iteration (synthesis -> R3 seat
objections -> integration -> final verdict) with crowd gold, so the
one-step dynamics were measured retrospectively:

- The loop CORRECTS conditionally: fix rate P(final ok | synthesis wrong)
  = 12.0% [7.1, 17.6] vs break rate 2.8% [1.7, 4.1]; rate gap +9.2pp
  [+4.0, +15.4].
- But an UNGATED loop turn is net-negative in counts (-0.7pp [-2.0, +0.7])
  because right syntheses outnumber wrong ones six to one. The loop needs
  a sensor -- the textbook control-theory situation.
- SENSORS EXIST, and they are the stake structure itself: a MIS-LOCALISED
  objection (a stake-bearing seat objecting although the synthesis did not
  undermine its stake) flags a wrong synthesis at precision 42.9% against
  a 14.3% base rate, lift +32.8pp [+18.9, +45.8]; the stakeless neutral
  seat's objection carries lift +20.7pp [+7.2, +35.9]; the composite flag
  reaches recall 54.4% at precision 39.5%. Objections that interest cannot
  explain are information about content -- the theoretical claim, now
  measured.
- The integration step already responds to the signal: fix rate among
  flagged wrong syntheses is +18.8pp [+9.8, +28.6] above unflagged.
- The gated counterfactual (integrate only flagged debates) improves on
  ungated but is still net ~zero (-0.1pp [-1.4, +1.1]): with precision
  ~0.4, flagged-right debates still absorb break risk. The two levers this
  isolates: sensor precision (severity filtering, REJECT-only vs AWM,
  later the activation stake-meter) and per-fix efficacy (tell the
  moderator WHY the debate was flagged; iterate).

Ladder from here: L1 (zero spend) rerun on the E5 tripled data + severity-
filtered sensors; L2 (small spend) a second label->integrate cycle applied
ONLY to flagged debates, moderator told the flag reason -- measures
marginal fix/break of a true gated iteration (~4 calls x ~20% of debates);
L3 stake-dose x loop-gain interaction and per-iteration disturbance
rejection (resistance across loop turns); L4 the activation stake-meter as
sensor on a local model.

### 8.2 L1 and L2 done (2026-09-01/02): the loop has no restoring force

Full numbers in prereg_embodiment_community.md (E5 RESULTS, L1 RESULTS,
L2 RESULTS). What they add to the roadmap:

- **L1 (k=4, 1,677 debates).** Phase 0 replicates with every interval
  narrower and no sign change. The loop's use of the sensor is now firm:
  flagged wrong syntheses are fixed +19.6pp [+13.6, +26.6] more often
  than unflagged. Severity filtering (REJECT-only) is a modest lever
  (+0.026 precision, -0.096 recall), not a different sensor.

- **L2 (337 flagged debates, three moderator sub-arms, ~5k calls).** The
  registered primary is null: an informed second iteration nets +0.012
  [-0.039, +0.063] in the flagged population; blind +0.009, generic
  +0.033, all intervals covering zero. Flag CONTENT does nothing
  (informed - blind +0.006 [-0.028, +0.041]). The kill criterion for
  "iteration does not pay at this sensor precision" fires; the
  "iteration is harmful" criterion does not (net >= 0 in all sub-arms).

- **The mechanism did not degrade -- the verdict just re-rolled.**
  Iteration-2 defeasibility is LARGER than iteration 1 (accept when
  addressed 0.629 vs 0.010; iteration 1: 0.366 vs 0.009). Yet within
  flagged debates both iterations look the same: iteration 1 fix 0.196 /
  break 0.151, iteration 2 fix 0.32 / break 0.21, fix-to-break ratio
  1.3-1.5 both times, net ~0 both times. Stability across passes does
  not predict correctness (S3 correct 0.60 whether or not it changed from
  S2), and a majority over S1/S2/S3 does not beat S3 alone (-0.018
  [-0.066, +0.034]). On contested items the passes are close to
  exchangeable draws at ~59% accuracy with a slight drift toward truth.

The control-theoretic reading is exact: the plant has a sensor (the
stake structure detects contested debates at lift +0.33) but no
actuator with a restoring force -- re-running the same three agents on
the same post is a re-sample of the same posterior, not new evidence.
Iterating the same protocol cannot converge. What this rules IN for L3/L4:
the next lever must add INFORMATION, not passes -- either a sharper
sensor (the activation stake-meter, precision well above 0.41 so the
break exposure of correct-but-flagged debates falls) or an escalation
path (flag-and-route to a different model / retrieval / human) rather
than self-repair. The dose-graded L3 keeps its rationale (does stake
dose move the sensor's precision?) but its "loop gain" half is moot
until the actuator changes.

### 8.3 The actuator ladder (2026-09-02): the sensor plus a decorrelated model is the loop

Registered as prereg Addendum 4 (spend-ordered, stop at first positive)
and stopped at the third zero-spend rung. Full table in the prereg.

- Within-community direction is absent. The against-interest objector's
  own verdict agrees with the synthesis 193/231 times (+0.000); the flip
  rule is harmful (-0.143); deferring to the neutral seat is +0.057 and
  fails replication. Mis-localised objections are about reasoning, not
  direction. The flag knows WHERE, not WHICH WAY.
- Direction from outside works, and only where the sensor says. Routing
  flagged debates to a second vendor's plain answer (haiku standard,
  majority of 3 cached samples) lifts them 0.582 -> 0.774, +0.193
  [+0.067, +0.310] at 98.75%, replicating in both item halves and both
  arms; the same rule on UNflagged debates is worse than the deliberation
  (-0.040). Sensor-specific gain +0.232.
- The composed system beats both components: 0.881 vs 0.843
  (deliberation alone, +0.039 [+0.018, +0.060]) and 0.850 (haiku alone,
  +0.032 [-0.004, +0.068]).

This is the thesis the PI stated at the outset, measured: agents whose
alignments are combined push toward objectivity in a way that yields a
SIGNAL, and the signal's use is to route. The embodied community is the
sensor; the decorrelated model is the actuator; the stake structure is the
wiring between them. The correction is asymmetric (mostly leniency
errors, gold-YTA 0.61 -> 0.93) and one model pair on one task family;
reverse routing and a second task are the next registrations.

### 8.4 Reinforcing the routed signal (prereg Addendum 5, 2026-09-02)

Three cheap levers tried on the grok panel, all resolved; S3
(generalisation to a nano community) resolved with the kill criterion
firing -- see the end of this section.

- S1 (sensor x actuator selection, half-0 select / half-1 confirm): the
  registered configuration stands. No candidate beats composite + haiku
  maj-3 on held-out items (best challenger -0.010 [-0.032, +0.011]).
  The instructive part is the sensor comparison at fixed actuator:
  routing the 18% flagged beats routing 85-98% of debates by ~0.04,
  because the unflagged majority is where the community is right and the
  second vendor is merely comparable. The flag is doing the work, not
  the vendor swap.
- S2 (crux escalation): showing the second vendor the community's three
  objection texts does not help (-0.036 [-0.095, +0.026] vs cold) and
  the loss sits on gold-NTA (-0.094), where the community's own
  over-blaming error lives. Content leaving the community re-correlates
  the outside judge with the community's criterion. Consistent with A1
  and the L2 informed sub-arm.
- S2b: haiku maj-5 vs maj-3 is +0.003; A3a at maj-5 is +0.196
  [+0.094, +0.291]. Already saturated at k=3.

Net statement of the mechanism, sharpened by these nulls: the signal the
opposed-alignment community yields is *addressing information* -- which
items, by whose stake -- and its correct consumer is a decorrelated judge
that is shown nothing else. The narrative/embodiment apparatus
manufactures the disagreement structure; the structure is a pointer, not
an argument.

- S3 (same protocol, gpt-5.4-nano as every seat and moderator, 828
  debates): the pointer disappears. The composite flag fires on 85% of
  debates (grok: 20%); seats object almost always, almost always as
  accept-with-modification, and regardless of whether their stake was
  undermined; stake concentration falls from +0.657 to +0.033; the flag's
  lift on synthesis error is +0.056 [-0.017, +0.119]. Routing
  nano-flagged debates to haiku is +0.035 [-0.025, +0.096] (null); to
  grok +0.091, but routing everything to grok is +0.082 and the composed
  system is no better than grok alone. Kill criterion: the stake
  structure is model-specific; the architecture claim is confined to grok
  until a second model shows grip.

What S3 adds to the embodiment theme, which is the PI's growth category:
embodiment is a control apparatus only where the model binds the
assigned stake to its behaviour. nano adopts the role's voice but not its
selectivity; it produces the form of stakeholder dissent without the
information. "Stake grip" -- fire rate, reject rate, stake concentration
on a small panel -- is therefore a measurable per-model precondition, and
the first per-model metric this project has for whether embodiment took.
Next registrations: a stake-grip screen across candidate community
models; haiku as the community to decide whether grok is the exception.

### 8.5 The stake-grip screen (prereg Addendum 6, 2026-09-02, $70.47 final)

Grip defined and measured (`scripts/analyze_stake_grip.py`): composite
fire rate <= 0.5, R4 reject share >= 0.05, stake-seat reject
concentration >= +0.20 with CI excluding zero. Four community models,
byte-identical protocol.

CORRECTION: the haiku and sonnet screens initially FAILED their own
round-level truncation guard (haiku R4 vote 81-97% truncated against a
512-token cap sized to grok's 91-token average; sonnet 0-19%), printed
plainly in each console log but missed at read time because the live
verification grepped `ERROR|guard` (lowercase), which matched the
separate passing OUTCOME guard but not the failing `GUARD FAILED`
line. Both were re-run with `--max-tokens-label/--vote 3072` and the
downstream integration cache purged (it is keyed on an unchanged
token cap and would otherwise silently replay output computed from the
truncated R3 text). Both corrected runs pass the guard cleanly.

Final, corrected numbers:

| model | fire | reject share | reject conc | grip |
|---|---|---|---|---|
| grok-4-1-fast-reasoning | 0.20 | 0.33 | +0.631 [+0.587, +0.673] | yes |
| claude-haiku-4-5 (n=192, corrected) | 0.46 | 0.07 | +0.062 [+0.016, +0.108] | no (real, weak) |
| gpt-5.4-nano | 0.85 | 0.02 | +0.034 [+0.018, +0.051] | no (real, weak, unusable) |
| claude-sonnet-4-6 (n=31, corrected) | 0.07 | 0.00 | 0 (no rejects) | no (none detected) |

Grok clears the registered threshold; the other three do not, but not
uniformly for the reason first reported. Nano's dissent is real but
untethered from stake (concentration barely above zero despite firing
on 85% of debates). Sonnet shows no measurable stake sensitivity at
n=16 items. Haiku is the interesting middle case, corrected from an
apparent "compliant, no signal" artifact to a REAL, CI-confirmed, but
small stake effect (+0.062, an order of magnitude below grok's +0.63),
and its composite flag itself replicated as a genuine (if weak) lift
signal on 60 fresh items (+0.148 [+0.024, +0.268]) -- though routing on
that flag did not clear a CI at this sample size (+0.060
[-0.059, +0.174]). Grip looks graded rather than binary: grok >> haiku
~ nano (real but too rare to use) > sonnet (none). For the embodiment
theme, the corrected result is more encouraging than the retracted one:
the mechanism is not confined to a single idiosyncratic model, it is
just far stronger in grok than anywhere else measured so far, and haiku
is the natural second candidate for a larger-item follow-up.

Two smaller findings. (a) nano's rare REJECT votes (2.4% of votes) ARE
informative -- routing on them is +0.226 to grok -- so nano has the
structure but almost never expresses it. (b) haiku's compliant community
produces a low-base-rate flag that pointed the right way on discovery
(precision 0.47) and on fresh items (lift +0.15, routing +0.17) but
both CIs bottom exactly at zero on 60 items; unconfirmed, and in any
case not a stake mechanism.

Where the mechanism claim now stands: opposed-alignment embodiment is a
sensor when, and only when, the model binds stake to behaviour; the
binding is model-specific and measurable; when present, the signal is
addressing information best consumed by an uninformed decorrelated
judge. The open scientific question is what makes grok bind and the
others not (reasoning-mode dissent? RLHF compliance priors?), and
whether prompt-level changes to the embodiment can induce binding in a
model that lacks it -- which would be the first true test of embodiment
as a control knob rather than a fixed property.
