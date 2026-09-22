# The pillared theory: objectivity from represented opposition

Written 2026-09-21 as the design document for the unified Narration-of-Thought paper
(`papers/unified/`). It fixes one theory, states what each of the paper's three outcomes
is a case of, and names the experiment that tests each prediction. Every citation key below
is in `papers/embodied_sensor/references.bib` and passed the primary-page check recorded in
`papers/embodied_sensor/related_works_verified.json`; keys marked TO VERIFY are not yet
fetched and are not cited in the paper until they are.

## 1. One statement

A model's answer is pulled toward the interest that is present in its context. On an
open-ended dilemma the interest is the asker's own account of the conflict; on a false
premise it is the premise the user owns; on an assigned side it is the side. The pull
cannot be removed from inside the same context, because the copy that would remove it is
the copy that carries it (16.15.4: three copies of one model vote as one voter; the
correlated-jury bound of [ladha1992condorcet], [boland1989majority],
[dietrich2013independent], [kim2025correlated]). What can be done is to place an opposed
interest into the process so that agreement across interests, or dissent against
interest, becomes informative.

This is the argumentative theory of reasoning [mercier2011argumentative,
mercier2017enigma] joined to the social account of objectivity [longino1990science,
longino2002fate, popper2011open]: individuals argue for a side, and truth is a property of
the structure of criticism among them rather than of any one mind. The paper's scaffold and
its collective are the two ways of placing the opposed interest.

- Internal opposition. The scaffold makes one trace represent the other parties: it must
  name every party with a stake, trace what each available action does to each of them,
  and state what remains unknown, before it commits. The represented counterparty is the
  content of the intervention.
- External opposition. The collective assigns the interests to separate copies (seats),
  reads their dissent by the credible-signal rule (a statement against the speaker's
  assigned interest is informative, a statement with it is not; [milgrom1986relying,
  lipman1995robust, glazer2001debates]), and lets a seat with no side decide.

The two are not the same claim, and the paper does not let one borrow the other's
evidence. The game-theoretic reading applies only where its primitives exist, an assigned
interest, a receiver whose verdict is separate from the message, and a decision read
independently of the speaker; those exist in the collective and not in a single trace
(`Guidance_Documents/theory_grounding_and_judge_rehabilitation.md` §1, §5). For the
single-agent scaffold the paper claims a change in the output distribution whose content is
the represented counterparty, and makes no signalling claim.

## 2. What each outcome is a case of

### Pillar 1, sycophancy in social contexts (internal opposition)

Sycophancy is the answer pulled toward the asker [sharma23, cheng25elephant,
chen2024yesmen]. The prediction is that the reduction under the scaffold is a counterparty
effect and not a manner effect.

- P1a. RESULT (17.3, run 2026-09-21): NEITHER, at the pooled level. The four-way section
  knockout on ELEPHANT OEQ validation found no single section's removal clears a 0.25 share
  of the intact reduction with a CI excluding zero, pooled across haiku, grok and nano.
  The result is model-heterogeneous: on haiku every one of the three candidate sections
  (stakeholders, consequences, uncertainty) removes 0.42 to 0.55 of the reduction, mostly
  with a CI excluding zero, so on haiku the whole scaffold is load-bearing rather than any
  one section, which does not separate the counterparty account from the competing
  hedging-register account (theory-grounding §1.1); on grok and nano every knockout's
  share sits near zero with a wide CI including zero, so the reduction there survives
  losing any single section almost intact. Neither the counterparty prediction nor the
  hedging-register prediction is confirmed at the section level. The item-level prediction
  (P1b) is unaffected by this and still holds.
- P1b. The reduction is concentrated on accounts that contain a party other than the asker
  and absent where none exists. Experiment: Z2, the counterparty moderator on the 150 OEQ
  items (zero spend). The propositional case is the no-counterparty case by construction
  (a false theorem has no second party), so the observed backfire there (BrokenMath +6.7
  points [3.3, 10.5]; 16.15.5 S0 +0.067 [+0.033, +0.103]) is the scope boundary the theory
  predicts, and 16.15.6's reading (the scaffold lowers false affirmation by abstaining
  more, not by rejecting more) fits it.
- P1c. The reduction is a property of the prompt's structure and so reproduces across
  vendors and on open-weight models. Experiment: A1 on Llama-3.3-70B-Instruct,
  Mistral-Large-3-2 and DeepSeek-V4-Pro; A1b the three-vendor judge panel.

What pillar 1 does not claim: a mechanism at the level of compliance (retracted, HANDOFF
2026-08-26 §R1), a length-independent effect beyond the length-matched estimates on nano and
grok, or any effect on the `aita_yta` validation metric (withdrawn).

### Pillar 2, represented minds (the same act, measured as theory of mind)

Naming what another party knows and wants is an expression-side theory-of-mind act. The
record already frames the live question as whether a narrative scaffold closes the gap
between what a model represents about other minds and what it expresses
(`Guidance_Documents/not_origins_review.md` L193-209). The prediction is sign-agnostic: the
scaffold should move questions that turn on another agent's knowledge state, if it moves
anything, and the false-belief stratum is the primary stratum. Experiment: B1, a public,
mechanically scored theory-of-mind benchmark, NoT against standard CoT, seven models. The
abstract's sentence is written from the pooled interval. Benchmark sources: TO VERIFY at
their primary pages before registration (candidate ToMBench, alternate FANToM).

### Pillar 3, objective collective judgement (external opposition)

When the interests are separate seats, three things follow and each has been measured.

- The dissent edge exists only where the interests are opposed: fire 0.196 with embodied
  roles against 0.005 with identical readers (16.10); it survives removing the exchange
  between seats (edges off, 0.129) and removing narration (plain-CoT seats, 0.131), so it is
  the assignment and not the conversation that carries it
  ([dewatripont1999advocates, krishna2001model]; 16.10, 16.16).
- Dissent against interest is the signal. Within one-loser verdicts a second objector marks
  a verdict that is wrong 9.83x as often (0.387 against 0.039, +0.348 [+0.201, +0.493],
  16.15.1). The counter floods where a verdict blames everyone or no one (P(2+) 0.603,
  16.12) and is silent where no seat departs from its side (binding plans, Dilemmas 6 of
  624, 16.13 B1). These are the three boundary conditions the credible-signal rule owns.
- The collective is a sensor and not a decider. No cell beats its best seat by vote; the
  majority of three grok seats is the neutral seat on 1,642 of 1,642 debates (16.15.4); the
  moderator's competence sets the verdict ([shin1998adversarial]; 16.17, sonnet +0.038,
  haiku -0.115). The gain is realised by routing: flagged cases to a stronger judge with no
  side raise accuracy over the collective by +0.070 [+0.048, +0.094] on grok (the paper's
  7.0 points) and over the strongest single model by +0.024 [+0.005, +0.045] (16.22).
  Independently recomputed from the CSVs on 2026-09-21 (`verify_pillar3_headline.py`).

What pillar 3 does not claim, after 16.23: the against-interest mechanism on a model that
does not lock to its role. On haiku (lock 0.42) the all-strata sensor and the routing gain
over the collective replicate (+0.149 [+0.053, +0.242]; +0.037 [+0.015, +0.059]) but the
within-one-loser lift includes zero and the advocates' objections are positively
correlated (phi +0.246 [+0.130, +0.360]); the seats behave as correlated readers, and the
counter is the ordinary disagreement signal of three copies, which the solo's own sample
disagreement gives at three calls against seventeen. The credible-signal rule needs the lock
it presupposes. This is the precise sense in which pillar 3 depends on pillar 1's control:
embodiment has to hold for opposition to be readable.

## 3. Where the three pillars meet

The paper's claim is one sentence long. Objectivity is obtained by placing an opposed
interest into the process, either represented inside one trace or assigned to a separate
seat, and it is a property of that arrangement and not of any copy. The three outcomes are
the three readouts of that arrangement: the asker's pull on a single trace (pillar 1), the
representation of other minds that the arrangement requires (pillar 2), and the readable
dissent that separate seats produce (pillar 3). Two results tie the pillars to each other
rather than merely side by side. First, 16.16: narration does not create the collective's
structure, so internal and external opposition are different objects with different
evidence. Second, 16.23: without role lock the external structure collapses to correlated
readers, so external opposition rests on the same control (assignment holds what a copy
will say) that pillar 1 exploits. Z4 adds the third tie if it holds: whether the external
counterparty removes the asker's pull that the internal one only reduces.

## 4. Theory-to-experiment table

| prediction | source | experiment | status (2026-09-21) |
|---|---|---|---|
| P1a counterparty sections/content carry the sycophancy reduction causally | argumentative theory; against the hedging-register account | 17.3 knockout; 17.5 isolated manipulation | NOT CONFIRMED, and 17.5 reverses: removing one section (17.3) retakes at most a small, model-specific share; adding counterparty-awareness alone with no other structure (17.5) RAISES validation, +6.1 [+2.1,+10.2] pooled, not pre-declared. The causal, section/instruction-level claim fails on both tests; only the item-level association (P1b) survives |
| P1b reduction concentrated where a counterparty exists | same; BrokenMath as the no-counterparty case | Z2 moderator; 16.15.5/16.15.6 | done: pooled diff -26.3 [-35.7,-17.0], HOLDS on 3 of 4 original models |
| P1c reduction reproduces across vendors and open weights | property of the prompt | 17.1, 17.2 | done: REPLICATES on all 3 new deployments (22-70 points, 31-87%, larger on open-weight models); judge-robust on 6 of 7 models under a second judge, NOT judge-robust in the strong sense (a third judge reverses sign on 3/7, traced to that judge's acquiescence bias) |
| P2 scaffold moves knowledge-state questions (sign-agnostic) | expression-side ToM | B1 | to register (18) |
| P3a dissent edge needs opposed interests, not exchange or narration | [dewatripont1999advocates, krishna2001model] | 16.10, 16.16, 16.25 noedge k=4 | done, quantified: within-one-loser lift 9.83x on-edge, 5.06x with edges cut (still CI-excluding-zero at 78 fired); phi -0.901 on-edge, -0.842 off-edge |
| P3b dissent against interest is informative within one-loser verdicts | [milgrom1986relying, lipman1995robust, glazer2001debates] | 16.15.1, 16.12, 16.13 B1 | done on grok (phi -0.901, opposed); NOT established on haiku (phi +0.246) or Llama (phi +0.148) despite Llama's high role-lock (0.862); the lift survives on both, the against-interest READING does not |
| P3c same-model seats aggregate to one voter | [ladha1992condorcet, boland1989majority, dietrich2013independent, kim2025correlated] | 16.15.4, aggregation_rules | done |
| P3d the gain is realised by routing, over the collective and the solo | [shin1998adversarial] for the decider; 16.22 | 16.15.1, 16.22, verify_pillar3_headline.py | over the collective: done, independently reproduced on 4/4 conditions (grok on/off-edge, haiku-249, Llama), never failed; over the strongest single model: grok only (16.22 R1), does not replicate on Llama (CI includes 0) |
| P3e the mechanism needs role lock | credible-signal rule presupposes an interest | 16.23 (haiku), 16.24 (Llama) | REVISED: role-lock is not sufficient for the against-interest (anti-correlated) reading -- Llama has grok-like lock (0.862) but haiku-like phi sign (positive); something other than lock alone decides the sign, still unidentified. Role-lock's absence (haiku) does coincide with a weak within-one-loser lift (+0.089, CI includes 0 at low power; the all-strata lift is the one that clearly holds there) |
| P3f power on the invariance cells | MDEs | 16.25 (noedge k=4 done; stdcot k=4, nperspective k=2 pending) | noedge k=4 done at full power (78 fired, gate 60 met); stdcot and nperspective still at k=1, under-gated |
| tie Z4 external counterparty removes the asker's pull | pillars 1 and 3 | Z4 on the three communities | to run (zero spend) |
