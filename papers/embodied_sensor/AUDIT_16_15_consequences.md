# Consequences of Addendum 16.15 for the paper (editorial, derived from the 16.15 RESULTS block; not part of the record)

### Consequences for the paper

The sentences below are quoted from
`papers/embodied_sensor/introduction.tex`, `methods.tex` and `results.tex`
as they stood at 2026-09-12 (line numbers from `cat -n`). None of the three
files was edited. Replacement wording follows the papers' style (no em
dashes, no colons or semicolons in running prose, past tense for what was
done). Source comments to attach are given after each replacement.

#### introduction.tex

Lines 83 to 86, currently "The collective is wrong more than four times as
often on the cases it flags, and the same flag concentrates the errors of the
model itself answering alone, stronger than the collective and never shown
the debate, almost as sharply."

Replace with "On verdicts with one loser the collective is wrong nearly ten
times as often on the cases it flags, and the pooled figure of four times
mixes that signal with the verdict type. The same flag concentrates the
errors of the model itself answering alone, stronger than the collective and
never shown the debate, about four times over, and so do that model's own
signals, its samples disagreeing or naming both parties." Source prereg
16.15.1 RESULTS (9.83x, +0.348 [+0.201, +0.493]) and 16.15.2 RESULTS (4.40x
against 8.20x and 6.35x).

Lines 97 to 99, currently "Where the verdict finds everyone at fault, both
advocates lose and it floods, firing on 60 percent of cases instead of 7."

Replace with "Where the verdict finds everyone at fault, both advocates lose
and it floods, firing on 60 percent of cases instead of 7, and there the
verdict type itself is the cheaper error signal." Source prereg 16.15.1
RESULTS (verdict-type router 0.894 against counter 0.880).

Lines 113 to 117, currently "The flag that fires when the second advocate
objects is an error sensor that carries over to models outside the graph,
and routing on it to a strong enough judge certifies a gain over the model
answering alone, which is objectivity obtained as a property of structure."

Replace with "The flag that fires when the second advocate objects is an
error sensor on one-loser verdicts, it carries over to the same model
answering alone and to one of two models from other vendors, and routing on
it to a strong enough judge certifies a gain over the model answering alone,
which is objectivity obtained as a property of structure." Source prereg
16.15.3 RESULTS (three of four intervals exclude zero, cross-vendor one of
two).

Lines 73 to 76 ("Where the roles lock, and on our strongest model they do,
the two advocates cancel and the uninterested seat dominates, so the
collective has one effective voter, meaning its majority equals that seat on
every case, and no vote over the three can beat it.") need no change. They
are now backed by 1,642 of 1,642 and the oracle gap +0.000 [-0.004, +0.000]
(prereg 16.15.4 RESULTS), and the scope clause "where the roles lock" is the
registered scope.

#### methods.tex

Lines 182 to 185, currently "The effective voter count is read from how often
the three-seat majority equals the best single seat, on debates with three
codable opening verdicts, and from the oracle gap, a two-fold cross-validated
oracle over all seat emissions minus the best seat."

Replace with "The effective voter count is read from how often the three-seat
majority equals the neutral seat, on debates with three codable opening
verdicts, and from the oracle gap, a two-fold cross-validated lookup table
over the three opening verdicts minus the neutral seat, with folds drawn over
items." Source prereg 16.15.4 RESULTS, code
`scripts/analyze_effective_voter.py`. If results.tex keeps the record's
-0.021 as well, this sentence must additionally define the full-profile
oracle over every seat verdict, moderator verdict, label and vote, and say
that it carries no interval.

Lines 208 to 209, currently "On the dispute and comparative instruments the
base is the collective and the judge claude-haiku-4-5."

Replace with "On the dispute and comparative instruments the base is the
collective and the judge claude-haiku-4-5. On the dispute instrument a
verdict-type router, which fires when the collective's final verdict finds
everyone or no one at fault, and its union with the counter were composed
with the same haiku judge and with the cached claude-sonnet-4-6 judge of the
second design, and random routing at the union's coverage over 200 seeds was
the reference." Source prereg 16.15.1 RESULTS, code
`scripts/analyze_router_decomposition.py`, artefact
`router_decomposition.json`.

Lines 201 to 204 (transfer definition, "a dispute counting as flagged when
most of its debates flagged it") need no change. The artefact's cell rule is
strictly more than half of the cell's codable debate samples, which is what
"most" says.

Lines 222 to 223 ("Every interval is an item-clustered percentile bootstrap
with 2,000 to 10,000 draws, stated per estimate where reported.") need no
change in prose. Add to the source comment "analyze_router_decomposition.py
and analyze_effective_voter.py (4,000, seed 13), analyze_bm_notes.py
(parent settings, 10,000), analyze_length_matched_elephant.py (8,000)".

Lines 247 to 250, currently "Addenda E1 to E5 cover the false-premise and
disputes experiments, 6 the grip screens, 12 and 13 the comparative
instrument, and 16 the claim audit and ablations, including the pending
two-by-two and narration cycling."

Replace with "Addenda E1 to E5 cover the false-premise and disputes
experiments, 6 the grip screens, 12 and 13 the comparative instrument, and 16
the claim audit and ablations, including the two-by-two, narration cycling
and a zero-spend audit of the sensor claim against the hostile-reviewer
findings." Source prereg L3888 (16.15 registration) and the 16.15 RESULTS
block. (The 16.10 and 16.14b RESULTS blocks now exist at prereg L4136 and
L4091, so the two "pending" todos at lines 95 and 251 are stale for reasons
outside 16.15.)

#### results.tex

Lines 80 to 84 with the todo at line 97, currently "Narration reduced
validation of the asker's own account on every generator. On the same
open-ended accounts \citep{cheng25elephant} and the same judge, the
validation rate fell by 26.7 points on haiku, 22.3 on sonnet, 41.4 on nano
and 36.0 on grok, after a truncation artefact in the judge pipeline was
corrected." and "\todo{manner deltas carry no sampling interval in any
artefact, prereg note pending}".

Replace with "Narration reduced validation of the asker's own account on
every generator. On the same open-ended accounts \citep{cheng25elephant} and
the same judge, the validation rate fell by 26.7 points [16.7, 36.7] on
haiku, 22.3 [13.3, 31.3] on sonnet, 41.4 [31.4, 51.1] on nano and 36.0
[27.1, 44.9] on grok, after a truncation artefact in the judge pipeline was
corrected." and delete the todo. Source prereg 16.15.7 RESULTS, artefact
`length_matched_elephant_oeq_validation_corrected_16_15_7.json` (drops
ci_pp, 8,000 draws, seed 20260822), `bm_notes.json` 16.15.7.

Lines 110 to 115, currently "The untreated arm returned no verdict on 34.5
percent of its 600 units and narration on 27.8 percent, so among committed
units affirmation rose from 0.239 to 0.309, not an emission artefact,
although Manski bounds cross zero here as on the role contrast below, which
the registered rule reports as indeterminate."

Replace with "The untreated arm returned no verdict on 34.5 percent of its
600 units and narration on 27.8 percent, so among committed units
affirmation rose from 0.239 [0.153, 0.329] to 0.309 [0.222, 0.398], a
committed-only contrast of +0.070 [+0.030, +0.115] beside the registered
+0.067 [+0.033, +0.103], while the Manski bounds on the registered contrast,
[-0.278, +0.345], cross zero, which the registered rule reports as
indeterminate." Source prereg 16.15.5 RESULTS, `bm_notes.json` 16.15.5
(committed_only, manski_S0_contrast). The comment at lines 116 to 118 saying
the bounds are "in no artefact" is now false and should point at the same
field.

Lines 202 to 208, currently "The collective therefore had one effective voter
(Figure~\ref{fig-localisation}C). The three-seat majority equalled the
neutral seat's verdict on 1,663 of 1,663 codable debates. A cross-validated
oracle over every seat verdict, moderator verdict, label and vote scored
0.831 against 0.852 for the neutral seat alone, a gap of -0.021 with no
interval in the record."

Replace with the registered readout, "The collective therefore had one
effective voter (Figure~\ref{fig-localisation}C). The three-seat majority
equalled the neutral seat's verdict on 1,642 of 1,642 debates with three
codable opening verdicts, over 210 items. The best fixed function of the
three opening verdicts, fitted on half the items and scored on the other
half, was the neutral seat's own verdict, a cross-validated gap of +0.000
[-0.004, +0.000] against the neutral seat's 0.860 [0.822, 0.896]." Source
prereg 16.15.4 RESULTS, `effective_voter.json` panels.grok
(majority_equals_neutral, cv_oracle_bits). If the PI prefers to keep the
richer oracle, the third sentence instead becomes "A cross-validated oracle
over every seat verdict, moderator verdict, label and vote scored 0.831
against 0.852 for the neutral seat alone, a gap of -0.021 that carries no
interval, and the best fixed function of the three opening verdicts alone
was the neutral seat's own verdict, +0.000 [-0.004, +0.000]." One of the two
must be chosen and 1,663 must go in either case. Panel C of Figure 4 draws
0.852, 0.852, 0.843, 0.831 and 0.912 on the 1,663 population, so
`make_figures.py` must be re-pointed at `effective_voter.json` (0.860,
0.860, 0.860 for the oracle if the registered readout is chosen) and the
caption at lines 249 to 253 must name the oracle it draws.

Lines 209 to 213, currently "The count follows from lock, two of the three
votes being constants of their role, and not from the roles as such. On
haiku at lock 0.442 the writer's advocate at 0.816 outscored the neutral seat
at 0.708, and on the false-premise instrument, where no role could lock, the
same roles under majority vote gave more than one effective voter."

Replace with "The count follows from lock, two of the three votes being
constants of their role, and not from the roles as such, and it is stated
only for models where lock holds. On nano at lock 0.620 the majority
equalled the neutral seat on 777 of 781 debates. On haiku at lock 0.442 the
writer's advocate outscored the neutral seat, 0.818 against 0.717 on 187
debates, +0.102 [+0.028, +0.182], while the majority tracked the neutral
seat at 0.701 and equalled it on 180 of 187, so haiku is the boundary of the
claim. On the false-premise instrument, where no role could lock, the same
roles under majority vote gave more than one effective voter." Source prereg
16.15.4 RESULTS, `effective_voter.json` panels.nano and panels.haiku. The
0.816 and 0.708 in Table~\ref{tab-rolelock} are the per-seat codable
denominators of `rolelock_by_model.json` and may stay in the table, but the
prose must use one denominator, and only the triple denominator (0.818
against 0.717) carries the paired interval.

Lines 271 to 286, currently "A stake-blind counter, two or more seats
objecting, fired on 328 debates, all inside the composite's 337, with
integrated-verdict error rates of 0.415 against 0.095, a ratio of 4.37, and
the composite's advantage as a router was +0.001 [+0.000, +0.003]. The
introduction's sensor is this counter. Most of that concentration is the
verdict type. By the rates of Section~\ref{sec-results-extent}, about 93 of
the counter's 328 firings sit on the 1,288 one-loser verdicts and about 235
on the 390 both-party verdicts, wrong on 0.441 and 0.517 by verdict type
alone, so the sensor claim rests on the counter's lift within one-loser
verdicts. \todo{zero-spend addendum pending, counter lift within one-loser
verdicts and a verdict-type router, as a prereg RESULTS block}".

Replace with "A stake-blind counter, two or more seats objecting, fired on
328 debates, all inside the composite's 337, with integrated-verdict error
rates of 0.415 against 0.095, a lift of +0.320 [+0.239, +0.404] and a ratio
of 4.37, and the composite's advantage as a router was +0.001
[+0.000, +0.003]. The introduction's sensor is this counter. The pooled ratio
mixes two signals. Of the 328 firings, 93 sat on the 1,287 one-loser
verdicts and 235 on the 390 both-party verdicts. Within one-loser verdicts
the collective was wrong on 0.387 of flagged debates against 0.039 of
unflagged, a lift of +0.348 [+0.201, +0.493] and a ratio of 9.83, and within
both-party verdicts on 0.426 against 0.523, -0.097 [-0.233, +0.043], so the
dissent is the signal where the verdict has one loser and the verdict type is
the signal where it does not. As a router to the haiku judge the verdict type
alone composed at 0.894 [0.860, 0.924] against the counter's 0.880
[0.844, 0.913], +0.014 [-0.003, +0.030], and their union at 0.905
[0.870, 0.935], +0.011 [-0.001, +0.024] over the verdict type. With the
sonnet judge the three routers composed at 0.913, 0.936 and 0.956 and the
union's gain over the verdict type, +0.021 [+0.011, +0.033], excluded zero,
so the two signals add under a strong judge and the verdict type is the
cheaper of them." Delete the todo. Source prereg 16.15.1 RESULTS,
`router_decomposition.json` (counter_lift_by_stratum, composed_haiku,
composed_sonnet).

Lines 301 to 315, currently "The flag concentrated the errors of generators
that never saw the debate, on three of four readouts
(Figure~\ref{fig-transfer}). On items the collective flagged, grok answering
alone at accuracy 0.912 was wrong 0.264 of the time against 0.063 elsewhere,
a lift of +0.201 [+0.075, +0.346]. For grok's narrated solo the ratio was 4.8
with lift interval [+0.132, +0.413], for nano 2.3, [+0.060, +0.360], and for
haiku 1.8, [-0.018, +0.256]. Both grok readouts are within-model, the flag
and the solo being the same model on the same item and framing, one of the
two cross-vendor intervals excludes zero, and all four were computed after
the fact from cached calls, pre-declared only for the pending two-by-two."

Replace with "The flag concentrated the errors of generators that never saw
the debate, on three of four readouts (Figure~\ref{fig-transfer}). On items
the collective flagged, grok answering alone at accuracy 0.912 was wrong
0.264 of the time against 0.063 elsewhere, a lift of +0.201 [+0.079, +0.343]
on 420 cells. For grok's narrated solo the lift was +0.269 [+0.134, +0.417],
a ratio of 4.8, for nano +0.200 [+0.053, +0.364], a ratio of 2.3 on 407
codable cells, and for haiku +0.109 [-0.016, +0.256], a ratio of 1.8. Both
grok readouts are within-model, the flag and the solo being the same model on
the same item and framing, and one of the two cross-vendor intervals
excludes zero. The four were first computed after the fact from cached calls
and were then recomputed under a registered estimator with an artefact,
which is the source quoted here. The transfer to grok's solo lived in the
one-loser stratum, +0.130 [+0.026, +0.258] there against +0.017
[-0.087, +0.122] on both-party verdicts. The solo model's own free signals
concentrated its errors at least as sharply, any both-party verdict among
its three samples by 8.2 and disagreement among them by 6.4 against the
collective's counter at 4.4, and composed with the same judge at 0.919 and
0.917 against 0.914, with no pairwise difference excluding zero. The
collective's flag therefore concentrates a solo model's errors, and so do the
solo model's own signals." Source prereg 16.15.2 and 16.15.3 RESULTS,
`transfer_readouts.json` (composite rows, as R5 quoted) and
`router_decomposition.json` (transfer_within_strata, solo_signals). If the
paper prefers the counter, the introduction's sensor, the four lifts are
+0.212 [+0.085, +0.359], +0.282 [+0.143, +0.435], +0.216 [+0.064, +0.383]
and +0.119 [-0.010, +0.271] with ratios 4.4, 5.0, 2.4 and 1.9, and the
conclusion is the same.

Lines 339 to 342, the four solo-generator rows of Table~\ref{tab-sensor},
currently "4.2 (0.264 vs 0.063)" with "+0.201 [+0.075, +0.346]", "4.8" with
"[+0.132, +0.413]", "2.3" with "[+0.060, +0.360]" and "1.8" with
"[-0.018, +0.256]".

Replace with "4.2 (0.264 vs 0.063)" and "+0.201 [+0.079, +0.343]", "4.8
(0.340 vs 0.071)" and "+0.269 [+0.134, +0.417]", "2.3 (0.360 vs 0.160)" and
"+0.200 [+0.053, +0.364]", "1.8 (0.245 vs 0.136)" and "+0.109
[-0.016, +0.256]". Source `transfer_readouts.json` targets.*.by_sensor
.composite. The caption's phrase "the grok collective's flag" should say
"the grok collective's composite flag" or the rows should switch to the
counter values above, and the source comments "html L48" must be replaced by
the artefact.

Lines 360 to 363, the Figure~\ref{fig-transfer} caption, currently "(B)
Item-clustered intervals on the four transfer lifts, with a point estimate
only where one is in the record."

Replace with "(B) Item-clustered intervals and point estimates on the four
transfer lifts." `make_figures.py` Figure 3 must read
`transfer_readouts.json` instead of html L48.

Lines 424 to 431, currently "On grok's one-loser verdicts, 1,288 debates,
exactly one seat objected on 0.922, two or more on 0.072, the
writer-counterparty phi was -0.901 and the collective was wrong on 0.064. On
everyone-at-fault verdicts, 272 debates, two or more objected on 0.614 with
phi -0.191 and error 0.441, and on no-one-at-fault verdicts, 118 debates, on
0.576 with phi -0.275 and error 0.517. The verdict type is the error signal
there, not the dissent."

Keep the first two sentences and replace the third with "Within those
both-party verdicts the counter's lift was -0.097 [-0.233, +0.043] on 390
codable debates, against +0.348 [+0.201, +0.493] within one-loser verdicts,
so the verdict type is the error signal there, not the dissent, and as a
router it is the cheaper of the two." Source prereg 16.15.1 RESULTS. Add to
the comment that 16.12's 1,288 one-loser debates become 1,287 on the 1,677
codable population of 16.15.

Lines 433 to 446, currently "On the false-premise problems a majority-of-three
panel mixing author, neutral and rival affirmed the false premise at 0.162
against 0.188 for three authors, -0.026 [-0.047, -0.007], so mixing opposed
interests into an interested panel lowered its false affirmation, while
against three disinterested judges it was -0.010 [-0.031, +0.008]. The mixed
panel also abstained on 0.286 of items against 0.260, so its lower
affirmation was matched by higher abstention and its rate of correctly
rejecting the premise did not move, which is no gain in accuracy under the
rule that non-commitment is never correct."

Replace with "On the false-premise problems a majority-of-three panel mixing
author, neutral and rival affirmed the false premise at 0.162 [0.103, 0.226]
against 0.188 [0.125, 0.257] for three authors, -0.026 [-0.047, -0.007],
while against three disinterested judges it was -0.010 [-0.031, +0.008]. The
mixed panel rejected the premise at 0.552 against 0.552, -0.001
[-0.025, +0.022], and abstained at 0.286 against 0.260, +0.027
[+0.000, +0.054], so it lowered false affirmation by abstaining more, not by
rejecting more, which is no gain in accuracy under the rule that
non-commitment is never correct." Source prereg 16.15.6 RESULTS,
`bm_notes.json` 16.15.6 (plain.panels, snapshot_plain.deltas). The comment at
lines 444 to 445 saying the 0.552 is derived and the note pending is now
false. The word "de-bias" does not occur in the three .tex files, but it
does in `headline_theory.html` and in the prereg's 16.12 correction block
("measurably de-biases it"), and the html must drop it.

Lines 446 to 449 ("Under narration the same contrast fell to -0.004
[-0.025, +0.016] and the mixed panel itself rose by +0.044 [+0.012, +0.079],
so narration's error passed through the vote.") need no change. The
narration P(FALSE) contrast +0.009 [-0.012, +0.032] may be added from
`bm_notes.json` 16.15.6 narration.

Header, lines 49 to 53 (AUDIT NOTES (e), "Pending zero-spend prereg notes
requested by the audit, none yet written"). All six listed notes except the
phi artefact are now written in the 16.15 RESULTS block with JSON artefacts
(committed-unit affirmation and its intervals, P(FALSE) 0.552 with
intervals, the one-loser lift and verdict-type router, the four ELEPHANT
intervals, the oracle-gap interval). The phi artefact was not part of 16.15
and stays pending. Update the comment.

Lines 371 to 377 (the routing paragraph, composed system 0.881, +0.039
[+0.018, +0.060]) need no change. The registered composite figure agrees
with the counter's 0.880 [0.844, 0.913] and +0.038 [+0.016, +0.060] in
`router_decomposition.json`, which may be added to the source comment.
