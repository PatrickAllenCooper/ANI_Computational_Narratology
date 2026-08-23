# Pre-registration: does majority aggregation amplify a sycophantic criterion shift?

**Written before the data were collected.** Branch `sycophancy-scenario-redesign`, 2026-08-22.
Committed prior to launching the run; the commit that adds this file contains no results.

## Why this is pre-registered

We already have a suggestive result in this direction — on the published ESH coding, majority-of-3
enlarged the stance-induced criterion shift on two of three same-model panels (haiku +0.117
[+0.027, +0.254], nano +0.096 [+0.009, +0.193]). It **failed** the mandatory coding-sensitivity
check: 0 of 6 panels were sign-stable and significant in ≥2 of the three ESH codings. We are now
deliberately going looking for this specific effect, which is exactly the situation in which
analyst degrees of freedom produce findings. Hence: predictions and kill criteria fixed in advance.

## The theoretical prediction, derived not assumed

Equal-variance signal detection. A voter says "at fault" with probability Φ(d′/2 − c) on
gold-at-fault items and Φ(−d′/2 − c) on gold-not-at-fault items. Sycophancy is modelled as a shift
of c by δ, applied identically to every voter — a **bias**, not added noise.

Majority-of-k of such voters has a steeper response function than a single voter, so the *same*
criterion shift produces a *larger* movement in the majority's implied criterion. Simulated at
d′ = 1.8, c₀ = 0.10, δ = 0.145 (the observed nano individual shift):

| panel k | majority Δc | pass-through | amplification |
|---:|---:|---:|---:|
| 1 | +0.1450 | — | 1.000 |
| 3 | +0.2155 | **+0.0705** | 1.486 |
| 5 | +0.2686 | **+0.1236** | 1.852 |
| 7 | +0.3129 | **+0.1679** | 2.158 |
| 9 | +0.3517 | **+0.2067** | 2.426 |

Voter correlation destroys it, because correlated voters behave more like a single voter and the
response function stops steepening. Simulated at k=3 with a shared latent factor:

| error correlation ρ | pass-through |
|---:|---:|
| 0.00 | +0.076 |
| 0.35 | +0.033 |
| 0.65 | +0.015 |
| 0.90 | +0.003 |

The amplification is therefore **not** a claim that groups are worse than individuals in general.
It is a claim about a specific mechanism with a specific functional form, and the functional form
is what we are testing.

## Primary hypothesis

**H-AMP.** Pass-through — Δc(majority-of-k) − Δc(individual) — increases monotonically in k.

Pre-registered as a **trend test**, not a set of pairwise contrasts: Spearman ρ between k and the
observed pass-through across k ∈ {3, 5, 7, 9}, with an item-clustered bootstrap CI. The prediction
is a monotone increasing curve, so a trend test is the correct instrument and pairwise contrasts
are secondary.

## Secondary hypothesis

**H-DECORR.** Pass-through at matched k is *smaller* on panels with higher error correlation.
Cross-vendor panels (measured ρ̄ = 0.336) should show *larger* pass-through than same-model panels
(measured ρ̄ = 0.652), because correlation attenuates the amplification.

**Note that this predicts the OPPOSITE of what we currently observe.** Same-model k=3 pass-through
came in at +0.096 to +0.117 against a ρ=0.65 prediction of +0.015 — six times too large — while
cross-vendor came in at +0.046 against a ρ=0.34 prediction of +0.033, a good match. If the existing
same-model result were real, H-DECORR is false. We are pre-registering the theory's prediction
rather than the one our pilot favours, and recording here that we expect the same-model excess to
regress toward the prediction.

## Design

- **Instrument.** Crowd-Gold AITA, 249 Scruples items, crowd gold on both classes, byte-identical
  post text under the stance reframing (`third_person` → `as_asker`).
- **Manipulation.** None new. Panels are constructed post hoc from independent samples of the
  untreated `standard` arm — no communication, majority vote. This is "N independent agents".
- **What is bought.** Samples 3–8 on the two arms, taking k from 3 to 9. 249 items × 2 arms ×
  6 samples × 3 models = 8,964 calls. Estimated **$1.54 nano + $0.87 grok + $6.57 haiku ≈ $9.00**.
- **Panels.** k ∈ {1, 3, 5, 7, 9} drawn from the 9 samples, same-model; and cross-vendor panels at
  k = 3, 6, 9 (3 samples from each of 3 models).
- **Estimator.** `scripts/analyze_aggregation_passthrough.py`, item-clustered paired bootstrap,
  B = 4000, seed 44. Majority verdicts collapse non-committal responses as abstentions; ties resolve
  to not-at-fault.
- **Codings.** All three ESH codings, mandatory. No result is reported under one coding.

## Kill criteria, fixed now

1. **H-AMP is rejected** if the k-trend Spearman CI includes zero, or if the point estimate is
   negative, under the preferred (published) coding. A flat curve kills it.
2. **The whole result is not reportable** if the trend fails to hold in sign under all three ESH
   codings. Same rule that killed the k=3 pilot result; it applies to its replacement.
3. **The result is uninterpretable, not supportive**, if observed pass-through at k=9 exceeds the
   iid prediction (+0.207). The iid case is the theoretical *ceiling* for this mechanism; exceeding
   it means something other than the modelled mechanism is operating, and we report that rather
   than claiming a stronger effect.
4. **If observed values sit systematically above the ρ-adjusted prediction** (as the current
   same-model k=3 values do), we report the discrepancy as unexplained rather than treating the
   direction alone as confirmation.

## What a positive result would and would not license

**Would license:** "Majority aggregation does not merely fail to remove a sycophantic bias; because
aggregation sharpens the group's response function, the same bias produces a larger distortion in
the group's decisions than in any member's — increasing with panel size, and attenuated by voter
diversity."

**Would not license:** any claim that deliberation amplifies bias. This tests *voting* over
independent samples. The deliberative protocol is a different object and is measured separately.

**Would not license:** any claim about accuracy. Aggregation demonstrably *improves* accuracy on
these panels (cross-vendor +0.042). The claim is specifically that it improves accuracy while
*worsening* the bias — two separable quantities, which is the point.

## Analysis order

1. Verify sample counts and truncation/parse guard before any contrast is read.
2. Compute the k-trend under the published coding. Report the Spearman CI.
3. Repeat under both other codings. Apply kill criterion 2.
4. Compare observed against predicted at each k. Apply kill criteria 3 and 4.
5. Only then, the cross-vendor vs same-model comparison for H-DECORR.
