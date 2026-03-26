# Topic Decision

## Objective

Lock the project to one defensible main question before any more model or paper work.

## Candidate A

`Residual momentum under regimes`

Question:
Does market state change FF3 residual momentum predictability?

Pros:

- strong finance framing
- existing notes and test scripts already cover much of the setup

Cons:

- overlaps heavily with momentum-crash literature
- `cycle_position` already looks weak in preliminary notes
- the user's real interest is not Fama-MacBeth; it is PE interpretation
- would force the architecture to become a side detail

Verdict:

- keep as application / appendix candidate
- do not use as the main paper center

## Candidate B

`Index-conditioned positional encoding`

Question:
Can broad-market index flow be used to condition the temporal coordinate system itself,
so that attention and local motif extraction become state-aware in an interpretable and
testable way?

Pros:

- matches the user's actual research interest
- gives PE a first-class role instead of treating it as a minor feature trick
- naturally explains why Transformer, TCN/CNN, and probabilistic output can coexist
- can reuse the current `cycle_pe`, hybrid fusion, and CVAE code with a cleaner story

Cons:

- weaker fit for top finance journals than a pure economics paper
- requires strong ablations against concat conditioning and non-attention baselines
- needs careful language: diagnostic evidence, not "attention = explanation"

Verdict:

- choose as the main topic

## Candidate C

`State-dependent uncertainty forecasting`

Question:
Can a probabilistic decoder recover regime-dependent predictive distributions better than
point forecasts?

Pros:

- clean evaluation via calibration and CRPS
- easy to package as a forecasting paper

Cons:

- not the user's true center of gravity
- decoder choice is secondary unless the representation claim already works
- CVAE alone is unlikely to carry the paper's novelty

Verdict:

- keep as a supporting layer
- do not make it the main claim

## Final Lock

The project is now locked to:

`Index-Conditioned Positional Encoding for State-Aware Pattern Routing in Financial Time Series`

Primary anchor:

- `S&P 500` as the main market-state index

Secondary robustness anchor:

- `Nasdaq` as a robustness / appendix condition signal

## One-Sentence Main Question

Does conditioning positional encoding on broad-market index state create a better temporal
coordinate system for routing global context and local price motifs than static PE or
simple input concatenation?

Current evidence status:

- stronger than `static`: yes
- stronger than `concat_a`: not yet
- current paper-facing claim is therefore downgraded to `explicit stock-index conditioning matters`

## What This Means

Main paper center:

- PE is the object of interest
- Transformer is used because PE directly modifies attention geometry
- TCN/CNN is used to test whether global routing helps local motif extraction
- the probabilistic decoder is an output head, not the paper's novelty core

What gets demoted:

- FF3 residual momentum
- 2-step Fama-MacBeth
- pure momentum-crash framing

These may still appear as:

- an application domain
- a robustness section
- a later separate finance paper

## Naming Decision

Use `IC-PE` as the project term:

- `IC-PE` = `Index-Conditioned Positional Encoding`

Do not keep `Cycle-PE` as the main paper name. It is too tied to the earlier regime paper
and is semantically broader than the new thesis.

## Non-Negotiable Rules

- one main question only
- no claim that attention weights are explanations
- no claim that CVAE is novel by itself
- no claim that S&P flow "is" PE; it conditions the coordinate system used by PE
