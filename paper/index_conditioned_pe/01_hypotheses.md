# Hypotheses

## Final Research Question

Can broad-market index state be used to condition temporal coordinates in a way that
improves forecasting and yields testable diagnostic changes in global-context routing and
local motif extraction?

## Core Architectural Interpretation

- `IC-PE`: state-conditioned temporal coordinate system
- `Transformer encoder`: global context router
- `TCN/CNN branch`: local motif extractor
- `Gate`: diagnostic summary of global-vs-local reliance
- `Probabilistic head`: distributional output layer

## State Signal

Primary index:

- `S&P 500`

Primary continuous state inputs:

- `index_position_t = (Index_t - MA200_t) / MA200_t`
- `index_intensity_t = quantile_rank(RV30_t ; 252)`

Robustness:

- replace `S&P 500` with `Nasdaq`
- use both as a two-index conditioning variant only after the single-index version is stable

## H1: Performance

`IC-PE improves predictive performance relative to static PE and concat conditioning.`

Formal version:

- `Loss(IC-PE) < Loss(static PE)`
- `Loss(IC-PE) < Loss(concat state)`

Current evidence status:

- supported versus `static`
- not supported versus `concat_a` on repeated-seed ranking checks
- still open only as an exploratory coordinate-warp question via `flow_pe`

Primary metrics:

- rank IC
- ICIR
- MAE
- CRPS, if using probabilistic output

Interpretation:

- the benefit must come from the injection point, not merely from adding the same state information

## H2: Routing and Motif Hypothesis

`IC-PE changes how the model routes information across time, and those routing changes have measurable local-pattern consequences.`

Required evidence:

- attention topology differs across low- and high-intensity states
- TCN/CNN activations or downstream errors shift with the conditioned coordinates
- masking timesteps with high diagnostic importance degrades IC-PE more than control baselines

Current evidence status:

- not supported yet
- current top-attention masking does not hurt more than random masking
- H2 therefore remains open and needs a stronger routing ablation

Important wording:

- this is a diagnostic hypothesis
- not an explanation claim

## H3: Distribution Hypothesis

`A probabilistic output head calibrated on top of IC-PE captures state-dependent spread better than a point head alone.`

Default implementation path:

- keep `CVAE` as the first probabilistic head

Interpretation rule:

- CVAE is not the main novelty
- if quantile regression or another simpler head matches or beats it, keep the paper and swap the decoder

Primary metrics:

- CRPS
- PI-80 coverage
- calibration / reliability
- regime-conditional spread

## Demoted Finance Application

Residual-return prediction can remain as an application case:

- `IC-PE -> residual target`
- `IC-PE -> raw return target`

But this is no longer the main paper question. It is an empirical setting used to test the
representation claim.
