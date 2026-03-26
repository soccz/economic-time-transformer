# Routing Mask Results

## Purpose

The paper should not rely on attention plots unless a quantitative routing diagnostic agrees.

This note tests the simplest version of that diagnostic:

- take the model's top-attention timesteps
- mask them
- compare the damage against masking the same number of random timesteps

If attention-guided routing is meaningful, top-mask damage should be larger than random-mask damage.

## Run

Script:

- `paper_test/icpe_mask_diagnostic.py`

Output file:

- `paper/index_conditioned_pe/results/hybrid_mask_gspc_residual_point_2020_2024_e3_summary.csv`

Setup:

- anchor: `S&P 500`
- target: `residual`
- decoder: `point`
- window: `2020-2024`
- epochs: `3`
- mask fraction: `20%` of the 30-step sequence
- models: `static`, `concat_a`, `flow_pe`

## Results

| pe_mode | base IC | top-mask IC | random-mask IC | IC drop top | IC drop random | MAE up top | MAE up random |
|---|---:|---:|---:|---:|---:|---:|---:|
| `static` | 0.0029 | 0.0747 | 0.0086 | -0.0718 | -0.0057 | 0.000023 | -0.000000 |
| `concat_a` | 0.0252 | 0.0360 | 0.0308 | -0.0107 | -0.0056 | -0.000001 | 0.000002 |
| `flow_pe` | -0.0150 | -0.0103 | -0.0268 | -0.0047 | 0.0118 | -0.000007 | 0.000008 |

Read:

- `static`: masking top-attention timesteps improved IC sharply
- `concat_a`: masking top-attention timesteps also improved IC slightly
- `flow_pe`: top-mask damage was smaller than random-mask damage

That is the opposite of what a strong routing diagnostic should show.

## Interpretation

Current attention importance is not a defendable causal-importance signal.

The likely reasons are:

- attention is still close to uniform in entropy terms
- the current `last-query attention` extraction is too weak as a routing proxy
- top-attention timesteps are not aligned with the timesteps that matter most for out-of-sample ranking

## Consequence For The Paper

What this result rules out:

- `attention heatmaps as explanation`
- `top-attention masking as current proof of routing importance`

What remains allowed:

- attention and gates as descriptive diagnostics
- state-swap as a minimal activity check
- future routing diagnostics based on stronger perturbations or branch-level ablations

## Updated Diagnostic Rule

Until a stronger routing test succeeds, the paper should say:

`attention and gate statistics are descriptive diagnostics, not validated explanation mechanisms`
