# Experiment Matrix

## Goal

Test whether `IC-PE` is better because of the coordinate-conditioning mechanism, not because
of extra parameters or extra market information.

## Data Tracks

Track 1:

- current local reproducible track using available public / existing data and the current codebase

Track 2:

- stock-focused track using broad-market index anchors

The paper should not claim more than the data track can support.

## Model Grid

### Representation Core

`M0` static Transformer

- standard PE
- no state conditioning

`M1` concat-conditioned Transformer

- same state variables appended to input channels
- control for information quantity

`M2` IC-PE Transformer

- state enters through positional conditioning
- main representation model
- practical variants:
  - additive state-in-PE (`cycle_pe`, `cycle_pe_full`)
  - true coordinate warp (`flow_pe`)

### Architecture Role Tests

`M3` TCN/CNN only + concat state

- local-pattern baseline

`M4` IC-PE Transformer only

- isolates the effect of the global branch

`M5` IC-PE Transformer + TCN/CNN hybrid

- main hybrid architecture

`M6` hybrid with random or uniform routing prior

- tests whether attention-guided local routing matters

## Output-Head Grid

`D0` point head

- baseline forecasting head

`D1` quantile head

- simple probabilistic baseline

`D2` CVAE head

- preferred distribution head, but not guaranteed to remain final

## Required Comparisons

Representation:

- `M0` vs `M1`
- `M1` vs `M2`
- `M2` vs `M4`
- `M4` vs `M5`

Routing diagnostics:

- `M5` vs `M6`

Probabilistic output:

- `D0` vs `D1`
- `D1` vs `D2`

Index anchor:

- `S&P 500` anchor vs `Nasdaq` anchor

## Metrics

Performance:

- IC
- ICIR
- MAE
- CRPS

Calibration:

- PI-80 coverage
- reliability curve
- regime-conditional spread error

Diagnostics:

- performance drop after masking top-diagnostic timesteps
- state-wise gate summary
- state-wise attention-distance summary
- local-branch error change under routing ablations

## Diagnostic Tests

### Perturbation Test

Mask or perturb top-importance timesteps:

- if IC-PE is meaningful, loss should rise more than in static or concat models

### State-Swap Test

Keep the same price sequence but swap index-state conditioning:

- if outputs change materially, the conditioning channel is active
- if outputs do not change, IC-PE is probably decorative

### Anchor Robustness

Swap `S&P 500` with `Nasdaq`:

- same direction is good
- a full collapse means the thesis is unstable

## Go / No-Go Rules

Go:

- `M2` beats both `M0` and `M1` on at least one primary forecast metric with consistent fold-wise direction
- perturbation and state-swap diagnostics show that conditioning is active

Conditional go:

- `M2` ties `M1` but both beat `M0`
- then the paper downgrades from "PE injection is better" to "explicit index conditioning matters"

No-go:

- `M2` does not beat `M1` and diagnostics do not show active conditioning
- then IC-PE is not the paper center

## Reporting Rule

Do not report attention plots alone.

Every qualitative figure must be paired with one quantitative diagnostic:

- masking
- state-swap
- routing ablation
