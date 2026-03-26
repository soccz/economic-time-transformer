# Pre-Registered Test Specification

Date fixed: 2026-03-09

This document freezes the confirmatory hypotheses and inference protocol for the
`path-2` paper direction before running the next replication experiment.

## Status

- `2022-2024` results are treated as exploratory only.
- The next `2020-2024` evaluation is the confirmatory replication.
- Any claim about "volatile-regime advantage" must be based on the confirmatory
  test below, not on the exploratory `2022-2024` findings.

## Paper Direction

Working paper question:

`How does the choice of conditioning space in financial Transformers change the
trade-off between cross-sectional ranking and absolute return prediction?`

Confirmatory comparison:

- `concat_a`: input-space conditioning
- `tau_rope`: coordinate-space conditioning
- `static`: sanity baseline only

## Confirmatory Hypotheses

### H1 (Primary)

In high-volatility regimes (`intensity > 0.5`), `tau_rope` achieves higher
daily cross-sectional IC than `concat_a`, evaluated on pooled `GSPC + IXIC`
out-of-sample predictions over `2020-01-01` to `2024-12-31`.

### H2 (Secondary)

In high-volatility regimes (`intensity > 0.5`), `tau_rope` achieves lower MAE
than `concat_a`, evaluated on the same pooled out-of-sample predictions.

## Inference Protocol

Primary IC test:

- Compute daily cross-sectional IC separately for each `market-date`.
- Restrict to days with `intensity > 0.5`.
- Form the difference series:
  `DeltaIC_t = IC_tau_rope,t - IC_concat_a,t`
- Test `E[DeltaIC_t] > 0` with a one-sided Newey-West t-test.
- Newey-West lag is fixed at `4` because the target uses a 5-day overlapping
  return horizon.

Secondary MAE test:

- Compute daily MAE separately for each `market-date`.
- Restrict to days with `intensity > 0.5`.
- Form the difference series:
  `DeltaMAE_t = MAE_concat_a,t - MAE_tau_rope,t`
- Test `E[DeltaMAE_t] > 0` with a one-sided Newey-West t-test.
- Same Newey-West lag `4`.

## Sanity Conditions

- `static` remains in every confirmatory table.
- If `tau_rope < static` in any seed-market configuration, that run is flagged
  and analyzed separately before any stronger claim is made.
- If H1 fails, the paper must not claim that coordinate conditioning improves
  ranking in high-volatility regimes.
- If H2 fails, the paper must not claim an absolute-prediction advantage for
  coordinate conditioning.

## Exploratory Analyses Only

The following are explicitly exploratory and must be labeled as such:

- `Bull` versus `Bear` trend splits
- portfolio/backtest results
- learned `tau` variants
- any analysis derived from the exploratory `2022-2024` window

## Interpretation Rule

- If H1 and H2 are both supported, the paper may claim a volatile-regime
  advantage for coordinate-space conditioning.
- If only H2 is supported, the paper may claim an absolute-prediction advantage
  but not a ranking advantage.
- If neither H1 nor H2 is supported, the paper reverts to the weaker claim that
  index conditioning is useful relative to `static`, while the conditioning
  space trade-off remains exploratory.
