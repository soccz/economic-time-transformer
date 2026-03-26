# Stock-Only Boundary

## Purpose

Prevent the paper track from drifting back into legacy crypto assumptions.

## Hard Boundary

For this paper, the allowed domain is:

- equities or equity portfolios
- broad-market stock indexes
- `S&P 500`
- `Nasdaq`
- Ken French portfolio and factor data

The excluded domain is:

- crypto assets
- BTC/ETH proxy factors
- exchange-specific coin universes
- any paper claim derived from legacy crypto feature engineering

## Allowed Paper-Facing Code

Only these files are authoritative for the current paper path:

- `paper/index_conditioned_pe/icpe_transformer.py`
- `paper/index_conditioned_pe/icpe_cvae.py`
- `paper/index_conditioned_pe/icpe_hybrid_model.py`
- `paper_test/icpe_proxy_ablation.py`
- `paper_test/icpe_hybrid_supervised.py`

## Disallowed Dependencies In The Paper Path

Paper-facing code must not depend on:

- `utils.config` crypto feature settings
- `btc_regime`
- `crypto_data.db`
- legacy GAN trainer logic
- legacy recommendation or trading modules

## Allowed Language

Use:

- `broad-market index`
- `S&P 500 anchor`
- `Nasdaq anchor`
- `index-conditioned positional encoding`
- `state-conditioned routing`

Do not use in the paper narrative:

- BTC regime
- coin universe
- crypto-native factor
- exchange tickers

## Decision Rule

If a result only works because of a legacy crypto assumption, it is excluded from the paper.

If a code path requires crypto config to run, it is not paper-authoritative until replaced or isolated.
