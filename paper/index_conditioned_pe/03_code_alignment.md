# Code Alignment

## Status

The paper path is now split into two layers:

- legacy repository context, which remains mixed-domain
- paper-specific implementation, which must remain stock-only

## Reusable Components

Current code that is directly relevant:

- `../models/transformer_encoder.py`
  - already has `static`, `concat_a`, `cycle_pe`, `cycle_pe_full`
- `../models/hybrid_model.py`
  - already has Transformer + attention-guided local branch + gate + FiLM
- `../models/cvae_decoder.py`
  - usable as the first probabilistic head
- `../aaa/paper_test/h2_h3_ablation.py`
  - already compares `static`, `concat_a`, `cycle_pe`

## Mismatch With New Thesis

### Naming / domain mismatch

The codebase still reflects a crypto project:

- `btc_regime`
- `crypto_data.db`
- BTC/ETH market index assumptions
- context names like `market_index_return`, `historical_similarity`

This prevents a clean stock-paper narrative.

### Conditioning mismatch

The new paper needs:

- explicit `S&P 500` anchor
- explicit `Nasdaq` robustness anchor
- continuous index state variables

The current implementation only partially approximates this through generic context channels.

## Paper-Specific Clean Boundary

The paper-facing implementation is now self-contained under:

- `paper/index_conditioned_pe/icpe_transformer.py`
- `paper/index_conditioned_pe/icpe_cvae.py`
- `paper/index_conditioned_pe/icpe_hybrid_model.py`
- `paper_test/icpe_hybrid_supervised.py`
- `paper_test/icpe_proxy_ablation.py`

These files are the only code that should be treated as authoritative for the paper path.
They do not depend on:

- `btc_regime`
- `crypto_data.db`
- legacy GAN decoder paths
- legacy crypto feature names

## Actual Local Data Finding

Current local files show:

- `../data/crypto_data.db`
- `../data/portfolio.db`
- no clearly prepared local stock-index research dataset for `S&P 500` / `Nasdaq`

Current paper prototype scripts such as `../aaa/paper_test/h2_h3_ablation.py` fetch data from:

- `Ken French` via `pandas_datareader`
- `Yahoo Finance` via `yfinance`

Implication:

- a stock-paper experiment is not yet fully reproducible from local files alone
- either a local stock data cache must be added, or the paper scripts must be allowed to fetch public data

### Training mismatch

`../training/trainer.py` is still centered on WGAN-GP training logic.

That is misaligned with the new thesis because:

- the paper center is not adversarial generation
- probabilistic output is secondary
- the current training loop makes the decoder story harder to defend

### Diagnostic gap

The current code exposes attention and gate values, but the new paper needs quantitative
diagnostics:

- perturbation / masking
- state-swap tests
- routing ablation

Those are not yet implemented as first-class evaluation steps.

## Immediate Refactor Priorities

1. Rename the paper-facing conditioning concept:
   - `cycle_pe` -> keep internally for now
   - paper term -> `IC-PE`

2. Add stock-index context builder:
   - produce `index_position`
   - produce `index_intensity`
   - allow `S&P 500` and `Nasdaq`

3. Build a paper-specific evaluation script:
   - `static` vs `concat_a` vs `IC-PE`
   - perturbation test
   - state-swap test
   - probabilistic metrics when enabled

4. Decouple paper experiments from the legacy GAN trainer:
   - add a simpler supervised trainer path for point / quantile / CVAE heads

## Active Implementation Files

- `paper/index_conditioned_pe/`
  - paper design docs
  - stock-only transformer / decoder / hybrid modules
- `../aaa/paper_test/`
  - stock-only proxy and supervised experiment scripts

## Decision Constraint

No more architecture additions until the following is true:

- `IC-PE` beats `static`
- `IC-PE` beats or ties `concat`
- state-swap shows conditioning is active

If those conditions fail, the paper must be reframed before more model work.
