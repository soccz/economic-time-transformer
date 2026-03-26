# True Coordinate-Warp Results

## Why This Follow-Up Was Needed

The original `cycle_pe` family was not a true coordinate warp.

- `cycle_pe`: sinusoidal PE plus additive intensity projection
- `cycle_pe_full`: sinusoidal PE plus additive intensity and position projections

That means the earlier negative result could still be attacked with:

`This is not really index-conditioned temporal coordinates. It is just another state embedding.`

To test the user's actual thesis more faithfully, a new `flow_pe` mode was added to:

- `paper/index_conditioned_pe/icpe_transformer.py`

Implementation:

- learn a context-driven monotone step size at each timestep
- cumulatively integrate those step sizes into a warped time coordinate
- evaluate the sinusoidal PE on the warped coordinate instead of the static integer index

This is a true coordinate-conditioning variant rather than another additive state injection.

## Parameter Fairness

Point-head hybrid parameter counts:

| pe_mode | parameters |
|---|---:|
| `static` | 45,202 |
| `concat_a` | 45,266 |
| `flow_pe` | 45,236 |
| `cycle_pe_full` | 45,330 |

Read:

- `flow_pe` is not winning because of a larger parameter budget
- `concat_a` and `flow_pe` are essentially matched on size

## Run A: One-Shot Smoke, S&P 500, Residual Target, Point Head, 2020-2024, 1 Epoch

Output file:

- `paper/index_conditioned_pe/results/hybrid_gspc_residual_point_2020_2024_e1_summary.csv`

| pe_mode | IC | ICIR | MAE | state-swap delta |
|---|---:|---:|---:|---:|
| `static` | -0.0174 | -0.0641 | 0.009476 | 0.000000 |
| `concat_a` | 0.0214 | 0.0784 | 0.009100 | 0.003204 |
| `cycle_pe_full` | -0.0555 | -0.1991 | 0.008031 | 0.004363 |
| `flow_pe` | 0.0372 | 0.1315 | 0.019071 | 0.000197 |

Read:

- `flow_pe` won IC on this single run
- but it had the worst MAE by a wide margin
- its state-swap response was almost zero
- this is not strong enough to claim active conditioning

## Run B: Stability Check, 2 Seeds x 3 Windows, Residual Target, Point Head, 1 Epoch

Output files:

- `paper/index_conditioned_pe/results/hybrid_stability_residual_point_summary.csv`
- `paper/index_conditioned_pe/results/hybrid_stability_residual_point_winners.csv`

Mean IC summary:

| index/window | `concat_a` | `flow_pe` | `cycle_pe_full` | `static` |
|---|---:|---:|---:|---:|
| `^GSPC`, `2020-2024` | 0.0272 | 0.0109 | 0.0003 | -0.0192 |
| `^GSPC`, `2022-2024` | 0.0360 | 0.0257 | -0.0127 | 0.0072 |
| `^IXIC`, `2020-2024` | 0.0278 | 0.0112 | 0.0006 | -0.0192 |

Winner summary:

- IC winner:
  - `concat_a` in all three windows
- MAE winner:
  - `flow_pe` in `^GSPC 2020-2024`
  - `concat_a` in `^GSPC 2022-2024`
  - `flow_pe` in `^IXIC 2020-2024`

Diagnostic read:

- `flow_pe` does improve MAE in two of the three stability windows
- but it does not win mean IC anywhere
- its average state-swap delta remains tiny:
  - `0.000193`
  - `0.000855`
  - `0.000211`

That swap magnitude is much closer to `static` than to `concat_a`.

## Run C: Fairer Training-Budget Check, S&P 500, Residual Target, Point Head, 2020-2024, 3 Epochs

Output file:

- `paper/index_conditioned_pe/results/hybrid_gspc_residual_point_2020_2024_e3_summary.csv`

| pe_mode | IC | ICIR | MAE | state-swap delta |
|---|---:|---:|---:|---:|
| `static` | 0.0029 | 0.0113 | 0.007485 | 0.000000 |
| `concat_a` | 0.0252 | 0.1044 | 0.007530 | 0.000188 |
| `flow_pe` | -0.0150 | -0.0641 | 0.007490 | 0.000258 |

Read:

- under a less noisy 3-epoch comparison, `flow_pe` loses its one-shot IC advantage
- `concat_a` remains clearly best on ranking
- `flow_pe` collapses back toward `static`
- state-swap is still too small to call the coordinate warp materially active

## Decision

Current conclusion:

- a true coordinate-warp implementation is now present
- that implementation is interesting enough to keep as an exploratory branch
- but it does not currently support the headline claim that PE injection beats concat conditioning

What survives:

- `stock-index conditioning matters`
- injection location changes the error trade-off
- `concat_a` is the current ranking leader
- `flow_pe` may help point error in some windows, but it is not an active or dominant ranking mechanism yet

What does not survive:

- `true coordinate warping already beats concat conditioning`
- `PE superiority is established`
