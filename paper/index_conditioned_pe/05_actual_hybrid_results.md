# Actual Hybrid Results

## Scope

These are the current paper-facing neural results for the stock-only path:

- `paper/index_conditioned_pe/icpe_transformer.py`
- `paper/index_conditioned_pe/icpe_cvae.py`
- `paper/index_conditioned_pe/icpe_hybrid_model.py`
- `paper_test/icpe_hybrid_supervised.py`
- `paper_test/icpe_hybrid_stability.py`

The older proxy experiments are no longer the decision center.

## Current Authoritative Read

What is robust:

- the paper path is stock-only and standalone
- explicit stock-index conditioning is useful relative to `static`
- conditioned variants produce positive state-swap response, while `static` stays at zero
- `concat_a` is the current best ranking baseline on repeated-seed checks

What is not robust:

- `cycle_pe > concat_a`
- `true coordinate warp > concat_a`
- `attention masking validates routing importance`
- CVAE-based H3

## Point-Head Read

### Stability Summary

Output files:

- `paper/index_conditioned_pe/results/hybrid_stability_residual_point_summary.csv`
- `paper/index_conditioned_pe/results/hybrid_stability_residual_point_winners.csv`

Mean IC by anchor/window:

| index/window | `concat_a` | `flow_pe` | `cycle_pe_full` | `static` |
|---|---:|---:|---:|---:|
| `^GSPC`, `2020-2024` | 0.0272 | 0.0109 | 0.0003 | -0.0192 |
| `^GSPC`, `2022-2024` | 0.0360 | 0.0257 | -0.0127 | 0.0072 |
| `^IXIC`, `2020-2024` | 0.0278 | 0.0112 | 0.0006 | -0.0192 |

Read:

- `concat_a` wins mean IC in all repeated-seed windows tested so far
- `flow_pe` is usually second on IC and sometimes best on MAE
- `cycle_pe_full` is not competitive in the current stock-only setup

### Single-Run Reference: S&P 500, Residual, Point, 2020-2024, 1 Epoch

Output file:

- `paper/index_conditioned_pe/results/hybrid_gspc_residual_point_2020_2024_e1_summary.csv`

| pe_mode | IC | ICIR | MAE | state-swap delta |
|---|---:|---:|---:|---:|
| `static` | -0.0174 | -0.0641 | 0.009476 | 0.000000 |
| `concat_a` | 0.0214 | 0.0784 | 0.009100 | 0.003204 |
| `cycle_pe_full` | -0.0555 | -0.1991 | 0.008031 | 0.004363 |
| `flow_pe` | 0.0372 | 0.1315 | 0.019071 | 0.000197 |

Read:

- single-run IC can still move around enough to create false optimism
- `flow_pe` looked strong on IC once, but the swap diagnostic was almost zero
- this is why repeated-seed stability became the real decision rule

### Fair-Budget Follow-Up: S&P 500, Residual, Point, 2020-2024, 3 Epochs

Output file:

- `paper/index_conditioned_pe/results/hybrid_gspc_residual_point_2020_2024_e3_summary.csv`

| pe_mode | IC | ICIR | MAE | state-swap delta |
|---|---:|---:|---:|---:|
| `static` | 0.0029 | 0.0113 | 0.007485 | 0.000000 |
| `concat_a` | 0.0252 | 0.1044 | 0.007530 | 0.000188 |
| `flow_pe` | -0.0150 | -0.0641 | 0.007490 | 0.000258 |

Read:

- under a less noisy training budget, `concat_a` still leads on ranking
- `flow_pe` no longer shows a ranking edge
- the current coordinate-warp implementation remains too weakly active to headline

See also:

- `paper/index_conditioned_pe/07_coordinate_warp_results.md`
- `paper/index_conditioned_pe/08_routing_mask_results.md`

## CVAE Read

Output file:

- `paper/index_conditioned_pe/results/hybrid_gspc_residual_cvae_2021_2024_e1_summary.csv`

| pe_mode | IC | ICIR | MAE | CRPS | PI80 | state-swap delta |
|---|---:|---:|---:|---:|---:|---:|
| `static` | -0.0370 | -0.1530 | 0.065742 | 0.043059 | 0.5196 | 0.061525 |
| `concat_a` | 0.0052 | 0.0237 | 0.011512 | 0.017902 | 1.0000 | 0.080146 |
| `cycle_pe` | -0.0130 | -0.0646 | 0.028597 | 0.021154 | 0.9977 | 0.069607 |

Read:

- the probabilistic path runs end-to-end
- `concat_a` still leads on ranking and CRPS in the current CVAE setup
- coverage is too wide to make H3 defensible yet

## Locked Claim

The paper can currently defend:

`Explicit stock-index conditioning changes representation and routing in a useful way, and input-concat conditioning is the strongest ranking baseline so far.`

The paper cannot yet defend:

- `PE injection is consistently superior to concat conditioning`
- `true coordinate warping is already active enough to carry the paper`
- `CVAE uncertainty is ready to headline`

## Consequence

The live paper center is now stricter than the original PE-superiority hope:

1. keep the stock-only boundary
2. keep repeated-seed repeated-window evaluation as the real decision rule
3. treat `concat_a` as the control-to-beat, not as a disposable baseline
4. treat `flow_pe` as an exploratory branch until its diagnostics become materially active
5. do not overclaim H3 until calibration is fixed
