# Smoke Test Results

## Scope

Fast proxy runs using:

- `paper_test/icpe_proxy_ablation.py`
- public Ken French 25 portfolios
- `S&P 500` anchor
- sample: `2020-01-01` to `2024-12-31`
- sparse walk-forward: `step=120`

These are not publication results. They are decision-stage signals only.

## Run A: Raw target

Command:

```bash
python -u paper_test/icpe_proxy_ablation.py --start 2020-01-01 --end 2024-12-31 --step 120 --target raw
```

Summary:

| variant | IC | ICIR | mean CRPS | PI80 |
|---|---:|---:|---:|---:|
| static | 0.1269 | 0.3007 | 0.007682 | 0.8733 |
| concat | 0.1318 | 0.3543 | 0.008850 | 0.7600 |
| icpe | 0.0926 | 0.2446 | 0.008933 | 0.7400 |

DM:

- static vs icpe: `t=-5.768`, `p=0.0000` -> `static` wins
- concat vs icpe: `t=-1.377`, `p=0.1686` -> not significant

Diagnostic:

- mean state-swap delta: `0.033160`

Interpretation:

- the conditioning channel is active
- but in this raw-target proxy, the conditioned representation does not beat the simpler baseline
- this is a `no-go` signal for a raw-return-centered version of the paper

## Run B: Residual target

Command:

```bash
python -u paper_test/icpe_proxy_ablation.py --start 2020-01-01 --end 2024-12-31 --step 120 --target residual
```

Summary:

| variant | IC | ICIR | mean CRPS | PI80 |
|---|---:|---:|---:|---:|
| static | 0.0160 | 0.0710 | 0.002744 | 0.7267 |
| concat | 0.0188 | 0.0534 | 0.002808 | 0.6733 |
| icpe | 0.0335 | 0.1201 | 0.002819 | 0.6867 |

DM:

- static vs icpe: `t=-1.209`, `p=0.2265` -> not significant
- concat vs icpe: `t=-0.517`, `p=0.6053` -> not significant

Diagnostic:

- mean state-swap delta: `0.004549`

Interpretation:

- IC/ICIR improve in the `icpe` proxy
- CRPS does not improve
- this is not a win, but it is enough to keep the representation thesis alive in the residual-target setting

## Current Read

The main topic stays alive, but only under a narrower statement:

- `IC-PE may help ranking-style representation on residual targets`
- `IC-PE is not yet supported as a raw-return forecasting win`

## Immediate Consequence

The next serious experiments should prioritize:

1. residual target
2. better local/global hybrid features than the current linear proxy
3. `concat` vs `IC-PE` comparisons under the actual hybrid model

Raw-target expansion can wait until the representation story is stronger.
