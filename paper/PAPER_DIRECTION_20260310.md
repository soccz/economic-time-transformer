# Paper Direction Lock (2026-03-10)

## Purpose

This document freezes the paper direction after reviewing the current repository state:

- top-level design and draft documents
- `paper/index_conditioned_pe/*`
- `paper/economic_time/*`
- `paper_test/*.py`
- result summaries under both paper tracks

The goal is not to preserve every old idea. The goal is to choose the strongest paper that the repository can defend now.

## Repository Read

The repository currently contains three paper layers.

### Layer 1: legacy finance / residual-momentum path

Files:

- `paper/paper_idea.md`
- `paper/paper_idea_ko.md`
- `paper/paper_ko.md`
- `paper/paper_final_1.md`
- `paper/paper_final_2.md`
- `paper_test/h1_*.py`
- `paper_test/h2_h3_ablation.py`

Read:

- this path is economically motivated
- but the main H1 story is not locked by stored confirmatory results
- it overlaps heavily with existing momentum-crash / regime literature
- it is no longer the cleanest match to the user's core idea

Decision:

- keep only as background or a separate future finance paper
- do not use as the main paper center

### Layer 2: `index_conditioned_pe` transitional path

Files:

- `paper/index_conditioned_pe/*`
- `paper_test/icpe_*`

Read:

- this path correctly moved the center from residual momentum to representation
- it clarified that the real question is conditioning the temporal coordinate system
- but the stored results are explicit: `concat_a` is the ranking baseline to beat
- `flow_pe` and `cycle_pe_full` do not robustly beat `concat_a`
- attention masking fails as an explanation diagnostic
- CVAE is not calibrated well enough to headline

Decision:

- keep as the conceptual bridge
- do not use `IC-PE > concat_a` as the headline claim

### Layer 3: `economic_time` path

Files:

- `paper/economic_time/00_preregistered_test_spec.md`
- `paper/economic_time/market_time_model.py`
- `paper_test/economic_time_supervised.py`
- `paper_test/economic_time_stability.py`
- `paper_test/economic_time_confirmatory.py`
- `paper_test/economic_time_regime_report.py`

Read:

- this is the most mature paper path
- it narrows the question from "is PE superior?" to "what does conditioning space change?"
- it contains a pre-registered confirmatory test
- it has stored confirmatory outputs for `2020-2024`
- it is the only path with a paper claim that is both narrow and empirically supported

Decision:

- this becomes the main paper path

## Locked Main Question

Use this as the paper question:

`How does the choice of conditioning space in financial Transformers change the trade-off between cross-sectional ranking and absolute return prediction, especially in high-volatility regimes?`

Short version:

`Input-space conditioning versus coordinate-space conditioning in financial Transformers`

## Locked Main Comparison

Only three models belong in the main paper:

- `static`: no market conditioning
- `concat_a`: input-space conditioning baseline
- `tau_rope`: coordinate-space conditioning model

Interpretation:

- `concat_a` = state is appended to the input representation
- `tau_rope` = state changes the temporal coordinate system used inside attention
- `static` = sanity baseline only

Do not make these main-paper models:

- `cycle_pe`
- `cycle_pe_full`
- `flow_pe`
- `econ_time`
- `econ_time:pe_only`
- `econ_time:qk_only`
- `learned_tau_rope`
- `CVAE`

They can appear only in appendix or future work.

## Locked Empirical Claim

The strongest defensible claim in the current repository is:

`Coordinate-space conditioning does not dominate input-space conditioning everywhere, but it improves absolute prediction in high-volatility regimes and shows a marginal pooled ranking advantage, while input-space conditioning remains a strong general ranking baseline.`

This is the honest hierarchy:

### Claim A: strong and safe

`Explicit market conditioning matters relative to static conditioning.`

Supported by:

- `index_conditioned_pe` results
- `economic_time` confirmatory summaries

### Claim B: main paper claim

`Coordinate-space conditioning (`tau_rope`) improves MAE in high-volatility regimes relative to `concat_a`, and may improve ranking there as well, but the ranking advantage is weaker than the MAE advantage.`

Supported by:

- pooled confirmatory Newey-West test:
  - H1 high-vol IC: `p = 0.0595`
  - H2 high-vol MAE: `p = 0.0019`

### Claim C: allowed nuance

`The two conditioning spaces induce different error trade-offs: input-space conditioning is a strong ranking baseline, while coordinate-space conditioning is more favorable for absolute prediction under stress.`

Supported by:

- preregistered question
- confirmatory pooled results
- exploratory regime reports

## Claims That Must Be Removed

The repository does not support these claims:

- `PE injection is consistently superior to concat conditioning`
- `attention weights are valid explanations`
- `top-attention masking proves routing importance`
- `CVAE uncertainty is ready to headline`
- `learned tau is better than rule-based tau`
- `residual momentum / Fama-MacBeth is the paper center`
- `this is a novel probabilistic sequence model paper`

## Evidence Matrix

### Transitional `index_conditioned_pe` evidence

What survives:

- explicit stock-index conditioning beats `static`
- `concat_a` is the most stable ranking model
- `flow_pe` sometimes helps MAE

What fails:

- `flow_pe > concat_a` on ranking
- active-routing proof via attention masking
- CVAE headline story

Key files:

- `paper/index_conditioned_pe/05_actual_hybrid_results.md`
- `paper/index_conditioned_pe/07_coordinate_warp_results.md`
- `paper/index_conditioned_pe/08_routing_mask_results.md`

### Main `economic_time` evidence

Pre-registered confirmatory setup:

- `paper/economic_time/00_preregistered_test_spec.md`

Confirmatory result:

- pooled high-vol IC difference (`tau_rope - concat_a`): positive, one-sided NW `p = 0.0595`
- pooled high-vol MAE difference (`concat_a - tau_rope`): positive, one-sided NW `p = 0.0019`

Per-market breakdown:

- MAE advantage holds in both `GSPC` and `IXIC`
- IC advantage is positive in both markets but not individually significant

Interpretation:

- the paper should headline MAE and stress-regime trade-off
- the ranking result should be described as marginal / supportive, not definitive

## Final Paper Positioning

Use this positioning:

`Applied ML / forecasting paper with financial structure`

Do not position as:

- pure finance theory paper
- pure explainability paper
- pure probabilistic forecasting paper

Recommended venue tier:

- FinML / forecasting / applied ML venues
- workshop, ECML-PKDD-style applied track, or solid forecasting journal

Avoid writing as though this is ready for a top finance journal. The stored data and experiments are not at that level yet.

## Final Title Options

Use one of these:

1. `Conditioning Space in Financial Transformers: Input-Space vs Coordinate-Space Conditioning under Market Volatility`
2. `Economic Time in Financial Transformers: Coordinate-Space Conditioning under High-Volatility Regimes`
3. `When Market Time Speeds Up: Coordinate-Space Conditioning in Financial Transformers`

Best default:

`Economic Time in Financial Transformers: Input-Space vs Coordinate-Space Conditioning under Market Volatility`

## Final Abstract Skeleton

Structure:

1. Problem:
   standard financial Transformers usually treat market state as an input feature, not as part of the temporal coordinate system.
2. Question:
   does conditioning space change ranking and absolute-prediction behavior?
3. Method:
   compare `static`, `concat_a`, and `tau_rope` on Ken French 25 portfolios with `S&P 500` and `Nasdaq` anchors.
4. Confirmatory result:
   in high-volatility regimes, `tau_rope` significantly improves MAE and yields a marginal pooled IC gain.
5. Interpretation:
   coordinate-space conditioning is not uniformly superior, but it changes the ranking-vs-MAE trade-off under stress.

## Final Contribution List

Use exactly three contributions.

1. `Problem reformulation`
   We recast market conditioning as a conditioning-space problem: input space versus temporal coordinate space.

2. `Method`
   We implement a coordinate-conditioned Transformer hybrid using economic-time RoPE (`tau_rope`) and compare it directly to input-space conditioning.

3. `Evidence`
   We provide a pre-registered confirmatory test showing that coordinate-space conditioning improves absolute prediction in high-volatility regimes, with weaker but directionally positive pooled ranking gains.

Do not list uncertainty decomposition or explainability as core contributions.

## Final Method Section Structure

### 1. Data

- Ken French 25 portfolios
- FF3 residual target
- `S&P 500` and `Nasdaq` anchor indexes
- `2022-2024` exploratory
- `2020-2024` confirmatory

### 2. State Construction

- `position = (Index - MA200) / MA200`
- `intensity = RV30 quantile rank over 252 days`

### 3. Models

- `static`
- `concat_a`
- `tau_rope`

### 4. Architecture

Main description should follow the actual implemented `tau_rope` hybrid:

- Transformer global branch
- TCN local branch
- scalar fusion gate
- rule-based economic-time coordinate builder
- RoPE with `tau`

### 5. Evaluation

- daily cross-sectional IC
- MAE
- Newey-West one-sided confirmatory tests
- market-specific breakdown

## Final Results Section Structure

### 6.1 Exploratory

Use `2022-2024` only to motivate the confirmatory test.

Message:

- `tau_rope` can approach or exceed `concat_a` in some windows
- but exploratory wins alone are not enough

### 6.2 Confirmatory

This is the core section.

Report:

- pooled high-vol IC delta
- pooled high-vol MAE delta
- market breakdown table
- static sanity comparison

### 6.3 Diagnostics

Allowed diagnostics:

- `tau` summary
- `step-intensity` alignment
- fusion gate distribution
- regime-wise IC / MAE table

Forbidden headline diagnostics:

- attention heatmaps as explanation
- masking-based causal claims

## Final Figures

Keep the figure set small.

1. architecture diagram for `tau_rope` vs `concat_a`
2. pooled confirmatory delta distribution plot
3. regime-split IC / MAE bar chart
4. example `tau` trajectories under low versus high volatility

Optional appendix figure:

- exploratory stability across seeds and windows

## Appendix Policy

Move the following to appendix or future work:

- `flow_pe`
- `cycle_pe_full`
- `econ_time` full model
- `learned_tau_rope`
- CVAE / quantile heads
- attention masking
- residual-momentum H1 scripts

This keeps the main paper coherent.

## Writing Rules

### Use

- `conditioning space`
- `input-space conditioning`
- `coordinate-space conditioning`
- `economic time`
- `high-volatility regime`
- `cross-sectional ranking`
- `absolute prediction`

### Avoid

- `attention explains`
- `market-state manifold` unless formally developed
- `novel PE paper` without qualification
- `probabilistic transformer` as the paper identity
- `finance theory contribution`

## Final One-Sentence Thesis

`In financial Transformers, the choice of conditioning space matters: input-space conditioning is a strong ranking baseline, while coordinate-space conditioning via economic time is more favorable for absolute prediction under high volatility and shows only a weaker, regime-local ranking advantage.`

## Immediate Next Steps

1. Rewrite the paper around `static` vs `concat_a` vs `tau_rope` only.
2. Promote `paper/economic_time/00_preregistered_test_spec.md` to the main Method/Results backbone.
3. Use `confirmatory_2020_2024` as the paper's core evidence table.
4. Demote all uncertainty material to appendix unless a new calibrated result is produced.
5. Remove any wording that implies validated attention explanation.
6. Keep residual-momentum material out of the main introduction.
