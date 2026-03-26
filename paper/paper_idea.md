# Regime-Conditional Predictability of Factor-Adjusted Returns: A Hybrid Deep Learning Approach with Decomposed Uncertainty Estimation

---

## Abstract (draft)

We investigate whether market regimes alter the predictability structure of factor-adjusted equity returns — the component of individual stock returns unexplained by standard asset pricing factors. Using S&P 500 constituents (1990–2024) and Fama-French six-factor returns from Ken French's Data Library, we construct a rolling factor-adjusted residual as the prediction target and propose a hybrid deep learning model that explicitly encodes market cycle position into its temporal representation. The model combines a Transformer encoder (global trend dependencies) and a Temporal Convolutional Network (local price motifs) via a learned gating mechanism, conditioned on a four-state market regime. Uncertainty is decomposed into epistemic (MC-Dropout) and aleatoric (GAN-based sampling) components. We validate three pre-registered hypotheses: (H1) regime moderates the predictability of factor-adjusted returns; (H2) cycle-aware positional encoding improves forecast accuracy over static encoding; (H3) epistemic and aleatoric uncertainty decomposition is empirically meaningful. Results show [TBD after experiments].

**Keywords:** factor model, APT, regime-conditional forecasting, uncertainty decomposition, Transformer, GAN, epistemic uncertainty, aleatoric uncertainty

**Target journal:** International Journal of Forecasting / Journal of Empirical Finance / Quantitative Finance

---

## 1. Introduction

### 1.1 Motivation

Standard asset pricing models — CAPM, Fama-French three-factor (FF3), and the five-factor extension (FF5) — decompose individual stock returns into a systematic component explained by common factors and an idiosyncratic residual:

```
r_i,t = α_i + β_i,t · F_t + ε_i,t
```

The residual `ε_i,t` (plus any persistent alpha `α_i`) represents the component that factor models cannot explain. If markets were fully efficient and factor models were complete, this residual would be unpredictable white noise. In practice, however, substantial evidence suggests that residual predictability exists and varies over time — particularly across different market regimes.

Two empirical regularities motivate this paper:

1. **Time-varying factor loadings**: APT (Ross 1976) explicitly allows `β_i,t` to vary over time. Rolling OLS estimates confirm that factor exposures shift substantially across bull/bear cycles and volatility regimes. A model that treats β as static misspecifies the return-generating process.

2. **Regime-dependent information structure**: During trending (Bull/quiet) markets, recent price momentum dominates. During stressed (Bear/volatile) markets, historical pattern recurrence and cross-sectional factor spreads dominate. A single inductive bias cannot capture both.

Existing ML approaches to return prediction (Gu, Kelly, Xiu 2020; Chen, Kelly, Wu 2023) demonstrate that nonlinear models outperform linear factor models in predicting raw returns. However, they do not: (a) explicitly condition on market regime in the model architecture, (b) decompose prediction uncertainty into epistemic and aleatoric components, or (c) target factor-adjusted residuals as a theoretically motivated prediction objective.

This paper addresses all three gaps.

### 1.2 Research Questions

- **RQ1 (Economic)**: Does market regime moderate the predictability of factor-adjusted returns? Specifically, does the interaction term `residual_momentum × regime` carry incremental predictive power beyond unconditional momentum? (→ H1)
- **RQ2 (ML)**: Does explicitly encoding market cycle position into the Transformer's positional encoding improve out-of-sample forecast accuracy? (→ H2)
- **RQ3 (Uncertainty)**: Can epistemic and aleatoric uncertainty be meaningfully separated, and do they respond differently to market conditions? (→ H3)

### 1.3 Contributions

1. **Theoretically grounded prediction target**: Factor-adjusted residual as prediction target, directly motivated by APT time-varying β framework. This separates the "known" systematic component from the "unknown" idiosyncratic component.

2. **Cycle-aware positional encoding (Cycle-PE)**: A novel PE design that injects two continuous market cycle signals — `(index - MA200d) / MA200d` (cycle position) and `30d realized volatility quantile` (cycle intensity) — into the Transformer's time representation. Pre-specified, not HPO-tuned.

3. **Decomposed uncertainty estimation**: MC-Dropout (epistemic) + GAN noise sampling (aleatoric) with empirical validation that the two components respond to different market conditions.

4. **Pre-registered hypotheses**: All three hypotheses and their test statistics are specified before data analysis, following López de Prado (2018) and Gu, Kelly, Xiu (2020) recommendations.

### 1.4 Related Work

| Stream | Key papers | Gap addressed |
|--------|-----------|---------------|
| ML for return prediction | Gu, Kelly, Xiu (2020); Chen, Kelly, Wu (2023) | No regime conditioning; no uncertainty decomposition |
| Factor timing | Kim (2022); Polk, Haghbin, de Longis (2020) | Rule-based, not probabilistic; no uncertainty |
| Time-varying β | Ferson & Harvey (1991); Ang & Kristensen (2012) | No ML; no uncertainty decomposition |
| Probabilistic forecasting | Lim et al. (2021 TFT); Rasul et al. (2021) | No factor model grounding; no regime conditioning |
| Uncertainty in finance | Bali, Brown, Tang (2017) | No epistemic/aleatoric decomposition |

---

## 2. Theoretical Framework

### 2.1 Factor Model Decomposition

Following Ross (1976) APT and Fama-French (1993, 2015), the return of stock i at time t is:

```
r_i,t = α_i,t + β_i,t · F_t + ε_i,t

where:
  F_t = [MKT_t, SMB_t, HML_t, RMW_t, CMA_t, WML_t]  (6-factor vector)
  β_i,t = time-varying factor loading (estimated via rolling 60d OLS)
  ε_i,t = idiosyncratic residual (prediction target)
  α_i,t = persistent alpha (absorbed into residual for prediction)
```

**Prediction target**: `y_i,t = Σ_{τ=t+1}^{t+5} ε_i,τ` — 5-day cumulative factor-adjusted residual.

This target is theoretically motivated: if factor loadings are correctly estimated, `y_i,t` represents pure idiosyncratic information not captured by systematic risk premia.

### 2.2 Regime Definition (Pre-specified)

Market regime is defined by two continuous signals, both computed from S&P 500 index data available at time t-1 (no hindsight bias):

```
cycle_position_t  = (SPX_t - MA200d_t) / MA200d_t     # trend position
cycle_intensity_t = quantile(RV_30d_t, history)        # volatility state
```

Four-state discrete regime label (for FiLM conditioning):
```
regime_t = 2 × I(cycle_position_t > 0) + I(cycle_intensity_t > 0.5)
         ∈ {0: Bear/quiet, 1: Bear/volatile, 2: Bull/quiet, 3: Bull/volatile}
```

**Pre-specification rationale**: MA200d and 30d RV are not HPO-tuned. Sensitivity analysis over {MA120d, MA200d, MA252d} × {RV20d, RV30d, RV60d} grid is reported to demonstrate robustness.

### 2.3 Hypothesis Formulation (Pre-registered)

**H1 — Regime-conditional predictability**:

```
y_i,t = β_0 + β_1 · mom_i,t + β_2 · (mom_i,t × regime_t) + γ · controls_i,t + u_i,t

Null:    β_2 = 0  (regime does not moderate momentum predictability)
Alternative: β_2 ≠ 0, with sign pattern:
  Bull/quiet regime: β_2 > 0  (momentum amplified)
  Bear/volatile regime: β_2 < 0  (momentum attenuated or reversed)

Test: HAC t-test (Newey-West, 10 lags) on panel regression
      Fama-MacBeth cross-sectional robustness check
```

**H2 — Cycle-aware PE improves forecast accuracy**:

```
Null:    IC(Cycle-PE model) = IC(Static-PE model)
Alternative: IC(Cycle-PE model) > IC(Static-PE model)

Test: Diebold-Mariano test on out-of-sample IC difference
      Regime-stratified IC comparison (4 regimes separately)
```

**H3 — Epistemic/Aleatoric decomposition is meaningful**:

```
H3a: Epistemic uncertainty (MC-Dropout std) is higher for stocks with
     shorter history / lower liquidity / recent IPO
     Test: Spearman correlation between epistemic_i,t and data_sparsity_i,t

H3b: Aleatoric uncertainty (GAN noise std) is higher in Bear/volatile regime
     Test: regime × aleatoric_uncertainty ANOVA, Tukey HSD

H3c: CRPS(GAN model) < CRPS(Quantile Regression baseline)
     Test: Diebold-Mariano test on CRPS difference
```

---

## 3. Model Architecture

### 3.1 Overview

```
Input: (B, 60, D)  — 60-day window, D features per stock-day
  ↓
[Cycle-aware Positional Encoding]
  PE_total = PE_static(position) + PE_cycle(cycle_position, cycle_intensity)
  ↓
[Transformer Encoder]  — global: trend/regime dependencies
  → attention_weights: (B, 60, 60)
  → last_token: (B, d_model)
  ↓
[Attention-guided TCN]  — local: price pattern motifs
  input weighted by softmax(attention_weights[:, -1, :])
  → tcn_features: (B, d_tcn)
  ↓
[Explainable Gated Fusion]
  gate = σ(f([transformer_feat, tcn_feat, proto_sim_transformer, proto_sim_tcn]))
  fused = gate * transformer_feat + (1 - gate) * tcn_feat
  ↓
[FiLM Regime Conditioning]
  fused = scale(regime) * fused + shift(regime)
  ↓
[WGAN-GP Decoder]
  z ~ N(0, I)  →  y_hat = G(fused, z)
  ↓
Output: scalar  y_hat_i,t  (5d cumulative factor-adjusted residual)
```

### 3.2 Cycle-aware Positional Encoding

Standard sinusoidal PE encodes only integer position (1, 2, ..., T). Cycle-PE additionally encodes where the current window sits in the market cycle:

```
PE_static(pos, d) = sin/cos(pos / 10000^(2d/D))   # standard

PE_cycle(t, d) = [cycle_position_t, cycle_intensity_t] @ W_phase  # (2,) → (D,)
  where W_phase ∈ R^{2×D} is learned

PE_total = PE_static + PE_cycle
```

**Key property**: `cycle_position_t` and `cycle_intensity_t` are computed from data available at t-1 only. No hindsight bias.

**Interpretation**: Two stocks with identical price sequences but different cycle positions receive different temporal representations, allowing the Transformer to learn regime-conditional attention patterns.

### 3.3 Gated Fusion and Diagnostic Variables

The fusion gate is treated as a diagnostic variable, not an explanation:

```
gate ≈ 1.0  →  Transformer path dominates  (global/trend structure)
gate ≈ 0.0  →  TCN path dominates          (local/pattern structure)
```

Whether gate correlates with regime is an empirical question (H2 ablation), not a design assumption.

Gate regularization (prototype diversity):
```
L_gate_reg = λ · mean(proto_sim_matrix[off-diagonal])
```

### 3.4 Uncertainty Decomposition

```
Epistemic uncertainty (model uncertainty):
  generator.train()  # dropout active
  {y_hat^(k)}_{k=1}^{K=100}  via MC-Dropout
  σ_epistemic = std({y_hat^(k)})

Aleatoric uncertainty (data uncertainty):
  generator.eval()  # dropout inactive
  {y_hat^(k)}_{k=1}^{K=100}  via z^(k) ~ N(0,I)  (GAN noise)
  σ_aleatoric = std({y_hat^(k)})

Prediction interval (80%):
  PI_80 = [quantile(samples, 0.10), quantile(samples, 0.90)]
  combined: K=100 MC-Dropout × K=100 GAN samples = 10,000 total
```

### 3.5 Loss Function

Generator loss (WGAN-GP framework):

```
L_G = L_adv + λ_recon · L_recon + λ_crps · L_crps + λ_dir · L_dir + L_gate_reg

L_adv   = -E[C(y_hat)]                          # adversarial (WGAN)
L_recon = MSE(y_hat_mean, y_true)               # reconstruction
L_crps  = CRPS(empirical_CDF, y_true)           # proper scoring rule (replaces ECE)
L_dir   = -E[tanh(k · y_hat) · sign(y_true)]   # directional (differentiable)
```

**Design decisions**:
- ECE removed from loss (non-differentiable binning metric → evaluation only)
- `sign(pred)` replaced by `tanh(k · pred)` (gradient = 0 almost everywhere for sign)
- CRPS is a proper scoring rule for univariate scalar targets → theoretically justified

Critic loss (standard WGAN-GP):
```
L_C = E[C(y_hat)] - E[C(y_true)] + λ_gp · GP
```

---

## 4. Data and Empirical Setup

### 4.1 Data Sources

| Source | Content | Period | Access |
|--------|---------|--------|--------|
| Yahoo Finance | S&P 500 daily OHLCV | 1990–2024 | Free (yfinance) |
| Ken French Data Library | FF6 daily factor returns | 1963–2024 | Free (direct download) |
| S&P 500 index (^GSPC) | Regime signal construction | 1990–2024 | Free (yfinance) |

**Universe**: S&P 500 constituents with ≥ 252 trading days of history at each fold's train-end date. Survivorship bias is controlled by using point-in-time constituent lists (CRSP or approximated via historical index membership).

### 4.2 Feature Construction

**Price-based features** (computed from stock i, window [t-60, t-1]):
```
log_return_d        daily log return
volume_log          log(volume), normalized cross-sectionally
realized_vol_20d    sqrt(sum(r^2, 20d))
price_momentum_60d  cumulative return over window
```

**Factor exposure features** (rolling 60d OLS, updated daily):
```
β_MKT, β_SMB, β_HML, β_RMW, β_CMA, β_WML   (6 loadings)
residual_momentum_20d   mean of past 20d residuals (ε_i,τ)
R²_60d                  fit quality of rolling regression
```

**Regime / cycle signals** (from S&P 500 index, t-1 only):
```
cycle_position      (SPX - MA200d) / MA200d
cycle_intensity     quantile(RV_30d, rolling 252d history)
regime_label        4-state discrete (for FiLM)
```

**Cross-sectional rank features** (normalized within universe at each t):
```
size_rank           market cap percentile
mom_rank            12-1 month return percentile
vol_rank            realized vol percentile
```

Total features D ≈ 20–25 per stock-day.

### 4.3 Prediction Target

```
# Rolling β estimation (no leakage: uses [t-60, t-1] only)
β̂_i,t = OLS(r_i,τ ~ F_τ, τ ∈ [t-60, t-1])

# Factor-adjusted residual for next 5 days
ε̂_i,t+h = r_i,t+h - β̂_i,t · F_t+h    for h = 1,...,5

# Prediction target (scalar)
y_i,t = Σ_{h=1}^{5} ε̂_i,t+h
```

**Parallel target** (ablation): `y_raw_i,t = Σ_{h=1}^{5} r_i,t+h` — raw 5d return, same model, same features. Comparison shows marginal contribution of factor adjustment.

### 4.4 Train/Validation/Test Split

Purged Walk-Forward CV (López de Prado 2018):

```
Total period: 1990–2024 (34 years, ~8,500 trading days)

Walk-forward folds (5 folds):
  Fold 1: Train 1990–2001, Embargo 10d, Val 2002–2004
  Fold 2: Train 1990–2004, Embargo 10d, Val 2005–2007
  Fold 3: Train 1990–2007, Embargo 10d, Val 2008–2010
  Fold 4: Train 1990–2010, Embargo 10d, Val 2011–2013
  Fold 5: Train 1990–2013, Embargo 10d, Val 2014–2016

Final held-out test: 2017–2024 (never touched during HPO/ablation)
```

**Embargo**: 10 trading days (2 × horizon=5d). Prevents leakage from overlapping return windows.

**Universe selection**: At each fold's train-end date, select stocks with ≥ 252d of non-missing data in that fold's train period. No forward-looking survivorship.

### 4.5 Evaluation Metrics

**Prediction quality** (model-only, no portfolio construction):
```
IC          Information Coefficient = Spearman(y_hat, y_true), cross-sectional mean
ICIR        IC / std(IC) — risk-adjusted predictive power
MAE         Mean Absolute Error
CRPS        Continuous Ranked Probability Score (proper scoring rule)
PI_80_cov   Empirical coverage of 80% prediction interval (target: 80%)
```

**Uncertainty quality**:
```
Reliability diagram    predicted quantile vs empirical frequency
Sharpness             mean PI_80 width (narrower = better, given coverage)
ECE                   Expected Calibration Error (evaluation only, not in loss)
```

**Portfolio performance** (funnel applied, reported separately):
```
Sharpe ratio (annualized, DSR-adjusted)
Maximum Drawdown
Calmar ratio
Information Ratio vs equal-weighted benchmark
```

---

## 5. Results

### 5.1 Table 1 — ML Ablation (H2 validation)

| Model | IC | ICIR | CRPS | PI_80_cov | Notes |
|-------|----|------|------|-----------|-------|
| Full model (Cycle-PE + Gate + FiLM + GAN) | — | — | — | — | Proposed |
| Static PE only | — | — | — | — | H2 test |
| Random importance (no attention-guided TCN) | — | — | — | — | Link 2 |
| Simple average fusion (no gate) | — | — | — | — | Link 3 |
| No FiLM | — | — | — | — | Link 4 |
| Transformer only | — | — | — | — | Ablation |
| TCN only | — | — | — | — | Ablation |
| DLinear (same input, same target) | — | — | — | — | External baseline |
| Quantile Regression (same input, same target) | — | — | — | — | GAN necessity |
| Full model, raw return target | — | — | — | — | Factor adj. contribution |

Statistical significance: Diebold-Mariano test vs Full model, Bonferroni-corrected (α/9).

### 5.2 Table 2 — Uncertainty Decomposition (H3 validation)

| Metric | Bull/quiet | Bull/volatile | Bear/quiet | Bear/volatile |
|--------|-----------|--------------|-----------|--------------|
| Mean σ_epistemic | — | — | — | — |
| Mean σ_aleatoric | — | — | — | — |
| PI_80 coverage | — | — | — | — |
| CRPS | — | — | — | — |

H3a: Spearman(σ_epistemic, data_sparsity) — reported with p-value
H3b: ANOVA F-stat for σ_aleatoric across 4 regimes
H3c: DM test: CRPS(GAN) vs CRPS(Quantile Regression)

Reliability diagram: GAN model vs Quantile Regression (Figure 1)

### 5.3 Table 3 — Regime-conditional Predictability (H1 validation)

Panel regression results:
```
y_i,t = β_0 + β_1 · mom_i,t + β_2 · (mom_i,t × regime_t) + γ · controls + u_i,t
```

| Coefficient | Estimate | HAC t-stat | p-value |
|-------------|----------|-----------|---------|
| β_1 (unconditional momentum) | — | — | — |
| β_2 (Bull/quiet × momentum) | — | — | — |
| β_2 (Bull/volatile × momentum) | — | — | — |
| β_2 (Bear/quiet × momentum) | — | — | — |
| β_2 (Bear/volatile × momentum) | — | — | — |

Fama-MacBeth cross-sectional robustness: reported in appendix.

### 5.4 Table 4 — Portfolio Performance

Long-top-decile strategy based on model predictions (PI_low_80 > 0 filter):

| Strategy | Ann. Return | Sharpe (DSR-adj.) | MDD | IR vs EW |
|----------|------------|-------------------|-----|---------|
| Full model | — | — | — | — |
| TFA baseline (Kim 2022 replication) | — | — | — | — |
| Equal-weighted | — | — | — | — |

Note: Portfolio results are reported separately from prediction results to prevent "the funnel did everything" critique.

### 5.5 Sensitivity Analysis

MA {120d, 200d, 252d} × RV {20d, 30d, 60d} grid — IC and CRPS across 9 configurations. If results are consistent across the grid, data snooping concern is mitigated.

---

## 6. Discussion

### 6.1 Why Cycle-aware PE Works (or Doesn't)

Attention heatmap analysis: average attention pattern per regime (Figure 2).
- Expected: Bull/quiet → attention concentrated on recent timesteps
- Expected: Bear/volatile → attention spread over longer history

If this pattern holds, it provides mechanistic evidence for H2 beyond IC improvement alone.

### 6.2 Gate Behavior as Diagnostic Variable

Spearman(gate_t, cycle_position_t) — reported with pseudo-replication correction (aggregated by date, not by stock).

Oracle consistency test: AUC of gate predicting which path (Transformer vs TCN) had lower error.

This is diagnostic, not causal. We do not claim gate "explains" regime — we report whether it correlates.

### 6.3 Limitations

- **β estimation error**: Rolling OLS β̂ contains estimation noise that propagates into the prediction target. Fama-MacBeth robustness check partially addresses this.
- **Survivorship bias**: Point-in-time S&P 500 membership is approximated; full CRSP access would eliminate this.
- **Single market**: Results are from US equities only. Generalization to other markets (Europe, APAC) is left for future work.
- **Backtest realism**: Portfolio results use simplified fill assumptions; actual transaction costs and market impact are not modeled.
- **Multiple testing**: 9 ablation comparisons + HPO trials constitute a large search space. Bonferroni correction and DSR adjustment are applied but cannot fully eliminate the concern.

---

## 7. Conclusion

We propose a regime-conditional hybrid deep learning framework for predicting factor-adjusted equity returns with decomposed uncertainty estimation. The key contributions are: (1) a theoretically grounded prediction target derived from APT time-varying β decomposition; (2) cycle-aware positional encoding that injects market cycle position into the Transformer's temporal representation; and (3) empirical validation that epistemic and aleatoric uncertainty respond to different market conditions.

Pre-registered hypotheses H1, H2, H3 are tested on S&P 500 data (1990–2024) using purged walk-forward cross-validation with a held-out test period (2017–2024).

---

## Appendix A — Data Snooping Defense

Pre-registration checklist (completed before any data analysis):
- [ ] H1, H2, H3 hypotheses and test statistics specified
- [ ] HPO objective function: IC only (Sharpe excluded from HPO loop)
- [ ] Held-out test period (2017–2024) locked — opened once only
- [ ] Sensitivity grid: MA {120d, 200d, 252d} × RV {20d, 30d, 60d}
- [ ] Bonferroni correction: α/9 for 9 ablation comparisons
- [ ] DSR adjustment for final Sharpe ratio reporting

## Appendix B — Implementation Details

- Framework: PyTorch
- Model size: ~3M parameters
- Training: WGAN-GP, Adam (β=(0.5, 0.9)), CosineAnnealingLR
- HPO: Optuna, Purged Walk-Forward CV, objective = IC
- Inference: MC-Dropout (K=100) × GAN sampling (K=100)
- Hardware: Single CPU server (inference), GPU optional for training

## Appendix C — Fama-MacBeth Robustness

Cross-sectional regression at each date t:
```
y_i,t = λ_0,t + λ_1,t · mom_i,t + λ_2,t · (mom_i,t × regime_t) + η_i,t
```
Time-series mean of {λ_k,t} with Newey-West standard errors.

---

## Open Questions / TODO

- [ ] Survivorship bias: obtain point-in-time S&P 500 constituent list (CRSP preferred)
- [ ] β estimation: compare rolling OLS vs Fama-MacBeth β — which gives cleaner residual?
- [ ] Confirm CRPS implementation for scalar target (properscoring library)
- [ ] Decide: include WML (momentum factor) in F_t or keep as separate feature?
- [ ] Sensitivity: test with FF5 only (no WML) as robustness check
- [ ] Future work section: extension to European/APAC markets using same framework
