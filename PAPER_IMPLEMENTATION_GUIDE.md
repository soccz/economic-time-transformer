# AETHER 논문 구현 가이드라인

> 코드 기준: `models/hybrid_model.py`, `utils/config.py`, `aaa/DESIGN_SPEC.md`
> 이 문서는 심사자 수준의 비판을 반영한 구현 + 논문 작성 로드맵이다.

---

## 0. 두 트랙 전략

**원칙: 공유 실험 코어를 한 번만 구현하고, 논문 프레이밍만 분기한다.**

### 0.1 트랙 비교

| | ML/엔지니어링 트랙 | 금융 트랙 |
|---|---|---|
| 제목 | "Attention-Guided Hybrid Encoder with Explainable Gated Fusion for Regime-Aware Time Series Forecasting" | "Regime-Conditioned Hybrid Forecasting with Explainable Gated Fusion for Cryptocurrency Alpha Generation" |
| 핵심 기여 | 아키텍처 참신성 (attention-guided TCN, explainable gate, uncertainty-as-constraint) | 팩터 프리미엄 + 체제 조건화 + cross-sectional ranking |
| Related Work | PatchTST, DLinear, TCN, MoE gating, MC-Dropout | Fama-French, crypto factor models, probabilistic forecasting |
| Experiment 강조 | ablation table, gate distribution, attention heatmap, calibration | regime × metric table, Sharpe/MDD, IC/ICIR, factor premium t-test |
| 도메인 처리 | crypto = 고난이도 비정상 시계열 벤치마크 + 도메인 일반화 1개 추가 | crypto = primary domain |
| 타겟 베뉴 | NeurIPS FinML Workshop → TMLR → IEEE TNNLS | JFDS → Quantitative Finance |
| 제출 순서 | **먼저** (BTC/ETH 7년으로 충분) | **나중** (알트 백필 완료 후) |

### 0.2 ML 트랙 강화: 도메인 일반화 실험

ML 리뷰어의 "왜 crypto만?" 질문 차단용. 도메인 1개 추가.

```
추가 도메인: S&P500 섹터 ETF (11개) or 주요 주가지수 (20개)
같은 파이프라인, 같은 모델, 다른 데이터
regime 정의: VIX 기반 4-state (BTC regime 대신)
```

### 0.3 제출 실행 순서

```
Phase A — 공유 코어 완성 (지금)
  leakage 버그 수정 → BTC/ETH 백필 → ablation 구현 → 지표 전부 계산

Phase B — ML 트랙 제출
  NeurIPS FinML Workshop (빠른 피드백) → 리뷰 반영 후 TMLR

Phase C — 금융 트랙 제출 (알트 백필 완료 후)
  ML 트랙 리뷰 피드백 반영 → JFDS
```

---

## 1. 데이터 파이프라인

### 1.1 백필 실행

현재: BTC/ETH 3.5개월 / 목표: BTC/ETH 2017~, 알트 2021~

```python
collect_market_data("KRW-BTC", days=3285)  # 2017-01-01~
collect_market_data("KRW-ETH", days=3285)
# 알트: 상장일 기준 available한 것만, 강제 백필 없음
```

### 1.2 Unbalanced Panel 처리

```
available_coins(t) = {c | listing_date(c) + 14d ≤ t AND tradeable(c)}
cross-sectional 정규화: available_coins(t) 기준으로만
생존편향 완화: 상장폐지 코인도 available 기간 동안 포함
```

### 1.3 Feature Engineering (33개, 순서 고정)

| 그룹 | 피처 | 논문 명시 필요 사항 |
|---|---|---|
| 기술적 지표 | close, volume, rsi, macd, macdsignal, macdhist, adx, obv | 각 지표의 window 파라미터 명시 |
| 컨텍스트 | market_index_return (idx=8), historical_similarity (idx=9) | similarity 정의 필수 (아래 1.4) |
| 기술적 지표 2 | bb_upper, bb_middle, bb_lower, volume_ma | window 명시 |
| 변동성 | volatility_24h, volatility_7d, volume_volatility | |
| 시장 중립 | alpha, beta | rolling closed-form, W∈{72,168,336}h, t-1까지만 사용 |
| 상대 지표 | price_position, volume_ratio, return_skew_24h, cross_corr_btc | |
| FF5 팩터 | factor_size, factor_mom, factor_vol, factor_liq | cross-sectional rank 정규화 |
| 체제 | btc_regime (0~3), btc_regime_rv, btc_ma_distance | 4-state |
| 상호작용 | factor_mom_x_bull, factor_liq_x_bear | FF5 × regime |

**예측 타겟**: `residual_return = coin_return - beta × btc_return`

**이상치 처리**:
```python
returns = returns.clip(lower=-5*rolling_std, upper=5*rolling_std)
beta = beta.clip(-3, 3)
```

### 1.4 historical_similarity 정의 (논문 필수 명시)

심사자가 반드시 묻는 항목. 아래 세 가지를 명확히 정의해야 한다:

```
(i)  표현 공간: 원시 피처 공간 (정규화된 168h × D 윈도우)
(ii) 거리 함수: cosine similarity or DTW (선택 후 고정)
(iii) 비교 대상: 메모리 뱅크 (과거 윈도우 슬라이딩 샘플)
     → 현재 윈도우 vs 메모리 뱅크 Top-K 유사도의 평균 or 최대값
(iv) 업데이트 방식: 추론 시 고정 메모리 뱅크 (leakage 방지)
```

### 1.5 Beta 추정 시간 정합성 (논문 필수 명시)

```python
# t 시점 의사결정이면 t-1까지의 데이터로만 beta 계산
# rolling window W는 [t-W, t-1] 구간
beta_t = cov(coin_ret[t-W:t-1], btc_ret[t-W:t-1]) / var(btc_ret[t-W:t-1])
# 미래 정보 절대 불가
```

### 1.6 BTC 4-state 체제 정의

```python
rv = btc_ret.rolling(168).std() * (8760 ** 0.5)
rv_low  = rv < rv.rolling(4320).quantile(0.4)
rv_high = rv > rv.rolling(4320).quantile(0.6)

ma200 = btc_price.rolling(200).mean()
delta = 0.02  # hysteresis band
trend_up   = btc_price > ma200 * (1 + delta)
trend_down = btc_price < ma200 * (1 - delta)
# 중간 구간 → 직전 상태 hold
# 0: Bull_quiet, 1: Bull_volatile, 2: Bear_quiet, 3: Bear_volatile
```

---

## 2. 모델 아키텍처

### 2.1 현재 구현 상태

`models/hybrid_model.py` → `HybridModel`:
- Transformer: `TransformerEncoder` (causal mask=False, 전체 시퀀스 참조)
- CNN: TCN-style 1D (dilation 1, 2, 4)
- Fusion: `ExplainableGatedFusion` (prototype bank 16개, cosine similarity)
- Regime: `FiLMRegimeConditioning` (4-state embedding)
- Decoder: `GANDecoder` (WGAN-GP, 3-step output)

### 2.2 Contextual Positional Encoding

```
PE_contextual(x, c) = x + PE_static + (c @ W_ctx)
```

- c: market_index_return + historical_similarity (2-dim)
- 효과: 시간 위치 + 컨텍스트 바이어스를 각 시점 토큰에 주입

**심사 방어 필요**: "피처에 컨텍스트 2개를 concat한 것"보다 왜 좋은지 실증 필요.
→ ablation: contextual PE vs 동일 파라미터 수의 concat 비교

**causal mask 결정**: 현재 False (전체 시퀀스 참조).
금융 시계열에서 causal mask=True가 기본값이어야 심사 통과가 편하다.
→ 논문에서 "예측 시점 t에서 t까지의 정보만 사용"임을 명시하거나, causal mask=True로 전환 검토.

### 2.3 Attention-guided TCN

```
importance = softmax(attn_weights[-1][:, -1, :])   # (B, 168)
guided_input = raw_input × importance.unsqueeze(-1) # (B, 168, D)
cnn_out = TCN(guided_input)
```

**심사 리스크**: "attention weights = importance"는 학계에서 논쟁이 많다.

방어 전략 (셋 중 하나 선택):

| 전략 | 내용 | 난이도 |
|---|---|---|
| A (채택 권장) | "설명"이 아니라 "학습 유도용 guidance prior"로 명명 + importance detach | 낮음 |
| B | attention rollout/flow로 누적 중요도 계산 | 중간 |
| C | 설명은 Integrated Gradients로 별도 제공, attention은 guidance만 | 높음 |

**수식 주의**: `attn_weights[-1][:, -1, :]`이 이미 post-softmax라면 softmax를 한 번 더 하면 분포가 과도하게 뾰족해진다.
→ head 평균 후 정규화 한 번으로 충분. pre/post-softmax 여부를 논문에 명시.

**재현성 필수 명시**:
- head aggregation 방식 (mean? max? last head?)
- importance의 detach 여부
- TCN: kernel_size, dilation, residual block 유무, normalization

### 2.4 Explainable Gated Fusion

```
sim_t = cosine_sim(transformer_out, prototype_bank)  # (B, 16)
sim_c = cosine_sim(cnn_proj(cnn_out), prototype_bank) # (B, 16)
gate = sigmoid(W · [transformer_out; cnn_proj(cnn_out); agg(sim_t); agg(sim_c)])
fused = gate ⊙ transformer_out + (1 - gate) ⊙ cnn_proj(cnn_out)
```

gate ∈ (0,1)^d_model: 차원별로 global/local 비율이 다름 → MoE 성격

**심사자 결정타 질문**: "프로토타입이 왜 explainable인가?"

latent prototype은 그 자체로 인간이 이해 가능한 패턴이 아니다.
explainability를 강하게 주장하려면 아래 중 하나 필요:

```
방법 1 (권장): 각 prototype에 대해 가장 가까운 실제 시계열 윈도우 Top-K 제시
  → "이 프로토타입은 Bear_volatile 국면의 이런 패턴에서 활성화"

방법 2: 성공/실패 구간 contrastive loss로 프로토타입 의미를 학습 제약으로 강제
```

**Gate Regularization 부호 검증 필요**:
```
L_gate = relu(0.3 - mean(gate)) + relu(mean(gate) - 0.7)  # range penalty (+)
       - 0.1 × std(mean(gate, dim=1))                       # diversity reward (-)
       + 0.01 × mean(proto_sim_matrix[off-diagonal])        # proto diversity penalty (+)
```
range penalty와 diversity reward가 상충할 수 있다. 학습 커브 + gate 분포 그림으로 설득 필요.

### 2.5 FiLM Regime Conditioning

```
h = h ⊙ scale(embed(regime)) + shift(embed(regime))
```

**재현성 필수 명시**:
- W_scale, W_shift: 선형 레이어인지 MLP인지
- scale을 (1 + δ) 형태로 안정화하는지 여부

**논문 강점**: regime이 4-state로 명시되어 있어 "상태별 성능/캘리브레이션/트레이드 특성" 분해 실험이 자연스럽게 연결된다.

### 2.6 GAN Decoder (WGAN-GP)

**장점**: point estimate가 아닌 조건부 분포(3-step joint) 생성 → 불확실성 필터와 자연스럽게 연결.

**심사 약점 3가지**:

1. 출력이 3차원(3-step)으로 매우 저차원 → GAN이 "과한 무기"로 보일 수 있음
2. MSE(재구성) + WGAN이 같이 있으면 분포 매칭과 점오차 최소화가 충돌 가능
3. ECE를 회귀에 어떻게 정의했는지 명시 필요

**방어 전략**:

```
ablation에 아래 비교 대상 포함:
  - Quantile regression (q10/q50/q90)  ← 가장 강력한 비교 대상
  - MDN (Mixture Density Network)
  - 결정론적 3-step head (GAN 제거)

"왜 MSE + WGAN을 같이 쓰는가" → 모드 붕괴 방지 + 중심 추정 안정화로 실험 설득
```

**회귀 ECE 정의 (논문 필수)**:
```
PI 커버리지 기반 ECE:
  bins = [0.1, 0.2, ..., 0.9] (nominal coverage levels)
  ECE = mean(|empirical_coverage(bin) - nominal_coverage(bin)|)

또는 Quantile Calibration Error (QCE):
  QCE = mean_q(|fraction(y ≤ q_hat) - q|)
```

### 2.7 MC-Dropout 불확실성 추정

**역할 분리 (논문 서술 강화)**:
```
GAN noise z  → aleatoric uncertainty (데이터 본질적 불확실성)
MC-Dropout   → epistemic uncertainty (모델 불확실성)
```

**재현성 필수 명시**:
```
샘플링 조합 명시:
  옵션 A: z 고정 + dropout on/off → epistemic만 측정
  옵션 B: z resample + dropout on → 전체 분산 (과대평가 위험)
  옵션 C: z resample + dropout off → aleatoric만 측정

권장: 옵션 A로 epistemic 추정, 옵션 C로 aleatoric 추정, 합산으로 total
```

---

## 3. 학습 / 평가 프로토콜

### 3.1 Purged Walk-Forward CV

```
전체 기간: 2017-01 ~ 2024-12
fold:
  train_size : 8,760h (1년)
  embargo    : 6h  ← horizon=3h 대비 충분한지 근거 필요 (López de Prado 인용)
  val_size   : 2,160h (90일)
  n_folds    : 5~7 (non-overlapping)

purge 구간: 라벨 horizon(3h)을 반영해야 함
  → train 마지막 3h는 val 라벨과 겹치므로 purge
```

### 3.2 손실 함수

```
L_total = L_recon + λ_gp·L_gp + λ_ece·L_ece + λ_dir·L_direction + L_gate_reg

L_recon     : MSE(pred_residual, true_residual)
L_gp        : WGAN-GP gradient penalty
L_ece       : Quantile Calibration Error (회귀 ECE, 위 2.6 정의)
L_direction : BCE(sign(pred), sign(true))
L_gate_reg  : gate range + diversity + prototype diversity
```

### 3.3 Optuna HPO

| 하이퍼파라미터 | 범위 | 비고 |
|---|---|---|
| beta_window W | {72, 168, 336} | |
| d_model | {64, 128, 256} | |
| n_heads | {4, 8} | |
| n_layers | {2, 3, 4} | |
| dropout_p | [0.05, 0.3] | |
| lambda_recon | [1, 100] | |
| lambda_gp | [1, 10] | |

목적함수: `val_sharpe` (또는 val IC — 트랙별 선택)

---

## 4. Baseline 및 Ablation

### 4.1 Baseline 목록

| Baseline | 트랙 | 비고 |
|---|---|---|
| DLinear cross-sectional | 양쪽 필수 | Zeng et al. 2022, multivariate ≠ cross-sectional |
| Quantile regression (q10/q50/q90) | 양쪽 필수 | GAN 필요성 방어용 |
| Transformer-only | 양쪽 필수 | ablation: no_cnn |
| CNN/TCN-only | 양쪽 필수 | ablation: no_transformer |
| No gate (simple avg fusion) | 양쪽 필수 | ablation: no_gate |
| No FiLM | 양쪽 필수 | ablation: no_film |
| Contextual PE → concat | ML 트랙 | PE 방어용 |
| No prototype bank | ML 트랙 | ablation: no_prototype |
| VAE variant | ML 트랙 | GAN vs VAE 비교 |
| Deterministic 3-step head | 양쪽 | GAN 제거 ablation |
| Buy-and-hold BTC | 금융 트랙 | 시장 수익률 기준선 |

### 4.2 Ablation Study (이 모델은 ablation이 논문 본체)

```
Full AETHER
  vs. - contextual PE       (PE → concat, 동일 파라미터 수)
  vs. - attention_guided    (random importance, detach 비교)
  vs. - gate                (simple average fusion)
  vs. - FiLM                (regime을 단순 피처로만)
  vs. - prototype_bank      (gate without similarity)
  vs. - GAN                 (deterministic 3-step head)
  vs. - MC-Dropout          (불확실성 없는 퍼널)
  vs. - funnel              (모델 순수 성능, 룰 필터 제거)
  vs. Transformer-only
  vs. TCN-only
  vs. Quantile regression
  vs. DLinear (cross-sectional)
  vs. VAE variant
```

모든 variant를 동일한 Purged Walk-Forward CV로 평가.

---

## 5. 평가 지표

### 5.1 예측 지표 (양쪽 공통)

| 지표 | 정의 | 비고 |
|---|---|---|
| MAE | mean(\|pred - true\|) | residual return 기준 |
| Directional Accuracy | mean(sign(pred) == sign(true)) | |
| IC | Spearman(pred_rank, true_rank) | cross-sectional |
| ICIR | IC.mean() / IC.std() | 안정성 |

### 5.2 확률 예측 지표 (양쪽 공통)

| 지표 | 정의 | 목표 |
|---|---|---|
| QCE | Quantile Calibration Error | < 0.05 |
| PICP | PI_80 Coverage | ≈ 0.80 |
| PINAW | PI_80 Width (normalized) | 작을수록 좋음 |
| Spearman(uncertainty, \|error\|) | 불확실성이 오차를 예측하는가 | > 0 |

### 5.3 트레이딩 지표 (양쪽 공통)

| 지표 | 정의 | 비고 |
|---|---|---|
| Sharpe Ratio | annualized | 체제별 분해 포함 |
| Sortino Ratio | downside deviation 기준 | |
| MDD | Maximum Drawdown | |
| Calmar Ratio | annualized_return / MDD | |
| Turnover | 포지션 변경 빈도 | 비용 포함 PnL과 함께 |
| Net Alpha | gross - fee(0.05%) - slippage(0.03%) | |

### 5.4 퍼널 단계별 기여 분해 (금융 트랙 필수)

심사자가 "모델 성능"과 "룰 기반 필터 성능"을 분리하지 못하면 공격 포인트가 생긴다.

```
필터 i 추가 전/후 Sharpe, turnover, hit-rate, drawdown 변화를 테이블로 보고
예:
  모델 raw output → Sharpe X
  + net_alpha > 0 → Sharpe Y
  + PI_low_80 > 0 → Sharpe Z
  + liquidity filter → Sharpe W
  ...
```

### 5.5 체제별 성능 분해 (금융 트랙 핵심)

```
Table: regime × metric (Sharpe, IC, Win Rate, MDD)
Figure: gate_value distribution per regime (violin plot)
  가설: Bull_quiet → gate ≈ 1 (Transformer 지배)
        Bear_volatile → gate ≈ 0 (TCN 지배)
```

### 5.6 Statistical Significance

```python
# 체제별 팩터 프리미엄 t-test
t_stat, p_val = stats.ttest_1samp(regime_returns[regime], 0)

# Bootstrap IC CI
ic_bootstrap = [spearman(pred[s], true[s]) for s in bootstrap_samples]
ci_95 = np.percentile(ic_bootstrap, [2.5, 97.5])
```

---

## 6. 설명 가능성 분석

ML 트랙: "interpretability mechanism" / 금융 트랙: "투자 근거 설명"으로 프레이밍.

### 6.1 Prototype Bank 시각화 (explainability 주장의 핵심 근거)

```python
# 각 prototype에 대해 가장 가까운 실제 시계열 윈도우 Top-K 제시
# "이 프로토타입은 Bear_volatile 국면의 이런 패턴에서 활성화"
for proto_id in range(16):
    top_k_windows = find_nearest_windows(prototype_bank[proto_id], memory_bank, k=5)
    # Figure: proto_id별 Top-5 실제 윈도우 오버레이
```

### 6.2 Gate Value × Regime

```python
gate_by_regime = {r: gate_values[regime_labels == r] for r in range(4)}
# Figure: violin plot per regime
# 가설 검증: Bull_quiet gate > Bear_volatile gate
```

### 6.3 Attention Top-3 패턴

```python
# 체제별 attention 집중 위치 분포
# Bull: 최근 시점 집중 / Bear: 원거리 시점 집중 가설 검증
# Figure: heatmap (regime × timestep position)
```

---

## 7. 재현성 필수 명시 항목

논문으로 제출하려면 아래가 빠지면 재현 불가능하다.

| 항목 | 현재 상태 | 논문 명시 내용 |
|---|---|---|
| 기술적 지표 window | config에 있음 | RSI=14, MACD=(12,26,9), ADX=14 등 전부 |
| historical_similarity 정의 | 불명확 | 표현 공간, 거리 함수, 메모리 뱅크 크기, 업데이트 방식 |
| beta 추정 시간 정합성 | 코드 확인 필요 | t-1까지만 사용 명시 |
| Transformer: FFN dim, head 수, 레이어 수 | config에 있음 | d_model=128, n_heads=8, n_layers=3 |
| causal mask 기본값 | False | 논문에서 이유 명시 필요 |
| attention head aggregation | 코드 확인 필요 | mean? last head? |
| importance detach 여부 | 코드 확인 필요 | stop-gradient 여부 |
| TCN: kernel_size, dilation, residual | 코드에 있음 | kernel=3, dilation=1/2/4 |
| GAN: noise_dim, critic:gen ratio, λ | config에 있음 | noise_dim=32, critic_iters=7 |
| ECE 정의 수식 | 불명확 | QCE 수식으로 재정의 |
| 퍼널 threshold 결정 방식 | config에 있음 | 고정값인지 CV 튜닝인지 |
| purge/embargo와 horizon 매칭 | 6h embargo | horizon=3h 대비 근거 명시 |
| MC-Dropout 샘플링 조합 | 코드 확인 필요 | z 고정/resample × dropout on/off |
| Optuna pruning 전략 | 코드 확인 필요 | MedianPruner? 없음? |

---

## 8. 알려진 버그 (제출 전 필수 수정)

| 우선순위 | 파일 | 문제 | 심각도 |
|---|---|---|---|
| 1 | `data/preprocessor.py:538` | fit_transform 전체 데이터 → train만 fit | 🔴 leakage |
| 2 | `inference/predictor.py` | crypto_factors_df 미전달 → factor 전부 0 | 🔴 |
| 3 | `data/preprocessor.py` | 팩터/패턴 전역 캐시 → backtest 미래 오염 | 🔴 leakage |
| 4 | `models/hybrid_model.py` | gate reg loss 부호 충돌 확인 | 🟠 |

1~3번은 backtest 결과를 무효화하는 leakage. 양쪽 트랙 모두 제출 전 필수.

---

## 9. 구현 체크리스트

```
Phase A — 공유 코어 (지금 당장)
  [ ] leakage 버그 1~3 수정
  [ ] BTC/ETH 백필 (2017-01-01~)
  [ ] historical_similarity 정의 확정 및 코드 명시
  [ ] beta 추정 시간 정합성 검증 (t-1까지만)
  [ ] causal mask 결정 (True 전환 or 논문에서 이유 명시)
  [ ] attention head aggregation + detach 방식 확정
  [ ] ECE → QCE로 재정의 및 코드 반영
  [ ] MC-Dropout 샘플링 조합 확정 (z 고정 + dropout on)
  [ ] ablation flag 구현 (no_cnn, no_gate, no_film, no_prototype, no_gan, no_funnel)
  [ ] Quantile regression baseline 구현
  [ ] DLinear cross-sectional baseline 구현
  [ ] 평가 지표 전부 계산 (IC, ICIR, Sharpe, Sortino, MDD, QCE, PICP, gate distribution)
  [ ] 퍼널 단계별 기여 분해 테이블
  [ ] Purged Walk-Forward CV purge 구간 검증 (horizon=3h 반영)

Phase B — ML 트랙 추가
  [ ] Prototype Top-K 실제 윈도우 시각화
  [ ] VAE variant 구현 (ablation용)
  [ ] 도메인 일반화 실험 (S&P500 섹터 ETF, VIX 기반 regime)
  [ ] gate × regime violin plot
  [ ] attention_top3 패턴 분석 (체제별 집중 위치)
  [ ] contextual PE vs concat ablation
  [ ] NeurIPS FinML Workshop 제출

Phase C — 금융 트랙 추가 (알트 백필 완료 후)
  [ ] 알트 백필 (2021-01-01~)
  [ ] regime × metric table
  [ ] 팩터 프리미엄 t-test / bootstrap IC CI
  [ ] ML 트랙 리뷰 피드백 반영
  [ ] JFDS 제출
```

---

## 10. 논문 구조 (트랙별 분기)

### ML/엔지니어링 트랙

```
1. Introduction
   단일 inductive bias의 한계 → hybrid + explainable gating + uncertainty-as-constraint

2. Related Work
   PatchTST, DLinear, TCN/WaveNet, MoE gating, MC-Dropout, prototype learning

3. Method
   3.1 Problem Formulation (non-stationary time series, residual return target)
   3.2 Contextual Positional Encoding
   3.3 Hybrid Encoder: Transformer + Attention-guided TCN
   3.4 Explainable Gated Fusion (prototype similarity)
   3.5 FiLM Regime Conditioning
   3.6 GAN Decoder + MC-Dropout (aleatoric/epistemic 분리)
   3.7 Uncertainty-as-constraint Decision Funnel

4. Experiments
   4.1 Datasets (crypto + 도메인 일반화 1개)
   4.2 Baselines + Ablation Table (이게 논문 본체)
   4.3 Explainability Analysis (prototype Top-K, gate distribution, attention heatmap)
   4.4 Uncertainty Calibration (QCE, PICP, Spearman)
   4.5 Domain Generalization

5. Conclusion + Limitations
```

### 금융 트랙

```
1. Introduction
   팩터 모델 + 체제 조건화 + uncertainty-as-constraint → 실행 가능한 의사결정

2. Related Work
   Fama-French, crypto factor models, regime-switching models, probabilistic forecasting

3. Method
   3.1 Problem Formulation (residual return, cross-sectional ranking, dynamic universe)
   3.2 Feature Engineering (33 features, 4-state regime, FF5 × regime interaction)
   3.3 Hybrid Encoder + Explainable Gated Fusion
   3.4 Recommendation Funnel (8-stage, PI_low_80 hard gate, net-alpha filter)

4. Experiments
   4.1 Data (2017-2024, unbalanced panel, survivorship bias handling)
   4.2 Baselines + Ablation
   4.3 Regime Analysis (regime × metric table, factor premium t-test)
   4.4 Funnel Contribution Decomposition (단계별 Sharpe 기여)
   4.5 Explainability (gate × regime, prototype match)

5. Conclusion + Limitations
   백테스트 현실성, 알트 데이터 길이, GAN 해석 한계
```

---

## 참고 문헌

- Fama & French (1993): cross-sectional factor model, dynamic universe
- Zeng et al. (2022) DLinear: linear baseline
- López de Prado (2018): Purged Walk-Forward CV, embargo
- Vaswani et al. (2017): Transformer attention
- Perez et al. (2018): FiLM conditioning
- Arjovsky et al. (2017): WGAN-GP
- Gal & Ghahramani (2016): MC-Dropout uncertainty
- Jain & Wallace (2019): "Attention is not Explanation" (attention 논쟁 선제 인용)
- Wiegreffe & Pinter (2019): "Attention is not not Explanation" (반론 인용)
