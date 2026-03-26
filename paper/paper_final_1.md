# 팩터 조정 잔차 수익률의 변동성 체제 조건부 예측:
# Cycle-aware Positional Encoding과 Hybrid Transformer–TCN–CVAE

---

## 초록

본 연구는 Fama–French 3팩터(FF3)로 조정한 잔차 수익률에서 단면 모멘텀 프리미엄(λ₁,t)이 시장 상태(state)에 따라 어떻게 변하는지, 그리고 해당 state 정보를 시퀀스 모델에 명시적으로 주입할 때 예측 및 불확실성 추정이 개선되는지를 검증한다.

예비 분석에서 S&P 500 200일 이동평균 기반 추세 신호(cycle_position)는 λ₁,t를 유의하게 설명하지 못했다(25 Size–B/M: p≈0.873, 49 Industry: p≈0.241). 반면 변동성 강도(cycle_intensity)로 구분한 체제별 비교에서 λ₁,t는 Bear/quiet에서 최대, Bear/volatile에서 급락(일부 반전)하는 일관된 패턴이 관찰되었으며, 이는 모멘텀 크래시 문헌과 정합적이다.

이에 따라 본 연구는 세 층위를 결합한다:
1. **H1-Vol (경제학)**: 변동성-비대칭 상태가 잔차 모멘텀 프리미엄을 조절한다 — 2-step Fama–MacBeth + stationary bootstrap으로 검정
2. **H2 (표현학습)**: cycle_intensity를 Cycle-aware Positional Encoding(Cycle-PE)으로 Transformer–TCN 인코더에 주입하면 표본 외 IC/CRPS가 개선된다 — Static PE / Concat-A 대비 Diebold–Mariano 검정
3. **H3 (확률예측)**: 예측 불확실성(분포 폭·꼬리·캘리브레이션)이 state에 따라 체계적으로 달라진다 — CVAE 디코더 + 체제별 CRPS/PI-80 coverage

**키워드**: FF3 residual, volatility regime, momentum crash, Cycle-PE, Transformer–TCN, gated fusion, CVAE, probabilistic forecasting

---

## 1. 서론

자산 수익률은 공통 요인(시장·스타일)과 개별 고유 성분으로 분해된다. FF3는 초과수익을 MKT, SMB, HML의 선형 결합으로 설명하고 남는 부분을 잔차로 정의한다. 본 연구의 초점은 "FF3가 맞냐/틀리냐"가 아니라, **FF3로 설명되지 않는 잔차 성분에서 예측 가능한 구조가 존재하며 그 구조가 시장 상태에 따라 달라지는가**이다.

기존 ML 기반 자산가격 연구(Gu et al. 2020 등)는 원시 수익률 예측에서 비선형 모델의 우위를 보였지만, (1) state를 모델 구조에 명시적으로 조건화하는 설계가 약하거나, (2) 예측을 분포로 다루지 않아 tail risk의 경제적 해석이 제한된다.

모멘텀 전략은 평균적으로 강하지만 특정 "공포/고변동성 상태"에서 급격한 손실(모멘텀 크래시)을 보이며, 이 크래시는 부분적으로 예측 가능하다(Daniel & Moskowitz 2016). 이는 모멘텀 프리미엄의 상태 의존성을 탐구할 경제적 동기를 제공한다.

본 논문의 기여:
- **경제학적 기여**: λ₁,t의 변동성-비대칭 조절을 2-step FMB + stationary bootstrap으로 확증
- **구조적 기여**: state를 PE 공간에 직접 주입하는 Cycle-PE 설계 및 Concat-A 대비 ablation
- **확률예측 기여**: CVAE로 체제별 조건부 분포를 생성하고 CRPS/PI-80으로 평가

---

## 2. 이론적 배경 및 가설

### 2.1 FF3 분해와 잔차 타겟

초과수익 r_{i,t}에 대해:

```
r_{i,t} = α_i + β_MKT,i · MKT_t + β_SMB,i · SMB_t + β_HML,i · HML_t + ε_{i,t}
```

FF3 팩터(일별 MKT, SMB, HML, RF)는 Ken French Data Library에서 제공된다. 팩터는 "모든 자산이 공유하는 공통 시계열"이고 β는 "자산별 노출도"다.

**예측 타겟**: h-일 누적 잔차 (h=5, 권장)

```
y_{i,t} = Σ_{h=1}^{5} ε_{i,t+h}
```

미래 구간 타겟을 사용하는 이유: 동일일자 설명이 아닌 예측가능성을 검증하기 위함. overlapping horizon(5일)으로 인해 추론은 stationary bootstrap 기반으로 수행한다.

**generated regressor 리스크**: 잔차 타겟은 관측값이 아니라 rolling OLS 추정으로 생성되므로 measurement error 공격을 받는다. 방어 전략은 §4.1 참조.

### 2.2 시장 상태(state) 정의

S&P 500 기반 두 연속 신호:

**추세 위치 (cycle_position)**:
```
cycle_position_t = (SPX_t - MA200_t) / MA200_t
```
200일 이동평균 대비 현재 가격의 상대적 위치.

**변동성 강도 (cycle_intensity)**:
```
cycle_intensity_t = QuantileRank(RV30_t ; 252)
```
30일 실현변동성의 최근 252일 분위수. [0,1] 범위로 정규화되어 비교 가능성이 높다.

**이산 체제 (2×2)**: 모델 조건화(FiLM) 및 체제별 비교를 위해 사용.

| 체제 | 정의 |
|------|------|
| Bear/quiet (0) | position < 0, intensity ≤ 0.5 |
| Bear/volatile (1) | position < 0, intensity > 0.5 |
| Bull/quiet (2) | position ≥ 0, intensity ≤ 0.5 |
| Bull/volatile (3) | position ≥ 0, intensity > 0.5 |

경제학적 검정(H1)은 연속 신호를 주 분석으로 두고, 이산 체제는 보조 비교에 사용한다.

### 2.3 H1: "추세 증폭"은 기각, "변동성-비대칭 조절"이 핵심

**예비 결과 요약**:

| 단면 | cycle_position 단독 p값 | 체제별 패턴 |
|------|------------------------|------------|
| 25 Size–B/M | p≈0.873 | Bear/quiet 최대, Bear/volatile 급락 |
| 49 Industry | p≈0.241 (부호 양수) | 동일 방향 |

이는 "Bull에서 모멘텀 증폭"이 아니라 "고변동성/공포 상태에서 모멘텀 크래시" 계열 가설과 정합적이다.

**H1-Pos (등록 가설, null result로 명시)**:
cycle_position이 λ₁,t를 선형 조절한다 → 예비 분석에서 지지되지 않음.

**H1-Vol (핵심 가설)**:
cycle_intensity가 λ₁,t를 조절하며, 특히 Bear × HighVol에서 λ₁,t가 급락/반전한다.

검정 가능한 형태:
```
λ_{1,t} = a + b·intensity_{t-1} + c·position_{t-1}
          + d·1(position_{t-1}<0)·intensity_{t-1} + u_t
```
**핵심 예측**: d < 0 (약세에서 변동성 증가가 모멘텀 프리미엄을 훼손)

선형이 약할 경우 intensity² 또는 상위 분위 더미(꼬리)로 비선형 확장 검정.

**주의**: 현재 체제 평균표 패턴은 탐색적 발견이다. "post-hoc binning" 공격을 막으려면 위 회귀식을 사전 고정하고 bootstrap/HAC로 재검정해야 한다(§8-A 참조).

### 2.4 H2: Cycle-PE의 역할 — "거리 메트릭 수정"

표준 PE는 토큰 임베딩에 순서 좌표(ordinal time)를 더한다. Cycle-PE는 여기에 시장 상태 좌표(state-space)를 추가한다:

```
x'_t = x_t + PE_time(t) + PE_state(s_t)
```

self-attention 스코어:
```
score(t,τ) ∝ (W_Q x'_t)ᵀ (W_K x'_τ)
```

state를 PE로 주입하면 attention 유사도 계산 자체에 state가 직접 개입한다. 반면 state를 입력 채널로 concat(Concat-A)하면 모델이 학습을 통해 state를 반영할 수 있지만, "시간 좌표계 자체가 바뀐다"는 해석은 상대적으로 약하다. 따라서 Cycle-PE vs Concat-A는 필수 ablation이다.

**H2**: Cycle-PE(intensity 주입)가 Static PE 또는 Concat-A 대비 표본 외 IC/CRPS를 개선한다.

검정: Diebold–Mariano(시계열 의존 반영) + fold-wise 일관성 보고.

구현된 variant (`models/transformer_encoder.py`, `CyclePE`):
- `static`: 표준 sinusoidal PE (baseline)
- `concat_a`: state를 입력 채널에 concat (Ablation A)
- `cycle_pe`: intensity를 PE 공간에 직접 주입 (H2 main)
- `cycle_pe_full`: intensity + position 모두 주입 (appendix)

### 2.5 H3: 분포 예측과 state-dependent uncertainty

CVAE를 "있으면 멋있다"가 아니라 "필수다"로 만들려면 H3가 경제적으로 서야 한다.

**H3**: 예측 불확실성(분포의 폭·꼬리·캘리브레이션)은 state에 따라 체계적으로 달라진다. 특히 high-intensity(고변동성)에서 예측분포가 넓어지고 tail risk가 증가한다.

이는 모멘텀 크래시/변동성-조건부 성과 문헌("high vol에서 리스크/왜도 문제가 커진다")과 정합적이다. H3가 성립하면 CVAE는 단순 점 예측 모델로 대체 불가능한 구성요소가 된다.

**논문 인과 사슬**:
```
H1-Vol (경제학)
  → 변동성-비대칭 상태가 잔차 모멘텀 프리미엄을 조절한다
      ↓
H2 (ML)
  → intensity state를 Cycle-PE로 주입하면 예측이 개선된다
      ↓
H3 (분포/불확실성)
  → 동일 state가 예측분포(폭·꼬리·캘리브레이션)에도 반영된다
  → CVAE가 필요한 이유
```

---

## 3. 데이터

### 3.1 최소 재현 가능 데이터 (포트폴리오 기반)

| 데이터 | 출처 | 용도 |
|--------|------|------|
| 25 Size–B/M daily 포트폴리오 수익률 | Ken French Data Library | H1 주 단면 |
| 49 Industry daily 포트폴리오 수익률 | Ken French Data Library | H1 robustness |
| FF3 daily factors + RF | Ken French Data Library | 잔차 생성 |
| Momentum factor (WML) | Ken French Data Library | H1 다중공선성 통제 |
| S&P 500 (^GSPC) | Yahoo Finance | state 계산 |
| VIX | FRED | intensity 대체 검정 |

표본 기간: 1990–2024 (H1), 2000–2024 (H2/H3 walk-forward).

Ken French Data Library와 VIX(FRED)는 공개 데이터로 재현성이 높다.

### 3.2 확장 데이터 (권장, 임팩트 상승)

CRSP point-in-time 유니버스 (생존편향 방지). 포트폴리오 기반 결과와 정성적 일관성을 확인하는 robustness 역할도 겸한다.

---

## 4. 방법론

### 4.1 잔차 생성: rolling β 추정과 generated-target 방어

rolling OLS로 β를 추정해 잔차를 생성한다:
- 기본 윈도우: 60일
- robustness: 120일, ridge β

**generated regressor 방어 (심사 핵심, 최소 2개 필수)**:

| 방어 전략 | 구현 |
|-----------|------|
| β-window 민감도 | 60 vs 120일 결과 비교 |
| ridge/shrinkage β | OLS 아티팩트 방어 |
| 포트폴리오 vs 종목 일관성 | 정성적 결론 일치 여부 |
| raw return 부록 | H1-Vol이 잔차 없이도 유사 패턴인지 |

`|mom|`과 `se(β̂_MKT)` 상관이 유의하게 나온 경우(예비 진단 결과), IV 접근(mom_{t-2} 도구변수) 또는 위 방어 전략 중 2개 이상을 반드시 보고해야 한다.

### 4.2 H1 추정: 2-step Fama–MacBeth + robust inference

**Step 1 (단면 회귀, 매일 t)**:
```
y_{i,t} = λ_{0,t} + λ_{1,t}·mom_{i,t} + γᵀ·controls_{i,t} + η_{i,t}
```
- mom_{i,t}: 12-1개월 누적 수익률의 단면 rank ([-0.5, 0.5] 정규화)
- controls: size, B/M (포트폴리오 기반에서는 포트폴리오 특성)
- 결과: λ₁,t 시계열

**Step 2 (시계열 회귀)**:
```
λ_{1,t} = a + b·intensity_{t-1} + c·position_{t-1}
          + d·1(position_{t-1}<0)·intensity_{t-1} + u_t
```
- 추론: HAC (maxlags=10) 또는 stationary bootstrap (overlapping horizon 대응)
- 비선형 확장: intensity² 추가 (Step 2-E)

**λ₁,t vs WML_t 상관 진단**: Spearman ρ 보고 (다중공선성 확인).

구현: `aaa/paper_test/h1_test.py` (Step 2-A~E 포함)

### 4.3 Cycle-aware Positional Encoding

**최소 버전 (H2 main, `cycle_pe`)**:
```
x'_t = x_t + PE_sin/cos(t) + W_intensity · intensity_t
```
- `W_intensity`: Linear(1 → d_model), 학습 파라미터
- intensity_t: context[:, :, 1] (historical_similarity 채널 사용)

**확장 버전 (appendix, `cycle_pe_full`)**:
```
x'_t = x_t + PE_sin/cos(t) + W_intensity · intensity_t + W_position · position_t
```

**Concat-A (ablation, `concat_a`)**:
```
x̃_t = [x_t ; state_t]  →  Linear(input_dim + context_dim → d_model)
```
state가 입력 채널로 들어가므로 PE 공간에는 개입하지 않는다.

구현: `models/transformer_encoder.py`, `CyclePE` 클래스

### 4.4 인코더: Transformer + attention-guided TCN + gated fusion

```
Input (T × D)
  ├─ Transformer encoder (global: regime-level dependencies)
  │    └─ attention_weights[-1] → (B, T) importance row
  ├─ Attention-guided TCN (local: shape primitives)
  │    └─ input weighted by softmax(attention_importance)
  └─ Explainable Gated Fusion
       └─ prototype similarity → gate → fused context
            └─ FiLM conditioning (4-state BTC regime)
```

**Gate 해석 주의**: gate는 "설명"이 아니라 진단 변수(diagnostic)로 취급한다. "attention is not explanation" 논쟁을 인지하고, oracle-consistency sanity check를 함께 보고한다.

구현: `models/hybrid_model.py`, `ExplainableGatedFusion`

### 4.5 디코더: CVAE 기반 조건부 분포 예측

**학습 (ELBO 최대화)**:
```
Encoder: q(z | y, c)  →  μ, log σ²
z ~ Reparameterize(μ, σ)
Decoder: p(y | z, c)  →  ŷ

Loss = Recon(y, ŷ) + β · KL(q(z|y,c) || N(0,I))
```

**추론 (MC sampling)**:
```
z ~ N(0, I)  (n_samples회)
ŷ_s = Decoder(z_s, c)
→ PI_80: [10th percentile, 90th percentile] of {ŷ_s}
→ CRPS: pinball loss approximation
```

GAN 디코더는 불안정성이 크므로 ablation(선택적)으로 두고, 메인은 CVAE로 고정한다.

구현: `models/cvae_decoder.py`, `CVAEDecoder`
- `forward(context, y)` → `(pred, kl_loss)` (학습)
- `forward(context)` → `pred` (추론)
- `predict_interval(context, n_samples, alpha)` → `(mean, pi_low, pi_high)`

### 4.6 학습 절차

**CVAE 학습 loss**:
```
L = MSE(y, ŷ) + λ_KL · KL_loss + λ_gate · gate_reg_loss
```

**Purged Walk-Forward CV**:
- train window: 252일 (또는 시계열 길이의 70%)
- embargo: 6일 (overlapping horizon 누출 방지)
- step: 5일

**HybridModel 학습 API**:
```python
pred = model.forward_train(src, y_true)   # CVAE encoder 활성화
kl   = model._cvae_kl_loss
loss = mse(pred, y_true) + lambda_kl * kl + model._gate_reg_loss
```
