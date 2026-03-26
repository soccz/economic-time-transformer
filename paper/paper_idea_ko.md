# 팩터 조정 수익률의 체제 조건부 예측 가능성: 불확실성 분해를 포함한 하이브리드 딥러닝 접근법

> **확정 스펙 기준 문서** (최종 업데이트)
> - 타겟: FF3 잔차 (MKT, SMB, HML만 제거)
> - WML(모멘텀): 타겟에서 제거하지 않고 입력 피처로 유지
> - FF5/FF6: robustness appendix
> - 디코더: CVAE 메인 (GAN은 ablation)
> - H1 회귀: 2-step Fama-MacBeth
> - 데이터: S&P 500 point-in-time (CRSP 기준, 불가시 Ken French 포트폴리오)

---

## 초록 (draft)

본 연구는 시장 체제(regime)가 Fama-French 3팩터 조정 잔차 수익률의 예측 가능성 구조를 변화시키는지 검토한다. S&P 500 구성종목(1990–2024, point-in-time)과 Ken French 데이터 라이브러리의 FF3 일별 팩터 수익률을 사용하여 롤링 팩터 조정 잔차를 예측 타겟으로 구성한다. 시장 사이클 위치를 시간적 표현에 명시적으로 인코딩하는 하이브리드 딥러닝 모델을 제안한다. 모델은 Transformer 인코더(글로벌 추세 의존성)와 Temporal Convolutional Network(로컬 가격 패턴)를 학습된 게이팅 메커니즘으로 결합하며, 4-상태 시장 체제로 조건화된다. 불확실성은 인식론적(MC-Dropout)과 우연적(CVAE 잠재 분산) 성분으로 분해된다. 사전 등록된 세 가지 가설을 검증한다: (H1) 체제가 단면 모멘텀 프리미엄을 시간적으로 조절한다; (H2) 사이클 인식 위치 인코딩이 정적 인코딩 대비 예측 정확도를 향상시킨다; (H3) 인식론적·우연적 불확실성 분해가 경제학적으로 의미있다.

**키워드:** Fama-French 3팩터, 체제 조건부 예측, 불확실성 분해, Transformer-TCN, CVAE, 인식론적 불확실성, 우연적 불확실성, Cycle-aware PE

**목표 저널:** International Journal of Forecasting / Journal of Empirical Finance / Quantitative Finance

---

## 1. 서론

### 1.1 연구 동기

Fama-French 3팩터 모델(FF3; Fama and French 1993)은 개별 주식 수익률을 체계적 성분과 고유 잔차로 분해한다:

```
r_i,t = α_i + β_MKT·MKT_t + β_SMB·SMB_t + β_HML·HML_t + ε_i,t
```

FF3는 25개 size-B/M 포트폴리오에서 R²가 0.9를 초과하는 경우가 다수임을 보여준다(Fama and French 1993, Table 9a). 그러나 이 설명력은 포트폴리오 수준의 결과이며, 개별 종목 수준에서는 잔차 α의 횡단면 분산이 상당하다. 본 연구는 FF3가 틀렸다고 주장하지 않는다 — FF3가 설명하지 못하는 고유 성분의 조건부 예측 가능성 구조를 탐색한다. "FF3 explains a large fraction of time-series variation in standard size–value portfolios, but our object is the conditional structure of firm-level factor-adjusted residual returns under observable market states."

두 가지 실증적 규칙성이 본 연구를 동기화한다:

1. **시간 가변 팩터 로딩**: APT(Ross 1976)는 β_i,t의 시간 변동을 허용한다. 롤링 OLS 추정값은 팩터 노출이 강세/약세 사이클과 변동성 체제에 걸쳐 실질적으로 변화함을 확인한다.

2. **체제 의존적 정보 구조**: 추세적(강세/조용한) 시장에서는 최근 가격 모멘텀이 지배한다. 스트레스(약세/변동성) 시장에서는 역사적 패턴 반복과 횡단면 팩터 스프레드가 지배한다. 단일 귀납적 편향으로는 두 가지를 모두 포착할 수 없다.

기존 ML 접근법(Gu, Kelly, Xiu 2020; Chen, Kelly, Wu 2023)은 비선형 모델이 원시 수익률 예측에서 선형 팩터 모델을 능가함을 보여준다. 그러나 이들은: (a) 모델 아키텍처에서 시장 체제를 명시적으로 조건화하지 않고, (b) 예측 불확실성을 인식론적·우연적 성분으로 분해하지 않으며, (c) 팩터 조정 잔차를 이론적으로 동기화된 예측 타겟으로 사용하지 않는다.

**FF3 선택 근거**: FF3는 파라미터 수(3개 계수 + 상수 = 4개)가 적어 rolling 60일 OLS에서 추정 노이즈가 최소화된다. WML(모멘텀 팩터)은 타겟에서 제거하지 않고 입력 피처로 유지한다 — 모멘텀을 타겟에서 제거하면 H1(체제가 모멘텀 프리미엄을 조절한다)의 검증 대상이 사라지기 때문이다. FF5/FF6 잔차 타겟을 사용한 robustness 분석은 부록에 보고한다.

### 1.2 연구 질문

- **RQ1 (경제학적)**: 시장 체제가 단면 모멘텀 프리미엄(λ_1,t)을 시간적으로 조절하는가? (→ H1)
- **RQ2 (ML)**: 시장 사이클 위치를 Transformer의 위치 인코딩에 명시적으로 인코딩하면 표본 외 예측 정확도가 향상되는가? (→ H2)
- **RQ3 (불확실성)**: 인식론적·우연적 불확실성이 의미있게 분리될 수 있으며, 시장 조건에 따라 다르게 반응하는가? (→ H3)

### 1.3 기여

1. **이론적으로 근거있는 예측 타겟**: FF3 잔차를 예측 타겟으로 사용. 체계적 성분(MKT, SMB, HML)을 제거하여 고유 정보만 예측 대상으로 삼음. WML은 피처로 유지하여 모멘텀 예측력을 보존.

2. **사이클 인식 위치 인코딩(Cycle-PE)**: 두 개의 연속적 시장 사이클 신호 — `(SPX - MA200d) / MA200d`(사이클 위치)와 `30일 실현 변동성 분위수`(사이클 강도) — 를 Transformer의 시간 표현에 주입하는 PE 설계. 사전 명세, HPO 튜닝 대상 아님. MA200d와 RV30d는 선행 연구(Faber 2007; Moskowitz et al. 2012)에서 확립된 기준을 채택.

3. **분해된 불확실성 추정**: MC-Dropout(인식론적) + CVAE conditional prior 분산(우연적). 우연적 불확실성은 conditional prior의 분산으로 정의한다: σ_aleatoric(c) := mean_d σ_p,d(c). Posterior 분산 σ_q는 학습에만 사용하며, 추론 단계의 불확실성 지표로 사용하지 않는다. 두 성분이 서로 다른 시장 조건에 반응한다는 실증적 검증 포함.

4. **2-step H1 검증**: Fama-MacBeth로 날짜별 λ_1,t(단면 모멘텀 프리미엄)를 추출한 후, λ_1,t ~ regime_t 시계열 회귀로 체제 조절 효과를 검증. regime_t가 모든 종목에 공통값이라는 식별 문제를 구조적으로 해결.

5. **사전 등록된 가설**: 모든 가설과 검정 통계량은 데이터 분석 전에 명세됨.

### 1.4 관련 연구

| 연구 흐름 | 주요 논문 | 해결하는 공백 |
|---------|---------|------------|
| 수익률 예측 ML | Gu, Kelly, Xiu (2020); Chen, Kelly, Wu (2023) | 체제 조건화 없음; 불확실성 분해 없음 |
| 팩터 타이밍 | Kim (2022); Polk, Haghbin, de Longis (2020) | 규칙 기반, 확률적 아님; 불확실성 없음 |
| 시간 가변 β | Ferson & Harvey (1991); Ang & Kristensen (2012) | ML 없음; 불확실성 분해 없음 |
| 확률적 예측 | Lim et al. (2021 TFT); Rasul et al. (2021) | 팩터 모델 근거 없음; 체제 조건화 없음 |
| 금융 불확실성 | Bali, Brown, Tang (2017) | 인식론적/우연적 분해 없음 |
| Cycle-aware PE | — | 금융 사이클 신호를 PE에 주입하는 구체적 적용 (단순 concat 대비 실증 검증) |


---

## 2. 이론적 프레임워크

### 2.1 팩터 모델 분해 및 예측 타겟

Fama-French 3팩터 모델(Fama and French 1993)을 따라:

```
r_i,t := R_i,t − RF_t  (초과수익, Ken French 데이터의 RF 사용)

r_i,t = α_i,t + β_MKT,i,t·MKT_t + β_SMB,i,t·SMB_t + β_HML,i,t·HML_t + ε_i,t

여기서:
  MKT_t := (Mkt−RF)_t  (Ken French Data Library 표기 일치)
  F_t = [MKT_t, SMB_t, HML_t]  (3팩터 벡터)
  β_i,t = 시간 가변 팩터 로딩 (rolling 60일 OLS로 추정, 4개 계수)
  ε_i,t = 고유 잔차 (예측 타겟)
```

**예측 타겟**: `y_i,t = Σ_{h=1}^{5} ε_i,t+h` — 5일 누적 FF3 조정 잔차.

**FF3 선택 이유**: rolling 60일 OLS에서 4개 계수(β_MKT, β_SMB, β_HML, α) 추정은 자유도 56으로 통계적으로 안정적이다. FF6(7개 계수)는 자유도 53으로 추정 노이즈가 증가하며, 특히 WML을 타겟에서 제거하면 H1 검증의 핵심 신호(모멘텀)가 사라진다.

**WML 처리**: WML(모멘텀 팩터)은 타겟 생성에 사용하지 않고 입력 피처(12-1개월 수익률)로 유지. 본 연구의 목적은 모멘텀의 존재를 '발견'하는 것이 아니라, 모멘텀 프리미엄의 체제 조건부 변동을 추정하는 것이다. WML 피처의 예측력이 단순 모멘텀 팩터 재발견이 아님을 보이기 위해, H1 Step 2 회귀에 WML_t(팩터 수익률 시계열)를 통제 변수로 추가하여 cycle_position의 독립적 기여를 분리한다.

**H1 내재 상관 문제 인정**: Step 1의 mom_i,t 피처(12-1개월 수익률)는 WML 팩터 구성 신호와 동일하므로, λ_1,t는 구조적으로 WML_t와 높은 상관관계를 가진다. Step 2에서 WML_t를 통제해도 이 상관관계를 완전히 제거할 수 없다. 따라서 H1의 주장은 "WML과 독립적인 모멘텀 효과"가 아니라 **"FF3 잔차에 잔류하는 모멘텀 성분의 체제 조건부 변동"**으로 한정한다. λ_1,t와 WML_t의 비교는 다음 방식으로 명시한다:
```
사전 진단 (데이터 확보 후 최우선 실행):
  Spearman(cycle_position_t, WML_t)  — 다중공선성 사전 측정
  |ρ| > 0.5이면 Step 2 회귀의 b/c 분리 추정이 불안정하므로 회귀 접근 포기.
  |ρ| ≤ 0.5이면 아래 비교 방법 진행.

비교 방법 (|ρ| ≤ 0.5 조건 충족 시):
  (1) 시계열 상관: Spearman(λ_1,t, WML_t)  — 두 시계열의 동조 정도 측정
  (2) 체제별 조건부 평균 비교:
      E[λ_1,t | regime] vs E[WML_t | regime]  — 각각 표준화(z-score) 후 비교
  (3) Step 2 회귀: λ_1,t = a + b·cycle_position_t + c·WML_t + e_t
      b의 유의성 = cycle_position의 독립 기여

|ρ| > 0.5인 경우 대안:
  Step 2를 단변량으로 유지 (λ_1,t = a + b·cycle_position_t + e_t).
  WML 통제는 포기하고, H1의 주장 범위를 더 좁힘:
  "cycle_position이 λ_1,t를 예측한다"로만 주장. WML과의 독립성은 주장하지 않음.

해석 기준: b가 유의하면 → H1 성립 (주장 범위 내에서)
           b가 유의하지 않으면 → H1 기각, 논문 프레임 재검토 필요
```

**Robustness**: FF4 잔차 타겟(FF3+WML 제거)을 사용한 동일 실험을 부록에 보고. "모멘텀을 타겟에서 제거해도 체제 조건부 구조가 일부 남는가"를 검증하여 WML 재발견 공격을 약화시킨다.

### 2.2 체제 정의 (사전 명세)

시장 체제는 t-1 시점까지 이용 가능한 데이터로 계산된 두 개의 연속 신호로 정의됨 (사후 편향 없음):

```
cycle_position_t  = (SPX_t - MA200d_t) / MA200d_t     # 추세 위치 (연속)
cycle_intensity_t = quantile_rank(RV_30d_t, 과거 252일)  # 변동성 상태 (0~1)
```

**MA200d 선택 근거**: Faber(2007), Moskowitz et al.(2012) 등 선행 연구에서 확립된 추세 기준선. HPO 탐색 대상이 아닌 사전 고정값.

**RV30d 선택 근거**: 변동성 측정의 표준 윈도우(20~30일). 추세 신호(MA200d)보다 빠른 전환을 포착하기 위해 의도적으로 다른 스케일 사용.

4-상태 이산 체제 레이블 (FiLM 조건화용):
```
regime_t = 2 × I(cycle_position_t > 0) + I(cycle_intensity_t > 0.5)
         ∈ {0: Bear/quiet, 1: Bear/volatile, 2: Bull/quiet, 3: Bull/volatile}
```

**체제 정의의 임의성 인정 및 대응**:

cycle_intensity는 quantile_rank이므로 임계값 0.5는 항상 데이터를 반반으로 나눈다. 이것은 임계값을 바꿔도 해결되지 않는 구조적 특성이다 — 0.3/0.7로 바꾸면 Bear/volatile 비율이 달라질 뿐, 분할의 임의성 자체는 유지된다.

이 문제에 대한 대응 계층:
```
1차 분석 (연속 신호 직접 사용):
   H1 Step 2: λ_1,t = a + b·cycle_position_t + c·cycle_intensity_t + e_t
   H3b: σ_aleatoric ~ f(cycle_position_t, cycle_intensity_t)  (연속 회귀)
   → 이산 분류 없이 연속 신호로 직접 검증. 이산화 임의성 문제를 우회.

2차 분석 (이산 4-상태, FiLM 조건화용):
   FiLM 레이어는 이산 레이블을 요구하므로 4-상태 분류를 유지.
   임계값 민감도: {0, 0.5} 기준 외에 {0, 0.3}, {0, 0.7} 대안 보고 (appendix).
   → 이산 분류는 모델 아키텍처 요구사항이며, 경제학적 주장(H1/H3)은 1차 분석(연속)에 의존.

**H1/H3(연속)과 H2(이산 FiLM) 간 체제 표현 불일치 처리**:
H1/H3는 연속 신호(cycle_position, cycle_intensity)로 검증하고, H2 모델은 이산 4-상태 FiLM을 포함한다. 이 불일치는 다음과 같이 처리한다:
- H2 ablation에서 FiLM을 연속 신호 조건화로 대체한 변형(FiLM-continuous)을 추가하여, 이산화가 예측 성능에 미치는 영향을 분리 측정.
- H1에서 연속 신호로 체제 조절 효과가 확인되고 H2 모델이 이산 FiLM을 사용하더라도, 두 가설은 서로 다른 질문을 다룬다: H1은 "체제가 모멘텀 프리미엄을 조절하는가"(경제학적), H2는 "사이클 신호를 PE에 주입하면 예측이 개선되는가"(ML). 체제 표현 방식의 차이는 두 가설의 독립성을 훼손하지 않는다.
- 단, FiLM-continuous ablation에서 이산 FiLM 대비 성능 차이가 크면, 이산화 선택의 영향을 한계 섹션에 추가한다.

보고 원칙: H1/H3의 주요 결과는 연속 신호 기반으로 보고. 이산 4-상태 결과는 보조.
```

**민감도 분석 (data snooping 방어)**: MA {120d, 200d, 252d} × RV {20d, 30d, 60d} 격자 9개 조합에서 H1 결과의 안정성을 부록에 보고. 특정 파라미터에서만 결과가 나오면 data snooping 의심; 넓은 범위에서 일관되면 방어 가능.

### 2.3 가설 수식화 (사전 등록)

**H1 — 체제가 단면 모멘텀 프리미엄을 조절하는가**

2-step 설계 (식별 문제 해결):

```
Step 1: 날짜별 cross-sectional regression (Fama-MacBeth)
  y_i,t = λ_0,t + λ_1,t · mom_i,t + γ · controls_i,t + η_i,t
  → λ_1,t 시계열 추출 (단면 모멘텀 프리미엄)

Step 2: time-series regression
  λ_1,t = a + b · cycle_position_t + e_t
  → b의 유의성 = H1 검증

귀무가설:    b = 0  (체제가 모멘텀 프리미엄을 조절하지 않음)
대립가설: b ≠ 0, 부호 예측:
  Bull 체제(cycle_position > 0): λ_1,t 높음 (모멘텀 증폭)
  Bear 체제(cycle_position < 0): λ_1,t 낮음 또는 음수 (모멘텀 약화/반전)

검정: HAC t-검정 (Newey-West, 10 래그) on Step 2
      regime 구간별 λ_1,t 평균 비교 (4개 체제)
```

**H1 식별 문제 해결 이유**: regime_t는 모든 종목에 동일한 시장 공통값이므로, 단일 cross-sectional 회귀에서 mom_i,t × regime_t 상호작용항은 식별이 어렵다. 2-step 설계는 "체제가 단면 모멘텀 프리미엄을 시간적으로 조절하는가"라는 질문으로 재프레이밍하여 이 문제를 구조적으로 해결한다.

**Generated regressor 문제 대응**:

Step 2에서 λ̂_1,t는 추정값이므로 두 가지 문제가 발생한다: (a) 표준오차 과소추정, (b) β̂_i,t 추정 오차가 ε̂_i,t를 통해 λ̂_1,t에 전파되는 편향.

**(a) 표준오차 문제**: 주 분석에 **stationary bootstrap**(Politis & Romano 1994, L=20)을 적용하여 Step 1→Step 2 추정 불확실성을 전파. HAC는 보조 검증으로 병행.

**(b) β̂ 추정 오차 전파 — 편향 교정 방안**:
```
진단 (사전 실행):
  Spearman(|mom_i,t|, se(β̂_MKT,i,t))  — 모멘텀 극단 종목의 β 추정 불안정성 측정
  목표: 상관관계가 낮으면(|ρ| < 0.1) 편향이 무시 가능하다고 주장

교정 방안 (상관관계가 유의한 경우):
  문제의 성격: β̂_i,t 추정 오차가 ε̂_i,t에 전파되어 λ̂_1,t의 피설명변수에 오차가 생긴다.
  피설명변수 측정오차는 계수 편향을 만들지 않지만, 그 오차가 설명변수(mom_i,t)와
  상관관계를 가질 경우 endogeneity 문제가 발생한다.
  → 고전적 EIV 공식(설명변수 오차 교정)은 이 경우에 적용 불가.

  주 방법: IV 접근
    mom_i,t의 도구변수로 t-2 시점 모멘텀(mom_i,t-2) 사용.
    조건: mom_i,t-2가 현재 β̂ 추정 오차와 독립적이어야 함.
    검증: Spearman(mom_i,t-2, se(β̂_MKT,i,t)) ≈ 0 확인 후 적용.
  보조 방법: 진단 상관관계가 낮으면(|ρ| < 0.1) 편향 무시 가능 주장.

보고 방식: 진단 결과를 appendix에 공개. 편향이 무시 가능하면 주 결과 유지.
           편향이 유의하면 IV 추정 결과를 주 결과로 대체.
```

**H2 — Cycle-PE가 예측 정확도를 향상시키는가**

```
귀무가설:    IC(Cycle-PE 모델) = IC(Static-PE 모델)
대립가설: IC(Cycle-PE 모델) > IC(Static-PE 모델)

검정: Diebold-Mariano 검정 (표본 외 IC 차이)
      체제별 IC 비교 (4개 체제 각각)
      Attention heatmap 분석: Bull vs Bear 체제에서 attention 패턴 차이
```

**H3 — 인식론적/우연적 분해가 경제학적으로 의미있는가**

```
H3a: σ_epistemic (MC-Dropout std)이 데이터 희소 구간에서 높아지는가
     프록시: (1) 1/R2_60d  (2) se(β_MKT)  (3) 1/listing_age (보조)
     검정: Spearman(σ_epistemic_i,t, 각 프록시)

H3b: σ_aleatoric := mean_d σ_p,d(c) (conditional prior 분산)이 Bear/volatile 체제에서 높아지는가
     검정: 체제별 σ_aleatoric 평균 비교, Tukey HSD
     보조 검증: partial Spearman(σ_aleatoric_t, VIX_t | cycle_intensity_t)

H3c: CRPS(CVAE 모델) < CRPS(분위수 회귀 baseline)
     검정: Diebold-Mariano 검정
     보조: Reliability diagram (calibration 시각화)
```

### 2.4 불확실성의 경제학적 해석

APT 프레임워크에서 수익률 분산은 두 성분으로 분해된다:

```
Var(r_i,t) = Var(β_i,t · F_t) + Var(ε_i,t)
             ↑                   ↑
             체계적 분산           고유 분산
```

이 분해가 불확실성 추정과 대응된다:

```
σ_epistemic (MC-Dropout):
  → 모델이 β̂_i,t를 얼마나 확신하는가
  → 데이터 희소 시 β 추정 불안정 → σ_epistemic 증가
  → 검증: σ_epistemic ↑ when listing_age ↓

σ_aleatoric := mean_d σ_p,d(c)  (conditional prior 분산, context 의존):
  → ε_i,t 자체의 고유 변동성
  → 시장 스트레스 시 고유 노이즈 증가 → σ_aleatoric 증가
  → 검증: σ_aleatoric ↑ when VIX ↑, Bear/volatile 체제
```

**CVAE 선택 이유**: GAN 방식에서 aleatoric uncertainty = std({G(context, z^(k))})는 generator 불안정성과 data noise를 구분할 수 없다. Conditional prior CVAE에서 σ_p(c)는 context에서 직접 산출되어 수학적으로 명확하다. Posterior 분산 σ_q는 학습에만 사용하며, 추론 단계의 불확실성 지표로 사용하지 않는다.

**σ_aleatoric의 구조적 한계 명시**: σ_p,d(c)는 prior 네트워크 p_ψ가 context c로부터 출력하는 분산이다. 이것이 "데이터 고유의 노이즈"를 측정한다는 주장은 다음 조건 하에서만 성립한다: (1) posterior collapse가 없어야 하고 (KL > free_bits 기준으로 감지), (2) σ_p(c)가 context 피처와 독립적인 외부 이벤트에서도 높게 나와야 한다. 조건 (2)의 검증:
```
어닝 서프라이즈 이벤트 스터디:
  이벤트 날짜: IBES 어닝 서프라이즈 상위/하위 10% 종목 (|SUE| > 1.5σ)
  대조군: 동일 종목, 동일 체제, 비-이벤트 날짜 (이벤트 ±30일 제외)
  검정: E[σ_p(c) | 이벤트] vs E[σ_p(c) | 대조군], paired t-test
  목표: 이벤트 날짜에서 σ_p가 유의하게 높고, 동일 context 피처를 가진
        비-이벤트 날짜에서는 낮아야 한다.
  한계 인정: context 피처(realized_vol, beta 등)가 이벤트 전후 변화하므로
             완전한 분리는 불가능. 이 검증은 σ_p의 해석 가능성을 지지하는
             증거이지, 인과적 증명이 아님을 명시.
```
이 검증의 근본 한계: context 피처(realized_vol, beta 등)가 이벤트 전후 변화하므로, σ_p가 높아지는 것이 "이벤트 자체의 불확실성"인지 "context 피처 변화에 대한 prior 네트워크의 반응"인지 구분이 불가능하다. 따라서 이 이벤트 스터디는 σ_aleatoric의 해석 가능성을 검증하는 것이 아니라, σ_p가 경제적으로 의미있는 상황에서 반응한다는 정황 증거를 제공하는 것으로 역할을 한정한다.

σ_aleatoric의 해석은 "데이터 고유 노이즈"가 아닌 **"prior 네트워크가 context로부터 추정하는 잠재 불확실성"**으로 고정한다. H3b의 주장은 이 정의 하에서만 성립하며, 이것이 aleatoric uncertainty의 이론적 정의와 완전히 일치하지 않음을 한계 섹션에 명시한다.


---

## 3. 모델 아키텍처

### 3.1 전체 구조 및 인과 사슬

```
입력: (B, T, D)  — T=60일 윈도우, D개 피처
  ↓
[Cycle-aware PE]
  PE_total = PE_static(position) + PE_cycle(cycle_position, cycle_intensity)
  → 각 토큰이 시장 사이클 내 위치를 인식
  ↓
[Transformer Encoder]  — 글로벌: 체제 인식 상태에서 장기 의존성 포착
  → attention_weights: (B, T, T)
  → last_token: (B, d_model)
  ↓ (attention_weights[-1] → guidance prior, stop-gradient)
[Attention-guided TCN]  — 로컬: Transformer가 중요하다고 한 시점의 패턴
  guided_input = src × softmax(attention_weights[:, -1, :]).unsqueeze(-1)
  → tcn_features: (B, d_tcn)
  ↓
[Explainable Gated Fusion]
  gate = σ(Linear([transformer_feat, tcn_feat]))
  fused = gate * transformer_feat + (1 - gate) * tcn_feat
  gate ≈ 1: Transformer 지배 (추세/글로벌)
  gate ≈ 0: TCN 지배 (패턴/로컬)
  ↓
[FiLM Regime Conditioning]
  fused = scale(regime_embed) * fused + shift(regime_embed)
  regime ∈ {0,1,2,3} → 4-state 체제별 affine 변환
  ↓
[CVAE Decoder]
  훈련: q_φ(z|c,y) → μ_q, σ_q  /  p_ψ(z|c) → μ_p, σ_p
        z ~ q_φ(z|c,y),  y_hat = Decoder(c, z)
        Loss += KL(q_φ(z|c,y) || p_ψ(z|c))
  추론: z^(k) ~ p_ψ(z|c)  (μ_p, σ_p를 context에서 직접 산출)
        {y_hat^(k)}_{k=1}^{K} → 예측 분포
  ↓
출력: y_hat_i,t (5일 누적 FF3 잔차), σ_aleatoric := mean_d σ_p,d(c), σ_epistemic (MC-Dropout)
```

**인과 사슬 논리**:
- Cycle-PE가 체제를 인코딩했다면 → Transformer attention이 체제별로 달라져야 함 (H2 검증)
- Attention이 체제별로 다르다면 → Gate도 체제와 상관관계를 가져야 함 (진단 변수)
- 체제가 정보 처리를 바꾼다면 → 불확실성 구조도 체제별로 달라져야 함 (H3 검증)

각 단계가 다음 단계를 논리적으로 요구하는 구조 → ablation이 자연스럽게 도출됨.

### 3.2 Cycle-aware Positional Encoding

표준 사인파 PE는 정수 위치(1, 2, ..., T)만 인코딩한다. 시점 t=30은 2008년 금융위기 직전이든 2021년 강세장 중반이든 동일한 PE 벡터를 받는다. 이 한계가 금융 시계열에서 특히 심각한 이유:

```
주식 A: 60일 윈도우, 현재 = 강세장 정점 (cycle_position = +0.35)
주식 B: 동일한 60일 가격 패턴, 현재 = 약세장 바닥 (cycle_position = -0.28)

Static PE: 두 시퀀스에 동일한 시간 표현 → Transformer가 구분 불가
Cycle-PE:  각 토큰에 시장 맥락 주입 → 체제 조건부 attention 학습 가능
```

**수식**:
```
PE_static(pos, d) = sin/cos(pos / 10000^(2d/D))   # 표준 sinusoidal

PE_cycle(t, d) = [cycle_position_t, cycle_intensity_t] @ W_phase
  W_phase ∈ R^{2×D}  (학습됨)

PE_total = PE_static + PE_cycle
```

**Cycle-PE 설계 원칙**:
1. 연속성: cycle_position은 연속값 → 체제 경계에서 불연속 없음
2. 인과성: t 시점의 PE는 t-1까지 데이터로만 계산 → 사후 편향 없음
3. 분리성: cycle_position(추세)과 cycle_intensity(변동성)는 독립적으로 변동

**Cycle-PE vs concat ablation (novelty 실증)**:
Cycle-PE는 cycle_position을 추가 입력 피처로 concat하는 것과 구조적으로 다르다. Baseline은 **Concat-A(per-token)**로 고정한다: 각 토큰 날짜 τ에 대해 (cycle_position_τ, cycle_intensity_τ)를 계산하여 입력 채널로 concat. 이 방식은 Cycle-PE와 동일한 정보량을 제공하며, 차이는 오직 주입 위치(입력 vs PE)에서만 발생한다. ablation 테이블에 "Cycle-PE vs Cycle concat-A" 항목을 포함하여 DM test로 비교한다.

**Cycle-PE novelty 실패 시 대비**: DM test에서 Cycle-PE와 Concat-A 간 유의한 차이가 없을 경우, 기여 2번(Cycle-PE)의 주장을 "PE 주입 방식의 우월성"에서 "사이클 신호 자체의 기여"로 재프레이밍한다. 즉 Cycle-PE와 Concat-A 모두 Static PE 대비 유의하게 개선된다면, 기여는 "어떻게 주입하느냐"가 아니라 "사이클 정보를 명시적으로 인코딩하는 것 자체"로 성립한다. 이 경우 ablation 테이블에서 Static PE vs {Cycle-PE, Concat-A} 비교를 1차 결과로 보고하고, Cycle-PE vs Concat-A는 보조 비교로 격하한다.

**FiLM과의 역할 분리**:
```
Cycle-PE: 입력 레벨 — 각 토큰의 시간적 표현 조정 ("언제")
FiLM:     표현 레벨 — 인코더 출력 전체를 체제별 affine 변환 ("어떤 체제에서")
```
두 메커니즘은 상호 보완적이며 역할이 겹치지 않는다.

### 3.3 Attention-guided TCN

Transformer attention의 마지막 토큰 행(attention_weights[:, -1, :])을 TCN 입력 가중치로 사용:

```
attention_importance = softmax(attention_weights[:, -1, :])  # (B, T)
guided_input = src * attention_importance.unsqueeze(-1)       # (B, T, D)
tcn_features = TCN(guided_input)
```

**stop-gradient 적용**: attention_importance는 TCN 경로에서 gradient를 차단. Transformer와 TCN이 독립적으로 학습되도록 보장.

**역할**: attention 가중치를 TCN 입력에 적용하여 두 경로가 동일한 입력 가중치를 공유하는 구조. stop-gradient로 인해 Transformer는 TCN이 잘 활용할 수 있는 attention을 학습할 인센티브가 없으므로, 이 결합이 실제로 유효한지는 설계 가정이 아닌 실증적 질문이다 (Random importance ablation으로 검증).

**한계 및 검증**: Transformer attention이 예측 정확도와 직접 연결되지 않을 수 있다 (Jain & Wallace 2019). 따라서 attention-guided TCN의 기여는 설계 가정이 아닌 실증적 질문으로 취급한다. 검증: Random importance baseline (attention 대신 uniform 가중치) 대비 DM test — ablation 테이블 "Random importance" 항목.

### 3.4 Gated Fusion (진단 변수)

```
gate = σ(Linear([transformer_feat; tcn_feat]))  # (B, 1)
fused = gate * transformer_feat + (1 - gate) * tcn_feat
```

gate는 설명 변수가 아닌 **진단 변수**로 취급:
- gate ≈ 1: Transformer 경로 가중치 높음 (글로벌/추세 구조)
- gate ≈ 0: TCN 경로 가중치 높음 (로컬/패턴 구조)

"We treat the gate as a diagnostic variable rather than a causal explanation; its utility is validated via oracle-consistency and regime-wise aggregation tests."

gate가 체제와 상관관계가 있는지는 실증적 질문 (설계 가정 아님):
- 검증 A: Spearman(gate_by_date, cycle_position_by_date) — pseudo-replication 방지를 위해 날짜별 집계
- 검증 B: Oracle consistency — gate_binary vs (transformer_error < tcn_error) AUC
- 검증 C: attention oracle consistency — attention_top1_timestep vs (해당 시점 제거 시 오차 증가) AUC (attention ≠ explanation 방어)

### 3.5 CVAE Decoder 및 불확실성 분해

Conditional prior CVAE 설계:

```
훈련 시:
  Posterior: q_φ(z | c, y) → μ_q, σ_q  (B, latent_dim)
  Prior:     p_ψ(z | c)    → μ_p, σ_p  (B, latent_dim)  ← context로부터 학습
  z ~ q_φ(z|c,y)  (reparameterization trick)
  y_hat = Decoder(c, z)
  L_ELBO = L_recon + β · KL(q_φ(z|c,y) || p_ψ(z|c))

추론 시:
  z^(k) ~ p_ψ(z|c),  k = 1,...,K  (μ_p, σ_p를 context에서 직접 산출)
  {y_hat^(k)} → 예측 분포
```

**불확실성 분해 (단일 정의)**:
```
정의:
  σ_aleatoric := mean_d σ_p,d(c)   # conditional prior 분산 (context 의존, 직접 해석 가능)
  σ_epistemic  := std_m({ŷ^(m)})   # MC-Dropout 샘플 m 간 예측 변동 (z = μ_p(c) 고정)

측정 방식 (두 성분을 독립적으로 측정):
  σ_epistemic 측정: dropout ON,  z = μ_p(c) 고정
    → {ŷ^(m)}_{m=1}^{M},  σ_epistemic = std_m(ŷ^(m))
  σ_aleatoric 측정: dropout OFF, z ~ p_ψ(z|c) 재샘플
    → {ŷ^(k)}_{k=1}^{K},  σ_aleatoric = mean_d σ_p,d(c)  (prior 분산 직접 사용)

참고: 위 두 측정은 독립 실행이며, law-of-total-variance 분해
  Var_total ≈ Var_m(E_z[ŷ]) + E_m(Var_z[ŷ])
는 동일 샘플 집합에서 성립하는 항등식이다. 본 설계는 각 성분을
해석 가능성을 위해 분리 측정하며, 합산 Var_total은 보조 보고에만 사용.

PI_80 = [quantile({ŷ^(m,k)}_{m,k}, 0.10), quantile({ŷ^(m,k)}_{m,k}, 0.90)]
  (dropout ON + z ~ p_ψ(z|c) 동시 샘플링, M×K 샘플)
```

**GAN_noise 표기 전면 삭제**

**손실 함수**:
```
L_total = L_recon + β·KL + λ_dir·L_dir

L_recon = MSE(y_hat_mean, y_true)
L_dir   = -E[tanh(k·y_hat) · sign(y_true)]  (sign() 대신 tanh 근사, gradient 보존)
```

**CRPS는 손실에서 제외**: CRPS를 훈련 손실에 포함하면 평가 지표로서의 독립성이 훼손된다 (모델이 CRPS를 직접 최적화했는지 vs 실제로 더 잘 보정된 건지 구분 불가). CRPS는 평가(보고)에만 사용.

**ECE는 손실에서 제외**: ECE는 binning 기반 비미분 지표 → 평가(보고)에만 사용.


---

## 4. 데이터 및 실험 설계

### 4.1 데이터 소스

| 소스 | 내용 | 기간 | 접근 |
|------|------|------|------|
| CRSP (WRDS) | S&P 500 point-in-time 구성종목 + delisting 수익률 | 1990–2024 | WRDS 구독 |
| Ken French Data Library | FF3 일별 팩터 수익률 (MKT, SMB, HML, WML) | 1926– | 무료 |
| Yahoo Finance (^GSPC) | S&P 500 지수 (체제 신호 계산용) | 1990– | 무료 |
| FRED | VIX 지수 (H3b 검증용) | 1990– | 무료 |

**CRSP 불가 시 대체안**: Ken French 25 Size-B/M 포트폴리오 또는 Industry 49 포트폴리오 cross-section 사용. 포트폴리오 단위에서도 H1/H2/H3 검증 가능하며, 생존편향과 상폐 처리가 포트폴리오 산출 과정에서 이미 처리됨.

**Point-in-time 필수 이유**: yfinance 기반 현재 S&P 500 구성종목으로 1990년까지 소급하면 생존편향(survivorship bias)이 발생한다. CRSP Security Index Membership 테이블(시작/종료일 포함)로 날짜별 구성종목을 재구성해야 한다.

### 4.2 피처 구성

**가격 기반 피처** (rolling [t-60, t-1]):
```
log_return_d        일별 로그 수익률
volume_log          log(거래량), cross-sectional 정규화
realized_vol_20d    sqrt(sum(r^2, 20일))
momentum_12_1       12-1개월 수익률 (WML 피처)
momentum_1          1개월 수익률
```

**팩터 노출도 피처** (rolling 60일 OLS):
```
beta_MKT, beta_SMB, beta_HML   (FF3 로딩 3개)
residual_momentum_20d           과거 20일 잔차 평균
R2_60d                          rolling 회귀 적합도
```

**체제/사이클 신호** (t-1까지 데이터만):
```
cycle_position      (SPX - MA200d) / MA200d  (연속)
cycle_intensity     quantile_rank(RV_30d, 252일 이력)  (0~1)
regime_label        4-state 이산 레이블 (FiLM용)
```

**Cross-sectional 정규화**: 종목별 피처만 적용. cycle_position, cycle_intensity는 시장 공통값이므로 cross-sectional rank 정규화 제외 → rolling z-score(시간축)로 스케일 조정.

총 피처 수 D ≈ 15–20개.

### 4.3 예측 타겟

```
# rolling β 추정 (leakage 없음: [t-60, t-1] 데이터만 사용)
β̂_i,t = OLS(r_i,τ ~ MKT_τ + SMB_τ + HML_τ, τ ∈ [t-60, t-1])
  여기서 r_i,τ := R_i,τ − RF_τ  (초과수익)

# FF3 조정 잔차
ε̂_i,t+h = r_i,t+h - β̂_i,t · F_t+h    for h = 1,...,5

# 예측 타겟 (스칼라)
y_i,t = Σ_{h=1}^{5} ε̂_i,t+h
```

**병렬 타겟** (ablation): `y_raw_i,t = Σ_{h=1}^{5} r_i,t+h` — raw 5일 수익률. 동일 모델, 동일 피처로 비교하여 FF3 조정의 기여를 실증.

### 4.4 Train/Validation/Test 분할

Purged Walk-Forward CV (López de Prado 2018):

```
전체 기간: 1990–2024 (약 34년, ~8,500 거래일)

Walk-forward folds (5개):
  Fold 1: Train 1990–2001, Embargo 10일, Val 2002–2004
  Fold 2: Train 1990–2004, Embargo 10일, Val 2005–2007
  Fold 3: Train 1990–2007, Embargo 10일, Val 2008–2010
  Fold 4: Train 1990–2010, Embargo 10일, Val 2011–2013
  Fold 5: Train 1990–2013, Embargo 10일, Val 2014–2016

최종 held-out test: 2017–2024 (HPO/ablation 중 절대 열람 금지, 최종 1회만 평가)
```

**Embargo 10일**: horizon=5일의 2배. 겹치는 수익률 윈도우로 인한 leakage 방지.

**유니버스 선정 시간정합성**: 각 fold의 train 구간 종료 시점 기준으로 구성종목 선정. 전체 기간 생존 종목 필터 금지 (미래 생존 look-ahead 발생).

### 4.5 평가 지표

**예측 품질** (모델 단독, funnel 없음):
```
IC          Information Coefficient = Spearman(y_hat, y_true), 날짜별 평균
ICIR        IC / std(IC)
MAE         Mean Absolute Error
CRPS        Continuous Ranked Probability Score (properscoring)
PI_80_cov   80% 예측 구간 실제 coverage (목표: 80%)
```

**불확실성 품질**:
```
Reliability diagram   예측 분위수 vs 실현 빈도
Sharpness            PI_80 폭 (coverage 동일 조건에서 좁을수록 좋음)
ECE                  Expected Calibration Error (평가 전용, 손실 아님)
```

**포트폴리오 성과** (funnel 적용, 예측 성능과 분리 보고):
```
전략 명세 (사전 고정):
  단방향: 매 거래일 t, 예측치 상위 10% 롱 / 하위 10% 숏 (달러 중립, equal-weight)
  홀딩: 5거래일, 매일 롤링 (5개 빈티지 격침)
  거래비용: one-way 10bp 기준, 5–15bp 민감도 분석 (appendix)
  가중: equal-weight

Sharpe ratio (연율화, DSR 보정)
Maximum Drawdown
Information Ratio vs equal-weighted benchmark
```

---

## 5. 실험 계획

### 5.1 ML Ablation 테이블 (H2 검증)

| 모델 | IC | ICIR | CRPS | PI_80_cov | 비고 |
|------|----|------|------|-----------|------|
| Full (Cycle-PE + Gate + FiLM + CVAE) | — | — | — | — | 제안 모델 |
| Static PE only | — | — | — | — | H2 검증 |
| Cycle-PE → Cycle concat (equal param) | — | — | — | — | PE novelty 실증 |
| Random importance (no attention-guided TCN) | — | — | — | — | 링크 2 |
| Simple average fusion (no gate) | — | — | — | — | 링크 3 |
| No FiLM | — | — | — | — | 링크 4 |
| Transformer only | — | — | — | — | 단일 경로 |
| TCN only | — | — | — | — | 단일 경로 |
| DLinear (동일 입력, 동일 타겟) | — | — | — | — | 외부 baseline |
| Quantile Regression (동일 입력) | — | — | — | — | CVAE 필요성 |
| Full, raw return target | — | — | — | — | FF3 조정 기여 |
| Full, FF4 residual target (appendix) | — | — | — | — | WML 재발견 robustness |

통계 유의성: Diebold-Mariano 검정 vs Full 모델, Bonferroni 보정 (α/9).

### 5.2 불확실성 분해 테이블 (H3 검증)

| 지표 | Bull/quiet | Bull/volatile | Bear/quiet | Bear/volatile |
|------|-----------|--------------|-----------|--------------|
| Mean σ_epistemic | — | — | — | — |
| Mean σ_aleatoric | — | — | — | — |
| PI_80 coverage | — | — | — | — |
| CRPS | — | — | — | — |

H3a: 멀티 프록시 기반 σ_epistemic 검증:
     (1) Spearman(σ_epistemic, 1/R2_60d) — 팩터회귀 적합도 낙을수록 β 추정 불안정
     (2) Spearman(σ_epistemic, se(β_MKT)) — rolling OLS 표준오차 직접 활용 (가능한 경우)
     (3) Spearman(σ_epistemic, 1/listing_age) — 보조 (분산 약할 수 있음 명시)
H3b: ANOVA F-stat for σ_aleatoric across 4 regimes, Tukey HSD
     (σ_aleatoric := mean_d σ_p,d(c) 기준)
     보조: partial Spearman(σ_aleatoric_t, VIX_t | cycle_intensity_t) — 입력 복사 방어
H3c: DM test: CRPS(CVAE) vs CRPS(Quantile Regression)
보조: Spearman(σ_aleatoric_t, VIX_t)

### 5.3 H1 검증 테이블

2-step Fama-MacBeth 결과:

| 계수 | 추정값 | HAC t-stat | p-value |
|------|--------|-----------|---------|
| a (절편) | — | — | — |
| b (cycle_position 계수) | — | — | — |

체제별 λ_1,t 평균 비교:

| 체제 | λ_1,t 평균 | std | t-stat vs Bear/quiet |
|------|-----------|-----|---------------------|
| Bull/quiet | — | — | — |
| Bull/volatile | — | — | — |
| Bear/quiet | — | — | — |
| Bear/volatile | — | — | — |

### 5.4 민감도 분석

MA {120d, 200d, 252d} × RV {20d, 30d, 60d} 격자 9개 조합에서 IC와 H1 b계수 보고. 결과가 넓은 범위에서 일관되면 data snooping 우려 완화.

---

## 6. 과적합 방어 및 Multiple Testing

**모델 규모**: ~3M 파라미터 (경량, 단일 CPU 서버에서 추론 가능)

**Multiple Testing 대응**:
1. 사전 등록: H1/H2/H3 및 검정 통계량을 데이터 분석 전 문서화 (이 문서)
2. HPO 목적함수 분리: Optuna는 IC/CRPS만 최적화. Sharpe는 최종 held-out에서만 보고
3. Bonferroni 보정: 사전 명세한 m개 주요 ablation 비교에 Holm(1979) step-down 절차 적용 (Bonferroni보다 덜 보수적, FWER 유지). "We control the family-wise error rate across the m pre-specified ablation comparisons using the Holm (1979) procedure."
4. DSR 보정: 최종 Sharpe 보고 시 Deflated Sharpe Ratio (Bailey & López de Prado 2014) 적용. trials = (ablation 비교 수) + (HPO에서 평가한 후보 수)로 사전 명세.
5. Held-out 고정: 2017–2024 test set은 최종 1회만 평가

**논문 1차 기여 재정의**: IC 향상이 DLinear 대비 미미하더라도 논문이 살아남는 구조를 유지한다. 제안 모델의 1차 기여는 정확도 SOTA가 아니라 **체제 조건부 예측 가능성 구조 + 불확실성 분해 + 검정 가능성**이다. DLinear가 IC에서 유사하더라도, 체제별 성능 분해, σ_epistemic/σ_aleatoric 분리, gate 진단 변수는 DLinear가 제공하지 못하는 추가 정보다.

**과적합 방어**: rolling window 겹침을 고려한 유효 표본 수 축소 문제를 인정하고, fold별 IC/CRPS를 별도 표로 보고하여 특정 fold 의존성을 투명하게 공개한다. 정규화: dropout(0.3), weight decay(1e-4), early stopping(patience=10) 사전 명세.

---

## 7. 한계

- **β 추정 오차**: rolling OLS β̂에 추정 노이즈가 포함되어 타겟에 오염될 수 있음. Fama-MacBeth robustness로 부분 대응.
- **생존편향**: CRSP point-in-time 없이는 완전한 통제 불가. CRSP 불가 시 Ken French 포트폴리오로 대체.
- **단일 시장**: 미국 주식 시장만 분석. 다른 시장으로의 일반화는 미래 연구.
- **CVAE 안정성**: posterior collapse 위험. β-annealing 및 free bits 기법으로 완화. Conditional prior p_ψ(z|c) 설계로 학습 압력 분산.
- **Multiple testing**: 완전 제거 불가. 사전 등록 + Bonferroni + DSR로 최대한 방어.

---

## 8. 미결 사항 (TODO)

- [ ] CRSP 접근 가능 여부 확인 → 불가 시 Ken French 포트폴리오로 전환
- [ ] **H1 선형 회귀 빠른 검증 (데이터 확보 후 최우선)** — 논문 존재 이유
- [ ] **β̂ 추정 오차-mom 상관관계 진단** (Spearman(|mom_i,t|, se(β̂_MKT))) — EIV 보정 필요 여부 판단
- [ ] 상관관계 유의 시 EIV 보정 구현 (또는 IV 접근 mom_i,t-2 도구변수)
- [ ] H1 Step 2에 **연속 신호 직접 회귀** (λ_1,t ~ cycle_position + cycle_intensity) 주 분석으로 구현
- [ ] H1 Step 2에 stationary bootstrap 구현 (L=20 기본, L∈{10,20,60} 민감도 appendix)
- [ ] H1 Step 2에 WML_t 통제 변수 추가 + λ_1,t vs WML_t 비교 방법론 실행
- [ ] Conditional prior CVAE 구현 (p_ψ(z|c) 학습, posterior collapse 방지: β-annealing + free bits)
- [ ] **어닝 서프라이즈 이벤트 스터디** (σ_p 해석 가능성 검증, IBES 데이터 필요)
- [ ] ablation: Cycle-PE vs Cycle concat (DM test)
- [ ] ablation: FF4 residual target (appendix, WML 재발견 robustness)
- [ ] H3a: 멀티 프록시 구현 (1/R2_60d, se(β_MKT), listing_age)
- [ ] H3b: partial Spearman(σ_aleatoric_t, VIX_t | cycle_intensity_t) 구현
- [ ] DSR trials 수 사전 명세 (ablation 수 + HPO 후보 수)
- [ ] FF5 robustness 실험 (부록용)
- [ ] 민감도 분석 격자 실험 (MA × RV 9개 조합)
- [ ] Survivorship bias 처리 방법 확정

