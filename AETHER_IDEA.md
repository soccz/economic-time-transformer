# AETHER 핵심 아이디어 문서

> 이 문서가 모든 설계 결정의 기준점이다.
> 코드 수정 전에 반드시 이 문서의 인과 사슬과 일치하는지 확인할 것.

---

## 0. 논문 포지셔닝

**이 논문은 새로운 경제학 이론을 제안하지 않는다. 기존 문헌에서 도출된 메커니즘 가설을 empirical ML 프레임워크로 검증한다.**

- 포지셔닝: **empirical ML for crypto return forecasting** (Gu, Kelly, Xiu 2020 계열) + **H1 가설 검증 (정보 전파 지연)**
- 주장 범위: BTC 사이클 상태가 알트코인 수익률 예측의 *조건부 예측력(conditional predictability)*에 영향을 미친다는 *실증적 관찰* — 이 관찰이 정보 전파 지연(Hong & Stein 1999) 가설과 일치하는지 검증
- 주장하지 않는 것:
  - BTC 사이클이 알트코인 가격에 *인과적으로* 영향을 미친다 (correlation ≠ causation)
  - arbitrage-free equilibrium 또는 새로운 systematic risk factor (이론 제안 아님)
  - FiLM이 경제학적으로 해석 가능하다 (조건부 표현 학습 메커니즘으로만 기술)
  - H2(유동성 공통 요인), H3(투자자 주의 배분) — 현재 데이터(Upbit 캔들)로 operationalize 불가
- 기여: 아키텍처 설계 + H1 가설의 실증 검증 (lead-lag predictive regression 기반)

**FF5 팩터 사용 입장**:
- FF5(size, mom, vol, liq)는 *feature engineering 레이어* — FF 이론 수용/비판/초월 주장 없음
- FF5 피처가 실제로 예측력에 기여하는지는 ablation으로 실증

---

## 1. 핵심 주장 (한 문장)

> BTC 사이클 상태(추세 위치 × 변동성 체제)를 조건으로 할 때,
> 장기 의존성 경로(Transformer)와 국소 패턴 경로(TCN)의 *조건부 예측력*이 달라진다는 실증적 관찰에 기반하여,
> 우리는 두 경로를 계층적으로 결합하고 gate를 통해 경로 의존도를 진단 가능하게 노출하는 아키텍처를 제안한다.

**"인과" 표현 금지**: 문서 전체에서 "BTC가 알트코인에 영향을 미친다"는 인과 표현 사용 금지.
대신: "BTC 사이클 상태를 조건으로 할 때 예측력이 달라진다" (조건부 예측력)로 일관 표현.

---

## 2. 이론적 프레임워크 (경제학 트랙 요구사항)

### 2.1 메커니즘 가설 (기존 문헌 기반)

**가설 H1: 정보 전파 지연 (Information Diffusion Lag)** — 유일한 검증 가설
- 근거: Hong & Stein (1999) — 정보는 큰 자산에서 작은 자산으로 점진적으로 전파됨
- Crypto 적용: BTC가 crypto 시장의 공통 정보를 먼저 반영하고, 알트코인은 시차를 두고 반응
- 검증 가능한 예측: BTC 사이클 상태(regime)를 조건으로 할 때 알트코인 수익률의 조건부 예측력이 달라진다
- Operationalization: lead-lag predictive regression으로 직접 검증 (아래 E1)

**H2/H3 제외 이유 (operationalization 실패)**:
- H2(유동성 공통 요인): FiLM scale로 liq 피처 중요도 측정 불가 — FiLM은 latent space에 작용, 입력 피처별 민감도와 무관. bid-ask spread 등 실제 유동성 측정치 없음
- H3(투자자 주의 배분): Google Trends 등 실제 attention proxy 없음. "Bull_volatile에서 MAE 증가"는 단순 변동성 효과와 구분 불가
- 측정 불가능한 가설을 포함하면 논문 전체 신뢰도 하락 → 제외

### 2.2 H1 Operationalization: Lead-lag Predictive Regression

```
# H1 직접 검증: regime-dependent lead-lag regression
alt_return[t+h] = α + β₁·BTC_return[t] + β₂·BTC_return[t]·regime[t] + ε

# β₂의 유의성 = BTC 사이클 상태에 따라 lead-lag 예측력이 달라지는가
# regime[t] = btc_ma_distance (연속) 또는 4-state 레이블 (이산)
# h = 1, 3, 6, 12h (시차별 검증)
# 검정: HAC 표준오차 (Newey-West) — 시계열 자기상관 보정
```

**사이클 전환점 정의 (MC3 방어 — 다중검정 폭탄 방지)**:
- MA200h 부호 전환은 너무 빈번 → 다중검정 폭탄
- 대신: **최소 구간 길이 강제** — 연속 168h(7일) 이상 동일 부호 유지 시에만 전환으로 인정
- 또는: btc_ma_distance를 이산 레이블 대신 **연속 변수**로 직접 regression에 투입 (전환점 정의 불필요)
- 채택: 연속 변수 투입 방식 — 전환점 정의 자의성 완전 제거

**Universe Confounding 통제 (MC4 방어)**:
- 알트코인 상장/폐지로 IC 분포가 기계적으로 변하는 문제
- 통제 방법: **고정 유니버스 서브샘플** — 분석 기간 전체에 걸쳐 지속 존재한 코인만으로 별도 분석
- 메인 분석(동적 유니버스) + 서브샘플 분석(고정 유니버스) 두 결과 모두 보고
- 두 결과가 일치하면 universe confounding이 결론을 바꾸지 않음을 입증

### 2.3 AETHER 아키텍처와 H1의 연결

```
H1 (정보 전파 지연)
  → Cycle-aware PE: btc_ma_distance를 연속 신호로 각 토큰에 주입
  → Transformer: 사이클 상태를 인식한 상태에서 장기 의존성 포착
  → FiLM: 체제별 조건부 표현 변환
  → 검증: lead-lag regression에서 β₂ 유의성 → AETHER가 이 조건부 예측력을 포착하는가
```

**E2: Markov Regime-Switching Baseline (H1 비교)**
```
방법: 2-state Markov Regime-Switching 모델 (statsmodels.tsa.regime_switching)
비교: AETHER 4-state 조건화 vs 단순 2-state Markov switching
주장: AETHER의 marginal improvement가 통계적으로 유의한가
검증: IC 차이에 Newey-West HAC t-test 적용
```

**E3: Feature Selection 정당화 (data snooping 방어)**
```
33개 피처 선택 근거 (사전 고정, HPO 탐색 아님):
  - BTC 사이클 변수 (btc_ma_distance, btc_regime_rv): H1 가설에서 직접 도출
  - FF5 스타일 (size, mom, vol, liq): Fama-French 문헌의 crypto 적용 (도메인 지식)
  - 기술적 지표 (RSI, MACD 등): 실무 표준 (도메인 지식)
  - 피처 선택 자체가 이론 가설에서 도출됨 → 암묵적 data snooping 아님
```

**E4: FiLM 체제별 기여도 측정 (MC5 방어 — latent space 측정 불가능성)**
```
문제: FiLM scale로 liq 피처 중요도 직접 측정 불가 — FiLM은 latent space에 작용
대안 (채택): 피처 그룹 ablation
  - AETHER without {liq features} vs full → 체제별 IC 감소폭 측정
  - Bear_volatile에서 liq 제거 시 IC 감소폭 > 다른 체제 → H2 방향성 지지
  - 이건 FiLM scale 직접 측정이 아니라 입력 피처 기여도 측정 → 측정 가능
주의: 이 분석은 H1 주 가설의 부수적 탐색 — 논문에서 명확히 구분
```

---

## 3. 아키텍처 인과 사슬 (구현의 생명줄)

```
[Cycle-aware PE]
  btc_ma_distance + btc_regime_rv → 각 시점 토큰이 BTC 사이클 내 위치를 인식
      ↓
[Transformer Encoder]
  사이클 위치를 인식한 상태에서 168h 전체 시퀀스의 장기 의존성 포착
  → attention_weights: "사이클 맥락에서 어느 과거 시점이 중요한가"
      ↓ (guidance prior, stop-gradient)
[Attention-guided TCN]
  Transformer가 중요하다고 한 시점을 중심으로 국소 패턴 집중 탐색
  → 두 경로가 독립 앙상블이 아니라 계층적으로 결합됨
      ↓
[Explainable Gated Fusion]
  gate = f(transformer_out, tcn_out, prototype_similarity)
  "지금 사이클 위치에서 장기 추세 vs 국소 패턴 중 어느 쪽이 더 예측력 있는가"
  gate ≈ 1: Transformer 지배 (추세 추종)
  gate ≈ 0: TCN 지배 (패턴 매칭)
      ↓
[FiLM Regime Conditioning]
  4-state 체제(Bull/Bear × quiet/volatile)로 표현 자체를 조건화
  "같은 사이클 위치라도 체제 상태에 따라 표현을 다르게 변환"
      ↓
[GAN Decoder + MC-Dropout]
  3-step residual return 분포 생성
  aleatoric (GAN noise) + epistemic (MC-Dropout) 불확실성 분리
      ↓
[Uncertainty-as-constraint Funnel]
  PI_low_80 > 0을 진입의 하드 게이트로 사용
  불확실성이 post-hoc 필터가 아니라 의사결정 조건
```

---

## 4. 각 컴포넌트의 역할 분리

| 컴포넌트 | 역할 | 표현 방식 ("경제학적 해석" 아님) |
|---|---|---|
| Cycle-aware PE | BTC 사이클 상태를 각 토큰에 조건화 | 조건부 입력 신호 (연속적) |
| Transformer | 조건화된 상태에서 장기 의존성 포착 | 시퀀스 전역 패턴 추출기 |
| TCN | attention이 가리킨 시점의 국소 패턴 탐색 | 국소 모티프 추출기 |
| Gate | 두 경로의 조건부 예측력 가중 혼합 | 진단 가능한 라우팅 변수 |
| FiLM | 이산 체제 레이블에 따른 조건부 표현 변환 | 조건부 표현 학습 (경제학적 해석 주장 안 함) |
| GAN noise | aleatoric uncertainty 모델링 | 분포 생성기 |
| MC-Dropout | epistemic uncertainty 추정 | 모델 불확실성 프록시 |

**PE와 FiLM의 역할 분리**:
- PE: 연속적 사이클 위상 신호 — btc_ma_distance, btc_regime_rv
- FiLM: 이산 체제 레이블(0~3)에 따른 affine 변환
- 둘은 다른 정보를 인코딩하므로 역할이 겹치지 않는다

**gate 순환논리 방어**:
- gate ≈ 1 = "Transformer 지배"는 *정의*가 아니라 *검증 대상 가설*이다
- 독립 검증: gate 값을 사후에 btc_ma_distance와 Spearman 상관 측정 (링크 3)
- gate 값을 모른 채 btc_ma_distance만으로 예측한 "추세 우세" 레이블과 gate 방향 일치율 측정
- 일치율이 우연 수준(50%)을 유의하게 초과하면 순환논리 반박 가능

---

## 5. Cycle-aware PE 설계 확정

**Cross-sectional Normalization 예외 규칙 (필수 명문화)**:
- cross-sectional rank normalization 적용 대상: **코인별 피처만** (size, mom, vol, liq, 기술적 지표)
- 적용 제외 대상: **전역 컨텍스트 변수** — btc_ma_distance, btc_regime_rv, market_index_return
  - 이유: 이 변수들을 cross-sectional rank에 포함하면 모든 코인이 동일한 값을 가져 rank 정보가 소실됨
  - 대신: rolling z-score (시간축 표준화)로 스케일 조정
- 논문 Method 섹션에 반드시 명시:
  > "Cross-sectional rank normalization is applied only to coin-specific features.
  > Global context variables (btc_ma_distance, btc_regime_rv, market_index_return)
  > are standardized temporally using rolling z-score and excluded from
  > cross-sectional ranking to preserve their information content."

**레짐 정의 윈도우 (스케일 불일치 방어 + data snooping 방어)**:
- `MA200` = 200시간 이동평균 (≈8.3일) — 시간봉 입력(168h)과 동일 단위
- `btc_ma_distance` = (price - MA200_h) / MA200_h — 단기 추세 내 위치, 연속적
- `btc_regime_rv` = 실현변동성 분위, rolling window = **720h (30일)**
  - 두 윈도우가 다른 이유: 추세는 빠른 전환, 변동성 체제는 느린 전환 — 스케일 분리는 의도적
- 4-state 레짐 레이블(FiLM용): Bull/Bear = MA200_h 기준, quiet/volatile = 720h RV 중앙값 기준

**윈도우 임의성 방어 (data snooping 공격 선제 차단)**:
- MA200(200h), RV(720h)는 HPO 탐색 대상이 아니라 **사전 고정(pre-specified)** 값
- 근거: 입력 시퀀스 168h의 약 1.2배(MA200) / 약 4.3배(RV720) — 입력 길이 대비 합리적 배수
- 이 값들이 최적이라고 주장하지 않는다. 주장: "이 값으로 정의된 사이클 상태가 조건부 예측력에 영향을 미치는가"
- 민감도 분석: MA {168h, 200h, 336h} × RV {504h, 720h, 1008h} 격자 실험으로 결과 안정성 보고
  → 특정 값에서만 결과가 나오면 data snooping 의심 → 넓은 범위에서 일관되면 방어 가능

```python
# 사이클 위치를 정의하는 두 연속 신호 (이미 피처에 있음)
cycle_position  = btc_ma_distance   # (price - MA200_h) / MA200_h  → 추세 내 위치
cycle_intensity = btc_regime_rv     # 720h rolling RV 분위 → 변동성 강도

# PE 구성
cycle_context = [cycle_position, cycle_intensity]  # (B, seq, 2)
PE_cycle = cycle_context @ W_phase                  # (B, seq, d_model)
x = x + PE_static + PE_cycle

# 핵심: t 시점의 PE는 t-1까지의 데이터로만 계산 → hindsight bias 없음
# btc_ma_distance, btc_regime_rv 모두 실시간 계산 가능한 객관적 지표
```

**historical_similarity의 역할**:
- PE의 세 번째 신호로 추가 가능: "현재 윈도우가 과거 BTC 사이클 중 어느 구간과 유사한가"
- 단, 정의를 먼저 확정해야 함 (표현 공간, 거리 함수, 메모리 뱅크)
- 현재는 btc_ma_distance + btc_regime_rv 2개로 시작, historical_similarity는 ablation으로 기여 검증

---

## 6. 검증 실험 전체 (ML ablation + 경제학 가설 검증)

각 링크가 실제로 작동하는지 검증하는 실험이 논문 본체다.

### 링크 1: PE가 사이클 위치를 실제로 인코딩하는가
```
검증: attention 패턴이 체제별로 달라지는가
  Bull_quiet → attention이 최근 시점에 집중 (추세 추종)
  Bear_volatile → attention이 원거리 시점에 집중 (패턴 매칭)
실험: attention heatmap per regime
```

### 링크 2: attention이 TCN을 실제로 가이드하는가
```
검증: attention-guided TCN vs random importance TCN 성능 비교
실험: ablation - attention_guided (random importance로 교체)
```

### 링크 3: gate가 사이클 위치와 실제로 상관관계가 있는가
```
검증 A — Spearman 상관 (pseudo-replication 방지):
  집계 단위: 코인별이 아니라 시간별 (timestamp 기준)
  gate_by_time = gate_values.groupby('timestamp').mean()
  btc_dist_by_time = btc_ma_distance.groupby('timestamp').first()
  Spearman(gate_by_time, btc_dist_by_time) > 0 검증
  → 동일 시점 t가 다중 코인에서 반복되는 pseudo-replication 방지

검증 B — Oracle Consistency Test (gate 유용성 실증):
  oracle_label = (transformer_error < tcn_error).astype(int)  # 어느 expert가 더 정확했는가
  gate_binary = (gate_values > 0.5).astype(int)
  consistency = accuracy_score(oracle_label, gate_binary)  # 우연(50%) 대비 유의한 초과 여부
  auc = roc_auc_score(oracle_label, gate_values)           # AUC > 0.5 검증
  → Spearman만으로는 "gate가 더 좋은 expert를 선택하는가" 검증 불가
  → Oracle test가 gate의 실용적 유용성을 직접 검증
실험: 두 검증 모두 보고 (Spearman + Oracle AUC)
```

### 링크 4: FiLM이 체제별로 다른 표현을 만드는가
```
검증: FiLM 있음 vs 없음 성능 비교, 체제별 분해
실험: ablation - no_film, regime × metric table
```

### ML ablation 테이블
```
Full AETHER
  vs. Cycle-aware PE → static PE only        (링크 1 검증)
  vs. attention_guided → random importance   (링크 2 검증)
  vs. gate → simple average fusion           (링크 3 검증)
  vs. FiLM → no regime conditioning          (링크 4 검증)
  vs. Transformer-only (no TCN)
  vs. TCN-only (no Transformer)
  vs. DLinear cross-sectional (외부 baseline)
  vs. Quantile regression (GAN 필요성 검증)
  vs. AETHER without FF5 features            (FF5 기여 검증)
  prototype NN 윈도우 해석 실험              (prototype 해석가능성 검증)
```

### 경제학 가설 검증 테이블 (H1만)
```
H1 (정보 전파 지연) — lead-lag predictive regression:
  β₂ 유의성: BTC_return[t] × regime[t] 상호작용항 HAC t-test
  시차별(h=1,3,6,12h) β₂ 패턴: 단기 시차에서 유의, 장기에서 소멸 (전파 창 확인)
  고정 유니버스 서브샘플 재현: universe confounding 통제
  AETHER vs Markov Regime-Switching baseline: marginal improvement (HAC t-test)
```

**결과 분리 원칙 (funnel이 결과를 주도하는 반박 방어)**:
- 순수 예측 성능 (IC, MAE, PI_80 coverage): 모델 단독 평가
- 전략 성과 (Sharpe, MDD, Calmar): funnel 적용 후 평가
- 두 테이블을 논문에서 반드시 분리 보고 — "funnel이 다 했다" 반박 차단

---

## 7. 과적합 방어 (심사자 우려 선제 차단)

```
모델 규모: ~3M 파라미터 (경량)
  → 단일 CPU 서버에서 실시간 추론 가능 (현재 운영 중)

Purged Walk-Forward CV + 6h embargo
  → 각 fold에서 미래 정보 없이 검증
  → horizon=3h 대비 embargo=6h는 충분한 분리 (López de Prado 기준)
```

**Multiple Testing 문제 (Gu, Kelly, Xiu 지적 정면 대응)**:
- 문제: 9 ablation × HPO trials = 수백 개 모델 테스트 → DSR 단독으로 불충분
- 대응 계층:
  1. **사전 등록(pre-registration)**: 핵심 4개 링크 검증 실험을 코드 실행 전 문서화 (이 문서가 그 역할)
  2. **HPO 목적함수 분리**: HPO는 IC/NLL만, Sharpe는 최종 held-out에서만 — HPO 루프에서 Sharpe 완전 배제
  3. **Bonferroni 보정**: 9개 ablation 비교에 α/9 적용 (보수적)
  4. **DSR 보정**: 최종 성과 보고 시 Deflated Sharpe Ratio (Bailey & López de Prado 2014) 적용
  5. **Out-of-sample 고정**: 마지막 fold는 HPO/ablation 탐색 중 절대 열람 금지 — 최종 1회만 평가
- 인정: 이 규모의 탐색에서 multiple testing을 완전히 제거할 수 없다. 논문에서 이를 명시적 한계로 기술.

**HPO 목적함수 원칙**:
- HPO(Optuna) 목적함수: **IC** 또는 **NLL** — 예측 품질 지표만
- Sharpe/MDD/Calmar는 최종 평가(held-out)에서만 보고

**유니버스 선정 시간정합성 원칙 (survivorship/look-ahead 방어)**:
- `valid_coins` 선정은 **각 fold의 train 구간 종료 시점 기준**으로만 수행
- 전체 기간 `notna().sum() > threshold` 필터 금지 — 미래 생존 코인 look-ahead 발생
- `market_index_return` 가중치: **각 fold train 구간의 rolling 30일 평균 거래대금** 기준

---

## 8. 진단 가능성 설계 ("Explainable" 주장 대신)

**표현 원칙**: "Explainable AI"를 주장하지 않는다. 대신 "diagnostic interpretability" — 모델 동작을 사후 진단할 수 있는 변수를 노출한다.

```
gate_value (진단 변수):
  두 경로의 혼합 비율 — 사후 분석 대상
  gate ≈ 1: Transformer 경로 가중치 높음
  gate ≈ 0: TCN 경로 가중치 높음
  → 이것이 "추세 지배"를 의미하는지는 링크 3 실험으로 검증 (정의가 아님)

  gate regularization: prototype diversity 강제
  L_gate_reg = +λ × mean(proto_sim_matrix[off-diagonal])
  → 부호: + (패널티), 최소화 목표 → prototype 다양성 강제 (부호 충돌 없음)

FiLM scale/shift (조건부 표현 변환):
  이산 체제 레이블에 따른 affine 변환 — 경제학적 해석 주장 안 함
  검증: FiLM 있음 vs 없음 성능 비교 (링크 4)

attention weights (guidance prior):
  explanation이 아님 — Jain & Wallace (2019) 논쟁 선제 회피
  TCN 입력 가중치로만 기능 — diagnostic 용도

btc_ma_distance:
  (price - MA200_h) / MA200_h  (MA200_h = 200시간)
  연속적, 실시간 계산 가능, hindsight bias 없음
```

**손실함수 설계 원칙 (미분가능성/학습 일관성)**:

**예측 타겟 정의 (수학적 명확성 필수)**:
- 타겟: `cumulative_3h_return` — 3시간 누적 수익률 **스칼라** (3-step vector 아님)
- 이유: CRPS는 univariate proper scoring rule. 3-step vector에 직접 적용 불가.
- 만약 multi-step 분포가 필요하면: Energy Score (Gneiting & Raftery 2007) 사용
  - `ES(F, y) = E||X - y|| - (1/2)E||X - X'||` (X, X' ~ F)
  - Energy Score는 multivariate proper scoring rule — 수학적으로 명확
- **채택: 스칼라 타겟(cumulative_3h_return) + CRPS** — 수학적으로 가장 명확하고 방어 용이

- `L_direction`: `sign(pred)` 직접 사용 금지 (거의 모든 점에서 gradient = 0)
  - 채택: `tanh(k · pred)` 연속 근사 (k는 하이퍼파라미터)
- `L_ece`: ECE는 binning 기반 평가 지표 → 비미분 → **학습 손실로 사용 금지**
  - ECE는 평가(보고) 지표로만 사용
  - 불확실성 학습: **CRPS** (univariate 스칼라 타겟에 적용, proper scoring rule)
- GAN 학습 절차: critic과 generator **분리 업데이트** (WGAN-GP 표준 절차)
  - `L_total` 표기는 generator loss만 지칭 — critic loss는 별도 루프
  - 논문 Method에 critic/generator 업데이트 절차를 pseudocode로 명시

**GAN vs Quantile Regression 비교 기준 (GAN 필요성 실증)**:
- ablation: AETHER(GAN) vs Quantile Regression — 동일 타겟, 동일 입력
- 비교 지표 3개 (모두 보고 필수):
  1. Calibration: Reliability diagram — 동일 coverage에서 어느 쪽이 더 잘 보정되는가
  2. Sharpness: 동일 coverage(80%)에서 예측 구간 폭 비교 — 좁을수록 좋음
  3. Uncertainty decomposition: MC-Dropout epistemic vs GAN aleatoric 분리가 의미있는가
     (epistemic이 데이터 희소 구간에서 높아지는지 확인)

**prototype 해석가능성 검증 (심사자 E2 방어)**:
- "prototype이 해석 가능하다"는 주장은 자동으로 성립하지 않음
- 검증 방법: 각 prototype에 대해 latent 거리 기준 nearest-neighbor 실제 시계열 윈도우 Top-5 제시
- 정량화: Top-5 윈도우의 공통 구조 레이블(추세/역추세/급락/거래대금 급증) 분포 보고
- ablation 섹션 5에 추가: prototype NN 윈도우 해석 실험

---

## 9. FF5 팩터의 역할 재정의

FF5 팩터(size, mom, vol, liq)는 논문의 핵심 기여가 아니라 **도메인 지식 피처**다.

```
역할: 33개 입력 피처 중 일부 (feature engineering 레이어)
주장: "시장 구조 정보를 모델에 제공하는 도메인 지식"
주장하지 않는 것: FF 이론을 수용하거나 비판하거나 넘어선다
ablation: AETHER with FF5 vs without FF5 → FF5 피처 기여 실증
```

**Occam's Razor 방어 (DLinear 공격 선제 차단)**:
- DLinear를 **동일 입력 · 동일 예측 타겟**으로 구현하여 공정하게 비교
- AETHER는 추가로 gate/attention 기반 진단 변수와 체제별 성능 분해를 제공한다
- DLinear가 제공하지 못하는 것(PI_80, gate 진단, 체제별 성능 분해)을 별도 테이블로 제시
- 통계적 유의성: IC 차이에 대해 Newey-West HAC t-test (시계열 자기상관 보정) 적용

---

## 10. 다음 단계 (순서 고정)

```
Phase 0 — 문서 확정 (지금)
  1. AETHER_IDEA.md 확정 (이 문서)

Phase 1 — 누수 제거 (선행 필수)
  2. leakage 버그 수정 (DESIGN_SPEC.md Phase 1) — 4개:
     a. 기존 명시된 버그 3개
     b. valid_coins 선정 look-ahead: fold train 구간 종료 시점 기준으로 교체
  3. market_index_return 가중치 산출을 fold-wise rolling 30일로 고정
  4. BTC/ETH 백필 실행 (2017-01-01~)

Phase 2 — 아키텍처 구현
  5. Cycle-aware PE 구현 확정
     - context_dim=2: btc_ma_distance (MA200_h) + btc_regime_rv (720h RV 분위)
     - historical_similarity는 ablation으로 기여 검증 후 포함 결정
  6. 손실함수 교체: L_direction → tanh(k·pred), L_ece → CRPS/quantile loss
  7. GAN 학습 절차 pseudocode 작성 (critic/generator 분리 업데이트 명시)
  8. ablation flag 구현 (prototype NN 해석 실험 포함)

Phase 3 — 경제학 검증 구현 (ML 트랙과 병렬 가능)
  9. Bai-Perron structural break test 구현 (H1 검증)
     - 라이브러리: ruptures 또는 statsmodels
     - 대상: BTC 사이클 전환점 전후 알트코인 IC 변화
  10. Markov Regime-Switching baseline 구현 (H1/H2 비교)
      - statsmodels.tsa.regime_switching.MarkovRegression
      - AETHER vs 2-state Markov, HAC t-test로 marginal improvement 검증
  11. 고정 유니버스 서브샘플 구성 (MC4 universe confounding 통제)

Phase 4 — 실험 실행 + 논문 작성
  12. ML ablation 실험 실행 + 인과 사슬 각 링크 검증
  13. H1 lead-lag regression 실험 (연속 btc_ma_distance 투입, HAC t-test)
  14. Markov Regime-Switching baseline 비교
  15. 민감도 분석: MA {168h, 200h, 336h} × RV {504h, 720h, 1008h} 격자
  16. 논문 작성:
      - 예측 성능 테이블 / 전략 성과 테이블 분리
      - H1 검증 테이블 (시차별 β₂, 고정 유니버스 재현)
      - Section 2: Theoretical Framework (H1 only)
      - Section 5: H1 Empirical Validation
```
