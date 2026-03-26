# 논문 3편 엄격 감사 및 실행 가능성 평가

> 작성일: 2026-03-26
> 목적: Paper A/B/C의 숫자 검증, 방법론 결함, 경쟁 논문 대조, 리뷰어 반박 시뮬레이션

---

## 0. 발견된 핵심 문제

### 문제 1: 데이터 누수 의혹 — 검증 결과 **경미**

에이전트가 `build_split()`에서 look-ahead를 지적했으나, 코드를 직접 확인한 결과:

```python
# 입력: market_arr[t - seq_len : t]  → Python 슬라이싱으로 인덱스 t-seq_len ~ t-1 (t 미포함)
# 타겟: target.iloc[t, asset_idx]
# target 정의: resid.rolling(horizon).sum().shift(-horizon)  → t시점의 target = t+1 ~ t+horizon의 미래 수익률 합
```

**판정:** 입력은 t-1까지, 타겟은 t+1~t+h. 직접적 look-ahead는 없다.

**그러나 남은 우려:**
1. `mean5`, `std5` = rolling(5) 통계가 `t-1` 시점에서 계산되지만, 이 rolling 자체가 미래를 포함하는지? → `raw[t-seq_len:t]`에서 가져오므로 t 미포함. **OK.**
2. `intensity`, `position` = MA200, RV30 기반. 이것들은 전체 시계열에서 한번에 계산됨. → rolling window 기반이므로 시점 t에서 과거만 사용. **OK.**
3. **진짜 문제:** train/test 분할이 70/15/15 날짜 기준인데, **embargo(purge gap) 없음.** 시퀀스 길이 30일 + 예측 horizon 5일 = 35일의 겹침 가능성. 훈련 마지막 날짜와 테스트 첫 날짜 사이에 최소 35일 간격이 필요하지만, 현재 코드에는 없다.

**심각도:** 중간. 완전한 look-ahead는 아니지만, purge/embargo 없는 설계는 리뷰어가 반드시 지적할 사항. 그러나 모든 모델이 동일한 분할을 사용하므로 **모델 간 비교의 상대적 순위는 유지**될 가능성이 높다.

### 문제 2: StretchTime (2026.02) — Paper B의 직접 경쟁자

**StretchTime: Adaptive Time Series Forecasting via Symplectic Attention (arXiv 2602.08983)**
- RoPE를 SO(2)에서 Sp(2,R)로 확장한 SyPE 제안
- **RoPE가 비선형(non-affine) 시간 워핑을 표현할 수 없음을 수학적으로 증명**
- 입력 의존적 적응형 워프 모듈 포함

**Paper B에 대한 영향:**
- tau-RoPE의 수학적 한계를 직접 증명한 논문이 이미 존재
- Paper B가 "tau-RoPE가 고변동성에서 MAE 개선"을 주장하는데, StretchTime은 "RoPE 자체가 비선형 워핑 불가"라고 증명
- **반드시 인용하고 차별화해야 함**

**차별화 가능한 부분:**
- StretchTime은 일반 시계열 벤치마크, Paper B는 금융 특화 + Clark(1973) 이론 동기
- StretchTime은 SyPE라는 새 아키텍처 제안, Paper B는 기존 RoPE의 한계 안에서 무엇이 되는지 실증
- 그러나 이 차별화가 충분한지는 의문

### 문제 3: 표본 크기

- 확증 실험 고변동성 관측치: **220일** (GSPC 109, IXIC 111)
- Ken French 25 포트폴리오 = 교차 단면 25개뿐
- 이것으로 p=0.0019를 주장하는 것이 통계적으로 충분한가?
- Newey-West lag=4로 자기상관 보정은 했으나, 다중비교 보정(Bonferroni 등)은 없음

---

## 1. Paper A 엄격 평가: "왜 concat이 이기는가"

### 1.1 숫자 검증 ✓

| 주장 | CSV 저장값 | 일치 |
|------|-----------|------|
| concat_a IC = 0.0571 | 0.057147 | ✓ |
| intensity_only IC = 0.0066 | 0.006596 | ✓ |
| intensity+indexret IC = 0.0592 | 0.059194 | ✓ |
| position_only IC = 0.0188 | 0.018780 | ✓ |
| shuffled_intensity IC = 0.0187 | 0.018662 | ✓ |
| no_intensity IC = 0.0026 | 0.002590 | ✓ |

**모든 핵심 숫자 정확히 일치.**

### 1.2 방법론 결함

**심각한 문제:**
1. **GSPC 단일 시장, 2022-2024만.** 채널 분해 실험이 GSPC에서만 수행됨. IXIC 재현 없음.
2. **F=14.335 (finance incremental)은 별도 데이터셋** (Fama-French WML). 채널 분해 실험(Ken French 25)과 데이터가 다름. 두 실험의 결론을 직접 연결하는 것은 비약.
3. **"early interaction hypothesis"는 검증이 아닌 사후 해석.** concat이 이기는 것을 관찰한 후 왜 이기는지 설명하는 것이지, 가설을 먼저 세우고 검증한 것이 아님.
4. **XIP가 concat을 못 이기는 것이 hypothesis를 지지하는가?** XIP가 explicit interaction을 포함하는데 concat과 동등하다면, concat이 이미 interaction을 발견한다는 증거일 수 있지만, interaction이 이유라는 직접 증거는 아님.

**경미한 문제:**
5. FiLM, learned_tau 결과의 시드 수가 일관적이지 않음 (일부 1시드)
6. 파라미터 수 매칭이 5% 이내라고 주장하나 정확한 수치 제시 부족

### 1.3 경쟁 논문 대조

| 경쟁 논문 | 위협도 | 이유 |
|----------|--------|------|
| DGFET (2025) | 낮음 | FiLM을 사용하지만 인터페이스 비교 안 함 |
| Kelly et al. NBER (2025) | 낮음 | 교차 자산 attention, 체제 주입 질문이 다름 |
| CB-APM (2025) | 낮음 | bottleneck 인터페이스, 비교 대상에 추가 가능 |

**novelty 판정:** 5가지 인터페이스의 체계적 비교 + early interaction hypothesis는 **현재까지 직접 경쟁 논문 없음**. Novelty 유지.

### 1.4 리뷰어 반박 시뮬레이션

**R1 (방법론):** "early interaction hypothesis는 사후 합리화에 불과하다. 가설을 먼저 세우고 검증하지 않았다."
→ **방어 어려움.** 대안: 새 데이터셋에서 사전등록 확증 실험 필요.

**R2 (일반성):** "25개 포트폴리오에서의 결과가 개별 주식 3000개에서도 성립하는가?"
→ **방어 불가.** 실험 확장 필요.

**R3 (인과):** "concat이 이기는 것과 interaction이 이유인 것은 다르다. Causal identification이 없다."
→ **부분 방어 가능.** XIP ablation + channel decomposition이 간접 증거. 그러나 완전한 인과 증명은 아님.

**R4 (trivial):** "concat이 더 많은 정보를 첫 층에 넣으니 이기는 건 당연하지 않나?"
→ **가장 위험한 반박.** "당연한 결과"로 치부될 위험. 대응: FiLM도 같은 정보를 넣지만 진다, XIP도 같은 정보 + explicit interaction을 넣지만 이기지 못한다 → 정보량이 아니라 접근 방식의 문제.

### 1.5 발행 가능성 판정

| 항목 | 상태 |
|------|------|
| 핵심 주장의 증거 | ✓ 숫자 정확, 패턴 일관 |
| 방법론 건전성 | △ purge 없음, 단일 시장 |
| Novelty | ✓ 직접 경쟁 없음 |
| 리뷰어 방어 | △ "trivial" 반박 대응 필요 |
| **필요한 추가 작업** | IXIC 재현, 개별주식 확장 실험, 사전등록 확증 |

**결론: 현재 상태 = workshop paper 수준. 추가 실험 2-3개로 TMLR/응용 저널 수준 가능.**

---

## 2. Paper B 엄격 평가: "경제적 시간 applied paper"

### 2.1 숫자 검증 ✓

| 주장 | CSV 저장값 | 일치 |
|------|-----------|------|
| H1 고변동성 IC p=0.0595 | 0.05948 | ✓ |
| H2 고변동성 MAE p=0.0019 | 0.00192 | ✓ |
| GSPC 고변동성 IC delta | +0.0378 | ✓ |
| IXIC 고변동성 IC delta | +0.0396 | ✓ |

**핵심 숫자 정확히 일치.**

### 2.2 방법론 결함

**심각한 문제:**
1. **purge/embargo 없음.** 동일 문제. seq_len=30 + horizon=5 = 35일 겹침 가능. 이것이 모델 간 비교를 왜곡하지는 않지만, 절대 성능 수치는 과대추정 가능.
2. **확증 기간의 실제 범위.** 파일명은 "confirmatory_2020_2024"이지만, 실제 OOS 테스트 데이터는 **2024년 4-12월만** (walk-forward 최신 fold). "2020-2024 전체"가 아님.
3. **IC p=0.0595가 5% 기준 미달.** 논문에서는 "방향적 지지"로 표현했지만, 엄격한 리뷰어는 이것을 실패로 볼 것.
4. **고변동성 관측 220일.** 표본 작음. 소수의 극단 이벤트에 의해 결과가 좌우될 수 있음.

**심각한 외부 위협:**
5. **StretchTime(2026.02)이 RoPE의 비선형 워핑 불가능을 증명.** tau-RoPE가 실제로 하는 일은 선형 스케일링에 가까움 (tau_corr=0.998). StretchTime의 증명이 tau-RoPE의 이론적 기반을 약화시킴.

### 2.3 경쟁 논문 대조

| 경쟁 논문 | 위협도 | 이유 |
|----------|--------|------|
| **StretchTime (2026.02)** | **높음** | RoPE 비선형 워핑 불가 증명, SyPE 제안 |
| KAIROS/DRoPE (2025.09) | 중간 | 적응형 RoPE이나 스펙트럴 기반 |
| ElasTST (NeurIPS 2024) | 중간 | 조정 가능 RoPE이나 horizon 적응 목적 |
| RoMAE (2025.05) | 낮음 | 불규칙 시계열용 시간 인식 RoPE |

**novelty 판정:** StretchTime 존재로 "RoPE 기반 시간 워핑" 자체의 novelty는 약화됨. 그러나 Clark(1973) 금융 이론 동기 + 체제별 트레이드오프 실증은 StretchTime에 없음.

### 2.4 리뷰어 반박 시뮬레이션

**R1 (통계):** "IC p=0.0595는 유의하지 않다. MAE만 유의한 논문이 publishable한가?"
→ **방어 가능하지만 약함.** MAE p=0.0019는 강하고 두 시장 재현. IC가 보조 지표라고 재프레이밍 가능.

**R2 (경쟁):** "StretchTime이 이미 RoPE의 한계를 증명하고 더 좋은 대안(SyPE)을 제안했다. 왜 제한된 tau-RoPE를 쓰는가?"
→ **방어 어려움.** 대안: "우리 논문의 기여는 방법이 아니라 conditioning space라는 개념적 프레이밍이다" 또는 SyPE를 구현하여 tau-SyPE로 확장.

**R3 (trivial):** "고변동성에서 MAE가 좋다는 것은 tau가 단순히 시계열을 더 smooth하게 만든 것일 수 있다."
→ **방어 필요.** Smoothing 효과와의 분리 실험 필요.

**R4 (범위):** "25 포트폴리오, 220일 고변동성 관측, 2개 시장. 너무 좁다."
→ **방어 불가.** 확장 필요.

### 2.5 발행 가능성 판정

| 항목 | 상태 |
|------|------|
| 핵심 주장의 증거 | ✓ MAE는 강함, IC는 약함 |
| 방법론 건전성 | △ purge 없음, 확증 기간 오해 소지 |
| Novelty | △ StretchTime으로 약화, 프레이밍으로 차별화 가능 |
| 리뷰어 방어 | ✗ StretchTime + 좁은 범위 이중 공격 |
| **필요한 추가 작업** | StretchTime 인용/차별화, SyPE 확장, 추가 시장, purge 추가 |

**결론: 현재 상태 = workshop paper 하단. StretchTime 대응 없이는 리젝 위험 높음. SyPE 확장 + 추가 시장으로 살릴 수 있으나 작업량 많음.**

---

## 3. Paper C 엄격 평가: "실패 분석"

### 3.1 숫자 검증

| 주장 | 검증 |
|------|------|
| learned_tau IC=0.0106 (smoke) | ⚠️ CSV에 직접 저장 안 됨. 방향 문서에서만 인용 |
| qk_ord_rate=0.992 | ⚠️ CSV에 직접 저장 안 됨 |
| concat_a IC=0.0571 | ✓ CSV 확인 |
| step_intensity_spearman=-0.50 | ⚠️ 로그에서만 |
| FiLM range -0.08~+0.08 | ⚠️ 시드별 저장 필요 |

**문제: 핵심 부정적 결과의 상당수가 CSV로 저장되어 있지 않다.** 재현 가능성을 위해 모든 결과를 재실행하고 저장해야 함.

### 3.2 방법론 결함

**심각한 문제:**
1. **재현 불가.** 핵심 실패 결과(learned_tau, FiLM 시드별, qk_ord_rate)의 원본 CSV가 없거나 불완전.
2. **"왜 실패하는가"의 인과 추론이 약함.** 세 병목을 식별했지만, 각 병목을 독립적으로 제거하고 성능이 개선되는지 보여주지 않았음. 예: softmax를 linear attention으로 바꾸면 tau_rope가 이기는가? 단조성을 제거하면? 이런 "수술적 개입" 실험이 없음.
3. **비교군 부족.** 다른 시간 워핑 방법(StretchTime의 SyPE, Neural ODE 등)과의 비교 없이 "시간 워핑이 실패한다"고 일반화하는 것은 과도함.

**경미한 문제:**
4. H1-H10 가설 중 일부(H5, H8)의 관련성이 핵심 스토리와 약함
5. 부정적 결과 논문의 타겟 venue가 제한적

### 3.3 경쟁 논문 대조

| 경쟁 논문 | 위협도 | 이유 |
|----------|--------|------|
| Curse of Attention (CPAL 2025) | 중간 | 다른 실패 메커니즘이지만 같은 "Transformer 실패" 카테고리 |
| Closer Look at Transformers (ICML 2025) | 중간 | "복잡한 것이 안 좋다" 결론 유사 |
| PE Intriguing Properties (2024) | 낮음 | PE 정보 소실 발견, 보완적 |
| StretchTime (2026) | **높음** | RoPE 비선형 워핑 불가를 이론적으로 증명 → Paper C의 경험적 실패를 이론적으로 설명 |

**novelty 판정:** StretchTime이 softmax/RoPE 한계를 이론적으로 증명했다면, Paper C의 경험적 실패 분석은 "이미 알려진 이론적 한계의 경험적 확인"에 그칠 수 있음. 반면, 세 가지 구체적 병목(특히 "상호작용 접근 제약")은 StretchTime에 없는 금융 도메인 특화 통찰.

### 3.4 리뷰어 반박 시뮬레이션

**R1 (so what):** "learned tau가 안 된다는 건 알겠다. 그래서 뭘 해야 하는가? 해결책 없는 실패 보고의 가치는?"
→ **부분 방어.** 세 가지 필요조건 제시. 그러나 실제로 하나라도 해결하고 "이렇게 하면 된다"를 보여주면 훨씬 강해짐.

**R2 (범위):** "RoPE 기반 tau만 테스트했다. ALiBi-tau, Linear Attention-tau는?"
→ **방어 불가.** 추가 실험 필요.

**R3 (StretchTime):** "StretchTime이 이미 RoPE의 한계를 이론적으로 증명했다. 경험적 실패 분석의 추가 기여가 뭔가?"
→ **방어 가능하지만 약함.** "이론적 한계와 실제 실패 메커니즘은 다르다" + "금융 도메인 특화 상호작용 병목은 새로운 발견."

### 3.5 발행 가능성 판정

| 항목 | 상태 |
|------|------|
| 핵심 주장의 증거 | △ 방향은 맞지만 CSV 부족 |
| 방법론 건전성 | ✗ 수술적 개입 실험 없음 |
| Novelty | △ StretchTime으로 부분 약화 |
| 리뷰어 방어 | △ "so what" + StretchTime 이중 공격 |
| **필요한 추가 작업** | 결과 재생성/저장, 수술적 개입 실험(linear attn, 비단조), 해결책 1개 제시 |

**결론: 현재 상태 = 아이디어 수준. 수술적 개입 실험 + 해결책 데모 추가 시 workshop ~ TMLR 가능.**

---

## 4. 종합 판정

### 현재 발행 가능성 순위

| 순위 | 논문 | 현재 수준 | 필요 작업량 | 목표 venue |
|------|------|----------|-----------|-----------|
| 1 | **Paper A** (인터페이스) | Workshop | 중간 (2-4주) | TMLR / 응용 ML |
| 2 | **Paper C** (실패 분석) | 아이디어 | 높음 (4-6주) | NeurIPS negative results |
| 3 | **Paper B** (경제적 시간) | Workshop 하단 | 매우 높음 (6-8주) | FinML workshop |

### Paper A가 1순위인 이유:
1. 직접 경쟁 논문이 없다
2. 숫자가 모두 정확하다
3. 추가 실험이 비교적 단순하다 (IXIC 재현 + 개별주식 확장)
4. "trivial" 반박만 대응하면 된다

### Paper B가 가장 어려운 이유:
1. StretchTime이 이론적 기반을 직접 공격
2. IC p=0.0595 미달
3. SyPE 확장이 상당한 구현 작업
4. 확증 기간 오해 소지 있는 라벨링

### Paper C의 잠재력:
- 수술적 개입 실험 1개만 성공하면(예: linear attention + tau → 성능 개선) 스토리가 급격히 강해짐
- "문제 식별 + 해결" 구조가 되면 negative results가 아닌 정규 논문으로 전환 가능

---

## 5. 각 논문에 필요한 구체적 추가 작업

### Paper A 필수 작업

| # | 작업 | 소요 | 위험 |
|---|------|------|------|
| A1 | IXIC에서 채널 분해 재현 | 1일 | 낮음 (동일 코드) |
| A2 | purge/embargo 추가 후 결과 변화 확인 | 2일 | 중간 |
| A3 | 개별 주식 확장 (S&P 500 컴포넌트) | 1주 | 높음 (데이터 필요) |
| A4 | "trivial" 반박 대응 실험 설계 | 1일 | 낮음 |
| A5 | finance incremental ↔ 채널 분해 연결 정당화 | 1일 | 낮음 |

### Paper B 필수 작업

| # | 작업 | 소요 | 위험 |
|---|------|------|------|
| B1 | StretchTime 인용 및 차별화 논거 | 1일 | 중간 |
| B2 | 확증 기간 라벨 정정 (2024 OOS) | 즉시 | 없음 |
| B3 | purge/embargo 추가 | 2일 | 중간 |
| B4 | 추가 시장 2개 이상 (Russell 2000, FTSE) | 2주 | 높음 |
| B5 | SyPE 구현 및 tau-SyPE 실험 (optional) | 3주 | 매우 높음 |
| B6 | smoothing 효과 분리 실험 | 3일 | 중간 |

### Paper C 필수 작업

| # | 작업 | 소요 | 위험 |
|---|------|------|------|
| C1 | 모든 핵심 결과 CSV 재생성/저장 | 3일 | 낮음 |
| C2 | 수술적 개입: softmax → linear attention + tau | 1주 | 높음 |
| C3 | 수술적 개입: 비단조 tau (cumsum 제거) | 1주 | 높음 |
| C4 | 수술적 개입: tau + concat 결합 (상호작용 접근 확보) | 3일 | 중간 |
| C5 | StretchTime 인용 및 이론적 맥락 | 1일 | 낮음 |
| C6 | ALiBi-tau 추가 실험 | 3일 | 중간 |

---

## 6. 가장 효율적인 경로

현 상태에서 가장 효율적인 경로는:

### 1단계 (1주): Paper A 보강
- A1 (IXIC 재현) + A2 (purge 추가) 실행
- 결과 변화 없으면 → Paper A를 주력으로

### 2단계 (2주): Paper C 수술적 개입
- C4 (tau + concat 결합) 먼저 — 가장 빠르고 성공 가능성 높음
- 성공 시 → Paper C가 "실패 분석 + 해결" 논문으로 승격
- C4 성공이면 Paper B도 강화됨 (concat + tau 앙상블이 최적이라는 스토리)

### 3단계 (이후): 통합 판단
- Paper A + C 결합 가능성 검토 (인터페이스 비교 + 실패 분석 + 해결)
- Paper B는 독립 진행 또는 보류

---

## 7. 읽어야 할 추가 논문 (반드시 인용)

| 논문 | arXiv/venue | 왜 필요 |
|------|-------------|---------|
| StretchTime (2026.02) | 2602.08983 | Paper B/C 직접 경쟁 |
| Curse of Attention (2025) | 2412.06061 | Paper C 관련 |
| PE Intriguing Properties (2024) | 2404.10337 | Paper A/C 관련 |
| PE Survey for TS (2025) | 2502.12370 | 전체 배경 |
| Closer Look at Transformers (ICML 2025) | openreview | Paper C 관련 |
| Re(Visiting) TSFMs in Finance (2025) | 2511.18578 | 전체 배경 |
| KAIROS/DRoPE (2025) | 2509.25826 | Paper B 경쟁 |
| ElasTST (NeurIPS 2024) | 2411.01842 | Paper B 경쟁 |
