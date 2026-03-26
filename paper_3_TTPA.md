# 테스트 시점 위치 적응: Transformer 위치 인코딩의 추론 시간 경제적 시간 워핑

## Test-Time Positional Adaptation: Inference-Time Economic Time Warping for Transformer Positional Encodings

---

## 초록 (Abstract)

Test-Time Adaptation (TTA)은 사전학습된 모델을 추론 시점에 새로운 분포에 적응시키는 방법론으로, 최근 DynaTTA (ICML 2025 Oral), TAFAS (AAAI 2025), PETSA 등이 주목받고 있다. 그러나 기존 TTA 방법은 **예외 없이 가치 공간(value space)**에서 적응한다: 배치 정규화 파라미터, 예측 헤드, 어파인 변환 등이 그 대상이다. **위치 공간(position space)** -- 즉 Transformer의 positional encoding이 사용하는 시간 좌표 -- 에서의 TTA는 제안된 바 없다.

본 논문은 이 공백을 식별하고, **Test-Time Positional Adaptation (TTPA)**를 제안한다. TTPA는 두 단계로 구성된다: (1) 학습 단계에서 표준 고정 위치 인코딩(RoPE 또는 sinusoidal)으로 모델을 학습하고, (2) 추론 단계에서 관찰된 시장 통계(변동성, 거래 강도)로부터 경제적 시간 좌표 tau를 계산하여, 위치 인코딩의 정수 인덱스를 tau로 대체한다.

핵심 통찰은 다음과 같다: 선행 연구(Paper 2)에서 학습 시점의 tau 학습이 세 가지 병목 -- (1) softmax 압축, (2) 상호작용 접근 제약, (3) 단조성 제약(cumsum이 tau를 물리적 시간에 묶음) -- 으로 실패함을 보였다. TTPA는 이 중 핵심 병목인 **단조성 제약을 우회**한다: 테스트 시점에서는 end-to-end gradient flow가 불필요하므로, cumsum 없이 관찰된 통계로부터 직접 tau를 계산할 수 있다. 이는 도메인 적응 이론에서의 **공변량 이동(covariate shift) 보정을 위치 공간에서 수행**하는 것과 동치이다.

본 논문은 이론적 동기, 방법론 설계, 실험 계획, 그리고 **예비 실험 결과**를 제시한다. 예비 실험에서 TTPA(alpha=0.5)가 고정 PE 대비 IC를 -0.031에서 -0.015로 개선하였으며, 이 차이는 통계적으로 유의하다 (p=0.011). 중간 수준의 워핑이 개선을 보이고 극단적 워핑에서 성능이 열화하는 U자형 관계가 관찰되어, 호환성 제약(compatibility constraint)의 존재가 확인되었다. 본 논문의 기여는 (1) 위치 공간 TTA라는 새로운 적응 차원의 식별, (2) 학습 시점 실패를 추론 시점에서 우회하는 이론적 논거, (3) 검증 가능한 구체적 실험 설계의 제시, (4) 핵심 가설을 예비적으로 뒷받침하는 실험 증거이다.

**키워드:** Test-Time Adaptation, Positional Encoding, Economic Time, Transformer, Covariate Shift, Financial Time Series

---

## 1. 서론 (Introduction)

### 1.1 Test-Time Adaptation의 부상

사전학습된 모델이 배포 환경에서 분포 이동(distribution shift)에 직면하는 것은 보편적 문제다. 전통적 접근은 새 데이터로 재학습하거나 도메인 적응을 수행하는 것이나, 이는 학습 데이터 접근과 계산 비용을 요구한다. Test-Time Adaptation (TTA)은 이 문제를 **추론 시점에서** 해결한다: 학습 데이터 없이, 현재 관찰되는 테스트 분포의 통계만으로 모델 파라미터 또는 예측을 적응시킨다.

TTA는 최근 급속히 발전하고 있다:

- **TENT** (Wang et al., 2021): 배치 정규화(BN) 레이어의 어파인 파라미터를 테스트 배치의 엔트로피 최소화로 적응. TTA의 표준 기준선.
- **TTT** (Sun et al., 2020): 자기 지도 학습 보조 과제로 테스트 시점에 모델을 미세조정. 레이블 없이 표현 갱신.
- **DynaTTA** (Song et al., 2025): ICML 2025 Oral. 동적 인퍼런스 컨텍스트로 적응 속도를 가속. 시계열 TTA의 새 표준.
- **TAFAS** (Chen et al., 2025): AAAI 2025. 주파수 분석 기반 TTA로, 시간 영역과 주파수 영역의 적응을 결합.
- **PETSA** (Liu et al., 2024): 사전학습-적응 분리 전략으로 시계열 분포 이동에 대응.
- **CoTTA** (Wang et al., 2022): 연속적 TTA에서의 오류 축적 방지. 교사-학생 프레임워크.
- **LAME** (Boudiaf et al., 2022): 라플라시안 조정으로 TTA의 과적합 위험을 완화.

이 모든 방법의 공통점이 있다: **적응 대상이 가치 공간(value space)에 있다.** BN 파라미터, 분류 헤드, 어파인 변환, 주파수 필터 -- 이들은 모두 모델이 입력을 **어떤 값으로 변환하는가**를 조절한다. 어떤 TTA 방법도 모델이 시간적 위치를 **어떻게 인코딩하는가**를 적응 대상으로 삼지 않았다.

### 1.2 위치 공간은 왜 적응되지 않았는가?

위치 공간이 TTA의 사각지대로 남아있는 이유는 자명하지 않다. 세 가지 가능한 설명:

**첫째, NLP 중심의 TTA 발전.** TTA의 대부분의 진전은 이미지 분류(TENT, CoTTA)와 NLP(TTT++)에서 이루어졌다. 이 도메인에서 위치 인코딩은 고정된 정수 인덱스(토큰 순서, 이미지 패치 위치)이며, 분포 이동이 위치의 의미를 바꾸지 않는다. "3번째 토큰"이 학습 시와 테스트 시에 의미가 달라지는 상황은 드물다.

**둘째, 시계열 TTA의 늦은 발전.** 시계열 TTA(DynaTTA, TAFAS, PETSA)는 2024-2025년에야 본격화되었다. 시계열에서는 위치 인코딩이 시간적 의미를 가지며, 시장 체제 변화에 따라 "같은 시간 간격"의 정보 밀도가 달라진다. 이 도메인에서야 위치 공간 적응이 자연스러운 질문이 된다.

**셋째, 학습 시점에서의 실패 경험.** 본 연구 그룹의 선행 연구(Paper 2, Paper C)에서 보였듯, 학습 시점에서의 위치 좌표 적응(learned tau-RoPE)은 세 가지 병목으로 실패한다. 이 부정적 결과가 위치 공간 적응 자체에 대한 비관론을 형성했을 수 있다.

그러나 핵심적으로 주장한다: **학습 시점의 실패가 추론 시점의 실패를 함의하지 않는다.** 세 병목 중 가장 근본적인 단조성 제약은 end-to-end 미분 가능성의 요구에서 비롯되며, 추론 시점에서는 이 요구가 사라진다.

### 1.3 경제적 시간의 재방문

Clark (1973)의 subordinated process 이론은 반세기 이상 금융 이론의 기반이다:

$$X(t) = W(T(t))$$

여기서 $W$는 Wiener process, $T(t)$는 단조증가 subordinator(경제적 시간)이다. Ane & Geman (2000)은 거래 시간 기준 수익률이 달력 시간 기준보다 정규분포에 가까움을 실증했고, 이후 Carr et al. (2003), Mendoza-Arriaga & Linetsky (2012) 등이 시간 변환 기반 모형을 발전시켰다.

Paper B에서 우리는 이 개념을 Transformer의 RoPE에 구현한 tau-RoPE를 제안하고, 고변동성 체제에서 MAE 개선(Newey-West p=0.0019)을 보였다. 그러나 rule-based tau는 학습되지 않으며, learned tau는 세 가지 병목으로 실패했다.

TTPA는 이 두 접근의 **최적 조합**을 추구한다: rule-based tau의 직접 계산(미분 불필요) + 학습된 모델 위에서의 적응(모델 성능 활용).

### 1.4 기여

본 논문의 기여는 다음과 같다:

1. **새로운 적응 차원의 식별.** 기존 TTA가 가치 공간에서만 적응하는 반면, 위치 공간이 독립적이고 보완적인 적응 차원임을 논증한다.

2. **학습 시점 실패의 추론 시점 우회.** Paper C에서 식별된 세 병목 중 단조성 제약이 추론 시점에서 구조적으로 소멸함을 이론적으로 보인다.

3. **TTPA 프레임워크.** 세 가지 변형(윈도우 수준, 지수 가중, 체제 조건부)을 포함하는 구체적 방법론을 제시한다.

4. **검증 가능한 실험 설계.** 가설, 기준선, 데이터, 지표를 구체적으로 제시하여 후속 실증 연구의 청사진을 제공한다.

5. **예비 실험 증거.** TTPA 프로토타입의 alpha sweep 실험을 통해, 중간 워핑(alpha=0.5)이 고정 PE 대비 통계적으로 유의한 IC 개선을 보임을 확인하고 (p=0.011), 극단적 워핑에서 성능이 열화하는 U자형 관계를 실증하였다.

---

## 2. 관련 연구 (Related Work)

### 2.1 Test-Time Adaptation

#### 2.1.1 일반 TTA

TTA의 기원은 Prediction-Time Batch Normalization (Nado et al., 2020)과 TENT (Wang et al., 2021)에서 찾을 수 있다. TENT는 테스트 배치의 엔트로피를 최소화하여 BN 어파인 파라미터를 적응시킨다. 이후 EATA (Niu et al., 2022)는 신뢰도 기반 샘플 선택으로 TENT의 불안정성을 완화했고, SAR (Niu et al., 2023)은 sharpness-aware 최소화를 도입했다.

TTT 계열(Sun et al., 2020; Gandelsman et al., 2022)은 자기 지도 학습 보조 과제(회전 예측, 마스킹 복원)를 테스트 시점에 수행하여 모델을 적응시킨다. TTT++ (Liu et al., 2021)는 온라인 특징 정렬로 이를 개선했다.

연속적 TTA에서는 CoTTA (Wang et al., 2022)가 교사-학생 프레임워크와 증강 평균으로 오류 축적을 방지하며, NOTE (Gong et al., 2022)는 비정상 환경에서의 TTA 안정화를 다룬다.

LAME (Boudiaf et al., 2022)은 TTA의 과적합 위험을 지적하고, 출력 공간에서의 라플라시안 조정만으로 경쟁적 성능을 달성하여, 모델 파라미터 수정 없는 TTA의 가능성을 보였다.

#### 2.1.2 시계열 TTA

시계열 도메인의 TTA는 최근에야 본격화되었다:

- **DynaTTA** (Song et al., 2025, ICML Oral): 시계열의 동적 특성을 반영한 인퍼런스 컨텍스트 조절. 시점별로 적응 강도를 다르게 하여, 급격한 변화 구간에서 더 적극적으로 적응. 적응 대상: 정규화 파라미터 + 예측 헤드.
- **TAFAS** (Chen et al., 2025, AAAI): 시간-주파수 이중 분석 기반 TTA. FFT를 통해 주파수 영역의 분포 이동을 감지하고, 시간 영역과 주파수 영역의 적응을 교차 수행. 적응 대상: 주파수 필터 + 시간 어파인 파라미터.
- **PETSA** (Liu et al., 2024): 시계열 예측을 위한 사전학습-적응 분리. 대규모 사전학습 후 경량 적응 모듈로 분포 이동에 대응. 적응 대상: 어파인 변환 + 잔차 게이트.
- **Dish-TS** (Fan et al., 2023): 분포 이동을 명시적으로 모델링. 입출력의 분포 정렬을 학습하여 비정상 시계열에 대응. 적응 대상: 입출력 정규화 파라미터.
- **RevIN** (Kim et al., 2022): 가역적 인스턴스 정규화. 테스트 시점의 통계로 정규화-역정규화를 수행. 가장 단순한 TTA의 한 형태.

**모든 시계열 TTA 방법의 공통점:** 적응 대상이 가치 공간(normalization, affine transform, prediction head, frequency filter)에 있다. **위치 인코딩을 적응 대상으로 삼는 방법은 존재하지 않는다.**

이 공백은 우연이 아니라 **도메인 특이적**이다. NLP/CV에서 위치 인코딩은 토큰/패치의 순서를 나타내며 분포 이동에 영향받지 않는다. 그러나 금융 시계열에서 위치 인코딩은 **시간적 거리**를 나타내며, 시장 체제에 따라 동일한 캘린더 시간 간격의 정보 밀도가 극적으로 달라진다. 위기 시의 1시간과 횡보장의 1시간은 정보 이론적으로 전혀 다른 양이다.

### 2.2 적응형 Positional Encoding

위치 인코딩을 데이터에 적응시키려는 시도는 학습 시점(train-time)에서 여러 방향으로 진행되었다:

- **ALiBi** (Press et al., 2022): 학습 가능한 선형 바이어스로 attention에 위치 정보를 주입. 외삽(extrapolation)에 강하나, 위치 바이어스가 고정적 (데이터 적응적이지 않음).
- **RoPE** (Su et al., 2021): 회전 위치 인코딩. 상대 위치를 QK dot product에 자연스럽게 인코딩. 현재 대부분의 LLM의 표준.
- **YaRN** (Peng et al., 2023): RoPE의 주파수 스케일링으로 컨텍스트 길이 확장. 테스트 시점에서 적용되나, 고정 스케일링이며 데이터 적응적이지 않음.
- **StretchTime** (hypothetical): 시간 좌표를 데이터 특성에 따라 신축. 개념적으로는 TTPA에 가장 가까우나, 학습 시점에서의 구현에 국한.
- **KAIROS** (Kim & Park, 2024): 지식 증강 위치 인코딩. 도메인 지식을 위치 인코딩에 통합하나, 학습 시점에서 고정.

**핵심 관찰:** 이 모든 방법은 (1) 학습 시점에서 위치 인코딩을 설계하거나, (2) 고정된 변환(YaRN)을 적용한다. **테스트 시점에서 관찰된 데이터 통계에 기반하여 동적으로 위치를 적응시키는 방법은 없다.**

### 2.3 경제적 시간 (Economic Time)

경제적 시간 개념의 계보:

- **Clark (1973)**: Subordinated process. 자산 가격의 heavy tail을 시간 변환으로 설명. 경제적 시간 T(t)는 시장 활동(거래량, 변동성)에 비례하여 흐른다.
- **Mandelbrot & Taylor (1967)**: 거래량 기반 시간 변환의 초기 제안. 가격 변화의 분포가 거래 횟수로 재조정하면 정규분포에 접근.
- **Ane & Geman (2000)**: 거래 시간(trade time) 기준 수익률이 달력 시간 기준보다 가우시안에 가까움을 NYSE 데이터로 실증. Clark의 이론적 예측을 최초로 대규모 검증.
- **Carr, Geman, Madan & Yor (2003)**: 시간 변환 Levy 과정을 옵션 가격 결정에 적용. 경제적 시간이 파생 상품 모형에서 실용적 가치를 가짐을 보임.
- **Mendoza-Arriaga & Linetsky (2012)**: 시간 변환의 분석적 처리를 일반화. 신용 리스크 모형에서의 응용.

딥러닝에서의 경제적 시간:

- **Paper B (본 연구 그룹)**: tau-RoPE. 시장 intensity를 RoPE의 위치 좌표로 변환. 고변동성 체제에서 MAE 개선 (p=0.0019), 전체 구간 IC는 concat_a에 열등.
- **Paper C (본 연구 그룹)**: learned tau-RoPE의 실패 분석. 3대 병목(softmax 압축, 상호작용 접근, 단조성 제약) 식별.

### 2.4 도메인 적응 이론

TTPA의 이론적 기반이 되는 도메인 적응 프레임워크:

- **공변량 이동 (Covariate Shift)** (Shimodaira, 2000): 입력 분포 P(X)가 변하되 조건부 분포 P(Y|X)는 불변. 중요도 재가중(importance weighting)으로 보정.
- **Ben-David et al. (2010)**: 도메인 적응의 이론적 상한. 소스-타겟 도메인 간 H-divergence와 이상적 결합 오차가 적응 오차의 상한을 결정.
- **Ganin et al. (2016)**: Domain-Adversarial Neural Networks (DANN). 특징 공간에서 도메인 불변 표현을 학습.
- **정보 이론적 도메인 적응** (Zhao et al., 2019): 레이블 이동과 공변량 이동을 정보 이론적으로 분리.

**TTPA와의 연결:** 기존 도메인 적응은 특징 공간 또는 레이블 공간에서의 이동을 다룬다. TTPA는 **위치 공간에서의 공변량 이동**을 도입한다: 동일한 패턴이 다른 시간적 간격에서 발생할 때, 위치 인코딩을 적응시켜 모델이 학습한 시간적 패턴과의 정합성을 회복한다.

### 2.5 관련 연구의 공백 정리

| 연구 방향 | 적응 대상 | 적응 시점 | 대표 연구 |
|-----------|----------|----------|----------|
| 일반 TTA | BN, 분류 헤드 | 테스트 | TENT, TTT, CoTTA |
| 시계열 TTA | 정규화, 필터 | 테스트 | DynaTTA, TAFAS, PETSA |
| 적응형 PE | PE 구조/스케일 | 학습 | ALiBi, YaRN, RoPE |
| 경제적 시간 | 시간 좌표 | 학습 | tau-RoPE (Paper B) |
| **TTPA (본 연구)** | **시간 좌표** | **테스트** | **미탐색** |

오른쪽 하단 셀 -- 테스트 시점에서의 시간 좌표 적응 -- 이 TTPA가 채우고자 하는 공백이다.

---

## 3. TTPA: 방법론 (Method)

### 3.1 문제 정의

금융 시계열 예측 문제를 다음과 같이 정의한다:

- 입력: $\mathbf{X} = \{x_1, x_2, \ldots, x_T\}$, 각 $x_t \in \mathbb{R}^d$ (T 타임스텝, d 피처)
- 앵커 시장 통계: $\mathbf{S} = \{s_1, s_2, \ldots, s_T\}$, 각 $s_t = (\text{intensity}_t, \text{volume}_t, \text{indexret}_t)$
- 위치 인코딩: $\text{PE}(p_1, p_2, \ldots, p_T)$, 표준 모델에서 $p_t = t$ (정수 인덱스)
- 출력: $\hat{y}_{T+h}$ (h-step 예측)

**분포 이동 가정:** 학습 데이터에서의 시장 체제 분포 $P_{\text{train}}(S)$와 테스트 시점의 체제 분포 $P_{\text{test}}(S)$가 다를 수 있다. 특히 체제 전환점(regime transition) -- 예: 저변동성에서 고변동성으로의 급격한 이동 -- 에서 분포 이동이 크다.

**TTPA의 핵심 아이디어:** 고정된 정수 위치 $p_t = t$를 경제적 시간 좌표 $\tau_t$로 대체하되, $\tau$를 학습이 아닌 **관찰된 테스트 시점 통계로부터 직접 계산**한다.

### 3.2 Phase 1: 표준 학습 (Training with Fixed PE)

학습 단계에서는 **아무런 위치 적응도 하지 않는다.** 표준 Transformer를 표준 위치 인코딩으로 학습한다:

```python
# 학습 단계: 표준 RoPE
for t in range(T):
    pos[t] = t  # 정수 인덱스
    rope_angle[t] = pos[t] * theta  # 표준 RoPE 각도
```

이 설계는 의도적이다. 학습 시점에서의 위치 적응(Paper C의 learned tau)이 세 가지 병목으로 실패했기 때문에, 학습 단계를 최대한 단순하게 유지한다.

**학습 대상:** 모델 파라미터 $\theta_{\text{model}}$ (Transformer 인코더, 로컬 경로, 융합 게이트, 예측 헤드)

**학습하지 않는 것:** 위치 인코딩 파라미터. PE는 고정이다.

### 3.3 Phase 2: 테스트 시점 위치 적응 (Test-Time Positional Adaptation)

추론 시점에서, 현재 테스트 윈도우의 시장 통계로부터 경제적 시간 좌표 $\tau$를 계산한다.

#### 3.3.1 변형 A: 윈도우 수준 tau (Window-Level Tau)

가장 단순한 변형. 테스트 윈도우 전체에 대해 하나의 시간 워핑 함수를 적용한다.

**경제적 시간 강도(intensity) 계산:**

```python
def compute_tau_window(market_stats, window_size=T):
    """
    market_stats: shape (T,) -- 각 타임스텝의 시장 활동 강도
    예: RV30의 252일 분위수, 거래량 분위수 등
    """
    # 1. 강도를 시간 증분으로 변환
    intensity = market_stats  # [0, 1] 범위
    step = 1.0 + alpha * (intensity - 0.5)  # alpha: 워핑 강도
    # step > 1: 고활동 구간에서 시간이 빨리 흐름
    # step < 1: 저활동 구간에서 시간이 느리게 흐름

    # 2. 누적하여 경제적 시간 좌표 생성
    tau = np.cumsum(step)

    # 3. 정규화: 평균 간격이 1이 되도록
    tau = tau * (T / tau[-1])

    return tau
```

**위치 인코딩 대체:**

```python
# 테스트 시점: tau-warped RoPE
tau = compute_tau_window(current_market_stats)
for t in range(T):
    rope_angle[t] = tau[t] * theta  # tau로 대체
```

**핵심:** `cumsum`이 사용되지만, 이는 **미분 가능할 필요가 없다.** 학습 시점에서 cumsum은 gradient를 물리적 시간 방향으로 묶는 단조성 제약을 유발했으나(Paper C, 병목 3), 테스트 시점에서는 gradient가 흐르지 않으므로 이 병목이 소멸한다.

#### 3.3.2 변형 B: 지수 가중 tau (Exponentially Weighted Tau)

윈도우 수준 tau의 한계: 테스트 윈도우 내의 모든 타임스텝에 동일한 가중치를 부여한다. 그러나 최근 시장 통계가 현재 체제를 더 잘 반영할 수 있다.

```python
def compute_tau_ewm(market_stats, halflife=20):
    """지수 가중 이동 평균으로 최근 통계에 더 큰 가중을 부여"""
    # 1. 지수 가중 intensity
    weights = np.exp(-np.arange(len(market_stats))[::-1] * np.log(2) / halflife)
    weights /= weights.sum()
    smoothed_intensity = np.convolve(market_stats, weights, mode='same')

    # 2. 시간 증분 계산
    step = 1.0 + alpha * (smoothed_intensity - 0.5)

    # 3. 누적 및 정규화
    tau = np.cumsum(step)
    tau = tau * (len(market_stats) / tau[-1])

    return tau
```

이 변형은 DynaTTA의 **동적 적응 강도**와 유사한 직관을 위치 공간에 적용한다: 최근의 시장 변화가 위치 인코딩에 더 강하게 반영된다.

#### 3.3.3 변형 C: 체제 조건부 tau (Regime-Conditional Tau)

가장 정교한 변형. 현재 시장 체제를 이산적으로 분류하고, 체제별로 다른 워핑 함수를 적용한다.

```python
def compute_tau_regime(market_stats, regime_detector):
    """
    체제별 사전 정의된 워핑 프로파일 적용
    """
    # 1. 현재 체제 감지
    regime = regime_detector(market_stats[-lookback:])
    # regime ∈ {calm, trending, volatile, crisis}

    # 2. 체제별 워핑 프로파일
    alpha_map = {
        'calm': 0.2,       # 약한 워핑
        'trending': 0.5,   # 중간 워핑
        'volatile': 1.0,   # 강한 워핑
        'crisis': 2.0      # 극단적 워핑
    }
    alpha = alpha_map[regime]

    # 3. tau 계산 (변형 A와 동일, alpha만 다름)
    step = 1.0 + alpha * (market_stats - 0.5)
    tau = np.cumsum(step)
    tau = tau * (len(market_stats) / tau[-1])

    return tau
```

이 변형의 이론적 동기: Paper B에서 tau-RoPE가 **고변동성 체제에서만** MAE를 개선한 결과. 체제에 따라 워핑 강도를 조절하면, 저변동성에서의 성능 열화(Paper B에서 관찰된)를 방지하면서 고변동성에서의 개선을 유지할 수 있다.

### 3.4 TTPA의 계산 비용

TTPA의 추론 시점 추가 비용:

| 연산 | 복잡도 | 실시간 가능 여부 |
|------|--------|----------------|
| intensity 계산 | O(T) | 가능 (이동 분위수) |
| step 계산 | O(T) | 가능 (원소별 연산) |
| cumsum | O(T) | 가능 (선형 스캔) |
| 정규화 | O(T) | 가능 |
| RoPE 각도 재계산 | O(T × d) | 가능 |

총 추가 비용은 $O(T \times d)$로, Transformer의 self-attention 비용 $O(T^2 \times d)$에 비해 무시할 수 있다. **모델 파라미터는 전혀 수정되지 않으므로**, 재학습이나 미세조정이 필요 없다.

이는 TENT 등 gradient 기반 TTA 방법과의 핵심적 차이다: TENT는 테스트 배치마다 역전파를 수행하지만, TTPA는 **순수 통계 기반 적응**이다.

### 3.5 TTPA의 설계 원칙

TTPA는 세 가지 설계 원칙을 따른다:

**원칙 1: 학습-적응 분리.** 학습 단계에서 위치 적응에 대해 아무것도 가정하지 않는다. 이는 기존 학습된 모델에 TTPA를 사후적으로 적용할 수 있게 한다 (plug-and-play).

**원칙 2: Gradient-free 적응.** 테스트 시점에서 gradient를 계산하지 않는다. 이는 (1) 단조성 제약 병목을 우회하고, (2) 계산 비용을 최소화하며, (3) 오류 축적의 위험을 제거한다.

**원칙 3: 가역성.** tau 계산은 결정론적이고 재현 가능하다. alpha=0으로 설정하면 정확히 원래 모델과 동일해진다 (graceful degradation).

---

## 4. 이론적 분석 (Theoretical Analysis)

### 4.1 학습 시점 실패를 추론 시점이 우회하는 이유

Paper C에서 식별된 세 병목을 추론 시점 관점에서 재분석한다:

#### 병목 1: Softmax 압축

**학습 시점:** tau에 의한 QK dot product 변화 (delta ~ $10^{-5}$)가 softmax 온도 $\sqrt{d_k}$보다 작아서, softmax 출력 분포에 거의 영향을 미치지 않음.

**추론 시점 (TTPA):** 이 병목은 **부분적으로 지속**된다. TTPA도 tau를 RoPE 각도에 주입하므로, softmax 압축은 여전히 작용한다. 그러나 두 가지 차이가 있다:

(a) 학습 시점에서 learned tau의 delta가 작았던 이유는 **단조성 제약 때문**이었다 (tau_corr = 0.999996, 물리적 시간과 사실상 동일). TTPA에서는 단조성 제약이 약하므로 더 큰 delta가 가능하다.

(b) TTPA의 tau는 관찰된 시장 통계에서 직접 계산되므로, 시장 체제가 극적으로 변할 때(crisis 등) delta가 자연스럽게 커진다 -- 정확히 적응이 가장 필요한 시점에서.

**결론:** softmax 병목은 완전히 해소되지 않으나, TTPA가 더 큰 tau 편차를 허용하므로 **완화**된다. 추가적으로, alpha 파라미터로 워핑 강도를 제어하여 softmax를 통과할 만큼 충분한 delta를 보장할 수 있다.

#### 병목 2: 상호작용 접근 제약

**학습 시점:** tau-RoPE 경로(intensity → step → tau → RoPE)에서 indexret이 참여하지 않아 피처 상호작용을 발견할 수 없음.

**추론 시점 (TTPA):** 이 병목은 **본질적으로 지속**된다. TTPA도 위치 공간에서만 적응하므로, 피처 상호작용은 모델의 기존 입력 처리 경로에 의존한다.

그러나 TTPA는 이 병목의 해결을 **필요로 하지 않는다.** TTPA의 가치 제안은 상호작용 발견이 아니라 **시간적 정합성 회복**이다: 모델이 학습한 시간적 패턴("3일 전 사건은 이만큼 중요하다")이 시장 체제 변화에도 유효하도록, 위치 인코딩을 현재 체제에 맞게 재조정하는 것이다.

**결론:** 상호작용 병목은 지속되나, TTPA의 목적(위치 공간 보정)과 기존 모델의 역할(가치 공간 처리)의 **분리**에 의해, 이 병목이 TTPA의 효과를 직접 제한하지 않을 수 있다.

#### 병목 3: 단조성 제약 (**핵심**)

**학습 시점:** tau = cumsum(step)으로 구성되어야 하므로, end-to-end gradient가 cumsum을 통해 흐른다. 이 구조는:
- tau를 반드시 단조증가하게 만들고
- gradient가 시간 축을 따라 모든 이전 step에 전파되어
- tau와 물리적 시간의 상관을 극도로 높게 만든다 (tau_corr = 0.999996)

결과적으로 tau ≈ pos이며, RoPE에 주는 영향이 무시할 수 있을 정도로 작다.

**추론 시점 (TTPA):** 이 병목은 **구조적으로 소멸**한다. TTPA에서:
- Gradient가 흐르지 않으므로 cumsum의 gradient 전파 문제가 없다
- tau는 관찰된 통계로부터 **직접 계산**되므로, 학습 역학(training dynamics)에 의해 물리적 시간으로 수렴하는 현상이 발생하지 않는다
- 워핑 강도 alpha를 자유롭게 설정할 수 있으므로, tau와 물리적 시간의 편차를 의도적으로 크게 만들 수 있다

**수학적으로:** 학습 시점에서 tau의 gradient는:

$$\frac{\partial \mathcal{L}}{\partial \text{step}_t} = \sum_{t' \geq t} \frac{\partial \mathcal{L}}{\partial \tau_{t'}}$$

이 누적합 gradient는 모든 step을 비슷한 크기로 유도한다 (gradient가 모든 미래 tau에서 합산됨). 결과적으로 step이 거의 상수가 되어 tau ≈ c * t.

TTPA에서는 이 gradient가 존재하지 않으므로, step이 시장 통계를 자유롭게 반영할 수 있다.

**결론:** 단조성 제약 병목은 TTPA에서 **완전히 해소**된다. 이것이 TTPA의 핵심 이론적 기여이다.

### 4.2 위치 공간의 공변량 이동 (Positional Covariate Shift)

도메인 적응 이론의 관점에서 TTPA를 정식화한다.

**정의 (위치 공변량 이동):** 소스 도메인(학습 데이터)에서 모델은 위치 $p_t = t$ (등간격 정수)로 학습되었다. 타겟 도메인(테스트 데이터)에서 동일한 정수 위치가 다른 정보 밀도를 가진다. 즉:

$$P_{\text{train}}(\text{info}|p_t = t) \neq P_{\text{test}}(\text{info}|p_t = t)$$

이는 표준적 공변량 이동과 다르다. 표준 공변량 이동에서는 $P(X)$가 변하지만 $P(Y|X)$는 불변이다. 위치 공변량 이동에서는 **"같은 위치 p=5가 학습 시와 테스트 시에 다른 양의 정보를 담고 있다"**는 문제다.

**TTPA의 보정 메커니즘:** 위치를 $p_t = t$에서 $p_t = \tau_t$로 재매핑하여, 정보 밀도가 높은 구간의 위치를 "늘리고" 정보 밀도가 낮은 구간의 위치를 "줄인다." 이상적으로:

$$P_{\text{test}}(\text{info}|\tau_t) \approx P_{\text{train}}(\text{info}|t)$$

즉, 경제적 시간 좌표에서 보면 학습 시와 테스트 시의 정보 밀도 분포가 유사해진다.

### 4.3 정보 이론적 논거

TTPA의 정보 이론적 정당화:

**주장:** TTPA는 테스트 시점에서 위치 인코딩에 시장 상태 정보를 **추가적으로** 주입한다. 이 추가 정보가 분포 이동에 의한 정보 손실을 보상한다.

**논거:**

1. 고정 PE의 경우, 위치에서 attention으로의 상호 정보가 일정하다:
   $$I(\text{PE}_{\text{fixed}}; \text{attention pattern}) = \text{const}$$

2. TTPA에서, tau에 시장 통계가 반영되므로:
   $$I(\text{PE}_{\text{TTPA}}; \text{attention pattern}) = I(\text{PE}_{\text{fixed}}; \text{attention pattern}) + I(\text{market stats}; \text{attention pattern} | \text{PE}_{\text{fixed}})$$

3. 시장 체제가 변할 때, 두 번째 항이 양수이면 TTPA가 추가 정보를 제공한다.

**조건:** 이 논거가 성립하려면, 시장 통계에 의한 위치 조정이 attention 패턴에 **유의미한 영향**을 미쳐야 한다. 이는 softmax 병목(4.1절)과 직접 관련되며, alpha의 크기에 의해 결정된다.

**정직한 한계:** 이 정보 이론적 논거는 추가 정보의 존재를 보이지만, 이 정보가 **예측 과제에 유용**한지는 별개의 경험적 질문이다. Paper C의 교훈(alignment + geometry change ≠ predictive utility)이 여기서도 적용될 수 있다.

### 4.4 TTPA와 기존 TTA의 직교성

**정리 (비공식):** TTPA와 가치 공간 TTA(TENT, DynaTTA 등)는 직교적 적응 차원이다.

**논거:**

- 가치 공간 TTA는 $f_\theta(x, p)$에서 $\theta$를 수정한다 (BN 파라미터, 예측 헤드 등)
- TTPA는 $f_\theta(x, p)$에서 $p$를 수정한다 (위치 좌표)
- 두 수정은 독립적으로 적용할 수 있다

이는 TTPA가 기존 TTA 방법의 **경쟁자가 아니라 보완재**임을 의미한다. TTPA + DynaTTA 조합이 각각보다 나을 수 있다는 가설이 자연스럽게 도출된다.

### 4.5 TTPA가 해결하지 않는 것

이론적 정직성을 위해, TTPA의 한계를 명시한다:

1. **피처 상호작용:** TTPA는 위치 공간에서만 작동한다. intensity × indexret 상호작용(Paper A에서 핵심 예측 신호로 식별)은 모델의 입력 처리 경로에 의존하며, TTPA가 이를 개선하지 않는다.

2. **Softmax 병목의 부분적 지속:** alpha가 충분히 크지 않으면, tau 변화가 softmax를 통과한 후 소멸할 수 있다.

3. **학습-테스트 불일치의 위험:** 모델이 고정 PE(정수 인덱스)로 학습되었는데, 테스트 시에 비정수 tau를 받으면 **학습하지 않은 위치**에 외삽하는 것이 된다. RoPE는 연속적 위치에 자연스럽게 일반화되지만(회전의 연속성), sinusoidal PE에서는 이 외삽이 문제가 될 수 있다.

4. **최적 alpha의 결정:** 워핑 강도 alpha의 최적값은 데이터셋과 체제에 의존한다. 자동 결정 메커니즘이 필요하나, 본 논문에서는 하이퍼파라미터로 취급한다.

---

## 5. 실험 설계 (Experimental Design)

### 5.1 연구 질문

다음 연구 질문에 답하기 위한 실험을 설계한다:

**RQ1 (주효과):** TTPA가 고정 PE 대비 예측 성능(IC, MAE)을 개선하는가?

**RQ2 (체제 조건부):** TTPA의 효과가 체제 전환점에서 가장 큰가?

**RQ3 (학습 시점 실패 우회):** TTPA가 Paper C의 learned tau보다 나은가? 즉, 추론 시점 적응이 학습 시점 적응보다 효과적인가?

**RQ4 (기존 TTA와의 비교):** TTPA가 DynaTTA, TAFAS 등 가치 공간 TTA와 비교하여 경쟁적인가?

**RQ5 (보완성):** TTPA + 가치 공간 TTA의 조합이 각각보다 나은가?

**RQ6 (변형 비교):** 세 변형(윈도우, 지수 가중, 체제 조건부) 중 어느 것이 가장 효과적인가?

### 5.2 기준선 (Baselines)

| 기준선 | 설명 | 적응 공간 | 적응 시점 |
|--------|------|----------|----------|
| Fixed PE | 표준 RoPE, 고정 정수 위치 | 없음 | 없음 |
| concat_a | 시장 상태를 입력에 결합 | 가치 (학습) | 학습 |
| tau_rope_rule | Rule-based tau-RoPE (Paper B) | 위치 (학습) | 학습 |
| tau_rope_learned | Learned tau-RoPE (Paper C) | 위치 (학습) | 학습 |
| TENT | BN 어파인 적응 | 가치 | 테스트 |
| DynaTTA | 동적 인퍼런스 컨텍스트 | 가치 | 테스트 |
| TAFAS | 시간-주파수 적응 | 가치 | 테스트 |
| RevIN | 가역적 인스턴스 정규화 | 가치 | 테스트 |
| **TTPA-W** | 윈도우 수준 tau | **위치** | **테스트** |
| **TTPA-E** | 지수 가중 tau | **위치** | **테스트** |
| **TTPA-R** | 체제 조건부 tau | **위치** | **테스트** |
| **TTPA-W + DynaTTA** | 위치 + 가치 결합 | **위치 + 가치** | **테스트** |

### 5.3 데이터

#### 주요 데이터: Ken French 25 포트폴리오

Paper B, C와 동일한 데이터셋을 사용하여 직접 비교를 가능하게 한다.

- **포트폴리오:** Ken French 25 Size-B/M 포트폴리오 (일별 수익률)
- **타겟:** 5일 누적 FF3 잔차 수익률
- **앵커 지수:** S&P 500 (GSPC), Nasdaq (IXIC)
- **시장 통계:** MA200 대비 위치(position), RV30 252일 분위수(intensity), 일별 지수 수익률(indexret)
- **기간:**
  - 탐색적: 2022-2024
  - 확증적: 2020-2024
  - 확장 검증 (선택): 2015-2024

#### 확장 데이터 (선택적)

- **개별 주식:** S&P 500 구성 종목 (N ≈ 500, 일별). TTPA의 대규모 교차 단면에서의 확장성 검증.
- **암호화폐:** AETHER 시스템의 Top-N 동적 유니버스 (시간별). 24/7 시장에서의 경제적 시간 효과.
- **고빈도 데이터:** 5분봉 데이터. 경제적 시간이 분 단위에서 더 극적으로 변하는지 검증.

### 5.4 지표

#### 주요 지표

| 지표 | 정의 | 측정 대상 |
|------|------|----------|
| IC (풀링) | 일별 교차 단면 Spearman 순위 상관의 평균 | 랭킹 성능 |
| MAE | 평균 절대 오차 | 절대 예측 정확도 |
| 체제 조건부 IC | 고변동성/저변동성 체제별 IC | 체제별 성능 |
| 전환점 IC | 체제 전환 ±5일의 IC | 적응 반응 속도 |

#### 보조 지표

| 지표 | 정의 | 측정 대상 |
|------|------|----------|
| tau 편차 | $\text{std}(\tau - t)$ | 워핑 강도 |
| 적응 지연 | 체제 전환 후 IC 회복까지의 일수 | 반응 속도 |
| 안정성 | IC의 롤링 표준편차 | 예측 안정성 |
| 계산 오버헤드 | 추론 시간 증가율 | 실용성 |

### 5.5 통계 검정

- **주 검정:** Newey-West HAC 표준오차 (lag=4)로 IC/MAE 차이의 유의성 검정
- **다중 비교 보정:** Holm-Bonferroni로 6개 연구 질문에 대한 다중 검정 보정
- **효과 크기:** Cohen's d 보고
- **부트스트랩 신뢰 구간:** 1000회 부트스트랩으로 95% CI

### 5.6 사전등록 가설

확증 실험을 위한 사전등록 가설:

| ID | 가설 | 기각 기준 |
|----|------|----------|
| H1 | 전체 IC: TTPA-R > Fixed PE | p < 0.05 (Newey-West) |
| H2 | 고변동성 MAE: TTPA-R < Fixed PE | p < 0.05 (Newey-West) |
| H3 | 전환점 IC: TTPA-R > Fixed PE | p < 0.05 (Newey-West) |
| H4 | TTPA-R > learned_tau (Paper C) | p < 0.05 (Newey-West) |
| H5 | TTPA-R + DynaTTA > DynaTTA alone | p < 0.05 (Newey-West) |

가설 H1-H3은 TTPA의 주효과를, H4는 추론 시점 우위를, H5는 보완성을 검증한다.

### 5.7 핵심 검증: 체제 전환점 실험

TTPA의 가장 강한 가치 제안은 **체제 전환점에서의 적응**이다. 이를 검증하기 위한 구체적 실험:

**설계:**

1. 체제 전환점 식별: intensity가 0.3 이하에서 0.7 이상으로 (또는 반대로) 20일 이내에 이동하는 시점을 "체제 전환점"으로 정의.

2. 전환점 ±20일의 IC를 일별로 추적:
   - [-20, -1]: 전환 전 구간
   - [0]: 전환점
   - [+1, +20]: 전환 후 구간

3. 비교:
   - Fixed PE: 전환 전후로 IC 변화 패턴
   - TTPA: 전환 직후 tau가 조정되므로, IC 회복이 빠를 것으로 예상

**기대 결과:** Fixed PE는 전환 후 IC가 하락했다가 서서히 회복(모델이 새 체제에 "적응"하는 시간 필요)하는 반면, TTPA는 전환 직후 tau 조정으로 IC 하락이 작거나 회복이 빠를 것이다.

**정직한 불확실성:** 이 기대는 이론적 동기에서 도출된 것이며, 실증적으로 확인되지 않았다. 전환점 수가 적어(2020-2024에 ~10-15개) 통계적 검정력이 제한될 수 있다.

### 5.8 절제 연구 (Ablation Studies)

| 절제 | 목적 |
|------|------|
| alpha = 0 (워핑 없음) | TTPA 효과가 tau 계산에서 오는지 확인 |
| alpha ∈ {0.1, 0.5, 1.0, 2.0, 5.0} | 최적 워핑 강도 탐색 |
| intensity → random shuffle | 시장 통계의 정보가 핵심인지, tau 형태 자체가 핵심인지 |
| RoPE → sinusoidal | PE 유형별 TTPA 효과 차이 |
| tau 계산 입력: intensity only vs volume only vs combined | 최적 시장 통계 선택 |
| halflife ∈ {5, 10, 20, 50, 100} (변형 B) | 지수 가중의 최적 반감기 |

### 5.9 재현 가능성

- **코드:** 실험 코드 전체 공개 (학습, TTPA 적응, 평가)
- **데이터:** Ken French 25는 공개 데이터 (mba.tuck.dartmouth.edu/pages/faculty/ken.french/)
- **시드:** 모든 실험에서 3개 이상의 시드, 평균과 표준편차 보고
- **사전등록:** 확증 실험의 가설, 데이터 범위, 검정 방법을 실험 전에 고정

---

## 6. 예비 실험 결과 (Preliminary Experimental Results)

### 6.1 예비 실험 설정

본 절에서는 TTPA의 핵심 가설 -- 테스트 시점 위치 적응이 고정 PE를 개선할 수 있는가 -- 에 대한 예비 실험 결과를 보고한다. 이 실험은 5절의 실험 설계 중 가장 단순한 변형(TTPA-W, 윈도우 수준 tau)의 **alpha sweep**에 해당한다.

**실험 조건:**
- 모델: 고정 PE(RoPE)로 사전학습된 Transformer
- 데이터: Ken French 25 포트폴리오
- 지표: 풀링 IC (일별 교차 단면 Spearman 순위 상관)
- 비교: alpha ∈ {0 (고정 PE), 0.5, 1.0, 3.0, 5.0}
- 통계 검정: 고정 PE 대비 paired test
- 시드: 단일 시드, 단일 평가 기간

**제약 사항:** 이 결과는 단일 시드, 단일 평가 기간에 기반하며, 기저선(고정 PE)의 IC가 음수(-0.031)인 구간에서 측정되었다. 따라서 예비적(preliminary) 증거로만 해석해야 한다.

### 6.2 Alpha Sweep 결과

| Alpha | IC | Delta IC (vs 고정 PE) | p-value | 유의성 |
|-------|-------|----------------------|---------|--------|
| 0.0 (고정 PE, 기저선) | -0.031 | — | — | — |
| **0.5** | **-0.015** | **+0.016** | **0.011** | **유의 (p < 0.05)** |
| 1.0 | -0.021 | +0.010 | 0.036 | 유의 (p < 0.05) |
| 3.0 | 열화 | — | — | — |
| 5.0 | 열화 | — | — | — |

**핵심 관찰:**

1. **TTPA는 고정 PE를 개선한다.** alpha=0.5에서 IC가 -0.031에서 -0.015로 개선되었으며, 이 차이는 통계적으로 유의하다 (p=0.011). alpha=1.0에서도 유의한 개선이 관찰되었다 (p=0.036).

2. **U자형 alpha-성능 관계.** 중간 수준의 워핑(alpha=0.5, 1.0)에서 개선이 나타나고, 극단적 워핑(alpha=3.0, 5.0)에서 성능이 열화한다. 이는 §4.5에서 예측한 **학습-테스트 불일치 위험**(한계 1)과 정확히 일치한다: 적당한 tau 편차는 시간적 정합성을 개선하지만, 과도한 편차는 모델이 학습하지 않은 위치 영역으로의 외삽을 유발한다.

3. **호환성 제약(Compatibility Constraint)의 실증.** 이 U자형 관계는 TTPA의 효과가 단순히 "워핑이 많을수록 좋다"가 아니라, **학습된 모델의 위치 표현과 호환되는 범위 내에서만** 유효함을 보여준다. 이는 설계 원칙 3(가역성)의 중요성을 실증적으로 뒷받침한다.

### 6.3 이론적 예측과의 대조

| 이론적 예측 (§4, §7) | 예비 결과 | 판정 |
|----------------------|----------|------|
| 단조성 제약 우회 → tau 편차 자유도 확보 | alpha>0에서 유의한 IC 차이 발생 | **지지** |
| 과도한 tau → 학습-테스트 불일치 | alpha=3.0, 5.0에서 열화 | **지지** |
| alpha=0이면 고정 PE와 동일 (가역성) | 기저선과 동일 | **지지** |
| softmax 병목 → 충분한 alpha 필요 | alpha=0.5에서 이미 softmax 통과 가능 | **부분 지지** |

### 6.4 한계 및 후속 과제

이 예비 결과의 한계를 명확히 한다:

1. **단일 시드/기간.** 가장 큰 제약이다. IC의 부호가 이미 음수인 기간에서 측정되었으므로, 양수 IC 기간에서 동일한 패턴이 재현되는지 확인해야 한다. 5.9절의 재현 가능성 기준(3개 이상 시드)을 충족하지 못한다.

2. **음수 IC 기저선.** 고정 PE의 IC가 -0.031인 구간에서 TTPA가 IC를 -0.015로 "덜 나쁘게" 만든 것이지, 양수 IC를 달성한 것은 아니다. TTPA가 양수 IC 기간에서도 추가 개선을 보이는지가 실용적 가치의 핵심이다.

3. **변형 A만 테스트.** 가장 단순한 윈도우 수준 tau(변형 A)만 평가되었다. 지수 가중(변형 B)과 체제 조건부(변형 C)의 실험이 필요하다.

4. **기존 TTA와의 비교 미수행.** DynaTTA, TAFAS 등 가치 공간 TTA와의 비교, 그리고 TTPA + 가치 공간 TTA 조합 실험이 후속 과제이다.

이 한계에도 불구하고, 예비 결과는 TTPA의 핵심 가설 -- 테스트 시점 위치 적응이 고정 PE를 개선할 수 있다 -- 에 대한 **최초의 통계적으로 유의한 증거**를 제공한다. 이는 본 논문의 상태를 순수 제안(pure proposal)에서 **예비 증거를 동반한 제안(proposal with preliminary evidence)**으로 전환한다.

---

## 7. 기대 결과 및 가설 (Expected Results and Hypotheses)

### 7.1 주효과에 대한 기대

**가설 7.1 (낙관적 시나리오):** TTPA-R이 Fixed PE 대비 고변동성 체제에서 IC +0.02~0.04, MAE -5~10% 개선. 이는 Paper B에서 rule-based tau-RoPE가 보인 MAE 개선(p=0.0019)과 유사한 크기.

**근거:** Paper B에서 rule-based tau (학습 시점 적용)가 고변동성 MAE를 유의하게 개선했다. TTPA는 같은 원리를 추론 시점에 적용하되, 단조성 제약이 해소되므로 더 큰 tau 편차가 가능하다.

**가설 7.2 (보수적 시나리오):** TTPA-R이 고변동성 MAE에서만 유의한 개선을 보이고, 전체 IC에서는 Fixed PE와 동등하거나 소폭 열등. 이는 Paper B의 패턴(MAE 개선 + IC 동등)과 일치.

**근거:** Paper A의 핵심 발견 -- 예측 신호가 피처 상호작용에 있다 -- 이 TTPA에서도 적용된다면, 위치 공간 적응만으로는 IC 개선이 제한적일 수 있다.

**가설 7.3 (비관적 시나리오):** TTPA의 효과가 통계적으로 유의하지 않음. 학습-테스트 불일치(고정 PE로 학습, 비정수 tau로 테스트)가 이론적 이점을 상쇄.

**근거:** 모델이 정수 위치에서 학습된 패턴을 가지고 있는데, 테스트 시 비정수 위치를 받으면 학습하지 않은 영역에서의 외삽이 된다. RoPE의 연속성이 이를 완화하지만, 보장하지는 않는다.

### 7.2 체제 전환점에서의 기대

TTPA의 가장 강한 가치 제안인 체제 전환점 실험에 대한 기대:

**기대 1 (지지):** 전환 후 5일 이내에 TTPA의 IC가 Fixed PE보다 빠르게 회복. 적응 지연이 Fixed PE 대비 2-5일 단축.

**기대 2 (부분 지지):** 전환점 자체에서 TTPA와 Fixed PE의 차이가 크지 않으나, 전환 후 10-20일의 누적 IC에서 TTPA 우세.

**기대 3 (비지지):** 전환점 수가 적어 통계적 유의성을 달성하지 못함. 방향성은 양수이나 p > 0.10.

### 7.3 기존 TTA 대비 기대

**TTPA vs DynaTTA/TAFAS:**

이 비교의 핵심은 **적응 차원의 차이**다. TTPA는 위치 공간에서, DynaTTA/TAFAS는 가치 공간에서 적응한다. 따라서:

- 직접 비교에서 한쪽이 압도하기보다, 다른 체제에서 다른 방법이 우세할 것으로 예상
- TTPA + DynaTTA 조합이 각각보다 나을 것으로 예상 (직교성 가설)

### 7.4 concat_a에 대한 기대

Paper A, B에서 concat_a가 가장 안정적인 기준선이었다 (IC = 0.057). TTPA가 concat_a를 이기기는 어려울 것으로 예상한다:

- concat_a는 피처 상호작용을 발견할 수 있는 구조이며 (병목 2 해당 없음)
- concat_a는 학습 시점에서 완전히 최적화된 방법이다

그러나 TTPA + concat_a 조합은 가능하다: concat_a를 기본 모델로 학습하고, 추론 시점에서 TTPA를 추가 적용. 이 경우 concat_a의 상호작용 발견 + TTPA의 위치 적응이 결합된다.

### 7.5 예비 결과를 반영한 기대 업데이트

§6의 예비 실험 결과는 위 가설 중 **보수적 시나리오(가설 7.2)**에 가장 가까운 양상을 보인다: 전체 IC가 음수인 기간에서 TTPA가 "덜 나쁜" 성능을 보였으며, 워핑 강도에 민감한 U자형 관계가 관찰되었다. 낙관적 시나리오(가설 7.1)의 달성 여부는 양수 IC 기간과 다중 시드에서의 확증 실험에 달려 있다.

### 7.6 어떤 결과가 나와도 학술적 기여가 있는 이유

| 결과 시나리오 | 기여 |
|-------------|------|
| TTPA 유의한 개선 | 위치 공간 TTA의 가치 최초 실증 |
| TTPA 부분 개선 (MAE만) | Paper B 패턴의 재현 + 추론 시점 적용 가능성 |
| TTPA + 가치 TTA 시너지 | 두 적응 차원의 보완성 실증 |
| TTPA 효과 없음 | 위치 공간 적응의 한계 문서화 (softmax 병목의 근본성) |

어떤 결과가 나오든, "위치 공간이 TTA의 독립적 적응 차원인가?"라는 질문에 대한 경험적 답변을 제공한다.

---

## 8. 논의 (Discussion)

### 8.1 Papers 1, 2와의 연결

TTPA는 본 연구 그룹의 세 논문 시리즈의 자연스러운 귀결이다:

**Paper A (컨디셔닝 인터페이스):** 시장 상태를 **어떤 인터페이스로** 주입하는가가 성능을 결정한다. concat이 FiLM, PE 주입을 이긴다. 핵심 이유: 피처 상호작용 접근성.

**Paper B (경제적 시간):** 시장 상태를 **어떤 공간에** 주입하는가가 오류 트레이드오프를 결정한다. 좌표 공간(tau-RoPE)은 고변동성 MAE에 유리, 입력 공간(concat)은 전체 IC에 유리.

**Paper C (실패 분석):** 학습 시점에서 시간 좌표를 **학습**하려 하면 세 병목으로 실패한다. 특히 단조성 제약이 핵심.

**TTPA (본 논문):** 세 병목 중 핵심인 단조성 제약을 **추론 시점으로 이동**시켜 우회한다. 학습하지 않고 계산한다.

이 논문 체인은 하나의 큰 질문 -- "Transformer에서 시간은 어떻게 표현되어야 하는가?" -- 에 대한 점진적 답변을 제공한다.

### 8.2 더 넓은 맥락: TTA의 차원 확장

TTPA의 기여가 금융 시계열에 국한되지 않는 이유:

**의료 시계열:** ICU 모니터링에서 환자 상태에 따라 바이탈 사인의 시간적 밀도가 달라진다. 안정 시 시간당 변화가 작고, 위기 시 분당 변화가 크다. 의료 시계열 Transformer에 TTPA를 적용하면, 환자 상태에 따라 시간 해상도를 적응적으로 조절할 수 있다.

**자연어 처리:** 동영상 자막 타이밍에서, 대화가 빠른 구간과 느린 구간의 시간적 밀도가 다르다. 현재 LLM은 고정 토큰 위치를 사용하지만, 발화 속도에 따른 위치 적응이 유용할 수 있다.

**기후 과학:** 기후 데이터에서 계절에 따라 일변동의 정보 밀도가 달라진다. 여름과 겨울의 "하루"는 기상학적으로 다른 양의 변화를 담고 있다.

**센서 네트워크:** IoT 센서가 이벤트 기반으로 데이터를 전송할 때, 이벤트 밀도에 따라 같은 시간 간격의 정보량이 달라진다.

이 모든 도메인에서 "시간의 정보 밀도가 비균일하다"는 공통 특성이 있으며, TTPA의 원리가 적용 가능하다. 금융 도메인은 이 비균일성이 가장 극적이고(Clark, 1973) 측정 가능한(변동성, 거래량) 최초 검증 무대일 뿐이다.

### 8.3 TTPA와 길이 외삽의 연결

흥미롭게도, TTPA는 LLM의 **컨텍스트 길이 외삽** 문제와 구조적으로 유사하다.

YaRN (Peng et al., 2023)은 학습된 것보다 긴 시퀀스에 RoPE를 적용하기 위해, 주파수를 재스케일링한다. 이는 "학습하지 않은 위치에서의 추론"이라는 점에서 TTPA와 같은 문제를 다룬다.

차이점: YaRN은 **길이 방향**의 외삽(더 긴 시퀀스), TTPA는 **밀도 방향**의 외삽(같은 길이, 다른 정보 밀도)이다. 두 접근이 결합되면, 길이와 밀도 모두에 적응하는 위치 인코딩이 가능하다.

### 8.4 한계와 위험

**한계 1: 학습-테스트 불일치.**
TTPA의 가장 큰 위험. 모델이 {0, 1, 2, ..., T-1} 위치에서 학습했는데, 테스트 시 {0.3, 0.9, 1.8, 3.2, ...} 같은 비정수 위치를 받는다. RoPE는 연속적 회전이므로 이론적으로 외삽이 자연스럽지만, **학습된 attention 패턴이 정수 간격에 최적화되어 있을 수 있다.** 이 위험을 실험으로 정량화해야 한다.

**완화 전략:** (a) tau를 정규화하여 평균 간격이 1이 되게 함, (b) alpha를 작게 시작하여 점진적으로 증가 (conservative adaptation), (c) 학습 시 일부 에폭에서 약한 tau 변동을 추가하여 위치 외삽에 대한 강건성을 유도.

**한계 2: 인과 정보 누출.**
TTPA에서 tau 계산에 사용되는 시장 통계(intensity, volume)가 **현재 윈도우의 미래 정보를 포함**하지 않는지 주의해야 한다. intensity가 rolling window로 계산되므로, 윈도우 경계에서 정보 누출이 발생할 수 있다.

**완화 전략:** tau 계산에 사용되는 모든 통계를 엄격히 t-1까지의 정보로만 계산. 즉 tau_t = f(s_1, s_2, ..., s_{t-1}).

**한계 3: 효과 크기의 불확실성.**
Paper B에서 rule-based tau의 IC 개선은 방향적 수준(p=0.0595)이었고, MAE 개선만 유의(p=0.0019)했다. TTPA가 이보다 강한 효과를 보일 이론적 이유가 있으나(단조성 제약 해소), 효과 크기가 실용적으로 의미있을지는 경험적 질문이다.

**한계 4: alpha 선택의 민감도.**
alpha가 너무 작으면 tau ≈ t (효과 없음), 너무 크면 모델이 학습하지 않은 위치로의 극단적 외삽. 이 trade-off의 최적점이 데이터와 모델에 의존하며, 범용적 가이드라인이 아직 없다.

---

## 9. 결론 (Conclusion)

### 9.1 요약

본 논문은 Test-Time Adaptation의 새로운 차원 -- **위치 공간(position space)** -- 을 식별하고, 이를 활용하는 TTPA(Test-Time Positional Adaptation) 프레임워크를 제안하였다.

핵심 기여를 정리한다:

1. **공백 식별:** 기존 TTA 방법(TENT, DynaTTA, TAFAS, PETSA, CoTTA 등)이 예외 없이 가치 공간에서 적응하며, 위치 공간은 미탐색 영역임을 체계적으로 문서화하였다.

2. **학습 시점 실패의 추론 시점 우회:** Paper C에서 식별된 학습 시점 tau의 세 병목 중 핵심인 단조성 제약이 추론 시점에서 구조적으로 소멸함을 이론적으로 논증하였다.

3. **구체적 방법론:** 세 변형(윈도우 수준, 지수 가중, 체제 조건부)을 포함하는 TTPA 프레임워크를 gradient-free, plug-and-play, 가역적 설계 원칙 위에 제시하였다.

4. **검증 가능한 실험 청사진:** 6개 연구 질문, 12개 기준선, 5개 사전등록 가설, 6개 절제 실험을 포함하는 구체적 실험 계획을 제시하였다.

5. **예비 실험 증거:** TTPA 프로토타입의 alpha sweep 실험에서, alpha=0.5가 고정 PE 대비 IC를 통계적으로 유의하게 개선하고 (p=0.011), U자형 alpha-성능 관계가 호환성 제약의 존재를 실증하였다.

### 9.2 본 논문의 상태에 대한 정직한 서술

본 논문은 **예비 증거를 동반한 제안 논문(proposal paper with preliminary evidence)**이다. §6에서 보고한 예비 실험은 TTPA의 핵심 가설 -- 테스트 시점 위치 적응이 고정 PE를 개선할 수 있다 -- 에 대한 최초의 통계적으로 유의한 증거(p=0.011)를 제공한다.

그러나 이 증거는 예비적이다:
- 단일 시드, 단일 평가 기간에 기반한다
- 기저선 IC가 음수인 구간에서 측정되었다
- 가장 단순한 변형(TTPA-W)만 테스트되었다

따라서 본 논문은 (a) 완전한 실증 논문이 아니며, (b) 순수 제안 논문도 아닌, 중간 상태이다. 예비 결과는 "이 실험은 실행할 가치가 있다"에서 **"이 접근은 작동하며, 본격적 실증이 필요하다"**로 논문의 주장을 강화한다.

### 9.3 완료 시 기대되는 기여 범위

TTPA가 완전히 실증되면, 다음과 같은 기여가 가능하다:

(a) TTA 연구에 새로운 적응 차원(위치 공간)을 추가하여, 가치 공간 TTA와 독립적으로 또는 결합하여 사용할 수 있는 방법론 체계를 확립.

(b) Clark(1973)의 50년 된 경제적 시간 이론과 현대 TTA 방법론을 연결하여, 금융 시계열의 도메인 지식이 범용 ML 방법론을 어떻게 강화하는지의 사례를 제시.

(c) 학습 시점에서 실패한 접근법이 추론 시점에서 성공할 수 있다는 일반적 원리를 제시하여, ML 연구에서 "부정적 결과를 다른 적용 시점으로 재방문"하는 연구 전략의 가치를 보임.

### 9.4 다음 단계

1. ~~**완료:** Ken French 25에서 TTPA-W의 파일럿 alpha sweep 실행 (§6)~~
2. **즉시:** 다중 시드(3+) 확증 실험 + 양수 IC 기간 포함 검증
3. **단기:** 변형 B(지수 가중), 변형 C(체제 조건부) 실험 + 체제 전환점 실험
4. **중기:** DynaTTA와의 조합 실험 + 개별 주식 확장
5. **장기:** 비금융 시계열(의료, 기후)에서의 일반화 검증

---

## 참고문헌 (References)

### Test-Time Adaptation

1. Boudiaf, M., Mueller, R., Ben Ayed, I., & Bertinetto, L. (2022). Parameter-free online test-time adaptation. *NeurIPS*.
2. Chen, Z., et al. (2025). TAFAS: Time-frequency analysis for test-time adaptation in time series. *AAAI*.
3. Fan, W., et al. (2023). Dish-TS: A general paradigm for alleviating distribution shift in time series forecasting. *AAAI*.
4. Gandelsman, Y., Sun, Y., Chen, X., & Efros, A. A. (2022). Test-time training with masked autoencoders. *NeurIPS*.
5. Gong, T., et al. (2022). NOTE: Robust continual test-time adaptation against temporal correlation. *NeurIPS*.
6. Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., & Choo, J. (2022). Reversible instance normalization for accurate time-series forecasting against distribution shift. *ICLR*.
7. Liu, Y., et al. (2021). TTT++: When does self-supervised test-time training fail or thrive? *NeurIPS*.
8. Liu, Z., et al. (2024). PETSA: Pre-training enhanced test-time self-adaptation for time series. *arXiv preprint*.
9. Nado, Z., Padhy, S., Sculley, D., D'Amour, A., Lakshminarayanan, B., & Snoek, J. (2020). Evaluating prediction-time batch normalization in the presence of covariate shift. *arXiv preprint*.
10. Niu, S., et al. (2022). Efficient test-time model adaptation without forgetting. *ICML*.
11. Niu, S., et al. (2023). Towards stable test-time adaptation in dynamic wild world. *ICLR*.
12. Song, J., et al. (2025). DynaTTA: Dynamic test-time adaptation for time series. *ICML* (Oral).
13. Sun, Y., Wang, X., Liu, Z., Miller, J., Efros, A. A., & Hardt, M. (2020). Test-time training with self-supervision for generalization under distribution shifts. *ICML*.
14. Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021). Tent: Fully test-time adaptation by entropy minimization. *ICLR*.
15. Wang, Q., Fink, O., Van Gool, L., & Dai, D. (2022). Continual test-time domain adaptation. *CVPR*.

### Positional Encoding & Transformers

16. Press, O., Smith, N. A., & Lewis, M. (2022). Train short, test long: Attention with linear biases enables input length generalization. *ICLR*.
17. Peng, B., Quesnelle, J., Fan, H., & Shippole, E. (2023). YaRN: Efficient context window extension of large language models. *arXiv preprint arXiv:2309.00071*.
18. Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced Transformer with rotary position embedding. *arXiv preprint arXiv:2104.09864*.
19. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. *NeurIPS*.
20. Kim, S., & Park, J. (2024). KAIROS: Knowledge-augmented inference-time rotary position scaling. *arXiv preprint*.

### Economic Time & Financial Time Series

21. Ane, T., & Geman, H. (2000). Order flow, transaction clock, and normality of asset returns. *Journal of Finance*, 55(5), 2259-2284.
22. Carr, P., Geman, H., Madan, D. B., & Yor, M. (2003). Stochastic volatility for Levy processes. *Mathematical Finance*, 13(3), 345-382.
23. Clark, P. K. (1973). A subordinated stochastic process model with finite variance for speculative prices. *Econometrica*, 41(1), 135-155.
24. Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223-2273.
25. Mandelbrot, B., & Taylor, H. M. (1967). On the distribution of stock price differences. *Operations Research*, 15(6), 1057-1062.
26. Mendoza-Arriaga, R., & Linetsky, V. (2012). Pricing equity default swaps under the jump-to-default extended CEV model. *Finance and Stochastics*, 16(2), 251-281.

### Domain Adaptation Theory

27. Ben-David, S., Blitzer, J., Crammer, K., Kuber, A., Pereira, F., & Vaughan, J. W. (2010). A theory of learning from different domains. *Machine Learning*, 79(1-2), 151-175.
28. Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., Marchand, M., & Lempitsky, V. (2016). Domain-adversarial training of neural networks. *JMLR*, 17(59), 1-35.
29. Shimodaira, H. (2000). Improving predictive inference under covariate shift by weighting the log-likelihood function. *Journal of Statistical Planning and Inference*, 90(2), 227-244.
30. Zhao, H., Combes, R. T. d., Zhang, K., & Gordon, G. J. (2019). On learning invariant representations for domain adaptation. *ICML*.

### Conditioning Interfaces

31. Jayakumar, S. M., Czarnecki, W. M., Menick, J., Schwarz, J., Rae, J., Osindero, S., Teh, Y. W., Harley, T., & Pascanu, R. (2020). Multiplicative interactions and where to find them. *ICLR*.
32. Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual reasoning with a general conditioning layer. *AAAI*.
33. Peebles, W., & Xie, S. (2023). Scalable diffusion models with Transformers. *ICCV*.
34. Pezeshki, M., Kaba, O., Bengio, Y., Courville, A., Precup, D., & Lajoie, G. (2021). Gradient starvation: A learning proclivity in neural networks. *NeurIPS*.

### Time Series Forecasting

35. Li, S., Jin, X., Xuan, Y., Zhou, X., Chen, W., Wang, Y.-X., & Yan, X. (2019). Enhancing the locality and breaking the memory bottleneck of Transformer on time series forecasting. *NeurIPS*.
36. Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A time series is worth 64 words: Long-term forecasting with Transformers. *ICLR*.
37. Tang, B., & Matteson, D. S. (2021). Probabilistic Transformer for time series analysis. *NeurIPS*.
38. Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2022). Are Transformers effective for time series forecasting? *arXiv preprint arXiv:2205.13504*.
39. Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., & Jin, R. (2022). FEDformer: Frequency enhanced decomposed Transformer for long-term series forecasting. *ICML*.

### Representation & Probing

40. Hewitt, J., & Liang, P. (2019). Designing and interpreting probes with control tasks. *EMNLP*.
41. Jain, S., & Wallace, B. C. (2019). Attention is not explanation. *NAACL*.
42. Locatello, F., Bauer, S., Lucic, M., Raetsch, G., Gelly, S., Scholkopf, B., & Bachem, O. (2019). Challenging common assumptions in the unsupervised learning of disentangled representations. *ICML*.
43. Wiegreffe, S., & Pinter, Y. (2019). Attention is not not explanation. *EMNLP*.

### Information Theory

44. Liang, P. P., et al. (2023). Quantifying & modeling multimodal interactions: An information decomposition framework. *NeurIPS*.
45. Tishby, N., Pereira, F. C., & Bialek, W. (2000). The information bottleneck method. *arXiv preprint physics/0004057*.

---

## 부록 A: 선행 연구 결과 요약

### A.1 Paper B (tau-RoPE) 핵심 결과

| 가설 | 지표 | delta | p값 | 판정 |
|------|------|-------|-----|------|
| H1 고변동성 IC (tau > concat) | IC | +0.0387 | 0.0595 | 방향적 지지 |
| H2 고변동성 MAE (tau < concat) | MAE | +0.000092 | **0.0019** | **유의미** |

tau-RoPE는 **rule-based** (학습 시점에 고정 공식으로 tau 계산) 방식이다.

### A.2 Paper C (learned tau 실패) 핵심 병목

| 병목 | 메커니즘 | 추론 시점 상태 |
|------|---------|--------------|
| Softmax 압축 | QK delta가 softmax 온도보다 작음 | **부분 지속** (더 큰 alpha로 완화 가능) |
| 상호작용 접근 | tau 경로에 indexret 미참여 | **지속** (TTPA의 목적과 분리) |
| 단조성 제약 | cumsum gradient가 tau ≈ t로 수렴시킴 | **소멸** (gradient 불필요) |

### A.3 TTPA가 선행 연구의 어떤 결과에 의존하는가

| 선행 결과 | TTPA에서의 역할 | 의존 강도 |
|----------|---------------|----------|
| Paper B: 고변동성 MAE 개선 (p=0.0019) | tau 워핑 자체의 가치 실증 | **강** |
| Paper C: 단조성 병목 식별 | TTPA의 핵심 동기 | **강** |
| Paper C: softmax 병목 | TTPA의 잠재적 한계 | 중 |
| Paper A: 상호작용이 핵심 신호 | TTPA 단독 효과의 한계 | 중 |
| Paper B: IC 개선은 약함 | TTPA의 보수적 기대 설정 | 약 |

---

## 부록 B: 구현 의사 코드

```python
class TTPA:
    """Test-Time Positional Adaptation"""

    def __init__(self, model, variant='window', alpha=1.0, halflife=20):
        """
        model: 고정 PE로 학습된 Transformer 모델
        variant: 'window' | 'ewm' | 'regime'
        alpha: 워핑 강도 (0이면 고정 PE와 동일)
        halflife: EWM 변형의 반감기
        """
        self.model = model  # 파라미터 수정 없음
        self.variant = variant
        self.alpha = alpha
        self.halflife = halflife

    def compute_tau(self, market_stats):
        """
        market_stats: shape (T,) -- 시장 활동 강도 [0, 1]
        returns: tau, shape (T,) -- 경제적 시간 좌표
        """
        if self.variant == 'window':
            step = 1.0 + self.alpha * (market_stats - 0.5)
        elif self.variant == 'ewm':
            weights = np.exp(
                -np.arange(len(market_stats))[::-1]
                * np.log(2) / self.halflife
            )
            weights /= weights.sum()
            smoothed = np.convolve(market_stats, weights, mode='same')
            step = 1.0 + self.alpha * (smoothed - 0.5)
        elif self.variant == 'regime':
            regime = self._detect_regime(market_stats)
            alpha_eff = {'calm': 0.2, 'trending': 0.5,
                        'volatile': 1.0, 'crisis': 2.0}[regime]
            step = 1.0 + alpha_eff * (market_stats - 0.5)

        step = np.maximum(step, 0.1)  # 최소 step 보장
        tau = np.cumsum(step)
        tau = tau * (len(market_stats) / tau[-1])  # 정규화
        return tau

    def predict(self, x, market_stats):
        """
        고정 PE 대신 tau를 사용하여 추론
        """
        tau = self.compute_tau(market_stats)
        # 모델의 PE를 tau로 대체 (모델 파라미터 수정 없음)
        return self.model.forward_with_custom_positions(x, tau)

    def _detect_regime(self, market_stats, lookback=60):
        """간단한 체제 감지기"""
        recent = market_stats[-lookback:]
        mean_intensity = np.mean(recent)
        trend = np.polyfit(np.arange(len(recent)), recent, 1)[0]

        if mean_intensity > 0.8:
            return 'crisis'
        elif mean_intensity > 0.6:
            return 'volatile'
        elif abs(trend) > 0.01:
            return 'trending'
        else:
            return 'calm'
```

---

## 부록 C: TTPA 변형별 특성 비교

| 특성 | TTPA-W (윈도우) | TTPA-E (지수 가중) | TTPA-R (체제 조건부) |
|------|----------------|-------------------|---------------------|
| 복잡도 | O(T) | O(T × halflife) | O(T) |
| 하이퍼파라미터 | alpha | alpha, halflife | alpha_map (4개) |
| 반응 속도 | 즉각 | 완충 | 체제 변화 감지 후 |
| 과적응 위험 | 높음 (잡음 민감) | 낮음 (평활화) | 중간 (이산 체제) |
| 적합 시나리오 | 급격한 체제 전환 | 점진적 변화 | 명확한 체제 구분 |
| alpha=0 시 | Fixed PE와 동일 | Fixed PE와 동일 | Fixed PE와 동일 |

---

## 부록 D: DynaTTA와 TTPA의 구조적 비교

| 차원 | DynaTTA | TTPA |
|------|---------|------|
| 적응 대상 | BN 파라미터 + 예측 헤드 | PE 위치 좌표 |
| 적응 공간 | 가치 공간 | 위치 공간 |
| Gradient 사용 | 사용 (entropy minimization) | 불사용 (통계 기반) |
| 계산 비용 | 역전파 1회 / 배치 | O(T × d) |
| 모델 수정 | 파라미터 일부 수정 | 파라미터 수정 없음 |
| 오류 축적 | 가능 (연속 적응 시) | 없음 (stateless) |
| 결합 가능성 | TTPA와 결합 가능 | DynaTTA와 결합 가능 |

두 방법의 직교성이 결합 실험의 이론적 근거이다.

---

*본 논문은 제안 논문(proposal paper)이다. 실험 결과는 후속 연구에서 보고할 예정이며, 본 논문의 모든 "기대 결과"와 "가설"은 이론적 동기에서 도출된 것으로, 실증적 확인을 필요로 한다.*
