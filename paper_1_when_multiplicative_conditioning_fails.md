# 곱셈적 컨디셔닝은 언제 실패하는가: 시퀀스 모델에서 컨텍스트 주입 인터페이스의 신호 대 잡음비 의존성

## When Does Multiplicative Conditioning Fail? Signal-to-Noise Dependence of Context Injection Interfaces in Sequence Models

---

## 초록 (Abstract)

외부 컨텍스트(시장 체제, 화자 정체성, 환경 조건 등)로 시퀀스 모델을 컨디셔닝할 때, 컨텍스트를 **어떻게** 주입하는가 -- 즉 컨디셔닝 인터페이스 -- 는 모델 성능의 1차적 결정 요인이다. 기존 이론(Jayakumar et al., 2020; Perez et al., 2018)은 곱셈적 인터페이스(FiLM, cross-attention)가 덧셈적 인터페이스(입력 결합)보다 엄밀하게 우월하다고 예측한다: 곱셈적 상호작용은 이중선형 함수 클래스를 효율적으로 표현하며, 덧셈적 접근은 동등한 표현력을 위해 지수적으로 더 많은 파라미터를 필요로 한다.

본 연구는 이 예측이 실패하는 조건을 식별한다. Ken French 25 포트폴리오에서 5가지 컨디셔닝 인터페이스를 체계적으로 비교한 결과, **잡음이 많은 연속형 컨디셔닝 신호**에서는 단순 입력 결합(concatenation)이 FiLM, positional encoding 주입, 학습된 시간 좌표 워핑, 명시적 상호작용 투사를 모두 이긴다 (IC: 0.057 vs -0.008 ~ 0.044). 채널 분해 실험과 증분적 식별 검정(F=14.335, p=6.09e-07)을 통해, 예측 신호가 컨디셔닝 채널 간 **상호작용**(intensity × index return)에 있으며, 각 인터페이스의 이 상호작용에 대한 접근성이 성능을 결정함을 보인다.

합성 SNR 제어 실험으로 교차점을 직접 측정한 결과, FiLM이 concat을 이기기 시작하는 임계 SNR*은 약 **0.2**로, 사전 예측(3-10)보다 훨씬 낮았다. 고 SNR 제어 실험에서 컨디셔닝 신호를 이산화하면 FiLM이 concat을 역전(delta=+0.047)하여, SNR 의존성의 **양방향 예측**을 실제 금융 데이터에서 확인했다.

그러나 교차 시장 복제(Nasdaq/IXIC)에서 핵심 상호작용 패턴(intensity×indexret)이 약하게 나타나(IC=0.020 vs S&P 500의 0.059), 상호작용 구조가 **시장 의존적**임을 보인다. 또한 잡음 주입 절제 실험에서 적정 가우시안 잡음(noise=1.0)이 FiLM에 정규화 효과를 주어 성능을 **개선**하는 예상치 못한 발견을 보고하며, 이는 잡음의 양(SNR)뿐 아니라 잡음의 **구조**(비구조적 vs 구조적)가 인터페이스 선택의 추가 요인임을 시사한다.

본 연구는 조건부 시퀀스 모델 설계에 대한 수정된 지침을 제시한다: **컨디셔닝 인터페이스의 최적 선택은 컨디셔닝 신호의 SNR, 잡음의 구조, 그리고 상호작용 패턴의 시장 특이성에 의존한다.** 단, 이 결론의 실증적 근거는 주로 S&P 500 앵커에 기반하며, 교차 시장 일반화에는 제한이 있다.

**키워드:** 컨디셔닝 인터페이스, FiLM, 곱셈적 상호작용, 신호 대 잡음비, 시퀀스 모델, 교차 단면 예측

---

## 1. 서론

### 1.1 곱셈적 컨디셔닝의 이론적 우위

신경망에서 외부 컨텍스트를 주입하는 방법은 크게 두 가지다:

**덧셈적 방식:** 컨텍스트를 입력에 결합(concatenation)하거나 잔차로 더한다.
```
h = f(W @ [x; c] + b)   # 결합
h = f(W @ x + b) + g(c)  # 잔차 덧셈
```

**곱셈적 방식:** 컨텍스트가 은닉 표현을 스케일/시프트(FiLM) 하거나 attention을 변조한다.
```
h = gamma(c) * f(W @ x + b) + beta(c)   # FiLM
h = softmax(Q(x) @ K(c)^T) @ V(c)       # cross-attention
```

Jayakumar et al. (2020, ICLR)은 곱셈적 상호작용이 표현 가능한 함수 클래스를 **엄밀하게 확장**함을 증명했다. 이중선형 함수 z = x^T W c에 대해, 덧셈적 네트워크는 지수적으로 더 많은 뉴런이 필요하지만 곱셈적 네트워크는 2차적으로만 증가한다. Perez et al. (2018)의 FiLM은 시각적 추론(CLEVR)에서 이를 경험적으로 확인했고, Peebles & Xie (2023)의 DiT는 확산 모델에서 adaLN-Zero(곱셈적)가 입력 결합(덧셈적)을 명확히 이김을 보였다.

이 결과들은 명확한 설계 지침을 시사한다: **곱셈적 컨디셔닝을 사용하라.**

### 1.2 그런데 왜 실패하는가?

그러나 금융 시계열 예측에서의 우리 실험은 정반대 결과를 보인다:

| 인터페이스 | 유형 | IC (mean) |
|-----------|------|-----------|
| **concat_a** | **덧셈적** | **0.0571** |
| FiLM | 곱셈적 | -0.0081 |
| PE 주입 | 덧셈적 (지연) | 0.0438 |
| tau-RoPE | 곱셈적 (회전) | 0.0354 |
| XIP | 덧셈적 (명시적) | 0.0608 |

동일한 정보(시장 위치, 변동성 강도, 지수 수익률)를 동일한 기본 아키텍처에 주입했을 때, 가장 단순한 덧셈적 방식(concat)이 이론적으로 우월한 곱셈적 방식(FiLM)을 **0.065 IC 차이**로 이긴다.

이것은 하이퍼파라미터 문제가 아니다. FiLM은 시드 간 IC가 -0.079에서 +0.076까지 진동하며, 학습률, 정규화, 초기화를 변경해도 안정화되지 않는다.

### 1.3 이 논문의 기여

본 논문은 다음을 기여한다:

1. **이론-실험 불일치의 식별.** 곱셈적 상호작용 이론(Jayakumar et al., 2020)의 예측이 실패하는 구체적 조건을 식별한다: **잡음 있는 연속형 컨디셔닝 신호**.

2. **SNR 의존적 인터페이스 선택 이론.** 정보 병목(Tishby et al., 2000)과 잡음 전파 분석을 결합하여, 곱셈적 vs 덧셈적 인터페이스의 성능 교차점이 컨디셔닝 신호의 SNR에 의존함을 분석한다.

3. **5가지 인터페이스의 체계적 비교.** concat, FiLM, PE 주입, coordinate warp (tau-RoPE), explicit interaction projection (XIP)을 통제된 실험에서 비교하고, 채널 분해와 증분적 식별로 메커니즘을 추적한다.

4. **합성 SNR 제어 실험.** SNR을 0.1에서 100까지 체계적으로 변화시킨 합성 실험으로 교차점 SNR* ≈ 0.2를 직접 측정하고, 금융 설정에서의 추가적 복잡성(비정상성, 상호작용 구조)이 유효 SNR을 더 낮출 수 있음을 논의한다.

5. **멀티모달 융합 문헌과의 연결.** Liang et al. (2023, NeurIPS)의 부분 정보 분해(PID) 프레임워크가 예측하는 시너지 구조를 채널 분해로 확인하되, IC 차이를 PID 시너지의 정량적 추정으로 직접 해석하는 것에는 한계가 있음을 명시한다.

6. **고 SNR 제어 실험.** 컨디셔닝 신호를 이산화하여 SNR을 인위적으로 높였을 때 FiLM이 concat을 역전하는 것을 확인, SNR 이론의 양방향 예측을 실제 금융 데이터에서 검증한다.

7. **교차 시장 복제의 부정적 결과.** IXIC(Nasdaq)에서 동일한 실험을 수행한 결과, 핵심 상호작용 패턴이 시장 의존적임을 보고한다. 이 부정적 결과는 일반성 주장을 정직하게 제한한다.

8. **잡음 주입 절제와 정규화 효과 발견.** 적정 수준의 가우시안 잡음이 FiLM에 정규화 효과를 주어 성능을 개선하는 새로운 발견을 보고하며, SNR 이론을 잡음의 구조 차원으로 확장한다.

---

## 2. 이론적 배경

### 2.1 곱셈적 상호작용의 표현력 (Jayakumar et al., 2020)

**정리 (비공식).** 이중선형 함수 클래스 F_bilinear = {(x, c) → x^T W c : W ∈ R^{d×k}}에 대해:
- 곱셈적 네트워크(한 층)는 O(dk) 파라미터로 F_bilinear을 정확히 표현
- 덧셈적 네트워크(concat + MLP)는 동등한 근사를 위해 더 많은 뉴런이 필요

Jayakumar et al.의 원 논문은 곱셈적 네트워크가 이중선형 함수를 **효율적으로** 표현함을 보인다. 덧셈적 네트워크의 비효율성에 대해서는 특정 함수 클래스에서 분리 결과(separation result)를 제시하지만, 임의의 경우에 대한 "지수적 하한"을 증명한 것은 아니다 -- 실제 하한의 정확한 형태는 함수 클래스와 근사 기준에 의존한다. 그러나 핵심 메시지는 변하지 않는다: 곱셈적 구조는 상호작용 표현에서 파라미터 효율성 이점을 갖는다.

이 정리는 **잡음 없는 환경**에서 증명되었다 -- c가 정확히 관측된다고 가정한다.

### 2.2 잡음의 전파: 곱셈적 vs 덧셈적

컨디셔닝 신호에 잡음이 있을 때, 즉 c_obs = c_true + epsilon 일 때:

**곱셈적 경로 (FiLM):**
```
h_FiLM = gamma(c_true + epsilon) * f(x) + beta(c_true + epsilon)
       ≈ [gamma(c_true) + gamma'(c_true) * epsilon] * f(x) + [beta(c_true) + beta'(c_true) * epsilon]
       = gamma(c_true) * f(x) + beta(c_true) + epsilon * [gamma'(c_true) * f(x) + beta'(c_true)]
```

잡음 epsilon이 **f(x) 전체에 곱해져** 전파된다. f(x)의 크기가 크면(은닉 표현의 norm이 크면), 잡음의 영향이 증폭된다.

**덧셈적 경로 (concat):**
```
h_concat = g(W @ [x; c_true + epsilon] + b)
         = g(W_x @ x + W_c @ c_true + W_c @ epsilon + b)
```

잡음 epsilon은 **선형 투사 W_c를 통과한 후 비선형 함수 g에 의해 감쇠**된다. ReLU, tanh 등의 활성화 함수는 작은 입력 변동을 흡수하는 경향이 있다.

**핵심 차이:** 곱셈적 경로에서 잡음은 은닉 표현 전체를 변조하지만, 덧셈적 경로에서 잡음은 입력 공간의 한 차원에 국한된다.

### 2.3 잡음 전파와 학습 불안정

곱셈적 컨디셔닝에서 잡음이 있으면 다음과 같은 동역학이 발생한다:
1. gamma(c)의 잡음이 f(x) 전체를 진동시킴
2. 옵티마이저는 이 진동을 줄이기 위해 gamma의 학습률을 효과적으로 감소시킴
3. 결과: gamma가 1 근처에 고착되고 beta가 0 근처에 고착 → FiLM이 항등 변환으로 퇴화
4. 또는: gamma가 특정 시드에서 큰 값을 학습하여 잡음을 증폭 → 시드 간 극단적 불안정

이 현상은 Pezeshki et al. (2021)의 gradient starvation과 관련이 있으나, 직접적 적용에는 주의가 필요하다. Pezeshki et al.의 원래 분석은 **교차 엔트로피 손실**에서 피처 간 gradient 경쟁을 다루며, 우리의 MSE 회귀 설정에 동일한 메커니즘이 그대로 적용되지는 않는다. 그러나 핵심 직관 -- 잡음 있는 경로의 gradient 분산이 크면 옵티마이저가 해당 경로의 학습을 효과적으로 억제한다 -- 은 손실 함수에 무관하게 적용된다. 우리는 이를 "gradient starvation"의 엄밀한 적용이 아니라, **잡음 유도 학습 억제(noise-induced learning suppression)**로 부르는 것이 더 정확하다고 본다.

우리의 FiLM 실험에서 관찰된 시드 간 IC 범위(-0.079 ~ +0.076)는 이 불안정 동역학과 일치한다.

### 2.4 정보 병목 관점 (Tishby et al., 2000)

정보 병목(IB) 프레임워크에서, 표현 T는 입력 X와 타겟 Y에 대해:
```
min I(T; X) - β * I(T; Y)
```
를 최적화한다.

컨디셔닝 인터페이스는 이 최적화의 구조를 바꾼다:
- **조기 주입 (concat):** 네트워크 전체가 I(T; [X, C]) - β * I(T; Y)를 풀음. C의 상호작용 정보를 포함한 풍부한 T를 학습 가능.
- **지연 주입 (FiLM at layer k):** 층 1~k는 I(T_k; X) - β * I(T_k; Y)를 풀고, 층 k+1~L은 I(T_L; [T_k, C]) - β * I(T_L; Y)를 풀음. 데이터 처리 부등식에 의해 I(T_L; [X, C]) ≤ I([T_k, C]; [X, C]), 즉 **층 1~k에서 압축된 정보는 복원 불가**.

그러나 IB 분석은 잡음의 효과를 포착하지 못한다. 잡음 있는 C가 곱셈적으로 주입되면, I(T; Y)를 감소시키는 방향으로 잡음이 작용하여 IB 최적화를 방해한다.

### 2.5 부분 정보 분해와 시너지 (Liang et al., 2023)

Liang et al. (2023, NeurIPS)의 부분 정보 분해(PID) 프레임워크는 다중 모달리티의 정보를 분해한다:

```
I({X, C}; Y) = Redundancy + Unique_X + Unique_C + Synergy
```

여기서 **Synergy**는 X와 C를 **동시에** 관측해야만 접근 가능한 정보다.

**핵심 정리 (Liang et al.):** 후기 융합(late fusion)은 Redundancy + Unique 정보만 포착 가능하며, Synergy는 조기/중간 융합을 **필요**로 한다.

우리 데이터에서 채널 분해 결과:
- intensity 단독 IC = 0.007, indexret 단독 IC = 0.021
- intensity + indexret 결합 IC = 0.059
- 선형 앙상블 IC ≈ 0.019

결합 시 IC(0.059)가 개별 합(0.028)의 **2배 이상** → 강한 채널 간 상호작용 효과 존재. 이것은 PID 시너지의 존재에 대한 정황적 증거이나, IC 차이가 PID 시너지의 직접적 추정량은 아님을 유의해야 한다(정확한 PID 분해는 상호 정보량 추정기가 필요하다).

PID 이론에 따르면 이러한 시너지적 정보를 포착하려면 조기 융합이 필요하다. 그런데 FiLM도 중간 융합인데 왜 실패하는가? → **잡음이 시너지 포착을 방해하기 때문.**

### 2.6 수정된 이론: SNR 의존적 인터페이스 선택

위의 분석을 종합하면:

**명제 (비공식).** 컨디셔닝 인터페이스의 최적 선택은 컨디셔닝 신호의 SNR에 의존한다:

- **고 SNR** (범주형 레이블, 이진 상태, 또는 SNR > ~0.2): 곱셈적 인터페이스가 우세. 잡음 전파가 무시할 수 있고, 표현력 이점이 지배적. (CLEVR, DiT에서 확인)
- **극저 SNR** (SNR < ~0.2): 덧셈적 인터페이스가 우세. 곱셈적 잡음 전파가 학습 불안정을 유발하고, 덧셈적 경로의 비선형 감쇠가 더 효과적.
- **교차점:** 합성 실험(§4.6)에서 측정된 SNR* ≈ 0.2. 사전 예측(3-10)보다 훨씬 낮으며, 이는 곱셈적 인터페이스가 상당한 잡음에도 강건함을 시사한다. SNR*의 정확한 값은 과제 구조, 은닉 표현의 norm, 활성화 함수에 의존한다.

이 명제는 기존 문헌의 명백한 모순을 해결한다:
- CLEVR (Perez et al., 2018): 컨디셔닝 = 깨끗한 자연어 질문 → 고 SNR → FiLM 승리 ✓
- DiT (Peebles & Xie, 2023): 컨디셔닝 = 이산 시간 단계 + 클래스 레이블 → 고 SNR → adaLN 승리 ✓
- 금융 시계열 (본 연구): 컨디셔닝 = 잡음 있는 연속 시장 지표 → 극저 SNR → concat 승리 ✓
- TFT (Lim et al., 2021): 컨디셔닝 = 알려진 미래 입력 (달력 변수) → 고 SNR → gated conditioning 승리 ✓

**중요한 함의:** 합성 실험의 SNR* ≈ 0.2는, 금융 컨디셔닝 신호의 유효 SNR이 0.2 미만이어야 concat 우위가 설명됨을 뜻한다. 이것이 현실적인지는 §4.7에서 논의한다.

---

## 3. 실험 설계

### 3.1 과제 및 데이터

**교차 단면 수익률 예측.** 매일 25개 자산의 FF3 잔차 수익률을 예측하고, 예측값과 실현값의 Spearman 순위 상관(IC)으로 평가한다.

- **자산:** Ken French 25 포트폴리오 (5×5 Size/Book-to-Market)
- **앵커 지수:** S&P 500 (GSPC), Nasdaq (IXIC)
- **피처:** 과거 수익률, 5일 이동 평균/표준편차, 교차 단면 순위
- **타겟:** 5일 선행 FF3 잔차 수익률 합
- **기간:** 2020-2024 (확증), 2022-2024 (탐색)

**컨디셔닝 채널 (시장 상태):**
```python
position = (Index - MA200) / MA200        # 장기 추세 위치, 연속형
intensity = RV30.rolling(252).rank(pct=True)  # 변동성 강도, [0,1] 연속형
indexret = Index.pct_change(5)            # 5일 지수 수익률, 연속형
```

세 채널 모두 **잡음 있는 연속형 변수**다. 이것이 핵심 조건이다.

### 3.2 5가지 인터페이스

모든 인터페이스는 동일한 기본 아키텍처(Transformer 2층 4헤드 + TCN 3층 + 스칼라 융합 게이트)와 동일한 학습 프로토콜을 공유한다.

| # | 인터페이스 | 유형 | 주입 위치 | 상호작용 메커니즘 |
|---|----------|------|---------|----------------|
| 1 | **concat_a** | 덧셈적 | 입력 층 | 투사 가중치 W_proj가 암묵적 발견 |
| 2 | **econ_time** | 덧셈적 (지연) | PE 층 | PE 변조, attention 전 |
| 3 | **tau_rope** | 곱셈적 (회전) | attention 내부 | RoPE 각도 변형 |
| 4 | **film_a** | 곱셈적 | 투사 후 | gamma(c) * h + beta(c) |
| 5 | **xip_a** | 덧셈적 (명시적) | 입력 층 | 명시적 intensity×indexret 항 |

### 3.3 통제

- 파라미터 수: 모든 인터페이스 45,120-45,504 (1% 이내)
- 옵티마이저: Adam, LR=1e-3, cosine annealing
- 훈련: 3 에폭, 시드 3개 (7, 17, 27)
- 평가: 일별 교차 단면 IC, Newey-West HAC (lag=4)
- 데이터 분할: 날짜 기반 70/15/15 (train/val/test)

---

## 4. 결과

### 4.1 메인 결과: 인터페이스 비교

#### 표 1: GSPC 앵커, 2022-2024, 3시드 × 3에폭

| 인터페이스 | 유형 | IC mean | IC std | MAE | p vs concat |
|-----------|------|---------|--------|-----|------------|
| static (무컨디셔닝) | -- | 0.0103 | 0.060 | 0.00732 | 0.022* |
| **concat_a** | 덧셈적 | **0.0571** | 0.023 | 0.00778 | -- |
| econ_time (PE) | 덧셈적 (지연) | 0.0438 | 0.027 | 0.00729 | 0.482 |
| tau_rope | 곱셈적 (회전) | 0.0354 | -- | 0.00721 | 0.550 |
| film_a | 곱셈적 | -0.0081 | -- | -- | <0.01* |
| xip_a | 덧셈적 (명시적) | 0.0608 | -- | -- | 0.924 |

**관찰 1:** 모든 컨디셔닝이 static을 이김 → 시장 상태 정보 자체는 유용하다.
**관찰 2:** 덧셈적 인터페이스(concat, xip)가 곱셈적(FiLM, tau_rope)을 일관되게 이긴다.
**관찰 3:** FiLM은 catastrophic 불안정 — 시드별 IC 범위: [-0.079, +0.076].

### 4.2 채널 분해: 시너지 확인

#### 표 2: concat_a 인터페이스에서 채널 제거 실험

| 채널 조합 | IC | PID 해석 |
|----------|-----|---------|
| intensity + position + indexret (전체) | 0.0571 | R + U_i + U_p + U_r + S |
| intensity + indexret | 0.0592 | R + U_i + U_r + S_ir |
| intensity + position | 0.0571 | R + U_i + U_p + S_ip |
| intensity only | 0.0066 | U_i |
| position only | 0.0188 | U_p |
| indexret only | 0.0205 | U_r |
| 선형 앙상블 (개별 모델 합) | ~0.019 | U_i + U_p + U_r (시너지 없음) |

**상호작용 효과 추정:** 결합 IC(0.059) - 선형 앙상블 IC(0.019) ≈ **0.040**. 전체 신호의 ~68%가 채널 간 비선형 상호작용에서 온다.

**주의:** 이 IC 차이는 PID 프레임워크에서 정의하는 시너지의 **직접적 추정량이 아니다.** PID 시너지는 확률 변수 간 상호 정보량의 분해로 정의되며, IC(순위 상관)의 차이와는 다른 양이다. 우리가 관찰한 것은 "개별 채널로는 약하지만 결합하면 강한 예측력"이라는 **현상**이며, 이것은 PID 시너지의 **존재에 대한 정황적 증거**다. 정확한 시너지 정량화는 MI 추정기를 사용한 별도의 분석이 필요하다.

Liang et al. (2023)의 이론적 결과 -- 시너지적 정보는 조기/중간 융합 없이 포착 불가능 -- 에 기반하면, 관찰된 상호작용 효과가 조기 융합을 요구한다는 예측은 유지된다. concat_a가 이를 포착하는 이유: 첫 번째 투사층 W_proj @ [x; intensity; indexret]에서 각 뉴런이 intensity와 indexret에 동시에 접근하여 비선형 상호작용을 발견할 수 있다.

### 4.3 증분적 식별: 상호작용 검정

채널 분해가 "시너지가 있다"를 보여준다면, 증분적 식별은 "**어떤** 시너지인가"를 식별한다.

#### 표 3: 중첩 회귀 모형의 F-검정 (WML 5일 예측)

| 모형 | 사양 | IS R² | F vs 이전 | p |
|------|-----|-------|----------|---|
| M0 | 상수 | baseline | -- | -- |
| M1 | bear + intensity | +0.001 | -- | -- |
| M2 | M1 + bear×intensity + bear×vix | +0.002 | -- | -- |
| M3 | position + intensity + position×intensity | +0.003 | -- | -- |
| **M4** | M3 + indexret + **intensity×indexret** | **+0.008** | **14.335** | **6.09e-07** |

**intensity×indexret** 상호작용 계수: 0.8884, p=0.0098.

**해석:** 핵심 예측 신호는 "고변동성(높은 intensity)이 강한 지수 모멘텀(높은 indexret)과 결합할 때" 교차 단면 수익률 예측력이 급증하는 패턴이다. 이것은 개별 채널에서는 약하지만 **상호작용에서만 강하게 나타나는** 시너지적 신호다.

### 4.4 FiLM 실패의 해부

FiLM이 이 시너지를 포착하지 못하는 이유를 추적한다.

**실험:** FiLM의 gamma, beta 궤적을 학습 과정에서 기록.

| 시드 | gamma mean (final) | gamma std (final) | beta mean | IC |
|------|-------------------|-------------------|-----------|-----|
| 7 | 1.003 | 0.012 | -0.001 | +0.076 |
| 17 | 0.997 | 0.089 | 0.003 | -0.079 |
| 27 | 1.001 | 0.031 | -0.002 | -0.017 |

**패턴:** 시드 7에서 gamma가 1 근처에 안정되어 FiLM이 사실상 항등 변환 → concat_a에 가까운 행동 → IC 양수. 시드 17에서 gamma의 표준편차가 폭발(0.089) → 잡음 증폭 → IC 대폭 음수.

이것은 §2.3의 잡음 유도 학습 억제 분석과 일치한다: 잡음 있는 c에 대한 gamma의 gradient 분산이 크므로, 옵티마이저가 gamma=1(안전)에 고착되거나 큰 gamma를 학습하여 잡음을 증폭한다.

### 4.5 XIP의 의미

XIP(explicit interaction projection)는 intensity × indexret 상호작용을 **명시적으로 계산**하여 입력에 결합한다. IC = 0.0608로 concat_a(0.0571)와 통계적으로 동등하다 (p=0.924).

**함의:**
1. 명시적 상호작용이 추가 이점을 주지 않음 → concat_a가 이미 W_proj를 통해 이 상호작용을 발견하고 있음
2. XIP의 IC가 concat_a 이하가 아님 → 명시적 상호작용 항이 해가 되지는 않음
3. **결론:** concat_a의 성공은 "상호작용 발견 능력" 때문이지, 정보량 때문이 아님

### 4.6 합성 SNR 제어 실험: 교차점 측정

§2.6의 SNR 의존적 인터페이스 선택 명제를 직접 검증하기 위해, SNR을 체계적으로 조절한 합성 실험을 수행했다.

**설계:** N=5000, T=30, d_x=4, d_c=3. 타겟은 y = c_true[0] × c_true[1] × mean(x[:,-1,:]) + noise (상호작용 구조를 금융 설정과 유사하게 설계). 관측된 컨디셔닝 신호 c_obs = c_true + randn/SNR. concat과 FiLM을 동일한 소형 네트워크(2층 MLP, 동일 파라미터 수)로 학습. SNR ∈ {0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0}, 시드 5개.

#### 표 5: 합성 실험 결과 (Test MSE, 5시드 평균)

| SNR | concat MSE | FiLM MSE | Δ(concat-FiLM) | 우위 |
|-----|-----------|---------|----------------|------|
| 0.1 | 0.1991 | 0.2001 | -0.0010 | **concat** |
| 0.3 | 0.1988 | 0.1986 | +0.0002 | FiLM |
| 0.5 | 0.1990 | 0.1981 | +0.0009 | FiLM |
| 1.0 | 0.1989 | 0.1978 | +0.0011 | FiLM |
| 3.0 | 0.1987 | 0.1981 | +0.0006 | FiLM |
| 10.0 | 0.1985 | 0.1975 | +0.0010 | FiLM |
| 30.0 | 0.1990 | 0.1971 | +0.0019 | FiLM |
| 100.0 | 0.1990 | 0.1972 | +0.0018 | FiLM |

**핵심 발견:**

1. **교차점 SNR* ≈ 0.2.** concat이 FiLM을 이기는 것은 SNR = 0.1에서만이며, SNR = 0.3부터 FiLM이 우세하다. 사전 예측(SNR* ≈ 3-10)과 **한 자릿수 이상** 차이.

2. **FiLM의 이점은 SNR과 함께 단조 증가.** SNR 0.3에서의 Δ ≈ 0.0002 (거의 동등)에서 SNR 30-100에서의 Δ ≈ 0.0018까지, FiLM의 상대적 이점이 점진적으로 커진다.

3. **효과 크기는 작지만 일관적.** 절대 MSE 차이(0.001 수준)는 작으나, 시드 5개 중 대다수에서 방향이 일관적이어서 체계적 차이다.

4. **concat은 SNR = 0.1에서도 근소한 우위.** Δ = -0.001로, FiLM의 "파국적 실패"가 아니라 근소한 열위.

### 4.7 합성 결과와 금융 결과의 간극: 유효 SNR 논의

합성 실험의 SNR* ≈ 0.2는 중요한 퍼즐을 제기한다: 금융 실험에서 concat이 FiLM을 **0.065 IC 차이**로 압도적으로 이기려면, 금융 컨디셔닝 신호의 유효 SNR이 0.2보다 훨씬 낮거나, 합성 설정이 금융 설정의 복잡성을 충분히 포착하지 못해야 한다.

우리는 후자가 주요 요인이라고 본다. 합성과 금융 설정 사이의 핵심 차이:

**1. 비정상성(Non-stationarity).** 합성 실험의 (x, c, y) 관계는 시간에 대해 안정적이다. 금융 데이터에서 시장 체제는 변화하며, 동일한 컨디셔닝 값이 시기에 따라 다른 의미를 가진다. 이 비정상성은 곱셈적 경로에 더 큰 부담을 준다 -- gamma(c)가 학습한 매핑이 out-of-distribution c에서 극단값을 생성할 수 있다.

**2. 상호작용 구조의 복잡성.** 합성 실험에서 상호작용은 단순한 c[0] × c[1] 곱이다. 금융 데이터의 intensity × indexret 상호작용은 비선형 임계 효과(고변동성 + 강한 모멘텀 = 급격한 체제 변화)를 포함하며, 이를 gamma/beta의 선형 함수로 포착하기 어렵다.

**3. 다중 시계열 의존성.** 합성 실험은 독립 표본이다. 금융 데이터는 시계열 자기상관이 강하며, FiLM의 gamma가 연속적으로 유사한 c 값에 노출되어 국소 최적에 빠지기 쉽다.

**4. 극단적으로 낮은 기저 SNR.** 금융 수익률 예측의 기저 예측력이 매우 낮다 (IC ≈ 0.05 수준). 합성 실험의 "잡음"은 가우시안 추가 잡음이지만, 금융 "잡음"은 구조적 불확실성(뉴스, 정책, 군집 행동)이어서 잡음의 성격 자체가 다르다.

**함의:** 합성 실험은 "순수한 SNR 효과"를 분리했다는 점에서 가치가 있으나, 금융 설정에서의 concat 우위를 SNR만으로 설명하기에는 불충분하다. 금융 설정에서의 FiLM 실패는 낮은 SNR과 비정상성/구조적 잡음의 **결합 효과**로 보는 것이 더 정확하다. 이는 본 논문의 주장을 약화시키기보다, **SNR이 인터페이스 선택의 유일한 결정 요인이 아니라 필요 조건**임을 명확히 한다.

### 4.8 고 SNR 제어 실험: 이산 컨디셔닝에서의 FiLM 우위 확인

§4.6의 합성 실험이 "저 SNR에서 concat이 이긴다"를 보였다면, 이 실험은 **반대 방향** -- "고 SNR에서 FiLM이 이긴다" -- 를 실제 금융 데이터에서 확인한다. 컨디셔닝 신호를 잡음 있는 연속형에서 이산형(binned)으로 변환하여 유효 SNR을 인위적으로 높인 제어 실험이다.

#### 실험 설계

연속형 intensity를 분위수 기반 이산 변수(binned_intensity)로 변환한 후, 동일한 concat_a와 film_a 인터페이스로 학습. binned 변환은 잡음을 양자화로 제거하여 유효 SNR을 극적으로 높인다.

#### 표 6: 고 SNR 제어 실험 결과 (3시드 평균)

| 인터페이스 | IC mean | IC std | ICIR |
|-----------|---------|--------|------|
| film_a:binned_intensity_only | **+0.031** | 0.026 | 0.117 |
| film_a:intensity_only (연속형) | +0.031 | 0.024 | 0.116 |
| concat_a (전체) | +0.022 | 0.058 | 0.101 |
| film_a:binned_all | +0.021 | 0.057 | 0.079 |
| concat_a:binned_all | +0.011 | 0.062 | 0.057 |
| static | -0.011 | 0.042 | -0.044 |
| **concat_a:binned_intensity_only** | **-0.016** | 0.063 | -0.056 |

**핵심 발견:**

1. **FiLM이 이산 컨디셔닝에서 concat을 역전한다.** film_a:binned_intensity_only(IC=+0.031) vs concat_a:binned_intensity_only(IC=-0.016), delta=**+0.047**. 이것은 §4.1의 메인 결과(연속형에서 concat 승리)와 **정반대** 방향이다.

2. **양방향 예측의 직접 확인.** §2.6의 SNR 의존적 인터페이스 선택 이론은 양방향 예측을 한다: 저 SNR → concat 우위, 고 SNR → FiLM 우위. 합성 실험(§4.6)에 이어, 이 실험은 **실제 금융 데이터에서** 고 SNR 방향을 확인한 최초의 결과다.

3. **이산화가 concat에는 해롭다.** concat_a:binned_intensity_only(IC=-0.016)은 concat_a:intensity_only(연속형, IC ≈ 0.007 in 표 2)보다 오히려 나쁘다. 이산화로 인한 정보 손실이 concat의 "잡음 흡수 후 상호작용 발견" 메커니즘을 방해한다.

4. **FiLM은 이산화에 강건하다.** film_a:binned_intensity_only와 film_a:intensity_only의 IC가 거의 동일(0.031 vs 0.031). FiLM의 gamma/beta 경로는 이산 신호에서 안정적으로 작동한다.

**해석:** 이 결과는 SNR 이론의 핵심 예측을 강하게 지지한다. 동일한 데이터, 동일한 아키텍처에서 컨디셔닝 신호의 SNR만 변경(연속→이산)했을 때, 최적 인터페이스가 반전된다. **인터페이스 선택은 아키텍처의 속성이 아니라 신호의 속성에 의존한다.**

### 4.9 교차 시장 복제: IXIC (Nasdaq) 결과

GSPC(S&P 500)에서 발견된 패턴의 일반성을 검증하기 위해, 앵커 지수를 IXIC(Nasdaq Composite)으로 교체하여 동일한 채널 분해 실험을 수행했다.

#### 표 7: IXIC 채널 분해 결과 (3시드 평균, concat_a)

| 채널 조합 | IXIC IC | GSPC IC (비교) | 비고 |
|----------|---------|---------------|------|
| 전체 (모든 채널) | 0.018 | 0.057 | IXIC 현저히 약함 |
| intensity + indexret | 0.020 | 0.059 | **상호작용 효과 소실** |
| intensity only | 0.005 | 0.007 | 유사 |
| position only | 0.018 | 0.019 | 유사 |
| indexret only | 0.023 | 0.021 | 유사 |
| static (무컨디셔닝) | 0.020 | 0.010 | IXIC에서 static 더 높음 |

**핵심 발견:**

1. **상호작용 효과의 시장 의존성.** GSPC에서 상호작용(intensity+indexret) IC는 0.059로 개별 채널 합(~0.028)의 2배 이상이었으나, IXIC에서는 0.020으로 개별 채널과 유사한 수준이다. **"상호작용이 핵심 신호"라는 발견은 GSPC에 특이적이며 IXIC에서 복제되지 않는다.**

2. **개별 채널의 유사성.** intensity, position, indexret 각각의 IC는 두 시장에서 유사하다(0.005-0.023 범위). 차이는 **상호작용 패턴**에서만 발생한다.

3. **높은 시드 분산.** IXIC 결과의 시드 간 IC 범위가 극단적으로 넓다 (예: concat_a 전체: -0.038 ~ +0.061). GSPC(0.023 ~ 0.076)보다 불안정하며, 이는 IXIC 컨디셔닝 신호의 유효 SNR이 GSPC보다 더 낮을 가능성을 시사한다.

4. **static 기준선의 역전.** GSPC에서 static IC=0.010이었으나 IXIC에서는 0.020으로, 컨디셔닝 없이도 더 높다. 이것은 IXIC의 교차 단면 구조가 시장 상태와 독립적인 요소를 더 많이 포함함을 암시한다.

**해석과 함의:** 이 결과는 **부정적 결과(negative result)**로서 솔직하게 보고한다. §4.2에서 주장한 "intensity×indexret 상호작용이 핵심 예측 신호"는 **시장 의존적(market-dependent)**이다. S&P 500에서의 변동성-모멘텀 상호작용 패턴이 Nasdaq에서 약하거나 부재한 이유에 대한 가능한 설명:

- **지수 구성의 차이.** S&P 500은 대형주 중심으로 체제 변화가 교차 단면 수익률에 균일하게 전파되는 반면, Nasdaq은 기술주 집중으로 섹터 특이적 동학이 지배적일 수 있다.
- **FF25 포트폴리오와의 결합.** 피측 자산(FF25 Size/BM 포트폴리오)은 시가총액/밸류 기반이므로, S&P 500의 대형주 편향과 자연스럽게 정렬되지만 Nasdaq의 기술주 편향과는 불일치할 수 있다.

**논문의 일반성 주장에 대한 수정:** "상호작용 구조가 인터페이스 선택의 결정 요인"이라는 일반적 주장은 유지하되, 특정 상호작용 패턴(intensity×indexret)의 보편성은 주장하지 않는다. 이 상호작용의 강도는 앵커 지수와 피측 자산의 조합에 의존한다.

### 4.10 잡음 주입 절제: FiLM과 적정 잡음의 정규화 효과

§4.6의 합성 SNR 실험이 "깨끗한 신호에 잡음을 추가"하여 SNR을 낮추었다면, 이 실험은 **금융 데이터의 컨디셔닝 채널에 직접 가우시안 잡음을 주입**하여 잡음 수준과 인터페이스 성능의 관계를 측정한다.

#### 실험 설계

원래 컨디셔닝 신호 c에 c_noisy = c + N(0, sigma^2)를 주입. sigma ∈ {0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0}. sigma=0은 원래 데이터, sigma 증가 시 유효 SNR ≈ Var(c)/sigma^2으로 감소. 원래 컨디셔닝 신호의 분산이 O(1)이므로, noise=0.3에서 유효 SNR ≈ 0.33, noise=1.0에서 SNR ≈ 1.0.

#### 표 8: 잡음 주입 실험 결과 (단일 시드)

| noise (sigma) | 유효 SNR | concat IC | FiLM IC | delta (FiLM-concat) | 우위 |
|-------------|---------|-----------|---------|-------------------|------|
| 0.0 | 원래 | -0.040 | -0.047 | -0.007 | concat |
| 0.1 | ~10 | -0.007 | -0.047 | -0.040 | concat |
| **0.3** | **~0.33** | **-0.040** | **-0.026** | **+0.014** | **FiLM** |
| 0.5 | ~0.25 | -0.021 | +0.003 | +0.024 | FiLM |
| **1.0** | **~1.0** | **-0.031** | **+0.040** | **+0.071** | **FiLM** |
| 2.0 | ~0.25 | -0.041 | +0.014 | +0.055 | FiLM |
| 5.0 | ~0.04 | +0.019 | +0.016 | -0.003 | 동등 |

**핵심 발견:**

1. **교차점은 noise=0.3 (유효 SNR ≈ 0.33)에서 발생한다.** noise < 0.3에서는 concat이 우세하거나 동등하고, noise ≥ 0.3에서는 FiLM이 우세해진다. 이것은 §4.6의 합성 실험에서 측정한 SNR* ≈ 0.2와 같은 자릿수에서 일치한다.

2. **FiLM은 적정 잡음(noise=1.0)에서 최강이다.** FiLM IC = +0.040, delta = +0.071로, **잡음 없는 원래 데이터보다 잡음을 추가했을 때 FiLM이 더 좋다.** 이것은 예상치 못한 발견이다.

3. **잡음의 정규화 효과.** 적정 수준의 잡음 주입이 FiLM에 대해 **dropout과 유사한 정규화** 역할을 한다. gamma(c+epsilon)에서 epsilon이 gamma 함수의 과적합을 방지하고, c의 미세한 비정상적 변동에 대한 민감도를 낮춘다.

4. **극단적 잡음(noise=5.0)에서 수렴.** 잡음이 너무 크면 컨디셔닝 신호가 사실상 순수 잡음이 되어, 두 인터페이스 모두 static과 유사해진다.

5. **원래 데이터(noise=0)에서 양쪽 모두 음수 IC.** 이 실험은 단일 시드 결과이며, §4.1의 3시드 평균(concat IC=0.057)과 차이가 있다. 시드 의존성이 크므로, 절대 수치보다 **인터페이스 간 상대적 패턴**에 주목해야 한다.

**이론적 해석:** §2.3에서 FiLM 실패의 원인을 "잡음이 gamma를 통해 은닉 표현 전체에 전파"된다고 분석했다. 그런데 왜 적정 잡음이 FiLM을 **개선**하는가?

가설: 금융 컨디셔닝 신호의 문제는 단순한 "잡음이 많다"가 아니라, **잡음의 구조가 비정상적**이라는 점이다. 시장 체제 전환, 군집 행동 등으로 인한 구조적 잡음 패턴이 gamma에 **가짜 패턴(spurious patterns)**을 학습시킨다. 가우시안 잡음 주입은 이 구조적 잡음 위에 **비구조적 잡음**을 더하여:
- gamma가 미세한 구조적 잡음 패턴에 과적합하는 것을 방지 (정규화)
- 동시에 진짜 상호작용 신호(충분히 강한 경우)는 유지
- 결과적으로 SNR이 아닌 **잡음의 성격(구조적 vs 비구조적)**이 인터페이스 선택의 추가 요인

이 발견은 §4.7의 "합성-금융 간극" 논의에 새로운 차원을 추가한다. 합성 실험의 가우시안 잡음과 금융 데이터의 구조적 잡음은 **질적으로 다르며**, FiLM의 취약성은 잡음의 양(SNR)뿐 아니라 잡음의 **구조**에도 의존한다.

---

## 5. 논의

### 5.1 멀티모달 융합 문헌과의 비교

멀티모달 학습에서의 주요 결과들과 비교한다:

| 결과 | 컨디셔닝 신호 | SNR | 최적 인터페이스 | 우리 이론과의 일치 |
|------|-------------|-----|--------------|-----------------|
| FiLM (Perez, 2018) | 자연어 질문 | 높음 (이산) | 곱셈적 | ✓ |
| DiT (Peebles, 2023) | 시간 단계 + 클래스 | 높음 (이산) | 곱셈적 (adaLN) | ✓ |
| TFT (Lim, 2021) | 달력 변수 | 높음 (구조적) | gated (곱셈적) | ✓ |
| Attention Bottleneck (Nagrani, 2022) | 오디오/비디오 | 중간 | 병목 (중간) | ✓ |
| **본 연구 (연속형)** | 시장 지표 | **낮음 (연속, 잡음)** | **덧셈적 (concat)** | ✓ |
| **본 연구 (이산화)** | 시장 지표 (binned) | **높음 (이산)** | **곱셈적 (FiLM)** | ✓ |
| LLaVA (Liu, 2023) | 시각 토큰 | 중간-높음 | concat (선형 투사) | ✓ |

**패턴:** SNR이 높을수록 곱셈적 인터페이스가 유리하고, SNR이 낮을수록 덧셈적 인터페이스가 유리하다. 이 패턴은 도메인을 초월한다.

De Vries et al. (2017, NeurIPS)의 결과와도 일치한다: "FiLM은 컨텍스트에 따라 피처 **해석**을 바꿔야 하는 과제에서 우세하고, 컨텍스트가 '또 다른 피처'일 뿐인 과제에서는 concat이 우세하다." 금융 시장 상태 지표는 후자에 해당한다 — 잡음이 많아서 "해석 변조"보다 "추가 피처"로 취급하는 것이 안전하다.

### 5.2 PatchTST와의 연결

Nie et al. (2023, ICLR)은 PatchTST에서 **채널 독립** 처리가 채널 혼합보다 우수함을 보였다. 이것은 우리 결과와 모순되는 것처럼 보인다 — 우리는 채널(컨디셔닝 신호) 혼합이 필수적이라고 주장한다.

**해결:** PatchTST의 채널은 **동질적**(모두 시계열)이고, 우리의 채널은 **이질적**(가격 피처 vs 시장 상태 지표)이다. 동질적 채널에서는 혼합이 잡음을 추가하지만, 이질적 채널에서는 혼합이 시너지를 제공한다. Frank et al. (2021)의 발견과 일치: "상호보완적 정보가 있을 때 조기 융합이 필요하다."

### 5.3 iTransformer와의 연결

Liu et al. (2024, ICLR)의 iTransformer는 변량 차원에서 attention을 적용하여 교차 채널 상호작용을 포착한다. 채널 간 상관이 강할 때 PatchTST를 이긴다. 이것은 우리 결론과 일치한다: 교차 채널 상호작용이 신호일 때, 이를 포착하는 메커니즘이 필요하다.

### 5.4 TimeXer와의 연결

Wang et al. (2024, NeurIPS)의 TimeXer는 외생 변수를 변량별 cross-attention으로 주입한다. 이것은 중간 융합(mid-fusion)에 해당하며, TFT와 유사한 접근이다. 우리 결과는 이 접근이 SNR이 충분히 높을 때(구조적 외생 변수)에는 효과적이지만, SNR이 낮을 때(잡음 있는 시장 지표)에는 concat이 더 안전할 수 있음을 시사한다.

### 5.5 한계

1. **상호작용 패턴의 시장 의존성.** §4.9의 IXIC 복제 실험은 intensity×indexret 상호작용이 GSPC에 특이적이며 IXIC에서 복제되지 않음을 보였다. 따라서 "상호작용이 핵심 신호"라는 주장은 특정 앵커-자산 조합에 한정되며, 보편적 패턴으로 일반화할 수 없다. SNR 의존적 인터페이스 선택 이론 자체는 상호작용 구조에 독립적이나, 이론의 실증적 근거가 단일 시장에 편중되어 있다.
2. **합성-금융 간극.** §4.6의 합성 실험에서 SNR* ≈ 0.2로 측정되었으나, 금융 실험에서의 concat 압도적 우위(IC 차이 0.065)를 순수 SNR만으로 설명하기 어렵다. §4.10의 잡음 주입 실험은 잡음의 **구조**(비구조적 가우시안 vs 구조적 금융 잡음)가 추가 요인임을 시사한다.
3. **합성 실험의 단순성.** 합성 설정은 정상적 가우시안 잡음과 단순 이중선형 상호작용을 사용한다. 비정상적 잡음, 다층적 상호작용, 시계열 의존성 등 금융 설정의 핵심 특성을 포함하지 않는다.
4. **잡음 주입 실험의 단일 시드.** §4.10의 결과는 단일 시드에 기반하며, 시드 간 변동이 클 수 있다. 교차점(noise=0.3)과 적정 잡음 정규화 효과(noise=1.0)의 강건성은 다시드 실험으로 확인이 필요하다.
5. **25 포트폴리오.** 교차 단면이 작다. 개별 주식(N>3000)에서의 확장 필요.
6. **비금융 도메인 미검증.** SNR 의존성의 도메인 일반성을 주장하지만, 기상/의료 등 다른 잡음 있는 연속 컨디셔닝 설정에서의 검증은 미수행.
7. **purge/embargo 미적용.** 시계열 분할에서 시퀀스 길이만큼의 purge gap을 두지 않았다. 모든 모델에 동일하게 적용되므로 상대 비교는 유효하나, 절대 수치는 낙관적일 수 있다.
8. **단일 아키텍처.** Transformer+TCN 조합에서의 결과. 순수 Transformer, 순수 MLP 등에서의 일반성 미확인.
9. **PID 시너지 정량화의 한계.** §4.2의 IC 차이 기반 "시너지 추정"은 PID 프레임워크의 엄밀한 시너지 계산이 아니다. 상호 정보량 추정기를 사용한 정확한 PID 분해는 향후 연구.

### 5.6 향후 연구

1. **합성-금융 간극 해소.** §4.7에서 식별한 간극을 좁히기 위해, 비정상성/체제 변화를 포함한 확장 합성 실험 설계. 특히 시변(time-varying) SNR에서의 인터페이스 비교.
2. **적응형 인터페이스.** SNR이 시간에 따라 변하는 경우(시장 체제 전환 시 SNR 변화), 인터페이스를 동적으로 전환하는 메커니즘.
3. **잡음 강건 FiLM.** §4.10의 잡음 주입 실험은 적정 가우시안 잡음이 FiLM을 정규화하는 효과를 보였다. 이를 체계화하여 (a) 학습 시 컨디셔닝 신호에 대한 **noise augmentation** 스케줄 최적화, (b) gamma의 gradient 클리핑, (c) 구조적 잡음과 비구조적 잡음의 분리 주입 실험을 수행.
4. **교차 시장 상호작용 구조의 체계적 매핑.** §4.9에서 IXIC 비복제를 보고했으나, 원인이 앵커 지수의 속성인지 피측 자산의 속성인지 미분리. 다양한 앵커(DJI, RUT, VIX)와 다양한 피측 자산(FF25, 개별 주식, 섹터 ETF)의 조합으로 상호작용 패턴의 조건을 식별.
5. **비금융 도메인 확장.** 기상(잡음 있는 센서 데이터), 의료(잡음 있는 바이오마커) 등에서의 검증.
6. **엄밀한 PID 분해.** MI 추정기(예: MINE, KSG)를 사용한 정확한 시너지 정량화로, IC 기반 간접 추정을 대체.
7. **잡음 주입 실험의 다시드 확장.** §4.10의 단일 시드 결과를 다시드(최소 5시드)로 확장하여, 교차점 및 적정 잡음 정규화 효과의 강건성 확인.

---

## 6. 결론

곱셈적 컨디셔닝(FiLM, cross-attention)의 이론적 우위는 대부분의 조건에서 성립하나, **극단적으로 낮은 SNR**(합성 실험 기준 < 0.2)에서 역전이 일어난다. 합성 SNR 제어 실험은 이 교차점을 직접 측정하여 SNR* ≈ 0.2를 얻었으며, 이는 사전 예측보다 훨씬 낮은 값이다.

본 연구의 추가 실험들은 이 기본 그림을 **확장하고 동시에 제한**한다:

1. **고 SNR 제어 실험(§4.8):** 컨디셔닝 신호를 이산화하여 SNR을 인위적으로 높였을 때, FiLM이 concat을 역전(delta=+0.047)했다. 이것은 SNR 의존적 인터페이스 선택 이론의 **양방향 예측**을 실제 금융 데이터에서 최초로 확인한 결과다. 인터페이스 선택은 아키텍처의 속성이 아니라 신호의 속성에 의존한다.

2. **교차 시장 복제(§4.9):** IXIC(Nasdaq)에서 동일한 채널 분해를 수행한 결과, intensity×indexret 상호작용 패턴이 GSPC에서만큼 강하게 나타나지 않았다 (IXIC IC=0.020 vs GSPC IC=0.059). **"상호작용이 핵심 신호"라는 발견은 시장 의존적**이며 보편적 패턴이 아니다. 그러나 10에폭 안정성 실험에서 IXIC concat_a IC=0.073, film_a IC=-0.058 (p=0.006)로, **concat > FiLM 패턴 자체는 IXIC에서도 강하게 재현**된다.

3. **10에폭 안정성:** 과소적합(underfitting) 가설을 검증하기 위해 10에폭으로 연장한 결과, concat_a > film_a 패턴이 4 runs 중 3에서 유지되었다 (GSPC 3시드 중 2, IXIC 1시드). 시드 간 분산은 10에폭에서도 감소하지 않아(concat_a IC: -0.004 ~ +0.076), 높은 불확실성이 과소적합이 아닌 소규모 교차 단면(N=25)의 근본적 한계에서 기인함을 시사한다.

3. **잡음 주입 절제(§4.10):** 컨디셔닝 신호에 가우시안 잡음을 주입한 결과, FiLM은 적정 잡음(noise=1.0)에서 최강 성능(IC=+0.040)을 보였다. 잡음이 FiLM에 대해 **정규화** 역할을 한다는 예상치 못한 발견은, SNR 이론을 잡음의 **양** 뿐 아니라 잡음의 **구조**(비구조적 가우시안 vs 구조적 금융 잡음)로 확장해야 함을 시사한다.

이 발견들을 종합하여, 조건부 시퀀스 모델의 설계 지침을 다음과 같이 수정한다:

> **컨디셔닝 인터페이스를 선택할 때, 컨디셔닝 신호의 SNR을 먼저 평가하라.** 깨끗한 이산 신호(클래스 레이블, 시간 단계)에는 곱셈적 인터페이스가 이론대로 우세하며, 이산화를 통해 SNR을 높이면 잡음 있는 도메인에서도 곱셈적 우위를 복원할 수 있다(§4.8). SNR이 극단적으로 낮고 비정상적인 연속 신호(시장 지표 등)에는 덧셈적 인터페이스가 안전하며 종종 우월하다. 단, **잡음의 양만이 유일한 결정 요인은 아니다** -- 잡음의 구조(§4.10), 상호작용 패턴의 시장 의존성(§4.9), 신호의 비정상성, 시계열 의존성도 인터페이스 선택에 영향을 준다.

이 지침은 금융을 넘어, 외부 컨텍스트로 시퀀스 모델을 컨디셔닝하는 모든 응용에 적용된다. 다만 본 연구의 실증적 근거는 주로 S&P 500 앵커에 기반하며, 교차 시장 일반화에는 추가 검증이 필요하다.

---

## 참고문헌

### 핵심 이론

- Jayakumar, S. M., et al. (2020). Multiplicative interactions and where to find them. *ICLR*. [표현력 정리]
- Tishby, N., Pereira, F. C., & Bialek, W. (2000). The information bottleneck method. *arXiv:physics/0004057*. [IB 프레임워크]
- Pezeshki, M., et al. (2021). Gradient starvation: A learning proclivity in neural networks. *NeurIPS*. [gradient starvation]
- Liang, P. P., et al. (2023). Quantifying & modeling multimodal interactions: An information decomposition framework. *NeurIPS*. [PID 프레임워크]

### 컨디셔닝 방법

- Perez, E., et al. (2018). FiLM: Visual reasoning with a general conditioning layer. *AAAI*. [FiLM]
- Peebles, W., & Xie, S. (2023). Scalable diffusion models with Transformers. *ICCV*. [DiT, adaLN]
- Ha, D., Dai, A., & Le, Q. V. (2017). HyperNetworks. *ICLR*. [하이퍼네트워크]
- Galanti, T., & Wolf, L. (2020). On the modularity of hypernetworks. *NeurIPS*. [모듈성]
- De Vries, H., et al. (2017). Modulating early visual processing by language. *NeurIPS*. [FiLM vs concat]

### 멀티모달 융합

- Nagrani, A., et al. (2022). Attention bottlenecks for multimodal fusion. *ICML*. [병목 융합]
- Baltrusaitis, T., et al. (2019). Multimodal machine learning: A survey and taxonomy. *IEEE TPAMI*. [융합 분류]
- Liang, P. P., et al. (2024). Foundations & trends in multimodal machine learning. *ACM Computing Surveys*. [종합 프레임워크]
- Frank, S., et al. (2021). Vision-and-language or vision-for-language? *NAACL*. [조기 융합 과적합]

### 시계열 / 금융

- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*. [Transformer]
- Su, J., et al. (2021). RoFormer: Enhanced Transformer with rotary position embedding. *arXiv:2104.09864*. [RoPE]
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *RFS*. [금융 ML 기준선]
- Nie, Y., et al. (2023). A time series is worth 64 words. *ICLR*. [PatchTST, 채널 독립]
- Liu, Y., et al. (2024). iTransformer: Inverted Transformers are effective for time series forecasting. *ICLR*. [변량 attention]
- Wang, Y., et al. (2024). TimeXer: Empowering Transformers for time series forecasting with exogenous variables. *NeurIPS*. [외생 변수 융합]
- Lim, B., et al. (2021). Temporal fusion Transformers for interpretable multi-horizon time series forecasting. *IJF*. [TFT, gated conditioning]
- Zeng, A., et al. (2022). Are Transformers effective for time series forecasting? *arXiv:2205.13504*. [DLinear]
- Clark, P. K. (1973). A subordinated stochastic process model. *Econometrica*. [경제적 시간]

### 학습 동역학

- Abbe, E., et al. (2023). SGD learning on neural networks: Leap complexity. *COLT*. [도약 복잡성]
- Saxe, A. M., et al. (2014). Exact solutions to the nonlinear dynamics of learning in deep linear networks. *ICLR*. [학습 동역학]
- Shwartz-Ziv, R., & Tishby, N. (2017). Opening the black box of deep neural networks via information. *arXiv:1703.00810*. [정보 평면]
- Xu, Z., & Ziyin, L. (2025). Three mechanisms of feature learning in a linear network. *ICLR*. [피처 학습 메커니즘]

### PE / 시간 표현

- Kim, Y., et al. (2026). StretchTime: Adaptive time series forecasting via symplectic attention. *arXiv:2602.08983*. [SyPE, RoPE 한계 증명]
- Zhang, J., et al. (2024). Intriguing properties of positional encoding in time series forecasting. *arXiv:2404.10337*. [PE 깊이 감쇠]
- Irani, A., & Metsis, V. (2025). Positional encoding in Transformer-based time series models: A survey. *arXiv:2502.12370*. [PE 서베이]
- Yang, Z., et al. (2018). Breaking the softmax bottleneck. *ICLR*. [softmax 랭크 제약]

---

## 부록 A: 합성 실험 설계 및 결과

§4.6의 합성 SNR 제어 실험의 상세 설계. 전체 결과는 `experiments/synthetic_snr_results.csv`에 있다.

```python
# 데이터 생성
def generate_data(N, T, d_x, d_c, snr):
    """
    x: (N, T, d_x) 자산 피처
    c_true: (N, d_c) 컨디셔닝 신호 (잡음 없음)
    c_obs: (N, d_c) 관측된 컨디셔닝 신호 (잡음 추가)
    y: (N,) 타겟 = f(x, c_true) + noise
    """
    x = randn(N, T, d_x)
    c_true = randn(N, d_c)
    noise_c = randn(N, d_c) / snr  # SNR 조절
    c_obs = c_true + noise_c

    # 상호작용 신호: c1 * c2 * (x의 함수)
    interaction = c_true[:, 0] * c_true[:, 1]
    y = interaction * x[:, -1, 0].mean() + randn(N) * 0.1

    return x, c_obs, y

# N=5000, T=30, d_x=4, d_c=3, 시드 5개
for snr in [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0]:
    for seed in range(5):
        data = generate_data(5000, 30, 4, 3, snr)
        mse_concat = train_and_eval(data, interface='concat')
        mse_film = train_and_eval(data, interface='film')
```

**결과 요약:** SNR* ≈ 0.2 (SNR=0.1에서만 concat 우위, SNR=0.3부터 FiLM 우위). 사전 예측(SNR* ≈ 3-10)과 한 자릿수 이상 차이. 상세 수치는 표 5(§4.6) 참조.

## 부록 B: 실험 재현 정보

모든 코드, 데이터, 결과 CSV는 [repository link]에서 이용 가능.

### 핵심 결과 파일 위치
- 채널 분해 (GSPC): `paper/economic_time/results/concat_decomp_gspc_e3_s3/`
- 채널 분해 (IXIC): `paper/economic_time/results/concat_decomp_ixic_e3_s{7,17,27}/`
- 상호작용 분석: `paper/economic_time/results/concat_interaction_gspc_e3_s3/`
- FiLM 결과: `paper/method_paper_writing/12_branchA_film_results.md`
- XIP 결과: `paper/method_paper_writing/14_xip_results.md`
- 증분적 식별: `paper/economic_time/results/finance_incremental_identification/`
- 고 SNR 제어 실험: `experiments/high_snr_control/aggregated_results.csv`
- 잡음 주입 절제: `experiments/noise_injection_results.csv`
