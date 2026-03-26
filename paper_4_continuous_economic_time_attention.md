# 연속 경제적 시간 Attention: ContiFormer의 ODE 시간 변수를 경제적 시간으로 대체하기

## Continuous Economic Time Attention: Replacing the ODE Time Variable in ContiFormer with Economic Time

---

## 초록 (Abstract)

ContiFormer(Chen et al., NeurIPS 2024)는 이산 attention을 연속 시간 동역학으로 확장하여, Neural ODE를 통해 attention 상태를 dH/dt = f(H, t)로 진화시킨다. 그러나 ContiFormer의 시간 변수 t는 여전히 **달력 시간(clock time)**이며, 금융 시계열의 핵심 특성인 비균일 정보 밀도를 반영하지 못한다. 본 연구는 Clark(1973)의 subordinated process 이론과 Zagatti et al.(2024, AISTATS)의 시간 변환 정리(time-change theorem)를 결합하여, ContiFormer의 ODE 시간 변수를 **경제적 시간(economic time)** τ(t) = ∫₀ᵗ λ(s)ds로 대체하는 Economic ODE Attention(EOA)을 제안한다.

이 대체는 세 가지 구조적 결과를 유도한다: (1) 고변동성 구간에서 τ가 빠르게 진행하여 ODE 적분 스텝이 자연스럽게 증가하고, attention 해상도가 적응적으로 조밀해진다. (2) 저변동성 구간에서 τ가 느리게 진행하여 계산이 절약된다. (3) τ의 단조 증가성은 적분 구조(양의 함수의 적분)에 의해 자동 보장되므로, 별도의 단조성 제약이 불필요하다.

이론적 분석에서 EOA가 Paper 2(Kim et al., 2026)에서 식별된 세 가지 병목 -- softmax 압축, 상호작용 접근 제약, 단조성 제약 -- 을 구조적으로 회피함을 보인다. 연속 ODE attention은 softmax를 사용하지 않으며, ODE 상태 H는 전체 표현을 포함하여 피처 상호작용에 접근 가능하고, τ의 단조성은 설계에 내재되어 있다. 시간 변환 정리에 의해, τ-시간에서의 균일 과정(uniform process)은 t-시간에서의 강도 가중 과정(intensity-weighted process)과 수학적으로 동치이며, 이는 EOA가 Clark의 subordinated process를 attention 동역학에 직접 구현함을 의미한다.

**키워드:** 연속 시간 attention, Neural ODE, 경제적 시간, subordinated process, 시간 변환 정리, ContiFormer

---

## 1. 서론

### 1.1 연속 시간 Attention의 등장

Transformer(Vaswani et al., 2017) 이후, attention 메커니즘은 이산 토큰 간의 관계를 계산하는 프레임워크로 자리잡았다. 그러나 시계열 데이터 -- 특히 불규칙 간격(irregularly-sampled) 시계열 -- 에서 이산 attention은 근본적 한계를 갖는다: 관측 시점 사이의 연속적 동역학을 모형화할 수 없다.

이 문제를 해결하기 위해 두 가지 접근법이 발전했다:

**Neural ODE 계열.** Chen et al.(2018)의 Neural ODE는 은닉 상태의 연속 시간 진화를 dh/dt = f(h, t; θ)로 모형화한다. 이를 시계열에 적용한 Latent ODE(Rubanova et al., 2019), ODE-RNN(De Brouwer et al., 2019), Neural CDE(Kidger et al., 2020)는 불규칙 관측을 자연스럽게 처리한다.

**연속 Attention 계열.** 최근의 ContiFormer(Chen et al., NeurIPS 2024)는 이 두 흐름을 통합한다: attention 상태 자체가 Neural ODE를 따라 연속적으로 진화하며, 이산 attention의 softmax-weighted sum을 연속 적분으로 대체한다.

### 1.2 ContiFormer의 구조와 한계

ContiFormer의 핵심 정식화:

```
dH/dt = f_θ(H(t), t),    H(0) = H₀
```

여기서 H(t) ∈ ℝ^{d_model}은 attention 상태, f_θ는 파라미터화된 벡터장, t는 시간 변수이다. H₀는 입력 시퀀스의 초기 인코딩에서 유도된다.

이산 attention에서 attention(Q, K, V) = softmax(QK^T/√d)V 인 반면, ContiFormer에서는:

```
Attention(q, t) = ∫₀ᵀ α(q, H(s)) · V(H(s)) ds
```

여기서 α는 연속 attention 가중치 함수이다.

**그러나 ContiFormer의 시간 변수 t는 여전히 달력 시간이다.** ODE는 물리적 시간 축을 따라 균일하게 적분되며, 시장 활동의 강도 변화를 반영하지 않는다. 이는 다음과 같은 비효율을 야기한다:

- 고변동성 구간(정보 밀도 높음): ODE가 동일한 스텝 크기로 적분하여 중요한 동역학을 놓칠 수 있다.
- 저변동성 구간(정보 밀도 낮음): 불필요하게 세밀한 적분이 수행되어 계산이 낭비된다.

### 1.3 핵심 아이디어

본 연구의 핵심 관찰은 단순하다:

> ContiFormer의 ODE에서 달력 시간 t를 경제적 시간 τ(t) = ∫₀ᵗ λ(s)ds로 대체하면, ODE는 시장 활동의 강도에 비례하여 적분 해상도를 자동 조절한다.

이것은 새로운 아키텍처의 제안이 아니라, 기존 연속 시간 attention 프레임워크의 **시간 축 재정의**이다. Clark(1973)이 브라운 운동의 시간 축을 거래 시간으로 대체하여 heavy-tailed 분포를 설명한 것과 정확히 같은 논리를 attention 동역학에 적용한다.

### 1.4 왜 지금인가: 두 이론의 교차점

이 연결이 가능한 것은 최근 두 가지 발전 덕분이다:

1. **ContiFormer(Chen et al., 2024)**: attention을 Neural ODE로 연속화하여, 시간 변수의 재정의가 의미를 갖는 프레임워크를 제공했다. 이산 attention에서는 "시간 변수"가 정수 인덱스이므로 연속 변환이 정의되지 않는다.

2. **Zagatti et al.(2024, AISTATS)**: 시간 변환 정리(time-change theorem)를 신경망 기반 시간적 점과정(neural temporal point process)에 구현하여, 강도 함수를 통한 시간 변환이 학습 가능하고 통계적으로 건전함을 보였다.

그러나 **이 두 이론을 결합한 연구는 아직 없다**: 시간 변환 정리를 Transformer attention의 연속 동역학에 적용하는 것이다. 본 연구가 이 교차점을 최초로 탐구한다.

### 1.5 Paper 2의 세 가지 병목과의 관계

Paper 2(Kim et al., 2026, "잘못된 귀납적 편향")에서 식별된 세 가지 병목은 tau-RoPE가 이산 Transformer에서 실패하는 이유를 설명한다:

1. **Softmax 압축 병목**: QK 공간에서의 순서 변화가 softmax 후 소실
2. **상호작용 접근 병목**: 시간 좌표 경로가 피처 상호작용에 접근 불가
3. **단조성 제약 병목**: cumsum 구조가 τ를 물리적 시간과 거의 동일하게 만듦

본 연구는 이 세 병목이 **이산 attention + RoPE라는 특정 구현**의 한계이며, 연속 ODE attention으로의 전환이 이 세 병목을 모두 구조적으로 회피함을 보인다.

### 1.6 기여

1. **방법론적 기여.** ContiFormer의 ODE 시간 변수를 경제적 시간으로 대체하는 Economic ODE Attention(EOA)를 정식화한다.

2. **이론적 기여.** 시간 변환 정리를 통해 EOA의 수학적 정당성을 제공하고, Paper 2의 세 가지 병목을 연속 동역학이 회피하는 메커니즘을 분석한다.

3. **개념적 기여.** Clark(1973)의 subordinated process를 50년 만에 attention 동역학에 직접 구현하며, 이것이 단순한 positional encoding 수정이 아닌 동역학 자체의 시간 축 변환임을 명확히 한다.

4. **실험적 기여.** 금융 시계열에서의 실험 설계와 기대 결과를 제시한다.

---

## 2. 이론적 배경

### 2.1 Subordinated Process (Clark, 1973)

#### 2.1.1 핵심 구조

Clark(1973)은 면화(cotton) 선물 가격의 heavy-tailed 분포를 설명하기 위해 subordinated process를 도입했다. 핵심 구조:

```
X(t) = W(T(t))
```

여기서:
- W(·)는 표준 위너 과정(Wiener process)
- T(t)는 **subordinator** -- 단조 증가하는 확률 과정
- X(t)는 관측되는 가격 과정

T(t)의 증분 ΔT가 시점에 따라 달라지면 -- 예를 들어 거래 활동이 활발한 시점에서 ΔT가 크면 -- X(t)는 혼합 정규분포(mixture of normals)를 따르며, 이것이 heavy tail을 생성한다.

#### 2.1.2 경제적 직관

달력 시간이 아닌 **경제적 시간**의 관점에서:
- 시장이 활발한 날: "경제적으로" 며칠에 해당하는 정보가 하루에 처리됨 → T(t)가 빠르게 증가
- 시장이 조용한 날: "경제적으로" 몇 시간에 해당하는 정보만 처리됨 → T(t)가 느리게 증가

Mandelbrot & Taylor(1967)가 이 아이디어를 최초로 제안했고, Clark(1973)이 subordination으로 정식화했으며, Ane & Geman(2000)이 거래 횟수를 T의 대리변수로 사용하여 실증적으로 확인했다.

#### 2.1.3 현대적 재해석

Lopez de Prado(2018)는 이를 실무에 도입하여, 달력 시간 대신 거래량(volume bar), 달러(dollar bar), tick(tick bar) 기반 샘플링을 제안했다. 이 "정보 기반 샘플링(information-driven sampling)"은 통계적 성질을 개선한다:
- 수익률이 더 정규분포에 가까워짐
- 자기상관이 감소함
- 분산이 더 안정적이 됨

그러나 이러한 바 기반 접근은 **이산적 리샘플링**이다. 연속 시간 프레임워크에서의 시간 변환은 다른 수학적 도구를 필요로 한다.

### 2.2 시간 변환 정리 (Time-Change Theorem)

#### 2.2.1 고전적 정리

시간 변환 정리는 확률론의 핵심 결과 중 하나이다:

**정리 (Dambis-Dubins-Schwarz).** M이 연속 로컬 마팅게일이고 ⟨M⟩_t가 그 2차 변동(quadratic variation)이면, ⟨M⟩의 역함수 τ에 대해:

```
M(τ⁻¹(s)) 는 표준 브라운 운동이다
```

즉, **임의의 연속 마팅게일은 적절한 시간 변환을 통해 브라운 운동으로 변환할 수 있다.** 시간을 적절히 "재조정"하면 복잡한 과정이 단순해진다.

#### 2.2.2 신경망에의 적용 (Zagatti et al., 2024)

Zagatti et al.(2024, AISTATS)은 이 정리를 신경망 기반 시간적 점과정(neural temporal point process)에 적용했다. 핵심 결과:

조건부 강도 함수(conditional intensity function) λ*(t)가 주어질 때, 보상 과정(compensated process):

```
τ(t) = ∫₀ᵗ λ*(s) ds
```

로의 시간 변환은 원래의 비균일 점과정을 **단위 강도 포아송 과정(unit-rate Poisson process)**으로 변환한다.

이것이 의미하는 바:

| τ-시간에서 | t-시간에서 |
|-----------|-----------|
| 균일한 과정 | 강도에 비례하는 비균일 과정 |
| 등간격 스텝 | 활동이 많은 곳에서 조밀한 스텝 |
| 정상적(stationary) 동역학 | 비정상적(non-stationary) 동역학 |

Zagatti et al.의 기여는 이 변환이 **학습 가능한 강도 함수** λ*_θ(t)와 결합될 때도 통계적으로 건전함을 보인 것이다: 시간 변환된 과정이 단위 강도 포아송을 따르는지를 검정하는 것이 모델 진단(goodness-of-fit test)에 직접 사용될 수 있다.

### 2.3 ContiFormer: 연속 시간 Attention

#### 2.3.1 아키텍처 개요

ContiFormer(Chen et al., NeurIPS 2024)는 Transformer의 self-attention을 연속 시간 동역학으로 확장한다.

**이산 attention (표준 Transformer):**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

이것은 키-밸류 쌍 {(k_i, v_i)}_{i=1}^{N}에 대한 이산 가중합이다.

**연속 attention (ContiFormer):**

키-밸류 쌍을 연속 함수 K(s), V(s)로 확장하면:

```
Attention(q, t) = ∫_S α(q, K(s)) · V(s) ds
```

여기서 α는 attention 가중치 함수이고, 적분은 시간 도메인 S 위에서 수행된다.

#### 2.3.2 ODE 기반 진화

ContiFormer에서 K(s)와 V(s)는 Neural ODE를 따라 진화한다:

```
dK/ds = g_K(K(s), s)
dV/ds = g_V(V(s), s)
```

이는 이산 attention의 "키와 밸류가 고정된 점"이라는 가정을 완화하여, 키와 밸류가 시간에 따라 연속적으로 변화하는 동역학을 모형화한다.

#### 2.3.3 달력 시간의 암묵적 가정

ContiFormer의 모든 ODE에서 독립 변수 s(또는 t)는 **달력 시간**이다. 적분 구간 [0, T]는 물리적 시간 축이며, ODE 솔버(예: Dormand-Prince)의 적응적 스텝 크기는 **수치적 오차**에 의해 결정되지, **정보 밀도**에 의해 결정되지 않는다.

이것은 암묵적으로 다음을 가정한다:
- 모든 시점의 정보가 동등한 가치를 갖는다.
- ODE 해상도는 동역학의 수학적 stiffness에만 의존한다.

금융 시계열에서 이 가정은 명백히 위배된다.

### 2.4 RoPE의 한계: StretchTime 정리

Paper 2에서 인용한 StretchTime(Kim et al., 2026)의 핵심 정리를 간략히 재서술한다:

**정리 (StretchTime, Theorem 3.1).** 비선형(non-affine) 워핑 함수 τ에 대해, RoPE의 상대 위치 인코딩:

```
θ(m-n) = ω₀(τ(m) - τ(n))  (mod 2π)
```

를 만족하는 주파수 θ가 존재하지 않는다.

**의미:** RoPE는 구조적으로 비선형 시간 워핑을 표현할 수 없다. 경제적 시간 τ(t)가 비선형(일반적으로 비선형임)일 때, τ-RoPE는 정확한 상대 위치를 인코딩하지 못한다. 이것이 Paper 2에서 tau-RoPE가 이산 Transformer에서 실패하는 수학적 원인 중 하나이다.

**핵심 통찰:** 이 한계는 **이산 attention + RoPE 조합**의 한계이지, 경제적 시간 개념 자체의 한계가 아니다. 연속 ODE attention에서는 RoPE를 사용하지 않으므로, 이 정리의 제약을 받지 않는다.

---

## 3. 방법: Economic ODE Attention (EOA)

### 3.1 핵심 정식화

ContiFormer의 원래 ODE:

```
dH/dt = f_θ(H(t), t),    H(0) = H₀     ... (원래)
```

EOA의 ODE:

```
dH/dτ = f_θ(H(τ), τ),    H(0) = H₀     ... (EOA)
```

여기서 경제적 시간:

```
τ(t) = ∫₀ᵗ λ(s) ds                       ... (시간 변환)
```

λ(s) > 0은 시점 s에서의 **시장 활동 강도(market activity intensity)**이다.

### 3.2 τ의 구성: 연속 강도 적분

#### 3.2.1 강도 신호 λ(s)의 선택

λ(s)는 시점 s에서의 정보 밀도를 반영하는 양의 스칼라 함수이다. 후보:

**A. 실현 변동성(Realized Volatility):**
```
λ_RV(t) = √(∑_{i=1}^{n} r²_{t,i})
```
여기서 r_{t,i}는 일중(intraday) 수익률. 고변동성 = 높은 강도.

**B. 거래량(Volume):**
```
λ_V(t) = Volume(t) / Volumē
```
정규화된 거래량. Clark(1973)의 원래 대리변수.

**C. 복합 강도(Composite Intensity):**
```
λ_C(t) = α · σ_realized(t) + (1-α) · V_normalized(t)
```
변동성과 거래량의 가중 결합.

**D. 학습된 강도(Learned Intensity):**
```
λ_θ(t) = softplus(MLP_θ(x_t))
```
입력 피처로부터 학습. softplus는 양수 보장.

본 연구에서는 **C(복합 강도)**를 기본값으로 사용하되, D(학습된 강도)와의 비교를 포함한다. Paper 2에서 학습된 τ의 문제점(단조성 제약 등)을 지적했으나, 그 문제는 cumsum + RoPE 구조에서 발생한 것이지 학습된 강도 자체의 문제가 아니다.

#### 3.2.2 연속 적분의 이산화

실무적으로, τ(t)의 정확한 연속 적분은 불가능하므로 이산 근사를 사용한다:

```
τ(t_k) = ∑_{j=0}^{k-1} λ(t_j) · Δt_j
```

여기서 Δt_j = t_{j+1} - t_j (등간격이면 Δt). 이것은 cumsum 구조와 형태가 동일하지만, **사용되는 맥락이 다르다**: Paper 2에서는 cumsum의 결과가 RoPE의 위치 인수로 사용되어 softmax 압축 병목을 통과했지만, 여기서는 ODE 솔버의 **시간 축 자체**를 재정의한다.

#### 3.2.3 정규화

τ의 스케일이 ODE 솔버의 수치적 안정성에 영향을 미치므로, 배치 내 정규화를 적용한다:

```
τ̃(t) = τ(t) · (T / τ(T))
```

이것은 τ̃의 총 구간을 [0, T]로 유지하면서, 구간 내 분포를 λ에 비례하게 변형한다.

### 3.3 ODE 솔버에서의 적응적 스텝

핵심적인 계산적 결과: τ-시간에서 등간격 ODE 스텝은 t-시간에서 **비등간격 스텝**이 된다.

τ-시간에서 ODE 솔버가 Δτ = ε의 고정 스텝을 사용한다고 하자. 이때 t-시간에서의 스텝 크기는:

```
Δt ≈ ε / λ(t)
```

따라서:
- λ(t)가 클 때 (고변동성): Δt가 작다 → t-시간에서 더 조밀한 스텝 → **더 세밀한 attention**
- λ(t)가 작을 때 (저변동성): Δt가 크다 → t-시간에서 더 성긴 스텝 → **계산 절약**

이것은 **설계한 것이 아니라 수학적 필연**이다. 시간 축을 τ로 변환하는 순간, ODE 솔버의 등간격 스텝이 자동으로 정보 밀도에 비례하는 적응적 스텝이 된다.

실제로는 Dormand-Prince와 같은 적응적 ODE 솔버를 사용하므로, 스텝 크기는 수치적 오차와 동역학의 복잡성에 의해 추가로 조절된다. 그러나 **기저 시간 축의 변환**이 1차적인 적응을 제공하고, 솔버의 적응은 2차적 보정을 수행한다.

### 3.4 연속 Attention의 τ-시간 정식화

ContiFormer의 연속 attention을 τ-시간으로 재정의한다:

**원래 ContiFormer:**
```
A(q, t) = ∫₀ᵀ α(q, K(s)) · V(s) ds
```

**EOA:**
```
A(q, τ) = ∫₀^{τ(T)} α(q, K(σ)) · V(σ) dσ
```

여기서 K(σ)와 V(σ)는 τ-시간에서의 ODE:
```
dK/dσ = g_K(K(σ), σ)
dV/dσ = g_V(V(σ), σ)
```

를 따라 진화한다.

**등가 관계:** 변수 변환 σ = τ(s)를 적용하면:

```
A(q, τ) = ∫₀ᵀ α(q, K(τ(s))) · V(τ(s)) · λ(s) ds
```

이것은 원래의 attention 적분에 **강도 가중치 λ(s)**가 곱해진 형태이다. 즉, τ-시간 ODE attention은 t-시간에서의 강도 가중 attention과 수학적으로 동치이다.

### 3.5 전체 아키텍처

EOA를 기존 Transformer 백본에 통합하는 전체 구조:

```
입력: x = {x₁, x₂, ..., x_N}  (시계열 관측)
      m = {m₁, m₂, ..., m_N}  (시장 활동 신호)

1. 초기 인코딩:
   H₀ = Encoder(x)              # 표준 Transformer 인코더 또는 MLP

2. 경제적 시간 구성:
   λ_k = IntensityNet(m_k)       # 양의 강도 함수
   τ_k = Σ_{j<k} λ_j · Δt       # 누적 적분

3. Economic ODE Attention:
   dH/dτ = f_θ(H(τ), τ)         # τ-시간 ODE
   H(τ_final) = ODESolve(f_θ, H₀, τ₀, τ_final)

4. 출력:
   ŷ = OutputHead(H(τ_final))    # 예측
```

#### 3.5.1 IntensityNet

강도 함수를 구성하는 네트워크:

```python
class IntensityNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, market_features):
        # market_features: [batch, seq_len, input_dim]
        # 출력: λ > 0, [batch, seq_len]
        return F.softplus(self.net(market_features)).squeeze(-1)
```

#### 3.5.2 Economic ODE Layer

ContiFormer의 ODE layer를 τ-시간으로 수정:

```python
class EconomicODEAttention(nn.Module):
    def __init__(self, d_model, intensity_dim):
        self.intensity_net = IntensityNet(intensity_dim)
        self.ode_func = ODEFunc(d_model)  # ContiFormer의 ODE 함수
        self.solver = 'dopri5'  # Dormand-Prince

    def forward(self, H0, market_seq, timestamps):
        # 1. 경제적 시간 구성
        lam = self.intensity_net(market_seq)         # [B, N]
        dt = timestamps[:, 1:] - timestamps[:, :-1]  # [B, N-1]
        tau = torch.cumsum(
            F.pad(lam[:, :-1] * dt, (1, 0)), dim=1   # [B, N]
        )

        # 2. τ-시간에서 ODE 적분
        # tau의 각 시점을 ODE의 evaluation point로 사용
        H_final = odeint(
            self.ode_func, H0, tau,
            method=self.solver
        )

        return H_final
```

### 3.6 학습

#### 3.6.1 손실 함수

기본 예측 손실에 두 가지 정규화를 추가한다:

```
L = L_pred + β₁ · L_alignment + β₂ · L_smoothness
```

**L_pred:** 표준 예측 손실 (MSE 또는 수익률 순위 손실).

**L_alignment:** 강도 함수 λ가 관측된 시장 활동과 정렬되도록 유도:
```
L_alignment = -Corr(λ_θ(t), σ_realized(t))
```

**L_smoothness:** λ의 급격한 변화를 억제:
```
L_smoothness = (1/N) Σ_k (λ(t_{k+1}) - λ(t_k))²
```

#### 3.6.2 그래디언트 계산

Neural ODE의 그래디언트는 adjoint method(Chen et al., 2018)를 통해 메모리 효율적으로 계산된다. EOA에서 τ가 입력(시장 활동 신호)에 의존하므로, τ에 대한 그래디언트도 adjoint를 통해 역전파된다:

```
dL/dλ = dL/dH · dH/dτ · dτ/dλ
```

여기서 dτ/dλ = Δt (적분 구조의 야코비안)이므로, 그래디언트가 자연스럽게 정의된다.

---

## 4. 이론적 분석

### 4.1 시간 변환 정리와 EOA의 동치 관계

#### 4.1.1 정리 1: 균일-비균일 동치

**정리 1.** EOA의 τ-시간 ODE:
```
dH/dτ = f(H, τ),    τ(t) = ∫₀ᵗ λ(s) ds
```

는 다음 t-시간 ODE와 수학적으로 동치이다:
```
dH/dt = λ(t) · f(H, τ(t))
```

**증명.** 연쇄 법칙(chain rule)에 의해:
```
dH/dt = (dH/dτ) · (dτ/dt) = f(H, τ) · λ(t)
```

τ = τ(t)를 대입하면 결론. □

**해석:** τ-시간에서의 "균일한" ODE 동역학은 t-시간에서 **강도 λ(t)로 가중된** 동역학과 동치이다. 이것은 Zagatti et al.(2024)의 시간 변환 정리를 ODE attention에 직접 적용한 결과이다.

#### 4.1.2 정리 2: 적응적 해상도의 자동 발생

**정리 2.** τ-시간에서 등간격 Δτ로 ODE를 이산화하면, t-시간에서의 스텝 크기 Δt_k는:

```
Δt_k ≈ Δτ / λ(t_k)
```

이며, NFE(Number of Function Evaluations)는 구간 [t_a, t_b]에서:

```
NFE([t_a, t_b]) ≈ (1/Δτ) ∫_{t_a}^{t_b} λ(s) ds
```

이다.

**증명.** τ(t_k + Δt_k) = τ(t_k) + Δτ. τ의 정의에서:
```
τ(t_k + Δt_k) - τ(t_k) = ∫_{t_k}^{t_k + Δt_k} λ(s) ds ≈ λ(t_k) · Δt_k
```

따라서 Δt_k ≈ Δτ / λ(t_k).

NFE는 전체 τ-구간을 Δτ로 나눈 것이므로:
```
NFE = (τ(t_b) - τ(t_a)) / Δτ = (1/Δτ) ∫_{t_a}^{t_b} λ(s) ds
```
□

**해석:** 고변동성 구간(λ 큼)에서 NFE가 자동으로 증가한다. 이것은 "중요한 구간에 더 많은 계산을 투입하라"는 직관을 수학적으로 보장한다.

#### 4.1.3 따름정리: 계산 효율

**따름정리.** λ가 비균일한 경우, EOA의 총 NFE는 λ의 분산에 의존한다:

| 시나리오 | λ 분포 | 총 NFE |
|---------|--------|--------|
| 균일 강도 (달력 시간 ≡ τ) | λ = c (상수) | N_base |
| 비균일 강도 (경제적 시간) | λ ∈ [λ_min, λ_max] | N_base · (1 + Var(λ)/E[λ]²에 의한 재분배) |

정확한 NFE는 동일하지만(같은 Δτ를 사용할 경우), **NFE의 분배**가 달라진다: 고강도 구간에 집중되고 저강도 구간에서 절약된다. 만약 저강도 구간에서의 동역학이 단순하다면(금융 시계열에서 일반적), 전체 수치적 오차가 감소하여 같은 정확도를 더 적은 총 NFE로 달성할 수 있다.

### 4.2 Paper 2의 세 가지 병목 회피 분석

Paper 2(Kim et al., 2026)에서 식별된 세 가지 병목이 EOA에서 어떻게 회피되는지를 분석한다.

#### 4.2.1 병목 1: Softmax 압축 — 해당 없음

**Paper 2의 병목:** QK dot product 공간에서의 순서 변화(qk_ord_rate=0.992)가 softmax를 통과하면서 소실된다. Softmax는 매핑 x → exp(x)/Σexp(x)이며, 이 매핑은 입력 차이를 지수적으로 증폭하면서도 정규화에 의해 평탄화한다.

**EOA에서의 상태:** EOA는 **softmax를 사용하지 않는다.** ContiFormer의 연속 attention은 ODE 적분으로 정의되며, attention 가중치 α는 softmax가 아닌 연속 함수이다.

구체적으로, ContiFormer에서 α(q, K(s))의 여러 가능한 형태:

```
α₁(q, K(s)) = exp(-‖q - K(s)‖² / 2σ²)           # Gaussian kernel
α₂(q, K(s)) = (1 + ‖q - K(s)‖²)^{-1}             # Cauchy kernel
α₃(q, K(s)) = σ(q^T K(s) / √d)                    # Sigmoid (soft)
```

이 중 어떤 것도 이산 softmax의 "winner-take-all" 정규화를 수행하지 않으므로, QK 공간의 미세한 순서 변화가 attention 출력에 직접 반영된다.

**결론:** Softmax 압축 병목은 연속 ODE attention에서 구조적으로 존재하지 않는다.

#### 4.2.2 병목 2: 상호작용 접근 — ODE 상태로 해결

**Paper 2의 병목:** tau-RoPE에서 시간 좌표 경로(intensity → τ → RoPE rotation)는 피처 상호작용(예: intensity × index_return)에 구조적으로 접근할 수 없다. RoPE 회전은 각도만 변경하며, 회전 각도는 피처 공간의 크기(magnitude)에 영향을 주지 않는다.

**EOA에서의 상태:** ODE 상태 H(τ) ∈ ℝ^{d_model}는 **전체 표현**을 포함한다. ODE 함수 f_θ(H, τ)는 H의 모든 차원에 접근 가능하며, τ는 H의 동역학 속도를 조절한다.

```
dH/dτ = f_θ(H(τ), τ)
```

f_θ가 MLP 또는 attention 기반일 때, H의 서로 다른 차원 간의 곱셈적 상호작용을 자연스럽게 학습할 수 있다. τ는 이 상호작용이 **얼마나 빠르게 진화하는지**를 조절한다.

**핵심 차이:** tau-RoPE는 "좌표 공간에서만" 시간 정보를 변경하고 "피처 공간"에는 영향을 주지 못하지만, EOA는 "시간 축 자체를 변환"하여 피처 공간의 **전체 동역학**에 영향을 준다.

**결론:** ODE 상태 H가 전체 표현을 포함하므로, 시간 변환이 피처 상호작용의 동역학에 직접 영향을 미친다. 상호작용 접근 병목은 해소된다.

#### 4.2.3 병목 3: 단조성 제약 — 자동 충족

**Paper 2의 병목:** cumsum 구조가 τ를 물리적 시간과 거의 동일하게 만든다(τ_corr > 0.998). τ = cumsum(softplus(x))에서 softplus의 출력이 1 근처에 집중되면, τ ≈ t가 된다.

**EOA에서의 상태:** τ(t) = ∫₀ᵗ λ(s) ds에서:

1. **단조 증가는 자동 보장된다.** λ(s) > 0 (softplus로 보장)이므로 τ는 엄밀히 단조 증가. 별도의 제약이 불필요하다.

2. **그러나 τ가 t에 가까워지는 문제는 여전히 존재할 수 있다.** 만약 λ(s) ≈ 상수이면, τ ≈ c·t가 되어 달력 시간과 동등하다.

3. **핵심 차이: 영향의 범위.** Paper 2에서 τ ≈ t가 문제인 이유는 RoPE에서 상대 위치 (τ(m)-τ(n))의 차이가 (m-n)과 거의 같아져서 실질적으로 영(null)이기 때문이다. **그러나 EOA에서는 τ ≈ t여도 ODE의 시간 축이 바뀌는 것이므로, f_θ(H, τ)에서 τ 인수가 다르게 해석된다.** ODE 함수는 τ를 직접 입력으로 받으며, τ의 스케일과 분포가 f_θ의 동작을 변경한다.

4. **더 중요한 점: ODE 솔버의 적응적 스텝.** τ ≈ c·t여도, τ의 국소적 곡률이 다르면 ODE 솔버의 스텝 크기가 달라진다. λ(s)가 비균일하면 τ(t)의 미분 dτ/dt = λ(t)가 변하므로, 등간격 τ-스텝이 비등간격 t-스텝이 된다. 이것은 Paper 2의 RoPE 맥락에서는 불가능한 효과이다.

**결론:** 단조성은 자동 보장되며, 더 중요하게는 τ가 ODE의 시간 축으로 사용될 때 RoPE의 상대 위치 인코딩과는 근본적으로 다른 메커니즘으로 작동한다.

### 4.3 수학적 정당성: Clark의 Subordinated Process로서의 EOA

**명제 3.** EOA의 attention 동역학:

```
dH/dτ = f_θ(H(τ), τ),    τ(t) = ∫₀ᵗ λ(s) ds
```

은 Clark(1973)의 subordinated process X(t) = W(T(t))의 구조적 아날로그이다:

| Clark (1973) | EOA |
|-------------|-----|
| W(·): 표준 위너 과정 | H(·): ODE 해 (τ-시간에서 "균일한" 동역학) |
| T(t): subordinator (경제적 시간) | τ(t): 강도 적분 (경제적 시간) |
| X(t) = W(T(t)): 관측 과정 | H(τ(t)): t-시간에서의 attention 상태 |

**해석:** Clark에서 W가 τ-시간에서 정규분포를 따르듯이, EOA에서 H는 τ-시간에서 "정규적인"(수치적으로 안정적인) 동역학을 갖는다. 이것이 t-시간으로 "관측"될 때, λ의 비균일성에 의해 비정상적(non-stationary) 동역학이 나타난다. 이것은 정확히 금융 시계열의 관측된 특성이다.

### 4.4 표현력 분석

#### 4.4.1 ContiFormer ⊂ EOA

**명제 4.** ContiFormer(달력 시간 ODE)는 EOA의 특수한 경우이다: λ(t) = 1 (상수).

**증명.** λ(t) = 1이면 τ(t) = t이므로 dH/dτ = dH/dt. □

따라서 EOA의 표현력은 ContiFormer 이상이다.

#### 4.4.2 EOA와 Neural CDE의 관계

Neural CDE(Kidger et al., 2020)는:
```
dH/dt = f_θ(H(t)) · dX/dt
```

여기서 dX/dt는 데이터 경로의 미분이다. EOA와 비교:
```
dH/dτ = f_θ(H(τ), τ)    ↔    dH/dt = λ(t) · f_θ(H(τ(t)), τ(t))
```

Neural CDE에서 dX/dt가 f_θ에 곱해지듯이, EOA에서 λ(t)가 f_θ에 곱해진다. 차이점:
- Neural CDE: 데이터 경로 전체가 동역학을 제어(벡터 값)
- EOA: 강도 함수가 시간 축만 제어(스칼라 값)

EOA는 Neural CDE의 특수한 경우로 볼 수 있다: dX/dt를 스칼라 강도 λ(t)로 제한한 것. 이 제한은 해석 가능성(시간 속도라는 명확한 의미)과 계산 효율(스칼라 곱)의 대가로 표현력을 일부 양보한다.

---

## 5. 구현 세부사항

### 5.1 ODE 솔버 선택

| 솔버 | 특성 | EOA 적합성 |
|------|------|-----------|
| Euler | 고정 스텝, 1차 | 빠르지만 부정확 |
| RK4 | 고정 스텝, 4차 | 안정적이나 비적응적 |
| **Dormand-Prince (dopri5)** | **적응 스텝, 4/5차** | **EOA와 시너지** |
| Adams | 적응, 다스텝 | 매끄러운 동역학에 적합 |

**추천:** Dormand-Prince(dopri5). 이유:
1. τ-시간에서의 적응적 스텝이 t-시간에서 이중으로 적응적이 됨 (λ에 의한 1차 적응 + 솔버에 의한 2차 적응)
2. torchdiffeq(Chen et al., 2018) 구현체에서 바로 사용 가능
3. 에러 제어가 자동화되어 수치적 안정성 보장

### 5.2 역전파: Adjoint Method

메모리 효율을 위해 adjoint method를 사용한다. τ-시간 ODE의 adjoint:

```
da/dτ = -a^T · ∂f/∂H
dH/dτ = f(H, τ)
```

여기서 a = dL/dH는 adjoint 상태. τ-시간에서의 adjoint는 t-시간에서의 adjoint와 동일한 메모리 복잡도 O(d_model)을 갖는다.

### 5.3 불규칙 관측 처리

금융 시계열에서 관측이 불규칙할 때(공휴일, 거래 중단 등), EOA는 자연스럽게 처리한다:

```
τ(t_k) = ∫₀^{t_k} λ(s) ds
```

관측이 없는 구간에서 λ(s)를 보간하면, τ의 연속성이 유지된다. ODE 솔버는 임의의 시점에서 evaluation할 수 있으므로, 관측 시점이 등간격일 필요가 없다.

---

## 6. 실험 설계

### 6.1 데이터셋

| 데이터셋 | 특성 | 목적 |
|---------|------|------|
| Ken French 25 Size-BM | 일별, 25 포트폴리오, FF3 잔차 | Paper 1/2/3과의 직접 비교 |
| S&P 500 개별 종목 | 일별, 500+ 종목, 거래량 포함 | 스케일 검증 |
| 합성 데이터 | 제어된 λ 프로파일 | 이론적 성질 검증 |

#### 6.1.1 합성 데이터 구성

이론적 예측을 정밀 검증하기 위해, 제어된 강도 함수를 가진 합성 시계열을 생성한다:

```python
def generate_synthetic(T=1000, regime_switch=True):
    if regime_switch:
        # 고변동성/저변동성 체제 전환
        regime = np.zeros(T)
        regime[200:400] = 1  # 고변동성
        regime[600:800] = 1  # 고변동성
        lam = np.where(regime, 5.0, 1.0)  # 고변동성에서 5배 빠른 경제적 시간
    else:
        lam = np.ones(T)  # 균일 (기준선)

    tau = np.cumsum(lam)
    # τ-시간에서 정규 AR(1) 과정 생성
    y_tau = generate_ar1(len(tau), phi=0.8, sigma=0.1)
    # t-시간으로 역변환하여 관측 시계열 생성
    y_t = np.interp(np.arange(T), tau / tau[-1] * T, y_tau[:T])
    return y_t, lam
```

**예측:** 합성 데이터에서 EOA는 ContiFormer를 유의하게 outperform해야 한다 (체제 전환이 있을 때). λ = 상수일 때는 두 모델이 동등해야 한다.

### 6.2 기준선 모델

| 모델 | 시간 처리 | Attention 유형 |
|------|----------|--------------|
| Transformer + RoPE | 달력 시간, 이산 | 표준 softmax |
| Transformer + tau-RoPE | 경제적 시간, 이산 | 표준 softmax |
| ContiFormer | 달력 시간, 연속 | ODE attention |
| Neural CDE | 데이터 제어, 연속 | 없음 (ODE만) |
| **EOA (본 연구)** | **경제적 시간, 연속** | **ODE attention** |

### 6.3 평가 지표

#### 6.3.1 예측 성능

- **IC (Information Coefficient):** 예측과 실현의 Spearman 상관 (교차 단면)
- **MAE, MSE:** 절대 예측 정확도
- **체제별 IC:** 고변동성/저변동성 구간별 분리 평가

#### 6.3.2 계산 효율

- **NFE (Number of Function Evaluations):** 전체 및 구간별
- **NFE 분배 비율:** 고변동성 구간의 NFE / 전체 NFE
- **벽시계 시간(wall-clock time):** 학습 및 추론

#### 6.3.3 이론적 성질 검증

- **τ-시간 정상성:** τ-시간에서의 attention 상태가 t-시간에서보다 더 정상적(stationary)인지 검정 (ADF test)
- **강도 정렬:** 학습된 λ_θ와 실현 변동성의 상관
- **스텝 분배:** ODE 솔버 스텝의 실제 분배가 이론적 예측(정리 2)과 일치하는지

### 6.4 가설

**H1 (핵심 성능):** 체제 전환이 있는 데이터에서 EOA > ContiFormer (IC 기준).
- 근거: τ-시간 변환이 체제 전환을 자동으로 처리

**H2 (병목 회피):** EOA에서 Paper 2의 세 병목이 관측되지 않음.
- H2a: Attention 가중치에 softmax 압축 없음 (엔트로피 측정)
- H2b: ODE 상태가 피처 상호작용을 포착함 (프로빙 실험)
- H2c: τ의 다양성이 tau-RoPE보다 큼 (τ_corr < 0.99)

**H3 (계산 효율):** EOA의 고변동성 구간 NFE > 저변동성 구간 NFE.
- 근거: 정리 2의 직접적 검증

**H4 (정상성):** τ-시간에서의 ODE 상태가 t-시간에서보다 더 정상적.
- 근거: 시간 변환 정리의 핵심 예측

**H5 (퇴화):** λ = 상수일 때 EOA ≡ ContiFormer.
- 근거: 명제 4의 직접적 검증

### 6.5 통계적 검정

- 모든 비교: Newey-West 표준 오차 (시계열 자기상관 보정)
- 다중 비교: Bonferroni 보정 (5개 가설)
- 시드: 최소 5개 (7, 17, 27, 37, 47)
- Embargo: 훈련-검증 간 최소 (seq_len + horizon) 일의 간격

---

## 7. 기대 결과

### 7.1 합성 데이터 실험

| 조건 | ContiFormer MSE | EOA MSE | 예상 Δ |
|------|----------------|---------|-------|
| λ = 상수 (균일) | 0.050 | 0.050 | ≈ 0 (동등) |
| 체제 전환 (λ ∈ {1, 5}) | 0.045 | 0.030 | -33% |
| 연속적 λ 변화 | 0.042 | 0.032 | -24% |

**근거:** 합성 데이터는 τ-시간에서 정상적으로 생성되므로, τ를 정확히 복원하는 EOA가 구조적 우위를 가진다. λ = 상수에서 두 모델이 동등한 것은 명제 4의 직접적 확인이다.

### 7.2 금융 데이터 실험

#### 7.2.1 전체 구간 성능

| 모델 | IC (전체) | MAE (전체) |
|------|----------|-----------|
| Transformer + RoPE | 0.042 | 0.0185 |
| Transformer + tau-RoPE | 0.035 | 0.0180 |
| ContiFormer | 0.048 | 0.0175 |
| Neural CDE | 0.046 | 0.0178 |
| **EOA** | **0.052** | **0.0168** |

**예상:** EOA가 전체 구간에서 소폭 개선. 연속 ODE attention의 기존 장점 + 경제적 시간의 추가 정보.

#### 7.2.2 체제별 성능

| 모델 | IC (고변동성) | IC (저변동성) | Δ (고-저) |
|------|------------|------------|----------|
| ContiFormer | 0.035 | 0.052 | -0.017 |
| **EOA** | **0.060** | **0.048** | **+0.012** |

**예상:** EOA의 핵심 강점은 고변동성 체제에서의 개선이다. ContiFormer는 고변동성에서 성능이 하락하지만, EOA는 τ-시간 변환에 의해 고변동성 구간을 더 세밀하게 처리하여 성능이 향상된다.

이것은 Paper 3(paper_B_economic_time_applied.md)에서 tau-RoPE가 고변동성 MAE만 개선한 결과와 일관적이지만, EOA에서는 **IC도 함께 개선**되어야 한다 (softmax 압축 병목이 없으므로).

### 7.3 NFE 분배

| 구간 | λ̄ (평균 강도) | ContiFormer NFE | EOA NFE |
|------|-------------|----------------|---------|
| 고변동성 (상위 25%) | 3.2 | 25% (균일) | 42% |
| 저변동성 (하위 25%) | 0.8 | 25% (균일) | 13% |

**예상:** EOA의 NFE가 자연스럽게 고변동성 구간에 집중된다. 총 NFE는 비슷하지만 분배가 달라진다.

### 7.4 τ-시간 정상성

**예상:** ODE 상태 H(τ)의 시계열이 H(t)의 시계열보다 더 낮은 ADF 통계량(더 정상적)을 보인다.

이것은 시간 변환 정리의 핵심 예측이다: τ-시간에서의 과정은 더 "균일"하므로, ODE 동역학이 더 안정적이어야 한다.

---

## 8. 관련 연구와의 차별화

### 8.1 ContiFormer와의 차이

| 측면 | ContiFormer | EOA |
|------|-----------|-----|
| 시간 변수 | 달력 시간 t | 경제적 시간 τ(t) |
| ODE 해상도 | 수치적 오차 기반 | 정보 밀도 + 수치적 오차 |
| 이론적 동기 | 연속 attention 일반화 | Clark(1973) subordination |
| 금융 특화 | 없음 | 있음 (강도 함수 = 시장 활동) |

### 8.2 tau-RoPE와의 차이

| 측면 | tau-RoPE (Paper 2/3) | EOA |
|------|---------------------|-----|
| 경제적 시간의 역할 | PE의 위치 인수 | ODE의 시간 축 |
| Attention 유형 | 이산 softmax | 연속 ODE |
| Softmax 병목 | 있음 (실패 원인) | 없음 |
| 상호작용 접근 | RoPE 회전만 | ODE 상태 전체 |
| 이론적 정당성 | 경험적 | 시간 변환 정리 |

### 8.3 Neural CDE와의 차이

| 측면 | Neural CDE | EOA |
|------|-----------|-----|
| 제어 신호 | 데이터 경로 dX/dt (벡터) | 강도 λ(t) (스칼라) |
| 해석 | 데이터가 동역학을 제어 | 정보 밀도가 시간 속도를 제어 |
| Attention | 없음 | 있음 (연속) |
| 금융 이론 연결 | 없음 | Clark(1973) |

### 8.4 StretchTime/SyPE와의 차이

| 측면 | StretchTime | EOA |
|------|-----------|-----|
| RoPE 한계 극복 | SO(2) → Sp(2,R) 확장 | ODE attention (RoPE 불필요) |
| 시간 변수 | 달력 시간 (워프 모듈은 PE 내부) | 경제적 시간 (동역학 자체) |
| 이론적 근거 | Symplectic geometry | Subordinated process |
| 적응 대상 | PE의 주파수/크기 | ODE의 시간 축 |

### 8.5 Zagatti et al. (2024)와의 차이

| 측면 | Zagatti et al. | EOA |
|------|---------------|-----|
| 대상 | 시간적 점과정 (TPP) | Transformer attention |
| 시간 변환 목적 | 모델 진단 (GoF test) | Attention 해상도 적응 |
| 아키텍처 | 점과정 신경망 | ContiFormer + τ |
| 금융 적용 | 주문 도착 모델링 | 수익률 예측 |

---

## 9. 논의

### 9.1 네 편의 논문 연결

본 연구는 네 편의 논문 시리즈의 네 번째로, 경제적 시간을 Transformer에 구현하는 과정에서의 발견과 해결을 문서화한다.

**Paper 1 (컨디셔닝 인터페이스):** 시장 상태를 모델에 주입하는 다섯 가지 인터페이스를 비교하고, SNR이 낮을 때 단순한 concat이 곱셈적 인터페이스를 이기는 메커니즘을 밝혔다. **EOA와의 연결:** EOA에서 λ(t)는 ODE의 시간 축에 곱해진다(dH/dt = λ(t)·f). 이것은 곱셈적 컨디셔닝의 일종이지만, 피처 공간(FiLM)이 아닌 시간 축에 적용되므로 Paper 1의 SNR 문제를 우회한다. λ가 잡음을 포함해도, 그것은 "시간의 속도"를 조절할 뿐 표현 자체를 왜곡하지 않는다.

**Paper 2 (표현-유용성 간극):** tau-RoPE가 attention geometry를 변형하지만 예측을 개선하지 못하는 세 가지 병목을 식별했다. **EOA와의 연결:** EOA는 이 세 병목을 모두 구조적으로 회피한다(4.2절). Paper 2의 부정적 결과는 "경제적 시간 자체의 실패"가 아니라 "이산 attention + RoPE라는 구현의 한계"임을 EOA가 보여준다.

**Paper 3 (경제적 시간의 조건부 우위):** tau-RoPE가 고변동성 체제에서만 MAE를 개선하는 부분적 성공을 보였다. **EOA와의 연결:** Paper 3의 부분적 성공은 경제적 시간의 이론적 타당성을 지지하며, EOA는 이 부분적 성공을 완전한 성공으로 확장할 구조적 조건(softmax 없음, 피처 접근 가능)을 갖춘다.

**흐름:**
```
Paper 1: 어떤 인터페이스가 좋은가? → concat이 이김 (SNR 문제)
Paper 2: 왜 tau-RoPE가 실패하는가? → 세 가지 병목
Paper 3: 어디서 부분적으로 성공하는가? → 고변동성 MAE
Paper 4: 세 병목을 회피하는 구조는? → 연속 ODE attention + 경제적 시간
```

### 9.2 한계

#### 9.2.1 계산 비용

Neural ODE는 이산 Transformer보다 계산 비용이 높다. adjoint method의 메모리 효율에도 불구하고, ODE 솔버의 반복적 함수 평가는 단일 forward pass보다 느리다. 금융 실무에서 추론 지연 시간(latency)이 중요한 경우, 이 비용은 장벽이 될 수 있다.

**완화:** (1) ODE 솔버의 허용 오차(atol, rtol)를 조절하여 정확도-속도 트레이드오프를 제어. (2) 저변동성 구간에서 자동으로 절약되는 NFE가 부분적으로 보상. (3) distillation을 통해 학습된 EOA의 동역학을 이산 모델로 전이.

#### 9.2.2 강도 함수의 선택

λ(s)의 올바른 구성은 도메인 지식에 의존한다. 잘못된 λ(예: 정보 밀도와 무관한 신호)를 사용하면, 시간 변환이 오히려 성능을 저하시킬 수 있다. 학습된 λ_θ는 이 문제를 완화하지만, 학습 자체의 어려움(Paper 2에서 관찰된 정렬 문제)이 남아 있다.

#### 9.2.3 일반화 범위

본 연구의 이론적 분석은 일반적이지만, 실험은 금융 시계열에 한정된다. 의료 시계열(예: ICU 모니터링), 지구물리학적 시계열(예: 지진 데이터) 등 다른 도메인에서의 검증이 필요하다. 이 도메인들도 비균일 정보 밀도를 가지므로, EOA가 적용 가능할 것으로 예상하지만 실증이 필요하다.

#### 9.2.4 ContiFormer 기반 구현의 현실적 문제

ContiFormer 자체가 아직 초기 단계의 아키텍처이며, 대규모 실험에서의 안정성과 하이퍼파라미터 민감성이 충분히 검증되지 않았다. EOA는 ContiFormer 위에 구축되므로, ContiFormer의 미성숙함이 직접적 위험 요인이다.

### 9.3 미래 방향

#### 9.3.1 다중 스케일 경제적 시간

금융 시장은 다중 스케일의 정보 밀도를 갖는다: 일중(초 단위), 일별, 주별. 현재의 단일 λ 대신, 다중 스케일 강도:

```
τ_multi(t) = ∫₀ᵗ [α₁λ₁(s) + α₂λ₂(s) + α₃λ₃(s)] ds
```

를 구성하면, 서로 다른 시간 스케일의 정보를 동시에 포착할 수 있다.

#### 9.3.2 확률적 경제적 시간

Clark(1973)의 원래 formulation에서 T(t)는 **확률 과정**이다. 현재 EOA에서는 λ(t)를 결정적(deterministic) 함수로 취급하지만, λ 자체를 확률 과정으로 모형화하면 불확실성 정량화가 가능하다:

```
dτ = λ(t)dt + σ_λ dW_λ
```

이것은 ODE를 SDE(확률 미분 방정식)로 확장하며, Tzen & Raginsky(2019)의 Neural SDE 프레임워크와 결합할 수 있다.

#### 9.3.3 경제적 시간의 비교 학습

서로 다른 자산의 경제적 시간을 비교하는 것도 흥미로운 방향이다. 동일한 달력 구간에서 자산 A와 B의 τ_A(t)와 τ_B(t)가 다르다면, 이 차이 자체가 상대 가치 정보를 포함할 수 있다.

---

## 10. 결론

본 연구는 ContiFormer의 연속 시간 attention에서 달력 시간을 경제적 시간으로 대체하는 Economic ODE Attention(EOA)을 제안했다. 이론적 분석에서 EOA가 시간 변환 정리에 의해 수학적으로 정당화되며, Paper 2에서 식별된 이산 attention의 세 가지 병목(softmax 압축, 상호작용 접근, 단조성 제약)을 구조적으로 회피함을 보였다.

EOA의 핵심 기여는 아키텍처의 참신성이 아니라, **50년 된 금융 이론(Clark, 1973)과 최신 연속 시간 attention(ContiFormer, 2024)의 연결**이다. 이 연결은 Zagatti et al.(2024)의 시간 변환 정리를 매개로 이루어지며, "경제적 시간에서 균일한 과정은 달력 시간에서 강도 가중 과정과 동치"라는 정리가 attention 동역학에 직접 적용된다.

네 편의 논문 시리즈를 통해, 우리는 경제적 시간을 Transformer에 구현하는 과정에서의 실패(Paper 2), 부분적 성공(Paper 3), 실패의 메커니즘(Paper 1, 2), 그리고 구조적 해결(Paper 4)을 문서화했다. 이 여정은 "이론적으로 올바른 아이디어 + 잘못된 구현 = 실패"에서 "올바른 아이디어 + 올바른 구현 = 성공"으로의 전환을 보여주며, 이 전환의 핵심은 이산 attention의 구조적 한계를 인식하고 연속 동역학으로 전환한 것이다.

---

## 참고문헌

### 경제적 시간 및 Subordinated Process

1. Clark, P. K. (1973). A subordinated stochastic process model with finite variance for speculative prices. *Econometrica*, 41(1), 135-155.

2. Mandelbrot, B. B., & Taylor, H. M. (1967). On the distribution of stock price differences. *Operations Research*, 15(6), 1057-1062.

3. Ane, T., & Geman, H. (2000). Order flow, transaction clock, and normality of asset returns. *Journal of Finance*, 55(5), 2259-2284.

4. Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

5. Carr, P., & Wu, L. (2004). Time-changed Levy processes and option pricing. *Journal of Financial Economics*, 71(1), 113-141.

### 시간 변환 정리 및 점과정

6. Zagatti, G. A., Ng, G. T., Cai, J., & Ooi, W. T. (2024). Time-change theorem for neural temporal point processes. *AISTATS 2024*.

7. Daley, D. J., & Vere-Jones, D. (2003). *An Introduction to the Theory of Point Processes*. Springer.

8. Dambis, K. E. (1965). On the decomposition of continuous submartingales. *Theory of Probability & Its Applications*, 10(3), 401-410.

9. Dubins, L. E., & Schwarz, G. (1965). On continuous martingales. *Proceedings of the National Academy of Sciences*, 53(5), 913-916.

10. Mei, H., & Eisner, J. (2017). The neural Hawkes process: A neurally self-modulating multivariate point process. *NeurIPS 2017*.

### Transformer 및 Attention

11. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. *NeurIPS 2017*.

12. Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced transformer with rotary position embedding. *arXiv:2104.09864*.

13. Yang, Z., Dai, Z., Yang, Y., Carbonell, J., Salakhutdinov, R., & Le, Q. V. (2018). XLNet: Generalized autoregressive pretraining for language understanding. *NeurIPS 2019*.

14. Han, X., Gao, J., & He, X. (2024). On the injectivity of softmax attention. *arXiv preprint*.

15. Kim, J., et al. (2026). StretchTime: Adaptive time series forecasting via symplectic attention. *arXiv:2602.08983*.

### ContiFormer 및 연속 시간 모델

16. Chen, Y., He, Y., Zhao, R., & Zhang, M. (2024). ContiFormer: Continuous-time Transformer for irregular time series modeling. *NeurIPS 2024*.

17. Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural ordinary differential equations. *NeurIPS 2018*.

18. Rubanova, Y., Chen, R. T. Q., & Duvenaud, D. (2019). Latent ordinary differential equations for irregularly-sampled time series. *NeurIPS 2019*.

19. Kidger, P., Morrill, J., Foster, J., & Lyons, T. (2020). Neural controlled differential equations for irregular time series. *NeurIPS 2020 Spotlight*.

20. De Brouwer, E., Simm, J., Arany, A., & Moreau, Y. (2019). GRU-ODE-Bayes: Continuous modeling of sporadically-observed time series. *NeurIPS 2019*.

### Neural SDE 및 확장

21. Tzen, B., & Raginsky, M. (2019). Neural stochastic differential equations: Deep latent Gaussian models in the diffusion limit. *arXiv:1905.09883*.

22. Li, X., Wong, T.-K. L., Chen, R. T. Q., & Duvenaud, D. (2020). Scalable gradients for stochastic differential equations. *AISTATS 2020*.

23. Jia, J., & Benson, A. R. (2019). Neural jump stochastic differential equations. *NeurIPS 2019*.

### 금융 머신러닝

24. Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223-2273.

25. Daniel, K., & Moskowitz, T. J. (2016). Momentum crashes. *Journal of Financial Economics*, 122(2), 221-247.

26. Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.

27. Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium: Empirical tests. *Journal of Political Economy*, 81(3), 607-636.

### 컨디셔닝 인터페이스

28. Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Bengio, Y. (2018). FiLM: Visual reasoning with a general conditioning layer. *AAAI 2018*.

29. Jayakumar, S. M., Czarnecki, W. M., Menick, J., et al. (2020). Multiplicative interactions and where to find them. *ICLR 2020*.

30. Peebles, W., & Xie, S. (2023). Scalable diffusion models with transformers. *ICCV 2023*.

### 표현-유용성 간극

31. Hewitt, J., & Liang, P. (2019). Designing and interpreting probes with control tasks. *EMNLP 2019*.

32. Locatello, F., Bauer, S., Lucic, M., et al. (2019). Challenging common assumptions in the unsupervised learning of disentangled representations. *ICML 2019 Best Paper*.

33. Jain, S., & Wallace, B. C. (2019). Attention is not explanation. *NAACL 2019*.

34. Pezeshki, M., Kaba, S.-O., Bengio, Y., Courville, A., Precup, D., & Lajoie, G. (2021). Gradient starvation: A learning proclivity in neural networks. *NeurIPS 2021*.

### Positional Encoding 적응

35. Zhang, G., et al. (2024). ElasTST: Towards robust varied-horizon forecasting with elastic time-series Transformer. *NeurIPS 2024*.

36. Zhang, Y., et al. (2024). T2B-PE: Position encoding diminishes through the network depth. *arXiv preprint*.

37. KAIROS/DRoPE (2025). Spectral-adaptive RoPE for time series. *arXiv preprint*.

### 경로 서명 및 시간 워핑

38. Kidger, P., & Lyons, T. (2021). Signatory: Differentiable computations of the signature and logsignature transforms, on both CPU and GPU. *ICLR 2021*.

39. Morrill, J., Kidger, P., Yang, L., & Lyons, T. (2021). Neural rough differential equations. *ICML 2021*.

### Softmax Bottleneck 및 Attention 분석

40. Yang, Z., Dai, Z., Salakhutdinov, R., & Cohen, W. W. (2018). Breaking the softmax bottleneck: A high-rank RNN language model. *ICLR 2018*.

41. Wiegreffe, S., & Pinter, Y. (2019). Attention is not not explanation. *EMNLP 2019*.

### 정보 이론

42. Tishby, N., Pereira, F. C., & Bialek, W. (2000). The information bottleneck method. *arXiv:physics/0004057*.

43. Liang, P. P., et al. (2023). Quantifying & modeling multimodal interactions: An information decomposition framework. *NeurIPS 2023*.

### 기타

44. Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708.

45. Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303-1313.

---

## 부록 A: ContiFormer에서 EOA로의 변환 레시피

기존 ContiFormer 구현을 EOA로 변환하는 최소 변경 사항:

### A.1 필요 변경

1. **시간 축 변환:** `timestamps` → `tau = cumulative_intensity(timestamps, market_features)`
2. **ODE 솔버 호출 변경:** `odeint(func, y0, timestamps)` → `odeint(func, y0, tau)`
3. **IntensityNet 추가:** 시장 활동 피처 → 양의 스칼라 강도

### A.2 불필요한 변경

1. ODE 함수 f_θ의 구조: **변경 불필요** (시간 인수만 τ로 바뀜)
2. Adjoint method: **변경 불필요** (τ-시간에서도 동일하게 작동)
3. 출력 헤드: **변경 불필요**

### A.3 의사 코드

```python
# 변경 전 (ContiFormer)
def forward(self, x, timestamps):
    h0 = self.encoder(x)
    h_final = odeint(self.ode_func, h0, timestamps, method='dopri5')
    return self.output_head(h_final[-1])

# 변경 후 (EOA)
def forward(self, x, timestamps, market_features):
    h0 = self.encoder(x)
    lam = self.intensity_net(market_features)  # [B, N], 양수
    dt = timestamps[:, 1:] - timestamps[:, :-1]
    tau = torch.cat([
        torch.zeros(x.size(0), 1, device=x.device),
        torch.cumsum(lam[:, :-1] * dt, dim=1)
    ], dim=1)
    tau = tau * (timestamps[:, -1:] / tau[:, -1:].clamp(min=1e-8))  # 정규화
    h_final = odeint(self.ode_func, h0, tau, method='dopri5')
    return self.output_head(h_final[-1])
```

총 변경: ~10줄 추가, 기존 코드 1줄 수정.

---

## 부록 B: 시간 변환 정리의 Attention 적용 증명

### B.1 전제

- H(τ): τ-시간에서의 ODE 해
- τ: [0, T] → [0, τ(T)], 엄밀 단조 증가
- λ(t) = dτ/dt > 0, ∀t

### B.2 정리

τ-시간에서의 연속 attention:
```
A_τ(q) = ∫₀^{τ(T)} α(q, K(σ)) · V(σ) dσ
```

는 t-시간에서의 강도 가중 attention:
```
A_t(q) = ∫₀^T α(q, K(τ(s))) · V(τ(s)) · λ(s) ds
```

과 수학적으로 동치이다.

### B.3 증명

변수 치환 σ = τ(s), dσ = λ(s)ds:

```
A_τ(q) = ∫₀^{τ(T)} α(q, K(σ)) · V(σ) dσ
        = ∫₀^T α(q, K(τ(s))) · V(τ(s)) · λ(s) ds
        = A_t(q)
```

□

### B.4 해석

이 동치 관계는 EOA의 물리적 의미를 명확히 한다: τ-시간에서 "균등하게" attention을 수행하는 것은 t-시간에서 "강도에 비례하여" attention을 수행하는 것과 정확히 같다. 즉, EOA는 **시장이 활발한 구간에 자동으로 더 많은 attention을 부여**한다.

---

## 부록 C: Paper 2 병목에 대한 형식적 회피 증명

### C.1 병목 1: Softmax 압축

**Paper 2의 주장:** ∀q, ∃ε > 0 s.t. |softmax(qk₁/√d) - softmax(qk₂/√d)| < ε 이어도 |qk₁ - qk₂| >> ε.

**EOA에서:** softmax가 없으므로 이 부등식 자체가 적용되지 않는다. 연속 attention 가중치 α(q, K(σ))는 Lipschitz 연속이면 충분하다:

```
|α(q, K₁) - α(q, K₂)| ≤ L · ‖K₁ - K₂‖
```

이것은 K 공간의 변화가 attention에 선형적으로 반영됨을 보장한다. □

### C.2 병목 2: 상호작용 접근

**Paper 2의 주장:** RoPE 경로 (λ → τ → rotation angle)는 피처 공간의 곱셈적 상호작용에 접근 불가.

**EOA에서:** ODE 함수 f_θ(H, τ)에서 H의 차원 간 상호작용은 f_θ의 구조(MLP, attention 등)에 의해 자유롭게 모형화된다. τ는 이 상호작용의 **진화 속도**를 조절한다:

```
dH_i/dτ = Σ_j W_{ij} · σ(H_j) + ...  (MLP 예시)
```

모든 차원 i, j 간의 상호작용이 W에 의해 매개되며, τ는 이 상호작용이 "얼마나 많이" 축적되는지를 결정한다. □

### C.3 병목 3: 단조성 제약

**Paper 2의 주장:** cumsum(softplus(x))에서 softplus 출력이 1 근처로 수렴하면 τ ≈ t.

**EOA에서:** (1) 동일한 cumsum 구조를 사용하지만, 결과가 RoPE의 위치 인수가 아니라 ODE의 시간 축이므로, τ ≈ c·t여도 ODE 솔버의 스텝 분배가 λ의 국소적 변화에 의해 달라진다. (2) 더 중요하게는, λ의 학습 목표가 다르다: tau-RoPE에서는 RoPE 회전 각도의 미세한 차이가 softmax를 통과해야 하므로 λ의 편차가 큰 영향을 미치지 못하지만, EOA에서는 λ가 ODE 동역학의 속도를 직접 조절하므로, λ의 편차가 H의 진화에 1차적 영향을 미친다. □

---

*본 논문은 Paper 1 (컨디셔닝 인터페이스), Paper 2 (표현-유용성 간극), Paper 3 (경제적 시간의 조건부 우위)에 이은 시리즈의 네 번째 논문이다.*
