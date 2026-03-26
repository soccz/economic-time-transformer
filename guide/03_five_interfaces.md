# Chapter 03: 다섯 가지 컨디셔닝 인터페이스

> *"같은 정보를 모델에 주더라도, 어디에 어떻게 넣느냐에 따라
> 모델이 완전히 다른 것을 배운다."*

---

## 이 장의 질문

> **시장 상태(position, intensity, indexret)를 모델에 넣는 최선의 방법은 무엇인가?**

"시장 상태"는 세 채널이다:
- **position**: S&P 500의 200일 이동평균 대비 위치 (추세)
- **intensity**: 30일 실현변동성의 252일 분위수 (활동 강도)
- **indexret**: 시장 지수 수익률 (시장 방향)

이 세 채널을 Transformer에 넣는 방법을 다섯 가지 시도했다.

---

## 인터페이스 1: concat_a — 그냥 붙이기

자산 피처 $x_t$에 시장 상태 $c_t$를 concatenate해서 같이 투사한다.

$$h_t = W \cdot [x_t ; c_t] + b$$

지도의 여백에 기온을 텍스트로 적어두는 것과 같다. 모델이 알아서 활용하길 기대한다. 시장 상태가 입력 공간에서 자산 피처와 즉시 섞이고, 추가 파라미터는 +64개뿐이다 (전체 45,266개).

이것이 기준선이다. 뒤에 나올 모든 방법은 이것을 이겨야 한다.

---

## 인터페이스 2: econ_time — PE에 시장 상태 더하기

시장 상태를 입력이 아니라 positional encoding에 직접 더한다.

$$\text{PE}'_t = \text{PE}_t + g \cdot W_c \cdot c_t$$

$g$는 주입 강도 스칼라, $W_c$는 투사 행렬이다. 시간의 흐름을 바꾸는 게 아니라 각 시점의 위치 표현에 시장 상태를 가산적으로 추가한다.

3 seeds x 3 epochs, S&P 500, 2022-2024:

| 모델 | IC mean | IC std |
|------|---------|--------|
| concat_a | **0.0571** | 0.0229 |
| econ_time | 0.0438 | 0.0273 |
| econ_time:pe_only | 0.0429 | 0.0273 |
| static | 0.0103 | 0.0604 |

t-test: econ_time vs concat_a, 모든 seed에서 p > 0.48.

`pe_scale_mean = 0.003~0.004` — PE에 대한 시장 상태의 기여가 사실상 0이다. $g$를 키우면 IC가 음수로 붕괴한다. **유효한 작동 범위가 없었다.**

---

## 인터페이스 3: tau_rope — 경제적 시간으로 RoPE 회전

Clark(1973)의 경제적 시간을 Transformer에 직접 구현한다. 시장 활동 강도로부터 시간 좌표 $\tau$를 만들고 RoPE에 사용한다.

$$\Delta\tau_t = \text{softplus}(\text{head}(h_t)), \quad \tau_t = \sum_{s=1}^{t} \Delta\tau_s$$

$$\text{RoPE}(q_t) = q_t \cdot e^{i \theta \cdot \tau_t}$$

$h_t$는 시장 경로를 처리하는 causal GRU의 hidden state다. 지도의 거리 척도를 교통량에 비례하게 바꾸는 것이다.

두 변형이 있다:
- **Rule-based**: $\Delta\tau_t = \text{softplus}(\alpha \cdot \text{intensity}_t)$
- **Learned**: GRU가 $\Delta\tau_t$를 end-to-end 학습

RoPE에서 attention score는 $f(\tau_t - \tau_s)$의 함수이므로, $\tau$를 경제적 시간으로 바꾸면 attention이 "경제적으로 가까운 시점"을 더 강하게 참조하게 된다. 이론적으로 가장 우아한 해법이다. 결과는 Chapter 05에서 다룬다.

---

## 인터페이스 4: film_a — FiLM 곱셈적 변조

Perez et al.(2018)의 FiLM을 적용한다. 시장 상태로부터 게인과 바이어스를 계산하고 은닉 표현을 곱셈적으로 변조한다:

$$h_t = \gamma(c_t) \odot h_x + \beta(c_t)$$

- $\gamma(c_t) = W_\gamma \cdot c_t + b_\gamma$ (게인), $\beta(c_t) = W_\beta \cdot c_t + b_\beta$ (바이어스)
- $h_x = W_x \cdot x_t$ (자산 피처 투사)

concat이 정보를 "추가"한다면, FiLM은 기존 표현을 "재구성"한다.

**왜 이론적으로 유력했나**: Jayakumar et al.(2020)에 따르면, 곱셈적 구조는 조건부 정보 전달에서 덧셈보다 효과적이다. $\gamma(c)$가 각 채널의 방향과 크기를 동시에 조절하기 때문이다. 시장 상태가 피처의 의미 자체를 바꾸는 금융에서는 곱셈적 변조가 자연스러웠다.

> "Bear 마켓의 모멘텀 주식과 Bull 마켓의 모멘텀 주식은 같은 '모멘텀'이 아니다."

이것이 우리의 사전 기대였다.

파라미터를 맞춰서(concat_a: 45,266, film_a: 45,330) 비교한 결과:
- concat_a:intensity_indexret IC = **0.0592**
- film_a:intensity_indexret IC = **-0.0081**

FiLM이 static(0.0103)보다도 나빴다. 이 역전의 원인은 Chapter 04에서 분석한다.

---

## 인터페이스 5: xip_a — 명시적 상호작용 항

concat_a가 왜 강한지를 분석하다가 도달한 설계다. 자산 피처와 시장 상태의 상호작용을 **명시적으로 분리**한다:

$$h_x = W_x x_t, \quad h_s = W_s c_t, \quad h_{\text{int}} = (U_x x_t) \odot (U_s c_t)$$

$$h_t = h_x + h_s + h_{\text{int}}$$

$U_x, U_s$는 rank-4 투사 행렬이다. 통계학에서 메인 효과와 상호작용 효과를 분리한 모형 행렬을 만드는 것과 같다.

이 설계에 도달한 이유 — concat_a 분해 실험(Chapter 07에서 상세):

| 구성 | IC mean |
|------|---------|
| intensity alone | 0.0066 |
| indexret alone | 0.0205 |
| intensity + indexret (concat) | **0.0592** |
| linear ensemble | 0.0184 |

단일 채널은 약하고, pair 모델은 강하고, 선형 결합으로 설명 안 된다. concat_a의 핵심은 **interaction-friendly interface**였다.

xip_a 결과 (3 seeds, 3 epochs):
- concat_a:intensity_indexret IC = 0.0592, std = 0.0226
- xip_a:intensity_indexret IC = **0.0608**, std = 0.0217
- pooled paired test: p = 0.924

붕괴하지 않았고 안정적이지만, 통계적으로 concat_a를 이기지는 못했다.

---

## 다섯 인터페이스 비교 요약

| 인터페이스 | 주입 위치 | 주입 방식 | IC mean |
|-----------|----------|----------|---------|
| concat_a | 입력 | 가산적 결합 | **0.0571** |
| econ_time | PE | 가산적 PE 변조 | 0.0438 |
| tau_rope | RoPE 각도 | 시간 좌표 교체 | 0.0354 |
| film_a | 은닉 표현 | 곱셈적 변조 | -0.0081 |
| xip_a | 입력 | 명시적 상호작용 | 0.0608 |

(tau_rope는 learned 3-epoch, film_a/xip_a는 intensity+indexret pair 기준)

---

## 사전 기대와 현실의 괴리

실험 전 기대: film_a > tau_rope > xip_a > econ_time > concat_a

실제 결과: xip_a $\approx$ concat_a >> econ_time > tau_rope >> film_a

**가장 단순한 방법이 가장 이론적인 방법을 이겼다.** 이것은 우연이 아니다.

---

## 이 장의 핵심 정리

1. 같은 시장 상태를 넣는 방법이 최소 다섯 가지 있고, 결과는 결정적으로 다르다
2. 주입 위치(입력/PE/RoPE/은닉)와 방식(가산적/곱셈적/좌표교체)이 핵심 변수다
3. 이론적으로 우아한 방법(FiLM, tau-RoPE)이 단순한 concat에 졌다
4. 이 역전은 금융 시계열의 낮은 SNR과 관련이 있다 (다음 장)

---

## 참고 문헌

- **Perez et al.** (2018). *FiLM: Visual reasoning with a general conditioning layer.* AAAI.
- **Jayakumar et al.** (2020). *Multiplicative interactions and where to find them.* ICLR.
- **Clark, P. K.** (1973). *A subordinated stochastic process model.* Econometrica, 41(1).
- **Su et al.** (2024). *RoFormer: Enhanced transformer with rotary position embedding.* Neurocomputing, 568.
