# Chapter 05: 학습된 시간 좌표의 세 가지 병목

> *"좌표가 바뀌었다. Attention이 바뀌었다. 그런데 예측은 안 바뀌었다.
> 이게 대체 어떻게 가능한 거지?"*

---

## 이 장의 질문

tau-RoPE는 이론적으로 가장 우아한 방법이다. 학습된 버전이 정렬에 성공하고 geometry를 바꾸면서도 **예측에 실패하는 이유**를 분석한다.

---

## Learned tau-RoPE 구현

시장 경로 데이터를 causal GRU에 넣어 경제적 시간 증분을 학습한다.

$$h_t = \text{GRU}(h_{t-1}, \text{market}\_t)$$

$$\Delta\tau_t = \text{softplus}(\text{head}(h_t)), \quad \tau_t = \sum_{s=1}^{t} \Delta\tau_s$$

$$q_t' = q_t \cdot e^{i\theta \cdot \tau_t}, \quad k_t' = k_t \cdot e^{i\theta \cdot \tau_t}$$

softplus는 $\Delta\tau > 0$을 보장하고(시간은 앞으로만), cumsum은 이를 누적한다. 학습 목표에 정렬 손실을 추가한다:

$$\mathcal{L} = \mathcal{L}_{\text{pred}} + \lambda \cdot (-\text{Pearson}(\Delta\tau_t, \text{intensity}_t))$$

이것 없이는 GRU가 경제적 의미와 무관한 좌표를 학습한다 (naive 버전: step-intensity Spearman = -0.50).

---

## 좋은 소식: 정렬은 됐다

3 seeds, 3 epochs, S&P 500, 2022-2024:

**step-intensity Spearman = +0.46 ~ +0.52** (세 시드 모두 안정적)

$\Delta\tau_t$가 큰 시점 = intensity가 높은 시점. 시장이 활발한 날에 경제적 시간이 빠르게 흐른다. Clark이 말한 종속 과정의 핵심 속성이 학습된 좌표에서 나타났다.

"좌표가 시장 활동과 맞춰졌다!"

---

## 좋은 소식 2: Geometry도 변했다

QK ordinal loss를 추가하면:

**qk_ord_rate = 0.992** — 경제적 시간으로 가까운 시점 쌍의 99.2%에서 pre-softmax QK 내적이 경제적 시간으로 먼 쌍보다 크다. Attention이 경제적 시간 거리에 따라 시점을 구분하고 있다.

"Attention 패턴이 바뀌었다!"

---

## 나쁜 소식: 예측은 안 된다

| 모델 | IC mean |
|------|---------|
| concat_a | **0.0571** |
| learned_tau_rope | 0.0354 |
| static | 0.0103 |

Geometry를 강제로 극대화하면? qk_swap_delta = 0.022, **IC = -0.0072** — 오히려 음수로 붕괴.

QK ordering을 거의 완벽하게 만들면? qk_ord_rate = 0.992, qk_swap_delta = 2e-05, **IC = 0.0002** — 사실상 0.

> **"예측은 안 된다. 왜?"**

세 가지 병목이 이 간극을 설명한다.

---

## 병목 1: Softmax가 미세한 QK 변화를 압축

| 진단 | rule-based tau | learned tau |
|------|---------------|-------------|
| qk_swap_delta (pre-softmax) | 1.6e-05 | 4.5e-07 |
| attn_swap_delta (post-softmax) | ~1e-05 | ~1e-07 |

Pre-softmax에서 post-softmax로 가면서 변화가 한 자릿수 더 작아진다.

Softmax 출력 $\alpha_{t,s} = \exp(q_t k_s / \sqrt{d}) / \sum_{s'} \exp(q_t k_{s'} / \sqrt{d})$에서, QK 변화 $\delta$에 대한 attention 변화는:

$$\Delta\alpha \approx \alpha(1-\alpha) \cdot \delta / \sqrt{d}$$

$T=168$, $d=32$이면 $\Delta\alpha \approx \delta / 950$. QK 변화가 softmax를 통과하면서 **약 1000배 압축**된다.

tau-RoPE가 만드는 QK 변화는 softmax의 감도 문턱보다 작다. 경제적 시간이 다른 ordering을 만들어도 그 차이가 정규화에 의해 씻겨나간다.

(이 병목의 해결책 — linear attention — 은 Chapter 08에서 다룬다.)

---

## 병목 2: 신호는 피처 상호작용에 있다

Chapter 03의 concat_a 분해에서 발견한 것:

| 모델 | IC mean |
|------|---------|
| intensity alone | 0.0066 |
| indexret alone | 0.0205 |
| intensity + indexret (concat) | **0.0592** |
| linear ensemble | 0.0184 |

예측 핵심 신호는 **intensity와 indexret의 비선형 상호작용**에 있다.

tau-RoPE는 이 상호작용에 접근 불가다. $\tau_t$는 스칼라 — 각 시점에 숫자 하나만 할당한다. intensity 정보를 담을 수 있지만, **intensity와 indexret의 상호작용**은 최소 2차원 정보다.

"변동성이 높으면서 시장이 하락하는" 조합과 "변동성이 높으면서 시장이 상승하는" 조합은 예측에 완전히 다른 의미를 가진다. 스칼라 시간 좌표 하나로는 이 구분을 표현할 수 없다.

---

## 병목 3: cumsum이 tau를 물리적 시간으로 만든다

**tau_corr ($\tau$와 $t$의 Pearson 상관) > 0.998**

사실상 $\tau \approx t$다.

이것은 cumsum의 구조적 결과다. $\tau_t = \sum \Delta\tau_s$에서 $\Delta\tau_s > 0$(softplus)이므로 $\tau$는 단조 증가. $t$도 단조 증가. 두 단조 증가 시퀀스의 상관은 구조적으로 높다.

상관을 0.9 아래로 낮추려면 $\sigma(\Delta\tau) / \mu(\Delta\tau) > 6.3$이어야 한다. 한 시점에서 다음까지 경제적 시간 증분이 평균의 6배 이상 흔들려야 한다는 뜻인데, 현실적이지 않다.

더 나아가, $\Delta\tau$의 분산이 커지면 RoPE 회전 각도가 불안정해져서 학습이 망가진다.

**순환 딜레마**: $\tau \neq t$가 되려면 분산이 커야 하고 → 분산이 커지면 학습 불안정 → 학습이 분산을 줄이는 방향으로 수렴 → $\tau \approx t$

---

## "표현이 변했는데 예측이 안 되는" 패턴

이 패턴은 다른 분야에서도 반복된다:

| 연구 | 바뀐 것 | 안 바뀐 것 |
|------|--------|-----------|
| Hewitt & Liang (2019) | 표현에 구문 정보 존재 | 모델이 사용하는지 |
| Locatello et al. (2019) | 표현이 인자별로 분리 | 다운스트림 성능 |
| Jain & Wallace (2019) | Attention 패턴 | 최종 예측 |
| 이 연구 | $\tau$ 정렬 + QK ordering | IC (예측 성능) |

Hewitt & Liang: BERT attention에서 구문 구조를 발견해도, 모델이 그것을 예측에 사용하는지는 별개. "Probe가 찾을 수 있다" $\neq$ "모델이 쓰고 있다".

Locatello et al.: 비지도 disentanglement에서 표현의 "의미적 품질"과 "예측적 유용성"은 다른 축이다.

Jain & Wallace: attention weight를 무작위로 섞어도 예측이 크게 안 바뀌는 경우가 많다.

이 패턴의 반복은 우리 실패가 구현 실수가 아니라 **구조적 간극**임을 시사한다.

---

## 이 시점에서 우리가 몰랐던 것

이 세 병목을 진단한 시점에서 우리는 **해결책이 무엇인지 몰랐다**. 병목 1(softmax 압축)에 대해 linear attention이 답이 될 수 있다는 것은 한참 뒤에야(Chapter 08) 발견했다. 당시에는 RoPE 각도 스케일링, 별도의 attention head 분리, QK 정규화 변형 등을 시도하고 있었고, softmax 자체를 제거하는 것은 고려하지 않았다.

더 중요한 오판은 **어떤 병목이 지배적인가**에 대한 것이었다. 이 장을 쓸 때 우리는 병목 2(상호작용)가 가장 근본적이라고 확신했다 — 스칼라 좌표로는 다차원 상호작용을 표현할 수 없다는 것이 가장 근본적인 한계로 보였기 때문이다. 하지만 나중에 돌아보면, 병목 1(softmax)에 대한 수술이 가장 큰 예측 개선을 가져왔다. 이론적으로 가장 "깊어 보이는" 병목이 실무적으로 가장 중요한 병목은 아니었다. 이 경험은 "먼저 가장 다루기 쉬운 병목부터 해결하라"는 교훈을 남겼다.

---

## 10개 ablation이 모두 실패한 과정

세 병목을 진단한 후, 이를 우회하거나 완화하는 10가지 방향을 시도했다. 모두 실패했다. 이 실패들이 누적되어 method paper NO-GO 판정의 근거가 됐다.

**Align-only learned tau** (A): 정렬 손실만으로 학습한 $\tau$. step-intensity 정렬은 성공(Spearman +0.46~+0.52)했지만, IC=0.0354로 concat_a(0.0571)에 크게 못 미쳤다. 정렬만으로는 geometry도 예측 이득도 불충분했다.

**Strong geometry objective** (B): geometry 목적함수를 강하게 걸어 qk_swap_delta를 0.022까지 올렸다. 하지만 IC=-0.0072로 음수 붕괴, 정렬도 Spearman 0.124로 무너졌다. geometry를 강제하면 표현이 예측에서 멀어진다.

**QK ordinal loss** (C): qk_ord_rate를 0.992까지 끌어올려 QK ordering을 거의 완벽하게 만들었다. IC=0.0002 — 사실상 0. 이것이 병목 1의 가장 직접적인 증거다: ordering은 맞는데 softmax가 그 차이를 지운다.

**Global-only ablation** (D): local branch를 제거하여 신호 경쟁을 없앴다. static_tau_rope(0.0346) > learned_tau_rope(0.0239). local이 덮는 것만이 문제가 아니었다.

**Window signature token** (E): shape signature와 simple summary token 모두 IC 음수(-0.0226, -0.0267). global-only로 바꿔도 shape은 0.0000, simple은 -0.0144. 윈도우 레벨 요약 토큰은 유효한 예측 인터페이스가 아니었다.

**Concat decomposition** (G/J): intensity 제거 시 IC 0.0571→0.0026, intensity만 넣으면 0.0066. intensity가 필수이지만 intensity만으로는 설명 불가. 상호작용이 핵심이라는 진단의 근거.

**Cycle PE** (H/I): PE 주입 경로로 intensity를 넣은 cycle_pe. linear 인코딩(IC=-0.0244)은 참담, embedding 인코딩(IC=0.0040)은 나았지만 concat(0.0066)을 여전히 못 넘었다. PE 경로 자체의 우위를 주장할 수 없었다.

**FiLM Branch A** (J): Chapter 04의 결과 재확인. IC=-0.0081, static(0.0103)보다 나쁨.

**Explicit interaction projection** (K): xip_a(IC=0.0608)가 처음으로 concat_a(0.0592)에 근접했다. 하지만 paired t-test p=0.9242로 유의하지 않았고, interaction head 제거 시 IC 하락은 0.0019에 불과. "stable match"이지 "winner"는 아니었다.

10번 시도하고 10번 모두 concat_a를 유의하게 넘지 못했다. 이 시점에서 현재 프레임워크(RoPE 기반 경제적 시간 좌표)로는 method paper를 쓸 수 없다는 판정을 내렸다. 실패 자체가 발견이다 — 이 10개의 ablation이 "왜 안 되는가"에 대한 체계적 증거를 구성한다.

---

## StretchTime이 증명한 것

우리가 경험적으로 부딪힌 병목 1 — RoPE 기반 시간 좌표가 softmax를 통과하면서 예측에 무의미해지는 현상 — 을 Kim et al. (2026)의 StretchTime이 수학적으로 증명했다. 그들의 Theorem 3.1은 **RoPE가 non-affine 시간 워핑(warping)을 표현할 수 없다**는 것을 보인다. 경제적 시간은 본질적으로 non-affine이다 — 시장 활동에 따라 비선형적으로 늘어나고 줄어든다. RoPE의 회전 구조는 균일한 간격의 선형 변환만 자연스럽게 표현하므로, 우리가 cumsum으로 만든 $\tau$가 결국 $t$에 붙어버리는 것은 구조적으로 불가피했다.

이것은 후행적 확인(retrospective validation)이다. 우리가 10개 ablation으로 경험적으로 "안 된다"는 것을 확인한 후에, 누군가 "왜 안 되는지"를 수학적으로 증명했다. 연구 순서로는 경험이 먼저이고 이론이 나중이다. 하지만 이 증명이 우리 발견의 의미를 바꾸지는 않는다 — 오히려 우리의 경험적 병목 진단이 수학적으로 견고했음을 확인해준다.

---

## 세 병목의 관계

세 병목은 서로 강화한다:

```
[병목 3] cumsum → τ ≈ t → QK 변화 작음
                            ↓
[병목 1] softmax가 작은 QK 변화를 더 압축 → attention 변화 ≈ 0
                            ↓
[병목 2] 어차피 신호는 피처 상호작용에 있음 → tau-RoPE는 접근 불가
```

어느 하나만 해결해서는 작동하지 않는다.

---

## Global-Only 분리 실험

Local branch(TCN)가 tau-RoPE 신호를 덮는가?

| 모델 | IC mean |
|------|---------|
| concat_a | **0.0713** |
| static_tau_rope:global_only | 0.0346 |
| learned_tau_rope:global_only | 0.0239 |

Local branch를 제거해도 learned는 static을 이기지 못한다. "local이 덮어서 안 되는 거야"라는 설명은 부분적으로만 맞다. 병목은 tau-RoPE 자체의 구조적 한계에도 있다.

---

## 이 장에서 배운 것

**교훈 1**: 정렬은 필요하지만 충분하지 않다. 정렬된 표현이 예측에 사용 가능한 경로를 통해 전달되어야 한다.

**교훈 2**: Geometry 변화와 예측 개선은 다른 문제다. QK ordering이 99.2% 올바르게 바뀌어도 IC는 0에 가깝다.

**교훈 3**: 스칼라 시간 좌표의 한계. 예측에 필요한 시장 상태 정보는 다차원이다. $\tau_t$ 하나로는 접근 불가능한 신호가 있다.

---

## 참고 문헌

- **Clark, P. K.** (1973). *A subordinated stochastic process model.* Econometrica, 41(1).
- **Hewitt, J., & Liang, P.** (2019). *Designing and interpreting probes with control tasks.* EMNLP.
- **Locatello, F., et al.** (2019). *Challenging common assumptions in unsupervised disentanglement.* ICML.
- **Jain, S., & Wallace, B. C.** (2019). *Attention is not explanation.* NAACL.
- **Kim, J., et al.** (2026). *StretchTime: Non-affine temporal warping beyond RoPE.* (Theorem 3.1: RoPE는 non-affine warping 불가)
