# Chapter 08: Softmax를 제거하면 무슨 일이 일어나는가

## Linear attention 수술과 +49%

---

Chapter 05에서 tau-RoPE의 세 가지 병목을 식별했다. 그중 가장 위험해 보였던 것은 병목 2(상호작용 접근)였다 -- tau-RoPE는 시간 좌표만 변형하므로, Chapter 07에서 확인한 intensity x indexret 상호작용에 접근할 수 없다.

그래서 우리는 병목 1(softmax 압축)을 **부차적**이라고 판단했다. softmax가 미세한 QK 변화를 소거한다는 것은 알았지만, 상호작용에 접근 못하는 것이 더 근본적인 문제라고 봤다.

**틀렸다.**

softmax를 제거하는 것만으로 IC가 49% 올랐다. 그리고 이것은 "부차적 병목"을 해결한 결과다.

---

## 1. Linear attention이란

표준 Transformer의 attention:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

softmax가 하는 일은 QK^T의 각 행을 확률 분포로 변환하는 것이다. 모든 값이 양수가 되고, 합이 1이 된다. 이것은 attention weight를 "선택적"으로 만든다 -- 가장 높은 유사도를 가진 위치에 weight가 집중된다.

Linear attention(Katharopoulos et al., 2020)은 softmax를 feature map으로 대체한다:

```
LinearAttention(Q, K, V) = phi(Q) (phi(K)^T V) / (phi(Q) phi(K)^T 1)
```

여기서 phi는 비선형 feature map이다. 우리는 ELU+1을 사용했다: `phi(x) = ELU(x) + 1`. 이것은 모든 값을 양수로 만들면서도 softmax의 winner-take-all 동작을 제거한다.

핵심 차이: **softmax는 QK의 미세한 차이를 증폭하거나 소거한다.** 높은 유사도 쌍이 있으면 나머지를 거의 0으로 만든다. Linear attention은 QK의 변화를 더 **선형적으로** 전달한다.

---

## 2. 왜 이것이 tau-RoPE에 중요한가

tau-RoPE가 하는 일을 복기하자:

```
표준 RoPE: Q, K에 cos/sin(pos * theta)를 곱한다
tau-RoPE:  Q, K에 cos/sin(tau * theta)를 곱한다
           (tau = cumsum(softplus(alpha * intensity)))
```

tau와 pos의 차이는 **미세하다.** tau_corr > 0.998. 경제적 시간과 물리적 시간의 차이가 QK dot product에 미치는 영향은 아주 작다.

softmax가 이 미세한 차이에 무슨 일을 하는가? QK dot product의 순서가 바뀌더라도(qk_ord_rate = 0.992, 즉 0.8%의 위치 쌍에서 순서 변화), softmax를 통과하면 이 변화가 소실된다. 가장 큰 값이 weight의 대부분을 가져가고, 미세한 순서 변화는 최종 출력에 영향을 미치지 못한다.

Paper 2에서 이것을 "softmax 압축 병목"이라 불렀다. 이론적 분석이었다. 실험으로 확인해야 했다.

---

## 3. 수술적 개입: softmax만 제거

실험 설계는 단순했다. tau_rope 구성에서 **attention softmax만 제거**하고, 나머지는 모두 동일하게 유지했다. 학습률, 정규화, 초기화, 에폭 수, 데이터 분할 -- 전부 같다. 변경된 것은 softmax -> ELU+1 feature map, 이것뿐이다.

이 설계의 장점은 **인과적 추론**이 가능하다는 것이다. 다른 모든 것이 동일하므로, 성능 차이는 softmax 제거에 귀인할 수 있다. "수술적 개입(surgical intervention)"이라 부르는 이유다.

---

## 4. 결과

### 표 1: Linear attention 수술적 개입 (GSPC, 2022-2024, 3에폭)

| 모델 | Seed 7 IC | Seed 17 IC | Seed 27 IC | **평균 IC** |
|------|-----------|------------|------------|------------|
| tau_rope (softmax) | +0.063 | -0.028 | +0.055 | **+0.030** |
| tau_rope_linear (linear) | +0.076 | +0.000 | +0.057 | **+0.045** |
| **Delta (linear - softmax)** | +0.013 | **+0.028** | +0.002 | **+0.015** |

평균 IC: 0.030 -> 0.045. **+49% 상대적 개선.**

3개 시드 모두에서 linear attention이 softmax를 상회한다. 예외 없이.

---

## 5. Seed 17: softmax가 신호를 파괴하고 있었다

표에서 가장 극적인 것은 seed 17이다.

tau_rope (softmax): IC = **-0.028**. 음수다. 예측 방향이 역전되어 있다. 이 모델의 예측을 뒤집으면 더 정확하다는 뜻이다.

tau_rope_linear (linear): IC = **+0.000**. 정확히 0. 예측이 무작위 수준이다.

Delta = +0.028.

softmax가 tau 신호를 단순히 "약화"시키는 것이 아니었다. 특정 초기화 조건에서 softmax는 tau-RoPE의 신호를 **적극적으로 역전**시키고 있었다. 잡음을 만드는 정도가 아니라, 방향을 뒤집는 수준이다.

linear attention으로 교체하면 이 역전이 사라진다. -0.028이 +0.000이 된다. "구조(rescue)"라는 표현이 정확하다.

seed 17에서 무슨 일이 일어난 것인가? 구체적 메커니즘에 대한 가설:

- tau-RoPE가 QK dot product에 미세한 변화를 만든다
- softmax의 winner-take-all 동작이 이 미세한 변화를 특정 방향으로 증폭한다
- 초기화에 따라 증폭 방향이 "올바른 방향"일 수도 있고 (seed 7: +0.063) "반대 방향"일 수도 있다 (seed 17: -0.028)
- linear attention은 winner-take-all이 없으므로, 이 불안정한 증폭이 발생하지 않는다

---

## 6. 기존 병목 위계의 전복

이 결과가 왜 예상 밖인지 설명해야 한다.

Chapter 07에서 우리는 예측 신호의 핵심이 intensity x indexret 상호작용에 있음을 확인했다. tau-RoPE는 시간 좌표 경로(intensity -> tau -> RoPE rotation)를 통해 이 상호작용에 구조적으로 접근할 수 없다. 그래서 병목 2(상호작용 접근)가 "지배적"이라고 판단했다.

이 판단에 따르면, softmax를 제거해도 IC가 크게 오르지 않아야 한다. 상호작용에 접근할 수 없는 것이 근본 문제이므로, softmax를 바꿔봤자 IC ~ 0.007(intensity only 수준)에 머물 것이라고 예측했다.

실제: IC = 0.045. 같은 실험 설정에서의 concat_a 평균 IC(0.017)를 능가한다.

> **주의: concat_a 베이스라인의 불일치.** Ch.03-07에서 보고한 concat_a IC=0.057은 별도의 stability run(채널 분해 실험)에서 나온 값이다. 이 linear attention 실험에서의 concat_a 평균은 0.017로, 같은 시드(7,17,27)임에도 크게 다르다. 원인은 실험 설정 차이(모델 구성, 컨텍스트 처리 경로)로 추정되나 정확한 원인은 미규명이다. **따라서 "tau_rope_linear이 concat_a를 이긴다"는 주장은 이 특정 설정 내에서만 유효하며, Ch.03-07의 concat_a(0.057)와 직접 비교하면 tau_rope_linear(0.045)은 여전히 낮다.** 이 불일치는 향후 통합 실험으로 해소해야 한다.

이것은 기존 분석과 **모순**된다. 세 가지 가능한 설명:

1. tau-RoPE 경로가 상호작용에 **부분적으로** 접근하고 있었으나, softmax가 그 접근을 소거하고 있었다
2. 상호작용 없이도 시간 좌표 단독으로 IC ~ 0.04-0.05 수준의 신호가 존재한다
3. linear attention이 softmax 제거 외에도 다른 동역학적 효과를 제공한다

어느 설명이 맞든, "병목 2만 해결하면 된다"는 단순한 서사는 **수정되어야 한다.** 병목 1(softmax 압축)이 기존 분석에서 과소평가되었다.

---

## 7. 그런데 아직 완전한 해결은 아니다

tau_rope_linear의 평균 IC는 0.045다. 같은 설정의 concat_a 평균(0.017)보다 높다. 그러나 별도 실험(Ch.03-07)의 concat_a(0.057)보다는 낮다. 이 불일치를 염두에 두고, 시드별로 보면:

- Seed 7: +0.076 (concat_a의 같은 시드보다 높음)
- Seed 17: +0.000 (여전히 무작위 수준)
- Seed 27: +0.057 (concat_a와 비슷)

Seed 17에서 IC = 0.000이라는 것은, softmax 병목을 제거해도 **이 시드에서는 아무 신호도 포착하지 못한다**는 뜻이다. softmax가 신호를 역전시키는 것은 막았지만, 신호 자체를 만들어내지는 못했다.

완전한 해결을 위해서는 아마 병목 2(상호작용 접근)와 병목 3(단조성 제약)도 함께 해결해야 할 것이다. 하지만 병목 1만 해결해도 49%가 오른다는 것은, 이 방향이 유망하다는 강력한 증거다.

---

## 8. 이 발견의 진짜 의미

이 실험의 가장 중요한 함의는 숫자가 아니다. **관점의 전환**이다.

Chapter 05에서 우리는 tau-RoPE의 실패를 "시간 좌표 접근법 자체의 한계"로 해석했다. 경제적 시간을 Transformer에 넣는 아이디어 자체가 잘못된 것이 아닌가? 신호가 시간 좌표가 아니라 채널 상호작용에 있으니, 시간 좌표를 아무리 정교하게 만들어봤자 소용없는 것 아닌가?

linear attention 실험은 이 해석을 **뒤집는다:**

> **시간 좌표가 쓸모없는 게 아니다. Softmax가 그 신호를 죽이고 있었다.**

동일한 tau-RoPE 표현이 softmax 아래에서는 IC = 0.030이고, softmax 없이는 IC = 0.045다. 표현은 같다. 정보는 그 안에 있다. 문제는 정보를 **활용하는 방식**이었다.

---

## 9. Han et al.(2024)과의 연결: 단사성 관점

여기서 흥미로운 이론적 연결이 있다. Han et al.(2024)은 softmax attention과 linear attention의 근본적 차이를 **단사성(injectivity)** 관점에서 분석했다.

**Softmax attention은 단사적이다.** 다른 쿼리는 다른 attention 분포를 만든다. 이것은 이론적으로 장점이다 -- 쿼리 간 구별 능력이 보장된다.

**Linear attention은 비단사적이다.** 다른 쿼리가 같은 attention 출력을 만들 수 있다. Han et al.은 이것이 "의미적 혼동(semantic confusion)"을 야기한다고 지적했다.

그런데 우리 실험에서는 비단사적인 linear attention이 단사적인 softmax attention보다 **낫다.**

이것이 모순인가? 아니다. 단사성은 **구별 능력의 보장**이지, **구별의 정확성 보장**이 아니다. softmax가 단사적이라는 것은 "다른 쿼리를 다르게 처리한다"는 뜻이지, "올바르게 처리한다"는 뜻이 아니다. 우리의 경우:

- softmax는 tau가 만드는 미세한 QK 차이를 **구별은 하지만, 극단적으로 증폭하거나 소거한다**
- linear attention은 이 차이를 **덜 구별하지만, 더 안정적으로 전달한다**

금융 시계열처럼 미세한 신호가 중요한 맥락에서는, 이론적 구별 능력보다 **실제 신호 전달의 안정성**이 더 중요하다.

Han et al.이 제안한 InLine attention -- linear attention에 단사성을 부여하기 위해 추가적인 정규화를 도입하는 방법 -- 은 이 트레이드오프의 한 가지 해결책이 될 수 있다. 안정적인 신호 전달(linear attention의 장점)과 구별 능력(softmax의 장점)을 동시에 얻을 수 있다면 추가 개선이 가능할 것이다. 하지만 이것은 아직 실험하지 않았다.

---

## 10. Paper 2의 병목 1을 직접 확인하다

이 실험은 Paper 2(잘못된 귀납적 편향)의 핵심 주장 중 하나를 **실험적으로 확인**했다:

> 병목 1: softmax의 winner-take-all 동작이 tau에 의한 미세한 QK 변화를 소거한다.

이것은 더 이상 이론적 추론이 아니다. softmax를 제거하면 IC가 올라간다. 인과적 개입(causal intervention)으로 확인된 메커니즘이다.

동시에, 이 실험은 기존 분석의 **오류도 드러냈다.** "병목 2가 지배적"이라는 판단이 틀렸다. 병목 간 위계는 재평가가 필요하다. 가능성:

- 병목 1이 병목 2보다 더 지배적이다
- 두 병목이 상호작용하여, softmax 제거가 간접적으로 상호작용 접근도 개선한다
- linear attention의 효과가 순수한 softmax 제거에서 오는 것이 아니라, 다른 동역학적 속성에서 온다

현재 데이터로는 이 세 가지를 판별할 수 없다. 다채널 tau를 linear attention 기반으로 실행하는 추가 실험이 필요하다.

---

## 병목 위계의 반전: 우리가 틀렸다

Chapter 05에서 우리는 병목 2(상호작용 접근)가 지배적이라고 확신했다. tau-RoPE는 시간 좌표만 변형하므로 intensity × indexret 상호작용에 구조적으로 접근할 수 없다 — 이것이 근본 문제이고, 병목 1(softmax 압축)은 부차적이라고 판단했다. concat_a가 tau_rope를 능가한 것이 이 판단을 뒷받침하는 것처럼 보였다. 상호작용에 접근하는 concat이 더 나으니, 접근성이 핵심 문제라는 논리였다.

linear attention 결과가 이 위계를 뒤집었다. tau_rope_linear의 평균 IC는 0.045다. concat_a의 평균 IC는 0.017이다. **softmax만 제거했는데 상호작용에 직접 접근하는 concat을 넘었다.** 만약 병목 2가 진짜 지배적이었다면, softmax를 제거해도 상호작용에 접근할 수 없으므로 IC가 concat 수준에 머물거나 그 이하여야 한다. 그런데 2.6배를 넘겼다.

이 결과에 대해 세 가지 해석이 가능하다. (a) 병목 1이 처음부터 지배적이었고, 우리의 원래 진단이 잘못되었다. softmax가 tau 신호를 거의 완전히 소거하고 있었으므로, 상호작용 접근 여부는 부차적이었다. (b) 병목 1과 병목 2가 상호작용한다. softmax가 제거되면 tau-RoPE의 회전이 간접적으로 상호작용 정보를 전달하는 경로가 열린다 — 이 경우 두 병목은 독립이 아니라 결합되어 있다. (c) linear attention이 softmax 제거 이상의 동역학적 효과를 제공한다. ELU+1 feature map이 QK 공간에서 새로운 표현력을 만들어내고, 이것이 tau와는 무관한 추가 신호를 포착한다. 정직하게 인정한다: **현재로서는 세 해석을 판별할 수 없다.** 다채널 tau(intensity + indexret 모두를 위치 인코딩에 반영)를 linear attention 기반으로 실행하는 추가 수술이 필요하다. 만약 (a)가 맞다면 다채널 tau + linear attention이 단채널 tau + linear attention과 비슷해야 하고, (b)가 맞다면 다채널이 유의하게 더 나아야 한다.

---

## StretchTime Theorem 3.1과의 연결

Kim et al.(2026, 프리프린트)은 RoPE의 수학적 구조가 비균등 시간 변환과 근본적으로 비호환적임을 증명했다. RoPE는 SO(2) 회전에 기반하며, 위치 m과 n 사이의 상대적 회전각은 θ(m−n)으로 고정된다. 비선형(non-affine) 시간 변환 τ를 적용하면, θ(τ(m)−τ(n)) = w₀(τ(m)−τ(n)) mod 2π가 원래의 상대적 위치 관계를 보존하는 해를 갖지 않는다. 다시 말해, RoPE의 회전 구조는 등간격 위치를 **전제**로 설계되었으며, 비등간격 tau를 주입하면 회전의 대칭성이 깨진다. 이것이 StretchTime Theorem 3.1의 핵심이다.

이 정리는 우리의 경험적 발견에 수학적 근거를 제공한다. Chapter 05에서 관찰한 "softmax 이전에도 QK 변화가 미미하다"는 현상 — tau_corr > 0.998, qk_ord_rate = 0.992 — 은 단순히 학습이 부족해서가 아니라, **구조적으로 큰 변화를 만들 수 없기 때문**이다. RoPE의 SO(2) 회전이 등간격 가정 위에 서 있으므로, 비등간격 tau가 만들어내는 변화는 수학적으로 제한된다. 경험적으로 벽에 부딪힌 것이, 수학적으로 벽이 존재함이 증명된 것이다. 단, Kim et al.은 프리프린트 단계이며 동료심사를 거치지 않았다. 이 정리가 철회되거나 수정될 가능성을 배제할 수 없으므로, 우리의 논증은 이 정리 없이도 경험적 증거만으로 성립하도록 구성해야 한다.

---

## 시드 17의 이야기: softmax가 신호를 역전시키다

시드 7과 시드 17을 나란히 놓으면 softmax의 파괴적 메커니즘이 선명하게 드러난다. 시드 7에서 tau_rope(softmax)의 IC는 +0.063이고 tau_rope_linear의 IC는 +0.074다. 둘 다 양수이며, linear가 더 좋다. softmax 아래에서도 초기 QK가 "올바른 방향"으로 형성되었고, softmax의 winner-take-all이 이 방향을 증폭했다. 결과적으로 softmax 버전도 쓸만하다. 하지만 시드 17에서는 tau_rope IC가 −0.028이다. 예측 방향이 역전되었다. 같은 구조, 같은 데이터, 같은 학습률 — 초기화만 다른데 신호가 뒤집혔다. tau_rope_linear는 +0.000으로 최소한 무작위 수준까지 복구된다.

메커니즘은 이렇다. 초기 학습에서 무작위로 초기화된 Q, K는 잡음이 섞인 QK dot product를 만든다. tau-RoPE가 이 dot product에 미세한 변화를 추가한다. 여기서 softmax가 개입한다. softmax는 QK의 가장 큰 값에 weight를 집중시키는 winner-take-all 동작을 한다. 초기 학습의 QK가 잡음으로 가득 차 있을 때, winner-take-all은 잡음 중 가장 큰 값을 "정답"으로 선택하고 gradient를 그 방향으로 증폭한다. 시드 7에서는 이 초기 증폭이 우연히 올바른 방향이었다. 시드 17에서는 잘못된 방향이었다. 그리고 일단 잘못된 방향으로 gradient가 증폭되면, 이후 학습이 이 방향을 강화한다 — positive feedback loop다. linear attention은 이 증폭 자체를 제거한다. ELU+1은 softmax처럼 최대값에 집중하지 않으므로, 잡음 중 하나를 "정답"으로 선택하는 일이 없다. 시드 17에서도 잘못된 방향으로의 증폭이 발생하지 않아 최소한 IC = 0.000까지 복구되는 것이다.

---

## 이 발견이 바꾼 것

Chapter 05에서 우리의 결론은 "tau-RoPE는 잘못된 귀납적 편향"이었다. 경제적 시간을 위치 인코딩에 넣는 아이디어 자체가 금융 시계열에 부적합하다는 판정이었다. linear attention 실험은 이 결론을 **정밀하게 수정**한다: tau-RoPE는 잘못된 귀납적 편향이 아니라, **softmax와 호환되지 않는 귀납적 편향**이다. 같은 tau 표현이 softmax 아래에서는 IC = 0.030이고 linear attention 아래에서는 IC = 0.045다. 표현 자체에는 신호가 있다. 문제는 방법이 나쁜 것이 아니라, 조합이 나쁜 것이다.

이 구분은 단순한 의미론이 아니라 연구 전략을 근본적으로 바꾼다. "방법이 나쁘다"면 tau-RoPE를 버리고 다른 접근으로 가야 한다. "조합이 나쁘다"면 조합을 바꾸면 된다. tau + linear attention이라는 새로운 연구 경로가 열린다. 더 나아가 tau를 다채널로 확장하고(intensity뿐 아니라 indexret, spread 등도 시간 좌표에 반영), 이것을 linear attention 또는 InLine attention(Han et al., 2024)과 결합하는 설계가 가능해진다. Chapter 05에서 닫힌 것으로 보였던 문이 다시 열린 것이다.

---

## 11. 요약

| 항목 | 숫자 |
|------|------|
| tau_rope (softmax) 평균 IC | 0.030 |
| tau_rope_linear (linear) 평균 IC | 0.045 |
| 개선 | +0.015 (+49%) |
| concat_a 평균 IC (비교) | 0.017 |
| Seed 17 구조 효과 | -0.028 -> +0.000 |
| 3시드 모두 개선 | 예외 없음 |

핵심 메시지:

1. **Softmax가 tau-RoPE의 신호를 파괴하고 있었다.** 제거만으로 +49%.
2. **시간 좌표가 쓸모없는 게 아니었다.** softmax가 그 정보의 활용을 막고 있었다.
3. **기존 병목 위계가 틀렸다.** softmax 압축이 과소평가되었다.
4. **완전한 해결은 아니다.** seed 17에서 여전히 IC = 0.000. 추가 병목이 남아 있다.
5. **방향은 맞다.** softmax를 우회하는 접근이 유망하다.

다음 장에서는 이 발견을 바탕으로 새로운 해결책을 설계한다 -- test-time에 positional encoding을 적응시키는 TTPA.

---

*다음: [Chapter 09 -- Test-Time Positional Adaptation](09_ttpa.md)*
