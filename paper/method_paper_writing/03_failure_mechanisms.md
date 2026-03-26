# Failure Mechanisms

이 파일은 지금까지 실험에서 반복적으로 드러난 실패 메커니즘만 정리한다.

## 1. Alignment Is Not Enough

`Δτ`와 시장 intensity의 정렬이 좋아져도 predictive gain은 자동으로 나오지 않았다.

의미:

- semantic alignment
- predictive usefulness

는 다른 문제다.

## 2. QK Geometry Change Is Not Enough

`QK ordinal loss`를 쓰면 pre-softmax QK ordering은 거의 완벽하게 만들 수 있었다.

하지만:

- post-softmax attention 변화는 작고
- 최종 prediction 변화는 더 작았다.

의미:

`geometry change != output change`

## 3. Softmax Bottleneck Exists

QK는 바뀌는데 attention map에서 그 차이가 눌렸다.

이는 다음 둘 중 하나를 뜻한다.

- QK 차이의 절대 규모가 작다
- softmax 이후 정규화가 차이를 희석한다

이 병목은 실제로 관측되었다.

## 4. Local Branch Suppression Exists

`learned_tau_rope:global_only`가 `learned_tau_rope`보다 낫다.

의미:

현재 hybrid 구조에서는 local branch가 regime-aware geometry 신호를 덮고 있다.

## 5. Local Removal Alone Does Not Solve The Problem

같은 global-only 조건에서 `static_tau_rope:global_only > learned_tau_rope:global_only` 였다.

즉:

- local branch suppression은 진짜 병목이지만
- 유일한 병목은 아니다.

## 6. Naive Window Signature Is Too Weak

`shape_signature_token`은 `simple_summary_token`보다 약간 나은 방향을 보일 때가 있었지만,
반복적으로 강한 baseline을 넘지 못했다.

의미:

`window-level signature`라는 아이디어 자체보다,
현재 signature 설계가 너무 약한 가능성이 크다.

## 7. Strong Baselines May Simply Use The Right Signal Better

`concat_a` intensity ablation은 중요한 단서를 준다.

- intensity 채널을 제거하면 `concat_a` 평균 `IC`가 `0.0571 -> 0.0249`로 크게 떨어진다.
- 즉 현재 strongest baseline의 힘은 `복잡한 temporal geometry`보다도, `simple but predictive intensity-linked signal`을 input path에서 바로 활용하는 데 있을 수 있다.

이건 다음 설계 방향에 직접적인 의미가 있다.

- 더 sophisticated한 representation을 만드는 것만으로는 부족하다.
- baseline이 이미 잘 쓰는 signal을 놓치면, geometry나 signature 개선도 predictive utility로 이어지지 않을 수 있다.

## 8. PE Injection Is Not Automatically Better Than Input Concat

`cycle_pe:intensity_only` vs `concat_a:intensity_only`의 공정 비교는 또 하나의 중요한 failure signal이다.

- 두 모델은 같은 intensity signal을 썼다.
- 추가 파라미터 수도 사실상 맞췄다.
- 그럼에도 현재 `cycle_pe`는 `concat_a`를 이기지 못했다.

의미:

- 문제는 단순히 `signal이 없어서`가 아니다.
- 같은 signal을 줘도 `PE injection interface`가 현재 forecasting objective에 더 잘 맞는다고 말할 수 없다.

따라서 현재 기준으로는 아래 문장이 더 정직하다.

`The current failure is not only about learning better signals, but also about whether positional injection is the right interface for using those signals.`

## 9. The PE Encoding Function Itself Matters

`cycle_pe:intensity_embed` sanity check는 중요한 보완 정보를 준다.

- binned embedding은 bias-free linear projection보다 확실히 나았다.
- 즉 현재 `cycle_pe` failure는 `PE interface` 문제만이 아니라, `intensity를 PE 공간으로 올리는 함수`가 너무 약했기 때문일 수도 있다.

하지만 여기서도 선을 그어야 한다.

- `embed > linear`는 맞다.
- 그러나 `embed`도 아직 `concat_a:intensity_only`를 안정적으로 넘지 못했다.

따라서 가장 정직한 해석은 아래와 같다.

`PE encoding quality matters, but improving the encoding function alone is not yet sufficient to overturn the concat baseline.`

## Current Best Explanation

현재까지 가장 정직한 설명은 아래 두 줄이다.

1. regime-aware representation은 geometry를 바꿀 수 있다.
2. 그러나 지금까지의 인터페이스는 그 geometry 변화를 예측적으로 유용한 신호로 변환하지 못했다.
