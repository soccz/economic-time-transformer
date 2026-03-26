# Next Method Design Options

이 문서는 현재 증거를 바탕으로, 다음 method를 어떤 질문으로 설계해야 하는지 고정한다.

## Locked Update

이제 질문은 더 이상 이것이 아니다.

`어떤 PE가 concat보다 더 나은가?`

현재 질문은 이것이다.

`intensity와 다른 market-state channel의 interaction을 구조적으로 살리면서, raw concat보다 더 잘 일반화하는 interface는 무엇인가?`

즉 `H2`의 중심이 바뀌었다.

- old `H2`: `PE injection superiority`
- new `H2*`: `interaction-friendly interface superiority`

## Why The Question Changed

현재 strongest empirical signal은 [10_concat_interaction_results.md](./10_concat_interaction_results.md) 에 있다.

핵심 요약:

- 단일 채널은 모두 약하다
- pair model은 강하다
- 단순 선형 결합 baseline은 pair model을 설명하지 못한다

따라서 현재 strongest baseline의 강점은

- `better scalar signal`
- `better ordering`
- `better PE`

보다는,

- `interaction-friendly input interface`

에 더 가깝다.

## Non-Negotiable Requirement For The Next Method

다음 method는 아래 중 적어도 하나를 보여야 한다.

1. `concat_a`와 같은 channel interaction을 더 구조적으로 표현한다.
2. 그 표현이 `concat_a`보다 더 잘 일반화된다.
3. 같은 interaction signal을 더 적은 capacity나 더 강한 해석 가능성으로 달성한다.

즉 단순히 geometry를 더 바꾸는 것으로는 부족하다.

## Two Main Design Branches

### Branch A. FiLM / Explicit Interaction Conditioning

아이디어:

- market-state summary를 만들어
- asset sequence representation 또는 Q/K/V에 modulation으로 주입
- interaction을 `명시적 구조`로 모델링

후보 예시:

- FiLM on input projection
- FiLM on transformer blocks
- Q/K-specific gating
- cross-feature gating between asset token and state summary
- explicit interaction projection (`x`, `s`, `x ⊙ s`-style low-rank term)

장점:

- interaction을 직접 모델링한다
- raw concat보다 구조적이다
- 해석이 비교적 쉽다

위험:

- 결국 concat의 다른 형태로 보일 수 있다
- 정말 더 일반화되는지 보여야 한다

### Branch B. Interaction-Aware Positional / Temporal Interface

아이디어:

- intensity와 다른 state channel의 interaction을 먼저 계산
- 그 interaction-derived signal을 temporal coordinate나 PE에 반영

예시:

- intensity x indexret interaction embedding
- state-pair-driven PE modulation
- profile-conditioned PE

장점:

- 원래 `economic time` 철학과 더 가깝다
- positional interface를 완전히 버리지는 않는다

위험:

- 다시 `PE idea`를 너무 일찍 밀 가능성이 있다
- 지금까지의 실패를 보면 interaction을 먼저 잡지 않으면 또 같은 문제가 반복될 수 있다

## Recommended Priority

현재 추천 우선순위는 아래와 같다.

1. `Branch A`를 먼저 본다
2. `Branch B`는 `Branch A`에서 interaction structure가 확인된 뒤에 본다

이유:

- 지금 strongest evidence는 `interaction`이지 `positional superiority`가 아니다
- 따라서 다음 method는 먼저 interaction-friendly interface를 정면으로 겨냥해야 한다

## Current Branch A Status

[12_branchA_film_results.md](./12_branchA_film_results.md) 기준으로, 최소 `FiLM` 구현은 아직 실패다.

- `concat_a:intensity_indexret IC mean = 0.0592`
- `film_a:intensity_indexret IC mean = -0.0081`

즉 `Branch A`라는 방향 자체는 유지되지만,

- `minimal FiLM on input projection`

은 현재 정답이 아니다.

따라서 다음 `Branch A` 설계는 단순 FiLM 재시도가 아니라,

1. 더 안정적인 interaction path
2. raw concat이 실제로 사용하는 channel interaction을 보존하는 구조
3. seed instability를 줄이는 injection interface

를 노려야 한다.

현재 가장 유력한 다음 후보는 [13_explicit_interaction_projection_plan.md](./13_explicit_interaction_projection_plan.md) 의

- `explicit interaction projection`

이다.

이 후보를 우선하는 이유:

- `concat_a`의 early shared projection 장점을 잃지 않는다
- interaction을 명시적으로 분리해 해석할 수 있다
- `FiLM`보다 raw concat의 작동 방식에 더 가깝다

[14_xip_results.md](./14_xip_results.md) 기준으로, 이 후보는 실제로 첫 positive signal을 보였다.

- `concat_a:intensity_indexret IC mean = 0.0592`
- `xip_a:intensity_indexret IC mean = 0.0608`

따라서 현재 `Branch A` 우선순위는 다음처럼 바뀐다.

1. `explicit interaction projection`을 먼저 반복 검증
2. 그 다음에만 더 복잡한 FiLM/gating 설계를 본다

## Success Criteria For The Next Method

다음 method는 아래 셋 중 최소 둘을 만족해야 한다.

1. `concat_a` 대비 평균 IC 개선
2. seed 반복에서 더 안정적
3. interaction ablation에서 raw concat보다 더 해석 가능한 구조를 보임

즉 다음 method는 `새롭다`보다 `concat_a의 핵심 강점을 더 잘 표현한다`를 먼저 보여야 한다.

## What We Should Not Do Next

현재 증거 기준으로 아래는 우선순위가 낮다.

- 더 복잡한 pointwise tau warping
- 더 강한 geometry objective만 추가
- intensity-only PE 개선을 method headline으로 밀기

이건 지금까지의 실패를 반복할 가능성이 높다.

## One-Sentence Direction

`The next method should be designed around interaction-friendly interfaces, not around positional novelty alone.`
