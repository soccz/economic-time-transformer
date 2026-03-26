# Concat Interaction Results

이 문서는 `intensity`가 다른 채널과 어떤 상호작용을 만들 때 strongest baseline이 살아나는지 정리한다.

결과 파일:

- [concat interaction summary](../economic_time/results/concat_interaction_gspc_e3_s3/economic_time_stability_summary.csv)
- [concat interaction details](../economic_time/results/concat_interaction_gspc_e3_s3/economic_time_stability_details.csv)

설정:

- `concat_a:intensity_only`
- `concat_a:position_only`
- `concat_a:indexret_only`
- `concat_a:intensity_indexret`
- `concat_a` (`intensity + position`)
- validation-fitted linear ensemble baselines:
  - `interaction:intensity+position`
  - `interaction:intensity+indexret`

## Main Numbers

- `concat_a:intensity_only IC mean = 0.0066`
- `concat_a:position_only IC mean = 0.0188`
- `concat_a:indexret_only IC mean = 0.0205`
- `concat_a IC mean = 0.0571`
- `concat_a:intensity_indexret IC mean = 0.0592`

linear ensemble baselines:

- `interaction:intensity+position IC mean = 0.0189`
- `interaction:intensity+indexret IC mean = 0.0184`

## Immediate Read

### 1. The strength of `concat_a` is genuinely interactive

단일 채널은 모두 약하다.

- intensity alone: `0.0066`
- position alone: `0.0188`
- index return alone: `0.0205`

그런데 pair model은 강하다.

- intensity + position: `0.0571`
- intensity + index return: `0.0592`

즉 strongest baseline의 성능은 `useful singleton signal 하나`가 아니라, 채널 간 interaction에서 나온다.

### 2. Simple linear ensembling does not explain the gain

validation으로 alpha를 고정한 test-time linear ensemble baseline은 둘 다 `~0.018` 수준에 그쳤다.

- pair model `0.057 ~ 0.059`
- ensemble baseline `0.018 ~ 0.019`

즉 이 성능 이득은 단순히 두 약한 단일 모델을 섞은 결과가 아니다.

### 3. Position is not uniquely necessary

`intensity + index return`이 `intensity + position`과 비슷하거나 약간 더 높다.

즉 현재 strongest baseline의 핵심은 `position`이라는 특정 채널보다, intensity가 다른 market-state channel과 결합될 수 있는 interface일 가능성이 더 크다.

## Best Current Interpretation

이 결과는 아래 문장으로 요약할 수 있다.

`The predictive advantage of concat_a appears to come from an interaction-friendly input interface that lets intensity combine with other market-state channels, rather than from intensity alone or from simple additive combination of weak single-channel predictors.`

## Method Implication

다음 method는 아래 질문을 겨냥해야 한다.

`How do we build an interaction-aware representation that preserves the strong channel coupling exploited by concat_a, while being more structured than raw input concatenation?`

즉 다음 설계는:

- 더 큰 geometry
- 더 복잡한 PE

보다 먼저,

- intensity와 다른 state channel의 coupling
- 그 coupling을 backbone이 쉽게 사용할 수 있는 interface

를 봐야 한다.

## What This Weakens

이 결과는 아래 주장을 약하게 만든다.

- `intensity-only PE`만 잘 만들면 concat을 이길 수 있다
- ordering-aware warping alone이 strongest baseline을 설명한다

## What This Strengthens

이 결과는 아래 방향을 강화한다.

- interaction-aware conditioning
- multi-channel state coupling
- representation/interface redesign around signal interactions
