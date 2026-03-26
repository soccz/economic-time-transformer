# Locked Question

## Main Question

현재 method paper의 핵심 질문은 이것이다.

`시장 상태를 반영한 regime-aware representation이 Transformer의 attention geometry를 실제로 바꾸는가, 그리고 그 변화가 예측 이득으로 이어지는가?`

이 질문은 두 단계로 나뉜다.

1. `representation -> geometry`
2. `geometry -> predictive utility`

지금까지의 실험은 1단계는 일부 성립하고, 2단계는 아직 성립하지 않는다는 쪽으로 기울어 있다.

## What This Paper Is Not

이 논문은 현재 기준으로 다음이 아니다.

- pure alpha discovery paper
- uncertainty / VAE paper
- TCN/CNN architecture paper
- generic adaptive PE paper

## Current Working Claim

지금 당장 가장 정직하게 쓸 수 있는 claim 후보는 이것이다.

`Regime-aware temporal representations can measurably alter Transformer attention geometry in financial forecasting, but such geometry change does not automatically translate into predictive gain.`

이 문장은 아직 `최종 headline`은 아니다.
하지만 현재 증거 수준과 가장 잘 맞는다.

## Minimal Paper Contribution If We Stop Here

만약 지금 단계에서 method path를 더 늘리지 않고 정리한다면, 최소 기여는 아래 세 가지다.

1. `learned τ`가 economic activity와 alignment를 가질 수 있음을 보였다.
2. `learned τ`가 QK geometry를 실제로 바꿀 수 있음을 보였다.
3. geometry change와 predictive gain 사이에 비자명한 간극이 있음을 보였다.

