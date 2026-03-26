# Decision Rules

이 파일은 앞으로 더 팔지, 여기서 정리할지 결정하는 기준이다.

## Stop Condition For Current Method Line

아래 둘이 동시에 성립하면, 현재 method line은 여기서 정리하는 쪽이 맞다.

1. `learned_tau_rope`가 `static_tau_rope`도 반복적으로 못 이긴다.
2. `window-level signature`도 `simple summary`를 안정적으로 못 이긴다.

현재 상태는 사실상 이 조건에 매우 가깝다.

## Continue Condition

계속 파려면 아래 중 하나는 나와야 한다.

1. `shape_signature_token > simple_summary_token`이 반복 seed에서 안정적으로 성립
2. `learned_tau_rope > static_tau_rope`가 global-only 기준에서 성립
3. 새로운 설계가 `geometry -> utility` 연결을 직접 개선

## Current Recommendation

현재 저장소 기준 추천은:

1. 지금까지의 method 결과를 `failure analysis + structural insight`로 정리
2. applied fallback 논문은 별도로 유지
3. 새 method를 판다면, 기존 line의 미세 수정이 아니라 더 큰 representation redesign로 간다

## One-Sentence Status

`방향은 살아 있지만, 현재 구현 family는 아직 predictive winner가 아니다.`

