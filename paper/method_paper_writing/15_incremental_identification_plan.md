# Incremental Identification Plan

이 문서는 현재 finance-first 분기에서 가장 현실적인 다음 질문을 고정한다.

핵심 질문:

`우리의 state interaction signal이 기존 bear/volatility/panic proxy를 넘어서는 추가 식별력을 가지는가`

지금 단계에서 중요한 점:

- 현재 데이터는 `conditional pattern`은 보여준다.
- 그러나 아직 `새 메커니즘 식별`까지는 아니다.
- 따라서 다음 실험의 목적은 새 모델 성능 경쟁이 아니라 `existing explanation을 넘어서는가`를 검증하는 것이다.

## Locked Finance Question

현재 finance 논문의 메인 질문은 아래처럼 좁힌다.

`Does the interaction between volatility-linked intensity and market-state variables identify momentum-crash risk beyond existing panic-state proxies?`

보다 직접적으로 쓰면:

`Bear/volatile regime에서 모멘텀 크래시를 설명하는 기존 proxy보다, intensity × state interaction이 추가 설명력을 제공하는가?`

## Why This Branch

지금까지의 결과는 아래를 보여준다.

1. `intensity`는 중요하다.
2. 그러나 `intensity`는 standalone strong predictor가 아니라 `interaction signal`에 가깝다.
3. strongest ML baseline의 힘도 `channel interaction`에서 나오는 것으로 보인다.
4. 따라서 finance 쪽에서도 단순 `bear dummy`나 `volatility level`보다 `interaction-based state identification`이 더 강한지 보는 것이 자연스럽다.

## Locked Hypotheses

### Primary

`F1`
`intensity × state interaction`은 기존 bear/volatility/panic proxy를 통제한 뒤에도 momentum-crash risk에 대해 추가 예측력을 가진다.

### Secondary

`F2`
위 추가 설명력은 특히 `bear/volatile` 구간에서 더 강하게 나타난다.

### Follow-up

`F3`
위 추가 설명력은 winner-leg보다는 loser-leg rebound 쪽과 더 강하게 연결된다.

현재 단계에서 `F1`만 confirmatory로 잡고, `F2`, `F3`는 후속 보강으로 둔다.

## Comparison Targets

다음 비교군을 반드시 같은 표에 둔다.

1. `bear dummy`
2. `realized volatility level`
3. `panic-style interaction proxy`
   - 예: `negative market state × high volatility`
4. `our interaction signal`
   - 기본 후보: `intensity × indexret`

중요:

- 지금 목적은 새 ML 구조 비교가 아니다.
- `concat_a`, `xip_a`, `FiLM`은 여기서 메인 객체가 아니라 `state interaction signal`을 어떻게 추출/사용했는지 보여주는 보조 도구다.

## Minimum Test Design

가장 먼저 할 실험은 단순해야 한다.

1. 종속변수:
   - future `WML` return
   - 또는 crash indicator / left-tail outcome

2. 기본 통제:
   - 기존 bear proxy
   - volatility proxy
   - panic-style proxy

3. 추가 변수:
   - `intensity × indexret`

4. 검정 질문:
   - 기존 proxy만 쓴 모델 대비
   - `intensity × indexret`를 추가했을 때
   - OOS 예측력 또는 state classification이 개선되는가

## Success Criteria

이 branch에서 먼저 필요한 성공 기준은 하나다.

`C-F1`
기존 panic-state proxy를 통제한 뒤에도 `our interaction signal`이 추가 설명력을 보여야 한다.

구체적 판정 예시:

- OOS `R^2` 또는 classification metric이 증가
- nested model test에서 유의
- portfolio sort에서 추가 분리력 존재

이 기준이 안 서면, 현재 finance-first 주장은 아직 pattern report 수준에 머문다.

## What Does Not Count

아래는 아직 식별 기여로 세면 안 된다.

- 단순히 `bear/volatile` 구간에서 패턴이 보인다
- `intensity`가 중요해 보인다
- ML 모델이 proxy를 잘 쓴다
- 특정 구간에서만 post-hoc하게 강한 결과가 나온다

즉 `기존 설명을 넘는 incremental value`가 없으면 top finance claim으로 쓰면 안 된다.

## Writing Implication

이 branch가 성공하면 논문 문장은 아래처럼 바뀐다.

`We show that momentum crash risk is not captured by volatility or bear-state proxies alone; an interaction-based state signal provides incremental identification of crash-prone regimes.`

이 branch가 실패하면 정직한 문장은 아래다.

`Our state interaction results are consistent with the existing panic-state explanation, but do not yet establish incremental identification beyond prior proxies.`

## Immediate Next Step

다음 액션은 하나다.

`intensity × indexret`를 기존 bear/volatility/panic proxy와 같은 regression / forecasting table에 올려 incremental value를 먼저 본다.

그 다음에만:

1. loser-leg decomposition
2. portfolio economic value
3. crash avoidance strategy

순으로 확장한다.
