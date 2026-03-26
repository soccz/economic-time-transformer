# Proxy Spec And Test Table

이 문서는 `15_incremental_identification_plan.md`를 실제 실험 스펙으로 내리는 문서다.

목적:

- 비교할 기존 proxy를 사전에 고정한다.
- 종속변수와 회귀/예측 표의 기본 형태를 잠근다.
- 이후 결과 해석이 변수 선택 post-hoc로 보이지 않도록 한다.

## Locked Data Objects

가능한 한 기존 H1 스크립트의 변수 정의를 재사용한다.

주요 참조:

- [paper_test/h1_search.py](../../paper_test/h1_search.py)
- [paper_test/h1_check12.py](../../paper_test/h1_check12.py)
- [paper_test/h1_v2.py](../../paper_test/h1_v2.py)

## Locked Proxy Definitions

### Core State Variables

1. `position`
   - 정의: `(SPX - MA200) / MA200`
   - 해석: market trend / bear-bull state proxy

2. `intensity`
   - 정의: `RV30`의 252일 rolling percentile rank
   - 해석: volatility-linked activity state

3. `bear`
   - 정의: `1[position < 0]`
   - 해석: negative market trend dummy

4. `high_vol`
   - 정의: `1[intensity > 0.5]`
   - 해석: high-volatility dummy

5. `vix_proxy`
   - 저장소 기존 정의를 따른다
   - 해석: market fear / panic proxy

6. `indexret`
   - 정의: 시장지수 단기 return 계열
   - 현재 finance-first 1차 후보 signal의 한 축

### Interaction Proxies

아래 4개를 비교 테이블에 고정한다.

1. `bear_x_intensity = bear * intensity`
2. `bear_x_vix = bear * vix_proxy`
3. `position_x_intensity = position * intensity`
4. `intensity_x_indexret = intensity * indexret`

여기서 `intensity_x_indexret`이 우리의 primary signal이다.

## Locked Dependent Variables

### Primary

`Y1 = future WML return`

이유:

- 지금 논문의 가장 강한 finance 질문은 momentum crash risk다.
- 따라서 1차 종속변수는 직접적으로 `future WML`이어야 한다.

### Secondary

`Y2 = crash indicator / left-tail indicator of future WML`

이건 1차 분석이 끝난 뒤에만 붙인다.

### Deferred

`winner-leg`, `loser-leg` 분해는 현재 문서의 1차 표에는 넣지 않는다.
이건 `F3` 보강 단계다.

## Locked First-Pass Table

가장 먼저 만들 표는 아래 5개 열로 충분하다.

1. `Model 0`: baseline
   - const only

2. `Model 1`: bear / volatility controls
   - `bear`
   - `intensity`

3. `Model 2`: panic-style benchmark
   - `bear`
   - `intensity`
   - `bear_x_intensity`
   - 또는 저장소 정의상 더 직접적인 `vix_proxy`, `bear_x_vix`

4. `Model 3`: market-state interaction benchmark
   - `position`
   - `intensity`
   - `position_x_intensity`

5. `Model 4`: our signal added
   - `position`
   - `intensity`
   - `position_x_intensity`
   - `indexret`
   - `intensity_x_indexret`

중요:

- `Model 4`는 `our interaction signal`의 incremental value를 보는 표다.
- 이 표가 1차 confirmatory 표다.

## Locked Testing Logic

### Confirmatory Test

다음 질문만 본다.

`Model 4`가 `Model 3`보다 유의하게 낫는가?

판정 기준 예시:

- nested model test
- OOS `R^2` 증가
- forecast loss 감소
- classification metric 증가

### Benchmark Test

추가로 아래도 같이 본다.

`Model 4`가 `panic-style benchmark`보다 낫는가?

즉:

- `bear_x_intensity`
- `bear_x_vix`
- `position_x_intensity`
- `intensity_x_indexret`

중 어떤 interaction이 가장 설명력이 큰지 비교한다.

## Locked Interpretation Rules

### Case A

`intensity_x_indexret`가 기존 interaction proxy를 통제한 뒤에도 유의하다.

해석:

`our signal`은 기존 panic-state 설명을 넘는 추가 식별력을 가진다.

### Case B

`intensity_x_indexret`가 기존 interaction proxy와 같이 넣었을 때 약해지거나 사라진다.

해석:

현재 결과는 기존 panic-state explanation과 정합적이지만, incremental identification까지는 아니다.

### Case C

`position_x_intensity`만으로 대부분 설명되고 `indexret` interaction은 추가 가치가 약하다.

해석:

새 기여는 `indexret`보다 broader interaction regime에 있을 수 있으며, current signal choice를 다시 봐야 한다.

## What Is Explicitly Out Of Scope

현재 1차 표에서 아래는 하지 않는다.

- 복잡한 ML 모델 비교
- `concat_a`, `xip_a`, `FiLM` 성능표
- PE / geometry 결과 재논쟁
- loser-leg decomposition
- portfolio utility

이것들은 모두 2차 보강이다.

## Immediate Implementation Order

1. `Y1 = future WML return`로 first-pass 표 작성
2. `Model 1 -> Model 4` nested 비교
3. `intensity_x_indexret`의 추가 설명력 확인
4. 그 다음에만
   - loser-leg decomposition
   - crash indicator
   - portfolio value

## Practical Note

현재 저장소 기준으로는 아래 조합이 가장 자연스럽다.

- market state:
  - `position`
  - `intensity`
  - `indexret`
  - `vix_proxy`
- core interaction:
  - `bear_x_intensity`
  - `bear_x_vix`
  - `position_x_intensity`
  - `intensity_x_indexret`

즉 구현도 이 변수들을 우선적으로 재사용해야 한다.
