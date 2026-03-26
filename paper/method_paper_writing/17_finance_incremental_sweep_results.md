# Finance Incremental Sweep Results

이 문서는 2026-03-12 기준 `finance-first incremental identification` 1차 실행 결과를 고정한다.

핵심 질문:

`intensity × indexret interaction`이 기존 bear/volatility/panic proxy를 넘는 추가 식별력을 가지는가?

실행 스크립트:

- [paper_test/finance_incremental_identification.py](../../paper_test/finance_incremental_identification.py)
- [paper_test/finance_incremental_sweep.py](../../paper_test/finance_incremental_sweep.py)

결과 폴더:

- [single-spec first pass](../economic_time/results/finance_incremental_identification/)
- [spec sweep summary](../economic_time/results/finance_incremental_sweep/finance_incremental_sweep_summary.csv)

## What Was Tested

아래 4개 조합을 같은 형식으로 비교했다.

1. `future WML 1d` + `indexret = ret1`
2. `future WML 1d` + `indexret = ret5`
3. `future WML 5d` + `indexret = ret1`
4. `future WML 5d` + `indexret = ret5`

여기서:

- `ret1`: 1일 log index return
- `ret5`: 5일 평균 log index return

공통 비교표는 [16_proxy_spec_and_test_table.md](./16_proxy_spec_and_test_table.md) 의 `Model 0` ~ `Model 4`를 따른다.

## Main Read

가장 강한 스펙은:

`future WML 5d` + `indexret = ret5`

즉 현재 신호는 `당일 시장수익률`보다,

`최근 며칠의 market move × intensity`

형태에서 더 잘 살아난다.

## Strongest Specification

주요 파일:

- [h5_ret5 tests](../economic_time/results/finance_incremental_sweep/h5_ret5/finance_incremental_tests.csv)
- [h5_ret5 coefficients](../economic_time/results/finance_incremental_sweep/h5_ret5/finance_incremental_coefficients.csv)
- [h5_ret5 oos summary](../economic_time/results/finance_incremental_sweep/h5_ret5/finance_incremental_oos_summary.csv)
- [h5_ret5 oos tests](../economic_time/results/finance_incremental_sweep/h5_ret5/finance_incremental_oos_tests.csv)

핵심 수치:

- `Model 4 > Model 3` joint test: `F = 14.335`, `p = 6.09e-07`
- `intensity_x_indexret` in `Model 4`: `coef = 0.8884`, `p = 0.0098`
- `indexret` in `Model 4`: `coef = -1.1166`, `p = 5.26e-06`
- `Model 4 OOS R^2 vs Model 0 = 0.00462`
- `Model 3 OOS R^2 vs Model 0 = -0.00036`

의미:

- in-sample 기준으로는 `our interaction signal`이 살아 있다.
- 특히 `ret5`와 `future WML 5d` 조합에서만 아니라, `ret5` 또는 `horizon=5` 쪽에서 전반적으로 더 강해진다.
- 따라서 현재 읽기는 `very short-run daily panic proxy`보다 `multi-day market-state interaction`에 가깝다.

## What Is Still Missing

중요하게도 아래는 아직 안 섰다.

- `Model 4`의 OOS abs loss가 `Model 3`보다 유의하게 낮다: `p = 0.204`
- `Model 4`의 OOS sq loss가 `Model 3`보다 유의하게 낮다: `p = 0.322`
- `Model 4`가 benchmark (`Model 2`)를 OOS에서 유의하게 이긴다: abs-loss 기준 `p = 0.070`

즉 현재는:

- `signal exists`: 말할 수 있다
- `confirmatory forecasting win is secured`: 아직 아니다

## Interpretation Update

현재 결과는 기존 `Case A/B/C` 중 완전한 `Case A`는 아니다.

가장 정직한 문장은 아래에 가깝다.

`The interaction signal is not strongest at the one-day panic proxy level. It becomes materially stronger when market returns and momentum outcomes are aggregated over a short multi-day window, suggesting that crash-prone regimes may be identified by broader market-state interaction rather than by a purely daily panic measure.`

## Practical Decision

지금 기준으로 메인 finance spec의 1차 후보는 아래로 옮긴다.

- target:
  - `future WML 5d`
- signal:
  - `intensity × ret5`

즉 다음 표와 반복 검정은 이 스펙을 우선한다.

## Immediate Next Step

다음 액션은 하나다.

`h5_ret5` 스펙을 다른 시장/표본 분할/secondary tail outcome에 복제해서 repeatability를 확인한다.

우선순위:

1. `IXIC` 반복
2. crash / left-tail indicator
3. 기간 분할 robustness

그 전까지는:

- `탑 finance claim 확정`
- `incremental identification 이미 증명`

처럼 쓰면 안 된다.
