# Explicit Interaction Projection Results

이 문서는 `xip_a` (`explicit interaction projection`)의 첫 결과를 고정한다.

결과 파일:

- [xip Branch A summary](../economic_time/results/xip_branchA_gspc_e3_s3/economic_time_stability_summary.csv)
- [xip Branch A details](../economic_time/results/xip_branchA_gspc_e3_s3/economic_time_stability_details.csv)
- [xip Branch A pooled t-tests](../economic_time/results/xip_branchA_gspc_e3_s3/economic_time_stability_pooled_ttests.csv)

## Compared Models

- `static`
- `concat_a:intensity_indexret`
- `film_a:intensity_indexret`
- `xip_a:intensity_indexret`

핵심 의도:

- strongest interaction pair였던 `intensity + indexret`를 유지
- raw concat의 early shared projection 장점을 보존하면서
- interaction term을 명시적으로 분리한 `xip_a`가 더 나은 interface인지 본다

## Main Result

대표 수치:

- `concat_a:intensity_indexret IC mean = 0.0592`
- `xip_a:intensity_indexret IC mean = 0.0608`
- `film_a:intensity_indexret IC mean = -0.0081`
- `static IC mean = 0.0103`

보조 수치:

- `concat_a:intensity_indexret IC std = 0.0226`
- `xip_a:intensity_indexret IC std = 0.0217`
- `concat_a:intensity_indexret param_count = 45266`
- `xip_a:intensity_indexret param_count = 45418`

즉 `xip_a`는 small extra capacity (`+152` params)로,

- 평균 IC는 `concat_a:intensity_indexret`보다 약간 높고
- seed variance도 약간 낮다.

## Pooled Paired Test

`seed x daily-fold` 기준 pooled paired test:

- `xip_a:intensity_indexret` vs `concat_a:intensity_indexret`
- `t = 0.0953`
- `p = 0.9242`
- `n = 294`

즉 현재 증거로는 `xip_a`가 `concat_a:intensity_indexret`를 통계적으로 이긴다고 말할 수 없다.

## Interaction Usage Diagnostic

summary 기준 추가 진단:

- `xip_h_int_norm_mean = 0.0633`
- `xip_h_int_ratio_mean = 0.0419`
- `xip_pred_delta_mean = 0.000036`
- `xip_ic_off_mean = 0.0589`
- `xip_ic_drop_mean = 0.0019`

해석:

- `h_int`는 완전히 collapse하지는 않았다.
- 하지만 전체 표현에서 차지하는 비중은 아직 작다.
- test-time에 `h_int=0`으로 꺼도 평균 IC 하락이 매우 작다.

즉 현재 `xip_a`의 match는

- `explicit interaction term` 하나가 강하게 작동해서라기보다,
- early state mixing + 작은 interaction 도움

에 더 가깝다.

## Seed-Level Read

- `seed 7`: `concat_a:intensity_indexret = 0.0734`, `xip_a:intensity_indexret = 0.0576`
- `seed 17`: `concat_a:intensity_indexret = 0.0331`, `xip_a:intensity_indexret = 0.0839`
- `seed 27`: `concat_a:intensity_indexret = 0.0711`, `xip_a:intensity_indexret = 0.0409`

읽는 방법:

- `xip_a`는 모든 seed에서 dominant하지는 않다.
- 하지만 `film_a`처럼 붕괴하지 않고, 세 seed 모두 `static`은 안정적으로 넘는다.
- 현재 `Branch A` family에서 처음 나온 positive signal이다.

## Strict Interpretation

지금은 아래처럼 쓰는 것이 가장 안전하다.

- `xip_a`는 `promising candidate`다.
- `concat_a:intensity_indexret`를 현재 기준으로 `clearly beats`한다고 쓰면 안 된다.
- 평균은 약간 높지만, pooled paired test에서는 사실상 `indistinguishable`이다.
- 또한 현재 interaction term 자체의 예측 기여는 아직 작다.

즉 현재 문장은:

`An explicit interaction projection provides the first stable positive signal within Branch A, matching the strongest concat-based interaction baseline under a comparable parameter budget, but not yet exceeding it in pooled paired tests. Its explicit interaction term is active but currently contributes only modestly to predictive utility.`

정도가 적절하다.

## Why This Result Matters

이 결과는 `Branch A` 질문 자체를 살린다.

- `minimal FiLM`은 실패했다.
- 하지만 `explicit interaction projection`은 baseline 근처까지 간다.
- 따라서 문제는 `structured conditioning` 전체가 틀린 것이 아니라,
  `which interaction interface preserves useful early mixing`의 문제일 가능성이 커진다.

## Immediate Next Step

다음으로 필요한 것은 하나다.

- `xip_a`가 현재 왜 `match`는 되는데 `win`은 못 하는지 분해하는 것
- interaction term `h_int`를 더 task-relevant하게 쓰게 만드는 것

즉 다음 질문은:

- interaction term `h_int`가 왜 살아 있는데도 utility 기여가 작은가
- `concat_a`와 다른 generalization 이점이 있는가
- seed/market을 넓혀도 `match`가 유지되는가
