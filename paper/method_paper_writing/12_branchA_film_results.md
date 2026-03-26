# Branch A FiLM Results

이 문서는 `Branch A = interaction-friendly interface`의 최소 구현으로 넣은 `FiLM` 결과를 고정한다.

결과 파일:

- [FiLM Branch A summary](../economic_time/results/film_branchA_gspc_e3_s3/economic_time_stability_summary.csv)
- [FiLM Branch A t-tests](../economic_time/results/film_branchA_gspc_e3_s3/economic_time_stability_ttests.csv)

## Compared Models

- `static`
- `concat_a`
- `concat_a:intensity_indexret`
- `film_a:intensity_indexret`

핵심 의도:

- strongest empirical interaction pair였던 `intensity + indexret`를 유지한 채
- raw input concatenation 대신
- `FiLM` modulation이 더 나은 interface인지 본다.

## Main Result

대표 수치:

- `concat_a IC mean = 0.0571`
- `concat_a:intensity_indexret IC mean = 0.0592`
- `film_a:intensity_indexret IC mean = -0.0081`
- `static IC mean = 0.0103`

보조 수치:

- `concat_a:param_count = 45266`
- `film_a:intensity_indexret:param_count = 45330`

## Seed-Level Read

- `seed 7`: `film_a:intensity_indexret = -0.0789`, `concat_a:intensity_indexret = 0.0734`
- `seed 17`: `film_a:intensity_indexret = 0.0760`, `concat_a:intensity_indexret = 0.0331`
- `seed 27`: `film_a:intensity_indexret = -0.0212`, `concat_a:intensity_indexret = 0.0711`

즉 `FiLM`은 한 seed에서 강하게 나오지만, 반복적으로는 매우 불안정하다.

## Interpretation

지금 증거로는 다음이 더 맞다.

- `interaction-friendly interface`라는 진단은 여전히 유효하다.
- 하지만 최소 `FiLM` 구현은 아직 `concat_a`보다 나은 해결책이 아니다.
- 따라서 `Branch A`는 아직 method winner가 아니라, 탐색 중인 interface family다.

## What This Means For The Paper

지금은 이렇게 쓰는 것이 안전하다.

`We find that simple concatenation remains a very strong interaction interface. A minimal FiLM-based alternative does not yet provide a stable improvement, suggesting that the core issue is not merely adding structure, but finding an interface that preserves useful channel interactions without introducing optimization instability.`

지금 쓰면 안 되는 문장:

- `FiLM solves the interaction problem`
- `structured conditioning is already better than concat`
- `Branch A is the final method`
