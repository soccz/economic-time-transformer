# Explicit Interaction Projection Plan

이 문서는 `concat_a`의 강점을 대체할 다음 `Branch A` 후보를 고정한다.

핵심 질문:

`raw concat이 강한 이유가 early shared projection에 있다면, 그 장점을 보존하면서 더 명시적이고 해석 가능한 interaction projection을 만들 수 있는가?`

## Why This Candidate Exists

현재까지의 진단은 아래로 수렴한다.

- `PE`는 현재 구현에서 `concat_a`를 못 이겼다.
- 최소 `FiLM`도 `concat_a`를 못 이겼다.
- strongest baseline의 핵심은 `single strong signal`이 아니라 `interaction-friendly input interface`에 가깝다.

특히 [10_concat_interaction_results.md](./10_concat_interaction_results.md) 기준으로:

- `concat_a:intensity_only = 0.0066`
- `concat_a:indexret_only = 0.0205`
- `concat_a:intensity_indexret = 0.0592`

즉 단일 채널을 early projection에 넣는 것보다, 두 채널을 함께 넣을 때 생기는 효과가 훨씬 크다.

## Correct Reading Of `concat_a`

중요한 점:

`concat_a`의 input projection은 `명시적 곱셈 interaction`을 하지는 않는다.

선형층 자체는 결국

`W_x x_t + W_s s_t + b`

형태다.

하지만 `concat_a`는 asset/state 정보를 매우 이른 단계에서 같은 latent channel 안에 섞는다.
즉 현재 baseline의 강점은

- explicit multiplication

보다는,

- early shared projection / early latent mixing

에 더 가깝다.

따라서 다음 method는 이 장점을 잃지 않으면서, interaction term을 더 명시적으로 드러내야 한다.

## Minimal Candidate

가장 작은 후보는 아래 구조다.

```text
h_x   = W_x x_t
h_s   = W_s s_t
h_int = (U_x x_t) ⊙ (U_s s_t)
h_t   = h_x + h_s + h_int
```

이후에는:

- static PE
- 기본 Transformer

만 사용한다.

즉 이 후보는:

- `concat_a`의 early mixing 장점을 유지하고
- interaction을 `h_int`로 명시적으로 분리한다.

## Why This Is Better Than Immediate Bilinear Full Model

처음부터 full bilinear tensor로 가면:

- 파라미터가 급증하고
- baseline 비교가 어려워지고
- 실패했을 때 해석이 모호해진다.

따라서 첫 버전은

- low-rank
- element-wise product
- gated interaction

수준의 최소 explicit interaction projection으로 제한한다.

## Fair Comparison Rule

비교는 아래 셋으로만 한다.

1. `concat_a:intensity_indexret`
2. `film_a:intensity_indexret`
3. `xip_a:intensity_indexret`  (`explicit interaction projection` 후보)

여기서 `xip_a`는 임시 이름이다.

공정 비교 조건:

- 같은 backbone
- 같은 PE (`static`)
- 같은 state pair (`intensity + indexret`)
- 비슷한 parameter budget
- 같은 seeds / epochs

## What Must Be Logged

이 후보는 성능만 보면 안 된다.

반드시 같이 볼 것:

- `IC / MAE`
- parameter count
- `h_int` norm mean
- `h_int` variance across samples
- `h_int`와 output 변화의 상관

즉 이 후보의 장점은 `성능`뿐 아니라 `interaction term을 직접 꺼내 보여줄 수 있음`에 있다.

## Success Criterion

최소 성공 기준은 하나다.

`Perf(xip_a:intensity_indexret) > Perf(concat_a:intensity_indexret)`

보조 성공 기준:

- seed variance가 더 낮다
- `h_int`가 collapse하지 않는다
- `h_int`가 실제 예측 변화와 연관된다

## Failure Interpretation

이 후보도 실패하면 해석은 명확하다.

- raw concat의 강점은 단순 explicit interaction term으로 복제되지 않는다
- 즉 문제는 `interaction term 유무`가 아니라, `raw shared projection이 주는 optimization/interface advantage`일 수 있다

그 경우 다음 질문은

`why is raw shared projection so effective in financial state conditioning?`

로 더 좁혀진다.

## One-Sentence Lock

`The next Branch A candidate should preserve concat_a's early shared projection advantage while making channel interaction explicit and measurable.`
