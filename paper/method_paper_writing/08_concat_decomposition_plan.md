# Concat Decomposition Plan

이 문서는 `concat_a`의 강점을 분해하기 위한 최소 실험 계획이다.

목적:

- strongest baseline이 실제로 어떤 signal을 쓰는지 확인한다.
- 그 결과를 다음 `PE / signature / interface` 재설계의 힌트로 쓴다.
- 결과 해석을 실험 전에 고정해, post-hoc 해석을 줄인다.

## Main Question

`concat_a`는 왜 강한가?

더 구체적으로는 아래를 묻는다.

1. intensity signal 자체가 핵심인가?
2. intensity의 level만 중요한가?
3. intensity의 temporal ordering/shape가 중요한가?
4. 비선형 intensity encoding이 필요한가?

## Minimal Variants

실험은 아래 variant만 먼저 본다.

1. `concat_a`
2. `concat_a:no_intensity`
3. `concat_a:intensity_only`
4. `concat_a:binned_intensity_only`
5. `concat_a:shuffled_intensity`

필요하면 그 다음에만 추가:

6. `concat_a:window_intensity_summary`

## Why These Variants

### `concat_a:no_intensity`

role:

- intensity signal 제거
- strongest baseline이 intensity에 얼마나 의존하는지 확인

### `concat_a:intensity_only`

role:

- 나머지 context 채널을 제거
- intensity 하나만으로 어느 정도 성능이 나오는지 확인

### `concat_a:binned_intensity_only`

role:

- raw scalar intensity보다 discrete/nonlinear encoding이 더 나은지 확인
- 다음 `PE embedding` 설계와 직접 연결됨

### `concat_a:shuffled_intensity`

role:

- intensity 값의 분포는 유지하고, 각 window 내부의 시간 순서만 파괴
- ordering/shape의 중요도를 직접 검정

중요:

- shuffle은 반드시 `각 window 내부`에서만 수행한다.
- 가능하면 1회가 아니라 2~3회 shuffle 평균으로 본다.

## Pre-Locked Interpretation Rules

### Case A. `shuffled_intensity ≈ intensity_only`

의미:

- intensity의 `level` 또는 marginal distribution이 핵심이다.
- 시간 순서 정보는 크게 중요하지 않을 수 있다.

다음 설계 방향:

- pointwise scalar encoding
- binned embedding
- simple feature injection

불리해지는 주장:

- `time-structure-aware PE`가 핵심이라는 주장

### Case B. `shuffled_intensity << intensity_only`

의미:

- intensity의 temporal ordering/shape가 실제로 중요하다.
- 같은 평균 intensity라도 시간적 배치가 predictive signal을 만든다.

다음 설계 방향:

- PE / profile encoding
- window-level signature
- shape-aware conditioning

유리해지는 주장:

- `economic-time profile`이나 `temporal shape signature`가 필요하다는 주장

### Case C. `binned_intensity_only > intensity_only`

의미:

- raw linear scalar보다 비선형/discrete encoding이 더 적절하다.
- 현재 `cycle_pe` linear intensity projection failure는 구현 문제일 가능성이 커진다.

다음 설계 방향:

- embedding-based PE
- bin-aware signature

### Case D. `intensity_only << concat_a`

의미:

- intensity 하나로는 부족하다.
- 다른 state channel이나 interaction이 중요하다.

다음 설계 방향:

- multi-channel conditioning
- cross-feature interaction analysis

## Primary Decision Rule

이 실험의 첫 번째 목적은 `concat_a`를 더 잘 이해하는 것이다.
두 번째 목적은 `다음 method가 어떤 interface를 써야 하는지` 정하는 것이다.

따라서 이 문서의 핵심 판정은 아래 한 줄이다.

`concat_a의 이점이 level signal에서 오는지, temporal ordering에서 오는지 먼저 가른다.`

## What This Plan Is Not

이 실험은 아직 새 method 제안이 아니다.

- `concat_a`를 이기기 위한 최종 해법은 아니다.
- `PE superiority`를 다시 주장하는 실험도 아니다.

이 실험은 다음 method를 위한 `diagnostic map`이다.

## Expected Output

실험이 끝나면 아래 질문에 답할 수 있어야 한다.

1. `concat_a`는 intensity를 얼마나 쓰는가?
2. intensity의 `크기`와 `순서` 중 무엇이 더 중요한가?
3. intensity는 raw scalar로 충분한가, 아니면 discrete/nonlinear encoding이 더 맞는가?

## One-Sentence Summary

`concat_a decomposition is not only a baseline analysis; it is a diagnostic experiment that tells us whether the next representation should focus on signal level, temporal ordering, or nonlinear encoding.`
