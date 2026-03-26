# Concat Decomposition Results

이 문서는 `concat_a` 분해 실험의 첫 결과만 요약한다.

결과 파일:

- [concat decomposition summary](../economic_time/results/concat_decomp_gspc_e3_s3/economic_time_stability_summary.csv)

설정:

- `static`
- `concat_a`
- `concat_a:no_intensity`
- `concat_a:intensity_only`
- `concat_a:binned_intensity_only`
- `concat_a:shuffled_intensity`

모두 `GSPC`, `2022-2024`, `3 seeds x 3 epochs` 기준이다.

## Main Numbers

- `concat_a IC mean = 0.0571`
- `concat_a:no_intensity IC mean = 0.0026`
- `concat_a:intensity_only IC mean = 0.0066`
- `concat_a:binned_intensity_only IC mean = 0.0070`
- `concat_a:shuffled_intensity IC mean = 0.0187`
- `static IC mean = 0.0103`

## Immediate Read

### 1. Intensity matters

`concat_a:no_intensity`가 거의 0에 가까워졌다.

즉 full `concat_a`의 예측 이점은 intensity signal과 강하게 연결되어 있다.

### 2. Intensity alone is not enough

`concat_a:intensity_only`는 `static`과 비슷하거나 조금 나은 수준에 그쳤다.

즉 strongest baseline의 힘은 `intensity 하나`보다, intensity와 다른 신호의 결합 또는 full input interface에서 나온다.

### 3. Binning does not materially help in concat space

`binned_intensity_only`는 `intensity_only`와 사실상 비슷했다.

즉 concat 경로에서는 raw scalar intensity를 discrete로 바꿔도 큰 차이가 없었다.

### 4. Ordering does not look like the main driver, at least in this minimal path

`shuffled_intensity`가 `intensity_only`보다 오히려 높았다.

즉 현재 결과만 보면, intensity의 fine-grained temporal ordering이 이 minimal concat path의 핵심 signal이라고 보기 어렵다.

하지만 주의:

- 이건 `intensity-only` path 안에서의 비교다.
- `shuffled_intensity`는 단일 shuffle realization 기반이다.
- 따라서 강한 일반 법칙으로 쓰면 안 되고, 현재는 `ordering is not obviously the primary driver` 정도로만 써야 한다.

## Current Best Interpretation

이 결과는 아래 문장으로 요약된다.

`The predictive strength of concat_a appears to depend more on the presence of intensity-linked signal and its interaction with the broader input interface than on the fine-grained temporal ordering of intensity alone.`

## Method Implication

다음 method 설계에 주는 힌트는 명확하다.

1. `intensity signal`은 반드시 잡아야 한다.
2. 하지만 `intensity-only pointwise path`로는 부족하다.
3. 따라서 다음 설계는 `더 복잡한 geometry`를 먼저 만들기보다, `intensity를 다른 useful signals와 어떻게 결합할지`를 먼저 봐야 한다.

## What This Does Not Yet Prove

이 결과만으로 아래를 단정하면 안 된다.

- `ordering never matters`
- `window-level shape is useless`
- `PE/profile methods are unnecessary`

현재 이 결과가 말해주는 최대치는 이것이다.

`In the current concat interface, level/distribution information seems more immediately useful than intensity ordering alone.`
