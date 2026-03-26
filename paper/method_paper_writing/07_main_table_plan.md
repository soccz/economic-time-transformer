# Main Table Plan

이 문서는 본문 메인 표를 어떻게 구성할지 미리 고정한 초안이다.

원칙:

- 표는 `무엇이 되는가`보다 `무엇이 안 되는가`를 같이 보여줘야 한다.
- 모델 수를 늘리지 않는다.
- 메인 claim과 직접 연결되는 열만 둔다.

## Main Table Goal

메인 표 하나로 아래 4가지를 동시에 보여준다.

1. `alignment`는 가능하다.
2. `geometry change`도 가능하다.
3. 그러나 `predictive gain`은 따라오지 않는다.
4. architecture interaction과 naive signature 대안도 winner는 아니다.

## Recommended Table Placement

- 본문 `Section 4. Result Chain`의 첫 표
- 제목 예시:
  - `Table 1. Regime-aware representations can align with market activity and alter geometry without consistently improving forecasting.`

## Recommended Rows

메인 표에는 아래 행만 둔다.

1. `static`
2. `concat_a`
3. `learned_tau_rope`
4. `static_tau_rope:global_only`
5. `learned_tau_rope:global_only`
6. `simple_summary_token`
7. `shape_signature_token`

이유:

- `static`, `concat_a`는 기준선
- `learned_tau_rope`는 main method line
- `static_tau_rope:global_only`, `learned_tau_rope:global_only`는 architecture interaction 분리
- `simple_summary_token`, `shape_signature_token`은 A 방향 minimal check

메인 표에서 제외:

- `rule-based tau_rope`
- `window signature global-only`
- intermediate ablations
- any VAE/CVAE variant

이들은 appendix 또는 supplementary로 보낸다.

## Recommended Columns

### Core Columns

1. `Model`
2. `Representation Family`
3. `Interface`
4. `IC`
5. `MAE`
6. `Activity Alignment`
7. `QK Geometry Shift`

### Concrete Mapping

- `Representation Family`
  - `physical time`
  - `input conditioning`
  - `learned temporal warping`
  - `window summary`
  - `window shape signature`

- `Interface`
  - `input concat`
  - `tau-RoPE`
  - `global-only tau-RoPE`
  - `conditioning token`

- `Activity Alignment`
  - `step-intensity spearman`
  - 없는 모델은 `-`

- `QK Geometry Shift`
  - `qk_swap_delta`
  - 해당 진단이 없는 token model은 `-`

## Suggested Table Layout

| Model | Representation Family | Interface | IC | MAE | Activity Alignment | QK Geometry Shift |
| --- | --- | --- | --- | --- | --- | --- |
| `static` | physical time | standard transformer | ... | ... | `-` | `-` |
| `concat_a` | input conditioning | input concat | ... | ... | `-` | `-` |
| `learned_tau_rope` | learned temporal warping | tau-RoPE | ... | ... | ... | ... |
| `static_tau_rope:global_only` | physical time | global-only tau-RoPE | ... | ... | `-` | `-` |
| `learned_tau_rope:global_only` | learned temporal warping | global-only tau-RoPE | ... | ... | ... | ... |
| `simple_summary_token` | window summary | conditioning token | ... | ... | `-` | `-` |
| `shape_signature_token` | window shape signature | conditioning token | ... | ... | `-` | `-` |

## How To Read The Table

본문에서는 표를 아래 순서로 읽는다.

1. `concat_a`가 strongest predictive baseline임을 먼저 인정한다.
2. `learned_tau_rope`는 alignment와 geometry shift를 만들 수 있음을 보여준다.
3. 하지만 같은 모델이 `IC`와 `MAE`에서 consistent winner가 아님을 보여준다.
4. `global_only` 비교로 local suppression이 병목 중 하나였음을 보인다.
5. `shape_signature_token`도 simple summary를 결정적으로 넘지 못해, naive window-level alternative 역시 아직 충분치 않음을 보인다.

## What The Table Must Not Imply

이 표는 아래 주장을 하면 안 된다.

- `learned temporal warping improves forecasting`
- `geometry change explains predictive gain`
- `window shape signatures are already superior`

이 표가 지지하는 최대 해석은 이것이다.

`Regime-aware representations can change alignment and geometry, but predictive utility remains unresolved under current interfaces.`

## Preferred Source Files

표를 채울 때 우선 참조할 결과 파일:

- [GSPC align-only summary](../economic_time/results/method_gspc_s3_e1e3/economic_time_stability_summary.csv)
- [static vs learned global-only](../economic_time/results/method_gspc_staticglobal_smoke/economic_time_stability_summary.csv)
- [shape/simple token](../economic_time/results/method_gspc_sig_s3_e1e3/economic_time_stability_summary.csv)
- [QK output smoke](../economic_time/results/method_gspc_qkout_smoke/economic_time_stability_summary.csv)

## Optional Appendix Table

appendix에는 별도 표를 둔다.

제목 예시:

`Appendix Table A1. Additional ablations and alternative interfaces`

여기에 넣을 것:

- `rule-based tau_rope`
- `window signature global-only`
- geometry-objective variants
- ordinal-loss variants

## One-Sentence Rule

메인 표는 `representation -> geometry -> utility gap`을 보여주는 표여야지, 여러 모델을 나열하는 실험 inventory 표가 되면 안 된다.
