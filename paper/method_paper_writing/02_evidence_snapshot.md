# Evidence Snapshot

이 파일은 지금까지의 핵심 결과만 남긴 요약본이다.

## A. Align-Only Learned Tau

결과 파일:

- [GSPC align-only](../economic_time/results/method_gspc_s3_e1e3/economic_time_stability_summary.csv)
- [IXIC align-only](../economic_time/results/method_ixic_s3_e1e3/economic_time_stability_summary.csv)

핵심:

- `learned_tau_rope`는 `step-intensity` 정렬은 잡힌다.
- 하지만 `qk/attn swap delta`는 거의 0에 가깝다.
- `concat_a`가 반복적으로 더 강하다.

대표 수치:

- `GSPC 3 epoch`: `concat_a IC=0.0571`, `learned_tau_rope IC=0.0354`
- `IXIC 3 epoch`: `concat_a IC=0.0515`, `learned_tau_rope IC=0.0357`

해석:

`alignment`만으로는 geometry도, predictive gain도 충분하지 않았다.

## B. Strong Geometry Objective

결과 파일:

- [GSPC geometry objective](../economic_time/results/method_gspc_geom_s3_e1e3/economic_time_stability_summary.csv)

핵심:

- geometry objective를 강하게 걸면 `qk_swap_delta`는 크게 증가한다.
- 하지만 alignment와 IC가 붕괴한다.

대표 수치:

- `learned_tau_rope 3 epoch`: `qk_swap_delta=0.022263`, `IC=-0.0072`, `step-intensity spearman=0.124`

해석:

`geometry를 강제로 키우는 것`과 `예측에 의미 있는 representation`은 다르다.

## C. QK Ordinal Loss

결과 파일:

- [QK output smoke](../economic_time/results/method_gspc_qkout_smoke/economic_time_stability_summary.csv)
- [QK output train history](../economic_time/results/method_gspc_qkout_smoke/economic_time_stability_train_history.csv)

핵심:

- `qk_ord_rate`는 거의 `1.0`까지 올라간다.
- `τ -> QK geometry` 경로는 분명히 존재한다.
- 하지만 `IC`는 여전히 거의 0이다.

대표 수치:

- `qk_ord_rate ≈ 0.992`
- `qk_swap_delta ≈ 2e-05`
- `IC ≈ 0.0002`

해석:

`QK geometry change`는 확인됐지만, 이것이 예측 이득으로 연결되지는 않았다.

## D. Global-Only Ablation

결과 파일:

- [static vs learned global-only](../economic_time/results/method_gspc_staticglobal_smoke/economic_time_stability_summary.csv)

핵심:

- local branch 제거는 분명 의미가 있다.
- 하지만 같은 global-only 조건에서 `static physical time`이 `learned τ`보다 낫다.

대표 수치:

- `concat_a IC=0.0713`
- `static_tau_rope:global_only IC=0.0346`
- `learned_tau_rope:global_only IC=0.0239`

해석:

`local branch suppression`은 병목 중 하나였지만, 그것만 해결해도 `learned τ`가 winner가 되지는 않았다.

## E. Window Signature Token (Direction A)

결과 파일:

- [shape/simple token](../economic_time/results/method_gspc_sig_s3_e1e3/economic_time_stability_summary.csv)
- [shape/simple token global-only](../economic_time/results/method_gspc_sig_global_s3_e1e3/economic_time_stability_summary.csv)

핵심:

- `shape_signature_token > simple_summary_token` 방향은 약하게 보일 때가 있다.
- 하지만 3 epoch 평균 기준 둘 다 `concat_a`를 전혀 못 넘는다.
- global-only로 바꿔도 winner는 아니다.

대표 수치:

- `3 epoch`: `shape_signature_token IC=-0.0226`, `simple_summary_token IC=-0.0267`
- `3 epoch global-only`: `shape_signature_token IC≈0.0000`, `simple_summary_token IC=-0.0144`
- `concat_a IC=0.0571`

해석:

naive한 window-level signature도 현재는 충분한 predictive interface가 아니다.

## F. Applied Fallback Confirmatory Result

결과 파일:

- [confirmatory 2020-2024](../economic_time/results/confirmatory_2020_2024/confirmatory_hypothesis_tests.csv)

핵심:

- applied fallback path는 confirmatory evidence가 이미 있다.

대표 수치:

- `H1 high-vol IC`: `mean ΔIC=0.0387`, `p=0.0595`
- `H2 high-vol MAE`: `mean ΔMAE=0.000092`, `p=0.0019`

의미:

method path가 아직 안 서더라도, 논문 전체가 완전히 빈 상태는 아니다.

## G. Concat Intensity Ablation

결과 파일:

- [concat no-intensity summary](../economic_time/results/concat_no_intensity_gspc_e3_s3/economic_time_stability_summary.csv)
- [concat no-intensity t-tests](../economic_time/results/concat_no_intensity_gspc_e3_s3/economic_time_stability_ttests.csv)

핵심:

- `concat_a`에서 intensity 채널만 제거하면 평균 `IC`가 크게 떨어진다.
- 따라서 strongest baseline의 장점은 단순한 `input interface` 자체보다는, 그 경로를 통해 전달되는 intensity-linked signal에 강하게 의존한다.
- 다만 모든 seed에서 완전히 같은 패턴은 아니므로, `concat_a`의 힘을 intensity 하나로 완전히 환원해서 쓰면 안 된다.

대표 수치:

- `concat_a IC mean=0.0571`
- `concat_a:no_intensity IC mean=0.0249`
- `static IC mean=0.0103`

seed별 paired t-test:

- `seed 7`: `concat_a:no_intensity < concat_a`, `p=0.0022`
- `seed 17`: `concat_a:no_intensity < concat_a`, `p=0.0149`
- `seed 27`: 차이 비유의

해석:

다음 method는 `새 geometry를 만드는 것`만으로는 부족하고, 현재 strongest baseline이 잘 활용하는 simple but useful signal을 먼저 설명하거나 포착해야 한다.

## H. Fair Intensity-Only PE vs Concat Comparison

결과 파일:

- [cycle_pe vs concat intensity-only summary](../economic_time/results/cyclepe_vs_concat_intensity_gspc_e3_s3/economic_time_stability_summary.csv)
- [cycle_pe vs concat intensity-only t-tests](../economic_time/results/cyclepe_vs_concat_intensity_gspc_e3_s3/economic_time_stability_ttests.csv)

설정:

- `concat_a:intensity_only`
- `cycle_pe:intensity_only`
- 둘 다 `context_dim=1`
- 추가 파라미터 수를 맞추기 위해 `cycle_pe:intensity_only`는 bias-free intensity projection 사용

대표 수치:

- `concat_a:intensity_only IC mean=0.0066`
- `cycle_pe:intensity_only IC mean=-0.0244`
- 두 모델의 `param_count = 45234`

seed별 결과:

- `seed 7`: `concat_a:intensity_only 0.0141 > cycle_pe:intensity_only -0.0331`
- `seed 17`: `concat_a:intensity_only -0.0518 < cycle_pe:intensity_only -0.0342`
- `seed 27`: `concat_a:intensity_only 0.0575 > cycle_pe:intensity_only -0.0060`

해석:

같은 intensity signal과 거의 같은 model capacity를 준 조건에서도, 현재 `cycle_pe`는 `concat_a`보다 낫지 않았다. 따라서 현재 저장소 증거로는 `PE injection superiority`를 주장할 수 없다.

## I. PE Encoding Function Sanity Check

결과 파일:

- [cycle_pe embed vs linear summary](../economic_time/results/cyclepe_embed_vs_linear_gspc_e3_s3/economic_time_stability_summary.csv)
- [cycle_pe embed vs linear t-tests](../economic_time/results/cyclepe_embed_vs_linear_gspc_e3_s3/economic_time_stability_ttests.csv)

설정:

- `cycle_pe:intensity_only`: bias-free linear intensity projection
- `cycle_pe:intensity_embed`: binned intensity embedding
- 둘 다 intensity-only context 사용

대표 수치:

- `concat_a:intensity_only IC mean=0.0066`
- `cycle_pe:intensity_only IC mean=-0.0244`
- `cycle_pe:intensity_embed IC mean=0.0040`

보조 수치:

- `cycle_pe:intensity_only param_count=45234`
- `cycle_pe:intensity_embed param_count=46226`

해석:

- `embed`는 `linear`보다 분명히 낫다.
- 즉 현재 `cycle_pe` 실패는 `PE idea` 전체보다 `PE encoding function`의 문제를 일부 포함한다.
- 그러나 `embed`로 바꿔도 평균적으로 `concat_a:intensity_only`를 넘지 못했다.

## J. Branch A FiLM Check

결과 파일:

- [FiLM Branch A summary](../economic_time/results/film_branchA_gspc_e3_s3/economic_time_stability_summary.csv)
- [FiLM Branch A t-tests](../economic_time/results/film_branchA_gspc_e3_s3/economic_time_stability_ttests.csv)

설정:

- strongest interaction pair였던 `intensity + indexret`를 유지
- `concat_a:intensity_indexret`와 `film_a:intensity_indexret`를 같은 조건으로 비교

대표 수치:

- `concat_a:intensity_indexret IC mean=0.0592`
- `film_a:intensity_indexret IC mean=-0.0081`
- `static IC mean=0.0103`

해석:

- `interaction-friendly interface`가 중요하다는 진단은 유지된다.
- 그러나 최소 `FiLM` 구현은 아직 raw concat보다 더 좋은 해결책이 아니다.
- 따라서 `Branch A`는 현재 `method winner`가 아니라, 아직 불안정한 대안 family다.

## K. Explicit Interaction Projection Check

결과 파일:

- [xip Branch A summary](../economic_time/results/xip_branchA_gspc_e3_s3/economic_time_stability_summary.csv)
- [xip Branch A details](../economic_time/results/xip_branchA_gspc_e3_s3/economic_time_stability_details.csv)

설정:

- strongest interaction pair였던 `intensity + indexret` 유지
- `concat_a:intensity_indexret`와 `xip_a:intensity_indexret`를 같은 조건으로 비교
- `xip_a`는 `r=4`, bias-free explicit interaction path 사용

대표 수치:

- `concat_a:intensity_indexret IC mean=0.0592`
- `xip_a:intensity_indexret IC mean=0.0608`
- `concat_a:intensity_indexret IC std=0.0226`
- `xip_a:intensity_indexret IC std=0.0217`
- `concat_a:intensity_indexret param_count=45266`
- `xip_a:intensity_indexret param_count=45418`

해석:

- `xip_a`는 `Branch A`에서 처음 나온 positive signal이다.
- 최소 `FiLM`과 달리 붕괴하지 않았고, 평균 IC와 seed variance 모두 `concat_a:intensity_indexret` 근처 또는 약간 위다.
- 다만 pooled paired test에서는 `concat_a:intensity_indexret`와 유의한 차이가 없다 (`p=0.9242`).
- 따라서 현재는 `winner`보다 `stable match / promising candidate`로만 써야 한다.
- 추가 진단상 `h_int`는 완전히 죽지 않았지만, `h_int=0` ablation 시 평균 IC 하락은 `0.0019`로 작다.
- 즉 현재 match는 `strong explicit interaction utility`보다는 `early mixing + modest interaction`에 가깝다.

의미:

현재 증거는 `linear PE`가 나쁘다는 것은 보여주지만, `better PE encoding`이 곧바로 `concat`보다 낫다고까지는 보여주지 못한다.

## J. Concat Decomposition

결과 파일:

- [concat decomposition summary](../economic_time/results/concat_decomp_gspc_e3_s3/economic_time_stability_summary.csv)

대표 수치:

- `concat_a IC mean=0.0571`
- `concat_a:no_intensity IC mean=0.0026`
- `concat_a:intensity_only IC mean=0.0066`
- `concat_a:binned_intensity_only IC mean=0.0070`
- `concat_a:shuffled_intensity IC mean=0.0187`

해석:

- intensity signal 자체는 중요하다.
- 하지만 intensity 하나만으로는 full `concat_a`의 강점을 설명할 수 없다.
- current concat path에서는 intensity ordering이 primary driver로 보이지는 않는다.

의미:

다음 method는 `geometry를 더 키우는 것`보다 `intensity signal을 다른 useful signals와 어떤 interface에서 결합할지`를 먼저 봐야 한다.

## K. Concat Interaction Structure

결과 파일:

- [concat interaction summary](../economic_time/results/concat_interaction_gspc_e3_s3/economic_time_stability_summary.csv)
- [concat interaction details](../economic_time/results/concat_interaction_gspc_e3_s3/economic_time_stability_details.csv)

대표 수치:

- `concat_a:intensity_only IC mean=0.0066`
- `concat_a:position_only IC mean=0.0188`
- `concat_a:indexret_only IC mean=0.0205`
- `concat_a IC mean=0.0571`
- `concat_a:intensity_indexret IC mean=0.0592`

linear ensemble baseline:

- `interaction:intensity+position IC mean=0.0189`
- `interaction:intensity+indexret IC mean=0.0184`

해석:

- strongest baseline의 힘은 단일 채널에서 나오지 않는다.
- 또한 단순한 선형 결합으로도 설명되지 않는다.
- 즉 `concat_a`의 핵심은 `interaction-friendly interface`일 가능성이 높다.

의미:

다음 method는 `better PE`보다 먼저 `intensity와 다른 state channel이 상호작용하는 구조를 어떻게 더 잘 표현할지`를 물어야 한다.
