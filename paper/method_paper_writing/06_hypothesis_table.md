# Hypothesis Table

이 문서는 현재 method line의 핵심 가설을 `주장 -> 증거 -> 판단` 형태로 고정한 표다.

목적:

- 지금 무엇이 지지됐고 무엇이 안 됐는지 한 눈에 본다.
- 논문 본문에서 과장하면 안 되는 부분을 분리한다.
- 다음 설계가 어떤 가설을 깨거나 살려야 하는지 명확히 한다.

## Current Hypotheses

| ID | Hypothesis | Current Evidence | Status | Main Files | Implication |
| --- | --- | --- | --- | --- | --- |
| `H1` | `regime-aware temporal representation`은 강한 baseline보다 더 나은 예측 성능을 만든다 | `learned_tau_rope`는 `concat_a`를 반복적으로 못 이겼고, `shape_signature_token`도 `simple_summary_token`과 `concat_a`를 안정적으로 못 넘었다 | `Not supported` | [evidence snapshot](./02_evidence_snapshot.md), [GSPC align-only](../economic_time/results/method_gspc_s3_e1e3/economic_time_stability_summary.csv), [shape/simple token](../economic_time/results/method_gspc_sig_s3_e1e3/economic_time_stability_summary.csv) | 현재 구현 family를 그대로 method winner로 쓰면 안 된다 |
| `H2` | attention geometry change는 predictive gain으로 이어진다 | `QK` 수준 ordering과 `qk_swap_delta` 증가는 확인됐지만, `IC`는 거의 0이거나 약했고 `qk_pred_corr`도 매우 작았다 | `Not supported` | [QK output smoke](../economic_time/results/method_gspc_qkout_smoke/economic_time_stability_summary.csv), [QK output history](../economic_time/results/method_gspc_qkout_smoke/economic_time_stability_train_history.csv) | `geometry change != utility`가 현재 핵심 structural insight다 |
| `H3` | economic activity와의 정렬은 representation을 유의미하게 만든다 | `step-intensity spearman`은 양수로 만들 수 있었지만, 그것만으로 geometry나 성능이 보장되지는 않았다 | `Supported as necessary but not sufficient` | [GSPC align-only](../economic_time/results/method_gspc_s3_e1e3/economic_time_stability_summary.csv), [IXIC align-only](../economic_time/results/method_ixic_s3_e1e3/economic_time_stability_summary.csv) | alignment는 유지해야 하지만, 단독 성공 기준으로 쓰면 안 된다 |
| `H4` | local branch는 regime-aware signal 전달의 병목이 될 수 있다 | `learned_tau_rope:global_only`가 hybrid보다 나았고, `qk -> global/fused`는 보였지만 `qk -> local/output`은 거의 없었다 | `Supported` | [global-only](../economic_time/results/method_gspc_globalonly_smoke/economic_time_stability_summary.csv), [QK output history](../economic_time/results/method_gspc_qkout_smoke/economic_time_stability_train_history.csv) | architecture interaction은 failure analysis의 핵심 축으로 써야 한다 |
| `H5` | pointwise temporal warping보다 window-level signature가 더 좋은 인터페이스일 수 있다 | `shape_signature_token > simple_summary_token` 방향은 약하게 있었지만, 3 epoch 기준 둘 다 winner가 아니었다 | `Inconclusive / currently unsupported` | [shape/simple token](../economic_time/results/method_gspc_sig_s3_e1e3/economic_time_stability_summary.csv), [shape/simple global-only](../economic_time/results/method_gspc_sig_global_s3_e1e3/economic_time_stability_summary.csv) | A 방향은 아직 아이디어 수준이며, 최소 구현 성공으로 보기는 이르다 |
| `H6` | strongest baseline의 예측 이점은 단순 interface보다 intensity-linked signal usage에서 더 많이 나온다 | `concat_a:no_intensity`는 평균 IC가 크게 떨어졌고, 두 seed에서 `concat_a` 대비 유의하게 나빴다 | `Supported with caveat` | [concat no-intensity summary](../economic_time/results/concat_no_intensity_gspc_e3_s3/economic_time_stability_summary.csv), [concat no-intensity t-tests](../economic_time/results/concat_no_intensity_gspc_e3_s3/economic_time_stability_ttests.csv) | 다음 method는 baseline이 잘 쓰는 simple signal을 먼저 설명해야 한다 |
| `H7` | 같은 intensity signal을 줄 때 PE injection이 input concat보다 더 낫다 | `cycle_pe:intensity_only`는 `concat_a:intensity_only`를 평균적으로 이기지 못했고, `cycle_pe:intensity_embed`도 개선은 있었지만 평균적으로 concat을 넘지 못했다 | `Not supported, but implementation-sensitive` | [cycle_pe vs concat intensity-only summary](../economic_time/results/cyclepe_vs_concat_intensity_gspc_e3_s3/economic_time_stability_summary.csv), [cycle_pe embed vs linear summary](../economic_time/results/cyclepe_embed_vs_linear_gspc_e3_s3/economic_time_stability_summary.csv) | 현재 증거로는 `PE superiority`를 주장하면 안 되지만, `linear PE` 실패만으로 PE idea 전체를 버리면 안 된다 |
| `H8` | concat baseline에서 intensity ordering이 핵심 predictive source다 | `shuffled_intensity`가 `intensity_only`보다 더 나쁘지 않았고, 오히려 평균 IC가 더 높았다 | `Currently not supported` | [concat decomposition summary](../economic_time/results/concat_decomp_gspc_e3_s3/economic_time_stability_summary.csv) | 현재 minimal path에서는 ordering보다 level/distribution과 interface interaction이 더 중요할 수 있다 |
| `H9` | strongest baseline의 예측 이점은 channel interaction에서 나온다 | 단일 채널들은 모두 약했지만, `intensity + position`과 `intensity + indexret` pair 모델은 강했고, validation-fitted linear ensemble baseline은 훨씬 약했다 | `Supported` | [concat interaction summary](../economic_time/results/concat_interaction_gspc_e3_s3/economic_time_stability_summary.csv), [concat interaction details](../economic_time/results/concat_interaction_gspc_e3_s3/economic_time_stability_details.csv) | 다음 method는 `interaction-friendly interface`를 핵심 설계 목표로 삼아야 한다 |
| `H10` | minimal FiLM conditioning은 raw concat보다 더 안정적인 interaction interface가 될 수 있다 | `film_a:intensity_indexret`는 한 seed에서 강했지만 평균적으로 `concat_a:intensity_indexret`보다 크게 약했고 분산도 컸다 | `Not supported` | [FiLM Branch A summary](../economic_time/results/film_branchA_gspc_e3_s3/economic_time_stability_summary.csv), [FiLM Branch A t-tests](../economic_time/results/film_branchA_gspc_e3_s3/economic_time_stability_ttests.csv) | `interaction-friendly interface`라는 질문은 살아 있지만, 최소 FiLM은 아직 답이 아니다 |
| `H11` | explicit interaction projection은 raw concat의 early mixing 장점을 유지하면서 더 해석 가능한 interface가 될 수 있다 | `xip_a:intensity_indexret`는 평균 IC와 seed variance에서 `concat_a:intensity_indexret`와 비슷하거나 약간 더 좋았지만, pooled paired test에서는 차이가 없었고 (`p=0.9242`), explicit interaction term을 꺼도 평균 IC 하락이 작았다 (`ΔIC≈0.0019`) | `Supported as a matching candidate, not superiority` | [xip Branch A summary](../economic_time/results/xip_branchA_gspc_e3_s3/economic_time_stability_summary.csv), [xip Branch A pooled t-tests](../economic_time/results/xip_branchA_gspc_e3_s3/economic_time_stability_pooled_ttests.csv) | `raw concat을 대체할 첫 후보`로는 의미가 있지만, 아직 superiority claim은 안 되고 interaction term 자체의 utility는 더 키워야 한다 |

## Working Read

현재 가장 정직한 읽기는 아래와 같다.

1. `alignment`는 만들 수 있다.
2. `QK geometry change`도 만들 수 있다.
3. 그러나 그 둘만으로는 `predictive utility`가 나오지 않는다.
4. 따라서 지금 남는 핵심 질문은 `왜 geometry change가 utility로 이어지지 않는가`이다.
5. strongest baseline은 우리가 만든 복잡한 geometry보다 `intensity-linked signal`을 더 직접적으로 잘 쓰고 있을 가능성이 높다.
6. 더 나아가, 같은 intensity signal을 줘도 현재는 `PE injection`이 `input concat`보다 좋은 interface라고 보이지 않는다.
7. 다만 `PE encoding function`을 바꾸면 결과가 달라지므로, 현재 실패는 `PE idea` 전체가 아니라 `current PE implementation`의 한계도 포함한다.
8. `concat_a`의 minimal intensity path에서는 ordering 자체보다 level/distribution과 interface interaction이 더 중요한 후보로 보인다.
9. strongest empirical baseline은 `단일 강한 signal`이 아니라 `channel interaction`을 잘 쓰는 구조일 가능성이 높다.
10. 하지만 그 interaction을 구조적으로 더 잘 살린다고 가정한 최소 `FiLM` 구현도 아직 raw concat을 이기지 못했다.
11. 반면 `explicit interaction projection`은 `Branch A`에서 처음 나온 positive signal이며, 현재 기준으로는 raw concat과 `match`하는 첫 후보로 보인다.
12. 하지만 현재 `explicit interaction term`의 직접적 예측 기여는 작아서, 아직은 `why it works`까지 완성된 상태는 아니다.

## What Can Be Claimed Now

지금 당장 method paper 초안에서 비교적 안전하게 쓸 수 있는 문장:

`We find that regime-aware temporal representations can be aligned with market activity and can measurably alter pre-softmax attention geometry, but these changes do not automatically translate into predictive gains.`

지금 쓰면 안 되는 문장:

- `learned economic time consistently improves forecasting`
- `attention geometry change explains predictive gain`
- `window-level economic signatures outperform simple market summaries`

## Next-Design Filter

앞으로 새 설계를 볼 때는 아래 한 줄로 걸러야 한다.

`새 설계는 H1 또는 H2 중 적어도 하나를 뒤집을 수 있어야 한다.`

즉:

- `alignment`만 더 좋아지는 설계는 부족하다.
- `geometry`만 더 커지는 설계도 부족하다.
- 최소한 `predictive gain` 또는 `geometry -> utility coupling` 중 하나를 더 강하게 보여야 한다.
