# Economic Time Research Log
# 전체 실험 흐름 및 현재 상태 정리

---

## 1. 원래 주장 (Starting Claim)

> 물리적 시간 t보다 시장 경로가 형성하는 경제적 시간 τ_t가 금융 시계열 표현에
> 더 적합하다. τ를 Transformer의 시간 좌표계로 사용하면 단순히 상태를 입력에
> 추가하는 것(concat)보다 예측 성능이 개선된다.

타겟: NeurIPS / ICLR 수준의 방법론 논문

---

## 2. 실험 흐름 요약

### Phase 1 — Additive PE 계열 (econ_time, cycle_pe)

**방법**: sinusoidal PE에 intensity를 additive하게 주입
- `econ_time`: PE + W_intensity · intensity_t
- `econ_time:pe_only`, `econ_time:qk_only`: 주입 위치 ablation

**핵심 결과** (stability_gspc_e3_s3, 3 seeds × 3 epochs):

| model | IC mean | IC std |
|---|---|---|
| concat_a | **0.0571** | 0.0229 |
| econ_time | 0.0438 | 0.0273 |
| econ_time:pe_only | 0.0429 | 0.0273 |
| econ_time:qk_only | 0.0430 | 0.0177 |
| static | 0.0103 | 0.0604 |

t-test: econ_time vs concat_a, 모든 seed에서 p > 0.48. 유의하지 않음.

**진단**:
- pe_scale_mean = 0.003~0.004 → PE 기여가 사실상 0
- g를 키우면 (g=5, g=10) IC가 음수로 붕괴
- 유효한 작동 범위 없음

**결론**: additive PE 계열은 concat_a를 이기지 못한다. 주입 방식의 구조적 한계.

---

### Phase 2 — Rule-based τ-RoPE

**방법**: intensity → step_t = softplus(α · intensity_t) → τ_t = cumsum(step_t) → RoPE(Q, K)
- additive PE 없이 Q/K rotation에 직접 τ 주입
- 이론적 정당화: attention score가 f(τ_t - τ_s)의 함수가 됨

**핵심 결과** (tau_rope_epoch_trace, seed 7):

| epochs | model | IC | attn_swap_delta |
|---|---|---|---|
| 1 | concat_a | 0.0419 | - |
| 1 | tau_rope | 0.0289 | 1.6e-05 |
| 3 | concat_a | **0.0713** | - |
| 3 | tau_rope | 0.0659 | 1.4e-05 |

t-test: tau_rope vs concat_a, p = 0.55 (1-epoch), p = 0.84 (3-epoch). 유의하지 않음.

**진단**:
- tau_corr_mean = 0.9979 → τ가 물리적 시간 t와 거의 동일
- attn_swap_delta = 1.6e-05 → geometry 변화 미미
- intensity-only rule로는 τ와 t의 괴리를 충분히 만들 수 없음

**Regime split 결과** (path2_regime_report, GSPC+IXIC, 2022-2024):

| market | regime | concat_a IC | tau_rope IC | tau_rope - concat_a |
|---|---|---|---|---|
| GSPC | Bull/quiet | 0.0640 | 0.0291 | -0.035 |
| GSPC | Bull/volatile | 0.0731 | **0.0748** | +0.002 |
| IXIC | Bull/quiet | 0.0389 | 0.0281 | -0.011 |
| IXIC | Bull/volatile | 0.0539 | **0.0784** | +0.025 |

MAE: 모든 구간에서 tau_rope ≤ concat_a (일관된 패턴)

**탐색적 발견**: Bull/volatile에서 tau_rope가 concat_a와 동등하거나 앞섬.
단, n_dates=79로 통계적 유의성 없음 (SE ≈ 0.11).

**결론**: rule-based τ-RoPE는 concat_a를 전체적으로 이기지 못한다.
geometry 진단이 너무 약하다. τ와 t의 상관이 너무 높다.

---

### Phase 3 — Learned τ-RoPE (naive GRU, regularizer 없음)

**방법**: market_seq → causal GRU → step_t = softplus(head(h_t)) → τ_t = cumsum → RoPE

**핵심 결과** (learned_tau_rope_smoke, GSPC 2022-2024):

| model | IC | MAE | step_intensity_spearman | tau_corr | attn_swap_delta |
|---|---|---|---|---|---|
| concat_a | **0.0419** | 0.007150 | - | - | - |
| learned_tau_rope | 0.0106 | 0.007966 | **-0.500** | 0.999996 | 4.5e-07 |
| tau_rope (rule) | 0.0289 | 0.016823 | +1.0 | 0.9979 | 1.6e-05 |

t-test: learned_tau_rope vs concat_a, p = 0.36. 유의하지 않음.

**Confirm 결과** (2020-2024, GSPC+IXIC):

| market | model | IC | t-stat vs concat_a | p |
|---|---|---|---|---|
| GSPC | tau_rope | 0.0076 | 1.33 | 0.185 |
| IXIC | tau_rope | 0.0093 | 1.55 | 0.124 |

**진단**:
- step_intensity_spearman = -0.50 → GRU가 intensity와 반대 방향으로 step 학습
- τ가 경제적 시간이 아닌 예측 loss 최소화를 위한 임의 좌표로 수렴
- tau_corr = 0.999996 → 사실상 물리적 시간으로 붕괴
- attn_swap_delta = 4.5e-07 → rule-based보다도 geometry 변화 작음

**결론**: naive GRU는 경제적 시간을 학습하지 못한다.
end-to-end 학습만으로는 τ generator가 경제적 의미를 가진 좌표를 생성하지 않는다.

---

## 3. 전체 패턴 요약

모든 실험에서 일관된 패턴:

```
static < tau_rope 계열 ≈ learned_tau_rope < concat_a (IC 기준)
```

- static은 항상 진다 (p = 0.02~0.05로 유의하게)
- index conditioning 자체는 유효하다
- τ를 통한 geometry 주입은 concat_a를 이기지 못한다
- τ 방식과 무관하게 (additive / RoPE / learned) 결론이 동일하다

---

## 4. 실패 원인 분석

### 구조적 원인 (이론)

τ_t = cumsum(step_t)는 단조증가 좌표다. 단조증가 좌표는 물리적 시간 t와 높은 상관을 가질 수밖에 없다. τ와 t의 괴리를 만들려면 step_t의 분산이 매우 커야 하는데, 그러면 RoPE rotation 각도가 불안정해져서 학습이 망가진다.

즉 **단조증가 τ + RoPE는 구조적으로 τ와 t의 상관을 낮추기 어렵다.**

### 학습 원인 (empirical)

end-to-end 학습에서 τ generator는 예측 loss만 보고 τ를 결정한다. 경제적 의미를 강제하는 제약이 없으면 τ가 임의의 좌표로 수렴한다. step_intensity_spearman = -0.50이 이를 보여준다.

---

## 5. 현재 선택지

### 선택 1 — Regularizer 추가 후 마지막 시도

L_total = L_pred + λ · L_align
L_align = -Pearson(Δτ_t, intensity_t)

목표: step_intensity_spearman > +0.3, attn_swap_delta > rule-based 수준
성공 시: concat_a 대비 성능 우위 확인 후 논문 진행
실패 시: τ-RoPE 계열 전체 종료

### 선택 2 — 현재 결과로 논문 작성 (Negative Result Paper)

**주장**:
> "We systematically investigate market-path-conditioned economic time as a
> temporal coordinate system for financial Transformers. Despite theoretical
> motivation from subordinated process theory (Clark 1973), we find that
> coordinate-space conditioning does not outperform input-space conditioning
> on ranking metrics, while consistently reducing point error. We analyze
> why: the monotone constraint on τ prevents sufficient divergence from
> physical time, and end-to-end training without alignment regularization
> causes τ to collapse toward arbitrary coordinates."

타겟: TMLR (rolling submission, negative/null result 환영)

방어 가능한 기여:
1. τ-RoPE의 체계적 구현 및 ablation (additive / rule-based / learned)
2. 실패 원인의 이론적 + 실증적 분석
3. conditioning 위치가 IC vs MAE trade-off를 바꾼다는 탐색적 발견
4. step_intensity_spearman 진단 지표 제안

---

## 6. 다음 결정

선택 1 또는 선택 2 중 하나를 결정해야 한다.

선택 1 기준: regularizer 추가 후 smoke에서
- step_intensity_spearman > +0.3
- attn_swap_delta > 1.6e-05 (rule-based 수준 이상)
- IC가 concat_a 방향으로 개선

이 세 가지 중 하나라도 안 되면 즉시 선택 2로 전환.

---

## 7. 파일 맵

| 실험 | 디렉토리 | 핵심 파일 |
|---|---|---|
| Additive PE stability | results/stability_gspc_e3_s3/ | economic_time_stability_summary.csv |
| Rule-based τ-RoPE epoch trace | results/tau_rope_epoch_trace/ | economic_time_stability_summary.csv |
| Rule-based τ-RoPE smoke | results/tau_rope_smoke/ | *_summary.csv, *_ttest.csv |
| Regime split | results/path2_regime_report.csv | - |
| Learned τ-RoPE smoke | results/learned_tau_rope_smoke/ | *_summary.csv, *_ttest.csv |
| Confirm GSPC 2020-2024 | results/confirm_gspc_2020_2024/ | *_summary.csv, *_ttest.csv |
| Confirm IXIC 2020-2024 | results/confirm_ixic_2020_2024/ | *_summary.csv, *_ttest.csv |

| 코드 | 파일 | 내용 |
|---|---|---|
| 모델 | economic_time/market_time_model.py | static, concat_a, econ_time, tau_rope, learned_tau_rope |
| 실험 runner | economic_time_supervised.py | 단일 실험 |
| Stability runner | economic_time_stability.py | multi-seed, multi-epoch |
| Regime report | economic_time_regime_report.py | 체제별 IC/MAE 분리 |
| 가설 문서 | economic_time/00_preregistered_test_spec.md | pre-registered H1, H2 |
