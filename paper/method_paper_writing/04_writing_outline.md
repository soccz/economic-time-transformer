# Writing Outline

이 outline은 `현재 결과로 실제로 방어 가능한 method-analysis paper` 기준이다.

핵심 원칙:

- 이 논문은 `winner paper`가 아니다.
- 이 논문은 `regime-aware representation -> geometry -> utility` 사슬을 분해한 분석 논문이다.
- 본문은 [06_hypothesis_table.md](./06_hypothesis_table.md)의 `H1~H5` 순서와 맞물려야 한다.

## Title Direction

후보:

- `When Regime-Aware Time Representations Change Geometry but Not Utility`
- `Economic-Time Representations in Financial Transformers: Geometry Without Predictive Gain`
- `Why Regime-Aware Attention Geometry Fails to Improve Financial Forecasting`

부제 후보:

- `A Failure Analysis of Learned Temporal Warping in Financial Transformers`

## Section 1. Introduction

### Core Question

`시장 상태를 반영한 시간 표현은 Transformer의 attention geometry를 실제로 바꾸고, 그 변화는 왜 예측 이득으로 이어지지 않는가?`

### Problem Framing

- 기존 regime-aware forecasting 계열은 representation change가 utility improvement로 이어진다고 암묵적으로 가정한다.
- 하지만 금융 시계열에서는 representation, geometry, utility가 분리될 수 있다.

### Paper Claim

`We show that regime-aware temporal representations can be aligned with market activity and can measurably alter pre-softmax attention geometry, yet these changes do not automatically translate into predictive gains.`

### Main Contributions

1. `learned τ`의 alignment와 `QK geometry` 변화를 분리해서 진단한다.
2. geometry change와 predictive gain 사이의 간극을 실험적으로 보인다.
3. local branch suppression과 naive window-signature insufficiency를 failure mechanism 후보로 제시한다.

## Section 2. Experimental Setup

### Task

- financial forecasting task
- main metrics: `IC`, `MAE`

### Baselines

- `static`
- `concat_a`

### Method Families

- `rule-based tau_rope`
- `learned_tau_rope`
- `window signature token`

### Why These Families

- `learned_tau_rope`: pointwise temporal warping 가설 검정
- `window signature token`: window-level economic signature 가설 검정

## Section 3. Hypotheses And Diagnostics

이 섹션은 본문에서 매우 중요하다. `무엇을 검정했고, 무엇으로 판정했는가`를 먼저 잠근다.

### Hypotheses

- `H1`: regime-aware temporal representation은 강한 baseline보다 더 나은 예측 성능을 만든다
- `H2`: attention geometry change는 predictive gain으로 이어진다
- `H3`: activity alignment는 useful representation의 필요 조건이다
- `H4`: architecture interaction, 특히 local branch가 regime-aware signal을 억제할 수 있다
- `H5`: window-level signature가 pointwise warping보다 더 나은 인터페이스일 수 있다

### Diagnostics

- `step-intensity spearman`
- `qk_swap_delta`
- `attn_swap_delta`
- `global-only vs hybrid`
- `simple summary vs shape signature`

## Section 4. Result Chain

이 섹션은 `H3 -> H2 -> H1 failure -> H4 -> H5` 흐름으로 쓴다.

### 4.1 Alignment Is Achievable But Insufficient

- `H3`를 다룸
- activity alignment는 만들 수 있지만 그것만으로 geometry와 utility가 나오지 않음을 보임

### 4.2 QK Geometry Can Be Changed

- `H2`의 전반부를 다룸
- `learned τ -> QK geometry` 경로는 실제로 존재함을 보임
- 단, post-softmax attention 및 output utility와는 분리됨

### 4.3 Geometry Change Does Not Yield Predictive Gain

- `H1`, `H2`의 핵심 결과
- geometry가 바뀌어도 `concat_a`를 이기지 못함
- `representation change != predictive utility`를 본문 headline insight로 둔다

### 4.4 Architecture Interaction Matters

- `H4`를 다룸
- global-only와 hybrid 비교를 통해 local branch suppression을 보임
- 그러나 suppression 제거만으로 winner가 되지는 않음을 같이 말한다

### 4.5 Naive Window Signatures Are Not Enough

- `H5`를 다룸
- shape signature가 simple summary보다 약한 방향성은 보여도, predictive winner는 아님을 정리

## Section 5. Failure Analysis

### Main Question

`왜 geometry change가 predictive utility로 이어지지 않았는가?`

### Failure Mechanism Candidates

이 섹션에서는 후보를 너무 많이 열지 않는다. 아래 3개만 쓴다.

1. `interface mismatch`
2. `architecture interaction`
3. `representation insufficiency`

### What We Can Rule Out

- gradient가 전혀 안 흐른 것은 아니다
- `QK` 변화 자체가 전혀 없었던 것도 아니다

### What We Cannot Yet Claim

- 어떤 단일 failure mechanism이 최종 원인이라고 단정할 수는 없다
- stronger method가 무엇인지까지 본문에서 해결했다고 쓰면 안 된다

## Section 6. Discussion

### Research Implication

- 금융 시계열에서 regime-aware representation은 `만드는 것`보다 `어떤 interface로 예측기와 결합되는가`가 더 중요할 수 있다
- geometry change를 관찰하는 것만으로 method utility를 주장하면 안 된다

### Narrow Future Directions

future work는 길게 쓰지 않는다. 딱 두 줄기만 둔다.

1. stronger window-level signatures
2. representation-to-utility coupling redesign

## Section 7. Conclusion

최종 문장 후보:

`In financial forecasting, regime-aware temporal representations can be aligned with market activity and can measurably alter Transformer geometry, but these changes do not automatically yield predictive gains. This gap suggests that the main challenge is not merely learning better representations, but learning interfaces that translate representation change into forecasting utility.`

## Tables And Figures To Anchor The Paper

### Main Table

- `static`
- `concat_a`
- `learned_tau_rope`
- `static_tau_rope:global_only`
- `learned_tau_rope:global_only`
- `simple_summary_token`
- `shape_signature_token`

지표:

- `IC`
- `MAE`
- `step-intensity spearman`
- `qk_swap_delta`

### Main Figure Sequence

1. alignment vs utility
2. QK geometry vs attention/output bottleneck
3. hybrid vs global-only
4. simple summary vs shape signature
