# Method Paper Direction Lock (2026-03-10)

## Purpose

이 문서는 앞으로의 메인 연구 방향을 `applied paper`가 아니라 `method paper`로 고정하기 위한 기준 문서다.

목표는 대화를 반복하는 것이 아니라, 앞으로 무엇을 만들고 무엇을 버릴지 명확히 정하는 것이다.

이 문서는 현재 저장소의 fallback 논문 방향인 [PAPER_DIRECTION_20260310.md](./PAPER_DIRECTION_20260310.md) 와 별개다.

---

## Locked Main Goal

메인 목표는 다음 한 문장으로 고정한다.

`금융 시계열에서 시장 경로로부터 학습된 economic time 좌표 τ를 만들고, 이 τ가 attention의 temporal geometry 자체를 바꾸도록 설계한다.`

짧게 쓰면:

`learned economic time for attention`

---

## Locked Novelty Statement

이 논문의 novelty는 아래처럼 잡는다.

`generic adaptive positional encoding`이 아니다.

`금융 시장 경로로부터 monotone economic time 좌표를 학습하고, 이 좌표가 attention의 temporal metric 자체를 바꾸도록 설계한다`가 핵심이다.

즉 novelty 포인트는 세 부분의 결합이다.

1. `market-path-conditioned`
2. `learned monotone τ`
3. `attention geometry modification`

다음처럼 쓰면 과장이다.

- 우리가 최초의 time-warped Transformer를 만들었다
- 우리가 최초의 adaptive positional encoding을 만들었다

다음처럼 써야 한다.

- 우리는 `economic time`을 attention geometry 수준으로 옮긴다
- 우리는 금융 시계열에서 `input conditioning`이 아니라 `temporal metric`을 직접 바꾸는 learned coordinate를 제안한다

---

## Locked Paper Type

이 논문은 다음이 아니다.

- 금융 applied comparison paper
- 단순 baseline benchmark paper
- TCN/CNN architecture paper
- VAE uncertainty paper

이 논문은 다음이다.

- temporal coordinate method paper
- attention geometry paper
- financial time representation paper

---

## Theory Anchor

이 논문의 이론적 출발점은 `economic time / business time`이다.

가장 직접적인 앵커는 Clark (1973)류의 subordinated process 관점이다.

이 문헌이 말하는 핵심은:

- 달력시간이 아니라 activity time이 가격 변동을 더 자연스럽게 설명할 수 있다
- 변동성이 큰 구간에서는 물리적 시간과 경제적 시간이 더 크게 어긋날 수 있다

이 논문은 Clark 이론 자체를 증명하는 논문은 아니다.

이 논문이 하는 일은:

- economic time이라는 개념을 Transformer attention의 temporal coordinate로 구현하는 것

즉 이론 사용 방식은 아래로 제한한다.

- `motivation`: 사용 가능
- `conceptual framing`: 사용 가능
- `strict financial theory proof`: 사용 불가

---

## Locked Main Question

메인 질문은 이것 하나다.

`Can we learn a market-path-conditioned monotone temporal coordinate τ that changes attention geometry and improves forecasting beyond input-space conditioning?`

이 질문에서 중요한 것은 세 가지다.

1. `learn`
2. `temporal coordinate`
3. `attention geometry`

성능만 오르는 모델은 충분하지 않다.

---

## Locked Comparative Frame

메인 비교 프레임은 아래로 고정한다.

- `static`: 물리적 시간 기준
- `concat_a`: input-space conditioning
- `rule-based tau_rope`: handcrafted economic time proxy
- `learned_tau_rope`: learned economic time

이 네 모델의 역할은 다르다.

- `static`: no-conditioning baseline
- `concat_a`: 강한 실용 baseline
- `rule-based tau_rope`: economic-time의 최소 구현
- `learned_tau_rope`: 메인 제안법

핵심 질문은 항상 아래 순서를 따른다.

1. `learned_tau_rope > static` 인가
2. `learned_tau_rope > rule-based tau_rope` 인가
3. `learned_tau_rope > concat_a` 인가

3번만 보고 판단하지 않는다.

---

## Locked Core Method

메인 방법은 아래 구조로 고정한다.

`market-path encoder -> monotone step process Δτ_t -> cumulative τ_t -> τ-RoPE / τ-relative attention`

현재 기본안:

- 입력: broad-market path
- encoder: 작은 causal sequence encoder
  - 우선순위: causal GRU or causal TCN
- 출력: `step_t = softplus(head(h_t))`
- 누적: `τ_t = cumsum(step_t)`
- 주입: additive PE가 아니라 `τ-RoPE` 또는 `τ-relative attention`

---

## What This Paper Must Not Become

다음 방향으로 새면 method paper 초점이 깨진다.

- `TCN/CNN + Transformer + VAE`를 한 편의 동등한 novelty로 밀기
- uncertainty paper로 바꾸기
- explainability paper로 바꾸기
- 단순 finance alpha discovery paper로 바꾸기
- “우리도 applied 결과 하나 좋다” 식 benchmark paper로 내려가기

즉 전체 시스템을 키우는 것보다 `learned τ`를 먼저 증명하는 것이 우선이다.

---

## Non-Negotiables

다음은 반드시 지킨다.

1. 메인 novelty는 `learned τ` 하나다.
2. additive PE superiority를 headline으로 쓰지 않는다.
3. `Transformer + TCN/CNN`은 backbone 또는 downstream module일 뿐, 이 논문의 주기여가 아니다.
4. `VAE`, `CVAE`, uncertainty decomposition은 이 논문 메인에서 뺀다.
5. `attention weights are explanations` 같은 주장은 금지한다.

---

## Required Evidence

이 논문이 method paper가 되려면 아래 네 축이 동시에 필요하다.

### 1. Semantic Alignment

학습된 `Δτ_t`가 경제적 activity와 정렬되어야 한다.

최소 진단:

- `Spearman(Δτ_t, intensity_t) > 0`

좋은 상태:

- seed 평균 기준 양수이고, 안정적으로 유지

의미:

- `τ`가 단순 예측용 임의 좌표가 아니라 economic time에 가깝다는 최소 증거

### 2. Geometry Activation

학습된 `τ`가 attention geometry를 실제로 바꿔야 한다.

최소 진단:

- `qk_swap_delta`
- `attn_swap_delta`
- random swap null 대비 유의미한 차이

의미:

- `τ`가 단순 latent feature가 아니라 temporal metric에 개입한다는 증거

### 3. Predictive Gain

`learned_tau_rope`가 강한 baseline보다 나아야 한다.

필수 baseline:

- `static`
- `concat_a`
- `rule-based tau_rope`

필수 조건:

- 최소 한 개의 핵심 목적에서 `concat_a`를 반복 seed 기준으로 이겨야 한다
- 단일 seed lucky run으로는 부족하다

### 4. Repeatability

패턴이 한 데이터셋 한 구간에만 나오면 안 된다.

최소 반복성:

- `GSPC`
- `IXIC`

이상적 반복성:

- 추가 금융 시계열
- 가능하면 일반 시계열 벤치마크

---

## Evidence Hierarchy

증거의 우선순위는 아래와 같다.

1. `alignment`
2. `geometry`
3. `repeatable predictive gain`
4. `broader generalization`

이 순서를 뒤집지 않는다.

예:

- 성능이 좋지만 alignment가 없다 -> 불충분
- alignment는 좋지만 geometry가 안 움직인다 -> 불충분
- geometry는 움직이지만 repeatability가 없다 -> 불충분

---

## Explicit Failure Conditions

아래 중 하나라도 지속되면 method path는 실패로 본다.

1. `Δτ`가 intensity와 정렬되지 않는다.
2. geometry 진단이 거의 0에 머문다.
3. `concat_a`를 안정적으로 못 이긴다.
4. seed가 바뀌면 결과가 무너진다.
5. `τ`가 해석 불가능한 임의 좌표처럼 동작한다.

실패하면 억지로 method paper를 밀지 않는다.

---

## Decision Rule

앞으로의 판단 기준은 이렇다.

### Go

아래가 함께 성립하면 method paper를 계속 민다.

- alignment 양수
- geometry activation 증가
- baseline 대비 반복 가능한 성능 우위

### No-Go

아래가 지속되면 method paper를 중단하고 fallback으로 복귀한다.

- alignment는 되지만 geometry가 안 움직임
- geometry는 움직이지만 성능 우위가 없음
- 성능은 좋아도 해석 가능성이 없음

---

## Publishable Outcome Ladder

현재 method path의 가능한 결과는 세 단계다.

### Level 1

`learned_tau_rope`가 alignment는 만들지만 geometry와 성능 우위는 약하다.

의미:

- method paper로는 약함
- fallback applied paper의 appendix / future work 재료

### Level 2

alignment와 geometry는 분명하지만, 성능 우위가 금융 도메인 안에서만 제한적으로 나온다.

의미:

- TMLR 수준의 technical report / method note 가능
- top ML venue는 어려움

### Level 3

alignment, geometry, 성능 우위, 반복성이 모두 선다.

의미:

- ICLR / NeurIPS 스타일 method paper를 노릴 수 있음

지금 목표는 Level 3이고, Level 2에서 멈추면 솔직히 그 수준으로 쓴다.

---

## Fallback

fallback은 이미 존재한다.

- 문서: [PAPER_DIRECTION_20260310.md](./PAPER_DIRECTION_20260310.md)
- 논문 성격: applied financial Transformer paper
- 메인 비교: `static / concat_a / tau_rope`

즉 method path는 실패해도 연구 전체가 무너지지 않는다.

---

## Immediate Experimental Sequence

앞으로의 실험 순서는 아래로 고정한다.

### Step 1

`learned_tau_rope` alignment regularizer 안정화

확인 지표:

- `step_intensity_spearman`
- `train_align_corr`

### Step 2

geometry activation 확인

확인 지표:

- `qk_swap_delta`
- `attn_swap_delta`
- random null 대비 차이

### Step 3

짧은 seed 반복

기본:

- `GSPC`
- `IXIC`
- `1 epoch / 3 epochs`
- 3 seeds 이상

### Step 4

Step 1-3가 통과한 경우에만 더 큰 구조 확장

예:

- better encoder
- τ-relative bias 보강
- broader benchmark

이 순서를 건너뛰지 않는다.

---

## Immediate Next Step

지금 당장 할 일은 이것 하나다.

`alignment regularizer가 들어간 learned_tau_rope를 3 seeds x (1,3 epochs)로 짧게 반복하고, alignment-geometry-performance를 한 표로 정리한다.`

---

## Final Reminder

이 연구의 큰 질문은 맞다.

하지만 method paper는 `좋은 철학`만으로 되지 않는다.

이 논문은 아래가 모두 있어야만 method paper다.

- learned τ
- geometry change
- repeatable gain

셋 중 하나라도 없으면 아직 연구 계획이지, method paper가 아니다.
