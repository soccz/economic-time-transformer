# 실패 연대기: 10개의 가설이 무너진 과정

> *이 챕터는 연구 가이드 전체에서 가장 중요하다. 성공한 실험은 결과만 보면 되지만, 실패한 실험은 과정을 봐야 한다. 여기에는 우리가 "이번엔 된다"고 믿었던 모든 순간과, 그 믿음이 숫자 앞에서 무너진 과정이 시간 순서대로 기록되어 있다.*

---

## 0. 이 기록을 읽는 법

이 챕터는 연구 일지다. 학술 논문이 아니다.

여기에 적힌 숫자들은 모두 실제 실험 결과다. 반올림하지 않았고, 꾸미지 않았다. 어떤 가설이 어떤 근거로 세워졌고, 어떤 숫자가 나왔고, 그 숫자를 보고 무엇을 느꼈는지를 있는 그대로 적었다.

실패를 기록하는 이유는 단순하다. 같은 실패를 반복하지 않기 위해서다. 그리고 더 솔직하게 말하면, 실패의 패턴 안에 성공의 단서가 숨어 있었기 때문이다.

---

## 1. 출발: method paper를 쓰려고 했다

### 목표

처음 세운 목표는 명확했다. 시장 경로로부터 **경제적 시간 좌표 τ를 학습**하고, 이 τ가 Transformer attention의 temporal geometry 자체를 바꾸도록 설계한다. 그리고 이것이 단순히 시장 상태를 입력에 붙이는 `concat_a`보다 낫다는 것을 보인다.

짧게 쓰면: **learned economic time for attention**.

논문의 novelty는 세 가지의 결합이었다:

1. **market-path-conditioned** — 시장 경로가 조건이 된다
2. **learned monotone τ** — 학습된 단조 시간 좌표
3. **attention geometry modification** — attention의 시간 구조 자체가 바뀐다

### 4축 증거 체계

method paper가 되려면 네 가지 축이 동시에 서야 한다고 정했다.

| 축 | 내용 | 최소 기준 |
|---|---|---|
| Alignment | Δτ가 시장 activity와 정렬 | Spearman > 0 |
| Geometry | τ가 attention geometry를 실제로 바꿈 | qk_swap_delta > random null |
| Predictive Gain | learned_tau_rope > concat_a | 반복 seed 기준 |
| Repeatability | GSPC + IXIC 모두에서 | seed 3개 이상 |

내부적으로 이런 합의가 있었다: "4개 다 서면 ICLR급. 2개면 TMLR급. 1개 이하면 method paper는 포기."

이 기준은 나중에 우리를 구했다. 기준이 없었다면 "아직 가능성이 있다"며 끝없이 실험을 돌렸을 것이다.

---

## 2. 첫 번째 벽: alignment는 됐는데 예측이 안 된다

### 가설

"learned τ의 Δτ가 시장 intensity와 정렬되면, 그 자체로 attention geometry가 바뀌고, 예측도 나아질 것이다."

### 첫 시도: regularizer 없이

처음에는 alignment regularizer 없이 learned_tau_rope를 돌렸다. GRU encoder가 시장 경로를 받아 Δτ를 출력하고, 이것이 RoPE 주파수를 조절한다.

결과: **Spearman = −0.50**.

마이너스다. GRU가 intensity와 **반대 방향**으로 학습했다. 변동성이 높을 때 시간을 늘려야 하는데, 오히려 줄이고 있었다.

이때 처음 느꼈다: 학습 가능한 파라미터를 주면 자동으로 올바른 방향을 찾을 거라는 가정은 순진했다.

### 두 번째 시도: regularizer 추가

alignment regularizer를 넣었다. Δτ와 intensity 사이에 양의 상관관계를 유도하는 soft constraint.

결과: **Spearman = +0.46 ~ +0.52**. 시드에 따라 다르지만 안정적으로 양수.

"됐다!" — 이때의 감정을 솔직히 기억한다. alignment가 서면 나머지도 따라올 것 같았다.

### 그런데

GSPC 3 epoch 기준:

- `concat_a IC = 0.0571`
- `learned_tau_rope IC = 0.0354`

IXIC 3 epoch 기준:

- `concat_a IC = 0.0515`
- `learned_tau_rope IC = 0.0357`

alignment는 됐다. 하지만 **예측 성능은 concat_a의 60%에도 미치지 못했다**.

### 교훈

> **Alignment는 필요조건이지 충분조건이 아니다.**

Δτ가 시장 상태를 올바르게 반영하고 있다는 것과, 그 반영이 예측에 도움이 된다는 것은 완전히 다른 문제였다. 이 구분을 처음부터 알았으면 최소 2주는 아꼈을 것이다.

---

## 3. 두 번째 벽: geometry를 바꿔도 소용없다

### 가설

"alignment가 부족한 게 아니라, geometry 변화가 약한 게 문제다. QK ordering을 직접 강제하면 attention이 경제적 시간을 따를 것이고, 그러면 예측도 나아질 것이다."

### 실험: QK ordinal loss

QK ordinal loss를 도입했다. τ 순서대로 pre-softmax QK 값이 정렬되도록 auxiliary loss를 걸었다.

결과:

- `qk_ord_rate ≈ 0.992` — 거의 완벽한 정렬
- `qk_swap_delta ≈ 2e-05`
- **`IC ≈ 0.000`**

이 숫자를 처음 봤을 때 뭔가 잘못 읽은 줄 알았다. QK ordering이 99.2%나 맞는데 IC가 0이라니.

### 왜?

두 가지 병목이 있었다:

1. **softmax 병목**: QK는 바뀌는데 softmax 정규화가 그 차이를 희석한다. pre-softmax에서 ordering이 바뀌어도 post-softmax attention map에서는 거의 같다.

2. **절대 규모 문제**: QK 차이의 절대 규모가 너무 작았다. ordering은 맞지만, 실제로 attention weight가 이동할 만큼의 에너지가 없었다.

### 추가 확인: strong geometry objective

geometry objective를 더 세게 걸어봤다.

결과:

- `qk_swap_delta = 0.022263` — 이전보다 1000배 큰 geometry 변화
- `IC = −0.0072` — 오히려 마이너스
- `step-intensity Spearman = 0.124` — alignment도 붕괴

geometry를 강제로 키우면 오히려 alignment가 무너지고, 예측도 망가졌다.

### 교훈

> **Representation change ≠ predictive utility.**

이 한 줄이 이 연구 전체에서 가장 비싼 교훈이었다. 표현을 바꾸는 것과 예측에 유용한 표현을 만드는 것은 다르다. geometry가 움직인다고 해서 예측이 나아지는 건 아니다. geometry는 도구이지 목적이 아니다.

---

## 4. 세 번째 벽: 로컬 브랜치가 방해한다

### 가설

"learned_tau_rope가 약한 건 하이브리드 구조에서 TCN 로컬 브랜치가 regime-aware geometry 신호를 덮기 때문이다. 로컬 브랜치를 끄면 learned τ의 진가가 드러날 것이다."

### 실험: global-only ablation

로컬 브랜치를 제거하고 Transformer global branch만 남겼다.

결과:

- `learned_tau_rope:global_only IC = 0.0239`
- `learned_tau_rope (hybrid) IC` — 이보다 낮음

맞다. 로컬 브랜치를 끄면 learned_tau_rope가 나아진다. 로컬 브랜치가 실제로 방해하고 있었다.

### 그런데

같은 global-only 조건에서:

- `static_tau_rope:global_only IC = 0.0346`
- `learned_tau_rope:global_only IC = 0.0239`
- `concat_a IC = 0.0713`

**로컬을 제거해도 static이 learned보다 낫다.** 그리고 둘 다 concat_a에 한참 못 미친다.

### 해석

로컬 브랜치 억제는 진짜 병목이었다. 하지만 **유일한 병목은 아니었다**. 문제가 하나가 아니라 여러 개가 겹쳐 있다는 것을 이때 깨달았다.

이건 연구에서 가장 위험한 상황이다. 병목 하나를 찾으면 "이것만 고치면 된다"고 믿게 된다. 실제로는 그 뒤에 또 다른 병목이 기다리고 있다. 우리는 이 패턴을 세 번 반복한 뒤에야 인정했다.

### 교훈

> **병목을 하나 찾았다고 문제가 하나인 것은 아니다.**

---

## 5. 네 번째 벽: window signature도 실패

### 가설

"문제는 τ를 만드는 방식이 아니라, τ 정보를 transformer에 전달하는 인터페이스가 약한 것이다. 시장 경로의 shape를 요약하는 signature token을 추가하면, 모델이 경제적 시간 맥락을 더 잘 이해할 것이다."

### 실험: shape signature token & simple summary token

시장 경로의 통계적 요약(shape signature)을 별도 토큰으로 만들어 시퀀스에 추가했다. 간단한 요약 토큰(simple summary)도 함께 테스트했다.

결과 (3 epoch):

- `shape_signature_token IC = −0.0226`
- `simple_summary_token IC = −0.0267`
- `concat_a IC = 0.0571`

마이너스다. concat_a의 **반대 방향**이다.

global-only로 바꿔봐도:

- `shape_signature_token:global_only IC ≈ 0.0000`
- `simple_summary_token:global_only IC = −0.0144`

여전히 못 쓴다.

### 해석

경로를 "요약"하는 것과 예측에 "활용"하는 것은 다르다. 요약 토큰은 정보를 압축하지만, 모델이 그 압축된 정보를 시퀀스의 다른 토큰과 상호작용시키는 방법을 배우지 못했다.

이때 "상호작용"이라는 키워드가 처음 등장했다. 단순히 정보를 넣는 것이 아니라, **어떤 구조로 넣느냐**가 중요하다.

### 교훈

> **경로 요약 ≠ 예측적 활용. 정보가 있어도 상호작용 구조가 없으면 죽은 정보다.**

---

## 6. 다섯 번째 벽: PE 주입이 concat보다 나쁘다

### 가설

"같은 intensity 정보를 positional encoding으로 넣으면, input concatenation보다 더 자연스럽게 attention의 시간 구조에 녹아들 것이다. PE는 attention의 언어이니까."

이건 연구 초기부터 가장 강하게 믿었던 가설이었다. PE가 attention의 temporal metric을 직접 조절하니까, 당연히 input에 그냥 붙이는 것보다 나아야 한다고 생각했다.

### 실험: intensity-only 공정 비교

같은 intensity signal, 같은 model capacity (param_count = 45,234), 같은 context_dim=1 조건에서:

- `concat_a:intensity_only IC mean = 0.0066`
- `cycle_pe:intensity_only IC mean = −0.0244`

**같은 정보를 다른 경로로 넣었을 뿐인데 방향이 뒤집혔다.**

시드별 결과:

| Seed | concat_a:intensity_only | cycle_pe:intensity_only |
|------|------------------------|------------------------|
| 7 | +0.0141 | −0.0331 |
| 17 | −0.0518 | −0.0342 |
| 27 | +0.0575 | −0.0060 |

### encoding 함수가 문제인가?

혹시 linear projection이 너무 약한 건 아닌지 확인했다. binned embedding으로 바꿔봤다.

- `cycle_pe:intensity_embed IC mean = 0.0040` (param_count = 46,226)
- `cycle_pe:intensity_only (linear) IC mean = −0.0244` (param_count = 45,234)

embedding이 linear보다 확실히 낫다. 즉 PE encoding 함수의 품질도 문제의 일부였다.

하지만 embedding으로 바꿔도 `concat_a:intensity_only (0.0066)`를 안정적으로 넘지 못했다.

### 해석

이 결과가 주는 메시지는 잔인하다:

1. PE injection이 자동으로 더 나은 건 아니다
2. 같은 신호도 경로에 따라 예측 방향이 뒤집힐 수 있다
3. encoding 함수의 품질이 PE 성능의 하한을 결정한다
4. 하지만 encoding을 개선해도 concat 인터페이스를 넘는 건 별개 문제다

### 교훈

> **이론적 자연스러움 ≠ 실증적 우위. "PE는 attention의 언어니까 더 나을 것이다"는 가설은 숫자 앞에서 무너졌다.**

---

## 7. 여섯 번째 벽: FiLM 참사

### 배경: 질문이 바뀌다

5번째 벽까지 거치면서 질문이 바뀌었다.

- 이전 질문: "어떤 PE가 concat보다 더 나은가?"
- 새 질문: "intensity와 다른 market-state channel의 interaction을 구조적으로 살리면서, raw concat보다 더 잘 일반화하는 interface는 무엇인가?"

이 질문 전환의 핵심 증거는 채널 분해 실험이었다:

- `concat_a IC mean = 0.0571`
- `concat_a:intensity_only IC mean = 0.0066`
- `concat_a:indexret_only IC mean = 0.0205`
- `concat_a:intensity_indexret IC mean = 0.0592`

intensity 하나로는 설명이 안 된다. indexret 하나로도 안 된다. 하지만 **둘을 함께 넣으면 full concat_a와 비슷하거나 더 낫다**. 그리고 이 결과는 선형 결합 baseline으로 설명되지 않았다:

- `linear ensemble: intensity+indexret IC mean = 0.0184`

즉 concat_a의 진짜 힘은 **채널 간 상호작용**에 있었다. 이것이 Branch A의 출발점이었다.

### 가설

"FiLM conditioning이 channel interaction을 더 구조적으로 표현할 수 있다. raw concat 대신 market-state summary를 modulation으로 주입하면, 더 깨끗하고 해석 가능한 interaction을 얻을 것이다."

이론적으로 FiLM은 가장 강해야 했다. affine transformation으로 feature를 직접 조절하니까.

### 실험

strongest interaction pair인 `intensity + indexret`를 유지한 채, concat과 FiLM을 비교.

결과:

- `concat_a:intensity_indexret IC mean = 0.0592`
- `film_a:intensity_indexret IC mean = −0.0081`
- `static IC mean = 0.0103`

**FiLM이 static보다 나쁘다.** 아무것도 안 한 것보다 못하다.

시드별 결과:

| Seed | concat_a:intensity_indexret | film_a:intensity_indexret |
|------|---------------------------|--------------------------|
| 7 | +0.0734 | −0.0789 |
| 17 | +0.0331 | +0.0760 |
| 27 | +0.0711 | −0.0212 |

seed 17에서는 FiLM이 +0.076으로 concat보다 낫다. 하지만 seed 7에서는 −0.079로 참사. 범위가 **−0.079 ~ +0.076**이다. 이것은 학습이 아니라 동전 던지기다.

### 이때의 기분

솔직히 말하면, 이 결과를 봤을 때 연구 전체에 대한 회의가 들었다. "이론적으로 가장 강해야 하는 방법이 가장 나쁘다"는 건, 우리의 이론 자체가 틀렸을 수 있다는 뜻이니까.

하지만 seed 17의 +0.076은 무시할 수 없었다. FiLM이 완전히 쓸모없는 건 아니다. 다만 **최적화가 극도로 불안정**하다. minimal FiLM on input projection이라는 구현이 문제일 수 있다.

### 교훈

> **이론적 우아함과 최적화 안정성은 별개다. "이 구조가 더 표현력이 높다"는 것이 "이 구조가 더 잘 학습된다"를 보장하지 않는다.**

---

## 8. 방향 전환: method paper를 포기하다 (2026-03-10)

### NO-GO 판정

4축 증거를 다시 점검했다.

| 축 | 상태 | 판정 |
|---|---|---|
| Alignment | Spearman +0.46~0.52 (regularizer 있을 때) | PASS |
| Geometry | qk_ord_rate 0.992, qk_swap_delta 존재 | PARTIAL |
| Predictive Gain | concat_a를 안정적으로 못 이김 | FAIL |
| Repeatability | GSPC, IXIC 모두에서 같은 패턴 | PASS (실패가 반복됨) |

Alignment는 됐고, geometry도 움직이긴 한다. 하지만 **성능 우위가 없다**. 그리고 이 실패는 한 시드, 한 데이터셋의 문제가 아니라 반복적으로 관찰되었다.

미리 정해둔 기준에 따르면, 이건 명확한 NO-GO다:

> "geometry는 움직이지만 성능 우위가 없음" → method paper 중단

### 판정문

그날 작성한 판정의 핵심:

- method paper를 억지로 밀지 않는다
- 현재 증거로는 `learned_tau_rope > concat_a`를 반복 가능하게 보여줄 수 없다
- fallback: applied paper로 전환

### fallback 논문

fallback은 이미 준비되어 있었다. `static` vs `concat_a` vs `tau_rope`를 비교하는 applied financial Transformer paper.

이 논문의 핵심 주장:

> "Coordinate-space conditioning은 input-space conditioning을 전면적으로 지배하지 않지만, 고변동성 구간에서 absolute prediction을 개선하고, 약하지만 방향적으로 양의 ranking advantage를 보인다."

confirmatory evidence도 있었다:

- H1 high-vol IC: `mean ΔIC = 0.0387`, `p = 0.0595`
- H2 high-vol MAE: `mean ΔMAE = 0.000092`, `p = 0.0019`

연구 전체가 무너진 건 아니었다. **질문을 바꾸는 것**이었다.

### 그 순간

방향 전환을 결정하는 건 기술적 판단이 아니라 심리적 판단이다. "아직 해볼 게 남았는데"라는 생각과 "이미 충분히 봤다"라는 생각 사이에서 결정해야 한다.

이때 미리 세운 4축 기준이 구했다. 기준이 없었다면 "다음 실험은 다를 거야"라며 3개월을 더 썼을 것이다. 기준이 있으니까 "기준에 안 맞으면 멈춘다"고 판단할 수 있었다.

### 교훈

> **방향 전환은 포기가 아니다. 미리 세운 기준에 따른 합리적 판단이다. 기준 없이 시작한 연구는 끝도 없다.**

---

## 9. 전환 후: "왜 실패하는가"가 새로운 질문이 되다

### method paper는 죽었지만

method paper를 포기한 뒤, 실험 데이터를 정리하다가 이상한 것을 발견했다.

실패 자체가 논문이 될 수 있었다.

### 채널 분해의 발견

concat_a를 분해하기 시작했다:

| 구성 | IC mean |
|------|---------|
| concat_a (full) | 0.0571 |
| intensity_only | 0.0066 |
| position_only | 0.0188 |
| indexret_only | 0.0205 |
| no_intensity | 0.0249 (→ 0.0026 in later run) |
| intensity_indexret | 0.0592 |
| shuffled_intensity | 0.0187 |

이 표에서 핵심 발견:

1. **단일 채널은 모두 약하다** — intensity만으로는 0.007, position만으로는 0.019
2. **intensity + indexret 조합이 full concat_a와 비슷하거나 더 낫다** — 0.059 vs 0.057
3. **이것은 선형 결합이 아니다** — linear ensemble baseline은 0.018에 불과

즉 concat_a의 힘은 단일 채널이 아니라 **채널 간 비선형 상호작용**에 있었다.

### 상호작용 → SNR 이론

왜 어떤 인터페이스는 이 상호작용을 살리고, 어떤 인터페이스는 죽이는가?

가설: **signal-to-noise ratio (SNR) 문제**. PE injection은 정보를 positional 차원에 압축하면서 SNR이 떨어진다. concat은 정보를 그대로 input 차원에 노출하니까 모델이 직접 interaction을 학습할 수 있다.

이 가설은 이전의 모든 실패를 설명했다:

- learned_tau_rope가 alignment는 잡지만 예측이 안 되는 이유 → PE 경로의 SNR이 낮아서
- geometry를 강제로 키워도 안 되는 이유 → geometry가 아니라 signal pathway의 문제
- FiLM이 불안정한 이유 → modulation이 interaction을 암묵적으로 요구하는데, minimal 구현이 이를 지원하지 못해서

### explicit interaction projection — 첫 번째 양성 신호

이 분석을 바탕으로 explicit interaction projection (xip)을 설계했다. concat의 early shared projection 장점을 유지하면서, interaction을 명시적으로 분리하는 구조.

결과:

- `concat_a:intensity_indexret IC mean = 0.0592`
- `xip_a:intensity_indexret IC mean = 0.0608`

처음으로 **concat_a를 넘는 숫자**가 나왔다. seed variance도 비슷하거나 약간 낫다:

- `concat_a:intensity_indexret IC std = 0.0226`
- `xip_a:intensity_indexret IC std = 0.0217`

하지만 pooled paired test: `p = 0.9242`. **통계적으로 유의하지 않다.**

이것은 winner가 아니라 promising candidate다. 하지만 6번의 실패 끝에 처음 나온 양성 방향이었다.

### 교훈

> **실패를 분석하다가 성공의 단서를 찾았다. method paper를 포기한 것이 오히려 더 깊은 질문으로 이어졌다.**

---

## 10. 교훈: 실패에서 배운 5가지

### 교훈 1: 이론적 우위 ≠ 실증적 우위

PE injection이 attention의 temporal metric을 직접 조절하니까 concat보다 낫다 — 이건 이론이다. 숫자는 달랐다. cycle_pe:intensity_only IC = −0.024 vs concat_a:intensity_only IC = +0.007. 이론이 숫자를 이긴 적은 한 번도 없다.

### 교훈 2: alignment + geometry ≠ prediction

alignment Spearman이 +0.5이고, qk_ord_rate가 0.992여도, IC는 0.000일 수 있다. representation을 바꾸는 것과 예측에 유용한 representation을 만드는 것은 다르다. 이 구분을 배우는 데 실험 20회와 GPU 시간 수십 시간이 들었다.

### 교훈 3: 실패 기록이 성공 기록보다 가치 있다

성공한 실험은 "이것이 됐다"로 끝난다. 실패한 실험은 "왜 안 됐는가"를 물어야 한다. 그리고 그 "왜"를 파다 보면 다음에 뭘 해야 하는지가 보인다. 채널 분해 → 상호작용 발견 → SNR 이론 → explicit interaction projection. 이 경로는 실패 분석 없이는 열리지 않았을 경로다.

### 교훈 4: 방향 전환은 포기가 아니다

method paper를 포기한 건 연구를 포기한 게 아니다. "learned τ가 concat_a를 이긴다"는 좁은 주장을 포기한 것이다. 그 대신 "왜 어떤 인터페이스가 더 잘 작동하는가"라는 더 넓고 더 흥미로운 질문을 얻었다. 그리고 applied paper의 confirmatory evidence (H2 high-vol MAE p = 0.0019)는 이미 충분히 강했다.

### 교훈 5: 부정적 결과도 논문이 된다

"learned economic time이 attention geometry를 바꿀 수 있다. 하지만 그것만으로는 예측 성능이 나아지지 않는다." 이것은 부정적 결과다. 하지만 **정보가 있는** 부정적 결과다. 왜 안 되는지, 어떤 조건에서 안 되는지, 대신 무엇이 작동하는지를 보여주니까. 실패를 정직하게 보고하는 논문은, 성공만 보고하는 논문보다 필드에 더 기여한다.

---

## 부록: 실패 타임라인 요약

| 순서 | 시도 | 핵심 수치 | 판정 |
|------|------|-----------|------|
| 1 | learned_tau (no reg) | Spearman = −0.50 | FAIL: 반대 방향 |
| 2 | learned_tau (with reg) | Spearman +0.5, IC = 0.035 | PARTIAL: alignment만 |
| 3 | QK ordinal loss | qk_ord_rate 0.992, IC ≈ 0.000 | FAIL: geometry ≠ prediction |
| 4 | strong geometry obj | qk_swap_delta 0.022, IC = −0.007 | FAIL: alignment 붕괴 |
| 5 | global-only ablation | learned:global 0.024 < static:global 0.035 | PARTIAL: 병목 확인, 해결 안 됨 |
| 6 | shape signature token | IC = −0.023 | FAIL: 반대 방향 |
| 7 | cycle_pe:intensity_only | IC = −0.024 vs concat 0.007 | FAIL: PE < concat |
| 8 | cycle_pe:intensity_embed | IC = 0.004 | PARTIAL: encoding 개선, 여전히 부족 |
| 9 | FiLM:intensity_indexret | IC = −0.008, seed range [−0.079, +0.076] | FAIL: 극심한 불안정 |
| 10 | xip_a:intensity_indexret | IC = 0.061 (p = 0.92 vs concat) | PROMISING: 첫 양성, 유의하지 않음 |

---

## 이 챕터를 닫으며

10번의 시도 중 명확한 성공은 없었다. 하지만 10번의 실패가 모여서 하나의 그림이 완성되었다.

> **concat_a가 강한 이유는 PE보다 나은 geometry를 만들어서가 아니다. 채널 간 비선형 상호작용을 input interface에서 자연스럽게 허용하기 때문이다.**

이 한 문장을 얻기 위해 10번의 실험이 필요했다. 그리고 이 한 문장이 다음 연구의 출발점이 되었다.

실패는 끝이 아니다. 실패는 질문이 정제되는 과정이다.

- "learned τ가 더 나은가?" → 아니다.
- "왜 아닌가?" → geometry가 아니라 signal pathway의 문제다.
- "그러면 무엇이 작동하는가?" → interaction-friendly interface다.
- "그것을 어떻게 만드는가?" → 이것이 다음 챕터의 질문이다.
