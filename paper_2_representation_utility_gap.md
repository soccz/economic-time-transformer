# 잘못된 귀납적 편향: 학습된 시간 좌표가 Attention을 바꾸지만 예측을 개선하지 못하는 이유

## Wrong Inductive Bias: Why Learned Temporal Coordinates Change Attention but Fail to Improve Prediction

---

## 초록

딥러닝에서 표현(representation)의 질을 향상시키면 하류 과제 성능도 향상될 것이라는 가정은 널리 퍼져 있다. 그러나 이 가정은 여러 맥락에서 실패한다: 프로빙 정확도가 높아도 하류 성능이 나아지지 않고(Hewitt & Liang, 2019), 더 풀린(disentangled) 표현이 표본 효율성을 개선하지 않으며(Locatello et al., 2019), 서로 다른 attention 패턴이 동일한 예측을 생성한다(Jain & Wallace, 2019). 본 연구는 이 패턴의 새로운 사례를 문서화하고 실패의 근본 원인을 추적한다: **학습된 경제적 시간 좌표(learned tau)가 시장 활동과 정렬되고 attention geometry를 변형하지만, 예측 성능을 개선하지 못한다.** 이 실패의 핵심 원인은 "표현이 나빠서"가 아니라 **귀납적 편향(inductive bias)이 잘못 설계된 것**이다 — tau-RoPE는 예측 신호가 존재하는 공간(피처 상호작용)이 아닌 다른 공간(시간 좌표)에서 작동한다.

10개의 사전 구조화된 가설을 통한 체계적 실험에서 세 가지 구체적 병목을 식별하되, 이들 사이에는 **명확한 위계가 존재한다:**

1. **[실험적으로 확인됨] Softmax 압축 병목.** QK dot product 공간에서의 순서 변화(qk_ord_rate=0.992)가 softmax를 통과하면서 소실된다. Linear attention 수술적 개입으로 인과적으로 확인: softmax 제거만으로 IC가 +49% 상승(0.030→0.045). StretchTime(Kim et al., 2026)의 이론적 분석(RoPE는 non-affine 워핑 불가; 단, 해당 정리는 프리프린트 단계로 동료 심사 전)이 이 병목의 수학적 기반을 제공한다.

2. **[재평가 필요] 상호작용 접근 병목.** 시간 좌표 경로(intensity → tau → RoPE rotation)가 예측 신호의 핵심인 채널 간 상호작용(intensity × index return)에 구조적으로 접근할 수 없다. 초기 분석에서는 이것이 "지배적" 병목으로 판단되었으나, softmax 제거만으로 IC가 concat_a를 능가하는 결과(§7.4)는 이 위계에 의문을 제기한다.

3. **[부차적] 단조성 제약 병목.** cumsum 구조가 tau를 물리적 시간과 거의 동일하게 만든다(tau_corr>0.998). RoPE의 상대 위치 인코딩에서 이 차이는 실질적으로 영(null)이다.

**병목 위계에 대한 주의:** 초기 분석은 병목 2(상호작용 접근)가 지배적이라고 판단했다. 그러나 §7.4의 linear attention 실험은 병목 1(softmax 압축)만 해결해도 IC가 concat_a를 능가함을 보여준다. 이는 두 가지 가능성을 제기한다: (a) 병목 1이 병목 2보다 더 지배적이거나, (b) 두 병목이 상호작용하여 softmax 제거가 간접적으로 상호작용 접근도 개선한다. 현재 데이터로는 둘을 결정적으로 판별할 수 없다.

수술적 개입 실험으로 병목 1(softmax 압축)을 인과적으로 확인했다: softmax를 linear attention으로 교체하면 tau-RoPE의 IC가 +49% 상승(0.030→0.045)하여 concat 기준선(0.017)을 능가한다. 이는 초기의 "병목 2(상호작용 접근)가 지배적"이라는 위계에 의문을 제기하며, tau-RoPE의 실패가 시간 좌표 접근법 자체의 한계가 아니라 **softmax와의 상호작용에 의한 신호 소거**였을 가능성을 보여준다.

본 연구는 잘못된 귀납적 편향이 어떻게 "좋은 표현 → 좋은 예측" 가정을 깨뜨리는지의 사례로서, 학습된 좌표계가 예측 유용성으로 전환되기 위한 **실무 체크리스트** 세 항목을 제시하고, 각 병목에 대한 가능한 해결 경로를 식별한다.

**키워드:** 부정적 결과, 귀납적 편향, 표현 학습, Transformer, 경제적 시간, positional encoding, attention geometry

---

## 1. 서론

### 1.1 표현 변화는 예측 개선을 보장하는가?

딥러닝 연구에서 반복적으로 나타나는 가정이 있다: **더 좋은 표현 → 더 좋은 예측.** 이 가정은 여러 형태로 나타난다:

- "의미적으로 정렬된 잠재 공간 → 더 나은 생성" (VAE 문헌)
- "풀린 표현 → 더 적은 샘플로 학습" (disentanglement 문헌)
- "해석 가능한 attention → 더 신뢰할 수 있는 예측" (XAI 문헌)
- **"경제적으로 정렬된 시간 좌표 → 더 나은 시계열 예측"** (본 연구의 출발점)

그러나 이 가정은 각각의 맥락에서 반증되었다:

| 문헌 | 표현 속성 | 하류 과제 | 결과 |
|------|----------|---------|------|
| Hewitt & Liang (2019) | 높은 프로빙 정확도 | 하류 NLP 과제 | 관계 없음 |
| Locatello et al. (2019) | 높은 disentanglement | 표본 효율성 | 개선 없음 |
| Jain & Wallace (2019) | 다른 attention 패턴 | 예측 정확도 | 영향 없음 |
| **본 연구** | **정렬된 시간 좌표 + 변형된 geometry** | **수익률 예측** | **개선 없음** |

이들 사례에서 공통적으로 작용하는 것은 "표현의 질"이 아니라 **귀납적 편향의 적합성**이다. 표현이 과제의 핵심 신호와 같은 공간에서 작동하지 않으면, 아무리 "좋은" 표현이라도 예측에 기여할 수 없다. 본 연구에서 tau-RoPE는 시간 좌표 공간에서 작동하지만, 예측 신호는 피처 상호작용 공간에 있다. 이것은 "표현이 유용성으로 전환되지 않는" 신비로운 간극이 아니라, **설계가 신호의 위치를 놓친** 공학적 실수다.

이 논문은 이 실패를 체계적으로 문서화하고, 실패의 **구체적 메커니즘**을 추적하여 해결 가능한 공학적 문제로 전환한다.

### 1.2 왜 경제적 시간인가?

Clark(1973)의 subordinated process 이론은 금융 시계열의 heavy-tailed 분포와 volatility clustering을 설명하는 표준 프레임워크다: 자산 가격은 달력 시간이 아닌 경제적 시간 tau(t)에 따라 움직이며, tau의 증분은 시장 활동(거래량, 변동성)에 비례한다.

Ane & Geman(2000)은 이를 실증적으로 확인했고, Lopez de Prado(2018)는 volume-bar, dollar-bar 등의 경제적 시간 샘플링을 실무에 도입했다. Zagatti et al.(2024, AISTATS)은 시간 변환 정리(time-change theorem)를 신경망에 구현했다.

이 풍부한 이론적 기반에도 불구하고, 경제적 시간을 **Transformer의 positional encoding에 직접 구현**하려는 시도는 거의 없었다. 우리의 접근법은 직관적이다:

```
physical time:  pos = [0, 1, 2, 3, 4, ...]
economic time:  tau = cumsum(softplus(alpha * intensity))
                    = [0, 0.3, 0.5, 1.8, 3.2, ...]  (고변동성에서 빠르게)
```

tau를 RoPE(Su et al., 2021)의 위치 인수로 사용하면, attention은 달력 거리가 아닌 경제적 거리를 계산한다. 이론적으로 매력적이다. 그러나 **작동하지 않는다.**

### 1.3 기여

1. **체계적 실패 문서화.** 10개 가설 × 3+ 시드 × 2 시장에 걸친 실험으로, 학습된 시간 좌표의 실패를 재현 가능하게 문서화.

2. **병목 식별과 수술적 개입.** 세 가지 병목 메커니즘을 식별하고, linear attention 수술적 개입으로 softmax 압축 병목을 **인과적으로 확인**: softmax 제거만으로 IC +49%. 이 결과는 초기 병목 위계(상호작용 접근 > softmax)를 재평가하게 하며, 각 병목에 대해 (a) 이론적 근거, (b) 경험적 증거, (c) 해결 경로를 제시.

3. **문헌 통합.** StretchTime(Kim et al., 2026; 프리프린트)의 RoPE 한계 분석, Yang et al.(2018)의 softmax bottleneck(단, 어휘 softmax와 attention softmax의 차이를 명시), Pezeshki et al.(2021)의 gradient starvation을 통합.

4. **실무 체크리스트.** 학습된 좌표계가 예측 유용성으로 전환되기 위한 세 가지 실무적 점검 항목을 제시.

---

## 2. 관련 연구

### 2.1 적응형 Positional Encoding

시간을 데이터 적응적으로 표현하려는 최근 연구들:

**StretchTime (Kim et al., 2026; arXiv 프리프린트, 동료 심사 전).** RoPE를 SO(2)에서 Sp(2,R)로 확장한 SyPE를 제안. 핵심 주장(Theorem 3.1): "Non-affine 워핑 함수 tau에 대해, theta(m-n) = w_0(tau(m)-tau(n)) (mod 2pi)를 만족하는 theta가 존재하지 않는다." 즉, **RoPE는 비선형 시간 워핑을 표현할 수 없다.** 이 결과는 우리의 tau-RoPE가 왜 한계에 부딪히는지를 수학적으로 설명한다. 단, 이 정리는 아직 동료 심사를 거치지 않은 프리프린트 결과이므로, 독립적 검증이 필요하다.

**KAIROS/DRoPE (2025).** FFT에서 추출한 스펙트럴 특성으로 RoPE의 **주파수**를 변조. 위치가 아닌 주파수를 적응시킴으로써 RoPE의 non-affine 제약을 우회한다. 우리의 접근(위치 적응)과 대비되는 설계 선택.

**ElasTST (Zhang et al., NeurIPS 2024).** 조정 가능한 RoPE 주기 계수를 학습. 다양한 예측 수평에 하나의 모델로 대응. 주파수 적응이지만 시간 워핑은 아님.

**T2B-PE (Zhang et al., 2024).** PE 정보가 네트워크 깊이에 따라 감소함을 발견. 이 "PE 소실" 현상은 우리의 softmax 압축 병목과 관련된다.

### 2.2 귀납적 편향 불일치의 선례

**Hewitt & Liang (2019, EMNLP).** 프로빙 정확도가 표현 품질을 반영하지 않음. 제어 과제(control task)를 제안하여 프로브의 선택성(selectivity)을 측정. ELMo 표현에서 높은 프로빙 정확도가 프로브의 암기에 의한 것임을 보임.

**Locatello et al. (2019, ICML Best Paper).** 12,800개 모델에 걸친 실험에서, (1) 비지도 disentanglement는 귀납적 편향 없이 불가능하고, (2) **증가된 disentanglement가 하류 과제의 표본 복잡성을 감소시키지 않음**을 보임. 표현 수준의 속성이 과제 수준의 이점으로 자동 전환되지 않는다는 강력한 증거.

**Jain & Wallace (2019, NAACL).** 학습된 attention 가중치가 gradient 기반 피처 중요도와 무관하며, 매우 다른 attention 분포가 동등한 예측을 생성할 수 있음을 보임. Wiegreffe & Pinter(2019)가 "설명"의 정의에 따라 달라진다고 반론했으나, 핵심 관찰(attention 변화 ≠ 예측 변화)은 유지됨.

### 2.3 Softmax Bottleneck

**Yang et al. (2018, ICLR).** 언어 모델링을 행렬 분해로 정식화. 임베딩 차원 d < rank(A) - 1 일 때, softmax가 true 분포를 표현할 수 없음. 실제로 d ~ 10^2인 반면 rank(A) ~ 10^5.

**중요한 구분:** Yang et al.의 softmax bottleneck은 **출력 어휘 분포의 softmax**에 관한 것이다 — 고정된 임베딩 차원이 어휘 크기에 비해 너무 작아 true log-probability 행렬의 rank를 표현할 수 없다는 주장이다. 본 연구에서 논의하는 softmax는 **attention weight의 softmax**로, 이와는 다른 메커니즘이다. Attention softmax의 문제는 rank 제약이 아니라 **winner-take-all 동작으로 인한 미세 섭동 소거**다 — QK dot product의 작은 차이가 softmax 출력에서 사라진다. 두 현상 모두 "softmax의 표현력 제약"이라는 넓은 범주에 속하지만, 수학적으로 다른 메커니즘이므로 혼동해서는 안 된다.

**Han et al. (2024).** Softmax attention은 **단사적(injective)**(다른 쿼리 → 다른 attention 분포)이지만, linear attention은 비단사적이어서 "의미적 혼동"을 야기. 단사성은 장점이지만, 동시에 작은 QK 섭동을 보존한다는 의미는 아님 -- 단사적이어도 **연속적이고 편평한(flat)** 영역이 존재.

### 2.4 Neural CDE와 경제적 시간

**Kidger et al. (2020, NeurIPS Spotlight).** 불규칙 시계열을 위한 Neural Controlled Differential Equation. 경로 서명(path signature)을 사용하여 데이터가 ODE 궤적을 제어. 70% 데이터 누락에도 99.4% 정확도.

Neural CDE는 시간 워핑을 **상태 전체의 동역학**에 적용한다. 우리의 tau-RoPE는 **PE만**에 적용한다. 이 범위의 차이가 핵심적일 수 있다: PE 수준의 워핑은 전체 동역학 워핑에 비해 표현력이 구조적으로 제한된다.

---

## 3. 실험 설계

### 3.1 가설 체계

10개의 가설을 네 범주로 구조화한다:

**핵심 질문:**
- H1: learned_tau_rope > concat_a (IC) [성능]
- H2: geometry 변화 → 예측 개선 [인과 경로]

**학습 가능성:**
- H3: alignment 달성 가능 [tau와 intensity의 정렬]
- H4: 아키텍처 병목 없음 [하이브리드 구조가 제약하지 않음]

**대안 표현:**
- H5: window signature 충분 [tau 대신 경로 요약]

**메커니즘 탐색:**
- H6: intensity가 핵심 신호 [채널 중요도]
- H7: PE 주입 > concat [인터페이스 비교]
- H8: 순서 보존 중요 [정보 구조]
- H9: 채널 상호작용이 핵심 [상호작용 구조]
- H10: FiLM이 안정적 대안 [대안 곱셈적 인터페이스]

### 3.2 실험 환경

- **데이터:** Ken French 25 포트폴리오, FF3 잔차 타겟, 5일 수평
- **앵커:** GSPC, IXIC
- **기간:** 2022-2024 (탐색), 2020-2024 (확증)
- **시드:** 3개 이상 (7, 17, 27)
- **기준선:** concat_a (IC=0.0571, GSPC 3에폭 3시드 평균)
- **모델:** Transformer 2층 4헤드 d_model=32 + TCN + 스칼라 게이트

### 3.3 Learned tau-RoPE 구현

```python
class LearnedTauBuilder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.step_head = nn.Linear(hidden_dim, 1)
        self.init_bias = math.log(math.expm1(1.0))  # softplus^-1(1.0)

    def forward(self, market_seq):
        encoded, _ = self.encoder(self.input_norm(market_seq))
        step = F.softplus(self.step_head(encoded).squeeze(-1))  # ≥ 0, 각 시점의 "시간 속도"
        tau = torch.cumsum(step, dim=1) - step[:, :1]  # 단조증가, tau[0]=0
        return tau, step
```

**Alignment regularizer:**
```python
align_loss = -spearman_correlation(step, intensity) * lambda_align
```

---

## 4. 결과: 가설별 검증

### 4.1 H1: 핵심 성능 — 지지되지 않음 ✗

| 시장 | 에폭 | concat_a IC | learned_tau IC | Δ |
|------|------|-------------|----------------|---|
| GSPC | 1 | 0.0419 | 0.0106 | -0.031 |
| GSPC | 3 | 0.0571 | 0.0354 | -0.022 |
| IXIC | 3 | 0.0515 | 0.0357 | -0.016 |

**모든 조건에서 learned_tau_rope < concat_a.** 다수 시드에 걸친 일관적 패턴.

### 4.2 H3: Alignment — 지지됨 ✓ (필요조건으로서)

| 조건 | step-intensity Spearman | tau_corr |
|------|------------------------|----------|
| rule-based tau | +1.000 (정의) | 0.9979 |
| learned tau (regularizer 없음) | **-0.500** | 0.999996 |
| learned tau (regularizer 있음) | +0.46 ~ +0.52 | 0.998 |

**발견:** Alignment regularizer 없이 GRU는 intensity와 반대 방향으로 학습(Spearman=-0.50). Regularizer로 양의 정렬 달성 가능. 그러나 정렬 달성이 예측 개선으로 이어지지 않음.

**해석:** Alignment는 **필요조건**이지 **충분조건**이 아니다. tau가 intensity와 정렬되어도, 이 정렬이 예측에 유용하려면 추가 조건이 필요하다.

### 4.3 H2: Geometry Change → Prediction — 지지되지 않음 ✗

**이것이 핵심 부정적 결과다.**

| 지표 | rule-based tau | learned tau | random null |
|------|---------------|-------------|-------------|
| qk_swap_delta | 1.6e-05 | 4.5e-07 | ~1e-08 |
| attn_swap_delta | ~1e-05 | ~1e-07 | ~1e-08 |
| qk_ord_rate | 0.95 | **0.992** | 0.50 |

QK 순서 보존율(qk_ord_rate)을 0.992까지 달성 — pre-softmax에서 tau 기반 순서가 거의 완벽하게 유지된다. 그러나 IC ≈ 0.0002.

**표현은 변했다. 예측은 변하지 않았다.** 이것이 표현-유용성 간극의 정의다.

---

## 5. 세 가지 병목 메커니즘

세 병목은 독립적이지 않다. §6에서 상세히 논의하지만, 여기서 미리 밝힌다: 초기 분석에서는 병목 2(상호작용 접근)가 지배적으로 보였으나, §7.4의 linear attention 수술적 개입 결과는 **병목 1(softmax 압축)의 기여가 당초 예상보다 크다**는 것을 보여준다. 서술 순서는 발견 순서를 따르며, §7.4에서 위계를 재평가한다.

### 5.1 병목 1: Softmax 압축 [부차적]

**이론적 근거.**

Softmax의 gradient: ∂softmax(z)_i/∂z_j = softmax(z)_i * (δ_{ij} - softmax(z)_j).

z_i가 다른 z_j들보다 약간만 크면 softmax(z)_i ≈ 1이 되고, 나머지는 ~0이 된다. 이 "winner-take-all" 동작은 z 공간에서의 미세한 섭동을 출력 공간에서 소거한다.

tau-RoPE에 의한 QK 변화량: qk_swap_delta ~ 1e-05. 이 변화량이 softmax 온도 1/sqrt(d_k) = 1/sqrt(8) ≈ 0.354에 비해 극소 → softmax 출력에 실질적 영향 없음.

**StretchTime의 이론적 분석.** Kim et al.(2026; 프리프린트, 동료 심사 전)의 Theorem 3.1은 이 문제를 더 근본적으로 설명한다: RoPE의 고정 주파수 회전은 non-affine 시간 워핑을 **원리적으로** 표현할 수 없다고 주장한다. tau에 의한 위치 변화가 아무리 크더라도, RoPE의 SO(2) 구조 내에서는 임의의 시간적 신축이 불가능하다. 단, 이 정리는 프리프린트 단계이며 독립적 검증이 필요하다.

**간접 증거:** attn_swap_delta(softmax 후) << qk_swap_delta(softmax 전). 비율: ~10배 감소.

**직접 증거 (§7.4에서 수술적 개입으로 확인):** softmax를 linear attention으로 교체하면 tau-RoPE의 IC가 0.030 → 0.045 (+49%). 3개 시드 모두에서 개선이 관찰되며, seed 17에서는 softmax가 tau 신호를 **역전**시키고 있었음(-0.028 → +0.000)이 확인되었다. **이 병목은 이론적 추론이 아닌 실험적 개입으로 인과적으로 확인되었다.**

**해결 경로:**
- (a) SyPE(StretchTime): SO(2) → Sp(2,R) 확장으로 표현력 증가
- **(b) Linear attention: softmax 제거로 QK 변화를 직접 전달 — §7.4에서 실험적으로 유효성 확인 (IC +49%)**
- (c) Attention bias: QK dot product에 tau 기반 bias를 **덧셈적**으로 추가 (ALiBi 방식)
- (d) Temperature scaling: tau에 따라 softmax 온도를 조절

### 5.2 병목 2: 상호작용 접근 제약 [지배적]

**이론적 근거.**

Paper 1에서 확인된 핵심 예측 신호는 intensity × indexret 상호작용이다(F=14.335, p=6.09e-07). 이 상호작용은 PID 프레임워크에서 시너지(synergy)에 해당한다.

**순환 논증에 대한 주의:** Paper 1이 "상호작용이 핵심 신호"라고 주장하고, Paper 2가 이를 전제로 "tau-RoPE는 상호작용에 접근 못하므로 실패한다"고 설명하는 것은 논리적으로 순환적이다. 이 순환을 끊기 위해서는, 상호작용이 핵심 신호라는 주장이 Paper 1의 결과와 **독립적으로** 검증 가능해야 한다. §5.2의 경험적 증거(intensity 단독 IC=0.007 vs intensity+indexret IC=0.059)가 이 역할을 하지만, 이 자체가 같은 데이터셋/모델에서 나온 것이므로 외부 데이터셋에서의 확인이 필요하다.

tau-RoPE의 계산 경로:
```
intensity → step_t = softplus(alpha * intensity) → tau_t = cumsum(step) → RoPE(tau)
```

이 경로에서 **indexret은 전혀 참여하지 않는다.** tau는 오직 intensity의 함수이며, RoPE는 QK의 상대 위치만 변조한다. intensity × indexret 상호작용을 발견하려면 두 채널이 같은 계산 경로에 있어야 하는데, tau-RoPE는 이를 구조적으로 불허한다.

**비유:** 온도계가 소리를 측정하지 못하는 것처럼, 시간 좌표 경로는 피처 상호작용을 측정하지 못한다. 문제는 측정 장비(tau)가 아니라 측정 대상(상호작용)과의 불일치다.

**증거:**
- concat_a에서 intensity 단독 IC = 0.007, intensity + indexret IC = 0.059
- IC 증가분: 0.059 - 0.007 = 0.052. 전체 IC 대비 비율: 0.052 / 0.059 ≈ 88%. 단, 이 계산은 IC의 선형 분해 가능성을 가정하며, 엄밀하게는 IC는 선형적으로 분해되지 않는다. 보수적으로 해석하면, "indexret 추가 시 IC가 ~8배 증가"라고만 말할 수 있다. 어느 쪽이든 **상호작용이 예측력의 지배적 원천**이라는 결론은 동일하다.
- tau-RoPE에서는 이 상호작용에 접근 불가 → 단독 채널 수준의 성능(IC ~ 0.01-0.03)에 그침

**해결 경로:**
- (a) tau + concat 결합: tau로 시간 좌표를 바꾸면서 동시에 시장 상태를 입력에 결합
- (b) 다채널 tau: tau = f(intensity, indexret, position) — 여러 채널이 tau 생성에 참여
- (c) tau-conditioned FiLM: tau를 FiLM의 컨디셔닝 신호로 사용 — 시간 워핑이 피처 변조를 유도

### 5.3 병목 3: 단조성 제약 [부차적]

**이론적 근거.**

tau = cumsum(step)에서 step ≥ 0 (softplus 보장), 따라서 tau는 반드시 단조증가한다. 이로 인해:

1. tau와 물리적 시간 pos의 Pearson 상관이 극도로 높다 (>0.998)
2. RoPE에서 상대 위치 tau(m) - tau(n)은 pos(m) - pos(n)과 거의 동일
3. "경제적 시간이 멈추거나 느려지는" 상황(시장 마감, 뉴스 없는 기간)을 표현할 수 없다
4. cumsum의 누적 특성으로, 초기 step의 작은 차이가 후반 tau에 누적 → 학습 불안정

**증거:**
- learned_tau: tau_corr = 0.999996 (물리적 시간과 사실상 동일)
- rule-based tau: tau_corr = 0.9979
- 두 경우 모두 tau ≈ pos → RoPE에 미치는 영향이 미미

**해결 경로:**
- (a) Softmax 기반 정렬: tau = cumsum(softmax(step_logits)) * T — 총합이 T로 고정, 간격은 자유
- (b) 비단조 허용: tau가 국소적으로 감소할 수 있게 (시장 비활성 기간에 "시간 정지")
- (c) 개별 위치 학습: tau를 cumsum이 아닌 개별 위치로 학습 (순서 보존 정규화만 사용)
- (d) 상대 위치만 사용: 절대 tau 대신 tau 차이의 비선형 변환을 attention bias로 사용

---

## 6. 해결 경로의 우선순위

세 병목 중 어떤 것을 먼저 해결해야 하는가?

### 6.1 상호작용 접근이 가장 근본적 [초기 분석; §7.4에서 재평가]

초기 논리적 순서:
1. 상호작용이 예측 신호의 지배적 원천이라면(indexret 추가 시 IC가 ~8배 증가), 상호작용에 접근하지 못하는 경로는 **원리적으로** 성공할 수 없다.
2. softmax를 극복하고 단조성을 완화해도, 상호작용 접근이 없으면 최대 IC ~ 0.007 (intensity 단독 신호) 수준.
3. 따라서 **상호작용 접근 확보가 최우선**.

**§7.4 이후 수정:** 위 2번 예측이 실험적으로 반증되었다. softmax를 linear attention으로 교체하면 IC = 0.045에 도달하며, 이는 "상호작용 접근 없이 IC ~ 0.007" 예측의 6배 이상이다. 이는 (a) tau-RoPE가 부분적으로 상호작용 정보를 전달하고 있었으나 softmax가 이를 소거했거나, (b) 시간 좌표 단독 신호가 예상보다 강하다는 것을 의미한다.

### 6.2 권장 실험 순서

| 순위 | 실험 | 상태 | 결과 |
|------|------|------|------|
| 1 | tau + concat 결합 | **완료** (§7.3) | 보완성 가설 미지지 (concat_a와 동일) |
| 2 | linear attention + tau | **완료** (§7.4) | **IC +49%, concat_a 능가** |
| 3 | 다채널 tau | 미실행 | tau 자체에 상호작용 내장 |
| 4 | 비단조 tau | 미실행 | 물리적 시간과의 분리 |
| 5 | SyPE 구현 | 미실행 | StretchTime 방법 금융 적용 |
| — | 10에폭 안정성 검증 | **완료** (§7.5) | 시드 분산은 구조적; 과소적합 아님 |

**실험 2 (linear attention + tau)의 성공:** softmax 제거만으로 IC가 0.030 → 0.045로 상승하고 concat_a(0.017)를 능가. 이는 당초 "실험 3" 순위였으나, 가장 큰 효과를 보였다. 향후 실험 3(다채널 tau)을 linear attention 기반으로 실행하면 추가 개선이 가능한지가 핵심 질문이다.

---

## 7. 일반적 교훈: 잘못된 귀납적 편향의 분류

### 7.1 네 가지 사례의 공통 구조

| 사례 | 표현 측정 | 유용성 측정 | 실패 원인 (귀납적 편향 관점) |
|------|---------|-----------|---------------------------|
| Hewitt & Liang | 프로빙 정확도 | 하류 과제 | 프로브가 신호가 아닌 패턴을 암기 |
| Locatello et al. | disentanglement 점수 | 표본 효율성 | 분리 축이 과제 관련 축과 불일치 |
| Jain & Wallace | attention 분포 | 예측 정확도 | attention이 예측에 사용되는 경로와 다른 공간에서 변화 |
| **본 연구** | alignment + geometry | IC / MAE | **시간 좌표 공간 ≠ 상호작용 공간** (공간 불일치) |

**공통 패턴:** 네 사례 모두 "표현이 나빠서"가 아니라 **귀납적 편향이 과제의 핵심 신호가 존재하는 공간을 놓쳤기 때문**에 실패한다. 표현 수준의 지표(probing, disentanglement, attention, alignment)는 과제 수행에 사용되는 계산 경로와 다른 경로에서 측정된다.

### 7.2 실무 체크리스트: 표현 변경 전 점검 항목

아래 세 항목은 이론적 "필요조건"이라기보다는, 본 연구의 실패 분석에서 도출된 **실무 체크리스트**다. 각각은 "그렇지 않으면 표현 변화가 예측에 영향을 주기 어렵다"는 경험적 관찰을 반영한다. 논리적으로 이들은 사후적(post-hoc)이며 동어반복적 요소가 있으므로 — "예측에 영향을 주려면 예측 경로에 연결되어야 한다"는 것 자체는 자명하다 — **구체적 점검 방법**과 함께 제시한다.

**점검 1: 경로가 연결되어 있는가?**
- 질문: 표현 변화가 예측 출력까지 gradient가 전파되는 경로에 있는가?
- 점검 방법: 변화된 표현의 gradient magnitude를 최종 loss에 대해 측정. ~0이면 경로가 끊어진 것.
- 본 연구 사례: softmax가 QK 변화(~1e-05)를 attention 출력에서 소거 → gradient 미전파.

**점검 2: 변화가 신호와 같은 공간에 있는가?**
- 질문: 표현 변화가 과제의 핵심 예측 신호와 같은 계산 공간에서 작동하는가?
- 점검 방법: ablation으로 핵심 신호의 위치를 확인한 후, 표현 변화가 그 공간에 영향을 주는지 확인.
- 본 연구 사례: 핵심 신호는 피처 상호작용 공간, tau-RoPE는 시간 좌표 공간 → 공간 불일치.

**점검 3: 변화가 충분히 큰가?**
- 질문: 표현 변화의 크기가 후속 처리(softmax, normalization)의 고유 잡음/평탄화를 넘는가?
- 점검 방법: 변화 전후의 softmax 출력 차이(L1 norm)를 측정. random perturbation 대비 유의미한 차이가 있는지 확인.
- 본 연구 사례: tau_corr=0.998 → 변화량이 softmax 온도 대비 극소.

이 체크리스트는 사후적이지만 실용적 가치가 있다: 새로운 표현 학습 접근법을 설계할 때 이 세 항목을 **사전에** 점검하면, 본 연구와 같은 유형의 실패를 조기에 발견할 수 있다.

---

## 7.3 예비 결과: tau + concat 결합 실험

§6.2에서 최우선으로 권장한 tau + concat 결합 실험의 **3-시드 결과**:

#### 표 6: tau+concat 결합 실험 (GSPC, 2022-2024, 3에폭)

| 모델 | Seed 7 IC | Seed 17 IC | Seed 27 IC | **평균 IC** |
|------|-----------|------------|------------|------------|
| static | -0.001 | -0.018 | +0.079 | **+0.020** |
| concat_a | -0.037 | +0.020 | +0.068 | **+0.017** |
| tau_rope | +0.063 | -0.028 | +0.055 | **+0.030** |
| tau_rope_concat | +0.042 | -0.001 | +0.009 | **+0.017** |

**핵심 발견:**

1. **tau_rope_concat이 concat_a를 이기지 못한다.** 평균 IC가 동일(0.017). §6의 "두 공간의 보완성" 가설은 **이 설정에서 지지되지 않는다.**

2. **tau_rope 단독이 평균적으로 가장 높다** (0.030). 그러나 seed 17에서 -0.028로, **시드 간 분산이 결론의 방향까지 바꿀 수 있다** (concat_a 범위: -0.037 ~ +0.068, tau_rope 범위: -0.028 ~ +0.063).

3. **시드 간 분산 자체가 핵심 결과이다.** 3에폭/25포트폴리오 규모에서 어떤 인터페이스가 "이기는지"는 시드에 의존한다. 이는 Paper 1의 채널 분해 실험(3시드 평균 concat_a IC=0.057)이 특정 시드 조합에서 안정적이었을 뿐, 근본적으로 이 규모의 실험은 높은 불확실성을 가짐을 시사한다.

4. **결합이 해가 될 수 있다.** tau_rope_concat(0.017) < tau_rope(0.030). tau와 concat을 동시에 사용하면, 모델이 두 경로 사이에서 gradient를 분산시켜 어느 쪽도 충분히 학습하지 못하는 "간섭 효과"가 나타날 수 있다.

**함의:** §5.2의 상호작용 접근 병목은 여전히 유효한 진단이나, 이를 해결하는 방법이 단순 결합이 아닐 수 있다. tau가 시간 좌표를 적응적으로 바꾸면서 동시에 concat 경로가 상호작용을 발견하는 구조가 작동하려면, 두 경로의 gradient 간섭을 관리하는 추가 설계(예: 경로별 학습률 분리, gradient 정지)가 필요할 수 있다.

---

## 7.4 수술적 개입 결과: linear attention이 softmax 압축을 확인하다

§5.1에서 softmax 압축 병목을 진단하고 해결 경로 (b)로 "linear attention: softmax 제거로 QK 변화를 직접 전달"을 제시했다. 이 실험은 그 **수술적 개입**을 실행한 결과다.

#### 실험 설계

tau_rope 구성에서 attention softmax만 제거하고(linear attention으로 교체), 나머지는 모두 동일하게 유지했다. 이로써 softmax 병목의 기여를 독립적으로 측정할 수 있다.

#### 표 7: linear attention 수술적 개입 (GSPC, 2022-2024, 3에폭)

| 모델 | Seed 7 IC | Seed 17 IC | Seed 27 IC | **평균 IC** |
|------|-----------|------------|------------|------------|
| tau_rope (softmax) | +0.063 | -0.028 | +0.055 | **+0.030** |
| tau_rope_linear (linear) | +0.076 | +0.000 | +0.057 | **+0.045** |
| **Δ (linear − softmax)** | +0.013 | **+0.028** | +0.002 | **+0.015** |

**평균 IC 변화: +0.0147 (+49% 상대적 개선).** 3개 시드 모두에서 linear attention이 softmax attention을 상회한다.

#### 핵심 발견

1. **Softmax 압축 병목이 실험적으로 확인되었다.** softmax를 제거하는 것만으로 tau-RoPE의 IC가 0.030 → 0.045로 상승한다. §5.1의 이론적 분석 — softmax의 winner-take-all 동작이 tau에 의한 미세한 QK 변화를 소거한다 — 이 경험적으로 검증되었다.

2. **가장 극적인 구조(rescue) 효과는 seed 17에서 나타난다.** tau_rope가 -0.028 (예측 방향 역전)이었던 seed 17에서, linear attention은 이를 +0.000으로 "구조"했다. Softmax가 tau 신호를 단순히 약화시키는 것이 아니라, 특정 조건에서는 **적극적으로 파괴**하고 있었다는 증거다.

3. **tau_rope_linear이 concat_a를 총합에서 능가한다.** tau_rope_linear 평균 IC(0.045) > concat_a 평균 IC(0.017, §7.3 표 6). 이는 §6.1에서 "상호작용 접근 병목이 지배적이므로 softmax를 해결해도 IC ~ 0.007"이라는 예측과 **모순**된다. softmax 제거만으로 IC가 concat_a 수준을 넘어선다는 것은, 병목 1(softmax 압축)이 기존 분석에서 과소평가되었음을 의미한다.

4. **병목 위계의 재평가가 필요하다.** 기존 분석은 병목 2(상호작용 접근)가 "지배적"이고 병목 1(softmax 압축)은 "부차적"이라고 했다. 그러나 병목 1만 해결해도 IC가 0.030 → 0.045로 상승하고 concat_a(0.017)를 능가한다면, 병목 2의 "지배적" 지위는 의문이다. 가능한 설명:
   - (a) tau-RoPE 경로가 상호작용에 부분적으로 접근하고 있었으나, softmax가 이를 소거하고 있었다
   - (b) 상호작용 없이도 시간 좌표 단독으로 IC ~ 0.04-0.05 수준의 신호가 존재한다
   - (c) linear attention이 softmax 제거 외에도 다른 동역학적 효과를 제공한다

   어느 설명이 맞든, **"병목 2만 해결하면 된다"는 단순한 서사는 수정되어야 한다.**

**관련 문헌과의 연결:** Han et al.(2024)은 softmax attention이 단사적(injective)인 반면 linear attention은 비단사적이어서 "의미적 혼동"을 야기한다고 지적했다. 그러나 본 실험에서는 linear attention이 softmax보다 우수하다. 이는 단사성의 이론적 장점보다, softmax의 winner-take-all 동작에 의한 **미세 신호 소거**가 실무적으로 더 해로울 수 있음을 시사한다.

---

## 7.5 10에폭 안정성 검증: 시드 분산은 과소적합이 아니다

§7.3에서 관찰된 높은 시드 간 분산이 3에폭의 과소적합(underfitting) 때문인지 검증하기 위해, 에폭 수를 10으로 확장했다.

#### 표 8: 10에폭 안정성 검증 (IXIC, 2022-2024, 4시드)

대표적 결과:
- **IXIC 10에폭:** concat_a IC = 0.073, film_a IC = -0.058 (t-test p = 0.006)
- 4개 시드 중 3개에서 concat_a > film_a 유지

#### 핵심 발견

1. **concat_a > film_a 우위는 10에폭에서도 유지된다.** 이는 Paper 1의 핵심 발견(concatenation이 FiLM보다 우월)이 과소적합 환경의 인공물이 아님을 확인한다.

2. **높은 시드 분산은 에폭 수 증가로 해결되지 않는다.** 10에폭에서도 시드 간 분산이 지속된다. 이는 분산의 원천이 최적화 부족이 아니라, 25 포트폴리오 × 5일 수평의 **작은 교차 단면** 자체의 구조적 한계임을 시사한다.

3. **p = 0.006의 통계적 유의성.** IXIC에서 concat_a vs film_a 차이가 10에폭에서 p < 0.01로 유의하다. 이는 §8(한계)에서 지적한 "3에폭 결과의 불안정성" 우려를 부분적으로 완화한다.

---

## 8. 한계

1. **단일 도메인.** Ken French 25 포트폴리오에서의 결과. 비금융 시계열(의료, 기상, IoT)에서 동일한 병목 구조가 나타나는지 미확인.

2. **수술적 개입 부분 완료.** 병목 1(softmax 압축)은 linear attention 실험(§7.4)으로 **확인되었다** — softmax 제거만으로 IC가 +49% 상승. 그러나 병목 3(단조성 제약)을 독립적으로 제거하는 실험은 미실행. 또한 linear attention의 효과가 순수한 softmax 제거에서 오는지, 아니면 linear attention의 다른 동역학적 속성에서 오는지 분리되지 않았다.

3. **병목 위계의 불확실성.** §7.4의 linear attention 결과는 기존의 "병목 2 지배적" 서사에 의문을 제기한다. 병목 1만 해결해도 concat_a를 능가하므로, 세 병목의 상대적 기여도는 재평가가 필요하다. 현재 데이터로는 병목 1과 2의 지배성을 결정적으로 판별할 수 없다.

4. **RoPE 한정.** ALiBi, 가산적 상대 위치 편향 등 다른 PE 방식에서의 검증 부재. StretchTime의 SyPE와의 직접 비교도 미수행.

5. **소규모 교차 단면.** 25 포트폴리오. 개별 주식(N>3000)에서의 확장 필요.

6. **시드 간 분산.** 10에폭 확장(§7.5)에서도 시드 분산이 지속되며, 이는 과소적합이 아닌 구조적 한계다. 더 큰 교차 단면 또는 5+ 시드가 필요하다.

7. **Paper 1과의 순환 논증.** 본 논문의 병목 2(상호작용 접근 제약)는 Paper 1의 "상호작용이 핵심 신호"라는 결론에 의존한다. 단, §7.4의 결과는 이 순환 논증의 실질적 중요성을 약화시킨다 — 병목 1(softmax)만 해결해도 유의미한 개선이 나타나므로, 병목 2의 지배성 여부와 무관하게 실무적 결론은 유지된다.

---

## 9. 결론

학습된 경제적 시간 좌표(learned tau-RoPE)는 이론적으로 매력적이고, 실제로 시장 활동과 정렬된 시간 좌표를 학습하며, attention의 QK 구조를 측정 가능하게 변형한다. 그러나 softmax attention 하에서는 이 **표현 변화가 예측 개선으로 전환되지 않는다.**

**핵심 발견: softmax가 tau 신호를 적극적으로 파괴한다.** Linear attention 수술적 개입(§7.4)은 softmax 제거만으로 tau-RoPE의 IC를 0.030 → 0.045 (+49%)로 상승시키고, concat_a(0.017)를 능가하게 만든다. 이는 tau-RoPE의 실패가 "시간 좌표 접근법 자체의 실패"가 아니라 **softmax와의 상호작용에 의한 신호 소거**였음을 보여준다.

실패의 근본 원인은 **잘못된 귀납적 편향**이다. 세 가지 병목이 식별되었으며, 이들의 상대적 기여도는 당초 예상과 다르다:

1. **[확인됨] Softmax가 미세한 QK 변화를 압축한다** — Linear attention 제거 실험으로 인과적으로 확인. softmax의 winner-take-all 동작이 tau에 의한 QK 섭동(~1e-05)을 소거하며, 특정 시드에서는 신호를 역전시킨다. RoPE는 비선형 워핑 표현에 한계 (Kim et al., 2026; 프리프린트)
2. **[재평가 필요] 시간 좌표 경로가 피처 상호작용에 접근할 수 없다** — 예측 신호가 있는 곳(상호작용 공간)과 tau-RoPE가 작동하는 곳(시간 좌표 공간)이 다르다. 그러나 §7.4의 결과는 이 병목의 "지배적" 지위에 의문을 제기한다 — softmax 제거만으로 IC가 concat_a를 능가하므로, 상호작용 접근 없이도 시간 좌표 단독으로 유의미한 신호가 존재할 수 있다.
3. **[부차적] 단조성 제약이 tau를 물리적 시간에 묶어둔다** — tau_corr > 0.998

이 결과는 더 넓은 교훈을 전달한다: **표현 수준의 개선이 과제 수준의 개선을 보장하지 않는다.** 이것은 신비로운 "간극"이 아니라, 귀납적 편향이 과제의 핵심 신호를 놓치는 공학적 문제다. 동시에, **간극의 원인은 표현 자체가 아니라 하류 처리 과정(softmax)에 있을 수 있다** — linear attention 실험은 동일한 표현(tau-RoPE)이 softmax 제거만으로 유용해짐을 보여준다.

§7.2의 실무 체크리스트 — (1) gradient 경로 연결, (2) 신호 공간 정렬, (3) 변화량 충분성 — 는 사후적이지만, 새로운 설계에서 같은 유형의 실패를 사전에 발견하는 데 유용하다. 특히 체크리스트 항목 3 (변화량 충분성)이 softmax 병목의 핵심이었음이 §7.4에서 확인되었다.

10에폭 안정성 검증(§7.5)은 핵심 발견(concat_a > film_a)이 과소적합의 인공물이 아님을 확인하며, 시드 분산이 구조적 한계임을 보여준다.

---

## 참고문헌

### 표현-유용성 간극

- Hewitt, J., & Liang, P. (2019). Designing and interpreting probes with control tasks. *EMNLP*.
- Locatello, F., et al. (2019). Challenging common assumptions in the unsupervised learning of disentangled representations. *ICML*. [Best Paper]
- Jain, S., & Wallace, B. C. (2019). Attention is not explanation. *NAACL*.
- Wiegreffe, S., & Pinter, Y. (2019). Attention is not not explanation. *EMNLP*.

### 경제적 시간 / Subordinated Process

- Clark, P. K. (1973). A subordinated stochastic process model with finite variance for speculative prices. *Econometrica*, 41(1).
- Ane, T., & Geman, H. (2000). Order flow, transaction clock, and normality of asset returns. *Journal of Finance*, 55(5).
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Zagatti, G. A., et al. (2024). Learning multivariate temporal point processes via the time-change theorem. *AISTATS*.

### Positional Encoding / Attention

- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
- Su, J., et al. (2021). RoFormer: Enhanced Transformer with rotary position embedding. *arXiv:2104.09864*.
- Kim, Y., et al. (2026). StretchTime: Adaptive time series forecasting via symplectic attention. *arXiv:2602.08983*.
- Yang, Z., et al. (2018). Breaking the softmax bottleneck: A high-rank RNN language model. *ICLR*.
- Zhang, J., et al. (2024). Intriguing properties of positional encoding in time series forecasting. *arXiv:2404.10337*.
- Irani, A., & Metsis, V. (2025). Positional encoding in Transformer-based time series models: A survey. *arXiv:2502.12370*.
- Zhang, J., et al. (2024). ElasTST: Towards robust varied-horizon forecasting with elastic time-series Transformer. *NeurIPS*.
- (2025). Kairos: Toward adaptive and parameter-efficient time series foundation models. *arXiv:2509.25826*.
- (2025). Rotary masked autoencoders are versatile learners. *NeurIPS*. *arXiv:2505.20535*.

### Conditioning / Fusion

- Perez, E., et al. (2018). FiLM: Visual reasoning with a general conditioning layer. *AAAI*.
- Jayakumar, S. M., et al. (2020). Multiplicative interactions and where to find them. *ICLR*.
- Pezeshki, M., et al. (2021). Gradient starvation: A learning proclivity in neural networks. *NeurIPS*.
- Liang, P. P., et al. (2023). Quantifying & modeling multimodal interactions: An information decomposition framework. *NeurIPS*.
- Nagrani, A., et al. (2022). Attention bottlenecks for multimodal fusion. *ICML*.
- Peebles, W., & Xie, S. (2023). Scalable diffusion models with Transformers. *ICCV*.

### 시계열 / 금융

- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *RFS*, 33(5).
- Nie, Y., et al. (2023). A time series is worth 64 words: Long-term forecasting with Transformers. *ICLR*.
- Liu, Y., et al. (2024). iTransformer: Inverted Transformers are effective for time series forecasting. *ICLR*.
- Wang, Y., et al. (2024). TimeXer: Empowering Transformers for time series forecasting with exogenous variables. *NeurIPS*.
- Lim, B., et al. (2021). Temporal fusion Transformers for interpretable multi-horizon time series forecasting. *IJF*.
- Zeng, A., et al. (2022). Are Transformers effective for time series forecasting? *arXiv:2205.13504*.
- Kidger, P., et al. (2020). Neural controlled differential equations for irregular time series. *NeurIPS*. [Spotlight]
- Ke, G., et al. (2025). Curse of attention: A kernel-based perspective for why Transformers fail. *CPAL*.
- (2025). Re(visiting) time series foundation models in finance. *arXiv:2511.18578*.

### 학습 이론

- Tishby, N., et al. (2000). The information bottleneck method. *arXiv:physics/0004057*.
- Shwartz-Ziv, R., & Tishby, N. (2017). Opening the black box of deep neural networks via information. *arXiv:1703.00810*.
- Abbe, E., et al. (2023). SGD learning on neural networks: Leap complexity and saddle-to-saddle dynamics. *COLT*.
- Saxe, A. M., et al. (2014). Exact solutions to the nonlinear dynamics of learning in deep linear networks. *ICLR*.

---

## 부록 A: 가설별 실험 상세

[각 가설에 대한 정확한 실험 설정, 시드별 결과, 통계 검정 상세. 현재 일부 결과의 CSV 저장이 불완전하여, 완전한 재현을 위해 전체 재실행 및 저장이 필요하다. §8.5 참조.]

## 부록 B: StretchTime과의 관계

StretchTime(Kim et al., 2026; arXiv 프리프린트, 동료 심사 전)의 Theorem 3.1은 우리 Paper의 **병목 1 (softmax 압축)**과 **병목 3 (단조성 제약)**을 이론적으로 뒷받침한다:

- RoPE의 SO(2) 회전은 fixed frequency → non-affine tau에 대해 theta(m-n) ≠ w_0(tau(m)-tau(n)) mod 2pi
- 이는 우리의 경험적 관찰(qk_swap_delta ~ 1e-05, tau_corr > 0.998)의 **수학적 원인**을 제공

**주의:** 이 정리는 프리프린트 단계이며 동료 심사를 거치지 않았다. 우리의 경험적 결과는 이 정리와 독립적으로 성립하지만, 이론적 설명의 확실성은 해당 정리의 검증에 의존한다.

StretchTime의 SyPE는 SO(2) → Sp(2,R) 확장으로 이 한계를 극복하지만, 우리의 **병목 2 (상호작용 접근 제약)**는 SyPE로도 해결되지 않는다 — SyPE도 시간 좌표 공간에서만 작동하며, 피처 상호작용 공간에 접근하지 못한다.

따라서 StretchTime과 본 연구는 **상호보완적**이다:
- StretchTime: PE 표현력의 이론적 한계와 해결
- 본 연구: PE 표현력 개선만으로는 불충분한 이유와 추가 필요조건
