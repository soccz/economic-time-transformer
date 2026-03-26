# paper_final_2.md — 실험·결과·심사 방어·부록
# (paper_final_1.md 이어서)

---

## 5. 실험 설계

### 5.1 검증 순서 (논문 리스크 최소화)

```
Step 1: H1-Vol 확정 (경제학)
  → 회귀식 사전 고정 → bootstrap/HAC 재검정
  → 통과 시 Step 2로 진행

Step 2: H2 검정 (ML 예측)
  → Cycle-PE vs Static vs Concat-A walk-forward
  → DM test + fold-wise IC 일관성

Step 3: H3 검정 (분포/불확실성)
  → CVAE 체제별 CRPS / PI-80 coverage
  → reliability diagram
```

H1이 약하면 H2/H3의 동기가 흔들린다. 순서를 지키는 것이 논문 리스크 관리의 핵심이다.

### 5.2 평가 지표

| 층위 | 지표 | 설명 |
|------|------|------|
| 점 예측 | IC | Spearman rank correlation (예측 vs 실현) |
| 점 예측 | ICIR | IC / std(IC), 안정성 |
| 점 예측 | MAE | 절대 오차 |
| 분포 예측 | CRPS | pinball loss 근사 (τ = 0.1, 0.25, 0.5, 0.75, 0.9) |
| 분포 예측 | PI-80 coverage | 80% 예측구간 실제 포함률 (목표: 0.80) |
| 분포 예측 | Reliability diagram | 캘리브레이션 시각화 |
| 경제 성과 | L/S 포트폴리오 SR | 분리 보고 (과대추정 방지) |
| 경제 성과 | DSR | Deflated Sharpe Ratio (다중검정 보정) |

**Purged CV / embargo / DSR**는 금융 백테스트 과대추정 방어에 효과적이다.

### 5.3 Ablation 구성

| 모델 | PE mode | Decoder | 목적 |
|------|---------|---------|------|
| M0 | static | CVAE | H2 baseline |
| M1 | concat_a | CVAE | H2 ablation A |
| M2 | cycle_pe | CVAE | H2 main |
| M3 | cycle_pe_full | CVAE | appendix |
| M4 | cycle_pe | GAN | decoder ablation |

M0 vs M2: H2 주 검정 (DM test)
M1 vs M2: Cycle-PE novelty 방어 (Concat-A 대비)
M2 vs M4: CVAE vs GAN 안정성 비교

### 5.4 H2 DM 검정 절차

```
diff_t = CRPS_static_t - CRPS_cycle_pe_t  (per prediction)
DM_t   = mean(diff) / (std(diff) / sqrt(n))
p      = 2 * (1 - Φ(|DM_t|))
```

fold-wise 일관성: 각 walk-forward fold에서 Cycle-PE가 Static PE를 이기는 비율 보고.

### 5.5 H3 체제별 분석 절차

```
for regime in [Bear/quiet, Bear/volatile, Bull/quiet, Bull/volatile]:
    CRPS_regime    = mean(CRPS | regime)
    PI80_coverage  = mean(1(y ∈ [q10, q90]) | regime)
    spread_proxy   = std(y_true | regime)  # 실현 변동성
```

H3 핵심 예측: Bear/volatile에서 CRPS가 가장 높고(예측 어려움), PI-80 coverage가 가장 낮거나 분포가 가장 넓어야 한다.

---

## 6. 예비 결과 및 논의

### 6.1 H1-Pos: 지지되지 않음 (null result)

| 단면 | b (cycle_position) | p값 | 판정 |
|------|-------------------|-----|------|
| 25 Size–B/M | — | ≈0.873 | 기각 |
| 49 Industry | — (부호 양수) | ≈0.241 | 기각 |

추세 신호 단독의 선형 조절 가설은 약하다. 이를 "null result"로 명시하고 논문에 포함하는 것이 오히려 신뢰도를 높인다.

### 6.2 H1-Vol: 체제 평균에서 강한 패턴 (탐색적)

| 체제 | λ₁,t 평균 | 특징 |
|------|-----------|------|
| Bear/quiet | 최대 | 모멘텀 프리미엄 가장 강함 |
| Bear/volatile | 급락 (일부 음수) | 모멘텀 크래시 |
| Bull/quiet | 중간 | |
| Bull/volatile | 중간~낮음 | |

25 Size–B/M, 49 Industry 두 단면에서 정성적으로 일관. Daniel & Moskowitz(2016) 모멘텀 크래시 문헌과 방향성 일치.

**현재 상태**: 탐색적 발견. Step 2-D 회귀식(상호작용항 d)으로 사전 고정 후 재검정 필요.

### 6.3 기술적 리스크: generated target + endogeneity

`|mom|`과 `se(β̂_MKT)` Spearman 상관이 유의하게 나온 경우, 잔차 타겟이 β 추정오차와 얽힐 수 있다. 이는 IV 하자로만 끝내면 안 되고 §4.1의 방어 전략 중 최소 2개를 본문에 보고해야 한다.

---

## 7. 심사 방어 체크리스트

아래는 리젝 사유로 직결되는 공격 포인트와 대응 전략이다.

### (A) H1-Vol: 탐색→확증 전환 절차

**공격**: "post-hoc binning — 체제 평균표를 보고 나서 가설을 만든 것 아닌가?"

**방어**:
1. H1-Vol 회귀식(Step 2-D)을 사전 고정하고 bootstrap/HAC로 재검정
2. 두 단면(25 포트폴리오, 49 Industry)에서 동일 방향 확인
3. 예비 분석과 확증 분석을 명확히 분리해서 서술

### (B) Generated residual target 방어

**공격**: "잔차가 추정 오차를 포함하므로 결과가 아티팩트일 수 있다"

**방어 (최소 2개 필수)**:
- β-window 민감도: 60일 vs 120일 결과 비교
- ridge β: OLS 아티팩트 방어
- 포트폴리오 vs 종목: 정성적 결론 일치 여부
- raw return 부록: H1-Vol이 잔차 없이도 유사 패턴인지

### (C) Cycle-PE novelty 방어

**공격**: "state를 입력에 concat하는 것과 뭐가 다른가?"

**방어**:
- Concat-A(M1) vs Cycle-PE(M2) ablation을 반드시 포함
- 이론적 차이 명시: PE 주입은 attention score 계산 자체에 state가 개입하지만, concat은 모델이 학습을 통해 반영하는 간접 경로
- DM test에서 M1 vs M2가 유의하면 novelty 방어 완성

### (D) CVAE 필요성 방어

**공격**: "점 예측 모델로 충분하지 않은가?"

**방어**:
- H3가 성립하면 (체제별 불확실성 구조가 다르면) 점 예측은 tail risk를 놓친다
- PI-80 coverage가 체제별로 유의하게 다름을 보여야 함
- reliability diagram으로 캘리브레이션 시각화

### (E) Attention 설명력 과장 방어

**공격**: "attention is not explanation (Jain & Wallace 2019)"

**방어**:
- gate와 attention_top3를 "설명"이 아닌 "진단 변수(diagnostic)"로 명시
- oracle-consistency sanity check: attention이 높은 timestep을 masking했을 때 성능 저하 여부 보고

### (F) 백테스트 과대추정 방어

**공격**: "Purged CV를 했어도 hyperparameter 탐색 과정에서 미래 정보가 샜을 수 있다"

**방어**:
- Optuna 튜닝과 최종 평가를 완전히 분리된 기간에서 수행
- embargo 6일 (overlapping 5일 horizon + 1일 버퍼)
- DSR(Deflated Sharpe Ratio)로 다중검정 보정
- 고정 end-time backtest로 ablation 공정성 확보

---

## 8. 구현 현황 및 파일 맵

### 8.1 경제학 검정 (H1)

| 파일 | 내용 |
|------|------|
| `aaa/paper_test/h1_test.py` | 25 Size–B/M, Step 2-A~E, 체제별 λ₁,t 비교 |
| `aaa/paper_test/h1_ind49_monthly.py` | 49 Industry robustness |
| `aaa/paper_test/h1_stock.py` | 개별 종목 확장 |

### 8.2 모델 구현 (H2/H3)

| 파일 | 내용 |
|------|------|
| `models/transformer_encoder.py` | `CyclePE` (static/concat_a/cycle_pe/cycle_pe_full) |
| `models/cvae_decoder.py` | `CVAEDecoder` (학습/추론/MC sampling/PI-80) |
| `models/hybrid_model.py` | `HybridModel` (pe_mode, decoder_mode, forward_train, predict_interval) |

### 8.3 Ablation 검정 (H2/H3 proxy)

| 파일 | 내용 |
|------|------|
| `aaa/paper_test/h2_h3_ablation.py` | Ken French 25 포트폴리오 walk-forward, Static/Concat-A/Cycle-PE 비교, DM test, 체제별 CRPS |
| `aaa/paper_test/proxy_crps.py` | state-blind vs state-aware CRPS 비교 (선행 검증) |

### 8.4 남은 구현 작업

| 우선순위 | 작업 | 파일 |
|----------|------|------|
| 1 | H1-Vol Step 2-D stationary bootstrap 재검정 | `h1_test.py` 확장 |
| 2 | β-window 민감도 (60 vs 120) robustness | `h1_robustness.py` (신규) |
| 3 | HybridModel CVAE 학습 루프 통합 | `training/trainer.py` 수정 |
| 4 | 체제별 CRPS/PI-80 전체 모델 평가 | `h2_h3_ablation.py` → 실제 모델로 교체 |
| 5 | Reliability diagram | `aaa/paper_test/calibration.py` (신규) |
| 6 | DSR 계산 | `utils/metrics.py` 확장 |

---

## 9. 논문 구조 (투고 버전)

```
1. Introduction
2. Related Work
   2.1 Momentum crashes and volatility regimes
   2.2 State-conditioned sequence models
   2.3 Probabilistic forecasting in finance
3. Data and Preliminary Analysis
   3.1 Data sources
   3.2 State signal construction
   3.3 Preliminary: H1-Pos null result
4. Hypotheses
   H1-Vol, H2, H3 (formally stated)
5. Methodology
   5.1 Residual target construction
   5.2 2-step Fama–MacBeth
   5.3 Cycle-PE
   5.4 Hybrid encoder (Transformer + TCN + gated fusion)
   5.5 CVAE decoder
   5.6 Training and evaluation protocol
6. Results
   6.1 H1-Vol: volatility-asymmetric modulation
   6.2 H2: Cycle-PE vs ablations (IC, CRPS, DM test)
   6.3 H3: regime-conditional uncertainty (CRPS, PI-80, reliability)
7. Robustness
   7.1 β-window sensitivity
   7.2 Ridge β
   7.3 VIX as alternative intensity
8. Discussion and Limitations
9. Conclusion
Appendix
   A. cycle_pe_full (position + intensity)
   B. GAN decoder ablation
   C. Raw return H1-Vol
   D. Oracle-consistency attention check
```

---

## 10. 핵심 판단 기준 (go/no-go)

| 가설 | go 조건 | no-go 시 대응 |
|------|---------|--------------|
| H1-Vol | Step 2-D: d < 0, p < 0.05 (HAC), 두 단면 일관 | 탐색적 발견으로 격하, H2/H3 동기 재서술 |
| H2 | DM test: Cycle-PE vs Static p < 0.10, fold 일관성 ≥ 60% | Concat-A와 동률이면 novelty 약화 → 이론 강화 |
| H3 | Bear/volatile CRPS > Bear/quiet CRPS (유의), PI-80 coverage 체제별 차이 | CVAE 필요성 약화 → 점 예측 모델로 격하 |

---

## 부록 A. 수식 정리

**FF3 잔차**:
```
ε_{i,t} = r_{i,t} - α_i - β_MKT,i·MKT_t - β_SMB,i·SMB_t - β_HML,i·HML_t
```

**5일 누적 타겟**:
```
y_{i,t} = Σ_{h=1}^{5} ε_{i,t+h}
```

**H1-Vol 회귀식**:
```
λ_{1,t} = a + b·intensity_{t-1} + c·position_{t-1}
          + d·1(position_{t-1}<0)·intensity_{t-1} + u_t
```
핵심 예측: d < 0

**Cycle-PE (cycle_pe mode)**:
```
x'_t = x_t + PE_sin/cos(t) + W_int · intensity_t
```

**CVAE ELBO**:
```
L = E[log p(y|z,c)] - β·KL(q(z|y,c) || N(0,I))
```

**CRPS (pinball 근사)**:
```
CRPS ≈ (1/|τ|) Σ_τ ρ_τ(y - q_τ)
ρ_τ(e) = e·τ if e≥0, e·(τ-1) if e<0
```

**PI-80**:
```
PI_low  = q_{0.10}({ŷ_s}_{s=1}^{S})
PI_high = q_{0.90}({ŷ_s}_{s=1}^{S})
coverage = mean(1(y ∈ [PI_low, PI_high]))  → target: 0.80
```

---

## 부록 B. 참고 문헌 (핵심)

- Daniel, K. & Moskowitz, T. (2016). Momentum crashes. *Journal of Financial Economics*.
- Gu, S., Kelly, B. & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*.
- Jain, S. & Wallace, B.C. (2019). Attention is not explanation. *NAACL*.
- Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS*.
- Kingma, D.P. & Welling, M. (2014). Auto-encoding variational bayes. *ICLR*.
- Bailey, D.H. & López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*.
- Politis, D.N. & Romano, J.P. (1994). The stationary bootstrap. *JASA*.
