# Aether 서비스 기술/기획 상세 보고서 (심층 확장판)

## 1. 서비스 명
**Aether (에테르)**: Explainable AI Partner for Crypto Assets

## 2. 서비스 배포 URL
*   **Web Dashboard**: [URL 기입 필요] (예: Cloudflare Tunnel 주소)
*   **GitHub Repository**: [https://github.com/soccz/Chrono-Trader-v2](https://github.com/soccz/Chrono-Trader-v2)

---

## 3. 해결하고자 했던 문제 (Problem Definition) (500자 이상)
**[1] 공공/금융 의사결정의 '설명 불가능성'과 사회적 비용**
AI 기반 의사결정 시스템은 금융, 행정, 안전 등 고위험(High-Stakes) 영역으로 빠르게 확산되고 있으나, 그 판단 과정이 불투명한 '블랙박스'로 남아있어 심각한 사회적 불신을 초래하고 있습니다. 데이터가 "매수하라" 혹은 "반려하라"고 지시할 때, 그 이유를 설명하지 못한다면 책임 소재가 모호해지고 결과에 승복하기 어렵습니다. 특히 실시간 판단이 요구되는 환경에서 AI의 불투명성은 단순한 불편을 넘어, 데이터 거버넌스의 붕괴와 사회적 갈등 비용 증가로 이어집니다. 저희는 이러한 문제의 핵심이 **"정확도에만 매몰되어 '설명 가능성(Explainability)'을 희생한 기존 AI 설계 방식"**에 있다고 진단했습니다.

**[2] 실험 환경으로서의 암호화폐 시장: '비마르코프적' 복잡계**
저희는 이 '설명 가능한 AI(XAI)' 구조를 검증하기 위한 최적의 실험 환경(Testbed)으로 **24시간 운영되는 암호화폐 시장**을 선택했습니다. 이곳은 극도로 변동성이 크고, 대중 심리와 거시 경제 데이터가 복잡하게 얽힌 대표적인 **'비마르코프적(Non-Markovian)'** 시스템입니다. 이 가혹한(Hostile) 환경에서 AI가 자신의 판단 근거(시장 맥락, 과거 유사 사례)를 인간에게 논리적으로 설명하고 설득할 수 있다면, 이 모델은 향후 **공공 정책 수립, 재난 안전 관리, 고위험 금융 자동화** 등 투명성이 필수적인 모든 데이터 거버넌스 영역으로 확장될 수 있다고 확신합니다. Aether는 단순한 투자 도구가 아닌, **"신뢰 가능한 AI 의사결정 표준"**을 정립하기 위한 도전입니다.

## 4. 서비스 소개 및 주요 기능 (Solution) (700자 이상)
**[개요] 설명 가능한 의사결정의 미래, Aether (에테르)**
Aether는 데이터 기반 의사결정의 투명성을 확보하기 위해 설계된 **'설명 가능한 AI(XAI) 리서치 플랫폼'**입니다. 블랙박스 모델의 한계를 극복하기 위해, 모델의 추론 과정을 시각화하고(Attention), 과거의 판단 기준을 참조하며(Reference), 인간과 대화(Dialogue)하는 3단계 검증 구조를 구현했습니다.

**[핵심 기능 상세]**

**1. Explainable Gated Fusion: "판단의 근거를 시각적으로 증명하다"**
Aether는 의사결정의 투명성을 위해 Prototype Learning을 시계열 추천 모델에 적용했습니다. AI는 학습 과정에서 과거 성공/위험 패턴(Prototype)을 추출하여 데이터베이스화합니다. 실시간 판단 시, 각 추천에는 가장 유사한 과거 성공 패턴의 ID와 유사도(`prototype_match`)가 함께 출력됩니다. 또한 Transformer attention 중요도 기준 상위 3개 시점(`attention_top3`)을 노출하여, 모델이 어느 과거 시점을 근거로 판단했는지 설명합니다. 사용자는 결과뿐만 아니라 판단 과정까지 검증할 수 있습니다.

**2. Context-Aware Time Perception: "맥락을 이해하는 적응형 시스템"**
단순한 데이터 수치 비교를 넘어, 상황의 '맥락(Context)'을 이해하는 기술을 구현했습니다. 시장 지수 수익률(`market_index_return`, BTC/ETH 기반)과 과거 패턴 유사도(`historical_similarity`)를 Contextual PE로 주입하고, 4-state BTC 체제(Bull/Bear × quiet/volatile)를 FiLM 레이어로 조건화합니다. 이를 통해 AI는 단순 수치가 같더라도 **"평온한 시기의 데이터"와 "위기 상황의 데이터"를 구분**하여 해석합니다. 이는 공공 영역에서 동일한 민원이라도 긴급 상황 여부에 따라 다르게 처리해야 하는 고도화된 의사결정 구조의 기반 기술이 됩니다.

**3. Interactive Research Lab: "투명한 소통을 위한 RAG 인터페이스" (계획 중)**
Aether는 일방적 통보가 아닌 '설명과 소통'을 지향합니다. 계획 중인 Research Lab은 거대언어모델(LLM)과 벡터 검색(RAG)을 결합하여, 사용자가 **"왜 이런 판단을 내렸는가?"**라고 물으면 모델의 내부 상태값(`attention_top3`, `prototype_match`)과 참조한 데이터를 종합하여 논리적으로 답변하는 인터페이스를 목표로 합니다. 이는 향후 공공 정책 결정 과정에서 시민들에게 **정책의 근거를 투명하게 설명하고 피드백을 수용하는 '디지털 거버넌스' 모델**로 발전할 수 있는 가능성을 보여줍니다.

---

## 4.5. 시스템 아키텍처 다이어그램

```text
[입력] Upbit/DB 시간봉 캔들 (동적 유니버스: 24h 거래대금 Top-N)
  |
  v
[Feature Engineering]  168h × 33 features
  - 기술적 지표: close, volume, RSI, MACD, ADX, OBV, BB 등
  - 컨텍스트: market_index_return (BTC/ETH 지수), historical_similarity
  - 팩터: factor_size, factor_mom, factor_vol, factor_liq
  - 체제: btc_regime (4-state), btc_regime_rv, btc_ma_distance
  - 상호작용: factor_mom_x_bull, factor_liq_x_bear
  - 시장 중립: rolling beta/alpha (W∈{72,168,336}h, Optuna 선택)
  - cross-sectional rank 정규화
  |
  v
┌─────────────────────────────────────────────┐
│           Hybrid Encoder (병렬)              │
│                                             │
│  [Transformer Encoder]                      │
│   causal mask 없음 → 전체 시퀀스 참조        │
│   attention_weights[-1]: (B, 168)           │
│          |                                  │
│          | softmax → attention_importance   │
│          ↓                                  │
│  [Attention-guided CNN]                     │
│   guided_input = raw_input                  │
│              × attention_importance         │
│   → local motif features                   │
└─────────────────────────────────────────────┘
  |                    |
  transformer_out      cnn_out
  |                    |
  └────────┬───────────┘
           v
  [Explainable Gated Fusion]
   gate = f(transformer_out, cnn_out)
   fused = gate × transformer_out
         + (1-gate) × cnn_out
   gate → 해석 변수 (Trend vs Pattern)
           |
           v
  [FiLM Regime Conditioning]
   regime_embed(btc_regime: 0~3)
   → scale, shift → fused 변환
   (Bull_quiet / Bull_volatile /
    Bear_quiet / Bear_volatile)
           |
           v
  [GAN Decoder]
   fused + noise → Generator
   → 3-step residual return path
     (t+1, t+2, t+3)
   Critic: WGAN-GP
           |
           v
  [MC-Dropout 반복 추론]
   N회 샘플 → individual_patterns
   → PI_low_80 (10th percentile)
   → PI_high_80 (90th percentile)
   → attention_top3 (mean attn 상위 3 timestep)
   → prototype_match (DTW vs 성공 패턴 DB)
           |
           v
┌─────────────────────────────────────────────┐
│         Recommendation Funnel               │
│                                             │
│  1. Tradeable 검증 (상장 상태)               │
│  2. Regime / Lead-Lag 필터                  │
│  3. Liquidity 필터 (24h 거래대금)            │
│  4. net_alpha > 0                           │
│     gross - 2×(fee 0.05% + slippage 0.03%) │
│  5. PI_low_80 > 0  ← 하드 게이트            │
│  6. Uncertainty 필터 (adaptive threshold)   │
│  7. Consensus 필터 (앙상블 동의율)           │
│  8. DTW 패턴 유사도 필터                    │
│                                             │
│  통과 → Recommended (position_size > 0)    │
│  전부 탈락 → MinRec Watch-only (≥1 보장)   │
└─────────────────────────────────────────────┘
           |
           v
  [출력 per 코인]
   market, signal, net_alpha,
   pi_low_80 / pi_high_80,
   attention_top3, prototype_match,
   gate_value, regime_state,
   position_size, status

  [Ops Contract]
   exit 0: 정상 출력
   exit 2: stale DB → watch-only 재실행
   exit 3: watchdog timeout

  [학습 / 튜닝]
   Optuna HPO
   CV: Purged Walk-Forward + 6h embargo
   타겟: residual return (beta × BTC 제거)
   유니버스: 동적 Top-N (매 실행 갱신)
```

---

## 5. 활용한 핵심 기술 및 AI 모델 (Tech Stack)

### 5.1. Core AI Architecture
*   **Dual-Path Hybrid Model**: 시계열의 장기 의존성(Long-term Dependency)을 포착하는 **Transformer Encoder**와 국소적 패턴 변동에 강한 **CNN (local motif extractor)**을 병렬로 배치합니다. Transformer의 attention 가중치가 CNN 입력을 직접 재가중(Attention-guided CNN)하여 두 경로가 결합됩니다.
*   **Generative Decoder (GAN)**: GAN 디코더를 사용하여 **3h 잔차 수익률 경로(t+1~t+3, 3-step residual return)**를 생성합니다. 예측 타겟은 코인 수익률에서 beta × BTC 수익률을 제거한 시장 중립적 알파(residual return)이며, MC-Dropout 반복 추론으로 PI_80(80% 예측 구간)을 산출합니다.

### 5.2. Explainability & Context Engineering
*   **Prototype Match + Attention Top-3**: 각 추천에는 과거 성공 패턴 중 가장 유사한 패턴의 ID와 유사도(`prototype_match`)가 함께 출력됩니다. 또한 Transformer attention 중요도 기준 상위 3개 시점(`attention_top3`)을 노출하여 모델이 어느 과거 시점을 근거로 판단했는지 설명합니다.
*   **Context Injection + FiLM Regime Conditioning**: `Market Index Return`과 `Historical Similarity Score`를 Contextual PE로 주입하고, 4-state BTC 체제(Bull/Bear × quiet/volatile)를 FiLM 레이어로 조건화하여 체제별 표현을 분리합니다.

### 5.3. MLOps & Infrastructure
*   **Optimization**: **Optuna** 기반 하이퍼파라미터 튜닝 파이프라인이 주기적으로 실행됩니다. CV 방식으로 **Purged Walk-Forward + 6h embargo**를 적용하여 look-ahead leak을 차단합니다.
*   **Uncertainty Quantification**: 다양한 구조의 모델 앙상블과 **MC-Dropout** 반복 추론을 결합하여 PI_80(10th~90th percentile) 예측 구간을 산출합니다. `PI_low_80 > 0`이 진입 조건의 하드 게이트로 작동합니다.
*   **Ops Contract**: 스케줄 실행은 exit code 계약(0: 성공 / 2: stale DB abort / 3: watchdog timeout)을 준수합니다. DB 신선도 게이트 실패 시 watch-only 모드로 재실행하며, MinRec 정책으로 매 실행마다 최소 1개 출력을 보장합니다. 학습 유니버스는 24h 거래대금 기준 동적 Top-N으로 구성됩니다(스테이블코인·레버리지 토큰·상장 14일 미만 제외).

---

## 6. 기술적 챌린지 및 해결 과정 (Technical Challenges) (700자 이상)
**[Challenge 1] 비정상성(Non-Stationarity) 데이터로 인한 모델의 조기 노화**
금융 시계열 데이터는 통계적 특성이 시간에 따라 계속 변하는 '비정상성'을 가집니다. 과거 3년치 데이터로 완벽하게 학습한 모델이라도, 내일 새로운 규제 뉴스가 터지거나 거시 경제 기조가 바뀌면 무용지물이 되기 십상입니다. 저희 팀은 초기 모델 개발 당시, 백테스트에서는 연 200% 수익률을 기록했으나 실전 투입 일주일 만에 -10% 손실을 기록하는 '과적합(Overfitting)의 덫'에 걸렸습니다. 모델은 과거의 특정 패턴을 단순 암기했을 뿐, 시장의 변화하는 맥락을 읽지 못했기 때문입니다.

**[Solution 1] Historical Similarity 기반 컨텍스트 주입**
이를 해결하기 위해 현재 입력 윈도우(168h)와 과거 윈도우들 간의 유사도를 계산하여 `historical_similarity` 피처로 모델에 주입했습니다. 모델은 현재 시장 상황이 과거 어느 시점과 유사한지를 컨텍스트로 참조하며 추론합니다. 또한 Purged Walk-Forward CV + 6h embargo를 적용하여 학습-검증 간 look-ahead leak을 구조적으로 차단하고, Optuna HPO로 beta 계산 윈도우(W∈{72, 168, 336}h)를 포함한 하이퍼파라미터를 데이터 기반으로 선택합니다. 이를 통해 모델이 특정 과거 구간을 암기하는 대신 시장 맥락에 적응하도록 설계했습니다.

**[Challenge 2] 사용자가 납득할 수 있는 수준의 '설명력' 확보**
설명 가능한 AI(XAI) 연구는 주로 이미지 인식 분야에 치우쳐 있어, 시계열 데이터에 적용하기가 매우 까다로웠습니다. 단순히 `t-5` 시점의 Attention Score가 높다고 해서, 그것이 "왜 매수 신호인가?"를 설명해주지는 않습니다. 단순 수치나 히트맵만으로는 금융 지식이 부족한 일반 투자자를 설득하기 어려웠습니다.

**[Solution 2] Explainable Gated Fusion + Prototype Match**
단순 attention 수치만으로는 "왜 매수인가"를 설명하기 어렵다는 문제를 두 가지 방식으로 해결했습니다. 첫째, Transformer와 CNN의 융합 비율을 나타내는 gate 값을 출력하여 모델이 거시 추세(Transformer)와 국소 패턴(CNN) 중 어느 쪽에 의존했는지 명시합니다. 둘째, 각 추천에 `prototype_match`(과거 성공 패턴 DB와의 DTW 유사도 + 패턴 ID)와 `attention_top3`(판단 근거가 된 상위 3개 시점)를 함께 출력합니다. 사용자는 현재 상황이 과거 어느 패턴과 유사하며, 모델이 어느 시점을 근거로 판단했는지를 구체적으로 확인할 수 있습니다.
