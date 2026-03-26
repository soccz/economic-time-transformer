# AETHER 확정 설계 스펙
> 이 파일이 모든 코드 수정의 기준. 새 대화 시작 시 @DESIGN_SPEC.md 로 컨텍스트 복원.

---

## 최종 목표

```
BTC 장기 흐름 → 4-state 체제 판단
→ 체제가 강세일 때 Top-100 알트를 FF5 팩터로 스크리닝
→ 상위 알트 단기(3h) 급등 포착
→ 각 추천의 근거를 Attention heatmap + Prototype으로 설명
```

핵심 아이디어:
- S&P500처럼 BTC를 포지셔널 기준축으로 사용
- 주기 신호를 입력 벡터에 주입 → Attention이 가중치로 수치화 → CNN이 재해석
- GAN 디코더로 시계열을 확률 분포 형태로 표현
- 설명 가능성: Attention heatmap + Prototype matching

---

## 아키텍처 흐름

```
[입력] 168h × feature_dim
  + BTC 체제 신호 (4-state, FiLM conditioning)
  + Contextual PE (market_index_return 다중 스케일)
       ↓
[Transformer Encoder] (causal mask 없음)
  → attention_weights[-1]: (B, 168) ← 마지막 토큰의 attention row
       ↓
[Attention-guided CNN]
  guided_input = raw_input × softmax(attention_weights[-1]).unsqueeze(-1)
  CNN(guided_input) → local features
       ↓
[Explainable Gated Fusion]
  gate = f(transformer_out, cnn_out, prototype_distances)
  fused = gate * transformer_out + (1-gate) * cnn_out
       ↓
[GAN Decoder]
  Generator: fused + noise → 3h return path (3-step)
  Critic: WGAN-GP
       ↓
[출력]
  predicted_alpha (gross/net), PI_80, regime_state,
  attention_top3, prototype_match
```

---

## Q1 — Beta/Alpha 계산 (확정)

```python
# Vectorized rolling closed-form (statsmodels.OLS 사용 안 함)
W = [72, 168, 336]  # Optuna 하이퍼파라미터로 선택
eps = 1e-8

mean_x  = btc_ret.rolling(W).mean()
mean_y  = coin_rets.rolling(W).mean()
cov_xy  = (coin_rets.mul(btc_ret, axis=0)).rolling(W).mean() - mean_y.mul(mean_x, axis=0)
var_x   = btc_ret.rolling(W).var().clip(lower=eps)
beta    = cov_xy.div(var_x, axis=0).clip(-3, 3)
alpha_t = mean_y - beta.mul(mean_x, axis=0)
resid   = coin_ret - alpha_t - beta * btc_ret  # ← 예측 타겟

# 이상치 전처리
returns = returns.clip(lower=-5*rolling_std, upper=5*rolling_std)
```

---

## Q2 — BTC 체제 (4-state, 확정)

```python
# Realized volatility (연율화)
rv = btc_ret.rolling(168).std() * (8760 ** 0.5)

# 상대적 분위수 임계 (절대값 아님)
rv_low  = rv < rv.rolling(4320).quantile(0.4)   # ~180일
rv_high = rv > rv.rolling(4320).quantile(0.6)

# 200h MA + hysteresis (flip 방지)
ma200 = btc_price.rolling(200).mean()
delta = 0.02
trend_up   = btc_price > ma200 * (1 + delta)
trend_down = btc_price < ma200 * (1 - delta)
# 중간 구간 → 직전 상태 hold

# 4-state
# 0: Bull_quiet    (trend_up   & rv_low)
# 1: Bull_volatile (trend_up   & rv_high)
# 2: Bear_quiet    (trend_down & rv_low)
# 3: Bear_volatile (trend_down & rv_high)
```

---

## Q3 — 체제 조건부 학습 (FiLM, 확정)

```python
# 단일 모델 + FiLM conditioning (별도 모델 분리 안 함)
regime_embed = nn.Embedding(4, d_model)
scale = nn.Linear(d_model, hidden_dim)
shift = nn.Linear(d_model, hidden_dim)
h = h * scale(regime_embed(regime)) + shift(regime_embed(regime))

# FF5 팩터 × regime 상호작용 피처 추가
# 예: factor_mom * bull_quiet_flag (상호작용 피처)
```

---

## Q4 — 추천 출력 (확정)

```
코인당 출력:
  - predicted_alpha_gross
  - predicted_alpha_net = gross - fee(0.05%) - slippage_est
  - PI_80: 80% prediction interval (MC-Dropout 20회)
  - regime_state: 현재 BTC 체제 (0~3)
  - attention_top3: 가장 중요한 과거 시점 3개 (설명용)
  - prototype_match: prototype ID + 유사도

진입 필터:
  predicted_alpha_net > 0 AND PI_low_80 > 0

출력:
  - Top 5 trade + Top 5 watch-only
  - 웹 대시보드: 상시 표시
  - 텔레그램: 조건 충족 시 + Top-N 변경 시만 푸시

비용 모델:
  net_alpha = gross_alpha - turnover * (fee + slippage)
  fee = 0.05%, slippage = 보수적 추정
```

---

## Q5 — 학습 코인 범위 (확정)

```
유니버스: 매일 t-24h 거래대금 기준 Top 100
제외: stablecoin / 레버리지 토큰 / 상장 14일 미만
TRAIN_COINS 하드코딩 제거 → 동적 유니버스로 교체
```

---

## Q6 — Prediction Horizon (확정)

```
horizon: 3h (6h에서 축소)
label: sum(log_return(t+1 ~ t+3))
CV: Purged Walk-Forward + embargo 6h
신호 생성 주기: 매 6h (재진입/중복 포지션 룰 명확화 필요)
```

---

## Q7 — Backfill 범위 (확정)

```
BTC/ETH: 2019-01-01~ (반감기 2회: 2020-05, 2024-04 포함)
알트: 2021-01-01~ (가능한 것만, 유동성 기준)
실행: collect_market_data(market, days=2600) 백그라운드
```

---

## 버그 수정 목록 (Phase 1, 우선순위 순)

| # | 파일 | 문제 | 심각도 |
|---|---|---|---|
| 1 | `data/preprocessor.py:538` | `fit_transform` 전체 데이터 → train만 fit | 🔴 |
| 2 | `inference/predictor.py` | 추론 시 `crypto_factors_df` 미전달 → factor 전부 0 | 🔴 |
| 3 | `data/preprocessor.py` | 팩터/패턴 전역 캐시 → backtest 미래 오염 | 🔴 |
| 4 | `models/transformer_encoder.py` | Encoder causal mask → attention 왜곡 | 🟠 |
| 5 | `models/hybrid_model.py` | gate reg loss 부호 충돌 | 🟠 |
| 6 | `training/trainer.py` | CNN_MODE 전역 변경 try/finally 없음 | 🟡 |

---

## 아키텍처 개선 목록 (Phase 2)

| # | 파일 | 내용 |
|---|---|---|
| A | `data/preprocessor.py` | resid 타겟 / rolling beta closed-form / clip 전처리 |
| B | `models/transformer_encoder.py` | Attention-guided CNN soft masking (방법 A) |
| C | `data/preprocessor.py` | 4-state regime 피처 / cross-sectional rank / 상호작용 피처 |
| D | `models/hybrid_model.py` | FiLM conditioning 추가 |
| E | `utils/config.py` | horizon 3h / W 후보 하이퍼파라미터화 / 동적 유니버스 |
| F | `inference/recommender.py` | net-alpha 필터 / PI 출력 / attention_top3 / prototype_match |

---

## 보류 항목 (데이터 2년+ 후)

- VAE probabilistic encoder (posterior collapse 위험)
- Autoformer 주기 자동 발견 (데이터 부족)
- Funding rate / Open Interest 외부 데이터
- RAG 인터페이스

---

## 현재 데이터 현황

| 마켓 | 행수 | 기간 |
|---|---|---|
| KRW-BTC | 11,911 | 2024-10-23 ~ 2026-03-04 |
| KRW-ETH | 11,842 | 2024-10-25 ~ 2026-03-04 |
| 알트 244개 | ~4,028 | 2025-09-17 ~ 2026-03-04 |
| 전체 | 827,295 | 244 마켓 |

backfill 후 목표: BTC/ETH 2019~, 알트 2021~
