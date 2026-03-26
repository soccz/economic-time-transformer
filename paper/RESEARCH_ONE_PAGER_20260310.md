# Research One-Pager

## 제목

Economic Time in Financial Transformers:
Input-Space Conditioning vs Coordinate-Space Conditioning under Market Volatility

## 한 줄 요약

이 연구는 금융 시계열 Transformer에서 시장 상태를 단순 입력 피처로 넣는 방식(`concat_a`)과 시간 좌표계 자체를 시장 상태로 조건화하는 방식(`tau_rope`)이 예측 특성을 어떻게 다르게 만드는지 검증한다.

## 문제 정의

기존 금융 시계열 모델은 대체로 시장 상태를 입력 채널에 추가하는 수준에서 조건화를 수행한다. 그러나 금융 시장에서는 같은 가격 패턴이라도 어떤 시장 상태에서 발생했는지에 따라 의미가 달라질 수 있다. 따라서 시장 상태를 단순 feature가 아니라 시간 흐름 자체를 바꾸는 조건으로 다룰 필요가 있다.

핵심 질문은 다음과 같다.

> 금융 Transformer에서 시장 상태를 어디에 주입하느냐가
> cross-sectional ranking과 absolute prediction의 trade-off를 바꾸는가?

## 연구 질문

1. 입력공간 조건화(`concat_a`)와 좌표공간 조건화(`tau_rope`)는 서로 다른 예측 특성을 만드는가?
2. 특히 고변동성 구간에서 좌표공간 조건화가 절대오차(MAE)와 순위예측(IC)에 이점을 가지는가?

## 핵심 비교 모델

- `static`: 시장 조건화 없음
- `concat_a`: 시장 상태를 입력공간에 concat
- `tau_rope`: 시장 상태로 economic time `tau`를 만들고, 이를 좌표공간 조건화에 사용

메인 비교는 위 세 모델로만 제한한다.

## 방법 요약

### 데이터

- Ken French 25 portfolios
- S&P 500 (`^GSPC`)와 Nasdaq (`^IXIC`)를 broad-market anchor로 사용
- 잔차 타겟: FF3 residual

### 상태 변수

- `position = (Index - MA200) / MA200`
- `intensity = RV30 quantile rank over 252 days`

### 모델

- 글로벌 경로: Transformer
- 로컬 경로: TCN/CNN
- 융합: scalar gate
- 좌표공간 조건화: `tau_rope`

`tau_rope`는 시장 경로에서 monotone한 `tau_t`를 생성하고, 이를 RoPE 기반 attention 좌표로 사용한다.

## 현재 저장소 기준 실증 결과

확인적 실험은 `2020-2024` 구간에서 수행되었고, 고변동성 구간만 대상으로 사전등록된 검정을 적용했다.

### Confirmatory pooled result

- H1 high-vol IC:
  - `tau_rope - concat_a = 0.0387`
  - one-sided Newey-West `p = 0.0595`
- H2 high-vol MAE:
  - `concat_a - tau_rope = 0.000092`
  - one-sided Newey-West `p = 0.0019`

### 해석

- IC 개선은 한계적으로만 지지된다
- MAE 개선은 유의하게 지지된다
- 따라서 `tau_rope`를 “전반적으로 더 좋은 모델”로 쓰면 과장이다
- 대신 “고변동성에서 absolute prediction에 더 유리한 conditioning space”로 쓰는 것이 정직하다

## 현재 방어 가능한 주장

이 논문이 현재 방어할 수 있는 가장 강한 주장은 다음과 같다.

> 금융 Transformer에서 conditioning space의 선택은 중요하며,
> input-space conditioning은 강한 ranking baseline이고,
> coordinate-space conditioning은 고변동성 구간에서 absolute prediction에 유리한 trade-off를 만든다.

## 버려야 하는 주장

현재 저장소 기준으로 본문에서 버려야 하는 주장은 다음과 같다.

- `PE injection이 concat보다 항상 우월하다`
- `attention이 설명 메커니즘이다`
- `masking 결과가 routing 중요도를 증명한다`
- `CVAE / uncertainty decomposition이 현재 논문의 핵심 기여다`
- `learned tau가 이미 rule-based tau보다 낫다`

## 연구 기여 3개

1. 금융 Transformer에서 시장 상태 조건화를 `입력공간 vs 좌표공간` 문제로 재정의했다.
2. market-conditioned economic time을 사용하는 `tau_rope` 비교 프레임을 구현했다.
3. 고변동성 구간에서 좌표공간 조건화가 MAE를 유의하게 개선하고 IC에서는 한계적 이점을 보인다는 확인적 증거를 제시했다.

## 한계

- 데이터가 Ken French 25 portfolios 중심이라 미시구조 수준 해석은 제한적이다
- IC 우위는 아직 강하게 유의하지 않다
- explainability와 probabilistic decoding은 현재 메인 주장으로 쓰기 어렵다

## 다음 단계

1. 논문 본문을 `static / concat_a / tau_rope` 3모델 중심으로 재작성
2. `2020-2024` confirmatory 결과를 메인 테이블로 고정
3. `flow_pe`, `CVAE`, `learned_tau_rope`는 appendix 또는 후속 과제로 이동
4. attention 관련 서술은 descriptive diagnostic 수준으로 제한

## 결론

현재 연구 방향은 올바르게 잡혀 있다. 다만 이 논문은 “시장 상태로 시간을 재정의하는 거대한 이론”이 아니라, “조건화 공간의 차이가 금융 Transformer의 예측 trade-off를 바꾼다”는 좁고 강한 주장으로 집필해야 가장 경쟁력이 높다.
