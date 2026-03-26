팩터 조정 잔차 수익률의 변동성 체제 조건부 예측: Cycle-aware Positional Encoding과 Hybrid Transformer–TCN–CVAE
초록

본 연구는 Fama–French 3팩터(FF3)로 조정한 잔차 수익률에서 단면 모멘텀 프리미엄이 **시장 상태(state)**에 따라 어떻게 변하는지, 그리고 해당 state 정보를 시퀀스 모델에 명시적으로 주입할 때 예측 및 불확실성 추정이 개선되는지를 검증한다. 예비 분석에서, S&P 500의 200일 이동평균 기반 **추세 신호(cycle_position)**는 단면 모멘텀 프리미엄(λ₁,t)의 시계열 변동을 유의하게 설명하지 못했다(예: Ken French 25 Size–B/M 포트폴리오 p≈0.873). 반면 **변동성 상태(cycle_intensity)**로 구분한 체제별 비교에서 λ₁,t는 “조용한 시장(quiet)”에서 높고 “고변동성(volatile)”에서 급락(일부 구간 반전)하는 일관된 패턴이 관찰되었으며, 이는 모멘텀 크래시 문헌과 정합적이다.
이에 따라 본 연구는 (i) 경제학적 H1을 “추세 증폭”이 아닌 “변동성-비대칭 조절(특히 Bear/volatile 급락)”로 재정의하고, (ii) state를 Cycle-aware Positional Encoding(Cycle-PE)으로 Transformer–TCN 하이브리드 인코더에 주입하며, (iii) CVAE 디코더로 잔차 수익률의 조건부 분포를 산출해 체제별 불확실성 구조를 추정한다. 실증은 Purged Walk-forward 교차검증과 stationary bootstrap 기반 추론을 사용한다.

키워드: FF3 residual, volatility regime, momentum crashes, Cycle-aware positional encoding, Transformer–TCN, gated fusion, CVAE, probabilistic forecasting

1. 서론

자산 수익률은 공통 요인(시장·스타일)과 개별 고유 성분으로 분해된다. FF3는 개별 자산의 초과수익을 시장(MKT), 사이즈(SMB), 가치(HML) 요인의 선형 결합으로 설명하고, 남는 부분을 잔차로 정의한다.
본 연구의 초점은 “FF3가 맞냐/틀리냐”가 아니라, FF3로 설명되지 않는 잔차 성분에서 예측 가능한 구조가 존재하며 그 구조가 시장 상태에 따라 달라지는가이다.

기존 예측 연구는 원시 수익률 또는 기대수익을 대상으로 비선형 모델의 예측력을 보여주었지만(예: ML 기반 자산가격) ,
(1) state를 모델 구조에 명시적으로 조건화하는 설계가 약하거나,
(2) 예측을 **분포(확률적 예측)**로 다루지 않아 tail risk/불확실성의 경제적 해석이 제한되는 경우가 많다.

또한 모멘텀 전략은 평균적으로 강하지만, 특정 “공포/고변동성 상태”에서 급격한 손실(모멘텀 크래시)을 보이며, 이 크래시는 부분적으로 예측 가능하다는 결과가 알려져 있다.
이는 “모멘텀 프리미엄의 상태 의존성”을 탐구할 경제적 동기를 제공한다.

본 논문은 다음을 결합한다:

경제학적 층: 단면 모멘텀 프리미엄(λ₁,t)의 state 의존성 검정(2-step Fama–MacBeth).

표현학습 층: state를 Positional Encoding에 주입해 attention의 거리/유사도 계산에 직접 반영하는 Cycle-PE.

확률예측 층: CVAE로 조건부 예측분포를 생성해 체제별 불확실성 구조를 비교.

2. 이론적 배경 및 가설
2.1 FF3 분해와 잔차 타겟

초과수익 
𝑟
𝑖
,
𝑡
r
i,t
	​

에 대해,

𝑟
𝑖
,
𝑡
=
𝛼
𝑖
+
𝛽
𝑀
𝐾
𝑇
,
𝑖
𝑀
𝐾
𝑇
𝑡
+
𝛽
𝑆
𝑀
𝐵
,
𝑖
𝑆
𝑀
𝐵
𝑡
+
𝛽
𝐻
𝑀
𝐿
,
𝑖
𝐻
𝑀
𝐿
𝑡
+
𝜖
𝑖
,
𝑡
r
i,t
	​

=α
i
	​

+β
MKT,i
	​

MKT
t
	​

+β
SMB,i
	​

SMB
t
	​

+β
HML,i
	​

HML
t
	​

+ϵ
i,t
	​


FF3 팩터(일별 
𝑀
𝐾
𝑇
,
𝑆
𝑀
𝐵
,
𝐻
𝑀
𝐿
,
𝑅
𝐹
MKT,SMB,HML,RF)는 Ken French Data Library에서 제공되며, 팩터는 “모든 자산이 공유하는 공통 시계열”이고 β는 “자산별 노출도”다.

본 연구의 예측 대상은 잔차(또는 잔차의 누적)이다.

1일 잔차: 
𝜖
𝑖
,
𝑡
ϵ
i,t
	​


h-일 누적 잔차(권장: 5일): 
𝑦
𝑖
,
𝑡
=
∑
ℎ
=
1
5
𝜖
𝑖
,
𝑡
+
ℎ
y
i,t
	​

=∑
h=1
5
	​

ϵ
i,t+h
	​


중요한 원칙: 예측가능성 논문으로 가려면 “동일일자 설명”이 아니라 “미래 구간(horizon)의 타겟”이 더 정합적이다(너의 스펙도 5일 누적을 선택).

2.2 시장 상태(state) 정의: 추세 vs 변동성

본 연구는 S&P 500 기반의 두 연속 신호로 state를 정의한다.

추세 위치(Trend position)

𝑐
𝑦
𝑐
𝑙
𝑒
_
𝑝
𝑜
𝑠
𝑖
𝑡
𝑖
𝑜
𝑛
𝑡
=
𝑆
𝑃
𝑋
𝑡
−
𝑀
𝐴
200
,
𝑡
𝑀
𝐴
200
,
𝑡
cycle_position
t
	​

=
MA
200,t
	​

SPX
t
	​

−MA
200,t
	​

	​


200일 이동평균은 대표적 장기 추세 기준선으로 널리 사용된다.

변동성 강도(Volatility intensity)

𝑐
𝑦
𝑐
𝑙
𝑒
_
𝑖
𝑛
𝑡
𝑒
𝑛
𝑠
𝑖
𝑡
𝑦
𝑡
=
QuantileRank
(
𝑅
𝑉
30
,
𝑡
;
252
)
cycle_intensity
t
	​

=QuantileRank(RV
30,t
	​

;252)

(30일 실현변동성의 최근 252일 분위수)

또한 모델 조건화(FiLM 등)를 위해 2×2 이산 체제를 사용할 수 있으나, 경제학적 검정(H1)은 연속 신호를 주 분석으로 둔다.

2.3 H1: “추세 증폭”은 기각, “변동성-비대칭 조절”이 핵심 후보

너의 예비 실험 결과(스펙 정렬 후)는 다음을 시사한다.

𝑐
𝑦
𝑐
𝑙
𝑒
_
𝑝
𝑜
𝑠
𝑖
𝑡
𝑖
𝑜
𝑛
𝑡
−
1
cycle_position
t−1
	​

 단독 회귀는 λ₁,t를 설명하지 못함
(25 Size–B/M: p≈0.873, 49 Industry: p≈0.241)

체제 평균 비교에서 공통 패턴:

Bear/quiet에서 λ₁,t가 가장 높음

Bear/volatile에서 λ₁,t가 급락(일부 음수/반전)

두 단면(25 Size–B/M, 49 Industry)에서 정성적으로 일관

이건 “Bull에서 모멘텀 증폭”이라기보다 “고변동성/공포 상태에서 모멘텀 크래시” 계열 가설과 더 정합적이다.

따라서 가설은 아래처럼 재정렬하는 게 논리적으로 깔끔하다.

(사전등록 H1-Pos: 추세 기반)

H1-Pos(등록 가설): 
𝑐
𝑦
𝑐
𝑙
𝑒
_
𝑝
𝑜
𝑠
𝑖
𝑡
𝑖
𝑜
𝑛
cycle_position이 λ₁,t를 조절한다

현 상태: 예비 분석에서 지지되지 않음(“null result”로 명시)

(핵심 가설 후보 H1-Vol: 변동성 기반)

H1-Vol(핵심): 
𝑐
𝑦
𝑐
𝑙
𝑒
_
𝑖
𝑛
𝑡
𝑒
𝑛
𝑠
𝑖
𝑡
𝑦
cycle_intensity가 λ₁,t를 조절하며, 특히 Bear × HighVol에서 λ₁,t가 급락/반전한다.

이를 검정 가능한 형태로 명확히 쓰면:

𝜆
1
,
𝑡
=
𝑎
+
𝑏
⋅
𝑖
𝑛
𝑡
𝑒
𝑛
𝑠
𝑖
𝑡
𝑦
𝑡
−
1
+
𝑐
⋅
𝑝
𝑜
𝑠
𝑖
𝑡
𝑖
𝑜
𝑛
𝑡
−
1
+
𝑑
⋅
1
(
𝑝
𝑜
𝑠
𝑖
𝑡
𝑖
𝑜
𝑛
𝑡
−
1
<
0
)
⋅
𝑖
𝑛
𝑡
𝑒
𝑛
𝑠
𝑖
𝑡
𝑦
𝑡
−
1
+
𝑢
𝑡
λ
1,t
	​

=a+b⋅intensity
t−1
	​

+c⋅position
t−1
	​

+d⋅1(position
t−1
	​

<0)⋅intensity
t−1
	​

+u
t
	​


핵심은 d < 0 (약세에서 변동성 증가가 모멘텀 프리미엄을 훼손)

선형이 약하면 
𝑖
𝑛
𝑡
𝑒
𝑛
𝑠
𝑖
𝑡
𝑦
2
intensity
2
 또는 상위 분위 더미(꼬리)로 비선형을 테스트

2.4 H2: Cycle-PE의 역할을 “거리(유사도) 메트릭 수정”으로 정의

네가 고민한 “sin/cos 파형 vs 다른 파형” 논쟁의 핵심은 이것이다.

표준 PE는 토큰 임베딩에 “순서 좌표(ordinal time)”를 더한다.

Cycle-PE는 여기에 “시장 상태 좌표(state-space)”를 더한다.

𝑥
𝑡
′
=
𝑥
𝑡
+
𝑃
𝐸
𝑡
𝑖
𝑚
𝑒
(
𝑡
)
+
𝑃
𝐸
𝑠
𝑡
𝑎
𝑡
𝑒
(
𝑠
𝑡
)
x
t
′
	​

=x
t
	​

+PE
time
	​

(t)+PE
state
	​

(s
t
	​

)

이때 self-attention의 스코어는 대략

score
(
𝑡
,
𝜏
)
∝
(
𝑊
𝑄
𝑥
𝑡
′
)
⊤
(
𝑊
𝐾
𝑥
𝜏
′
)
score(t,τ)∝(W
Q
	​

x
t
′
	​

)
⊤
(W
K
	​

x
τ
′
	​

)

이므로, state를 PE로 주입하면 attention 유사도 계산 자체에 state가 직접 개입한다.
반면 state를 입력 채널로 concat하면 모델이 학습을 통해 state를 반영할 수는 있지만, “시간 좌표계 자체가 바뀐다”는 해석은 상대적으로 약해진다(그래서 Cycle-PE vs Concat-A가 필수 ablation).

여기서 H2는 이렇게 명시한다:

H2: Cycle-PE(특히 intensity 주입)가 Static PE 또는 Concat-A 대비 표본 외 IC/CRPS를 개선한다.

검정: Diebold–Mariano(시계열 의존 반영 형태로) + fold-wise 일관성 보고.

2.5 H3: 분포 예측과 state-dependent uncertainty

CVAE를 “있으면 멋있다”가 아니라 “필수다”로 만들려면, H3가 경제적으로 서야 한다.

H3: 예측 불확실성(분포의 폭/꼬리/캘리브레이션)은 state에 따라 체계적으로 달라진다.
특히 high-intensity(고변동성)에서 예측분포가 넓어지고 tail risk가 증가한다.

모멘텀 크래시/변동성-조건부 성과 문헌은 “high vol에서 리스크/왜도 문제가 커진다”는 방향과 정합적이다.

3. 데이터
3.1 최소 재현 가능 데이터(포트폴리오 기반)

Ken French 25 Size–B/M daily 포트폴리오 수익률

Ken French 49 Industry daily 포트폴리오 수익률

FF3 daily factors + RF, Momentum factor(WML)

S&P 500 (^GSPC) 지수 (state 계산)

VIX (대체/보조 intensity 검정)

Ken French Data Library와 VIX(FRED)는 공개 데이터로 재현성이 높다.

3.2 확장 데이터(주식 개별 종목, 권장)

CRSP point-in-time 유니버스(생존편향 방지)
이 확장은 논문 임팩트를 크게 올리지만, 구현 비용이 상승한다.

4. 방법론
4.1 잔차 생성: rolling β 추정과 generated-target 리스크

너의 구현처럼 rolling OLS로 β를 추정해 잔차를 만든다:

윈도우: 60일(기본), robustness로 120일/ ridge β 등

비판적 포인트(반드시 해결해야 함):
잔차 타겟은 “관측”이 아니라 “추정으로 생성”되므로 generated regressor/measurement error 공격을 받는다.
따라서 β-window 민감도, ridge/축소 추정, 포트폴리오 vs 종목 결과 일관성을 반드시 보고해야 한다(심사 방어 핵심).

4.2 H1 추정: 2-step Fama–MacBeth + robust inference

Step 1(단면):

𝑦
𝑖
,
𝑡
=
𝜆
0
,
𝑡
+
𝜆
1
,
𝑡
⋅
𝑚
𝑜
𝑚
𝑖
,
𝑡
+
𝛾
⊤
𝑐
𝑜
𝑛
𝑡
𝑟
𝑜
𝑙
𝑠
𝑖
,
𝑡
+
𝜂
𝑖
,
𝑡
y
i,t
	​

=λ
0,t
	​

+λ
1,t
	​

⋅mom
i,t
	​

+γ
⊤
controls
i,t
	​

+η
i,t
	​


Step 2(시계열):

𝜆
1
,
𝑡
=
𝑎
+
𝑓
(
𝑠
𝑡
𝑎
𝑡
𝑒
𝑡
−
1
)
+
𝑢
𝑡
λ
1,t
	​

=a+f(state
t−1
	​

)+u
t
	​


Fama–MacBeth 틀 자체는 고전적이며, λ 시계열의 추론은 시계열 의존을 반영해야 한다.
또한 overlapping horizon(예: 5일 누적) 때문에 block 기반 추론이 유리하므로 stationary bootstrap을 기본으로 둔다.

4.3 Cycle-aware Positional Encoding

Cycle-PE(time + intensity 중심 버전):

최소 버전: intensity만 PE로 주입(“핵심 신호가 intensity”라는 예비 결과와 정합)

확장 버전: position + intensity 모두 주입(appendix)

4.4 인코더: Transformer + attention-guided TCN + gated fusion

Transformer: 글로벌 구조/장기 의존

TCN: 로컬 패턴/단기 구조

Gate: “설명”이 아니라 **진단 변수(diagnostic)**로 취급

주의: attention을 설명으로 과장하면 역공 포인트가 된다. “attention is not explanation” 논쟁을 인지하고, oracle-consistency류 sanity check를 함께 둔다.

4.5 디코더: CVAE 기반 조건부 분포 예측

CVAE는 
𝑝
(
𝑦
∣
𝑐
)
p(y∣c)를 직접 생성할 수 있고, CRPS/coverage/calibration으로 평가 가능하다.
GAN은 불안정성이 커서 **ablation(선택적)**로 두고, 메인은 CVAE로 고정하는 편이 논문 완성도가 높다.

5. 실험 설계
5.1 검증 순서(논문 리스크 최소화)

경제학(H1-Vol) 먼저 확정

예측(H2)로 연결

분포/불확실성(H3)로 확장

5.2 검증 지표

점 예측: IC, ICIR, MAE

분포 예측: CRPS, PI coverage, reliability diagram

경제 성과(분리 보고): long-short 포트폴리오 성과(+ DSR)

Purged CV/embargo/DSR는 금융 백테스트 과대추정 방어에 효과적이다.

6. 예비 결과(현재까지) 및 논의
6.1 “추세 기반 H1-Pos”는 지지되지 않음

𝑐
𝑦
𝑐
𝑙
𝑒
_
𝑝
𝑜
𝑠
𝑖
𝑡
𝑖
𝑜
𝑛
𝑡
−
1
cycle_position
t−1
	​

 단독:

25 Size–B/M: p≈0.873

49 Industry: p≈0.241(부호 양수로 전환)

→ 추세 신호 단독의 선형 조절 가설은 약하다(또는 틀렸다).

6.2 “변동성 기반 H1-Vol” 후보: 체제 평균에서 강한 패턴

Bear/quiet에서 λ₁,t 최대

Bear/volatile에서 급락(일부 반전)

25/49 단면에서 정성적으로 일관

이 패턴은 모멘텀 크래시가 “시장 하락 이후 + 변동성 고조 + 반등 국면”에서 나타난다는 결과와 방향성이 맞는다.
또한 변동성 조건부 모멘텀 리스크 관리(스케일링) 문헌과도 연결된다.

엄격한 결론(현 시점):

“추세가 λ₁,t를 선형 조절한다”는 H1-Pos는 기각 쪽

“변동성과 약세-비대칭이 핵심”이라는 H1-Vol은 강한 후보

단, H1-Vol은 지금 단계에선 탐색적 발견 → 사전 명세된 형태로 재검정이 필요

6.3 가장 큰 기술적 리스크: generated target + endogeneity 신호

너의 진단에서 
∣
𝑚
𝑜
𝑚
∣
∣mom∣과 
𝑠
𝑒
(
𝛽
^
𝑀
𝐾
𝑇
)
se(
β
^
	​

MKT
	​

) 상관이 유의하게 나왔다는 건, 잔차 타겟이 β 추정오차와 얽힐 수 있음을 의미한다.
이건 “IV 하자”로만 끝나면 안 되고, 더 강한 방어는 아래 중 최소 2개가 필요하다:

β-window 확대/축소(60 vs 120) 민감도

ridge/shrinkage β로 잔차 생성(OLS 아티팩트 방어)

포트폴리오 vs 종목에서 정성적 결론 일치 여부

H1-Vol이 잔차가 아니라 raw return에서도 유사 패턴을 보이는지(부록)

7. 이 논문의 “최종 메시지”를 단단하게 만드는 설계 재정렬

너의 처음 설계는 “position 중심”이었는데, 현재 데이터가 말하는 건 “intensity 중심”이다.
그러면 논문 전체 인과 사슬을 이렇게 다시 묶는 게 가장 깔끔하다:

H1-Vol(경제학): 변동성-비대칭 상태가 잔차 모멘텀 프리미엄을 조절한다(quiet에서 높고, Bear/volatile에서 크래시).

H2(ML): 따라서 intensity state를 Cycle-PE로 주입하면 예측이 개선된다(Static PE/Concat 대비).

H3(분포/불확실성): 동일 state가 예측분포(폭/꼬리/캘리브레이션)에도 반영된다(CVAE가 필요해지는 이유).

이렇게 하면 CVAE는 “멋”이 아니라 “논문 필수 구성요소”가 된다(분포/꼬리/캘리브레이션을 본문 기여로 만들기 때문).

8. 심사 관점에서 “반드시 잠가야 하는” 공격 포인트 체크리스트

아래는 리젝 사유로 직결되는 것들만 모은 거야(중요도 순).

(A) H1-Vol을 ‘탐색→확증’으로 전환하는 절차

지금 체제 평균표 패턴은 설득력 있지만, 그대로 주장하면 “post-hoc binning” 공격이 가능

해결: H1-Vol을 회귀식(상호작용/비선형)로 사전 고정하고 bootstrap/HAC로 재검정

(B) generated residual target 방어

최소 2개 robustness 필수:

β-window / ridge β / 종목 데이터 확장 중 택2

(C) Cycle-PE novelty 방어

Static PE vs Cycle-PE만 하면 부족

**Concat-A(per-token state feature)**를 꼭 넣고, Cycle-PE가 concat 대비 왜/언제 낫는지 보여야 함
(안 나오면 기여를 “PE 방식”이 아니라 “state encoding 자체”로 재정의)

(D) “attention/gate 해석” 과장 금지

attention이 설명이라는 주장은 공격받기 쉬움.

해결: gate/attention은 “진단”으로 두고 sanity check(oracle-consistency, perturbation test) 포함

(E) CVAE 필요성 방어(가장 중요)

금융 심사자가 물을 질문: “왜 quantile regression이 아니라 VAE인가?”

해결: CVAE를 쓰는 이유를 **‘state-dependent distribution + CRPS + calibration + tail risk’**로 고정하고, 반드시 강한 베이스라인(quantile regression, mixture density 등)과 비교

9. 결론

현재까지의 예비 증거는, 추세(position) 기반 state로 단면 모멘텀 프리미엄의 시간 변동을 설명하려는 접근보다, 변동성(intensity) 기반 state에서 “quiet vs volatile, 특히 Bear/volatile에서 급락/반전” 구조를 탐구하는 방향이 더 데이터와 정합적임을 시사한다. 이는 모멘텀 크래시 및 변동성 조건부 리스크 관리 문헌과 연결되며, state 정보를 시퀀스 모델에 주입하는 Cycle-PE 및 분포 예측(CVAE)을 결합할 경제적 동기를 제공한다.
남은 핵심 과제는 (i) H1-Vol의 확증적 검정(비선형/상호작용 포함), (ii) 잔차 타겟 생성 과정의 오염/내생성 방어, (iii) Cycle-PE vs Concat-A 차별성, (iv) CVAE의 “필수성”을 CRPS/캘리브레이션으로 입증하는 것이다.

참고문헌(핵심)

Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds.

Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium: Empirical tests.

Daniel, K., & Moskowitz, T. J. (2016). Momentum crashes.

Barroso, P., & Santa-Clara, P. (2015). Momentum has its moments.

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning.

Chen, L., Kelly, B., & Wu, D. (2023). Deep Learning in Asset Pricing.

López de Prado, M. (2018). Advances in Financial Machine Learning (Purged CV/Embargo).

Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap.

Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio.

Lim, B., Arik, S. O., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers.

Jain, S., & Wallace, B. C. (2019). Attention is not Explanation.

Kenneth R. French Data Library (Factors/Portfolios).

FRED: CBOE Volatility Index (VIX).

“네가 바로 들고 가야 할” 실행용 정리(1페이지)

H1-Vol 확증 회귀 3종 세트(사전 고정)

(1) 
𝜆
1
,
𝑡
∼
𝑖
𝑛
𝑡
𝑒
𝑛
𝑠
𝑖
𝑡
𝑦
𝑡
−
1
λ
1,t
	​

∼intensity
t−1
	​


(2) 
+
𝑝
𝑜
𝑠
𝑖
𝑡
𝑖
𝑜
𝑛
𝑡
−
1
+position
t−1
	​


(3) 
+
1
(
𝑝
𝑜
𝑠
𝑖
𝑡
𝑖
𝑜
𝑛
<
0
)
×
𝑖
𝑛
𝑡
𝑒
𝑛
𝑠
𝑖
𝑡
𝑦
+1(position<0)×intensity (+ intensity² 또는 상위 q 더미 중 택1)

generated residual 방어 2개

β-window(60/120) + ridge β(또는 shrinkage)

가능하면 종목 데이터(최소 subset)로 정성 재현

H2 핵심 ablation 2개는 무조건

Cycle-PE vs Concat-A

intensity-only PE vs (position+intensity) PE

CVAE “필수성”을 만드는 보고 패키지

CRPS + coverage + reliability diagram

Quantile regression baseline 필수

해석 가능성은 ‘진단’으로만

gate/attention은 oracle-consistency/perturbation test 같이 보고(과장 금지)