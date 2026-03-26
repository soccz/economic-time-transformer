# Appendix A1: 수식 사전

본 가이드에서 반복 등장하는 핵심 수식을 정리한다. 정의, 직관, 등장 챕터 포함.

---

## 1. Clark의 종속 과정
$$X(t) = W(\tau(t)), \quad \tau(t) = \int_0^t \lambda(s) \, ds$$
$W$: 브라운 운동, $\tau$: 경제적 시간, $\lambda$: 시간 강도. 가격은 시장 활동 강도에 비례하는 경제적 시간에 따라 움직인다. — Ch.01, 10

## 2. 조건부 수익률 분포
$$r_t \mid \tau_t \sim \mathcal{N}(0, \sigma^2 \tau_t)$$
$\tau_t$ 크면 분산 크고(활발한 시장), 작으면 분산 작다. Fat tail과 volatility clustering의 기원. — Ch.01

## 3. RoPE
$$\text{RoPE}(x_m, m) = x_m \cdot e^{im\theta}, \quad \theta_k = 10000^{-2k/d}$$
위치 $m$에서 벡터를 $m\theta$ 만큼 회전. QK dot product에서 상대 위치 $(m-n)$만 영향. — Ch.02, 03

## 4. tau-RoPE
$$\text{tau-RoPE}(x_m, \tau_m) = x_m \cdot e^{i\tau_m\theta}, \quad \tau_m = \text{cumsum}(\text{softplus}(\alpha \cdot \text{intensity}))_m$$
정수 위치를 경제적 시간으로 대체. 고변동성에서 $\tau$ 빠르게 증가. — Ch.03, 05

## 5. FiLM
$$h = \gamma(c) \odot f(Wx + b) + \beta(c)$$
컨텍스트 $c$가 은닉 표현을 채널별 스케일/시프트. 이중선형 함수를 효율적으로 표현. — Ch.03, 04

## 6. Softmax Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$
작은 QK 섭동($\sim 10^{-5}$)이 softmax 통과 후 소멸 (winner-take-all). — Ch.02, 05, 08

## 7. Linear Attention
$$\text{LinearAttn}(Q, K, V) = \frac{\phi(Q) \cdot (\phi(K)^T V)}{\phi(Q) \cdot (\phi(K)^T \mathbf{1})}$$
$\phi$: 커널 특징 맵. Softmax 없이 시간 좌표 변화가 보존됨. — Ch.08

## 8. IC (Information Coefficient)
$$\text{IC}_t = \rho_{\text{Spearman}}(\hat{y}_t, y_t), \quad \text{Mean IC} = \frac{1}{T} \sum_t \text{IC}_t$$
일별 교차 단면 순위 상관 평균. 모델의 랭킹 능력 측정. — 전체

## 9. Newey-West HAC 표준오차
$$\hat{\sigma}_{\text{NW}}^2 = \hat{\Gamma}_0 + 2 \sum_{j=1}^{L} \left(1 - \frac{j}{L+1}\right) \hat{\Gamma}_j$$
자기상관 보정 표준오차. 본 연구에서 $L=4$. — Ch.04, 06, 09

## 10. PID (부분 정보 분해)
$$I(X_1, X_2; Y) = R + U_1 + U_2 + S$$
$R$: 중복, $U$: 고유, $S$: 시너지. intensity와 indexret 개별 IC 낮지만 결합 IC 높으면 시너지 $S$ 큼. — Ch.07

## 11. F-test (증분적 식별)
$$F = \frac{(R^2_{\text{full}} - R^2_{\text{reduced}}) / q}{(1 - R^2_{\text{full}}) / (n - p)}$$
상호작용 항이 예측력을 유의하게 개선하는지 검정. $F = 14.335, p = 6.09 \times 10^{-7}$. — Ch.07

## 12. TTPA tau 계산
$$\tau_t = \text{normalize}\left(\text{cumsum}\left(1 + \alpha(\text{intensity}_t - 0.5)\right)\right)$$
alpha=0이면 $\tau_t = t$. alpha>0이면 고활동 구간에서 시간 빨리 흐름. 추론 시점에만 적용. — Ch.09

## 13. EOA (Economic ODE Attention)
$$\frac{dH}{d\tau} = f_\theta(H(\tau), \tau), \quad \tau(t) = \int_0^t \lambda(s) \, ds$$
ContiFormer ODE 시간을 경제적 시간으로 대체. 고변동성에서 자동 해상도 증가. — Ch.10

## 14. 시간 변환 정리
$$\tau(t) = \int_0^t \lambda^*(s) \, ds \implies N(\tau) \sim \text{Poisson}(1)$$
적절한 시간 변환으로 비균일 과정이 균일 과정으로 변환. — Ch.10

## 15. cumsum gradient (단조성 제약)
$$\frac{\partial \mathcal{L}}{\partial \text{step}_t} = \sum_{t' \geq t} \frac{\partial \mathcal{L}}{\partial \tau_{t'}}$$
누적합 gradient가 모든 step을 상수로 유도 → $\tau \approx c \cdot t$. Learned tau 실패의 핵심 원인. — Ch.05, 09

## 16. StretchTime 정리 (프리프린트)
Non-affine 워핑 $\tau$에 대해: $\nexists \theta : \theta(m-n) = w_0(\tau(m) - \tau(n)) \pmod{2\pi}$.
RoPE는 비선형 시간 워핑 표현 불가. — Ch.05

## 17. 이중선형 함수 클래스
$$\mathcal{F}_{\text{bilinear}} = \{(x, c) \mapsto x^T W c\}$$
곱셈적 네트워크: $O(dk)$로 표현. 덧셈적: 더 많은 뉴런 필요. **잡음 없는 환경에서 증명.** — Ch.04

---

## 18. Cohen's d (효과 크기)
$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}$$
두 조건 간 차이를 표준편차 단위로 표현. |d| < 0.2: 작음, 0.2-0.8: 중간, > 0.8: 큼. — Ch.06, 09

## 19. Holm-Bonferroni 보정
다중 비교에서 1종 오류 제어. p-value를 오름차순 정렬 후, $k$번째 p-value를 $\alpha / (m - k + 1)$과 비교. — Ch.09 (6개 RQ에 적용)

---

## 색인

| # | 키워드 | 챕터 |
|---|--------|------|
| 1 | 종속 과정 | 01, 10 |
| 2 | 조건부 분포 | 01 |
| 3 | RoPE | 02, 03 |
| 4 | tau-RoPE | 03, 05 |
| 5 | FiLM | 03, 04 |
| 6-7 | Softmax / Linear Attention | 02, 05, 08 |
| 8 | IC | 전체 |
| 9 | Newey-West | 04, 06, 09 |
| 10-11 | PID / F-test | 07 |
| 12 | TTPA | 09 |
| 13-14 | EOA / 시간 변환 | 10 |
| 15 | cumsum gradient | 05, 09 |
| 16 | StretchTime | 05 |
| 17 | 이중선형 | 04 |
| 18 | Cohen's d | 06, 09 |
| 19 | Holm-Bonferroni | 09 |
