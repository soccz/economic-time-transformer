# Chapter 10: 연속 시간 경제적 Attention

> *"이산 attention의 시간 축을 바꾸는 대신,*
> *시간 축이 의미를 갖는 프레임워크로 이동하면 어떨까?"*

---

## 이 장의 질문

Ch.05의 세 병목: (1) softmax 압축, (2) 상호작용 접근, (3) 단조성 제약.
Ch.08에서 softmax 제거로 +49%, Ch.09의 TTPA는 병목 3을 해소하되 1-2는 부분적.

> **세 병목을 전부 구조적으로 우회하는 프레임워크가 존재하는가?**

---

## ContiFormer: Neural ODE로 attention을 연속화하다

Chen et al.(NeurIPS 2024)의 ContiFormer는 이산 attention을 연속 시간 동역학으로 확장한다.

**이산 attention:**
$\text{Attention}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d_k}) V$ — 이산 가중합.

**연속 attention (ContiFormer):**
$$\text{Attention}(q, t) = \int_{\mathcal{S}} \alpha(q, K(s)) \cdot V(s) \, ds$$

$K(s)$와 $V(s)$는 Neural ODE를 따라 진화하고, attention 상태도 ODE를 따른다:
$$\frac{dH}{dt} = f_\theta(H(t), t), \qquad H(0) = H_0$$

핵심: **softmax가 사라진다.** 그리고 시간 변수 $t$가 정수 인덱스가 아니라 실수 값의 연속 좌표가 된다. 시간 변수의 *재정의*가 수학적으로 의미를 갖는 프레임워크다.

---

## 핵심 아이디어: ODE의 $t$를 $\tau$로 대체

Clark(1973): $X(t) = W(\tau(t))$, $\tau(t) = \int_0^t \lambda(s) \, ds$

ContiFormer: $dH/dt = f_\theta(H(t), t)$

**제안 — Economic ODE Attention (EOA):**

$$\frac{dH}{d\tau} = f_\theta(H(\tau), \tau), \qquad \tau(t) = \int_0^t \lambda(s) \, ds$$

이 대체의 세 가지 구조적 결과:

1. **적응적 해상도.** 고변동성에서 $\lambda$ 크고 $\tau$ 빠르게 진행 → ODE 스텝이 $t$축에서 조밀해짐. 정보 많은 곳에서 더 세밀하게 계산.

2. **계산 절약.** 저변동성에서 $\lambda$ 작고 $\tau$ 느리게 진행 → $\tau$축의 같은 구간이 $t$축의 더 긴 구간 커버.

3. **자동 단조성.** $\lambda(s) > 0$이면 $\tau$는 자동 단조증가. 별도 제약 불필요.

---

## 시간 변환 정리와의 연결

**고전적 정리 (Dambis-Dubins-Schwarz):** 연속 로컬 마팅게일 $M$에 대해, 적절한 시간 변환 $\tau$를 적용하면 $M(\tau^{-1}(s))$는 표준 브라운 운동이다. **적절한 시간 변환으로 복잡한 과정이 단순해진다.**

**신경망 적용 (Zagatti et al., AISTATS 2024):** 강도 함수 $\lambda^*(t)$에 의한 시간 변환 $\tau(t) = \int_0^t \lambda^*(s) \, ds$가 비균일 점과정을 단위 강도 포아송으로 변환. 학습 가능한 $\lambda$와 결합해도 통계적으로 건전함을 증명.

| $\tau$-시간 | $t$-시간 |
|:---:|:---:|
| 균일한 과정 | 강도 가중 비균일 과정 |
| 등간격 스텝 | 활동 많은 곳에서 조밀한 스텝 |

**EOA는 이 두 이론의 교차점이다.** ContiFormer가 시간 변수 재정의가 의미를 갖는 프레임워크를 제공하고, Zagatti et al.이 시간 변환의 학습 가능성을 증명했다. **두 이론을 결합한 연구는 아직 없다.** Clark(1973)의 subordinated process를 50년 만에 attention 동역학에 직접 구현하는 것이다.

---

## 왜 3대 병목을 전부 우회하는가

| 병목 | 이산 Transformer + RoPE | TTPA (Ch.09) | **EOA** |
|------|------------------------|-------------|---------|
| Softmax 압축 | 치명적 | 부분 완화 | **구조적 회피** (softmax 없음) |
| 상호작용 접근 | 차단됨 | 지속 | **구조적 회피** (ODE 상태 $H$가 전체 표현 포함) |
| 단조성 제약 | 치명적 | 완전 해소 | **설계에 내재** ($\lambda > 0$ → 자동 단조) |

**병목 1:** 연속 ODE attention은 softmax를 사용하지 않는다. Ch.08에서 softmax 제거만으로 +49%가 나왔다는 사실이 이 병목의 심각성을 보여준다.

**병목 2:** ODE 상태 $H(\tau)$는 모든 피처 정보를 포함하며, $f_\theta(H(\tau), \tau)$에서 시간과 피처가 같은 동역학 안에서 상호작용한다.

**병목 3:** $\tau = \int_0^t \lambda(s) \, ds$에서 $\lambda > 0$이면 적분 구조 자체가 단조성을 보장한다. cumsum + softplus 같은 인위적 제약 불필요.

---

## 강도 함수 $\lambda$의 설계 옵션

**A. Rule-based:** $\lambda(t) = \text{softplus}(\alpha \cdot \text{intensity}(t))$. 학습 파라미터 없음.

**B. 학습 가능:** $\lambda_\theta^*(t) = g_\theta(H(t), t)$. ODE 프레임워크 내에서 adjoint method로 gradient 처리 — cumsum gradient 문제 발생하지 않음.

**C. 하이브리드:** $\lambda(t) = \lambda_{\text{rule}}(t) + \epsilon_\theta(t)$. Rule-based로 초기화, 잔차만 학습. $\epsilon_\theta$가 작은 보정만 담당하므로 학습이 안정적이다.

---

## Clark의 종속 과정을 attention에 구현하는 의미

이 연결이 단순한 기술적 트릭이 아닌 이유:

Clark(1973)의 핵심 통찰은 "가격이 정규분포를 따르지 않는 것은 시간이 균일하게 흐르지 않기 때문"이라는 것이었다. 50년 동안 이 통찰은 옵션 가격 결정(Carr et al., 2003), 신용 리스크(Mendoza-Arriaga & Linetsky, 2012), 정보 기반 샘플링(Lopez de Prado, 2018)에 적용되었지만, **attention 동역학에는 적용되지 않았다.** 이산 attention에서 "시간 변수"가 정수 인덱스이므로 연속 시간 변환이 정의되지 않았기 때문이다.

ContiFormer가 이 장벽을 무너뜨렸다. Attention을 연속화한 순간, Clark의 시간 변환이 적용 가능해졌다. EOA는 50년 된 금융 이론, 2년 된 연속 attention, 1년 된 신경망 시간 변환의 교차점에 서 있다.

---

## EOA vs TTPA: 상보적 접근

| 측면 | TTPA | EOA |
|------|------|-----|
| 복잡도 | 낮음 (몇 줄 코드) | 높음 (ODE 솔버 필요) |
| 기존 모델 호환 | 완전 (plug-and-play) | 없음 (새 아키텍처) |
| 병목 해결 | 3번만 완전 | 1-2-3번 전부 |
| 구현 상태 | 예비 실험 완료 | **제안 단계** |
| 실용성 | 즉시 적용 가능 | 연구 단계 |

**경쟁이 아니라 상보적.** TTPA는 지금 바로 적용 가능한 해결책, EOA는 근본적이지만 구현 비용이 높은 이론적 해결책이다.

---

## 정직한 현재 상태: 제안 단계

EOA는 이론만 있고 **구현도 실험도 없다.** 구현상 도전:

1. **ODE 솔버 비용.** Neural ODE의 forward pass는 이산 attention보다 느리다.
2. **수치 안정성.** $\lambda$가 급변하는 금융 데이터에서 ODE stiffness 문제 가능.
3. **ContiFormer 재구현.** 공개 코드를 금융 시계열에 적응시키는 공학적 작업 필요.

예상 소요: ContiFormer 재구현(3-4주) + $\lambda$ 구현(1주) + 기본 실험(1-2주) + 학습 가능 $\lambda$(2-3주) = **최소 2-3개월.** 박사 프로젝트 규모.

---

## ContiFormer 너머

EOA의 아이디어는 다른 연속 시간 변종에도 적용 가능하다:
- **Neural CDE** (Kidger et al., 2020): 제어 경로에 경제적 시간 반영
- **Mamba/S4** (Gu et al., 2022, 2024): 상태 공간 모델의 이산화 스텝 $\Delta t$를 경제적 시간 간격 $\Delta \tau$로 대체. SSM이 시장 활동에 비례하는 해상도를 갖게 된다. EOA보다 구현이 단순할 수 있으며, 실용적 첫 번째 실험 대상으로 유망하다.

---

## 이 장의 핵심

1. ContiFormer(NeurIPS 2024)가 시간 변수 재정의가 의미를 갖는 프레임워크를 제공했다.
2. EOA는 ODE 시간을 $\tau = \int \lambda(s) \, ds$로 대체. Clark(1973)을 attention에 직접 구현.
3. 시간 변환 정리(Zagatti et al., AISTATS 2024)가 수학적 정당성을 제공한다.
4. 3대 병목 전부를 구조적으로 우회한다.
5. 현재 상태는 **제안 단계** — 최소 2-3개월의 구현 작업이 필요.
6. TTPA와 상보적: TTPA는 실용적 해결, EOA는 이론적 해결.

다음 장에서는 한 발 물러서서, 연구 전체의 현재 위치와 남은 거리를 정직하게 평가한다.
