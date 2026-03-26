# Appendix A2: 논문 카드 39선

본 연구에서 가장 중요한 참고 논문 39편. 저자, 연도, venue, 핵심 기여, 우리 연구와의 관계.

---

## I. 경제적 시간과 시간 변환

**1. Clark (1973)** — *Econometrica.* 종속 과정 $X(t)=W(\tau(t))$로 heavy tail 설명. **연구 전체의 이론적 출발점.**

**2. Mandelbrot & Taylor (1967)** — *Operations Research.* 거래량 기반 시간 변환 최초 제안. Clark의 직접 선행.

**3. Ane & Geman (2000)** — *J. Finance.* 거래 시간 기준 수익률이 가우시안에 가까움을 NYSE에서 실증. **tau-RoPE의 경험적 근거.**

**4. Zagatti et al. (2024)** — *AISTATS.* 시간 변환 정리를 신경망에 구현. 학습 가능 $\lambda$의 통계적 건전성 증명. **EOA(Ch.10)의 수학적 기반.**

**5. Lopez de Prado (2018)** — *Wiley.* Volume/dollar bar 등 정보 기반 샘플링. 경제적 시간의 이산적 구현.

**6. Carr et al. (2003)** — *Mathematical Finance.* 시간 변환 기반 확률적 변동성 모델. Clark의 종속 과정을 옵션 가격 결정에 적용. **Ch.10에서 EOA의 50년 적용사 근거.**

**7. Mendoza-Arriaga & Linetsky (2012)** — *Finance and Stochastics.* 시간 변환 마르코프 과정으로 신용 리스크 모델링. **Ch.10에서 Clark 통찰의 금융 확장 사례.**

## II. Transformer와 Positional Encoding

**8. Vaswani et al. (2017)** — *NeurIPS.* Transformer. Self-attention, sinusoidal PE. **모든 것의 기반.**

**9. Su et al. (2021)** — *arXiv.* RoPE. 상대 위치를 회전으로 인코딩. **tau-RoPE의 기저.**

**10. StretchTime / Kim et al. (2026)** — *arXiv 프리프린트.* SyPE(Sp(2,R) 확장). RoPE가 non-affine 워핑 표현 불가 증명. **직접 경쟁 논문.**

**11. KAIROS/DRoPE (2025)** — *arXiv.* FFT 스펙트럴로 RoPE 주파수 변조. 위치가 아닌 주파수 적응.

**12. ElasTST (2024)** — *NeurIPS.* 조정 가능 RoPE 주기. 다양한 예측 수평에 하나의 모델.

**13. T2B-PE (2024)** — *arXiv.* PE 정보가 깊이에 따라 감소("PE 소실"). softmax 압축과 관련.

**14. ALiBi (Press et al., 2022)** — *ICLR.* 선형 바이어스 PE. 외삽에 강하나 데이터 적응적이지 않음.

**15. Katharopoulos et al. (2020)** — *ICML.* Linear Transformer. Softmax를 커널 feature map $\phi$로 대체하여 $O(N)$ attention 구현. **Ch.08에서 선형 attention의 softmax 제거 효과 분석 기반.**

**16. Zeng et al. (2022)** — *AAAI.* "Are Transformers Effective for Time Series Forecasting?" 단순 선형 모델이 Transformer를 능가함을 보여 PE의 실질적 무효성 비판. **Ch.02의 핵심 문제 제기.**

**17. Nie et al. (2023)** — *ICLR.* PatchTST. 시계열을 패치 단위로 토큰화하여 PE 문제를 우회. **Ch.02에서 시간 표현 문제의 실용적 대안으로 논의.**

**18. Kazemi (2019)** — *arXiv.* Time2Vec. 학습 가능 주파수 $\sin(\omega t + \phi)$로 시간 표현. 달력 시간 $t$ 기반의 한계. **Ch.02 PE 비교 테이블에서 기존 접근 사례.**

## III. 컨디셔닝 인터페이스

**19. Perez et al. (2018)** — *AAAI.* FiLM. 시각적 추론에서 곱셈적 컨디셔닝 효과 실증. **Paper 1 핵심 기준선.**

**20. Jayakumar et al. (2020)** — *ICLR.* 곱셈적 상호작용의 이중선형 함수 효율적 표현 증명. **"곱셈적이 항상 좋다"는 주장의 원천.**

**21. Peebles & Xie (2023)** — *ICCV.* DiT. adaLN-Zero(곱셈적)가 concat을 이김. **고 SNR(이미지)에서 이론이 맞는 사례.**

## IV. 귀납적 편향과 표현 학습

**22. Hewitt & Liang (2019)** — *EMNLP.* 프로빙 정확도 ≠ 표현 품질. **"좋은 표현 ≠ 좋은 예측"의 선례.**

**23. Locatello et al. (2019)** — *ICML Best Paper.* 12,800 모델. disentanglement 증가가 하류 과제 미개선. **가장 강력한 표현-과제 괴리 증거.**

**24. Jain & Wallace (2019)** — *NAACL.* 다른 attention 분포 → 동등 예측. **attention 변화 ≠ 예측 변화.**

**25. Pezeshki et al. (2021)** — *NeurIPS.* Gradient starvation. 쉬운 피처 먼저 학습, 어려운 피처 "굶김".

## V. Test-Time Adaptation

**26. TENT (Wang et al., 2021)** — *ICLR.* BN 어파인을 엔트로피 최소화로 적응. TTA 표준 기준.

**27. DynaTTA (Song et al., 2025)** — *ICML Oral.* 동적 인퍼런스 컨텍스트. 시계열 TTA SOTA. **TTPA의 가장 직접적 비교 대상.**

**28. TAFAS (Chen et al., 2025)** — *AAAI.* 시간-주파수 이중 분석 TTA.

**29. CoTTA (Wang et al., 2022)** — *CVPR.* 교사-학생으로 연속 TTA 오류 축적 방지.

**30. RevIN (Kim et al., 2022)** — *ICLR.* 가역적 인스턴스 정규화. 가장 단순한 TTA.

**31. PETSA / Liu et al. (2024)** — *arXiv.* 파라미터 효율적 TTA. 어파인 변환 + 잔차 게이트로 소수 파라미터만 적응. **Ch.09 TTPA 비교 테이블의 직접 대상.**

## VI. 연속 시간 모델

**32. ContiFormer (Chen et al., 2024)** — *NeurIPS.* 이산 attention을 Neural ODE로 연속화. **EOA의 기저 프레임워크.**

**33. Neural ODE (Chen et al., 2018)** — *NeurIPS.* 은닉 상태의 연속 진화. Adjoint method.

**34. Neural CDE (Kidger et al., 2020)** — *NeurIPS.* CDE 기반 비규칙 관측 처리.

**35. S4/Mamba — Gu et al. (2022, 2024)** — *ICLR/arXiv.* 구조화된 상태 공간 모델(S4) 및 선택적 SSM(Mamba). 이산화 스텝 $\Delta t$를 경제적 시간 간격 $\Delta\tau$로 대체 가능. **Ch.10에서 EOA보다 구현이 단순한 실용적 대안으로 논의.**

## VII. 금융 시계열

**36. Gu et al. (2020)** — *Review of Financial Studies.* 금융에서 비선형 모델 체계적 비교. **교차 단면 IC 방법론의 근거.**

**37. Fama & French (1993)** — *J. Financial Economics.* 3-factor 모형과 25 포트폴리오. **데이터셋과 타겟 변수의 원천.**

**38. Daniel & Moskowitz (2016)** — *J. Financial Economics.* 모멘텀 크래시 분석. 모멘텀 전략이 시장 반등 시 급격히 붕괴하는 현상 실증. **Ch.04에서 이론 vs 현실 괴리의 핵심 증거.**

**39. Yang et al. (2018)** — *ICLR.* 어휘 softmax bottleneck. **주의: 어휘 softmax와 attention softmax는 다른 메커니즘.** Ch.05와 구별 필요.

---

## 읽는 순서 추천

**첫 번째 읽기 (경제적 시간 배경):** 1 → 3 → 5 → 6 → 7 → 8 → 9

**Transformer/PE 맥락:** 10 → 16 → 17 → 18 → 15

**인터페이스 비교 맥락:** 19 → 20 → 21 → 25

**실패 분석 맥락:** 22 → 23 → 24 → 10 → 39

**TTA 맥락:** 26 → 27 → 28 → 30 → 31

**연속 시간 맥락:** 32 → 33 → 4 → 34 → 35

---

## 카테고리 색인

| 카테고리 | 번호 |
|---------|------|
| 경제적 시간 | 1-7 |
| Transformer/PE | 8-18 |
| 컨디셔닝 | 19-21 |
| 귀납적 편향 | 22-25 |
| TTA | 26-31 |
| 연속 시간 | 32-35 |
| 금융 ML | 36-39 |
