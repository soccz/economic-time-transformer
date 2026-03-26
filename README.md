# Economic Time in Financial Transformers

Clark(1973)의 경제적 시간을 금융 Transformer에 구현하려다 배운 것들.

## 연구 개요

시장 상태(변동성, 모멘텀)를 Transformer에 **어떻게** 주입하느냐에 따라 예측 성능이 극적으로 달라진다. 이론적으로 우월한 방법(FiLM, 곱셈적 컨디셔닝)이 실제로는 단순 concatenation에 졌고, 학습된 시간 좌표가 attention을 바꾸면서도 예측은 개선하지 못했다. 이 실패를 분석하는 과정에서 발견한 것들을 4편의 논문 초고와 ~400회 실험으로 정리했다.

## 핵심 발견

| 발견 | 근거 |
|------|------|
| SNR이 낮으면 곱셈적 컨디셔닝이 실패한다 | 합성 실험 SNR*≈0.2, 고SNR 대조군 역전 |
| 예측 신호는 채널 간 상호작용에 있다 | F=14.335, p=6.09e-07 (intensity×indexret) |
| Softmax가 시간 좌표 신호를 압축한다 | linear attention 수술 +49% (3시드 일관) |
| test-time PE 적응이 유의미한 개선을 준다 | TTPA alpha=0.5, p=0.011 |

## 논문 초고 (4편)

| # | 제목 | 파일 |
|---|------|------|
| 1 | 곱셈적 컨디셔닝이 실패하는 조건 | `paper_drafts/paper_1_when_multiplicative_conditioning_fails.md` |
| 2 | 잘못된 귀납적 편향: 3대 병목 | `paper_drafts/paper_2_representation_utility_gap.md` |
| 3 | Test-Time Positional Adaptation | `paper_drafts/paper_3_TTPA.md` |
| 4 | Continuous Economic Time Attention | `paper_drafts/paper_4_continuous_economic_time_attention.md` |

## 레포 구조

```
economic-time-transformer/
├── guide/              ← 연구 해설집 마크다운 16챕터
│   ├── 00_overview.md ~ 11_next_steps.md  (13 chapters)
│   └── A1~A3  (appendices)
├── models/             ← 모델 구현
│   ├── market_time_model.py   (tau-RoPE, linear attention, FiLM 등)
│   ├── icpe_transformer.py    (기본 Transformer)
│   └── icpe_hybrid_model.py   (하이브리드 아키텍처)
├── experiments/        ← 실험 스크립트 + 결과
│   ├── synthetic_snr_experiment.py
│   ├── noise_injection_experiment.py
│   ├── ttpa_prototype.py
│   └── results/        ← CSV 결과 (~480개)
├── paper/              ← 연구 과정 문서
│   ├── economic_time/  ← 핵심 실험 + 60개 결과 디렉토리
│   ├── method_paper_writing/  ← 17개 판단 기록
│   └── index_conditioned_pe/  ← 기반 실험
├── paper_drafts/       ← 논문 초고 4편 + 감사 보고서
├── paper_test/         ← 추가 실험 스크립트
└── *.md                ← AETHER_IDEA, DESIGN_SPEC 등
```

## 연구 해설집

> [soccz.github.io/projects/economic-time-research-guide/](https://soccz.github.io/projects/economic-time-research-guide/)

실패 과정, 사고 흔적, 방향 전환을 블로그 형태로 정리한 13챕터 + 3부록.

## 실험 재현

```bash
# 환경
pip install torch pandas scipy statsmodels

# 채널 분해 실험 (Paper 1)
PYTHONPATH=. python -m experiments.economic_time_supervised \
  --index-symbol "^GSPC" --start 2022-01-01 --end 2024-12-31 \
  --epochs 3 --seed 7 \
  --model-kinds "static,concat_a,concat_a:intensity_only,concat_a:indexret_only"

# 합성 SNR 실험
python experiments/synthetic_snr_experiment.py

# TTPA 프로토타입
python experiments/ttpa_prototype.py
```

상세: `guide/A3_reproduction.md` 참조.

## 참고 논문

39편 → `guide/A2_paper_cards.md`
