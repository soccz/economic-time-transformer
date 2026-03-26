# Appendix A3: 실험 재현 가이드

---

## 1. 환경 설정

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch numpy pandas scipy scikit-learn matplotlib yfinance statsmodels
```

### 필수 버전
| 라이브러리 | 최소 버전 |
|-----------|----------|
| Python | 3.10+ |
| PyTorch | 2.0+ |
| pandas | 3.0+ |
| scipy | 1.17+ |
| statsmodels | 0.14+ |

- CUDA 권장 (CPU도 가능), VRAM 8GB+ 권장

### PYTHONPATH 설정

모든 실험 스크립트는 `aaa/` 디렉토리를 패키지 루트로 사용한다. 실행 전 반드시 설정:

```bash
# aaa/ 디렉토리로 이동
cd /path/to/aaa

# PYTHONPATH 설정 (세션 동안 유지)
export PYTHONPATH=$(pwd)

# 또는 매 명령 앞에 붙이기
PYTHONPATH=. python -m paper_test.economic_time_supervised ...
```

## 2. 디렉토리 구조

```
aaa/
├── paper/index_conditioned_pe/   # 모델 코드
│   ├── icpe_transformer.py        # Transformer + PE
│   └── icpe_hybrid_model.py       # 하이브리드 모델
├── paper_test/                    # 실험 스크립트
│   ├── icpe_hybrid_supervised.py   # Paper 1 핵심
│   ├── economic_time_supervised.py # Paper 2 핵심
│   ├── economic_time_confirmatory.py
│   └── finance_incremental_*.py
└── experiments/                   # 추가 실험
    ├── ttpa_prototype.py           # TTPA (Ch.09)
    ├── synthetic_snr_experiment.py  # SNR 제어 (Ch.06)
    ├── noise_injection_experiment.py
    └── high_snr_control/
```

## 3. 데이터 준비

**Ken French 25:** https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- 25_Portfolios_5x5_Daily.csv, F-F_Research_Data_Factors_daily.csv

**앵커 지수:** Yahoo Finance에서 ^GSPC (S&P 500), ^IXIC (Nasdaq) 다운로드.

파생 변수(position, intensity, indexret, mean5, std5)는 실험 코드에서 자동 계산.

## 4. 실험 실행

### Paper 1: 인터페이스 비교 (Ch.03-04)
```bash
python paper_test/icpe_hybrid_supervised.py --mode concat_a
python paper_test/icpe_hybrid_supervised.py --mode film
python paper_test/icpe_hybrid_supervised.py --mode pe_inject
python paper_test/icpe_hybrid_supervised.py --mode tau_rope
python paper_test/icpe_hybrid_supervised.py --mode xip
```

### Paper 1: 채널 분해 (Ch.07)
```bash
python paper_test/icpe_hybrid_supervised.py --mode intensity_only
python paper_test/icpe_hybrid_supervised.py --mode intensity_indexret
python paper_test/icpe_hybrid_supervised.py --mode shuffled_intensity
```

### Paper 1: 증분적 F-test (Ch.07)
```bash
python paper_test/finance_incremental_identification.py
```

### Paper 1: SNR 제어 (Ch.06)
```bash
python experiments/synthetic_snr_experiment.py        # 합성 SNR sweep (시드: {0, 1, 2, 3, 4})
python experiments/noise_injection_experiment.py       # 잡음 주입 절제
bash experiments/high_snr_control/run_experiment.sh    # 고 SNR (이산화)
```

### Paper 2: 경제적 시간 — 채널 분해 (Ch.05)
```bash
PYTHONPATH=. python -m paper_test.economic_time_supervised \
  --index-symbol "^GSPC" --start 2022-01-01 --end 2024-12-31 \
  --epochs 3 --seed 7 \
  --model-kinds "static,concat_a,concat_a:intensity_only,concat_a:position_only,concat_a:indexret_only,concat_a:intensity_indexret,concat_a:no_intensity"
```

### Paper 2: 경제적 시간 — Linear Attention (Ch.05)
```bash
PYTHONPATH=. python -m paper_test.economic_time_supervised \
  --index-symbol "^GSPC" --start 2022-01-01 --end 2024-12-31 \
  --epochs 3 --seed 7 \
  --model-kinds "static,concat_a,tau_rope,tau_rope_linear"
```

### Paper 2: 다중 시드 실험
논문의 보고값은 시드 {7, 17, 27}의 평균이다. 각 시드별로 위 명령을 반복 실행:
```bash
for SEED in 7 17 27; do
  PYTHONPATH=. python -m paper_test.economic_time_supervised \
    --index-symbol "^GSPC" --start 2022-01-01 --end 2024-12-31 \
    --epochs 3 --seed $SEED \
    --model-kinds "static,concat_a,tau_rope"
done
```

### Paper 2: 확인적 분석
```bash
PYTHONPATH=. python -m paper_test.economic_time_confirmatory
```

### Paper 3: TTPA (Ch.09)
```bash
python experiments/ttpa_prototype.py
```
고정 PE 모델 학습 후, alpha sweep {0, 0.5, 1.0, 3.0, 5.0} 자동 평가.

### Synthetic SNR 실험 (Ch.06)
```bash
python experiments/synthetic_snr_experiment.py
```
5개 시드 {0, 1, 2, 3, 4}로 SNR 수준별 IC 차이를 측정한다.

## 5. 핵심 하이퍼파라미터

| 파라미터 | 값 |
|---------|-----|
| seq_len | 30 |
| horizon | 5 |
| d_model | 64 |
| n_heads | 4 |
| n_layers | 2 |
| lr | 1e-3 |
| batch_size | 64 |
| train/val/test | 70/15/15 (날짜 기준) |
| Newey-West lag | 4 |

**알려진 제한:** 현재 구현에는 purge/embargo가 없습니다. seq_len(30) + horizon(5) = 35일의 겹침이 있을 수 있으며, 이는 절대 IC 값을 과대추정할 수 있지만 모델 간 상대 비교에는 영향이 적습니다.

## 6. 결과 파일 위치

| 실험 | 결과 파일 |
|------|----------|
| SNR 제어 | `experiments/synthetic_snr_results.csv` |
| 잡음 주입 | `experiments/noise_injection_results.csv` |
| 고 SNR | `experiments/high_snr_control/aggregated_results.csv` |
| 기타 | 실행 디렉토리에 CSV 출력 |

## 7. 검증 체크리스트

| 수치 | 기대값 | 허용 오차 |
|------|--------|----------|
| concat_a IC | 0.0571 | +/- 0.005 |
| intensity+indexret IC | 0.0592 | +/- 0.005 |
| F-test | 14.335 | +/- 0.5 |
| SNR 교차점 | ~0.2 | +/- 0.1 |
| MAE p-value (고변동성) | 0.0019 | < 0.01 |
| TTPA alpha=0.5 p-value | 0.011 | < 0.05 |

시드에 따라 정확한 수치는 달라지나, 방향성과 유의성은 재현되어야 한다.

## 8. 트러블슈팅

- **데이터 다운로드 실패:** 웹사이트에서 직접 ZIP → CSV 변환.
- **GPU 메모리 부족:** batch_size를 16으로, d_model을 32로 줄여도 방향성 유지.
- **수치 불일치:** 기본 시드는 7 (`--seed 7`). `torch.backends.cudnn.deterministic = True` 확인. 다중 시드 실험은 {7, 17, 27}을 사용.
- **Newey-West p-value 불일치:** lag=3~5로 변경해도 유의성 뒤집히지 않으면 견고.
