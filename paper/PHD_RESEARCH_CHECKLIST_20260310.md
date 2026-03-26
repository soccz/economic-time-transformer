# PhD Research Checklist

## 목적

이 문서는 현재 연구가 박사과정 수준에서 얼마나 정리되어 있는지 냉정하게 점검하기 위한 체크리스트다.  
평가 기준은 "아이디어가 멋있는가"가 아니라 "논문으로 방어 가능한가"다.

평가 표기:

- `[완료]` 현재 저장소 기준 충족
- `[부분]` 방향은 맞지만 아직 약함
- `[미완료]` 아직 본문 주장으로 쓰면 위험

---

## 1. 문제 정의

### 1.1 메인 질문이 하나로 고정되어 있는가

- `[완료]`
- 현재 가장 좋은 메인 질문:
  - `conditioning space가 ranking과 absolute prediction trade-off를 어떻게 바꾸는가`

메모:

- 초반의 residual momentum, PE, uncertainty, interpretation, non-Markov 아이디어가 많았지만
- 최근 경로에서는 `input-space vs coordinate-space`로 충분히 수렴했다

### 1.2 연구 질문이 데이터와 결과로 답할 수 있는 질문인가

- `[완료]`

메모:

- `tau_rope vs concat_a vs static`은 현재 저장소의 코드와 결과로 직접 답할 수 있다
- 반면 “시장을 manifold로 본다”, “latent regime을 완전히 해석한다” 수준은 아직 데이터/실험이 부족하다

### 1.3 기존 문헌과의 차별점이 과장 없이 설명되는가

- `[부분]`

현재 가능한 설명:

- novelty는 “완전히 새로운 architecture”가 아니라
- `conditioning space` 자체를 문제로 삼았다는 데 있다

주의:

- `PE superiority`로 쓰면 과장
- `new theory of time`로 쓰면 과장

---

## 2. 주장 설계

### 2.1 메인 claim이 결과로 지지되는가

- `[완료]`

현재 메인 claim:

- `coordinate-space conditioning은 high-volatility에서 MAE에 유리하고, IC는 한계적 개선을 보인다`

이 claim은 confirmatory 결과와 일치한다.

### 2.2 claim이 결과보다 앞서가지 않는가

- `[부분]`

위험한 과장 포인트:

- `tau_rope가 concat_a보다 전반적으로 우월하다`
- `attention이 흐름 해석을 증명한다`
- `CVAE가 핵심 기여다`

이 셋은 현재 본문 주장으로 쓰면 안 된다.

### 2.3 negative result를 정직하게 처리하고 있는가

- `[완료]`

좋은 점:

- `IC-PE > concat_a`가 안 된다는 사실을 문서에서 이미 인정하고 있다
- attention masking 실패도 기록되어 있다

이건 오히려 박사과정 수준에서는 강점이다.

---

## 3. 실험 설계

### 3.1 exploratory와 confirmatory가 구분되어 있는가

- `[완료]`

근거:

- `2022-2024` exploratory
- `2020-2024` confirmatory
- preregistration 문서가 존재함

### 3.2 비교 대상이 공정한가

- `[완료]`

현재 메인 비교:

- `static`
- `concat_a`
- `tau_rope`

이 셋은 문제 정의와 정합적이다.

### 3.3 baseline이 강한가

- `[완료]`

메모:

- `concat_a`를 약한 strawman으로 두지 않고 진짜 control-to-beat로 유지한 점이 좋다

### 3.4 실험이 너무 많아서 메시지가 흐려지지 않는가

- `[부분]`

위험:

- `flow_pe`
- `econ_time`
- `pe_only`
- `qk_only`
- `learned_tau_rope`
- `CVAE`

이걸 본문에 다 넣으면 메시지가 깨진다.

대응:

- 메인 본문은 `static / concat_a / tau_rope`만 유지

---

## 4. 통계적 방어력

### 4.1 메인 결과에 통계 검정이 있는가

- `[완료]`

현재 확보:

- Newey-West one-sided confirmatory test

### 4.2 결과 해석이 p-value에 맞게 절제되어 있는가

- `[부분]`

정확한 해석:

- IC: marginal
- MAE: supported

부정확한 해석:

- IC도 강하게 입증됐다

### 4.3 시장별 breakdown이 있는가

- `[완료]`

좋은 점:

- pooled만이 아니라 `GSPC`, `IXIC` 개별 breakdown도 존재

### 4.4 robustness가 충분한가

- `[부분]`

있는 것:

- anchor robustness (`GSPC`, `IXIC`)
- exploratory vs confirmatory split

아직 약한 것:

- seed stability를 본문 main result와 더 직접 연결하는 정리
- 더 넓은 데이터셋 확장

---

## 5. 모델-이론 정합성

### 5.1 아키텍처가 질문에 맞게 설계되어 있는가

- `[완료]`

현재 정합성:

- `concat_a`: 입력공간 조건화
- `tau_rope`: 좌표공간 조건화

이 비교는 질문과 정확히 맞물린다.

### 5.2 불필요한 모듈이 섞여 있지 않은가

- `[부분]`

문제:

- 현재 저장소에는 옛 경로의 흔적이 많다
- residual momentum, CVAE, explainability, coordinate warp 등이 동시에 남아 있다

대응:

- 본문 경계만 엄격히 자르면 해결 가능

### 5.3 attention 해석이 과도하지 않은가

- `[부분]`

현재 올바른 서술:

- descriptive diagnostic

잘못된 서술:

- validated explanation

---

## 6. 박사과정 수준의 연구 감각

### 6.1 아이디어를 버릴 줄 아는가

- `[완료]`

좋은 점:

- `IC-PE > concat_a`를 억지로 밀지 않고 다운그레이드했다
- negative result를 숨기지 않았다

### 6.2 큰 철학과 작은 실험 사이를 연결하는가

- `[완료]`

현재 연결:

- 큰 철학: 시장 상태로 시간을 다시 본다
- 작은 실험: conditioning space에 따라 IC/MAE trade-off가 달라지는가

이건 박사과정 연구로 충분히 괜찮은 연결이다.

### 6.3 논문 하나에 모든 걸 넣으려는 유혹을 통제하고 있는가

- `[부분]`

현재 가장 큰 리스크:

- VAE
- explainability
- non-Markov
- finance theory
- PE
- routing

를 한 편에 다 넣고 싶어지는 것

판단:

- 연구 프로그램으로는 좋다
- 단일 논문으로는 과하다

---

## 7. 지금 당장 본문에 넣어도 되는 것

- `[완료]`

넣어도 되는 것:

- conditioning space 문제 정의
- `static`, `concat_a`, `tau_rope` 비교
- high-volatility confirmatory result
- MAE 우위 + IC 한계적 우위
- `static` 대비 explicit conditioning의 필요성

---

## 8. 지금 당장 appendix로 보내야 하는 것

- `[완료]`

appendix / future work:

- `flow_pe`
- `learned_tau_rope`
- `econ_time`
- `pe_only`
- `qk_only`
- CVAE / uncertainty
- attention masking
- residual momentum H1 라인

---

## 9. 현재 상태 총평

### 강점

- 방향성이 분명하다
- 문제 정의가 점점 좋아졌다
- negative result를 정직하게 반영했다
- confirmatory test까지 간 점이 좋다
- `concat_a`를 강한 baseline으로 인정한 점이 좋다

### 약점

- 아직도 아이디어가 많아서 본문이 쉽게 퍼질 수 있다
- IC 우위는 약하다
- explainability와 uncertainty는 아직 본문 중심이 아니다
- 상위 저널급 finance framing으로는 아직 부족하다

### 냉정한 판정

- 연구 방향: 맞다
- 박사과정 프로젝트로서의 일관성: 충분하다
- 현재 논문 상태: “좋은 방향의 exploratory + confirmatory 혼합 상태”
- 지금 필요한 것: 새 아이디어 추가가 아니라 메인 주장 축소와 집필 고정

---

## 10. 최종 판정

### 질문: 현재 제대로 방향은 잡아진 상태로 연구하고 있는가?

- `예`

단, 정확한 표현은 이렇다.

> 방향은 제대로 잡혔다.
> 이제 필요한 것은 방향 탐색이 아니라,
> 논문으로 방어 가능한 범위까지 주장을 줄이고 고정하는 일이다.

### 질문: 박사과정 수준에서 괜찮은가?

- `예`

이유:

- 아이디어가 representation 문제로 정리되었다
- 모델과 실험이 그 질문에 맞게 정렬되었다
- 결과가 약한 부분도 이미 구분되어 있다

### 질문: 지금 당장 제일 중요한 한 가지는 무엇인가?

- `본문에서 무엇을 버릴지 결정하는 것`

가장 우선순위 높은 실행 항목:

1. 메인 모델을 `static / concat_a / tau_rope`로 고정
2. confirmatory `2020-2024` 결과를 메인 테이블로 고정
3. MAE 중심, IC 보조의 서술로 수정
4. attention/CVAE/non-Markov는 후속으로 분리
