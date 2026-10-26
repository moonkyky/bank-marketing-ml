# Bank Marketing ML

**정기예금 가입 가능성에 따른 은행 마케팅 레코드의 재현 가능한 순위 예측 프로젝트.**

[UCI Bank Marketing 데이터셋](https://archive.ics.uci.edu/dataset/222/bank+marketing)을 사용해, 연락 가능한 용량이 제한된 상황에서 아웃바운드 마케팅 레코드의 우선순위를 정하는 데이터 누수 방지 중심의 머신러닝 워크플로를 개발한다.

현재 데이터 이해, 전처리, baseline 모델링과 모델 실험을 다룬다. 독립적인 Test 평가, 재사용 가능한 패키징, 배치 추론과 오프라인 모니터링은 이후 작업으로 계획되어 있다.

## Project Status

| 영역 | 상태 |
|---|---|
| 문제와 데이터(1~3단계) | 완료 |
| 4단계 — Baseline 모델링 | 완료 |
| 5단계 — 모델 실험 | Sentinel을 분리한 baseline 기준으로 수정 진행 중 |
| 6단계 — 최종 평가 | 수정된 모델 실험의 선택 결과 대기 |
| 엔지니어링과 배포(7~9단계) | 미시작 |
| 운영과 개선(10~12단계) | 미시작 |

상세 진행 상태, 결정, 검증과 다음 작업 인계는 [프로젝트 진행 기록](docs/project-status.md)에서 관리한다.

## Key Validation Results

Notebook 03은 `pdays`의 sentinel을 분리한 전처리로 새 커널에서 실행되었다. 아래 값은 현재의 **baseline Validation 결과**이며, 최종 모델 선택 결과나 Test 평가 결과가 아니다. 이번 한국어 문서 변경에서는 모델을 재실행하지 않았다.

| 후보 | Average Precision | ROC-AUC | Lift@10% |
|---|---:|---:|---:|
| Sentinel을 분리한 Logistic Regression baseline | 0.288234 | 0.688023 | 2.705123 |

상위 10% 용량에서 baseline은 Validation 가입 레코드 1,142개 중 309개를 상위 824개 안에 배치한다. Precision@10%는 `0.375000`, Recall@10%는 `0.270578`이다.

이전의 원본 `pdays` baseline AP는 `0.297231`이었다. 현재 표현은 AP가 `0.008997` 낮아졌지만, 변수의 의미를 보존하기 위해 sentinel 여부와 경과 일수를 분리한다. Notebook 04에서 이 baseline을 재현했으며, 모델 계열 비교 코드는 작성했지만 실행하지 않았다. Tuning과 최종 선택은 아직 남아 있다. Test는 최종 평가를 위해 보류한다.

## Problem Definition

| 항목 | 정의 |
|---|---|
| 업무 의사결정 | 제한된 연락 용량 안에서 연락 대상 자격을 갖춘 아웃바운드 마케팅 레코드의 우선순위 결정 |
| ML 과제 | 레코드 순위 산정에 사용하는 이진 분류 |
| 예측 시점 | 예정된 연락 직전 |
| 분석 단위 | 캠페인 레코드 1개이며 고유 고객 식별자는 없음 |
| Target | 정기예금 가입 여부 `y` |
| 양성 클래스 | `yes = 1` |
| 모델 출력 | 양성 클래스 순위 점수: baseline은 `predict_proba`, SVM 후보는 `decision_function` 사용 |
| 주요 선택 지표 | Average Precision |

고정 임계값 `0.5`로 운영 결정을 내리는 대신 점수에 따라 레코드의 순위를 정한다. 점수를 검증된 가입 확률로 취급하지 않으며, 예측 성능은 연락의 인과효과나 수익성을 입증하지 않는다.

## Data

UCI Bank Marketing 데이터셋의 `bank-additional-full.csv`를 사용한다.

| 속성 | 값 |
|---|---|
| 레코드 수 | 41,188 |
| 원본 예측 변수 | 20개 |
| 모델링 예측 변수 | 19개 |
| 제외 변수 | `duration` |
| Target | `y` |
| 파일 구분자 | 세미콜론 (`;`) |
| 수집 기간 | 2008년 5월~2010년 11월 |
| 로컬 경로 | `data/raw/bank-additional/bank-additional-full.csv` |

원본 데이터는 Git에서 제외한다. 제공된 다운로드 스크립트로 로컬에 준비한다.

### Feature and Leakage Policy

- 현재 연락의 소요 시간은 연락이 끝나기 전에 알 수 없으므로 `duration`을 제외한다.
- 기존의 예측 시점 정책에 따라 나머지 예측 변수 19개를 유지한다.
- 연락 채널은 점수 계산 전에 결정되므로 `contact`를 사용할 수 있다고 본다.
- 거시경제 값은 점수 계산 시점에 공개되어 사용할 수 있었다고 가정한다. 실제 운영을 위한 데이터 계약에서는 발표 지연과 수정 이력을 확인해야 한다.
- 문자열 `"unknown"`은 임의로 결측값으로 바꾸지 않고 관측된 범주로 유지한다.

## Methodology

### Chronological Split

원본 행 순서를 유지한다. 달력상 기간을 재구성하지 않고 레코드 비율에 따라 분할한다.

| 구분 | 행 수 | 용도 |
|---|---:|---|
| Train | 28,831 | 전처리와 개발 모델 학습 |
| Validation | 8,238 | 실험 비교와 최종 구성 선택 |
| Test | 4,119 | 확정한 최종 구성 평가 |

셔플, 층화 분할, `month`나 `day_of_week`에 따른 재정렬을 하지 않는다. 양성 비율은 Train 약 5.57%, Validation 약 13.86%로, 시간에 따른 분포와 양성 비율의 변화가 주요 한계이다.

과거 EDA에서 Test 집계 정보를 확인했다. 따라서 Test가 모든 측면에서 완전히 미관측이라고 표현하지 않는다. 다만 feature engineering, tuning, 모델 선택, 순위 정책 선택에는 사용하지 않는다.

### Preprocessing

개발 중에는 baseline의 scikit-learn `Pipeline`에 포함된 모든 학습 단계를 Train에서만 fit한다.

| 분기 | 설정 |
|---|---|
| 수치형 | 일반 수치형 feature 8개에 `StandardScaler` 적용 |
| `pdays` | `pdays_previously_contacted = (pdays != 999)`와 `pdays_elapsed` 파생. Sentinel 행의 경과 일수는 `0`, 나머지는 기록된 일수로 두고 두 열 모두 scaling |
| 범주형 | `OneHotEncoder(handle_unknown="ignore")` |
| 결측값 대체 | 개발 입력에서 인식된 결측값이 없어 적용하지 않음 |
| 기타 feature engineering | 적용하지 않음 |

원본 입력 19개는 수치형 8개, one-hot 범주형 52개, sentinel을 분리한 `pdays` 2개로 변환되어 총 62개 feature가 된다. 원본의 `999`를 경과 일수로 사용하지 않는다.

교차 검증에서는 각 training fold 안에서 전처리와 모델을 포함한 전체 Pipeline을 독립적으로 fit한다.

### Model Selection

과거 실험에서는 Train 내부의 확장형 `TimeSeriesSplit` fold 4개로 모델 계열과 hyperparameter를 비교한 뒤, 소수 후보를 Validation에서 비교했다. 원본 `pdays`를 사용한 당시 선택은 수정된 표현 정책에 따라 대체되었으며 현재 최종 구성으로 사용할 수 없다.

현재는 동일한 sentinel 전처리와 Train 내부 fold를 사용해 Logistic Regression, Random Forest, RBF SVM을 비교하는 코드를 작성했으며 실행은 대기 중이다. 모든 후보와 fold에 sentinel을 분리한 전처리를 적용해 비교와 tuning을 다시 수행해야 한다. 최종 평가를 위한 구성은 아직 확정하지 않았다. 모델과 평가 규칙을 확정한 뒤 모든 학습 단계를 Train+Validation에서 다시 fit하고 Test를 한 번 평가한다.

### Evaluation

- 주요 지표: Average Precision
- 보조 지표: ROC-AUC
- 용량별 지표: 상위 5%, 10%, 20%의 Precision@K, Recall@K, Lift@K
- 순위: 반올림하지 않은 점수를 내림차순으로 정렬하고, 동점은 stable sort로 원본 행 순서 유지
- 용량 크기: `K = ceil(n_records × fraction)`
- Lift 기준: 평가 대상 데이터의 양성 비율

Top-K는 레코드 단위이며 평가 데이터 전체를 하나의 대상 집합으로 순위화한다. 일일 할당량, 반복 연락 규칙, 고유 고객 단위 제약을 재현하지 않는다.

## Repository Structure

```text
bank-marketing-ml/
├── data/                       # 로컬 원본 데이터; Git에서 제외
├── docs/
│   └── project-status.md       # 진행 상태, 결정, 인계 기록
├── notebooks/
│   ├── 01-data-understanding.ipynb
│   ├── 02-preprocessing.ipynb
│   ├── 03-baseline-modeling.ipynb
│   └── 04-model-experiments.ipynb
├── scripts/
│   └── download_data.py        # UCI 데이터 다운로드와 압축 해제
├── src/bank_marketing_ml/      # 패키지 골격; 재사용 코드 정리는 이후 진행
├── tests/                      # 현재 최소한의 smoke test
├── pyproject.toml
├── uv.lock
└── README.md
```

재사용 가능한 모듈과 의미 있는 단위·통합 테스트는 코드 리팩터링과 패키징 과정에서 구축한다. 현재 `src/` 진입점과 smoke test는 초기 골격이며, 출시된 학습·추론 인터페이스가 아니다.

## Getting Started

### Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Git

Windows 11과 WSL2 Ubuntu 24.04 환경에서 개발했다. 아래 명령은 저장소 루트에서 실행한다.

### Installation

```bash
git clone https://github.com/moonkyky/bank-marketing-ml.git
cd bank-marketing-ml
uv sync --locked
```

### Data Preparation

```bash
uv run python scripts/download_data.py
```

생성될 파일 경로:

```text
data/raw/bank-additional/bank-additional-full.csv
```

### Notebook Workflow

저장소 루트에서 Jupyter를 시작한다.

```bash
uv run jupyter lab
```

Notebook 01~03을 순서대로 확인한다. Notebook 04는 baseline 재현까지 실행했으며, 모델 계열 비교 코드는 실행 대기 중이고 tuning은 아직 구현하지 않았다.

1. [데이터 이해](notebooks/01-data-understanding.ipynb)
2. [전처리](notebooks/02-preprocessing.ipynb)
3. [Baseline 모델링](notebooks/03-baseline-modeling.ipynb)
4. [모델 실험](notebooks/04-model-experiments.ipynb) — 수정 진행 중

각 노트북은 데이터를 직접 로딩하며 새 커널에서 위에서 아래로 실행할 수 있어야 한다. Notebook 05와 정식 학습·점수 계산 명령은 아직 구현하지 않았다.

## Verification

현재 제공되는 저장소 검사를 실행한다.

```bash
uv run pytest
uv run ruff check .
```

현재 테스트 범위는 최소한이며 모델링 Pipeline을 검증하지 않는다. 코드 리팩터링과 패키징 과정에서 임시 테스트를 의미 있는 단위·통합 테스트로 대체할 계획이다.

## Intended Use

- 정기예금 가입 순위의 오프라인 분석
- 명시적인 연락 용량 시나리오별 레코드 비교
- 재현 가능한 전체 ML 생명주기의 구현 예시
- 출시 준비 이후 로컬 CSV 배치 점수 계산

## Prohibited or Unsupported Use

- 연락이 가입을 유발한다고 주장하는 것
- Lift@K만으로 수익성이나 투자수익률을 주장하는 것
- 보정하지 않은 점수를 검증된 확률로 취급하는 것
- 레코드 하나가 고유 고객 한 명이라고 단정하는 것
- 운영 검증과 거버넌스 없이 실제 의사결정을 자동화하는 것
- 현재 패키지 골격을 출시된 학습·추론 서비스로 사용하는 것

## Limitations

- 고유 고객 ID가 없어 반복 고객과 데이터 분할 간 고객 중복을 신뢰성 있게 확인할 수 없다.
- 완전한 연락 시각이 없어 정확한 달력상 경계와 운영 일정을 재구성할 수 없다.
- 시간 순서에 따른 단일 Validation 기간만으로 모든 미래 기간의 안정성을 입증할 수 없다.
- 과거에 연락한 레코드의 결과가 다른 연락 대상 집단에도 일반화된다고 볼 수 없다.
- 예측 시점의 거시경제 값 가용성, 발표 지연, 수정 이력은 운영 관점에서 확인해야 한다.
- 확률 보정과 연락의 인과효과는 확인하지 않았다.
- 연락 용량, 비용, 가입 가치, 업무 수용 기준은 정해지지 않았다.
- 독립적인 Test 평가는 아직 완료하지 않았다.

## Roadmap

- [x] 1단계 — 문제 정의
- [x] 2단계 — 데이터 이해
- [x] 3단계 — 데이터 전처리
- [x] 4단계 — Baseline 모델링
- [ ] 5단계 — 모델 실험: sentinel을 분리한 baseline 기준으로 수정
- [ ] 6단계 — 최종 평가
- [ ] 7단계 — 리팩터링과 패키징
- [ ] 8단계 — ML Pipeline과 출시 준비
- [ ] 9단계 — 모델 배포와 통합
- [ ] 10단계 — 운영 모니터링 시뮬레이션
- [ ] 11단계 — 재학습과 지속적인 개선
- [ ] 12단계 — 유지보수와 거버넌스

## Documentation

- [프로젝트 진행 기록](docs/project-status.md): 현재 진행 상태, 확인된 결정, 다음 작업 인계
- [데이터 이해](notebooks/01-data-understanding.ipynb): 데이터 확인, EDA, 누수 검토, 분할 설계
- [전처리](notebooks/02-preprocessing.ipynb): feature 그룹과 Train에서만 학습하는 전처리
- [Baseline 모델링](notebooks/03-baseline-modeling.ipynb): baseline 학습과 Validation 평가
- [모델 실험](notebooks/04-model-experiments.ipynb): 모델 비교, tuning, 분석, 최종 선택

## References

- [UCI Bank Marketing 데이터셋](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [데이터셋 DOI: 10.24432/C5K306](https://doi.org/10.24432/C5K306)
