# Project Status

Last updated: 2026-11-05

이 문서는 12단계의 결정, 검증 근거, 인계 사항을 기록한다. 프로젝트 소개와 실행 방법은 [README](../README.md)에 정리했다.

## Lifecycle Progress

| Phase | Stage | Status | Primary evidence |
|---|---|---|---|
| 1 — Problem & Data | 1. Problem Definition | Complete | `README.md` |
| 1 — Problem & Data | 2. Data Understanding | Complete | `notebooks/01-data-understanding.ipynb` |
| 1 — Problem & Data | 3. Data Preprocessing | Complete | `notebooks/02-preprocessing.ipynb` |
| 2 — Model Development | 4. Baseline Modeling | Complete | `notebooks/03-baseline-modeling.ipynb` |
| 2 — Model Development | 5. Model Experiments | Complete | `notebooks/04-model-experiments.ipynb` |
| 2 — Model Development | 6. Final Evaluation | Complete | `notebooks/05-final-evaluation.ipynb` |
| 3 — Engineering & Deployment | 7. Refactoring & Packaging | Complete | `src/bank_marketing_ml/`, `tests/` |
| 3 — Engineering & Deployment | 8. ML Pipeline & Release Readiness | Complete | `train.py`, `artifact.py`, `inference.py`, `tests/test_release.py` |
| 3 — Engineering & Deployment | 9. Deployment & Integration | Complete | [CSV 배치 CLI](../src/bank_marketing_ml/cli.py), [통합 테스트](../tests/test_cli.py) |
| 4 — Operations & Iteration | 10. Monitoring | Complete — offline simulation | [모니터링 시뮬레이션](../scripts/simulate_monitoring.py) |
| 4 — Operations & Iteration | 11. Retraining & Improvement | Protocol complete — pending new-period labels | [재학습·교체 절차](#stage-11--retraining--improvement) |
| 4 — Operations & Iteration | 12. Maintenance & Governance | Complete | [유지보수 문서](maintenance.md) |

## Invariant Modeling Decisions

- Prediction time: immediately before a planned contact.
- Unit: one campaign record; no unique customer identifier is available.
- Target: `y`, mapped from `no`/`yes` to `0`/`1`.
- Retained prediction-time features: the established 19 predictors.
- Excluded feature: `duration`, because it is unavailable before the contact ends.
- `pdays = 999` is a sentinel. Derive `pdays_previously_contacted` and `pdays_elapsed` (zero for sentinel rows); do not scale raw `pdays` as elapsed days. This is a representation constraint, not a Validation-selected option.
- Split: source row order, Train 70% (`0:28,831`), Validation 20% (`28,831:37,069`), Test 10% (`37,069:41,188`).
- Selection metric: Average Precision; ROC-AUC and stable record-based Top-5/10/20% metrics are supporting evidence.
- Top-K: `ceil(n × fraction)`, unrounded scores, descending stable sort, original row order for ties.
- Test is reserved for Stage 6. Earlier EDA inspected aggregate Test information, so it is not described as completely unseen, but it is excluded from development decisions.

## Stage 1 — Problem Definition

### Objective and Decisions

- Business problem: prioritize planned outbound marketing contacts when contact capacity is limited.
- Prediction objective: rank campaign records by the positive-class score for term-deposit subscription.
- Prediction time: immediately before the planned contact attempt.
- Unit of analysis: one campaign record, not a verified unique customer.
- Intended use: decision support for ranking an already eligible contact population.
- Prohibited interpretations: do not claim causal contact uplift, profitability, or verified absolute subscription probabilities from the predictive scores.
- Technical success: reproducible leakage-safe development, AP-led model selection, and supporting ROC-AUC and Top-K evaluation.
- Business acceptance remains undefined because contact costs, conversion value, eligibility constraints, and operational capacity are unavailable.

### Evidence and Limitations

- Primary evidence: `README.md` sections `Problem Definition`, `Data`, and `Methodology`.
- The dataset represents historically contacted records and may not generalize to a different eligible population.
- No unique customer identifier is available, so ranking and later monitoring remain record-based.
- Exact operational contact schedules and financial thresholds cannot be reconstructed from the dataset.

## Stage 2 — Data Understanding

### Data and Quality Findings

| Item | Finding |
|---|---|
| Source | UCI `bank-additional-full.csv`, semicolon-delimited and documented as date ordered |
| Shape | 41,188 rows, 20 original predictors, and target `y` |
| Missing values | No pandas-recognized missing values in Train |
| Literal unknowns | `default` is approximately 25.33% `"unknown"` in Train; unknowns are retained as categories |
| Exact duplicates | Nine exact duplicate Train rows; retained because erroneous duplication is not established |
| Target prevalence | Train 5.57%; Validation 13.86%, indicating material temporal/prevalence shift |
| Skew and sparsity | `campaign` is right-skewed; `previous` is zero for approximately 96.52% of Train |
| Previous-contact history | Only 53 Train rows have `pdays != 999`; 949 rows have `previous > 0` with `pdays = 999` |
| Correlation | `emp.var.rate` and `euribor3m` have approximately 0.95 Train correlation |

### Leakage and Split Decisions

- Exclude `duration`; it is only known after the current contact.
- Retain `contact` because the channel is selected before scoring.
- Retain the other candidate predictors; do not remove features based only on EDA association or correlation.
- Preserve source row order and use boundaries `28,831` and `37,069` for the 70/20/10 split.
- Do not shuffle, stratify, or reconstruct dates from `month` and `day_of_week`.
- Detailed EDA uses Train. Development-period aggregate diagnostics may use Validation.
- Historical aggregate Test inspection remains disclosed, but Test is excluded from subsequent development decisions.

### Interpretation and Limitations

- Subscription-rate differences and feature associations are descriptive, not causal.
- Strong macroeconomic correlations may complicate coefficient interpretation but do not independently justify feature removal.
- The history variables contain related but non-equivalent information; do not overwrite or infer corrected `pdays` values.
- Missing client identifiers prevent an audit of repeated customers or cross-split customer overlap.
- Incomplete timestamps prevent verification of exact calendar boundaries and point-in-time macroeconomic publication availability.

Primary evidence: `notebooks/01-data-understanding.ipynb`. The simplified notebook was executed from a fresh kernel on 2026-10-12; subsequent interpretation edits retained those outputs. No EDA was rerun during this documentation update.

## Stage 3 — Data Preprocessing

### Current Baseline Representation

- Modeling inputs: 19 predictors after excluding target `y` and `duration`.
- Numeric branch: `StandardScaler` on eight ordinary numerical predictors.
- Sentinel branch: derive `pdays_previously_contacted = (pdays != 999)` and `pdays_elapsed = 0` for sentinel rows, otherwise recorded days; apply `StandardScaler` to both derived columns inside the Pipeline.
- Categorical branch: `OneHotEncoder(handle_unknown="ignore")` on ten categorical predictors.
- Preserve literal `"unknown"` values as observed categories.
- Do not add imputation because the inspected development inputs contain no recognized missing values.
- Do not initially apply clipping, winsorization, log transforms, rare-category merging, outlier removal, or correlation-based deletion.
- Fit preprocessing on Train only and transform Validation without refitting.
- Leave unspecified columns dropped by the `ColumnTransformer`.

### Implementation and Verification

- Notebook 02 was simplified from 36 to 14 cells; assertion blocks and synthetic validation tests were removed.
- Binary target encoding uses `no = 0` and `yes = 1`.
- Train output shape: `(28831, 62)`; Validation output shape: `(8238, 62)`.
- Outputs comprise eight numerical, 52 one-hot categorical, and two sentinel-aware columns.
- Only Train and Validation inputs are prepared; Test is not transformed or evaluated.
- Fresh Kernel execution succeeded for all six code cells; file-level Ruff checks passed.

Primary evidence: `notebooks/02-preprocessing.ipynb`, executed on 2026-10-12. No execution was repeated during this documentation update.

## Stage 4 — Baseline Modeling

### Configuration

- Full scikit-learn `Pipeline`: baseline `ColumnTransformer` followed by `LogisticRegression(max_iter=1000)`.
- Sentinel-aware preprocessing matches Notebook 02 and produces 62 transformed features.
- All preprocessing statistics and model coefficients are fitted on Train only.
- Positive-class probabilities rank Validation records; no hard-label threshold is selected.
- Test remains excluded from training, prediction, and evaluation.

### Validation Results

| Metric | Result |
|---|---:|
| Average Precision | 0.288234 |
| ROC-AUC | 0.688023 |
| Validation prevalence | 0.138626 |

| Capacity | K | Captured positives | Precision@K | Recall@K | Lift@K |
|---|---:|---:|---:|---:|---:|
| Top 5% | 412 | 179 | 0.434466 | 0.156743 | 3.134090 |
| Top 10% | 824 | 309 | 0.375000 | 0.270578 | 2.705123 |
| Top 20% | 1,648 | 499 | 0.302791 | 0.436953 | 2.184233 |

### Interpretation and Limitations

- The baseline provides a simple, maintainable ranking reference for Stage 5.
- Logistic Regression does not automatically capture nonlinear numeric effects or complex interactions.
- AP and Top-K metrics are sensitive to the higher Validation prevalence and temporal shift.
- Whole-period Top-K evaluation does not simulate daily capacity, repeat-contact rules, or unique-customer constraints.
- Scores are used for ranking; probability calibration and causal business impact are not established.

The earlier raw-`pdays` baseline AP was `0.297231`; the new baseline is lower by `0.008997`. The sentinel-aware constraint was adopted for semantic consistency, with this measured trade-off recorded rather than hidden.

Primary evidence: `notebooks/03-baseline-modeling.ipynb`. Simplified from 25 to 16 cells, with assertion blocks removed. Fresh Kernel execution succeeded for all eight code cells on 2026-10-12 and reproduced these metrics on a second execution. File-level Ruff checks passed. No model was rerun during this documentation update.

## Stage 5 — Model Experiments

**Current status: complete — 2026-10-26, before Test access.** Sentinel-aware 전처리로 모델 비교, tuning, Validation 선택과 평가 규칙 확정을 완료했다. 아래 새 결과와 과거 raw-`pdays` 결과는 구분한다.

### Current Experiments and Selection

- 원본 앞 37,069행만 로딩했다. Train 내부의 확장형 `TimeSeriesSplit(n_splits=4, gap=0)`와 복제한 전체 Pipeline을 모든 후보에 동일하게 적용했다.
- 초기 CV 평균 AP: Logistic Regression `0.083139`, Random Forest `0.066477`, RBF SVM `0.060447`. SVM은 AP와 계산 비용을 고려해 추가 tuning에서 제외했다.
- 탐색은 결과를 보고 확장하지 않았다. Logistic Regression `C=[0.01,0.1,1,10]`, Random Forest `max_depth=[5,10,None]`, `min_samples_leaf=[5,20]`; RF는 tree 100개, `random_state=42`, worker 하나를 유지했다.
- Train-only tuning: Logistic Regression `C=0.1`, CV AP `0.085400`; Random Forest `max_depth=None`, `min_samples_leaf=20`, CV AP `0.065949`.
- 기존 선택 규칙을 재사용했다: Validation AP 최고값에서 0.005 이내, 이어 Lift@10% 최고값의 98% 이상인 후보 중 단순성 우선. 이 범위는 통계적 유의성 기준이 아니다.

| Validation 후보 | AP | ROC-AUC | Lift@10% |
|---|---:|---:|---:|
| Baseline Logistic Regression | 0.288234 | 0.688023 | 2.705123 |
| Tuned Logistic Regression | 0.282831 | 0.686609 | 2.643841 |
| Tuned Random Forest | 0.160494 | 0.552186 | 1.304412 |

Baseline을 선택했다. Tuned LR의 AP 감소 `0.005404`는 AP 허용 범위 밖이다. Baseline 구간별 Validation AP `0.108697`~`0.386684`와 양성 비율 5.73~23.51%는 기간 변화의 한계를 보여준다. 이 결정과 아래 평가 규칙은 Test 접근 전에 확정했고, 이후 별도 Notebook 05에서 최종 평가했다.

### Frozen Configuration and Evaluation Protocol

- 확정 시점: 2026-10-26, 이번 Test 접근 전. 과거 Test target 집계 정보 확인 이력은 계속 공개한다.
- Feature: 원본 19개, `duration`·`y` 제외; `contact` 유지. 수치형 8개 scaling, `pdays` indicator·elapsed 파생 후 두 열 scaling, 범주형 10개 one-hot (`handle_unknown="ignore"`), `unknown` 유지. 세부 목록과 함수는 Notebook 04의 확정 코드가 근거이다.
- Estimator: `LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, tol=0.0001)`, 나머지는 Notebook 04에 출력된 전체 설정 유지. Class weight·calibration 변경 없음. `lbfgs`에서는 무작위 추출을 사용하지 않는다.
- 환경: Python 3.12.15, scikit-learn 1.9.1, NumPy 2.5.3, pandas 3.0.6, SciPy 1.18.1; 의존성 변경 없음.
- 최종 학습: 새로운 동일 구성의 전체 Pipeline을 Train+Validation 37,069행에 fit. Test 4,119행은 학습하지 않는다.
- 최종 평가: 확정한 모델 하나로 Test를 한 번 평가. AP·ROC-AUC·양성 비율과 Top-5/10/20% Precision·Recall·Lift. `ceil(n*p)`, 반올림하지 않은 양성 클래스 점수, 내림차순 stable sort, 원본 순서 tie break, Test 양성 비율 기준 Lift.
- Test 평가 후 tuning·재선택·평가 정책 변경 없음. 수익성·인과효과·검증된 확률은 주장하지 않는다.

### Verification

새 커널에서 Notebook 04의 코드 셀 21개를 위에서 아래로 실행해 baseline 재현, CV, tuning과 Validation 선택을 확인했다. 이후 수정은 결과 해석과 확정 기록 Markdown이다. 모델 개발에는 Test 행을 로딩하지 않았다. 사용자가 삭제한 저장값 비교 절은 복원하지 않았다.

### Historical Raw-pdays Run — Superseded

다음 내용은 과거 실행의 근거로 보존한다. 현재 구성과 결과로 사용하지 않는다.

#### Objective and Method

- Reproduce the Stage 4 baseline in the locked environment.
- Compare raw `pdays` with an explicit previous-contact indicator plus elapsed-day representation.
- Compare Logistic Regression, Random Forest, and HistGradientBoosting using four expanding Train-only `TimeSeriesSplit` folds.
- Tune Logistic Regression and the strongest nonlinear family using full leakage-safe Pipelines fitted within each fold.
- Evaluate only the baseline and two Train-tuned frozen candidates on Validation.
- Select by AP first; when AP differs by at most `0.005`, compare Lift@10%; if Lift@10% is within 2% relatively, prefer the simpler model.

Stage 5 reads only the first 37,069 CSV rows with `nrows`; Test rows are not materialized, inspected, or evaluated.

#### Results

Baseline reproduction:

| Metric | Validation result |
|---|---:|
| Average Precision | 0.297231 |
| ROC-AUC | 0.692923 |
| Precision@10% | 0.385922 |
| Recall@10% | 0.278459 |
| Lift@10% | 2.783913 |

Feature representation:

- Engineered `pdays` Validation AP: `0.288234`.
- AP change relative to raw `pdays`: `-0.008997`.
- Decision: retain raw `pdays`.

Train-only model-family comparison:

| Family | Mean CV AP | CV AP standard deviation |
|---|---:|---:|
| Logistic Regression | 0.083139 | 0.054310 |
| HistGradientBoosting | 0.064833 | 0.020020 |
| Random Forest | 0.063195 | 0.019331 |

Frozen Validation comparison:

| Candidate | AP | ROC-AUC | Lift@10% |
|---|---:|---:|---:|
| Baseline Logistic Regression | 0.297231 | 0.692923 | 2.783913 |
| Tuned Logistic Regression | 0.293450 | 0.693192 | 2.678859 |
| Tuned HistGradientBoosting | 0.169968 | 0.568961 | 1.365693 |

Validation temporal-block AP ranged from `0.094411` to `0.399525`. This is a diagnostic warning about temporal and prevalence sensitivity, not evidence for selecting a different model.

#### Historical Configuration — Superseded

- Features: the established 19 predictors; exclude `duration` and target `y`.
- Numeric preprocessing: `StandardScaler` on `age`, `campaign`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`, and raw `pdays`.
- Categorical preprocessing: `OneHotEncoder(handle_unknown="ignore")` on `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, and `poutcome`.
- Estimator: `LogisticRegression(max_iter=1000)` with otherwise default scikit-learn 1.9.1 parameters.
- Selection decision: baseline Logistic Regression. It had the highest Validation AP and materially higher Lift@10% than the tuned Logistic Regression within the AP tie margin.
- Stage 6 training rule: create a fresh Pipeline with this exact configuration and fit all learned steps on Train+Validation.
- Stage 6 evaluation rule: evaluate Test once with AP, ROC-AUC, and Top-5/10/20%; diagnostics must not reopen model or feature selection.

#### Verification

| Check | Status | Evidence |
|---|---|---|
| Notebook JSON and Python syntax | Passed | 16 cells parsed; all code cells compiled |
| Fresh Kernel top-to-bottom execution | Passed | `uv run jupyter execute notebooks/04-model-experiments.ipynb --inplace --timeout=-1` |
| Baseline reproduction assertions | Passed | AP and ROC-AUC matched stored results within `5e-6` |
| Test isolation | Passed | Development prefix only loaded; final notebook assertion/output records no Test use |
| Existing unit tests | Passed | `uv run pytest`: 1 passed |
| Stage 5 notebook lint / format | Passed | `uv run ruff check notebooks/04-model-experiments.ipynb`; `uv run ruff format --check notebooks/04-model-experiments.ipynb` |
| Repository-wide lint | Passed | `uv run ruff check .` after focused import cleanup in notebooks `01`, `03`, and `scripts/download_data.py` |

## Stage 6 — Final Evaluation

**Current status: complete — 2026-10-26.** Test 접근 전에 확정한 sentinel-aware Logistic Regression 하나를 평가했다. Test 결과로 모델이나 규칙을 수정하지 않았다.

### Final Training and Data Provenance

- Notebook 05는 다른 노트북의 실행 상태에 의존하지 않고 동일한 feature 정의와 전처리를 직접 구성한다.
- Train+Validation 37,069행, 가입 레코드 2,748개(양성 비율 `0.074132`)에서 전체 Pipeline을 새로 fit했다. 변환 후 feature 62개, `lbfgs` 46회 반복으로 학습했다.
- 원본 CSV SHA-256: `74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8`.
- Test 접근 전 최초 확정본 Notebook 04 SHA-256: `187186e12721f75e7f8d5c1dabdd8207b49d0e010ff3da35af489e75d6a4c953`. 이는 당시 파일의 hash이며, 이후 Markdown을 정리한 현재 파일의 hash가 아니다. 모델·평가 코드와 저장 출력은 그대로 보존했다.
- 모델 1개, Test score 계산 1회. 그 점수를 AP·ROC-AUC·Top-K와 PR 곡선에 재사용했다. 과거 EDA의 Test 집계 정보 확인 이력은 계속 공개한다.

### Newly Verified Test Results

Test는 4,119행, 가입 레코드 1,892개이며 양성 비율은 `0.459335`이다. AP `0.549042`, ROC-AUC `0.620009`를 얻었다.

| 용량 | K | 포함한 가입 레코드 | Precision@K | Recall@K | Lift@K |
|---|---:|---:|---:|---:|---:|---:|
| 상위 5% | 206 | 128 | 0.621359 | 0.067653 | 1.352737 |
| 상위 10% | 412 | 211 | 0.512136 | 0.111522 | 1.114951 |
| 상위 20% | 824 | 432 | 0.524272 | 0.228330 | 1.141372 |

Validation보다 높은 Test AP를 개선으로 해석하지 않는다. 양성 비율이 13.86%에서 45.93%로 크게 바뀌었고 학습 범위도 달라졌다. Test의 ROC-AUC와 Lift@10%는 후기 기간의 순위 성능이 제한적임을 보여준다. 누적 Precision은 관측 구간에 따라 단조 감소하지 않을 수 있다. 수익성, 인과효과, 검증된 확률이나 운영 수용 가능성은 입증하지 않았다.

### Verification

- Notebook 04: 편집 전 새 커널에서 코드 셀 21개 전체 실행 성공. Baseline 재현, CV, tuning, Validation 선택 확인. Test 행 미로딩. 이후 코드 셀 내용·실행 순서·출력은 보존하고 Markdown을 정리했다.
- Notebook 05: 편집 전 새 커널에서 코드 셀 9개 전체 실행 성공. 확정 구성으로 최종 학습·Test 평가 1회. 이후 데이터 준비가 전처리보다 먼저 오도록 기존 코드 셀을 재배치하고 Markdown을 정리했다. 코드 내용과 저장 출력·실행 번호는 당시 실행 이력으로 보존했다. 재배치 후 모델과 Test는 재실행하지 않았다.
- Test 접근 전 Notebook 04/05 feature 정의와 모델 전체 파라미터 일치를 확인했다. 이후 편집에서 모든 코드 셀과 출력의 동일성을 다시 확인했다. Notebook 04의 코드 셀 source를 실행 순서대로 `\n\n`으로 연결한 UTF-8 SHA-256은 편집 전후 동일한 `4290e8216a99c7ca70583ae3aa5ac9752cb60b5baf442f8dfce6e315a74efd0c`이다.
- 저장된 Test PR 곡선 표시 확인. Notebook JSON·Python 문법, 실행 순서와 오류 없는 출력 확인 통과. `uv run --frozen --offline ruff check .`, Notebook 04/05 format 검사와 `git diff --check` 통과. 모델 재실행 없이 결과 해석 Markdown을 보완했다.
- Notebook assertion suite나 검증 framework는 추가하지 않았다. 기존 raw-`pdays` 근거는 위에 보존했고, 삭제 요청한 저장값 비교 절은 복원하지 않았다.

### Notebook Presentation Alignment

Notebook 01–03의 목적 → 준비 → 실행 결과 → 해석 → 요약 흐름에 맞춰 Notebook 04/05를 정리했다. 제목은 영어, 설명은 한국어로 유지했다. Notebook 04는 64개에서 43개 셀로 줄여 인접한 설명을 묶고, Notebook 05는 데이터·Pipeline 준비 → 최종 학습 → 평가 → 요약 순서로 재구성했다. 요약 표에는 선택 구성과 주요 지표를 남기고 실행 이력·hash 의미는 이 문서에서 관리한다. 삭제 요청한 저장값 비교 절은 추가하지 않았다. 정책·파라미터·수치·출력의 변경 없이 편집했으며 새 실험 결과로 표현하지 않는다.

## Stage 7 — Refactoring & Packaging

**Current status: complete — 2026-11-05.** Notebook에서 확정한 로직을 기존 Python 패키지로 옮겼다. Notebook 코드와 저장 출력은 보존했다.

- `data.py`: 원본 CSV의 앞 37,069행만 로딩해 Train 28,831행과 Validation 8,238행으로 분리한다. 확정한 19개 입력을 선택하고 `y`를 `no = 0`, `yes = 1`로 인코딩한다.
- `features.py`, `train.py`: sentinel-aware 전처리와 확정한 Logistic Regression 설정으로 미학습 Pipeline을 생성한다. 호출할 때마다 독립적인 객체를 반환한다.
- `evaluate.py`: 순서가 맞는 이진 label과 점수를 받아 AP, ROC-AUC, 양성 비율 및 Top-5/10/20% 지표를 계산한다. 기존의 올림, 반올림 전 점수, 동점 원본 순서, 평가 구간 양성 비율 기준을 유지한다.
- 기존 smoke test를 데이터 준비, 전처리·모델 통합, 순위 계산 테스트로 대체했다. 패키지 설정과 잠금 의존성은 변경하지 않았다.

### Verification

- `uv run --locked --offline pytest -q`: 13개 통과. `ruff check .` 및 새 Python 코드의 format 검사 통과.
- 합성 데이터에서 Notebook 05의 정의와 패키지 Pipeline의 모델 파라미터, 변환 feature 이름·값, 예측 점수가 정확히 일치했다.
- Wheel 빌드 성공. 생성한 wheel에 네 모델링 모듈이 포함됨을 확인했다.
- 원본 데이터를 로딩하거나 Notebook 04의 실험·Notebook 05의 Test를 재실행하지 않았다. 기존 성능 수치는 과거 실행 결과이며 이번 단계에서 다시 측정하지 않았다.

## Stage 8 — ML Pipeline & Release Readiness

**Current status: complete.** 확정 Pipeline을 Train+Validation에 학습해 저장하고, 별도 실행에서 불러와 CSV 점수를 계산하는 경로를 구현했다.

- `train_and_save`는 기존 개발 구간 로더를 사용해 앞 37,069행만 학습한다. Test 행은 사용하지 않는다.
- `artifact.py`는 Pipeline과 모델 버전, 학습 파일명·행 수, 입력 열·양성 클래스, 실행 환경 버전을 하나의 ZIP으로 저장한다. 로딩 전 형식·입력 schema·환경 버전·체크섬을 검사하고, 로딩 후 학습 상태와 입력·클래스 순서를 확인한다. 표준 라이브러리를 사용해 의존성은 변경하지 않았다. Pickle은 신뢰하는 로컬 모델 파일만 로딩하며 체크섬은 출처 검증 수단이 아니다.
- `score_csv`는 레코드 순서를 보존하고 필수 열 누락·중복, 행 길이, 비수치 또는 무한값, 빈 범주를 검사한다. 추가 열을 무시하고 미관측 범주는 기존 정책으로 처리한다. 결과는 반올림하지 않은 점수이며 순위와 출력 CSV는 아직 생성하지 않는다.

### Verification

- 합성 데이터에서 학습, 저장, 별도 프로세스 로딩 및 점수 계산을 검증했다. 잘못된 CSV, 손상된 모델, 환경·schema 불일치도 검사했다. 전체 테스트 28개 통과.
- 실제 Train+Validation 37,069행으로 `models/bank-marketing-v1.zip`을 생성했다. Git에서 제외한 로컬 파일이다. 개발 구간에서 뽑은 점검용 네 행의 점수가 저장 전과 별도 프로세스 로딩 후 정확히 일치했다. 성능 지표를 새로 측정하거나 Test를 재평가하지 않았다.
- Ruff lint·format 및 diff 검사 통과. Notebook과 잠금 의존성은 변경하지 않았다.

## Stage 9 — Deployment & Integration

**Current status: complete — 2026-11-05.** `bank-marketing-ml` 명령에서 입력 CSV, 모델 ZIP, 출력 경로를 받아 기존 `score_csv`를 실행한다. 기본 입력은 세미콜론이며 `--sep ','`로 쉼표 입력을 지정한다.

- 결과는 UTF-8 쉼표 CSV이며 입력 레코드 순서로 `input_row`, `score`, `rank`를 쓴다. 순위는 반올림 전 점수의 내림차순이며 정확히 같은 점수는 입력 순서를 따른다.
- 입력 또는 모델과 동일한 파일을 출력으로 지정하면 거부한다. 결과는 대상 디렉터리에서 임시 파일에 완전히 기록한 후 교체한다. 입력·모델·출력 오류는 표준 오류에 표시하고 실패 종료 코드를 반환한다.
- 롤백은 보존한 이전 ZIP을 `--model`로 명시하고 같은 입력을 다시 처리한다. 이전 ZIP도 현재 실행 환경과 호환되어야 한다.

### Verification

- `uv run --no-cache --locked --offline pytest -q` 통과. 전체 40개 중 CLI 통합 테스트 12개가 두 입력 구분자, 원본 행 추적, 동점과 미세한 점수 차이, 입력·모델 오류, 기존 출력 보호, 출력 교체 실패 후 정리, 경로 별칭 차단을 확인했다.
- 합성 모델로 `v1 → v2 → v1` 전환을 실행했다. `v2`에서 결과가 달라졌고, `v1`을 다시 선택하자 첫 CSV가 바이트 단위로 재현됐다.
- 로컬 `models/bank-marketing-v1.zip`으로 합성 입력 3행을 처리해 출력 3행과 순위를 확인했다. 실제 데이터 재학습이나 Test 재평가는 수행하지 않았다.
- 저장소 전체 Ruff lint, 변경한 Python 파일의 Ruff format 검사, `git diff --check` 통과.

## Stage 10 — Monitoring

**Current status: complete — 2026-11-05.** `scripts/simulate_monitoring.py`가 기존 배치 CLI를 별도 프로세스로 실행하고 점검 결과를 JSON으로 출력한다. 합성 데이터는 seed `42`로 생성한다. 로컬 `v1` ZIP을 사용했으며 원본 데이터, 재학습, 최종 Test 재평가는 사용하지 않았다.

### Offline Simulation Results

| 시나리오 | 관찰 |
|---|---|
| 정상 | 200행 입력과 출력의 레코드 번호가 일치했다. |
| 필수 열 누락과 복구 | `age` 누락 시 종료 코드 1과 오류 메시지를 반환했고 결과 파일을 생성하지 않았다. 열을 복구해 다시 실행하자 정상 배치의 점수와 순위가 재현됐다. |
| 입력 구성 변화 | 200행 모두 처리했다. `job=unknown`은 4%에서 52%, `contact=cellular`는 45%에서 85.5%, 나이 중앙값은 45세에서 51세로 바뀌었다. 점수 중앙값은 0.19175에서 0.20129로 변했다. |
| 10배 크기 | 기준 입력을 반복한 2,000행을 모두 처리했다. 결과 크기는 5,550바이트에서 59,248바이트로 증가했다. 첫 점검 실행에서 200행과 2,000행은 각각 약 1.48초와 1.81초, 실패와 복구의 합은 약 3.87초였다. |
| 지연 라벨 | 기준 배치의 합성 라벨을 역순으로 전달한 뒤 같은 배치의 `input_row`로 연결했다. 200행, 양성 비율 20%, AP 0.23482, ROC-AUC 0.52578을 얻었다. 상위 10%는 20행 중 6건이 양성이며 Precision 0.30, Recall 0.15, Lift 1.50이었다. 기존 평가 함수로 Top-5/10/20%를 모두 계산했다. |

처리 시간은 프로세스 시작, 모델 로딩, 점수 계산과 출력 저장을 포함한 로컬 단일 실행 관측치다. 지연 라벨은 나이에 따라 확률을 달리해 생성한 시뮬레이션 값이다. 성능 수치는 라벨 연결과 지표 계산 흐름의 확인에 사용했다.

### Review Criteria

| 신호 | 확인할 사항 |
|---|---|
| 실행 실패, 행 누락 또는 순서 불일치 | 결과 게시 여부와 오류를 확인한다. 입력·모델·출력 경로 중 원인을 찾아 수정한 뒤 같은 배치를 재실행하고 행 대응을 확인한다. |
| 입력 품질 또는 분포 변화 | 같은 모델을 쓴 비교 가능한 배치와 범주 비율, 수치형 중앙값, 점수 분위수를 비교한다. 평소 변동 폭을 벗어나면 입력 생성 과정과 캠페인 구성 변화를 먼저 확인한다. 운영 경보 임계값을 정하려면 여러 기간의 기준 데이터가 필요하다. |
| 라벨 확보 후 성능 변화 | 배치와 레코드 키를 함께 관리하고 라벨 범위·양성 비율을 확인한 뒤 AP와 Top-K를 비교한다. 이번 시뮬레이션은 라벨이 완전한 경우를 사용한다. 실제 연락 대상에만 라벨이 관측된다면 전체 후보의 성능으로 일반화할 수 없다. |
| 처리 비용 증가 | 비슷한 크기의 배치끼리 경과 시간, 초당 처리 행 수, 출력 크기, 재시도 시간을 비교한다. |

### Verification

- 로컬 `v1` 모델을 지정한 전체 시뮬레이션이 종료 코드 0으로 완료됐다. 누락 모델을 지정한 별도 실행은 오류가 담긴 JSON을 출력하고 종료 코드 1을 반환했다.
- 순서가 뒤섞인 라벨을 레코드 번호로 연결했고, 누락되거나 중복된 라벨 키는 거부하는 것을 별도 점검했다.
- `uv run --no-cache --locked --offline pytest -q` 전체 40개, 저장소 전체 Ruff lint, 시뮬레이션 스크립트의 Ruff format 검사, `git diff --check`를 통과했다.

## Stage 11 — Retraining & Improvement

**Current status: protocol complete — 2026-11-05.** 로컬에는 기존 UCI 데이터와 `v1` ZIP이 있다. 새 기간의 운영 라벨은 저장소에서 확인되지 않았다. 아래 절차를 수립했으며 실제 재학습, 신규 기간 백테스트와 모델 교체는 라벨 확보 후 진행한다. 운영 기준값은 10단계 합성 결과에서 정하지 않는다.

### Retraining Triggers

| 관측 신호 | 조치 |
|---|---|
| 필수 열 누락, 레코드 대응 오류, 배치 실패 | 입력 생성 과정이나 실행 환경을 고친 뒤 같은 배치를 재실행한다. |
| 입력 또는 점수 분포 변화 | 입력 생성 변경, 캠페인 구성과 라벨 확보 범위를 조사한다. 라벨이 성숙한 비교 기간에서 순위 성능을 확인한다. |
| 라벨이 성숙한 비교 가능한 연속 두 기간에서 AP가 사전 검토 기준을 밑돎 | 입력 품질과 라벨 관측 범위를 확인한 뒤 재학습 검토를 시작한다. 기간별 양성 비율, 표본 수, Top-K 결과를 함께 기록한다. |
| 새 기간의 라벨이 없거나 비교에 필요한 범위를 충족하지 못함 | `v1`을 유지하고 라벨 수집과 검증을 계속한다. |

연속 두 기간은 초기 검토 규칙이다. 실제 검토 전에 기간 길이, 라벨 관측 기간, 최소 표본·양성 건수, AP 기준값과 업무 연락 용량을 정해 기록한다. 이전과 비교할 때 라벨 정의와 관측 방식이 같은지 확인한다. 연락한 레코드만 라벨이 있으면 관측 집단의 결과로 범위를 명시하며, 미관측 레코드를 `no`로 채우거나 전체 후보 성능으로 일반화하지 않는다.

### New-Period Backtest

1. 새 데이터의 출처, 연락·스코어링 시점, 라벨 확정 시점, 라벨을 관측할 수 있는 집단을 기록한다. 배치와 레코드 키로 중복·누락 및 연결을 확인하고, 예정된 연락 직전에 확보할 수 있는 19개 입력만 사용한다. 원본 Test는 신규 후보의 학습·선택·평가에 사용하지 않는다.
2. 새 데이터를 시간순으로 앞쪽 개발 기간과 가장 최근의 별도 평가 기간으로 나눈다. 개발 기간에서는 앞 기간으로 학습해 다음 기간을 검증하는 방식으로 비교한다. 각 학습 시점에 이미 확정된 라벨만 학습에 사용하고, 전처리는 매 학습 구간의 Pipeline 안에서 fit한다.
3. 첫 후보는 기존 sentinel-aware Logistic Regression 설정을 유지하고 학습 데이터 갱신만 비교한다. 개발 비교 후 후보 구성, 학습 범위, 평가 기간과 교체 기준을 확정한다. 고정된 `v1` ZIP과 최종 후보 ZIP을 같은 최신 평가 레코드에 적용하고, 라벨과 원본 행 순서를 맞춘 뒤 기존 `evaluate_ranking`으로 평가한다.
4. AP를 주 지표로 사용하고 ROC-AUC, 양성 비율, Precision·Recall·Lift@5/10/20%와 기간별 결과를 함께 기록한다. 최소 AP 개선 폭, 업무 용량에 해당하는 Top-K의 허용 하락 폭, 처리 시간 한도는 최신 평가를 열기 전에 정한다. 새 기간 데이터가 확보되면 표본 규모와 운영 요구를 근거로 이 값을 결정한다.

### Promotion and Rollback

1. 후보가 사전에 정한 AP 개선 폭과 Top-K·처리 시간 기준을 모두 충족하고, CLI 호환성 및 결과 파일 검사를 통과하면 교체를 검토한다. 결과가 불충분하면 `v1`을 유지한다. 결정 기록에 데이터 기간, 비교 집단, 평가 지표, 모델 파일 체크섬, 실행 환경과 선택 이유를 남긴다.
2. 후보를 별도 버전 ZIP으로 보관한다. 동일한 점검 입력을 두 모델로 실행해 행 대응과 출력 계약을 비교하고, 이전 ZIP을 기존 호환 환경에서 다시 실행할 수 있는지 확인한다. 검토가 끝나면 다음 배치의 `--model`에 승인된 후보 경로를 명시한다.
3. 후보로 전환한 뒤 후보에만 발생하는 호환성·처리 오류가 있거나, 성숙 라벨에서 사전에 정한 성능 기준을 위반하면 이전 ZIP을 `--model`로 지정한다. 같은 입력을 별도 결과 경로로 다시 처리해 행 대응과 결과를 확인하고, 영향받은 배치와 복구 시점을 기록한다.

## Stage 12 — Maintenance & Governance

**Current status: complete — 2026-11-05.** [유지보수 문서](maintenance.md)에 코드·모델 버전 기록, 파일 취급, 의존성 변경 전후 점검, 인계와 복구 절차를 정리했다.

### Baseline and Verification

- 로컬 `models/bank-marketing-v1.zip`의 모델 버전 `v1`, 포맷 버전 `1`, 학습 행 수 37,069를 확인했다. 저장 환경은 Python `3.12.15`, 패키지 `0.1.0`이며, 점검한 ZIP의 SHA-256은 `db4e13062d4b16d2ccc610b5526b38756bfb5b912353e103b45986f94597b4f4`다.
- `uv lock --check --offline --no-cache`, 전체 pytest 40개, Ruff lint, `git diff --check` 통과. 이 잠금 파일 점검은 프로젝트 설정과의 일관성을 확인하며 외부 보안 공지 조회는 수행하지 않았다.
- 로컬 `v1`과 합성 데이터로 10단계 시뮬레이션을 다시 실행했다. 기준·복구·구성 변화 배치는 각각 200행, 대형 배치는 2,000행을 처리했다. 필수 열 누락 배치는 종료 코드 1과 결과 0행을 확인했다.
- `data/`, `models/`, `artifacts/`, `.venv/` 및 `.env`의 대표 경로가 Git에서 제외됨을 확인했다. 사용하지 않는 빈 `.env.example`은 프로젝트 정리 시 제거했다.

### Project Review and Handoff

로컬 CSV 배치 실행과 합성 데이터 기반 오프라인 모니터링까지 프로젝트 범위를 마무리했다. 실제 운영을 검토할 경우 새 기간의 라벨과 관측 범위, 예측 시점의 입력 가용성, 업무 연락 용량, 데이터 취급 책임을 먼저 확정한다. 새 라벨이 확보되면 11단계의 시간순 백테스트로 후보를 비교한다. 이번 점검에서 모델 재학습과 기존 Test 재평가는 수행하지 않았다.

### Portfolio Documentation

- README에서 문제와 평가 결과, 설계 결정, 합성 CSV 시연을 먼저 볼 수 있게 정리했다. 로컬 `v1` ZIP으로 시연을 실행해 출력 두 행과 순위가 문서 예시와 일치함을 확인했다.
- Quick Start에 단계별 결과를 명시하고 Verification에서 데이터·전처리, 순위·모델 저장, 배치 CLI 테스트 명령을 구분했다. `pytest --collect-only`에서 테스트 40개가 인식됐다. README 변경 시 테스트 본 실행과 모델 평가는 수행하지 않았다.
