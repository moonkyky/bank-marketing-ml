# Project Status

Last updated: 2026-11-05

This file is the cross-chat handoff for the 12-stage ML lifecycle. It records current progress, evidence, decisions, verification, and the next stage. Project methodology is documented in `README.md` and the relevant notebooks.

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
| 3 — Engineering & Deployment | 9. Model Deployment & Integration | Not started | — |
| 4 — Operations & Iteration | 10. Production Monitoring | Not started | — |
| 4 — Operations & Iteration | 11. Retraining & Continuous Improvement | Not started | — |
| 4 — Operations & Iteration | 12. Maintenance & Governance | Not started | — |

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

## Next Handoff — Stage 9 Deployment & Integration

검증한 학습·저장·점수 계산 기능을 로컬 CSV Batch CLI에 연결한다. 점수·순위 CSV, 실패 시 출력 처리, 이전 모델 버전 지정과 롤백을 통합 검증한다.

기존 Test는 이미 평가됐다. 같은 Test로 tuning하거나 재선택하지 않는다. 새 모델 개선에는 새로운 기간의 데이터 또는 별도로 설계한 backtest와 평가 계획이 필요하다.
