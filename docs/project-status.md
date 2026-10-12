# Project Status

Last updated: 2026-10-12

This file is the cross-chat handoff for the 12-stage ML lifecycle. `AGENTS.md` defines invariant working rules; this document records current progress, evidence, decisions, verification, and the next stage.

## Lifecycle Progress

| Phase | Stage | Status | Primary evidence |
|---|---|---|---|
| 1 — Problem & Data | 1. Problem Definition | Complete | `README.md` |
| 1 — Problem & Data | 2. Data Understanding | Complete | `notebooks/01-data-understanding.ipynb` |
| 1 — Problem & Data | 3. Data Preprocessing | Complete | `notebooks/02-preprocessing.ipynb` |
| 2 — Model Development | 4. Baseline Modeling | Complete | `notebooks/03-baseline-modeling.ipynb` |
| 2 — Model Development | 5. Model Experiments | Complete | `notebooks/04-model-experiments.ipynb` |
| 2 — Model Development | 6. Final Evaluation | Next | `notebooks/05-final-evaluation.ipynb` |
| 3 — Engineering & Deployment | 7. Refactoring & Packaging | Not started | — |
| 3 — Engineering & Deployment | 8. ML Pipeline & Release Readiness | Not started | — |
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

- Primary evidence: `README.md` Sections 1–4.
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

Primary evidence: `notebooks/01-data-understanding.ipynb`. These are stored Notebook findings; Stage 2 was not rerun during the current documentation update.

## Stage 3 — Data Preprocessing

### Frozen Baseline Representation

- Modeling inputs: 19 predictors after excluding target `y` and `duration`.
- Numeric branch: `StandardScaler` on eight numerical predictors plus raw `pdays`.
- Categorical branch: `OneHotEncoder(handle_unknown="ignore")` on ten categorical predictors.
- Preserve literal `"unknown"` values as observed categories.
- Do not add imputation because the inspected development inputs contain no recognized missing values.
- Do not initially apply clipping, winsorization, log transforms, rare-category merging, outlier removal, or correlation-based deletion.
- Fit preprocessing on Train only and transform Validation without refitting.
- Leave unspecified columns dropped by the `ColumnTransformer`.

### Stored Validation Checks

- Chronological boundaries, feature/target alignment, and complete non-overlapping feature groups were checked.
- Binary target encoding uses `no = 0` and `yes = 1`.
- Transformed Train and Validation matrices have consistent width and contain no NaN or infinite values.
- The fitted representation produces 61 transformed features: nine numeric outputs and 52 one-hot outputs.
- A synthetic unseen job category transforms successfully without changing output width.
- Test was partitioned to preserve boundaries but was not transformed or used to fit preprocessing.

Primary evidence: `notebooks/02-preprocessing.ipynb`. These are stored Notebook results; Stage 3 was not rerun during the current documentation update.

## Stage 4 — Baseline Modeling

### Configuration

- Full scikit-learn `Pipeline`: baseline `ColumnTransformer` followed by `LogisticRegression(max_iter=1000)`.
- All preprocessing statistics and model coefficients are fitted on Train only.
- Positive-class probabilities rank Validation records; no hard-label threshold is selected.
- Test remains excluded from training, prediction, and evaluation.

### Validation Results

| Metric | Result |
|---|---:|
| Average Precision | 0.297231 |
| ROC-AUC | 0.692923 |
| Validation prevalence | 0.138626 |

| Capacity | K | Captured positives | Precision@K | Recall@K | Lift@K |
|---|---:|---:|---:|---:|---:|
| Top 5% | 412 | 177 | 0.429612 | 0.154991 | 3.099072 |
| Top 10% | 824 | 318 | 0.385922 | 0.278459 | 2.783913 |
| Top 20% | 1,648 | 503 | 0.305218 | 0.440455 | 2.201742 |

### Interpretation and Limitations

- The baseline provides a simple, maintainable ranking reference for Stage 5.
- Logistic Regression does not automatically capture nonlinear numeric effects or complex interactions.
- AP and Top-K metrics are sensitive to the higher Validation prevalence and temporal shift.
- Whole-period Top-K evaluation does not simulate daily capacity, repeat-contact rules, or unique-customer constraints.
- Scores are used for ranking; probability calibration and causal business impact are not established.

Primary evidence: `notebooks/03-baseline-modeling.ipynb`. The core AP, ROC-AUC, and Top-K results were reproduced in the Stage 5 Fresh Kernel run.

## Stage 5 — Model Experiments

### Objective and Method

- Reproduce the Stage 4 baseline in the locked environment.
- Compare raw `pdays` with an explicit previous-contact indicator plus elapsed-day representation.
- Compare Logistic Regression, Random Forest, and HistGradientBoosting using four expanding Train-only `TimeSeriesSplit` folds.
- Tune Logistic Regression and the strongest nonlinear family using full leakage-safe Pipelines fitted within each fold.
- Evaluate only the baseline and two Train-tuned frozen candidates on Validation.
- Select by AP first; when AP differs by at most `0.005`, compare Lift@10%; if Lift@10% is within 2% relatively, prefer the simpler model.

Stage 5 reads only the first 37,069 CSV rows with `nrows`; Test rows are not materialized, inspected, or evaluated.

### Results

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

### Frozen Configuration for Stage 6

- Features: the established 19 predictors; exclude `duration` and target `y`.
- Numeric preprocessing: `StandardScaler` on `age`, `campaign`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`, and raw `pdays`.
- Categorical preprocessing: `OneHotEncoder(handle_unknown="ignore")` on `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, and `poutcome`.
- Estimator: `LogisticRegression(max_iter=1000)` with otherwise default scikit-learn 1.9.1 parameters.
- Selection decision: baseline Logistic Regression. It had the highest Validation AP and materially higher Lift@10% than the tuned Logistic Regression within the AP tie margin.
- Stage 6 training rule: create a fresh Pipeline with this exact configuration and fit all learned steps on Train+Validation.
- Stage 6 evaluation rule: evaluate Test once with AP, ROC-AUC, and Top-5/10/20%; diagnostics must not reopen model or feature selection.

### Verification

| Check | Status | Evidence |
|---|---|---|
| Notebook JSON and Python syntax | Passed | 16 cells parsed; all code cells compiled |
| Fresh Kernel top-to-bottom execution | Passed | `uv run jupyter execute notebooks/04-model-experiments.ipynb --inplace --timeout=-1` |
| Baseline reproduction assertions | Passed | AP and ROC-AUC matched stored results within `5e-6` |
| Test isolation | Passed | Development prefix only loaded; final notebook assertion/output records no Test use |
| Existing unit tests | Passed | `uv run pytest`: 1 passed |
| Stage 5 notebook lint / format | Passed | `uv run ruff check notebooks/04-model-experiments.ipynb`; `uv run ruff format --check notebooks/04-model-experiments.ipynb` |
| Repository-wide lint | Passed | `uv run ruff check .` after focused import cleanup in notebooks `01`, `03`, and `scripts/download_data.py` |

## Next Handoff — Stage 6

Work only in `notebooks/05-final-evaluation.ipynb` plus necessary status/README documentation. Reconstruct the frozen configuration above, fit on Train+Validation, and perform the first and only final Test evaluation. Do not tune, change features, change preprocessing, or select a different model based on Test results.
