# Project Status

Last updated: 2026-10-11

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
- Retained prediction-time features: `contact` and `campaign` under the documented availability assumptions.
- Excluded feature: `duration`, because it is unavailable before the contact ends.
- Split: source row order, Train 70% (`0:28,831`), Validation 20% (`28,831:37,069`), Test 10% (`37,069:41,188`).
- Selection metric: Average Precision; ROC-AUC and stable record-based Top-5/10/20% metrics are supporting evidence.
- Top-K: `ceil(n × fraction)`, unrounded scores, descending stable sort, original row order for ties.
- Test is reserved for Stage 6. Earlier EDA inspected aggregate Test information, so it is not described as completely unseen, but it is excluded from development decisions.

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
| Repository-wide lint | Failed on pre-existing files | Existing issues remain in notebooks `01`, `03`, and `scripts/download_data.py`; they were not modified in Stage 5 |

## Next Handoff — Stage 6

Work only in `notebooks/05-final-evaluation.ipynb` plus necessary status/README documentation. Reconstruct the frozen configuration above, fit on Train+Validation, and perform the first and only final Test evaluation. Do not tune, change features, change preprocessing, or select a different model based on Test results.
