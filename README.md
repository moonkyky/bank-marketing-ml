# Bank Marketing ML

**Term deposit subscription ranking with reproducible machine learning.**

This project uses the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) to investigate whether historical customer and campaign information can help prioritize records for a bank's outbound marketing contacts. The practical question is **which records should be prioritized when contact capacity is limited?**

The current scope covers **Phase 1 — Problem & Data** and **Phase 2 — Model Development**. The project documentation reports that the first three notebooks are complete; model experimentation and final evaluation remain to be done. Deployment and production monitoring are outside the current implementation scope.

## 1. Problem Definition

### 1.1 Business Problem

Outbound campaigns require staff time and other resources. When the number of available contacts is limited, the bank needs a defensible way to prioritize its planned contacts rather than treating every record as equally promising.

The model assigns a subscription score to each campaign record. Records are ranked in descending score order, and evaluation measures how many observed subscribers appear within the highest-ranked portion of the list.

| Item | Definition |
|---|---|
| Decision supported | Prioritization of planned outbound marketing contacts |
| Prediction time | Immediately before the planned contact attempt |
| Unit of analysis | One customer-related campaign record (one dataset row) |
| Target | Term deposit subscription, `y` |
| Positive / negative class | `yes` = 1 / `no` = 0 |
| Model output | Positive-class score from `predict_proba` |
| Intended use | Rank eligible records by score, highest first |

**Important distinction:** The dataset has no unique customer identifier. One row cannot be assumed to represent one unique customer, so the current ranking and Top-K metrics are **record-based**, not unique-customer-based.

### 1.2 Modeling Objective and Boundaries

The learning task is **binary classification**, but the primary decision task is **ranking**: concentrate observed subscriptions among the highest-scoring records under a fixed contact capacity.

- Rank using positive-class scores rather than defaulting to hard labels or a `0.5` classification threshold.
- Do not interpret uncalibrated model scores as verified real-world subscription probabilities; probability calibration has not yet been assessed.
- Do not infer the **incremental effect of calling a customer**. Historical subscription prediction is not uplift modeling or causal evaluation.

## 2. Success Criteria

Success is evaluated at three different levels:

| Level | Criterion | Status |
|---|---|---|
| Reproducible development | Document prediction-time assumptions, fixed splits, leakage-safe preprocessing, and a repeatable baseline | Reported complete in notebooks `01`–`03` |
| Model selection | Compare candidates using the same validation protocol, emphasizing AP and checking Top-K performance, stability, and complexity | Planned in notebook `04` |
| Final model assessment | Freeze the selected configuration and report held-out test performance and limitations | Planned in notebook `05` |
| Operational business acceptance | Demonstrate value at a realistic contact capacity, accounting for cost, conversion value, eligibility, and contact constraints | Not yet defined |

### 2.1 Technical Selection Criteria

**Average Precision (AP)** is the primary validation-based model selection metric. The reported Logistic Regression baseline achieved **validation AP = 0.297231**, which serves as the reference for subsequent experiments, **not** as a fixed test-set pass threshold.

Secondary measures are ROC-AUC and **Precision@K, Recall@K, and Lift@K**. Model selection should consider:

- Whether improvements are meaningful under the **same validation records, target definition, and ranking rules**.
- Whether AP improvements align with the contact-capacity scenarios relevant to the business.
- Error patterns, time-related distribution shift, reproducibility, inference cost, and implementation complexity.
- Whether the baseline remains preferable when differences are small or trade-offs are unfavorable.

An improvement in every metric is **not** required, and small numerical differences alone do not establish a better model.

### 2.2 Business Acceptance Criteria

The **top 5%, 10%, and 20%** scenarios are evaluation conventions, **not approved operational contact policies**. Before setting numerical business targets or claiming return on investment, the bank would need to specify:

- Eligible contact population and daily or campaign-level capacity.
- Cost per contact and expected value per subscription.
- Rules for repeat contacts and other eligibility restrictions.
- Required subscription capture or conversion concentration.
- Outcome observation window and operational evaluation unit.

No revenue uplift, minimum Precision@K, or other business threshold is invented without these inputs. In particular, **Lift@K greater than 1 does not prove profitability or incremental causal impact**.

## 3. Data and Prediction-Time Feature Policy

### 3.1 Dataset

The source is UCI's `bank-additional-full.csv`.

| Property | Value |
|---|---|
| Records | 41,188 |
| Original predictors | 20 |
| Target | `y` |
| File delimiter | Semicolon (`;`) |
| Covered period | May 2008–November 2010 |
| Original order | Chronological, as documented by UCI |
| Candidate modeling predictors | 19, after excluding `duration` |
| Local data path | `data/raw/bank-additional/bank-additional-full.csv` |

Detailed inspection and EDA belong in `notebooks/01-data-understanding.ipynb`.

### 3.2 Feature Availability at Prediction Time

The intended decision is made **before a planned contact**, so only information available at that moment is admissible. The project uses the following feature-availability policy. Operational data contracts, especially point-in-time availability of macroeconomic data, must still be verified before deployment:

| Features | Working assumption or restriction |
|---|---|
| Customer attributes | Available before the contact |
| `pdays`, `previous`, `poutcome` | Refer to prior campaign history available before the current contact |
| `month`, `day_of_week` | The planned contact date is known when scoring |
| `campaign` | Retained: the current campaign contact count for the client is treated as the current planned attempt number and is known at scoring time |
| `contact` | Retained: the communication channel is decided before scoring |
| Macroeconomic indicators | Values were published and available at scoring time; verify publication lags and revisions |
| `duration` | **Excluded**: current-call duration is unavailable before the call |

Keeping `duration` in the raw dataset for inspection does not make it an eligible model input. The retained `campaign` feature may reflect the historical contact policy, so a material change in campaign workflow can cause distribution shift and requires a new feature-availability review.

## 4. Data Splitting and Evaluation Protocol

### 4.1 Chronological Holdout

The project preserves the source row order and partitions records **70% / 20% / 10%** for training, validation, and testing. These are proportions of **rows**, not calendar-time intervals.

| Split | Rows | Role |
|---|---:|---|
| Train | 28,831 | Fit preprocessing and candidate models |
| Validation | 8,238 | Compare experiments and select configurations or policies |
| Test | 4,119 | Evaluate the frozen final approach |

Do not sort again by `month` or `day_of_week`: they do not uniquely identify dates or reconstruct the original chronological sequence.

The reported positive-class rates are approximately **5.57% for Train** and **13.86% for Validation**. Performance must therefore be interpreted in the context of temporal and prevalence shift; AP and Precision@K are sensitive to class prevalence.

**Test-set transparency:** Aggregate test-target information was inspected during earlier EDA, so the test set should not be described as completely unseen in every respect. It is reserved from further development decisions and must not be used for feature engineering, tuning, model selection, or ranking-policy selection.

### 4.2 Leakage Controls

- Fit learned preprocessing, feature selection, and any resampling **only on training data** (or within the training fold during cross-validation).
- Apply the fitted pipeline to Validation without refitting.
- Keep preprocessing and estimator behavior consistent, preferably through a single scikit-learn `Pipeline`.
- Select models, hyperparameters, and ranking rules **without consulting Test**.
- After selection, optionally refit on Train + Validation, including all learned preprocessing; apply the final fitted pipeline to Test only for final evaluation.
- Record split boundaries and model settings; run notebooks from a fresh kernel without relying on another notebook's in-memory variables.

### 4.3 Metrics and Top-K Rules

| Metric | Definition / purpose |
|---|---|
| **Average Precision** | Primary ranking-oriented selection metric, summarizing precision across recall levels |
| ROC-AUC | Secondary measure of overall score discrimination |
| Precision@K | Observed subscription rate among the top K records |
| Recall@K | Fraction of all observed subscribers captured among the top K records |
| Lift@K | Precision@K divided by the positive-class rate of the **evaluation split** |

For each predefined contact fraction `p` in **{5%, 10%, 20%}**:

1. Set `K = ceil(n_evaluation_records × p)`.
2. Rank by **unrounded positive-class score**, descending.
3. Break score ties by preserving original row order (stable sort).
4. Select exactly K records and calculate metrics on that selected group.

The current Top-K analysis ranks each entire validation split as a single pool. It **does not** reproduce daily contact quotas, unique-customer constraints, or a live campaign assignment policy.

## 5. Established Baseline

The following settings and results are **reported in the existing baseline notebook**; they have not been rerun as part of this README edit.

| Component | Baseline configuration |
|---|---|
| Model | `LogisticRegression(max_iter=1000)` |
| Numeric preprocessing | `StandardScaler`, including raw `pdays` |
| Categorical preprocessing | `OneHotEncoder(handle_unknown="ignore")` |
| Inputs | 19 predictors, excluding `duration` |
| Training / evaluation | Train / Validation |

The baseline retains `"unknown"` as an explicit category. It does not initially apply rare-category grouping, outlier removal/clipping, log transforms, or correlation-based feature deletion. The value `pdays = 999` is a sentinel; alternative representations belong in the experiment notebook.

### 5.1 Reported Validation Results

| Metric | Result |
|---|---:|
| Average Precision | **0.297231** |
| ROC-AUC | **0.692923** |
| Positive-class prevalence | Approximately **13.86%** |

| Capacity | K records | Subscribers captured | Precision@K | Recall@K | Lift@K |
|---|---:|---:|---:|---:|---:|
| Top 5% | 412 | 177 | 0.4296 | 0.1550 | 3.0991 |
| Top 10% | 824 | 318 | 0.3859 | 0.2785 | 2.7839 |
| Top 20% | 1,648 | 503 | 0.3052 | 0.4405 | 2.2017 |

At the **top 10%** capacity, 824 validation records contain 318 observed subscribers, capturing approximately **27.85%** of subscribers in the validation split. The observed subscription rate in this group is **38.59%**, approximately **2.78×** the validation prevalence.

Values are rounded for display. Experiments should compare full-precision computed metrics under the same protocol.

## 6. Notebook Workflow and Status

The filenames below follow the **existing README's notebook naming convention**.

| Notebook | Responsibility | Status reported in project notes |
|---|---|---|
| `01-data-understanding.ipynb` | Data inspection, EDA, leakage assessment, chronological split | Completed |
| `02-preprocessing.ipynb` | Feature definitions and baseline preprocessing | Completed |
| `03-baseline-modeling.ipynb` | Baseline pipeline and validation results | Completed |
| `04-model-experiments.ipynb` | Feature/preprocessing experiments, model comparison, tuning and selection | Next |
| `05-final-evaluation.ipynb` | Final training, held-out evaluation and limitations | Planned |

Problem definition and acceptance criteria are documented here. Experimental hypotheses, detailed analyses, and decisions belong in the corresponding notebooks. Stable shared logic may be moved incrementally to `src/bank_marketing_ml/`, with focused tests where appropriate.

## 7. Reproducibility

The existing project notes specify:

- **Environment:** Windows 11 + WSL2 Ubuntu 24.04; VS Code (WSL)
- **Python:** 3.12
- **Dependency manager:** `uv`
- **Dependency files:** `pyproject.toml` and `uv.lock`

From the project root:

```bash
uv sync
uv run python --version
```

Use the project's Python environment as the Jupyter kernel. Each notebook should run top-to-bottom from a fresh kernel and load required data or shared code explicitly.

Keep data locations, dependency versions, preprocessing decisions, model settings, split boundaries, and evaluation rules documented. Set and record random seeds whenever randomness is introduced. Do not assume numerical results are unchanged after changes to the dataset, dependencies, or protocol.

## 8. Limitations and Next Steps

**Known limitations**

- There is **no unique customer ID**, so repeated customers and cross-split customer overlap cannot be reliably audited.
- The data lacks full contact timestamps; exact day boundaries and operational contact schedules cannot be reconstructed.
- A single chronological validation period does not establish performance across every future period.
- Historical contacted-customer outcomes may not generalize to an entirely different eligible customer pool.
- Point-in-time availability, publication lags, and revisions of macroeconomic features remain to be verified against an operational data source.
- Score calibration and causal effects of contacting customers have not been established.
- Contact capacity, costs, conversion value, and business acceptance thresholds are not yet specified.

**Next steps**

1. Complete `04-model-experiments.ipynb`: compare feature representations and candidate models using the fixed validation protocol; select and document the final configuration.
2. Complete `05-final-evaluation.ipynb`: freeze the configuration, perform the predetermined final training procedure, evaluate on Test, and report generalization limits.
3. Consider software packaging, deployment, and monitoring **only after** the modeling phase and actual operational requirements justify them.

## 9. References

- [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [UCI dataset DOI: 10.24432/C5K306](https://doi.org/10.24432/C5K306)
- Internal project evidence: notebooks `01`–`03` (data understanding, preprocessing, baseline modeling)
