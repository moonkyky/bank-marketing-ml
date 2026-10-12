# Bank Marketing ML

**Reproducible ranking of bank marketing records by term-deposit subscription likelihood.**

This project uses the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) to develop a leakage-aware machine learning workflow for prioritizing outbound marketing records when contact capacity is limited.

The project currently covers data understanding, preprocessing, baseline modeling, and model experiments. Independent Test evaluation, reusable packaging, batch inference, and offline monitoring are planned in later lifecycle stages.

## Project Status

| Area | Status |
|---|---|
| Phase 1 — Problem & Data | Complete |
| Stage 4 — Baseline Modeling | Complete |
| Stage 5 — Model Experiments | Revision required for sentinel-aware baseline |
| Stage 6 — Final Evaluation | Pending revised Stage 5 selection |
| Phase 3 — Engineering & Deployment | Not started |
| Phase 4 — Operations & Iteration | Not started |

Detailed lifecycle state, decisions, verification, and the next-stage handoff are maintained in [Project Status](docs/project-status.md).

## Key Validation Results

Notebook 03 was rerun from a fresh kernel with sentinel-aware `pdays` preprocessing. These are current **baseline Validation results**, not a final model selection or Test evaluation.

| Candidate | Average Precision | ROC-AUC | Lift@10% |
|---|---:|---:|---:|
| Sentinel-aware Logistic Regression baseline | 0.288234 | 0.688023 | 2.705123 |

At the top 10% capacity, the baseline ranks 309 of 1,142 observed Validation subscribers within 824 records. Precision@10% is `0.375000` and Recall@10% is `0.270578`.

The earlier raw-`pdays` baseline recorded AP `0.297231`. The current representation separates sentinel status from elapsed days as a semantic constraint, despite the AP reduction of `0.008997`. Notebook 04 still contains the historical experiments and must be revised before final selection. Test remains reserved for Stage 6.

## Problem Definition

| Item | Definition |
|---|---|
| Business decision | Prioritize eligible outbound marketing records under limited contact capacity |
| ML task | Binary classification used for record ranking |
| Prediction time | Immediately before a planned contact |
| Unit | One campaign record; no unique customer identifier is available |
| Target | Term-deposit subscription, `y` |
| Positive class | `yes = 1` |
| Model output | Positive-class score from `predict_proba` |
| Primary selection metric | Average Precision |

The model ranks records by score rather than making an operational decision at a fixed `0.5` threshold. Scores are not treated as verified subscription probabilities, and predictive performance does not establish the causal effect or profitability of contacting a record.

## Data

The project uses `bank-additional-full.csv` from the UCI Bank Marketing dataset.

| Property | Value |
|---|---|
| Records | 41,188 |
| Original predictors | 20 |
| Modeling predictors | 19 |
| Excluded predictor | `duration` |
| Target | `y` |
| File delimiter | Semicolon (`;`) |
| Covered period | May 2008–November 2010 |
| Local path | `data/raw/bank-additional/bank-additional-full.csv` |

Raw data is excluded from Git. Use the provided download script to prepare it locally.

### Feature and Leakage Policy

- Exclude `duration` because the current-contact duration is unavailable before the contact finishes.
- Retain the remaining 19 predictors under the established prediction-time feature policy.
- Treat `contact` as available because the communication channel is decided before scoring.
- Assume macroeconomic values were published and available at scoring time; an operational data contract must verify publication lags and revisions.
- Preserve literal `"unknown"` values as observed categories rather than silently treating them as missing.

## Methodology

### Chronological Split

The source row order is preserved. The split uses proportions of records rather than reconstructed calendar periods.

| Split | Rows | Purpose |
|---|---:|---|
| Train | 28,831 | Fit preprocessing and development models |
| Validation | 8,238 | Compare experiments and select one configuration |
| Test | 4,119 | Evaluate the frozen final approach in Stage 6 |

The project does not shuffle, stratify, or re-sort by `month` or `day_of_week`. Train prevalence is approximately 5.57%, while Validation prevalence is approximately 13.86%, so temporal and prevalence shift are central limitations.

Aggregate Test information was inspected during earlier EDA. Test is therefore not described as completely unseen in every respect, but it remains excluded from feature engineering, tuning, model selection, and ranking-policy selection.

### Preprocessing

The baseline scikit-learn `Pipeline` fits all learned steps on Train only during development.

| Branch | Configuration |
|---|---|
| Numeric | `StandardScaler` on eight ordinary numerical features |
| `pdays` | Derive `pdays_previously_contacted = (pdays != 999)` and `pdays_elapsed = 0` for sentinel rows, otherwise the recorded days; scale both derived columns |
| Categorical | `OneHotEncoder(handle_unknown="ignore")` |
| Missing-value imputation | Not applied; no recognized missing values were found in development inputs |
| Other feature engineering | None applied |

The 19 raw modeling inputs produce 62 transformed features: eight numerical, 52 one-hot categorical, and two sentinel-aware `pdays` columns. Raw `999` is not used as an elapsed-day value.

Cross-validation fits the complete preprocessing and estimator Pipeline independently within each training fold.

### Model Selection

The historical Stage 5 run used four expanding `TimeSeriesSplit` folds within Train for model-family comparison and hyperparameter search, followed by a small Validation comparison. Its raw-`pdays` selection is superseded by the revised representation policy.

Stage 5 must be rerun with sentinel-aware preprocessing in every candidate and fold. No current configuration is frozen for Stage 6. Once Stage 5 selects and freezes a configuration, Stage 6 will refit all learned steps on Train+Validation and evaluate Test once.

### Evaluation

- Primary: Average Precision
- Secondary: ROC-AUC
- Capacity-oriented: Precision@K, Recall@K, and Lift@K at 5%, 10%, and 20%
- Ranking: descending unrounded scores with stable tie handling
- Capacity size: `K = ceil(n_records × fraction)`
- Lift reference: positive-class prevalence of the evaluated split

Top-K results are record-based and rank an entire evaluation split as one pool. They do not reproduce daily quotas, repeat-contact rules, or unique-customer constraints.

## Repository Structure

```text
bank-marketing-ml/
├── data/                       # Local raw data; ignored by Git
├── docs/
│   └── project-status.md       # Lifecycle status, decisions, and handoff
├── notebooks/
│   ├── 01-data-understanding.ipynb
│   ├── 02-preprocessing.ipynb
│   ├── 03-baseline-modeling.ipynb
│   └── 04-model-experiments.ipynb
├── scripts/
│   └── download_data.py        # UCI dataset download and extraction
├── src/bank_marketing_ml/      # Package scaffold; refactoring begins in Stage 7
├── tests/                      # Minimal current smoke test
├── AGENTS.md                   # Repository-wide development rules
├── pyproject.toml
├── uv.lock
└── README.md
```

Reusable package modules and meaningful unit/integration tests are planned for Stage 7. The current `src/` entry point and smoke test are scaffolding, not a released training or inference interface.

## Getting Started

### Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Git

The project was developed with Windows 11 and WSL2 Ubuntu 24.04. Commands below run from the repository root.

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

Expected file:

```text
data/raw/bank-additional/bank-additional-full.csv
```

### Notebook Workflow

Start Jupyter from the project root:

```bash
uv run jupyter lab
```

Run notebooks 01–03 in order. Notebook 04 is historical and awaits revision:

1. [Data Understanding](notebooks/01-data-understanding.ipynb)
2. [Preprocessing](notebooks/02-preprocessing.ipynb)
3. [Baseline Modeling](notebooks/03-baseline-modeling.ipynb)
4. [Model Experiments](notebooks/04-model-experiments.ipynb) — revision required

Each notebook loads its own data and should run top-to-bottom from a fresh kernel. Notebook 05 and the supported training/scoring commands have not yet been implemented.

## Verification

Run the currently available repository checks:

```bash
uv run pytest
uv run ruff check .
```

Current test coverage is intentionally minimal and does not yet validate the modeling pipeline. Stage 7 will replace the placeholder smoke test with focused unit and integration tests.

## Intended Use

- Offline analysis of term-deposit subscription ranking
- Comparison of records at explicit contact-capacity scenarios
- Demonstration of a reproducible end-to-end ML lifecycle
- Future local CSV batch scoring after the release-readiness stages are complete

## Prohibited or Unsupported Use

- Claiming that contact causes subscription
- Claiming profitability or return on investment from Lift@K alone
- Treating uncalibrated scores as verified probabilities
- Treating one record as a guaranteed unique customer
- Automated production decisions without operational validation and governance
- Using the current package scaffold as a released training or inference service

## Limitations

- No unique customer ID is available, so repeated customers and cross-split customer overlap cannot be audited reliably.
- Complete contact timestamps are unavailable, so exact calendar boundaries and operational schedules cannot be reconstructed.
- A single chronological Validation period does not establish stability across all future periods.
- Historical contacted-record outcomes may not generalize to a different eligible population.
- Point-in-time macroeconomic availability, publication lags, and revisions require operational verification.
- Probability calibration and causal contact effects have not been established.
- Contact capacity, costs, subscription value, and business acceptance thresholds are not specified.
- Independent Test evaluation has not yet been completed.

## Roadmap

- [x] Stage 1 — Problem Definition
- [x] Stage 2 — Data Understanding
- [x] Stage 3 — Data Preprocessing
- [x] Stage 4 — Baseline Modeling
- [ ] Stage 5 — Model Experiments (revise for sentinel-aware baseline)
- [ ] Stage 6 — Final Evaluation
- [ ] Stage 7 — Refactoring & Packaging
- [ ] Stage 8 — ML Pipeline & Release Readiness
- [ ] Stage 9 — Model Deployment & Integration
- [ ] Stage 10 — Production Monitoring Simulation
- [ ] Stage 11 — Retraining & Continuous Improvement
- [ ] Stage 12 — Maintenance & Governance

## Documentation

- [Project Status](docs/project-status.md): current lifecycle state, verified decisions, and next-stage handoff
- [Repository Instructions](AGENTS.md): invariant development and ML methodology rules
- [Data Understanding](notebooks/01-data-understanding.ipynb): data inspection, EDA, leakage analysis, and split design
- [Preprocessing](notebooks/02-preprocessing.ipynb): feature groups and train-only preprocessing
- [Baseline Modeling](notebooks/03-baseline-modeling.ipynb): baseline training and Validation evaluation
- [Model Experiments](notebooks/04-model-experiments.ipynb): model comparison, tuning, analysis, and final selection

## References

- [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [Dataset DOI: 10.24432/C5K306](https://doi.org/10.24432/C5K306)
