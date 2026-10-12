# Bank Marketing ML — Repository Instructions

These instructions apply to the entire repository.

## Project Objective

Build a scientifically valid, reproducible, and maintainable end-to-end
machine learning project using the UCI Bank Marketing dataset. The model ranks
campaign records by the likelihood of term-deposit subscription immediately
before a planned contact.

Use `README.md` for the current project definition and documented decisions.
Use the relevant notebook or source file as evidence for implementation and
experimental results. Do not treat prior chat context as the sole source of
truth.

## Environment

- OS target: Windows 11 with WSL2 Ubuntu 24.04.
- Python: 3.12.
- Package and environment manager: `uv`.
- Prefer `uv run` for Python commands, tests, linting, and scripts.
- Respect `pyproject.toml` and `uv.lock`; do not add or upgrade dependencies
  without explaining the need and trade-offs first.
- Do not commit raw data, generated model artifacts, secrets, caches, or the
  virtual environment.

## ML Methodology — Non-Negotiable Rules

- Target: `y`, encoded as `yes = 1` and `no = 0` when modeling.
- Prediction time: immediately before the planned marketing contact.
- Exclude `duration`; it is unavailable before the contact finishes.
- Retain `contact`; the communication channel is decided before scoring.
- Interpret `campaign` according to the UCI definition: the cumulative number
  of contacts performed for the client during the current campaign, including
  the last contact. UCI does not provide a separate planned contact-attempt
  variable.
- Retain `campaign`; the project assumes this cumulative count of contacts
  performed so far is known operational history at scoring time. Do not
  reinterpret it as the ordinal number of a future planned contact.
- Treat changes in the campaign workflow as a possible distribution-shift
  risk, not as evidence that `campaign` is inherently target leakage.
- Preserve the source row order and the established chronological split:
  Train 70%, Validation 20%, Test 10%.
- Do not shuffle or stratify the established chronological holdout.
- Do not reconstruct chronology by sorting `month` or `day_of_week`.
- Fit learned preprocessing, feature selection, resampling, and models only on
  Train during development, or inside each training fold when cross-validation
  is used.
- Use time-aware folds for cross-validation; never fit preprocessing outside
  the fold pipeline.
- Use Validation for experiments, comparison, tuning, and final model
  selection.
- Do not inspect or use Test features, targets, metrics, or distributions for
  development decisions. Test may be used only in the final-evaluation stage,
  after the full model configuration and evaluation protocol are frozen.
- The historical aggregate inspection of the Test target must remain disclosed;
  do not describe Test as completely unseen in every respect.
- Set and record random seeds whenever randomness is introduced.
- Compare experiments under the same split, target, feature-availability
  policy, and evaluation rules.

## Evaluation Protocol

- Primary model-selection metric: Average Precision (AP).
- Supporting discrimination metric: ROC-AUC.
- Business-oriented ranking metrics: Precision@K, Recall@K, and Lift@K at the
  top 5%, 10%, and 20% of the evaluation records.
- For a fraction `p`, use `K = ceil(n_evaluation_records * p)`.
- Rank by unrounded positive-class scores in descending order.
- Break score ties by preserving original row order with a stable sort.
- Compute Lift@K relative to the positive-class prevalence of the evaluation
  split.
- Treat Top-K results as record-based because no unique customer identifier is
  available.
- Do not claim profitability, causal uplift, or the incremental effect of
  contact from predictive metrics.
- Do not describe uncalibrated scores as verified subscription probabilities.
- Never invent experimental results or claim an improvement without executed
  evidence.
- Clearly distinguish values stored in an existing notebook from results newly
  verified during the current task.

## Experiment Design

- Record experiments using the sequence:
  Hypothesis → Experiment → Evaluation → Decision.
- Change as few experimental factors as practical at one time.
- Reproduce the established baseline before interpreting improvements when the
  task requires a new experimental run.
- Consider validation performance together with temporal shift, Top-K behavior,
  stability, inference cost, implementation complexity, and maintainability.
- Prefer the simpler model when performance differences are small or do not
  justify added complexity.
- Keep Test sealed even if a development result is inconclusive.

## Notebook Conventions

Preserve the existing notebook filenames and their responsibilities:

- `01-data-understanding.ipynb`: inspection, EDA, leakage, split, findings.
- `02-preprocessing.ipynb`: feature definitions and preprocessing.
- `03-baseline-modeling.ipynb`: baseline training and validation evaluation.
- `04-model-experiments.ipynb`: experiments, tuning, analysis, selection.
- `05-final-evaluation.ipynb`: frozen final training and Test evaluation.

Within notebooks:

- Use Markdown cells for Objective, Methodology, Results, Interpretation,
  Decision, and Summary.
- Use code cells for computation rather than lengthy narrative.
- Keep notebooks executable top-to-bottom from a fresh kernel.
- Do not rely on state created by another notebook.
- Load data and shared code explicitly.
- Avoid unnecessary duplicate code, but move logic to `src/` only after it is
  stable and genuinely reusable.
- Add a new notebook or module only when its responsibility is clear and cannot
  be handled cleanly by the existing structure.
- Do not erase or replace valid saved outputs without a task-specific reason.

## Engineering Conventions

- Keep solutions simple, focused, and reproducible.
- Preserve existing names, structure, and conventions unless a change is
  justified.
- Use type hints where they improve reusable code.
- Prefer focused functions and meaningful tests over premature abstraction.
- Avoid unnecessary infrastructure. FastAPI, Docker, cloud services, experiment
  trackers, and orchestration tools require a demonstrated project need.
- Adjust Phase 3 and Phase 4 scope to actual deployment requirements and
  available resources.
- Validate input schemas, preprocessing parity, artifact compatibility, and
  failure behavior before declaring training or inference code release-ready.

## Working Agreement

- Inspect the relevant existing files and Git status before making changes.
- Read only the project material needed for the current task; do not perform a
  full repository audit for a small, isolated change.
- For assessment, explanation, diagnosis, or review requests, remain read-only
  unless the user also asks for implementation.
- Before editing, briefly explain the reason for the change and its expected
  file scope.
- Modify only files relevant to the requested lifecycle stage or section.
- Preserve unrelated user changes in a dirty worktree.
- Do not silently broaden the task into later lifecycle stages.
- Ask for direction when a choice would materially alter scientific scope,
  evaluation policy, dependencies, deployment strategy, or external state.
- Write explanations primarily in Korean. Keep code, filenames, notebook
  headings, metric names, and established technical terms in English.

## Verification and Reporting

- Use verification proportional to the change.
- Prefer `uv run pytest` for tests and `uv run ruff check .` for linting when
  relevant.
- Execute or retrain notebooks only when the task authorizes experimental
  execution; do not run expensive modeling merely to edit documentation.
- When a notebook is executed, verify fresh-kernel, top-to-bottom behavior when
  practical and report the exact scope that was run.
- Report checks as succeeded, failed, or not run.
- Summarize changed files, methodological decisions, measured results, and
  remaining limitations.
- Do not commit, push, create a pull request, or modify remote systems unless
  the user explicitly requests it.
