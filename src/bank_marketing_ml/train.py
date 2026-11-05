"""Construction of the frozen estimator pipeline."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from bank_marketing_ml.artifact import save_model
from bank_marketing_ml.data import load_development_splits, prepare_xy
from bank_marketing_ml.features import build_preprocessor


def build_final_pipeline() -> Pipeline:
    """Return a fresh, unfitted copy of the selected model configuration."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "model",
                LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, tol=0.0001),
            ),
        ]
    )


def train_and_save(
    source_csv: str | Path, artifact_path: str | Path, *, model_version: str
) -> Pipeline:
    """Fit the frozen configuration on the development prefix and save it."""
    if Path(source_csv).resolve() == Path(artifact_path).resolve():
        raise ValueError("The model artifact must not overwrite the training CSV.")
    train, validation = load_development_splits(source_csv)
    features, target = prepare_xy(pd.concat([train, validation]))
    pipeline = build_final_pipeline().fit(features, target)
    save_model(
        artifact_path,
        pipeline,
        model_version=model_version,
        training_source=Path(source_csv).name,
        training_rows=len(features),
    )
    return pipeline
