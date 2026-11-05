"""Construction of the frozen estimator pipeline."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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
