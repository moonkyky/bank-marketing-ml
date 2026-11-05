"""Feature representation frozen in the modeling notebooks."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

MODEL_FEATURES = (
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
)
NUMERIC_FEATURES = (
    "age",
    "campaign",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
)
CATEGORICAL_FEATURES = (
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
)


def split_pdays(values: ArrayLike) -> NDArray[np.float64]:
    """Separate recorded prior contact from elapsed days for a single column."""
    pdays = np.asarray(values).reshape(-1)
    previously_contacted = (pdays != 999).astype(float)
    elapsed_days = np.where(pdays == 999, 0.0, pdays.astype(float))
    return np.column_stack([previously_contacted, elapsed_days])


def _pdays_feature_names(transformer, input_features) -> NDArray[np.object_]:
    return np.asarray(["pdays_previously_contacted", "pdays_elapsed"], dtype=object)


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted preprocessor for use inside a model Pipeline."""
    pdays_transformer = Pipeline(
        [
            (
                "derive",
                FunctionTransformer(
                    split_pdays, validate=False, feature_names_out=_pdays_feature_names
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("num", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURES),
            ),
            ("pdays", pdays_transformer, ["pdays"]),
        ],
        remainder="drop",
    )
