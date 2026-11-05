"""Development data preparation for the canonical UCI CSV."""

from pathlib import Path

import pandas as pd

from bank_marketing_ml.features import MODEL_FEATURES

TRAIN_END = 28_831
VALIDATION_END = 37_069


def load_development_splits(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read only the canonical Train and Validation prefix in source order."""
    development = pd.read_csv(path, sep=";", nrows=VALIDATION_END)
    if len(development) != VALIDATION_END:
        raise ValueError(f"Expected at least {VALIDATION_END} development rows.")
    return development.iloc[:TRAIN_END].copy(), development.iloc[TRAIN_END:].copy()


def prepare_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select the frozen raw inputs and encode the observed target labels."""
    features = frame.loc[:, list(MODEL_FEATURES)].copy()
    target = frame["y"].map({"no": 0, "yes": 1})
    if target.isna().any():
        raise ValueError("Target y must contain only 'no' and 'yes'.")
    return features, target
