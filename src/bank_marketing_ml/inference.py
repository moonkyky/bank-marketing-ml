"""Read and score an unlabeled CSV using an already fitted Pipeline."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from bank_marketing_ml.artifact import load_model
from bank_marketing_ml.features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
)


def read_scoring_csv(path: str | Path, *, sep: str = ";") -> pd.DataFrame:
    """Return the required inputs in model order, preserving CSV record order."""
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source, delimiter=sep, strict=True)
            header = next(reader, [])
            if not header:
                raise ValueError("Input CSV must have a header.")
            if len(set(header)) != len(header):
                raise ValueError("Input CSV contains duplicate column names.")
            missing = [name for name in MODEL_FEATURES if name not in header]
            if missing:
                raise ValueError(f"Input CSV is missing required columns: {missing}")
            indices = [header.index(name) for name in MODEL_FEATURES]
            rows = []
            for row in reader:
                if not row:
                    continue
                if len(row) != len(header):
                    raise ValueError(
                        f"CSV line {reader.line_num} has the wrong number of fields."
                    )
                rows.append([row[index] for index in indices])
    except (csv.Error, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not parse input CSV: {path}") from exc

    if not rows:
        raise ValueError("Input CSV must contain at least one record.")
    features = pd.DataFrame(rows, columns=MODEL_FEATURES)
    for name in (*NUMERIC_FEATURES, "pdays"):
        values = pd.to_numeric(features[name], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Column {name!r} must contain finite numbers.")
        features[name] = values
    for name in CATEGORICAL_FEATURES:
        if features[name].str.strip().eq("").any():
            raise ValueError(f"Column {name!r} must not contain blank categories.")
    return features


def score_csv(
    input_csv: str | Path, artifact_path: str | Path, *, sep: str = ";"
) -> pd.Series:
    """Score records in original order with a trusted local artifact."""
    features = read_scoring_csv(input_csv, sep=sep)
    pipeline, _ = load_model(artifact_path)
    return pd.Series(pipeline.predict_proba(features)[:, 1], name="score")
