"""Discrimination and record-based ranking metrics for one labeled split."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, roc_auc_score

TOP_FRACTIONS = (0.05, 0.10, 0.20)


def evaluate_ranking(
    y_true: ArrayLike, scores: ArrayLike
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate aligned 1-D inputs with both labels present and finite scores.

    Inputs are matched by position. Ties preserve that input order. Rounding
    is left to the caller for presentation.
    """
    labels = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("Labels and scores must be aligned one-dimensional arrays.")
    if not np.isin(labels, [0, 1]).all() or not (
        np.any(labels == 0) and np.any(labels == 1)
    ):
        raise ValueError("Labels must contain both binary classes, 0 and 1.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite.")

    prevalence = float(labels.mean())
    total_positives = int(labels.sum())
    ranked_labels = labels[np.argsort(-scores, kind="stable")]
    cumulative_positives = np.cumsum(ranked_labels)
    ranking_rows = []
    for fraction in TOP_FRACTIONS:
        k = int(np.ceil(len(labels) * fraction))
        captured = int(cumulative_positives[k - 1])
        precision = captured / k
        ranking_rows.append(
            {
                "top_fraction": fraction,
                "k": k,
                "captured_subscribers": captured,
                "precision_at_k": precision,
                "recall_at_k": captured / total_positives,
                "lift_at_k": precision / prevalence,
            }
        )

    summary = {
        "prevalence": prevalence,
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }
    return summary, pd.DataFrame(ranking_rows)
