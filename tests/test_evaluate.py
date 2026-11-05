import numpy as np
import pytest

from bank_marketing_ml.evaluate import evaluate_ranking


def test_metrics_preserve_tie_order_and_round_capacity_up():
    labels = np.zeros(21, dtype=int)
    labels[[0, 2, 4, 20]] = 1

    summary, top_k = evaluate_ranking(labels, np.full(21, 0.5))

    assert summary == pytest.approx(
        {"prevalence": 4 / 21, "average_precision": 4 / 21, "roc_auc": 0.5}
    )
    assert top_k["top_fraction"].tolist() == [0.05, 0.10, 0.20]
    assert top_k["k"].tolist() == [2, 3, 5]
    assert top_k["captured_subscribers"].tolist() == [1, 2, 3]
    assert top_k["precision_at_k"].tolist() == pytest.approx([0.5, 2 / 3, 0.6])
    assert top_k["recall_at_k"].tolist() == pytest.approx([0.25, 0.5, 0.75])
    assert top_k["lift_at_k"].tolist() == pytest.approx(
        [0.5 / (4 / 21), (2 / 3) / (4 / 21), 0.6 / (4 / 21)]
    )


def test_close_scores_are_ranked_without_rounding():
    summary, top_k = evaluate_ranking([0, 1], [0.50000001, 0.50000002])

    assert summary["average_precision"] == pytest.approx(1.0)
    assert summary["roc_auc"] == pytest.approx(1.0)
    assert top_k["captured_subscribers"].tolist() == [1, 1, 1]


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        ([0, 1], [[0.1], [0.9]], "one-dimensional"),
        ([0, 1], [0.1], "aligned"),
        ([0, 0], [0.1, 0.9], "both binary classes"),
        ([0, 2], [0.1, 0.9], "binary"),
        ([0, 1], [0.1, np.nan], "finite"),
    ],
)
def test_evaluation_rejects_undefined_inputs(labels, scores, message):
    with pytest.raises(ValueError, match=message):
        evaluate_ranking(labels, scores)
