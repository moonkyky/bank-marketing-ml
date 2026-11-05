import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from bank_marketing_ml.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    split_pdays,
)
from bank_marketing_ml.train import build_final_pipeline


def test_pdays_distinguishes_sentinel_from_real_zero():
    np.testing.assert_array_equal(
        split_pdays([[999], [0], [6]]), [[0, 0], [1, 0], [1, 6]]
    )


def test_frozen_pipeline_fits_on_training_rows_and_handles_unseen_category():
    train = pd.DataFrame({name: [10.0, 20.0, 30.0, 40.0] for name in NUMERIC_FEATURES})
    for name in CATEGORICAL_FEATURES:
        train[name] = "unknown"
    train["pdays"] = [999, 0, 6, 999]
    train["duration"] = [5, 20, 100, 120]
    validation = train.iloc[[0, 1]].copy()
    validation["age"] = 999.0
    validation["contact"] = "telephone"
    validation["duration"] = 50_000

    pipeline = build_final_pipeline().fit(train, [0, 1, 1, 0])
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(validation)
    feature_names = preprocessor.get_feature_names_out().tolist()

    assert model.C == 1.0
    assert model.solver == "lbfgs"
    assert model.max_iter == 1000
    assert model.tol == 0.0001
    assert model.class_weight is None
    assert preprocessor.named_transformers_["num"].mean_[0] == pytest.approx(25.0)
    assert not any("duration" in name for name in feature_names)
    assert "pdays__pdays_elapsed" in feature_names
    contact_index = feature_names.index("cat__contact_unknown")
    np.testing.assert_array_equal(transformed[:, contact_index], [0.0, 0.0])

    scores = pipeline.predict_proba(validation)[:, 1]
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(
        scores, pipeline.predict_proba(validation.drop(columns="duration"))[:, 1]
    )
    check_is_fitted(pipeline)
    assert not hasattr(clone(pipeline).named_steps["model"], "classes_")
    assert not hasattr(build_final_pipeline().named_steps["model"], "classes_")
