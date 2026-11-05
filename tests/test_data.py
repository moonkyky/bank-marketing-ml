import pandas as pd
import pytest

from bank_marketing_ml.data import load_development_splits, prepare_xy
from bank_marketing_ml.features import MODEL_FEATURES


def test_load_development_preserves_chronological_boundaries(tmp_path):
    path = tmp_path / "synthetic.csv"
    pd.DataFrame({"row": range(37_071), "y": "no"}).to_csv(path, sep=";", index=False)

    train, validation = load_development_splits(path)

    assert train["row"].tolist() == list(range(28_831))
    assert validation["row"].tolist() == list(range(28_831, 37_069))


def test_short_development_prefix_is_rejected(tmp_path):
    path = tmp_path / "short.csv"
    path.write_text("row;y\n0;no\n", encoding="utf-8")
    with pytest.raises(ValueError, match="37069"):
        load_development_splits(path)


def test_prepare_xy_excludes_duration_and_keeps_contact():
    frame = pd.DataFrame({name: [1, 2] for name in MODEL_FEATURES}, index=[5, 3])
    frame["contact"] = ["cellular", "telephone"]
    frame["duration"] = [60, 120]
    frame["y"] = ["yes", "no"]

    features, target = prepare_xy(frame)

    assert features.shape == (2, 19)
    assert "duration" not in features and "y" not in features
    assert features["contact"].tolist() == ["cellular", "telephone"]
    assert features.index.tolist() == [5, 3]
    assert target.tolist() == [1, 0]
    assert target.index.equals(features.index)


def test_prepare_xy_rejects_unmapped_target():
    frame = pd.DataFrame({name: [0] for name in MODEL_FEATURES})
    frame["y"] = ["maybe"]
    with pytest.raises(ValueError, match="Target y"):
        prepare_xy(frame)
