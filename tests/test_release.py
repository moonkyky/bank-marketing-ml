import csv
import json
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from bank_marketing_ml.artifact import load_model
from bank_marketing_ml.data import VALIDATION_END, prepare_xy
from bank_marketing_ml.features import CATEGORICAL_FEATURES, MODEL_FEATURES
from bank_marketing_ml.inference import read_scoring_csv, score_csv
from bank_marketing_ml.train import train_and_save


@pytest.fixture(scope="module")
def released_model(tmp_path_factory):
    directory = tmp_path_factory.mktemp("release")
    training_csv = directory / "training.csv"
    model_path = directory / "model.zip"
    example = pd.DataFrame({name: [20, 40, 35, 50] for name in MODEL_FEATURES})
    for name in CATEGORICAL_FEATURES:
        example[name] = ["a", "b", "unknown", "a"]
    example["pdays"] = [999, 0, 6, 999]
    example["y"] = ["no", "yes", "yes", "no"]
    development = example.iloc[np.arange(VALIDATION_END) % len(example)]
    reserved = example.iloc[[0]].assign(y="must_not_be_loaded")
    pd.concat([development, reserved]).to_csv(training_csv, sep=";", index=False)
    model = train_and_save(training_csv, model_path, model_version="test-v1")
    return model, model_path, example


def test_training_artifact_scores_identically_in_separate_process(
    released_model, tmp_path
):
    fitted, model_path, example = released_model
    features, _ = prepare_xy(example)
    features.loc[0, "contact"] = "new category"
    scoring_csv = tmp_path / "scoring.csv"
    features.assign(duration=500).iloc[:, ::-1].to_csv(
        scoring_csv, sep=",", index=False
    )

    expected = fitted.predict_proba(features)[:, 1]
    _, metadata = load_model(model_path)
    assert metadata["model_version"] == "test-v1"
    assert metadata["training_source"] == "training.csv"
    assert metadata["training_rows"] == VALIDATION_END
    assert metadata["input_features"] == list(MODEL_FEATURES)
    np.testing.assert_array_equal(score_csv(scoring_csv, model_path, sep=","), expected)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from bank_marketing_ml.inference import score_csv; "
            "print(score_csv(*sys.argv[1:], sep=',').to_json(double_precision=15))",
            str(scoring_csv),
            str(model_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    np.testing.assert_allclose(
        list(json.loads(result.stdout).values()), expected, rtol=0, atol=1e-15
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("version", "runtime versions"),
        ("schema", "input schema"),
        ("format", "artifact format"),
        ("payload", "checksum"),
        ("missing", "Invalid model artifact"),
    ],
)
def test_artifact_rejected_before_deserialization(
    released_model, tmp_path, monkeypatch, change, message
):
    _, model_path, _ = released_model
    with ZipFile(model_path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        payload = archive.read("pipeline.pkl")
    if change == "version":
        metadata["versions"]["scikit-learn"] = "incompatible"
    elif change == "schema":
        metadata["input_features"].reverse()
    elif change == "format":
        metadata["format_version"] += 1
    elif change == "payload":
        payload = b"damaged"
    bad_path = tmp_path / "bad.zip"
    with ZipFile(bad_path, "w") as archive:
        archive.writestr("metadata.json", json.dumps(metadata))
        if change != "missing":
            archive.writestr("pipeline.pkl", payload)

    def must_not_load(_):
        pytest.fail("Incompatible artifact reached pickle deserialization")

    monkeypatch.setattr("bank_marketing_ml.artifact.pickle.loads", must_not_load)
    with pytest.raises(ValueError, match=message):
        load_model(bad_path)


def test_missing_or_non_zip_artifact_has_clear_failure(tmp_path):
    model_path = tmp_path / "model.zip"
    with pytest.raises(FileNotFoundError):
        load_model(model_path)
    model_path.write_text("not an archive", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid model artifact"):
        load_model(model_path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing", "missing required columns.*age"),
        ("duplicate", "duplicate column names"),
        ("non_numeric", "age.*finite numbers"),
        ("infinite", "pdays.*finite numbers"),
        ("blank", "contact.*blank categories"),
        ("wide", "wrong number of fields"),
        ("empty", "at least one record"),
    ],
)
def test_invalid_scoring_csv_has_actionable_failure(tmp_path, change, message):
    path = tmp_path / "invalid.csv"
    header = list(MODEL_FEATURES)
    row = ["1"] * len(header)
    if change == "missing":
        header.remove("age")
        row.pop(0)
    elif change == "duplicate":
        header.append("age")
        row.append("2")
    elif change == "non_numeric":
        row[header.index("age")] = "oops"
    elif change == "infinite":
        row[header.index("pdays")] = "inf"
    elif change == "blank":
        row[header.index("contact")] = " "
    elif change == "wide":
        row.append("extra")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(header)
        if change != "empty":
            writer.writerow(row)
    with pytest.raises(ValueError, match=message):
        read_scoring_csv(path)


def test_training_cannot_replace_source_csv(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("y\nno\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must not overwrite"):
        train_and_save(source, source, model_version="test-v1")
    assert source.read_text(encoding="utf-8") == "y\nno\n"
