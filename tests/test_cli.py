import csv
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from bank_marketing_ml.artifact import save_model
from bank_marketing_ml.features import CATEGORICAL_FEATURES, MODEL_FEATURES
from bank_marketing_ml.inference import score_csv
from bank_marketing_ml.train import build_final_pipeline


@pytest.fixture(scope="module")
def batch_models(tmp_path_factory):
    directory = tmp_path_factory.mktemp("batch-models")
    features = pd.DataFrame({name: [20, 40] for name in MODEL_FEATURES})
    for name in CATEGORICAL_FEATURES:
        features[name] = ["a", "b"]
    features["pdays"] = [999, 0]
    for version, labels in (("v1", [0, 1]), ("v2", [1, 0])):
        pipeline = build_final_pipeline().fit(features, labels)
        # Keep distinct scores that would tie if rounded to six decimals.
        pipeline.named_steps["model"].coef_ *= 1e-8
        save_model(
            directory / f"{version}.zip",
            pipeline,
            model_version=version,
            training_source="synthetic.csv",
            training_rows=len(features),
        )
    return features.iloc[[0, 1, 1]], directory / "v1.zip", directory / "v2.zip"


def run_batch(source, model, output, *options):
    command = Path(sys.executable).with_name("bank-marketing-ml")
    return subprocess.run(
        [
            str(command),
            "--input",
            str(source),
            "--model",
            str(model),
            "--output",
            str(output),
            *options,
        ],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("sep", [";", ","])
def test_batch_scores_and_ranks_in_input_order(batch_models, tmp_path, sep):
    features, model, _ = batch_models
    source = tmp_path / "contacts.csv"
    output = tmp_path / "scores.csv"
    features.to_csv(source, sep=sep, index=False)
    options = ("--sep", ",") if sep == "," else ()
    result = run_batch(source, model, output, *options)
    assert result.returncode == 0, result.stderr
    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [int(row["input_row"]) for row in rows] == [1, 2, 3]
    assert [int(row["rank"]) for row in rows] == [3, 1, 2]
    scores = [float(row["score"]) for row in rows]
    assert len(set(round(score, 6) for score in scores)) == 1
    np.testing.assert_array_equal(scores, score_csv(source, model, sep=sep))


def test_batch_rollback_by_model_path(batch_models, tmp_path):
    features, first_model, second_model = batch_models
    source = tmp_path / "contacts.csv"
    output = tmp_path / "scores.csv"
    features.to_csv(source, sep=";", index=False)
    snapshots = []
    for model in (first_model, second_model, first_model):
        result = run_batch(source, model, output)
        assert result.returncode == 0, result.stderr
        snapshots.append(output.read_bytes())
    assert snapshots[0] != snapshots[1]
    assert snapshots[0] == snapshots[2]


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("csv", "missing required columns"),
        ("missing", "No such file"),
        ("damaged", "Invalid model artifact"),
        ("incompatible", "runtime versions are incompatible"),
    ],
)
def test_batch_failure_preserves_existing_output(
    batch_models, tmp_path, problem, message
):
    features, model, _ = batch_models
    source = tmp_path / "contacts.csv"
    output = tmp_path / "scores.csv"
    output.write_bytes(b"previous result\n")
    features.to_csv(source, sep=";", index=False)
    if problem == "csv":
        source.write_text("age\n20\n", encoding="utf-8")
    elif problem == "missing":
        model = tmp_path / "missing.zip"
    elif problem == "damaged":
        model = tmp_path / "damaged.zip"
        model.write_bytes(b"not a ZIP")
    else:
        with ZipFile(model) as archive:
            metadata = json.loads(archive.read("metadata.json"))
            payload = archive.read("pipeline.pkl")
        metadata["versions"]["python"] = "incompatible"
        model = tmp_path / "incompatible.zip"
        with ZipFile(model, "w") as archive:
            archive.writestr("metadata.json", json.dumps(metadata))
            archive.writestr("pipeline.pkl", payload)
    before = set(tmp_path.iterdir())

    result = run_batch(source, model, output)

    assert result.returncode == 1
    assert message in result.stderr
    assert output.read_bytes() == b"previous result\n"
    assert set(tmp_path.iterdir()) == before


def test_batch_cleans_temporary_file_when_output_replace_fails(batch_models, tmp_path):
    features, model, _ = batch_models
    source = tmp_path / "contacts.csv"
    output = tmp_path / "scores.csv"
    features.to_csv(source, sep=";", index=False)
    output.mkdir()
    existing = output / "previous"
    existing.write_bytes(b"keep")
    before = set(tmp_path.iterdir())

    result = run_batch(source, model, output)

    assert result.returncode == 1
    assert "Cannot write output CSV" in result.stderr
    assert existing.read_bytes() == b"keep"
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("target", ["input", "model", "symlink", "hardlink"])
def test_batch_rejects_output_aliases(batch_models, tmp_path, target):
    features, model, _ = batch_models
    source = tmp_path / "contacts.csv"
    features.to_csv(source, sep=";", index=False)
    output = tmp_path / "alias.csv"
    if target == "input":
        output = source
    elif target == "model":
        output = model
    elif target == "symlink":
        output.symlink_to(source)
    else:
        output.hardlink_to(model)
    source_before, model_before = source.read_bytes(), model.read_bytes()

    result = run_batch(source, model, output)

    assert result.returncode == 1
    assert "must not overwrite" in result.stderr
    assert source.read_bytes() == source_before
    assert model.read_bytes() == model_before
