"""Integration checks for the optional UI and local MLOps entry points."""

import importlib
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from bank_marketing_ml import data
from bank_marketing_ml.artifact import load_model, save_model
from bank_marketing_ml.train import build_final_pipeline

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def example_batch(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    features, _ = importlib.import_module("simulate_monitoring").make_reference()
    features = features.head(12)
    labels = [0, 1] * 6
    pipeline = build_final_pipeline().fit(features, labels)
    model = tmp_path / "models" / "bank-marketing-v1.zip"
    model.parent.mkdir()
    save_model(
        model,
        pipeline,
        model_version="v1",
        training_source="example.csv",
        training_rows=len(features),
    )
    source = tmp_path / "contacts.csv"
    features.to_csv(source, sep=";", index=False)
    return source, model, features


def test_dashboard_resets_results_after_input_changes(example_batch, tmp_path):
    pytest.importorskip("gradio")
    dashboard_module = importlib.import_module("run_dashboard")
    _, model, _ = example_batch
    dashboard = dashboard_module.build_dashboard(tmp_path, model.parent)
    functions = {block.fn.__name__: block for block in dashboard.fns.values()}
    score = functions["on_run"]
    cards, rows, download, status = score.fn("합성 예제", None, model.name, ";")
    assert len(rows) == 10
    assert download["interactive"]
    assert len(pd.read_csv(download["value"])) == 12
    assert model.name in status and "example.csv" in status

    reset = functions["reset_results"]
    assert set(reset.targets) == {
        (component._id, "change") for component in score.inputs
    }
    _, rows, download, status = reset.fn()
    assert not rows and not status
    assert download["value"] is None and not download["interactive"]


def test_prefect_publishes_only_checked_output(example_batch, tmp_path, monkeypatch):
    pytest.importorskip("prefect")
    from prefect.testing.utilities import prefect_test_harness

    workflow = importlib.import_module("run_batch_flow")
    source, model, _ = example_batch
    output = tmp_path / "scores.csv"
    image = os.environ.get("BANK_MARKETING_TEST_IMAGE")
    with prefect_test_harness(server_startup_timeout=90):
        report = workflow.batch_flow(source, model, output, image=image)
        assert report["input_rows"] == report["output_rows"] == 12
        original = output.read_bytes()

        def corrupt_output(source, model, candidate, sep, image):
            candidate.write_text("input_row,score,rank\n9,0.5,1\n")
            return 0.1

        monkeypatch.setattr(workflow, "run_cli", corrupt_output)
        with pytest.raises(ValueError, match="invalid scores, ranks or row numbers"):
            workflow.batch_flow(source, model, output, image=image)
        assert output.read_bytes() == original
        assert not list(tmp_path.glob(".bank-batch-*"))


def test_mlflow_build_uses_one_development_snapshot(
    example_batch, tmp_path, monkeypatch
):
    mlflow = pytest.importorskip("mlflow")
    tracking = importlib.import_module("track_training")
    _, _, features = example_batch
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(data, "TRAIN_END", 8)
    monkeypatch.setattr(data, "VALIDATION_END", 12)
    source = tmp_path / "development.csv"
    features.assign(y=["no", "yes"] * 6).to_csv(source, sep=";", index=False)
    output = tmp_path / "candidate.zip"
    uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "track_training.py",
            "--data",
            str(source),
            "--output",
            str(output),
            "--model-version",
            "candidate",
            "--tracking-uri",
            uri,
            "--register-as",
            "bank-marketing",
        ],
    )
    read_development = tracking.load_development_splits

    def read_then_remove(path):
        split = read_development(path)
        path.unlink()
        return split

    monkeypatch.setattr(tracking, "load_development_splits", read_then_remove)
    tracking.main()
    client = mlflow.MlflowClient(tracking_uri=uri)
    (version,) = client.search_model_versions("name='bank-marketing'")
    run = client.get_run(version.run_id)
    assert run.info.status == "FINISHED"
    assert "validation_average_precision" in run.data.metrics
    assert [
        artifact.path for artifact in client.list_artifacts(run.info.run_id, "batch")
    ] == ["batch/candidate.zip"]
    _, metadata = load_model(output)
    assert metadata["training_rows"] == 12
