"""Persistence for trusted, locally produced model artifacts."""

import hashlib
import json
import os
import pickle
import platform
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile, ZipFile

from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from bank_marketing_ml.features import MODEL_FEATURES

FORMAT_VERSION = 1


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        **{
            package: version(package)
            for package in (
                "bank-marketing-ml",
                "scikit-learn",
                "numpy",
                "scipy",
                "pandas",
            )
        },
    }


def _check_pipeline(pipeline: Pipeline) -> None:
    if not isinstance(pipeline, Pipeline) or list(pipeline.named_steps) != [
        "preprocessor",
        "model",
    ]:
        raise ValueError("Artifact does not contain the expected model Pipeline.")
    for step in pipeline.named_steps.values():
        check_is_fitted(step)
    if tuple(pipeline.feature_names_in_) != MODEL_FEATURES:
        raise ValueError("Pipeline features do not match the frozen input schema.")
    if pipeline.classes_.tolist() != [0, 1]:
        raise ValueError("Pipeline classes must be ordered as [0, 1].")


def save_model(
    path: str | Path,
    pipeline: Pipeline,
    *,
    model_version: str,
    training_source: str,
    training_rows: int,
) -> None:
    """Save a fitted Pipeline and its metadata as one atomically replaced file."""
    if not model_version.strip():
        raise ValueError("Model version must not be empty.")
    _check_pipeline(pipeline)
    payload = pickle.dumps(pipeline, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {
        "format_version": FORMAT_VERSION,
        "model_version": model_version,
        "training_source": training_source,
        "training_rows": training_rows,
        "training_scope": "Train+Validation",
        "input_features": list(MODEL_FEATURES),
        "positive_class": 1,
        "versions": _runtime_versions(),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    path = Path(path)
    temporary_path = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            with ZipFile(temporary, "w") as archive:
                archive.writestr("metadata.json", json.dumps(metadata, indent=2))
                archive.writestr("pipeline.pkl", payload)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_model(path: str | Path) -> tuple[Pipeline, dict]:
    """Load only an artifact from a trusted source; pickle can execute code."""
    try:
        with ZipFile(path) as archive:
            metadata = json.loads(archive.read("metadata.json"))
            if not isinstance(metadata, dict) or metadata.get("format_version") != (
                FORMAT_VERSION
            ):
                raise ValueError("Unsupported model artifact format.")
            if (
                metadata.get("input_features") != list(MODEL_FEATURES)
                or metadata.get("positive_class") != 1
            ):
                raise ValueError(
                    "Model artifact input schema or classes are incompatible."
                )
            installed = _runtime_versions()
            saved = metadata.get("versions")
            if saved != installed:
                raise ValueError(
                    f"Model runtime versions are incompatible: saved={saved}, "
                    f"installed={installed}."
                )
            payload = archive.read("pipeline.pkl")
    except (BadZipFile, KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid model artifact: {path}") from exc
    if hashlib.sha256(payload).hexdigest() != metadata.get("payload_sha256"):
        raise ValueError("Model artifact checksum does not match.")
    try:
        pipeline = pickle.loads(payload)
    except (pickle.UnpicklingError, EOFError, ImportError, AttributeError) as exc:
        raise ValueError(f"Could not deserialize model artifact: {path}") from exc
    _check_pipeline(pipeline)
    return pipeline, metadata
