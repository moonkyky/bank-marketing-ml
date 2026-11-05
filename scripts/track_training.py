"""Track a frozen model build in MLflow using the development data only."""

import argparse
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from bank_marketing_ml.data import load_development_splits, prepare_xy
from bank_marketing_ml.evaluate import evaluate_ranking
from bank_marketing_ml.features import NUMERIC_FEATURES
from bank_marketing_ml.train import build_final_pipeline, train_and_save


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--register-as", help="Optional MLflow registered model name")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path to preserve previous model versions.")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("bank-marketing")
    train, validation = load_development_splits(args.data)
    x_train, y_train = prepare_xy(train)
    x_validation, y_validation = prepare_xy(validation)
    development = pd.concat([train, validation])
    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(development, index=False).to_numpy().tobytes()
    ).hexdigest()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        mlflow.start_run(run_name=args.model_version) as run,
        TemporaryDirectory(dir=args.output.parent) as staging,
    ):
        validation_model = build_final_pipeline().fit(x_train, y_train)
        metrics, top_k = evaluate_ranking(
            y_validation, validation_model.predict_proba(x_validation)[:, 1]
        )
        mlflow.log_params(
            {
                **validation_model.named_steps["model"].get_params(),
                "data_file": args.data.name,
                "development_fingerprint": fingerprint,
                "validation_fit_rows": len(train),
                "validation_rows": len(validation),
                "artifact_fit_rows": len(development),
                "artifact_version": args.model_version,
            }
        )
        mlflow.log_metrics(
            {f"validation_{key}": value for key, value in metrics.items()}
        )
        mlflow.log_dict(
            {"validation_top_k": top_k.to_dict("records")},
            "validation-top-k.json",
        )

        candidate = Path(staging) / args.output.name
        release = train_and_save(args.data, candidate, model_version=args.model_version)
        mlflow.log_artifact(str(candidate), artifact_path="batch")
        if args.register_as:
            schema_sample = x_train.head(1).astype(
                dict.fromkeys((*NUMERIC_FEATURES, "pdays"), float)
            )
            mlflow.sklearn.log_model(
                release,
                name="pipeline",
                registered_model_name=args.register_as,
                signature=infer_signature(
                    schema_sample, release.predict(schema_sample)
                ),
                skops_trusted_types=[
                    "bank_marketing_ml.features.split_pdays",
                    "bank_marketing_ml.features._pdays_feature_names",
                ],
                metadata={
                    "batch_zip": f"runs:/{run.info.run_id}/batch/{args.output.name}"
                },
            )
        os.link(candidate, args.output)
    print(f"MLflow run: {run.info.run_id}")


if __name__ == "__main__":
    main()
