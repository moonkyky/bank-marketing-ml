"""Simulate offline monitoring for the CSV batch command with synthetic records."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd

from bank_marketing_ml.evaluate import evaluate_ranking


def make_reference() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    size = 200
    age = rng.integers(25, 66, size)
    features = pd.DataFrame(
        {
            "age": age,
            "job": rng.choice(
                ["admin.", "technician", "unknown"], size, p=[0.55, 0.4, 0.05]
            ),
            "marital": "married",
            "education": "university.degree",
            "default": "no",
            "housing": "yes",
            "loan": "no",
            "contact": rng.choice(["cellular", "telephone"], size),
            "month": "may",
            "day_of_week": "mon",
            "campaign": rng.integers(1, 6, size),
            "pdays": rng.choice([999, 999, 3, 6], size),
            "previous": 0,
            "poutcome": "nonexistent",
            "emp.var.rate": -1.8,
            "cons.price.idx": 92.893,
            "cons.conf.idx": -46.2,
            "euribor3m": 1.299,
            "nr.employed": 5099.1,
        }
    )
    features["previous"] = (features["pdays"] != 999).astype(int)
    features.loc[features["previous"] == 1, "poutcome"] = "success"
    # Simulate delayed outcomes keyed by the batch-local record number.
    labels = pd.Series(
        (rng.random(size) < age / 250).astype(int),
        index=pd.RangeIndex(1, size + 1, name="input_row"),
        name="y",
    )
    return features, labels


def run_batch(
    features: pd.DataFrame, model: Path, directory: Path, name: str
) -> tuple[dict, pd.DataFrame | None]:
    source = directory / f"{name}-input.csv"
    output = directory / f"{name}-scores.csv"
    features.to_csv(source, sep=";", index=False)
    command = Path(sys.executable).with_name("bank-marketing-ml")
    started = perf_counter()
    result = subprocess.run(
        [command, "--input", source, "--model", model, "--output", output],
        capture_output=True,
        text=True,
    )
    elapsed = perf_counter() - started
    report = {
        "input_rows": len(features),
        "exit_code": result.returncode,
        "elapsed_seconds": elapsed,
        "output_exists": output.exists(),
        "output_bytes": output.stat().st_size if output.exists() else 0,
    }
    if result.returncode:
        report.update(status="failed", output_rows=0, error=result.stderr.strip())
        return report, None

    scored = pd.read_csv(output, float_precision="round_trip")
    if scored["input_row"].tolist() != list(range(1, len(features) + 1)):
        raise ValueError(f"{name}: output records do not match the input order.")
    report.update(
        status="ok",
        output_rows=len(scored),
        rows_per_second=len(scored) / elapsed,
        unknown_job_rate=float(features["job"].eq("unknown").mean()),
        cellular_rate=float(features["contact"].eq("cellular").mean()),
        age_median=float(features["age"].median()),
        score_quantiles={
            f"p{percent}": float(scored["score"].quantile(percent / 100))
            for percent in (10, 50, 90)
        },
    )
    return report, scored


def labeled_performance(scored: pd.DataFrame, labels: pd.Series) -> dict:
    input_rows = pd.Index(scored["input_row"])
    if not input_rows.is_unique or not labels.index.is_unique:
        raise ValueError("Scores and labels must have unique input_row values.")
    aligned = labels.reindex(input_rows)
    if len(labels) != len(scored) or aligned.isna().any():
        raise ValueError("Labels must cover every scored record in this batch.")
    metrics, top_k = evaluate_ranking(aligned.to_numpy(), scored["score"].to_numpy())
    return {**metrics, "labeled_rows": len(aligned), "top_k": top_k.to_dict("records")}


def simulate(model: Path) -> tuple[dict, bool]:
    reference, labels = make_reference()
    shifted = reference.copy()
    shifted.loc[:99, "job"] = "unknown"
    shifted.loc[:139, "contact"] = "cellular"
    shifted.loc[:99, "age"] += 10
    with TemporaryDirectory(prefix="bank-marketing-monitoring-") as temporary:
        directory = Path(temporary)
        baseline, scored = run_batch(reference, model, directory, "reference")
        if scored is None:
            return {"model": str(model), "reference": baseline}, False

        invalid, _ = run_batch(
            reference.drop(columns="age"), model, directory, "invalid"
        )
        recovered, retry_scores = run_batch(reference, model, directory, "recovered")
        drift, shifted_scores = run_batch(shifted, model, directory, "shifted")
        larger, larger_scores = run_batch(
            pd.concat([reference] * 10, ignore_index=True), model, directory, "larger"
        )

    report = {
        "model": str(model),
        "synthetic_seed": 42,
        "reference": baseline,
        "invalid": invalid,
        "recovered": recovered,
        "shifted": drift,
        "larger": larger,
        "failure_and_retry_seconds": (
            invalid["elapsed_seconds"] + recovered["elapsed_seconds"]
        ),
        "labeled_reference": labeled_performance(scored, labels.iloc[::-1]),
    }
    if shifted_scores is not None:
        report["shift_vs_reference"] = {
            "unknown_job_percentage_points": 100
            * (drift["unknown_job_rate"] - baseline["unknown_job_rate"]),
            "cellular_percentage_points": 100
            * (drift["cellular_rate"] - baseline["cellular_rate"]),
            "age_median_years": drift["age_median"] - baseline["age_median"],
            "score_median_delta": (
                drift["score_quantiles"]["p50"] - baseline["score_quantiles"]["p50"]
            ),
        }

    ok = (
        invalid["exit_code"] == 1
        and not invalid["output_exists"]
        and "missing required columns" in invalid["error"]
        and scored.equals(retry_scores)
        and shifted_scores is not None
        and larger_scores is not None
    )
    return report, ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Trusted model ZIP")
    args = parser.parse_args()
    report, ok = simulate(args.model)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
