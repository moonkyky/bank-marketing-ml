"""Run and check one CSV scoring batch with Prefect."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
from prefect import flow, get_run_logger, task

from bank_marketing_ml.cli import _check_output_path
from bank_marketing_ml.inference import read_scoring_csv


@task
def run_cli(
    source: Path, model: Path, output: Path, sep: str, image: str | None
) -> float:
    command = [str(Path(sys.executable).with_name("bank-marketing-ml"))]
    input_arg, model_arg, output_arg = str(source), str(model), str(output)
    if image:
        _check_output_path(output, source, model)
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--mount",
            f"type=bind,src={source.resolve()},dst=/input.csv,readonly",
            "--mount",
            f"type=bind,src={model.resolve()},dst=/model.zip,readonly",
            "--mount",
            f"type=bind,src={output.parent.resolve()},dst=/batch-output",
            image,
        ]
        input_arg, model_arg, output_arg = (
            "/input.csv",
            "/model.zip",
            (f"/batch-output/{output.name}"),
        )
    start = perf_counter()
    result = subprocess.run(
        [
            *command,
            "--input",
            input_arg,
            "--model",
            model_arg,
            "--output",
            output_arg,
            "--sep",
            sep,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"CLI exited {result.returncode}.")
    return perf_counter() - start


@task
def check_batch(source: Path, output: Path, sep: str, elapsed: float) -> dict:
    features = read_scoring_csv(source, sep=sep)
    results = pd.read_csv(output, float_precision="round_trip")
    scores = results["score"].to_numpy()
    if (
        results["input_row"].tolist() != list(range(1, len(features) + 1))
        or sorted(results["rank"]) != list(range(1, len(features) + 1))
        or not np.isfinite(scores).all()
    ):
        raise ValueError("Batch output contains invalid scores, ranks or row numbers.")

    report = {
        "input_rows": len(features),
        "output_rows": len(results),
        "elapsed_seconds": elapsed,
        "output_bytes": output.stat().st_size,
        "unknown_job_rate": float(features["job"].eq("unknown").mean()),
        "score_median": float(np.median(scores)),
    }
    get_run_logger().info("Batch report: %s", report)
    return report


@flow(name="bank-marketing-batch")
def batch_flow(
    source: Path, model: Path, output: Path, sep: str = ";", image: str | None = None
) -> dict:
    get_run_logger().info("Scoring with %s", model)
    _check_output_path(output, source, model)
    with TemporaryDirectory(dir=output.parent, prefix=".bank-batch-") as staging:
        candidate = Path(staging) / "scores.csv"
        elapsed = run_cli(source, model, candidate, sep, image)
        report = check_batch(source, candidate, sep, elapsed)
        _check_output_path(output, source, model)
        candidate.replace(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sep", choices=(";", ","), default=";")
    parser.add_argument("--image", help="Run the batch CLI in this local Docker image")
    args = parser.parse_args()
    batch_flow(args.input, args.model, args.output, args.sep, args.image)


if __name__ == "__main__":
    main()
