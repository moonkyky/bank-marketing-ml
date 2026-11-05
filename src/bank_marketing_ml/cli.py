"""Score a local CSV batch with a saved model."""

import argparse
import csv
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from bank_marketing_ml.inference import score_csv


def _check_output_path(output: Path, *protected_paths: Path) -> None:
    resolved_output = output.resolve()
    for source in protected_paths:
        if resolved_output == source.resolve() or (
            output.exists() and source.exists() and output.samefile(source)
        ):
            raise ValueError(f"Output CSV must not overwrite {source}.")


def _write_results(output: Path, scores: np.ndarray) -> None:
    ranked_indices = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(ranked_indices)
    ranks[ranked_indices] = np.arange(1, len(scores) + 1)

    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=output.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.writer(temporary)
            writer.writerow(("input_row", "score", "rank"))
            for input_row, (score, rank) in enumerate(zip(scores, ranks), start=1):
                writer.writerow((input_row, float(score), rank))
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    """Parse batch arguments and report operational errors to stderr."""
    parser = argparse.ArgumentParser(
        description="Score and rank CSV records using a trusted local model ZIP."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV path")
    parser.add_argument("--model", type=Path, required=True, help="Trusted model ZIP")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument(
        "--sep", choices=(";", ","), default=";", help="Input delimiter (default: ';')"
    )
    args = parser.parse_args()

    try:
        _check_output_path(args.output, args.input, args.model)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Cannot use output CSV {args.output}: {exc}", file=sys.stderr)
        return 1

    try:
        scores = score_csv(args.input, args.model, sep=args.sep)
    except (OSError, ValueError) as exc:
        print(
            f"Cannot score {args.input} with model {args.model}: {exc}", file=sys.stderr
        )
        return 1

    try:
        _write_results(args.output, scores.to_numpy())
    except OSError as exc:
        print(f"Cannot write output CSV {args.output}: {exc}", file=sys.stderr)
        return 1
    return 0
