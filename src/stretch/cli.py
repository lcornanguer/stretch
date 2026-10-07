"""Command-line interface for STRETCH."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .estimator import STRETCH


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stretch",
        description="Learn a causal graph with time-varying causal delays from CSV data.",
    )
    parser.add_argument("input", type=Path, help="CSV file with observations in rows")
    parser.add_argument(
        "--search",
        choices=("greedy", "globe", "exhaustive", "topic"),
        default="greedy",
    )
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--max-parents", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("stretch-results"))
    parser.add_argument("--verbose", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Run the STRETCH CSV command."""

    args = _parser().parse_args(argv)
    frame = pd.read_csv(args.input)
    result = STRETCH(
        search=args.search,
        max_lag=args.max_lag,
        max_parents=args.max_parents,
        verbose=args.verbose,
    ).fit_result(frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    adjacency_path = args.output_dir / "adjacency.csv"
    delays_path = args.output_dir / "delays.csv"
    arrays_path = args.output_dir / "delay-functions.npz"
    metadata_path = args.output_dir / "metadata.json"

    result.adjacency.to_csv(adjacency_path)
    result.delays_frame().to_csv(delays_path, index=False)
    edge_records = []
    arrays = {}
    for edge_index, ((source, target), delays) in enumerate(result.delay_functions.items()):
        key = f"edge_{edge_index:04d}"
        arrays[key] = delays
        edge_records.append({"array": key, "source": str(source), "target": str(target)})
    np.savez_compressed(arrays_path, **arrays)

    metadata = {
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "search": result.search,
        "max_lag": result.max_lag,
        "variables": [str(name) for name in result.variable_names],
        "delay_arrays": edge_records,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
