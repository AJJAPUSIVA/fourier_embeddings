"""Aggregate deterministic Kronecker/Fourier dimension-sweep runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from experiments.aggregate_training import (
        MATCHED_FIELDS,
        mean_std,
        mean_std_ci95,
        nested,
    )
except ModuleNotFoundError:  # Direct execution as experiments/aggregate_dimension_sweep.py.
    from aggregate_training import (  # type: ignore[no-redef]
        MATCHED_FIELDS,
        mean_std,
        mean_std_ci95,
        nested,
    )


def load_runs(root: Path) -> list[dict]:
    runs = []
    for path in sorted(root.rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema_version") == 1 and record.get("embedding") in {"kronecker", "fourier"}:
            record["source_file"] = str(path)
            runs.append(record)
    if not runs:
        raise ValueError(f"No sweep result JSON files found below {root}")
    return runs


def validate_sweep(runs: list[dict], dimensions: list[int], seeds: list[int]) -> None:
    keys = []
    for run in runs:
        dimension = run["fourier_dim"] if run["embedding"] == "fourier" else None
        keys.append((run["embedding"], dimension, run["seed"]))
    expected = {("kronecker", None, seed) for seed in seeds}
    expected.update(("fourier", dimension, seed) for dimension in dimensions for seed in seeds)
    duplicates = sorted(key for key in set(keys) if keys.count(key) > 1)
    missing = sorted(expected - set(keys), key=str)
    extra = sorted(set(keys) - expected, key=str)
    if duplicates or missing or extra:
        raise ValueError(f"Invalid sweep matrix: duplicates={duplicates}, missing={missing}, extra={extra}")
    baseline = runs[0]
    for run in runs[1:]:
        mismatches = [field for field in MATCHED_FIELDS if nested(run, field) != nested(baseline, field)]
        if mismatches:
            raise ValueError(f"Unmatched settings in {run['source_file']}: {mismatches}")


def aggregate_sweep(runs: list[dict], dimensions: list[int], seeds: list[int]) -> dict:
    kronecker = {run["seed"]: run for run in runs if run["embedding"] == "kronecker"}
    kron_ppl = [kronecker[seed]["final_validation_perplexity"] for seed in seeds]
    kron_params = next(iter(kronecker.values()))["parameters"]["embedding"]
    rows = []
    for dimension in dimensions:
        fourier = {
            run["seed"]: run for run in runs
            if run["embedding"] == "fourier" and run["fourier_dim"] == dimension
        }
        ppl = [fourier[seed]["final_validation_perplexity"] for seed in seeds]
        regressions = [100 * (fourier[seed]["final_validation_perplexity"] / kronecker[seed]["final_validation_perplexity"] - 1) for seed in seeds]
        params = next(iter(fourier.values()))["parameters"]["embedding"]
        rows.append({
            "dimension": dimension,
            "seeds": seeds,
            "validation_perplexity": mean_std(ppl),
            "paired_regression_pct": mean_std_ci95(regressions),
            "embedding_parameters": params,
            "parameter_reduction_vs_kronecker": kron_params / params,
            "tokens_per_second": mean_std([fourier[seed]["tokens_per_second"] for seed in seeds]),
        })
    first = runs[0]
    return {
        "schema_version": 1,
        "status": "DESCRIPTIVE_ONLY",
        "status_reason": "The dimension sweep is exploratory and has no post-hoc PASS/FAIL threshold.",
        "experiment": {
            "dataset": first["dataset"],
            "dataset_source": first["dataset_source"],
            "dataset_revision": first["dataset_revision"],
            "tokenizer": first["tokenizer"],
            "seeds": seeds,
            "dimensions": dimensions,
            "model_config": first["model_config"],
            "training": first["training"],
        },
        "kronecker": {
            "validation_perplexity": mean_std(kron_ppl),
            "embedding_parameters": kron_params,
            "tokens_per_second": mean_std([kronecker[seed]["tokens_per_second"] for seed in seeds]),
        },
        "fourier_dimensions": rows,
    }


def markdown(report: dict) -> str:
    kron = report["kronecker"]
    lines = [
        "# Deterministic Fourier dimension sweep", "",
        "**DESCRIPTIVE ONLY:** no post-hoc PASS/FAIL threshold is assigned.", "",
        f"Kronecker validation PPL: {kron['validation_perplexity']['mean']:.2f} ± {kron['validation_perplexity']['std']:.2f}", "",
        "| Fourier D | Validation PPL | Mean paired regression | 95% paired CI | Embedding params | Reduction vs Kronecker | Tokens/s |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["fourier_dimensions"]:
        ppl = row["validation_perplexity"]
        paired = row["paired_regression_pct"]
        ci = paired["ci95"]
        ci_text = "n/a" if ci is None else f"[{ci[0]:.2f}%, {ci[1]:.2f}%]"
        speed = row["tokens_per_second"]
        lines.append(
            f"| {row['dimension']} | {ppl['mean']:.2f} ± {ppl['std']:.2f} | "
            f"{paired['mean']:+.2f}% | {ci_text} | {row['embedding_parameters']:,} | "
            f"{row['parameter_reduction_vs_kronecker']:.1f}× | {speed['mean']:.1f} ± {speed['std']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dimensions", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--json-output", type=Path, default=Path("dimension_sweep.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("dimension_sweep.md"))
    args = parser.parse_args()
    runs = load_runs(args.input)
    validate_sweep(runs, args.dimensions, args.seeds)
    report = aggregate_sweep(runs, args.dimensions, args.seeds)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
