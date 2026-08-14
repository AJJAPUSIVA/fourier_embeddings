"""Aggregate matched training JSON files into machine- and human-readable reports."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


MATCHED_FIELDS = (
    "dataset", "dataset_source", "dataset_revision", "tokenizer", "deterministic", "model_config",
    "training.max_tokens", "training.batch_size", "training.grad_accumulation",
    "training.max_steps", "training.learning_rate", "training.weight_decay",
)


def nested(record: dict, path: str):
    value = record
    for part in path.split("."):
        value = value[part]
    return value


def load_runs(root: Path) -> list[dict]:
    runs = []
    for path in sorted(root.rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema_version") == 1 and "embedding" in record:
            record["source_file"] = str(path)
            runs.append(record)
    if not runs:
        raise ValueError(f"No training result JSON files found below {root}")
    return runs


def validate_matrix(runs: list[dict], arms: list[str], seeds: list[int]) -> None:
    expected = {(arm, seed) for arm in arms for seed in seeds}
    actual = [(run["embedding"], run["seed"]) for run in runs]
    duplicates = sorted(pair for pair in set(actual) if actual.count(pair) > 1)
    missing = sorted(expected - set(actual))
    extra = sorted(set(actual) - expected)
    if duplicates or missing or extra:
        raise ValueError(f"Invalid run matrix: duplicates={duplicates}, missing={missing}, extra={extra}")
    baseline = runs[0]
    for run in runs[1:]:
        mismatches = [field for field in MATCHED_FIELDS if nested(run, field) != nested(baseline, field)]
        if mismatches:
            raise ValueError(f"Unmatched settings in {run['source_file']}: {mismatches}")
    for run in runs:
        normalized_rms = run["initial_embedding_scale"]["normalized_rms"]
        if not 0.99 <= normalized_rms <= 1.01:
            raise ValueError(
                f"Embedding scale check failed in {run['source_file']}: "
                f"normalized_rms={normalized_rms}"
            )


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def aggregate(
    runs: list[dict],
    arms: list[str],
    *,
    max_fourier_ppl_regression_pct: float = 3.0,
    min_fourier_reduction_vs_kronecker: float = 16.0,
) -> dict:
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["embedding"]].append(run)
    summary = {}
    for arm in arms:
        arm_runs = sorted(grouped[arm], key=lambda item: item["seed"])
        summary[arm] = {
            "seeds": [run["seed"] for run in arm_runs],
            "validation_loss": mean_std([run["final_validation_loss"] for run in arm_runs]),
            "validation_perplexity": mean_std([run["final_validation_perplexity"] for run in arm_runs]),
            "elapsed_s": mean_std([run["elapsed_s"] for run in arm_runs]),
            "tokens_per_second": mean_std([run["tokens_per_second"] for run in arm_runs]),
            "embedding_parameters": arm_runs[0]["parameters"]["embedding"],
            "total_parameters": arm_runs[0]["parameters"]["total"],
            "raw_embedding_rms": mean_std([run["initial_embedding_scale"]["raw_rms"] for run in arm_runs]),
            "normalized_embedding_rms": mean_std([run["initial_embedding_scale"]["normalized_rms"] for run in arm_runs]),
        }
    if "dense" in summary:
        dense_ppl = summary["dense"]["validation_perplexity"]["mean"]
        dense_params = summary["dense"]["embedding_parameters"]
        for arm, values in summary.items():
            values["perplexity_change_vs_dense_pct"] = 100 * (values["validation_perplexity"]["mean"] / dense_ppl - 1)
            values["embedding_parameter_reduction_vs_dense"] = dense_params / max(values["embedding_parameters"], 1)
    criteria = []
    if "fourier" in summary and "kronecker" in summary:
        fourier = summary["fourier"]
        kronecker = summary["kronecker"]
        ppl_regression = 100 * (
            fourier["validation_perplexity"]["mean"]
            / kronecker["validation_perplexity"]["mean"]
            - 1
        )
        parameter_reduction = (
            kronecker["embedding_parameters"] / fourier["embedding_parameters"]
        )
        criteria.extend([
            {
                "id": "fourier_quality_vs_kronecker",
                "description": "Fourier mean validation PPL regression versus Kronecker",
                "measured": ppl_regression,
                "operator": "<=",
                "threshold": max_fourier_ppl_regression_pct,
                "unit": "percent",
                "status": "PASS" if ppl_regression <= max_fourier_ppl_regression_pct else "FAIL",
            },
            {
                "id": "fourier_embedding_reduction_vs_kronecker",
                "description": "Kronecker/Fourier embedding parameter ratio",
                "measured": parameter_reduction,
                "operator": ">=",
                "threshold": min_fourier_reduction_vs_kronecker,
                "unit": "times",
                "status": "PASS" if parameter_reduction >= min_fourier_reduction_vs_kronecker else "FAIL",
            },
        ])
    overall = "PASS" if criteria and all(item["status"] == "PASS" for item in criteria) else "FAIL"
    baseline = runs[0]
    return {
        "schema_version": 2,
        "run_count": len(runs),
        "experiment": {
            "dataset": baseline["dataset"],
            "dataset_source": baseline["dataset_source"],
            "dataset_revision": baseline["dataset_revision"],
            "tokenizer": baseline["tokenizer"],
            "deterministic": baseline["deterministic"],
            "seeds": sorted({run["seed"] for run in runs}),
            "model_config": baseline["model_config"],
            "training": baseline["training"],
        },
        "overall_status": overall,
        "criteria": criteria,
        "arms": summary,
    }


def markdown_report(report: dict) -> str:
    lines = [
        f"# Deterministic matched-training results — {report['overall_status']}", "",
        "Values are mean ± sample standard deviation across seeds.", "",
        "| Embedding | Seeds | Validation PPL | PPL vs dense | Raw RMS | Normalized RMS | Embedding params | Reduction vs dense | Tokens/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, values in report["arms"].items():
        ppl = values["validation_perplexity"]
        speed = values["tokens_per_second"]
        raw_rms = values["raw_embedding_rms"]["mean"]
        normalized_rms = values["normalized_embedding_rms"]["mean"]
        change = values.get("perplexity_change_vs_dense_pct", 0.0)
        reduction = values.get("embedding_parameter_reduction_vs_dense", 1.0)
        lines.append(
            f"| {arm} | {len(values['seeds'])} | {ppl['mean']:.2f} ± {ppl['std']:.2f} "
            f"| {change:+.2f}% | {raw_rms:.4f} | {normalized_rms:.4f} "
            f"| {values['embedding_parameters']:,} | {reduction:.1f}× "
            f"| {speed['mean']:.1f} ± {speed['std']:.1f} |"
        )
    lines.extend([
        "", "## Acceptance criteria", "",
        "| Criterion | Measured | Required | Status |",
        "|---|---:|---:|:---:|",
    ])
    for item in report["criteria"]:
        suffix = "%" if item["unit"] == "percent" else "×"
        lines.append(
            f"| {item['description']} | {item['measured']:.2f}{suffix} "
            f"| {item['operator']} {item['threshold']:.2f}{suffix} | **{item['status']}** |"
        )
    lines.extend(["", "The verdict applies only to these predeclared empirical criteria; it is not a proof of injectivity.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=["dense", "kronecker", "fourier"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1337, 2027, 3407])
    parser.add_argument("--json-output", type=Path, default=Path("training_results.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("training_summary.md"))
    parser.add_argument("--max-fourier-ppl-regression-pct", type=float, default=3.0)
    parser.add_argument("--min-fourier-reduction-vs-kronecker", type=float, default=16.0)
    args = parser.parse_args()
    runs = load_runs(args.input)
    validate_matrix(runs, args.arms, args.seeds)
    report = aggregate(
        runs,
        args.arms,
        max_fourier_ppl_regression_pct=args.max_fourier_ppl_regression_pct,
        min_fourier_reduction_vs_kronecker=args.min_fourier_reduction_vs_kronecker,
    )
    args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(markdown_report(report), encoding="utf-8")


if __name__ == "__main__":
    main()
