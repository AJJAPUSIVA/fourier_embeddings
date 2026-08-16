"""Aggregate matched training JSON files into machine- and human-readable reports."""

from __future__ import annotations

import argparse
import json
import math
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


_T_CRITICAL_975 = (
    12.706204736, 4.302652730, 3.182446305, 2.776445105, 2.570581836,
    2.446911851, 2.364624252, 2.306004135, 2.262157163, 2.228138852,
    2.200985160, 2.178812830, 2.160368656, 2.144786688, 2.131449546,
    2.119905299, 2.109815578, 2.100922040, 2.093024054, 2.085963447,
    2.079613845, 2.073873068, 2.068657610, 2.063898562, 2.059538553,
    2.055529439, 2.051830516, 2.048407142, 2.045229642, 2.042272456,
)


def mean_std_ci95(values: list[float]) -> dict[str, object]:
    """Return sample statistics and a two-sided 95% Student-t interval."""
    result: dict[str, object] = mean_std(values)
    result["values"] = values
    result["n"] = len(values)
    if len(values) < 2:
        result["ci95"] = None
        result["ci95_method"] = "unavailable with fewer than two paired observations"
        return result
    degrees_of_freedom = len(values) - 1
    if degrees_of_freedom <= len(_T_CRITICAL_975):
        critical = _T_CRITICAL_975[degrees_of_freedom - 1]
    elif degrees_of_freedom <= 40:
        critical = _T_CRITICAL_975[-1]
    elif degrees_of_freedom <= 60:
        critical = 2.021075390
    elif degrees_of_freedom <= 120:
        critical = 2.000297822
    else:
        critical = 1.979930406
    margin = critical * float(result["std"]) / math.sqrt(len(values))
    result["ci95"] = [float(result["mean"]) - margin, float(result["mean"]) + margin]
    result["ci95_method"] = "two-sided Student t interval"
    result["degrees_of_freedom"] = degrees_of_freedom
    return result


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
    comparisons = {}
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
        fourier_by_seed = {
            run["seed"]: run for run in grouped["fourier"]
        }
        kronecker_by_seed = {
            run["seed"]: run for run in grouped["kronecker"]
        }
        paired_seeds = sorted(set(fourier_by_seed) & set(kronecker_by_seed))
        paired_regressions = [
            100 * (
                fourier_by_seed[seed]["final_validation_perplexity"]
                / kronecker_by_seed[seed]["final_validation_perplexity"]
                - 1
            )
            for seed in paired_seeds
        ]
        absolute_ppl_difference = (
            fourier["validation_perplexity"]["mean"]
            - kronecker["validation_perplexity"]["mean"]
        )
        fourier_memory_bytes = fourier["embedding_parameters"] * 4
        kronecker_memory_bytes = kronecker["embedding_parameters"] * 4
        comparisons["fourier_vs_kronecker"] = {
            "paired_seeds": paired_seeds,
            "validation_perplexity": {
                "absolute_aggregate_mean_difference": absolute_ppl_difference,
                "aggregate_mean_regression_pct": ppl_regression,
                "paired_regression_pct": mean_std_ci95(paired_regressions),
                "worst_seed_regression_pct": max(paired_regressions),
            },
            "embedding_projection": {
                "kronecker_parameters": kronecker["embedding_parameters"],
                "fourier_parameters": fourier["embedding_parameters"],
                "reduction_factor": parameter_reduction,
                "reduction_pct": 100 * (1 - 1 / parameter_reduction),
                "float32_weight_memory_bytes": {
                    "kronecker": kronecker_memory_bytes,
                    "fourier": fourier_memory_bytes,
                },
                "float32_weight_memory_mib": {
                    "kronecker": kronecker_memory_bytes / (1024 ** 2),
                    "fourier": fourier_memory_bytes / (1024 ** 2),
                },
            },
        }
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
        "comparisons": comparisons,
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
    comparison = report.get("comparisons", {}).get("fourier_vs_kronecker")
    if comparison:
        ppl = comparison["validation_perplexity"]
        paired = ppl["paired_regression_pct"]
        projection = comparison["embedding_projection"]
        ci = paired["ci95"]
        ci_text = "unavailable" if ci is None else f"[{ci[0]:.2f}%, {ci[1]:.2f}%]"
        lines.extend([
            "", "## Derived Fourier–Kronecker comparison", "",
            "These descriptive statistics do not add or change acceptance criteria.", "",
            "| Quantity | Value |", "|---|---:|",
            f"| Absolute aggregate-mean validation PPL difference | {ppl['absolute_aggregate_mean_difference']:.2f} |",
            f"| Aggregate-mean validation PPL regression | {ppl['aggregate_mean_regression_pct']:.2f}% |",
            f"| Mean paired validation PPL regression | {paired['mean']:.2f}% |",
            f"| Paired-regression sample standard deviation | {paired['std']:.2f}% |",
            f"| Paired-regression 95% Student-t interval | {ci_text} |",
            f"| Worst-seed validation PPL regression | {ppl['worst_seed_regression_pct']:.2f}% |",
            f"| Projection-parameter reduction | {projection['reduction_factor']:.2f}× ({projection['reduction_pct']:.2f}% fewer) |",
            f"| Raw float32 projection-weight storage | {projection['float32_weight_memory_mib']['kronecker']:.2f} MiB → {projection['float32_weight_memory_mib']['fourier']:.2f} MiB |",
        ])
    lines.extend(["", "The verdict applies only to these explicit committed empirical criteria; it is not a proof of injectivity.", ""])
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
