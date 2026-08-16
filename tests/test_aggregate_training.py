import copy

import pytest

from experiments.aggregate_training import aggregate, mean_std_ci95, validate_matrix


def run(arm, seed, ppl):
    return {
        "embedding": arm, "seed": seed, "dataset": "synthetic",
        "dataset_source": "generated:synthetic-v1", "dataset_revision": None,
        "tokenizer": "gpt2",
        "deterministic": True, "model_config": {"d_model": 8},
        "training": {"max_tokens": 100, "batch_size": 2, "grad_accumulation": 2,
                     "max_steps": 3, "learning_rate": 0.001, "weight_decay": 0.0},
        "final_validation_loss": 1.0, "final_validation_perplexity": ppl,
        "elapsed_s": 2.0, "tokens_per_second": 10.0,
        "parameters": {"embedding": 100 if arm == "dense" else 10, "total": 200},
        "initial_embedding_scale": {"raw_rms": 0.5, "normalized_rms": 1.0},
        "source_file": f"{arm}-{seed}.json",
    }


def test_aggregate_matched_matrix():
    runs = [run(arm, seed, 10.0 if arm == "dense" else 11.0)
            for arm in ("dense", "fourier") for seed in (1, 2)]
    validate_matrix(runs, ["dense", "fourier"], [1, 2])
    report = aggregate(runs, ["dense", "fourier"])
    assert report["arms"]["fourier"]["perplexity_change_vs_dense_pct"] == pytest.approx(10.0)
    assert report["arms"]["fourier"]["embedding_parameter_reduction_vs_dense"] == 10.0


def test_result_based_criteria_report_pass_and_fail():
    passing = [run("dense", 1, 20.0), run("kronecker", 1, 10.0), run("fourier", 1, 10.2)]
    passing[1]["parameters"]["embedding"] = 160
    passing[2]["parameters"]["embedding"] = 10
    report = aggregate(passing, ["dense", "kronecker", "fourier"])
    assert report["overall_status"] == "PASS"
    assert [item["status"] for item in report["criteria"]] == ["PASS", "PASS"]

    failing = copy.deepcopy(passing)
    failing[2]["final_validation_perplexity"] = 11.0
    report = aggregate(failing, ["dense", "kronecker", "fourier"])
    assert report["overall_status"] == "FAIL"
    assert report["criteria"][0]["status"] == "FAIL"


def test_validation_rejects_unmatched_hyperparameters():
    runs = [run("dense", 1, 10.0), run("fourier", 1, 11.0)]
    runs[1] = copy.deepcopy(runs[1])
    runs[1]["training"]["max_steps"] = 4
    with pytest.raises(ValueError, match="Unmatched settings"):
        validate_matrix(runs, ["dense", "fourier"], [1])


def test_validation_rejects_bad_normalized_scale():
    runs = [run("dense", 1, 10.0), run("fourier", 1, 11.0)]
    runs[1]["initial_embedding_scale"]["normalized_rms"] = 0.5
    with pytest.raises(ValueError, match="Embedding scale check failed"):
        validate_matrix(runs, ["dense", "fourier"], [1])


def test_validation_rejects_missing_seed_pair():
    runs = [
        run("dense", 1, 20.0),
        run("dense", 2, 20.0),
        run("kronecker", 1, 10.0),
        run("kronecker", 2, 100.0),
        run("fourier", 1, 11.0),
    ]
    with pytest.raises(ValueError, match=r"missing=.*\('fourier', 2\)"):
        validate_matrix(runs, ["dense", "kronecker", "fourier"], [1, 2])


def test_paired_statistics_match_by_seed_not_input_order():
    runs = [
        run("fourier", 2, 102.0),
        run("kronecker", 1, 10.0),
        run("dense", 2, 200.0),
        run("fourier", 1, 11.0),
        run("dense", 1, 200.0),
        run("kronecker", 2, 100.0),
    ]
    for item in runs:
        if item["embedding"] == "kronecker":
            item["parameters"]["embedding"] = 160
        elif item["embedding"] == "fourier":
            item["parameters"]["embedding"] = 10
    validate_matrix(runs, ["dense", "kronecker", "fourier"], [1, 2])
    report = aggregate(runs, ["dense", "kronecker", "fourier"])
    comparison = report["comparisons"]["fourier_vs_kronecker"]
    paired = comparison["validation_perplexity"]["paired_regression_pct"]

    assert comparison["paired_seeds"] == [1, 2]
    assert paired["values"] == pytest.approx([10.0, 2.0])
    assert paired["mean"] == pytest.approx(6.0)
    assert comparison["embedding_projection"]["reduction_factor"] == 16.0
    assert comparison["embedding_projection"]["reduction_pct"] == 93.75
    assert comparison["embedding_projection"]["float32_weight_memory_bytes"] == {
        "kronecker": 640,
        "fourier": 40,
    }


def test_aggregate_mean_and_mean_paired_regressions_are_distinct():
    runs = [
        run("dense", 1, 200.0), run("dense", 2, 200.0),
        run("kronecker", 1, 10.0), run("kronecker", 2, 100.0),
        run("fourier", 1, 11.0), run("fourier", 2, 102.0),
    ]
    for item in runs:
        if item["embedding"] == "kronecker":
            item["parameters"]["embedding"] = 160
    report = aggregate(runs, ["dense", "kronecker", "fourier"])
    ppl = report["comparisons"]["fourier_vs_kronecker"]["validation_perplexity"]

    assert ppl["aggregate_mean_regression_pct"] == pytest.approx(100 * (56.5 / 55.0 - 1))
    assert ppl["paired_regression_pct"]["mean"] == pytest.approx(6.0)
    assert ppl["aggregate_mean_regression_pct"] != pytest.approx(
        ppl["paired_regression_pct"]["mean"]
    )


def test_student_t_interval_is_reproducible():
    values = [1.7577252813141042, 2.879995438857308, 3.569007094249743]
    first = mean_std_ci95(values)
    second = mean_std_ci95(values)

    assert first == second
    assert first["n"] == 3
    assert first["degrees_of_freedom"] == 2
    assert first["mean"] == pytest.approx(2.7355759381403852)
    assert first["std"] == pytest.approx(0.9142364002862431)
    assert first["ci95"] == pytest.approx([0.46448681860221575, 5.006665057678555])
    assert mean_std_ci95([2.0])["ci95"] is None


def test_descriptive_worst_seed_does_not_change_declared_verdict():
    runs = [
        run("dense", 1, 200.0), run("dense", 2, 200.0),
        run("kronecker", 1, 100.0), run("kronecker", 2, 100.0),
        run("fourier", 1, 96.0), run("fourier", 2, 109.0),
    ]
    for item in runs:
        if item["embedding"] == "kronecker":
            item["parameters"]["embedding"] = 160
    report = aggregate(runs, ["dense", "kronecker", "fourier"])
    ppl = report["comparisons"]["fourier_vs_kronecker"]["validation_perplexity"]

    assert ppl["aggregate_mean_regression_pct"] == pytest.approx(2.5)
    assert ppl["worst_seed_regression_pct"] == pytest.approx(9.0)
    assert report["overall_status"] == "PASS"
    assert [item["status"] for item in report["criteria"]] == ["PASS", "PASS"]
