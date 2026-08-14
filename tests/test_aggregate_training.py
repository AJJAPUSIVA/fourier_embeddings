import copy

import pytest

from experiments.aggregate_training import aggregate, validate_matrix


def run(arm, seed, ppl):
    return {
        "embedding": arm, "seed": seed, "dataset": "synthetic", "tokenizer": "gpt2",
        "deterministic": True, "model_config": {"d_model": 8},
        "training": {"max_tokens": 100, "batch_size": 2, "grad_accumulation": 2,
                     "max_steps": 3, "learning_rate": 0.001, "weight_decay": 0.0},
        "final_validation_loss": 1.0, "final_validation_perplexity": ppl,
        "elapsed_s": 2.0, "tokens_per_second": 10.0,
        "parameters": {"embedding": 100 if arm == "dense" else 10, "total": 200},
        "source_file": f"{arm}-{seed}.json",
    }


def test_aggregate_matched_matrix():
    runs = [run(arm, seed, 10.0 if arm == "dense" else 11.0)
            for arm in ("dense", "fourier") for seed in (1, 2)]
    validate_matrix(runs, ["dense", "fourier"], [1, 2])
    report = aggregate(runs, ["dense", "fourier"])
    assert report["arms"]["fourier"]["perplexity_change_vs_dense_pct"] == pytest.approx(10.0)
    assert report["arms"]["fourier"]["embedding_parameter_reduction_vs_dense"] == 10.0


def test_validation_rejects_unmatched_hyperparameters():
    runs = [run("dense", 1, 10.0), run("fourier", 1, 11.0)]
    runs[1] = copy.deepcopy(runs[1])
    runs[1]["training"]["max_steps"] = 4
    with pytest.raises(ValueError, match="Unmatched settings"):
        validate_matrix(runs, ["dense", "fourier"], [1])
