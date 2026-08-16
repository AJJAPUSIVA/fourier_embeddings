import xml.etree.ElementTree as ET

import pytest
import torch

from experiments.analyze_representation import (
    nearest_distinct_euclidean,
    neighbor_analysis,
    norm_analysis,
    pearson_correlation,
    write_cosine_histogram,
    write_norm_plot,
)


def test_pearson_marks_near_constant_norm_unavailable():
    lengths = torch.arange(1, 6, dtype=torch.float32)
    nearly_constant = torch.tensor(
        [22.605277, 22.605278, 22.605276, 22.605277, 22.605278]
    )
    result = pearson_correlation(lengths, nearly_constant)
    assert result["value"] is None
    assert result["status"] == "unavailable_near_constant_right_variable"


def test_pearson_computes_when_variance_is_meaningful():
    result = pearson_correlation(
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([2.0, 4.0, 6.0]),
    )
    assert result["status"] == "computed"
    assert result["value"] == pytest.approx(1.0)


def test_norm_analysis_separates_normalization_stages():
    raw = [b"a", b"aa", b"aaa"]
    stages = {
        "raw_sum": torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        "length_normalized": torch.tensor([[1.0, 0.0], [1.4142, 0.0], [1.7321, 0.0]]),
        "z_normalized": torch.tensor([[1.0, -1.0], [1.0, -1.0], [1.0, -1.0]]),
    }
    report = norm_analysis(raw, stages)
    assert sum(group["count"] for group in report["by_byte_length"]) == 3
    assert report["stages"]["raw_sum"]["norm_length_pearson"]["value"] == pytest.approx(1.0)
    assert report["stages"]["z_normalized"]["norm_length_pearson"]["value"] is None


def test_nearest_euclidean_excludes_duplicate_byte_strings():
    codes = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ])
    raw = [b"same", b"same", b"different", b"other"]
    values, neighbors = nearest_distinct_euclidean(codes, raw, block_size=2)
    assert neighbors[0].item() == 2
    assert neighbors[1].item() == 2
    assert values[0].item() == pytest.approx(2 ** 0.5 / 10)


def test_neighbor_analysis_handles_repetitions_and_permutations():
    codes = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.9, 0.1],
    ])
    raw = [b"aaa", b"aab", b"aba", b"baa"]
    report, cosines = neighbor_analysis(
        codes, raw, [0, 1, 2, 3], ["aaa", "aab", "aba", "baa"], block_size=2
    )
    assert report["analyzed_tokens"] == 4
    assert report["distinct_byte_strings"] == 4
    assert cosines.numel() == 4
    assert report["nearest_distinct_euclidean"]["summary"]["min"] > 0
    assert report["cosine_self_retrieval_margin"]["summary"]["min"] > 0


def test_diagnostic_svg_writers_produce_valid_xml(tmp_path):
    raw = [b"a", b"aa"]
    stages = {
        "raw_sum": torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
        "length_normalized": torch.tensor([[1.0, 0.0], [1.4, 0.0]]),
        "z_normalized": torch.tensor([[1.0, -1.0], [1.0, -1.0]]),
    }
    norm_path = tmp_path / "norm.svg"
    cosine_path = tmp_path / "cosine.svg"
    write_norm_plot(norm_analysis(raw, stages), norm_path)
    write_cosine_histogram(torch.tensor([0.1, 0.5, 0.9]), cosine_path)
    ET.parse(norm_path)
    ET.parse(cosine_path)
